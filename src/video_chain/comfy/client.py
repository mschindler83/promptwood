"""The ComfyUI client: HTTP + websocket API, uploads, one clip (ComfyGenerator.generate).

ComfyGenerator is the base of everything that talks to ComfyUI: it also carries refine (refine.RefineMixin) and the
SeedVR2 upscale (seedvr.SeedvrMixin); ComfyImageEditor (image_edit.py) and ComfyPromptRefiner (prompt_refiner.py)
extend it.

ComfyUI loads the model once and keeps it as long as memory allows; free() asks it to unload.
Needs no torch: only the standard library (+ websocket-client for step progress; without it the result is
polled and there is no per-step progress)."""
from __future__ import annotations

from collections.abc import Collection, Mapping
import hashlib
import http.client
import json
import logging
import mimetypes
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePath

from .. import media, net
from .errors import ComfyError, ComfyJobFailed, ComfyUnavailable, LocalWriteError
from .loras import comfy_lora_roots, lora_name_for, trains_adaln
from .refine import RefineMixin, pick_refine_files
from .seedvr import SEEDVR_PRECISION_LOCAL, SeedvrMixin
from .settings import (DEFAULT_FRAMES, DEFAULT_HEIGHT, DEFAULT_REF_SIZE, DEFAULT_STEPS, DEFAULT_WIDTH,
                       ClipSettings)
from .video import (AUDIO_VAE, DISTILLED_CHECKPOINT, TEXT_ENCODER, TWO_PASS_NODES, UNET, UNET_REF, VEDA_NODE,
                    VIDEO_VAE, build_two_pass_workflow, build_workflow, clip_recipe,
                    pick_veda_predictor, with_ref_hint)
from .wfb import H3_LORA_NODE, SAVE_LAST_ID, SAVE_VIDEO_ID, SUBFOLDER, combo_options

logger = logging.getLogger(__name__)

try:
    import websocket  # websocket-client (optional: without it the result is polled)
except ImportError:
    websocket = None   # type: ignore[assignment]  # (optional dependency: the code checks for None)

DEFAULT_URL = "http://127.0.0.1:8188"
UPLOAD_MIN_RATE = 100 * 1024          # bytes/s an upload may be as slow as before it times out ...
UPLOAD_MIN_TIMEOUT = 120              # ... (and an upload may always take this long, seconds)
POLL_SECONDS = 5                      # the history is polled this often next to the websocket ...
POLL_SECONDS_NO_WS = 2                # ... and this often without one
POLL_SECONDS_FINISHED = 0.5           # ... and this often once the end was announced
POLL_ERRORS_MAX = 5                   # failed polls in a row before ComfyUI counts as gone
DOWNLOAD_TRIES = 3
REQUEST_TIMEOUT = 30                  # seconds one plain HTTP request to ComfyUI may take
STOP_TIMEOUT = 10                     # ... one of the requests that stop a job (they must not hang the Stop)
DOWNLOAD_TIMEOUT = 300                # ... the connection of a download (the body is streamed)
CHUNK_BYTES = 1 << 20                 # uploads and downloads are streamed in pieces of this size
WS_CONNECT_TIMEOUT = 5                # the websocket: connecting ...
WS_RECV_TIMEOUT = 2.0                 # ... and waiting for a message (the wait loop looks at Stop between them)
WS_RETRY_SECONDS = 10                 # a dropped websocket is connected again this often
WAIT_SLEEP_SECONDS = 0.5              # the wait loop's pause without a websocket
def _setting_dict(s: Mapping[str, object], key: str) -> dict:
    """A clip setting that is a dict ({"strength": ...}); another shape is said, not a KeyError / AttributeError."""
    v = s.get(key)
    if not isinstance(v, dict):
        raise ComfyError(f"the {key} setting is damaged (expected a dict, got {type(v).__name__})")
    return v


def _strength(d: dict) -> float:
    try:
        return float(d.get("strength", 1.0))
    except (TypeError, ValueError) as e:
        raise ComfyError(f"a LoRA strength is not a number: {d.get('strength')!r}") from e


_UPLOADS: dict[tuple[str, str], int] = {}     # (ComfyUI url, name there) -> jobs using that uploaded copy
_UPLOADS_LOCK = threading.Lock()


def free_models(url: str, unload_models: bool = True, timeout: float = STOP_TIMEOUT, opener=None) -> bool:
    """POST ComfyUI's /free (the one place that knows it; the app's windows call this too): unload_models False drops
    only the cached results. True when ComfyUI took it, False when it did not (it is not reachable, or refused)."""
    try:
        net.request_json(url.rstrip("/") + "/free", payload={"unload_models": bool(unload_models), "free_memory": True},
                         method="POST", timeout=timeout, opener=opener or net.urlopen)
        return True
    except (net.NetError, ValueError) as e:
        logger.warning("ComfyUI /free failed: %s", e)
        return False


def _arg(argv: list, name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


class ComfyGenerator(RefineMixin, SeedvrMixin):
    """Generator (see chain.Generator) that renders clips in ComfyUI. The SeedVR2 options are set by the app."""

    def __init__(self, url: str = DEFAULT_URL, log=print, progress=None, timeout: float = 3600,
                 comfy_dir: Path | None = None, cleanup: bool = True):
        """comfy_dir: ComfyUI's folder on this machine (for its LoRA folders and, with cleanup, to delete the
        copies it keeps of every clip and start image after they were downloaded)."""
        self.url = url.rstrip("/")
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ComfyError(f"ComfyUI's address must be http:// or https://, not {url!r}")
        self.log, self.progress, self.timeout = log, progress, timeout
        self._info: dict[str, dict] = {}          # /object_info answers (a node that is there stays there)
        self.node_times: dict = {}                # seconds per workflow node of the last job
        self._running_node: tuple | None = None
        self._last_seedvr = False
        self.seedvr_tiled = True                  # SeedVR2 options (the app sets them)
        self.seedvr_precision = SEEDVR_PRECISION_LOCAL
        self.seedvr_keep_cache = False
        self.client_id = uuid.uuid4().hex
        self.current: str | None = None          # prompt id being rendered
        self._cancelled = threading.Event()      # Stop was pressed: _wait gives up (a queued job has no history)
        # HyperFlow's patch stays on ComfyUI's cached H3 model and breaks the turbo LoRA's adapters after it
        # (measured: HyperFlow -> Turbo fails, every other order gives the very same clips): the models are
        # unloaded only for that switch. None: not known (another process may have used HyperFlow).
        self.last_hyperflow: bool | None = None
        self._loras: list[str] | None = None
        self.stats = self._json("GET", "/system_stats")          # raises ComfyUnavailable if not running
        self._check_models()
        self.comfy_dir = Path(comfy_dir).expanduser() if comfy_dir else None
        if self.comfy_dir is not None and not (self.comfy_dir / "main.py").is_file():
            self.log(f"{self.comfy_dir} does not look like ComfyUI's folder: no cleanup, LoRAs by name only")
            self.comfy_dir = None
        self.cleanup = bool(cleanup and self.comfy_dir is not None)
        self.roots = comfy_lora_roots(self.comfy_dir) if self.comfy_dir else []
        argv = list((self.stats.get("system") or {}).get("argv") or [])
        base = self.comfy_dir
        base_arg, out_arg, in_arg = (_arg(argv, f"--{n}-directory") for n in ("base", "output", "input"))
        if base is not None and base_arg:
            base = Path(base_arg).expanduser()
        self.output_dir: PurePath | None = (Path(out_arg).expanduser() if out_arg
                                            else (base / "output" if base else None))
        self.input_dir: PurePath | None = (Path(in_arg).expanduser() if in_arg
                                           else (base / "input" if base else None))

    # ---- HTTP
    def _open(self, req, timeout: float):
        """urlopen; LAN and loopback addresses never go through a proxy from the environment."""
        return net.urlopen(req, timeout)

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _request(self, method: str, path: str, data=None, headers: dict | None = None,
                 timeout: float = REQUEST_TIMEOUT) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers or {})
        try:
            with self._open(req, timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise ComfyError(f"ComfyUI {path}: HTTP {e.code} {body[:600]}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:
            raise ComfyUnavailable(f"ComfyUI is not reachable at {self.url} ({e})") from e

    def _json(self, method: str, path: str, payload=None, timeout: float = REQUEST_TIMEOUT):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        raw = self._request(method, path, data, headers, timeout)
        try:
            return json.loads(raw or b"null")
        except ValueError as e:
            raise ComfyError(f"ComfyUI {path}: an answer that is not JSON: {raw[:200]!r}") from e

    def _object_info(self, node: str, fresh: bool = False) -> dict:
        if not fresh and node in self._info:
            return self._info[node]
        info = ((self._json("GET", f"/object_info/{node}") or {}).get(node)) or {}
        if info:
            self._info[node] = info              # (a missing node is asked again: ComfyUI may get it restarted)
        return info

    def _check_models(self):
        """The H3 files must be in ComfyUI's model folders (and its H3 nodes present)."""
        if not self._object_info("MiniMaxH3ImageToVideo"):
            raise ComfyError("this ComfyUI has no MiniMax H3 nodes (update ComfyUI)")
        need = (("UNETLoader", "unet_name", UNET), ("CLIPLoader", "clip_name", TEXT_ENCODER),
                ("VAELoader", "vae_name", VIDEO_VAE), ("VAELoader", "vae_name", AUDIO_VAE))
        for node, field, name in need:
            req = self._object_info(node).get("input", {}).get("required", {})
            if name not in combo_options(req.get(field)):
                raise ComfyError(f"ComfyUI does not list {name} ({node}); is it in its model folders?")

    def ref_key(self):
        """How this ComfyUI names the reference image inputs in the API (from its node description)."""
        opt = self._object_info("MiniMaxH3ReferenceToVideo").get("input", {}).get("optional", {})
        spec = opt.get("ref_images")
        prefix = "ref_image_"
        if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
            prefix = (spec[1].get("template") or {}).get("prefix", prefix)
        return lambda i: f"ref_images.{prefix}{i}"

    def audio_key(self):
        """How this ComfyUI names the reference sound inputs (raises ComfyError if its H3 nodes take none)."""
        opt = self._object_info("MiniMaxH3ReferenceToVideo").get("input", {}).get("optional", {})
        spec = opt.get("ref_audios")
        if spec is None:
            raise ComfyError("this ComfyUI's MiniMax H3 Reference to Video node takes no sound references "
                             "(update ComfyUI)")
        prefix = "ref_audio_"
        if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
            prefix = (spec[1].get("template") or {}).get("prefix", prefix)
        return lambda i: f"ref_audios.{prefix}{i}"

    def video_key(self):
        """How this ComfyUI names the reference video inputs (raises ComfyError if its H3 nodes take none)."""
        opt = self._object_info("MiniMaxH3ReferenceToVideo").get("input", {}).get("optional", {})
        spec = opt.get("ref_videos")
        if spec is None:
            raise ComfyError("this ComfyUI's MiniMax H3 Reference to Video node takes no reference videos "
                             "(update ComfyUI)")
        prefix = "ref_video_"
        if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
            prefix = (spec[1].get("template") or {}).get("prefix", prefix)
        return lambda i: f"ref_videos.{prefix}{i}"

    def check_ref_model(self, checkpoint: bool = False):
        """The reference model is there (checkpoint: a chosen H3 checkpoint makes these clips, no ref2va file)."""
        req = self._object_info("UNETLoader").get("input", {}).get("required", {})
        if not checkpoint and UNET_REF not in combo_options(req.get("unet_name")):
            raise ComfyError(f"ComfyUI does not list {UNET_REF} (needed for reference images)")
        if not self._object_info("MiniMaxH3ReferenceToVideo"):
            raise ComfyError("this ComfyUI has no MiniMax H3 Reference to Video node")

    def check_checkpoint(self, ck: dict) -> None:
        """A chosen H3 checkpoint must be listed by ComfyUI (UNETLoader) and its sigma shift node be there."""
        req = self._object_info("UNETLoader").get("input", {}).get("required", {})
        if ck.get("name") not in combo_options(req.get("unet_name")):
            raise ComfyError(f"ComfyUI does not list the H3 checkpoint {ck.get('name')} (UNETLoader); is it in its "
                             "diffusion model folders?")
        if ck.get("distilled") and not self._object_info("MiniMaxH3SigmaShift"):
            raise ComfyError("this ComfyUI has no MiniMaxH3SigmaShift node, which distilled H3 checkpoints sample "
                             "with (update ComfyUI)")

    def two_pass_upscaler(self) -> str:
        """The H3 latent upscaler a two-pass clip upscales pass 1 with (its nodes must be there too)."""
        for node in TWO_PASS_NODES:
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node, which two-pass clips need (update ComfyUI)")
        req = self._object_info("LatentUpscaleModelLoader").get("input", {}).get("required", {})
        name = pick_refine_files([], combo_options(req.get("model_name")))["upscaler"]
        if not name:
            raise ComfyError("ComfyUI lists no H3 latent upscaler for two-pass clips "
                             "(minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors in models/latent_upscale_models)")
        return name

    def adaln_names(self, loras: list[tuple[str, str]]) -> set[str]:
        """The ComfyUI names of these LoRAs ((local file, name)) that train the adaln projections, to be loaded by
        Promptwood's H3 LoRA node (it also applies that part on the pruned H3 files). Without the node: none, and a
        log line (ComfyUI's own loader applies the rest and skips that part)."""
        names = {n for p, n in loras if trains_adaln(Path(p))}
        if names and not self._object_info(H3_LORA_NODE):
            self.log(f"ComfyUI has no {H3_LORA_NODE} node: the adaln part of {', '.join(sorted(names))} is skipped "
                     "(copy video-chain's comfy_nodes/video_chain_nodes into ComfyUI/custom_nodes and restart "
                     "ComfyUI)")
            return set()
        return names

    def veda_predictor(self, refs: bool) -> str:
        """The Veda predictor to use (the node must be there: a clip recorded as made with Veda is never made
        without it)."""
        info = self._object_info(VEDA_NODE)
        if not info:
            raise ComfyError("this clip is large enough for Veda sparse attention, but this ComfyUI has no Veda node "
                             "(install veda-sparse-attention: comfy node install veda-sparse-attention, then restart "
                             "ComfyUI), or switch off \u201cFast attention (Veda)\u201d in the project settings")
        names = combo_options(info.get("input", {}).get("required", {}).get("predictor"))
        name = pick_veda_predictor(names, refs)
        if name is None:
            raise ComfyError("ComfyUI lists no Veda predictor in models/veda "
                             "(https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)")
        return name

    def check_guides(self):
        for node in ("MiniMaxH3AddGuide", "LoadVideo", "GetVideoComponents"):
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node (update ComfyUI)")

    def hyperflow_file(self) -> str:
        """The HyperFlow weights ComfyUI lists for ApplyHyperFlowH3: the pruned build for the pruned H3 files."""
        info = self._object_info("ApplyHyperFlowH3")
        if not info:
            raise ComfyError("this ComfyUI has no ApplyHyperFlowH3 node (install ComfyUI-HyperFlow-H3)")
        names = combo_options(info.get("input", {}).get("required", {}).get("hyperflow_file"))
        names = [n for n in names if n.endswith(".safetensors")]
        pruned = [n for n in names if "pruned" in n.lower()]
        if not pruned:                           # this app uses the pruned H3 files only
            raise ComfyError("ComfyUI lists no pruned HyperFlow weights in models/hyperflow "
                             "(custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors)")
        return sorted(pruned)[0]

    def pika_lora(self, refs: bool) -> str:
        """ComfyUI's name of the Pika H3 Flex LoRA for the fl2va model (ref2va with face references)."""
        kind = "ref2va" if refs else "fl2va"
        names = [n for n in self.lora_names(refresh=True) if "pika_h3_flex" in n.lower() and kind in n.lower()]
        if not names:
            raise ComfyError(f"ComfyUI lists no Pika H3 Flex LoRA for {kind} (pika_h3_flex_{kind}_v1_bf16"
                             ".safetensors in a LoRA folder)")
        return sorted(names, key=len)[0]

    def check_speed(self, s: ClipSettings) -> None:
        """Before a run: the HyperFlow weights / Pika LoRA these settings need are listed by ComfyUI (raises
        ComfyError otherwise, so the run stops up front instead of failing clip by clip)."""
        if s.get("hyperflow"):
            self.hyperflow_file()
        if s.get("pika"):
            self.pika_lora(bool(s.get("refs") or s.get("ref_paths") or s.get("audio_refs") or s.get("audio_paths")
                                  or s.get("ref_videos")))

    def check_turbo(self):
        for node in ("MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler"):
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node (install ComfyUI-MiniMax-H3-Turbo)")

    def lora_names(self, refresh: bool = False) -> list[str]:
        if self._loras is None or refresh:
            req = self._object_info("LoraLoaderModelOnly", fresh=refresh).get("input", {}).get("required", {})
            self._loras = combo_options(req.get("lora_name"))
        return self._loras

    def name_for(self, lora: str) -> str | None:
        return (lora_name_for(lora, self.lora_names(), self.roots)
                or lora_name_for(lora, self.lora_names(refresh=True), self.roots))

    def missing_loras(self, loras: list[str]) -> list[str]:
        """LoRA files ComfyUI cannot load (not in its LoRA folders)."""
        return [lo for lo in loras if self.name_for(lo) is None]

    def note(self, text: str) -> None:
        """A line about what is going on (the worker sets its own)."""

    def free(self) -> bool:
        """Ask ComfyUI to unload its models and free the memory; False (and a log line) if that failed."""
        ok = free_models(self.url, unload_models=True, opener=self._open)
        if not ok:
            self.log("ComfyUI could not unload its models")
        return ok

    def free_cache(self) -> bool:
        """Ask ComfyUI to drop its cached results (the last job's frames), keeping the models loaded."""
        ok = free_models(self.url, unload_models=False, opener=self._open)
        if not ok:
            self.log("ComfyUI could not drop its cache")
        return ok

    def _save(self, f: dict, dest: Path) -> None:
        """Download a result into dest, written under a temporary name first (never a truncated file)."""
        dest = Path(dest)
        with media.part_file(dest) as tmp:
            self._download_to(f, tmp)

    def _download_to(self, f: dict, dest: Path) -> None:
        """Stream a result into dest in chunks (clips are hundreds of MB); a cut connection is tried again."""
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        for attempt in range(DOWNLOAD_TRIES):
            try:
                req = urllib.request.Request(f"{self.url}/view?{q}")
                with self._open(req, DOWNLOAD_TIMEOUT) as r:
                    try:
                        out = open(dest, "wb")
                    except OSError as e:
                        raise LocalWriteError(e) from e
                    with out:
                        while chunk := r.read(CHUNK_BYTES):
                            try:
                                out.write(chunk)
                            except OSError as e:
                                raise LocalWriteError(e) from e
                    want = r.headers.get("Content-Length")
                if want is not None and dest.stat().st_size != int(want):
                    raise ComfyUnavailable(f"the download of {f['filename']} was cut off")
                return
            except LocalWriteError as e:                # (a full disk or no right to write: no retry, no
                assert e.__cause__ is not None           # (LocalWriteError(e) always chains the OSError)
                raise e.__cause__ from None              #  "ComfyUI is unavailable"; the OSError itself)
            except urllib.error.HTTPError as e:
                raise ComfyError(f"ComfyUI /view: HTTP {e.code}") from e
            except (ComfyUnavailable, urllib.error.URLError, ConnectionError, TimeoutError, OSError,
                    http.client.HTTPException) as e:
                if attempt == DOWNLOAD_TRIES - 1:
                    raise ComfyUnavailable(f"the download of {f['filename']} failed ({e})") from e
                self.log(f"download of {f['filename']} failed ({e}): trying again")
                self._sleep(2 ** attempt)

    def _drop_uploads(self, uploads: list[tuple[str, Path]]) -> None:
        """Remove ComfyUI's copies of uploaded files, unless another job still uses the same copy."""
        if not self.cleanup:
            return
        for up, local in uploads:
            with _UPLOADS_LOCK:
                key = (self.url, up)
                left = _UPLOADS.get(key, 1) - 1
                if left > 0:
                    _UPLOADS[key] = left
                    continue
                _UPLOADS.pop(key, None)
            self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                         Path(local), same_size=False)

    def _queue(self, wf: dict) -> str:
        """Queue a workflow under an id chosen here, set as current before it is sent: a Stop while the request is
        on its way (or after a lost answer) can still cancel the job ComfyUI may have accepted."""
        pid = str(uuid.uuid4())
        self.current = pid
        self.node_times, self._running_node = {}, None
        res = self._json("POST", "/prompt", {"prompt": wf, "client_id": self.client_id, "prompt_id": pid})
        if not isinstance(res, dict):
            raise ComfyError(f"ComfyUI /prompt: an unexpected answer: {str(res)[:200]}")
        if res.get("node_errors"):
            self.current = None
            raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(res['node_errors'])[:600]}")
        self.current = res.get("prompt_id") or pid
        return self.current

    def _begin_job(self) -> None:
        """Start of a job: an earlier Stop is over, and ComfyUI's node descriptions are asked afresh (they are
        kept only while one job checks them, as ComfyUI's nodes may change between jobs)."""
        self._cancelled.clear()
        self._info.clear()

    def _run_workflow(self, wf: dict) -> dict:
        """Queue a workflow and wait for its history entry. The websocket is connected first (the job's messages
        are not missed) and closed after; a job that went wrong other than by ComfyUI's own error is cancelled so
        it cannot hold up the next one."""
        ws = self._connect_ws()
        try:
            self._queue(wf)
            if self._cancelled.is_set():         # Stop came before the job was queued: take it out again
                self.interrupt()
                raise ComfyJobFailed("stopped")
            return self._wait(self.current or "", ws)   # (_queue set it)
        except ComfyJobFailed:
            raise
        except BaseException:                    # (also a stop by KeyboardInterrupt / SystemExit: cancel the job)
            self.interrupt()
            raise
        finally:
            self._close_ws(ws)
            self.current = None

    def _combo(self, node: str, field: str) -> list[str]:
        """The values ComfyUI offers for a combo input of a node (empty when it has none)."""
        return combo_options(self._object_info(node).get("input", {}).get("required", {}).get(field))

    def interrupt(self) -> None:
        """Stop the clip being rendered (and drop it from the queue)."""
        self._cancelled.set()                    # also with no job queued yet: the job about to start gives up
        pid = self.current
        if not pid:
            return
        for path, payload in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
            try:
                self._json("POST", path, payload, timeout=STOP_TIMEOUT)
            except ComfyError as e:
                self.log(f"ComfyUI {path} (Stop) failed: {e}")

    # ---- one clip
    def upload(self, image: Path) -> str:
        """Upload a file (an image, or a guide clip) into ComfyUI's input/video-chain; returns its name there.
        The file is streamed, never held in memory. A copy another job uploaded and still uses is not sent again
        (the name is the content's hash); _drop_uploads removes it when the last job is done."""
        image = Path(image)
        digest, size = hashlib.sha1(), 0
        with open(image, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
                digest.update(chunk)
                size += len(chunk)
        suffix = image.suffix.lower() or ".png"
        if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
            suffix = ".bin"                          # (never an odd suffix into the multipart header)
        name = digest.hexdigest()[:20] + suffix
        with _UPLOADS_LOCK:
            have = _UPLOADS.get((self.url, f"{SUBFOLDER}/{name}"), 0) if self.cleanup else 0
            if have:
                _UPLOADS[(self.url, f"{SUBFOLDER}/{name}")] = have + 1
                return f"{SUBFOLDER}/{name}"
        ctype = {".png": "image/png", ".mp4": "video/mp4", ".wav": "audio/wav"}.get(suffix) \
            or mimetypes.guess_type(name)[0] or "application/octet-stream"
        boundary = uuid.uuid4().hex
        head = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode()
                        for field, value in (("subfolder", SUBFOLDER), ("type", "input"), ("overwrite", "true")))
        head += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n").encode()
        tail = f"\r\n--{boundary}--\r\n".encode()

        def body():
            yield head
            with open(image, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
                    yield chunk
            yield tail
        raw = self._request("POST", "/upload/image", body(),
                            {"Content-Type": f"multipart/form-data; boundary={boundary}",
                             "Content-Length": str(len(head) + size + len(tail))},
                            max(UPLOAD_MIN_TIMEOUT, size / UPLOAD_MIN_RATE))
        try:
            res = json.loads(raw)
            got = f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise ComfyError(f"ComfyUI /upload/image: an unexpected answer: {raw[:200]!r}") from e
        if self.cleanup:
            with _UPLOADS_LOCK:
                _UPLOADS[(self.url, got)] = _UPLOADS.get((self.url, got), 0) + 1
        return got

    def generate(self, prompt: str, start_image: Path | None, out_video: Path, out_last_frame: Path, seed: int,
                 settings: ClipSettings | None = None) -> None:
        self._begin_job()
        s: ClipSettings = settings or {}
        self._last_seedvr = False                  # (a SeedVR2 job after this one frees the H3 models first)
        loras = []
        stack = ([(s["lora"], s.get("strength", 1.0))] if s.get("lora") else []) + \
            [(e["lora"], e.get("strength", 1.0)) for e in s.get("extra_loras") or []]
        for path, strength in stack:                 # chained in this order
            name = self.name_for(path)
            if name is None:
                raise ComfyError(f"{Path(path).name} is not in ComfyUI's LoRA folders")
            loras.append((name, float(strength)))
        adaln = self.adaln_names([(p, n) for (p, _st), (n, _st2) in zip(stack, loras)])
        turbo = None
        if s.get("turbo"):
            self.check_turbo()
            t = _setting_dict(s, "turbo")
            if not t.get("lora"):
                raise ComfyError("the turbo setting has no LoRA")
            name = self.name_for(t["lora"])
            if name is None:
                raise ComfyError(f"{Path(t['lora']).name} is not in ComfyUI's LoRA folders")
            turbo = (name, _strength(t))
        if s.get("checkpoint"):
            self.check_checkpoint(_setting_dict(s, "checkpoint"))
        hyper = (self.hyperflow_file(), _strength(_setting_dict(s, "hyperflow"))) if s.get("hyperflow") else None
        pika = (self.pika_lora(bool(s.get("ref_paths") or s.get("audio_paths") or s.get("ref_videos"))),
                _strength(_setting_dict(s, "pika"))) \
            if s.get("pika") else None
        if turbo and self.last_hyperflow:
            self.free()                              # HyperFlow's patch would break the turbo LoRA
        if hyper:
            self.last_hyperflow = True               # patched from now on, even if this clip fails
        uploads: list[tuple[str, Path]] = []        # (ComfyUI name, local file); removed again even on errors
        try:
            try:
                self._run(prompt, start_image, out_video, out_last_frame, seed, s, loras, turbo, uploads, hyper, pika,
                          adaln)
            except ComfyJobFailed as e:
                # a HyperFlow-patched model left from before this process: unload once and try again
                if not (turbo and self.last_hyperflow is None and "shapes cannot be multiplied" in str(e)):
                    raise
                self.log("ComfyUI still had a HyperFlow-patched model: unloading it and trying again")
                self.free()
                self._run(prompt, start_image, out_video, out_last_frame, seed, s, loras, turbo, uploads, hyper, pika,
                          adaln)
            if not hyper:
                self.last_hyperflow = False          # a clip without HyperFlow ran: the model is unpatched
        finally:
            self._drop_uploads(uploads)

    def _run(self, prompt, start_image, out_video, out_last_frame, seed, s: ClipSettings, loras, turbo, uploads,
             hyper=None, pika=None, adaln: Collection[str] = ()) -> None:
        ck = _setting_dict(s, "checkpoint") if s.get("checkpoint") else {}
        own = bool(ck)                             # a chosen H3 checkpoint makes every clip (no ref2va file)
        distilled = DISTILLED_CHECKPOINT if ck.get("distilled") else None
        image = self.upload(Path(start_image)) if start_image is not None else None
        if image:
            uploads.append((image, start_image))
        guides = []
        if s.get("guides"):
            self.check_guides()
            for g in s["guides"]:
                name = self.upload(Path(g["video"]))
                uploads.append((name, g["video"]))
                guides.append({**g, "name": name})
        keyframes = []
        if s.get("keyframes"):
            self.check_guides()
            for kf in s["keyframes"]:
                name = self.upload(Path(kf["image"]))
                uploads.append((name, kf["image"]))
                keyframes.append((name, int(kf["frame_idx"])))
        end_image = None
        if s.get("end_image") and not any(g.get("at") == "end" for g in guides):
            end_image = self.upload(Path(s["end_image"]))
            uploads.append((end_image, s["end_image"]))
        refs = []
        frame_refs = [f for f in ("start", "end") if f in (s.get("frame_pictures") or [])]
        if s.get("ref_paths") or frame_refs:
            self.check_ref_model(own)
            for r in s.get("ref_paths") or []:
                refs.append(self.upload(Path(r)))
                uploads.append((refs[-1], r))
            if refs and s.get("ref_hint", True) and s.get("prompt_format") != "h3":   # (h3: the prompt ties them)
                prompt = with_ref_hint(prompt)
        sounds = []
        if s.get("audio_paths"):
            if not refs and not frame_refs:
                self.check_ref_model(own)
            for a in s["audio_paths"]:
                sounds.append(self.upload(Path(a)))
                uploads.append((sounds[-1], a))
        videos = []
        if s.get("ref_videos"):                    # the end of the clip before (a continuation): ref2va as well
            if not refs and not frame_refs and not sounds:
                self.check_ref_model(own)
            self.check_guides()                    # (LoadVideo + GetVideoComponents)
            for v in s["ref_videos"]:
                videos.append(self.upload(Path(v)))
                uploads.append((videos[-1], v))
        prefix = f"{SUBFOLDER}/{uuid.uuid4().hex[:12]}"
        two = _setting_dict(s, "two_pass") if s.get("two_pass") else None
        build = (lambda **kw: build_two_pass_workflow(two_pass=two, upscaler=self.two_pass_upscaler(), **kw)) \
            if two else build_workflow
        try:
            wf = build(prompt=prompt, image=image, seed=seed, width=s.get("width", DEFAULT_WIDTH),
                                height=s.get("height", DEFAULT_HEIGHT), frames=s.get("frames", DEFAULT_FRAMES),
                                steps=s.get("steps", DEFAULT_STEPS),
                                loras=loras, audio=bool(s.get("audio", True)), prefix=prefix, refs=refs,
                                ref_size=s.get("ref_size", DEFAULT_REF_SIZE),
                                ref_key=self.ref_key() if refs or frame_refs else None, frame_refs=frame_refs,
                                turbo=turbo, end_image=end_image, guides=guides, hyperflow=hyper, pika=pika,
                                cfg=float(s.get("cfg", 1.0)),
                                audio_refs=sounds, **({"audio_key": self.audio_key()} if sounds else {}),
                                start_pin=int(s.get("start_pin") or 1),
                                recipe=clip_recipe(s.get("recipe"), distilled["recipe"] if distilled else None),
                                keyframes=keyframes, unet=ck.get("name") or None, adaln_loras=adaln,
                                ref_videos=videos, **({"video_key": self.video_key()} if videos else {}),
                                veda=self.veda_predictor(bool(refs or sounds or videos))
                                if s.get("veda") and not distilled else None)
        except ValueError as e:                # (a setting combination that cannot be built: say why)
            raise ComfyError(str(e)) from e
        entry = self._run_workflow(wf)
        outputs = entry.get("outputs") or {}
        video = self._first_file(outputs.get(SAVE_VIDEO_ID))
        frame = self._first_file(outputs.get(SAVE_LAST_ID))
        if video is None or frame is None:
            raise ComfyError("ComfyUI finished without the video or the last frame")
        self._save(video, Path(out_video))
        self._save(frame, Path(out_last_frame))
        if self.cleanup:
            self._remove(self.output_dir, video, Path(out_video))
            self._remove(self.output_dir, frame, Path(out_last_frame))

    def _remove(self, top: PurePath | None, f: dict, local: Path, same_size: bool = True) -> None:
        """Delete ComfyUI's copy of a file, only inside its video-chain subfolder and (for outputs) only if the
        downloaded file has the same size."""
        if top is None:
            return
        try:
            area = (Path(top) / SUBFOLDER).resolve()
            target = (Path(top) / (f.get("subfolder") or "") / f["filename"]).resolve()
            if area not in target.parents or not target.is_file():
                return
            if same_size and target.stat().st_size != local.stat().st_size:
                return
            target.unlink()
        except OSError as e:
            logger.warning("could not remove the uploaded copy: %s", e)

    @staticmethod
    def _first_file(out: dict | None) -> dict | None:
        if not out:
            return None
        files = out.get("images") or out.get("videos") or out.get("gifs") or []
        return files[0] if files else None

    def _download(self, f: dict) -> bytes:
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        return self._request("GET", f"/view?{q}", timeout=DOWNLOAD_TIMEOUT)

    def _connect_ws(self):
        if websocket is None:                    # (websocket-client is not installed: polling)
            return None
        u = urllib.parse.urlparse(self.url)
        scheme = "wss" if u.scheme == "https" else "ws"
        try:
            ws = websocket.create_connection(f"{scheme}://{u.netloc}/ws?clientId={self.client_id}", timeout=WS_CONNECT_TIMEOUT)
            ws.settimeout(WS_RECV_TIMEOUT)
            return ws
        except Exception as e:  # noqa: BLE001  (no websocket: fall back to polling)
            logger.debug("no websocket, polling instead: %s", e)
            return None

    def _history(self, pid: str) -> dict | None:
        return (self._json("GET", f"/history/{pid}") or {}).get(pid)

    def handle_message(self, msg: dict, pid: str) -> str | None:
        """One websocket message: reports progress; returns 'done' or raises on an error for this prompt."""
        t, d = msg.get("type"), msg.get("data") or {}
        if d.get("prompt_id") not in (None, pid):
            return None                                  # (another job's message)
        mine = d.get("prompt_id") == pid
        if t == "executing" and mine:
            self._node_clock(d.get("node"))
            if d.get("node") is None:
                return "done"
        elif t == "progress" and self.progress is not None and d.get("max"):
            self.progress(int(d["value"]), int(d["max"]))
        elif t == "execution_success" and mine:
            self._node_clock(None)
            return "done"
        elif t == "execution_error" and mine:
            raise ComfyJobFailed(f"ComfyUI: {d.get('exception_type', 'error')}: {d.get('exception_message', '')}"
                                 f" (node {d.get('node_type', '?')})")
        elif t == "execution_interrupted" and mine:
            raise ComfyJobFailed("ComfyUI: the job was interrupted")
        return None

    def _node_clock(self, node) -> None:
        """Seconds per workflow node (node_times), from ComfyUI's "executing" messages: a node runs until the next
        one starts (None: the job ended). Cached nodes send none and are not counted."""
        now = time.monotonic()
        last = self._running_node
        if last is not None:
            self.node_times[last[0]] = self.node_times.get(last[0], 0.0) + now - last[1]
        self._running_node = (node, now) if node is not None else None

    @staticmethod
    def _is_ws_timeout(e: Exception) -> bool:
        return isinstance(e, TimeoutError) or (websocket is not None
                                               and isinstance(e, websocket.WebSocketTimeoutException))

    def _wait(self, pid: str, ws) -> dict:
        """Wait for the job's history entry. Messages come over the websocket (progress, the end); the history is
        polled next to it, and alone (every POLL_SECONDS_NO_WS) when there is no socket or it dropped (then a new
        one is tried every 10 s)."""
        end = time.monotonic() + self.timeout
        last_poll = 0.0
        poll_errors = 0
        finished = False                                 # the end was announced: fetch the result now
        reconnect_at = float("inf") if ws is None else 0.0
        own_ws = None
        try:
            while time.monotonic() < end:
                if self._cancelled.is_set():
                    raise ComfyJobFailed("stopped")
                if ws is not None:
                    try:
                        raw = ws.recv()
                    except Exception as e:  # noqa: BLE001
                        raw = None
                        if not self._is_ws_timeout(e):   # a dropped socket: poll, and try to connect again
                            self.log(f"ComfyUI's websocket dropped ({e}): polling the result")
                            ws, reconnect_at = None, time.monotonic()
                    if isinstance(raw, str):
                        try:
                            msg = json.loads(raw)
                        except ValueError:
                            msg = None
                        if isinstance(msg, dict):
                            try:
                                if self.handle_message(msg, pid) == "done":
                                    finished = True
                            except ComfyError:
                                raise
                            except Exception as e:  # noqa: BLE001  (a bug in a progress callback: not the socket's fault)
                                self.log(f"progress report failed: {e}")
                else:
                    self._sleep(WAIT_SLEEP_SECONDS)
                    if not finished and time.monotonic() >= reconnect_at:
                        new = self._connect_ws()
                        if new is None:                      # (the dropped socket is never used again)
                            reconnect_at = time.monotonic() + WS_RETRY_SECONDS
                        else:
                            self._close_ws(own_ws)           # (one this wait opened before, now dead)
                            ws = own_ws = new
                            reconnect_at = float("inf")
                every: float = POLL_SECONDS if ws is not None and not finished else POLL_SECONDS_NO_WS
                if finished:
                    every = POLL_SECONDS_FINISHED
                if time.monotonic() - last_poll >= every or last_poll == 0.0:
                    last_poll = time.monotonic()
                    try:
                        entry = self._history(pid)
                        poll_errors = 0
                    except ComfyUnavailable:
                        poll_errors += 1
                        if poll_errors >= POLL_ERRORS_MAX:                 # ~5 tries in a row: ComfyUI is really gone
                            raise
                        continue
                    if entry is not None:
                        st = entry.get("status") or {}
                        if st.get("status_str") == "error":
                            msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                            detail = msgs[-1][1] if msgs else {}
                            raise ComfyJobFailed(f"ComfyUI: {detail.get('exception_type', 'error')}: "
                                                 f"{detail.get('exception_message', '')}")
                        if st.get("completed", True):
                            return entry
            raise ComfyError(f"ComfyUI did not finish within {self.timeout / 60:.0f} min")
        finally:
            self._node_clock(None)                           # (the last node's time, also when the job failed)
            self._close_ws(own_ws)

    @staticmethod
    def _close_ws(ws) -> None:
        if ws is not None:
            try:
                ws.close()
            except Exception as e:  # noqa: BLE001  (a socket that is gone already)
                logger.debug("closing the websocket: %s", e)
