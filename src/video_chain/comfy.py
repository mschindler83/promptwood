"""MiniMax-H3 image-to-video through a running ComfyUI (its HTTP + websocket API).

The workflow follows ComfyUI's own blueprint "Image to Video (MiniMax H3)": UNETLoader / CLIPLoader (minimax) /
the video and audio VAE, MiniMaxH3ImageToVideo with the start image as first frame, SamplerCustomAdvanced with
res_multistep + the simple scheduler and guidance 1 (BasicGuider), VAEDecode + VAEDecodeAudio, CreateVideo at
24 fps. LoRAs are chained with LoraLoaderModelOnly. The clip's last frame is saved as a PNG by ComfyUI itself
(ImageFromBatch + SaveImage), so the next clip starts from the decoded frame, not from the H.264 file.

ComfyUI loads the model once and keeps it as long as memory allows; free() asks it to unload.
Needs no torch: only the standard library (+ websocket-client for step progress; without it the result is
polled and there is no per-step progress).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8188"
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SAMPLER = "res_multistep"
SCHEDULER = "simple"
SUBFOLDER = "video-chain"


class ComfyError(RuntimeError):
    pass


class ComfyUnavailable(ComfyError):
    pass


def _combo_options(spec) -> list[str]:
    """Options of a combo input in /object_info, old ([list]) or new (["COMBO", {"options": list}]) format."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return list(spec[0])
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return list(spec[1].get("options", []))
    return []


def lora_name_for(path: str, names: list[str], roots: list[Path] = ()) -> str | None:
    """ComfyUI's name for a LoRA file: the listed name the path ends with (longest match), else the listed name
    that is the same file (followed links) in one of ComfyUI's LoRA folders (roots)."""
    p = Path(path).as_posix()
    hits = [n for n in names if p.endswith("/" + n.replace("\\", "/")) or p == n]
    if hits:
        return max(hits, key=len)
    try:
        real = os.path.realpath(path)
    except OSError:
        return None
    for root in roots:
        for n in names:
            cand = Path(root) / n
            if cand.exists() and os.path.realpath(cand) == real:
                return n
    return None


def comfy_lora_roots(comfy_dir: Path) -> list[Path]:
    """ComfyUI's LoRA folders: models/loras plus the 'loras' entries of extra_model_paths.yaml."""
    comfy_dir = Path(comfy_dir).expanduser()
    roots = [comfy_dir / "models" / "loras"]
    cfg = comfy_dir / "extra_model_paths.yaml"
    if cfg.is_file():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            data = {}
        for section in data.values():
            if not isinstance(section, dict) or "loras" not in section:
                continue
            base = Path(os.path.expanduser(str(section.get("base_path", ""))))
            for entry in str(section["loras"]).split("\n"):
                entry = entry.strip()
                if entry:
                    root = Path(os.path.expanduser(entry))
                    roots.append(root if root.is_absolute() else (comfy_dir / base / root if not base.is_absolute()
                                                                  else base / root))
    return [r for r in roots if r.is_dir()]


def _arg(argv: list, name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def build_workflow(*, prompt: str, image: str, seed: int, width: int, height: int, frames: int, steps: int,
                   loras: list[tuple[str, float]], audio: bool, prefix: str) -> dict:
    """The API-format workflow for one clip (image = the uploaded start image, as LoadImage names it)."""
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "start": {"class_type": "LoadImage", "inputs": {"image": image}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
                            "height": int(height), "length": int(frames), "first_frame": ["start", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "last": {"class_type": "ImageFromBatch", "inputs": {"image": ["decode", 0], "batch_index": int(frames) - 1,
                                                             "length": 1}},
        "save_last": {"class_type": "SaveImage", "inputs": {"images": ["last", 0],
                                                            "filename_prefix": f"{prefix}_last"}},
    }
    model = ["unet", 0]
    for k, (name, strength) in enumerate(loras):
        wf[f"lora{k}"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [f"lora{k}", 0]
    wf["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": SCHEDULER,
                                                                "steps": int(steps), "denoise": 1.0}}
    wf["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}}
    video_inputs: dict = {"images": ["decode", 0], "fps": 24.0}
    if audio:
        wf["audio_vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}
        wf["decode_audio"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0],
                                                                         "vae": ["audio_vae", 0]}}
        video_inputs["audio"] = ["decode_audio", 0]
    wf["video"] = {"class_type": "CreateVideo", "inputs": video_inputs}
    wf["save_video"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix,
                                                              "format": "mp4"}}
    return wf


class ComfyGenerator:
    """Generator (see chain.Generator) that renders clips in ComfyUI."""

    restart_helps = False            # a missing LoRA is not fixed by starting a new worker

    def __init__(self, url: str = DEFAULT_URL, log=print, progress=None, timeout: float = 3600,
                 comfy_dir: Path | None = None, cleanup: bool = True):
        """comfy_dir: ComfyUI's folder on this machine (for its LoRA folders and, with cleanup, to delete the
        copies it keeps of every clip and start image after they were downloaded)."""
        self.url = url.rstrip("/")
        self.log, self.progress, self.timeout = log, progress, timeout
        self.client_id = uuid.uuid4().hex
        self.current: str | None = None          # prompt id being rendered
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
        if base is not None and _arg(argv, "--base-directory"):
            base = Path(_arg(argv, "--base-directory")).expanduser()
        self.output_dir = (Path(_arg(argv, "--output-directory")).expanduser() if _arg(argv, "--output-directory")
                           else (base / "output" if base else None))
        self.input_dir = (Path(_arg(argv, "--input-directory")).expanduser() if _arg(argv, "--input-directory")
                          else (base / "input" if base else None))

    # ---- HTTP
    def _request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None,
                 timeout: float = 30) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise ComfyError(f"ComfyUI {path}: HTTP {e.code} {body[:600]}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ComfyUnavailable(f"ComfyUI is not reachable at {self.url} ({e})") from e

    def _json(self, method: str, path: str, payload=None, timeout: float = 30):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        return json.loads(self._request(method, path, data, headers, timeout) or b"null")

    def _object_info(self, node: str) -> dict:
        return (self._json("GET", f"/object_info/{node}") or {}).get(node) or {}

    def _check_models(self):
        """The H3 files must be in ComfyUI's model folders (and its H3 nodes present)."""
        if not self._object_info("MiniMaxH3ImageToVideo"):
            raise ComfyError("this ComfyUI has no MiniMax H3 nodes (update ComfyUI)")
        need = (("UNETLoader", "unet_name", UNET), ("CLIPLoader", "clip_name", TEXT_ENCODER),
                ("VAELoader", "vae_name", VIDEO_VAE), ("VAELoader", "vae_name", AUDIO_VAE))
        for node, field, name in need:
            req = self._object_info(node).get("input", {}).get("required", {})
            if name not in _combo_options(req.get(field)):
                raise ComfyError(f"ComfyUI does not list {name} ({node}); is it in its model folders?")

    def lora_names(self, refresh: bool = False) -> list[str]:
        if self._loras is None or refresh:
            req = self._object_info("LoraLoaderModelOnly").get("input", {}).get("required", {})
            self._loras = _combo_options(req.get("lora_name"))
        return self._loras

    def name_for(self, lora: str) -> str | None:
        return (lora_name_for(lora, self.lora_names(), self.roots)
                or lora_name_for(lora, self.lora_names(refresh=True), self.roots))

    def missing_loras(self, loras: list[str]) -> list[str]:
        """LoRA files ComfyUI cannot load (not in its LoRA folders)."""
        return [lo for lo in loras if self.name_for(lo) is None]

    def can_use(self, loras: list[str]) -> bool:
        return not self.missing_loras(loras)

    def free(self) -> None:
        """Ask ComfyUI to unload its models and free the memory."""
        try:
            self._json("POST", "/free", {"unload_models": True, "free_memory": True})
        except ComfyError:
            pass

    def interrupt(self) -> None:
        """Stop the clip being rendered (and drop it from the queue)."""
        pid = self.current
        if not pid:
            return
        for path, payload in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
            try:
                self._json("POST", path, payload, timeout=10)
            except ComfyError:
                pass

    # ---- one clip
    def upload(self, image: Path) -> str:
        data = Path(image).read_bytes()
        name = hashlib.sha1(data).hexdigest()[:20] + ".png"
        boundary = uuid.uuid4().hex
        parts = []
        for field, value in (("subfolder", SUBFOLDER), ("type", "input"), ("overwrite", "true")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                      f"Content-Type: image/png\r\n\r\n").encode() + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        res = json.loads(self._request("POST", "/upload/image", b"".join(parts),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}, 120))
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def generate(self, prompt: str, start_image: Path, out_video: Path, out_last_frame: Path, seed: int,
                 settings: dict | None = None) -> None:
        s = settings or {}
        loras = []
        if s.get("lora"):
            name = self.name_for(s["lora"])
            if name is None:
                raise ComfyError(f"{Path(s['lora']).name} is not in ComfyUI's LoRA folders")
            loras.append((name, float(s.get("strength", 1.0))))
        image = self.upload(Path(start_image))
        prefix = f"{SUBFOLDER}/{uuid.uuid4().hex[:12]}"
        wf = build_workflow(prompt=prompt, image=image, seed=seed, width=s.get("width", 512),
                            height=s.get("height", 512), frames=s.get("frames", 73), steps=s.get("steps", 20),
                            loras=loras, audio=bool(s.get("audio", True)), prefix=prefix)
        ws = self._connect_ws()
        try:
            res = self._json("POST", "/prompt", {"prompt": wf, "client_id": self.client_id})
            if res.get("node_errors"):
                raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(res['node_errors'])[:600]}")
            self.current = res["prompt_id"]
            entry = self._wait(self.current, ws)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass
            self.current = None
        outputs = entry.get("outputs") or {}
        video = self._first_file(outputs.get("save_video"))
        frame = self._first_file(outputs.get("save_last"))
        if video is None or frame is None:
            raise ComfyError("ComfyUI finished without the video or the last frame")
        Path(out_video).write_bytes(self._download(video))
        Path(out_last_frame).write_bytes(self._download(frame))
        if self.cleanup:
            self._remove(self.output_dir, video, Path(out_video))
            self._remove(self.output_dir, frame, Path(out_last_frame))
            self._remove(self.input_dir, {"filename": Path(image).name, "subfolder": str(Path(image).parent)},
                         Path(start_image), same_size=False)

    def _remove(self, top: Path | None, f: dict, local: Path, same_size: bool = True) -> None:
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
        except OSError:
            pass

    @staticmethod
    def _first_file(out: dict | None) -> dict | None:
        if not out:
            return None
        files = out.get("images") or out.get("videos") or out.get("gifs") or []
        return files[0] if files else None

    def _download(self, f: dict) -> bytes:
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        return self._request("GET", f"/view?{q}", timeout=300)

    def _connect_ws(self):
        try:
            import websocket  # websocket-client
        except ImportError:
            return None
        u = urllib.parse.urlparse(self.url)
        scheme = "wss" if u.scheme == "https" else "ws"
        try:
            ws = websocket.create_connection(f"{scheme}://{u.netloc}/ws?clientId={self.client_id}", timeout=5)
            ws.settimeout(2.0)
            return ws
        except Exception:  # noqa: BLE001  (no websocket: fall back to polling)
            return None

    def _history(self, pid: str) -> dict | None:
        return (self._json("GET", f"/history/{pid}") or {}).get(pid)

    def handle_message(self, msg: dict, pid: str) -> str | None:
        """One websocket message: reports progress; returns 'done' or raises on an error for this prompt."""
        t, d = msg.get("type"), msg.get("data") or {}
        if d.get("prompt_id") not in (None, pid):
            return None
        if t == "progress" and self.progress is not None and d.get("max"):
            self.progress(int(d["value"]), int(d["max"]))
        elif t == "executing" and d.get("node") is None and d.get("prompt_id") == pid:
            return "done"
        elif t == "execution_success" and d.get("prompt_id") == pid:
            return "done"
        elif t == "execution_error" and d.get("prompt_id") == pid:
            raise ComfyError(f"ComfyUI: {d.get('exception_type', 'error')}: {d.get('exception_message', '')}"
                             f" (node {d.get('node_type', '?')})")
        elif t == "execution_interrupted" and d.get("prompt_id") == pid:
            raise ComfyError("ComfyUI: the job was interrupted")
        return None

    def _wait(self, pid: str, ws) -> dict:
        end = time.monotonic() + self.timeout
        last_poll = 0.0
        while time.monotonic() < end:
            if ws is not None:
                try:
                    raw = ws.recv()
                    if isinstance(raw, str) and self.handle_message(json.loads(raw), pid) == "done":
                        ws = None                       # finished: fetch the result below
                except ValueError:
                    pass
                except ComfyError:
                    raise
                except Exception:  # noqa: BLE001  (timeout or a dropped socket: keep polling)
                    pass
            else:
                time.sleep(0.5)
            if ws is None or time.monotonic() - last_poll > 5:
                last_poll = time.monotonic()
                entry = self._history(pid)
                if entry is not None:
                    st = entry.get("status") or {}
                    if st.get("status_str") == "error":
                        msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                        detail = msgs[-1][1] if msgs else {}
                        raise ComfyError(f"ComfyUI: {detail.get('exception_type', 'error')}: "
                                         f"{detail.get('exception_message', '')}")
                    if st.get("completed", True):
                        return entry
        raise ComfyError(f"ComfyUI did not finish within {self.timeout / 60:.0f} min")
