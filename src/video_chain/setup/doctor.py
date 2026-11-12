"""`video-chain doctor`: what is installed, what is missing and how to get it.

check_all() looks at files and programs only (fast, no network, the GPU is not used). deep_checks() asks a running
ComfyUI what it lists (starting ComfyUI when needed); render_test() makes one short real clip. Standard library
only, plus the app's own ComfyUI client.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from . import manifest as M
from . import system as S
from .download import human

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
MARK = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]", INFO: "[ -- ]"}
SETUP = "video-chain setup"


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str = ""
    fix: str = ""


def installed_components() -> list[str] | None:
    """The components the last setup run installed (None: setup never ran here)."""
    raw = paths.load_config().get("components")
    return [c for c in raw.split(",") if c] if raw else None


def comfy_python() -> Path | None:
    cfg = paths.load_config().get("comfy_python")
    if cfg and Path(cfg).is_file():
        return Path(cfg)
    for venv in (".venv", "venv"):
        p = paths.venv_python(paths.comfy_dir() / venv)
        if p.is_file():
            return p
    return None


def model_roots(comfy_dir: Path, folder: str) -> list[Path]:
    """Every place ComfyUI looks for models/<folder> (extra_model_paths.yaml included)."""
    from ..comfy import comfy_model_roots
    kind, _, rest = folder.partition("/")
    roots = comfy_model_roots(comfy_dir, kind)
    own = Path(comfy_dir) / "models" / kind
    if own not in roots:
        roots.insert(0, own)
    return [r / rest if rest else r for r in roots]


def find_file(f: dict, comfy_dir: Path, hf_cache: Path) -> list[Path] | None:
    """Where the file(s) of a manifest entry are on this computer, or None when one is missing or incomplete."""
    if f.get("hf"):
        p = M.hf_snapshot(f, hf_cache) / f["path"]
        return [p] if p.is_file() and p.stat().st_size == f["size"] else None
    names = f.get("extract") or [f["name"]]
    out = []
    for name in names:
        hit = None
        for root in model_roots(comfy_dir, f["dir"]):
            p = root / name
            if p.is_file() and ("extract" in f or p.stat().st_size == f["size"]):
                hit = p
                break
        if hit is None:
            return None
        out.append(hit)
    return out


def component_state(man: dict, cid: str, comfy_dir: Path, hf_cache: Path) -> tuple[int, int, int]:
    """(files there, files in all, bytes still to download)."""
    files = man["components"][cid]["files"]
    have = [f for f in files if find_file(f, comfy_dir, hf_cache) is not None]
    return len(have), len(files), sum(f["size"] for f in files if f not in have)


def node_dir(man: dict, node: str, comfy_dir: Path) -> Path:
    n = man["nodes"][node]
    return Path(comfy_dir) / "custom_nodes" / (n.get("dir") or Path(n["bundled"]).name)


def node_installed(man: dict, node: str, comfy_dir: Path) -> bool:
    d = node_dir(man, node, comfy_dir)
    if (d / "__init__.py").is_file():
        return True
    custom = Path(comfy_dir) / "custom_nodes"
    want = d.name.lower().replace("_", "-")
    try:
        return any(p.name.lower().replace("_", "-") == want and (p / "__init__.py").is_file()
                   for p in custom.iterdir())
    except OSError:
        return False


# ----------------------------------------------------------------------------- the quick checks
def check_system() -> list[Check]:
    g, out = "This computer", []
    osi = S.os_info()
    out.append(Check(g, "System", OK if osi["system"] == "Linux" and osi["machine"] in ("x86_64", "amd64") else WARN,
                     f"{osi['name']} ({osi['machine']})",
                     "" if osi["system"] == "Linux" else "Promptwood is tested on Linux (x86-64) only"))
    out.append(Check(g, "Python", OK if S.python_ok() else FAIL, platform.python_version(),
                     "" if S.python_ok() else "Python 3.12 or newer is needed: run install.sh, it brings its own"))
    ff = S.ffmpeg_state()
    if not ff["ffmpeg"] or not ff["ffprobe"]:
        out.append(Check(g, "ffmpeg", FAIL, "ffmpeg / ffprobe not found",
                         f"{SETUP} --only system   (downloads one), or: {S.install_hint('ffmpeg', osi)}"))
    elif not ff["x264"] or not ff["aac"]:
        lack = " and ".join(n for n, ok in (("libx264 (H.264)", ff["x264"]), ("aac", ff["aac"])) if not ok)
        out.append(Check(g, "ffmpeg", FAIL, f"{ff['ffmpeg']} {ff['version']} cannot encode {lack}",
                         f"{SETUP} --only system   (downloads a full ffmpeg), or: {S.install_hint('ffmpeg', osi)}"))
    else:
        out.append(Check(g, "ffmpeg", OK, f"{ff['ffmpeg']} {ff['version']}"))
    gpu = S.gpu()
    if gpu is None:
        out.append(Check(g, "Graphics card", FAIL, "no NVIDIA card found (nvidia-smi is missing or reports none)",
                         S.install_hint("nvidia", osi)))
    else:
        text = f"{gpu['name']}, {gpu['vram_mb'] / 1024:.0f} GB, driver {gpu['driver']}"
        if gpu["vram_mb"] < S.MIN_VRAM_MB:
            out.append(Check(g, "Graphics card", FAIL, text,
                             f"the video model needs about {S.MIN_VRAM_MB // 1000} GB of graphics memory or more"))
        elif gpu["vram_mb"] < S.GOOD_VRAM_MB:
            out.append(Check(g, "Graphics card", WARN, text, "tested with 24 GB; with less, clips take longer "
                                                             "(ComfyUI moves parts of the model to main memory)"))
        else:
            out.append(Check(g, "Graphics card", OK, text))
    libs = S.qt_missing_libs()
    if libs:
        out.append(Check(g, "Window libraries (Qt)", FAIL, "missing: " + ", ".join(libs), S.install_hint("qt", osi)))
    elif libs is not None:
        out.append(Check(g, "Window libraries (Qt)", OK))
    return out


def check_app() -> list[Check]:
    g, out = "Promptwood", []
    lacking = []
    for mod, dist in (("PySide6", "PySide6"), ("websocket", "websocket-client"), ("yaml", "pyyaml"),
                      ("keyring", "keyring")):
        try:
            __import__(mod)
        except ImportError:
            lacking.append(dist)
    out.append(Check(g, "The app's Python packages", FAIL if lacking else OK,
                     "missing: " + ", ".join(lacking) if lacking else "",
                     "run install.sh again" if lacking else ""))
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        out.append(Check(g, "Face detection (onnxruntime)", OK))
    except ImportError:
        out.append(Check(g, "Face detection (onnxruntime)", WARN, "not installed: no “crop to face”",
                         "run install.sh again (it installs the app with its faces extra)"))
    out.append(Check(g, "Home folder", INFO, str(paths.home())))
    return out


def check_comfy(man: dict, components: list[str]) -> list[Check]:
    g, out = "ComfyUI", []
    d = paths.comfy_dir()
    if not (d / "main.py").is_file():
        return [Check(g, "ComfyUI", FAIL, f"not found in {d}",
                      f"{SETUP}   (or, for a ComfyUI you already have: {SETUP} --comfy-dir /path/to/ComfyUI)")]
    out.append(Check(g, "ComfyUI", OK, str(d)))
    py = comfy_python()
    if py is None:
        out.append(Check(g, "ComfyUI's Python", WARN, "not known (a ComfyUI that was not installed by setup)",
                         f"{SETUP} --comfy-dir {d} --comfy-python /path/to/its/python"))
    else:
        r = S.run([str(py), "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"], 120)
        if r is None or r.returncode != 0:
            out.append(Check(g, "PyTorch", FAIL, f"{py} cannot import torch", f"{SETUP} --only comfyui"))
        elif r.stdout.strip().endswith("False"):
            out.append(Check(g, "PyTorch", FAIL, f"torch {r.stdout.split()[0]} does not see the graphics card",
                             f"update the NVIDIA driver ({S.install_hint('nvidia')}), then {SETUP} --only comfyui"))
        else:
            out.append(Check(g, "PyTorch", OK, f"torch {r.stdout.split()[0]} with CUDA"))
    launcher = paths.comfy_launcher()
    out.append(Check(g, "Launcher", OK if launcher.is_file() else FAIL, str(launcher),
                     "" if launcher.is_file() else f"{SETUP} --only comfyui"))
    for node in M.nodes_for(man, components):
        ok = node_installed(man, node, d)
        out.append(Check(g, f"Custom node {node_dir(man, node, d).name}", OK if ok else FAIL,
                         "" if ok else "not in ComfyUI's custom_nodes", "" if ok else f"{SETUP} --only nodes"))
    return out


def check_components(man: dict, components: list[str]) -> list[Check]:
    g, out = "Models", []
    d, cache = paths.comfy_dir(), paths.hf_cache()
    for cid, c in man["components"].items():
        have, total, left = component_state(man, cid, d, cache)
        size = human(c["bytes"])
        if have == total:
            out.append(Check(g, c["title"], OK, f"{cid}, {size}"))
        elif cid in components:
            out.append(Check(g, c["title"], FAIL, f"{cid}: {total - have} of {total} files missing "
                                                  f"({human(left)} to download)", f"{SETUP} --only {cid}"))
        else:
            out.append(Check(g, c["title"], INFO, f"{cid}: not installed ({size})", f"{SETUP} --with {cid}"))
    return out


def check_software(man: dict, components: list[str]) -> list[Check]:
    g, out = "Programs", []
    soft = M.software_for(man, components)
    if "tts" in soft:
        py = paths.tts_python()
        if not py.is_file():
            out.append(Check(g, "Text to speech environment", FAIL, f"{py} is missing", f"{SETUP} --only tts"))
        else:
            r = S.run([str(py), "-c", "import qwen_tts, soundfile, torch; print(torch.__version__)"], 180)
            ok = r is not None and r.returncode == 0
            out.append(Check(g, "Text to speech environment", OK if ok else FAIL,
                             f"torch {r.stdout.strip()}" if ok else f"{py} cannot import qwen_tts",
                             "" if ok else f"{SETUP} --only tts"))
    if "llama" in soft:
        server = paths.llama_server()
        r = S.run([server, "--version"], 30) if Path(server).is_file() or S.shutil.which(server) else None
        ok = r is not None and r.returncode == 0
        ver = ((r.stderr or r.stdout).strip().splitlines() or [""])[0] if ok else ""
        out.append(Check(g, "llama.cpp server", OK if ok else FAIL, f"{server} ({ver})" if ok else
                         f"{server} is missing or does not start", "" if ok else f"{SETUP} --only llm"))
    return out


def check_all(components: list[str] | None = None) -> list[Check]:
    man = M.load()
    if components is None:
        components = installed_components()
    never = components is None
    components = components or M.select(man, "standard")
    out = check_system() + check_app()
    if never:
        out.append(Check("Promptwood", "Setup", WARN, "setup has not run on this computer yet", SETUP))
    return out + check_comfy(man, components) + check_software(man, components) + check_components(man, components)


# ----------------------------------------------------------------------------- with a running ComfyUI
def comfy_up(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
        return False


class ComfySession:
    """ComfyUI for the checks: the running one is used as it is; else it is started with the launcher and stopped
    again afterwards (only the process started here is ever stopped)."""

    def __init__(self, url: str | None = None, launcher: Path | None = None, wait: float = 900, log=print):
        self.url, self.launcher = (url or paths.comfy_url()).rstrip("/"), Path(launcher or paths.comfy_launcher())
        self.wait, self.log = wait, log
        self.proc: subprocess.Popen | None = None
        self.logfile: Path | None = None

    def __enter__(self):
        if comfy_up(self.url):
            self.log(f"ComfyUI is running at {self.url}")
            return self
        if not self.launcher.is_file():
            raise RuntimeError(f"ComfyUI is not running at {self.url} and its launcher {self.launcher} is missing")
        paths.cache_dir().mkdir(parents=True, exist_ok=True)
        self.logfile = paths.cache_dir() / "comfyui-setup.log"
        self.log(f"Starting ComfyUI ({self.launcher}; its output goes to {self.logfile}) …")
        with open(self.logfile, "wb") as f:
            self.proc = subprocess.Popen([str(self.launcher)], stdout=f, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        end = time.monotonic() + self.wait
        while time.monotonic() < end:
            if comfy_up(self.url, 3):
                self.log("ComfyUI is up")
                return self
            if self.proc.poll() is not None:
                tail, code = self.logfile.read_text(errors="replace")[-3000:], self.proc.returncode
                self.proc = None
                raise RuntimeError(f"ComfyUI stopped while starting (exit code {code}):\n{tail}")
            time.sleep(2)
        self.__exit__(None, None, None)
        raise RuntimeError(f"ComfyUI did not answer at {self.url} within {int(self.wait)} s (see {self.logfile})")

    def __exit__(self, *exc):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            import signal
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    p.kill()
            self.log("ComfyUI stopped again")
        return False


def _info(url: str, node: str) -> dict:
    with urllib.request.urlopen(f"{url}/object_info/{node}", timeout=60) as r:
        return (json.loads(r.read()) or {}).get(node) or {}


def _listed(url: str, node: str, field: str) -> list[str]:
    from ..comfy import _combo_options
    return _combo_options(_info(url, node).get("input", {}).get("required", {}).get(field))


# which loader lists a models/<folder>: (node, input)
LOADERS = {"diffusion_models": ("UNETLoader", "unet_name"), "text_encoders": ("CLIPLoader", "clip_name"),
           "vae": ("VAELoader", "vae_name"), "loras": ("LoraLoaderModelOnly", "lora_name"),
           "latent_upscale_models": ("LatentUpscaleModelLoader", "model_name"),
           "model_patches": ("ModelPatchLoader", "name"), "clip_vision": ("CLIPVisionLoader", "clip_name"),
           "hyperflow": ("ApplyHyperFlowH3", "hyperflow_file"), "veda": ("VedaSparseAttention", "predictor")}
# the nodes each component's workflows are built from (the app's own lists)
def _component_nodes() -> dict[str, tuple[str, ...]]:
    from .. import animate, comfy
    return {"video": ("MiniMaxH3ImageToVideo", "SamplerCustomAdvanced", "CreateVideo", "SaveVideo", "VAEDecodeAudio"),
            "video-refs": ("MiniMaxH3ReferenceToVideo",),
            "video-turbo": ("MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler"),
            "video-2k": tuple(comfy.REFINE_NODES), "upscale": tuple(comfy.SEEDVR_NODES),
            "image": (*comfy.QWEN21_NODES, comfy.INPAINT21_NODE), "faces": (animate.FACE_NODE,),
            "video-speed": ("ApplyHyperFlowH3", comfy.VEDA_NODE), "headswap-wan": tuple(animate.ANIMATE_NODES),
            "headswap-h3": ("MiniMaxH3SigmaShift", "MiniMaxH3TurboLoRA")}


def deep_checks(url: str, components: list[str]) -> list[Check]:
    """Ask the running ComfyUI: are the nodes there, does it list every model file of the installed components."""
    man, g, out = M.load(), "ComfyUI (running)", []
    nodes = _component_nodes()
    lists: dict[str, list[str]] = {}
    for cid in components:
        c = man["components"][cid]
        problems = [f"no {n} node" for n in nodes.get(cid, ()) if not _info(url, n)]
        for f in c["files"]:
            if f.get("hf") or "extract" in f or f["dir"] not in LOADERS:
                continue
            if f["dir"] not in lists:
                lists[f["dir"]] = _listed(url, *LOADERS[f["dir"]])
            if not any(Path(n.replace("\\", "/")).name == f["name"] for n in lists[f["dir"]]):
                problems.append(f"{f['name']} is not listed")
        if cid == "faces" and not problems:
            from ..animate import FACE_NODE
            sw = _listed(url, FACE_NODE, "swapper")
            if not any("hyperswap" in s.lower() for s in sw):
                problems.append("no HyperSwap model listed")
        if any(f.get("hf") for f in c["files"]) and not nodes.get(cid):
            continue                                       # (not ComfyUI's business: text to speech, llama.cpp)
        out.append(Check(g, c["title"], FAIL if problems else OK, "; ".join(problems),
                         f"{SETUP} --only {cid}" if problems else ""))
    if "video" in components:
        try:
            from ..comfy import ComfyGenerator
            ComfyGenerator(url, comfy_dir=str(paths.comfy_dir()), cleanup=False, log=lambda *_: None)
            out.append(Check(g, "The app's own check of the video model", OK))
        except Exception as e:  # noqa: BLE001
            out.append(Check(g, "The app's own check of the video model", FAIL, str(e)))
    return out


def render_test(url: str, log=print, steps: int = 8) -> Check:
    """One short real clip from a generated picture, through the app's own command line."""
    g, name = "Proof", "A real clip (39 frames, with sound)"
    paths.use_private_tools()
    with tempfile.TemporaryDirectory(prefix="video-chain-test-") as tmp:
        tmp = Path(tmp)
        image, prompts, out = tmp / "start.png", tmp / "prompts.txt", tmp / "out"
        r = S.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=512x512:rate=1", "-frames:v",
                   "1", str(image)], 60)
        if r is None or r.returncode != 0:
            return Check(g, name, FAIL, "ffmpeg could not make the start picture")
        prompts.write_text("A colourful test pattern slowly dissolves into drifting clouds, a soft chime plays.\n")
        cmd = [sys.executable, "-m", "video_chain", str(image), str(prompts), "--out", str(out), "--frames", "39",
               "--steps", str(steps), "--resolution", "384", "--comfy-url", url, "--comfy-dir",
               str(paths.comfy_dir())]
        log("Rendering a test clip (the first one loads 35 GB of models: a few minutes) …")
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return Check(g, name, FAIL, "no clip after an hour")
        video = out / "chain.mp4"
        if p.returncode != 0 or not video.is_file():
            return Check(g, name, FAIL, (p.stderr or p.stdout).strip()[-1500:])
        probe = S.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,"
                       "nb_frames", "-of", "json", str(video)], 60)
        streams = json.loads(probe.stdout).get("streams", []) if probe and probe.returncode == 0 else []
        v = next((s for s in streams if s.get("codec_type") == "video"), None)
        a = next((s for s in streams if s.get("codec_type") == "audio"), None)
        if v is None:
            return Check(g, name, FAIL, "the result has no video stream")
        detail = (f"{v.get('width')}x{v.get('height')}, {v.get('nb_frames')} frames {v.get('codec_name')}, "
                  f"sound: {a.get('codec_name') if a else 'none'}, {video.stat().st_size // 1024} KB, "
                  f"{time.monotonic() - t0:.0f} s")
        return Check(g, name, OK if a else FAIL, detail)


def speech_test(log=print) -> Check:
    """One sentence spoken by the local text to speech (a designed voice), through the app's own TTS client."""
    g, name = "Proof", "A spoken line (Qwen3-TTS voice design)"
    from .. import tts
    paths.use_private_tools()
    local = tts.LocalTTS(start_timeout=600)
    bad = local.problems(design=True)
    if bad:
        return Check(g, name, FAIL, "; ".join(bad))
    with tempfile.TemporaryDirectory(prefix="video-chain-test-") as tmp:
        wav = Path(tmp) / "line.wav"
        log("Speaking a test line (loads the 1.7B voice model) …")
        try:
            ans = local.request({"cmd": "design", "model": str(tts.snapshot(tts.QWEN_DESIGN, local.cache)),
                              "text": "Hello, this is a test of the voice.", "language": "English",
                              "instruct": "A calm adult voice.", "out": str(wav), "seed": 1}, timeout=900.0)
        except Exception as e:  # noqa: BLE001
            return Check(g, name, FAIL, str(e))
        finally:
            local.stop()
        if not ans.get("ok") or not wav.is_file():
            return Check(g, name, FAIL, str(ans.get("error", "no sound file")))
        return Check(g, name, OK, f"{ans.get('secs')} s of sound at {ans.get('sr')} Hz")


def llm_test(log=print) -> Check:
    """llama.cpp's server answers with the installed language model."""
    g, name = "Proof", "The prompt refiner's language model (llama.cpp)"
    from ..llm_server import LlamaServer, find_mmproj
    man = M.load()
    f = next(f for f in man["components"]["llm"]["files"] if "mmproj" not in f["path"])
    model = M.hf_snapshot(f, paths.hf_cache()) / f["path"]
    if not model.is_file():
        return Check(g, name, FAIL, f"{model} is missing", f"{SETUP} --only llm")
    paths.cache_dir().mkdir(parents=True, exist_ok=True)
    port = 8091
    server = LlamaServer(paths.llama_server(), str(model), port, False, find_mmproj(str(model)),
                         log_dir=paths.cache_dir())
    log("Starting llama.cpp with the language model …")
    try:
        server.start()
        body = json.dumps({"messages": [{"role": "user", "content": "Answer with the single word: ready"}],
                           "max_tokens": 200, "temperature": 0}).encode()
        req = urllib.request.Request(f"{server.url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        return Check(g, name, OK if text else FAIL, f"it answered: {text[:60]!r}")
    except Exception as e:  # noqa: BLE001
        return Check(g, name, FAIL, str(e))
    finally:
        server.stop()


# ----------------------------------------------------------------------------- report
def report(checks: list[Check], colour: bool = False) -> str:
    lines, group = [], None
    tint = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", INFO: "\033[2m"}
    for c in checks:
        if c.group != group:
            group = c.group
            lines.append(f"\n{group}")
        mark = f"{tint[c.status]}{MARK[c.status]}\033[0m" if colour else MARK[c.status]
        lines.append(f"  {mark} {c.name}" + (f": {c.detail}" if c.detail else ""))
        if c.fix and c.status != OK:
            lines.append(f"         -> {c.fix}")
    return "\n".join(lines).lstrip("\n")


def summary(checks: list[Check]) -> str:
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    if fails:
        return f"{fails} problem{'s' if fails != 1 else ''} to fix (see the arrows above)."
    return "Everything is in place." + (f" ({warns} note{'s' if warns != 1 else ''})" if warns else "")


def failed(checks: list[Check]) -> bool:
    return any(c.status == FAIL for c in checks)
