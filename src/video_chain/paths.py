"""Where Promptwood keeps and finds things on this computer. Nothing here is tied to one person's machine.

One folder, the *home*, holds everything the installer sets up (ComfyUI, the model cache, the TTS environment,
llama.cpp, a private ffmpeg): $VIDEO_CHAIN_HOME, else the platform's data folder (~/.local/share/video-chain).
`config.json` in it records what the installer chose (e.g. an existing ComfyUI somewhere else). Every value can be
overridden by an environment variable, and the app's own preferences (QSettings) override the defaults shown there:

    key             environment variable          default
    comfy_dir       VIDEO_CHAIN_COMFY_DIR         <home>/ComfyUI
    comfy_launcher  VIDEO_CHAIN_COMFY_LAUNCHER    <home>/bin/comfyui
    comfy_url       VIDEO_CHAIN_COMFY_URL         http://127.0.0.1:8188
    hf_cache        VIDEO_CHAIN_HF_CACHE          <home>/hf-models   (Hugging Face cache layout)
    llama_server    VIDEO_CHAIN_LLAMA_SERVER      <home>/tools/llama.cpp/llama-server, else llama-server on PATH
    tts_python      VIDEO_CHAIN_TTS_PYTHON        <home>/tts-venv/bin/python
    fish_dir        VIDEO_CHAIN_FISH_DIR          <home>/fish-speech   (fish-speech's code with its own .venv)

Standard library only (the installer imports this before anything else is installed).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

APP = "video-chain"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
WINDOWS = os.name == "nt"
_BIN = "Scripts" if WINDOWS else "bin"
_EXE = ".exe" if WINDOWS else ""


def home() -> Path:
    """The folder the installer fills (created on demand by whoever writes into it)."""
    env = os.environ.get("VIDEO_CHAIN_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP


def cache_dir() -> Path:
    """Logs and other things that can be thrown away."""
    if WINDOWS:
        return home() / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / APP


def config_file() -> Path:
    return home() / "config.json"


def load_config() -> dict:
    try:
        data = json.loads(config_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(**values) -> dict:
    """Merge values into config.json (None removes a key)."""
    data = load_config()
    for k, v in values.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = str(v)
    config_file().parent.mkdir(parents=True, exist_ok=True)
    tmp = config_file().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(config_file())
    return data


def venv_python(venv: Path) -> Path:
    return Path(venv) / _BIN / ("python" + _EXE)


def _defaults() -> dict:
    h = home()
    return {
        "comfy_dir": h / "ComfyUI",
        "comfy_launcher": h / "bin" / ("comfyui" + (".bat" if WINDOWS else "")),
        "comfy_url": DEFAULT_COMFY_URL,
        "hf_cache": h / "hf-models",
        "llama_server": h / "tools" / "llama.cpp" / ("llama-server" + _EXE),
        "tts_python": venv_python(h / "tts-venv"),
        "fish_dir": h / "fish-speech",
    }


def setting(key: str) -> str:
    """Environment variable, else config.json, else the default (see the module text)."""
    env = os.environ.get("VIDEO_CHAIN_" + key.upper(), "").strip()
    if env:
        return os.path.expanduser(env)
    cfg = load_config().get(key)
    if cfg:
        return os.path.expanduser(str(cfg))
    return str(_defaults()[key])


def comfy_dir() -> Path:
    return Path(setting("comfy_dir"))


def comfy_launcher() -> Path:
    return Path(setting("comfy_launcher"))


def comfy_url() -> str:
    return setting("comfy_url").rstrip("/")


def hf_cache() -> Path:
    return Path(setting("hf_cache"))


def tts_python() -> Path:
    return Path(setting("tts_python"))


def fish_dir() -> Path:
    return Path(setting("fish_dir"))


def fish_python() -> Path:
    return venv_python(fish_dir() / ".venv")


def llama_server() -> str:
    """The llama-server program: the configured / installed one, else the one on PATH (the configured path is
    returned when neither exists, so messages can name where it was expected)."""
    want = setting("llama_server")
    if Path(want).is_file():
        return want
    return shutil.which("llama-server") or want


def model_cache_dirs() -> list[Path]:
    """Hugging Face caches searched for GGUF language models: ours, then the user's own ($HF_HUB_CACHE, $HF_HOME,
    ~/.cache/huggingface/hub)."""
    out = [hf_cache()]
    if os.environ.get("HF_HUB_CACHE"):
        out.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        out.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    out.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def tools_bin() -> Path:
    """Programs the installer put next to the app (a private ffmpeg) when the system had none."""
    return home() / "tools" / "bin"


def use_private_tools() -> None:
    """Put <home>/tools/bin first on PATH: the installer puts an ffmpeg there only when the system has none or one
    that cannot write H.264 (libx264), so what is there is meant to be used."""
    d = str(tools_bin())
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if tools_bin().is_dir() and d not in parts:
        os.environ["PATH"] = os.pathsep.join([d] + [p for p in parts if p])
