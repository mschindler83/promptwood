"""Where things live on this machine: one place for the paths that used to be written into several modules. Every value
can be overridden by an environment variable; the defaults are the user's own folders, so nothing changes without one."""
from __future__ import annotations

import os
from pathlib import Path

_AI = "/path/to/ai"


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _xdg(var: str, fallback: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / fallback) / "video-chain"


def data_dir() -> Path:
    """The app's own data (LoRA presets and trigger words): $XDG_DATA_HOME/video-chain."""
    return _xdg("XDG_DATA_HOME", ".local/share")


def state_dir() -> Path:
    """What the app keeps about its runs (the log): $XDG_STATE_HOME/video-chain."""
    return _xdg("XDG_STATE_HOME", ".local/state")


def config_dir() -> Path:
    """Settings files (the RunPod key): $XDG_CONFIG_HOME/video-chain."""
    return _xdg("XDG_CONFIG_HOME", ".config")


def hf_cache() -> Path:
    """The Hugging Face cache with the models (loaded offline). VIDEO_CHAIN_HF_CACHE."""
    return Path(_env("VIDEO_CHAIN_HF_CACHE", f"{_AI}/hf-models"))


def llama_server() -> str:
    """The llama.cpp server program. VIDEO_CHAIN_LLAMA_SERVER."""
    return _env("VIDEO_CHAIN_LLAMA_SERVER", f"{_AI}/llama.cpp/build/bin/llama-server")
