"""Where the app keeps its own files and finds its tools: the folders of this computer's user (XDG on Linux,
%LOCALAPPDATA%\\video-chain on Windows) and what the installer set up (paths.py: <home>/config.json, environment
variables). Nothing here is tied to one person's machine."""
from __future__ import annotations

import os
from pathlib import Path

from . import paths


def _xdg(var: str, fallback: str) -> Path:
    if paths.WINDOWS:                               # (one folder there: %LOCALAPPDATA%\\video-chain)
        return paths.home()
    return Path(os.environ.get(var) or Path.home() / fallback) / "video-chain"


def cache_dir() -> Path:
    """Logs and caches (the llama.cpp log, thumbnails): $XDG_CACHE_HOME/video-chain."""
    return paths.cache_dir()


def data_dir() -> Path:
    """The app's own data (LoRA presets and trigger words, the libraries): $XDG_DATA_HOME/video-chain."""
    return _xdg("XDG_DATA_HOME", ".local/share")


def state_dir() -> Path:
    """What the app keeps about its runs (the log): $XDG_STATE_HOME/video-chain."""
    return _xdg("XDG_STATE_HOME", ".local/state") / ("state" if paths.WINDOWS else "")


def config_dir() -> Path:
    """Settings files (the RunPod key): $XDG_CONFIG_HOME/video-chain."""
    return _xdg("XDG_CONFIG_HOME", ".config") / ("config" if paths.WINDOWS else "")


def hf_cache() -> Path:
    """The Hugging Face cache with the models (loaded offline): VIDEO_CHAIN_HF_CACHE, the installer's choice, else
    <home>/hf-models (paths.hf_cache)."""
    return paths.hf_cache()


def llama_server() -> str:
    """The llama.cpp server program: VIDEO_CHAIN_LLAMA_SERVER, the installed one, else llama-server on PATH."""
    return paths.llama_server()
