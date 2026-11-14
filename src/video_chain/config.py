"""Where things live on this machine: one place for the paths that used to be written into several modules. Every value
can be overridden by an environment variable; the defaults are the user's own folders, so nothing changes without one."""
from __future__ import annotations

import os
from pathlib import Path

_AI = "/path/to/ai"


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def hf_cache() -> Path:
    """The Hugging Face cache with the models (loaded offline). VIDEO_CHAIN_HF_CACHE."""
    return Path(_env("VIDEO_CHAIN_HF_CACHE", f"{_AI}/hf-models"))


def llama_server() -> str:
    """The llama.cpp server program. VIDEO_CHAIN_LLAMA_SERVER."""
    return _env("VIDEO_CHAIN_LLAMA_SERVER", f"{_AI}/llama.cpp/build/bin/llama-server")
