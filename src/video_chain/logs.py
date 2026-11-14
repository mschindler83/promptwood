"""Logging, set up once by the entry points (app.py, __main__.py). Modules only do `log = logging.getLogger(__name__)`."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def default_log_file() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "video-chain" / "video-chain.log"


def setup(file: Path | None = None, level: int = logging.INFO, to_file: bool = True) -> None:
    """stderr and a rotating log file (to_file=False: stderr only; file=None: the default one; a folder that cannot be written: stderr only).
    Calling it again changes nothing."""
    root = logging.getLogger()
    if any(getattr(h, "_video_chain", False) for h in root.handlers):
        return
    fmt = logging.Formatter(FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        if not to_file:
            raise OSError("no file wanted")
        path = Path(file) if file else default_log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=1, encoding="utf-8"))
    except OSError as e:
        if to_file:
            print(f"video-chain: no log file ({e})", file=sys.stderr)
    for h in handlers:
        h.setFormatter(fmt)
        h._video_chain = True          # noqa: SLF001  (marks ours: setup() is idempotent)
        root.addHandler(h)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
