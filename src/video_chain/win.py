"""Everything that is different on Windows, in one place. Nothing here runs on Linux or macOS: every caller asks
`if win.WINDOWS:` first (or the function returns at once), so the app behaves there exactly as without this file.

Status: written from Windows' documented behaviour and covered by tests that run on Linux with the Windows calls
replaced; it has not been run on a Windows computer yet.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200


def hide_consoles() -> None:
    """(the windowed app) Programs it runs in the background (ffmpeg, ffprobe, nvidia-smi …) would each flash a
    console window: they are started without one, unless a caller sets creation flags itself."""
    if not WINDOWS or getattr(subprocess.Popen, "_video_chain_quiet", False):
        return
    plain = subprocess.Popen.__init__

    def quiet(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = CREATE_NO_WINDOW
        plain(self, *args, **kwargs)
    subprocess.Popen.__init__ = quiet
    subprocess.Popen._video_chain_quiet = True


def ensure_utf8(module: str, argv: list[str] | None = None) -> int | None:
    """Windows reads and writes text files in an old code page unless Python runs in UTF-8 mode; the app's files
    (projects, prompts in any language) are UTF-8, as on Linux. Programs started from here inherit the mode. When
    this process is not in it, the same command is run again in UTF-8 mode: its exit code is returned (the caller
    ends with it); None: all is well, go on."""
    if not WINDOWS:
        return None
    os.environ["PYTHONUTF8"] = "1"
    if sys.flags.utf8_mode:
        return None
    args = sys.argv[1:] if argv is None else list(argv)
    return subprocess.call([sys.executable, "-X", "utf8", "-m", module, *args])


def python_args() -> list[str]:
    """Arguments that put a Python the app starts into UTF-8 mode (also one started isolated with -I)."""
    return ["-X", "utf8"] if WINDOWS else []


def popen_group() -> dict:
    """Popen arguments that give a background server its own process group (what start_new_session does on
    Linux), so it can be stopped with everything it started."""
    return {"creationflags": CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW} if WINDOWS else {}


def kill_tree(pid: int, force: bool = True) -> bool:
    """End a process and the processes it started (Linux: a signal to the process group)."""
    try:
        r = subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", *(["/F"] if force else [])],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def link_dir(target: Path, link: Path) -> None:
    """A folder link like the relative symlink used on Linux. Windows allows symlinks only with developer mode
    or as administrator; without, a junction is made (it holds the absolute path: after moving the whole project
    folder the app makes it anew when the project is opened or saved as a copy)."""
    target, link = Path(target), Path(link)
    try:
        os.symlink(os.path.relpath(target, link.parent), link, target_is_directory=True)
    except OSError:
        import _winapi
        _winapi.CreateJunction(str(target.resolve()), str(link))


def is_link(path: Path) -> bool:
    """A symlink or (Windows) a junction."""
    p = Path(path)
    return p.is_symlink() or (WINDOWS and os.path.isjunction(p))


def remove_link(path: Path) -> None:
    p = Path(path)
    try:
        p.unlink()
    except OSError:
        os.rmdir(p)                                   # (a junction is removed like an empty folder)


def data_dir() -> Path:
    """Where the app keeps its own small files on Windows (%LOCALAPPDATA%\\video-chain); Linux keeps its XDG
    folders and never asks."""
    from . import paths
    return paths.home()


class LineReader:
    """Lines from a pipe with a timeout. Windows cannot wait on a pipe with select(), so a thread reads and hands
    the lines over. on_stop: called (in that thread) as soon as a {"cmd": "stop"} line arrives, also while a job
    is running: Windows has no signal the app could send to interrupt its worker."""

    def __init__(self, stream, on_stop=None):
        self.lines: queue.Queue = queue.Queue()
        self.on_stop = on_stop
        self._next: str | None = None
        raw = getattr(stream, "buffer", stream)
        threading.Thread(target=self._read, args=(raw,), daemon=True).start()

    def _read(self, raw):
        try:
            for line in iter(raw.readline, b""):
                text = line.decode(errors="replace") if isinstance(line, bytes) else line
                if self.on_stop is not None and '"stop"' in text and text.replace(" ", "").startswith('{"cmd":"stop"'):
                    self.on_stop()
                    continue
                self.lines.put(text if text.endswith("\n") else text + "\n")
        except (OSError, ValueError):
            pass
        self.lines.put("")                            # end of input

    def ready(self, timeout: float) -> bool:
        if self._next is None:
            try:
                self._next = self.lines.get(timeout=timeout)
            except queue.Empty:
                return False
        return True

    def readline(self) -> str:
        if self._next is None:
            self._next = self.lines.get()
        line, self._next = self._next, None
        return line
