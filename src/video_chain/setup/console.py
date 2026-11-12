"""What setup prints: short lines for the person, everything for the log file.

The screen gets the steps ("[3/7] ComfyUI"), one line per piece of work with the time it took, and questions. The
output of the tools setup runs (uv, pip) goes to the log file only; when one fails, its last lines are shown.
Standard library only.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import paths

_log = None
_log_path: Path | None = None
_steps = {"n": 0, "total": 0}


class CommandFailed(RuntimeError):
    pass


def open_log() -> Path:
    """Start this run's log file (<home>/logs/setup-<date>-<time>.log); the 10 newest are kept."""
    global _log, _log_path
    if _log is not None:
        return _log_path
    d = paths.home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    for old in sorted(d.glob("setup-*.log"))[:-9]:
        old.unlink(missing_ok=True)
    _log_path = d / time.strftime("setup-%Y%m%d-%H%M%S.log")
    _log = open(_log_path, "a", encoding="utf-8", buffering=1)      # noqa: SIM115
    return _log_path


def log_path() -> Path | None:
    return _log_path


def log_only(text: str) -> None:
    if _log is not None:
        _log.write(text if text.endswith("\n") else text + "\n")


def short(text) -> str:
    """Paths as a person writes them: the home folder as ~."""
    home = str(Path.home())
    return str(text).replace(home, "~") if len(home) > 1 else str(text)


def say(text: str = "") -> None:
    print(text, flush=True)
    log_only(text)


def plan_steps(total: int) -> None:
    _steps["n"], _steps["total"] = 0, int(total)


def step(text: str) -> None:
    _steps["n"] += 1
    head = f"[{_steps['n']}/{_steps['total']}]" if _steps["total"] >= _steps["n"] else "==>"
    say(f"\n{head} {text}")


def ask(question: str, default: bool = True) -> bool:
    try:
        a = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    except EOFError:
        return default
    log_only(f"{question} -> {a or '(default)'}")
    return default if not a else a in ("y", "yes", "j", "ja")


def ask_text(question: str, default: str = "") -> str:
    try:
        a = input(f"{question}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        return default
    log_only(f"{question} -> {a or '(default)'}")
    return a or default


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class working:
    """Shows that something long is still going on: in a terminal one line that counts the time (with the tool's
    last line of output, when note() is fed); otherwise a line when it starts, one a minute, one when it ends."""

    def __init__(self, what: str, expect: str = "", quiet_end: bool = False):
        self.what, self.expect, self.quiet_end = what, expect, quiet_end
        self.tty = sys.stdout.isatty()
        self.last = ""
        self.start = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def note(self, line: str) -> None:
        self.last = line.strip()

    def _line(self) -> str:
        text = f"    {self.what} … {clock(time.monotonic() - self.start)}"
        if self.last:
            width = max(20, (os.get_terminal_size().columns if self.tty else 100) - len(text) - 4)
            text += "  " + self.last[:width]
        return text

    def _tick(self):
        every = 1.0 if self.tty else 60.0
        while not self._stop.wait(every):
            if self.tty:
                sys.stdout.write("\r\033[K" + self._line())
            else:
                sys.stdout.write(f"    … still at it ({clock(time.monotonic() - self.start)})\n")
            sys.stdout.flush()

    def __enter__(self):
        self.start = time.monotonic()
        hint = f" ({self.expect})" if self.expect else ""
        log_only(f"    {self.what} …{hint}")
        if self.tty:
            sys.stdout.write(f"    {self.what} …{hint}")
        else:
            sys.stdout.write(f"    {self.what} …{hint}\n")
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, kind, *rest):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3)
        took = clock(time.monotonic() - self.start)
        end = f"    {self.what}: {'failed after' if kind else 'done in'} {took}"
        log_only(end)
        if self.tty:
            sys.stdout.write("\r\033[K" + ("" if self.quiet_end and not kind else end + "\n"))
        elif not (self.quiet_end and not kind):
            sys.stdout.write(end + "\n")
        sys.stdout.flush()
        return False


def run(cmd: list, what: str, env: dict | None = None, cwd: Path | None = None, expect: str = "") -> None:
    """Run a tool quietly: its output goes to the log, the screen shows `what` and the time. CommandFailed carries
    the tool's last lines when it ends with an error."""
    cmd = [str(c) for c in cmd]
    log_only("    $ " + " ".join(cmd))
    tail: list[str] = []
    with working(what, expect) as w:
        try:
            p = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, errors="replace")
        except OSError as e:
            raise CommandFailed(f"{what}: {e}") from e
        for line in p.stdout:
            log_only("      " + line.rstrip())
            if line.strip():
                tail.append(line.rstrip())
                del tail[:-25]
                w.note(line)
        code = p.wait()
        if code != 0:
            raise CommandFailed(f"{what} failed (exit code {code}). Its last lines:\n      "
                                + "\n      ".join(tail[-25:]))
