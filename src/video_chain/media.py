"""The one place that runs ffmpeg / ffprobe, probes media files and writes files safely (no Qt, stdlib only).

Every failure is a MediaError (a RuntimeError) with a plain message: a missing binary says which one, a failed run
carries the last lines of the tool's own message, a run that does not finish in time says so. Callers that want
their own error type catch MediaError and re-raise (comfy.py does: ComfyError).

Writes: atomic_write_* put a small file down whole or not at all; part_file(out) is for a tool that writes a big
file (ffmpeg): it writes next to out under a ".part" name and renames it when the block ends without an error.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

FFMPEG_TIMEOUT = 3600           # seconds an ffmpeg job on a clip may take
FFPROBE_TIMEOUT = 120           # seconds ffprobe may take (it reads the container, not the frames)
DEFAULT_FPS = 24.0


class MediaError(RuntimeError):
    """ffmpeg / ffprobe is missing, failed or did not finish."""


def tail(text: str | bytes | None, lines: int = 5, chars: int = 600) -> str:
    """The last lines of a tool's message (its reason is at the end)."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return "\n".join((text or "").strip().splitlines()[-lines:])[-chars:]


def _missing(tool: str) -> MediaError:
    return MediaError(f"{tool} is not installed (it comes with ffmpeg)" if tool == "ffprobe"
                      else f"{tool} is not installed (it is needed to read, prepare and join videos)")


def require_ffmpeg() -> None:
    """MediaError naming what is missing when ffmpeg or ffprobe is not on the PATH (checked once at the start)."""
    gone = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if gone:
        raise MediaError(" and ".join(gone) + (" are" if len(gone) > 1 else " is")
                         + " not installed (ffmpeg is needed to read, prepare and join videos)")


def run(cmd: list[str], *, timeout: float | None = None, text: bool = True) -> str | bytes:
    """Run a command, return its stdout (text or bytes). MediaError: the binary is missing, it exits non-zero
    (the message ends with its last stderr lines) or it does not finish within timeout seconds."""
    name = Path(str(cmd[0])).name
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                           text=text, errors="replace" if text else None)
    except FileNotFoundError as e:
        raise _missing(name) from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{name} did not finish within {timeout / 60:.0f} min" if timeout and timeout >= 120
                         else f"{name} did not finish within {timeout:g} s") from e
    except OSError as e:
        raise MediaError(f"{name} could not be started: {e}") from e
    if r.returncode != 0:
        raise MediaError(f"{name} failed: {tail(r.stderr) or 'exit code %d' % r.returncode}")
    return r.stdout


def ffmpeg(args: list, *, timeout: float | None = FFMPEG_TIMEOUT, text: bool = True) -> str | bytes:
    """ffmpeg without stdin, quiet (-nostdin -v error), with a time limit. Pass -y yourself to overwrite."""
    return run(["ffmpeg", "-nostdin", "-v", "error", *map(str, args)], timeout=timeout, text=text)


def ffprobe(args: list, *, timeout: float = FFPROBE_TIMEOUT) -> str:
    return run(["ffprobe", "-v", "error", *map(str, args)], timeout=timeout)


def ffprobe_json(path: str | Path, *args: str, timeout: float = FFPROBE_TIMEOUT) -> dict:
    """ffprobe's JSON for a file (args: -show_streams, -show_format ... as for ffprobe)."""
    out = ffprobe([*args, "-print_format", "json", str(path)], timeout=timeout)
    try:
        return json.loads(out or "{}")
    except ValueError as e:
        raise MediaError(f"ffprobe gave no usable answer for {Path(str(path)).name}") from e


def popen(cmd: list[str], **kwargs) -> subprocess.Popen:
    """subprocess.Popen whose missing binary is a MediaError (for streaming readers and writers)."""
    try:
        return subprocess.Popen([str(c) for c in cmd], **kwargs)
    except FileNotFoundError as e:
        raise _missing(Path(str(cmd[0])).name) from e
    except OSError as e:
        raise MediaError(f"{Path(str(cmd[0])).name} could not be started: {e}") from e


# ----------------------------------------------------------------------------- probing

@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    frames: int             # 0: unknown (probe(count_frames=True) counts them: exact, but decodes the file)
    fps: float
    duration: float         # seconds, 0.0: unknown
    has_audio: bool

    def as_dict(self) -> dict:
        """The old dict shape ("audio" for has_audio)."""
        d = asdict(self)
        d["audio"] = d.pop("has_audio")
        return d


def _rate(text) -> float:
    num, _, den = (text or "0/1").partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):     # "0/0" for some streams
        return 0.0


def probe(path: str | Path, *, count_frames: bool = False) -> MediaInfo:
    """Size (as the player shows it: a phone video's rotation flag is applied), frames, frame rate, duration and
    sound of a video. count_frames decodes the whole file to count its frames (exact); otherwise the container's
    own count is used (0 when it does not say). MediaError without a video stream."""
    data = ffprobe_json(path, *(["-count_frames"] if count_frames else []), "-show_streams", "-show_format")
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None or "width" not in v or "height" not in v:
        raise MediaError(f"{Path(str(path)).name} has no video stream")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list") or []:            # phone videos: stored sideways + a rotation flag
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    try:
        rot = rot or int(float((v.get("tags") or {}).get("rotate", 0) or 0))
    except ValueError:
        pass
    if rot % 180:
        w, h = h, w
    try:
        frames = int(v.get("nb_read_frames" if count_frames else "nb_frames") or 0)
    except ValueError:
        frames = 0
    try:
        duration = float(v.get("duration") or (data.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
    return MediaInfo(w, h, frames, fps, duration, any(s.get("codec_type") == "audio" for s in streams))


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) of a picture or of a video's first stream."""
    data = ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "stream=width,height")
    s = (data.get("streams") or [{}])[0]
    if "width" not in s or "height" not in s:
        raise MediaError(f"{Path(str(path)).name} has no picture in it")
    return int(s["width"]), int(s["height"])


def audio_seconds(path: str | Path) -> float | None:
    """Length of a file's first audio stream in seconds (None: it has no sound)."""
    info = ffprobe_json(path, "-select_streams", "a:0", "-show_entries", "stream=duration:format=duration")
    if not info.get("streams"):
        return None
    d = info["streams"][0].get("duration") or (info.get("format") or {}).get("duration")
    return float(d) if d not in (None, "N/A") else None


# ----------------------------------------------------------------------------- frames

def iter_frames(path: str | Path, size: tuple[int, int], pix_fmt: str = "rgb24", *, flags: str | None = "area",
                passthrough: bool = False, cancelled: Callable[[], bool] | None = None,
                on_start: Callable[[subprocess.Popen], None] | None = None,
                timeout: float | None = None) -> Iterator[bytes]:
    """The frames of a video, one by one, scaled to size, as raw bytes of pix_fmt ("rgb24", "gray": width x height
    x bytes per pixel each). The reader is streamed (little memory) and killed when the iterator is closed,
    cancelled() turns true (the iteration just ends) or timeout seconds have passed. passthrough: every frame
    exactly as stored (no repeats or drops). on_start gets the ffmpeg process (so a closing dialog can end it).
    MediaError when ffmpeg cannot start or exits with an error."""
    w, h = size
    per = w * h * {"rgb24": 3, "gray": 1}.get(pix_fmt, 3)
    vf = f"scale={w}:{h}" + (f":flags={flags}" if flags else "") + (",format=gray" if pix_fmt == "gray" else "")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0"]
    if passthrough:
        cmd += ["-fps_mode", "passthrough"]
    cmd += ["-vf", vf, "-f", "rawvideo", *(["-pix_fmt", pix_fmt] if pix_fmt != "gray" else []), "-"]
    err = tempfile.TemporaryFile()              # (a file, not a pipe: a chatty ffmpeg cannot fill it up)
    try:
        proc = popen(cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL)
    except BaseException:
        err.close()
        raise
    timed_out = threading.Event()

    def too_long():
        timed_out.set()
        proc.kill()
    timer = threading.Timer(timeout, too_long) if timeout else None
    if timer:
        timer.daemon = True
        timer.start()
    stopped = False
    try:
        if on_start is not None:
            on_start(proc)
        while True:
            if cancelled is not None and cancelled():
                stopped = True
                return
            buf = proc.stdout.read(per)
            if len(buf) < per:
                break
            yield buf
    finally:
        if timer:
            timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        code = proc.wait()
        err.seek(0)
        msg = tail(err.read(65536))
        err.close()
    if timed_out.is_set():
        raise MediaError(f"ffmpeg did not finish reading {Path(str(path)).name} within {timeout:g} s")
    if code != 0 and not stopped:
        raise MediaError(f"cannot read {Path(str(path)).name}: {msg or 'ffmpeg failed'}")


# ----------------------------------------------------------------------------- safe writes

def _sync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write a file whole or not at all: to a name of its own next to it, on disk, then renamed over the old file.
    A crash or an error leaves the old file (or none), never a cut-off one, and no stray part file."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, data, *, indent: int | None = 2, **dump) -> None:
    """json.dumps first (a value that cannot be written changes nothing), then atomic_write_text."""
    atomic_write_text(path, json.dumps(data, indent=indent, ensure_ascii=dump.pop("ensure_ascii", False), **dump))


@contextlib.contextmanager
def part_file(out: str | Path) -> Iterator[Path]:
    """`with part_file(out) as tmp:` write the file to tmp (next to out, "name.part.ext"); when the block ends
    without an error and tmp exists, it is renamed to out. Otherwise it is removed and out stays as it was."""
    out = Path(out)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        yield tmp
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def file_digest(path: str | Path, algo: str = "sha256", stop: Callable[[], bool] | None = None,
                chunk: int = 1 << 20) -> str | None:
    """Hex digest of a file's content (streamed). OSError when it cannot be read; None when stop() turned true."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            if stop is not None and stop():
                return None
            h.update(block)
    return h.hexdigest()


def text_digest(text: str, algo: str = "sha256") -> str:
    """Hex digest of a text (utf-8)."""
    return hashlib.new(algo, text.encode("utf-8")).hexdigest()


def scratch(name: str, dir: str | Path | None = None) -> tempfile.TemporaryDirectory:
    """A temporary folder named video-chain-<name>-... (one prefix style everywhere); use it as a context manager
    or call cleanup()."""
    return tempfile.TemporaryDirectory(prefix=f"video-chain-{name}-", dir=str(dir) if dir else None)
