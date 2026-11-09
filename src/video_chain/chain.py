"""Chained image-to-video: prompt 1 starts from the reference image, every next prompt from the last frame of the
clip before. Writes every clip on its own and all of them joined into one video.

This module knows nothing about the model: a generator object makes one clip from (prompt, start image, seed) and
saves the clip's last frame as a lossless PNG. The MiniMax-H3 generator (through ComfyUI) lives in comfy.py.

Output folder:
  start.png          the reference image as given (converted to PNG)
  01.mp4 01.png      clip 1 and its last frame (the start image of clip 2)
  02.mp4 02.png      ...
  chain.mp4          all clips joined; the first frame of clips 2..n is left out, because it repeats the last
                     frame of the clip before (the model starts from exactly that image)
  chain.json         prompts, seeds, settings and inputs of every clip (also used to resume)
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

FPS = 24
HYPERFLOW_STEPS = 8      # HyperFlow's trained sigma grid
PIKA_STEPS = 6           # Pika H3 Flex (its template's 6-step sigmas)
# Veda sparse attention (veda-sparse-attention node) pays off only for large clips: measured on an RTX 4090 it made a
# 5 s clip at 672x384 (37k tokens) 25-30 % slower and a 10 s clip at 1344x768 (290k tokens) 2.3x faster; the
# predictor's own RTX 4090 numbers give 1.77x at 1344x768 for 5 s (149k). So it is used from this many tokens on.
VEDA_MIN_TOKENS = 120_000


def h3_tokens(width: int, height: int, frames: int) -> int:
    """The video tokens H3 attends over for a clip (latent frames x 16-px patches), as ComfyUI counts them."""
    f = int(frames)
    while f > 5 and f % 17 != 5:                # (off the 17n+5 grid: rounded up, as ComfyUI does)
        f += 1
    latent = 2 if f <= 5 else ((f - 5) // 17) * 5 + 2
    return latent * (int(width) // 16) * (int(height) // 16)


def veda_useful(width: int, height: int, frames: int) -> bool:
    """Veda makes a clip of this size faster (see VEDA_MIN_TOKENS)."""
    return h3_tokens(width, height, frames) >= VEDA_MIN_TOKENS


def veda_from_seconds(width: int, height: int) -> float | None:
    """The clip length (s) from which Veda is used at this size (None: not even at 15 s)."""
    for frames in range(5, 15 * 24 + 3, 17):
        if veda_useful(width, height, frames):
            return frames / 24
    return None
DIVISIBILITY = 32
MAX_RATIO = 2.5          # H3 custom sizes go from 2:5 to 5:2


class Generator(Protocol):
    def generate(self, prompt: str, start_image: Path, out_video: Path, out_last_frame: Path, seed: int,
                 settings: dict | None = None) -> None:
        """Make one clip that starts at start_image; write it to out_video and its last frame to out_last_frame.
        settings (chain.settings_dict) carries per-clip lora, strength, frames and steps; None: the generator's own."""


# ----------------------------------------------------------------------------- input

def read_prompts(path: Path) -> list[str]:
    """One prompt per line; empty lines and lines starting with # are skipped."""
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            prompts.append(s)
    return prompts


def canvas_size(width: int, height: int, resolution: int = 512, divisibility: int = DIVISIBILITY) -> tuple[int, int]:
    """Video size for a start image: about resolution² pixels, the image's aspect ratio (clamped to 2:5 … 5:2),
    both sides a multiple of 32. Same rule as ai-toolkit's training buckets, so the clips match the LoRA's
    training sizes (1:1 -> 512x512, 16:9 -> 672x384, 9:16 -> 384x672 at 512).

    H3 stretches the start image onto the canvas, so a canvas with the image's own ratio keeps it undistorted."""
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    ratio = min(MAX_RATIO, max(1 / MAX_RATIO, width / height))
    w = math.sqrt(resolution * resolution * ratio)
    h = w / ratio
    w = max(divisibility, int(w // divisibility * divisibility))
    h = max(divisibility, int(h // divisibility * divisibility))
    return w, h


def parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    try:
        size = int(w), int(h)
    except ValueError:
        raise ValueError(f"size must look like 512x512, got {text!r}") from None
    if min(size) < DIVISIBILITY or any(v % DIVISIBILITY for v in size):
        raise ValueError(f"both sides must be multiples of {DIVISIBILITY} (at least {DIVISIBILITY}), got {text}")
    return size


# ----------------------------------------------------------------------------- ffmpeg helpers

def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"{cmd[0]} failed: {tail}")
    return r.stdout


def probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-count_frames", "-print_format", "json", "-show_streams", str(path)])
    streams = json.loads(out).get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"{path} has no video stream")
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):     # "0/0" for some streams
        fps = 0.0
    return {"frames": int(v.get("nb_read_frames") or 0), "fps": fps,
            "width": int(v["width"]), "height": int(v["height"]),
            "audio": any(s.get("codec_type") == "audio" for s in streams)}


def last_frame(video: Path, png: Path) -> None:
    """The exact last frame of a video as PNG (decoded, so it went through the video codec once)."""
    n = probe(video)["frames"]
    if n < 1:
        raise RuntimeError(f"{video} has no frames")
    Path(png).unlink(missing_ok=True)              # an older PNG there must not pass as this frame
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{n - 1})", "-frames:v", "1",
          str(png)])
    if not Path(png).is_file():
        raise RuntimeError(f"could not read the last frame of {Path(video).name}")


def to_png(image: Path, png: Path, crop: tuple[int, int, int, int] | None = None) -> None:
    """Convert to PNG (crop: x, y, width, height in its pixels); written under a temporary name first, so a failed
    conversion never leaves a broken file."""
    png = Path(png)
    tmp = png.with_name(png.stem + ".part.png")
    vf = ["-vf", "crop={2}:{3}:{0}:{1}".format(*(int(v) for v in crop))] if crop else []
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(image), "-frames:v", "1", *vf, str(tmp)])
        tmp.replace(png)
    finally:
        tmp.unlink(missing_ok=True)


def audio_seconds(path: Path) -> float | None:
    """Length of a file's first audio stream in seconds (None: it has no sound)."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration:format=duration",
                "-print_format", "json", str(path)])
    info = json.loads(out or "{}")
    if not info.get("streams"):
        return None
    d = info["streams"][0].get("duration") or (info.get("format") or {}).get("duration")
    return float(d) if d not in (None, "N/A") else None


def to_wav(source: Path, wav: Path, start: float = 0.0, seconds: float | None = None, loop: float | None = None,
           offset: float = 0.0, pad: float = 0.0, fill: bool = False) -> None:
    """The sound of an audio or video file as 48 kHz stereo WAV (a part of it with start / seconds); written under
    a temporary name first. loop: only `loop` seconds from start are taken and repeated over and over, the WAV
    begins `offset` seconds into that loop. pad: seconds of silence first. fill: a sound that ends before `seconds` is
    filled up with silence."""
    wav = Path(wav)
    tmp = wav.with_name(wav.stem + ".part.wav")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{float(start):.3f}"]
    if loop:
        cmd += ["-t", f"{float(loop):.3f}"]
    cmd += ["-i", str(source), "-map", "0:a:0", "-vn"]
    if loop:
        cmd += ["-af", "aloop=loop=-1:size=2147483647"]
        if offset > 0:
            cmd += ["-ss", f"{float(offset):.3f}"]
    else:
        filters = [f"adelay={round(float(pad) * 1000)}:all=1"] if pad > 0 else []
        if fill and seconds is not None:
            filters.append("apad")
        if filters:
            cmd += ["-af", ",".join(filters)]
    if seconds is not None:
        cmd += ["-t", f"{float(seconds):.3f}"]
    try:
        _run(cmd + ["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(tmp)])
        tmp.replace(wav)
    finally:
        tmp.unlink(missing_ok=True)


def image_size(path: Path) -> tuple[int, int]:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0", str(path)])
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def probe_source(path: Path) -> dict:
    """Size, frame rate, duration and sound of any video file, without decoding it (fast, also on a share)."""
    out = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    data = json.loads(out)
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"{Path(path).name} has no video stream")

    def rate(text):
        num, _, den = (text or "0/1").partition("/")
        try:
            return float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            return 0.0
    fps = rate(v.get("avg_frame_rate")) or rate(v.get("r_frame_rate")) or float(FPS)
    try:
        duration = float(v.get("duration") or (data.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list") or []:            # phone videos: stored sideways + a rotation flag
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float((v.get("tags") or {}).get("rotate", 0) or 0))
    if rot % 180:
        w, h = h, w
    return {"width": w, "height": h, "fps": fps, "duration": duration,
            "audio": any(s.get("codec_type") == "audio" for s in streams)}


def range_frames(start: float, end: float, fps: int = FPS) -> int:
    """Frames of a time range at the chain's frame rate."""
    return max(0, round((float(end) - float(start)) * fps))


def convert_range(source: Path, out: Path, start: float, end: float, width: int, height: int,
                  fps: int = FPS) -> int:
    """Cut start..end (seconds) out of a real video and turn it into a chain clip: fps frames per second (frames
    dropped or repeated, no interpolation), scaled to fill width x height and cut to it at the centre, sound as
    48 kHz stereo (if the source has sound). Written under a temporary name first. Returns the frame count."""
    info = probe_source(source)
    if info["duration"] and float(end) > info["duration"]:
        end = info["duration"]                       # the sound is cut to the frames, not to the asked range
    n = range_frames(start, end, fps)
    if n < 1:
        raise ValueError("the range is shorter than one frame")
    out = Path(out)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    vf = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height},setsar=1,format=yuv420p")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{float(start):.6f}", "-i", str(source), "-map", "0:v:0",
           "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-crf", "12", "-preset", "slow"]
    if info["audio"]:
        cmd += ["-map", "0:a:0", "-af", f"aresample=48000,aformat=channel_layouts=stereo,apad,"
                                        f"atrim=0:{n / fps:.6f}", "-c:a", "aac", "-b:a", "256k"]
    cmd += ["-map_metadata", "-1", "-r", str(fps), str(tmp)]
    try:
        _run(cmd)
        made = probe(tmp)["frames"]
        if made < 1:
            raise RuntimeError(f"no frames in {Path(source).name} between {start:.2f} s and {end:.2f} s")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return made


def frame_at(video: Path, index: int, png: Path) -> None:
    """Frame number index (0-based) of a video as PNG."""
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{int(index)})", "-frames:v", "1",
          "-update", "1", str(png)])
    if not Path(png).is_file():
        raise RuntimeError(f"{Path(video).name} has no frame {index}")


def replace_audio(video: Path, wav: Path) -> None:
    """Put this sound into a clip instead of its own (the picture is copied, not re-encoded); the sound is cut or
    padded with silence to the clip's length. Written under a temporary name first."""
    video = Path(video)
    tmp = video.with_name(video.stem + ".part" + video.suffix)
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
              "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-b:a", "256k", "-shortest", str(tmp)])
        tmp.replace(video)
    finally:
        tmp.unlink(missing_ok=True)


def cut_frames(video: Path, out: Path, first: int, count: int, fps: int = FPS) -> bool:
    """Frames first .. first+count-1 of a chain clip (and their sound) as a clip of their own, nearly lossless.
    Returns whether it has sound."""
    has_audio = probe(video)["audio"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0",
           "-vf", f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS",
           "-c:v", "libx264", "-crf", "8", "-preset", "slow", "-pix_fmt", "yuv420p"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-af", f"atrim=start={first / fps:.6f}:end={(first + count) / fps:.6f},"
                                        "asetpts=PTS-STARTPTS", "-c:a", "aac", "-b:a", "256k"]
    out = Path(out)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        _run(cmd + ["-r", str(fps), str(tmp)])
        if probe(tmp)["frames"] != count:
            raise RuntimeError(f"could not cut {count} frames from {Path(video).name}")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return has_audio


def concat(videos: list[Path], out: Path, drop_first: bool = True, fps: int = FPS,
           cuts: list[tuple[int, int]] | None = None, size: tuple[int, int] | None = None) -> None:
    """Join clips into one video. With drop_first, clips 2..n lose their first frame (and its 1/24 s of sound),
    which repeats the last frame of the clip before. cuts instead gives (frames cut at the start, frames cut at
    the end) per clip. Clips without sound get silence, so sound stays in sync. size: fit every clip into this
    frame (aspect kept, black bars), for parts of different sizes (a film's scenes)."""
    if not videos:
        raise ValueError("nothing to join")
    infos = [probe(v) for v in videos]
    if cuts is None:
        cuts = [(1 if (drop_first and k > 0) else 0, 0) for k in range(len(videos))]
    any_audio = any(i["audio"] for i in infos)
    args: list[str] = []
    for v in videos:
        args += ["-i", str(v)]
    filters, labels = [], []
    extra = len(videos)
    for k, info in enumerate(infos):
        skip, cut_end = cuts[k]
        if info["frames"] - skip - cut_end < 1:              # never cut a clip away completely
            skip, cut_end = 0, 0
        stop = info["frames"] - cut_end
        fit = (f",scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease:flags=lanczos,"
               f"pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1") if size else ""
        filters.append(f"[{k}:v]trim=start_frame={skip}:end_frame={stop},setpts=PTS-STARTPTS,fps={fps}{fit}[v{k}]")
        labels.append(f"[v{k}]")
        if any_audio:
            start, end = skip / fps, stop / fps
            if info["audio"]:
                src = f"[{k}:a]"
            else:
                args += ["-f", "lavfi", "-t", f"{end:.6f}", "-i", "anullsrc=r=48000:cl=stereo"]
                src = f"[{extra}:a]"
                extra += 1
            # pad short sound with silence, then cut exactly to the clip's frames
            filters.append(f"{src}aresample=48000,aformat=channel_layouts=stereo,apad,"
                           f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{k}]")
            labels.append(f"[a{k}]")
    n = len(videos)
    filters.append("".join(labels) + f"concat=n={n}:v=1:a={1 if any_audio else 0}" +
                   ("[v][a]" if any_audio else "[v]"))
    maps = ["-map", "[v]"] + (["-map", "[a]"] if any_audio else [])
    codec = ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]
    codec += ["-c:a", "aac", "-b:a", "192k"] if any_audio else []
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        _run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", ";".join(filters), *maps, *codec,
              "-r", str(fps), str(tmp)])
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def join_parts(videos: list[Path], out: Path, size: tuple[int, int], joins: list[str], fade: float = 0.5,
               fps: int = FPS) -> None:
    """Join a film's parts (each scene joined already) in one frame size. joins[k]: how part k follows part k-1
    ("cut", "fade": a crossfade of `fade` seconds, picture and sound, "continue": its first frame repeats the end of
    the part before and is left out); joins[0] is ignored. Parts without sound get silence."""
    if not videos:
        raise ValueError("nothing to join")
    if not any(j == "fade" for j in joins[1:]):
        concat(videos, out, cuts=[(1 if k and joins[k] == "continue" else 0, 0) for k in range(len(videos))],
               size=size, fps=fps)
        return
    w, h = size
    infos = [probe(v) for v in videos]
    args: list[str] = []
    for v in videos:
        args += ["-i", str(v)]
    filters, extra, durs = [], len(videos), []
    for k, info in enumerate(infos):
        skip = 1 if k and joins[k] == "continue" and info["frames"] > 1 else 0
        n = info["frames"] - skip
        durs.append(n / fps)
        filters.append(f"[{k}:v]trim=start_frame={skip}:end_frame={info['frames']},setpts=PTS-STARTPTS,fps={fps},"
                       f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                       f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p,settb=AVTB[v{k}]")
        if info["audio"]:
            src = f"[{k}:a]"
        else:
            args += ["-f", "lavfi", "-t", f"{n / fps + 1:.6f}", "-i", "anullsrc=r=48000:cl=stereo"]
            src = f"[{extra}:a]"
            extra += 1
        filters.append(f"{src}aresample=48000,aformat=channel_layouts=stereo,apad,"
                       f"atrim=start={skip / fps:.6f}:end={info['frames'] / fps:.6f},asetpts=PTS-STARTPTS[a{k}]")
    cv, ca, t = "[v0]", "[a0]", durs[0]
    for k in range(1, len(videos)):
        nv, na = f"[cv{k}]", f"[ca{k}]"
        if joins[k] == "fade":
            d = max(0.04, min(fade, durs[k - 1] / 2, durs[k] / 2))
            filters.append(f"{cv}[v{k}]xfade=transition=fade:duration={d:.6f}:offset={t - d:.6f}{nv}")
            filters.append(f"{ca}[a{k}]acrossfade=d={d:.6f}{na}")
            t += durs[k] - d
        else:                                   # (concat output gets its frame rate again: a fade may follow)
            filters.append(f"{cv}{ca}[v{k}][a{k}]concat=n=2:v=1:a=1[cc{k}]{na}")
            filters.append(f"[cc{k}]fps={fps},settb=AVTB{nv}")
            t += durs[k]
        cv, ca = nv, na
    out = Path(out)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    try:
        _run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", ";".join(filters), "-map", cv, "-map", ca,
              "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac",
              "-b:a", "192k", "-r", str(fps), str(tmp)])
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


_SHA_CACHE: dict[tuple, str] = {}


def file_signature(path: str | Path | None) -> str | None:
    """SHA-1 of a file's content, cached by (path, size, mtime); None for no file or a missing one."""
    if not path:
        return None
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha1_file(Path(path))
    return _SHA_CACHE[key]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- the chain

@dataclass
class Clip:
    num: int
    prompt: str
    seed: int
    start_image: str          # file name in the output folder
    start_sha1: str
    video: str
    last_frame: str
    settings: dict = field(default_factory=dict)
    reused: bool = False


def seeds_for(n: int, seed: int, same_seed: bool) -> list[int]:
    return [seed] * n if same_seed else [seed + k for k in range(n)]


def _load_previous(out_dir: Path) -> dict[int, dict]:
    p = out_dir / "chain.json"
    if not p.exists():
        return {}
    try:
        return {c["num"]: c for c in json.loads(p.read_text(encoding="utf-8")).get("clips", [])}
    except (ValueError, KeyError, TypeError):
        return {}


START_PIN = 5        # frames the start image is held for in a clip with faces (the reference model cuts away from
                     # a start frame anchored on one frame only when the story does not fit it)


def settings_dict(*, model: str, lora: str | None, strength: float, frames: int, steps: int, width: int,
                  height: int, audio: bool, backend: str = "comfyui", refs: list | None = None,
                  ref_size: str = "match", ref_hint: bool = True, extra_loras: list | None = None,
                  turbo: tuple[str, float] | None = None, guide: int = 0,
                  style: tuple[str, str] | None = None, hyperflow: float | None = None,
                  pika: float | None = None, audio_refs: list | None = None, cast: list | None = None,
                  voice_language: str | None = None, lora_triggers: list | None = None,
                  source_sound: dict | None = None, veda: bool = False) -> dict:
    """What a clip depends on besides prompt, seed and start image; a clip is reused only if this is unchanged.
    The LoRA counts by content (lora_sha1), so a LoRA retrained into the same file makes its clips change.
    extra_loras: further LoRAs stacked after the first ([(path, strength)]), only present when there are any, so
    a single-LoRA setting is recorded exactly as before.
    refs: the face pictures this clip is made with ([{"file": path in the project, "sha1": ...}]), only present
    when it uses any; such a clip holds its start image for START_PIN frames (start_pin). cast: with @mentions, the characters the model prompt describes ([{"name", "desc"}]).
    voice_language: a character speaks with its voice (MiniMax's structured format, lines tagged with this
    language); such a clip holds its start image like one with faces.
    style: the project's text before / after every prompt (only present when there is any)."""
    s = {"model": model, "lora": lora or None, "lora_sha1": file_signature(lora) if lora else None,
         "strength": float(strength) if lora else 1.0,
         "frames": int(frames), "steps": int(steps), "width": int(width), "height": int(height),
         "audio": bool(audio), "backend": backend}
    if hyperflow is not None:                   # HyperFlow at this strength; its own 8-step sigma grid
        s["hyperflow"] = {"strength": float(hyperflow)}
    if pika is not None:                        # Pika H3 Flex LoRA at this strength; lcm on its 6 sigmas
        s["pika"] = {"strength": float(pika)}
    if turbo:                                   # (turbo LoRA, strength); steps are already the turbo steps
        s["turbo"] = {"lora": turbo[0], "sha1": file_signature(turbo[0]), "strength": float(turbo[1])}
    if extra_loras:
        s["extra_loras"] = [{"lora": p, "sha1": file_signature(p), "strength": float(st)} for p, st in extra_loras]
    if audio_refs:                              # sound references (<Audio 1> ...) this clip's prompt names
        s["audio_refs"] = [{"file": a["file"], "sha1": a["sha1"]} for a in audio_refs]
    if refs:
        s.update({"refs": [{"file": r["file"], "sha1": r["sha1"]} for r in refs], "ref_size": ref_size,
                  "ref_hint": bool(ref_hint), "start_pin": START_PIN})
    if (refs or voice_language) and cast:
        s["cast"] = [{"name": c["name"], "desc": c.get("desc", "")} for c in cast]
    if voice_language:
        s["voice_language"] = voice_language
        s["start_pin"] = START_PIN
    if source_sound:                            # the sound of the real video after its range, as the soundtrack
        src = Path(source_sound["file"]) if source_sound.get("file") else None
        try:                                    # (a clip of the project: its name is unique per version)
            st = src.stat() if src is not None and src.is_absolute() else None
            key = [st.st_size, st.st_mtime_ns] if st is not None else None
        except OSError:
            key = None
        s["source_sound"] = {"file": str(src) if src is not None else "", "start": float(source_sound["start"]),
                             "secs": float(source_sound["secs"]), "file_key": key}
        if source_sound.get("mode") == "previous":     # the clip before's sound, repeated (loop / offset)
            s["source_sound"].update({"mode": "previous", "loop": float(source_sound["loop"]),
                                      "offset": float(source_sound.get("offset") or 0.0)})
        elif source_sound.get("mode") == "track":      # an audio track of the project, carried on along the path
            s["source_sound"].update({"mode": "track", "track": str(source_sound["track"]),
                                      "pad": float(source_sound.get("pad") or 0.0)})
        if source_sound.get("overlay"):                # laid over the made clip; the model never hears it
            s["source_sound"]["overlay"] = True
    if lora_triggers:                           # the LoRAs' trigger words put before the prompt
        s["lora_triggers"] = [str(w) for w in lora_triggers]
    if veda:                                    # made with Veda sparse attention (a large clip, see veda_useful)
        s["veda"] = True
    if guide:                                   # motion guide: frames of the neighbouring clips anchored
        s["guide"] = int(guide)
    prefix, suffix = ((style[0] or "").strip(), (style[1] or "").strip()) if style else ("", "")
    if prefix:
        s["prompt_prefix"] = prefix
    if suffix:
        s["prompt_suffix"] = suffix
    return s


def _write_record(out_dir: Path, settings: dict, clips: list[Clip], done: bool, image: Path | None = None) -> None:
    data = {"tool": "video-chain", "image": str(image) if image else None, "settings": settings, "done": done,
            "chain": "chain.mp4" if done else None, "clips": [asdict(c) for c in clips]}
    tmp = out_dir / "chain.json.part"
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_dir / "chain.json")


def run_chain(generator: Generator, image: Path, prompts: list[str], out_dir: Path, *, seed: int = 42,
              same_seed: bool = False, settings: dict | None = None, resume: bool = True,
              drop_first: bool = True, log: Callable[[str], None] = print,
              events: Callable[[dict], None] | None = None) -> Path:
    """Make the clips one after another and join them. With resume, a clip whose prompt, seed, start image and
    settings are unchanged since the last run (and whose files exist) is reused instead of generated again."""
    if not prompts:
        raise ValueError("no prompts")
    image = Path(image)
    if not image.is_file():
        raise FileNotFoundError(image)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(settings or {})
    emit = events or (lambda _e: None)
    previous = _load_previous(out_dir) if resume else {}

    start = out_dir / "start.png"
    to_png(image, start)
    seeds = seeds_for(len(prompts), seed, same_seed)
    width = len(str(len(prompts)))
    clips: list[Clip] = []
    current = start
    for k, (prompt, s) in enumerate(zip(prompts, seeds), start=1):
        name = f"{k:0{max(2, width)}d}"
        video, last = out_dir / f"{name}.mp4", out_dir / f"{name}.png"
        clip = Clip(k, prompt, s, current.name, sha1_file(current), video.name, last.name, settings)
        old = previous.get(k)
        same = (old is not None and video.exists() and last.exists()
                and {key: old.get(key) for key in ("prompt", "seed", "start_sha1", "settings")}
                == {"prompt": prompt, "seed": s, "start_sha1": clip.start_sha1, "settings": settings})
        if same:
            clip.reused = True
            log(f"[{k}/{len(prompts)}] unchanged, reusing {video.name}")
            emit({"e": "clip", "k": k, "n": len(prompts), "state": "reused", "video": str(video), "frame": str(last)})
        else:
            log(f"[{k}/{len(prompts)}] generating {video.name} (seed {s}) from {current.name}")
            emit({"e": "clip", "k": k, "n": len(prompts), "state": "generating"})
            new_video, new_last = video.with_name(video.stem + ".new.mp4"), last.with_name(last.stem + ".new.png")
            for f in (new_video, new_last):
                f.unlink(missing_ok=True)
            generator.generate(prompt, current, new_video, new_last, s, settings=settings or None)
            if not new_video.exists() or not new_last.exists():
                raise RuntimeError(f"the generator did not write {video.name} and {last.name}")
            new_video.replace(video)              # the old clip is replaced only once the new one exists
            new_last.replace(last)
            emit({"e": "clip", "k": k, "n": len(prompts), "state": "done", "video": str(video), "frame": str(last)})
        clips.append(clip)
        _write_record(out_dir, settings, clips, done=False, image=image.resolve())
        current = last

    for old_num in sorted(n for n in previous if n > len(prompts)):      # clips of a longer earlier chain
        for key in ("video", "last_frame"):
            if previous[old_num].get(key):
                (out_dir / previous[old_num][key]).unlink(missing_ok=True)

    chain = out_dir / "chain.mp4"
    log(f"joining {len(clips)} clips into {chain.name}")
    emit({"e": "joining"})
    concat([out_dir / c.video for c in clips], chain, drop_first=drop_first)
    _write_record(out_dir, settings, clips, done=True, image=image.resolve())
    emit({"e": "done", "chain": str(chain)})
    return chain


def copy_prompts(prompts_file: Path, out_dir: Path) -> None:
    target = Path(out_dir) / "prompts.txt"
    if target.exists() and target.resolve() == Path(prompts_file).resolve():
        return
    shutil.copyfile(prompts_file, target)


def load_record(out_dir: Path) -> dict | None:
    """chain.json of an earlier run, or None."""
    p = Path(out_dir) / "chain.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
