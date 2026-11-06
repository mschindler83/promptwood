"""Chained image-to-video: prompt 1 starts from the reference image, every next prompt from the last frame of the
clip before. Writes every clip on its own and all of them joined into one video.

This module knows nothing about the model: a generator object makes one clip from (prompt, start image, seed) and
saves the clip's last frame as a lossless PNG. The MiniMax-H3 generator lives in h3.py.

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
    return {"frames": int(v.get("nb_read_frames") or 0), "fps": float(num) / float(den or 1),
            "width": int(v["width"]), "height": int(v["height"]),
            "audio": any(s.get("codec_type") == "audio" for s in streams)}


def last_frame(video: Path, png: Path) -> None:
    """The exact last frame of a video as PNG (decoded, so it went through the video codec once)."""
    n = probe(video)["frames"]
    if n < 1:
        raise RuntimeError(f"{video} has no frames")
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{n - 1})", "-frames:v", "1",
          str(png)])


def to_png(image: Path, png: Path) -> None:
    """Convert to PNG; written under a temporary name first, so a failed conversion never leaves a broken file."""
    png = Path(png)
    tmp = png.with_name(png.stem + ".part.png")
    try:
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(image), "-frames:v", "1", str(tmp)])
        tmp.replace(png)
    finally:
        tmp.unlink(missing_ok=True)


def image_size(path: Path) -> tuple[int, int]:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0", str(path)])
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def concat(videos: list[Path], out: Path, drop_first: bool = True, fps: int = FPS) -> None:
    """Join clips into one video. With drop_first, clips 2..n lose their first frame (and its 1/24 s of sound),
    which repeats the last frame of the clip before. Clips without sound get silence, so sound stays in sync."""
    if not videos:
        raise ValueError("nothing to join")
    infos = [probe(v) for v in videos]
    any_audio = any(i["audio"] for i in infos)
    args: list[str] = []
    for v in videos:
        args += ["-i", str(v)]
    filters, labels = [], []
    extra = len(videos)
    for k, info in enumerate(infos):
        skip = 1 if (drop_first and k > 0 and info["frames"] > 1) else 0
        filters.append(f"[{k}:v]trim=start_frame={skip},setpts=PTS-STARTPTS,fps={fps}[v{k}]")
        labels.append(f"[v{k}]")
        if any_audio:
            start, end = skip / fps, info["frames"] / fps
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
    _run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", ";".join(filters), *maps, *codec,
          "-r", str(fps), str(tmp)])
    tmp.replace(out)


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


def settings_dict(*, model: str, lora: str | None, strength: float, frames: int, steps: int, width: int,
                  height: int, audio: bool) -> dict:
    """What a clip depends on besides prompt, seed and start image; a clip is reused only if this is unchanged.
    The LoRA counts by content (lora_sha1), so a LoRA retrained into the same file makes its clips change."""
    return {"model": model, "lora": lora or None, "lora_sha1": file_signature(lora) if lora else None,
            "strength": float(strength) if lora else 1.0,
            "frames": int(frames), "steps": int(steps), "width": int(width), "height": int(height),
            "audio": bool(audio)}


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
            generator.generate(prompt, current, new_video, new_last, s)
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
