"""Smooth joins at a carry-over (pure Python and ffmpeg, no Qt).

A generated clip that carries over the last frames of the clip before does not contain those frames: MiniMax H3
gets them as conditioning (MiniMaxH3AddGuide) and paints its own copy of them at the same moments. The copy is
close, but its timing can be a frame or two off and its colour slightly different, so a hard cut after the
carry-over shows a small jump. measure() compares the copy with the original:

  timing   the copy's frames at offsets -SEARCH .. +SEARCH against the original's last frames (the luma, each frame's
           mean taken off, so a colour shift does not count); the best offset moves the cut, so the frame after
           the clip before's last one shows the next moment (cut = carry-over + offset)
  colour   a gain and an offset per Y, U, V plane that map the copy onto the original (only where it helps)
  blend    the last frames of the clip before are blended into the copy's frames of the same moments

None when the copy does not match the original at all (another picture): that join is left as it is. lut() gives
the ffmpeg filter of the colour correction; chain.concat applies it with the blend.
"""
from __future__ import annotations

from pathlib import Path

from . import media

SEARCH = 3              # frames: the copy's timing is looked for this far each way
WINDOW = 12             # frames: the original's last frames compared (the ones nearest the cut matter)
BLEND = 6               # frames blended at a carry-over join
MIN_GUIDE = 5           # shorter carry-overs (the last frame only): colour only
WIDTH = 128             # px: frames are compared this wide
MAX_ERROR = 10.0        # mean luma difference (0-255) above which the copy is not one of the original
GAIN = (0.85, 1.18)     # the colour correction never goes beyond these gains ...
SHIFT = 30.0            # ... and offsets (0-255)
TIE = 1.03              # an offset must be this much better than one nearer 0 to be taken
VERSION = 1             # of what measure returns (a kept measure of another version is measured again)


def read_yuv(video: str | Path, first: int, count: int, width: int = WIDTH) -> list[bytes]:
    """Frames first .. first+count-1 of a video, scaled to `width` (the height in proportion, even), as YUV 4:4:4
    planes: each frame Y, then U, then V (the values as stored, no colour conversion). Fewer at the video's end."""
    if count <= 0:
        return []
    info = media.probe(video)
    height = max(2, round(width * info.height / max(1, info.width) / 2) * 2)
    vf = (f"trim=start_frame={max(0, first)}:end_frame={max(0, first) + count},setpts=PTS-STARTPTS,"
          f"scale={width}:{height}:flags=area,format=yuv444p")
    raw = media.ffmpeg(["-i", str(video), "-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "yuv444p",
                        "-fps_mode", "passthrough", "-"], text=False)
    per = width * height * 3
    return [raw[k:k + per] for k in range(0, len(raw) - per + 1, per)]


def _luma_error(a: bytes, b: bytes, n: int) -> float:
    """Mean |a - b| of the luma planes, each frame's mean taken off (a colour shift does not count)."""
    ya, yb = a[:n], b[:n]
    c = (sum(ya) - sum(yb)) / n
    return sum(abs(x - y - c) for x, y in zip(ya, yb)) / n


def _fit(xs: list[int], ys: list[int]) -> tuple[float, float]:
    """Gain and offset with ys ~ g * xs + o (mean and spread matched), (1, 0) unless it brings ys closer."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    g = min(GAIN[1], max(GAIN[0], sy / sx)) if sx > 1.0 else 1.0
    o = min(SHIFT, max(-SHIFT, my - g * mx))
    before = sum(abs(y - x) for x, y in zip(xs, ys)) / n
    after = sum(abs(y - min(255.0, max(0.0, g * x + o))) for x, y in zip(xs, ys)) / n
    return (round(g, 4), round(o, 3)) if after < before * 0.97 and before - after > 0.2 else (1.0, 0.0)


def measure(a_video: str | Path, a_stop: int, b_video: str | Path, guide: int) -> dict | None:
    """How the clip after (b) carries on from the clip before (a), whose frames up to a_stop (exclusive) are shown;
    guide: b's carry-over frames (0 / 1: b's first frame repeats a's last one). {"offset": frames to add to the
    cut, "planes": [[gain, offset]] for Y, U, V (b onto a), "blend": frames to blend, "error"}, or None when b's
    start is no copy of a's end."""
    g = int(guide) if int(guide) >= MIN_GUIDE else 1
    a = read_yuv(a_video, a_stop - g, g)
    b = read_yuv(b_video, 0, g + (SEARCH if g > 1 else 0))
    if len(a) < g or not b:
        return None
    n = len(a[0]) // 3                                   # pixels per plane
    window = range(max(0, g - WINDOW), g)
    errors: dict[int, float] = {}
    for d in (range(-SEARCH, SEARCH + 1) if g > 1 else (0,)):
        pairs = [(i, i + d) for i in window if 0 <= i + d < len(b)]
        if len(pairs) < max(1, len(window) // 2):
            continue
        errors[d] = sum(_luma_error(a[i], b[j], n) for i, j in pairs) / len(pairs)
    if not errors:
        return None
    low = min(errors.values())
    if low > MAX_ERROR:
        return None
    offset = min((d for d, e in errors.items() if e <= low * TIE + 0.05), key=abs)
    pairs = [(i, i + offset) for i in window if 0 <= i + offset < len(b)]
    planes = []
    for p in range(3):
        xs: list[int] = []
        ys: list[int] = []
        for i, j in pairs:
            xs += b[j][p * n:(p + 1) * n:2]
            ys += a[i][p * n:(p + 1) * n:2]
        planes.append(list(_fit(xs, ys)))
    cut = g + offset if g > 1 else 1
    blend = min(BLEND, g - 1, cut) if g > 1 else 0
    return {"offset": offset if g > 1 else 0, "planes": planes, "blend": max(0, blend), "error": round(low, 2),
            "version": VERSION}


def lut(planes) -> str:
    """The ffmpeg filter of a colour correction ([[gain, offset]] for Y, U, V), "" when it changes nothing."""
    exprs, used = [], False
    for name, (g, o) in zip("yuv", planes):
        if abs(g - 1.0) < 0.004 and abs(o) < 0.4:
            exprs.append(f"{name}=val")
        else:
            used = True
            exprs.append(f"{name}='clip(val*{g:.4f}+{o:.3f},0,255)'")
    return "lutyuv=" + ":".join(exprs) if used else ""
