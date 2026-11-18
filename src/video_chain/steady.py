"""Steady look along a chain (pure Python and ffmpeg, no Qt).

Every clip that starts from the frame the clip before handed over paints that frame again, and MiniMax H3 (most of
all with a speed LoRA) adds a little fine texture and shifts its colour while doing so: 14-24 % more texture per clip
with Turbo, measured. The next clip starts from that and adds its own, so a chain gets sharper and sharper.

A clip's first frame shows the very picture it was given, so the two tell exactly what the model added:

  measure()  texture: how much the fine texture grew (first / given, of texture()); planes: a gain and an offset per
             Y, U, V plane that map the first frame's colour back onto the given one (joinmatch's fit); None when
             the first frame is no copy of the given one (a cut, another picture)
  correct()  the frame a clip hands to the next one, with that taken off again: the colour mapped back (lutyuv), then
             the texture brought to texture(frame) / measure's texture with ffmpeg's unsharp (a negative amount
             softens, a positive one sharpens back what a softening model lost), the amount found by bisection
  correct_clip()  the same for the frames a clip carries over (a motion guide, the end of a continued clip)

The clips stay as made; only what the next clip starts from is corrected.
"""
from __future__ import annotations

import shutil
from array import array
from pathlib import Path

from . import joinmatch, media

VERSION = 1             # of what measure returns
BLUR_SIGMA = 1.2        # texture: the luma minus this blur of it (the fine detail a model sharpens)
KERNEL = 5              # unsharp's matrix (pixels) for the correction
AMOUNT = (-2.0, 5.0)    # unsharp's range of amounts
STEPS = 14              # bisection steps for the amount
MAX_GAIN = 1.6          # a texture change beyond this (or below its inverse) is not taken as it is
SAME = 0.02             # a texture change within this counts as none
WIDTH = 128             # px: colour and likeness are compared this wide (as joinmatch does)
MAX_ERROR = 10.0        # mean luma difference (0-255, mean taken off) above which the first frame is no copy


def texture(image: str | Path, vf: str = "") -> float:
    """The fine texture of a picture: the mean |luma - its blur| (0-255 scale), after the filter vf."""
    graph = ("[0:v:0]" + (vf + "," if vf else "") + "format=gray16le,split[a][b];[b]gblur=sigma=" + str(BLUR_SIGMA)
             + "[c];[a][c]blend=all_mode=difference[out]")
    raw = media.ffmpeg(["-i", str(image), "-filter_complex", graph, "-map", "[out]", "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], text=False)
    a = array("H")
    a.frombytes(raw[:len(raw) // 2 * 2])
    return sum(a) / max(1, len(a)) / 257.0


def measure(given: str | Path, first: str | Path) -> dict | None:
    """What the model added to the picture it was given (its first frame): {"texture": first / given (clamped),
    "planes": [[gain, offset]] for Y, U, V (first onto given), "version"}, or None when first is no copy of given."""
    a = joinmatch.read_yuv(given, 0, 1, WIDTH)
    b = joinmatch.read_yuv(first, 0, 1, WIDTH)
    if not a or not b or len(a[0]) != len(b[0]):
        return None
    n = len(a[0]) // 3
    if joinmatch._luma_error(a[0], b[0], n) > MAX_ERROR:
        return None
    planes = [list(joinmatch._fit(list(b[0][p * n:(p + 1) * n]), list(a[0][p * n:(p + 1) * n])))
              for p in range(3)]
    tg = texture(given)
    ratio = texture(first) / tg if tg > 0.05 else 1.0
    ratio = min(MAX_GAIN, max(1.0 / MAX_GAIN, ratio))
    if abs(ratio - 1.0) < SAME:
        ratio = 1.0
    return {"texture": round(ratio, 4), "planes": planes, "version": VERSION}


def is_identity(look: dict | None) -> bool:
    return not look or (float(look.get("texture", 1.0)) == 1.0 and not joinmatch.lut(look.get("planes") or []))


def correction_filter(src: str | Path, look: dict) -> str:
    """The ffmpeg filter that takes look off a picture like src: the colour, then the texture (the amount found on
    src; the same filter serves the frames carried over with it). "" when there is nothing to take off."""
    colour = joinmatch.lut(look.get("planes") or [])
    ratio = float(look.get("texture", 1.0))
    if ratio == 1.0:
        return colour
    target = texture(src) / ratio
    pre = colour + "," if colour else ""

    def unsharp(amount: float) -> str:
        return f"{pre}unsharp=lx={KERNEL}:ly={KERNEL}:la={amount:.4f}:cx={KERNEL}:cy={KERNEL}:ca=0"
    lo, hi = (AMOUNT[0], 0.0) if ratio > 1.0 else (0.0, AMOUNT[1])
    for _ in range(STEPS):                      # texture grows with the amount
        mid = (lo + hi) / 2
        if texture(src, unsharp(mid)) > target:
            hi = mid
        else:
            lo = mid
    return unsharp((lo + hi) / 2)


def correct(src: str | Path, out: str | Path, look: dict | None, vf: str | None = None) -> None:
    """The picture src with look (measure) taken off, as PNG at out (a copy when there is nothing to take off).
    vf: the correction_filter already worked out (for a picture like src)."""
    out = Path(out)
    if is_identity(look):
        shutil.copyfile(src, out)
        return
    assert look is not None
    vf = correction_filter(src, look) if vf is None else vf
    if not vf:
        shutil.copyfile(src, out)
        return
    with media.part_file(out) as tmp:
        media.ffmpeg(["-y", "-i", str(src), "-map", "0:v:0", "-frames:v", "1", "-vf", vf + ",format=rgb24",
                      "-update", "1", tmp])


def correct_clip(src: str | Path, out: str | Path, look: dict | None, vf: str | None = None) -> None:
    """A carried-over clip with look taken off every frame (the amount found on its last frame, the one the next
    clip goes on from, unless vf gives the correction_filter already worked out); its sound kept as it is. A copy
    when there is nothing to take off. (The colour fit is of RGB pictures, applied here to the clip's own YUV: the
    same to within a level or so.)"""
    out = Path(out)
    if is_identity(look):
        shutil.copyfile(src, out)
        return
    assert look is not None
    if vf is None:
        from . import chain
        with media.scratch("steady") as tmp:
            last = Path(tmp) / "last.png"
            chain.frame_at(Path(src), chain.probe(Path(src))["frames"] - 1, last)
            vf = correction_filter(last, look)
    if not vf:
        shutil.copyfile(src, out)
        return
    with media.part_file(out) as tmp_out:
        media.ffmpeg(["-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-vf", vf, "-c:v", "libx264",
                      "-crf", "8", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-fps_mode",
                      "passthrough", tmp_out])
