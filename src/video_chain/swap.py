"""Replace a person's head in a real video with another person (Wan 2.2 Animate, "Mix" mode, in ComfyUI).

The head is found here, frame by frame, with the app's face detector (faces.py): one person is followed through the
clip (the largest face first, then the face nearest to it), the face box is grown to the whole head (hair, chin) and
smoothed over time, and written as a mask video (white = the head, drawn anew). ComfyUI then draws the new head into
that area, driven by the clip's pose and face motion (DWPose), and lays it back into the original frames along a
soft edge: everything outside the head stays as it was, frame for frame, with the clip's sound.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import faces

# the head around a face box (forehead to chin): this much of the face box's width to each side, of its height above
# (hair) and below (chin, the start of the neck); scaled by the dialog's "head size"
HEAD_SIDE, HEAD_TOP, HEAD_BOTTOM = 0.45, 0.75, 0.3
SMOOTH = 5                                  # frames averaged for the head's position and size (no jitter)


def probe(video: Path) -> tuple[int, int, int, float]:
    """(width, height, frames, fps) of a clip."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames,r_frame_rate", "-print_format", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    s = (json.loads(out).get("streams") or [{}])[0]
    num, _, den = str(s.get("r_frame_rate") or "24/1").partition("/")
    return int(s["width"]), int(s["height"]), int(s.get("nb_read_frames") or 0), float(num) / float(den or 1)


def detect_faces(video: Path, model: Path, progress=None, cancelled=lambda: False) -> list[list[tuple]]:
    """The faces of every frame, [(x1, y1, x2, y2, score)] in the clip's pixels (largest first)."""
    import numpy as np
    w, h, n, _fps = probe(video)
    sw, sh = faces.fit_size(w, h)
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vf",
                             f"scale={sw}:{sh}:flags=bilinear", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, size = [], sw * sh * 3
    try:
        while True:
            if cancelled():
                raise InterruptedError("cancelled")
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            out.append(faces.detect_rgb(np.frombuffer(raw, np.uint8).reshape(sh, sw, 3), w, h, model))
            if progress is not None:
                progress(len(out), n)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    return out


def track(found: list[list[tuple]]) -> list[tuple | None]:
    """One person through the clip: the largest face of the first frame that has one, then in every frame the face
    nearest to where it was (None where it is not found; a face far away is someone else)."""
    path: list[tuple | None] = []
    last = None
    for fs in found:
        pick = None
        if fs:
            if last is None:
                pick = fs[0]
            else:
                lw = last[2] - last[0]
                cx, cy = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2

                def dist(f):
                    return ((f[0] + f[2]) / 2 - cx) ** 2 + ((f[1] + f[3]) / 2 - cy) ** 2
                near = min(fs, key=dist)
                pick = near if dist(near) <= (1.5 * lw) ** 2 else None
        path.append(tuple(pick[:4]) if pick else None)
        if pick:
            last = pick
    return path


def head_boxes(faces_track: list[tuple | None], width: int, height: int, scale: float = 1.0) -> list[tuple]:
    """The head box of every frame (x1, y1, x2, y2), from the tracked face boxes: frames without a face take the
    nearest ones (between two: interpolated), then smoothed over SMOOTH frames. scale: the head's size (1 = as
    found)."""
    n = len(faces_track)
    known = [i for i, f in enumerate(faces_track) if f is not None]
    if not known:
        raise ValueError("no face found in this clip")
    filled = []
    for i in range(n):
        if faces_track[i] is not None:
            filled.append(faces_track[i])
            continue
        before = max((k for k in known if k < i), default=None)
        after = min((k for k in known if k > i), default=None)
        if before is None or after is None:
            filled.append(faces_track[after if before is None else before])
        else:
            t = (i - before) / (after - before)
            a, b = faces_track[before], faces_track[after]
            filled.append(tuple(a[j] + (b[j] - a[j]) * t for j in range(4)))
    half = SMOOTH // 2
    out = []
    for i in range(n):
        win = filled[max(0, i - half):i + half + 1]
        x1, y1, x2, y2 = (sum(f[j] for f in win) / len(win) for j in range(4))
        fw, fh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        side = fw * (0.5 + HEAD_SIDE) * scale
        top, bottom = (y1 + y2) / 2 - fh * (0.5 + HEAD_TOP) * scale, (y1 + y2) / 2 + fh * (0.5 + HEAD_BOTTOM) * scale
        out.append((max(0.0, cx - side), max(0.0, top), min(float(width), cx + side), min(float(height), bottom)))
    return out


def mask_frame(box: tuple, width: int, height: int):
    """One frame of the mask: an ellipse filling the head box, white (255) on black (uint8, height x width)."""
    import numpy as np
    x1, y1, x2, y2 = box
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, max(1.0, (x2 - x1) / 2), max(1.0, (y2 - y1) / 2)
    ys, xs = np.ogrid[:height, :width]
    return ((((xs + 0.5 - cx) / rx) ** 2 + ((ys + 0.5 - cy) / ry) ** 2) <= 1.0).astype(np.uint8) * 255


def write_mask(boxes: list[tuple], width: int, height: int, out: Path, fps: float = 24.0) -> None:
    """The head mask as a lossless video (one frame per box), white = the head."""
    out = Path(out)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                             "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-qp", "0",
                             "-pix_fmt", "yuv444p", str(tmp)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for b in boxes:
            proc.stdin.write(mask_frame(b, width, height).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("could not write the head mask: " + proc.stderr.read().decode(errors="replace")[-300:])
        tmp.replace(out)
    finally:
        if proc.poll() is None:
            proc.kill()
        tmp.unlink(missing_ok=True)


def find_heads(video: Path, model: Path, scale: float = 1.0, progress=None, cancelled=lambda: False) -> dict:
    """Faces found, tracked and grown to head boxes: {"boxes", "found" (frames with the face), "frames", "size",
    "fps"}."""
    w, h, _n, fps = probe(video)
    found = detect_faces(video, model, progress, cancelled)
    tr = track(found)
    return {"boxes": head_boxes(tr, w, h, scale), "found": sum(1 for t in tr if t is not None),
            "frames": len(found), "size": (w, h), "fps": fps, "track": tr}


# ---- the video node: its source becomes the swapped clip (the original is kept to go back to)
SWAPS_DIR = "swaps"


def replaced(node) -> dict | None:
    """What a video node's person was replaced with ({"original", "reference", ...}), or None."""
    r = (node.source or {}).get("replaced")
    return r if isinstance(r, dict) and isinstance(r.get("original"), dict) else None


def apply(node, file: Path, frames: int, info: dict, fps: float = 24.0) -> None:
    """The video node plays the swapped clip (all of it) from now on; its source before is kept."""
    before = replaced(node)
    original = before["original"] if before else {k: v for k, v in (node.source or {}).items() if k != "replaced"}
    node.source = {"file": str(Path(file).resolve()), "start": 0.0, "end": round(int(frames) / fps, 4),
                   "replaced": {**info, "original": original}}


def restore(node) -> bool:
    """Back to the video as it was before its person was replaced (False if it was not)."""
    r = replaced(node)
    if r is None:
        return False
    node.source = dict(r["original"])
    return True
