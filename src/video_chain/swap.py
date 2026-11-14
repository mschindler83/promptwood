"""Replace a person's head in a real video with another person (Wan 2.2 Animate, "Mix" mode, in ComfyUI).

The head is found here, frame by frame, with the app's face detector (faces.py): one person is followed through the
clip (the largest face first, then the face nearest to it), the face box is grown to the whole head (hair, chin) and
smoothed over time, and written as a mask video (white = the head, drawn anew). ComfyUI then draws the new head into
that area, driven by the clip's pose and face motion (DWPose), and lays it back into the original frames along a
soft edge: everything outside the head stays as it was, frame for frame, with the clip's sound.
"""
from __future__ import annotations

import logging
import bisect
import json
import shutil
import contextlib
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Callable, TypedDict

from . import chain, faces, media

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]            # x1, y1, x2, y2 in pixels
Face = tuple[float, float, float, float, float]    # a Box and the detector's score

# the head around a face box (forehead to chin): this much of the face box's width to each side, of its height above
# (hair) and below (chin, the start of the neck); scaled by the dialog's "head size"
HEAD_SIDE, HEAD_TOP, HEAD_BOTTOM = 0.45, 0.75, 0.3
SMOOTH = 5                                  # frames averaged for the head's position and size (no jitter)
LOST = 12                                   # frames without the person before the largest face is taken again


def probe(video: Path) -> tuple[int, int, int, float]:
    """(width, height, frames, fps) of a clip (its frames counted: exact). RuntimeError (MediaError) with a plain
    message when it cannot be read."""
    p = chain.probe(video)
    return p["width"], p["height"], p["frames"], p["fps"] or 24.0


def _ffmpeg(args: list[str], timeout: float = 3600) -> None:
    """ffmpeg without stdin and with a time limit; RuntimeError (MediaError) with its own message when it fails."""
    media.ffmpeg(args, timeout=timeout)


def detect_faces(video: Path, model: Path, progress: Callable[[int, int], None] | None = None,
                 cancelled: Callable[[], bool] = lambda: False,
                 info: tuple[int, int, int, float] | None = None) -> list[list[Face]]:
    """The faces of every frame, [(x1, y1, x2, y2, score)] in the clip's pixels (largest first). info: the
    probe() of the clip when the caller has it already (the clip is then not read for it a second time)."""
    import numpy as np
    w, h, n, _fps = info or probe(video)
    sw, sh = faces.fit_size(w, h)
    out = []
    try:
        with contextlib.closing(media.iter_frames(video, (sw, sh), "rgb24", flags="bilinear", passthrough=True)) as frames:
            for raw in frames:
                if cancelled():
                    raise InterruptedError("cancelled")
                out.append(faces.detect_rgb(np.frombuffer(raw, np.uint8).reshape(sh, sw, 3), w, h, model))
                if progress is not None:
                    progress(len(out), n)
    except media.MediaError as e:
        if not out:
            raise RuntimeError(f"cannot read the clip: {e}") from e
    if n and len(out) != n:
        raise RuntimeError(f"read {len(out)} frames of the clip, it has {n}")
    return out


def track(found: list[list[Face]]) -> list[Box | None]:
    """One person through the clip: the largest face of the first frame that has one, then in every frame the face
    nearest to where it was (None where it is not found; a face far away is someone else)."""
    path: list[Box | None] = []
    last: Face | None = None
    gap = 0

    def dist(f, cx, cy):
        return ((f[0] + f[2]) / 2 - cx) ** 2 + ((f[1] + f[3]) / 2 - cy) ** 2
    for fs in found:
        pick: Face | None = None
        if fs:
            if last is None:
                pick = fs[0]
            else:
                lw = last[2] - last[0]
                cx, cy = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
                near = min(fs, key=lambda f: dist(f, cx, cy))
                pick = near if dist(near, cx, cy) <= ((1.5 + 0.5 * gap) * lw) ** 2 else None
        if fs and pick is None and gap > LOST:      # lost for long: the largest face again (after a cut)
            pick = fs[0]
        path.append((pick[0], pick[1], pick[2], pick[3]) if pick else None)
        if pick:
            last, gap = pick, 0
        else:
            gap += 1
    return path


def head_boxes(faces_track: list[Box | None], width: int, height: int, scale: float = 1.0) -> list[Box]:
    """The head box of every frame (x1, y1, x2, y2), from the tracked face boxes: frames without a face take the
    nearest ones (between two: interpolated), then smoothed over SMOOTH frames. scale: the head's size (1 = as
    found)."""
    n = len(faces_track)
    known = [i for i, f in enumerate(faces_track) if f is not None]
    if not known:
        raise ValueError("no face found in this clip")
    filled: list[Box] = []
    for i in range(n):
        here = faces_track[i]
        if here is not None:
            filled.append(here)
            continue
        at = bisect.bisect_left(known, i)             # (i itself is not known here: known[at - 1] < i < known[at])
        before = known[at - 1] if at > 0 else None
        after = known[at] if at < len(known) else None
        if before is None or after is None:
            edge = faces_track[after] if before is None and after is not None else \
                faces_track[before] if before is not None else None
            assert edge is not None                   # (known holds only frames with a face)
            filled.append(edge)
        else:
            t = (i - before) / (after - before)
            a, b = faces_track[before], faces_track[after]
            assert a is not None and b is not None
            filled.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                           a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t))
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


def write_mask(boxes: list[Box], width: int, height: int, out: Path, fps: float = 24.0) -> None:
    """The head mask as a lossless video (one frame per box), white = the head."""
    out = Path(out)
    rate = Fraction(fps).limit_denominator(1001)          # (30000/1001 stays 30000/1001, not 29.97)
    with media.part_file(out) as tmp, tempfile.TemporaryFile() as err:
        proc = media.popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                            "-s", f"{width}x{height}", "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
                            "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(tmp)],
                           stdin=subprocess.PIPE, stderr=err)
        try:
            try:
                assert proc.stdin is not None             # (stdin=PIPE above)
                for b in boxes:
                    proc.stdin.write(mask_frame(b, width, height).tobytes())
                proc.stdin.close()
            except BrokenPipeError:                       # (ffmpeg stopped early: its message says why)
                pass
            code = proc.wait()
            if code != 0:
                err.seek(0)
                raise RuntimeError("could not write the head mask: " + media.tail(err.read()))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class Heads(TypedDict):
    boxes: list[Box]           # the head box of every frame
    found: int                 # frames with the person's face
    frames: int
    size: tuple[int, int]
    fps: float
    track: list[Box | None]    # the face box per frame (None: not found)


def find_heads(video: Path, model: Path, scale: float = 1.0, progress: Callable[[int, int], None] | None = None,
               cancelled: Callable[[], bool] = lambda: False) -> Heads:
    """Faces found, tracked and grown to head boxes: {"boxes", "found" (frames with the face), "frames", "size",
    "fps", "track" (the face box of every frame)}."""
    info = probe(video)
    w, h, _n, fps = info
    found = detect_faces(video, model, progress, cancelled, info)
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


# ---- DWPose cache: the pose and face motion of a clip, bound to its exact video, range and size
POSE_CACHE_DIR = f"{SWAPS_DIR}/pose_cache"
POSE_VERSION = 1                            # bumped when the DWPose settings of the workflow change


def pose_key(clip: Path, source: dict, gen: tuple[int, int]) -> str:
    """The cache key of a clip's DWPose videos: its source file and range, the converted clip's content and the
    size the head is made at (any change: another key)."""
    data = {"v": POSE_VERSION, "file": str((source or {}).get("file")), "start": (source or {}).get("start"),
            "end": (source or {}).get("end"), "clip": chain.file_signature(Path(clip)), "size": list(gen)}
    return media.text_digest(json.dumps(data, sort_keys=True), "sha1")[:20]


def cached_pose(project: Path, key: str) -> dict:
    """The stored DWPose videos for key: {"pose": file, "face": file}, the parts there are (may be empty)."""
    d = Path(project) / POSE_CACHE_DIR / key
    if not (d / "info.json").is_file():
        return {}
    return {name: d / f"{name}.mkv" for name in ("pose", "face") if (d / f"{name}.mkv").is_file()}


def store_pose(project: Path, key: str, parts: dict, info: dict) -> dict:
    """Keep a run's DWPose videos for key ({"pose"/"face": file}, moved in; info.json last, so a half-written
    entry is never used). Returns all parts stored for key now."""
    d = Path(project) / POSE_CACHE_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    for name, f in parts.items():
        if Path(f).is_file():
            shutil.move(str(f), d / f"{name}.mkv")        # (replace() fails across file systems)
            moved.append((Path(f), d / f"{name}.mkv"))
    try:                                        # (the info of the run that made a part first stays)
        before = json.loads((d / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        before = {}
    try:
        media.atomic_write_json(d / "info.json",
                                {**info, **before, "parts": sorted(set(before.get("parts", [])) | set(parts))})
    except BaseException:
        if not before:                          # (no entry without info.json: the parts go back, no orphan files)
            for src, dest in moved:
                try:
                    shutil.move(str(dest), src)
                except OSError:
                    dest.unlink(missing_ok=True)
        raise
    return cached_pose(project, key)


# ---- H3 head swap (the BFS "H3 Head Swap - Ref2VA" LoRA, minimax_h3_head_swap_v1.0_r32): the reference model with
# the new person's picture as <Picture 1> and the whole clip (with its sound) as the guide from its first frame, the
# LoRA at strength 1 and its structured prompt (as in the LoRA's own workflow: the face details in the middle)
H3_MAX_FRAMES = chain.H3_MAX_FRAMES         # H3's 17n+5 grid up to 15 s
# the creator's recipe (Henge, backend/workflows.py, profile "headswap"): the DMD 8-step turbo LoRA (drbaph) under the
# head swap LoRA, both at strength 1; MiniMaxH3SigmaShift 12 / 3, euler, the beta scheduler, 8 steps, cfg 1
H3_SWAP_STEPS = 8
H3_SWAP_TURBO = "minimax_h3_dmd_ref2va_8step_turbo_pruned"
H3_SWAP_RECIPE = {"sampler": "euler", "scheduler": "beta", "shift": [12.0, 3.0]}   # (a list: as stored in JSON)
HEAD_CROP_MARGIN = 2.2                      # the head square for the H3 head swap: 2.2 x the face box (forehead to
                                            # chin), so the hair, ears and chin are in it
H3_REF_SIZE = 256                           # the face picture letterboxed to 256 x 256 on black (as the LoRA's creator
                                            # prepares it, Henge's edit_modes.head_reference)
HEAD_SWAP_HEAD = 'head_swap:\n\nsubject_definitions:\n\n<Subject 1> is the reference head identity from <Picture 1>, including the complete head, facial identity, facial structure, head shape, hairstyle, hairline, skin appearance, proportions, and all distinctive visual features.\n\nFacial details of <Subject 1>:\n'
HEAD_SWAP_TAIL = (
    "\n\n"
    "<Subject 2> is the original person in <Video 1>, whose body, neck, clothing, pose, physical proportions below the head, actions, motion, timing, and interaction with the environment must remain unchanged.\n\n"
    "<Video 1> is the source video for the target video edit, providing the original body performance, motion, camera movement, framing, timing, lighting, environment, scene composition, and temporal structure.\n\n"
    "summary:\n\n"
    "[video editing + reference generation]\n\n"
    "The target video is an edited version of <Video 1>. Replace only the entire head of <Subject 2> with the reference head identity defined by <Subject 1> from <Picture 1>. Preserve the source video's body, clothing, performance, motion, camera behavior, environment, lighting, timing, and composition.\n\n"
    "retention_analysis:\n\n"
    "<Subject 1> (appears throughout the target video): attribute_transfer - the complete head identity from <Picture 1>, including facial identity, facial structure, hairstyle, head shape, proportions, skin appearance, and distinctive features, is transferred consistently onto the body of <Subject 2>.\n\n"
    "<Subject 2> (appears throughout the target video): partially_preserved - the original body, neck position, clothing, pose, motion, gestures, actions, and physical performance are preserved, while only the original head is replaced.\n\n"
    "<Video 1> (entire target timeline): partially_preserved - the original video timing, body motion, camera movement, framing, lighting, background, environment, scene composition, and temporal structure are preserved; only the subject's head appearance is modified.\n\n"
    "detailed_description:\n\n"
    "Preserve <Video 1> as the complete temporal and compositional foundation of the target video. <Subject 2> retains the exact original body, clothing, pose, body proportions, movement, gestures, actions, speed, timing, and interaction with the environment.\n\n"
    "Replace only the entire original head of <Subject 2> with <Subject 1>, the reference head identity from <Picture 1>. Preserve the complete recognizable identity of <Subject 1>, including facial structure, skull and head shape, hairstyle, hairline, skin appearance, facial proportions, eyes, eyebrows, nose, mouth, jawline, ears, and distinctive identifying features.\n\n"
    "The transferred head follows the original head motion and performance of <Subject 2> throughout <Video 1>, including head rotation, tilt, pitch, yaw, natural translation, facial orientation, gaze direction, and perspective changes. Maintain anatomically correct head scale and a seamless connection between the transferred head and the original neck.\n\n"
    "Preserve the lighting conditions of <Video 1> on the transferred head. The head naturally responds to changes in illumination, shadows, highlights, color temperature, exposure, depth of field, motion blur, occlusion, camera angle, and perspective while continuously maintaining the identity established by <Picture 1>.\n\n"
    "Maintain strong temporal consistency across every frame. The identity, facial geometry, hairstyle, head size, skin appearance, and distinguishing features of <Subject 1> remain stable during motion and viewpoint changes. Prevent identity drift, facial morphing, hairstyle changes, head-size fluctuations, geometry instability, texture crawling, temporal flickering, frame-to-frame inconsistency, or sudden changes in facial appearance.\n\n"
    "Do not alter the body, clothing, hands, pose, gestures, action, background, objects, environment, camera trajectory, framing, lens behavior, scene timing, or composition inherited from <Video 1>. The final result should appear as though <Subject 1>'s head naturally belonged to <Subject 2>'s body during the original recorded performance.\n\n"
    "overall_soundscape:\n\n"
    "Preserve the original audiovisual timing and visible action structure of <Video 1>. No visual modifications are made outside the head replacement.\n\n"
    "non_diegetic_music:\nN/A")


def head_swap_prompt(details: str = "") -> str:
    """The head swap LoRA's prompt; details: the new person's face in a few words (gender, face shape, skin, eyes,
    hair, distinctive features), as the LoRA's workflow writes them."""
    return HEAD_SWAP_HEAD + " ".join((details or "").split()) + HEAD_SWAP_TAIL


def h3_length(frames: int) -> int:
    """The H3 length (17n+5) that holds a clip of this many frames."""
    return chain.h3_frames(frames)


def trim_with_sound(made: Path, clip: Path, out: Path, frames: int, fps: float = 24.0) -> None:
    """The made video cut to the clip's frame count (at the clip's frame rate), with the clip's own sound
    (exactly as recorded)."""
    out = Path(out)
    rate = Fraction(fps).limit_denominator(1001)

    def run(audio: list[str], tmp: Path):
        _ffmpeg(["-y", "-i", str(made), "-i", str(clip), "-map", "0:v:0", "-map", "1:a:0?", "-frames:v",
                 str(int(frames)), "-r", f"{rate.numerator}/{rate.denominator}", "-c:v", "libx264", "-crf", "12",
                 "-pix_fmt", "yuv420p", *audio, str(tmp)])
    with media.part_file(out) as tmp:
        try:
            run(["-c:a", "copy"], tmp)          # the clip's sound exactly
        except RuntimeError as e:
            if not any(w in str(e).lower() for w in ("not currently supported in container", "could not find tag")):
                raise                           # (not about the sound: a second encode would only repeat it)
            run(["-c:a", "aac", "-b:a", "256k"], tmp)   # (a sound mp4 cannot carry as it is)
