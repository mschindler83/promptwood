"""Faces in a picture, for cropping a character's face picture tight around the face.

The detector is SCRFD (insightface's det_10g.onnx, as ComfyUI face nodes keep it in models/insightface/models/
buffalo_l), run on the CPU with onnxruntime. Both are optional: without them the crop is set by hand.
The picture is read with ffmpeg (scaled to fit the detector's 640 x 640), so no image library is needed."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

DETECTOR_FILES = ("models/buffalo_l/det_10g.onnx", "models/antelopev2/scrfd_10g_bnkps.onnx")
SIZE = 640
FACE_MARGIN = 1.5          # the square crop is this many times the face box (forehead to chin): the hair and
                           # chin, but (almost) no scene or pose for the model to drift back to
MIN_SCORE = 0.5

_SESSIONS: dict[str, object] = {}


def find_detector(comfy_dir: Path | None) -> Path | None:
    """The SCRFD model in ComfyUI's insightface folders (models/insightface and extra_model_paths.yaml)."""
    if comfy_dir is None:
        return None
    from .comfy import comfy_model_roots
    for root in comfy_model_roots(Path(comfy_dir), "insightface"):
        for rel in DETECTOR_FILES:
            if (root / rel).is_file():
                return root / rel
    return None


def available(model: Path | None) -> str | None:
    """Why faces cannot be found ('' never): None when the detector can run."""
    if model is None or not Path(model).is_file():
        return ("no face detector found (insightface's det_10g.onnx in ComfyUI's models/insightface/models/"
                "buffalo_l)")
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime is not installed (uv pip install onnxruntime numpy)"
    return None


def image_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-print_format", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = (json.loads(out).get("streams") or [{}])[0]
    return int(s["width"]), int(s["height"])


def _session(model: Path):
    key = str(model)
    if key not in _SESSIONS:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _SESSIONS[key] = ort.InferenceSession(key, opts, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def _nms(boxes, scores, thresh=0.4):
    import numpy as np
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thresh]
    return keep


def detect(path: Path, model: Path) -> list[tuple[float, float, float, float, float]]:
    """The faces in a picture, [(x1, y1, x2, y2, score)] in its pixels, largest first. The picture should be a
    PNG (no EXIF rotation): ffprobe gives the stored size, ffmpeg would turn a rotated photo first."""
    import numpy as np
    w, h = image_size(path)
    scale = SIZE / max(w, h)
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-vf",
                          f"scale={sw}:{sh}:flags=bilinear", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    img = np.frombuffer(raw, np.uint8)[: sw * sh * 3].reshape(sh, sw, 3)
    return detect_rgb(img, w, h, model)


def fit_size(w: int, h: int) -> tuple[int, int]:
    """The size a picture is scaled to for the detector (its long side 640)."""
    scale = SIZE / max(w, h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def detect_rgb(img, w: int, h: int, model: Path) -> list[tuple[float, float, float, float, float]]:
    """detect() for a picture already scaled to fit_size(w, h) (an RGB uint8 array): faces in w x h pixels."""
    import numpy as np
    sh, sw = img.shape[:2]
    canvas = np.zeros((SIZE, SIZE, 3), np.float32)
    canvas[:sh, :sw] = img
    blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
    sess = _session(Path(model))
    outs = sess.run(None, {sess.get_inputs()[0].name: blob})
    boxes, scores = [], []
    for k, stride in enumerate((8, 16, 32)):
        sc, bb = outs[k].reshape(-1), outs[k + 3].reshape(-1, 4) * stride
        n = SIZE // stride
        ys, xs = np.mgrid[:n, :n]
        centers = np.repeat(np.stack([xs, ys], -1).reshape(-1, 2) * stride, len(sc) // (n * n), axis=0)
        ok = sc >= MIN_SCORE
        c, d = centers[ok], bb[ok]
        boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1))
        scores.append(sc[ok])
    if not boxes or not sum(len(s) for s in scores):
        return []
    b, s = np.concatenate(boxes), np.concatenate(scores)
    b = b / np.array([sw / w, sh / h, sw / w, sh / h])          # the exact factors of the scaled picture
    found = [(*map(float, b[i]), float(s[i])) for i in _nms(b, s)]
    found = [(max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2), sc) for x1, y1, x2, y2, sc in found]
    found = [f for f in found if f[2] - f[0] >= 1 and f[3] - f[1] >= 1]   # e.g. a hit in the padding
    return sorted(found, key=lambda f: -(f[2] - f[0]) * (f[3] - f[1]))


def face_square(face, width: int, height: int, margin: float = FACE_MARGIN) -> tuple[int, int, int]:
    """The square crop (x, y, side) around a face box, inside the picture; the face a little above the middle."""
    x1, y1, x2, y2 = face[:4]
    side = max(1, min(int(round(max(x2 - x1, y2 - y1) * margin)), width, height))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + (y2 - y1) * 0.05
    x = int(round(min(max(cx - side / 2, 0), width - side)))
    y = int(round(min(max(cy - side / 2, 0), height - side)))
    return x, y, side


def centre_square(width: int, height: int) -> tuple[int, int, int]:
    side = min(width, height)
    return (width - side) // 2, (height - side) // 2, side
