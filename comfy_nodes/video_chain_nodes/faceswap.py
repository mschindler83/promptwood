"""Face swap on video frames with FaceFusion's HyperSwap model (256 px), by the method FaceFusion uses (Video
Chain's own code: no FaceFusion code is in it).

Per frame: the faces (SCRFD det_10g, with their 5 landmarks), the person followed through the clip (the largest
face first, then the face nearest to it), the face aligned to the arcface_128 template at 256 px, HyperSwap with
the new person's identity (ArcFace w600k_r50 of the reference pictures, averaged and normalised), optionally GPEN
face restoration at 512 px, pasted back with a soft box mask. numpy, OpenCV and onnxruntime only (no ComfyUI), so
it can be used and tested on its own. Frames are RGB uint8 (h, w, 3).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

TEMPLATES = {      # the alignments the models were trained with (5 landmarks as a share of the crop; FaceFusion's)
    "arcface_112_v2": [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536],
                       [0.37097589, 0.82469196], [0.63151696, 0.82325089]],
    "arcface_128": [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                    [0.38710391, 0.72160547], [0.61507734, 0.72034453]],
    "ffhq_512": [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                 [0.39308822, 0.72541100], [0.61150205, 0.72490465]],
}
log = logging.getLogger(__name__)
DET_SIZE, DET_SCORE, MASK_BLUR = 640, 0.5, 0.3
LOST = 12                                          # frames without the face before the largest face is taken again


# NOTE: _nms (and the idea of _session) also exist in src/video_chain/faces.py. They are NOT shared on purpose: this
# package is copied into ComfyUI's custom_nodes and runs in ComfyUI's own Python, where video_chain is not installed
# (and must not be imported: it pulls in Qt-free but app-only modules). tests/test_errors_foundation.py compares the
# two _nms so they cannot drift apart. _session differs too (this one uses CUDA when it can, faces.py is CPU only).
def _session(path: str, gpu: bool = True):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()]
    if gpu and "CUDAExecutionProvider" not in providers:
        log.warning("video-chain face swap: onnxruntime has no CUDA provider, %s runs on the CPU (much slower)",
                    Path(path).name)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=providers)


def _nms(boxes, scores, thresh=0.4):
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= thresh]
    return keep


def warp(frame, landmarks, template: str, size: int):
    """The face aligned to a template: a similarity transform (turn, scale, shift) fitted to the 5 landmarks, the
    frame resampled through it. (crop, matrix frame -> crop); (None, None) when the points cannot be fitted."""
    import cv2
    src = np.asarray(landmarks, np.float32).reshape(-1, 2)
    dst = np.asarray(TEMPLATES[template], np.float32) * size
    matrix = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    if matrix is None:                             # degenerate landmarks: no alignment for this face
        return None, None
    crop = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return crop, matrix


def box_mask(size: int, blur: float = MASK_BLUR):
    """A square mask for a size x size crop: 1 inside, a band at the border set to 0, then softened with a Gaussian
    so the pasted face fades into the frame (blur: the share of the half crop the fade takes)."""
    import cv2
    fade = int(size * 0.5 * blur)                  # px the fade reaches into the crop
    band = max(fade // 2, 1)
    mask = np.zeros((size, size), np.float32)
    mask[band:size - band, band:size - band] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), fade / 4) if fade > 0 else mask


def paste_back(frame, crop, mask, matrix):
    """The crop laid back into the frame through the inverse of matrix (frame -> crop), weighted by mask. Only the
    part of the frame the crop covers is resampled; a new frame comes back."""
    import cv2
    h, w = frame.shape[:2]
    ch, cw = crop.shape[:2]
    back = cv2.invertAffineTransform(matrix)
    corners = cv2.transform(np.array([[[0, 0], [cw, 0], [cw, ch], [0, ch]]], np.float32), back)[0]
    x1, y1 = np.clip(np.floor(corners.min(axis=0)), 0, [w, h]).astype(int)
    x2, y2 = np.clip(np.ceil(corners.max(axis=0)), 0, [w, h]).astype(int)
    if x2 <= x1 or y2 <= y1:                       # (the crop lies outside the frame)
        return frame
    to_region = back.copy()
    to_region[:, 2] -= (x1, y1)                    # (into the covered part's own coordinates)
    area = (int(x2 - x1), int(y2 - y1))
    alpha = cv2.warpAffine(mask, to_region, area).clip(0, 1)[..., None]
    face = cv2.warpAffine(crop, to_region, area, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = frame.copy()
    under = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (under * (1 - alpha) + face * alpha).round().clip(0, 255).astype(np.uint8)
    return out


class FaceSwapper:
    """detector: SCRFD det_10g.onnx; recognizer: ArcFace w600k_r50.onnx; swapper: hyperswap_1*_256.onnx;
    enhancer: GPEN-BFR-512.onnx or None."""

    def __init__(self, detector: str, recognizer: str, swapper: str, enhancer: str | None = None,
                 gpu: bool = True):
        self.det = _session(detector, gpu)
        self.rec = _session(recognizer, gpu)
        self.swap = _session(swapper, gpu)
        self.enh = _session(enhancer, gpu) if enhancer else None

    # ---- faces
    def faces(self, frame) -> list[tuple]:
        """[(x1, y1, x2, y2, score, landmarks 5x2)] in the frame's pixels, largest first."""
        import cv2
        h, w = frame.shape[:2]
        k = DET_SIZE / max(w, h)
        sw, sh = max(1, round(w * k)), max(1, round(h * k))
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), np.float32)
        canvas[:sh, :sw] = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            sc, bb, kp = outs[i].reshape(-1), outs[i + 3].reshape(-1, 4) * stride, outs[i + 6].reshape(-1, 10) * stride
            n = DET_SIZE // stride
            ys, xs = np.mgrid[:n, :n]
            centers = np.repeat(np.stack([xs, ys], -1).reshape(-1, 2) * stride, len(sc) // (n * n), axis=0)
            ok = sc >= DET_SCORE
            c, d, p = centers[ok], bb[ok], kp[ok]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1))
            kpss.append(np.stack([c[:, 0:1] + p[:, 0::2], c[:, 1:2] + p[:, 1::2]], -1))
            scores.append(sc[ok])
        if not sum(len(s) for s in scores):
            return []
        b, s, kps = np.concatenate(boxes) / k, np.concatenate(scores), np.concatenate(kpss) / k
        found = [(*map(float, b[i]), float(s[i]), kps[i]) for i in _nms(b, s)]
        return sorted(found, key=lambda f: -(f[2] - f[0]) * (f[3] - f[1]))

    def embedding(self, frame, landmarks):
        """ArcFace identity of a face (normalised)."""
        crop, _m = warp(frame, landmarks, "arcface_112_v2", 112)
        if crop is None:
            raise ValueError("the face could not be aligned")
        x = (crop.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        e = self.rec.run(None, {self.rec.get_inputs()[0].name: x})[0].ravel()
        return e / np.linalg.norm(e)

    def identity(self, pictures: list) -> np.ndarray:
        """The new person: the largest face of every picture, their identities averaged."""
        es = []
        for pic in pictures:
            fs = self.faces(pic)
            if fs:
                try:
                    es.append(self.embedding(pic, fs[0][5]))
                except ValueError:
                    pass
        if not es:
            raise ValueError("no face found in the picture(s) of the new person")
        e = np.mean(es, axis=0)
        return (e / np.linalg.norm(e)).astype(np.float32)

    # ---- swapping
    def swap_face(self, frame, landmarks, identity, enhance: float = 0.0):
        crop, m = warp(frame, landmarks, "arcface_128", 256)
        if crop is None:
            return frame                               # (one degenerate frame: left as it is)
        x = ((crop.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feeds = {}
        for inp in self.swap.get_inputs():
            feeds[inp.name] = identity.reshape(1, -1) if inp.name == "source" else x
        out = self.swap.run(None, feeds)[0][0].transpose(1, 2, 0)
        out = ((out * 0.5 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)
        frame = paste_back(frame, out, box_mask(256), m)
        if enhance > 0 and self.enh is not None:
            frame = self.enhance(frame, landmarks, enhance)
        return frame

    def enhance(self, frame, landmarks, blend: float):
        """GPEN face restoration (512 px), blended in by blend (0..1)."""
        crop, m = warp(frame, landmarks, "ffhq_512", 512)
        if crop is None:
            return frame
        assert self.enh is not None                    # (swap_face only calls this with an enhancer)
        x = ((crop.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.enh.run(None, {self.enh.get_inputs()[0].name: x})[0][0]
        out = ((np.clip(out, -1, 1) + 1) / 2 * 255).round().astype(np.uint8).transpose(1, 2, 0)
        pasted = paste_back(frame, out, box_mask(512), m)
        return (frame.astype(np.float32) * (1 - blend) + pasted.astype(np.float32) * blend).round().astype(np.uint8)

    def run(self, frames, identity, enhance: float = 0.0, progress=None, out=None, total: int | None = None):
        """Swap the face of one person through the frames (followed from frame to frame: the nearest face, the
        search widening while the face is lost, the largest face again after LOST frames); frames without that
        face stay as they are. out(k, frame) takes each frame as it is done (else they are returned as a list).
        total is the number of frames when frames is a generator (for progress).
        Returns (frames or None, how many were swapped)."""
        last, kept, done, gap = None, [], 0, 0
        for k, f in enumerate(frames):
            fs = self.faces(f)
            pick = None
            if fs:
                if last is None or gap > LOST:
                    pick = fs[0]
                else:
                    lw = last[2] - last[0]
                    cx, cy = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
                    near = min(fs, key=lambda g: ((g[0] + g[2]) / 2 - cx) ** 2 + ((g[1] + g[3]) / 2 - cy) ** 2)
                    reach = lw * (2 + 0.5 * gap)
                    if ((near[0] + near[2]) / 2 - cx) ** 2 + ((near[1] + near[3]) / 2 - cy) ** 2 <= reach ** 2:
                        pick = near
            if pick is not None:
                f = self.swap_face(f, pick[5], identity, enhance)
                last, gap = pick, 0
                done += 1
            else:
                gap += 1
            if out is not None:
                out(k, f)
            else:
                kept.append(f)
            if progress is not None:
                progress(k + 1, total if total is not None else len(frames))
        return (None if out is not None else kept), done
