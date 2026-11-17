"""The face swap node's geometry (comfy_nodes/video_chain_nodes/faceswap.py): the face aligned to a template, the
soft box mask, the swapped face laid back into the frame. Synthetic frames; needs OpenCV (ComfyUI's Python has it),
skipped without."""
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes"))
import faceswap as F  # noqa: E402

UPRIGHT = np.array([[260, 200], [380, 198], [320, 270], [275, 330], [365, 328]], np.float32)


def _frame(h=480, w=640, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("size,blur", [(256, 0.3), (512, 0.3), (112, 0.1), (256, 0.0), (256, 0.6)])
def test_the_box_mask_is_one_inside_and_fades_out_at_the_border(size, blur):
    m = F.box_mask(size, blur)
    assert m.shape == (size, size) and m.dtype == np.float32
    assert m.min() >= 0 and m.max() <= 1.0001 and m[size // 2, size // 2] == pytest.approx(1, abs=1e-3)
    assert m[0].max() < 0.5 and m[:, 0].max() < 0.5                         # (the very border is faded)
    assert np.allclose(m, m[::-1]) and np.allclose(m, m[:, ::-1]) and np.allclose(m, m.T)
    if blur == 0:
        assert set(np.unique(m)) == {0.0, 1.0} and m[1, 1] == 1 and m[0, 5] == 0   # (a hard box, one pixel in)


@pytest.mark.parametrize("template,size", [("arcface_128", 256), ("arcface_112_v2", 112), ("ffhq_512", 512)])
def test_the_face_is_aligned_to_the_template(template, size):
    crop, m = F.warp(_frame(), UPRIGHT, template, size)
    assert crop.shape == (size, size, 3) and m.shape == (2, 3)
    moved = UPRIGHT @ m[:, :2].T + m[:, 2]
    want = np.array(F.TEMPLATES[template], np.float32) * size
    assert np.abs(moved - want).max() < 0.06 * size                         # (a similarity fit of 5 points)
    assert F.warp(_frame(), np.zeros((5, 2), np.float32), template, size) == (None, None)   # degenerate


def _smooth(h=480, w=640):
    ys, xs = np.mgrid[:h, :w]
    return np.stack([xs * 255 // w, ys * 255 // h, (xs + ys) * 255 // (w + h)], -1).astype(np.uint8)


def test_the_swapped_face_goes_back_where_it_came_from():
    smooth = _smooth()
    crop, m = F.warp(smooth, UPRIGHT, "arcface_128", 256)
    same = F.paste_back(smooth, crop, np.ones((256, 256), np.float32), m)
    face = (slice(205, 325), slice(275, 365))                                # (well inside the face's box)
    assert np.abs(same[face].astype(int) - smooth[face].astype(int)).max() <= 3   # (the crop, warped back)
    frame = _frame()
    small = (UPRIGHT - UPRIGHT.mean(axis=0)) * 0.3 + [320, 265]                  # (a small face: its crop ~130 px)
    crop, m = F.warp(frame, small, "arcface_128", 256)
    red = np.zeros_like(crop)
    red[..., 0] = 255
    out = F.paste_back(frame, red, F.box_mask(256), m)
    assert out[265, 320, 0] == 255 and out[265, 320, 1] == 0               # the face's middle: the new face
    assert np.array_equal(out[:40], frame[:40]) and np.array_equal(out[:, :150], frame[:, :150])  # far away: kept
    assert np.array_equal(F.paste_back(frame, red, np.zeros((256, 256), np.float32), m), frame)   # no mask: kept
    assert out is not frame and frame[265, 320, 0] != 255 or frame[265, 320, 1] != 0       # (the frame itself kept)


def test_a_face_partly_outside_the_frame_is_pasted_without_error():
    lm = np.array([[600, 400], [700, 410], [650, 450], [610, 500], [690, 505]], np.float32)
    frame = _frame()
    crop, m = F.warp(frame, lm, "arcface_128", 256)
    out = F.paste_back(frame, np.full_like(crop, 200), F.box_mask(256), m)
    assert out.shape == frame.shape and out.dtype == np.uint8
