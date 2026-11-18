"""Steady look along a chain (steady.py): what a clip's model added to the frame it started from (its first frame
against the frame it was given: the same picture), and the correction of the frame it hands to the next clip, so the
fine texture and the colour do not ratchet up from clip to clip. Synthetic pictures: "the model" here is an unsharp
mask and a brightness shift."""
import numpy as np
import pytest
from PIL import Image, ImageFilter

from video_chain import chain, steady as S


def _picture(path, seed=1, size=(320, 180)):
    rng = np.random.default_rng(seed)
    w, h = size
    noise = Image.fromarray((rng.random((h, w)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    base = np.asarray(noise, dtype=np.float64) * 0.5
    yy, xx = np.mgrid[:h, :w]
    base += 60 * np.sin(xx / 23.0) + 40 * np.cos(yy / 17.0) + 80
    rgb = np.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    return path


def _model(src, out, amount=80, brighter=12):
    """What a model does to the frame it starts from: sharper fine texture, a little brighter."""
    im = Image.open(src).convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.5, percent=amount, threshold=0))
    a = (np.asarray(im, dtype=np.int16) + brighter).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(out)
    return out


def _mean(path):
    return float(np.asarray(Image.open(path).convert("L"), dtype=np.float64).mean())


def test_texture_rises_with_sharpening(tmp_path):
    a = _picture(tmp_path / "a.png")
    b = _model(a, tmp_path / "b.png", brighter=0)
    assert S.texture(b) > S.texture(a) * 1.1


def test_what_the_model_added_is_measured_and_taken_off_the_handed_over_frame(tmp_path):
    given = _picture(tmp_path / "given.png")
    first = _model(given, tmp_path / "first.png")
    look = S.measure(given, first)
    assert look is not None and look["texture"] > 1.1 and look["version"] == S.VERSION
    assert look["planes"][0] != [1.0, 0.0]                                 # (brighter: the luma is mapped back)
    last = _model(_picture(tmp_path / "later.png", seed=2), tmp_path / "last.png")   # (another moment, same look)
    out = tmp_path / "next_start.png"
    S.correct(last, out, look)
    assert S.texture(out) == pytest.approx(S.texture(last) / look["texture"], rel=0.06)
    assert abs(_mean(out) - (_mean(last) - 12)) < 3.0
    assert Image.open(out).size == Image.open(last).size


def test_a_softening_model_is_sharpened_back(tmp_path):
    given = _picture(tmp_path / "given.png")
    first = tmp_path / "first.png"
    Image.open(given).filter(ImageFilter.GaussianBlur(0.8)).save(first)
    look = S.measure(given, first)
    assert look is not None and look["texture"] < 0.9
    out = tmp_path / "out.png"
    S.correct(first, out, look)
    assert S.texture(out) == pytest.approx(S.texture(first) / look["texture"], rel=0.08)


def test_no_correction_for_another_picture_or_an_unchanged_look(tmp_path):
    a = _picture(tmp_path / "a.png", seed=1)
    other = _picture(tmp_path / "b.png", seed=9, size=(320, 180))
    Image.open(other).transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=False).save(other)
    assert S.measure(a, other) is None                                     # (a cut: the first frame is no copy)
    same = S.measure(a, a)
    assert same is not None and S.is_identity(same)
    out = tmp_path / "out.png"
    S.correct(a, out, same)
    assert np.array_equal(np.asarray(Image.open(out)), np.asarray(Image.open(a)))


def test_a_handed_over_clip_is_corrected_frame_by_frame(tmp_path):
    """Carry-over frames and the continued clip's end (<Video 1>) get the same correction as a start frame."""
    from conftest import settings  # noqa: F401  (the test environment)
    given = _picture(tmp_path / "given.png")
    first = _model(given, tmp_path / "first.png")
    look = S.measure(given, first)
    frames = tmp_path / "f%02d.png"
    for k in range(5):
        _model(_picture(tmp_path / f"p{k}.png", seed=k + 3), tmp_path / f"f{k:02d}.png")
    clip = tmp_path / "carry.mp4"
    chain.media.ffmpeg(["-y", "-framerate", "24", "-i", str(frames), "-c:v", "libx264", "-crf", "8",
                        "-pix_fmt", "yuv420p", str(clip)])
    out = tmp_path / "carry_steady.mp4"
    S.correct_clip(clip, out, look)
    assert chain.probe(out)["frames"] == 5
    last_in, last_out = tmp_path / "li.png", tmp_path / "lo.png"
    chain.frame_at(clip, 4, last_in)
    chain.frame_at(out, 4, last_out)
    assert S.texture(last_out) < S.texture(last_in) * 0.95


def test_the_measure_is_clamped(tmp_path):
    given = _picture(tmp_path / "given.png")
    first = _model(given, tmp_path / "first.png", amount=500, brighter=0)
    look = S.measure(given, first)
    assert look is not None and look["texture"] <= S.MAX_GAIN
