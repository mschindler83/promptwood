"""Smooth joins at a carry-over: the model's copy of the carry-over frames is measured against the clip before
(timing offset, colour), the cut follows the measured timing, the colour is matched and the last frames are
blended. Synthetic videos (ffmpeg lavfi), no model."""
import json
import subprocess
from pathlib import Path

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"      # never open windows on the real desktop
import pytest  # noqa: E402

from conftest import settings  # noqa: E402,F401
from ui_helpers import app, image, new_project, win  # noqa: E402,F401
from video_chain import chain, joinmatch  # noqa: E402
from video_chain import tree as T  # noqa: E402

FPS = 24
W, H = 320, 180
SHIFT = "lutyuv=y='clip(val*1.06+4,0,255)':u='clip(val+5,0,255)'"     # the model's slightly other colour


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """150 frames of moving test picture, nearly lossless."""
    d = tmp_path_factory.mktemp("scene")
    s = d / "scene.mp4"
    _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}", "-frames:v", 150, "-c:v", "libx264", "-qp", 0,
            "-pix_fmt", "yuv420p", s)
    return s


def _cut(src: Path, out: Path, first: int, count: int, vf: str = "") -> Path:
    f = f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS" + (f",{vf}" if vf else "")
    _ffmpeg("-i", src, "-vf", f, "-c:v", "libx264", "-qp", 0, "-pix_fmt", "yuv420p", "-r", FPS, out)
    return out


def _clips(scene, d: Path, guide: int, ahead: int, colour: bool = True):
    """A: scene frames 0..59. B: the model's clip after it with a `guide`-frame carry-over whose copy runs `ahead`
    frames ahead of the original (B[i] = scene[60 - guide + ahead + i]), in a slightly other colour."""
    a = _cut(scene, d / "a.mp4", 0, 60)
    b = _cut(scene, d / "b.mp4", 60 - guide + ahead, 70, SHIFT if colour else "")
    return a, b


def _frame(video: Path, k: int) -> bytes:
    return joinmatch.read_yuv(video, k, 1, 80)[0]


def _diff(x: bytes, y: bytes) -> float:
    return sum(abs(p - q) for p, q in zip(x, y)) / len(x)


@pytest.mark.parametrize("ahead", [0, 1, 2, -1])
def test_the_timing_of_the_models_copy_is_measured(scene, tmp_path, ahead):
    a, b = _clips(scene, tmp_path, 22, ahead)
    m = joinmatch.measure(a, 60, b, 22)
    assert m is not None and m["offset"] == -ahead and m["blend"] == 6


def test_the_colour_of_the_clip_before_is_matched(scene, tmp_path):
    a, b = _clips(scene, tmp_path, 22, 0)
    m = joinmatch.measure(a, 60, b, 22)
    (gy, oy), (gu, ou), (gv, ov) = m["planes"]
    assert gy == pytest.approx(1 / 1.06, abs=0.02) and oy == pytest.approx(-4 / 1.06, abs=2.5)
    assert ou == pytest.approx(-5, abs=1.5) and gv == pytest.approx(1, abs=0.02) and abs(ov) < 1.5
    same = joinmatch.measure(a, 60, _cut(scene, tmp_path / "same.mp4", 38, 70), 22)
    assert joinmatch.lut(same["planes"]) == ""                   # nothing to correct: no filter
    assert "lutyuv" in joinmatch.lut(m["planes"])


def test_another_picture_is_left_alone(scene, tmp_path):
    """The clip after does not copy the clip before (another scene): nothing is changed at that join."""
    a = _cut(scene, tmp_path / "a.mp4", 0, 60)
    other = tmp_path / "other.mp4"
    _ffmpeg("-f", "lavfi", "-i", f"mandelbrot=size={W}x{H}:rate={FPS}", "-frames:v", 70, "-c:v", "libx264",
            "-qp", 0, "-pix_fmt", "yuv420p", other)
    assert joinmatch.measure(a, 60, other, 22) is None


def test_a_single_frame_join_only_matches_the_colour(scene, tmp_path):
    a = _cut(scene, tmp_path / "a.mp4", 0, 60)
    b = _cut(scene, tmp_path / "b.mp4", 59, 70, SHIFT)                # its first frame repeats a's last one
    m = joinmatch.measure(a, 60, b, 1)
    assert m["offset"] == 0 and m["blend"] == 0 and joinmatch.lut(m["planes"])


def test_the_join_is_smooth(scene, tmp_path):
    """Joined with the measured cut, colour and blend: right after the join comes the scene's next frame in the
    clip before's colour, and the joined video is as long as the cuts say."""
    a, b = _clips(scene, tmp_path, 22, 1)
    m = joinmatch.measure(a, 60, b, 22)
    cut = 22 + m["offset"]
    out = tmp_path / "joined.mp4"
    chain.concat([a, b], out, cuts=[(0, 0), (cut, 0)],
                 smooth=[None, {"blend": m["blend"], "lut": joinmatch.lut(m["planes"])}])
    assert chain.probe(out)["frames"] == 60 + 70 - cut
    after = _frame(out, 60)
    assert _diff(after, _frame(scene, 60)) < 2.0                     # the right moment, in the right colour
    assert _diff(_frame(out, 59), _frame(scene, 59)) < 2.0          # the blend's end: a copy of the same moment
    assert _diff(_frame(out, 50), _frame(scene, 50)) < 0.5           # before the blend: the clip before as it is
    late = _frame(out, 100)                                          # after the fade: the model's own colour again
    assert _diff(late, _frame(b, cut + 40)) < 1.0
    naive = tmp_path / "naive.mp4"
    chain.concat([a, b], naive, cuts=[(0, 0), (22, 0)])
    assert _diff(_frame(naive, 60), _frame(scene, 60)) > 3 * _diff(after, _frame(scene, 60))


def _made(t, parent, video: Path, frames: int, guide: int):
    n = t.add(parent, "x")
    n.made = {"video": video.name, "frames": frames, "settings": {"guide": guide, "frames": frames}}
    return n


def test_the_path_takes_the_measured_cut_and_keeps_it(scene, tmp_path, monkeypatch):
    a, b = _clips(scene, tmp_path, 22, 1)
    t = T.Tree()
    na = _made(t, None, a, 60, 0)
    nb = _made(t, na.id, b, 70, 22)
    path = [na, nb]
    videos = [a, b]
    cuts = T.path_cuts(t, path)
    assert cuts == [(0, 0), (22, 0)]
    got, smooth = T.smooth_cuts(tmp_path, t, path, videos, cuts)
    assert got == [(0, 0), (21, 0)] and smooth[0] is None and smooth[1]["blend"] == 6 and smooth[1]["lut"]
    assert (tmp_path / "paths" / "joins.json").is_file()
    monkeypatch.setattr(joinmatch, "measure", lambda *a, **k: pytest.fail("measured again"))
    assert T.smooth_cuts(tmp_path, t, path, videos, cuts)[0] == got             # (kept)
    assert T.smooth_cuts(tmp_path, t, path, videos, cuts, measure=False)[0] == got
    spans = T.path_spans(t, nb.id, got)
    assert spans[1][2] == pytest.approx((60 + 70 - 21) / FPS)
    t.options["smooth_joins"] = False                                           # off: as before
    assert T.smooth_cuts(tmp_path, t, path, videos, cuts) == (cuts, [None, None])
    t.options["smooth_joins"] = True
    t.options["keep_repeated"] = True                                           # the repeated frames kept: as made
    assert T.smooth_cuts(tmp_path, t, path, videos, [(0, 0), (0, 0)]) == ([(0, 0), (0, 0)], [None, None])


def test_a_video_after_a_clip_is_not_measured(scene, tmp_path, monkeypatch):
    t = T.Tree()
    a = _cut(scene, tmp_path / "a.mp4", 0, 60)
    na = _made(t, None, a, 60, 0)
    v = t.add_video(na.id, str(a), 0.0, 1.0)
    monkeypatch.setattr(joinmatch, "measure", lambda *a, **k: pytest.fail("a real video is never changed"))
    cuts = [(0, 1), (0, 0)]
    assert T.smooth_cuts(tmp_path, t, [na, v], [a, a], cuts) == (cuts, [None, None])


def test_the_kept_measure_belongs_to_the_files(scene, tmp_path):
    a, b = _clips(scene, tmp_path, 22, 0)
    t = T.Tree()
    na, nb = _made(t, None, a, 60, 0), None
    nb = _made(t, na.id, b, 70, 22)
    T.smooth_cuts(tmp_path, t, [na, nb], [a, b], [(0, 0), (22, 0)])
    kept = json.loads((tmp_path / "paths" / "joins.json").read_text())
    assert len(kept) == 1
    _cut(scene, b, 60 - 22 + 2, 70, SHIFT)                                      # the clip made again: other timing
    got, _ = T.smooth_cuts(tmp_path, t, [na, nb], [a, b], [(0, 0), (22, 0)])
    assert got[1] == (20, 0)



def test_the_project_option(app, win, tmp_path, image):  # noqa: F811
    from video_chain.ui import ui_logic
    assert T.DEFAULT_OPTIONS["smooth_joins"] is True
    assert not any(k == "smooth_joins" for k, _ in ui_logic.project_setting_changes({"smooth_joins": True}))
    assert ("smooth_joins", "Smooth joins") in ui_logic.project_setting_changes({"smooth_joins": False})
    new_project(win, tmp_path / "p", image, ("one",))
    assert win.smooth_joins.isChecked()
    win.smooth_joins.setChecked(False)
    win._options_changed()
    assert win.tree.options["smooth_joins"] is False


def test_a_short_carry_over_is_matched_against_its_first_frame(scene, tmp_path):
    """2-4 carry-over frames: H3 anchors only the first of them (MiniMaxH3AddGuide), so b's first frame copies
    a's frame stop - guide; that is what the colour is matched with."""
    a = _cut(scene, tmp_path / "a.mp4", 0, 60)
    b = _cut(scene, tmp_path / "b.mp4", 57, 70, SHIFT)                    # b[0] = a[57]: a 3-frame carry-over
    m = joinmatch.measure(a, 60, b, 3)
    assert m is not None and m["offset"] == 0 and m["blend"] == 0
    assert m["planes"][0][0] == pytest.approx(1 / 1.06, abs=0.02)
    assert m["planes"][1] == [1.0, pytest.approx(-5, abs=0.5)]           # (the right frame: the exact U shift)


def test_a_damaged_kept_measure_is_measured_again(scene, tmp_path):
    a, b = _clips(scene, tmp_path, 22, 1)
    t = T.Tree()
    na = _made(t, None, a, 60, 0)
    nb = _made(t, na.id, b, 70, 22)
    T.smooth_cuts(tmp_path, t, [na, nb], [a, b], [(0, 0), (22, 0)])
    f = tmp_path / "paths" / "joins.json"
    kept = json.loads(f.read_text())
    f.write_text(json.dumps({k: {"offset": "x", "planes": 7} for k in kept}))
    got, smooth = T.smooth_cuts(tmp_path, t, [na, nb], [a, b], [(0, 0), (22, 0)])
    assert got == [(0, 0), (21, 0)] and smooth[1]["lut"]
    f.write_text("[1, 2]")                                                  # not even a dict
    assert T.smooth_cuts(tmp_path, t, [na, nb], [a, b], [(0, 0), (22, 0)])[0] == got
