"""Steady look along a chain in the tree (option "steady_look"): a clip's look change (its first frame against the
frame it was given) is recorded, and the frame it hands to the next clip goes there with it taken off; existing
projects keep making their clips as before."""
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageFilter

from video_chain import chain, steady as S, tree as T

pytestmark = pytest.mark.steady_default


def _picture(path, size=(192, 108)):
    rng = np.random.default_rng(3)
    w, h = size
    noise = Image.fromarray((rng.random((h, w)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    yy, xx = np.mgrid[:h, :w]
    base = np.asarray(noise, dtype=np.float64) * 0.5 + 50 * np.sin(xx / 13.0) + 30 * np.cos(yy / 9.0) + 90
    Image.fromarray(np.stack([base] * 3, axis=-1).clip(0, 255).astype(np.uint8)).save(path)
    return path


class SharpeningGenerator:
    """Like the model: every frame of a clip is the frame it was given, sharper and a little brighter."""

    def __init__(self, keep):
        self.keep, self.calls = keep, []

    def generate(self, prompt, start_image, out_video, out_last_frame, seed, settings=None):
        k = len(self.calls)
        given = self.keep / f"given{k}.png"
        shutil.copyfile(start_image, given)
        guides = []
        for i, g in enumerate((settings or {}).get("guides") or []):
            copy = self.keep / f"guide{k}_{i}.mp4"
            shutil.copyfile(g["video"], copy)
            guides.append(copy)
        self.calls.append({"given": given, "settings": settings, "guides": guides})
        im = Image.open(start_image).convert("RGB")
        w, h = (settings or {}).get("width", 192), (settings or {}).get("height", 108)
        im = im.resize((w, h)).filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=0))
        im = Image.fromarray((np.asarray(im, dtype=np.int16) + 6).clip(0, 255).astype(np.uint8))
        frame = self.keep / f"frame{k}.png"
        im.save(frame)
        chain.media.ffmpeg(["-y", "-loop", "1", "-i", str(frame), "-frames:v", "17", "-r", "24", "-c:v", "libx264",
                            "-crf", "4", "-pix_fmt", "yuv444p", str(out_video)])
        shutil.copyfile(frame, out_last_frame)


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    t = T.Tree()
    T.set_start_image(p, _picture(tmp_path / "in.png"), t)
    a = t.add(None, "a")
    b = t.add(a.id, "b")
    c = t.add(b.id, "c")
    T.save(p, t)
    return p, a.id, b.id, c.id


def _run(p, ids, tmp_path):
    keep = tmp_path / f"keep{len(list(tmp_path.glob('keep*')))}"
    keep.mkdir()
    gen = SharpeningGenerator(keep)
    T.run_tree(gen, p, ids, log=lambda _m: None)
    return gen


def test_new_projects_keep_the_look_steady_and_existing_ones_do_not():
    assert T.DEFAULT_OPTIONS["steady_look"] is True and T.Tree().options["steady_look"] is True
    old = T.Tree.from_json({"options": {"frames": 17}, "nodes": []})       # (saved before the option existed)
    assert old.options["steady_look"] is False
    on = T.Tree.from_json({"options": {"steady_look": True}, "nodes": []})
    assert on.options["steady_look"] is True


def test_the_handed_over_frame_is_corrected_along_the_chain(project, tmp_path):
    p, a, b, c = project
    gen = _run(p, [c], tmp_path)
    t = T.load(p)
    look = t.nodes[a].made["look"]
    assert look["texture"] > 1.05 and look["version"] == S.VERSION        # (what the model added to its start)
    last_a = p / t.nodes[a].made["frame"]
    given_b = gen.calls[1]["given"]
    assert S.texture(given_b) < S.texture(last_a) * 0.97                 # (b starts with that taken off)
    assert S.texture(given_b) == pytest.approx(S.texture(gen.calls[0]["given"]), rel=0.08)   # (as a was given)
    assert S.texture(gen.calls[2]["given"]) == pytest.approx(S.texture(gen.calls[0]["given"]), rel=0.1)
    assert "steady_look" not in gen.calls[0]["settings"]                 # (the first clip: nothing handed over)
    assert gen.calls[1]["settings"]["steady_look"] == S.VERSION
    assert np.asarray(Image.open(last_a)).mean() > 0                      # (the clips stay as made)
    ch = T.Checker(p, t)
    assert all(ch.status(n) == "done" for n in (a, b, c))


def test_an_existing_project_hands_over_the_frame_as_it_is(project, tmp_path):
    p, a, b, c = project
    f = p / T.TREE_FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    del d["options"]["steady_look"]                                       # (a project from before)
    f.write_text(json.dumps(d), encoding="utf-8")
    gen = _run(p, [b], tmp_path)
    t = T.load(p)
    assert "look" not in t.nodes[a].made and "steady_look" not in gen.calls[1]["settings"]
    assert np.array_equal(np.asarray(Image.open(gen.calls[1]["given"]).convert("RGB")),
                          np.asarray(Image.open(p / t.nodes[a].made["frame"]).convert("RGB")))
    t.options["steady_look"] = True                                       # switched on: the clips after change,
    T.save(p, t)                                                          # the first one stays
    ch = T.Checker(p, T.load(p))
    assert ch.status(a) == "done" and ch.status(b) == "changed"
    gen = _run(p, [b], tmp_path)                                          # (a's look measured afterwards)
    assert S.texture(gen.calls[0]["given"]) < S.texture(p / T.load(p).nodes[a].made["frame"]) * 0.97


def test_carried_over_frames_are_corrected_too(project, tmp_path):
    p, a, b, c = project
    t = T.load(p)
    t.nodes[b].overrides = {"guide": 5}
    T.save(p, t)
    gen = _run(p, [b], tmp_path)
    guide = gen.calls[1]["guides"][0]
    t = T.load(p)
    plain = tmp_path / "plain_cut.mp4"                                     # (the same carry-over, uncorrected)
    chain.cut_frames(p / t.nodes[a].made["video"], plain, 12, 5)
    got, want = tmp_path / "got.png", tmp_path / "want.png"
    chain.frame_at(guide, 4, got)
    chain.frame_at(plain, 4, want)
    look = t.nodes[a].made["look"]
    assert look["texture"] > 1.05
    assert S.texture(got) == pytest.approx(S.texture(want) / look["texture"], rel=0.04)


def test_a_linear_project_from_before_opens_with_it_off(tmp_path):
    """Review: a project from before the tree file (chain.json) got the new default and its clips were remade."""
    old = tmp_path / "linear"
    old.mkdir()
    (old / "chain.json").write_text(json.dumps({"tool": "video-chain", "settings": {}, "clips": []}))
    assert T.load(old).options["steady_look"] is False


def test_a_failed_correction_makes_the_clip_from_the_plain_frame(project, tmp_path, monkeypatch):
    """Review: an ffmpeg failure in the correction stopped the run; it is optional, the clip is made anyway."""
    p, a, b, c = project
    _run(p, [a], tmp_path)

    def broken(*a, **k):
        raise chain.media.MediaError("ffmpeg failed: boom")
    monkeypatch.setattr(S, "correction_filter", broken)
    logs = []
    keep = tmp_path / "keep_broken"
    keep.mkdir()
    gen = SharpeningGenerator(keep)
    T.run_tree(gen, p, [b], log=logs.append)
    t = T.load(p)
    assert t.nodes[b].made and any("look" in m and "boom" in m for m in logs)
    assert np.array_equal(np.asarray(Image.open(gen.calls[0]["given"]).convert("RGB")),
                          np.asarray(Image.open(p / t.nodes[a].made["frame"]).convert("RGB")))


def test_the_correction_is_worked_out_once_per_clip(project, tmp_path, monkeypatch):
    """Review: the start frame, the carry-over and the continuation each searched the same amount again (and a start
    frame the carry-over replaces was corrected for nothing)."""
    p, a, b, c = project
    t = T.load(p)
    t.nodes[b].overrides = {"guide": 5}
    T.save(p, t)
    _run(p, [a], tmp_path)
    real, seen = S.correction_filter, []
    monkeypatch.setattr(S, "correction_filter", lambda src, look: seen.append(src) or real(src, look))
    _run(p, [b], tmp_path)
    assert len(seen) == 1
