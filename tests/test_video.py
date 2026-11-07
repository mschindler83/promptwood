"""Real-video nodes: conversion, end frames, motion guides, joins (synthetic videos, fake generator)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fakes import FakeGenerator
from PIL import Image

from video_chain import __main__ as cli
from video_chain import chain
from video_chain import tree as T

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def make_source(path: Path, colours=("red", "blue"), secs=1.0, size="320x180", fps=30, audio=True) -> Path:
    """A synthetic 'real' video: one second per colour at another frame rate and aspect ratio than the chain."""
    args, labels = [], ""
    for k, c in enumerate(colours):
        args += ["-f", "lavfi", "-i", f"color={c}:s={size}:r={fps}:d={secs}"]
        labels += f"[{k}:v]"
    cmd = ["ffmpeg", "-v", "error", "-y", *args]
    filt = f"{labels}concat=n={len(colours)}:v=1:a=0[v]"
    maps = ["-map", "[v]"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=500:d={secs * len(colours)}"]
        maps += ["-map", f"{len(colours)}:a", "-c:a", "aac"]
    subprocess.run(cmd + ["-filter_complex", filt, *maps, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return path


def rgb(png: Path):
    return Image.open(png).convert("RGB").getpixel((5, 5))


def is_colour(px, name):
    want = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 128, 0), "yellow": (255, 255, 0)}[name]
    return all(abs(a - b) < 40 for a, b in zip(px, want))


@pytest.fixture
def source(tmp_path):
    return make_source(tmp_path / "real.mp4")


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    T.save(p, T.Tree())
    return p


def with_start(project, tmp_path):
    img = tmp_path / "start.jpg"
    Image.new("RGB", (96, 54), (0, 160, 0)).save(img)
    t = T.load(project)
    T.set_start_image(project, img, t)
    T.save(project, t)


def test_convert_range(tmp_path, source):
    out = tmp_path / "c.mp4"
    n = chain.convert_range(source, out, 0.5, 1.5, 384, 224)
    info = chain.probe(out)
    assert n == 24 == info["frames"]                        # 30 fps source -> 24 fps, exactly the range
    assert (info["width"], info["height"], info["fps"]) == (384, 224, 24)
    assert info["audio"]
    chain.frame_at(out, 0, tmp_path / "f.png")
    chain.frame_at(out, n - 1, tmp_path / "l.png")
    assert is_colour(rgb(tmp_path / "f.png"), "red") and is_colour(rgb(tmp_path / "l.png"), "blue")


def test_video_first_without_start_image(project, source):
    t = T.load(project)
    v = t.add_video(None, str(source), 0.0, 1.0)
    a = t.add(v.id, "after the video")
    T.save(project, t)
    ch = T.Checker(project, t)
    base = ch.base()
    assert (base["width"], base["height"]) == chain.canvas_size(320, 180, 512)       # size from the video
    assert T.plan(t, ch, [a.id]) == [v.id, a.id]
    gen = FakeGenerator(frames=17, size=None)
    T.run_tree(gen, project, [a.id], log=lambda _m: None)
    t = T.load(project)
    assert t.nodes[v.id].made["frames"] == 24
    assert is_colour(gen.calls[0]["start_rgb"], "red")    # the clip starts from the video's last frame
    assert gen.calls[0]["end_rgb"] is None
    ch = T.Checker(project, t)
    assert ch.status(v.id) == ch.status(a.id) == "done"
    assert T.path_cuts(t, t.path_to(a.id)) == [(0, 0), (1, 0)]
    joined = T.join_path(project, t, a.id)
    assert chain.probe(joined)["frames"] == 24 + 16


def test_clip_into_video_gets_end_frame(project, tmp_path, source):
    with_start(project, tmp_path)
    t = T.load(project)
    a = t.add(None, "toward the video")
    v = t.add_video(a.id, str(source), 1.0, 2.0)          # blue part
    b = t.add(v.id, "after")
    T.save(project, t)
    ch = T.Checker(project, t)
    assert T.plan(t, ch, [b.id]) == [v.id, a.id, b.id]
    assert T.plan(t, ch, [a.id]) == [v.id, a.id]           # the video is needed even off the path
    gen = FakeGenerator(frames=17, size=None)
    T.run_tree(gen, project, [b.id], log=lambda _m: None)
    assert is_colour(gen.calls[0]["start_rgb"], "green")  # start image
    assert is_colour(gen.calls[0]["end_rgb"], "blue")     # = the video's first frame
    assert is_colour(gen.calls[1]["start_rgb"], "blue")   # b continues from the video's last frame
    t = T.load(project)
    assert t.nodes[a.id].made["end_key"] == t.nodes[v.id].made["key"]
    assert T.path_cuts(t, t.path_to(b.id)) == [(0, 1), (0, 0), (1, 0)]
    assert chain.probe(T.join_path(project, t, b.id))["frames"] == 16 + 24 + 16
    # another range: the video and everything that depends on it is made again, nothing else
    t.nodes[v.id].source["start"] = 0.5
    T.save(project, t)
    ch = T.Checker(project, t)
    assert (ch.status(v.id), ch.status(a.id), ch.status(b.id)) == ("changed", "changed", "changed")
    assert T.plan(t, ch, [a.id]) == [v.id, a.id]


def test_motion_guide(project, source):
    t = T.load(project)
    v1 = t.add_video(None, str(source), 0.0, 1.0)
    a = t.add(v1.id, "bridge")
    a.overrides = {"guide": 5}
    v2 = t.add_video(a.id, str(source), 1.0, 2.0)
    T.save(project, t)
    gen = FakeGenerator(frames=17, size=None)
    T.run_tree(gen, project, [v2.id], log=lambda _m: None)
    g = gen.calls[0]["guides"]
    assert [(x["at"], x["frame_idx"], x["frames"], x["probe"]["frames"], x["audio"]) for x in g] == \
        [("start", 0, 5, 5, True), ("end", 73 - 5, 5, 5, True)]
    assert gen.calls[0]["end_rgb"] is None                 # the guide replaces the single end frame
    assert gen.calls[0]["settings"]["guide"] == 5
    t = T.load(project)
    assert T.path_cuts(t, t.path_to(v2.id)) == [(0, 0), (5, 5), (0, 0)]
    assert chain.probe(T.join_path(project, t, v2.id))["frames"] == 24 + 7 + 24
    assert not list((project / "nodes").glob("video-chain-*"))           # the guide clips are gone
    # the guide comes from v1's clip: a new clip of v1 (same last frame) makes the bridge out of date
    assert T.Checker(project, t).status(a.id) == "done"
    old = project / t.nodes[v1.id].made["video"]
    new = old.with_name(f"{v1.id}_0badcafe.mp4")
    shutil.copyfile(old, new)
    t.nodes[v1.id].made["video"] = f"nodes/{new.name}"
    c = T.Checker(project, t)
    assert c.status(v1.id) == "done" and c.status(a.id) == "changed"


def test_guide_too_long(project, source):
    t = T.load(project)
    v1 = t.add_video(None, str(source), 0.0, 0.1)          # 2 frames: shorter than the guide
    a = t.add(v1.id, "x")
    a.overrides = {"guide": 5}
    T.save(project, t)
    with pytest.raises(cli.JobError, match="shorter than the 5-frame"):
        cli.prepare_job(project, [a.id])
    gen = FakeGenerator(frames=17, size=None)
    with pytest.raises(ValueError, match="shorter than the 5-frame"):
        T.run_tree(gen, project, [a.id], log=lambda _m: None)
    t = T.load(project)                                    # found before anything was made
    assert t.nodes[a.id].made is None and t.nodes[v1.id].made is None and not gen.calls


def test_range_past_the_end_of_the_source(tmp_path, source):
    out = tmp_path / "c.mp4"
    n = chain.convert_range(source, out, 1.5, 5.0, 64, 64)            # the source is 2 s long
    info = chain.probe(out)
    assert n == info["frames"] == 12
    longest = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(out)], capture_output=True, text=True, check=True).stdout
    assert float(longest) < 0.6                                      # the sound is not padded to 3.5 s


def test_one_video_per_clip_and_json(project, source):
    t = T.load(project)
    a = t.add(None, "a")
    t.add_video(a.id, str(source), 0, 1)
    assert not t.can_add_video(a.id)
    with pytest.raises(ValueError):
        t.add_video(a.id, str(source), 0, 1)
    with pytest.raises(ValueError):
        t.add_video(None, str(source), 1, 1)
    t2 = T.Tree.from_json(json.loads(json.dumps(t.to_json())))
    assert t2.to_json() == t.to_json()
    d = t.to_json()
    d["nodes"][1]["source"] = None
    with pytest.raises(ValueError, match="no source"):
        T.Tree.from_json(d)


def test_source_gone_keeps_clip(project, source, tmp_path):
    t = T.load(project)
    v = t.add_video(None, str(source), 0, 1)
    T.save(project, t)
    T.run_tree(FakeGenerator(frames=17, size=None), project, [v.id], log=lambda _m: None)
    source.rename(tmp_path / "moved.mp4")
    t = T.load(project)
    ch = T.Checker(project, t)
    assert ch.base() is not None and ch.status(v.id) == "done"
    t.nodes[v.id].source["end"] = 0.5                       # a new range cannot be made without the file
    T.save(project, t)
    with pytest.raises(FileNotFoundError, match="video file not found"):
        T.run_tree(FakeGenerator(frames=17, size=None), project, [v.id], log=lambda _m: None)


def test_only_videos_need_no_model(project, source):
    t = T.load(project)
    v = t.add_video(None, str(source), 0, 1)
    T.save(project, t)
    job = cli.prepare_job(project, [v.id])
    assert job["todo"] == [v.id] and job["ai_todo"] == []
    cli.run_job(job, None, lambda _e: None, {"id": None, "k": 0})
    assert T.Checker(project, T.load(project)).status(v.id) == "done"
