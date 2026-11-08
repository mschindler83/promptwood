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


def test_own_end_image(project, tmp_path, source):
    with_start(project, tmp_path)
    t = T.load(project)
    a = t.add(None, "toward the end image")
    b = t.add(a.id, "after")
    yellow = tmp_path / "yellow.png"
    Image.new("RGB", (40, 30), (255, 255, 0)).save(yellow)
    rel = T.set_end_image(project, t, a.id, yellow)
    assert rel.startswith("ends/") and (project / rel).is_file()
    T.save(project, t)
    assert T.load(project).nodes[a.id].end_image == rel                       # kept in tree.json
    gen = FakeGenerator(frames=17, size=None)
    T.run_tree(gen, project, [b.id], log=lambda _m: None)
    assert is_colour(gen.calls[0]["end_rgb"], "yellow") and gen.calls[1]["end_rgb"] is None
    t = T.load(project)
    assert T.path_cuts(t, t.path_to(b.id)) == [(0, 0), (1, 0)]              # nothing cut at the end image
    ch = T.Checker(project, t)
    assert (ch.status(a.id), ch.status(b.id)) == ("done", "done")
    # another image: the clip (and what follows) is made again; the made copy stays for Undo changes
    Image.new("RGB", (40, 30), (0, 0, 255)).save(yellow)
    T.set_end_image(project, t, a.id, yellow)
    assert (project / rel).exists()
    ch = T.Checker(project, t)
    assert (ch.status(a.id), ch.status(b.id)) == ("changed", "changed")
    # a missing image is found before anything is made
    (project / t.nodes[a.id].end_image).unlink()
    T.save(project, t)
    with pytest.raises(cli.JobError, match="end image of clip"):
        cli.prepare_job(project, [b.id])
    # a video after the clip wins; set_end_image refuses then
    t = T.load(project)
    T.clear_end_image(project, t, a.id)
    assert t.nodes[a.id].end_image is None
    c = t.add(None, "x")
    T.set_end_image(project, t, c.id, yellow)
    t.add_video(c.id, str(source), 0.0, 1.0)
    assert T.end_image_used(t, t.nodes[c.id]) is None
    with pytest.raises(ValueError, match="ends at the video"):
        T.set_end_image(project, t, c.id, yellow)
    end = project / t.nodes[c.id].end_image
    t.remove(c.id, project)                                                  # its end image goes with it
    assert not end.exists()


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


def _main_hz(path: Path) -> float:
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le",
                          "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * 16000 / len(x))


def test_continue_the_real_videos_sound(project, tmp_path):
    """'Continue the real video's sound': the clip's sound is the file's sound right after the range (and after the
    clips before it), handed to the model as its soundtrack and put into the clip exactly."""
    src = tmp_path / "talk.mp4"                     # 700 Hz until 1 s (the range), 1200 Hz after (what comes next)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=320x180:r=30:d=5", "-f", "lavfi",
                    "-i", "aevalsrc=0.5*sin(2*PI*if(lt(t\\,1)\\,700\\,1200)*t):d=5", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    t = T.load(project)
    t.options["frames"] = 17
    v = t.add_video(None, str(src), 0.0, 1.0)
    a = t.add(v.id, "she keeps talking")
    b = t.add(a.id, "and talking")
    a.overrides = {"source_sound": True}
    b.overrides = {"source_sound": True, "guide": 5}
    other = t.add(None, "no video before")
    T.save(project, t)
    assert T.source_sound_span(t, a) == {"file": str(src), "start": round(1.0 - 1 / 24, 3), "secs": round(17 / 24, 3),
                                         "video": v.id}
    # b: after a's 17 frames (its first repeats the video's last one), its own first 5 frames repeat a's end
    assert T.source_sound_span(t, b)["start"] == round(1.0 + 16 / 24 - 5 / 24, 3)
    assert T.source_sound_span(t, other) is None
    gen = FakeGenerator(frames=17, size=None)
    T.run_tree(gen, project, [a.id], log=lambda _m: None)
    call = next(c for c in gen.calls if c["prompt"].startswith("she keeps"))
    assert call["settings"]["audio_paths"][-1].endswith("source_sound.wav")
    assert call["prompt"].endswith(T.SOURCE_SOUND_TEXT.format(k=1))
    t = T.load(project)
    rec = t.nodes[a.id].made["settings"]["source_sound"]
    assert rec["start"] == round(1.0 - 1 / 24, 3) and rec["file_key"] is not None
    clip = project / t.nodes[a.id].made["video"]
    assert abs(_main_hz(clip) - 1200) < 30                     # the recorded sound, not the generator's
    ch = T.Checker(project, t)
    assert ch.status(a.id) == "done"
    t.nodes[v.id].source["end"] = 1.5                         # another range: the clip's sound moves with it
    assert T.Checker(project, t).status(a.id) == "changed"
    t.nodes[v.id].source["end"] = 1.0
    t.nodes[b.id].overrides["frames"] = 24 * 15                # longer than the file has sound for
    assert any("ends at 0:05.0" in m for m in T.source_sound_problems(t, [b.id]))
    t.nodes[other.id].overrides = {"source_sound": True}
    assert T.source_sound_problems(t, [other.id]) == [f"clip #{other.id[1:]} continues a real video's sound, but "
                                                      "there is no real video before it"]
