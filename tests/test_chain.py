"""Chain logic with a fake generator (synthetic ffmpeg clips); no model, no GPU."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from video_chain import chain
from video_chain.__main__ import main
from fakes import COLOURS, FakeGenerator  # noqa: F401

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)

@pytest.fixture
def start(tmp_path):
    p = tmp_path / "start.jpg"
    Image.new("RGB", (96, 54), (250, 250, 250)).save(p)
    return p


def near(rgb, target, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


def run(gen, start, prompts, out, **kw):
    return chain.run_chain(gen, start, prompts, out, log=lambda _m: None, **kw)


def test_read_prompts(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# my chain\nfirst prompt\n\n   second prompt  \n#skip\nthird\n", encoding="utf-8")
    assert chain.read_prompts(f) == ["first prompt", "second prompt", "third"]


def test_canvas_size_matches_training_buckets():
    assert chain.canvas_size(1024, 1024) == (512, 512)
    assert chain.canvas_size(1920, 1080) == (672, 384)
    assert chain.canvas_size(1080, 1920) == (384, 672)
    assert chain.canvas_size(4000, 3000) == (576, 416)        # 4:3 keeps its ratio, about 512² pixels
    w, h = chain.canvas_size(5000, 500)                       # 10:1 is clamped to 5:2
    assert w % 32 == 0 and h % 32 == 0 and 2.3 < w / h <= 2.6
    assert chain.parse_size("640x384") == (640, 384)
    for bad in ("640", "650x384", "0x0", "axb"):
        with pytest.raises(ValueError):
            chain.parse_size(bad)


def test_chain_links_each_clip_to_the_last_frame_before(tmp_path, start):
    gen = FakeGenerator(frames=17)
    out = tmp_path / "out"
    result = run(gen, start, ["one", "two", "three"], out, seed=7)
    assert result == out / "chain.mp4"
    assert [c["start"] for c in gen.calls] == ["start.png", "01.png", "02.png"]
    assert [c["seed"] for c in gen.calls] == [7, 8, 9]
    assert near(gen.calls[0]["start_rgb"], (250, 250, 250))
    assert near(gen.calls[1]["start_rgb"], (255, 0, 0))          # last frame of clip 1 (red)
    assert near(gen.calls[2]["start_rgb"], (0, 128, 0))          # last frame of clip 2 (green)
    for name in ("01", "02", "03"):
        assert (out / f"{name}.mp4").exists() and (out / f"{name}.png").exists()
    info = chain.probe(out / "chain.mp4")
    assert info["frames"] == 17 + 16 + 16                        # repeated first frames left out
    assert info["audio"] and info["fps"] == 24
    rec = json.loads((out / "chain.json").read_text())
    assert rec["done"] and [c["prompt"] for c in rec["clips"]] == ["one", "two", "three"]


def test_keep_repeated_frames_and_same_seed(tmp_path, start):
    gen = FakeGenerator(frames=17)
    run(gen, start, ["a", "b"], tmp_path / "o", seed=3, same_seed=True, drop_first=False)
    assert [c["seed"] for c in gen.calls] == [3, 3]
    assert chain.probe(tmp_path / "o" / "chain.mp4")["frames"] == 34


def test_chain_sound_stays_in_sync(tmp_path, start):
    gen = FakeGenerator(frames=17)
    out = tmp_path / "out"
    run(gen, start, ["a", "b", "c"], out)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json",
                        str(out / "chain.mp4")], capture_output=True, text=True, check=True)
    d = {s["codec_type"]: float(s["duration"]) for s in json.loads(r.stdout)["streams"]}
    assert abs(d["video"] - 49 / 24) < 0.05 and abs(d["audio"] - d["video"]) < 0.06


def test_clip_without_sound_gets_silence(tmp_path, start):
    class Mixed(FakeGenerator):
        def generate(self, *a, **kw):
            self.audio = len(self.calls) != 1                    # clip 2 has no sound
            super().generate(*a, **kw)
    out = tmp_path / "out"
    run(Mixed(frames=17), start, ["a", "b", "c"], out)
    info = chain.probe(out / "chain.mp4")
    assert info["audio"] and info["frames"] == 49


def test_no_sound_at_all(tmp_path, start):
    out = tmp_path / "out"
    run(FakeGenerator(frames=17, audio=False), start, ["a", "b"], out)
    info = chain.probe(out / "chain.mp4")
    assert not info["audio"] and info["frames"] == 33


def test_resume_reuses_unchanged_clips(tmp_path, start):
    out = tmp_path / "out"
    gen = FakeGenerator(frames=17)
    run(gen, start, ["a", "b", "c"], out, settings={"steps": 28})
    assert len(gen.calls) == 3
    gen2 = FakeGenerator(frames=17)
    run(gen2, start, ["a", "b", "c"], out, settings={"steps": 28})
    assert gen2.calls == []                                      # nothing changed
    gen3 = FakeGenerator(frames=17)
    run(gen3, start, ["a", "B", "c"], out, settings={"steps": 28})
    assert [c["prompt"] for c in gen3.calls] == ["B", "c"]       # clip 2 changed, so clip 3's start changed too
    gen4 = FakeGenerator(frames=17)
    run(gen4, start, ["a", "B", "c"], out, settings={"steps": 20})
    assert len(gen4.calls) == 3                                  # other settings: everything again
    gen5 = FakeGenerator(frames=17)
    run(gen5, start, ["a", "B", "c"], out, settings={"steps": 20}, resume=False)
    assert len(gen5.calls) == 3


def test_shorter_chain_removes_old_clips(tmp_path, start):
    out = tmp_path / "out"
    run(FakeGenerator(frames=17), start, ["a", "b", "c"], out)
    run(FakeGenerator(frames=17), start, ["a", "b"], out)
    assert not (out / "03.mp4").exists() and not (out / "03.png").exists()
    assert chain.probe(out / "chain.mp4")["frames"] == 33


def test_generator_must_write_its_files(tmp_path, start):
    class Lazy:
        def generate(self, *a, **kw):
            pass
    with pytest.raises(RuntimeError, match="did not write"):
        run(Lazy(), start, ["a"], tmp_path / "o")


def test_last_frame_is_exact(tmp_path):
    gen = FakeGenerator(frames=17)
    v, png = tmp_path / "v.mp4", tmp_path / "v.png"
    gen.generate("x", _img(tmp_path), v, png, 1)
    assert near(Image.open(png).convert("RGB").getpixel((10, 10)), (255, 0, 0))


def _img(tmp_path):
    p = tmp_path / "i.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(p)
    return p


def test_cli_rejects_bad_input_before_loading_anything(tmp_path, start, capsys):
    p = tmp_path / "p.txt"
    p.write_text("a prompt\n")
    assert main([str(start), str(p), "--out", str(tmp_path / "o"), "--frames", "60"]) == 2
    assert "17n+5" in capsys.readouterr().err
    empty = tmp_path / "e.txt"
    empty.write_text("# nothing\n\n")
    assert main([str(start), str(empty), "--out", str(tmp_path / "o")]) == 2
    assert main([str(tmp_path / "missing.png"), str(p), "--out", str(tmp_path / "o")]) == 2
    assert main([str(start), str(p), "--out", str(tmp_path / "o"), "--size", "500x500"]) == 2


def test_recorded_cast_names_the_kind_of_a_non_person():
    kw = dict(model="m", lora=None, strength=1.0, frames=73, steps=28, width=672, height=384, audio=True,
              refs=[{"file": "refs/ref1.png", "sha1": "x"}])
    s = chain.settings_dict(cast=[{"name": "Anna", "desc": "woman"}], **kw)
    assert s["cast"] == [{"name": "Anna", "desc": "woman"}]                    # a person: as before
    s = chain.settings_dict(cast=[{"name": "Anna", "desc": "woman"}, {"name": "Hall", "desc": "", "kind": "place"}],
                            **kw)
    assert s["cast"][1] == {"name": "Hall", "desc": "", "kind": "place"}
