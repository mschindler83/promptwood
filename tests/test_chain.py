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


def test_joined_seconds_reads_the_container(tmp_path):
    """The length of a joined video without decoding it (Play path on a joined path must not count frames)."""
    import subprocess
    v = tmp_path / "a.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24", "-frames:v", "48",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    assert abs(chain.joined_seconds(v) - 2.0) < 1e-6 and chain.probe(v)["frames"] == 48


def _clip(path: Path, hz: int | None, frames: int = 48, amp: float = 0.5) -> Path:
    """A 24 fps clip of `frames` frames, with a stereo sine of hz at amplitude amp (None: no sound). Stereo, as
    ffmpeg's mono-to-stereo upmix would lower the level by 3 dB."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x64:rate=24:d={frames / 24:.6f}"]
    if hz:
        e = f"{amp}*sin(2*PI*{hz}*t)"
        cmd += ["-f", "lavfi", "-i", f"aevalsrc={e}|{e}:s=48000:d={frames / 24:.6f}", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _wav(path: Path, hz: int, secs: float, amp: float = 0.5) -> Path:
    e = f"{amp}*sin(2*PI*{hz}*t)"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"aevalsrc={e}|{e}:s=48000:d={secs}",
                    "-c:a", "pcm_s16le", str(path)], check=True)
    return path


def _pcm(path: Path, start: float = 0.0, secs: float | None = None):
    """The sound's first channel as float samples at 48 kHz (start / secs: a part of it); not a downmix, which
    would add up the channels."""
    import numpy as np
    cut = ["-ss", f"{start}"] + (["-t", f"{secs}"] if secs else [])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), *cut, "-vn", "-af", "pan=mono|c0=c0", "-ar",
                          "48000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def _band(x, hz: float) -> float:
    """The amplitude of the sine at hz in x (about 0.5 for a 0.5 sine)."""
    import numpy as np
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / (len(x) / 4)
    k = round(hz * len(x) / 48000)
    return float(spec[k - 3:k + 4].max())


def _video_stream(path: Path) -> str:
    """The video stream's codec, duration and frame count as ffprobe states them (to see it was copied)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=duration,nb_frames,codec_name", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out


def _audio_streams(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.split()
    return len(out)


def test_mix_audio_keeps_both_sounds(tmp_path):
    """mix_audio: the clip's own 440 Hz and the given 1000 Hz are both heard at their own level, in one audio
    stream; the picture is copied (same frames, same length); padded to the clip's length like replace_audio."""
    v = _clip(tmp_path / "clip.mp4", 440)
    before = _video_stream(v)
    chain.mix_audio(v, _wav(tmp_path / "given.wav", 1000, 2.0, amp=0.25))
    assert _video_stream(v) == before and chain.probe(v)["frames"] == 48 and _audio_streams(v) == 1
    assert abs(chain.audio_seconds(v) - 2.0) < 0.05
    x = _pcm(v, 0.2, 1.6)
    assert abs(_band(x, 440) - 0.5) < 0.06 and abs(_band(x, 1000) - 0.25) < 0.04     # not normalised (halved)
    v2 = _clip(tmp_path / "short.mp4", 440)
    chain.mix_audio(v2, _wav(tmp_path / "short.wav", 1000, 0.8))                       # shorter: silence after it
    assert abs(chain.audio_seconds(v2) - 2.0) < 0.05 and chain.probe(v2)["frames"] == 48
    assert _band(_pcm(v2, 0.1, 0.6), 1000) > 0.4 and _band(_pcm(v2, 1.1, 0.8), 1000) < 0.02
    assert _band(_pcm(v2, 1.1, 0.8), 440) > 0.4                                        # the clip's own goes on
    v3 = _clip(tmp_path / "long.mp4", 440)
    chain.mix_audio(v3, _wav(tmp_path / "long.wav", 1000, 5.0))                        # longer: cut at the clip's end
    assert abs(chain.audio_seconds(v3) - 2.0) < 0.05 and chain.probe(v3)["frames"] == 48


def test_mix_audio_never_clips_and_takes_a_level(tmp_path):
    """Two full-scale sounds together stay under the limit; the level changes only the given sound's share."""
    import numpy as np
    v = _clip(tmp_path / "loud.mp4", 440, amp=1.0)
    chain.mix_audio(v, _wav(tmp_path / "loud.wav", 1000, 2.0, amp=1.0))
    x = _pcm(v)
    assert np.abs(x).max() < 1.0 and np.abs(x).max() > 0.6          # limited, not silenced
    assert _band(_pcm(v, 0.2, 1.6), 440) > 0.25 and _band(_pcm(v, 0.2, 1.6), 1000) > 0.25
    a, b = _clip(tmp_path / "a.mp4", 440, amp=0.25), _clip(tmp_path / "b.mp4", 440, amp=0.25)
    given = _wav(tmp_path / "g.wav", 1000, 2.0, amp=0.25)
    chain.mix_audio(a, given)
    chain.mix_audio(b, given, level_db=-12)
    xa, xb = _pcm(a, 0.2, 1.6), _pcm(b, 0.2, 1.6)
    assert abs(_band(xa, 440) - _band(xb, 440)) < 0.02                              # the clip's own: unchanged
    assert abs(20 * np.log10(_band(xb, 1000) / _band(xa, 1000)) + 12) < 1.0          # the given one: -12 dB


def test_mix_audio_without_a_sound_of_its_own(tmp_path):
    """A clip without sound (the project's sound off) gets just the given sound, as replace_audio would."""
    v = _clip(tmp_path / "mute.mp4", None)
    chain.mix_audio(v, _wav(tmp_path / "g.wav", 1000, 1.0, amp=0.5))
    assert _audio_streams(v) == 1 and chain.probe(v)["frames"] == 48 and abs(chain.audio_seconds(v) - 2.0) < 0.05
    assert abs(_band(_pcm(v, 0.1, 0.8), 1000) - 0.5) < 0.06


def test_mix_is_recorded_only_with_mix():
    """The overlay's record stays as before; a mix adds "mix" (and its level only when not 0)."""
    common = dict(model="m", lora=None, strength=1.0, frames=48, steps=8, width=64, height=64, audio=True)
    ss = {"file": "", "start": 0.0, "secs": 2.0, "overlay": True}
    plain = chain.settings_dict(**common, source_sound=ss)["source_sound"]
    assert plain == {"file": "", "start": 0.0, "secs": 2.0, "file_key": None, "overlay": True}
    mixed = chain.settings_dict(**common, source_sound={**ss, "mix": True})["source_sound"]
    assert mixed == {**plain, "mix": True}
    assert chain.settings_dict(**common, source_sound={**ss, "mix": True, "mix_level": -6})["source_sound"] == \
        {**plain, "mix": True, "mix_level": -6.0}
    assert "mix" not in chain.settings_dict(**common, source_sound={**ss, "overlay": False, "mix": True})["source_sound"]
