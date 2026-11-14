"""chain.py file handling: kept-aside clips, stale frames, the hash cache (synthetic data)."""
import shutil
import subprocess
from pathlib import Path

import pytest
from fakes import FakeGenerator
from PIL import Image

from video_chain import chain

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


@pytest.fixture
def start(tmp_path):
    p = tmp_path / "start.jpg"
    Image.new("RGB", (96, 54), (250, 250, 250)).save(p)
    return p


def test_run_chain_does_not_overwrite_a_clip_it_could_not_move_aside(start, tmp_path, monkeypatch):
    out = tmp_path / "out"
    chain.run_chain(FakeGenerator(frames=17), start, ["a"], out, log=lambda _m: None)
    video = out / "01.mp4" if (out / "01.mp4").exists() else sorted(out.glob("*.mp4"))[0]
    before = video.read_bytes()
    monkeypatch.setattr(chain, "keep_aside", lambda *_a, **_k: False)      # open in a player on a share
    with pytest.raises(RuntimeError, match="could not be moved"):
        chain.run_chain(FakeGenerator(frames=17), start, ["b"], out, log=lambda _m: None)
    assert video.read_bytes() == before


def test_frame_at_past_the_end_does_not_pass_an_old_png(tmp_path):
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x48:r=24:d=0.5", "-pix_fmt",
                    "yuv420p", str(clip)], check=True)
    png = tmp_path / "f.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(png)                       # an older picture under that name
    with pytest.raises(RuntimeError, match="no frame"):
        chain.frame_at(clip, 500, png)


def test_the_hash_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "_SHA_CACHE_MAX", 5)
    chain._SHA_CACHE.clear()
    for k in range(20):
        f = tmp_path / f"{k}.bin"
        f.write_bytes(bytes([k]) * 10)
        chain.file_signature(f)
    assert len(chain._SHA_CACHE) <= 5
