"""ComfyGenerator guards that had no test: _remove stays inside video-chain's folder, _join_pieces checks frames."""
import shutil
import subprocess

import pytest
from fake_comfy import FakeComfy

from video_chain import comfy as C

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def test_remove_never_leaves_the_video_chain_folder(tmp_path):
    s = FakeComfy(tmp_path / "comfy")
    try:
        gen = C.ComfyGenerator(s.url, comfy_dir=s.root)
        (s.root / "main.py").write_text("x")
        outside = s.root / "output" / "precious.png"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"keep")
        mine = s.root / "output" / "video-chain" / "a.png"
        mine.parent.mkdir(parents=True, exist_ok=True)
        mine.write_bytes(b"keep")
        local = tmp_path / "local.png"
        local.write_bytes(b"keep")
        for sub in ("..", "video-chain/../..", "../output", "/etc"):
            gen._remove(s.root / "output", {"filename": "precious.png", "subfolder": sub}, local)
            gen._remove(s.root / "output", {"filename": "../precious.png", "subfolder": "video-chain"}, local)
        assert outside.exists()
        gen._remove(s.root / "output", {"filename": "a.png", "subfolder": "video-chain"}, local)
        assert not mine.exists()                                    # (and the real one goes)
    finally:
        s.close()


def test_join_pieces_refuses_a_wrong_frame_count(tmp_path):
    from video_chain.comfy.seedvr import _join_pieces

    def clip(name, n):
        p = tmp_path / name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24", "-frames:v",
                        str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True)
        return p
    a, b, draft = clip("a.mp4", 20), clip("b.mp4", 20), clip("d.mp4", 30)
    _join_pieces([a, b], [(0, 20), (10, 20)], draft, tmp_path / "ok.mp4", 30)       # 20 + 20 - 10 overlap = 30
    with pytest.raises(C.ComfyError, match="instead of"):
        _join_pieces([a, b], [(0, 20), (10, 20)], draft, tmp_path / "bad.mp4", 45)
    assert not (tmp_path / "bad.mp4").exists()
    with pytest.raises(C.ComfyError, match="overlap"):
        _join_pieces([a, b], [(0, 20), (20, 20)], draft, tmp_path / "no.mp4", 40)


def test_one_frame_rate_everywhere():
    from video_chain import chain, media, swap
    from video_chain.comfy import wfb
    assert chain.FPS == media.DEFAULT_FPS == wfb.FPS == 24
    import inspect
    assert inspect.signature(swap.write_mask).parameters["fps"].default == chain.FPS
