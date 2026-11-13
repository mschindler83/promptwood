"""video-chain's lanczos node: the very same frames as ComfyUI's ImageScale (lanczos, crop disabled), bit for bit."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

NODES = Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes"
sys.path.insert(0, str(NODES))
import scale as S  # noqa: E402


def comfy_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """comfy.utils.lanczos (ComfyUI v0.45.3) for one frame, its numpy steps verbatim:
    Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)), resize LANCZOS,
    np.array(image).astype(np.float32) / 255.0."""
    image = frame[..., 0] if frame.shape[-1] == 1 else frame
    im = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))
    im = im.resize((width, height), resample=Image.Resampling.LANCZOS)
    t = np.array(im).astype(np.float32) / 255.0
    return t if t.ndim == 3 else t[..., None]


@pytest.mark.parametrize("shape,size", [((7, 37, 53, 3), (101, 80)),      # up, odd sizes
                                        ((5, 120, 90, 3), (41, 59)),      # down
                                        ((3, 30, 20, 4), (60, 45)),       # RGBA
                                        ((2, 30, 20, 1), (31, 17))])      # grayscale
def test_same_frames_as_comfy_bit_for_bit(shape, size):
    rng = np.random.default_rng(7)
    frames = rng.random(shape, dtype=np.float32) * 1.2 - 0.1           # (outside 0..1 too: clipped the same way)
    want = np.stack([comfy_frame(f, *size) for f in frames])
    for threads in (1, 4):
        got = S.lanczos_frames(frames, *size, threads=threads)
        assert got.dtype == np.float32 and got.shape == want.shape
        assert np.array_equal(got, want)
    out = np.full((shape[0], size[1], size[0], shape[3]), -1, dtype=np.float32)
    assert S.lanczos_frames(frames, *size, out=out, threads=3) is out and np.array_equal(out, want)


def test_threads_from_the_process_setting(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "23")
    assert S.workers() == 23
    monkeypatch.setenv("OMP_NUM_THREADS", "500")
    assert S.workers() == 64
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert 1 <= S.workers() <= 64


TTS_PY = Path(__file__).resolve().parents[1] / ".venv-tts" / "bin" / "python"   # (a venv with torch)


@pytest.mark.skipif(not TTS_PY.is_file(), reason="needs a Python with torch (.venv-tts)")
def test_the_node_equals_comfy_imagescale_with_torch(tmp_path):
    """The whole node against ComfyUI's ImageScale path, torch layout included: ImageScale.upscale's movedims and
    comfy.utils.lanczos (both copied verbatim from ComfyUI v0.45.3) on synthetic frames."""
    script = tmp_path / "check.py"
    script.write_text(f'''
import sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, {str(NODES)!r})
import scale as S
def lanczos(samples, width, height):
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    images = [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(t).movedim(-1, 0) if (t := np.array(image).astype(np.float32) / 255.0).ndim == 3 else torch.from_numpy(t) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)
def imagescale(image, width, height):                  # ImageScale.upscale, crop disabled -> common_upscale -> lanczos
    samples = image.movedim(-1, 1)
    return lanczos(samples, width, height).movedim(1, -1)
torch.manual_seed(3)
for shape, (w, h) in (((9, 61, 47, 3), (130, 170)), ((4, 200, 150, 3), (77, 99))):
    img = torch.rand(shape)
    want = imagescale(img, w, h)
    with torch.inference_mode():                           # (as ComfyUI runs its nodes)
        got, = S.VideoChainLanczosScale().scale(img, w, h)
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape)
    assert torch.equal(got, want), (got - want).abs().max()
print("same")
''')
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    r = subprocess.run([str(TTS_PY), "-I", str(script)], capture_output=True, text=True, timeout=120, env=env,
                       check=False)
    assert r.returncode == 0 and r.stdout.strip() == "same", r.stderr[-2000:]
