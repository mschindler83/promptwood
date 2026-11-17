"""VideoChainLanczosScale: ComfyUI's lanczos resize (ImageScale, upscale_method "lanczos", crop disabled) with the
very same arithmetic per frame (comfy.utils.lanczos: x 255 in float32, clipped, truncated to uint8, PIL's LANCZOS,
back to float32 / 255), so the frames are bit for bit the same, but many frames at once on the CPU's cores (PIL and
numpy release the GIL) and written into one buffer instead of a list of new arrays stacked at the end.

ComfyUI's loop runs one frame after another on one core: on a RunPod H100 pod it took 728 s of a 1035 s SeedVR2
job for one clip at 1568x2304 (2026-11-13), while the GPU waited.

Needs numpy and Pillow only (torch only in the node wrapper). This file is a whole ComfyUI node package by itself
(the app copies it to a pod as custom_nodes/video_chain_scale/__init__.py); video_chain_nodes registers it too.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image


_BUFFERS = threading.local()                      # per thread: the working arrays, reused frame after frame


def _buffer(name: str, shape: tuple, dtype) -> np.ndarray:
    b = getattr(_BUFFERS, name, None)
    if b is None or b.shape != shape or b.dtype != dtype:
        b = np.empty(shape, dtype=dtype)
        setattr(_BUFFERS, name, b)
    return b


def lanczos_frame(frame: np.ndarray, width: int, height: int, out: np.ndarray) -> None:
    """One frame (h, w, c float32 in 0..1, c = 1, 3 or 4) resized into out (height, width, c float32), as
    comfy.utils.lanczos does it: np.clip(255. * x, 0, 255).astype(np.uint8), PIL LANCZOS,
    np.array(im).astype(np.float32) / 255.0, the same float32 operations, into reused buffers (no new arrays per
    frame but PIL's own image)."""
    f32 = _buffer("f32", frame.shape, np.float32)
    np.multiply(frame, np.float32(255.0), out=f32)
    np.clip(f32, 0, 255, out=f32)
    u8 = _buffer("u8", frame.shape, np.uint8)
    np.copyto(u8, f32, casting="unsafe")              # (= astype(np.uint8): the same truncating cast)
    im = Image.fromarray(u8[..., 0] if u8.shape[-1] == 1 else u8)   # (comfy squeezes grayscale: PIL "L")
    im = im.resize((width, height), resample=Image.Resampling.LANCZOS)
    r32 = _buffer("r32", out.shape, np.float32)
    np.copyto(r32, np.asarray(im).reshape(out.shape), casting="unsafe")   # uint8 -> float32, exact
    np.divide(r32, np.float32(255.0), out=out)


def workers() -> int:
    """Threads to use: the process's thread setting (OMP_NUM_THREADS: on a pod, its share of the host's CPUs),
    else the CPUs this process may run on; at most 64."""
    try:
        n = int(os.environ.get("OMP_NUM_THREADS") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        try:
            n = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n = os.cpu_count() or 1
    return max(1, min(64, n))


def lanczos_frames(frames: np.ndarray, width: int, height: int, out: np.ndarray | None = None,
                   threads: int | None = None) -> np.ndarray:
    """All frames (n, h, w, c float32) resized to (n, height, width, c), in parallel, into out (made if None)."""
    n, _h, _w, c = frames.shape
    if out is None:
        out = np.empty((n, height, width, c), dtype=np.float32)
    threads = threads or workers()
    if threads == 1 or n == 1:
        for k in range(n):
            lanczos_frame(frames[k], width, height, out[k])
        return out
    with ThreadPoolExecutor(max_workers=min(threads, n)) as pool:
        for f in [pool.submit(lanczos_frame, frames[k], width, height, out[k]) for k in range(n)]:
            f.result()                                 # (raises what a frame raised)
    return out


class VideoChainLanczosScale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "width": ("INT", {"default": 1920, "min": 1, "max": 16384, "step": 1}),
                             "height": ("INT", {"default": 1080, "min": 1, "max": 16384, "step": 1})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"
    CATEGORY = "image/upscaling"
    DESCRIPTION = ("Lanczos resize, the same frames as Upscale Image (lanczos, crop disabled), on all CPU cores")

    def scale(self, image, width, height):
        import torch
        src = image.detach().to("cpu", torch.float32).contiguous().numpy()
        out = torch.empty((src.shape[0], int(height), int(width), src.shape[3]), dtype=torch.float32)
        lanczos_frames(src, int(width), int(height), out.numpy())
        return (out.to(image.device, image.dtype),)


NODE_CLASS_MAPPINGS = {"VideoChainLanczosScale": VideoChainLanczosScale}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoChainLanczosScale": "Lanczos Scale, all cores (Promptwood)"}
