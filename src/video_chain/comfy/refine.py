"""The 2K refine (Open-H3-Regenerate-2K): the workflow, its pickers and ComfyGenerator.refine (RefineMixin)."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from .. import media
from ..chain import canvas_size
from .errors import ComfyError
from .ffutil import with_audio
from .loras import is_ic_lora
from .video import AUDIO_VAE, TEXT_ENCODER, UNET, VIDEO_VAE
from .wfb import (FPS, REFINE_SAVE_ID, SUBFOLDER, combo_options, loader_clip, loader_unet, loader_vae, lora_chain,
                  save_video)

# ----------------------------------------------------------------------------- 2K refine (Open-H3-Regenerate-2K)

REFINE_RELEASE = (0.909, 0.857, 0.667, 0.0)  # the release's 3 Euler steps, re-noised from 0.909, no CFG
# How much noise the refine adds before its 3 steps: the release's schedule scaled down to start at this sigma.
# Measured on a 1.6 s face clip at 1080p (RTX 4090): from 0.909 (the release) the refine repaints faces (closed eyes
# drawn half open with lashes, eye and hair colour changed, frizz and grain); 0.8 keeps the person, crisp; 0.7 calm
# and clean; 0.6 softer, little more than a scale-up. Grain on flat areas: 1.42 at 0.909, 1.01-1.07 at 0.6-0.8.
REFINE_STRENGTHS = {"soft": 0.6, "natural": 0.7, "crisp": 0.8}
REFINE_DEFAULT = "natural"


def refine_sigmas(strength: str = REFINE_DEFAULT) -> str:
    """The refine's sigmas for a strength (soft / natural / crisp)."""
    start = REFINE_STRENGTHS[strength]
    return ", ".join(f"{s * start / REFINE_RELEASE[0]:.4g}" for s in REFINE_RELEASE)


SOL_TAU = 1.3                               # Sol-Attn threshold (ComfyUI's default); measured at 2K on an RTX 4090:
                                            # 2.2x faster refine, PSNR 33 dB against dense attention
# The refine pass at 2K runs out of memory on a 24 GB GPU for long clips (measured on an RTX 4090: 10 s = 880k tokens
# fails in the model's forward pass). Longer clips are refined in MiniMax H3 context windows (ComfyUI core): windows
# of at most this many tokens, overlapping by CONTEXT_OVERLAP frames and blended.
WINDOW_TOKENS = 530_000                    # a 124-frame window at 2528x1440 (37 latent frames = 526k)
CONTEXT_OVERLAP = 39


def latent_frames(frames: int) -> int:
    """H3's latent length for a frame count (ComfyUI's video_latent_t: 5 latents per 17 frames, plus 2; frame counts
    off the 17n+5 grid are rounded up first, as ComfyUI does)."""
    frames = int(frames)
    while frames > 5 and frames % 17 != 5:
        frames += 1
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def context_window(frames: int, width: int, height: int) -> int | None:
    """The context window (frames, 17n+5) for refining a clip of this size, or None when it fits in one pass."""
    per = (int(width) // 16) * (int(height) // 16)
    if latent_frames(frames) * per <= WINDOW_TOKENS:
        return None
    chunks = (WINDOW_TOKENS // per - 2) // 5           # 17-frame chunks whose latent fits
    length = max(56, chunks * 17 + 5)                   # at least 56 frames (overlap 39 + 17)
    return length if length < frames else None


# Sol-Attn's other settings (ComfyUI's BlockSparseAttention node) and the tiles the refined frames are decoded in
SPARSE_SETTINGS = {"start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288,
                   "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}
REFINE_DECODE_TILES = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}

REFINE_NODES = ("LatentUpscaleModelLoader", "MiniMaxH3LatentUpscaleWithModel", "LTXVConcatAVLatent", "VAEEncodeAudio",
                "ManualSigmas", "VAEDecodeTiled", "LoadVideo", "GetVideoComponents", "MiniMaxH3ContextWindows",
                "MiniMaxH3ImageToVideo", "CreateVideo", "SaveVideo", "VAEEncode")


def pick_refine_files(loras: list[str], upscalers: list[str]) -> dict:
    """The Open-H3-Regenerate-2K refiner LoRA (highest rank) and the H3 latent upscaler (bf16 preferred)."""
    ref = [lo for lo in loras if ("regenerate_2k" in lo.lower() or "regenerate-2k" in lo.lower()) and not is_ic_lora(lo)]

    def rank(n: str) -> int:
        m = re.search(r"rank_?(\d+)", n.lower())
        return int(m.group(1)) if m else 0
    ups = [u for u in upscalers if "minimax_h3" in u.lower() or "h3_latent" in u.lower()]
    ups.sort(key=lambda u: ("bf16" not in u.lower(), u))
    return {"refiner": max(ref, key=rank) if ref else None, "upscaler": ups[0] if ups else None}


def build_refine_workflow(*, prompt: str, draft: str, seed: int, width: int, height: int, frames: int,
                          loras: list[tuple[str, float]], refiner: str, upscaler: str, prefix: str,
                          first_frame: str | None = None, last_frame: str | None = None,
                          sparse: float | None = None, strength: str = REFINE_DEFAULT) -> dict:
    """Re-render a finished clip (the uploaded draft video, with sound) at width x height: its video and sound are
    encoded into H3's AV latent, the video upscaled with the latent upscaler, then refined in 3 steps with the
    refiner LoRA (and the clip's own LoRAs, as it was made); the clip's own sound is kept.
    first_frame / last_frame (uploaded images): pinned as keyframes, so a refined clip starts exactly where the
    refined clip before it hands over (and ends at a real video's refined first frame).
    sparse: Sol-Attn block-sparse attention at this threshold (ComfyUI's BlockSparseAttention; not exact).
    strength: how much is drawn anew (REFINE_STRENGTHS)."""
    wf: dict = {
        "unet": loader_unet(UNET),
        "clip": loader_clip(TEXT_ENCODER, "minimax"),
        "vae": loader_vae(VIDEO_VAE),
        "audio_vae": loader_vae(AUDIO_VAE),
        "draft": {"class_type": "LoadVideo", "inputs": {"file": draft}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["draft", 0]}},
        "venc": {"class_type": "VAEEncode", "inputs": {"pixels": ["parts", 0], "vae": ["vae", 0]}},
        "aenc": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["parts", 1], "vae": ["audio_vae", 0]}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["venc", 0], "audio_latent": ["aenc", 0]}},
        "upmodel": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": upscaler}},
        "up": {"class_type": "MiniMaxH3LatentUpscaleWithModel",
               "inputs": {"model": ["upmodel", 0], "samples": ["av", 0], "width": int(width), "height": int(height)}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
                            "height": int(height), "length": int(frames),
                            **({"first_frame": ["first", 0]} if first_frame else {}),
                            **({"last_frame": ["last", 0]} if last_frame else {})}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": refine_sigmas(strength)}},
    }
    if first_frame:
        wf["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
    if last_frame:
        wf["last"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
    model = lora_chain(wf, ["unet", 0], list(loras) + [(refiner, 1.0)])
    window = context_window(frames, width, height)
    if window:                                  # a long clip: refined in overlapping windows (memory)
        wf["windows"] = {"class_type": "MiniMaxH3ContextWindows",
                         "inputs": {"model": model, "context_length": int(window), "context_overlap": CONTEXT_OVERLAP,
                                    "fuse_method": "pyramid", "freenoise": True, "split_conds_to_windows": False}}
        model = ["windows", 0]
    if sparse:
        wf["sparse"] = {"class_type": "BlockSparseAttention",
                        "inputs": {"model": model, "selection": "sol-attn", "selection.tau": float(sparse),
                                   **SPARSE_SETTINGS}}
        model = ["sparse", 0]
    wf["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["up", 0]}}
    wf["decode"] = {"class_type": "VAEDecodeTiled",
                    "inputs": {"samples": ["sample", 0], "vae": ["vae", 0], **REFINE_DECODE_TILES}}
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": FPS,
                                                          "audio": ["parts", 1]}}
    wf[REFINE_SAVE_ID] = save_video(prefix)
    return wf


def refine_size(width: int, height: int, target: str) -> tuple[int, int]:
    """The refined size for a clip of width x height: '1080p' (~1920x1080 pixels) or '2k' (~2560x1440), the
    clip's aspect ratio, multiples of 32."""
    return canvas_size(width, height, {"1080p": 1440, "2k": 1920}[target])



class RefineMixin:
    """ComfyGenerator.refine (needs the generator's upload / _run_workflow / _save ...)."""

    def refine(self, draft: Path, prompt: str, out: Path, seed: int, size: tuple[int, int], frames: int,
               loras: list[tuple[str, float]], first_frame: Path | None = None, last_frame: Path | None = None,
               sparse: float | None = None, strength: str = REFINE_DEFAULT) -> None:
        """Re-render one finished clip at size (see build_refine_workflow)."""
        self._begin_job()
        self._last_seedvr = False
        for node in REFINE_NODES:
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node (update ComfyUI: the H3 2K nodes are in its "
                                 "master since 2026-10-28)")
        if self.last_hyperflow is not False:
            # a HyperFlow-patched H3 model (or one this process does not know about) must not refine: load it fresh
            if self.free():
                self.last_hyperflow = False
        if sparse and not self._object_info("BlockSparseAttention"):
            raise ComfyError("this ComfyUI has no BlockSparseAttention node (Sol-Attn; update ComfyUI)")
        req = self._object_info("LatentUpscaleModelLoader").get("input", {}).get("required", {})
        files = pick_refine_files(self.lora_names(refresh=True), combo_options(req.get("model_name")))
        if not files["refiner"]:
            raise ComfyError("ComfyUI lists no Open-H3-Regenerate-2K refiner LoRA "
                             "(minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors in a LoRA folder)")
        if not files["upscaler"]:
            raise ComfyError("ComfyUI lists no H3 latent upscaler (minimax_h3_latent_upscaler_3d_conv_v1_bf16"
                             ".safetensors in models/latent_upscale_models)")
        names = []
        for path, weight in loras:
            name = self.name_for(path)
            if name is None:
                raise ComfyError(f"{Path(path).name} is not in ComfyUI's LoRA folders")
            names.append((name, float(weight)))
        uploads: list[tuple[str, Path]] = []
        with media.scratch("refine") as tmp:
            src = with_audio(Path(draft), Path(tmp) / "draft.mp4")
            try:
                name = self.upload(src)
                uploads.append((name, src))
                pins = {}
                for key, img in (("first_frame", first_frame), ("last_frame", last_frame)):
                    if img is not None:
                        pins[key] = self.upload(Path(img))
                        uploads.append((pins[key], Path(img)))
                prefix = f"{SUBFOLDER}/refine_{uuid.uuid4().hex[:12]}"
                wf = build_refine_workflow(prompt=prompt, draft=name, seed=seed, width=size[0], height=size[1],
                                           frames=frames, loras=names, refiner=files["refiner"],
                                           upscaler=files["upscaler"], prefix=prefix, sparse=sparse,
                                           strength=strength, **pins)
                entry = self._run_workflow(wf)
                video = self._first_file((entry.get("outputs") or {}).get(REFINE_SAVE_ID))
                if video is None:
                    raise ComfyError("ComfyUI finished without the refined clip")
                self._save(video, Path(out))
                if self.cleanup:
                    self._remove(self.output_dir, video, Path(out))
            finally:
                self._drop_uploads(uploads)
