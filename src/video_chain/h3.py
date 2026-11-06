"""MiniMax-H3 image-to-video through ai-toolkit, the same code path the trainer uses for its samples.

Runs inside ai-toolkit's venv (it needs torch, the H3 extension and the model files in ai-toolkit's MODELS_PATH).

How the LoRA is applied: H3's load_model loads `assistant_lora_path` as a live (never merged) LoRA module, because
the transformer is pre-quantized. ai-toolkit's sampler switches that module off for samples, unless
`inference_lora_path` is set, which switches it back on. So the LoRA goes into both fields: loaded like the
training adapter, active while generating. The training adapter itself is NOT loaded; it is only for training.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_AITK = Path("/path/to/ai/ai-toolkit")
DEFAULT_MODEL = "Comfy-Org/MiniMax-H3"


def model_config_dict(model: str = DEFAULT_MODEL, lora: str | None = None, offload: float = 0.9,
                      low_vram: bool = True, audio: bool = True) -> dict:
    """The model block of step 6's training config, without the training adapter, plus the LoRA to apply."""
    cfg = {"name_or_path": model, "arch": "minimax_h3", "quantize": True, "qtype": "convrot8",
           "quantize_te": True, "qtype_te": "nvfp4", "low_vram": low_vram, "dtype": "bf16",
           "model_kwargs": {"sample_audio": bool(audio)}}
    if offload > 0:
        cfg.update({"layer_offloading": True, "layer_offloading_transformer_percent": float(offload),
                    "layer_offloading_text_encoder_percent": 1.0})
    if lora:
        cfg["assistant_lora_path"] = str(lora)
        cfg["inference_lora_path"] = str(lora)
    return cfg


def free_gpu_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:  # noqa: BLE001
        return None


class H3Generator:
    def __init__(self, *, width: int, height: int, frames: int = 73, steps: int = 28, lora: Path | None = None,
                 strength: float = 1.0, model: str = DEFAULT_MODEL, offload: float = 0.9, low_vram: bool = True,
                 audio: bool = True, aitk: Path = DEFAULT_AITK, device: str = "cuda:0", min_free_gb: float = 18.0,
                 log=print, progress=None):
        self.width, self.height, self.frames, self.steps = width, height, frames, steps
        self.log = log
        aitk = Path(aitk).resolve()
        if not (aitk / "toolkit").is_dir():
            raise FileNotFoundError(f"{aitk} is not an ai-toolkit folder")
        if lora is not None and not Path(lora).is_file():
            raise FileNotFoundError(lora)

        free = free_gpu_gb()
        if free is not None and free < min_free_gb:
            raise RuntimeError(f"only {free:.1f} GB GPU memory free (need {min_free_gb:g} GB, see --min-free-gb). "
                               "Is a training or another model still running?")

        # ai-toolkit resolves its extensions and MODELS_PATH relative to its own folder
        sys.path.insert(0, str(aitk))
        os.chdir(aitk)
        import torch  # noqa: F401
        from safetensors.torch import load_file
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class

        cfg = ModelConfig(**model_config_dict(model, str(lora) if lora else None, offload, low_vram, audio))
        cls = get_model_class(cfg)
        self.sd = cls(device=device, model_config=cfg, dtype="bf16", noise_scheduler=cls.get_train_scheduler())
        log("loading MiniMax-H3 (this takes a few minutes)")
        self.sd.load_model()

        if lora is not None:
            net = self.sd.assistant_lora
            weights = self.sd.convert_lora_weights_before_load(load_file(str(lora)))
            unknown = net.load_weights(weights)          # same weights again; returns what did not fit
            if unknown:
                raise RuntimeError(f"{len(unknown)} of {len(weights)} LoRA weights do not fit this model "
                                   f"(e.g. {next(iter(unknown))}); is it an H3 LoRA for this checkpoint?")
            if not any(float(p.detach().abs().sum()) > 0 for n, p in net.named_parameters() if "lora_up" in n):
                raise RuntimeError("the LoRA loaded as all zeros")
            net.multiplier = float(strength)
            net._update_torch_multiplier()
            log(f"LoRA {Path(lora).name} applied (strength {strength:g}, {len(weights)} weights)")

        if progress is not None:          # H3's sampler reports every denoising step through this hook
            self.sd.sample_step_hook = lambda idx, num, _latents: progress(idx + 1, num)

        # keep the decoded frames: the last one becomes the next start image without a trip through H.264
        import extensions_built_in.diffusion_models.minimax_h3.minimax_h3 as h3mod
        self._h3mod = h3mod
        self._orig_save = h3mod.new_save_image_function
        self._last_png: Path | None = None
        gen = self

        def save_and_keep_last(cfg_, image, count=0, max_count=0, **kw):
            if gen._last_png is not None and image.get("video") is not None:
                from PIL import Image
                frames = image["video"]                       # (T, H, W, C) uint8
                last = frames[-1]
                last = last.numpy() if hasattr(last, "numpy") else last
                Image.fromarray(last).save(gen._last_png)
            return gen._orig_save(cfg_, image, count, max_count, **kw)

        h3mod.new_save_image_function = save_and_keep_last

    def generate(self, prompt: str, start_image: Path, out_video: Path, out_last_frame: Path, seed: int) -> None:
        from toolkit.config_modules import GenerateImageConfig

        with tempfile.TemporaryDirectory(prefix="video-chain-", dir=str(Path(out_video).parent)) as tmp:
            cfg = GenerateImageConfig(
                prompt=prompt, negative_prompt="", width=self.width, height=self.height,
                num_inference_steps=self.steps, guidance_scale=1.0,      # H3 is guidance-distilled
                seed=int(seed), num_frames=self.frames, fps=24, ctrl_img=str(start_image),
                output_folder=tmp, output_ext="mp4")
            self._last_png = Path(out_last_frame)
            try:
                self.sd.generate_images([cfg])
            finally:
                self._last_png = None
            made = [p for p in Path(tmp).iterdir() if p.suffix == ".mp4"]
            if len(made) != 1:
                raise RuntimeError(f"expected one video from ai-toolkit, found {len(made)}")
            shutil.move(str(made[0]), str(out_video))
