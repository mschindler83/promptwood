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


def lora_file_info(path: Path) -> tuple[int, float]:
    """(rank, scale) of a LoRA file, read from the tensor shapes and alpha values without loading the weights.
    scale = alpha / rank; files without alpha values (ai-toolkit's own format) use alpha = rank, scale 1."""
    from safetensors import safe_open

    ranks: dict[str, int] = {}
    alpha_keys: list[str] = []
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            if key.endswith((".lora_A.weight", ".lora_down.weight")):
                ranks[key.rsplit(".", 2)[0]] = int(f.get_slice(key).get_shape()[0])
            elif key.endswith(".alpha"):
                alpha_keys.append(key)
    alphas: dict[str, float] = {}
    if alpha_keys:
        try:
            with safe_open(str(path), framework="numpy") as f:
                for key in alpha_keys:
                    alphas[key[: -len(".alpha")]] = float(f.get_tensor(key).reshape(-1)[0])
        except (TypeError, ValueError):     # e.g. bf16 alphas, which numpy cannot hold
            with safe_open(str(path), framework="pt") as f:
                for key in alpha_keys:
                    alphas[key[: -len(".alpha")]] = float(f.get_tensor(key).float().reshape(-1)[0])
    if not ranks:
        raise ValueError(f"{Path(path).name} contains no LoRA weights")
    scales = {round(alphas.get(m, r) / r, 6) for m, r in ranks.items()}
    if len(scales) != 1:
        raise ValueError(f"{Path(path).name} uses different alpha/rank ratios per layer; not supported")
    return max(ranks.values()), scales.pop()


class H3Generator:
    """MiniMax-H3 image-to-video. Several LoRAs can be used in one run (one at a time, per clip): the LoRA
    network is built at the largest rank among them; a smaller LoRA is padded with zeros, which is exact. Before
    every swap all LoRA weights are zeroed (nothing of the previous LoRA stays) and every module's scale is set to
    the new file's alpha / rank (ai-toolkit computes the scale only once, when the network is built)."""

    def __init__(self, *, width: int, height: int, frames: int = 73, steps: int = 28, lora: Path | None = None,
                 strength: float = 1.0, model: str = DEFAULT_MODEL, offload: float = 0.9, low_vram: bool = True,
                 audio: bool = True, aitk: Path = DEFAULT_AITK, device: str = "cuda:0", min_free_gb: float = 18.0,
                 log=print, progress=None, loras: list | tuple = ()):
        self.width, self.height, self.frames, self.steps = width, height, frames, steps
        self.default_lora = str(lora) if lora else None
        self.default_strength = float(strength)
        self.log = log
        aitk = Path(aitk).resolve()
        if not (aitk / "toolkit").is_dir():
            raise FileNotFoundError(f"{aitk} is not an ai-toolkit folder")
        wanted = []
        for lo in ([self.default_lora] if self.default_lora else []) + [str(x) for x in loras if x]:
            if lo not in wanted:
                wanted.append(lo)
        for lo in wanted:
            if not Path(lo).is_file():
                raise FileNotFoundError(lo)
        # absolute paths: ai-toolkit needs the working directory changed below
        self.paths = {lo: str(Path(lo).resolve()) for lo in wanted}
        self.info = {lo: lora_file_info(Path(lo)) for lo in wanted}       # rank, scale
        base = max(wanted, key=lambda lo: self.info[lo][0]) if wanted else None

        free = free_gpu_gb()
        if free is not None and free < min_free_gb:
            raise RuntimeError(f"only {free:.1f} GB GPU memory free (need {min_free_gb:g} GB, see --min-free-gb). "
                               "Is a training or another model still running?")

        # ai-toolkit resolves its extensions and MODELS_PATH relative to its own folder
        sys.path.insert(0, str(aitk))
        os.chdir(aitk)
        import torch  # noqa: F401
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class

        cfg = ModelConfig(**model_config_dict(model, self.paths[base] if base else None, offload, low_vram, audio))
        cls = get_model_class(cfg)
        self.sd = cls(device=device, model_config=cfg, dtype="bf16", noise_scheduler=cls.get_train_scheduler())
        log("loading MiniMax-H3 (this takes a few minutes)")
        self.sd.load_model()
        self.net = self.sd.assistant_lora if base else None
        self.net_rank = self.info[base][0] if base else 0
        self.current: str | None = None
        if self.net is not None:
            self.use_lora(base, 1.0)                      # checks that it fits; the strength is set per clip

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

    def can_use(self, loras: list[str]) -> bool:
        """Whether these LoRAs can be swapped into the loaded model (preparing new ones of a rank that fits)."""
        for lo in loras:
            if lo in self.info:
                continue
            if self.net is None or not Path(lo).is_file():
                return False
            info = lora_file_info(Path(lo))
            if info[0] > self.net_rank:
                return False
            self.info[lo] = info
            self.paths[lo] = str(Path(lo).resolve())
        return True

    def use_lora(self, lora: str | None, strength: float) -> None:
        """Make lora (or none) the active LoRA at this strength."""
        import torch
        from safetensors.torch import load_file

        net = self.net
        if lora is None:
            if net is not None:
                net.multiplier = 0.0                      # ai-toolkit skips the LoRA path entirely at 0
                net._update_torch_multiplier()
            return
        if net is None or lora not in self.info:
            raise RuntimeError(f"LoRA {Path(lora).name} was not prepared for this run")
        if lora != self.current:
            weights = self.sd.convert_lora_weights_before_load(load_file(self.paths[lora]))
            with torch.no_grad():
                for p in net.parameters():
                    p.zero_()
            unknown = net.load_weights(weights)           # pads a smaller rank with zeros
            if unknown:
                raise RuntimeError(f"{len(unknown)} of {len(weights)} weights of {Path(lora).name} do not fit this "
                                   f"model (e.g. {next(iter(unknown))}); is it an H3 LoRA for this checkpoint?")
            if not any(float(p.detach().abs().sum()) > 0 for n, p in net.named_parameters() if "lora_up" in n):
                raise RuntimeError(f"{Path(lora).name} loaded as all zeros")
            rank, scale = self.info[lora]
            for m in net.get_all_modules():
                m._set_runtime_scale(scale)
            self.current = lora
            self.log(f"LoRA {Path(lora).name} loaded (rank {rank}, scale {scale:g})")
        net.multiplier = float(strength)
        net._update_torch_multiplier()
        # ai-toolkit moves the LoRA to the CPU after every sample and back before the next one, but builds this
        # tensor on the device the weights are on right now; it is a plain attribute, so moving the network
        # does not move it. Put it where the model runs, or the next clip fails with a CPU/CUDA mix.
        net.torch_multiplier = net.torch_multiplier.to(self.sd.device_torch)

    def generate(self, prompt: str, start_image: Path, out_video: Path, out_last_frame: Path, seed: int,
                 settings: dict | None = None) -> None:
        from toolkit.config_modules import GenerateImageConfig

        s = settings or {}
        frames = int(s.get("frames", self.frames))
        steps = int(s.get("steps", self.steps))
        width, height = int(s.get("width", self.width)), int(s.get("height", self.height))
        if settings is not None:
            self.use_lora(s.get("lora"), float(s.get("strength", 1.0)))
        else:
            self.use_lora(self.default_lora, self.default_strength)
        with tempfile.TemporaryDirectory(prefix="video-chain-", dir=str(Path(out_video).parent)) as tmp:
            cfg = GenerateImageConfig(
                prompt="", negative_prompt="", width=width, height=height,
                num_inference_steps=steps, guidance_scale=1.0,      # H3 is guidance-distilled
                seed=int(seed), num_frames=frames, fps=24, ctrl_img=str(start_image),
                output_folder=tmp, output_ext="mp4")
            # set afterwards: ai-toolkit parses '--w 512', '--seed 3' etc. out of a prompt given to the constructor
            # and cuts the prompt at the first '--'
            cfg.prompt = cfg.prompt_2 = prompt
            self._last_png = Path(out_last_frame)
            try:
                self.sd.generate_images([cfg])
            finally:
                self._last_png = None
            made = [p for p in Path(tmp).iterdir() if p.suffix == ".mp4"]
            if len(made) != 1:
                raise RuntimeError(f"expected one video from ai-toolkit, found {len(made)}")
            shutil.move(str(made[0]), str(out_video))
