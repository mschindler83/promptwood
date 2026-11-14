"""Pieces every workflow builder needs (video, image edit, refine, SeedVR2, prompt refiner, and imagegen.py /
animate.py): node ids, the loader and LoRA nodes, the video saver, the sampler settings of the text-generate node.

ComfyUI renames inputs now and then (the code already tracks "old vs new combo format"): a node dict that is built
here is fixed in one place."""
from __future__ import annotations

from typing import Sequence

SUBFOLDER = "video-chain"        # ComfyUI's input / output subfolder for everything the app uploads or makes
FPS = 24.0                       # every video the app makes (CreateVideo)
QWEN_SHIFT = 3.1                 # ModelSamplingAuraFlow's shift in ComfyUI's Qwen blueprints (edit and text to image)

# The ids of the nodes whose output the caller reads back (the builder names them, the caller looks them up).
SAVE_VIDEO_ID = "save_video"     # the clip (build_workflow)
SAVE_LAST_ID = "save_last"       # the clip's last frame (build_workflow)
REFINE_SAVE_ID = "refine_save"   # the refined clip
SEEDVR_SAVE_ID = "seedvr_save"   # the SeedVR2 clip
EDIT_SAVE_ID = "edit_save"       # the edited picture
TEXT_SHOW_ID = "show"            # PreviewAny: the language model's answer
TEXT_GEN_ID = "gen"              # TextGenerate

# Every video ComfyUI writes for the app: H.264 re-encoded at this quality. SaveVideo's automatic setting wrote a
# 672x832 clip at 2 Mbit/s (0.15 bit per pixel): visible 8-px blocks (borders 29 % stronger than inside) and
# flicker that SeedVR2 then enlarged faithfully into 2K grain (its flicker followed the draft's at r = 0.993).
VIDEO_CRF = 12.0

# The sampling the text-generate node uses for a language model (the prompt refiner and the face description)
TEXT_TOP_K = 64
TEXT_TOP_P = 0.95
TEXT_MIN_P = 0.05
TEXT_REPETITION_PENALTY = 1.05


def save_video(prefix: str, video=("video", 0)) -> dict:
    """A SaveVideo node writing an mp4 at VIDEO_CRF (never the automatic quality)."""
    return {"class_type": "SaveVideo", "inputs": {"video": list(video), "filename_prefix": prefix, "format": "mp4",
                                                  "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                  "format.codec.encoding.crf": VIDEO_CRF}}


def combo_options(spec) -> list[str]:
    """Options of a combo input in /object_info, old ([list]) or new (["COMBO", {"options": list}]) format."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return list(spec[0])
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return list(spec[1].get("options", []))
    return []


def low_name(name: str) -> str:
    """A file name as the pickers compare it: lower case, '-' as '_'."""
    return name.lower().replace("-", "_")


def loader_unet(name: str) -> dict:
    return {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}


def loader_clip(name: str, kind: str) -> dict:
    """A CLIPLoader; kind is its "type" (minimax, qwen_image, wan, ...)."""
    return {"class_type": "CLIPLoader", "inputs": {"clip_name": name, "type": kind, "device": "default"}}


def loader_vae(name: str) -> dict:
    return {"class_type": "VAELoader", "inputs": {"vae_name": name}}


def lora_node(model: list, name: str, strength: float) -> dict:
    return {"class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}


def lora_chain(wf: dict, model: list, loras: Sequence[tuple[str, float]], prefix: str = "lora") -> list:
    """Chain the LoRAs after `model` in this order (nodes `<prefix>0`, `<prefix>1` ... into wf); the last model."""
    for k, (name, strength) in enumerate(loras):
        wf[f"{prefix}{k}"] = lora_node(model, name, strength)
        model = [f"{prefix}{k}", 0]
    return model


def text_generate_inputs(*, clip: list, prompt: str, system_prompt: str, max_length: int, temperature: float,
                         seed: int, image: list | None = None) -> dict:
    """The inputs of a TextGenerate node (a language model in ComfyUI) with the app's sampling settings."""
    inputs = {"clip": clip, "prompt": prompt, "max_length": int(max_length), "sampling_mode": "on",
              "sampling_mode.temperature": float(temperature), "sampling_mode.top_k": TEXT_TOP_K,
              "sampling_mode.top_p": TEXT_TOP_P, "sampling_mode.min_p": TEXT_MIN_P,
              "sampling_mode.repetition_penalty": TEXT_REPETITION_PENALTY, "sampling_mode.seed": int(seed),
              "thinking": False, "use_default_template": True, "system_prompt": system_prompt}
    if image is not None:
        inputs["image"] = image
    return inputs
