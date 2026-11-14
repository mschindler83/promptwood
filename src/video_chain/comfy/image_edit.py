"""Image edit and new pictures with the Qwen-Image-Edit 2511, Qwen-Image 2.1 and MOSS-Image workflows, and
ComfyImageEditor (the same connection, upload and cleanup as the clip generator)."""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Sequence

from ..chain import image_size
from .client import ComfyGenerator
from .errors import ComfyError
from .wfb import (EDIT_SAVE_ID, QWEN_SHIFT, SUBFOLDER, combo_options, loader_clip, loader_unet, loader_vae, lora_chain,
                  lora_node, low_name)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- image edit (Qwen Image Edit)

QWEN_NODES = ("TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod", "FluxKontextImageScale",
              "ModelSamplingAuraFlow", "CFGNorm", "KSampler", "VAEEncode")
QUALITY = {"fast": None, "standard": (20, 4.0), "best": (40, 4.0)}   # fast: the Lightning LoRA's steps, cfg 1


def _steps_in(name: str) -> int | None:
    m = re.search(r"(\d+)\s*-?\s*steps?", name.lower())
    return int(m.group(1)) if m else None


def pick_qwen_files(unets: list[str], clips: list[str], vaes: list[str], loras: list[str]) -> dict:
    """ComfyUI's Qwen Image Edit files: the edit model (2511 preferred), its text encoder and VAE, and the Lightning
    LoRA of the same model version (fewest steps) if there is one. Missing files are None."""
    edit = [u for u in unets if "qwen_image_edit" in low_name(u)]
    unet = next((u for u in edit if "2511" in u), None) or (sorted(edit)[-1] if edit else None)
    clip = next((c for c in clips if "qwen_2.5_vl_7b" in low_name(c)), None)
    vae = next((v for v in vaes if "qwen_image_vae" in low_name(v)), None)
    version = next((v for v in ("2511", "2509") if unet and v in unet), "")
    light = [lo for lo in loras if "lightning" in lo.lower() and "edit" in lo.lower() and _steps_in(lo)
             and (not version or version in lo)]
    light.sort(key=lambda lo: (_steps_in(lo), "fp32" in lo.lower()))
    return {"unet": unet, "clip": clip, "vae": vae,
            "lightning": (light[0], _steps_in(light[0])) if light else None}


def build_edit_workflow(*, prompt: str, image: str, refs: Sequence[str] = (), seed: int, steps: int, cfg: float,
                        files: dict, lightning: str | None, prefix: str,
                        loras: Sequence[tuple[str, float]] = ()) -> dict:
    """One Qwen Image Edit: the source image (scaled to the model's size) edited by the prompt; refs: up to two more
    pictures (Picture 2, Picture 3 in the prompt), e.g. a face. As ComfyUI's blueprint 'Image Edit (Qwen 2511)'.
    loras: (ComfyUI name, strength) chained after the model, in this order, before the Lightning LoRA."""
    wf: dict = {
        "clip": loader_clip(files["clip"], "qwen_image"),
        "vae": loader_vae(files["vae"]),
        "unet": loader_unet(files["unet"]),
        "src": {"class_type": "LoadImage", "inputs": {"image": image}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["src", 0]}},
    }
    model = lora_chain(wf, ["unet", 0], loras)
    if lightning:
        wf["lightning"] = lora_node(model, lightning, 1.0)
        model = ["lightning", 0]
    wf["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": QWEN_SHIFT}}
    wf["cfgnorm"] = {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}}
    pics = {"image1": ["scale", 0]}
    for k, r in enumerate(list(refs)[:2]):
        wf[f"ref{k}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
        pics[f"image{k + 2}"] = [f"ref{k}", 0]
    for name, text in (("pos", prompt), ("neg", "")):
        wf[name] = {"class_type": "TextEncodeQwenImageEditPlus",
                    "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], **pics}}
        wf[name + "_ref"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                             "inputs": {"conditioning": [name, 0], "reference_latents_method": "index_timestep_zero"}}
    wf["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}}
    wf["sample"] = {"class_type": "KSampler",
                    "inputs": {"model": ["cfgnorm", 0], "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                               "sampler_name": "euler", "scheduler": "simple", "positive": ["pos_ref", 0],
                               "negative": ["neg_ref", 0], "latent_image": ["latent", 0], "denoise": 1.0}}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    wf[EDIT_SAVE_ID] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return wf


# Qwen-Image 2.1: one model for generation and editing (ComfyUI's templates image_qwen_image_2_1_image_edit / _t2i
# and their Turbo versions): the base samples 25 steps at cfg 1, Turbo its 8 fixed sigmas at cfg 1
QWEN21_NODES = ("TextEncodeQwenImage21", "QwenImage21Cache", "ManualSigmas", "CFGGuider", "SamplerCustomAdvanced")
QWEN21_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
QWEN21_STEPS = 25
QWEN21_RESOLUTION = 1024           # pictures are seen at about 1024 x 1024 (multiples of 32, aspect kept)


def _quantized(name: str) -> int:
    """Sort key, accuracy first: full precision, then fp8 / int8, then 4-bit."""
    n = name.lower()
    return 2 if any(k in n for k in ("w4a8", "nvfp4", "int4", "q4", "gguf")) else \
        1 if any(k in n for k in ("int8", "fp8", "q8")) else 0


def is_qwen21_name(name: str) -> bool:
    """A Qwen-Image 2.1 model file by its name (2.1 and 2512 must not be mixed up)."""
    n = name.lower().replace("-", "_")
    return "qwen_image_2.1" in n or "qwen_image_21" in n or "qwenimage2.1" in n or "qwenimage21" in n


def pick_qwen21_files(unets: list[str], clips: list[str], vaes: list[str]) -> dict | None:
    """Qwen-Image 2.1's files ComfyUI lists: {"base", "turbo", "models", "clip", "vae"}, or None without a model,
    its text encoder (Qwen3-VL 8B) or its VAE."""
    q = [u for u in unets if is_qwen21_name(u)]
    base = sorted([u for u in q if "turbo" not in low_name(u)], key=lambda u: (_quantized(u), u))
    turbo = sorted([u for u in q if "turbo" in low_name(u)], key=lambda u: (_quantized(u), u))
    clip = sorted([c for c in clips if "qwen3vl_8b" in low_name(c)], key=lambda c: (_quantized(c), c))
    vae = sorted([v for v in vaes if "qwen_image_2.1_vae" in low_name(v) or "qwen_image_21_vae" in low_name(v)])
    if not (base or turbo) or not clip or not vae:
        return None
    return {"base": base[0] if base else None, "turbo": turbo[0] if turbo else None, "models": base + turbo,
            "clip": clip[0], "vae": vae[0]}


# Inpainting with Qwen-Image 2.1 (a painted mask): the Qwen-Image 2.1 Fun ControlNet Union (alibaba-pai, in
# models/model_patches) draws the masked part anew and sees the rest; Promptwood's node (comfy_nodes/) hands it the
# source and the mask, since ComfyUI's QwenImageDiffsynthControlnet has no input for the image to keep. Afterwards
# the result is laid into the source along a soft edge, so everything outside the mask stays exactly as it was.
INPAINT21_NODE = "VideoChainQwen21FunInpaint"
INPAINT_FEATHER = 12                        # px: the soft edge (the mask grown by half of it, then blurred), at
                                            # a 2 MP picture; it grows with the picture (inpaint_feather)
INPAINT_FEATHER_MIN, INPAINT_FEATHER_MAX = 4, 20   # px: the clamp of inpaint_feather
INPAINT_BLUR_SIGMA_MAX = 10.0               # ImageBlur's sigma never goes above this
GREY = 0x808080                             # the grey the masked part is filled with before the edit


def inpaint_feather(width: int, height: int) -> int:
    """The soft edge for a picture: about 1/170 of its short side, 4 to 20 px (ImageBlur's sigma stays <= 10)."""
    return max(INPAINT_FEATHER_MIN, min(INPAINT_FEATHER_MAX, round(min(width, height) / 170)))


def pick_qwen21_inpaint(patches: list[str]) -> str | None:
    """The Qwen-Image 2.1 Fun ControlNet among ComfyUI's model patches (not the older Qwen-Image's)."""
    hits = [p for p in patches if is_qwen21_name(p) and "controlnet" in p.lower().replace("_", "").replace("-", "")]
    return sorted(hits, key=lambda p: (_quantized(p), p))[0] if hits else None


def build_qwen21_workflow(*, files: dict, prompt: str, seed: int, quality: str, prefix: str, save_as: str,
                          images: Sequence[str] = (), size: tuple[int, int] | None = None, negative: str = "",
                          loras: Sequence[tuple[str, float]] = (), model: str | None = None,
                          inpaint: dict | None = None) -> dict:
    """One Qwen-Image 2.1 picture. images: uploaded pictures (image_1 ... in the text encoder: an edit's source
    first); size: the new picture's (width, height); None: the size of images[0] as the model sees it (an edit).
    quality 'fast': the Turbo model (8 steps), else the base (25 steps); model: a chosen file (a fine-tune) instead.
    inpaint: {"patch": the Fun ControlNet, "mask": an uploaded mask at the source's size (white = draw anew),
    "strength", "feather"}: only the masked part of images[0] is drawn anew, the result laid into the source.
    save_as: the SaveImage node's id."""
    if model and model not in (files.get("base"), files.get("turbo")):
        unet = model                               # a chosen fine-tune: Turbo sampling only for a Turbo build
        turbo = "turbo" in model.lower()
    else:
        turbo = quality == "fast" and bool(files.get("turbo")) or not files.get("base")
        unet = files["turbo"] if turbo else files["base"]
    wf: dict = {
        "unet": loader_unet(unet),
        "clip": loader_clip(files["clip"], "qwen_image"),
        "vae": loader_vae(files["vae"]),
    }
    m = lora_chain(wf, ["unet", 0], loras)
    te = {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": negative or "", "resolution": QWEN21_RESOLUTION}
    if images:
        te["vae"] = ["vae", 0]
        for k, name in enumerate(images[:10]):
            wf[f"pic{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            te[f"images.image_{k + 1}"] = [f"pic{k}", 0]
    if inpaint:
        if not images:
            raise ValueError("inpainting needs the source picture")
        wf["ipatch"] = {"class_type": "ModelPatchLoader", "inputs": {"name": inpaint["patch"]}}
        wf["imask_pic"] = {"class_type": "LoadImage", "inputs": {"image": inpaint["mask"]}}
        wf["imask"] = {"class_type": "ImageToMask", "inputs": {"image": ["imask_pic", 0], "channel": "red"}}
        wf["inpaint"] = {"class_type": INPAINT21_NODE,
                         "inputs": {"model": m, "model_patch": ["ipatch", 0], "vae": ["vae", 0],
                                    "inpaint_image": ["pic0", 0], "mask": ["imask", 0],
                                    "strength": float(inpaint.get("strength", 1.0))}}
        m = ["inpaint", 0]
        # measured (Qwen-Image 2.1 Turbo, a head masked, "She is looking up"): with the whole source as the edit's
        # picture the masked part comes back as it was (the prompt is ignored); without a picture it follows the
        # prompt but loses the person; with the masked part grey it follows the prompt and keeps the person
        ref = inpaint.get("reference", "hole")
        if ref == "hole":                      # the edit's picture with the masked part grey: nothing there to copy
            wf["igrey"] = {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1,
                                                                   "color": GREY}}
            wf["ihole"] = {"class_type": "ImageCompositeMasked",
                           "inputs": {"destination": ["pic0", 0], "source": ["igrey", 0], "x": 0, "y": 0,
                                      "resize_source": True, "mask": ["imask", 0]}}
            te["images.image_1"] = ["ihole", 0]
    if inpaint:                                # (the ControlNet's block patches turn the prefix cache off anyway)
        sampling_model = m
    else:
        wf["cache"] = {"class_type": "QwenImage21Cache", "inputs": {"model": m, "device": "auto",
                                                                    "dtype": "default"}}
        sampling_model = ["cache", 0]
    wf["text"] = {"class_type": "TextEncodeQwenImage21", "inputs": te}
    if size is not None or not images:
        w, h = size or (1024, 1024)
        wf["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": int(w), "height": int(h),
                                                                     "batch_size": 1}}
        latent = ["latent", 0]
    else:
        latent = ["text", 2]                       # the source's size, as the model sees it
    if turbo:
        wf["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
        wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        wf["sigmas"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN21_SIGMAS}}
        wf["guider"] = {"class_type": "CFGGuider", "inputs": {"model": sampling_model, "positive": ["text", 0],
                                                              "negative": ["text", 1], "cfg": 1.0}}
        wf["sample"] = {"class_type": "SamplerCustomAdvanced",
                        "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                   "sigmas": ["sigmas", 0], "latent_image": latent}}
    else:
        wf["sample"] = {"class_type": "KSampler",
                        "inputs": {"model": sampling_model, "seed": int(seed), "steps": QWEN21_STEPS, "cfg": 1.0,
                                   "sampler_name": "euler", "scheduler": "simple", "positive": ["text", 0],
                                   "negative": ["text", 1], "latent_image": latent, "denoise": 1.0}}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    out = ["decode", 0]
    if inpaint:                                # laid into the source: outside the (softened) mask it stays exact
        feather = max(0, min(INPAINT_FEATHER_MAX, int(inpaint.get("feather", INPAINT_FEATHER))))
        wf["igrow"] = {"class_type": "GrowMask", "inputs": {"mask": ["imask", 0], "expand": feather,
                                                            "tapered_corners": True}}
        mask = ["igrow", 0]
        if feather:
            wf["isoft_pic"] = {"class_type": "MaskToImage", "inputs": {"mask": mask}}
            wf["isoft_blur"] = {"class_type": "ImageBlur", "inputs": {"image": ["isoft_pic", 0],
                                                                      "blur_radius": feather,
                                                                      "sigma": min(INPAINT_BLUR_SIGMA_MAX, max(1.0, feather / 2))}}
            wf["isoft"] = {"class_type": "ImageToMask", "inputs": {"image": ["isoft_blur", 0], "channel": "red"}}
            mask = ["isoft", 0]
        wf["icompose"] = {"class_type": "ImageCompositeMasked",
                          "inputs": {"destination": ["pic0", 0], "source": out, "x": 0, "y": 0,
                                     "resize_source": True, "mask": mask}}
        out = ["icompose", 0]
    wf[save_as] = {"class_type": "SaveImage", "inputs": {"images": out, "filename_prefix": prefix}}
    return wf


# MOSS-Image (OpenMOSS, 5B, Apache 2.0; ComfyUI's templates image_moss_image_t2i / _base_t2i / _edit): new images
# with the SFT model (25 steps, cfg 2.5, er_sde; the base model with skip-layer guidance on layer 22 for the first
# half), edits with its edit LoRA (30 steps, cfg 4, euler, skip-layer guidance throughout; the picture at ~1 MP)
MOSS_NODES = ("TextEncodeMOSSImageEdit", "SkipLayerGuidanceDiTSimple", "EmptyFlux2LatentImage",
              "ImageScaleToTotalPixels", "GetImageSize")
MOSS_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
MOSS_STEPS, MOSS_EDIT_STEPS = 25, 30


def pick_moss_files(unets: list[str], clips: list[str], vaes: list[str], loras: list[str]) -> dict | None:
    """MOSS-Image's files ComfyUI lists: {"sft", "base", "models", "clip", "vae", "edit_lora"} (edit_lora None:
    no edits), or None without a model, its text encoder (Qwen3.5 9B) or its VAE (FLUX.2's)."""
    m = [u for u in unets if "moss_image" in low_name(u)]
    sft = sorted([u for u in m if "base" not in low_name(u)], key=lambda u: (_quantized(u), u))
    base = sorted([u for u in m if "base" in low_name(u)], key=lambda u: (_quantized(u), u))
    clip = sorted([c for c in clips if "qwen3.5_9b" in low_name(c) and "qwen_image" not in low_name(c)],
                  key=lambda c: (_quantized(c), c))
    vae = sorted([v for v in vaes if "flux2_vae" in low_name(v)], key=lambda v: ("small" not in v, v))
    edit = sorted([lo for lo in loras if "moss_image_edit_lora" in low_name(lo)],     # the official edit LoRA
                  key=lambda lo: ("rank_64" not in low_name(lo), "bf16" not in low_name(lo), lo))
    if not (sft or base) or not clip or not vae:
        return None
    return {"sft": sft[0] if sft else None, "base": base[0] if base else None, "models": sft + base,
            "clip": clip[0], "vae": vae[0], "edit_lora": edit[0] if edit else None}


def build_moss_workflow(*, files: dict, prompt: str, seed: int, prefix: str, save_as: str, images: Sequence[str] = (),
                        size: tuple[int, int] | None = None, negative: str = "", loras: Sequence[tuple[str, float]] = (),
                        model: str | None = None) -> dict:
    """One MOSS-Image picture. Without images: text to image at size. With images (uploaded; an edit's source
    first): the edit LoRA and TextEncodeMOSSImageEdit; size None: the size of the first picture scaled to ~1 MP."""
    unet = model if model and model in (files.get("models") or []) else (files.get("sft") or files["base"])
    if images:
        unet = files.get("sft") or unet            # the edit LoRA was trained on the SFT model
    wf: dict = {
        "unet": loader_unet(unet),
        "clip": loader_clip(files["clip"], "moss_image"),
        "vae": loader_vae(files["vae"]),
    }
    m = lora_chain(wf, ["unet", 0], loras)
    neg = negative or MOSS_NEGATIVE
    if images:
        if not files.get("edit_lora"):
            raise ComfyError("MOSS-Image's edit LoRA (moss_image_edit_lora_rank_64_bf16.safetensors) is missing")
        wf["edit_lora"] = lora_node(m, files["edit_lora"], 1.0)
        wf["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple",
                     "inputs": {"model": ["edit_lora", 0], "double_layers": "", "single_layers": "22",
                                "start_percent": 0.0, "end_percent": 1.0}}
        pics = {}
        for k, name in enumerate(images[:10]):
            wf[f"pic{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wf[f"scale{k}"] = {"class_type": "ImageScaleToTotalPixels",
                               "inputs": {"image": [f"pic{k}", 0], "upscale_method": "lanczos", "megapixels": 1.0,
                                          "resolution_steps": 16}}
            pics[f"images.image_{k}"] = [f"scale{k}", 0]   # (0-based, as MOSS's own template; Qwen 2.1 counts from 1)
        for nid, text in (("pos", prompt), ("neg", neg)):
            wf[nid] = {"class_type": "TextEncodeMOSSImageEdit",
                       "inputs": {"clip": ["clip", 0], "prompt": text, "edit_format": "Moss-Image-Edit",
                                  "image_strength": 1.0, "vae": ["vae", 0], **pics}}
        if size is None:
            wf["size"] = {"class_type": "GetImageSize", "inputs": {"image": ["scale0", 0]}}
            wf["latent"] = {"class_type": "EmptyFlux2LatentImage",
                            "inputs": {"width": ["size", 0], "height": ["size", 1], "batch_size": 1}}
        steps, cfg, sampler, sched, sm = MOSS_EDIT_STEPS, 4.0, "euler", "normal", ["slg", 0]
    else:
        wf["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
        wf["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": neg}}
        sm = m
        if unet == files.get("base"):              # the base template: skip-layer guidance for the first half
            wf["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple",
                         "inputs": {"model": m, "double_layers": "", "single_layers": "22", "start_percent": 0.0,
                                    "end_percent": 0.5}}
            sm = ["slg", 0]
        steps, cfg, sampler, sched = MOSS_STEPS, 2.5, "er_sde", "simple"
    if "latent" not in wf:
        w, h = size or (1024, 1024)
        wf["latent"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": int(w), "height": int(h),
                                                                          "batch_size": 1}}
    wf["sample"] = {"class_type": "KSampler",
                    "inputs": {"model": sm, "seed": int(seed), "steps": steps, "cfg": cfg, "sampler_name": sampler,
                               "scheduler": sched, "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["latent", 0], "denoise": 1.0}}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    wf[save_as] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    return wf


EDIT_MODELS = {"2511": "Qwen-Image-Edit 2511", "qwen21": "Qwen-Image 2.1", "moss": "MOSS-Image Edit"}
EDIT21_QUALITIES = ("fast", "best")   # Turbo (8 steps) / base (25 steps)


class ComfyImageEditor(ComfyGenerator):
    """Edits an image by a prompt in ComfyUI (same connection, upload and cleanup as the clip generator), with
    Qwen-Image-Edit 2511 or Qwen-Image 2.1 (EDIT_MODELS)."""

    def _check_models(self):
        def combo(node, field):
            return combo_options(self._object_info(node).get("input", {}).get("required", {}).get(field))
        unets, clips, vaes = combo("UNETLoader", "unet_name"), combo("CLIPLoader", "clip_name"), \
            combo("VAELoader", "vae_name")
        self.files = pick_qwen_files(unets, clips, vaes, combo("LoraLoaderModelOnly", "lora_name"))
        q21 = pick_qwen21_files(unets, clips, vaes)
        self.files21 = q21 if q21 and all(self._object_info(n) for n in QWEN21_NODES) else None
        patch = pick_qwen21_inpaint(combo("ModelPatchLoader", "name"))
        self.inpaint_missing = (
            "Qwen-Image 2.1 is not available" if self.files21 is None else
            "the Qwen-Image 2.1 Fun ControlNet Union is not in ComfyUI's models/model_patches "
            "(Qwen-Image-2.1-Fun-Controlnet-Union.safetensors from alibaba-pai)" if patch is None else
            f"ComfyUI has no {INPAINT21_NODE} node: copy video-chain's comfy_nodes/video_chain_nodes into "
            "ComfyUI/custom_nodes and restart ComfyUI" if not self._object_info(INPAINT21_NODE) else "")
        self.inpaint_patch = None if self.inpaint_missing else patch
        moss = pick_moss_files(unets, clips, vaes, combo("LoraLoaderModelOnly", "lora_name"))
        self.files_moss = moss if moss and moss.get("edit_lora") and all(self._object_info(n) for n in MOSS_NODES) \
            else None
        missing = [what for what, key in (("the Qwen Image Edit model (e.g. qwen_image_edit_2511_*.safetensors in "
                                           "models/diffusion_models)", "unet"),
                                          ("its text encoder qwen_2.5_vl_7b*.safetensors", "clip"),
                                          ("its VAE qwen_image_vae.safetensors", "vae")) if not self.files[key]]
        self.missing_2511 = "ComfyUI does not list " + "; ".join(missing) if missing else ""
        if not self.missing_2511:
            for node in QWEN_NODES:
                if not self._object_info(node):
                    self.missing_2511 = f"this ComfyUI has no {node} node (update ComfyUI)"
                    break
        if self.missing_2511 and self.files21 is None and self.files_moss is None:
            raise ComfyError(self.missing_2511)

    def models(self) -> list[str]:
        """The edit models this ComfyUI has (keys of EDIT_MODELS)."""
        return [m for m, ok in (("2511", not self.missing_2511), ("qwen21", self.files21 is not None),
                                ("moss", self.files_moss is not None)) if ok]

    def qualities(self, model: str = "2511") -> list[str]:
        if model == "moss":
            return ["standard"]
        if model == "qwen21":
            return list(EDIT21_QUALITIES) if self.files21 and self.files21.get("turbo") else ["best"]
        return (["fast"] if self.files.get("lightning") else []) + ["standard", "best"]

    def can_inpaint(self, model: str) -> str:
        """Why a painted mask cannot be used with this edit model ("" when it can)."""
        if model != "qwen21":
            return "painting a mask works with Qwen-Image 2.1"
        return self.inpaint_missing

    def edit(self, image: Path, prompt: str, out_png: Path, seed: int, quality: str = "fast",
             refs: Sequence[Path] = (), loras: Sequence[tuple[str, float]] = (), model: str = "2511",
             mask: Path | None = None) -> dict:
        """Edit image by prompt; the result goes to out_png. loras: (file, strength) in ComfyUI's LoRA folders.
        model: a key of EDIT_MODELS. mask: a picture at the image's size, white where it is drawn anew (inpaint;
        Qwen-Image 2.1 only). Returns what it was made with."""
        self._begin_job()
        if mask is not None and self.can_inpaint(model):
            raise ComfyError(self.can_inpaint(model))
        names = []
        for path, strength in loras:
            name = self.name_for(path)
            if name is None:
                raise ComfyError(f"{Path(path).name} is not in ComfyUI's LoRA folders")
            names.append((name, float(strength)))
        if model == "qwen21":
            return self._edit21(image, prompt, out_png, seed, quality, refs, loras, names, mask)
        if model == "moss":
            return self._edit_moss(image, prompt, out_png, seed, refs, loras, names)
        if self.missing_2511:
            raise ComfyError(self.missing_2511)
        light = self.files.get("lightning") if quality == "fast" else None
        if quality == "fast" and not light:
            quality = "standard"
        fallback = QUALITY.get(quality) or QUALITY["standard"]
        assert fallback is not None                   # ("standard" always has its steps)
        steps, cfg = (light[1], 1.0) if light else fallback
        uploads: list[tuple[str, Path]] = []
        try:
            name = self.upload(Path(image))
            uploads.append((name, Path(image)))
            ref_names = []
            for r in list(refs)[:2]:
                ref_names.append(self.upload(Path(r)))
                uploads.append((ref_names[-1], Path(r)))
            prefix = f"{SUBFOLDER}/edit_{uuid.uuid4().hex[:12]}"
            wf = build_edit_workflow(prompt=prompt, image=name, refs=ref_names, seed=seed, steps=steps, cfg=cfg,
                                     files=self.files, lightning=light[0] if light else None, prefix=prefix,
                                     loras=names)
            self._run_saving(wf, EDIT_SAVE_ID, out_png, [])
        finally:
            self._drop_uploads(uploads)
        return {"quality": quality, "steps": steps, "cfg": cfg, "model": self.files["unet"], "edit_model": "2511",
                "lightning": light[0] if light else None,
                "loras": [{"path": str(p), "strength": float(st)} for p, st in loras]}

    def _run_saving(self, wf: dict, node: str, out_png: Path, uploads: list[tuple[str, Path]]) -> None:
        """Queue a workflow, wait, download what its SaveImage node `node` wrote to out_png; clean up after."""
        try:
            entry = self._run_workflow(wf)
            out = self._first_file((entry.get("outputs") or {}).get(node))
            if out is None:
                raise ComfyError("ComfyUI finished without the edited image")
            self._save(out, Path(out_png))
            if self.cleanup:
                self._remove(self.output_dir, out, Path(out_png))
        finally:
            self._drop_uploads(uploads)

    def _edit_moss(self, image, prompt, out_png, seed, refs, loras, names) -> dict:
        if self.files_moss is None:
            raise ComfyError("MOSS-Image is not available for edits (moss_image_sft_*.safetensors, its edit LoRA "
                             "moss_image_edit_lora_*, qwen3.5_9b and flux2-vae in ComfyUI's models)")
        uploads: list[tuple[str, Path]] = []
        pics = []
        try:
            for p in [image, *list(refs)[:2]]:
                uploads.append((self.upload(Path(p)), Path(p)))
                pics.append(uploads[-1][0])
            wf = build_moss_workflow(files=self.files_moss, prompt=prompt, seed=seed,
                                     prefix=f"{SUBFOLDER}/edit_{uuid.uuid4().hex[:12]}", save_as=EDIT_SAVE_ID,
                                     images=pics, loras=names)
        except BaseException:
            self._drop_uploads(uploads)              # (_run_saving cleans up only once it runs)
            raise
        self._run_saving(wf, EDIT_SAVE_ID, out_png, uploads)
        return {"quality": "standard", "steps": MOSS_EDIT_STEPS, "cfg": 4.0, "edit_model": "moss",
                "model": wf["unet"]["inputs"]["unet_name"], "lightning": None,
                "loras": [{"path": str(p), "strength": float(st)} for p, st in loras]}

    def _edit21(self, image, prompt, out_png, seed, quality, refs, loras, names, mask=None) -> dict:
        if self.files21 is None:
            raise ComfyError("Qwen-Image 2.1 is not available (qwen_image_2.1_*.safetensors, qwen3vl_8b and "
                             "qwen_image_2.1_vae in ComfyUI's models, and a ComfyUI with its nodes)")
        if quality not in self.qualities("qwen21"):
            quality = self.qualities("qwen21")[-1]
        uploads: list[tuple[str, Path]] = []
        pics = []
        try:
            for p in [image, *list(refs)[:2]]:
                uploads.append((self.upload(Path(p)), Path(p)))
                pics.append(uploads[-1][0])
            inpaint = None
            feather = INPAINT_FEATHER
            if mask is not None:
                mw, mh = image_size(Path(mask))
                iw, ih = image_size(Path(image))
                if (mw, mh) not in ((iw, ih), (ih, iw)):           # (a turned photo: its upright size)
                    raise ComfyError(f"the mask is {mw}x{mh}, the picture {iw}x{ih}: paint the mask again")
                uploads.append((self.upload(Path(mask)), Path(mask)))
                try:
                    feather = inpaint_feather(mw, mh)
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning("mask feather not computed: %s", e)
                inpaint = {"patch": self.inpaint_patch, "mask": uploads[-1][0], "strength": 1.0,
                           "feather": feather, "reference": "hole"}
            wf = build_qwen21_workflow(files=self.files21, prompt=prompt, seed=seed, quality=quality,
                                       prefix=f"{SUBFOLDER}/edit_{uuid.uuid4().hex[:12]}", save_as=EDIT_SAVE_ID,
                                       images=pics, loras=names, inpaint=inpaint)
        except BaseException:
            self._drop_uploads(uploads)
            raise
        self._run_saving(wf, EDIT_SAVE_ID, out_png, uploads)
        turbo = wf["sample"]["class_type"] == "SamplerCustomAdvanced"
        meta = {"quality": quality, "steps": 8 if turbo else QWEN21_STEPS, "cfg": 1.0, "edit_model": "qwen21",
                "model": wf["unet"]["inputs"]["unet_name"], "lightning": None,
                "loras": [{"path": str(p), "strength": float(st)} for p, st in loras]}
        if mask is not None:
            meta["inpaint"] = {"patch": self.inpaint_patch, "feather": feather}
        return meta
