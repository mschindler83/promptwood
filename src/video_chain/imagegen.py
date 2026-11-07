"""Create new images in ComfyUI: text to image, or from reference pictures, in four model families. Each workflow
follows ComfyUI's own template for that model (settings from the template; LoRAs of the family stacked in order):

  krea2       Krea-2 Turbo        Text to Image (Krea-2 Turbo); references: Image Style Reference (Krea-2 Turbo)
  qwen        Qwen-Image 2512     Text to Image (Qwen-Image 2512) (+ its Lightning LoRAs); references:
                                  Qwen-Image-Edit 2511 composing a new picture from them (empty latent)
  flux3       FLUX.3 Image Klein  image_flux3_klein_t2i / _distilled_t2i; references: ReferenceMultiLatent
  flux1       FLUX.1 dev          Text to Image (Flux.1 Dev) (your FLUX.1 LoRAs); references: image to image
  sdxl        Illustrious (SDXL)  image_sdxl_simple with Illustrious settings; references: image to image

Which model files a family uses is found in ComfyUI's lists (pick_files); a family whose files are missing is
offered with the reason (e.g. FLUX.3 needs its gated download)."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from .comfy import SUBFOLDER, ComfyError, ComfyGenerator, ComfyJobFailed, _combo_options

QWEN_NEGATIVE = ("低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲")
SDXL_NEGATIVE = ("lowres, worst quality, low quality, bad anatomy, bad hands, extra digits, fewer digits, jpeg artifacts, "
                 "blurry, watermark, signature, text")
SDXL_QUALITY = "masterpiece, best quality, amazing quality, absurdres"

FAMILIES = {
    "krea2": {"label": "Krea-2 Turbo", "pixels": 1024 * 1024, "multiple": 16, "negative": False,
              "qualities": {"standard": "Turbo, 8 steps"}, "refs": 3,
              "ref_text": "Style reference: the new image takes the look of the pictures"},
    "qwen": {"label": "Qwen-Image 2512", "pixels": 1328 * 1328, "multiple": 16, "negative": True,
             "qualities": {"fast": "Fast (Lightning, 4 steps)", "balanced": "Balanced (Lightning, 8 steps)",
                           "best": "Best (50 steps, slow)"}, "refs": 3,
             "ref_text": "The new image is composed from the pictures (Qwen-Image-Edit 2511); name them as "
                         "Picture 1, Picture 2 … in the prompt"},
    "flux3": {"label": "FLUX.3 Image Klein", "pixels": 1024 * 1024, "multiple": 16, "negative": True,
              "qualities": {"fast": "Fast (distilled, 16 steps)", "best": "Best (base, 50 steps)"}, "refs": 3,
              "ref_text": "Name the pictures in the prompt as <ref_image_0>, <ref_image_1> …"},
    "flux1": {"label": "FLUX.1 dev", "pixels": 1024 * 1024, "multiple": 16, "negative": False,
              "qualities": {"standard": "20 steps"}, "refs": 1,
              "ref_text": "Image to image: the new image starts from the picture (strength below)"},
    "sdxl": {"label": "Illustrious (SDXL)", "pixels": 1024 * 1024, "multiple": 64, "negative": True,
             "qualities": {"fast": "Fast (Euler a, 20 steps)", "best": "Best (DPM++ 2M Karras, 30 steps)"},
             "refs": 1, "ref_text": "Image to image: the new image starts from the picture (strength below)"},
}


def _low(n: str) -> str:
    return n.lower().replace("-", "_")


def pick_files(unets: list[str], ckpts: list[str], clips: list[str], vaes: list[str], loras: list[str]) -> dict:
    """The model files of every family ComfyUI lists: {family: {"models": [names, best first], ...} or
    {"missing": reason}}."""
    out: dict = {}

    def first(names, *keys):
        for k in keys:
            hit = [n for n in names if k in _low(n)]
            if hit:
                return sorted(hit, key=len)[0]
        return None
    # Krea-2: the official Turbo build first, then other Krea-2 diffusion models (fine-tunes)
    krea = [u for u in unets if "krea2" in _low(u) or "krea_2" in _low(u)]
    krea.sort(key=lambda u: (0 if "krea2_turbo_fp8_scaled" in _low(u) else 1 if "krea2_turbo" in _low(u) else 2, u))
    te = first(clips, "qwen3vl_4b_fp8", "qwen3vl_4b", "qwen_3_vl_4b")
    vae = first(vaes, "qwen_image_vae")
    style = first(loras, "krea2_style_reference")
    out["krea2"] = ({"models": krea, "clip": te, "vae": vae, "style": style} if krea and te and vae else
                    {"missing": "Krea-2 Turbo (krea2_turbo_fp8_scaled) and its text encoder qwen3vl_4b"})
    # Qwen-Image 2512 (and other Qwen-Image generation models; not the edit models)
    qwen = [u for u in unets if "qwen" in _low(u) and "edit" not in _low(u) and "2.1" not in u
            and ("qwen_image" in _low(u) or "qwenimage" in _low(u))]
    qwen.sort(key=lambda u: (0 if "qwen_image_2512" in _low(u) else 1 if "2512" in u else 2, u))
    qte = first(clips, "qwen_2.5_vl_7b")
    light = sorted([lo for lo in loras if "qwen_image_2512_lightning" in _low(lo)], key=len)
    edit = first(unets, "qwen_image_edit_2511", "qwen_image_edit")
    out["qwen"] = ({"models": qwen, "clip": qte, "vae": vae, "lightning4": next((lo for lo in light if "4steps" in lo), None),
                    "lightning8": next((lo for lo in light if "8steps" in lo), None), "edit": edit}
                   if qwen and qte and vae else {"missing": "Qwen-Image 2512 (qwen_image_2512_fp8) and qwen_2.5_vl_7b"})
    # FLUX.3 Image Klein
    f3 = [u for u in unets if "flux3_image_klein" in _low(u)]
    f3te = first(clips, "qwen3vl_4b_int8", "qwen3vl_4b_fp8", "qwen3vl_4b")
    f3vae = first(vaes, "flux2_vae", "flux2-vae")
    out["flux3"] = ({"base": next((u for u in sorted(f3) if "base" in u.lower()), None),
                     "distilled": next((u for u in sorted(f3) if "base" not in u.lower()), None), "models": sorted(f3),
                     "clip": f3te, "vae": f3vae} if f3 and f3te and f3vae else
                    {"missing": "FLUX.3 Image Klein: accept its licence on huggingface.co/black-forest-labs/"
                                "flux-3-image-klein, then download it (flux3-image-klein-*-int8-convrot, "
                                "qwen3vl_4b_int8_convrot, flux2-vae)"})
    # FLUX.1 dev
    f1 = [u for u in unets if ("flux1" in _low(u) or "flux_dev" in _low(u) or "flux1_dev" in _low(u))
          and "schnell" not in _low(u) and "fill" not in _low(u) and "kontext" not in _low(u)]
    f1.sort(key=lambda u: (0 if "krea" in _low(u) else 1, u))
    t5 = first(clips, "t5xxl_fp8", "t5xxl")
    cl = first(clips, "clip_l")
    ae = next((v for v in sorted(vaes, key=len) if Path(v).name == "ae.safetensors"), None)
    out["flux1"] = ({"models": f1, "t5": t5, "clip_l": cl, "vae": ae} if f1 and t5 and cl and ae else
                    {"missing": "FLUX.1 dev with t5xxl, clip_l and ae.safetensors"})
    # Illustrious / SDXL checkpoints
    sd = [c for c in ckpts if any(k in _low(c) for k in ("illustr", "noob", "sdxl", "pony"))]
    out["sdxl"] = {"models": sorted(sd)} if sd else {"missing": "an Illustrious (SDXL) checkpoint"}
    return out


def _lora_chain(wf: dict, model: list, loras: list[tuple[str, float]], prefix: str = "lora") -> list:
    for k, (name, strength) in enumerate(loras):
        wf[f"{prefix}{k}"] = {"class_type": "LoraLoaderModelOnly",
                              "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [f"{prefix}{k}", 0]
    return model


def build_create_workflow(*, family: str, files: dict, model: str, prompt: str, negative: str, width: int,
                          height: int, seed: int, quality: str, loras: list[tuple[str, float]], refs: list[str],
                          ref_strength: float, prefix: str, quality_tags: bool = True) -> dict:
    """The API workflow for one new image (refs: uploaded picture names; ref_strength: for image to image, how far
    the new image may move away from the picture, 0..1)."""
    w, h, seed = int(width), int(height), int(seed)
    wf: dict = {}
    save = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    if family == "krea2":
        wf["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}}
        wf["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "krea2",
                                                             "device": "default"}}
        wf["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}}
        wf["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        m = _lora_chain(wf, ["unet", 0], loras)
        if refs:                                   # Image Style Reference (Krea-2 Turbo)
            if not files.get("style"):
                raise ComfyError("the Krea-2 style reference LoRA (krea2_style_reference.safetensors) is missing")
            m = _lora_chain(wf, m, [(files["style"], 1.0)], "style")
            wf["shift"] = {"class_type": "ModelSamplingFlux", "inputs": {"model": m, "max_shift": 1.15,
                                                                         "base_shift": 0.5, "width": w, "height": h}}
            pics = {f"image{k + 1}": [f"ref{k}", 0] for k in range(len(refs[:3]))}
            for k, r in enumerate(refs[:3]):
                wf[f"ref{k}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
            wf["pos0"] = {"class_type": "TextEncodeQwenImageEditPlus",
                          "inputs": {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0], **pics}}
            wf["pos"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                         "inputs": {"conditioning": ["pos0", 0], "reference_latents_method": "index_timestep_zero"}}
            wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
            wf["guider"] = {"class_type": "CFGGuider", "inputs": {"model": ["shift", 0], "positive": ["pos", 0],
                                                                  "negative": ["neg", 0], "cfg": 1.0}}
            wf["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
            wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
            wf["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": ["shift", 0], "scheduler": "simple",
                                                                       "steps": 8, "denoise": 1.0}}
            wf["sample"] = {"class_type": "SamplerCustomAdvanced",
                            "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                       "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
        else:                                      # Text to Image (Krea-2 Turbo)
            wf["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
            wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
            wf["sample"] = {"class_type": "KSampler",
                            "inputs": {"model": m, "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                       "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0],
                                       "latent_image": ["latent", 0], "denoise": 1.0}}
    elif family == "qwen":
        if refs:                                   # a new picture composed from the references (edit model)
            if not files.get("edit"):
                raise ComfyError("Qwen-Image-Edit 2511 is missing (needed for pictures as references)")
            model = files["edit"]
        wf["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}}
        wf["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "qwen_image",
                                                             "device": "default"}}
        wf["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}}
        wf["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        m = _lora_chain(wf, ["unet", 0], loras)
        light = {"fast": files.get("lightning4"), "balanced": files.get("lightning8")}.get(quality)
        if refs:
            steps, cfg = 40, 4.0
        elif light:
            m = _lora_chain(wf, m, [(light, 1.0)], "lightning")
            steps, cfg = (4 if quality == "fast" else 8), 1.0
        else:
            steps, cfg = 50, 4.0
        wf["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": m, "shift": 3.1}}
        if refs:
            wf["cfgnorm"] = {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}}
            pics = {f"image{k + 1}": [f"ref{k}", 0] for k in range(len(refs[:3]))}
            for k, r in enumerate(refs[:3]):
                wf[f"ref{k}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
            for name, text in (("pos", prompt), ("neg", negative or " ")):
                wf[name + "0"] = {"class_type": "TextEncodeQwenImageEditPlus",
                                  "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], **pics}}
                wf[name] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                            "inputs": {"conditioning": [name + "0", 0], "reference_latents_method": "index_timestep_zero"}}
            sampler_model = ["cfgnorm", 0]
        else:
            wf["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
            wf["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative or QWEN_NEGATIVE}}
            sampler_model = ["shift", 0]
        wf["sample"] = {"class_type": "KSampler",
                        "inputs": {"model": sampler_model, "seed": seed, "steps": steps, "cfg": cfg,
                                   "sampler_name": "euler", "scheduler": "simple", "positive": ["pos", 0],
                                   "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}}
    elif family == "flux3":
        # the chosen file decides: a base model samples like the template's base (50 steps, cfg 3.5), a distilled one
        # like the distilled template (16 steps, cfg 1); 'best' / 'fast' pick the other kind if no file was chosen
        unet = model if model in (files.get("models") or []) else \
            (files.get("base") if quality == "best" else files.get("distilled")) or (files.get("models") or [model])[0]
        base = "base" in unet.lower()
        steps, cfg = (50, 3.5) if base else (16, 1.0)
        wf["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        wf["clip"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "flux3",
                                                             "device": "default"}}
        wf["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}}
        wf["latent"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        m = _lora_chain(wf, ["unet", 0], loras)
        wf["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
        wf["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative or ""}}
        pos, neg = ["pos", 0], ["neg", 0]
        if refs:                                   # FLUX.3: reference latents (like the template's Image Edit)
            pics = {}
            for k, r in enumerate(refs[:3]):
                wf[f"ref{k}"] = {"class_type": "LoadImage", "inputs": {"image": r}}
                pics[f"images.image{k}"] = [f"ref{k}", 0]
            wf["refs"] = {"class_type": "ReferenceMultiLatent",
                          "inputs": {"positive": pos, "negative": neg, "vae": ["vae", 0], "resolution": "resize",
                                     "resolution.megapixels": 1.0, **pics}}
            pos, neg = ["refs", 0], ["refs", 1]
        wf["sample"] = {"class_type": "KSampler",
                        "inputs": {"model": m, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
                                   "scheduler": "simple", "positive": pos, "negative": neg,
                                   "latent_image": ["latent", 0], "denoise": 1.0}}
    elif family == "flux1":
        wf["unet"] = {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}}
        wf["clip"] = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": files["clip_l"], "clip_name2": files["t5"],
                                                                 "type": "flux", "device": "default"}}
        wf["vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}}
        m = _lora_chain(wf, ["unet", 0], loras)
        wf["pos0"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}}
        wf["pos"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["pos0", 0], "guidance": 3.5}}
        wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos0", 0]}}
        denoise = 1.0
        if refs:
            wf["ref0"] = {"class_type": "LoadImage", "inputs": {"image": refs[0]}}
            wf["refscale"] = {"class_type": "ImageScale", "inputs": {"image": ["ref0", 0], "upscale_method": "lanczos",
                                                                     "width": w, "height": h, "crop": "center"}}
            wf["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["refscale", 0], "vae": ["vae", 0]}}
            denoise = float(ref_strength)
        else:
            wf["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        wf["sample"] = {"class_type": "KSampler",
                        "inputs": {"model": m, "seed": seed, "steps": 20, "cfg": 1.0, "sampler_name": "euler",
                                   "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0],
                                   "latent_image": ["latent", 0], "denoise": denoise}}
    elif family == "sdxl":
        wf["ckpt"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}}
        m, c = ["ckpt", 0], ["ckpt", 1]
        for k, (name, strength) in enumerate(loras):          # SDXL LoRAs act on the text encoders too
            wf[f"lora{k}"] = {"class_type": "LoraLoader", "inputs": {"model": m, "clip": c, "lora_name": name,
                                                                     "strength_model": float(strength),
                                                                     "strength_clip": float(strength)}}
            m, c = [f"lora{k}", 0], [f"lora{k}", 1]
        text = f"{SDXL_QUALITY}, {prompt}" if quality_tags else prompt
        wf["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": c, "text": text}}
        wf["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": c, "text": negative or SDXL_NEGATIVE}}
        denoise = 1.0
        if refs:
            wf["ref0"] = {"class_type": "LoadImage", "inputs": {"image": refs[0]}}
            wf["refscale"] = {"class_type": "ImageScale", "inputs": {"image": ["ref0", 0], "upscale_method": "lanczos",
                                                                     "width": w, "height": h, "crop": "center"}}
            wf["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["refscale", 0], "vae": ["ckpt", 2]}}
            denoise = float(ref_strength)
        else:
            wf["latent"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        steps, cfg, sampler, sched = ((30, 6.0, "dpmpp_2m", "karras") if quality == "best" else
                                      (20, 5.5, "euler_ancestral", "normal"))
        wf["sample"] = {"class_type": "KSampler",
                        "inputs": {"model": m, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                                   "scheduler": sched, "positive": ["pos", 0], "negative": ["neg", 0],
                                   "latent_image": ["latent", 0], "denoise": denoise}}
        wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}}
        wf["create_save"] = save
        return wf
    else:
        raise ValueError(f"unknown family {family!r}")
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    wf["create_save"] = save
    return wf


def size_for(family: str, aspect: float) -> tuple[int, int]:
    """Width x height for a family at an aspect ratio (w/h): its trained pixel count, its size multiple."""
    import math
    f = FAMILIES[family]
    m = f["multiple"]
    w = math.sqrt(f["pixels"] * aspect)
    h = w / aspect
    return max(m, round(w / m) * m), max(m, round(h / m) * m)


# ----------------------------------------------------------------------------- LoRA families (header names)

_SDXL = re.compile(r"(lora_te2_|text_encoder_2|input_blocks[._]4[._]1[._]transformer_blocks[._]1)")
_FLUX1 = re.compile(r"(double_blocks|single_blocks|single_transformer_blocks)")


def lora_family(keys) -> str | None:
    """The model family a LoRA is for, from its tensor names: 'qwen', 'flux1', 'sdxl', 'krea2' or None."""
    keys = list(keys)
    joined = "\n".join(keys)
    if re.search(r"transformer_blocks[._]\d+[._]img_mlp", joined) and re.search(r"txt_mlp", joined) and \
            "single_transformer_blocks" not in joined:
        return "qwen"
    if _FLUX1.search(joined) and ("img_mlp" in joined or "img_attn" in joined or "single_transformer_blocks" in joined
                                  or "double_blocks" in joined):
        return "flux1"
    if _SDXL.search(joined):
        return "sdxl"
    # Krea-2 (checked on the official krea2_style_reference LoRA): blocks.N.attn.wq/wk/wv/wo/gate + mlp.gate/down;
    # Wan has blocks.N.self_attn.q, Z-Image / Lumina layers.N.attention: not Krea-2
    if re.search(r"blocks[._]\d+[._]attn[._](wq|wk|wv|wo)([._]|$)", joined, re.M) and \
            re.search(r"blocks[._]\d+[._]mlp[._](gate|down)", joined):
        return "krea2"
    return None


class ComfyImageCreator(ComfyGenerator):
    """Makes new images in ComfyUI (see build_create_workflow)."""

    def _check_models(self):
        def combo(node, field):
            return _combo_options(self._object_info(node).get("input", {}).get("required", {}).get(field))
        self.files = pick_files(combo("UNETLoader", "unet_name"), combo("CheckpointLoaderSimple", "ckpt_name"),
                                combo("CLIPLoader", "clip_name"), combo("VAELoader", "vae_name"),
                                combo("LoraLoaderModelOnly", "lora_name"))

    def create(self, family: str, model: str | None, prompt: str, out_png: Path, seed: int, quality: str,
               width: int, height: int, negative: str = "", loras: list[tuple[str, float]] = (),
               refs: list[Path] = (), ref_strength: float = 0.65, quality_tags: bool = True) -> dict:
        files = self.files.get(family) or {}
        if "missing" in files:
            raise ComfyError(f"{FAMILIES[family]['label']} is not available: {files['missing']}")
        model = model if model in (files.get("models") or []) else (files.get("models") or [None])[0]
        names = []
        for path, strength in loras:
            name = self.name_for(path)
            if name is None:
                raise ComfyError(f"{Path(path).name} is not in ComfyUI's LoRA folders")
            names.append((name, float(strength)))
        uploads: list[tuple[str, Path]] = []
        try:
            ref_names = []
            for r in list(refs)[:FAMILIES[family]["refs"]]:
                ref_names.append(self.upload(Path(r)))
                uploads.append((ref_names[-1], Path(r)))
            prefix = f"{SUBFOLDER}/create_{uuid.uuid4().hex[:12]}"
            wf = build_create_workflow(family=family, files=files, model=model, prompt=prompt, negative=negative,
                                       width=width, height=height, seed=seed, quality=quality, loras=names,
                                       refs=ref_names, ref_strength=ref_strength, prefix=prefix,
                                       quality_tags=quality_tags)
            ws = self._connect_ws()
            try:
                res = self._json("POST", "/prompt", {"prompt": wf, "client_id": self.client_id})
                if res.get("node_errors"):
                    raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(res['node_errors'])[:600]}")
                self.current = res["prompt_id"]
                entry = self._wait(self.current, ws)
            except ComfyJobFailed:
                raise
            except Exception:
                self.interrupt()
                raise
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.current = None
            out = self._first_file((entry.get("outputs") or {}).get("create_save"))
            if out is None:
                raise ComfyError("ComfyUI finished without the image")
            Path(out_png).write_bytes(self._download(out))
            if self.cleanup:
                self._remove(self.output_dir, out, Path(out_png))
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)
        used = (wf.get("unet") or {}).get("inputs", {}).get("unet_name") or \
            (wf.get("ckpt") or {}).get("inputs", {}).get("ckpt_name") or model
        smp = wf["sample"]["inputs"] if wf["sample"]["class_type"] == "KSampler" else {}
        return {"mode": "create", "family": family, "model": used, "chosen_model": model,
                "steps": smp.get("steps", 8), "cfg": smp.get("cfg", 1.0), "quality": quality, "width": width,
                "height": height, "negative": negative, "ref_strength": ref_strength if refs else None,
                "loras": [{"path": str(p), "strength": float(s)} for p, s in loras]}


_FAMILY_CACHE: dict[tuple, str | None] = {}


def lora_file_family(path: str) -> str | None:
    """lora_family of a .safetensors file (header only, cached by path, size and date)."""
    import os
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _FAMILY_CACHE:
        fam = None
        try:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if 0 < n < 100 * 2**20:
                    head = json.loads(f.read(n))
                    fam = lora_family(k for k in head if k != "__metadata__")
                    if fam == "krea2":                 # Krea-2's width is 6144 (another model with such names: not)
                        k = next((k for k in head if re.search(r"attn[._]wq[._]lora_(A|down)", k)), None)
                        if k is not None and (head[k].get("shape") or [0, 0])[-1] != 6144:
                            fam = None
        except (OSError, ValueError):
            fam = None
        _FAMILY_CACHE[key] = fam
    return _FAMILY_CACHE[key]


def find_image_loras(roots: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """The image-model LoRAs in ComfyUI's LoRA folders by family ('flux1', 'sdxl', 'krea2'; Qwen ones are found by
    comfy.find_qwen_loras), as (path, name in its folder), sorted by name. Speed and style-reference LoRAs that the
    workflows apply themselves are left out."""
    import os
    out: dict[str, list[tuple[str, str]]] = {"flux1": [], "sdxl": [], "krea2": []}
    seen = set()
    for root in roots:
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for f in files:
                full = os.path.join(dirpath, f)
                if not f.endswith(".safetensors") or full in seen:
                    continue
                seen.add(full)
                fam = lora_file_family(full)
                name = os.path.relpath(full, root)
                if fam in out and "style_reference" not in name.lower():
                    out[fam].append((full, name))
    return {k: sorted(v, key=lambda x: x[1].lower()) for k, v in out.items()}
