"""Wan 2.2 Animate in ComfyUI: draw another person's head into a real video ("Mix" mode, as ComfyUI's template
video_wan2_2_14B_animate): the clip's pose and face motion (DWPose from comfyui_controlnet_aux) drive the new head,
the reference picture gives its look, the background video (the clip with the head blacked out) and the head mask
keep everything else. Long clips are made in windows of 77 frames, each going on from the last 5 of the one before.
The generated frames are laid into the original ones along a soft edge of the head mask, at the clip's own size."""
from __future__ import annotations

import math
import uuid
from pathlib import Path

from .comfy import SUBFOLDER, ComfyError, ComfyGenerator, ComfyImageEditor, _combo_options

ANIMATE_NODES = ("WanAnimateToVideo", "TrimVideoLatent", "DWPreprocessor", "BlockifyMask", "DrawMaskOnImage",
                 "CLIPVisionEncode", "CLIPVisionLoader", "ModelSamplingSD3", "LoadVideo", "GetVideoComponents",
                 "CreateVideo", "SaveVideo", "ImageFromBatch", "ImageBatch", "GrowMask", "ImageToMask", "MaskToImage",
                 "ImageBlur", "ImageCompositeMasked", "ImageScale")
NODE_HELP = {"DWPreprocessor": "comfyui_controlnet_aux (custom node)",
             "BlockifyMask": "ComfyUI-KJNodes (custom node)", "DrawMaskOnImage": "ComfyUI-KJNodes (custom node)"}
WINDOW, OVERLAP = 77, 5                     # frames per window; frames of the window before it goes on from
STEPS, SHIFT = 6, 8.0                       # with the lightx2v distill LoRA (the template's settings)
SIZES = {"480p": 480, "720p": 720}          # the short side the head is made at
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走")
DEFAULT_PROMPT = "A person in the scene, moving and speaking naturally."


def window_count(frames: int) -> int:
    """Windows needed for a clip: the first gives WINDOW frames, every further one WINDOW - OVERLAP new ones."""
    return 1 + max(0, math.ceil((int(frames) - WINDOW) / (WINDOW - OVERLAP)))


def gen_size(width: int, height: int, short: int) -> tuple[int, int]:
    """The size the head is made at: the short side `short` (at most the clip's), both multiples of 16."""
    short = min(short, min(width, height))
    k = short / min(width, height)
    return max(16, round(width * k / 16) * 16), max(16, round(height * k / 16) * 16)


def pick_animate_files(unets, clips, vaes, loras, visions) -> dict:
    """Wan 2.2 Animate's files ComfyUI lists: {"unet", "clip", "vae", "vision", "lightx2v", "relight"}, or
    {"missing": what}."""
    def low(n):
        return n.lower().replace("-", "_")
    unet = sorted(u for u in unets if "animate" in low(u) and "wan2_2" in low(u).replace(".", "_"))
    clip = sorted((c for c in clips if "umt5_xxl" in low(c)), key=lambda c: ("fp8" not in low(c), c))
    vae = [v for v in vaes if Path(v).name.lower() in ("wan_2.1_vae.safetensors", "wan2.1_vae.safetensors")]
    vision = [v for v in visions if "clip_vision_h" in low(v)]
    light = sorted((x for x in loras if "lightx2v" in low(x) and "i2v_14b" in low(x) and "distill" in low(x)),
                   key=lambda x: ("rank64" not in low(x), x))
    relight = [x for x in loras if "wananimate_relight" in low(x)]
    missing = [what for what, have in (
        ("the Wan 2.2 Animate model (Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors in models/diffusion_models)",
         unet), ("its text encoder umt5_xxl", clip), ("the VAE wan_2.1_vae", vae),
        ("clip_vision_h (models/clip_vision)", vision),
        ("the lightx2v LoRA (lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16)", light),
        ("the relight LoRA (WanAnimate_relight_lora_fp16)", relight)) if not have]
    if missing:
        return {"missing": "ComfyUI does not list " + "; ".join(missing)}
    return {"unet": unet[0], "clip": clip[0], "vae": vae[0], "vision": vision[0], "lightx2v": light[0],
            "relight": relight[0]}


def build_animate_workflow(*, files: dict, video: str, mask: str, reference: str, frames: int,
                           src_size: tuple[int, int], gen: tuple[int, int], prompt: str, seed: int, prefix: str,
                           fps: float = 24.0, feather: int = 12, grow: int = 10) -> dict:
    """video / mask / reference: uploaded names (the clip, its head mask, the new person's picture); frames: the
    clip's frame count; src_size: its size; gen: the size the head is made at."""
    W, H = gen
    sw, sh = src_size
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
        "lora_speed": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0],
                                                                       "lora_name": files["lightx2v"],
                                                                       "strength_model": 1.0}},
        "lora_light": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["lora_speed", 0],
                                                                       "lora_name": files["relight"],
                                                                       "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_light", 0], "shift": SHIFT}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "wan",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "vision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": files["vision"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt or DEFAULT_PROMPT}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEGATIVE}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": reference}},
        "ref_vision": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["vision", 0],
                                                                    "image": ["ref", 0], "crop": "none"}},
        # the clip, at the size the head is made at (pose, face motion, background)
        "src": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "src_parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["src", 0]}},
        "small": {"class_type": "ImageScale", "inputs": {"image": ["src_parts", 0], "upscale_method": "lanczos",
                                                         "width": W, "height": H, "crop": "disabled"}},
        "face": {"class_type": "DWPreprocessor", "inputs": {"image": ["small", 0], "detect_hand": "disable",
                                                            "detect_body": "disable", "detect_face": "enable",
                                                            "resolution": min(W, H),
                                                            "bbox_detector": "yolox_l.onnx",
                                                            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                                                            "scale_stick_for_xinsr_cn": "disable"}},
        "pose": {"class_type": "DWPreprocessor", "inputs": {"image": ["small", 0], "detect_hand": "enable",
                                                            "detect_body": "enable", "detect_face": "disable",
                                                            "resolution": min(W, H),
                                                            "bbox_detector": "yolox_l.onnx",
                                                            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                                                            "scale_stick_for_xinsr_cn": "disable"}},
        # the head mask: blocky and a little larger for the model; soft for laying the head into the original
        "mask_vid": {"class_type": "LoadVideo", "inputs": {"file": mask}},
        "mask_parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["mask_vid", 0]}},
        "mask_small_pic": {"class_type": "ImageScale", "inputs": {"image": ["mask_parts", 0],
                                                                  "upscale_method": "bilinear", "width": W,
                                                                  "height": H, "crop": "disabled"}},
        "mask_small": {"class_type": "ImageToMask", "inputs": {"image": ["mask_small_pic", 0], "channel": "red"}},
        "mask_grow": {"class_type": "GrowMask", "inputs": {"mask": ["mask_small", 0], "expand": int(grow),
                                                           "tapered_corners": True}},
        "mask_block": {"class_type": "BlockifyMask", "inputs": {"masks": ["mask_grow", 0], "block_size": 32}},
        "background": {"class_type": "DrawMaskOnImage", "inputs": {"image": ["small", 0], "mask": ["mask_block", 0],
                                                                   "color": "0, 0, 0"}},
    }
    acc = None
    for k in range(window_count(frames)):
        a = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": W, "height": H,
             "length": WINDOW, "batch_size": 1, "continue_motion_max_frames": OVERLAP,
             "video_frame_offset": 0 if k == 0 else [f"anim{k - 1}", 5],
             "clip_vision_output": ["ref_vision", 0], "reference_image": ["ref", 0], "face_video": ["face", 0],
             "pose_video": ["pose", 0], "background_video": ["background", 0], "character_mask": ["mask_block", 0]}
        if acc is not None:
            a["continue_motion"] = acc
        wf[f"anim{k}"] = {"class_type": "WanAnimateToVideo", "inputs": a}
        wf[f"sample{k}"] = {"class_type": "KSampler",
                            "inputs": {"model": ["shift", 0], "seed": int(seed), "steps": STEPS, "cfg": 1.0,
                                       "sampler_name": "euler", "scheduler": "simple",
                                       "positive": [f"anim{k}", 0], "negative": [f"anim{k}", 1],
                                       "latent_image": [f"anim{k}", 2], "denoise": 1.0}}
        wf[f"trim{k}"] = {"class_type": "TrimVideoLatent", "inputs": {"samples": [f"sample{k}", 0],
                                                                      "trim_amount": [f"anim{k}", 3]}}
        wf[f"decode{k}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"trim{k}", 0], "vae": ["vae", 0]}}
        wf[f"new{k}"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"decode{k}", 0],
                                                                    "batch_index": [f"anim{k}", 4], "length": 4096}}
        if acc is None:
            acc = [f"new{k}", 0]
        else:
            wf[f"join{k}"] = {"class_type": "ImageBatch", "inputs": {"image1": acc, "image2": [f"new{k}", 0]}}
            acc = [f"join{k}", 0]
    wf["made"] = {"class_type": "ImageFromBatch", "inputs": {"image": acc, "batch_index": 0, "length": int(frames)}}
    # laid into the original frames: only the (softened) head changes
    wf["made_big"] = {"class_type": "ImageScale", "inputs": {"image": ["made", 0], "upscale_method": "lanczos",
                                                             "width": sw, "height": sh, "crop": "disabled"}}
    wf["soft_mask"] = {"class_type": "ImageToMask", "inputs": {"image": ["mask_parts", 0], "channel": "red"}}
    wf["soft_grow"] = {"class_type": "GrowMask", "inputs": {"mask": ["soft_mask", 0], "expand": max(0, feather // 2),
                                                            "tapered_corners": True}}
    wf["soft_pic"] = {"class_type": "MaskToImage", "inputs": {"mask": ["soft_grow", 0]}}
    wf["soft_blur"] = {"class_type": "ImageBlur", "inputs": {"image": ["soft_pic", 0], "blur_radius": max(1, feather),
                                                             "sigma": min(10.0, max(1.0, feather / 2))}}
    wf["soft"] = {"class_type": "ImageToMask", "inputs": {"image": ["soft_blur", 0], "channel": "red"}}
    wf["compose"] = {"class_type": "ImageCompositeMasked",
                     "inputs": {"destination": ["src_parts", 0], "source": ["made_big", 0], "x": 0, "y": 0,
                                "resize_source": False, "mask": ["soft", 0]}}
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["compose", 0], "audio": ["src_parts", 1],
                                                           "fps": float(fps)}}
    wf["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": prefix,
                                                        "format": "mp4"}}
    return wf


class ComfyAnimator(ComfyGenerator):
    """Wan 2.2 Animate in ComfyUI: replace() draws a new head into a clip."""

    def _check_models(self):
        def combo(node, field):
            return _combo_options(self._object_info(node).get("input", {}).get("required", {}).get(field))
        self.files = pick_animate_files(combo("UNETLoader", "unet_name"), combo("CLIPLoader", "clip_name"),
                                        combo("VAELoader", "vae_name"), combo("LoraLoaderModelOnly", "lora_name"),
                                        combo("CLIPVisionLoader", "clip_name"))
        lacking = next((n for n in ANIMATE_NODES if not self._object_info(n)), None)
        self.missing = self.files.get("missing") or (
            f"this ComfyUI has no {lacking} node" + (f" (install {NODE_HELP[lacking]})" if lacking in NODE_HELP
                                                     else " (update ComfyUI)") if lacking else "")

    _run_saving = ComfyImageEditor._run_saving

    def replace(self, video: Path, mask: Path, reference: Path, out: Path, seed: int, frames: int,
                src_size: tuple[int, int], size: str = "480p", prompt: str = "", fps: float = 24.0) -> dict:
        """Draw the person of `reference` into the head mask of `video`; the result (the clip's size, frames and
        sound) goes to out. Returns what it was made with."""
        if self.missing:
            raise ComfyError(self.missing)
        gen = gen_size(*src_size, SIZES.get(size, 480))
        uploads: list[tuple[str, Path]] = []
        try:
            names = []
            for p in (video, mask, reference):
                uploads.append((self.upload(Path(p)), Path(p)))
                names.append(uploads[-1][0])
            feather = max(4, min(20, round(min(src_size) / 90)))
            wf = build_animate_workflow(files=self.files, video=names[0], mask=names[1], reference=names[2],
                                        frames=frames, src_size=src_size, gen=gen, prompt=prompt, seed=seed,
                                        prefix=f"{SUBFOLDER}/swap_{uuid.uuid4().hex[:12]}", fps=fps,
                                        feather=feather)
        except BaseException:
            self._drop_uploads(uploads)
            raise
        self._run_saving(wf, "save", Path(out), uploads)
        return {"model": self.files["unet"], "size": size, "gen_size": list(gen), "windows": window_count(frames),
                "steps": STEPS, "seed": int(seed), "prompt": prompt or DEFAULT_PROMPT, "feather": feather}
