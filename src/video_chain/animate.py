"""Wan 2.2 Animate in ComfyUI: draw another person's head into a real video ("Mix" mode, as ComfyUI's template
video_wan2_2_14B_animate): the clip's pose and face motion (DWPose from comfyui_controlnet_aux) drive the new head,
the reference picture gives its look, the background video (the clip with the head blacked out) and the head mask
keep everything else. Long clips are made in windows of 77 frames, each going on from the last 5 of the one before.
The generated frames are laid into the original ones along a soft edge of the head mask, at the clip's own size."""
from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path

from .comfy import SUBFOLDER, ComfyError, ComfyGenerator, ComfyJobFailed, _combo_options, save_video

logger = logging.getLogger(__name__)

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
                           fps: float = 24.0, feather: int = 12, grow: int = 10, pose: dict | None = None,
                           keep_pose: bool = False, body: bool = True, lock: dict | None = None) -> dict:
    """video / mask / reference: uploaded names (the clip, its head mask, the new person's picture); frames: the
    clip's frame count; src_size: its size; gen: the size the head is made at. pose: uploaded DWPose videos of an
    earlier run, {"pose": name, "face": name} (a part given is not made again); keep_pose: the DWPose videos made
    are saved too ("pose_save", "face_save"; lossless) to be stored for the next run. body: the body skeleton
    drives the motion too (False: only the face). lock: the face swap pass after Animate that locks the identity,
    {"swapper", "enhancer", "enhance", "persons": [uploaded pictures]} (see face_lock_nodes)."""
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
        "dw_face": {"class_type": "DWPreprocessor", "inputs": {"image": ["small", 0], "detect_hand": "disable",
                                                            "detect_body": "disable", "detect_face": "enable",
                                                            "resolution": min(W, H),
                                                            "bbox_detector": "yolox_l.onnx",
                                                            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                                                            "scale_stick_for_xinsr_cn": "disable"}},
        "dw_pose": {"class_type": "DWPreprocessor", "inputs": {"image": ["small", 0], "detect_hand": "enable",
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
    pose = dict(pose or {})
    inputs = {}
    for name in ("pose", "face"):
        if name == "pose" and not body:        # no body skeleton: DWPose's body pass is not run at all
            del wf["dw_pose"]
            inputs[name] = None
        elif pose.get(name):                   # the motion of an earlier run of this very clip (cached)
            del wf[f"dw_{name}"]
            wf[f"{name}_vid"] = {"class_type": "LoadVideo", "inputs": {"file": pose[name]}}
            wf[f"{name}_parts"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"{name}_vid", 0]}}
            inputs[name] = [f"{name}_parts", 0]
        else:
            inputs[name] = [f"dw_{name}", 0]
    pose_in, face_in = inputs["pose"], inputs["face"]
    if keep_pose:                              # lossless (H.264 at crf 0, 4:4:4): the drawings exactly as made
        for name, src in (("pose", pose_in), ("face", face_in)):
            if src is not None and src[0].startswith("dw_"):
                wf[f"{name}_keep"] = {"class_type": "CreateVideo",
                                      "inputs": {"images": src, "fps": float(fps), "bit_depth": 8,
                                                 "chroma_subsampling": "444"}}
                wf[f"{name}_save"] = {"class_type": "SaveVideo",
                                      "inputs": {"video": [f"{name}_keep", 0],
                                                 "filename_prefix": f"{prefix}_{name}", "format": "mkv",
                                                 "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                 "format.codec.encoding.crf": 0.0}}
    acc = None
    for k in range(window_count(frames)):
        a = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": W, "height": H,
             "length": WINDOW, "batch_size": 1, "continue_motion_max_frames": OVERLAP,
             "video_frame_offset": 0 if k == 0 else [f"anim{k - 1}", OUT_NEXT_OFFSET],
             "clip_vision_output": ["ref_vision", 0], "reference_image": ["ref", 0], "face_video": face_in,
             "background_video": ["background", 0], "character_mask": ["mask_block", 0]}
        if pose_in is not None:
            a["pose_video"] = pose_in
        if acc is not None:
            a["continue_motion"] = acc
        wf[f"anim{k}"] = {"class_type": "WanAnimateToVideo", "inputs": a}
        wf[f"sample{k}"] = {"class_type": "KSampler",
                            "inputs": {"model": ["shift", 0], "seed": int(seed), "steps": STEPS, "cfg": 1.0,
                                       "sampler_name": "euler", "scheduler": "simple",
                                       "positive": [f"anim{k}", 0], "negative": [f"anim{k}", 1],
                                       "latent_image": [f"anim{k}", 2], "denoise": 1.0}}
        wf[f"trim{k}"] = {"class_type": "TrimVideoLatent", "inputs": {"samples": [f"sample{k}", 0],
                                                                      "trim_amount": [f"anim{k}", OUT_TRIM]}}
        wf[f"decode{k}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"trim{k}", 0], "vae": ["vae", 0]}}
        wf[f"new{k}"] = {"class_type": "ImageFromBatch", "inputs": {"image": [f"decode{k}", 0],
                                                                    "batch_index": [f"anim{k}", OUT_NEW_START],
                                                                    "length": MAX_BATCH}}
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
    out = ["compose", 0]
    if lock:                                   # the identity locked with a face swap on every frame
        out = face_lock_nodes(wf, out, lock)
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": out, "audio": ["src_parts", 1],
                                                           "fps": float(fps)}}
    wf["save"] = save_video(prefix)
    return wf


# WanAnimateToVideo's outputs by slot: 0 positive, 1 negative, 2 latent, then these
OUT_TRIM, OUT_NEW_START, OUT_NEXT_OFFSET = 3, 4, 5      # (latents to trim off, first new frame, next video_frame_offset)
MAX_BATCH = 4096                                        # ImageFromBatch: "all of it" (more than any clip has)
MAX_PERSONS = 4                                         # pictures of the new person the face swap node takes

FACE_NODE = "VideoChainFaceSwap"
FACE_HELP = ("copy video-chain's comfy_nodes/video_chain_nodes into ComfyUI/custom_nodes and restart ComfyUI; "
             "hyperswap_1b_256.onnx in models/insightface")


def face_lock_nodes(wf: dict, images: list, lock: dict) -> list:
    """Promptwood's face swap node on images (the new person's identity from lock["persons"]); its output."""
    persons = list(lock["persons"])[:MAX_PERSONS]      # (the node takes four pictures of the new person)
    inputs = {"images": images, "swapper": lock["swapper"], "enhancer": lock.get("enhancer") or "none",
              "enhance": float(lock.get("enhance", 0.8))}
    for k, name in enumerate(persons):
        wf[f"lock_person{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs["person" if k == 0 else f"person_{k + 1}"] = [f"lock_person{k}", 0]
    wf["lock"] = {"class_type": FACE_NODE, "inputs": inputs}
    return ["lock", 0]


def build_faceswap_workflow(*, video: str, lock: dict, prefix: str, fps: float = 24.0) -> dict:
    """Face only: the clip's face swapped frame by frame (no Animate; every motion is the clip's own)."""
    wf = {"src": {"class_type": "LoadVideo", "inputs": {"file": video}},
          "src_parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["src", 0]}}}
    out = face_lock_nodes(wf, ["src_parts", 0], lock)
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": out, "audio": ["src_parts", 1], "fps": float(fps)}}
    wf["save"] = save_video(prefix)
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
        # the face swap pass (Promptwood's node; HyperSwap, GPEN optional)
        info = self._object_info(FACE_NODE)
        info = info.get("input", {}).get("required", {}) if info else {}
        swappers = [n for n in _combo_options(info.get("swapper")) if "hyperswap" in n.lower()]
        enhancers = [n for n in _combo_options(info.get("enhancer")) if n != "none"]
        if not self._object_info(FACE_NODE):
            self.face_missing, self.face_files = f"this ComfyUI has no {FACE_NODE} node ({FACE_HELP})", {}
        elif not swappers:
            self.face_missing, self.face_files = f"no HyperSwap model ({FACE_HELP})", {}
        else:
            pick = next((n for n in swappers if "1b" in n), swappers[0])
            gpen = sorted(enhancers, key=lambda n: ("512" not in n, n))
            self.face_missing, self.face_files = "", {"swapper": pick, "enhancer": gpen[0] if gpen else None}

    def _run_outputs(self, wf: dict, outs: dict[str, Path], uploads: list[tuple[str, Path]]) -> None:
        """Queue a workflow, wait, download what its save nodes wrote ({node: local file}); clean up after."""
        try:
            ws = self._connect_ws()
            try:
                self._queue(wf)
                entry = self._wait(self.current, ws)
            except ComfyJobFailed:
                raise
            except BaseException:                # (also a stop by KeyboardInterrupt / SystemExit: cancel the job)
                self.interrupt()
                raise
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.current = None
            for node, dest in outs.items():
                f = self._first_file((entry.get("outputs") or {}).get(node))
                if f is None:
                    raise ComfyError(f"ComfyUI finished without the output of {node}")
                self._save(f, Path(dest))
                if self.cleanup:
                    self._remove(self.output_dir, f, Path(dest))
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)

    def replace(self, video: Path, mask: Path, reference: Path, out: Path, seed: int, frames: int,
                src_size: tuple[int, int], size: str = "480p", prompt: str = "", fps: float = 24.0,
                pose: dict | None = None, keep_pose: dict | None = None, body: bool = True,
                lock: list[Path] | None = None, enhance: float = 0.8) -> dict:
        """Draw the person of `reference` into the head mask of `video`; the result (the clip's size, frames and
        sound) goes to out. pose: {"pose": file, "face": file} stored from an earlier run of this clip at this size
        (those DWPose parts are skipped); keep_pose: {"pose": file, "face": file}, where this run's DWPose videos
        go (to be stored); body: the body skeleton drives the motion too; lock: pictures of the new person for the
        face swap pass that locks the identity (None: no pass). Returns what it was made with."""
        if self.missing:
            raise ComfyError(self.missing)
        if lock and self.face_missing:
            raise ComfyError(self.face_missing)
        gen = gen_size(*src_size, SIZES.get(size, 480))
        pose = {k: Path(v) for k, v in (pose or {}).items() if v and (k == "face" or body)}
        uploads: list[tuple[str, Path]] = []
        try:
            def up(p):
                uploads.append((self.upload(Path(p)), Path(p)))
                return uploads[-1][0]
            names = [up(p) for p in (video, mask, reference)]
            pose_names = {k: up(v) for k, v in pose.items()}
            lock_cfg = self._lock_cfg([up(p) for p in lock], enhance) if lock else None
            feather = max(4, min(20, round(min(src_size) / 90)))
            wf = build_animate_workflow(files=self.files, video=names[0], mask=names[1], reference=names[2],
                                        frames=frames, src_size=src_size, gen=gen, prompt=prompt, seed=seed,
                                        prefix=f"{SUBFOLDER}/swap_{uuid.uuid4().hex[:12]}", fps=fps,
                                        feather=feather, pose=pose_names, keep_pose=keep_pose is not None,
                                        body=body, lock=lock_cfg)
            outs = {"save": Path(out)}
            made = []
            for name in ("pose", "face"):              # the DWPose videos this run made (to be stored)
                if keep_pose and keep_pose.get(name) and f"{name}_save" in wf:
                    outs[f"{name}_save"] = Path(keep_pose[name])
                    made.append(name)
        except BaseException:
            self._drop_uploads(uploads)
            raise
        self._run_outputs(wf, outs, uploads)
        return {"method": "animate+lock" if lock else "animate", "model": self.files["unet"], "size": size,
                "gen_size": list(gen), "windows": window_count(frames), "pose_cached": sorted(pose),
                "pose_made": made, "body": bool(body), "steps": STEPS, "seed": int(seed),
                "prompt": prompt or DEFAULT_PROMPT, "feather": feather,
                **({"lock": {"swapper": self.face_files["swapper"], "enhancer": self.face_files.get("enhancer"),
                             "enhance": float(enhance) if self.face_files.get("enhancer") else 0.0}} if lock else {})}

    def _lock_cfg(self, persons: list[str], enhance: float) -> dict:
        return {"swapper": self.face_files["swapper"], "enhancer": self.face_files.get("enhancer") or "none",
                "enhance": float(enhance), "persons": persons}

    def face_only(self, video: Path, out: Path, persons: list[Path], fps: float = 24.0,
                  enhance: float = 0.8) -> dict:
        """Face only: the clip's face swapped frame by frame with HyperSwap (all motion is the clip's own)."""
        if self.face_missing:
            raise ComfyError(self.face_missing)
        uploads: list[tuple[str, Path]] = []
        try:
            for p in [video, *persons]:
                uploads.append((self.upload(Path(p)), Path(p)))
            wf = build_faceswap_workflow(video=uploads[0][0], lock=self._lock_cfg([u for u, _ in uploads[1:]],
                                                                                    enhance),
                                         prefix=f"{SUBFOLDER}/face_{uuid.uuid4().hex[:12]}", fps=fps)
        except BaseException:
            self._drop_uploads(uploads)
            raise
        self._run_outputs(wf, {"save": Path(out)}, uploads)
        return {"method": "face", "lock": {"swapper": self.face_files["swapper"],
                                           "enhancer": self.face_files.get("enhancer"),
                                           "enhance": float(enhance) if self.face_files.get("enhancer") else 0.0}}
