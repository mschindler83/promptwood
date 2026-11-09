"""A small stand-in for ComfyUI's HTTP API (no websocket): enough for ComfyGenerator's requests.
/prompt checks the workflow's node types and links, then 'renders' it with ffmpeg (a colour clip with the
requested frame count, size and sound, plus the last frame as PNG)."""
import json
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

KNOWN = {"UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "MiniMaxH3ImageToVideo", "RandomNoise",
         "KSamplerSelect", "BasicScheduler", "BasicGuider", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio",
         "ImageFromBatch", "SaveImage", "CreateVideo", "SaveVideo", "LoraLoaderModelOnly",
         "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide", "MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler",
         "LoadVideo", "GetVideoComponents", "LoadAudio", "RepeatImageBatch",
         # Qwen Image Edit
         "FluxKontextImageScale", "ModelSamplingAuraFlow", "CFGNorm", "TextEncodeQwenImageEditPlus",
         "FluxKontextMultiReferenceLatentMethod", "VAEEncode", "KSampler", "ApplyHyperFlowH3",
         # 2K refine
         "VAEEncodeAudio", "LTXVConcatAVLatent", "LatentUpscaleModelLoader", "MiniMaxH3LatentUpscaleWithModel",
         "ManualSigmas", "VAEDecodeTiled", "BlockSparseAttention", "TextGenerate", "PreviewAny",
         # image creation
         "CLIPTextEncode", "ConditioningZeroOut", "EmptyLatentImage", "EmptySD3LatentImage", "CFGGuider",
         "ModelSamplingFlux", "DualCLIPLoader", "FluxGuidance", "CheckpointLoaderSimple", "LoraLoader", "ImageScale",
         "EmptyFlux2LatentImage", "ReferenceMultiLatent", "BatchImagesNode", "MiniMaxH3ContextWindows",
         # Qwen-Image 2.1
         "TextEncodeQwenImage21", "QwenImage21Cache",
         # Qwen-Image 2.1 inpaint
         "ModelPatchLoader", "VideoChainQwen21FunInpaint", "ImageToMask", "MaskToImage", "ImageBlur", "GrowMask",
         "ImageCompositeMasked", "EmptyImage",
         "VedaSparseAttention",
         # SeedVR2
         "SeedVR2Preprocess", "SeedVR2Conditioning", "SeedVR2PostProcessing", "SeedVR2TemporalChunk",
         "SeedVR2TemporalMerge", "VAEEncodeTiled",
         # Wan 2.2 Animate (replace a person)
         "WanAnimateToVideo", "TrimVideoLatent", "DWPreprocessor", "BlockifyMask", "DrawMaskOnImage",
         "CLIPVisionEncode", "CLIPVisionLoader", "ModelSamplingSD3", "ImageBatch", "ImageScale",
         "VideoChainFaceSwap",
         # MOSS-Image
         "T5TokenizerOptions", "TextEncodeMOSSImageEdit", "SkipLayerGuidanceDiTSimple", "ImageScaleToTotalPixels", "GetImageSize"}
IMAGE_UNETS = ["Chroma1-HD.safetensors", "Chroma1-HD-Flash.safetensors",
               "krea2/krea2_turbo_fp8_scaled.safetensors", "qwen-image/qwen_image_2512_fp8_e4m3fn.safetensors",
               "FLUX1/flux1-dev-fp8.safetensors"]
IMAGE_CKPTS = ["illustrij_v20.safetensors"]
IMAGE_CLIPS = ["qwen3vl_4b_fp8_scaled.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "clip_l.safetensors"]
IMAGE_VAES = ["FLUX1/ae.safetensors"]
IMAGE_LORAS = ["qwen-image/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
               "qwen-image/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors", "krea2/krea2_style_reference.safetensors",
               "pika-h3-flex/pika_h3_flex_fl2va_v1_bf16.safetensors", "pika-h3-flex/pika_h3_flex_ref2va_v1_bf16.safetensors"]
LLM_FILES = ["gemma-3-12b-abliterated-text-encoder.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors"]
REFINER = "minimax-h3-refiner/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
HYPERFLOW = "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors"
# the refine strengths: the release's 0.909, 0.857, 0.667 scaled to start at 0.6 / 0.7 / 0.8
SEEDVR_UNETS = ["seedvr2_3b_int8_convrot.safetensors", "seedvr2_7b_int8_convrot.safetensors",
                "seedvr2_7b_sharp_int8_convrot.safetensors", "seedvr2_7b_fp16.safetensors"]
SEEDVR_VAE = ["seedvr2_ema_vae_fp16.safetensors"]
VEDA_PREDICTORS = ["minimax_h3_r2va_veda_preview_fp8.safetensors",
                   "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"]
C_REFINE_SIGMAS = {"0.6, 0.5657, 0.4403, 0": "soft", "0.7, 0.66, 0.5136, 0": "natural", "0.8, 0.7542, 0.587, 0": "crisp"}
QWEN_UNET, QWEN_CLIP, QWEN_VAE = ("qwen-image/qwen_image_edit_2511_fp8mixed.safetensors",
                                  "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image_vae.safetensors")
QWEN_LIGHTNING = "qwen-image-edit/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN21_UNETS = ["qwen-image/qwen_image_2.1_int8_convrot.safetensors",
                "qwen-image/qwen_image_2.1_turbo_int8_convrot.safetensors"]
QWEN21_CLIP, QWEN21_VAE = "qwen3vl_8b_int8_convrot.safetensors", "qwen_image_2.1_vae_bf16.safetensors"
MOSS_UNET, MOSS_CLIP, MOSS_VAE = ("moss-image/moss_image_sft_int8_convrot.safetensors",
                                  "qwen3.5_9b_w4a8_gptq.safetensors", "flux2-vae_small.safetensors")
MOSS_EDIT = "moss-image/moss_image_edit_lora_rank_64_bf16.safetensors"


# required and optional inputs (as ComfyUI's node definitions) of the nodes whose inputs the tests check
SCHEMAS = {
    "EmptyImage": ({"width", "height", "batch_size", "color"}, set()),
    "ImageBlur": ({"image", "blur_radius", "sigma"}, set()),
    "GrowMask": ({"mask", "expand", "tapered_corners"}, set()),
    "ImageCompositeMasked": ({"destination", "source", "x", "y", "resize_source"}, {"mask"}),
    "ImageToMask": ({"image", "channel"}, set()),
    "MaskToImage": ({"mask"}, set()),
    "ModelPatchLoader": ({"name"}, set()),
    "VideoChainQwen21FunInpaint": ({"model", "model_patch", "vae", "inpaint_image", "mask", "strength"},
                                   {"control_image"}),
    "VideoChainFaceSwap": ({"images", "person", "swapper", "enhancer", "enhance"},
                           {"person_2", "person_3", "person_4"}),
    "MiniMaxH3SigmaShift": ({"model", "shift_video", "shift_audio"}, set()),
    "VedaSparseAttention": ({"model", "predictor", "generated_sparsity", "reference_sparsity", "full_attention_layers",
                             "full_attention_steps", "verbose"}, set()),
    "SeedVR2Preprocess": ({"resized_images"}, set()),
    "SeedVR2Conditioning": ({"model", "vae_conditioning"}, set()),
    "SeedVR2PostProcessing": ({"images", "original_resized_images", "color_correction_method"}, set()),
    "SeedVR2TemporalChunk": ({"latent", "temporal_overlap", "chunking_mode"}, set()),
    "SeedVR2TemporalMerge": ({"latents", "temporal_overlap"}, set()),
    "VAEEncodeTiled": ({"pixels", "vae", "tile_size", "overlap", "temporal_size", "temporal_overlap"}, set()),
    "ImageScale": ({"image", "upscale_method", "width", "height", "crop"}, set()),
    "WanAnimateToVideo": ({"positive", "negative", "vae", "width", "height", "length", "batch_size",
                           "continue_motion_max_frames", "video_frame_offset"},
                          {"clip_vision_output", "reference_image", "face_video", "pose_video", "continue_motion",
                           "background_video", "character_mask"}),
    "TrimVideoLatent": ({"samples", "trim_amount"}, set()),
    "BlockifyMask": ({"masks", "block_size"}, {"device"}),
    "DrawMaskOnImage": ({"image", "mask", "color"}, {"device"}),
    "DWPreprocessor": ({"image"}, {"detect_hand", "detect_body", "detect_face", "resolution", "bbox_detector",
                                   "pose_estimator", "scale_stick_for_xinsr_cn"}),
}


class FakeComfy:
    def __init__(self, root: Path, loras=("MiniMax H3/glow.safetensors",), fail_with: str | None = None,
                 models_missing: bool = False):
        self.root = Path(root)
        (self.root / "input").mkdir(parents=True, exist_ok=True)
        (self.root / "output").mkdir(parents=True, exist_ok=True)
        self.loras, self.fail_with, self.models_missing = list(loras), fail_with, models_missing
        self.turbo_nodes = True
        self.qwen = True                           # the Qwen Image Edit files and nodes are there
        self.qwen21 = True                         # Qwen-Image 2.1: its files and nodes
        self.moss = True                           # MOSS-Image: its files, edit LoRA and nodes
        self.inpaint21 = True                      # the Qwen-Image 2.1 Fun ControlNet and Promptwood's node
        self.animate = True                        # Wan 2.2 Animate's files, LoRAs and nodes (controlnet_aux too)
        self.faceswap = True                       # Promptwood's face swap node with HyperSwap and GPEN
        self.hyperflow = True                      # ComfyUI-HyperFlow-H3 with its weights
        self.refine = True                         # the H3 2K nodes, the latent upscaler and the refiner LoRA
        self.refine_strengths: list[str] = []      # the strength of every refine job
        self.seedvr = True                         # SeedVR2's 7B / 3B models and VAE
        self.veda = True                           # the Veda node and its two predictors
        self.seedvr_jobs: list[str] = []           # the model of every SeedVR2 job
        self.audio_refs = True                     # MiniMaxH3ReferenceToVideo takes reference sounds
        self.fail_next: str | None = None          # the next job fails with this message (once)
        self.prompts, self.calls, self.history = [], [], {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def object_info(self, node):
        from video_chain import comfy as C
        q, q21, mo = self.qwen, self.qwen21, self.moss
        files = {"UNETLoader": ("unet_name", [C.UNET, C.UNET_REF] + ([QWEN_UNET] if q else []) + IMAGE_UNETS
                                + (QWEN21_UNETS if q21 else []) + ([MOSS_UNET] if mo else [])
                                + (SEEDVR_UNETS if self.seedvr else [])),
                 "CLIPLoader": ("clip_name", [C.TEXT_ENCODER] + ([QWEN_CLIP] if q else []) + LLM_FILES + IMAGE_CLIPS
                                + ([QWEN21_CLIP] if q21 else []) + ([MOSS_CLIP] if mo else [])),
                 "VAELoader": ("vae_name", [C.VIDEO_VAE, C.AUDIO_VAE] + ([QWEN_VAE] if q else []) + IMAGE_VAES
                               + ([QWEN21_VAE] if q21 else []) + ([MOSS_VAE] if mo else [])
                               + (SEEDVR_VAE if self.seedvr else [])),
                 "CheckpointLoaderSimple": ("ckpt_name", IMAGE_CKPTS),
                 "LoraLoaderModelOnly": ("lora_name", self.loras + ([QWEN_LIGHTNING] if q else [])
                                         + ([REFINER] if self.refine else []) + IMAGE_LORAS
                                         + ([MOSS_EDIT] if mo else [])),
                 "LatentUpscaleModelLoader": ("model_name", [UPSCALER] if self.refine else []),
                 "ModelPatchLoader": ("name", ["qwen_image_inpaint_diffsynth_controlnet.safetensors"]
                                      + (["Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"] if self.inpaint21
                                         else []))}
        an = self.animate
        if node == "UNETLoader" and an:
            files["UNETLoader"] = ("unet_name", files["UNETLoader"][1] + ["Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"])
        if node == "CLIPLoader" and an:
            files["CLIPLoader"] = ("clip_name", files["CLIPLoader"][1] + ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"])
        if node == "VAELoader" and an:
            files["VAELoader"] = ("vae_name", files["VAELoader"][1] + ["wan_2.1_vae.safetensors"])
        if node == "LoraLoaderModelOnly" and an:
            files["LoraLoaderModelOnly"] = ("lora_name", files["LoraLoaderModelOnly"][1] + [
                "wan2-2-Lightning/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
                "WanAnimate_relight_lora_fp16.safetensors"])
        if node == "VideoChainFaceSwap":
            return {node: {"input": {"required": {
                "swapper": [["hyperswap_1a_256.onnx", "hyperswap_1b_256.onnx"]],
                "enhancer": [["none", "GPEN-BFR-1024.onnx", "GPEN-BFR-512.onnx"]]}}}} if self.faceswap else {}
        if node == "CLIPVisionLoader":
            return {node: {"input": {"required": {"clip_name": ["COMBO", {"options": ["clip_vision_h.safetensors"]
                                                                          if an else []}]}}}}
        if node in ("WanAnimateToVideo", "TrimVideoLatent", "DWPreprocessor", "BlockifyMask", "DrawMaskOnImage",
                    "CLIPVisionEncode", "ModelSamplingSD3", "ImageBatch", "ImageScale", "ImageFromBatch",
                    "CreateVideo", "SaveVideo", "GrowMask", "ImageToMask", "MaskToImage", "ImageBlur",
                    "ImageCompositeMasked"):
            return {node: {"input": {"required": {}}}} if an or node not in ("WanAnimateToVideo", "DWPreprocessor") \
                else {}
        if node in C.SEEDVR_NODES and node not in ("LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo",
                                                   "ImageScale", "VAEDecodeTiled"):
            return {node: {"input": {"required": {}}}} if self.seedvr else {}
        if node == "VedaSparseAttention":
            return {node: {"input": {"required": {"predictor": ["COMBO", {"options": VEDA_PREDICTORS}]}}}} \
                if self.veda else {}
        if node == "VideoChainQwen21FunInpaint":
            return {node: {"input": {"required": {}}}} if self.inpaint21 else {}
        if node in C.REFINE_NODES + ("BlockSparseAttention",) and node != "LatentUpscaleModelLoader":
            return {node: {"input": {"required": {}}}} if self.refine else {}
        if node == "TextGenerate":
            return {node: {"input": {"required": {}}}}
        if node == "ApplyHyperFlowH3":
            return {node: {"input": {"required": {"hyperflow_file": [[HYPERFLOW, "custom_node_hyperflow_8step_v1.0_comfyui.safetensors"]]}}}} \
                if self.hyperflow else {}
        if node in C.QWEN21_NODES:
            return {node: {"input": {"required": {}}}} if q21 else {}
        if node in C.MOSS_NODES:
            return {node: {"input": {"required": {}}}} if mo else {}
        if node in C.QWEN_NODES:
            return {node: {"input": {"required": {}}}} if q else {}
        if node in ("MiniMaxH3ImageToVideo", "MiniMaxH3TurboSampler", "MiniMaxH3TurboLoRA", "MiniMaxH3AddGuide",
                    "LoadVideo", "GetVideoComponents"):
            return {} if node.startswith("MiniMaxH3Turbo") and not self.turbo_nodes else {node: {"input": {"required": {}}}}
        if node == "MiniMaxH3ReferenceToVideo":
            opt = {"ref_images": ["COMFY_AUTOGROW_V3", {"template": {"prefix": "ref_image_", "min": 0, "max": 9}}]}
            if self.audio_refs:
                opt["ref_audios"] = ["COMFY_AUTOGROW_V3", {"template": {"prefix": "ref_audio_", "min": 0, "max": 3}}]
            return {node: {"input": {"required": {}, "optional": opt}}}
        if node in files:
            field, names = files[node]
            names = [] if self.models_missing and node == "UNETLoader" else names
            return {node: {"input": {"required": {field: ["COMBO", {"options": names}]}}}}
        return {}

    def render(self, wf: dict) -> dict:
        for nid, n in wf.items():
            assert n["class_type"] in KNOWN, n["class_type"]
            spec = SCHEMAS.get(n["class_type"])
            if spec is not None:                     # the inputs real ComfyUI takes for these nodes
                need, may = spec
                keys = set(n["inputs"])
                assert need <= keys, f"{nid} ({n['class_type']}) lacks {need - keys}"
                assert keys <= need | may, f"{nid} ({n['class_type']}) has unknown inputs {keys - need - may}"
            for v in n["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                    assert v[0] in wf, f"{nid} links to missing node {v[0]}"
        if "create_save" in wf:                      # a new image at the latent's size (or the scaled reference)
            for nid, node in wf.items():
                if node["class_type"] == "LoadImage":
                    assert (self.root / "input" / node["inputs"]["image"]).is_file(), f"{nid} not uploaded"
            lat = wf.get("refscale") or wf["latent"]
            w, h = lat["inputs"]["width"], lat["inputs"]["height"]
            prefix = wf["create_save"]["inputs"]["filename_prefix"]
            out = self.root / "output" / (prefix + "_00001_.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=orange:s={w}x{h}",
                            "-frames:v", "1", str(out)], check=True)
            self.created = wf
            return {"create_save": {"images": [{"filename": out.name, "subfolder": str(Path(prefix).parent),
                                                "type": "output"}]}}
        if "gen" in wf and wf["gen"]["class_type"] == "TextGenerate":     # the prompt refiner: a fake answer
            g = wf["gen"]["inputs"]
            self.refined = g
            return {"show": {"text": [f"<think>planning</think>Here is the refined prompt: \"REFINED {g['prompt'][:40]}\""]}}
        if "lock" in wf and "anim0" not in wf and wf.get("save", {}).get("class_type") == "SaveVideo":
            self.faced = wf                          # face only: the clip again (its face "swapped")
            src = self.root / "input" / wf["src"]["inputs"]["file"]
            prefix = wf["save"]["inputs"]["filename_prefix"]
            out = self.root / "output" / (prefix + "_00001_.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "hue=h=180", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)
            return {"save": {"images": [{"filename": out.name, "subfolder": str(Path(prefix).parent),
                                         "type": "output"}]}}
        if "anim0" in wf and wf.get("save", {}).get("class_type") == "SaveVideo":   # Wan Animate: the clip again
            self.animated = wf
            src = self.root / "input" / wf["src"]["inputs"]["file"]
            mask = self.root / "input" / wf["mask_vid"]["inputs"]["file"]
            assert src.is_file() and mask.is_file(), "clip or mask not uploaded"
            n = wf["made"]["inputs"]["length"]
            prefix = wf["save"]["inputs"]["filename_prefix"]
            out = self.root / "output" / (prefix + "_00001_.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "hue=h=90", "-frames:v", str(n),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)
            res = {"save": {"images": [{"filename": out.name, "subfolder": str(Path(prefix).parent),
                                        "type": "output"}]}}
            for name in ("pose", "face"):            # the DWPose videos kept for the cache
                if f"{name}_vid" in wf:
                    assert (self.root / "input" / wf[f"{name}_vid"]["inputs"]["file"]).is_file()
                if f"{name}_save" in wf:
                    sv = wf[f"{name}_save"]["inputs"]
                    assert sv["format"] == "mkv" and sv["format.codec.encoding.crf"] == 0.0
                    f = self.root / "output" / (sv["filename_prefix"] + "_00001_.mkv")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", str(n), "-c:v",
                                    "libx264", "-qp", "0", str(f)], check=True)
                    res[f"{name}_save"] = {"images": [{"filename": f.name, "subfolder": str(Path(prefix).parent),
                                                       "type": "output"}]}
            return res
        if "seedvr_save" in wf:                      # SeedVR2: the draft at the target size, its frames and sound
            draft = self.root / "input" / wf["load"]["inputs"]["file"]
            assert draft.is_file(), "draft not uploaded"
            assert wf["sample"]["inputs"]["steps"] == 1 and wf["chunk"]["inputs"]["chunking_mode"] == "auto"
            assert wf["merge"]["inputs"]["temporal_overlap"] == ["chunk", 1]
            self.seedvr_jobs.append(wf["unet"]["inputs"]["unet_name"])
            rs = wf["resize"]["inputs"]
            prefix = wf["seedvr_save"]["inputs"]["filename_prefix"]
            out = self.root / "output" / (prefix + "_00001_.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(draft), "-vf",
                            f"scale={rs['width']}:{rs['height']}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", str(out)], check=True)
            return {"seedvr_save": {"images": [{"filename": out.name, "subfolder": str(Path(prefix).parent),
                                                "type": "output"}]}}
        if "refine_save" in wf:                      # 2K refine: the draft at the target size, its frames and sound
            draft = self.root / "input" / wf["draft"]["inputs"]["file"]
            assert draft.is_file(), "draft not uploaded"
            assert wf["sigmas"]["inputs"]["sigmas"] in C_REFINE_SIGMAS, wf["sigmas"]["inputs"]["sigmas"]
            self.refine_strengths.append(C_REFINE_SIGMAS[wf["sigmas"]["inputs"]["sigmas"]])
            up = wf["up"]["inputs"]
            prefix = wf["refine_save"]["inputs"]["filename_prefix"]
            out = self.root / "output" / (prefix + "_00001_.mp4")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(draft), "-vf",
                            f"scale={up['width']}:{up['height']}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", str(out)], check=True)
            return {"refine_save": {"images": [{"filename": out.name, "subfolder": str(Path(prefix).parent),
                                                "type": "output"}]}}
        if "edit_save" in wf:                        # Qwen Image Edit: a 1024x576 picture
            self.edited = wf
            for nid, node in wf.items():
                if node["class_type"] == "LoadImage":
                    assert (self.root / "input" / node["inputs"]["image"]).is_file(), f"{nid} not uploaded"
            prefix = wf["edit_save"]["inputs"]["filename_prefix"]
            out = self.root / "output" / (prefix + "_00001_.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=1024x576",
                            "-frames:v", "1", str(out)], check=True)
            return {"edit_save": {"images": [{"filename": out.name, "subfolder": str(Path(prefix).parent),
                                              "type": "output"}]}}
        if "refcond" in wf:                         # identity references: ref2va + the start image as guide
            cond = wf["refcond"]["inputs"]
            assert wf["unet"]["inputs"]["unet_name"].startswith("minimax_h3_ref2va")
            if "start" in wf:
                assert wf["cond"]["class_type"] == "MiniMaxH3AddGuide"
                still = wf.get("start_still")      # faces: the start image held as a short still
                assert wf["cond"]["inputs"]["image"] == (["start_still", 0] if still else ["start", 0])
                assert still is None or still["inputs"]["image"] == ["start", 0]
            for k, v in cond.items():
                if k.startswith("ref_images."):
                    assert (self.root / "input" / wf[v[0]]["inputs"]["image"]).is_file(), "reference not uploaded"
                if k.startswith("ref_audios."):
                    assert self.audio_refs and wf[v[0]]["class_type"] == "LoadAudio"
                    assert (self.root / "input" / wf[v[0]]["inputs"]["audio"]).is_file(), "sound not uploaded"
                    assert cond["audio_vae"] == ["audio_vae", 0]
        else:
            cond = wf["cond"]["inputs"]
            assert ("first_frame" in cond) == ("start" in wf) and ("last_frame" in cond) == ("end" in wf)
        for name in ("start", "end"):
            if name in wf:
                assert (self.root / "input" / wf[name]["inputs"]["image"]).is_file(), f"{name} image not uploaded"
        for nid, node in wf.items():
            if node["class_type"] == "LoadVideo":
                assert (self.root / "input" / node["inputs"]["file"]).is_file(), "guide clip not uploaded"
        w, h, n = cond["width"], cond["height"], cond["length"]
        prefix = wf["save_video"]["inputs"]["filename_prefix"]
        out = self.root / "output" / (prefix + "_00001_.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        args = ["-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=24"]
        if "audio" in wf["video"]["inputs"]:
            args += ["-f", "lavfi", "-i", "sine=f=440", "-c:a", "aac", "-t", f"{n / 24:.4f}"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-frames:v", str(n), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        last = self.root / "output" / (wf["save_last"]["inputs"]["filename_prefix"] + "_00001_.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=red:s={w}x{h}",
                        "-frames:v", "1", str(last)], check=True)
        sub = str(Path(prefix).parent)
        return {"save_video": {"images": [{"filename": out.name, "subfolder": sub, "type": "output"}],
                               "animated": [True]},
                "save_last": {"images": [{"filename": last.name, "subfolder": sub, "type": "output"}]}}

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                fake.calls.append(("GET", u.path))
                if u.path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "fake"}})
                if u.path.startswith("/object_info/"):
                    return self._send(200, fake.object_info(u.path.rsplit("/", 1)[1]))
                if u.path.startswith("/history/"):
                    pid = u.path.rsplit("/", 1)[1]
                    return self._send(200, {pid: fake.history[pid]} if pid in fake.history else {})
                if u.path == "/view":
                    q = urllib.parse.parse_qs(u.query)
                    f = fake.root / q["type"][0] / q.get("subfolder", [""])[0] / q["filename"][0]
                    return self._send(200, f.read_bytes(), "application/octet-stream")
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                u = urllib.parse.urlparse(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.calls.append(("POST", u.path))
                if u.path == "/upload/image":
                    ctype = self.headers["Content-Type"]
                    boundary = ctype.split("boundary=")[1].encode()
                    fields, fname, data = {}, None, None
                    for part in body.split(b"--" + boundary):
                        if b"\r\n\r\n" not in part:
                            continue
                        head, val = part.split(b"\r\n\r\n", 1)
                        val = val[:-2] if val.endswith(b"\r\n") else val
                        name = head.split(b'name="')[1].split(b'"')[0].decode()
                        if b"filename=" in head:
                            fname = head.split(b'filename="')[1].split(b'"')[0].decode()
                            data = val
                        else:
                            fields[name] = val.decode()
                    d = fake.root / "input" / fields.get("subfolder", "")
                    d.mkdir(parents=True, exist_ok=True)
                    (d / fname).write_bytes(data)
                    return self._send(200, {"name": fname, "subfolder": fields.get("subfolder", ""),
                                            "type": "input"})
                payload = json.loads(body or b"{}")
                if u.path == "/prompt":
                    fake.prompts.append(payload["prompt"])
                    pid = f"p{len(fake.prompts)}"
                    if fake.fail_with or fake.fail_next:
                        msg, fake.fail_next = fake.fail_with or fake.fail_next, None
                        fake.history[pid] = {"outputs": {}, "status": {"status_str": "error", "completed": False,
                                             "messages": [["execution_error", {"exception_type": "RuntimeError",
                                                                               "exception_message": msg}]]}}
                    else:
                        fake.history[pid] = {"outputs": fake.render(payload["prompt"]),
                                             "status": {"status_str": "success", "completed": True}}
                    return self._send(200, {"prompt_id": pid, "number": 1, "node_errors": {}})
                if u.path in ("/free", "/interrupt", "/queue"):
                    fake.calls.append(("BODY", u.path, payload))
                    return self._send(200, {})
                return self._send(404, {"error": "not found"})
        return H
