"""MiniMax-H3 image-to-video through a running ComfyUI (its HTTP + websocket API).

The workflow follows ComfyUI's own blueprint "Image to Video (MiniMax H3)": UNETLoader / CLIPLoader (minimax) /
the video and audio VAE, MiniMaxH3ImageToVideo with the start image as first frame, SamplerCustomAdvanced with
res_multistep + the simple scheduler and guidance 1 (BasicGuider), VAEDecode + VAEDecodeAudio, CreateVideo at
24 fps. LoRAs are chained with LoraLoaderModelOnly. The clip's last frame is saved as a PNG by ComfyUI itself
(ImageFromBatch + SaveImage), so the next clip starts from the decoded frame, not from the H.264 file.
A clip that leads into a real video gets the video's first frame as last_frame; motion guides (short clips with
their sound) are anchored with MiniMaxH3AddGuide (LoadVideo + GetVideoComponents).

ComfyUI loads the model once and keeps it as long as memory allows; free() asks it to unload.
Needs no torch: only the standard library (+ websocket-client for step progress; without it the result is
polled and there is no per-step progress).

This is a package; the names everything else uses (and the tests) are all available here, as before it was split:

  errors          ComfyError and its kinds
  wfb             the pieces every workflow builder shares (node ids, loaders, LoRA chains, the video saver)
  settings        ClipSettings: the settings dict of one clip
  loras           ComfyUI's LoRA / model folders; telling LoRA files apart by their headers
  video           build_workflow (one H3 clip)
  refine          the 2K refine: workflow, pickers, RefineMixin.refine
  seedvr          the SeedVR2 upscale: workflow, pieces, SeedvrMixin.seedvr
  client          ComfyGenerator (HTTP + websocket, uploads, generate)
  image_edit      Qwen / Qwen 2.1 / MOSS image edit workflows, ComfyImageEditor
  prompt_refiner  the prompt refiner (a language model in ComfyUI), ComfyPromptRefiner

Tests that replace a function patch it in the module that uses it (e.g. comfy.seedvr, comfy.loras), not here.
"""
from __future__ import annotations

from .errors import (ComfyError, ComfyJobFailed, ComfyUnavailable, LocalWriteError)  # noqa: F401
from .wfb import (combo_options, EDIT_SAVE_ID, FPS, H3_LORA_NODE, loader_clip, loader_unet, loader_vae, lora_chain, lora_node,
                  low_name, QWEN_SHIFT, REFINE_SAVE_ID, SAVE_LAST_ID, save_video, SAVE_VIDEO_ID, SEEDVR_SAVE_ID,
                  SUBFOLDER, TEXT_GEN_ID, text_generate_inputs, TEXT_MIN_P, TEXT_REPETITION_PENALTY, TEXT_SHOW_ID,
                  TEXT_TOP_K, TEXT_TOP_P, VIDEO_CRF)  # noqa: F401
from .settings import (ClipSettings, DEFAULT_FRAMES, DEFAULT_HEIGHT, DEFAULT_REF_SIZE, DEFAULT_STEPS, DEFAULT_WIDTH,
                       Guide, Keyframe, LoraEntry)  # noqa: F401
from .loras import (comfy_lora_roots, comfy_model_roots, find_h3_checkpoints, find_h3_loras, is_h3_checkpoint, trains_adaln, _find_loras, find_qwen_loras, _H3_ATTN,
                    _H3_MLP, is_h3_lora, is_ic_lora, is_qwen_lora, is_turbo_lora, lora_name_for, _QWEN_IMG,
                    _QWEN_TXT, _read_safetensors_keys, _safetensors_keys, _safetensors_keys_cached)  # noqa: F401
from .ffutil import (ffmpeg, FFMPEG_TIMEOUT, with_audio)  # noqa: F401
from .video import (AUDIO_VAE, DISTILLED_CHECKPOINT, build_workflow, default_audio_key, default_ref_key, default_video_key, pick_veda_predictor,
                    PIKA_SIGMAS,
                    REF_HINT, SAMPLER, SCHEDULER, TEXT_ENCODER, UNET, UNET_REF, VEDA_NODE, VEDA_SETTINGS, VIDEO_VAE,
                    with_ref_hint)  # noqa: F401
from .refine import (build_refine_workflow, CONTEXT_OVERLAP, context_window, latent_frames, pick_refine_files,
                     REFINE_DECODE_TILES, REFINE_DEFAULT, REFINE_NODES, REFINE_RELEASE, refine_sigmas, refine_size,
                     REFINE_STRENGTHS, RefineMixin, SOL_TAU, SPARSE_SETTINGS, WINDOW_TOKENS)  # noqa: F401
from .seedvr import (build_seedvr_workflow, _denoise, _join_pieces, pick_seedvr_files, SCALE_NODE, seedvr_budget,
                     SEEDVR_DEFAULT_METHOD, SEEDVR_DENOISE, SEEDVR_DOWNLOAD, SEEDVR_FRAME_COPIES, SEEDVR_FULL_VRAM,
                     SEEDVR_INTERMEDIATE_CRF, SEEDVR_JOIN_CRF, SEEDVR_METHODS, SEEDVR_NODES, SEEDVR_OVERLAP,
                     seedvr_piece_frames, SEEDVR_PIECE_OVERLAP, seedvr_pieces, SEEDVR_PRECISION_FULL,
                     SEEDVR_PRECISION_LOCAL, SEEDVR_RAM_BUDGET, SEEDVR_RAM_MIN, SEEDVR_RAM_RESERVE, SEEDVR_TILES,
                     seedvr_timing, SeedvrMixin, SeedvrProgress)  # noqa: F401
from .client import (_arg, CHUNK_BYTES, ComfyGenerator, DEFAULT_URL, DOWNLOAD_TIMEOUT, DOWNLOAD_TRIES, free_models,
                     POLL_ERRORS_MAX, POLL_SECONDS, POLL_SECONDS_FINISHED, POLL_SECONDS_NO_WS, REQUEST_TIMEOUT,
                     STOP_TIMEOUT, UPLOAD_MIN_RATE, UPLOAD_MIN_TIMEOUT, _UPLOADS, _UPLOADS_LOCK, WAIT_SLEEP_SECONDS,
                     WS_CONNECT_TIMEOUT, WS_RECV_TIMEOUT, WS_RETRY_SECONDS)  # noqa: F401
from .image_edit import (build_edit_workflow, build_moss_workflow, build_qwen21_workflow,
                         ComfyImageEditor, EDIT21_QUALITIES, EDIT_MODELS, INPAINT21_NODE,
                         INPAINT_BLUR_SIGMA_MAX, INPAINT_FEATHER, inpaint_feather, INPAINT_FEATHER_MAX,
                         INPAINT_FEATHER_MIN, is_qwen21_name, MOSS_EDIT_STEPS, MOSS_NEGATIVE, MOSS_NODES,
                         MOSS_STEPS, pick_moss_files, pick_qwen21_files, pick_qwen21_inpaint, pick_qwen_files,
                         QUALITY, _quantized, QWEN21_NODES, QWEN21_RESOLUTION, QWEN21_SIGMAS, QWEN21_STEPS,
                         QWEN_NODES, _steps_in)  # noqa: F401
from .prompt_refiner import (build_describe_face_workflow, build_refine_prompt_workflow, cast_instruction,
                             clean_refined, clip_instruction, ComfyPromptRefiner, FACE_INSTRUCTION, FACE_MAX_TOKENS,
                             FACE_TEMPERATURE, _KIND_TEXT, pick_refiner_models, REFINE_INSTRUCTIONS, REFINE_LENGTH,
                             REFINE_LENGTH_DEFAULT, refine_messages, REFINER_MODELS, _SAFETY, SOUND_FIELDS)  # noqa: F401

_combo_options = combo_options        # (the names these had before the split)
_ffmpeg = ffmpeg
_with_audio = with_audio
