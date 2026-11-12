# What setup downloads

Written by `tools/build_manifest.py` from the installer's list (`src/video_chain/setup/manifest.json`). Every file is fetched from the address below at a fixed revision and checked by size and SHA-256. Each model comes under its own licence, shown on its page; you accept it by downloading the model.

| Profile | Parts | Download |
|---|---|---|
| `core` | video | 39.6 GB |
| `standard` | video, video-refs, video-turbo, video-2k, upscale, image, faces, tts, llm | 125.8 GB |
| `full` | video, video-refs, video-turbo, video-2k, upscale, image, faces, tts, llm, video-speed, image-krea2, headswap-h3, headswap-wan, upscale-3b | 183.6 GB |

## video: Video clips (MiniMax H3, image to video with sound)

Profile `core`, 39.6 GB. Licence with its own terms: MiniMax H3 Community License.

| File | Size | Source |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |
| `minimax_h3_video_vae_fp16.safetensors` | 4.9 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |

## video-refs: Face and sound references in clips (H3 reference-to-video)

Profile `standard`, 19.5 GB. Needs: video. Licence with its own terms: MiniMax H3 Community License.

| File | Size | Source |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |

## video-turbo: Turbo clips (8 / 4 steps)

Profile `standard`, 3.6 GB. Needs: video. Custom nodes: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.

| File | Size | Source |
|---|---|---|
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1.8 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.8 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |

## video-2k: 2K refine pass (Open-H3-Regenerate-2K)

Profile `standard`, 8.4 GB. Needs: video.

| File | Size | Source |
|---|---|---|
| `open_h3_regenerate_2k_refiner_lora.safetensors` | 7.7 GB | https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-LoRA |
| `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | 0.6 GB | https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K |

## upscale: Export upscaling (SeedVR2 7B)

Profile `standard`, 8.2 GB.

| File | Size | Source |
|---|---|---|
| `seedvr2_7b_int8_convrot.safetensors` | 7.8 GB | https://huggingface.co/Comfy-Org/SeedVR2 |
| `seedvr2_ema_vae_fp16.safetensors` | 0.5 GB | https://huggingface.co/Comfy-Org/SeedVR2 |

## image: Pictures: create, edit and inpaint (Qwen-Image 2.1)

Profile `standard`, 26.4 GB. Custom nodes: bundled with Promptwood.

| File | Size | Source |
|---|---|---|
| `qwen_image_2.1_int8_convrot.safetensors` | 6.8 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1 |
| `qwen_image_2.1_turbo_int8_convrot.safetensors` | 6.8 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1 |
| `qwen3vl_8b_int8_convrot.safetensors` | 8.7 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1 |
| `qwen_image_2.1_vae_bf16.safetensors` | 0.6 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1 |
| `qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors` | 3.5 GB | https://huggingface.co/Comfy-Org/Qwen-Image-2.1 |

## faces: Faces: crop to face, face swap (SCRFD, ArcFace, HyperSwap, GPEN)

Profile `standard`, 0.9 GB. Custom nodes: bundled with Promptwood.

| File | Size | Source |
|---|---|---|
| det_10g.onnx, w600k_r50.onnx | 0.3 GB | https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip |
| `hyperswap_1b_256.onnx` | 0.4 GB | https://huggingface.co/facefusion/models-3.3.0 |
| `GPEN-BFR-512.onnx` | 0.3 GB | https://huggingface.co/facefusion/models-3.0.0 |

## tts: Spoken lines and voices (Qwen3-TTS, Whisper)

Profile `standard`, 10.0 GB.

| File | Size | Source |
|---|---|---|
| 11 files (Hugging Face cache) | 4.2 GB | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| 11 files (Hugging Face cache) | 4.2 GB | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign |
| 11 files (Hugging Face cache) | 1.5 GB | https://huggingface.co/openai/whisper-large-v3-turbo |

## llm: Prompt refiner (llama.cpp with Qwen3-VL 8B, sees pictures)

Profile `standard`, 9.2 GB.

| File | Size | Source |
|---|---|---|
| 2 files (Hugging Face cache) | 9.2 GB | https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF |

## video-speed: More speed modes (HyperFlow, Pika H3 Flex, Veda sparse attention)

Profile `full`, 7.5 GB. Needs: video. Custom nodes: https://github.com/Adudeguyman/ComfyUI-HyperFlow-H3, https://github.com/veda-sparse/Veda-on-ComfyUI.

| File | Size | Source |
|---|---|---|
| `custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors` | 3.6 GB | https://huggingface.co/drbaph/Hyperflow-Comfyui |
| `pika_h3_flex_fl2va_v1_bf16.safetensors` | 1.8 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |
| `pika_h3_flex_ref2va_v1_bf16.safetensors` | 1.8 GB | https://huggingface.co/Comfy-Org/MiniMax-H3 |
| `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors` | 0.3 GB | https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview |

## image-krea2: Pictures with Krea-2 Turbo (and style references)

Profile `full`, 17.8 GB.

| File | Size | Source |
|---|---|---|
| `krea2_turbo_fp8_scaled.safetensors` | 12.2 GB | https://huggingface.co/Comfy-Org/Krea-2 |
| `qwen3vl_4b_fp8_scaled.safetensors` | 4.9 GB | https://huggingface.co/Comfy-Org/Krea-2 |
| `qwen_image_vae.safetensors` | 242 MB | https://huggingface.co/Comfy-Org/Krea-2 |
| `krea2_style_reference.safetensors` | 0.4 GB | https://huggingface.co/Comfy-Org/Krea-2 |

## headswap-h3: Head swap in clips (H3 head swap LoRA)

Profile `full`, 2.4 GB. Needs: video-refs, faces. Custom nodes: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.

| File | Size | Source |
|---|---|---|
| `minimax_h3_head_swap_v1.0_r32.safetensors` | 0.3 GB | https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video |
| `minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors` | 2.1 GB | https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI |

## headswap-wan: Head swap in clips (Wan 2.2 Animate)

Profile `full`, 26.9 GB. Needs: faces. Custom nodes: https://github.com/Fannovel16/comfyui_controlnet_aux, https://github.com/kijai/ComfyUI-KJNodes.

| File | Size | Source |
|---|---|---|
| `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` | 17.1 GB | https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3 GB | https://huggingface.co/Comfy-Org/Wan-Animate-2 |
| `clip_vision_h.safetensors` | 1.2 GB | https://huggingface.co/Comfy-Org/Wan-Animate-2 |
| `wan_2.1_vae.safetensors` | 242 MB | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | 0.7 GB | https://huggingface.co/Comfy-Org/Wan-Animate-2 |
| `WanAnimate_relight_lora_fp16.safetensors` | 1.3 GB | https://huggingface.co/Kijai/WanVideo_comfy |

## upscale-3b: Export upscaling, smaller model (SeedVR2 3B)

Profile `full`, 3.2 GB. Needs: upscale.

| File | Size | Source |
|---|---|---|
| `seedvr2_3b_int8_convrot.safetensors` | 3.2 GB | https://huggingface.co/Comfy-Org/SeedVR2 |

## Programs

| Program | Source |
|---|---|
| ComfyUI at `1332fabf` | https://github.com/comfyanonymous/ComfyUI |
| PyTorch (CUDA build chosen from the graphics driver) | https://download.pytorch.org/whl/ |
| llama.cpp `b12317` (release build) | https://github.com/ggml-org/llama.cpp |
| Qwen3-TTS (`qwen-tts==0.1.1`) | https://pypi.org/project/qwen-tts/ |
| ffmpeg (only when the system has none with libx264) | https://github.com/BtbN/FFmpeg-Builds |
| uv (only when the system has none) | https://docs.astral.sh/uv/ |
