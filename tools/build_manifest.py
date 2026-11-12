#!/usr/bin/env python3
"""Maintainer tool: writes src/video_chain/setup/manifest.json from the list below.

For every model file it asks Hugging Face for the size, the SHA-256 and the commit, so the installer can check each
download and the list never silently points at a changed file. Run it after changing SPEC:

    python tools/build_manifest.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "video_chain" / "setup" / "manifest.json"
DOC = Path(__file__).resolve().parents[1] / "docs" / "MODELS.md"


def m(repo, path, folder, name=None, alt=None):
    """A file for ComfyUI's models/<folder>. alt: a pattern (lower-case file name) of files the app uses just as
    well (another precision or build of the same model): one of those on the computer and nothing is downloaded."""
    return {"repo": repo, "path": path, "dir": folder, "name": name or path.rsplit("/", 1)[-1],
            **({"alt": alt} if alt else {})}


def hf(repo, *patterns, alt=None):
    """Files of a repo for the Hugging Face cache (all of them, or those whose name contains a pattern).
    alt "gguf-chat": any GGUF chat model with a vision projector in a Hugging Face cache does as well."""
    return {"hf_repo": repo, "patterns": list(patterns), **({"alt": alt} if alt else {})}


H3 = "Comfy-Org/MiniMax-H3"
Q21 = "Comfy-Org/Qwen-Image-2.1"
KREA = "Comfy-Org/Krea-2"
WAN = "Comfy-Org/Wan-Animate-2"

# profile: the smallest profile a component belongs to (core < standard < full)
SPEC = {
    "video": {
        "title": "Video clips (MiniMax H3, image to video with sound)", "profile": "core",
        "licence": "MiniMax H3 Community License", "licence_url": f"https://huggingface.co/{H3}",
        "files": [m(H3, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models"),
                  m(H3, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders"),
                  m(H3, "vae/minimax_h3_video_vae_fp16.safetensors", "vae"),
                  m(H3, "vae/minimax_h3_audio_vae_fp32.safetensors", "vae")],
    },
    "video-refs": {
        "title": "Face and sound references in clips (H3 reference-to-video)", "profile": "standard",
        "needs": ["video"], "licence": "MiniMax H3 Community License",
        "files": [m(H3, "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models")],
    },
    "video-turbo": {
        "title": "Turbo clips (8 / 4 steps)", "profile": "standard", "needs": ["video"],
        "nodes": ["h3-turbo"],
        "files": [m(H3, "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "loras",
                    alt=r"minimax_h3_fl2v.*turbo.*\.safetensors$"),
                  m(H3, "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras",
                    alt=r"minimax_h3_ref2v.*turbo.*\.safetensors$")],
    },
    "video-2k": {
        "title": "2K refine pass (Open-H3-Regenerate-2K)", "profile": "standard", "needs": ["video"],
        "files": [m("Efficient-Large-Model/Open-H3-Regenerate-2K-LoRA", "open_h3_regenerate_2k_refiner_lora.safetensors",
                    "loras", alt=r"regenerate[_-]2k.*refiner.*\.safetensors$"),
                  m("Efficient-Large-Model/Open-H3-Regenerate-2K",
                    "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                    "latent_upscale_models", alt=r"(minimax_h3|h3_latent).*upscal.*\.safetensors$")],
    },
    "upscale": {
        "title": "Export upscaling (SeedVR2 7B)", "profile": "standard",
        "files": [m("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_7b_int8_convrot.safetensors", "diffusion_models"),
                  m("Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", "vae")],
    },
    "image": {
        "title": "Pictures: create, edit and inpaint (Qwen-Image 2.1)", "profile": "standard",
        "nodes": ["video-chain"],
        "files": [m(Q21, "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models",
                    alt=r"^(?!.*turbo)qwen[_-]?image[_-]?2\\.?1.*\\.safetensors$"),
                  m(Q21, "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", "diffusion_models",
                    alt=r"qwen[_-]?image[_-]?2\\.?1.*turbo.*\\.safetensors$"),
                  m(Q21, "text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders",
                    alt=r"qwen3vl_8b(?!.*joyimage).*\.safetensors$"),
                  m(Q21, "vae/qwen_image_2.1_vae_bf16.safetensors", "vae", alt=r"qwen_image_2\.?1_vae.*\.safetensors$"),
                  m(Q21, "model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors",
                    "model_patches", alt=r"qwen[_-]?image[_-]?2\\.?1.*controlnet.*\\.safetensors$")],
    },
    "faces": {
        "title": "Faces: crop to face, face swap (SCRFD, ArcFace, HyperSwap, GPEN)", "profile": "standard",
        "nodes": ["video-chain"], "app_extra": "faces",
        "files": [{"url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                   "dir": "insightface/models/buffalo_l", "extract": ["det_10g.onnx", "w600k_r50.onnx"]},
                  m("facefusion/models-3.3.0", "hyperswap_1b_256.onnx", "insightface", alt=r"hyperswap.*\.onnx$"),
                  m("facefusion/models-3.0.0", "gpen_bfr_512.onnx", "facerestore_models", "GPEN-BFR-512.onnx",
                    alt=r"gpen.*\.onnx$")],
    },
    "tts": {
        "title": "Spoken lines and voices (Qwen3-TTS, Whisper)", "profile": "standard", "software": ["tts"],
        "files": [hf("Qwen/Qwen3-TTS-12Hz-1.7B-Base"), hf("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
                  hf("openai/whisper-large-v3-turbo", "model.safetensors", ".json", ".txt")],
    },
    "llm": {
        "title": "Prompt refiner (llama.cpp with Qwen3-VL 8B, sees pictures)", "profile": "standard",
        "software": ["llama"],
        "files": [hf("unsloth/Qwen3-VL-8B-Instruct-GGUF", "Qwen3-VL-8B-Instruct-Q8_0.gguf", "mmproj-BF16.gguf",
                     alt="gguf-chat")],
    },
    "video-speed": {
        "title": "More speed modes (HyperFlow, Pika H3 Flex, Veda sparse attention)", "profile": "full",
        "needs": ["video"], "nodes": ["hyperflow", "veda"],
        "files": [m("drbaph/Hyperflow-Comfyui", "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors",
                    "hyperflow"),
                  m(H3, "loras/pika_h3_flex_fl2va_v1_bf16.safetensors", "loras"),
                  m(H3, "loras/pika_h3_flex_ref2va_v1_bf16.safetensors", "loras"),
                  m("Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
                    "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors", "veda")],
    },
    "image-krea2": {
        "title": "Pictures with Krea-2 Turbo (and style references)", "profile": "full",
        "files": [m(KREA, "diffusion_models/krea2_turbo_fp8_scaled.safetensors", "diffusion_models"),
                  m(KREA, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
                  m(KREA, "vae/qwen_image_vae.safetensors", "vae"),
                  m(KREA, "loras/krea2_style_reference.safetensors", "loras")],
    },
    "headswap-h3": {
        "title": "Head swap in clips (H3 head swap LoRA)", "profile": "full", "needs": ["video-refs", "faces"],
        "nodes": ["h3-turbo"],
        "files": [m("Alissonerdx/BFS-Best-Face-Swap-Video", "h3/minimax_h3_head_swap_v1.0_r32.safetensors", "loras"),
                  m("drbaph/MiniMax-H3-Turbo-Lora-ComfyUI",
                    "experimental/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors", "loras")],
    },
    "headswap-wan": {
        "title": "Head swap in clips (Wan 2.2 Animate)", "profile": "full", "needs": ["faces"],
        "nodes": ["controlnet-aux", "kjnodes"],
        "files": [m("Kijai/WanVideo_comfy_fp8_scaled", "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
                    "diffusion_models"),
                  m(WAN, "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
                  m(WAN, "clip_vision/clip_vision_h.safetensors", "clip_vision"),
                  m("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae"),
                  m(WAN, "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
                  m("Kijai/WanVideo_comfy", "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors", "loras")],
    },
    "upscale-3b": {
        "title": "Export upscaling, smaller model (SeedVR2 3B)", "profile": "full", "needs": ["upscale"],
        "files": [m("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors", "diffusion_models")],
    },
}

# Custom nodes, pinned to the commits the app was tested with
NODES = {
    "video-chain": {"bundled": "comfy_nodes/video_chain_nodes", "pip": ["onnxruntime-gpu", "opencv-python-headless"]},
    "h3-turbo": {"git": "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo", "rev": "4274783",
                 "dir": "ComfyUI-MiniMax-H3-Turbo"},
    "hyperflow": {"git": "https://github.com/Adudeguyman/ComfyUI-HyperFlow-H3", "rev": "9977890",
                  "dir": "ComfyUI-HyperFlow-H3"},
    "veda": {"git": "https://github.com/veda-sparse/Veda-on-ComfyUI", "rev": "67e4eb5", "dir": "Veda-on-ComfyUI"},
    "controlnet-aux": {"git": "https://github.com/Fannovel16/comfyui_controlnet_aux", "rev": "a587a4c",
                       "dir": "comfyui_controlnet_aux"},
    "kjnodes": {"git": "https://github.com/kijai/ComfyUI-KJNodes", "rev": "d3cfe21", "dir": "comfyui-kjnodes"},
}

COMFY = {"git": "https://github.com/comfyanonymous/ComfyUI", "rev": "1332fabf", "python": "3.12"}
# Programs next to the app: the TTS environment and llama.cpp's server (a release build, no compiler needed)
SOFTWARE = {
    "tts": {"python": "3.12", "pip": ["qwen-tts==0.1.1", "soundfile"]},
    "llama": {"repo": "ggml-org/llama.cpp", "tag": "b12317"},
}


def _get(url: str, tries: int = 5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "video-chain-setup"}),
                                        timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(3 * (i + 1))


_TREES: dict = {}


def repo_info(repo: str) -> tuple[str, dict]:
    if repo not in _TREES:
        sha = _get(f"https://huggingface.co/api/models/{repo}")["sha"]
        files, url = {}, f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=1&limit=1000"
        for f in _get(url):
            if f["type"] == "file":
                files[f["path"]] = {"size": (f.get("lfs") or f)["size"], "sha256": (f.get("lfs") or {}).get("oid")}
        _TREES[repo] = (sha, files)
        time.sleep(0.5)
    return _TREES[repo]


def resolve(entry: dict) -> list[dict]:
    if "url" in entry:
        req = urllib.request.Request(entry["url"], method="HEAD", headers={"User-Agent": "video-chain-setup"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return [{**entry, "size": int(r.headers["Content-Length"])}]
    if "hf_repo" in entry:
        sha, files = repo_info(entry["hf_repo"])
        skip = (".md", ".gitattributes", ".png", ".jpg", ".mp4", ".wav")
        out = []
        for path, info in sorted(files.items()):
            if path.endswith(skip):
                continue
            if entry["patterns"] and not any(p in path for p in entry["patterns"]):
                continue
            out.append({"repo": entry["hf_repo"], "rev": sha, "path": path, "hf": True, **info,
                        **({"alt": entry["alt"]} if entry.get("alt") else {})})
        if not out:
            raise SystemExit(f"{entry['hf_repo']}: nothing matches {entry['patterns']}")
        return out
    sha, files = repo_info(entry["repo"])
    if entry["path"] not in files:
        raise SystemExit(f"{entry['repo']} has no {entry['path']}")
    return [{**entry, "rev": sha, **files[entry["path"]]}]


def models_doc(comps: dict) -> str:
    """docs/MODELS.md: every download with its source, so the licences can be read before installing."""
    gb = lambda n: f"{n / 2**30:.1f} GB" if n >= 2**28 else f"{n / 2**20:.0f} MB"   # noqa: E731
    out = ["# What setup downloads", "",
           "Written by `tools/build_manifest.py` from the installer's list (`src/video_chain/setup/manifest.json`). "
           "Every file is fetched from the address below at a fixed revision and checked by size and SHA-256. "
           "Each model comes under its own licence, shown on its page; you accept it by downloading the model.", "",
           "| Profile | Parts | Download |", "|---|---|---|"]
    order = ("core", "standard", "full")
    for i, prof in enumerate(order):
        ids = [c for c, v in comps.items() if order.index(v["profile"]) <= i]
        out.append(f"| `{prof}` | {', '.join(ids)} | {gb(sum(comps[c]['bytes'] for c in ids))} |")
    for cid, c in comps.items():
        out += ["", f"## {cid}: {c['title']}", "",
                f"Profile `{c['profile']}`, {gb(c['bytes'])}."
                + (f" Needs: {', '.join(c['needs'])}." if c.get("needs") else "")
                + (f" Custom nodes: {', '.join(NODES[n].get('git', 'bundled with Promptwood') for n in c['nodes'])}."
                   if c.get("nodes") else "")
                + (f" Licence with its own terms: {c['licence']}." if c.get("licence") else ""), "",
                "| File | Size | Source |", "|---|---|---|"]
        repos: dict = {}
        for f in c["files"]:
            if f.get("hf"):
                r = repos.setdefault(f["repo"], [0, 0])
                r[0] += 1
                r[1] += f["size"]
            elif "url" in f:
                out.append(f"| {', '.join(f['extract'])} | {gb(f['size'])} | {f['url']} |")
            else:
                out.append(f"| `{f['name']}` | {gb(f['size'])} | https://huggingface.co/{f['repo']} |")
        for repo, (n, size) in repos.items():
            out.append(f"| {n} files (Hugging Face cache) | {gb(size)} | https://huggingface.co/{repo} |")
    out += ["", "## Programs", "", "| Program | Source |", "|---|---|",
            f"| ComfyUI at `{COMFY['rev']}` | {COMFY['git']} |",
            "| PyTorch (CUDA build chosen from the graphics driver) | https://download.pytorch.org/whl/ |",
            f"| llama.cpp `{SOFTWARE['llama']['tag']}` (release build) | https://github.com/{SOFTWARE['llama']['repo']} |",
            f"| Qwen3-TTS (`{SOFTWARE['tts']['pip'][0]}`) | https://pypi.org/project/qwen-tts/ |",
            "| ffmpeg (only when the system has none with libx264) | https://github.com/BtbN/FFmpeg-Builds |",
            "| uv (only when the system has none) | https://docs.astral.sh/uv/ |", ""]
    return "\n".join(out)


def main() -> int:
    comps = {}
    for cid, c in SPEC.items():
        files = [f for e in c["files"] for f in resolve(e)]
        for f in files:
            if not f.get("sha256"):
                f.pop("sha256", None)
        comps[cid] = {**c, "files": files, "bytes": sum(f["size"] for f in files)}
        print(f"{cid:14} {comps[cid]['bytes'] / 2**30:7.1f} GB  {len(files)} files")
    for cid, c in comps.items():
        for n in c.get("needs", []) + c.get("nodes", []):
            assert n in comps or n in NODES, (cid, n)
    OUT.write_text(json.dumps({"comfy": COMFY, "software": SOFTWARE, "nodes": NODES, "components": comps}, indent=1) + "\n")
    print("wrote", OUT)
    DOC.write_text(models_doc(comps))
    print("wrote", DOC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
