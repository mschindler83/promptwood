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
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8188"
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"      # reference-to-video (identity references)
REF_HINT = "The person has exactly the face and look of the person in <Picture 1>."
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SAMPLER = "res_multistep"
PIKA_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"   # ComfyUI's Pika H3 Flex templates
SCHEDULER = "simple"
SUBFOLDER = "video-chain"


class ComfyError(RuntimeError):
    pass


class ComfyUnavailable(ComfyError):
    pass


class ComfyJobFailed(ComfyError):
    """ComfyUI itself ended the job with an error (or it was interrupted): nothing is left to cancel."""


# Every video ComfyUI writes for the app: H.264 re-encoded at this quality. SaveVideo's automatic setting wrote a
# 672x832 clip at 2 Mbit/s (0.15 bit per pixel): visible 8-px blocks (borders 29 % stronger than inside) and
# flicker that SeedVR2 then enlarged faithfully into 2K grain (its flicker followed the draft's at r = 0.993).
VIDEO_CRF = 12.0


def save_video(prefix: str, video=("video", 0)) -> dict:
    """A SaveVideo node writing an mp4 at VIDEO_CRF (never the automatic quality)."""
    return {"class_type": "SaveVideo", "inputs": {"video": list(video), "filename_prefix": prefix, "format": "mp4",
                                                  "format.codec": "h264", "format.codec.encoding": "re-encode",
                                                  "format.codec.encoding.crf": VIDEO_CRF}}


def _combo_options(spec) -> list[str]:
    """Options of a combo input in /object_info, old ([list]) or new (["COMBO", {"options": list}]) format."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return list(spec[0])
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return list(spec[1].get("options", []))
    return []


def lora_name_for(path: str, names: list[str], roots: list[Path] = ()) -> str | None:
    """ComfyUI's name for a LoRA file: the listed name the path ends with (longest match), else the listed name
    that is the same file (followed links) in one of ComfyUI's LoRA folders (roots)."""
    p = Path(path).as_posix()
    try:
        real = os.path.realpath(path)
    except OSError:
        real = None
    hits = [n for n in names if p.endswith("/" + n.replace("\\", "/")) or p == n]
    if hits and roots:
        # with the folders known, a name only counts if it is this very file (a 'style.safetensors' elsewhere
        # is another LoRA than ComfyUI's 'style.safetensors')
        hits = [n for n in hits if any((Path(r) / n).exists() and os.path.realpath(Path(r) / n) == real
                                       for r in roots)]
    if hits:
        return max(hits, key=len)
    if real is None:
        return None
    for root in roots:
        for n in names:
            cand = Path(root) / n
            if cand.exists() and os.path.realpath(cand) == real:
                return n
    return None


def comfy_lora_roots(comfy_dir: Path) -> list[Path]:
    """ComfyUI's LoRA folders: models/loras plus the 'loras' entries of extra_model_paths.yaml."""
    return comfy_model_roots(comfy_dir, "loras")


def comfy_model_roots(comfy_dir: Path, kind: str) -> list[Path]:
    """ComfyUI's folders for one kind of model: models/<kind> plus the <kind> entries of extra_model_paths.yaml."""
    comfy_dir = Path(comfy_dir).expanduser()
    roots = [comfy_dir / "models" / kind]
    cfg = comfy_dir / "extra_model_paths.yaml"
    if cfg.is_file():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            data = {}
        if not isinstance(data, dict):
            data = {}
        for section in data.values():
            if not isinstance(section, dict) or not section.get(kind):
                continue
            base = Path(os.path.expanduser(str(section.get("base_path", ""))))
            for entry in str(section[kind]).split("\n"):
                entry = entry.strip()
                if entry:
                    root = Path(os.path.expanduser(entry))
                    roots.append(root if root.is_absolute() else (comfy_dir / base / root if not base.is_absolute()
                                                                  else base / root))
    return [r for r in roots if r.is_dir()]


_TURBO_CACHE: dict[tuple, bool] = {}


def is_turbo_lora(path: Path) -> bool:
    """A step-distilled H3 turbo LoRA (needs ComfyUI-MiniMax-H3-Turbo's own loader and sampler): 'turbo' in the
    name, or it trains final_layer, which normal H3 LoRAs do not."""
    if "turbo" in Path(path).name.lower():
        return True
    try:
        st = Path(path).stat()
    except OSError:
        return False
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _TURBO_CACHE:
        ok = False
        try:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if 0 < n < 100 * 2**20:
                    ok = any(k.startswith("final_layer") or ".final_layer" in k for k in json.loads(f.read(n)))
        except (OSError, ValueError):
            ok = False
        _TURBO_CACHE[key] = ok
    return _TURBO_CACHE[key]


_H3_ATTN = re.compile(r"blocks[._]\d+[._]attn[._]qkv_proj")
_H3_MLP = re.compile(r"blocks[._]\d+[._]mlp[._]fc1")
_H3_CACHE: dict[tuple, bool] = {}


def is_h3_lora(path: Path) -> bool:
    """Whether a .safetensors file is a MiniMax-H3 LoRA (any format: ai-toolkit/peft, kohya, LoKr), from the
    tensor names in its header only: H3's blocks have attn.qkv_proj and mlp.fc1."""
    try:
        st = Path(path).stat()
    except OSError:
        return False
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _H3_CACHE:
        ok = False
        try:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if 0 < n < 100 * 2**20:
                    names = json.loads(f.read(n)).keys()
                    ok = any(_H3_ATTN.search(k) for k in names) and any(_H3_MLP.search(k) for k in names)
        except (OSError, ValueError):
            ok = False
        _H3_CACHE[key] = ok
    return _H3_CACHE[key]


_QWEN_IMG = re.compile(r"transformer_blocks[._]\d+[._]img_mlp")
_QWEN_TXT = re.compile(r"transformer_blocks[._]\d+[._]txt_mlp")
_QWEN_CACHE: dict[tuple, bool] = {}


def is_qwen_lora(path: Path) -> bool:
    """Whether a .safetensors file is a Qwen-Image / Qwen-Image-Edit LoRA, from the tensor names in its header: the
    MMDiT's blocks have img_mlp and txt_mlp (Flux's have no such names; it also has single_transformer_blocks)."""
    try:
        st = Path(path).stat()
    except OSError:
        return False
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _QWEN_CACHE:
        ok = False
        try:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if 0 < n < 100 * 2**20:
                    names = json.loads(f.read(n)).keys()
                    ok = (any(_QWEN_IMG.search(k) for k in names) and any(_QWEN_TXT.search(k) for k in names)
                          and not any("single_transformer_blocks" in k for k in names))
        except (OSError, ValueError):
            ok = False
        _QWEN_CACHE[key] = ok
    return _QWEN_CACHE[key]


def find_qwen_loras(roots: list[Path]) -> list[tuple[str, str]]:
    """Qwen Image (Edit) LoRAs in these folders, without the Lightning speed LoRAs (Fast mode applies those)."""
    return [(p, n) for p, n in _find_loras(roots, is_qwen_lora) if "lightning" not in n.lower()]


def is_ic_lora(name: str) -> bool:
    """An Open-H3-Regenerate-2K-IC LoRA (or a converted copy): an upscaler part for the Ref2VA model, never a clip's
    LoRA nor the 2K refiner (e.g. open_h3_regenerate_2k_ic_lora, ..._regen_2k_IC_comfy_fp32)."""
    n = Path(name).name.lower()
    return ("regenerate_2k" in n or "regenerate-2k" in n or "regen_2k" in n) and \
        re.search(r"(^|[_\-.])ic([_\-.]|$)", n) is not None


def find_h3_loras(roots: list[Path]) -> list[tuple[str, str]]:
    """All H3 LoRAs in these folders (links followed) as (path, name relative to its folder), sorted by name.
    Turbo LoRAs are included; is_turbo_lora tells them apart. Pika H3 Flex is left out: its speed mode applies it;
    so are Open-H3-Regenerate-2K-IC LoRAs (upscaler parts, is_ic_lora)."""
    return [(p, n) for p, n in _find_loras(roots, is_h3_lora) if "pika_h3_flex" not in n.lower() and not is_ic_lora(n)]


def _find_loras(roots: list[Path], test) -> list[tuple[str, str]]:
    out, seen = [], set()
    for root in roots:
        for dirpath, _dirs, files in os.walk(root, followlinks=True):
            for f in files:
                if not f.endswith(".safetensors"):
                    continue
                full = os.path.join(dirpath, f)
                if full in seen or not test(Path(full)):
                    continue
                seen.add(full)
                out.append((full, os.path.relpath(full, root)))
    return sorted(out, key=lambda x: x[1].lower())


def _arg(argv: list, name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def with_ref_hint(prompt: str) -> str:
    """Add the sentence that ties the person to the reference, unless the prompt already names <Picture ...>."""
    return prompt if "<Picture" in prompt else f"{prompt.rstrip()} {REF_HINT}"


VEDA_NODE = "VedaSparseAttention"            # veda-sparse-attention (Veda-on-ComfyUI): learned sparse attention
VEDA_SETTINGS = {"generated_sparsity": "32", "reference_sparsity": "32", "full_attention_layers": "",
                 "full_attention_steps": "", "verbose": False}   # the node's defaults (as benchmarked)


def pick_veda_predictor(names: list[str], refs: bool) -> str | None:
    """The Veda predictor for a clip: the R2VA one for clips with references (faces, sounds), else the T2VA / FL2VA
    one (start / end frames); either when only one is there."""
    want, other = ("r2va", "t2va") if refs else ("t2va", "r2va")
    return next((n for n in names if want in n.lower()), None) or next((n for n in names if other in n.lower()),
                                                                        names[0] if names else None)


def build_workflow(*, prompt: str, image: str | None, seed: int, width: int, height: int, frames: int, steps: int,
                   loras: list[tuple[str, float]], audio: bool, prefix: str, refs: list[str] = (),
                   ref_size: str = "match", ref_key=lambda i: f"ref_images.ref_image_{i}",
                   turbo: tuple[str, float] | None = None, end_image: str | None = None, cfg: float = 1.0,
                   guides: list[dict] = (), hyperflow: tuple[str, float] | None = None,
                   pika: tuple[str, float] | None = None, audio_refs: list[str] = (),
                   audio_key=lambda i: f"ref_audios.ref_audio_{i}", start_pin: int = 1,
                   recipe: dict | None = None, veda: str | None = None,
                   keyframes: list[tuple[str, int]] = (), frame_refs: list[str] = ()) -> dict:
    """The API-format workflow for one clip (image = the uploaded start image, as LoadImage names it; None: a
    clip without a start frame).

    end_image: the uploaded frame the clip must end at (the first frame of the video after it).
    guides: motion guides, [{"name": uploaded clip, "frame_idx", "audio": bool, "at": "start" | "end"}]; a guide
    at the start replaces the start image, one at the end replaces end_image (the guide clip ends / begins with it).
    keyframes: middle keyframes, [(uploaded picture, frame)]: MiniMaxH3AddGuide pins each at its frame (H3 places
    it on the clip's timeline; one it cannot reach by motion makes it cut there).

    With refs (uploaded reference images): the ref2va model; MiniMaxH3ReferenceToVideo gets the references
    (<Picture 1> ... in the prompt) and MiniMaxH3AddGuide anchors the start image at frame 0, so the clip starts
    where the previous one ended and sees the same original face. start_pin > 1: the start image is anchored as a
    still of that many frames (RepeatImageBatch): one frame alone does not hold the start in the reference model when
    the prompt's story does not fit it, a short still does (as a motion guide does).
    frame_refs ("start" and / or "end", the h3 prompt format's settings "frame_pictures"): the plain start / end
    frame is also a reference picture, after all refs (start first), so the prompt can name it <Picture N>
    (MiniMaxH3AddGuide only pins a frame, the text encoder never sees it); the pinning stays as it is. The frames
    go whole (the node scales a reference picture down to the canvas area, never crops it).

    With audio_refs (uploaded sound files, <Audio 1> ... in the prompt): the ref2va model too (also without
    reference images); MiniMaxH3ReferenceToVideo gets them with the audio VAE, the prompt says what each one is for
    (a voice, a soundtrack to copy, a beat).

    With turbo (ComfyUI name of the turbo LoRA, strength): ComfyUI-MiniMax-H3-Turbo's loader (it handles the
    pruned model's adaln layers) right after the model, the other LoRAs after it, and its own sampler.

    With hyperflow (its weights file in models/hyperflow, strength): ComfyUI-HyperFlow-H3's ApplyHyperFlowH3 right
    after the model (curve refit on: the exact correction for the pruned int8 H3 files), the other LoRAs after it,
    Euler on the 8-step sigma grid it returns (no scheduler).

    With pika (ComfyUI name of the Pika H3 Flex LoRA for this model, strength): that LoRA right after the model,
    the other LoRAs after it, the lcm sampler on ManualSigmas PIKA_SIGMAS (ComfyUI's Pika H3 Flex templates).

    recipe: a LoRA's own sampling (e.g. the head swap LoRA's: {"sampler": "euler", "scheduler": "beta",
    "shift": (12.0, 3.0)}): MiniMaxH3SigmaShift after the LoRAs, that sampler and scheduler (not with turbo,
    hyperflow or pika)."""
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_REF if refs or audio_refs else UNET,
                                                         "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": ({"class_type": "MiniMaxH3TurboSampler", "inputs": {}} if turbo else
                    {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}}),
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "last": {"class_type": "ImageFromBatch", "inputs": {"image": ["decode", 0], "batch_index": int(frames) - 1,
                                                             "length": 1}},
        "save_last": {"class_type": "SaveImage", "inputs": {"images": ["last", 0],
                                                            "filename_prefix": f"{prefix}_last"}},
    }
    guided = {g.get("at") for g in guides}
    use_start = image is not None and "start" not in guided
    use_end = end_image is not None and "end" not in guided
    if use_start:
        wf["start"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    if use_end:
        wf["end"] = {"class_type": "LoadImage", "inputs": {"image": end_image}}
    if audio or audio_refs:
        wf["audio_vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}
    if refs or audio_refs:
        inputs = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
                  "height": int(height), "length": int(frames), "ref_image_size": ref_size}
        for i, ref in enumerate(refs):
            wf[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
            inputs[ref_key(i)] = [f"ref{i}", 0]
        for i, f in enumerate([f for f in ("start", "end") if f in frame_refs], start=len(refs)):
            if not (use_start if f == "start" else use_end):
                raise ValueError(f"the {f} frame is to be a reference picture, but the clip has no plain {f} frame")
            inputs[ref_key(i)] = [f, 0]                # the same LoadImage the guide pins
        for i, sound in enumerate(audio_refs):
            wf[f"aref{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": sound}}
            inputs[audio_key(i)] = [f"aref{i}", 0]
        if audio_refs:
            inputs["audio_vae"] = ["audio_vae", 0]
        wf["refcond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}
        latent, positive = ["refcond", 1], ["refcond", 0]
        if use_start:
            pin = max(1, min(int(start_pin), int(frames) - 1))
            if pin > 1:
                wf["start_still"] = {"class_type": "RepeatImageBatch", "inputs": {"image": ["start", 0], "amount": pin}}
            wf["cond"] = {"class_type": "MiniMaxH3AddGuide",
                          "inputs": {"positive": positive, "vae": ["vae", 0], "latent": latent,
                                     "image": ["start_still" if pin > 1 else "start", 0], "frame_idx": 0}}
            positive = ["cond", 0]
        if use_end:
            wf["end_guide"] = {"class_type": "MiniMaxH3AddGuide",
                               "inputs": {"positive": positive, "vae": ["vae", 0], "latent": latent,
                                          "image": ["end", 0], "frame_idx": -1}}
            positive = ["end_guide", 0]
    else:
        if frame_refs:
            raise ValueError("frames are reference pictures only with the reference model (pictures or sounds)")
        inputs = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
                  "height": int(height), "length": int(frames)}
        if use_start:
            inputs["first_frame"] = ["start", 0]
        if use_end:
            inputs["last_frame"] = ["end", 0]
        wf["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}
        latent, positive = ["cond", 1], ["cond", 0]
    for k, g in enumerate(guides):                 # motion guides: frames (+ sound) of the neighbouring clips
        wf[f"gvideo{k}"] = {"class_type": "LoadVideo", "inputs": {"file": g["name"]}}
        wf[f"gparts{k}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"gvideo{k}", 0]}}
        gin = {"positive": positive, "vae": ["vae", 0], "latent": latent, "image": [f"gparts{k}", 0],
               "frame_idx": int(g["frame_idx"])}
        if audio and g.get("audio"):
            gin.update({"audio": [f"gparts{k}", 1], "audio_vae": ["audio_vae", 0]})
        wf[f"guide{k}"] = {"class_type": "MiniMaxH3AddGuide", "inputs": gin}
        positive = [f"guide{k}", 0]
    for k, (name, idx) in enumerate(keyframes):    # middle keyframes
        wf[f"key{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf[f"key_guide{k}"] = {"class_type": "MiniMaxH3AddGuide",
                               "inputs": {"positive": positive, "vae": ["vae", 0], "latent": latent,
                                          "image": [f"key{k}", 0], "frame_idx": int(idx)}}
        positive = [f"key_guide{k}", 0]
    model = ["unet", 0]
    if pika and (turbo or hyperflow):
        raise ValueError("Pika H3 Flex cannot be combined with Turbo or HyperFlow")
    if pika:
        wf["pika"] = {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": model, "lora_name": pika[0], "strength_model": float(pika[1])}}
        model = ["pika", 0]
    if hyperflow:
        wf["hyperflow"] = {"class_type": "ApplyHyperFlowH3",
                           "inputs": {"model": model, "hyperflow_file": hyperflow[0], "strength": float(hyperflow[1]),
                                      "lora_mode": "bypass", "variant": "auto", "download_if_missing": False,
                                      "verbose": False, "experimental_curve_refit": True}}
        model = ["hyperflow", 0]
    if turbo:
        wf["turbo"] = {"class_type": "MiniMaxH3TurboLoRA",
                       "inputs": {"model": model, "lora_name": turbo[0], "strength": float(turbo[1]),
                                  "low_vram": False}}
        model = ["turbo", 0]
    for k, (name, strength) in enumerate(loras):
        wf[f"lora{k}"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [f"lora{k}", 0]
    if recipe and (turbo or hyperflow or pika):
        raise ValueError("a LoRA's own sampling recipe cannot be combined with Turbo, HyperFlow or Pika")
    if recipe and recipe.get("shift"):
        wf["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                       "inputs": {"model": model, "shift_video": float(recipe["shift"][0]),
                                  "shift_audio": float(recipe["shift"][1])}}
        model = ["shift", 0]
    if recipe and recipe.get("sampler"):
        wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": recipe["sampler"]}}
    if pika:
        wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
        wf["sigmas"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
        sigmas = ["sigmas", 0]
    elif hyperflow:
        wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        sigmas = ["hyperflow", 1]
    else:
        wf["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": model,
                                                                    "scheduler": (recipe or {}).get("scheduler")
                                                                    or SCHEDULER,
                                                                    "steps": int(steps), "denoise": 1.0}}
        sigmas = ["sigmas", 0]
    if veda:                                       # Veda sparse attention (after every LoRA, before the guider)
        wf["veda"] = {"class_type": VEDA_NODE, "inputs": {"model": model, "predictor": veda, **VEDA_SETTINGS}}
        model = ["veda", 0]
    if abs(cfg - 1.0) > 1e-6:                      # real CFG: a negative without text, the same frames (keyframes stay
        wf["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": positive}}   # in the cond)
        wf["guider"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": positive,
                                                              "negative": ["neg", 0], "cfg": float(cfg)}}
    else:                                          # guidance 1 (the reference workflows): one pass per step
        wf["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": positive}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                               "sigmas": sigmas, "latent_image": latent}}
    video_inputs: dict = {"images": ["decode", 0], "fps": 24.0}
    if audio:
        wf["decode_audio"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0],
                                                                         "vae": ["audio_vae", 0]}}
        video_inputs["audio"] = ["decode_audio", 0]
    wf["video"] = {"class_type": "CreateVideo", "inputs": video_inputs}
    wf["save_video"] = save_video(prefix)
    return wf


class ComfyGenerator:
    """Generator (see chain.Generator) that renders clips in ComfyUI."""

    def __init__(self, url: str = DEFAULT_URL, log=print, progress=None, timeout: float = 3600,
                 comfy_dir: Path | None = None, cleanup: bool = True):
        """comfy_dir: ComfyUI's folder on this machine (for its LoRA folders and, with cleanup, to delete the
        copies it keeps of every clip and start image after they were downloaded)."""
        self.url = url.rstrip("/")
        self.log, self.progress, self.timeout = log, progress, timeout
        self.client_id = uuid.uuid4().hex
        self.current: str | None = None          # prompt id being rendered
        # HyperFlow's patch stays on ComfyUI's cached H3 model and breaks the turbo LoRA's adapters after it
        # (measured: HyperFlow -> Turbo fails, every other order gives the very same clips): the models are
        # unloaded only for that switch. None: not known (another process may have used HyperFlow).
        self.last_hyperflow: bool | None = None
        self._loras: list[str] | None = None
        self.stats = self._json("GET", "/system_stats")          # raises ComfyUnavailable if not running
        self._check_models()
        self.comfy_dir = Path(comfy_dir).expanduser() if comfy_dir else None
        if self.comfy_dir is not None and not (self.comfy_dir / "main.py").is_file():
            self.log(f"{self.comfy_dir} does not look like ComfyUI's folder: no cleanup, LoRAs by name only")
            self.comfy_dir = None
        self.cleanup = bool(cleanup and self.comfy_dir is not None)
        self.roots = comfy_lora_roots(self.comfy_dir) if self.comfy_dir else []
        argv = list((self.stats.get("system") or {}).get("argv") or [])
        base = self.comfy_dir
        if base is not None and _arg(argv, "--base-directory"):
            base = Path(_arg(argv, "--base-directory")).expanduser()
        self.output_dir = (Path(_arg(argv, "--output-directory")).expanduser() if _arg(argv, "--output-directory")
                           else (base / "output" if base else None))
        self.input_dir = (Path(_arg(argv, "--input-directory")).expanduser() if _arg(argv, "--input-directory")
                          else (base / "input" if base else None))

    # ---- HTTP
    def _request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None,
                 timeout: float = 30) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise ComfyError(f"ComfyUI {path}: HTTP {e.code} {body[:600]}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ComfyUnavailable(f"ComfyUI is not reachable at {self.url} ({e})") from e

    def _json(self, method: str, path: str, payload=None, timeout: float = 30):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        return json.loads(self._request(method, path, data, headers, timeout) or b"null")

    def _object_info(self, node: str) -> dict:
        return (self._json("GET", f"/object_info/{node}") or {}).get(node) or {}

    def _check_models(self):
        """The H3 files must be in ComfyUI's model folders (and its H3 nodes present)."""
        if not self._object_info("MiniMaxH3ImageToVideo"):
            raise ComfyError("this ComfyUI has no MiniMax H3 nodes (update ComfyUI)")
        need = (("UNETLoader", "unet_name", UNET), ("CLIPLoader", "clip_name", TEXT_ENCODER),
                ("VAELoader", "vae_name", VIDEO_VAE), ("VAELoader", "vae_name", AUDIO_VAE))
        for node, field, name in need:
            req = self._object_info(node).get("input", {}).get("required", {})
            if name not in _combo_options(req.get(field)):
                raise ComfyError(f"ComfyUI does not list {name} ({node}); is it in its model folders?")

    def ref_key(self):
        """How this ComfyUI names the reference image inputs in the API (from its node description)."""
        opt = self._object_info("MiniMaxH3ReferenceToVideo").get("input", {}).get("optional", {})
        spec = opt.get("ref_images")
        prefix = "ref_image_"
        if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
            prefix = (spec[1].get("template") or {}).get("prefix", prefix)
        return lambda i: f"ref_images.{prefix}{i}"

    def audio_key(self):
        """How this ComfyUI names the reference sound inputs (raises ComfyError if its H3 nodes take none)."""
        opt = self._object_info("MiniMaxH3ReferenceToVideo").get("input", {}).get("optional", {})
        spec = opt.get("ref_audios")
        if spec is None:
            raise ComfyError("this ComfyUI's MiniMax H3 Reference to Video node takes no sound references "
                             "(update ComfyUI)")
        prefix = "ref_audio_"
        if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict):
            prefix = (spec[1].get("template") or {}).get("prefix", prefix)
        return lambda i: f"ref_audios.{prefix}{i}"

    def check_ref_model(self):
        req = self._object_info("UNETLoader").get("input", {}).get("required", {})
        if UNET_REF not in _combo_options(req.get("unet_name")):
            raise ComfyError(f"ComfyUI does not list {UNET_REF} (needed for reference images)")
        if not self._object_info("MiniMaxH3ReferenceToVideo"):
            raise ComfyError("this ComfyUI has no MiniMax H3 Reference to Video node")

    def veda_predictor(self, refs: bool) -> str:
        """The Veda predictor to use (the node must be there: a clip recorded as made with Veda is never made
        without it)."""
        info = self._object_info(VEDA_NODE)
        if not info:
            raise ComfyError("this clip is large enough for Veda sparse attention, but this ComfyUI has no Veda node "
                             "(install veda-sparse-attention: comfy node install veda-sparse-attention, then restart "
                             "ComfyUI), or switch off \u201cFast attention (Veda)\u201d in the project settings")
        names = _combo_options(info.get("input", {}).get("required", {}).get("predictor"))
        name = pick_veda_predictor(names, refs)
        if name is None:
            raise ComfyError("ComfyUI lists no Veda predictor in models/veda "
                             "(https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)")
        return name

    def check_guides(self):
        for node in ("MiniMaxH3AddGuide", "LoadVideo", "GetVideoComponents"):
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node (update ComfyUI)")

    def hyperflow_file(self) -> str:
        """The HyperFlow weights ComfyUI lists for ApplyHyperFlowH3: the pruned build for the pruned H3 files."""
        info = self._object_info("ApplyHyperFlowH3")
        if not info:
            raise ComfyError("this ComfyUI has no ApplyHyperFlowH3 node (install ComfyUI-HyperFlow-H3)")
        names = _combo_options(info.get("input", {}).get("required", {}).get("hyperflow_file"))
        names = [n for n in names if n.endswith(".safetensors")]
        pruned = [n for n in names if "pruned" in n.lower()]
        if not pruned:                           # this app uses the pruned H3 files only
            raise ComfyError("ComfyUI lists no pruned HyperFlow weights in models/hyperflow "
                             "(custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors)")
        return sorted(pruned)[0]

    def pika_lora(self, refs: bool) -> str:
        """ComfyUI's name of the Pika H3 Flex LoRA for the fl2va model (ref2va with face references)."""
        kind = "ref2va" if refs else "fl2va"
        names = [n for n in self.lora_names(refresh=True) if "pika_h3_flex" in n.lower() and kind in n.lower()]
        if not names:
            raise ComfyError(f"ComfyUI lists no Pika H3 Flex LoRA for {kind} (pika_h3_flex_{kind}_v1_bf16"
                             ".safetensors in a LoRA folder)")
        return sorted(names, key=len)[0]

    def check_speed(self, s: dict) -> None:
        """Before a run: the HyperFlow weights / Pika LoRA these settings need are listed by ComfyUI (raises
        ComfyError otherwise, so the run stops up front instead of failing clip by clip)."""
        if s.get("hyperflow"):
            self.hyperflow_file()
        if s.get("pika"):
            self.pika_lora(bool(s.get("refs") or s.get("ref_paths") or s.get("audio_refs") or s.get("audio_paths")))

    def check_turbo(self):
        for node in ("MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler"):
            if not self._object_info(node):
                raise ComfyError(f"this ComfyUI has no {node} node (install ComfyUI-MiniMax-H3-Turbo)")

    def lora_names(self, refresh: bool = False) -> list[str]:
        if self._loras is None or refresh:
            req = self._object_info("LoraLoaderModelOnly").get("input", {}).get("required", {})
            self._loras = _combo_options(req.get("lora_name"))
        return self._loras

    def name_for(self, lora: str) -> str | None:
        return (lora_name_for(lora, self.lora_names(), self.roots)
                or lora_name_for(lora, self.lora_names(refresh=True), self.roots))

    def missing_loras(self, loras: list[str]) -> list[str]:
        """LoRA files ComfyUI cannot load (not in its LoRA folders)."""
        return [lo for lo in loras if self.name_for(lo) is None]

    def free(self) -> None:
        """Ask ComfyUI to unload its models and free the memory."""
        try:
            self._json("POST", "/free", {"unload_models": True, "free_memory": True})
        except ComfyError:
            pass

    def free_cache(self) -> None:
        """Ask ComfyUI to drop its cached results (the last job's frames), keeping the models loaded."""
        try:
            self._json("POST", "/free", {"unload_models": False, "free_memory": True})
        except ComfyError:
            pass

    def _save(self, f: dict, dest: Path) -> None:
        """Download a result into dest, written under a temporary name first (never a truncated file)."""
        dest = Path(dest)
        tmp = dest.with_name(dest.stem + ".part" + dest.suffix)
        try:
            tmp.write_bytes(self._download(f))
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)

    def _drop_uploads(self, uploads: list[tuple[str, Path]]) -> None:
        if self.cleanup:
            for up, local in uploads:
                self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                             Path(local), same_size=False)

    def _queue(self, wf: dict) -> str:
        """Queue a workflow under an id chosen here, set as current before it is sent: a Stop while the request is
        on its way (or after a lost answer) can still cancel the job ComfyUI may have accepted."""
        pid = str(uuid.uuid4())
        self.current = pid
        self.node_times, self._running_node = {}, None
        res = self._json("POST", "/prompt", {"prompt": wf, "client_id": self.client_id, "prompt_id": pid})
        if res.get("node_errors"):
            self.current = None
            raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(res['node_errors'])[:600]}")
        self.current = res.get("prompt_id") or pid
        return self.current

    def interrupt(self) -> None:
        """Stop the clip being rendered (and drop it from the queue)."""
        pid = self.current
        if not pid:
            return
        for path, payload in (("/queue", {"delete": [pid]}), ("/interrupt", {"prompt_id": pid})):
            try:
                self._json("POST", path, payload, timeout=10)
            except ComfyError:
                pass

    # ---- one clip
    def upload(self, image: Path) -> str:
        """Upload a file (an image, or a guide clip) into ComfyUI's input/video-chain; returns its name there."""
        data = Path(image).read_bytes()
        suffix = Path(image).suffix.lower() or ".png"
        name = hashlib.sha1(data).hexdigest()[:20] + suffix
        ctype = {".png": "image/png", ".mp4": "video/mp4", ".wav": "audio/wav"}.get(suffix, "application/octet-stream")
        boundary = uuid.uuid4().hex
        parts = []
        for field, value in (("subfolder", SUBFOLDER), ("type", "input"), ("overwrite", "true")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                      f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        res = json.loads(self._request("POST", "/upload/image", b"".join(parts),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}, 120))
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def generate(self, prompt: str, start_image: Path, out_video: Path, out_last_frame: Path, seed: int,
                 settings: dict | None = None) -> None:
        s = settings or {}
        self._last_seedvr = False                  # (a SeedVR2 job after this one frees the H3 models first)
        loras = []
        stack = ([(s["lora"], s.get("strength", 1.0))] if s.get("lora") else []) + \
            [(e["lora"], e.get("strength", 1.0)) for e in s.get("extra_loras") or []]
        for path, strength in stack:                 # chained in this order
            name = self.name_for(path)
            if name is None:
                raise ComfyError(f"{Path(path).name} is not in ComfyUI's LoRA folders")
            loras.append((name, float(strength)))
        turbo = None
        if s.get("turbo"):
            self.check_turbo()
            name = self.name_for(s["turbo"]["lora"])
            if name is None:
                raise ComfyError(f"{Path(s['turbo']['lora']).name} is not in ComfyUI's LoRA folders")
            turbo = (name, float(s["turbo"].get("strength", 1.0)))
        hyper = (self.hyperflow_file(), float(s["hyperflow"].get("strength", 1.0))) if s.get("hyperflow") else None
        pika = (self.pika_lora(bool(s.get("ref_paths") or s.get("audio_paths"))), float(s["pika"].get("strength", 1.0))) \
            if s.get("pika") else None
        if turbo and self.last_hyperflow:
            self.free()                              # HyperFlow's patch would break the turbo LoRA
        if hyper:
            self.last_hyperflow = True               # patched from now on, even if this clip fails
        uploads: list[tuple[str, Path]] = []        # (ComfyUI name, local file); removed again even on errors
        try:
            try:
                self._run(prompt, start_image, out_video, out_last_frame, seed, s, loras, turbo, uploads, hyper, pika)
            except ComfyJobFailed as e:
                # a HyperFlow-patched model left from before this process: unload once and try again
                if not (turbo and self.last_hyperflow is None and "shapes cannot be multiplied" in str(e)):
                    raise
                self.log("ComfyUI still had a HyperFlow-patched model: unloading it and trying again")
                self.free()
                self._run(prompt, start_image, out_video, out_last_frame, seed, s, loras, turbo, uploads, hyper, pika)
            if not hyper:
                self.last_hyperflow = False          # a clip without HyperFlow ran: the model is unpatched
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)

    def _run(self, prompt, start_image, out_video, out_last_frame, seed, s, loras, turbo, uploads,
             hyper=None, pika=None) -> None:
        image = self.upload(Path(start_image)) if start_image is not None else None
        if image:
            uploads.append((image, start_image))
        guides = []
        if s.get("guides"):
            self.check_guides()
            for g in s["guides"]:
                name = self.upload(Path(g["video"]))
                uploads.append((name, g["video"]))
                guides.append({**g, "name": name})
        keyframes = []
        if s.get("keyframes"):
            self.check_guides()
            for kf in s["keyframes"]:
                name = self.upload(Path(kf["image"]))
                uploads.append((name, kf["image"]))
                keyframes.append((name, int(kf["frame_idx"])))
        end_image = None
        if s.get("end_image") and not any(g.get("at") == "end" for g in guides):
            end_image = self.upload(Path(s["end_image"]))
            uploads.append((end_image, s["end_image"]))
        refs = []
        frame_refs = [f for f in ("start", "end") if f in (s.get("frame_pictures") or [])]
        if s.get("ref_paths") or frame_refs:
            self.check_ref_model()
            for r in s.get("ref_paths") or []:
                refs.append(self.upload(Path(r)))
                uploads.append((refs[-1], r))
            if refs and s.get("ref_hint", True) and s.get("prompt_format") != "h3":   # (h3: the prompt ties them)
                prompt = with_ref_hint(prompt)
        sounds = []
        if s.get("audio_paths"):
            if not refs and not frame_refs:
                self.check_ref_model()
            for a in s["audio_paths"]:
                sounds.append(self.upload(Path(a)))
                uploads.append((sounds[-1], a))
        prefix = f"{SUBFOLDER}/{uuid.uuid4().hex[:12]}"
        wf = build_workflow(prompt=prompt, image=image, seed=seed, width=s.get("width", 512),
                            height=s.get("height", 512), frames=s.get("frames", 73), steps=s.get("steps", 20),
                            loras=loras, audio=bool(s.get("audio", True)), prefix=prefix, refs=refs,
                            ref_size=s.get("ref_size", "match"),
                            ref_key=self.ref_key() if refs or frame_refs else None, frame_refs=frame_refs,
                            turbo=turbo, end_image=end_image, guides=guides, hyperflow=hyper, pika=pika,
                            cfg=float(s.get("cfg", 1.0)),
                            audio_refs=sounds, **({"audio_key": self.audio_key()} if sounds else {}),
                            start_pin=int(s.get("start_pin") or 1), recipe=s.get("recipe"), keyframes=keyframes,
                            veda=self.veda_predictor(bool(refs or sounds)) if s.get("veda") else None)
        ws = self._connect_ws()
        try:
            self._queue(wf)
            entry = self._wait(self.current, ws)
        except ComfyJobFailed:
            raise
        except Exception:
            # timed out, lost the connection, a bad answer: the job may still run in ComfyUI and would hold up
            # the next clip, so cancel it (Stop cancels through the signal handler)
            self.interrupt()
            raise
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass
            self.current = None
        outputs = entry.get("outputs") or {}
        video = self._first_file(outputs.get("save_video"))
        frame = self._first_file(outputs.get("save_last"))
        if video is None or frame is None:
            raise ComfyError("ComfyUI finished without the video or the last frame")
        self._save(video, Path(out_video))
        self._save(frame, Path(out_last_frame))
        if self.cleanup:
            self._remove(self.output_dir, video, Path(out_video))
            self._remove(self.output_dir, frame, Path(out_last_frame))

    def _remove(self, top: Path | None, f: dict, local: Path, same_size: bool = True) -> None:
        """Delete ComfyUI's copy of a file, only inside its video-chain subfolder and (for outputs) only if the
        downloaded file has the same size."""
        if top is None:
            return
        try:
            area = (Path(top) / SUBFOLDER).resolve()
            target = (Path(top) / (f.get("subfolder") or "") / f["filename"]).resolve()
            if area not in target.parents or not target.is_file():
                return
            if same_size and target.stat().st_size != local.stat().st_size:
                return
            target.unlink()
        except OSError:
            pass

    @staticmethod
    def _first_file(out: dict | None) -> dict | None:
        if not out:
            return None
        files = out.get("images") or out.get("videos") or out.get("gifs") or []
        return files[0] if files else None

    def _download(self, f: dict) -> bytes:
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        return self._request("GET", f"/view?{q}", timeout=300)

    def _connect_ws(self):
        try:
            import websocket  # websocket-client
        except ImportError:
            return None
        u = urllib.parse.urlparse(self.url)
        scheme = "wss" if u.scheme == "https" else "ws"
        try:
            ws = websocket.create_connection(f"{scheme}://{u.netloc}/ws?clientId={self.client_id}", timeout=5)
            ws.settimeout(2.0)
            return ws
        except Exception:  # noqa: BLE001  (no websocket: fall back to polling)
            return None

    def _history(self, pid: str) -> dict | None:
        return (self._json("GET", f"/history/{pid}") or {}).get(pid)

    def handle_message(self, msg: dict, pid: str) -> str | None:
        """One websocket message: reports progress; returns 'done' or raises on an error for this prompt."""
        t, d = msg.get("type"), msg.get("data") or {}
        if d.get("prompt_id") not in (None, pid):
            return None
        if t == "executing" and d.get("prompt_id") == pid:
            self._node_clock(d.get("node"))
        if t == "progress" and self.progress is not None and d.get("max"):
            self.progress(int(d["value"]), int(d["max"]))
        elif t == "executing" and d.get("node") is None and d.get("prompt_id") == pid:
            return "done"
        elif t == "execution_success" and d.get("prompt_id") == pid:
            self._node_clock(None)
            return "done"
        elif t == "execution_error" and d.get("prompt_id") == pid:
            raise ComfyJobFailed(f"ComfyUI: {d.get('exception_type', 'error')}: {d.get('exception_message', '')}"
                             f" (node {d.get('node_type', '?')})")
        elif t == "execution_interrupted" and d.get("prompt_id") == pid:
            raise ComfyJobFailed("ComfyUI: the job was interrupted")
        return None

    def _node_clock(self, node) -> None:
        """Seconds per workflow node (node_times), from ComfyUI's "executing" messages: a node runs until the next
        one starts (None: the job ended). Cached nodes send none and are not counted."""
        now = time.monotonic()
        last = getattr(self, "_running_node", None)
        if not hasattr(self, "node_times"):
            self.node_times = {}
        if last is not None:
            self.node_times[last[0]] = self.node_times.get(last[0], 0.0) + now - last[1]
        self._running_node = (node, now) if node is not None else None

    def _wait(self, pid: str, ws) -> dict:
        end = time.monotonic() + self.timeout
        last_poll = 0.0
        poll_errors = 0
        while time.monotonic() < end:
            if ws is not None:
                try:
                    raw = ws.recv()
                    if isinstance(raw, str) and self.handle_message(json.loads(raw), pid) == "done":
                        ws = None                       # finished: fetch the result below
                except ValueError:
                    pass
                except ComfyError:
                    raise
                except Exception as e:  # noqa: BLE001  (a timeout: go on; a dropped socket: poll instead)
                    if not (isinstance(e, TimeoutError) or type(e).__name__ == "WebSocketTimeoutException"):
                        ws = None                       # (recv would fail at once forever: no busy loop)
            else:
                time.sleep(0.5)
            if ws is None or time.monotonic() - last_poll > 5:
                last_poll = time.monotonic()
                try:
                    entry = self._history(pid)
                    poll_errors = 0
                except ComfyUnavailable:
                    poll_errors += 1
                    if poll_errors >= 5:                 # ~5 tries in a row: ComfyUI is really gone
                        raise
                    continue
                if entry is not None:
                    st = entry.get("status") or {}
                    if st.get("status_str") == "error":
                        msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                        detail = msgs[-1][1] if msgs else {}
                        raise ComfyJobFailed(f"ComfyUI: {detail.get('exception_type', 'error')}: "
                                         f"{detail.get('exception_message', '')}")
                    if st.get("completed", True):
                        return entry
        raise ComfyError(f"ComfyUI did not finish within {self.timeout / 60:.0f} min")


# ----------------------------------------------------------------------------- image edit (Qwen Image Edit)

QWEN_NODES = ("TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod", "FluxKontextImageScale",
              "ModelSamplingAuraFlow", "CFGNorm", "KSampler", "VAEEncode")
QWEN_SHIFT = 3.1                   # ComfyUI's blueprint "Image Edit (Qwen 2511)"
QUALITY = {"fast": None, "standard": (20, 4.0), "best": (40, 4.0)}   # fast: the Lightning LoRA's steps, cfg 1


def _steps_in(name: str) -> int | None:
    m = re.search(r"(\d+)\s*-?\s*steps?", name.lower())
    return int(m.group(1)) if m else None


def pick_qwen_files(unets: list[str], clips: list[str], vaes: list[str], loras: list[str]) -> dict:
    """ComfyUI's Qwen Image Edit files: the edit model (2511 preferred), its text encoder and VAE, and the Lightning
    LoRA of the same model version (fewest steps) if there is one. Missing files are None."""
    def low(n):
        return n.lower().replace("-", "_")
    edit = [u for u in unets if "qwen_image_edit" in low(u)]
    unet = next((u for u in edit if "2511" in u), None) or (sorted(edit)[-1] if edit else None)
    clip = next((c for c in clips if "qwen_2.5_vl_7b" in low(c)), None)
    vae = next((v for v in vaes if "qwen_image_vae" in low(v)), None)
    version = next((v for v in ("2511", "2509") if unet and v in unet), "")
    light = [lo for lo in loras if "lightning" in lo.lower() and "edit" in lo.lower() and _steps_in(lo)
             and (not version or version in lo)]
    light.sort(key=lambda lo: (_steps_in(lo), "fp32" in lo.lower()))
    return {"unet": unet, "clip": clip, "vae": vae,
            "lightning": (light[0], _steps_in(light[0])) if light else None}


def build_edit_workflow(*, prompt: str, image: str, refs: list[str] = (), seed: int, steps: int, cfg: float,
                        files: dict, lightning: str | None, prefix: str,
                        loras: list[tuple[str, float]] = ()) -> dict:
    """One Qwen Image Edit: the source image (scaled to the model's size) edited by the prompt; refs: up to two more
    pictures (Picture 2, Picture 3 in the prompt), e.g. a face. As ComfyUI's blueprint 'Image Edit (Qwen 2511)'.
    loras: (ComfyUI name, strength) chained after the model, in this order, before the Lightning LoRA."""
    wf: dict = {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "qwen_image",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
        "src": {"class_type": "LoadImage", "inputs": {"image": image}},
        "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["src", 0]}},
    }
    model = ["unet", 0]
    for k, (name, strength) in enumerate(loras):
        wf[f"lora{k}"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [f"lora{k}", 0]
    if lightning:
        wf["lightning"] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"model": model, "lora_name": lightning, "strength_model": 1.0}}
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
    wf["edit_save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
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
    def low(n):
        return n.lower().replace("-", "_")
    q = [u for u in unets if is_qwen21_name(u)]
    base = sorted([u for u in q if "turbo" not in low(u)], key=lambda u: (_quantized(u), u))
    turbo = sorted([u for u in q if "turbo" in low(u)], key=lambda u: (_quantized(u), u))
    clip = sorted([c for c in clips if "qwen3vl_8b" in low(c)], key=lambda c: (_quantized(c), c))
    vae = sorted([v for v in vaes if "qwen_image_2.1_vae" in low(v) or "qwen_image_21_vae" in low(v)])
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


def inpaint_feather(width: int, height: int) -> int:
    """The soft edge for a picture: about 1/170 of its short side, 4 to 20 px (ImageBlur's sigma stays <= 10)."""
    return max(4, min(20, round(min(width, height) / 170)))


def pick_qwen21_inpaint(patches: list[str]) -> str | None:
    """The Qwen-Image 2.1 Fun ControlNet among ComfyUI's model patches (not the older Qwen-Image's)."""
    hits = [p for p in patches if is_qwen21_name(p) and "controlnet" in p.lower().replace("_", "").replace("-", "")]
    return sorted(hits, key=lambda p: (_quantized(p), p))[0] if hits else None


def build_qwen21_workflow(*, files: dict, prompt: str, seed: int, quality: str, prefix: str, save_as: str,
                          images: list[str] = (), size: tuple[int, int] | None = None, negative: str = "",
                          loras: list[tuple[str, float]] = (), model: str | None = None,
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
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "qwen_image",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
    }
    m = ["unet", 0]
    for k, (name, strength) in enumerate(loras):
        wf[f"lora{k}"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": m, "lora_name": name, "strength_model": float(strength)}}
        m = [f"lora{k}", 0]
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
                                                                   "color": 0x808080}}
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
        feather = max(0, min(20, int(inpaint.get("feather", INPAINT_FEATHER))))
        wf["igrow"] = {"class_type": "GrowMask", "inputs": {"mask": ["imask", 0], "expand": feather,
                                                            "tapered_corners": True}}
        mask = ["igrow", 0]
        if feather:
            wf["isoft_pic"] = {"class_type": "MaskToImage", "inputs": {"mask": mask}}
            wf["isoft_blur"] = {"class_type": "ImageBlur", "inputs": {"image": ["isoft_pic", 0],
                                                                      "blur_radius": feather,
                                                                      "sigma": min(10.0, max(1.0, feather / 2))}}
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
    def low(n):
        return n.lower().replace("-", "_")
    m = [u for u in unets if "moss_image" in low(u)]
    sft = sorted([u for u in m if "base" not in low(u)], key=lambda u: (_quantized(u), u))
    base = sorted([u for u in m if "base" in low(u)], key=lambda u: (_quantized(u), u))
    clip = sorted([c for c in clips if "qwen3.5_9b" in low(c) and "qwen_image" not in low(c)],
                  key=lambda c: (_quantized(c), c))
    vae = sorted([v for v in vaes if "flux2_vae" in low(v)], key=lambda v: ("small" not in v, v))
    edit = sorted([lo for lo in loras if "moss_image_edit_lora" in low(lo)],     # the official edit LoRA
                  key=lambda lo: ("rank_64" not in low(lo), "bf16" not in low(lo), lo))
    if not (sft or base) or not clip or not vae:
        return None
    return {"sft": sft[0] if sft else None, "base": base[0] if base else None, "models": sft + base,
            "clip": clip[0], "vae": vae[0], "edit_lora": edit[0] if edit else None}


def build_moss_workflow(*, files: dict, prompt: str, seed: int, prefix: str, save_as: str, images: list[str] = (),
                        size: tuple[int, int] | None = None, negative: str = "", loras: list[tuple[str, float]] = (),
                        model: str | None = None) -> dict:
    """One MOSS-Image picture. Without images: text to image at size. With images (uploaded; an edit's source
    first): the edit LoRA and TextEncodeMOSSImageEdit; size None: the size of the first picture scaled to ~1 MP."""
    unet = model if model in (files.get("models") or []) else (files.get("sft") or files["base"])
    if images:
        unet = files.get("sft") or unet            # the edit LoRA was trained on the SFT model
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["clip"], "type": "moss_image",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": files["vae"]}},
    }
    m = ["unet", 0]
    for k, (name, strength) in enumerate(loras):
        wf[f"lora{k}"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": m, "lora_name": name, "strength_model": float(strength)}}
        m = [f"lora{k}", 0]
    neg = negative or MOSS_NEGATIVE
    if images:
        if not files.get("edit_lora"):
            raise ComfyError("MOSS-Image's edit LoRA (moss_image_edit_lora_rank_64_bf16.safetensors) is missing")
        wf["edit_lora"] = {"class_type": "LoraLoaderModelOnly",
                           "inputs": {"model": m, "lora_name": files["edit_lora"], "strength_model": 1.0}}
        wf["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple",
                     "inputs": {"model": ["edit_lora", 0], "double_layers": "", "single_layers": "22",
                                "start_percent": 0.0, "end_percent": 1.0}}
        pics = {}
        for k, name in enumerate(images[:10]):
            wf[f"pic{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wf[f"scale{k}"] = {"class_type": "ImageScaleToTotalPixels",
                               "inputs": {"image": [f"pic{k}", 0], "upscale_method": "lanczos", "megapixels": 1.0,
                                          "resolution_steps": 16}}
            pics[f"images.image_{k}"] = [f"scale{k}", 0]
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
            return _combo_options(self._object_info(node).get("input", {}).get("required", {}).get(field))
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
             refs: list[Path] = (), loras: list[tuple[str, float]] = (), model: str = "2511",
             mask: Path | None = None) -> dict:
        """Edit image by prompt; the result goes to out_png. loras: (file, strength) in ComfyUI's LoRA folders.
        model: a key of EDIT_MODELS. mask: a picture at the image's size, white where it is drawn anew (inpaint;
        Qwen-Image 2.1 only). Returns what it was made with."""
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
        steps, cfg = (light[1], 1.0) if light else QUALITY.get(quality) or QUALITY["standard"]
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
            ws = self._connect_ws()
            try:
                self._queue(wf)
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
            out = self._first_file((entry.get("outputs") or {}).get("edit_save"))
            if out is None:
                raise ComfyError("ComfyUI finished without the edited image")
            self._save(out, Path(out_png))
            if self.cleanup:
                self._remove(self.output_dir, out, Path(out_png))
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)
        return {"quality": quality, "steps": steps, "cfg": cfg, "model": self.files["unet"], "edit_model": "2511",
                "lightning": light[0] if light else None,
                "loras": [{"path": str(p), "strength": float(st)} for p, st in loras]}

    def _run_saving(self, wf: dict, node: str, out_png: Path, uploads: list[tuple[str, Path]]) -> None:
        """Queue a workflow, wait, download what its SaveImage node `node` wrote to out_png; clean up after."""
        try:
            ws = self._connect_ws()
            try:
                self._queue(wf)
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
            out = self._first_file((entry.get("outputs") or {}).get(node))
            if out is None:
                raise ComfyError("ComfyUI finished without the edited image")
            self._save(out, Path(out_png))
            if self.cleanup:
                self._remove(self.output_dir, out, Path(out_png))
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)

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
                                     prefix=f"{SUBFOLDER}/edit_{uuid.uuid4().hex[:12]}", save_as="edit_save",
                                     images=pics, loras=names)
        except BaseException:
            self._drop_uploads(uploads)              # (_run_saving cleans up only once it runs)
            raise
        self._run_saving(wf, "edit_save", out_png, uploads)
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
                from . import chain
                mw, mh = chain.image_size(Path(mask))
                iw, ih = chain.image_size(Path(image))
                if (mw, mh) not in ((iw, ih), (ih, iw)):           # (a turned photo: its upright size)
                    raise ComfyError(f"the mask is {mw}x{mh}, the picture {iw}x{ih}: paint the mask again")
                uploads.append((self.upload(Path(mask)), Path(mask)))
                try:
                    feather = inpaint_feather(mw, mh)
                except (RuntimeError, ValueError, OSError):
                    pass
                inpaint = {"patch": self.inpaint_patch, "mask": uploads[-1][0], "strength": 1.0,
                           "feather": feather, "reference": "hole"}
            wf = build_qwen21_workflow(files=self.files21, prompt=prompt, seed=seed, quality=quality,
                                       prefix=f"{SUBFOLDER}/edit_{uuid.uuid4().hex[:12]}", save_as="edit_save",
                                       images=pics, loras=names, inpaint=inpaint)
        except BaseException:
            self._drop_uploads(uploads)
            raise
        self._run_saving(wf, "edit_save", out_png, uploads)
        turbo = wf["sample"]["class_type"] == "SamplerCustomAdvanced"
        meta = {"quality": quality, "steps": 8 if turbo else QWEN21_STEPS, "cfg": 1.0, "edit_model": "qwen21",
                "model": wf["unet"]["inputs"]["unet_name"], "lightning": None,
                "loras": [{"path": str(p), "strength": float(st)} for p, st in loras]}
        if mask is not None:
            meta["inpaint"] = {"patch": self.inpaint_patch, "feather": feather}
        return meta


# ----------------------------------------------------------------------------- 2K refine (Open-H3-Regenerate-2K)

REFINE_RELEASE = (0.909, 0.857, 0.667, 0.0)  # the release's 3 Euler steps, re-noised from 0.909, no CFG
# How much noise the refine adds before its 3 steps: the release's schedule scaled down to start at this sigma.
# Measured on a 1.6 s face clip at 1080p (RTX 4090): from 0.909 (the release) the refine repaints faces (closed eyes
# drawn half open with lashes, eye and hair colour changed, frizz and grain); 0.8 keeps the person, crisp; 0.7 calm
# and clean; 0.6 softer, little more than a scale-up. Grain on flat areas: 1.42 at 0.909, 1.01-1.07 at 0.6-0.8.
REFINE_STRENGTHS = {"soft": 0.6, "natural": 0.7, "crisp": 0.8}
REFINE_DEFAULT = "natural"


def refine_sigmas(strength: str = REFINE_DEFAULT) -> str:
    """The refine's sigmas for a strength (soft / natural / crisp)."""
    start = REFINE_STRENGTHS[strength]
    return ", ".join(f"{s * start / REFINE_RELEASE[0]:.4g}" for s in REFINE_RELEASE)
SOL_TAU = 1.3                               # Sol-Attn threshold (ComfyUI's default); measured at 2K on an RTX 4090:
                                            # 2.2x faster refine, PSNR 33 dB against dense attention
# The refine pass at 2K runs out of memory on a 24 GB GPU for long clips (measured on an RTX 4090: 10 s = 880k tokens
# fails in the model's forward pass). Longer clips are refined in MiniMax H3 context windows (ComfyUI core): windows
# of at most this many tokens, overlapping by CONTEXT_OVERLAP frames and blended.
WINDOW_TOKENS = 530_000                    # a 124-frame window at 2528x1440 (37 latent frames = 526k)
CONTEXT_OVERLAP = 39


def latent_frames(frames: int) -> int:
    """H3's latent length for a frame count (ComfyUI's video_latent_t: 5 latents per 17 frames, plus 2; frame counts
    off the 17n+5 grid are rounded up first, as ComfyUI does)."""
    frames = int(frames)
    while frames > 5 and frames % 17 != 5:
        frames += 1
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def context_window(frames: int, width: int, height: int) -> int | None:
    """The context window (frames, 17n+5) for refining a clip of this size, or None when it fits in one pass."""
    per = (int(width) // 16) * (int(height) // 16)
    if latent_frames(frames) * per <= WINDOW_TOKENS:
        return None
    chunks = (WINDOW_TOKENS // per - 2) // 5           # 17-frame chunks whose latent fits
    length = max(56, chunks * 17 + 5)                   # at least 56 frames (overlap 39 + 17)
    return length if length < frames else None


REFINE_NODES = ("LatentUpscaleModelLoader", "MiniMaxH3LatentUpscaleWithModel", "LTXVConcatAVLatent", "VAEEncodeAudio",
                "ManualSigmas", "VAEDecodeTiled", "LoadVideo", "GetVideoComponents", "MiniMaxH3ContextWindows",
                "MiniMaxH3ImageToVideo", "CreateVideo", "SaveVideo", "VAEEncode")


def pick_refine_files(loras: list[str], upscalers: list[str]) -> dict:
    """The Open-H3-Regenerate-2K refiner LoRA (highest rank) and the H3 latent upscaler (bf16 preferred)."""
    ref = [lo for lo in loras if ("regenerate_2k" in lo.lower() or "regenerate-2k" in lo.lower()) and not is_ic_lora(lo)]
    rank = lambda n: int(m.group(1)) if (m := re.search(r"rank_?(\d+)", n.lower())) else 0   # noqa: E731
    ups = [u for u in upscalers if "minimax_h3" in u.lower() or "h3_latent" in u.lower()]
    ups.sort(key=lambda u: ("bf16" not in u.lower(), u))
    return {"refiner": max(ref, key=rank) if ref else None, "upscaler": ups[0] if ups else None}


def build_refine_workflow(*, prompt: str, draft: str, seed: int, width: int, height: int, frames: int,
                          loras: list[tuple[str, float]], refiner: str, upscaler: str, prefix: str,
                          first_frame: str | None = None, last_frame: str | None = None,
                          sparse: float | None = None, strength: str = REFINE_DEFAULT) -> dict:
    """Re-render a finished clip (the uploaded draft video, with sound) at width x height: its video and sound are
    encoded into H3's AV latent, the video upscaled with the latent upscaler, then refined in 3 steps with the
    refiner LoRA (and the clip's own LoRAs, as it was made); the clip's own sound is kept.
    first_frame / last_frame (uploaded images): pinned as keyframes, so a refined clip starts exactly where the
    refined clip before it hands over (and ends at a real video's refined first frame).
    sparse: Sol-Attn block-sparse attention at this threshold (ComfyUI's BlockSparseAttention; not exact).
    strength: how much is drawn anew (REFINE_STRENGTHS)."""
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax",
                                                         "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "draft": {"class_type": "LoadVideo", "inputs": {"file": draft}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["draft", 0]}},
        "venc": {"class_type": "VAEEncode", "inputs": {"pixels": ["parts", 0], "vae": ["vae", 0]}},
        "aenc": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["parts", 1], "vae": ["audio_vae", 0]}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["venc", 0], "audio_latent": ["aenc", 0]}},
        "upmodel": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": upscaler}},
        "up": {"class_type": "MiniMaxH3LatentUpscaleWithModel",
               "inputs": {"model": ["upmodel", 0], "samples": ["av", 0], "width": int(width), "height": int(height)}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo",
                 "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
                            "height": int(height), "length": int(frames),
                            **({"first_frame": ["first", 0]} if first_frame else {}),
                            **({"last_frame": ["last", 0]} if last_frame else {})}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": refine_sigmas(strength)}},
    }
    if first_frame:
        wf["first"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
    if last_frame:
        wf["last"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
    model = ["unet", 0]
    for k, (name, strength) in enumerate(list(loras) + [(refiner, 1.0)]):
        wf[f"lora{k}"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)}}
        model = [f"lora{k}", 0]
    window = context_window(frames, width, height)
    if window:                                  # a long clip: refined in overlapping windows (memory)
        wf["windows"] = {"class_type": "MiniMaxH3ContextWindows",
                         "inputs": {"model": model, "context_length": int(window), "context_overlap": CONTEXT_OVERLAP,
                                    "fuse_method": "pyramid", "freenoise": True, "split_conds_to_windows": False}}
        model = ["windows", 0]
    if sparse:
        wf["sparse"] = {"class_type": "BlockSparseAttention",
                        "inputs": {"model": model, "selection": "sol-attn", "selection.tau": float(sparse),
                                   "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288,
                                   "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["sparse", 0]
    wf["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                               "sigmas": ["sigmas", 0], "latent_image": ["up", 0]}}
    wf["decode"] = {"class_type": "VAEDecodeTiled",
                    "inputs": {"samples": ["sample", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 64,
                               "temporal_size": 64, "temporal_overlap": 8}}
    wf["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 24.0,
                                                          "audio": ["parts", 1]}}
    wf["refine_save"] = save_video(prefix)
    return wf


def refine_size(width: int, height: int, target: str) -> tuple[int, int]:
    """The refined size for a clip of width x height: '1080p' (~1920x1080 pixels) or '2k' (~2560x1440), the
    clip's aspect ratio, multiples of 32."""
    from .chain import canvas_size
    return canvas_size(width, height, {"1080p": 1440, "2k": 1920}[target])


def _with_audio(video: Path, out: Path) -> Path:
    """The clip with a sound track (silence if it has none): the refine encodes its sound into the AV latent."""
    import subprocess
    from .chain import probe
    if probe(video)["audio"]:
        return video
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(out)], check=True, capture_output=True)
    return out


def _refine(self, draft: Path, prompt: str, out: Path, seed: int, size: tuple[int, int], frames: int,
            loras: list[tuple[str, float]], first_frame: Path | None = None, last_frame: Path | None = None,
            sparse: float | None = None, strength: str = REFINE_DEFAULT) -> None:
    """ComfyGenerator.refine: re-render one finished clip at size (see build_refine_workflow)."""
    self._last_seedvr = False
    for node in REFINE_NODES:
        if not self._object_info(node):
            raise ComfyError(f"this ComfyUI has no {node} node (update ComfyUI: the H3 2K nodes are in its "
                             "master since 2026-10-28)")
    if self.last_hyperflow is not False:
        # a HyperFlow-patched H3 model (or one this process does not know about) must not refine: load it fresh
        self.free()
        self.last_hyperflow = False
    if sparse and not self._object_info("BlockSparseAttention"):
        raise ComfyError("this ComfyUI has no BlockSparseAttention node (Sol-Attn; update ComfyUI)")
    req = self._object_info("LatentUpscaleModelLoader").get("input", {}).get("required", {})
    files = pick_refine_files(self.lora_names(refresh=True), _combo_options(req.get("model_name")))
    if not files["refiner"]:
        raise ComfyError("ComfyUI lists no Open-H3-Regenerate-2K refiner LoRA "
                         "(minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors in a LoRA folder)")
    if not files["upscaler"]:
        raise ComfyError("ComfyUI lists no H3 latent upscaler (minimax_h3_latent_upscaler_3d_conv_v1_bf16"
                         ".safetensors in models/latent_upscale_models)")
    names = []
    for path, weight in loras:
        name = self.name_for(path)
        if name is None:
            raise ComfyError(f"{Path(path).name} is not in ComfyUI's LoRA folders")
        names.append((name, float(weight)))
    import tempfile
    uploads: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory(prefix="video-chain-refine-") as tmp:
        src = _with_audio(Path(draft), Path(tmp) / "draft.mp4")
        try:
            name = self.upload(src)
            uploads.append((name, src))
            pins = {}
            for key, img in (("first_frame", first_frame), ("last_frame", last_frame)):
                if img is not None:
                    pins[key] = self.upload(Path(img))
                    uploads.append((pins[key], Path(img)))
            prefix = f"{SUBFOLDER}/refine_{uuid.uuid4().hex[:12]}"
            wf = build_refine_workflow(prompt=prompt, draft=name, seed=seed, width=size[0], height=size[1],
                                       frames=frames, loras=names, refiner=files["refiner"],
                                       upscaler=files["upscaler"], prefix=prefix, sparse=sparse,
                                       strength=strength, **pins)
            ws = self._connect_ws()
            try:
                self._queue(wf)
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
            video = self._first_file((entry.get("outputs") or {}).get("refine_save"))
            if video is None:
                raise ComfyError("ComfyUI finished without the refined clip")
            part = Path(out).with_name(Path(out).stem + ".part.mp4")
            part.write_bytes(self._download(video))
            part.replace(out)
            if self.cleanup:
                self._remove(self.output_dir, video, Path(out))
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)


ComfyGenerator.refine = _refine


# ----------------------------------------------------------------------------- SeedVR2 upscale (video restoration)
# ComfyUI's core SeedVR2 nodes, as its blueprint "Upscale Video (SeedVR2 3B Int8)": the clip scaled to the export
# size (lanczos), padded, encoded with SeedVR2's VAE, restored in one step, decoded and colour-matched to the input.
# Unlike the refine it restores what is there instead of drawing it anew: measured on a 1.6 s face clip at 1080p
# (RTX 4090, ~33 s each) it changed the clip least of all (7B 4.1, 3B 4.7, refine "natural" 7.0), closed eyes
# stayed closed, 7B was the cleanest, 3B crisper but harder. Long clips are split in time (SeedVR2TemporalChunk,
# auto: the largest chunk that fits the free GPU memory) and crossfaded back together.
SEEDVR_NODES = ("SeedVR2Preprocess", "SeedVR2Conditioning", "SeedVR2PostProcessing", "SeedVR2TemporalChunk",
                "SeedVR2TemporalMerge", "VAEEncodeTiled", "VAEDecodeTiled", "ImageScale", "LoadVideo",
                "GetVideoComponents", "CreateVideo", "SaveVideo")
SEEDVR_METHODS = {"seedvr2_7b": "7b", "seedvr2_3b": "3b"}       # export method -> model size
SEEDVR_OVERLAP = 2                         # latent frames (8 frames) crossfaded between the chunks of a long clip
SEEDVR_DOWNLOAD = "https://huggingface.co/Comfy-Org/SeedVR2 (diffusion_models/seedvr2_{}_int8_convrot.safetensors)"


SEEDVR_PRECISION_LOCAL = ("int8", "fp8", "fp16")   # 24 GB: the int8 copy fits with the frames (measured on a 4090)
SEEDVR_PRECISION_FULL = ("fp16", "int8", "fp8")    # 40 GB and more: the full-precision weights, nothing quantized
SEEDVR_FULL_VRAM = 40 * 2**30                      # from this much VRAM: fp16 weights and untiled VAE passes


def pick_seedvr_files(unets: list[str], vaes: list[str], size: str,
                      precision: tuple[str, ...] = SEEDVR_PRECISION_LOCAL) -> dict:
    """SeedVR2's model of this size ('7b' / '3b'; not the "sharp" variant; in the order of precision, then any;
    never mxfp8 / nvfp4, made for newer GPUs) and its VAE."""
    cands = [u for u in unets if f"seedvr2_{size}" in Path(u).name.lower() and "sharp" not in u.lower()
             and not any(k in u.lower() for k in ("mxfp8", "nvfp4"))]
    order = lambda u: next((i for i, k in enumerate(precision) if k in u.lower()), 9)   # noqa: E731
    vae = next((v for v in vaes if "seedvr2" in Path(v).name.lower() and "vae" in Path(v).name.lower()), None)
    return {"unet": min(cands, key=order) if cands else None, "vae": vae}


SCALE_NODE = "VideoChainLanczosScale"      # comfy_nodes/video_chain_nodes/scale.py: ImageScale's lanczos on all cores


def build_seedvr_workflow(*, draft: str, width: int, height: int, unet: str, vae: str, prefix: str,
                          seed: int = 42, tiled: bool = True, fast_scale: bool = False) -> dict:
    """Upscale the uploaded clip (with its sound) to width x height with SeedVR2 (see SEEDVR_NODES).
    tiled: the VAE passes in 512 px tiles (as the blueprint; for 24 GB). Off (40 GB and more): the plain
    VAEEncode / VAEDecode, which run SeedVR2's VAE on whole frames, split in time only as far as the free VRAM
    needs, with its causal cache carried between the slices: exact, no tile seams, the GPU kept busy (ComfyUI
    falls back to tiles by itself on an out-of-memory).
    fast_scale: the lanczos resize by SCALE_NODE (the same frames, bit for bit, on all CPU cores) instead of
    ImageScale (one frame after another on one core)."""
    tiles = {"tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}
    enc = ({"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0], **tiles}} if tiled
           else {"class_type": "VAEEncode", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0]}})
    dec = ({"class_type": "VAEDecodeTiled", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0], **tiles}} if tiled
           else {"class_type": "VAEDecode", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0]}})
    return {
        "load": {"class_type": "LoadVideo", "inputs": {"file": draft}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "resize": ({"class_type": SCALE_NODE, "inputs": {"image": ["parts", 0], "width": int(width),
                                                         "height": int(height)}} if fast_scale else
                   {"class_type": "ImageScale", "inputs": {"image": ["parts", 0], "upscale_method": "lanczos",
                                                           "width": int(width), "height": int(height),
                                                           "crop": "disabled"}}),
        "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["resize", 0]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "enc": enc,
        "chunk": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": ["enc", 0],
                                                                   "temporal_overlap": SEEDVR_OVERLAP,
                                                                   "chunking_mode": "auto"}},
        "cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["unet", 0],
                                                                 "vae_conditioning": ["chunk", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["cond", 0],
                                                        "negative": ["cond", 1], "latent_image": ["chunk", 0],
                                                        "seed": int(seed), "steps": 1, "cfg": 1.0,
                                                        "sampler_name": "euler", "scheduler": "simple",
                                                        "denoise": 1.0}},
        "merge": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["sample", 0],
                                                                   "temporal_overlap": ["chunk", 1]}},
        "dec": dec,
        "post": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["dec", 0],
                                                                   "original_resized_images": ["resize", 0],
                                                                   "color_correction_method": "lab"}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["post", 0], "fps": 24.0,
                                                          "audio": ["parts", 1]}},
        "seedvr_save": save_video(prefix),
    }


SEEDVR_RAM_BUDGET = 12 * 2**30           # frame RAM for one SeedVR2 job when ComfyUI does not say what is free
SEEDVR_RAM_RESERVE = 10 * 2**30          # RAM left for the system, this app and the browser
SEEDVR_RAM_MIN = 4 * 2**30               # never less frame RAM than this per job (very short pieces otherwise)
SEEDVR_FRAME_COPIES = 7                  # copies of the clip ComfyUI holds at once (scaled, padded, decoded, matched,
                                         # the video; float32): a 10 s 2K clip as one job took 65 GB RAM + 36 GB swap
                                         # (about 5-6 copies besides the cached models) and systemd-oomd killed it
SEEDVR_PIECE_OVERLAP = 8                 # frames two neighbouring pieces share, crossfaded
# a light denoise before SeedVR2 (optional): mostly temporal (noise that changes from frame to frame), little
# spatial blur, so SeedVR2 does not sharpen the draft's fine noise into texture. Measured on synthetic noise at the
# drafts' level (about 1.5 grey levels frame to frame): 1.56 -> 0.66; strong noise (8.5) it leaves (7.8): hqdn3d
# takes large changes for motion, so real motion is not smeared either
SEEDVR_DENOISE = "hqdn3d=1:1:4:4"


def seedvr_budget(ram_free: int | None) -> int:
    """The frame RAM one SeedVR2 job may use: what ComfyUI has free less SEEDVR_RAM_RESERVE (SEEDVR_RAM_BUDGET when
    it does not say; never below SEEDVR_RAM_MIN)."""
    if not ram_free:
        return SEEDVR_RAM_BUDGET
    return max(SEEDVR_RAM_MIN, int(ram_free) - SEEDVR_RAM_RESERVE)


def seedvr_piece_frames(width: int, height: int, budget: int | None = None) -> int:
    """How many frames of this size one SeedVR2 job gets within budget bytes (4n+1, at least 17)."""
    budget = SEEDVR_RAM_BUDGET if budget is None else int(budget)
    n = budget // (SEEDVR_FRAME_COPIES * int(width) * int(height) * 3 * 4)
    return max(17, 4 * ((int(n) - 1) // 4) + 1)


class SeedvrProgress:
    """One steady progress for a SeedVR2 upscale: ComfyUI reports each node of each piece on its own (tiled encode,
    the step, tiled decode: bars restarting many times). Here a piece counts 1 and its phases a third each: a phase
    starts when ComfyUI's counter goes back or changes its total. Reported as (done, total) in 1000ths, never back."""
    PHASES = 3

    def __init__(self, pieces: int, out=None):
        self.pieces, self.out = max(1, int(pieces)), out
        self.k, self.phase, self.last, self.best = 0, 0, None, 0

    def piece(self, k: int, note=None):
        self.k, self.phase, self.last = int(k), 0, None
        if note is not None:
            note(f"SeedVR2 piece {k + 1} of {self.pieces}" if self.pieces > 1 else "SeedVR2")
        self._send(0.0)

    def step(self, i: int, n: int):
        if not n:
            return
        if self.last is not None and (n != self.last[1] or i < self.last[0]):
            self.phase = min(self.phase + 1, self.PHASES - 1)
        self.last = (i, n)
        self._send((self.phase + min(1.0, i / n)) / self.PHASES)

    def finish(self):
        self.k, self.phase, self.last = self.pieces - 1, self.PHASES - 1, None
        self._send(1.0)

    def _send(self, sub: float):
        done = round(1000 * (self.k + min(1.0, max(0.0, sub))) / self.pieces)
        self.best = max(self.best, done)                 # never back
        if self.out is not None:
            self.out(self.best, 1000)


def seedvr_pieces(total: int, per: int, overlap: int = SEEDVR_PIECE_OVERLAP) -> list[tuple[int, int]]:
    """(first frame, frame count) of the pieces a clip of `total` frames is upscaled in: at most `per` frames each,
    neighbours sharing `overlap` frames; one piece when it fits."""
    total, per = int(total), int(per)
    if total <= per:
        return [(0, total)]
    out, start = [], 0
    while True:
        if start + per >= total:
            out.append((max(0, total - per), min(per, total)))   # (the last one full length, ending at the end)
            break
        out.append((start, per))
        start += per - overlap
    return out


def _seedvr(self, draft: Path, out: Path, size: tuple[int, int], method: str = "seedvr2_7b",
            seed: int = 42, denoise: bool = False) -> None:
    """ComfyGenerator.seedvr: upscale one finished clip (or video) to size with SeedVR2 (build_seedvr_workflow).
    The first SeedVR2 job unloads ComfyUI's other models (their RAM); a long clip goes in pieces that fit
    SEEDVR_RAM_BUDGET, crossfaded over SEEDVR_PIECE_OVERLAP frames, with the clip's own sound. Progress: one bar
    for the whole clip (SeedvrProgress) and a note per piece. denoise: SEEDVR_DENOISE on the clip first."""
    import tempfile
    from .chain import probe
    if denoise:
        with tempfile.TemporaryDirectory(prefix="video-chain-denoise-") as tmp:
            clean = Path(tmp) / "denoised.mp4"
            _denoise(Path(draft), clean)
            return self.seedvr(clean, out, size, method, seed)
    files = self._seedvr_files(method)
    if not getattr(self, "_last_seedvr", False):  # the job before was another model: its RAM goes to SeedVR2
        self.free()
        self.last_hyperflow = False
    self._last_seedvr = True
    total = probe(Path(draft))["frames"]
    ram_free = self._ram_free()                    # (after the unload: what SeedVR2 can really use)
    pieces = seedvr_pieces(total, seedvr_piece_frames(*size, seedvr_budget(ram_free)))
    outer = self.progress
    track = SeedvrProgress(len(pieces), outer)
    self.progress = track.step                     # (ComfyUI's per-node bars become one bar for the whole clip)
    try:
        if len(pieces) == 1:
            track.piece(0, self.note)
            self._seedvr_job(files, Path(draft), Path(out), size, seed)
            if not getattr(self, "seedvr_keep_cache", False):
                self.free_cache()
            track.finish()
            return
        self._seedvr_pieces(files, Path(draft), Path(out), size, seed, pieces, total, track)
    finally:
        self.progress = outer


def _ram_free(self) -> int:
    """The RAM ComfyUI has free now, in bytes (0: it does not say)."""
    try:
        return int((self._json("GET", "/system_stats").get("system") or {}).get("ram_free") or 0)
    except (ComfyError, ValueError, TypeError):
        return 0


def _seedvr_pieces(self, files, draft, out, size, seed, pieces, total, track) -> None:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="video-chain-seedvr-pieces-") as tmp:
        tmp, made = Path(tmp), []
        for i, (first, count) in enumerate(pieces):
            track.piece(i, self.note)
            part = tmp / f"in{i}.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(draft), "-vf",
                            f"trim=start_frame={first}:end_frame={first + count},setpts=PTS-STARTPTS", "-an",
                            "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p", str(part)],
                           check=True, capture_output=True)
            done = tmp / f"out{i}.mp4"
            self._seedvr_job(files, part, done, size, seed)
            if not getattr(self, "seedvr_keep_cache", False):
                self.free_cache()                  # (ComfyUI keeps the last job's frames cached otherwise)
            made.append(done)
        self.note("joining the upscaled pieces")
        _join_pieces(made, pieces, Path(draft), Path(out), total)
        track.finish()


def _denoise(src: Path, out: Path) -> None:
    """The clip with SEEDVR_DENOISE applied, nearly lossless, its sound copied."""
    import subprocess
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", SEEDVR_DENOISE, "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p", "-c:a", "copy",
                    str(out)], check=True, capture_output=True)


def _join_pieces(made: list[Path], pieces: list[tuple[int, int]], draft: Path, out: Path, total: int) -> None:
    """The upscaled pieces as one clip: each overlap crossfaded (ffmpeg xfade, frame-exact at 24 fps), the draft's
    own sound; checked to have the draft's frame count."""
    import subprocess
    from .chain import probe
    args, filt, last, length = [], [], None, 0
    for i, (f, (first, count)) in enumerate(zip(made, pieces)):
        args += ["-i", str(f)]
        filt.append(f"[{i}:v]settb=1/24,setpts=N,fps=24,format=yuv420p[p{i}]")
        if last is None:
            last, length = f"[p{i}]", count
            continue
        overlap = length - first                   # frames this piece shares with the clip so far
        filt.append(f"{last}[p{i}]xfade=transition=fade:duration={overlap / 24:.6f}:"
                    f"offset={(length - overlap) / 24:.6f}[x{i}]")
        last, length = f"[x{i}]", first + count
    args += ["-i", str(draft)]
    part = out.with_name(out.stem + ".part.mp4")
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args, "-filter_complex", ";".join(filt),
                        "-map", last, "-map", f"{len(made)}:a:0?", "-frames:v", str(int(total)), "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-r", "24", str(part)],
                       check=True, capture_output=True)
        got = probe(part)["frames"]
        if got != int(total):
            raise ComfyError(f"the upscaled pieces joined to {got} frames instead of {total}")
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)


def _seedvr_files(self, method: str) -> dict:
    """The SeedVR2 model and VAE ComfyUI lists for this method (clear errors when something is missing)."""
    for node in SEEDVR_NODES:
        if not self._object_info(node):
            raise ComfyError(f"this ComfyUI has no {node} node (update ComfyUI: SeedVR2 is in its core nodes)")
    model = SEEDVR_METHODS[method]
    unets = _combo_options(self._object_info("UNETLoader").get("input", {}).get("required", {}).get("unet_name"))
    vaes = _combo_options(self._object_info("VAELoader").get("input", {}).get("required", {}).get("vae_name"))
    files = pick_seedvr_files(unets, vaes, model, getattr(self, "seedvr_precision", SEEDVR_PRECISION_LOCAL))
    files["fast_scale"] = bool(self._object_info(SCALE_NODE))   # (video-chain's node, when ComfyUI has it)
    if not files["unet"]:
        raise ComfyError(f"ComfyUI lists no SeedVR2 {model.upper()} model in its diffusion_models: "
                         + SEEDVR_DOWNLOAD.format(model))
    if not files["vae"]:
        raise ComfyError("ComfyUI lists no SeedVR2 VAE (seedvr2_ema_vae_fp16.safetensors in models/vae, "
                         "https://huggingface.co/Comfy-Org/SeedVR2)")
    return files


def seedvr_timing(size, unet, up_bytes, t_up, t_run, down_bytes, t_down, node_times: dict, wf: dict) -> str:
    """One log line: where a SeedVR2 job's time went (upload, ComfyUI by node type, download)."""
    def rate(b, t):
        return f"{b / 1e6:.1f} MB in {t:.1f} s ({b / 1e6 / t:.1f} MB/s)" if t > 0 else f"{b / 1e6:.1f} MB"
    by_type: dict[str, float] = {}
    for node, secs in node_times.items():
        kind = (wf.get(str(node)) or {}).get("class_type", str(node))
        by_type[kind] = by_type.get(kind, 0.0) + secs
    nodes = ", ".join(f"{k} {v:.1f} s" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]) if v >= 0.05)
    return (f"SeedVR2 {size[0]}x{size[1]} ({unet}): upload {rate(up_bytes, t_up)}; ComfyUI {t_run:.1f} s"
            + (f" [{nodes}]" if nodes else "") + f"; download {rate(down_bytes, t_down)}")


def _seedvr_job(self, files: dict, draft: Path, out: Path, size: tuple[int, int], seed: int) -> None:
    """One SeedVR2 job in ComfyUI: draft (with its sound, silence if none) upscaled to size into out."""
    import tempfile
    uploads: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory(prefix="video-chain-seedvr-") as tmp:
        src = _with_audio(Path(draft), Path(tmp) / "draft.mp4")
        clock = time.monotonic()
        try:
            name = self.upload(src)
            uploads.append((name, src))
            t_up = time.monotonic() - clock
            prefix = f"{SUBFOLDER}/seedvr_{uuid.uuid4().hex[:12]}"
            wf = build_seedvr_workflow(draft=name, width=size[0], height=size[1], unet=files["unet"],
                                       vae=files["vae"], prefix=prefix, seed=seed,
                                       tiled=getattr(self, "seedvr_tiled", True), fast_scale=files.get("fast_scale",
                                                                                                         False))
            ws = self._connect_ws()
            clock = time.monotonic()
            try:
                self._queue(wf)
                entry = self._wait(self.current, ws)
                t_run = time.monotonic() - clock
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
            video = self._first_file((entry.get("outputs") or {}).get("seedvr_save"))
            if video is None:
                raise ComfyError("ComfyUI finished without the upscaled clip")
            part = Path(out).with_name(Path(out).stem + ".part.mp4")
            clock = time.monotonic()
            part.write_bytes(self._download(video))
            t_down = time.monotonic() - clock
            part.replace(out)
            self.log(seedvr_timing(size, files["unet"], src.stat().st_size, t_up, t_run, Path(out).stat().st_size,
                                   t_down, getattr(self, "node_times", {}), wf))
            if self.cleanup:
                self._remove(self.output_dir, video, Path(out))
        finally:
            if self.cleanup:
                for up, local in uploads:
                    self._remove(self.input_dir, {"filename": Path(up).name, "subfolder": str(Path(up).parent)},
                                 Path(local), same_size=False)


ComfyGenerator.seedvr = _seedvr
ComfyGenerator._ram_free = _ram_free
ComfyGenerator._seedvr_files = _seedvr_files
ComfyGenerator._seedvr_job = _seedvr_job
ComfyGenerator._seedvr_pieces = _seedvr_pieces
ComfyGenerator.note = lambda self, text: None   # (the worker sets its own)


# ----------------------------------------------------------------------------- prompt refiner (an LLM in ComfyUI)

# Language models ComfyUI can run with its TextGenerate node, loaded as text encoders: (file name contains, loader
# type, label, uncensored). Most preferred first.
# Measured on an RTX 4090 (ComfyUI 0.44): Qwen 2.5-VL 7B cannot generate in ComfyUI (no stop tokens) and the nvfp4
# Qwen3-VL 32B gives garbage on a 4090 (no fp4 hardware), so they are not offered. One file per model.
REFINER_MODELS = (
    ("gemma-3-12b-abliterated", "ltxv", "Gemma 3 12B abliterated (uncensored)", True),
    ("qwen3vl_4b_fp8", "krea2", "Qwen3-VL 4B", False),
    ("qwen-3-vl-4b", "krea2", "Qwen3-VL 4B", False),
    ("qwen3vl_4b", "krea2", "Qwen3-VL 4B", False),
    ("gemma_3_12b_it_fp8_scaled", "ltxv", "Gemma 3 12B", False),
    ("gemma_3_12b_it_fp8", "ltxv", "Gemma 3 12B", False),
)
_SAFETY = ("Every person you describe is an adult. Never write sexual or suggestive content involving minors; if the "
           "input asks for that, return the input unchanged.")
REFINE_INSTRUCTIONS = {
    "video": ("You improve prompts for an image-to-video model (MiniMax H3) that continues a video from a given start "
              "frame and also makes the sound. Rewrite the user's prompt into one clear paragraph in present tense "
              "that describes what happens: the action and motion of each subject, the camera movement and framing, "
              "and the sound (ambient sounds, music, and any spoken words in quotes). Do not describe at length how "
              "things look at the start, the start frame already shows that. Keep every subject, action and detail "
              "the user gave. Do not add new people, animals, objects or events, and do not change what happens; "
              "only make it clearer and more concrete. "
              "At most 120 words. Answer with the prompt only, no preface, no quotes around it, no markdown."),
    # the official MiniMax H3 prompt format (base guide, T2VA / I2VA): the body of
    # integrated_multimodal_description and the two sound fields; the app adds the header, the field label, the
    # <Subject> / <Picture> tags and the speaker ids (h3_prompt), so the refiner writes none of them
    "video_h3": ("You write prompts for MiniMax H3, a model that makes a video together with its sound. Rewrite the "
                 "user's prompt into a description of the video along its timeline, in English and in present "
                 "tense, where every detail is something visible or audible. Begin with the overall style (e.g. "
                 "Live-action, cinematic; 2D-animated; 3D CG; claymation; vintage film) and the initial composition: "
                 "the framing, the subjects, where they are and the place. Then describe what happens in order: how "
                 "the action begins, how it develops, and its result or the reaction to it. Write camera movement as "
                 "a natural sentence with its motion type and, where it matters, its amplitude and speed, e.g. \"The "
                 "camera pushes in with small amplitude at slow speed toward her hands.\" or \"The camera holds a "
                 "static shot.\" Keep every subject, action, detail and spoken line the user gave. Do not add new "
                 "people, animals, objects, events or spoken lines, and do not change what happens; only make it "
                 "clearer, more concrete and more vivid. Spoken lines stay word for word in their language, in "
                 "double quotes. Usually the video is one shot. Only if the user asks for a cut or the action clearly "
                 "needs another view, begin with [Shot 1] and start each later shot on a new line as \"[Shot 2] At "
                 "00:03.500, the camera cuts to ...\" with strictly increasing cut times inside the clip's length; "
                 "a cut shows something new, a slight change of distance or angle is a camera movement instead. "
                 "Write in detail: up to about 350-500 words for a long clip, fewer for a short one, and never more "
                 "action than fits into the clip's length. After the description write exactly two more lines: "
                 "\"overall_soundscape: \" with 1-4 sentences on the ambient sound, the sounds of the actions and "
                 "non-verbal human sounds such as breathing or laughter (not the spoken lines), and "
                 "\"non_diegetic_music: \" with 1-3 sentences on background music only the audience hears "
                 "(instruments, tempo, rhythm, changes in volume), or N/A when there is none. Do not write a header "
                 "line, any other field names, tags in angle brackets or speaker ids like (S1); the app adds them. "
                 "Answer with the prompt only, no preface, no quotes around it, no markdown."),
    "image": ("You improve prompts for text-to-image models. Expand the user's prompt into one cohesive paragraph: the "
              "subject(s) with their attributes and actions, the setting, lighting, composition and framing, the "
              "medium and style. Keep every subject, colour, action and spatial relation the user gave and the medium "
              "they asked for; do not add new people, animals or objects; add only visual details that serve their "
              "intent; if their prompt is already detailed, only polish it. If they want visible text, give the "
              "exact words in quotes. Answer with the prompt only, no preface, no markdown."),
    "edit": ("You improve instructions for an image editing model. Rewrite the user's request as one clear, direct "
             "edit instruction: what exactly to change and how, and what must stay the same (identity, pose, "
             "composition, background) unless they want it changed. Do not add changes they did not ask for. "
             "Answer with the instruction only, no preface, no markdown."),
}


def pick_refiner_models(clip_names: list[str]) -> list[dict]:
    """The language models ComfyUI lists that the refiner can use, most preferred (uncensored) first:
    [{"file", "type", "label", "uncensored"}]."""
    out, seen, labels = [], set(), set()
    for key, kind, text, free in REFINER_MODELS:
        for name in sorted(clip_names, key=len):
            if key in name.lower() and name not in seen and text not in labels:
                seen.add(name)
                labels.add(text)
                out.append({"file": name, "type": kind, "label": text, "uncensored": free})
    return out


SOUND_FIELDS = ("overall_soundscape", "non_diegetic_music")     # the two lines a "video_h3" refine ends with
REFINE_LENGTH = {"video_h3": 1024}       # tokens a refine may write (~500 words + the sound lines); others: 512


def clean_refined(text: str, lines: bool = False) -> str:
    """The model's answer without thinking blocks, prefaces, wrapping quotes or markdown, on one line.
    lines (the "video_h3" kind): the body on one line (a line break only before each later [Shot N]), then the
    sound lines the model wrote (overall_soundscape: / non_diegetic_music:), each on its own line; one it left out
    stays out (the prompt builder has defaults). What the app adds itself is taken out when the model wrote it
    anyway: a header line, the integrated_multimodal_description: label, <d> blocks (back to quotes), (S1) ids."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    text = re.sub(r"^\s*<think>.*", "", text, flags=re.S | re.I).strip()     # a thought cut off: no prompt in it
    text = re.sub(r"^(here(?:'s| is)[^:\n]*:|refined prompt:|prompt:)\s*", "", text, flags=re.I).strip()
    if lines:                                  # markdown anywhere: code fences, bold, headings
        text = re.sub(r"^\s*```\w*\s*$", "", text, flags=re.M)
        text = re.sub(r"^\s*#+\s*", "", text.replace("**", "").replace("__", ""), flags=re.M).strip()
    text = re.sub(r"^\*\*|\*\*$", "", text).strip()
    pairs = {'"': '"', "'": "'", "“": "”", "„": "“", "‘": "’", "«": "»"}
    inner = text[1:-1]
    if len(text) > 1 and pairs.get(text[0]) == text[-1] and text[0] not in inner and text[-1] not in inner:
        text = inner.strip()                       # only quotes around the whole answer, not spoken lines
    if not lines:
        return " ".join(text.split())
    text = "\n".join(x for x in text.splitlines()      # an official header line: the app writes its own
                     if not re.match(r"\s*(For the target video, at 0\.00 seconds|How the reference pictures align)",
                                     x, re.I))
    text = " ".join(text.split())
    text = re.sub(r"\bintegrated_multimodal_description\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"<d>\s*(?:\[[^\]]{1,30}\]\s*)?(.*?)\s*</d>", r'"\1"', text, flags=re.S | re.I)
    text = re.sub(r"\s*\(S\d+(?:\s*,\s*S\d+)*\)", "", text)
    parts = re.split(rf"\b({'|'.join(SOUND_FIELDS)})\s*:\s*", text, flags=re.I)
    body, sounds = parts[0].strip(), {}
    for name, value in zip(parts[1::2], parts[2::2]):
        value = value.strip()
        if value and name.lower() not in sounds:
            sounds[name.lower()] = value
    body = re.sub(r"\s*(\[Shot \d+\])", r"\n\1", body).strip()
    return "\n".join([body] + [f"{k}: {sounds[k]}" for k in SOUND_FIELDS if k in sounds]).strip()


_KIND_TEXT = {"place": "a place", "object": "an object", "style": "a visual style"}


def cast_instruction(cast: list[dict]) -> str:
    """What the refiner is told about the cast ([{"name", "desc", "voice": bool, "kind": person (also missing) |
    place | object | style}]): the names stay @Name, and a spoken line goes in quotes into a sentence that starts
    with its speaker's @Name (the app finds the speaker there and gives the line that character's voice). A place,
    object or style is said to be one, so the refiner does not make it a person."""
    if not cast:
        return ""

    def one(c: dict) -> str:
        desc = c.get("desc", "").strip().rstrip(".") if c.get("desc") else ""
        what = _KIND_TEXT.get(c.get("kind") or "person")
        if what:
            return f"@{c['name']} ({what}" + (f": {desc}" if desc else "") + ")"
        return (f"@{c['name']}" + (f" ({c['desc'].strip().rstrip('.')})" if c.get("desc") else "")
                + (", has a voice" if c.get("voice") else ""))
    people = "; ".join(one(c) for c in cast)
    things = [f"@{c['name']}" for c in cast if c.get("kind") in _KIND_TEXT]
    if things:
        people += (f". {', '.join(things)} {'is' if len(things) == 1 else 'are'} not a person: never give "
                   f"{'it' if len(things) == 1 else 'them'} a spoken line or a person's actions")
    return (f"The cast, written as @Name: {people}. The descriptions are for your information; do not repeat "
            "them in the prompt. Write a cast member always as @Name, exactly as written (not "
            "as a description or a plain name), at least once in each sentence about them. Mention a cast member only "
            "if the user's prompt or request does. Each spoken line: in double quotes, in its own sentence that "
            "starts with the speaker's @Name, never with he, she or they. Wrong: @A and @B sit down. She says "
            "\"Hi.\" He answers \"Hello.\" Right: @A and @B sit down. @A says \"Hi.\" @B answers \"Hello.\" "
            "Keep spoken lines in their language and word for word unless the user asks to change them, and do not add "
            "spoken lines. Do not write any tags in angle brackets.")


def clip_instruction(seconds: float | None, start: str | None) -> str:
    """What a "video_h3" refine is told about the clip: its length (cut times and the amount of action fit it) and
    its start frame. start: "seen" (the picture is attached), "unseen" (there is one, the model cannot see it) or
    None (the video starts from nothing given)."""
    out = []
    if seconds:
        s = float(seconds)
        out.append(f"The clip is {s:.2f} seconds long: describe only as much action as fits into {s:.2f} seconds, "
                   f"and every cut time is before {int(s // 60):02d}:{s % 60:06.3f}.")
    if start == "seen":
        out.append("The attached picture is the video's first frame (at 0.00 seconds). Anchor the first sentences "
                   "on what is visible in it: the style, the subjects with their look and clothing, their positions, "
                   "the place and the key objects; keep them consistent, then describe what happens next.")
    elif start == "unseen":
        out.append("The video starts from a given first frame that you cannot see. Do not invent how it looks (faces, "
                   "clothing, colours, the place); describe the style and the initial composition only as far as the "
                   "user's prompt gives them, and concentrate on what happens.")
    return " ".join(out)


def refine_messages(*, prompt: str, kind: str, hint: str = "", cast: list[dict] | None = None,
                    seconds: float | None = None, start: str | None = None, loras: str = "") -> tuple[str, str]:
    """The refiner's (system instruction, user message) for this kind of prompt (any backend).
    cast: the project's characters (video prompts), see cast_instruction. seconds, start: the clip ("video_h3"
    only), see clip_instruction. loras: what is known of the LoRAs the result is made with (lora_notes.guidance),
    put last."""
    system = REFINE_INSTRUCTIONS[kind] + " " + _SAFETY
    if cast and kind in ("video", "video_h3"):
        system += " " + cast_instruction(cast)
    if kind == "video_h3" and (seconds or start):
        system += " " + clip_instruction(seconds, start)
    tokens = re.findall(r"<[^<>]{1,40}>", prompt)          # <Picture 1>, <ref_image_0> …: only then mentioned
    tokens += re.findall(r"(?<![\w@])@\w[\w-]*", prompt)    # @Anna: the cast, mentioned by name
    if tokens:
        system += f" Keep these tokens exactly as they are, where they are: {', '.join(dict.fromkeys(tokens))}."
    if loras.strip():
        system += "\n\n" + loras.strip()
    user = prompt.strip() + (f"\n\nWhat the user wants changed in the prompt: {hint.strip()}" if hint.strip() else "")
    return system, user


def build_refine_prompt_workflow(*, prompt: str, kind: str, model: dict, seed: int, hint: str = "",
                                 temperature: float = 0.5, cast: list[dict] | None = None,
                                 seconds: float | None = None, start: str | None = None,
                                 image: str | None = None, loras: str = "") -> dict:
    """CLIPLoader (the LLM) -> TextGenerate with the instruction for this kind of prompt -> PreviewAny (text out).
    image: an uploaded picture the model sees (a vision model: the clip's first frame, start "seen")."""
    system, user = refine_messages(prompt=prompt, kind=kind, hint=hint, cast=cast, seconds=seconds, start=start,
                                   loras=loras)
    wf = {
        "llm": {"class_type": "CLIPLoader", "inputs": {"clip_name": model["file"], "type": model["type"],
                                                       "device": "default"}},
        "gen": {"class_type": "TextGenerate",
                "inputs": {"clip": ["llm", 0], "prompt": user, "max_length": REFINE_LENGTH.get(kind, 512),
                           "sampling_mode": "on",
                           "sampling_mode.temperature": float(temperature), "sampling_mode.top_k": 64,
                           "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                           "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": int(seed),
                           "thinking": False, "use_default_template": True, "system_prompt": system}},
        "show": {"class_type": "PreviewAny", "inputs": {"source": ["gen", 0]}},
    }
    if image:                                  # (as describe_face passes its picture)
        wf["pic"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        wf["gen"]["inputs"]["image"] = ["pic", 0]
    return wf


# The head swap LoRA's face description (the creator's instruction, Henge backend/edit_modes.py FACE_INSTRUCTION)
FACE_INSTRUCTION = ('Describe this face in a short comma-separated attribute list, in exactly this style: "Male, oval face '
                    'shape, average-sized head with strong jawline, light brown skin, dark eyes, black tousled hair, silver '
                    'hoop earring." Cover, in order: gender, head/face shape and proportions (e.g. oval/round/square/'
                    'heart-shaped, narrow/wide, jaw structure, whether the head reads as small/average/large relative to '
                    'the shoulders), skin tone, eye color, hair color and style, and any distinctive features (facial '
                    'hair, jewelry, makeup, glasses, etc.). Never name or guess real people, celebrities or fictional '
                    'characters. Only output the description, nothing else.')


def build_describe_face_workflow(*, image: str, model: dict, seed: int = 0) -> dict:
    """A vision language model describes the face in an uploaded picture (TextGenerate with the image)."""
    return {
        "llm": {"class_type": "CLIPLoader", "inputs": {"clip_name": model["file"], "type": model["type"],
                                                       "device": "default"}},
        "pic": {"class_type": "LoadImage", "inputs": {"image": image}},
        "gen": {"class_type": "TextGenerate",
                "inputs": {"clip": ["llm", 0], "prompt": FACE_INSTRUCTION, "image": ["pic", 0], "max_length": 160,
                           "sampling_mode": "on", "sampling_mode.temperature": 0.2, "sampling_mode.top_k": 64,
                           "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                           "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": int(seed),
                           "thinking": False, "use_default_template": True, "system_prompt": ""}},
        "show": {"class_type": "PreviewAny", "inputs": {"source": ["gen", 0]}},
    }


class ComfyPromptRefiner(ComfyGenerator):
    """Refines a prompt with a language model in ComfyUI (TextGenerate)."""

    def _check_models(self):
        if not self._object_info("TextGenerate"):
            raise ComfyError("this ComfyUI has no TextGenerate node (update ComfyUI)")
        req = self._object_info("CLIPLoader").get("input", {}).get("required", {})
        self.models = pick_refiner_models(_combo_options(req.get("clip_name")))
        if not self.models:
            raise ComfyError("ComfyUI lists no language model the prompt refiner can use (e.g. the Qwen3-VL 4B or a "
                             "Gemma 3 12B text encoder)")

    def vision_models(self) -> list[dict]:
        """The models that can look at a picture (Qwen3-VL)."""
        return [m for m in self.models if "vl" in m["file"].lower().replace("-", "")]

    def describe_face(self, picture: Path, seed: int = 0) -> str:
        """The face in a picture as the head swap LoRA's face details (a short attribute list)."""
        models = self.vision_models()
        if not models:
            raise ComfyError("ComfyUI lists no vision model to describe the face (e.g. the Qwen3-VL 4B text "
                             "encoder); write the face details yourself")
        uploads = [(self.upload(Path(picture)), Path(picture))]
        try:
            wf = build_describe_face_workflow(image=uploads[0][0], model=models[0], seed=seed)
            ws = self._connect_ws()                    # (before queueing: the job's messages are not missed)
            try:
                self._queue(wf)
                entry = self._wait(self.current, ws)
            except ComfyJobFailed:
                raise
            except Exception:
                self.interrupt()
                raise
            finally:
                self.current = None
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            if self.cleanup:
                self._drop_uploads(uploads)
        out = (entry.get("outputs") or {}).get("show") or {}
        text = (out.get("text") or [""])[0]
        text = " ".join(clean_refined(text if isinstance(text, str) else str(text)).split()).strip().strip('"')
        if not text:
            raise ComfyError("the vision model gave no face description")
        return text[:600]

    def complete(self, system: str, user: str, max_tokens: int = 512, model_file: str | None = None,
                 seed: int = 0, temperature: float = 0.2) -> str:
        """The model's raw answer to one question (text only; e.g. a LoRA's note, lora_notes)."""
        model = next((m for m in self.models if m["file"] == model_file), self.models[0])
        wf = build_refine_prompt_workflow(prompt=user, kind="image", model=model, seed=seed,
                                          temperature=temperature)
        wf["gen"]["inputs"].update({"system_prompt": system, "prompt": user, "max_length": int(max_tokens)})
        self._queue(wf)
        ws = self._connect_ws()
        try:
            entry = self._wait(self.current, ws)
        except ComfyJobFailed:
            raise
        except Exception:
            self.interrupt()
            raise
        finally:
            self.current = None
            if ws is not None:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass
        text = (((entry.get("outputs") or {}).get("show") or {}).get("text") or [""])[0]
        return text if isinstance(text, str) else str(text)

    def refine_prompt(self, prompt: str, kind: str, model_file: str | None = None, seed: int = 0,
                      hint: str = "", cast: list[dict] | None = None, seconds: float | None = None,
                      start_image: Path | str | None = None, loras: str = "") -> str:
        """seconds, start_image (the "video_h3" kind): the clip's length and the frame it starts from; a vision
        model sees that frame, any other is told not to invent how it looks. loras: lora_notes.guidance."""
        model = next((m for m in self.models if m["file"] == model_file), self.models[0])
        start, uploads = None, []
        if kind == "video_h3" and start_image:     # (a frame still to be made: there, but not to be seen yet)
            seen = Path(start_image).is_file() and model in self.vision_models()
            start = "seen" if seen else "unseen"
            if seen:
                uploads = [(self.upload(Path(start_image)), Path(start_image))]
        try:
            wf = build_refine_prompt_workflow(prompt=prompt, kind=kind, model=model, seed=seed, hint=hint, cast=cast,
                                              seconds=seconds if kind == "video_h3" else None, start=start,
                                              image=uploads[0][0] if uploads else None, loras=loras)
            self._queue(wf)
            ws = self._connect_ws()
            try:
                entry = self._wait(self.current, ws)
            except ComfyJobFailed:
                raise
            except Exception:
                self.interrupt()
                raise
            finally:
                self.current = None
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            if uploads and self.cleanup:
                self._drop_uploads(uploads)
        out = (entry.get("outputs") or {}).get("show") or {}
        text = (out.get("text") or [""])[0]
        text = clean_refined(text if isinstance(text, str) else str(text), lines=kind == "video_h3")
        if not text:
            raise ComfyError("the language model gave no answer")
        return text
