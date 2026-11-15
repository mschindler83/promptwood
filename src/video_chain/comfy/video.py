"""The workflow of one MiniMax-H3 clip, as ComfyUI's blueprint "Image to Video (MiniMax H3)" (see __init__)."""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .wfb import (FPS, SAVE_LAST_ID, SAVE_VIDEO_ID, loader_clip, loader_unet, loader_vae, lora_chain, save_video)

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"      # reference-to-video (identity references)
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SAMPLER = "res_multistep"
PIKA_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"   # ComfyUI's Pika H3 Flex templates
SCHEDULER = "simple"
REF_HINT = "The person has exactly the face and look of the person in <Picture 1>."


def with_ref_hint(prompt: str) -> str:
    """Add the sentence that ties the person to the reference, unless the prompt already names <Picture ...>."""
    return prompt if "<Picture" in prompt else f"{prompt.rstrip()} {REF_HINT}"


# A distilled H3 checkpoint (a few-step fine-tune with its turbo built in, as FastVideo's FastH3 OmniRef): the
# sampling of ComfyUI's FastH3 OmniRef template (8 steps, euler, simple, sigma shift 12 / 3), with exact attention.
# (Its template adds VSA sparse attention; measured with an H3 fine-tune without VSA gates, a 3 s clip at 672x384
# came out pixel-identical with and without it, so the approximation is left out.)
DISTILLED_CHECKPOINT: dict[str, Any] = {"steps": 8, "recipe": {"sampler": "euler", "scheduler": "simple",
                                                               "shift": (12.0, 3.0)}}

VEDA_NODE = "VedaSparseAttention"            # veda-sparse-attention (Veda-on-ComfyUI): learned sparse attention
VEDA_SETTINGS = {"generated_sparsity": "32", "reference_sparsity": "32", "full_attention_layers": "",
                 "full_attention_steps": "", "verbose": False}   # the node's defaults (as benchmarked)


def pick_veda_predictor(names: list[str], refs: bool) -> str | None:
    """The Veda predictor for a clip: the R2VA one for clips with references (faces, sounds), else the T2VA / FL2VA
    one (start / end frames); either when only one is there."""
    want, other = ("r2va", "t2va") if refs else ("t2va", "r2va")
    return next((n for n in names if want in n.lower()), None) or next((n for n in names if other in n.lower()),
                                                                        names[0] if names else None)


def default_ref_key(i: int) -> str:
    return f"ref_images.ref_image_{i}"


def default_audio_key(i: int) -> str:
    return f"ref_audios.ref_audio_{i}"


def default_video_key(i: int) -> str:
    return f"ref_videos.ref_video_{i}"


def build_workflow(*, prompt: str, image: str | None, seed: int, width: int, height: int, frames: int, steps: int,
                   loras: Sequence[tuple[str, float]], audio: bool, prefix: str, refs: Sequence[str] = (),
                   ref_size: str = "match", ref_key: Callable[[int], str] | None = None,
                   turbo: tuple[str, float] | None = None, end_image: str | None = None, cfg: float = 1.0,
                   guides: Sequence[dict] = (), hyperflow: tuple[str, float] | None = None,
                   pika: tuple[str, float] | None = None, audio_refs: Sequence[str] = (),
                   audio_key: Callable[[int], str] | None = None, start_pin: int = 1,
                   recipe: dict | None = None, veda: str | None = None,
                   keyframes: Sequence[tuple[str, int]] = (), frame_refs: Sequence[str] = (),
                   ref_videos: Sequence[str] = (), video_key: Callable[[int], str] | None = None,
                   unet: str | None = None) -> dict:
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
    ref_key / audio_key: how this ComfyUI names the n-th reference picture / sound input (None: ComfyUI's usual).
    frame_refs ("start" and / or "end", the h3 prompt format's settings "frame_pictures"): the plain start / end
    frame is also a reference picture, after all refs (start first), so the prompt can name it <Picture N>
    (MiniMaxH3AddGuide only pins a frame, the text encoder never sees it); the pinning stays as it is. The frames
    go whole (the node scales a reference picture down to the canvas area, never crops it).

    With audio_refs (uploaded sound files, <Audio 1> ... in the prompt): the ref2va model too (also without
    reference images); MiniMaxH3ReferenceToVideo gets them with the audio VAE, the prompt says what each one is for
    (a voice, a soundtrack to copy, a beat).

    With ref_videos (uploaded short clips, <Video 1> ... in the prompt; the end of the clip before that this clip
    continues): the ref2va model too; each goes in through LoadVideo + GetVideoComponents, its pictures only. Its
    soundtrack (ref_video_audios) is NOT wired: the node would give it an <Audio j> of its own right before the
    <Video k>, which would renumber the sounds and voices the prompt names. The node keeps a reference video's
    first 17n+5 frames, at most the clip's length, so the caller sends exactly such a length.

    With turbo (ComfyUI name of the turbo LoRA, strength): ComfyUI-MiniMax-H3-Turbo's loader (it handles the
    pruned model's adaln layers) right after the model, the other LoRAs after it, and its own sampler.

    With hyperflow (its weights file in models/hyperflow, strength): ComfyUI-HyperFlow-H3's ApplyHyperFlowH3 right
    after the model (curve refit on: the exact correction for the pruned int8 H3 files), the other LoRAs after it,
    Euler on the 8-step sigma grid it returns (no scheduler).

    With pika (ComfyUI name of the Pika H3 Flex LoRA for this model, strength): that LoRA right after the model,
    the other LoRAs after it, the lcm sampler on ManualSigmas PIKA_SIGMAS (ComfyUI's Pika H3 Flex templates).

    recipe: a LoRA's own sampling (e.g. the head swap LoRA's: {"sampler": "euler", "scheduler": "beta",
    "shift": (12.0, 3.0)}): MiniMaxH3SigmaShift after the LoRAs, that sampler and scheduler (not with turbo,
    hyperflow or pika).

    unet: another H3 checkpoint (ComfyUI's name of a fine-tune) instead of the standard files, for plain clips and
    clips with references alike (a distilled one samples with DISTILLED_CHECKPOINT's recipe).

    The parts: _check_combination (every invalid mix is refused before a node is built), _skeleton,
    _conditioning (the start / end frames, references, sounds), _guides, _keyframes, _model_chain (speed modes,
    LoRAs), _sampling and _output."""
    ref_key = ref_key or default_ref_key
    audio_key = audio_key or default_audio_key
    video_key = video_key or default_video_key
    guided = {g.get("at") for g in guides}
    use_start = image is not None and "start" not in guided
    use_end = end_image is not None and "end" not in guided
    has_refs = bool(refs or audio_refs or ref_videos)
    _check_combination(has_refs=has_refs, frame_refs=frame_refs, use_start=use_start, use_end=use_end,
                       turbo=turbo, hyperflow=hyperflow, pika=pika, recipe=recipe)
    wf = _skeleton(seed=seed, frames=frames, prefix=prefix, turbo=turbo, ref_model=has_refs, unet=unet)
    if use_start:
        wf["start"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    if use_end:
        wf["end"] = {"class_type": "LoadImage", "inputs": {"image": end_image}}
    if audio or audio_refs:
        wf["audio_vae"] = loader_vae(AUDIO_VAE)
    cond: dict[str, Any] = dict(prompt=prompt, width=width, height=height, frames=frames, use_start=use_start, use_end=use_end)
    if has_refs:
        latent, positive = _conditioning_refs(wf, **cond, refs=refs, ref_size=ref_size, ref_key=ref_key,
                                              audio_refs=audio_refs, audio_key=audio_key, frame_refs=frame_refs,
                                              start_pin=start_pin, ref_videos=ref_videos,
                                              video_key=video_key)
    else:
        latent, positive = _conditioning_plain(wf, **cond)
    positive = _guides(wf, positive, latent, guides, audio)
    positive = _keyframes(wf, positive, latent, keyframes)
    model = _model_chain(wf, loras=loras, turbo=turbo, hyperflow=hyperflow, pika=pika, recipe=recipe)
    _sampling(wf, model=model, positive=positive, latent=latent, steps=steps, cfg=cfg, hyperflow=hyperflow,
              pika=pika, recipe=recipe, veda=veda)
    _output(wf, audio=audio, prefix=prefix)
    return wf


def _check_combination(*, has_refs: bool, frame_refs: Sequence[str], use_start: bool, use_end: bool, turbo, hyperflow,
                       pika, recipe) -> None:
    """Refuse the settings that cannot be built together (a ValueError with the reason), before any node exists."""
    if has_refs:
        for f in [f for f in ("start", "end") if f in frame_refs]:
            if not (use_start if f == "start" else use_end):
                raise ValueError(f"the {f} frame is to be a reference picture, but the clip has no plain {f} frame")
    elif frame_refs:
        raise ValueError("frames are reference pictures only with the reference model (pictures or sounds)")
    if pika and (turbo or hyperflow):
        raise ValueError("Pika H3 Flex cannot be combined with Turbo or HyperFlow")
    if recipe and (turbo or hyperflow or pika):
        raise ValueError("a LoRA's own sampling recipe cannot be combined with Turbo, HyperFlow or Pika")


def _skeleton(*, seed: int, frames: int, prefix: str, turbo, ref_model: bool, unet: str | None = None) -> dict:
    """The nodes every clip has: the model files (unet: a chosen checkpoint for every clip), the noise, the sampler
    (replaced later by a speed mode or a recipe), the decode and the last frame saved as a PNG."""
    return {
        "unet": loader_unet(unet or (UNET_REF if ref_model else UNET)),
        "clip": loader_clip(TEXT_ENCODER, "minimax"),
        "vae": loader_vae(VIDEO_VAE),
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": ({"class_type": "MiniMaxH3TurboSampler", "inputs": {}} if turbo else
                    {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}}),
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "last": {"class_type": "ImageFromBatch", "inputs": {"image": ["decode", 0], "batch_index": int(frames) - 1,
                                                             "length": 1}},
        SAVE_LAST_ID: {"class_type": "SaveImage", "inputs": {"images": ["last", 0],
                                                             "filename_prefix": f"{prefix}_last"}},
    }


def _conditioning_refs(wf: dict, *, prompt: str, width: int, height: int, frames: int, use_start: bool,
                       use_end: bool, refs: Sequence[str], ref_size: str, ref_key: Callable[[int], str],
                       audio_refs: Sequence[str], audio_key: Callable[[int], str], frame_refs: Sequence[str],
                       start_pin: int, ref_videos: Sequence[str] = (),
                       video_key: Callable[[int], str] | None = None) -> tuple[list, list]:
    """The reference model: MiniMaxH3ReferenceToVideo with the pictures and sounds, the start frame pinned at 0 and
    the end frame at -1 by MiniMaxH3AddGuide. Returns (latent, positive) as node outputs."""
    inputs = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
              "height": int(height), "length": int(frames), "ref_image_size": ref_size}
    for i, ref in enumerate(refs):
        wf[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        inputs[ref_key(i)] = [f"ref{i}", 0]
    for i, f in enumerate([f for f in ("start", "end") if f in frame_refs], start=len(refs)):
        inputs[ref_key(i)] = [f, 0]                # the same LoadImage the guide pins
    for i, clip in enumerate(ref_videos):          # pictures only
        assert video_key is not None                # (given with the videos) (see build_workflow: no ref_video_audios)
        wf[f"rvideo{i}"] = {"class_type": "LoadVideo", "inputs": {"file": clip}}
        wf[f"rvparts{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"rvideo{i}", 0]}}
        inputs[video_key(i)] = [f"rvparts{i}", 0]
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
    return latent, positive


def _conditioning_plain(wf: dict, *, prompt: str, width: int, height: int, frames: int, use_start: bool,
                        use_end: bool) -> tuple[list, list]:
    """The plain model: MiniMaxH3ImageToVideo with the start frame as first and the end frame as last frame."""
    inputs = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "width": int(width),
              "height": int(height), "length": int(frames)}
    if use_start:
        inputs["first_frame"] = ["start", 0]
    if use_end:
        inputs["last_frame"] = ["end", 0]
    wf["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}
    return ["cond", 1], ["cond", 0]


def _guides(wf: dict, positive: list, latent: list, guides: Sequence[dict], audio: bool) -> list:
    """Motion guides: frames (+ sound) of the neighbouring clips. Returns the new positive conditioning."""
    for k, g in enumerate(guides):
        wf[f"gvideo{k}"] = {"class_type": "LoadVideo", "inputs": {"file": g["name"]}}
        wf[f"gparts{k}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"gvideo{k}", 0]}}
        gin = {"positive": positive, "vae": ["vae", 0], "latent": latent, "image": [f"gparts{k}", 0],
               "frame_idx": int(g["frame_idx"])}
        if audio and g.get("audio"):
            gin.update({"audio": [f"gparts{k}", 1], "audio_vae": ["audio_vae", 0]})
        wf[f"guide{k}"] = {"class_type": "MiniMaxH3AddGuide", "inputs": gin}
        positive = [f"guide{k}", 0]
    return positive


def _keyframes(wf: dict, positive: list, latent: list, keyframes: Sequence[tuple[str, int]]) -> list:
    """Middle keyframes. Returns the new positive conditioning."""
    for k, (name, idx) in enumerate(keyframes):
        wf[f"key{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf[f"key_guide{k}"] = {"class_type": "MiniMaxH3AddGuide",
                               "inputs": {"positive": positive, "vae": ["vae", 0], "latent": latent,
                                          "image": [f"key{k}", 0], "frame_idx": int(idx)}}
        positive = [f"key_guide{k}", 0]
    return positive


def _model_chain(wf: dict, *, loras: Sequence[tuple[str, float]], turbo, hyperflow, pika, recipe) -> list:
    """The model with its speed mode (pika, hyperflow or turbo, in that order), the LoRAs and the recipe's sigma
    shift. Returns the last model output."""
    model = ["unet", 0]
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
    model = lora_chain(wf, model, loras)
    if recipe and recipe.get("shift"):
        wf["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                       "inputs": {"model": model, "shift_video": float(recipe["shift"][0]),
                                  "shift_audio": float(recipe["shift"][1])}}
        model = ["shift", 0]
    return model


def _sampling(wf: dict, *, model: list, positive: list, latent: list, steps: int, cfg: float, hyperflow, pika,
              recipe, veda: str | None) -> None:
    """The sampler and sigmas of the speed mode or recipe, Veda sparse attention, the guider and the sampling node."""
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


def _output(wf: dict, *, audio: bool, prefix: str) -> None:
    """The video: CreateVideo (with the decoded sound when the clip has one) and its SaveVideo."""
    video_inputs: dict = {"images": ["decode", 0], "fps": FPS}
    if audio:
        wf["decode_audio"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0],
                                                                         "vae": ["audio_vae", 0]}}
        video_inputs["audio"] = ["decode_audio", 0]
    wf["video"] = {"class_type": "CreateVideo", "inputs": video_inputs}
    wf[SAVE_VIDEO_ID] = save_video(prefix)
