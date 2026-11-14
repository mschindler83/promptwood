"""The model prompt of a clip and its settings (LoRAs, frames, guide) worked out from the tree."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .. import camera as _camera
from .. import chain
from .. import h3_prompt as _h3
from .consts import DEFAULT_FRAMES, MAX_CAST_PICTURES
from .model import Node, Tree
from .timeline import full_prompt
from .files import node_loras
from .cast import audio_mentioned, is_h3
from .clipcast import _language, _resolver, clip_cast, lead_on
from .keyframes import end_image_used
from .starts import start_source
from .audio_tracks import _soundtrack_sentence, _stack_triggers

if TYPE_CHECKING:
    from .checker import Checker


def model_prompt(options: dict, node: Node, settings: dict | None, *, start: bool = False,
                 end: bool = False) -> str:
    """What the model gets for a generated clip: the cast's prompt (clip_cast) with its LoRAs' trigger words, and
    with "Continue the video's sound" the sentence that makes the last sound reference its soundtrack.
    start / end: the clip is given a plain start / end frame (plain_frames; not one a motion guide replaces). Only
    the h3 format uses them (the header of the base formats, or "begins exactly from the given start frame"); a
    frame the settings number ("frame_pictures") counts as given anyway."""
    if is_h3(options):
        return _h3_model_prompt(options, node, settings or {}, start, end)
    from ..triggers import with_triggers
    prompt = _camera.apply(node.prompt, (settings or {}).get("camera"))    # #tokens and the clip's camera move
    ss = (settings or {}).get("source_sound")
    given = bool(ss) and not (isinstance(ss, dict) and ss.get("overlay"))      # (then without the voices)
    text = with_triggers(clip_cast(options, prompt, voices=not given, lead=lead_on(node))["prompt"],
                         list((settings or {}).get("lora_triggers") or []))
    if ss and not (isinstance(ss, dict) and ss.get("overlay")):   # (a sound laid over is not the model's)
        k = len((settings or {}).get("audio_refs") or []) + 1
        text = f"{text.rstrip()} {_soundtrack_sentence(ss, k)}"
    notes = [f"At {k['at']:.1f} s: {k['note'].strip().rstrip('.')}." for k in node.keyframes if k["note"].strip()]
    if notes:                                      # what happens at each keyframe, in time order
        text = f"{text.rstrip()} {' '.join(notes)}"
    return text


def frame_numbers(n_pictures: int, frames: list[str] | tuple[str, ...] | None) -> dict[str, int]:
    """{"start": N, "end": M}: the <Picture N> of the frames sent as reference pictures (settings
    "frame_pictures"), numbered after the clip's n_pictures subject pictures, start first (the workflow's order)."""
    out, k = {}, n_pictures
    for f in ("start", "end"):
        if f in (frames or ()):
            k += 1
            out[f] = k
    return out


def _h3_model_prompt(options: dict, node: Node, s: dict, start: bool, end: bool) -> str:
    """model_prompt in the h3 format: h3_prompt.compose with the cast plan of _h3_cast, the clip's frames (numbered
    ones from s["frame_pictures"]), its seconds, trigger words, and as further sentences of the description the
    camera move, the keyframe notes and the soundtrack sentence (its <Audio N> after the sounds and voices)."""
    ss = s.get("source_sound")
    given = bool(ss) and not (isinstance(ss, dict) and ss.get("overlay"))
    use = clip_cast(options, _camera.expand(node.prompt), voices=not given, lead=lead_on(node))
    h = use["h3"]
    if h["structured"]:
        return h["body"]
    extra = [_camera.sentence(s.get("camera"))]
    extra += [f"At {k['at']:.1f} s: {k['note'].strip().rstrip('.')}." for k in node.keyframes if k["note"].strip()]
    soundtrack = None
    if given:
        soundtrack = len(s.get("audio_refs") or []) + 1
        extra.append(_soundtrack_sentence(ss, soundtrack))
    frames = int(s.get("frames") or (node.overrides or {}).get("frames") or options.get("frames", DEFAULT_FRAMES))
    return _h3.compose(h["body"], start=start, end=end, seconds=frames / chain.FPS, subjects=h["subjects"],
                       frame_pictures=frame_numbers(len(use["images"]), s.get("frame_pictures")),
                       sounds=h["sounds"], soundtrack=soundtrack, lines=h["lines"], language=_language(options),
                       lead_implicit=h["lead_implicit"], triggers=list(s.get("lora_triggers") or []),
                       extra=[x for x in extra if x], soundscape=h["soundscape"], music=h["music"],
                       resolve=_resolver(options),
                       reference=bool(use["images"] or use["voices"] or h["sounds"] or soundtrack))


def plain_frames(project: Path, tree: Tree, node: Node) -> tuple[bool, bool]:
    """(start, end): a generated clip is given a plain start frame / end frame, as run_tree and the workflow send
    them: a start frame unless it is a first clip without a start image or its start is a motion guide (one
    after another clip), an end frame when a video follows (without a motion guide) or it has its own end image."""
    if node.is_video:
        return False, False
    guide = int((node.overrides or {}).get("guide") or 0)
    start = start_source(project, node) is not None if node.parent is None else not guide
    vc = tree.video_child(node.id)
    end = not guide if vc is not None else bool(end_image_used(tree, node))
    return start, end


def node_settings(base: dict, options: dict, node: Node, audio_pool: list[dict] | None = None,
                  picture_sha: Callable[[str], str | None] | None = None,
                  source_sound: dict | None = None, frames_given: tuple[bool, bool] = (False, False)) -> dict:
    """The settings one node is made with: the project's, with the node's own LoRA stack and overrides.
    The first LoRA of the stack is recorded as "lora", the others as "extra_loras" (as before).
    audio_pool: the project's sound references ({"file", "sha1"}); the clip records those up to the highest
    <Audio N> its prompt names (none: nothing recorded, as before), then the voices of the characters who speak.
    picture_sha: the content hash of a face picture (project-relative name); the clip records the pictures it is
    made with (see clip_cast), and with @mentions what the model prompt says about the characters.
    In the h3 format (is_h3) it also records "prompt_format": "h3", the cast entries the prompt describes (also
    without pictures: their descriptions are in it), and "frame_pictures": frames_given = plain_frames (start,
    end) that are also sent as numbered reference pictures: with the reference model only, after the subject
    pictures, within MAX_CAST_PICTURES (the end frame's number goes first, then the start frame's)."""
    ov = node.overrides or {}
    n_audio = 0 if node.is_video else audio_mentioned(full_prompt(options, node.prompt))
    given = source_sound is not None and not source_sound.get("overlay")       # (then without the voices)
    h3 = is_h3(options) and not node.is_video
    use = clip_cast(options, node.prompt, voices=not given, lead=lead_on(node)) if not node.is_video \
        else {"images": [], "record": None, "voices": []}
    sha = picture_sha or (lambda rel: None)
    refs = [{"file": r, "sha1": sha(r)} for r in use["images"]]
    sounds = (audio_pool or [])[:n_audio] if n_audio else []
    sounds = sounds + [{"file": r, "sha1": sha(r)} for r in use.get("voices", [])]
    ident = options.get("identity") or {}
    stack = node_loras(node)
    lora, strength = stack[0] if stack else (None, 1.0)
    s = chain.settings_dict(model=base["model"], lora=lora, extra_loras=stack[1:],
                               strength=strength,
                               frames=ov.get("frames", options.get("frames", DEFAULT_FRAMES)),
                               # turbo / HyperFlow: their steps for every clip (a clip's own steps would break it)
                               steps=(base["steps"] if base.get("turbo") or base.get("hyperflow") or base.get("pika")
                                      else ov.get("steps", options.get("steps", 28))),
                               hyperflow=(base["hyperflow"]["strength"] if base.get("hyperflow") else None),
                               pika=(base["pika"]["strength"] if base.get("pika") else None),
                               width=base["width"], height=base["height"], audio=base["audio"],
                               turbo=(base["turbo"]["lora"], base["turbo"]["strength"]) if base.get("turbo") else None,
                               cfg=float(base.get("cfg", 1.0)),
                               backend="comfyui", refs=refs or None, cast=use["record"],
                               ref_size=ident.get("size", "match"), ref_hint=True if h3 else ident.get("hint", True),
                               guide=int(ov.get("guide") or 0),
                               style=(base.get("prompt_prefix", ""), base.get("prompt_suffix", "")),
                               audio_refs=sounds or None, voice_language=use.get("language"),
                               lora_triggers=_stack_triggers(options, node, stack),
                               source_sound=source_sound,
                               veda=veda_for(options, base["width"], base["height"],
                                             int(ov.get("frames", options.get("frames", DEFAULT_FRAMES)))),
                               camera=_camera.clean(ov.get("camera")))
    if not h3:
        return s
    s["prompt_format"] = "h3"
    if use["record"]:
        s["cast"] = [dict(c) for c in use["record"]]
    if (refs or sounds or given) and not use["h3"]["structured"]:     # the reference model: frames get numbers
        room = max(0, MAX_CAST_PICTURES - len(refs))
        frames = [f for f, on in zip(("start", "end"), frames_given) if on][:room]
        if frames:
            s["frame_pictures"] = frames
    return s


def veda_for(options: dict, width: int, height: int, frames: int) -> bool:
    """A clip of this size is made with Veda sparse attention: the project allows it and it is large enough."""
    return bool((options.get("veda") or {"on": True}).get("on", True)) and chain.veda_useful(width, height, frames)


def loras_needed(tree: Tree, checker: Checker, nids: list[str]) -> list[str]:
    """The LoRA files these nodes are made with (each once, in order)."""
    out: list[str] = []
    for nid in nids:
        if tree.nodes[nid].is_video:
            continue
        s = checker.settings(nid)
        for lora in ([s["lora"]] + [e["lora"] for e in s.get("extra_loras") or []]
                     + ([s["turbo"]["lora"]] if s.get("turbo") else [])):
            if lora and lora not in out:
                out.append(lora)
    return out


def empty_prompts(tree: Tree, targets: list[str]) -> list[str]:
    """Generated clips on the target paths without a prompt."""
    return [n.id for t in targets for n in tree.path_to(t) if not n.is_video and not n.prompt.strip()]


def plan(tree: Tree, checker: Checker, targets: list[str]) -> list[str]:
    """Nodes to make for these targets: every not-done node on their paths, parents first, each once. A video a
    generated clip leads into comes before that clip (its first frame is the clip's end), also when the video
    itself is not on the path."""
    out: list[str] = []

    def need(nid: str):
        if nid not in out and checker.status(nid) != "done":
            out.append(nid)
    for t in targets:
        for n in tree.path_to(t):
            vc = tree.video_child(n.id)
            if vc is not None:
                need(vc.id)
            need(n.id)
    return out
