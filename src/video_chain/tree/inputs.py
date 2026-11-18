"""The model prompt of a clip and its settings (LoRAs, frames, guide) worked out from the tree."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..triggers import with_triggers
from .. import camera as _camera
from .. import chain
from .. import h3_prompt as _h3
from .. import steady as _steady
from .consts import CONTINUE_CHOICES, DEFAULT_FRAMES, MAX_CAST_PICTURES, checkpoint_of, clean_sampling
from .model import Node, Tree
from .timeline import clip_length, full_prompt, handover_index, made_frames
from .files import node_loras
from .cast import audio_mentioned, is_h3
from .clipcast import _language, _resolver, clip_cast, lead_on
from .keyframes import end_image_used
from .starts import start_source
from .audio_tracks import _soundtrack_sentence, _stack_triggers
from .schema import Options

if TYPE_CHECKING:
    from .checker import Checker


def model_prompt(options: Options, node: Node, settings: dict | None, *, start: bool = False,
                 end: bool = False) -> str:
    """What the model gets for a generated clip: the cast's prompt (clip_cast) with its LoRAs' trigger words, and
    with "Continue the video's sound" the sentence that makes the last sound reference its soundtrack.
    start / end: the clip is given a plain start / end frame (plain_frames; not one a motion guide replaces). Only
    the h3 format uses them (the header of the base formats, or "begins exactly from the given start frame"); a
    frame the settings number ("frame_pictures") counts as given anyway."""
    if is_h3(options):
        return _h3_model_prompt(options, node, settings or {}, start, end)
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


def _h3_model_prompt(options: Options, node: Node, s: dict, start: bool, end: bool) -> str:
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
    continues = 1 if s.get("continue") else None     # the only reference video: <Video 1>
    return _h3.compose(h["body"], start=start, end=end, seconds=frames / chain.FPS, subjects=h["subjects"],
                       frame_pictures=frame_numbers(len(use["images"]), s.get("frame_pictures")),
                       sounds=h["sounds"], soundtrack=soundtrack, lines=h["lines"], language=_language(options),
                       lead_implicit=h["lead_implicit"], triggers=list(s.get("lora_triggers") or []),
                       extra=[x for x in extra if x], soundscape=h["soundscape"], music=h["music"],
                       resolve=_resolver(options), continues=continues,
                       reference=bool(use["images"] or use["voices"] or h["sounds"] or soundtrack or continues))


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


PIN_NOISE_OFF = 0.999   # H3's own level (comfy.video.PIN_NOISE_OFF): not recorded


def pin_noise_of(options) -> float | None:
    """The project's pinned-frame noise level (option "pin_noise"), None when it is H3's own (not set) or cannot be
    used (not a number, outside 0.5-1.0)."""
    v = options.get("pin_noise")
    try:
        v = None if v is None else round(float(v), 3)
    except (TypeError, ValueError):
        return None
    if v is None or v != v or not 0.5 <= v <= 1.0 or abs(v - PIN_NOISE_OFF) < 1e-9:
        return None
    return v


def node_settings(base: dict, options: Options, node: Node, audio_pool: list[dict] | None = None,
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
    pictures, within MAX_CAST_PICTURES (the end frame's number goes first, then the start frame's), and
    "continue": {"frames": n} for a clip that continues the clip before (continue_frames; it takes the reference
    model, so its frames are numbered too)."""
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
    distilled = bool((base.get("checkpoint") or {}).get("distilled"))   # (its own steps)
    stack = node_loras(node)
    lora, strength = stack[0] if stack else (None, 1.0)
    # the clip's own sampling; a speed mode samples its own way (as it has its own steps)
    speed = base.get("turbo") or base.get("hyperflow") or base.get("pika")
    own = {} if speed or node.is_video else clean_sampling(ov.get("sampling"))
    s = chain.settings_dict(model=base["model"], lora=lora, extra_loras=stack[1:],
                               strength=strength,
                               frames=ov.get("frames", options.get("frames", DEFAULT_FRAMES)),
                               # turbo / HyperFlow: their steps for every clip (a clip's own steps would break it)
                               steps=(base["steps"] if base.get("turbo") or base.get("hyperflow") or base.get("pika")
                                      or distilled else ov.get("steps", options.get("steps", 28))),
                               hyperflow=(base["hyperflow"]["strength"] if base.get("hyperflow") else None),
                               pika=(base["pika"]["strength"] if base.get("pika") else None),
                               width=base["width"], height=base["height"], audio=base["audio"],
                               turbo=(base["turbo"]["lora"], base["turbo"]["strength"]) if base.get("turbo") else None,
                               cfg=float(own.get("cfg", base.get("cfg", 1.0))),
                               backend="comfyui", refs=refs or None, cast=use["record"],
                               ref_size=ident.get("size", "match"), ref_hint=True if h3 else ident.get("hint", True),
                               guide=int(ov.get("guide") or 0),
                               style=(base.get("prompt_prefix", ""), base.get("prompt_suffix", "")),
                               audio_refs=sounds or None, voice_language=use.get("language"),
                               lora_triggers=_stack_triggers(options, node, stack),
                               source_sound=source_sound,
                               veda=veda_for(options, base["width"], base["height"],
                                             int(ov.get("frames", options.get("frames", DEFAULT_FRAMES)))),
                               checkpoint=base.get("checkpoint"), two_pass=base.get("two_pass"),
                               camera=_camera.clean(ov.get("camera")),
                               steady=_steady.VERSION if options.get("steady_look") and node.parent is not None
                               and not node.is_video else None,
                               pin_noise=pin_noise_of(options) if not node.is_video else None)
    recipe = {k: own[k] for k in ("sampler", "scheduler", "shift") if k in own}
    if recipe:                                   # (only when the clip sets any: other clips are recorded as before)
        s["recipe"] = recipe
    if not h3:
        return s
    s["prompt_format"] = "h3"
    if use["record"]:
        s["cast"] = [dict(c) for c in use["record"]]
    cont = continue_frames(options, node)
    if cont:
        s["continue"] = {"frames": cont}
    if (refs or sounds or given or cont) and not use["h3"]["structured"]:     # the reference model: numbered frames
        room = max(0, MAX_CAST_PICTURES - len(refs))
        frames = [f for f, on in zip(("start", "end"), frames_given) if on][:room]
        if frames:
            s["frame_pictures"] = frames
    return s


def continue_frames(options: Options, node: Node) -> int:
    """How many frames of the end of the clip before a clip is shown as <Video 1> (override "continue"); 0: none,
    also in the classic format (its prompt cannot say what the video is) and for a first clip (nothing before it)."""
    if node.is_video or node.parent is None or not is_h3(options):
        return 0
    return int((node.overrides or {}).get("continue") or 0)


def frames_before(tree: Tree, node: Node) -> int:
    """Frames of the clip before a node that lead into it: up to its hand-over frame (its made length, else its
    length setting / the video's range)."""
    assert node.parent is not None                   # (only asked for a clip with one)
    parent = tree.nodes[node.parent]
    n = made_frames(parent) or clip_length(tree, parent)
    k = handover_index(tree, parent, n)
    return k + 1 if k is not None else n


def continue_choices(tree: Tree, node: Node) -> list[int]:
    """The continuation lengths that fit a clip: no longer than what the clip before hands over with, and no longer
    than the clip itself (the node cuts a longer reference video to the target's frame count, keeping its START,
    so the end that matters would be lost)."""
    if node.is_video or node.parent is None:
        return []
    have, own = frames_before(tree, node), clip_length(tree, node)
    return [c for c in CONTINUE_CHOICES if c <= have and c <= own]


def veda_for(options: Options, width: int, height: int, frames: int) -> bool:
    """A clip of this size is made with Veda sparse attention: the project allows it, it is large enough, and no
    distilled checkpoint is chosen (that one samples with exact attention)."""
    ck = checkpoint_of(options)
    return bool((options.get("veda") or {"on": True}).get("on", True)) and not (ck and ck["distilled"]) \
        and chain.veda_useful(width, height, frames)


def loras_needed(tree: Tree, checker: Checker, nids: list[str]) -> list[str]:
    """The LoRA files these nodes are made with (each once, in order)."""
    out: list[str] = []
    for nid in nids:
        if tree.nodes[nid].is_video:
            continue
        s = checker.settings(nid)
        if s is None:
            continue
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
