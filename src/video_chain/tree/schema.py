"""The shapes of the dicts the project file holds. They are plain dicts at run time (tree.json is read and written
as it always was); these types only tell readers and the type checker which keys exist. Every key is optional
(total=False): older files lack newer keys."""
from __future__ import annotations

from typing import Any, TypedDict


class Keyframe(TypedDict):
    image: str                          # ends/... (project-relative)
    at: float                           # seconds into the clip
    note: str


class Source(TypedDict, total=False):
    """A video node's source (video_source)."""
    file: str                           # absolute path
    start: float                        # seconds
    end: float
    rotate: int                         # clockwise degrees; left out at 0
    replaced: dict[str, Any]            # swap.apply: what the person was replaced with, with "original" (a Source)


# (the functional form: "continue" is a Python keyword)
Overrides = TypedDict("Overrides", {
    "loras": list[dict[str, Any]],      # [{"path", "strength", "off"?}], applied in order; "off": kept, not used
    "frames": int,
    "steps": int,
    "guide": int,                       # one of GUIDE_CHOICES
    "continue": int,                    # frames of the clip before shown to the model as <Video 1> (h3 format)
    "source_sound": str | None,         # "previous", "track:<id>" or a real video's sound
    "sound_over": bool | str,
    "sound_mix_level": float,
    "camera": Any,
    "lead": Any,
    "sampling": dict[str, Any],         # the clip's own sampler / scheduler / shift / cfg (consts.clean_sampling)
}, total=False)
"""A clip's own settings over the project's (OVERRIDE_KEYS)."""


class MadeRecord(TypedDict, total=False):
    """What a node's current files were made from (Node.made); the keys of a video node are kind, key, frames,
    video, frame and first."""
    prompt: str
    seed: int
    settings: dict[str, Any]
    overrides: Overrides
    end_image: str | None
    keyframes: list[Keyframe]
    start_sha1: str | None
    video: str                          # nodes/... (project-relative)
    frame: str
    first: str
    frames: int
    own_sound: str                      # the model's own sound of a clip whose sound was laid over
    sound_rev: int                      # counts the remixes of the sound
    cuts: list[int]                     # hard cuts H3 made (frame indexes)
    end_key: Any
    keyframe_key: Any
    guide_from: Any
    continue_from: Any
    kind: str                           # a video node's record
    key: dict[str, Any]


class Eleven(TypedDict, total=False):
    id: str
    name: str


class VoiceRef(TypedDict):
    file: str
    secs: float


class Voice(TypedDict, total=False):
    file: str                           # refs/... (cut to VOICE_MAX_SECS)
    name: str
    secs: float
    text: str                           # what clone_file says
    ref: VoiceRef                       # a longer part of the source, for text to speech to clone


class CastEntry(TypedDict, total=False):
    name: str
    images: list[str]                   # refs/...
    desc: str
    lead: bool
    kind: str                           # CAST_KINDS; a person has none
    voice: Voice
    tts: dict[str, Eleven]              # {"eleven": {...}}: its ElevenLabs voice


class AudioRef(TypedDict, total=False):
    file: str
    name: str
    secs: float


class Track(TypedDict, total=False):
    id: str
    name: str
    file: str                           # the cut sound as WAV in refs/
    secs: float
    source: str
    start: float
    end: float
    tts: dict[str, Any]                 # a track made by speak_lines: {"clip", "provider", "language", "lines"}


class Options(TypedDict, total=False):
    """tree.options: the project's settings (DEFAULT_OPTIONS and what is added by use)."""
    model: str
    frames: int
    steps: int
    resolution: int
    audio: bool
    keep_repeated: bool
    seed: int
    aspect: str                         # the shape of a project without a start picture or video (shape unset)
    shape: str                          # the clips' shape (a sources.ASPECTS key); unset: the start image's / video's
    crops: dict[str, dict[str, Any]]    # the cut of each start picture of another shape (crops.py)
    size: list[int]                     # [width, height]: the clips' exact size (chain.exact_size); none: worked out
    turbo: dict[str, Any]
    hyperflow: dict[str, Any]
    pika: dict[str, Any]
    veda: dict[str, Any]
    two_pass: dict[str, Any]            # {"on": bool}: every clip in two passes (consts.TWO_PASS)
    checkpoint: dict[str, Any]          # another H3 checkpoint: {"name", "distilled"} (consts.checkpoint_of)
    cast: list[CastEntry]
    identity: dict[str, Any]
    audio_refs: list[AudioRef]
    tracks: list[Track]
    style: dict[str, str]
    lora_triggers: bool
    prompt_format: str
    tts: dict[str, Any]
