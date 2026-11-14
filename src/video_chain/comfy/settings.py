"""The settings dict of one clip as ComfyGenerator.generate reads it (tree.py builds it from the project)."""
from __future__ import annotations

from typing import TypedDict

# What a clip gets when its settings say nothing (tree.py has its own DEFAULT_FRAMES for the project's clips)
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FRAMES = 73
DEFAULT_STEPS = 20
DEFAULT_REF_SIZE = "match"


class LoraEntry(TypedDict, total=False):
    lora: str                  # the LoRA file
    strength: float            # (1.0 when missing)


class Guide(TypedDict, total=False):
    video: str                 # the neighbouring clip's file
    frame_idx: int
    audio: bool
    at: str                    # "start" | "end"


class Keyframe(TypedDict):
    image: str                 # the picture's file
    frame_idx: int


class ClipSettings(TypedDict, total=False):
    """Every key is optional: a missing key means the default noted here. Speed modes (turbo / hyperflow / pika):
    a LoraEntry-shaped dict ({"lora", "strength"}, hyperflow and pika: only "strength"), missing or empty: off."""
    width: int                 # DEFAULT_WIDTH
    height: int                # DEFAULT_HEIGHT
    frames: int                # DEFAULT_FRAMES
    steps: int                 # DEFAULT_STEPS
    cfg: float                 # 1.0
    audio: bool                # True: the clip has sound
    lora: str                  # the first LoRA of the stack ...
    strength: float            # ... and its strength (1.0)
    extra_loras: list[LoraEntry]    # the others, chained after it
    turbo: LoraEntry
    hyperflow: dict
    pika: dict
    veda: bool                 # Veda sparse attention
    recipe: dict               # a LoRA's own sampling: {"sampler", "scheduler", "shift": (video, audio)}
    start_pin: int             # frames the start picture is held as a still (1)
    end_image: str             # a file: the frame the clip must end at
    guides: list[Guide]
    keyframes: list[Keyframe]
    ref_paths: list[str]       # reference pictures (the reference model)
    ref_size: str              # DEFAULT_REF_SIZE
    ref_hint: bool             # True: the prompt gets the sentence that ties the person to <Picture 1>
    audio_paths: list[str]     # reference sounds
    ref_videos: list[str]      # reference videos (the end of the clip before: a continuation), pictures only
    frame_pictures: list[str]  # "start" and / or "end": that frame is also a reference picture
    prompt_format: str         # "h3": the prompt ties the pictures itself
    refs: list                 # (check_speed only: tells that the clip has references)
    audio_refs: list           # (check_speed only)
