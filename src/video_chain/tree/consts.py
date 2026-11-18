"""Names and limits of the project file and its data (no code)."""
from __future__ import annotations

import re
from typing import Any

from .schema import Options



TREE_FILE = "tree.json"


TREE_VERSION = 3


class NewerTreeError(ValueError):
    """tree.json was written by a newer app: not opened (and not replaced by the backup)."""


START_FILE = "start.png"


NO_LORA = "none"                   # old override value: this node without any LoRA


OVERRIDE_KEYS = ("loras", "frames", "steps", "guide", "source_sound", "sound_over", "sound_mix_level", "camera",
                 "lead", "continue", "sampling")


CFG_MAX = 30.0


def clean_sampling(s: Any) -> dict:
    """A clip's own sampling as it counts: {"sampler", "scheduler" (ComfyUI's names), "shift" ([video, audio]
    sigma shift, both above 0), "cfg" (above 0, up to CFG_MAX; 1 is the usual one and not kept)}, each only when
    set. Anything else is left out, so {} means: nothing of its own."""
    out: dict = {}
    if not isinstance(s, dict):
        return out
    for key in ("sampler", "scheduler"):
        if isinstance(s.get(key), str) and s[key].strip():
            out[key] = s[key].strip()
    shift = s.get("shift")
    if isinstance(shift, (list, tuple)) and len(shift) == 2 and \
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in shift):
        out["shift"] = [float(shift[0]), float(shift[1])]
    cfg = s.get("cfg")
    if isinstance(cfg, (int, float)) and not isinstance(cfg, bool) and 0 < cfg <= CFG_MAX and abs(cfg - 1.0) > 1e-6:
        out["cfg"] = round(float(cfg), 2)
    return out


_OLD_LORA_KEYS = ("lora", "strength", "extra_loras")


AI, VIDEO = "ai", "video"


GUIDE_CHOICES = (5, 22, 39)        # motion guide lengths on H3's 17n+5 grid


# lengths of the clip before's end shown as a reference video ("continue"): MiniMaxH3ReferenceToVideo keeps the
# first 17n+5 frames of a reference video (at most the target's frame count), so only these lengths arrive whole
CONTINUE_CHOICES = (22, 39, 56, 73, 90, 107, 124)


DEFAULT_FRAMES = 73                 # a clip's length when nothing else is set


DEFAULT_OPTIONS: Options = {"model": "Comfy-Org/MiniMax-H3", "frames": DEFAULT_FRAMES, "steps": 28,
                   "resolution": 512, "audio": True, "keep_repeated": False, "seed": 42,
                   # a join after a carry-over: the cut follows the model's copy, its colour matched, blended
                   "smooth_joins": True,
                   # (no "shape": the clips take the start image's / first video's; without either 16:9, or the
                   # "aspect" an older version stored for such a project)
                   # turbo: a step-distilled LoRA + its sampler, few steps for every clip (ComfyUI only)
                   "turbo": {"on": False, "lora": None, "steps": 6, "strength": 1.0},
                   # HyperFlow: Video Rebirth's 8-step LoRA with its own sigma grid (ComfyUI-HyperFlow-H3 nodes)
                   "hyperflow": {"on": False, "strength": 1.0},
                   # Pika H3 Flex: Pika's post-trained H3 as a LoRA, 6 steps (ComfyUI template recipe)
                   "pika": {"on": False, "strength": 1.0},
                   # Veda sparse attention: used automatically for clips large enough to gain (chain.veda_useful)
                   "veda": {"on": True},
                   # the cast: characters with face pictures (copies in refs/); a prompt mentions them as @Name and
                   # the clip is made with exactly their pictures. A clip that mentions nobody gets the lead's.
                   # [{"name", "images": [rel], "desc", "lead": bool, "voice": {"file", "name", "secs", "text",
                   #   "ref": {"file", "secs"} (a longer part of the source, for text to speech to clone)}
                   #   (optional), "tts": {"eleven": {"id", "name"}} (optional: its ElevenLabs voice)}]
                   "cast": [],
                   # how the face pictures are used: their size for the model, and the sentence tying a person to
                   # their picture, the language of spoken lines (dialogue tags for voices)
                   "identity": {"size": "match", "hint": True, "language": "English"},
                   # sound references (WAV copies in refs/): <Audio 1> ... in a clip's prompt uses them (a voice,
                   # a soundtrack to copy, a beat); only the clips that name one are made with them
                   "audio_refs": [],
                   # style text put before / after every prompt (e.g. look, camera, a LoRA's trigger word)
                   "style": {"prefix": "", "suffix": ""},
                   # the trigger words of a clip's LoRAs (triggers.py: from the files and CivitAI) before its prompt
                   "lora_triggers": True,
                   # how the model prompt is written: "h3" (H3's official formats, h3_prompt.compose) or "classic"
                   # (the app's own sentences, as before). New projects: h3; a file without it is classic (from_json)
                   "prompt_format": "h3"}


PROMPT_FORMATS = ("h3", "classic")


CAST_KINDS = ("person", "place", "object", "style")   # what a cast entry is (no "kind": a person)


REFS_DIR = "refs"


ENDS_DIR = "ends"


STARTS_DIR = "starts"              # first clips' own start pictures (Node.start_image)


MAX_CAST_PICTURES = 9              # MiniMax H3 ref2va takes up to 9 reference images


MAX_PICTURES_PER_CHARACTER = 3


PICTURE_ROLES = ("head_sheet", "body_sheet")   # what a person's picture is (none: a plain picture, as before)
SHEET_SOURCE = "src"                            # refs/src<k>.png: the photo a person's sheets were made from


MAX_AUDIO_REFS = 3                 # MiniMax H3 ref2va: up to 3 sounds, each 2-15 s, together at most 15 s


AUDIO_REF_MIN, AUDIO_REF_TOTAL = 2.0, 15.0


AUDIO_TAG = re.compile(r"<Audio\s*(\d+)>", re.IGNORECASE)


MENTION = re.compile(r"(?<![\w@])@(\w[\w-]*)")        # @Anna in a prompt


NAME_OK = re.compile(r"^[^\W\d_][\w-]*$")              # a character name: a letter first, no spaces


PICTURE_TAG = re.compile(r"<Picture\s*\d+>", re.IGNORECASE)


# a spoken line: "...", “...”, „...“, «...» or MiniMax's own <d>[Language] ...</d>
DIALOGUE = re.compile(r'<d>\s*(?:\[[^\]\n]*\]\s*)?(.*?)</d>|"([^"\n]+)"|“([^”\n]+)”|„([^“”\n]+)[“”]|«([^»\n]+)»',
                      re.DOTALL)


# how a line is spoken, in square brackets inside it: "[angry] Lass mich!", "Ja [laughs] genau". Text to speech that
# hears them gets them (tts.spoken_text); the clip model never does (clip_cast takes them out of the lines).
SPEECH_TAG = re.compile(r"\[[^\[\]\n]{1,80}\]")


SAID_BY = re.compile(r"[ \t]*,?[ \t]*(says|said|asks|asked|answers|answered|replies|replied|whispers|whispered|"
                     r"shouts|shouted|calls|called|adds|added|cries|cried|murmurs|murmured|mutters|muttered|laughs|"
                     r"laughed|sagt|fragt|antwortet|ruft|flüstert)[ \t]+@(\w[\w-]*)",
                     re.IGNORECASE)   # "...", says @Anna


VOICE_MAX_SECS = 7.5               # a character's voice is cut to this (two voices fit the 15 s a clip may have)


VOICE_REF_MAX_SECS = 30.0          # ... and text to speech clones this much of a longer source (voice["ref"])


WRONG_TAG = re.compile(r"<(Image|Img|Face|Person|Photo|Pic|Ref|Reference)\s*(\d+)>", re.IGNORECASE)


SEED_MAX = 2_000_000_000            # (new random seeds come from 1 … this)
SEED_LIMIT = 2**64 - 1             # the largest seed a clip can have (what ComfyUI's noise takes)


REFINED = ("1080p", "2k")          # refined copies of a clip version for export: nodes/<id>_<tag>_<size>.mp4


# a clip file of this app (also the fixed names of older versions)
_CLIP_NAME = re.compile(r"^n\d+(_[0-9a-f]{8})?((_first|_f\d+)?\.(mp4|png)|_(1080p|2k)\.(mp4|json)|\.own\.(m4a|mka))$")


# the files of a clip version its record names (own_sound: the model's sound of a clip with a sound laid over, kept
# for remix_sound); they travel together: kept aside, made again, removed as one
CLIP_KEYS = ("video", "frame", "first", "own_sound")


# another H3 checkpoint (options["checkpoint"]: {"name": ComfyUI's name, "distilled": bool}; no name: the standard
# files); a distilled one (its turbo built in) makes every clip in this many steps, without a speed mode or Veda
DISTILLED_STEPS = 8
# two passes (options["two_pass"]["on"]): pass 1 at this share of the size, pass 2 this share of the schedule at the
# full size (comfy.TWO_PASS); what a clip records
TWO_PASS = {"scale": 0.5, "denoise": 0.3}


def checkpoint_of(o: Any) -> dict | None:
    """The H3 checkpoint the options choose ({"name", "distilled"}), None for the standard model."""
    c = o.get("checkpoint") if hasattr(o, "get") else None
    name = str(c.get("name") or "").strip() if isinstance(c, dict) else ""
    return {"name": name, "distilled": bool(c.get("distilled"))} if name and isinstance(c, dict) else None


SPEED_MODES = ("normal", "turbo", "hyperflow", "pika")      # the speed modes; each but "normal" has its own option dict


def speed_mode(o: Any) -> str:
    """Which speed mode the options switch on: "turbo", "hyperflow", "pika" or "normal"."""
    for mode in SPEED_MODES[1:]:
        if (o.get(mode) or {}).get("on"):
            return mode
    return "normal"


def speed_mode_used(o: Any) -> str:
    """The speed mode clips are made with: "normal" while a distilled checkpoint is chosen (its turbo is built in;
    the stored mode stays for when it is not), else speed_mode."""
    ck = checkpoint_of(o)
    return "normal" if ck and ck["distilled"] else speed_mode(o)


def set_speed_mode(o: Any, mode: str) -> None:
    """Switch exactly one speed mode on (or none, for "normal"); the other settings of each mode stay."""
    if mode not in SPEED_MODES:
        raise ValueError(f"unknown speed mode {mode!r}")
    for m in SPEED_MODES[1:]:
        o[m] = {**(o.get(m) or dict(DEFAULT_OPTIONS[m])), "on": m == mode}   # type: ignore[literal-required]
