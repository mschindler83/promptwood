"""Names and limits of the project file and its data (no code)."""
from __future__ import annotations

import re



TREE_FILE = "tree.json"


TREE_VERSION = 3


class NewerTreeError(ValueError):
    """tree.json was written by a newer app: not opened (and not replaced by the backup)."""


START_FILE = "start.png"


NO_LORA = "none"                   # old override value: this node without any LoRA


OVERRIDE_KEYS = ("loras", "frames", "steps", "guide", "source_sound", "sound_over", "sound_mix_level", "camera",
                 "lead")


_OLD_LORA_KEYS = ("lora", "strength", "extra_loras")


AI, VIDEO = "ai", "video"


GUIDE_CHOICES = (5, 22, 39)        # motion guide lengths on H3's 17n+5 grid


DEFAULT_FRAMES = 73                 # a clip's length when nothing else is set


DEFAULT_OPTIONS = {"model": "Comfy-Org/MiniMax-H3", "frames": DEFAULT_FRAMES, "steps": 28,
                   "resolution": 512, "audio": True, "keep_repeated": False, "seed": 42,
                   # the aspect ratio of a project without a start image or video (its first clip starts from text)
                   "aspect": "16:9",
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


MAX_AUDIO_REFS = 3                 # MiniMax H3 ref2va: up to 3 sounds, each 2-15 s, together at most 15 s


AUDIO_REF_MIN, AUDIO_REF_TOTAL = 2.0, 15.0


AUDIO_TAG = re.compile(r"<Audio\s*(\d+)>", re.IGNORECASE)


MENTION = re.compile(r"(?<![\w@])@(\w[\w-]*)")        # @Anna in a prompt


NAME_OK = re.compile(r"^[^\W\d_][\w-]*$")              # a character name: a letter first, no spaces


PICTURE_TAG = re.compile(r"<Picture\s*\d+>", re.IGNORECASE)


# a spoken line: "...", “...”, „...“, «...» or MiniMax's own <d>[Language] ...</d>
DIALOGUE = re.compile(r'<d>\s*(?:\[[^\]\n]*\]\s*)?(.*?)</d>|"([^"\n]+)"|“([^”\n]+)”|„([^“”\n]+)[“”]|«([^»\n]+)»',
                      re.DOTALL)


SAID_BY = re.compile(r"[ \t]*,?[ \t]*(says|said|asks|asked|answers|answered|replies|replied|whispers|whispered|"
                     r"shouts|shouted|calls|called|adds|added|cries|cried|murmurs|murmured|mutters|muttered|laughs|"
                     r"laughed|sagt|fragt|antwortet|ruft|flüstert)[ \t]+@(\w[\w-]*)", re.IGNORECASE)   # "...", says @Anna


VOICE_MAX_SECS = 7.5               # a character's voice is cut to this (two voices fit the 15 s a clip may have)


VOICE_REF_MAX_SECS = 30.0          # ... and text to speech clones this much of a longer source (voice["ref"])


WRONG_TAG = re.compile(r"<(Image|Img|Face|Person|Photo|Pic|Ref|Reference)\s*(\d+)>", re.IGNORECASE)


SEED_MAX = 2_000_000_000            # (the seed field's range)


REFINED = ("1080p", "2k")          # refined copies of a clip version for export: nodes/<id>_<tag>_<size>.mp4


# a clip file of this app (also the fixed names of older versions)
_CLIP_NAME = re.compile(r"^n\d+(_[0-9a-f]{8})?((_first|_f\d+)?\.(mp4|png)|_(1080p|2k)\.(mp4|json)|\.own\.(m4a|mka))$")


# the files of a clip version its record names (own_sound: the model's sound of a clip with a sound laid over, kept
# for remix_sound); they travel together: kept aside, made again, removed as one
CLIP_KEYS = ("video", "frame", "first", "own_sound")
