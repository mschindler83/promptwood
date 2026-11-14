"""Pure helpers and constants of the main window (no Qt widgets): texts, number formats, calculations on options."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple, cast

from .. import chain, config, llm_server as LLM, tree as T, tts as TTS
from .graph import START
from .lora_stack import LORA_MAX, LORA_MIN


DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_COMFY_LAUNCHER = str(Path.home() / "bin" / "comfyui")
DEFAULT_COMFY_DIR = str(Path.home() / "ComfyUI")
LLM_DEFAULT_PORT, LLM_DEFAULT_SERVER = LLM.DEFAULT_PORT, LLM.DEFAULT_SERVER
LLM_MODEL_DIRS = (config.hf_cache(), Path.home() / ".cache" / "huggingface" / "hub")
CLOSE_GRACE = 3.0          # seconds llama-server gets to end after SIGTERM when the window closes
CHAT_HINTS = ("instruct", "chat", "-it-", "-it.", "_it_", "abliterated", "uncensored")
MAX_RECENT = 10
UNLOAD_TIMEOUT = 10        # seconds ComfyUI is given to answer "unload your models"
VRAM_POLL_SECONDS = 0.5    # how often the free VRAM is looked at while ComfyUI lets go
AUDIO_REF_RANGE = f"{T.AUDIO_REF_MIN:g}-{T.AUDIO_REF_TOTAL:g} s"   # what one sound reference may last
_SELECTED: Any = object()         # "the selected character" as a default argument (None means: no character)
MAX_RECENT_LORAS = 12      # LoRAs kept in the "recent" list of the pickers
COMFY_START_TIMEOUT = 180  # seconds ComfyUI is given to come up after it was started
KILL_GRACE_MS = 8000       # how long the generator process may take to end after Stop
CLOUD_KILL_GRACE_MS = 60000  # the same for a cloud worker: it ends its pod first
QWIDGETSIZE_MAX = 16777215   # Qt's "no maximum width"
CARD_MIN_WIDTH = 300         # the narrowest an inspector card is made
THUMB_CACHE_MAX = 400        # timeline thumbnails kept
TIMELINE_THUMB_HEIGHT = 48
DEFAULT_SIZE = (672, 384)    # a clip's size when nothing says another (width, height)
ASPECT_TOLERANCE = 0.05      # an end image this far from the canvas' aspect ratio is cut to it
# H3's 17n+5 frame grid from 3 s to 15 s (362 frames = 15.1 s, the top of its trained range of ~124-362)
FRAME_CHOICES = [17 * n + 5 for n in range(4, 22)]
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"


def app_cache_dir() -> Path:
    """The folder of this app's logs and caches (the llama.cpp server's log, the speech takes' scratch files)."""
    return Path.home() / ".cache" / "video-chain"


def find_gguf_models(roots=LLM_MODEL_DIRS) -> list[Path]:
    """GGUF chat language models in the model folders (Hugging Face cache layout: the snapshots' links): those
    with a vision projector next to them or a chat / instruct name; not the projectors, imatrix files, or GGUF
    image / video models and text encoders."""
    out: list[Path] = []
    for root in roots:
        try:
            by_folder: dict[Path, list[Path]] = {}              # (one look at each snapshot folder, not 3 per file)
            for f in sorted(Path(root).glob("models--*/snapshots/*/*.gguf")):
                by_folder.setdefault(f.parent, []).append(f)
            for files in by_folder.values():
                has_mmproj = any("mmproj" in f.name.lower() for f in files)
                for f in files:
                    low = f.name.lower()
                    if "mmproj" in low or "imatrix" in low or not f.exists():
                        continue
                    if has_mmproj or any(h in low for h in CHAT_HINTS):
                        out.append(f)
        except OSError:
            continue
    return out


def frames_text(n: int) -> str:
    return f"{n / chain.FPS:.1f} s ({n} frames)"


STATE_TEXT = {"done": "DONE", "reused": "DONE", "changed": "CHANGED", "new": "NEW", "generating": "GENERATING",
              "failed": "FAILED", "stopped": "STOPPED", "converting": "CONVERTING"}
STATE_TIP = {"done": "Made, and nothing it depends on changed: it is reused, not made again",
             "reused": "Made, and nothing it depends on changed: it is reused, not made again",
             "changed": "Something changed since it was made: it is made again with the path",
             "new": "Not made yet: Generate path makes it",
             "generating": "Being made now", "converting": "The video range is being converted",
             "failed": "Making it failed: press Details on the card", "stopped": "Stopped before it was finished"}
PIXMAP_CACHE_MAX = 200                           # pictures kept for the graph and lists
STATE_STYLE = {"done": "done", "reused": "done", "generating": "target", "failed": "failed", "stopped": "missing",
               "changed": "missing", "new": "", "converting": "target"}
SPEAK_TIP = ("<p>Speak this clip's lines (\"...\" after @Name, e.g. @Anna says \"Hello.\") with text to speech, "
             "each in its speaker's voice, and make them the clip's sound: the picture is made to it, lips in sync."
             "<br>A window opens: make a few takes, set how steady or lively and how fast the lines are spoken, "
             "listen, and use the best take. The words can be changed there too, to try other sentences."
             "<br>Who speaks (Qwen3-TTS on this computer, or ElevenLabs with your API key) is set in the Cast & "
             "references tab, under Text to speech; the API key in Settings.</p>")
MIX_TIP = ("<p>What you hear: the given sound exactly as it is, plus the sound the model makes for this clip "
           "(steps, voices, the sound of the room). Good when music or a background goes on and the clip should "
           "still sound alive, e.g. the real video's music under the new clip's footsteps.<br>The model does not hear "
           "the given sound, so its own sound may not follow its rhythm.<br>A made clip keeps the model's sound next "
           "to it, so switching this or changing the level only mixes its sound again (seconds, no new generation); "
           "a clip made before the app kept that sound is made again.</p>")
REMIX_DELAY_MS = 400    # a made clip's sound is mixed again this long after its mix level last changed
MIX_OFF_TIP = ("'Generate sound' is off in the project's clip settings, so the model makes no sound to mix with; the "
               "given sound is laid over alone.")
FREEZE_TIP = ("Freeze: keep this clip as it is made, and every clip before it. They count as done whatever "
              "the project's settings say (size, speed LoRA, steps …), are not made again and cannot be "
              "edited; the clips after it follow the settings as before. Click again to unfreeze")
GUIDE_TIP = ("How much this clip takes over from the clip before it (and from a real video after it). Last frame "
             "only: it starts at that frame. 5-39 frames: those frames, with their sound, are put into this clip, so "
             "motion and voice carry over; they are left out when the path is played or exported")
GUIDE_TEXT = [(None, "last frame only"), (5, "5 frames (0.2 s)"), (22, "22 frames (0.9 s)"), (39, "39 frames (1.6 s)")]
CONTINUE_TIP = ("Continue the clip before: the model is shown the end of the clip before (as <Video 1>) and the prompt "
                "tells it to continue it, so motion, camera and scene carry on. Carry-over is different: it fixes the "
                "first frames of this clip exactly; this shows the model the clip before as a reference. It uses the "
                "reference model, so the clip takes longer to make. New: whether it helps has not been tested here "
                "yet. (ComfyUI describes reference videos as 2-15 s long; shorter ones are taken too.)")
CONTINUE_CLASSIC_TIP = ("Continue the clip before needs the official prompt format (Project → Prompt format): only "
                        "that prompt can tell the model what the reference video is.")
CONTINUE_FIRST_TIP = "The first clip has no clip before it to continue."
CONTINUE_SHORT_TIP = ("The clip before or this clip is shorter than the shortest continuation (22 frames, 0.9 s): "
                      "there is nothing that fits.")


def continue_text(frames: int) -> str:
    """A continuation length in words: 'last 1.6 s (39 frames)'."""
    return f"last {frames / 24:.1f} s ({frames} frames)"


def fmt_left(seconds: float) -> str:
    """A remaining time in words: 'a few seconds', 'about 40 s', 'about 3 min', 'about 1 h 05 min'."""
    s = max(0.0, float(seconds))
    if s < 10:
        return "a few seconds"
    if s < 90:
        return f"about {int(round(s / 5) * 5)} s"
    m = int(round(s / 60))
    return f"about {m} min" if m < 60 else f"about {m // 60} h {m % 60:02d} min"


def plural(n: int, word: str, many: str | None = None) -> str:
    """"1 clip", "3 clips" (many: the plural when it is not word + "s")."""
    return f"{n} {word if n == 1 else many or word + 's'}"


def time_left(elapsed: float, frac: float) -> float | None:
    """The time still needed at this progress (None while too early to say)."""
    if frac < 0.03 or elapsed < 3:
        return None
    return elapsed * (1 - min(frac, 1.0)) / frac


def row_start(nid: str) -> str:
    """The id of the node-view node that shows a first clip's own start image."""
    return f"{START}:{nid}"


def row_start_of(gid) -> str | None:
    """The clip whose own start image a node-view node shows (None: another node)."""
    return gid[len(START) + 1:] if isinstance(gid, str) and gid.startswith(START + ":") else None


def variant_strengths(loras: list[dict], steps: list[float], k: int) -> list[dict]:
    """The LoRA stack of variant k (1, 2, ...): each strength moved by k times its step (within the LoRA range).
    steps go with the entries that have a path, in order; other entries are kept as they are."""
    out = copy.deepcopy(loras)
    for e, step in zip([e for e in out if e.get("path")], steps):
        if step:
            e["strength"] = round(min(LORA_MAX, max(LORA_MIN, float(e.get("strength", 1.0)) + k * step)), 4)
    return out


def repeated_variants(loras: list[dict], steps: list[float], count: int) -> list[int]:
    """Variants (1..count) whose stepped strengths equal the clip's own or an earlier variant's (the range limit
    reached): with the same seed they would be the same clip."""
    seen = [[float(e.get("strength", 1.0)) for e in loras if e.get("path")]]
    out = []
    for k in range(1, count + 1):
        st = [float(e.get("strength", 1.0)) for e in variant_strengths(loras, steps, k) if e.get("path")]
        if st in seen:
            out.append(k)
        seen.append(st)
    return out


class SourceSoundInfo(NamedTuple):
    """What the clip card's 'Sound' row shows about where the clip's sound comes from."""
    avail: bool          # a real video comes before the clip
    text: str
    max_guide: int       # the longest carry-over that fits
    can_repeat: bool     # a clip before it has a sound that can be repeated


def label_loras(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(path, name relative to the LoRA folder) -> (path, label): just the file name; the folder only where two
    LoRAs share a name."""
    count = Counter(Path(n).stem for _p, n in items)
    out = []
    for p, n in items:
        stem, folder = Path(n).stem, str(Path(n).parent)
        out.append((p, f"{stem} ({folder})" if count[stem] > 1 and folder != "." else stem))
    return out


END_TIP = ("End image: a picture this clip steers its end to (any image, or a frame of a clip in the project). "
           "The clip's last frame is what the model actually made; the next clip starts from that")
STEPS_TIP = ("Steps: how many times the model refines each clip (sampling steps). More is slower and rarely "
             "better; 20-30 is typical")
STEPS_UNSET = 3        # the per-clip steps box at its lowest value means "the project's steps" (nothing set)

LAYOUT_VERSION = 1
# preset: shares of the width (left panel, viewer, inspector) and of the centre's height for the viewer
LAYOUTS = {"edit": (0.20, 0.58, 0.22, 0.62), "write": (0.17, 0.51, 0.32, 0.55), "watch": (0.0, 0.84, 0.16, 0.70)}

# a clip's sound: (shown, override value)
SOUND_MODES = (("The model makes it", None), ("Continue the real video's sound", True),
               ("Repeat the sound of the clip before", "previous"))


def plain_problem(raw: str, nid: str) -> str:
    """A check's message about one clip as a sentence on that clip's card ("clip #3 is shorter …" -> "This clip is
    shorter …")."""
    num = T.short_id(nid)
    t = raw.replace(f"clip #{num}'s", "this clip's").replace(f" (clip #{num})", "").replace(f"clip #{num}", "this clip")
    t = " ".join(t.split())
    t = t[:1].upper() + t[1:]
    return t if t.endswith((".", "?", "!")) else t + "."


def speed_mode(o: dict) -> str:
    """Which speed mode the project's options switch on: "turbo", "hyperflow", "pika" or "normal"."""
    for mode in ("turbo", "hyperflow", "pika"):
        if (o.get(mode) or {}).get("on"):
            return mode
    return "normal"


def project_setting_changes(o: dict) -> list[tuple[str, str]]:
    """The project's settings (tree.options) that differ from a new project's (tree.DEFAULT_OPTIONS) and change how
    its clips are made: [(key, name shown)]. A setting that has no effect now (the steps in a speed mode with its
    own, the ElevenLabs model with local text to speech) is left out. ui keys, not option keys: see
    MainWindow._project_controls."""
    d, out = T.DEFAULT_OPTIONS, []
    mode = speed_mode(o)
    if mode == "normal" and int(o.get("steps", d["steps"])) != d["steps"]:
        out.append(("steps", "Steps"))
    if mode != "normal":
        out.append(("speed", "Speed"))
    if mode == "turbo":
        t, dt = o.get("turbo") or {}, d["turbo"]
        if int(t.get("steps", dt["steps"])) != dt["steps"] or float(t.get("strength", 1.0)) != dt["strength"]:
            out.append(("turbo", "Turbo steps / strength"))
    cfg = (o if mode == "normal" else (o.get(mode) or {})).get("cfg", 1.0)
    if abs(float(cfg) - 1.0) > 1e-6:
        out.append(("cfg", "CFG"))
    if bool((o.get("veda") or {"on": True}).get("on", True)) != d["veda"]["on"]:
        out.append(("veda", "Fast attention (Veda)"))
    if bool(o.get("keep_repeated", d["keep_repeated"])) != d["keep_repeated"]:
        out.append(("keep_repeated", "Keep repeated frames"))
    style = o.get("style") or {}
    for k, name in (("prefix", "Style before"), ("suffix", "Style after")):
        if (style.get(k) or "").strip():
            out.append(("style_" + k, name))
    if T.is_h3(o) != (d["prompt_format"] == "h3"):
        out.append(("prompt_format", "Prompt format"))
    if bool(o.get("lora_triggers", d["lora_triggers"])) != d["lora_triggers"]:
        out.append(("lora_triggers", "LoRA trigger words"))
    ident, di = o.get("identity") or {}, d["identity"]
    if ident.get("size", di["size"]) != di["size"]:
        out.append(("ref_size", "Reference size (Cast & references)"))
    if bool(ident.get("hint", di["hint"])) != di["hint"]:
        out.append(("ref_hint", "Tie each person to their face (Cast & references)"))
    topt = TTS.options(o)
    if topt["provider"] == "elevenlabs" and topt["eleven_model"] != TTS.DEFAULT_OPTIONS["eleven_model"]:
        out.append(("eleven_model", "ElevenLabs model (Cast & references)"))
    return out


def as_dict(typed: Any) -> dict[str, Any]:
    """A TypedDict (an Options, MadeRecord ...) as the plain dict it is, for keys that are variables."""
    return cast(dict[str, Any], typed)


CARD_WIDTH = 330                           # a card's width in a row; the header must fit it
