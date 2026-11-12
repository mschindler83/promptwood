"""Promptwood window: a start image and a tree of clips. Every clip continues from the last frame of the clip
before it; any clip can be continued in several ways (branches). A path from the start to a clip is one video.
Real videos can be put into the chain too (a range of a file): a generated clip before one ends at its first frame,
one after it starts from its last frame.

The project folder holds tree.json, start.png and the clips (see tree.py). Clips are rendered by ComfyUI; a small
worker process (python -m video_chain --serve) talks to it and reports '@@ {json}' events; Stop ends that process
and cancels the clip in ComfyUI.
"""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QInputDialog, QLabel,
                               QFrame, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from .. import chain
from .. import film as F
from .. import camera as CAM
from .. import swap
from .. import tree as T
from .. import tts as TTS
from ..comfy import REF_HINT
from .graph import START, GraphView, NodeInfo
from .image_picker import pick_image
from .clip_dialogs import CompareDialog, HandoverDialog, SplitDialog, VideoFrameDialog
from .keyframe_strip import KeyframeStrip
from .lora_stack import LORA_MAX, LORA_MIN, LoraStack, fill_lora_combo, lora_label, lora_path
from .swap_dialog import SwapDialog
from .mentions import MentionHelper
from .cast_strip import CastStrip
from .timeline import Block, Timeline
from .storyboard import Storyboard
from .face_crop import FaceCropDialog
from .prompt_refiner import RefineDialog, replace_text
from .tts_dialog import VoiceTtsDialog, run_dialog as run_voice_dialog
from .image_edit import EDITS_DIR, ImageEditDialog, project_edits, record_use, uses_of
from .video_editor import VideoRangeDialog, choose_video
from .theme import icon, repolish, theme
from .widgets import FrameTile, VideoView, guard_wheel, icon_button, label, section_bar

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
from ..llm_server import DEFAULT_PORT as LLM_DEFAULT_PORT, DEFAULT_SERVER as LLM_DEFAULT_SERVER  # noqa: E402
LLM_MODEL_DIRS = (Path("/path/to/ai/hf-models"), Path.home() / ".cache" / "huggingface" / "hub")


CHAT_HINTS = ("instruct", "chat", "-it-", "-it.", "_it_", "abliterated", "uncensored")


def find_gguf_models(roots=LLM_MODEL_DIRS) -> list[Path]:
    """GGUF chat language models in the model folders (Hugging Face cache layout: the snapshots' links): those
    with a vision projector next to them or a chat / instruct name; not the projectors, imatrix files, or GGUF
    image / video models and text encoders."""
    out: list[Path] = []
    for root in roots:
        try:
            for f in sorted(Path(root).glob("models--*/snapshots/*/*.gguf")):
                low = f.name.lower()
                if "mmproj" in low or "imatrix" in low or not f.exists():
                    continue
                if any(f.parent.glob("*mmproj*.gguf")) or any(h in low for h in CHAT_HINTS):
                    out.append(f)
        except OSError:
            continue
    return out
DEFAULT_COMFY_LAUNCHER = str(Path.home() / "bin" / "comfyui")
DEFAULT_COMFY_DIR = str(Path.home() / "ComfyUI")
# H3's 17n+5 frame grid from 3 s to 15 s (362 frames = 15.1 s, the top of its trained range of ~124-362)
FRAME_CHOICES = [17 * n + 5 for n in range(4, 22)]


def frames_text(n: int) -> str:
    return f"{n / 24:.1f} s ({n} frames)"


def select_frames(combo: QComboBox, frames: int | None) -> None:
    """Select a frame count; a saved one that is no longer offered is shown as it is, not replaced."""
    if frames is None:
        combo.setCurrentIndex(0)
        return
    i = combo.findData(int(frames))
    if i < 0:
        combo.addItem(frames_text(int(frames)) + " – no longer offered", int(frames))
        i = combo.count() - 1
    combo.setCurrentIndex(i)
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
STATE_TEXT = {"done": "DONE", "reused": "DONE", "changed": "CHANGED", "new": "NEW", "generating": "GENERATING",
              "failed": "FAILED", "stopped": "STOPPED", "converting": "CONVERTING"}
STATE_TIP = {"done": "Made, and nothing it depends on changed: it is reused, not made again",
             "reused": "Made, and nothing it depends on changed: it is reused, not made again",
             "changed": "Something changed since it was made: it is made again with the path",
             "new": "Not made yet: Generate path makes it",
             "generating": "Being made now", "converting": "The video range is being converted",
             "failed": "Making it failed (see the status bar)", "stopped": "Stopped before it was finished"}
STATE_STYLE = {"done": "done", "reused": "done", "generating": "target", "failed": "failed", "stopped": "missing",
               "changed": "missing", "new": "", "converting": "target"}
SPEAK_TIP = ("<p>Speak this clip's lines (\"...\" after @Name, e.g. @Anna says \"Hello.\") with text to speech, "
             "each in its speaker's voice, and make them the clip's sound: the picture is made to it, lips in sync."
             "<br>Who speaks (Qwen3-TTS on this computer, or ElevenLabs with your API key) is set in the Cast tab, "
             "under Text to speech; the API key in Settings.</p>")
FREEZE_TIP = ("Freeze: keep this clip as it is made, and every clip before it. They count as done whatever "
              "the project's settings say (size, speed LoRA, steps …), are not made again and cannot be "
              "edited; the clips after it follow the settings as before. Click again to unfreeze")
GUIDE_TIP = ("How much this clip takes over from the clip before it (and from a real video after it). Last frame "
             "only: it starts at that frame. 5-39 frames: those frames, with their sound, are put into this clip, so "
             "motion and voice carry over; they are left out when the path is played or exported")
GUIDE_TEXT = [(None, "last frame only"), (5, "5 frames (0.2 s)"), (22, "22 frames (0.9 s)"), (39, "39 frames (1.6 s)")]


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:04.1f}"


def fmt_left(seconds: float) -> str:
    """A remaining time in words: 'a few seconds', 'about 40 s', 'about 3 min', 'about 1 h 05 min'."""
    s = max(0.0, float(seconds))
    if s < 10:
        return "a few seconds"
    if s < 90:
        return f"about {int(round(s / 5) * 5)} s"
    m = int(round(s / 60))
    return f"about {m} min" if m < 60 else f"about {m // 60} h {m % 60:02d} min"


def time_left(elapsed: float, frac: float) -> float | None:
    """The time still needed at this progress (None while too early to say)."""
    if frac < 0.03 or elapsed < 3:
        return None
    return elapsed * (1 - min(frac, 1.0)) / frac


MAX_RECENT = 10


class NewProjectDialog(QDialog):
    """Name + location (+ an optional start image) for a new project."""

    def __init__(self, parent, location: str, image: str = ""):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(520)
        f = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. smoke-scene-1")
        f.addRow("Name", self.name)
        lrow = QHBoxLayout()
        self.location = QLineEdit(location)
        b = icon_button("fopen", "Choose the folder the project is created in")
        b.clicked.connect(self._choose_location)
        lrow.addWidget(self.location, 1)
        lrow.addWidget(b)
        f.addRow("Location", lrow)
        irow = QHBoxLayout()
        self.image = QLineEdit(image)
        self.image.setPlaceholderText("optional: start image")
        b2 = icon_button("fopen", "Choose the start image")
        b2.clicked.connect(self._choose_image)
        irow.addWidget(self.image, 1)
        irow.addWidget(b2)
        f.addRow("Start image", irow)
        self.target = label("", level="tertiary")
        self.target.setWordWrap(True)
        f.addRow("", self.target)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Create")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        f.addRow(self.buttons)
        for w in (self.name, self.location):
            w.textChanged.connect(self._check)
        self._check()

    def folder(self) -> Path | None:
        name = "".join(ch for ch in self.name.text().strip() if ch not in '/\\:*?"<>|').strip()
        loc = self.location.text().strip()
        return Path(os.path.expanduser(loc)) / name if name and loc else None

    def _check(self):
        f = self.folder()
        problem = ""
        if f is None:
            problem = "Enter a name and a location."
        elif (f / T.TREE_FILE).exists():
            problem = f"{f} is already a project (use Open)."
        elif f.exists() and any(f.iterdir()):
            problem = f"{f} exists and is not empty."
        self.target.setText(problem or f"Creates {f}")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not problem)

    def _choose_location(self):
        d = QFileDialog.getExistingDirectory(self, "Location", self.location.text().strip())
        if d:
            self.location.setText(d)

    def _choose_image(self):
        start = str(Path(self.image.text()).parent) if self.image.text().strip() else None
        f = pick_image(self, "Start image", getattr(self.parent(), "qs", None), start)
        if f:
            self.image.setText(f)


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


class VariantsDialog(QDialog):
    """How many variants of a clip, and whether to generate them right away. Advanced: move each LoRA's strength
    by a step per variant (a strength sweep), and keep the clip's seed so only the strengths differ."""

    def __init__(self, parent, node):
        super().__init__(parent)
        self.setWindowTitle("Create variants")
        self.stack = list((node.overrides or {}).get("loras") or [])
        self.loras = [e for e in self.stack if e.get("path")]
        f = QFormLayout(self)
        text = node.prompt if len(node.prompt) < 90 else node.prompt[:87] + "…"
        info = label(f"Clip #{node.id[1:]}: “{text}”\nEach variant is a new branch from the same point with the same "
                     "prompt and settings and another seed.", level="secondary")
        info.setWordWrap(True)
        f.addRow(info)
        self.count = QSpinBox()
        self.count.setRange(1, 20)
        self.count.setValue(3)
        f.addRow("Variants", self.count)
        self.generate = QCheckBox("Generate them now")
        self.generate.setChecked(True)
        f.addRow("", self.generate)

        self.advanced = QGroupBox("Advanced: change the LoRA strengths per variant")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        self.advanced.setToolTip("Variant 1 gets each strength plus its step, variant 2 plus twice the step … "
                                 "(a negative step lowers it), e.g. to find the strength that works best")
        av = QFormLayout(self.advanced)
        self.steps: list[QDoubleSpinBox] = []
        if not self.loras:
            av.addRow(label("This clip has no LoRAs.", level="tertiary"))
        for e in self.loras:
            spin = QDoubleSpinBox()
            spin.setRange(-5.0, 5.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setPrefix("± ")
            spin.setSuffix(" per variant")
            spin.setMinimumWidth(170)
            spin.setToolTip(f"Now {float(e.get('strength', 1.0)):g}; variant k gets {float(e.get('strength', 1.0)):g} "
                            "+ k × this step")
            spin.valueChanged.connect(self._preview)
            guard_wheel(spin)
            self.steps.append(spin)
            av.addRow(f"{lora_label(e['path'])[:32]} (×{float(e.get('strength', 1.0)):g})", spin)
        self.same_seed = QCheckBox("Keep the clip's seed (only the strengths differ)")
        self.same_seed.setToolTip("Off: every variant gets a new seed too. On: the same seed as the clip, so the "
                                  "variants differ only by their LoRA strengths")
        self.same_seed.setEnabled(bool(self.loras))
        self.same_seed.toggled.connect(self._preview)
        av.addRow("", self.same_seed)
        self.preview = label("", level="tertiary")
        self.preview.setWordWrap(True)
        av.addRow(self.preview)
        f.addRow(self.advanced)
        self.advanced.setEnabled(bool(self.loras))
        if not self.loras:
            self.advanced.setToolTip("This clip has no LoRAs of its own (add them in its settings first)")
        self.advanced.toggled.connect(self._preview)
        self.count.valueChanged.connect(self._preview)
        self._preview()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Create")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        f.addRow(buttons)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        self._preview()

    def strength_steps(self) -> list[float]:
        """The step per variant of each LoRA (all 0 when the advanced settings are off)."""
        on = self.advanced.isChecked()
        return [s.value() if on else 0.0 for s in self.steps]

    def keep_seed(self) -> bool:
        return self.advanced.isChecked() and self.same_seed.isChecked() and any(self.strength_steps())

    def _preview(self, *_):
        steps = self.strength_steps()
        if not self.advanced.isChecked() or not any(steps):
            self.preview.setText("")
            if hasattr(self, "ok"):
                self.ok.setEnabled(True)
            return
        rows = []
        for k in range(1, self.count.value() + 1):
            st = variant_strengths(self.loras, steps, k)
            rows.append(f"#{k}: " + ", ".join(f"×{float(e.get('strength', 1.0)):g}" for e, s in zip(st, steps) if s))
        text = "Strengths: " + " · ".join(rows)
        same = repeated_variants(self.loras, steps, self.count.value())
        if same:
            text += (f"\nVariant{'s' if len(same) > 1 else ''} {', '.join(f'#{k}' for k in same)} "
                     f"repeat{'s' if len(same) == 1 else ''} strengths already there (the range ends at "
                     f"{LORA_MIN:g}…{LORA_MAX:g})" + (": with the same seed that is the same clip again." if
                                                       self.same_seed.isChecked() else "."))
        self.preview.setText(text)
        if hasattr(self, "ok"):
            self.ok.setEnabled(not (same and self.same_seed.isChecked()))


class ExportDialog(QDialog):
    """Where to save a path, and at which size: as made (quick: the clips are joined), or at 1080p / 2K, upscaled in
    ComfyUI by SeedVR2 (restores the clips as made) or re-rendered by the Open-H3 refine (draws detail anew); every
    clip once, the copies are kept for the next export. ready(size, method, strength, denoise) -> True when every
    clip has a kept copy for that choice (a dict {size: bool} works too)."""

    METHODS = (("seedvr2_7b", "SeedVR2 7B — restores the clips as made, clean (default)"),
               ("seedvr2_3b", "SeedVR2 3B — crisper, a little harder"),
               ("refine", "Open-H3 refine — draws detail anew"))

    def __init__(self, parent, default: str, made_size: tuple[int, int] | None, refined,
                 swapped: int = 0, strength: str = "natural", method: str = "seedvr2_7b", denoise: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Export path")
        self.setMinimumWidth(620)
        self.ready = refined if callable(refined) else (lambda size, _m, _s, _d=False: bool(refined.get(size)))
        self.made_size = made_size
        f = QFormLayout(self)
        row = QHBoxLayout()
        self.file = QLineEdit(default)
        b = icon_button("fopen", "Choose where to save the video")
        b.clicked.connect(self._choose)
        row.addWidget(self.file, 1)
        row.addWidget(b)
        f.addRow("File", row)
        self.size = QComboBox()
        w, h = made_size or (0, 0)
        self.size.addItem(f"As made ({w}×{h}) — quick" if made_size else "As made — quick", "made")
        for key in ("1080p", "2k"):
            self.size.addItem(key, key)
        self.size.setToolTip("As made: the clips are joined as they are. 1080p / 2K: every clip of the path is "
                             "upscaled in ComfyUI (see Upscaler); the copies are kept, so the next export of the "
                             "same clips is quick. Real videos are converted again from their file at that size")
        f.addRow("Size", self.size)
        self.scale_note = label("", level="tertiary")
        self.scale_note.setWordWrap(True)
        f.addRow("", self.scale_note)
        self.method = QComboBox()
        for key, text in self.METHODS:
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(0, self.method.findData(method)))
        self.method.setToolTip(
            "SeedVR2 (video restoration, ComfyUI's own nodes): sharpens and cleans every frame of the clip as it was "
            "made — faces, eyes and colours stay. Measured on an RTX 4090 for a 1.6 s face clip at 1080p: about 33 s, "
            "it changed the clip least of all; 7B was the cleanest, 3B crisper but harder. Long clips are split in "
            "time to fit the GPU memory.\nOpen-H3 refine: re-renders each clip with its prompt and LoRAs after a "
            "latent upscale (3 steps); it invents more new detail but can change faces (see Refine). About 1.7 min "
            "(1080p) and 4-5 min (2K) per 3 s clip")
        f.addRow("Upscaler", self.method)
        self.denoise = QCheckBox("Light denoise before SeedVR2")
        self.denoise.setChecked(bool(denoise))
        self.denoise.setToolTip("Calms the draft's fine noise first (mostly the noise that changes from frame to frame, "
                                "little blur), so SeedVR2 does not sharpen it into grain or a painted texture on "
                                "smooth areas (walls, long hair, skin). Upscaled copies are kept per choice")
        f.addRow("", self.denoise)
        self.strength = QComboBox()
        for key, text in (("soft", "Soft — calm, little new detail"),
                          ("natural", "Natural — clean, keeps the person (default)"),
                          ("crisp", "Crisp — sharper hair and texture, busier")):
            self.strength.addItem(text, key)
        self.strength.setCurrentIndex(max(0, self.strength.findData(strength)))
        self.strength.setToolTip("How much the refine draws anew. Natural keeps faces, eyes and hair as made and "
                                 "adds clean detail; Crisp adds more texture (hair, knit) and some grain; Soft "
                                 "stays closest to the clip but is softer. (The release's own setting repainted "
                                 "faces: closed eyes drawn half open, eye and hair colour changed.) Refined copies "
                                 "are kept per strength")
        f.addRow("Refine", self.strength)
        self.fast = QCheckBox("Faster refine with Sol-Attn (about 2× faster at 2K, very slightly different)")
        self.fast.setToolTip("Block-sparse attention in the refine pass. Measured on an RTX 4090 at 2K: 120 s instead "
                             "of 267 s per 3 s clip, PSNR 33 dB against exact attention (side by side practically the "
                             "same; skin texture very slightly smoother). Off: exact attention")
        f.addRow("", self.fast)
        self.swapped = QCheckBox(f"Upscale the video{'s' if swapped != 1 else ''} with a replaced person too")
        self.swapped.setToolTip("A video whose person was replaced was made at the generation size. On: it is "
                                "upscaled like a generated clip (SeedVR2 restores it; the refine re-renders it with "
                                "the new person's face description), its recorded sound kept. Off: it is only "
                                "scaled up from its made size")
        self.swapped.setChecked(True)
        self.swapped.setVisible(bool(swapped))
        f.addRow("", self.swapped)
        note = label("Needs ComfyUI with SeedVR2 (its 7B / 3B model and VAE from Comfy-Org/SeedVR2) or, for the "
                     "refine, the H3 2K nodes, the latent upscaler and the refiner LoRA. Generating and the preview "
                     "stay at the normal size.", level="tertiary")
        note.setWordWrap(True)
        f.addRow("", note)
        for w_ in (self.size, self.method, self.strength):
            w_.currentIndexChanged.connect(lambda *_: self._update())
        self.denoise.toggled.connect(lambda *_: self._update())
        self._update()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        f.addRow(buttons)
        guard_wheel(self)

    def _ready(self, size: str) -> bool:
        how, st = self.method.currentData(), self.strength.currentData()
        return bool(self.ready(size, how, st, self.denoise.isChecked() and how != "refine"))

    def _update(self):
        up = self.size.currentData() != "made"
        refine = self.method.currentData() == "refine"
        self.method.setEnabled(up)
        self.strength.setEnabled(up and refine)
        self.fast.setEnabled(up and refine)
        self.denoise.setEnabled(up and not refine)
        self.swapped.setEnabled(up)
        w, h = self.made_size or (0, 0)
        from ..comfy import refine_size
        for i in range(1, self.size.count()):
            key = self.size.itemData(i)
            text = {"1080p": "1080p", "2k": "2K"}[key]
            if self.made_size:
                rw, rh = refine_size(w, h, key)
                text += f" — {rw}×{rh} · {rw / max(1, w):.1f}× the draft"
            if self._ready(key):
                text += " · already upscaled, quick"
            self.size.setItemText(i, text)
        size = self.size.currentData()
        factor = refine_size(w, h, size)[0] / w if self.made_size and size != "made" else 0.0
        calmer = self.made_size and size == "2k" and refine_size(w, h, "1080p")[0] / w <= 2.05
        self.scale_note.setText(
            f"{factor:.1f}× the draft: above about 2× SeedVR2 invents texture that shimmers in motion (measured: "
            "+29 % in a pan at 2.5× against 1.9×). " + ("1080p stays within 2× here. " if calmer else "")
            + "Larger drafts help: Clips → Size \u201c768p — H3's native size\u201d." if factor > 2.05 else "")
        self.scale_note.setVisible(factor > 2.05)

    def _choose(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export path", self.file.text(), "Video (*.mp4)")
        if f:
            self.file.setText(f)

    def target(self) -> Path:
        f = self.file.text().strip()
        return Path(os.path.expanduser(f if f.endswith(".mp4") else f + ".mp4"))


class _Bridge(QObject):
    joined = Signal(object, object, str, str, str)  # path or None, error or None, purpose, status prefix, project
    film = Signal(object, object, str)               # the joined film or None, error or None, 'play' / 'export'
    duplicated = Signal(object, object, str)         # (scene id, title) or None, error or None, film folder
    comfy = Signal(bool)                       # ComfyUI reachable
    loras = Signal(list, bool)                 # H3 LoRAs found in ComfyUI's LoRA folders, report them
    triggers = Signal(dict)                    # the trigger word scan finished (its counts)
    copied = Signal(str, str)                  # Save as: target folder, error ('' = done)
    converted = Signal(str, object, str, str, str)  # video node id, record or None, error, token, project
    llm_room = Signal(str)                     # ComfyUI's models were unloaded for the llama.cpp server (why)
    spoken = Signal(str, object, str, str)     # clip id, rendered lines (tts.render_lines) or None, error, project
    tts_log = Signal(str)                      # a step of text to speech
    eleven_models = Signal(object, str)        # ElevenLabs' TTS models or None, error
    eleven_cloned = Signal(str, object, str, str)   # character, cloned voice (tts.clone_sample) or None, error, project


# ----------------------------------------------------------------------------- path card

END_TIP = ("End image: a picture this clip steers its end to (any image, or a frame of a clip in the project). "
           "The clip's last frame is what the model actually made; the next clip starts from that")
STEPS_TIP = "Sampling steps per clip: more is slower and rarely better; 20-30 is typical"
STEPS_DEFAULT = 3


LAYOUT_VERSION = 1
# preset: shares of the width (left panel, viewer, inspector) and of the centre's height for the viewer
LAYOUTS = {"edit": (0.20, 0.58, 0.22, 0.62), "write": (0.17, 0.51, 0.32, 0.55), "watch": (0.0, 0.84, 0.16, 0.70)}


# a clip's sound: (shown, override value)
SOUND_MODES = (("The model makes it", None), ("Continue the real video's sound", True),
               ("Repeat the sound of the clip before", "previous"))


class _CurrentStack(QStackedWidget):
    """A stack whose size is that of the page shown (not of its largest hidden page)."""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class PromptCard(QFrame):
    """One clip of the selected path: prompt, seed, LoRAs, own length / steps, status."""
    settings_open = False                      # the own-settings part is open (the same on every card)
    edited = Signal(str)
    seedChanged = Signal(str, int)
    overridesChanged = Signal(str, dict)
    chooseLora = Signal(str)
    lora_picked = Signal(str, str)             # node id, LoRA path chosen in the picker
    action = Signal(str, str)                  # action, node id
    keyframe = Signal(str, str, object)        # keyframe action, node id, {"index", "at"}
    play = Signal(str)

    def __init__(self, node: T.Node, parent=None, names=lambda: []):
        super().__init__(parent)
        self.nid = node.id
        self.setObjectName("promptCard")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setFixedWidth(330)
        self.state = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 8, 10)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = label("Clip", type_="heading")
        self.sub = label(f"#{node.id[1:]}", level="tertiary")
        self.tag = label("", role="statusTag")
        head.addWidget(self.title)
        head.addWidget(self.sub)
        head.addWidget(self.tag)
        head.addStretch(1)
        self.buttons = []
        for act, ic, tip in (("variant", "clone", "Variant: same prompt, other seed, as a new branch (Ctrl+D)"),
                             ("variants", "layers", "Variants…: several new branches with the same prompt and "
                                                    "settings but other seeds, generated at once"),
                             ("compare", "cmp", "Compare: play this clip and its made variants side by side, keep one"),
                             ("branch", "split", "Branch: another continuation from the same point (Ctrl+B)"),
                             ("delete", "trash", "Delete this clip and everything after it")):
            b = icon_button(ic, tip)
            b.clicked.connect(lambda _=False, a=act: self.action.emit(a, self.nid))
            head.addWidget(b)
            self.buttons.append(b)
            if act == "compare":
                self.b_compare = b
        self._can_compare = False
        self.frozen, self._on = False, True
        self.has_lines = False                   # its prompt has a spoken line (text to speech can speak it)
        self.b_freeze = icon_button("lock", FREEZE_TIP, checkable=True)
        self.b_freeze.clicked.connect(lambda _=False: self.action.emit("freeze_toggle", self.nid))
        head.insertWidget(head.count() - 1, self.b_freeze)
        v.addLayout(head)
        self.span = label("", level="tertiary")
        v.addWidget(self.span)
        self.thumb = FrameTile(312, 176)
        self.thumb.setToolTip("Last frame of this clip (or its hand-over frame); click to play the clip")
        self.thumb.clicked.connect(lambda _t: self.play.emit(self.nid))
        v.addWidget(self.thumb)
        self.edit = QPlainTextEdit(node.prompt)
        self.edit.setPlaceholderText("What happens in this clip? It starts from the last frame of the clip before "
                                     "(and ends at the first frame of a real video after it).")
        self.edit.setTabChangesFocus(True)
        self.edit.setMinimumHeight(96)
        v.addWidget(self.edit, 1)
        self.mentions = MentionHelper(self.edit, names)
        self.faces = label("", level="tertiary")
        self.faces.setWordWrap(True)
        self.faces.hide()
        v.addWidget(self.faces)
        foot = QHBoxLayout()
        self.words = label("", level="tertiary")
        foot.addWidget(self.words)
        self.b_refine = icon_button("gen", "Refine this prompt with a language model (motion, camera, sound)")
        self.b_refine.clicked.connect(lambda: self.action.emit("refine", self.nid))
        foot.addWidget(self.b_refine)
        foot.addStretch(1)
        self.veda_mark = label("⚡ Veda", level="tertiary")
        self.veda_mark.setToolTip("This clip is large enough for Veda sparse attention: it is made with it (much "
                                  "faster for long / large clips, very slightly different from full attention). "
                                  "Switch it off in the project settings: “Fast attention (Veda)”")
        self.veda_mark.hide()
        foot.addWidget(self.veda_mark)
        foot.addWidget(label("Seed", level="tertiary"))
        self.seed = QSpinBox()
        self.seed.setRange(0, T.SEED_MAX)
        self.seed.setValue(node.seed)
        self.seed.setToolTip("Seed of this clip; another seed gives another version")
        foot.addWidget(self.seed)
        self.new_seed = None
        self.b_seed = icon_button("rotate", "New random seed for this clip (another version when it is made again)")
        self.b_seed.clicked.connect(self.random_seed)
        foot.addWidget(self.b_seed)
        v.addLayout(foot)

        # own settings of this clip; "default" = the project's
        self.ov = dict(node.overrides or {})
        self.steps_locked = False                 # turbo mode: the turbo steps apply to every clip
        self.settings_toggle = QToolButton()
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setAutoRaise(True)
        self.settings_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.settings_toggle.setToolTip("This clip's own length and steps (unset: the project's)")
        self.settings_toggle.toggled.connect(self._show_settings)
        v.addWidget(self.settings_toggle)
        self.own = QWidget()
        grid = QFormLayout(self.own)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.loras = LoraStack(self)
        self.loras.setToolTip("The LoRAs of this clip and their strengths (none: the base model)")
        self.loras.picked.connect(lambda p: self.lora_picked.emit(self.nid, p))
        self.length = QComboBox()
        self.length.setToolTip("Length of this clip")
        self.steps = QSpinBox()
        self.steps.setRange(STEPS_DEFAULT, 80)
        self.steps.setSpecialValueText("default")
        self.steps.setToolTip("Sampling steps of this clip (lowest value = the project's)")
        srow = QHBoxLayout()
        srow.addWidget(self.length, 1)
        srow.addWidget(label("Steps", level="tertiary"))
        srow.addWidget(self.steps)
        grid.addRow("Length", srow)
        v.addWidget(self.own)
        link = QFormLayout()
        link.setContentsMargins(0, 2, 0, 0)
        link.setHorizontalSpacing(8)
        link.setVerticalSpacing(4)
        link.addRow("LoRAs", self.loras)
        self.guide = QComboBox()
        for n, text in GUIDE_TEXT:
            self.guide.addItem(text, n)
        self.guide.setToolTip(GUIDE_TIP)
        link.addRow("Carry-over", self.guide)
        crow = QHBoxLayout()
        crow.setSpacing(6)
        self.cam_move = QComboBox()
        self.cam_move.addItem("none", None)
        for slug, (text, _w, _m) in CAM.MOVES.items():
            self.cam_move.addItem(text, slug)
        self.cam_move.setToolTip("A camera move for this clip, in the words of MiniMax H3's prompt guide: it is "
                                 "added as a sentence after the prompt (\u201cThe camera pushes in with small "
                                 "amplitude at slow speed.\u201d). For a move at a certain point, type # in the "
                                 "prompt instead (#push-in-slow \u2026)")
        self.cam_amp = QComboBox()
        for text, data in (("medium", None), ("small", "small"), ("large", "large")):
            self.cam_amp.addItem(text, data)
        self.cam_amp.setToolTip("How far the camera moves (amplitude); medium is left out of the prompt")
        self.cam_speed = QComboBox()
        for text, data in (("normal", None), ("slow", "slow"), ("fast", "fast")):
            self.cam_speed.addItem(text, data)
        self.cam_speed.setToolTip("How fast the camera moves; normal is left out of the prompt")
        crow.addWidget(self.cam_move, 2)
        crow.addWidget(self.cam_amp, 1)
        crow.addWidget(self.cam_speed, 1)
        link.addRow("Camera", crow)
        self.sound_box = QWidget()
        sb = QVBoxLayout(self.sound_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(2)
        self.src_sound = QComboBox()
        for text, data in SOUND_MODES:
            self.src_sound.addItem(text, data)
        self.src_sound.setToolTip(
            "<p><b>The model makes it</b>: the clip's sound is made with the picture.<br>"
            "<b>Continue the real video's sound</b>: the real video's own sound from right after its range (what was "
            "said next), exactly as recorded.<br><b>Repeat the sound of the clip before</b>: the clip before's sound "
            "once more (looped if this clip is longer), e.g. music or a background that goes on.<br>"
            "With either sound the picture is made to match it (lips in sync)</p>")
        self.src_sound_info = label("", level="tertiary")
        self.src_sound_info.setWordWrap(True)
        self.sound_over = QCheckBox("Lay the sound over (full motion, no lip sync)")
        self.sound_over.setToolTip(
            "<p><b>On</b>: the clip is made with the normal image-to-video model, as without a sound (it moves "
            "freely), and the exact sound is put into it afterwards. Nothing in the picture follows the sound: good "
            "for music, a background or a voice-over.<br><b>Off</b>: the reference model hears the sound and makes "
            "the picture to it (lips in sync), which often keeps the body still.</p>")
        sb.addWidget(self.src_sound)
        self.b_speak = QPushButton("Speak the lines (text to speech)")
        self.b_speak.setToolTip(SPEAK_TIP)
        self.b_speak.clicked.connect(lambda: self.action.emit("speak", self.nid))
        sb.addWidget(self.b_speak)
        sb.addWidget(self.sound_over)
        sb.addWidget(self.src_sound_info)
        link.addRow("Sound", self.sound_box)
        self._link = link
        link.setRowVisible(self.sound_box, True)           # (the sound choice in it only when there is one)
        self._max_guide = 0
        erow = QHBoxLayout()
        erow.setSpacing(6)
        self.end_thumb = QLabel()
        self.end_thumb.setFixedSize(48, 27)
        self.end_thumb.setScaledContents(True)
        self.end_text = label("", level="secondary")
        self.b_end = icon_button("target", END_TIP)
        self.b_end.clicked.connect(lambda: self.action.emit("end_image", self.nid))
        self.b_end_clear = icon_button("x", "No end image: the clip ends freely")
        self.b_end_clear.clicked.connect(lambda: self.action.emit("end_clear", self.nid))
        self.b_end_ai = icon_button("brush", "Make an end image with AI: edit this clip's frame (or any image) by a "
                                             "prompt, e.g. “she now sits on the sofa”")
        self.b_end_ai.clicked.connect(lambda: self.action.emit("end_ai", self.nid))
        self.b_end_video = icon_button("frame", "End image from a video: choose a project or any video, then the "
                                                "frame of it to end at")
        self.b_end_video.clicked.connect(lambda: self.action.emit("end_video", self.nid))
        erow.addWidget(self.end_thumb)
        erow.addWidget(self.end_text, 1)
        erow.addWidget(self.b_end)
        erow.addWidget(self.b_end_video)
        erow.addWidget(self.b_end_ai)
        erow.addWidget(self.b_end_clear)
        link.addRow("End", erow)
        kbox = QVBoxLayout()
        kbox.setSpacing(2)
        krow = QHBoxLayout()
        krow.setSpacing(6)
        self.keys = KeyframeStrip()
        self.keys.addAt.connect(self._add_key_menu)
        self.keys.moved.connect(lambda i, at: self.keyframe.emit("move", self.nid, {"index": i, "at": at}))
        self.keys.wanted.connect(lambda act, i: self.keyframe.emit(act, self.nid, {"index": i}))
        self.b_key = icon_button("plus", "Add a middle keyframe: a picture the clip passes through at a moment "
                                         "(or click the strip where it should be)")
        self.b_key.clicked.connect(lambda: self._add_key_menu(None))
        krow.addWidget(self.keys, 1)
        krow.addWidget(self.b_key)
        kbox.addLayout(krow)
        self.key_warn = label("", level="secondary")
        self.key_warn.setWordWrap(True)
        self.key_warn.hide()
        kbox.addWidget(self.key_warn)
        link.addRow("Keyframes", kbox)
        hrow = QHBoxLayout()
        hrow.setSpacing(6)
        self.hand_text = label("last frame", level="secondary")
        self.b_hand = icon_button("keyf", "Hand over at an earlier frame: the clips after this one start from the "
                                          "frame you choose; the rest of this clip is left out")
        self.b_hand.clicked.connect(lambda: self.action.emit("handover", self.nid))
        self.b_hand_clear = icon_button("x", "Hand over at the last frame again")
        self.b_hand_clear.clicked.connect(lambda: self.action.emit("handover_clear", self.nid))
        hrow.addWidget(self.hand_text, 1)
        hrow.addWidget(self.b_hand)
        hrow.addWidget(self.b_hand_clear)
        link.addRow("Hand-over", hrow)
        self._can_hand, self._has_hand = False, False
        self._end_free = True                    # no video after this clip: its own end image can be set
        self._has_end = False
        self._guide_useful = True                # there is a clip before it or a video after it
        v.addLayout(link)
        self._show_settings(PromptCard.settings_open)
        self.guide.currentIndexChanged.connect(self._ov_changed)
        for w_ in (self.cam_move, self.cam_amp, self.cam_speed):
            w_.currentIndexChanged.connect(self._ov_changed)
        self.src_sound.currentIndexChanged.connect(self._sound_toggled)
        self.sound_over.toggled.connect(self._sound_toggled)
        self.loras.changed.connect(self._ov_changed)
        self.length.currentIndexChanged.connect(self._ov_changed)
        self.steps.valueChanged.connect(self._ov_changed)

        strow = QHBoxLayout()
        self.step = label("", level="tertiary")
        self.step.setWordWrap(True)
        self.b_revert = QPushButton("Undo changes")
        self.b_revert.clicked.connect(lambda: self.action.emit("revert", self.nid))
        self.b_revert.hide()
        strow.addWidget(self.step, 1)
        strow.addWidget(self.b_revert, 0, Qt.AlignTop)
        v.addLayout(strow)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.hide()
        v.addWidget(self.bar)
        self.edit.textChanged.connect(self._text)
        self.seed.valueChanged.connect(lambda s: self.seedChanged.emit(self.nid, s))
        self.guide.currentIndexChanged.connect(lambda *_: self._enable_guide())
        guard_wheel(self)
        self._words()
        self._restyle()

    def set_choices(self, project_frames: int, project_steps: int, known: list[tuple[str, str]]):
        """Fill the setting widgets (without emitting changes). known: (path, label) of the LoRAs to offer."""
        widgets = (self.loras, self.length, self.steps, self.guide, self.src_sound, self.sound_over, self.cam_move,
                   self.cam_amp, self.cam_speed)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.loras.set_items(known, self.ov.get("loras") or [])
            self.length.clear()
            self.length.addItem(f"default ({project_frames / 24:.1f} s)", None)
            for n in FRAME_CHOICES:
                self.length.addItem(frames_text(n), n)
            select_frames(self.length, self.ov.get("frames"))
            self.steps.setValue(int(self.ov.get("steps", STEPS_DEFAULT)))
            g = self.guide.findData(self.ov.get("guide"))
            if g < 0:
                self.guide.addItem(f"{self.ov['guide']} frames", self.ov["guide"])
                g = self.guide.count() - 1
            self.guide.setCurrentIndex(g)
            k = self.src_sound.findData(self.ov.get("source_sound") or None)
            self.src_sound.setCurrentIndex(max(k, 0))
            self.sound_over.setChecked(bool(self.ov.get("sound_over")))
            cam = CAM.clean(self.ov.get("camera")) or {}
            self.cam_move.setCurrentIndex(max(0, self.cam_move.findData(cam.get("move"))))
            self.cam_amp.setCurrentIndex(max(0, self.cam_amp.findData(cam.get("amplitude"))))
            self.cam_speed.setCurrentIndex(max(0, self.cam_speed.findData(cam.get("speed"))))
            self._enable_camera()
            self._enable_guide()
            self.steps.setToolTip(f"Sampling steps of this clip (lowest value = the project's: {project_steps})")
            self.steps.setEnabled(not self.edit.isReadOnly() and not self.steps_locked)
            if self.steps_locked:
                self.steps.setToolTip("Turbo / HyperFlow / Pika H3 Flex: every clip uses their own steps (Clips → Speed)")
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.loras.setEnabled(not self.edit.isReadOnly())

    def _ov_changed(self, *_):
        ov = {}
        loras = self.loras.items()
        if loras:
            ov["loras"] = loras
        if self.length.currentData() is not None:
            ov["frames"] = self.length.currentData()
        if self.steps.value() > STEPS_DEFAULT:
            ov["steps"] = self.steps.value()
        if self.guide.currentData():
            ov["guide"] = self.guide.currentData()
        if self.src_sound.currentData():
            ov["source_sound"] = self.src_sound.currentData()
            if self.sound_over.isChecked():
                ov["sound_over"] = True
        cam = CAM.clean({"move": self.cam_move.currentData(), "amplitude": self.cam_amp.currentData(),
                         "speed": self.cam_speed.currentData()})
        if cam:
            ov["camera"] = cam
        self._enable_camera()
        self.ov = ov
        self.overridesChanged.emit(self.nid, ov)

    def _enable_camera(self):
        """Amplitude and speed only for a move that takes them (not for none, static, shake, POV)."""
        move = self.cam_move.currentData()
        on = bool(move) and CAM.MOVES[move][2] and not self.edit.isReadOnly()
        self.cam_amp.setEnabled(on)
        self.cam_speed.setEnabled(on)
        self.cam_move.setEnabled(not self.edit.isReadOnly())

    def text(self) -> str:
        return " ".join(self.edit.toPlainText().split())

    def random_seed(self):
        """A new random seed in the field (new_seed(): one no other clip has, else any)."""
        self.seed.setValue(self.new_seed() if self.new_seed is not None else random.randint(1, T.SEED_MAX))

    def _text(self):
        self._words()
        self._enable_speak()
        self.edited.emit(self.nid)

    def _enable_speak(self):
        """"Speak the lines" is always on the card; it can be used once the prompt has a line in quotes."""
        self.has_lines = bool(T.spoken_lines(self.text()))
        self.b_speak.setEnabled(not self.edit.isReadOnly() and self.has_lines)
        self.b_speak.setToolTip(SPEAK_TIP if self.has_lines else
                                "Write what is said in quotes first, e.g. @Anna says \"Hello, nice to see you.\" "
                                "Then this speaks it with text to speech and makes it the clip's sound.")

    def _words(self):
        n = len(self.text().split())
        self.words.setText(f"{n} words" if n else "empty prompt")

    def set_state(self, state: str, pixmap: QPixmap | None, step: str = ""):
        self.state = state
        self.tag.setText(STATE_TEXT.get(state, state.upper()))
        self.tag.setProperty("state", STATE_STYLE.get(state, ""))
        repolish(self.tag)
        if pixmap is not None and not pixmap.isNull():
            self.thumb.set_image(pixmap.toImage())
        else:
            self.thumb.clear_frame("not made yet")
        self.step.setText(step)
        self.bar.setVisible(state == "generating")
        self._restyle()

    def set_progress(self, i: int, n: int, text: str | None = None):
        if n:
            self.bar.setValue(round(100 * i / n))
            self.step.setText(text if text is not None else f"step {i} / {n}")

    def set_editable(self, on: bool):
        """on: nothing runs (the card can be used); a frozen clip's own inputs stay locked anyway."""
        self._on = bool(on)
        edit = self._on and not self.frozen
        self.edit.setReadOnly(not edit)
        for w in (self.seed, self.b_seed, self.length, self.steps, self.loras, self.guide, self.src_sound,
                  self.sound_over, self.b_refine, self.b_speak):
            w.setEnabled(edit)
        self._enable_speak()
        for w in self.buttons:
            w.setEnabled(self._on)
        self.steps.setEnabled(edit and not self.steps_locked)
        self._enable_camera()
        self._enable_guide()
        self._enable_end()
        self._enable_hand()
        self.b_compare.setEnabled(self._on and self._can_compare)
        self.b_revert.setEnabled(edit)
        self.b_freeze.setEnabled(self._on and (self.frozen or self.state == "done"))

    def set_frozen(self, frozen: bool):
        """A frozen clip: its prompt, seed, settings, end, keyframes and hand-over are locked."""
        self.frozen = bool(frozen)
        self.b_freeze.setChecked(self.frozen)
        self.b_freeze.setToolTip(FREEZE_TIP if not self.frozen else
                                 "Frozen: kept as made, whatever the project's settings say. Click to unfreeze it "
                                 "(and the frozen clips after it): it can be edited and follows the settings again")
        self.set_editable(self._on)

    def _show_settings(self, on: bool):
        PromptCard.settings_open = bool(on)
        self.own.setVisible(bool(on))
        self.settings_toggle.blockSignals(True)
        self.settings_toggle.setChecked(bool(on))
        self.settings_toggle.blockSignals(False)
        self.settings_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def set_faces(self, text: str):
        self.faces.setText(text)
        self.faces.setVisible(bool(text))

    def set_veda(self, on: bool):
        self.veda_mark.setVisible(bool(on))

    def set_summary(self, text: str):
        """The own settings in a few words, shown on the (closed) settings toggle."""
        self.settings_toggle.setText("Own settings: " + (text or "project defaults"))

    def set_revert(self, edited: list[str]):
        """Offer Undo changes when the prompt / seed / settings / end image differ from the made clip."""
        self.b_revert.setVisible(bool(edited))
        self.b_revert.setEnabled(not self.edit.isReadOnly())
        if edited:
            self.b_revert.setToolTip(f"Back to the {' and '.join(edited)} this clip was made with, so it counts as "
                                     "done again and is not made again")

    def set_handover(self, text: str, can: bool, has: bool, tip: str = ""):
        """Where the clips after this one start: 'last frame' or an earlier frame. can: it can be chosen (made,
        no video after it); has: an earlier frame is set."""
        self.hand_text.setText(text)
        self.hand_text.setToolTip(tip)
        self._can_hand, self._has_hand = bool(can), bool(has)
        self._enable_hand()

    def _enable_hand(self):
        on = not self.edit.isReadOnly()
        self.b_hand.setEnabled(on and self._can_hand)
        self.b_hand_clear.setEnabled(on and self._has_hand)

    def set_compare(self, n: int):
        """n: made clips next to this one (its variants / branches) that can be compared with it."""
        self._can_compare = n > 0
        self.b_compare.setEnabled(self._can_compare and self._on)
        self.b_compare.setToolTip(f"Compare: play this clip and its {n} made variant{'s' if n != 1 else ''} side by "
                                  "side, keep one" if n else
                                  "Compare: needs this clip and at least one other made clip from the same point "
                                  "(Variant / Variants…)")

    def set_guide_useful(self, useful: bool):
        """Carry-over needs a clip before this one or a real video after it; a first clip without a video after it
        has nothing to carry over."""
        self._guide_useful = bool(useful)
        self._enable_guide()

    def _sound_toggled(self, *_):
        """With the video's sound the picture should continue too: no carry-over yet -> the longest one that fits
        (the clip before must be that long, and this clip longer)."""
        if not self.src_sound.currentData() and self.sound_over.isChecked():
            self.sound_over.blockSignals(True)         # (no sound: nothing to lay over; off when one is chosen again)
            self.sound_over.setChecked(False)
            self.sound_over.blockSignals(False)
        if self.src_sound.currentData() is True and not self.sound_over.isChecked() \
                and not self.guide.currentData() and self._max_guide:
            k = self.guide.findData(self._max_guide)
            if k >= 0:
                self.guide.blockSignals(True)
                self.guide.setCurrentIndex(k)
                self.guide.blockSignals(False)
                self._enable_guide()
        self._ov_changed()

    def set_tracks(self, tracks: list[tuple[str, str]]):
        """The project's audio tracks as choices of the Sound dropdown: (id, name)."""
        names = [name for _tid, name in tracks]
        want = [(f"Audio track: {name}" + (f" ({names[:k].count(name) + 1})" if names.count(name) > 1 else ""),
                 T.TRACK_PREFIX + tid) for k, (tid, name) in enumerate(tracks)]
        have = [(self.src_sound.itemText(k), self.src_sound.itemData(k)) for k in range(len(SOUND_MODES),
                                                                                       self.src_sound.count())]
        cur = (getattr(self, "ov", None) or {}).get("source_sound") or None     # (the clip's own setting)
        if isinstance(cur, str) and cur.startswith(T.TRACK_PREFIX) and cur not in [d for _t, d in want]:
            want.append(("Audio track: (removed)", cur))   # kept, so the setting shows and can be taken back
        if want == have:
            if self.src_sound.currentData() != cur and self.src_sound.findData(cur) >= 0:
                self.src_sound.blockSignals(True)
                self.src_sound.setCurrentIndex(self.src_sound.findData(cur))
                self.src_sound.blockSignals(False)
            return
        self.src_sound.blockSignals(True)
        try:
            while self.src_sound.count() > len(SOUND_MODES):
                self.src_sound.removeItem(len(SOUND_MODES))
            for text, data in want:
                self.src_sound.addItem(text, data)
            self.src_sound.setCurrentIndex(max(0, self.src_sound.findData(cur)))
        finally:
            self.src_sound.blockSignals(False)

    def set_source_sound(self, available: bool, info: str, max_guide: int = 0, repeat: bool = False):
        """The 'Sound' row: shown for a clip after another clip (or when it is set). available: a real video comes
        before it (its sound can be continued); repeat: there is a clip before it (its sound can be repeated); info:
        where the sound comes from; max_guide: the longest carry-over that fits (set when the video's sound is
        switched on)."""
        self._max_guide = max_guide
        cur = self.src_sound.currentData()
        model = self.src_sound.model()
        for k in range(self.src_sound.count()):        # a choice in use stays choosable (to be taken back)
            d = self.src_sound.itemData(k)
            track = isinstance(d, str) and d.startswith(T.TRACK_PREFIX)
            ok = d is None or d == cur or track or (available if d is True else repeat)
            model.item(k).setEnabled(ok)
        some = available or repeat or bool(cur) or self.src_sound.count() > len(SOUND_MODES)
        self._link.setRowVisible(self.sound_box, True)   # (always: Speak the lines)
        self.src_sound.setVisible(some)
        self.src_sound.setEnabled(not self.edit.isReadOnly() and some)
        self._enable_speak()
        self.src_sound_info.setText(info if cur else "")
        self.src_sound_info.setVisible(bool(cur) and bool(info))
        self.sound_over.setVisible(bool(cur))
        self.sound_over.setEnabled(not self.edit.isReadOnly())

    def _enable_guide(self):
        on = not self.edit.isReadOnly()
        has = bool(self.guide.currentData())
        self.guide.setEnabled(on and (self._guide_useful or has))   # a set value can always be taken back
        self.guide.setToolTip(GUIDE_TIP if self._guide_useful else
                              "The first clip has no clip before it (and no real video after it): there is "
                              "nothing to carry over" + (" — this setting has no effect; set it back to "
                                                         "'last frame only'" if has else ""))

    def _add_key_menu(self, at: float | None):
        """Where the new keyframe's picture comes from (at: the moment clicked; None: the biggest free gap)."""
        if at is None:
            times = [0.0] + [it["at"] for it in self.keys.items] + [self.keys.duration]
            gap = max(range(len(times) - 1), key=lambda i: times[i + 1] - times[i])
            at = self.keys.at_of(self.keys.x_of((times[gap] + times[gap + 1]) / 2))
        m = QMenu(self)
        acts = {m.addAction(f"A picture at {at:.2f} s…"): "add", m.addAction("A frame of a video…"): "add_video"}
        if self.keys.made:
            acts[m.addAction("This clip's frame at that moment (keep it, re-roll the rest)…")] = "add_moment"
        a = self._exec_menu(m)
        if a in acts:
            self.keyframe.emit(acts[a], self.nid, {"at": at})

    def _exec_menu(self, m):
        return m.exec(QCursor.pos())

    def set_keyframes(self, duration: float, items: list[dict], cuts: list[float], made: bool):
        self.keys.set_data(duration, items, cuts, made)
        if cuts:
            self.key_warn.setText("⚠ The clip cuts at " + ", ".join(f"{c:.2f} s" for c in cuts) +
                                  ": a keyframe (or the end image) is probably too far from what comes before it "
                                  "— move it later, or use a picture closer to the scene.")
        self.key_warn.setVisible(bool(cuts))

    def _enable_end(self):
        on = not self.edit.isReadOnly()
        self.keys.setEnabled(on)
        self.b_key.setEnabled(on)
        self.b_end.setEnabled(on and self._end_free)
        self.b_end_video.setEnabled(on and self._end_free)
        self.b_end_ai.setEnabled(on and self._end_free)
        self.b_end_clear.setEnabled(on and self._has_end)

    def set_end(self, image: Path | None, video_after: str | None, pixmap: QPixmap | None):
        """How this clip ends: at the video after it, at its own end image, or freely."""
        self._end_free, self._has_end = video_after is None, image is not None
        self.b_end.setToolTip(END_TIP if video_after is None else
                              f"This clip ends at the first frame of video #{video_after[1:]}; remove that video "
                              "to give it an end image of its own")
        if video_after is not None:
            self.end_text.setText(f"first frame of video #{video_after[1:]}"
                                  + (" (own end image not used)" if image is not None else ""))
        elif image is not None:
            self.end_text.setText("own end image" if image.is_file() else "end image missing")
        else:
            self.end_text.setText("free")
        self.end_text.setToolTip(str(image) if image is not None else "")
        if pixmap is not None and not pixmap.isNull():
            self.end_thumb.setPixmap(pixmap)
            self.end_thumb.show()
        else:
            self.end_thumb.clear()
            self.end_thumb.hide()
        self._enable_end()

    def _restyle(self):
        t = theme()
        border = t.accent.name() if self.state == "generating" else t.lineSubtle.name()
        self.setStyleSheet(f"QFrame#promptCard {{ background: {t.bgRaised.name()}; border: 1px solid {border};"
                           f" border-radius: 6px; }} QFrame#promptCard QLabel {{ background: transparent; }}")


class VideoCard(QFrame):
    """A real video in the selected path: its file and range, status; edit the range, play, delete."""
    action = Signal(str, str)                  # action, node id
    play = Signal(str)

    def __init__(self, node: T.Node, parent=None, names=lambda: []):
        super().__init__(parent)
        self.nid = node.id
        self.setObjectName("promptCard")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setFixedWidth(330)
        self.state = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 8, 10)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = label("Video", type_="heading")
        self.sub = label(f"#{node.id[1:]}", level="tertiary")
        self.tag = label("", role="statusTag")
        head.addWidget(self.title)
        head.addWidget(self.sub)
        head.addWidget(self.tag)
        head.addStretch(1)
        self.buttons = []
        for act, ic, tip in (("edit", "pen", "Edit range… (or choose another file)"),
                             ("branch", "split", "Branch: another continuation from the same point (Ctrl+B)"),
                             ("delete", "trash", "Delete this video and everything after it")):
            b = icon_button(ic, tip)
            b.clicked.connect(lambda _=False, a=act: self.action.emit(a, self.nid))
            head.addWidget(b)
            self.buttons.append(b)
        self.frozen, self._on = False, True
        self.b_freeze = icon_button("lock", FREEZE_TIP.replace("this clip", "this video"), checkable=True)
        self.b_freeze.clicked.connect(lambda _=False: self.action.emit("freeze_toggle", self.nid))
        head.insertWidget(head.count() - 3, self.b_freeze)
        v.addLayout(head)
        self.span = label("", level="tertiary")
        v.addWidget(self.span)
        self.thumb = FrameTile(312, 176)
        self.thumb.setToolTip("Last frame of the range; click to play the converted clip")
        self.thumb.clicked.connect(lambda _t: self.play.emit(self.nid))
        v.addWidget(self.thumb)
        self.file = label("", level="secondary")
        self.file.setWordWrap(True)
        v.addWidget(self.file)
        self.range = label("", level="tertiary")
        self.range.setWordWrap(True)
        v.addWidget(self.range)
        self.b_edit = QPushButton("Edit range…")
        self.b_edit.clicked.connect(lambda: self.action.emit("edit", self.nid))
        v.addWidget(self.b_edit, 0, Qt.AlignLeft)
        v.addStretch(1)
        self.step = label("", level="tertiary")
        v.addWidget(self.step)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        v.addWidget(self.bar)
        self.set_source(node)
        self._restyle()

    def set_source(self, node: T.Node):
        src = node.source or {}
        f = Path(src.get("file", ""))
        self.file.setText(f.name + ("" if f.is_file() else "  (file not found)"))
        self.file.setToolTip(str(f))
        n = chain.range_frames(src.get("start", 0), src.get("end", 0))
        self.range.setText(f"{fmt_time(src.get('start', 0))} – {fmt_time(src.get('end', 0))} of the file · "
                           f"{n} frames at 24 fps")

    def set_state(self, state: str, pixmap: QPixmap | None, step: str = ""):
        self.state = state
        self.tag.setText(STATE_TEXT.get(state, state.upper()))
        self.tag.setProperty("state", STATE_STYLE.get(state, ""))
        repolish(self.tag)
        if pixmap is not None and not pixmap.isNull():
            self.thumb.set_image(pixmap.toImage())
        else:
            self.thumb.clear_frame("not converted yet")
        self.step.setText(step)
        self.bar.setVisible(state in ("generating", "converting"))
        self._restyle()

    def set_progress(self, i: int, n: int, text: str | None = None):
        pass

    def set_editable(self, on: bool):
        self._on = bool(on)
        for w in self.buttons:
            w.setEnabled(self._on)
        self.b_edit.setEnabled(self._on and not self.frozen)
        self.buttons[0].setEnabled(self._on and not self.frozen)       # (edit range)
        self.b_freeze.setEnabled(self._on and (self.frozen or self.state == "done"))

    def set_frozen(self, frozen: bool):
        self.frozen = bool(frozen)
        self.b_freeze.setChecked(self.frozen)
        self.b_freeze.setToolTip(FREEZE_TIP.replace("this clip", "this video") if not self.frozen else
                                 "Frozen: kept as converted, whatever the project's size says. Click to unfreeze it")
        self.set_editable(self._on)

    def _restyle(self):
        t = theme()
        border = t.accent.name() if self.state in ("generating", "converting") else t.lineSubtle.name()
        self.setStyleSheet(f"QFrame#promptCard {{ background: {t.bgSunken.name()}; border: 1px solid {border};"
                           f" border-radius: 6px; }} QFrame#promptCard QLabel {{ background: transparent; }}")


# ----------------------------------------------------------------------------- start image tile, preview

class StartTile(FrameTile):
    dropped = Signal(str)

    def __init__(self, w, h, parent=None):
        super().__init__(w, h, parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.clear_frame("Click or drop an image")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.dropped.emit(u.toLocalFile())
                break


class Preview(QWidget):
    """Video player for clips and paths. The media player is created on first use."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = None
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.stack = QStackedWidget()
        self.placeholder = label("Nothing to play yet.\nDouble-click a clip, or play the selected path.",
                                 level="tertiary")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setMinimumHeight(200)
        self.stack.addWidget(self.placeholder)
        v.addWidget(self.stack, 1)
        self.name = label("", level="secondary")
        v.addWidget(self.name)
        row = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause")
        self.b_play.setEnabled(False)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.time = label("0:00.0 / 0:00.0", level="tertiary", type_="mono")
        row.addWidget(self.b_play)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time)
        v.addLayout(row)
        self.b_play.clicked.connect(self.toggle)
        self.slider.sliderMoved.connect(lambda ms: self.player and self.player.setPosition(ms))

    def _ensure_player(self) -> bool:
        if self.player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except ImportError:
            self.placeholder.setText("Video playback is not available (QtMultimedia missing).")
            return False
        self.video = VideoView()                    # (painted frames: no native GPU window, see VideoView)
        self.video.setMinimumHeight(200)
        self.stack.addWidget(self.video)
        self.audio = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video.sink)
        self.player.positionChanged.connect(self._pos)
        self.player.durationChanged.connect(lambda d: self.slider.setRange(0, int(d)))
        self.player.playbackStateChanged.connect(self._state)
        return True

    def open(self, path: Path, title: str, autoplay: bool = True):
        if not path or not Path(path).exists() or not self._ensure_player():
            return
        self.player.stop()
        self.player.setSource(QUrl())
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.stack.setCurrentWidget(self.video)
        self.name.setText(title)
        self.b_play.setEnabled(True)
        self.slider.setEnabled(True)
        if autoplay:
            self.player.play()

    def toggle(self):
        if self.player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop(self):
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
            self.video.clear()

    def _pos(self, ms):
        if not self.slider.isSliderDown():
            self.slider.setValue(int(ms))
        self.time.setText(f"{fmt_time(ms / 1000)} / {fmt_time(self.player.duration() / 1000)}")

    def _state(self, st):
        from PySide6.QtMultimedia import QMediaPlayer
        self.b_play.setIcon(icon("pause" if st == QMediaPlayer.PlayingState else "play"))


# ----------------------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None, command: list[str] | None = None):
        super().__init__()
        self.setWindowTitle("Promptwood")
        self._build_menu()
        self.qs = settings or QSettings()
        self.command = command                  # tests replace the worker with a fake one
        self.project: Path | None = None
        self.tree: T.Tree | None = None
        self.status_of: dict[str, str] = {}
        self.why_of: dict[str, str] = {}             # why a changed clip is changed, in words
        self.live: dict[str, str] = {}          # states during a run (generating / failed / stopped)
        self.progress_of: dict[str, float] = {}
        self.cards: list[PromptCard | VideoCard] = []
        self._pix: dict[tuple, QPixmap] = {}
        self.proc: QProcess | None = None
        self._buf, self._ebuf = b"", ""
        self._stopping = False
        self._phase = ""
        self._error = ""
        self._plan: list[str] = []
        self._made_now: set[str] = set()
        self._targets: list[str] = []
        self._loading = False
        self._tree_stamp: tuple | None = None   # tree.json as this window last read or wrote it
        self._copying: str | None = None        # Save as in progress (the source project)
        self._worker_leaving = False            # the worker said bye (idle timeout) and is quitting
        self._job: dict | None = None           # the generation in progress
        self._worker_idle = False
        self._worker_bound: tuple | None = None
        self._replace_worker = False
        self._idle_since, self._idle_timeout = 0.0, None
        self._model_state = ""
        self._joins = 0                         # path joins still running
        self._converts: dict[str, str] = {}     # video node -> token of its conversion in progress
        self._kill_timer = QTimer(self, singleShot=True, interval=8000, timeout=self._kill)
        self.bridge = _Bridge()
        self.bridge.joined.connect(self._joined)
        self.bridge.film.connect(self._film_joined)
        self.bridge.duplicated.connect(self._scene_duplicated)
        self._film_queue: list[str] | None = None    # Generate film: the scenes still to make
        self.bridge.comfy.connect(self._comfy_state)
        self.bridge.loras.connect(self._loras_found)
        self.bridge.triggers.connect(self._triggers_found)
        self._trigger_scan = None
        self.bridge.copied.connect(self._copied)
        self.bridge.converted.connect(self._converted)
        self.bridge.llm_room.connect(self._llm_room_made)
        self.bridge.spoken.connect(self._lines_spoken)
        self.bridge.eleven_models.connect(self._fill_eleven_models)
        self.bridge.eleven_cloned.connect(self._eleven_cloned)
        self.bridge.tts_log.connect(lambda m: (self.log.appendPlainText("TTS: " + m), self.status.setText(m + "…")))
        self._tts_local: TTS.LocalTTS | None = None
        self._tts_busy = False
        self._eleven_voices: list | None = None
        self.h3_loras: list[tuple[str, str]] = []
        self.qwen_loras: list[tuple[str, str]] = []
        self.image_loras: dict[str, list[tuple[str, str]]] = {}
        self.turbo_loras: list[tuple[str, str]] = []
        self._scanned_at = 0.0                  # last LoRA scan (monotonic)
        self.comfy_up: bool | None = None
        self._comfy_polling = False
        self._comfy_starting = 0.0
        self._waiting_comfy: list[str] | None = None   # targets to generate once ComfyUI answers
        # the model this app last used in ComfyUI ("h3" clips, "qwen" image edits, None: none / unloaded). ComfyUI
        # is asked to unload only when it changes: switching models frees the other one first instead of letting
        # both fight over the memory; the same model again is used as it is (no reload)
        self._comfy_model: str | None = None
        self._quit_frees = True                 # the worker's next quit unloads the model
        self._qwen_timer = QTimer(self, singleShot=True, timeout=self._qwen_idle)
        self._join_lock = threading.Lock()      # joins run one after another (play after generating + export)
        self._save_timer = QTimer(self, singleShot=True, interval=400, timeout=self._save_tree)
        self._refresh_timer = QTimer(self, singleShot=True, interval=250, timeout=self.refresh)
        self._build()
        self._load_settings()
        self._gpu_proc: QProcess | None = None
        self._gpu_timer = QTimer(self, interval=4000, timeout=self._poll_gpu)
        self._gpu_timer.start()
        self._comfy_timer = QTimer(self, interval=4000, timeout=self._poll_comfy)
        self._comfy_timer.start()
        QTimer.singleShot(0, self._poll_comfy)
        QTimer.singleShot(0, self.scan_loras)
        self.comfy_dir.editingFinished.connect(self.scan_loras)
        QTimer.singleShot(0, self._poll_gpu)

    # ------------------------------------------------------------------ layout
    def _build(self):
        self.resize(1560, 940)
        split = QSplitter(Qt.Horizontal, self)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        self.split = split

        # panels (an editor's layout): left tabs (clips graph, cast, sounds), the viewer in the middle with the
        # film / path / log below, the project's settings on the right; ComfyUI's settings in Preferences
        def panel():
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            return w, v

        def scrolled(w):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(w)
            return sa
        left = QWidget()                                    # (parent of the section bars below)
        proj_w, proj_v = panel()
        cast_w, cast_v = panel()
        sound_w, sound_v = panel()
        self.prefs = QDialog(self)
        self.prefs.setWindowTitle("Preferences")
        prefs_v = QVBoxLayout(self.prefs)

        bar, _ = section_bar("Project", left)
        proj_v.addWidget(bar)
        pw = QWidget()
        pf = QFormLayout(pw)
        pf.setContentsMargins(10, 6, 10, 8)
        prow = QHBoxLayout()
        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText("a folder for this chain (new or existing)")
        self.project_edit.setReadOnly(True)
        self.b_project = icon_button("fopen", "Open or create a project folder")
        self.b_project.clicked.connect(self._choose_project)
        self.b_open_dir = icon_button("folder", "Show the project folder")
        self.b_open_dir.clicked.connect(self._open_dir)
        prow.addWidget(self.project_edit, 1)
        prow.addWidget(self.b_project)
        prow.addWidget(self.b_open_dir)
        pf.addRow("Folder", prow)
        brow = QHBoxLayout()
        self.b_new = QPushButton("New…")
        self.b_new.setToolTip("Create a new project (Ctrl+Shift+N)")
        self.b_new.clicked.connect(self.new_project)
        self.b_open = QPushButton("Open…")
        self.b_open.setToolTip("Open a project folder (Ctrl+O)")
        self.b_open.clicked.connect(self._choose_project)
        self.b_recent = QPushButton("Recent")
        self.b_recent.setToolTip("Open one of the last projects")
        self.recent_menu = QMenu(self)
        self.b_recent.setMenu(self.recent_menu)
        for b in (self.b_new, self.b_open, self.b_recent):
            brow.addWidget(b)
        brow.addStretch(1)
        pf.addRow("", brow)
        proj_v.addWidget(pw)

        bar, _ = section_bar("Start image", left)
        proj_v.addWidget(bar)
        sw = QWidget()
        sv = QVBoxLayout(sw)
        sv.setContentsMargins(10, 8, 10, 8)
        self.start_tile = StartTile(320, 180)
        sv.addWidget(self.start_tile, 0, Qt.AlignHCenter)
        self.size_info = label("", level="tertiary")
        self.size_info.setWordWrap(True)
        sv.addWidget(self.size_info)
        self.b_ai_edit = QPushButton("Edit with AI…")
        self.b_ai_edit.setToolTip("Change an image by a prompt (Qwen Image Edit in ComfyUI) and use the result as "
                                  "the start image or as a clip's end image")
        self.b_ai_edit.clicked.connect(lambda: self.edit_image(start=True, mode="edit"))
        self.b_ai_create = QPushButton("Create a new image…")
        self.b_ai_create.setToolTip("Make a new image from a prompt or from reference pictures (Krea-2, Qwen-Image, "
                                    "FLUX, Illustrious) and use it as the start image, an end image or a face "
                                    "reference")
        self.b_ai_create.clicked.connect(lambda: self.edit_image(start=True, mode="create"))
        airow = QHBoxLayout()
        airow.addWidget(self.b_ai_edit)
        airow.addWidget(self.b_ai_create)
        airow.addStretch(1)
        self.b_start_clear = icon_button("trash", "Remove the start image: the first clip then starts from its "
                                                  "prompt (the size follows the Aspect setting or a real video)")
        self.b_start_clear.clicked.connect(lambda: self.clear_image(confirm=True))
        airow.addWidget(self.b_start_clear)
        sv.addLayout(airow)
        proj_v.addWidget(sw)

        bar, _ = section_bar("Cast", left)
        cast_v.addWidget(bar)
        iw = QWidget()
        iv = QVBoxLayout(iw)
        iv.setContentsMargins(10, 6, 10, 8)
        iv.setSpacing(6)
        self.cast_list = QListWidget()
        self.cast_list.setIconSize(QSize(40, 40))
        self.cast_list.setFixedHeight(110)
        self.cast_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cast_list.setToolTip("The characters of this video. In a prompt write @Name (typing @ offers the "
                                  "names): the clip is made with exactly the faces it mentions. A clip that "
                                  "mentions nobody uses the lead's face.")
        self.cast_list.currentRowChanged.connect(lambda _r: self._show_character())
        self.cast_list.itemDoubleClicked.connect(lambda _i: self._rename_character())
        iv.addWidget(self.cast_list)
        crow = QHBoxLayout()
        self.b_char_new = QPushButton("New…")
        self.b_char_new.setToolTip("A new character: a name to mention as @Name, then its face pictures")
        self.b_char_new.clicked.connect(self._new_character)
        self.b_char_rename = QPushButton("Rename…")
        self.b_char_rename.setToolTip("Rename the character; every @mention in the prompts is renamed too")
        self.b_char_rename.clicked.connect(self._rename_character)
        self.b_char_del = QPushButton("Remove")
        self.b_char_del.setToolTip("Remove the character and its face pictures")
        self.b_char_del.clicked.connect(self._remove_character)
        crow.addWidget(self.b_char_new)
        crow.addWidget(self.b_char_rename)
        crow.addStretch(1)
        crow.addWidget(self.b_char_del)
        iv.addLayout(crow)
        self.char_desc = QLineEdit()
        self.char_desc.setPlaceholderText("Description (optional), e.g. woman, 30s, grey knit sweater")
        self.char_desc.setToolTip("Goes into the prompt the model gets for clips that mention this character")
        self.char_desc.editingFinished.connect(self._character_desc_changed)
        iv.addWidget(self.char_desc)
        self.char_lead = QCheckBox("Lead: clips that mention nobody get this face")
        self.char_lead.toggled.connect(self._character_lead_changed)
        iv.addWidget(self.char_lead)
        vrow = QHBoxLayout()
        self.char_voice = label("Voice: none", level="tertiary")
        self.char_voice.setToolTip("A character with a voice speaks its lines with it: write the line in quotes in "
                                   "the sentence of its @Name, e.g. @Anna looks up and says \"Good morning.\" "
                                   "Use a voice only with the speaker's consent.")
        self.b_voice_file = QPushButton("Voice…")
        self.b_voice_file.setToolTip("Set the voice from an audio or video file: choose the part of it where only "
                                     "this speaker talks, with little music or echo (the clip model hears the first "
                                     f"{T.VOICE_MAX_SECS:g} s of the part; text to speech clones the voice from up "
                                     f"to {T.VOICE_REF_MAX_SECS:g} s of it)")
        self.b_voice_file.clicked.connect(self._voice_from_file)
        self.b_voice_clip = QPushButton("From clip")
        self.b_voice_clip.clicked.connect(self._voice_from_selected)
        self.b_voice_tts = QPushButton("TTS…")
        self.b_voice_tts.setToolTip("Make the voice with text to speech: designed from a description (Qwen3-TTS, "
                                    "local) or one of your ElevenLabs voices. Its lines can then be spoken with it "
                                    "(Speak the lines on a clip)")
        self.b_voice_tts.clicked.connect(self._voice_with_tts)
        self.b_voice_clone = QPushButton("Clone")
        self.b_voice_clone.setToolTip("Clone the voice at ElevenLabs: the voice sample is uploaded to your ElevenLabs "
                                      "account and becomes a voice there (an instant voice clone; needs a paid "
                                      "plan). The character's lines are then spoken with it when ElevenLabs is the "
                                      "text to speech provider. Only with the speaker's consent.")
        self.b_voice_clone.clicked.connect(self._clone_voice_at_eleven)
        self.b_voice_play = icon_button("play", "Listen to the voice")
        self.b_voice_play.clicked.connect(self._play_voice)
        self.b_voice_del = icon_button("x", "Remove the voice")
        self.b_voice_del.clicked.connect(self._remove_voice)
        vrow.addWidget(self.char_voice, 1)
        for b in (self.b_voice_file, self.b_voice_clip, self.b_voice_tts, self.b_voice_clone, self.b_voice_play,
                  self.b_voice_del):
            vrow.addWidget(b)
        iv.addLayout(vrow)
        self.refs = QListWidget()
        self.refs.setViewMode(QListWidget.IconMode)
        self.refs.setIconSize(QSize(64, 64))
        self.refs.setFixedHeight(92)
        self.refs.setMovement(QListWidget.Static)
        self.refs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.refs.setToolTip("Face pictures of the selected character (up to 3, e.g. front and side): best a "
                             "sharp close-up of the face, without the scene around it")
        iv.addWidget(self.refs)
        rrow = QHBoxLayout()
        self.b_ref_add = QPushButton("Image…")
        self.b_ref_add.setToolTip("Add an image file as a face picture of the selected character")
        self.b_ref_add.clicked.connect(self._add_reference)
        self.b_ref_create = QPushButton("Create…")
        self.b_ref_create.setToolTip("Create a face picture with AI (the Create image window); there “Add as face "
                                     "reference” adds a result to the selected character")
        self.b_ref_create.clicked.connect(lambda: self.create_reference())   # (clicked passes checked=False)
        self.b_ref_start = QPushButton("Start image")
        self.b_ref_start.clicked.connect(self._reference_from_start)
        self.b_ref_clip = QPushButton("Selected clip")
        self.b_ref_clip.clicked.connect(self._reference_from_selected)
        self.b_ref_del = QPushButton("Remove")
        self.b_ref_del.setToolTip("Remove the selected face picture")
        self.b_ref_del.clicked.connect(self._remove_reference)
        self.b_ref_crop = QPushButton("Crop…")
        self.b_ref_crop.setToolTip("Crop the selected face picture again (from the full picture it came from)")
        self.b_ref_crop.clicked.connect(self._recrop_reference)
        self.refs.itemDoubleClicked.connect(lambda _i: self._recrop_reference())
        self.refs.currentRowChanged.connect(lambda _r: self._update_cast_buttons())
        rrow.addWidget(label("Add face:", level="tertiary"))
        for b in (self.b_ref_add, self.b_ref_create, self.b_ref_start, self.b_ref_clip):
            rrow.addWidget(b)
        rrow.addStretch(1)
        rrow.addWidget(self.b_ref_crop)
        rrow.addWidget(self.b_ref_del)
        iv.addLayout(rrow)
        irow = QFormLayout()
        irow.setContentsMargins(0, 0, 0, 0)
        self.ref_size = QComboBox()
        self.ref_size.addItem("match (fast)", "match")
        self.ref_size.addItem("max 2048 px (best likeness, slower)", "max")
        irow.addRow("Size", self.ref_size)
        self.dialogue_lang = QComboBox()
        self.dialogue_lang.setEditable(True)
        self.dialogue_lang.addItems(["English", "German", "French", "Spanish", "Italian", "Portuguese", "Japanese",
                                     "Chinese", "Korean"])
        self.dialogue_lang.setToolTip("The language of the spoken lines (\"...\" in a prompt); the model gets each "
                                      "line as <d>[Language] ...</d> when a character speaks with its voice")
        irow.addRow("Speech", self.dialogue_lang)
        self.tts_provider = QComboBox()
        for k, text in TTS.PROVIDERS.items():
            self.tts_provider.addItem(text, k)
        self.tts_provider.setToolTip("Who speaks the lines with text to speech (Speak the lines on a clip, TTS… for a "
                                     "voice): Qwen3-TTS on this computer, cloning each character's voice sample, or "
                                     "ElevenLabs with your API key (Settings), in each character's ElevenLabs voice")
        irow.addRow("Text to speech", self.tts_provider)
        mrow = QHBoxLayout()
        self.eleven_model = QComboBox()
        self.eleven_model.setToolTip("The ElevenLabs model the lines are spoken with")
        self.eleven_model.addItem(TTS.DEFAULT_OPTIONS["eleven_model"], TTS.DEFAULT_OPTIONS["eleven_model"])
        self.b_eleven_models = icon_button("rotate", "Ask ElevenLabs for its models")
        self.b_eleven_models.clicked.connect(self._load_eleven_models)
        mrow.addWidget(self.eleven_model, 1)
        mrow.addWidget(self.b_eleven_models)
        self._eleven_row = QWidget()
        self._eleven_row.setLayout(mrow)
        mrow.setContentsMargins(0, 0, 0, 0)
        irow.addRow("ElevenLabs model", self._eleven_row)
        self._tts_form = irow
        iv.addLayout(irow)
        self.ref_hint = QCheckBox("Tie each person to their face in the prompt")
        self.ref_hint.setChecked(True)
        self.ref_hint.setToolTip("With @mentions: \"Anna has exactly the face of the person in <Picture 1>.\" "
                                 "and that the pictures only show who the people are. Without: "
                                 f"\"{REF_HINT}\"")
        iv.addWidget(self.ref_hint)
        self.ident_info = label("", level="tertiary")
        self.ident_info.setWordWrap(True)
        iv.addWidget(self.ident_info)
        cast_v.addWidget(iw)
        self.ref_size.currentIndexChanged.connect(self._identity_changed)
        self.ref_hint.toggled.connect(self._identity_changed)
        self.dialogue_lang.activated.connect(self._identity_changed)          # a finished choice, not each key
        self.tts_provider.currentIndexChanged.connect(self._tts_changed)
        self.eleven_model.currentIndexChanged.connect(self._tts_changed)
        self.dialogue_lang.lineEdit().editingFinished.connect(self._identity_changed)

        bar, _ = section_bar("Audio tracks", left)
        sound_v.addWidget(bar)
        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.setContentsMargins(10, 6, 10, 8)
        tv.setSpacing(6)
        self.track_list = QListWidget()
        self.track_list.setFixedHeight(96)
        self.track_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.track_list.setToolTip("Sound the clips play as their soundtrack (a clip's Sound: “Audio track: …”). "
                                   "Double-click to cut it again")
        self.track_list.itemDoubleClicked.connect(lambda _i: self._recut_track())
        self.track_list.currentRowChanged.connect(lambda _r: self._update_track_buttons())
        tv.addWidget(self.track_list)
        trow = QHBoxLayout()
        self.b_track_add = QPushButton("Import…")
        self.b_track_add.setToolTip("Import the sound of an MP3, WAV, OGG, FLAC … or a video file and cut it to the "
                                    "part you need")
        self.b_track_add.clicked.connect(lambda: self.import_track())
        self.b_track_clip = QPushButton("From selected clip…")
        self.b_track_clip.clicked.connect(self._track_from_selected)
        self.b_track_cut = QPushButton("Cut…")
        self.b_track_cut.setToolTip("Change the part of the file the track plays (or its name); the clips playing it "
                                    "count as changed")
        self.b_track_cut.clicked.connect(self._recut_track)
        self.b_track_play = icon_button("play", "Listen to the selected audio track")
        self.b_track_play.clicked.connect(self._play_track)
        self.b_track_del = QPushButton("Remove")
        self.b_track_del.setToolTip("Remove the selected audio track")
        self.b_track_del.clicked.connect(self._remove_track)
        for w in (self.b_track_add, self.b_track_clip):
            trow.addWidget(w)
        trow.addStretch(1)
        for w in (self.b_track_cut, self.b_track_play, self.b_track_del):
            trow.addWidget(w)
        tv.addLayout(trow)
        self.track_info = label("", level="tertiary")
        self.track_info.setWordWrap(True)
        tv.addWidget(self.track_info)
        sound_v.addWidget(tw)

        bar, _ = section_bar("Sound reference", left)
        sound_v.addWidget(bar)
        aw = QWidget()
        av = QVBoxLayout(aw)
        av.setContentsMargins(10, 6, 10, 8)
        av.setSpacing(6)
        self.sounds = QListWidget()
        self.sounds.setFixedHeight(70)
        self.sounds.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sounds.setToolTip("Sound references: in a clip's prompt the first one is <Audio 1>, the second "
                               "<Audio 2> ... Double-click to listen")
        self.sounds.itemDoubleClicked.connect(lambda _i: self._play_sound())
        self.sounds.currentRowChanged.connect(lambda _r: self._update_sound_buttons())
        av.addWidget(self.sounds)
        arow = QHBoxLayout()
        self.b_sound_add = QPushButton("File…")
        self.b_sound_add.setToolTip("Add the sound of an audio or video file (WAV, MP3, FLAC, MP4 …)")
        self.b_sound_add.clicked.connect(self._add_sound)
        self.b_sound_clip = QPushButton("Selected clip")
        self.b_sound_clip.clicked.connect(self._sound_from_selected)
        self.b_sound_play = icon_button("play", "Listen to the selected sound reference")
        self.b_sound_play.clicked.connect(self._play_sound)
        self.b_sound_del = QPushButton("Remove")
        self.b_sound_del.setToolTip("Remove the selected sound reference")
        self.b_sound_del.clicked.connect(self._remove_sound)
        arow.addWidget(label("Add:", level="tertiary"))
        arow.addWidget(self.b_sound_add)
        arow.addWidget(self.b_sound_clip)
        arow.addStretch(1)
        arow.addWidget(self.b_sound_play)
        arow.addWidget(self.b_sound_del)
        av.addLayout(arow)
        self.sound_info = label("", level="tertiary")
        self.sound_info.setWordWrap(True)
        av.addWidget(self.sound_info)
        sound_v.addWidget(aw)

        bar, _ = section_bar("Clip settings (the project's defaults)", left)
        proj_v.addWidget(bar)
        cw = QWidget()
        c = QFormLayout(cw)
        c.setContentsMargins(10, 6, 10, 8)
        self.frames = QComboBox()
        for n in FRAME_CHOICES:
            self.frames.addItem(frames_text(n), n)
        self.frames.setCurrentIndex(FRAME_CHOICES.index(73))
        self.frames.setToolTip("Length of every clip (a clip can have its own). Longer clips mean fewer hand-overs "
                               "between clips, so the face drifts less; they take longer and need more memory")
        c.addRow("Length", self.frames)
        self.resolution = QComboBox()
        for r, txt in ((384, "384 (fast draft)"), (512, "512 (like the training)"), (640, "640"), (768, "768"),
                       (chain.H3_NATIVE, "768p — H3's native size (16:9 = 1344×768; slower, Veda helps)")):
            self.resolution.addItem(txt, r)
        self.resolution.setCurrentIndex(1)
        self.resolution.setToolTip("About this many pixels squared, in the start image's aspect ratio")
        c.addRow("Size", self.resolution)
        self.aspect = QComboBox()
        for key in T.ASPECTS:
            self.aspect.addItem(key, key)
        self.aspect.setToolTip("The picture format of a project without a start image or real video: its first clip "
                               "starts from the prompt alone (with the cast's face pictures as references when it "
                               "mentions someone). With a start image or video, their format is used instead")
        c.addRow("Aspect", self.aspect)
        self.steps = QSpinBox()
        self.steps.setRange(4, 80)
        self.steps.setValue(28)
        self.steps.setToolTip(STEPS_TIP)
        c.addRow("Steps", self.steps)
        self.speed = QComboBox()
        self.speed.addItem("Normal", "normal")
        self.speed.addItem("Turbo (turbo LoRA, 4-8 steps)", "turbo")
        self.speed.addItem("HyperFlow (8 steps)", "hyperflow")
        self.speed.addItem("Pika H3 Flex (6 steps)", "pika")
        self.speed.setToolTip("HyperFlow: Video Rebirth's 8-step LoRA, about 2× faster than Normal (measured on an "
                              "RTX 4090: 19 s instead of 42 s for a 3 s clip at 672×384; needs the "
                              "ComfyUI-HyperFlow-H3 nodes and its pruned weights). "
                              "Pika H3 Flex: Pika's post-trained H3 as a LoRA, 6 steps (lcm), reported close to "
                              "the full model; your own LoRAs can be added. "
                              "Turbo: a step-distilled LoRA with its own sampler renders every clip in 4-8 steps "
                              "instead of ~20 (needs the ComfyUI-MiniMax-H3-Turbo nodes)")
        c.addRow("Speed", self.speed)
        crow = QHBoxLayout()
        crow.setContentsMargins(0, 0, 0, 0)
        self.clip_cfg = QDoubleSpinBox()
        self.clip_cfg.setRange(0.0, 10.0)
        self.clip_cfg.setSingleStep(0.05)
        self.clip_cfg.setDecimals(2)
        self.clip_cfg.setValue(1.0)
        self.clip_cfg.setToolTip(
            "CFG of the chosen speed mode (each mode keeps its own value). 1: off, one pass per step, as H3 and its "
            "speed LoRAs (turbo, HyperFlow, Pika) are made for. Above 1: real classifier-free guidance against an "
            "empty prompt (start / end frames stay): the prompt is followed more strongly, but every step runs twice "
            "(about twice as slow) and the step-distilled LoRAs can burn colours or over-sharpen from about 1.5-2; "
            "try 1.2-1.5 first. Below 1 (e.g. 0.9): the prompt is followed a little less, also two passes per step")
        self.cfg_note = label("", level="tertiary")
        crow.addWidget(self.clip_cfg)
        crow.addWidget(self.cfg_note, 1)
        c.addRow("CFG", crow)
        self._cfg_mode = "normal"
        self.veda = QCheckBox("Fast attention (Veda) for large and long clips")
        self.veda.setChecked(True)
        self.veda.setToolTip(
            "Veda sparse attention (the veda-sparse-attention node in ComfyUI) computes only the attention that "
            "matters. It is used automatically for clips large enough to gain, marked “⚡ Veda” on the clip. "
            "Measured on an RTX 4090: a 10 s clip at 1344×768 took 146 s instead of 338 s (2.3× faster) and stayed "
            "coherent; a 5 s clip at 672×384 got 25–30 % slower, so small clips keep full attention. A clip made "
            "with Veda differs slightly from one made without (switching this makes such clips CHANGED)")
        c.addRow("", self.veda)
        self.turbo_lora = QComboBox()
        self.turbo_lora.setToolTip("The turbo LoRA (only turbo LoRAs are listed here, and they are left out of the "
                                   "normal LoRA lists); v4 step600 ema is the recommended one")
        c.addRow("Turbo LoRA", self.turbo_lora)
        trow = QHBoxLayout()
        self.turbo_steps = QSpinBox()
        self.turbo_steps.setRange(4, 8)
        self.turbo_steps.setValue(6)
        self.turbo_steps.setSuffix(" steps")
        self.turbo_steps.setToolTip("4 is the minimum, 6-8 look clearly better; above 8 it gets over-sharp")
        self.turbo_strength = QDoubleSpinBox()
        self.turbo_strength.setRange(-10.0, 10.0)
        self.turbo_strength.setSingleStep(0.05)
        self.turbo_strength.setValue(1.0)
        self.turbo_strength.setToolTip("Keep 1.0; smear/ghosting -> a bit higher (1.05-1.2), over-sharp grain -> "
                                       "a bit lower (0.8-0.95)")
        trow.addWidget(self.turbo_steps)
        trow.addWidget(label("strength", level="tertiary"))
        trow.addWidget(self.turbo_strength)
        trow.addStretch(1)
        self._turbo_row = QWidget()
        self._turbo_row.setLayout(trow)
        trow.setContentsMargins(0, 0, 0, 0)
        c.addRow("", self._turbo_row)
        self._clips_form = c
        self.audio = QCheckBox("Generate sound")
        self.audio.setChecked(True)
        c.addRow("", self.audio)
        self.keep_rep = QCheckBox("Keep repeated frames in paths")
        self.keep_rep.setToolTip("Each clip starts with the last frame of the clip before; by default that repeated "
                                 "frame is left out when a path is joined")
        c.addRow("", self.keep_rep)
        self.style_prefix = QLineEdit()
        self.style_prefix.setPlaceholderText("e.g. cinematic, soft daylight, handheld camera")
        self.style_prefix.setToolTip("Put before every prompt (look, camera, light, a LoRA's trigger word). "
                                     "Changing it makes every generated clip again")
        self.style_suffix = QLineEdit()
        self.style_suffix.setPlaceholderText("optional")
        self.style_suffix.setToolTip("Put after every prompt. Changing it makes every generated clip again")
        c.addRow("Style before", self.style_prefix)
        c.addRow("Style after", self.style_suffix)
        for w in (self.style_prefix, self.style_suffix):
            w.editingFinished.connect(self._style_changed)
        self.lora_triggers = QCheckBox("Put the LoRAs' trigger words before the prompt")
        self.lora_triggers.setChecked(True)
        self.lora_triggers.setToolTip("A clip with LoRAs gets their trigger words in front of its prompt (from the "
                                      "LoRA files and CivitAI; shown under each LoRA, right-click to change them). "
                                      "Clips record them, so new words make those clips again")
        self.lora_triggers.toggled.connect(self._lora_triggers_changed)
        c.addRow("", self.lora_triggers)
        proj_v.addWidget(cw)

        bar, _ = section_bar("ComfyUI", left)
        prefs_v.addWidget(bar)
        aw = QWidget()
        a = QFormLayout(aw)
        a.setContentsMargins(10, 6, 10, 8)
        self.comfy_url = QLineEdit(DEFAULT_COMFY_URL)
        self.comfy_url.setToolTip("Address of the ComfyUI that renders the clips")
        a.addRow("Address", self.comfy_url)
        crow = QHBoxLayout()
        self.comfy_launcher = QLineEdit(DEFAULT_COMFY_LAUNCHER)
        self.comfy_launcher.setToolTip("Command that starts ComfyUI (your app-drawer launcher)")
        self.b_start_comfy = QPushButton("Start ComfyUI")
        self.b_start_comfy.setToolTip("Start ComfyUI in the background with this command")
        self.b_start_comfy.clicked.connect(self.start_comfy)
        crow.addWidget(self.comfy_launcher, 1)
        crow.addWidget(self.b_start_comfy)
        a.addRow("Start with", crow)
        self.comfy_dir = QLineEdit(DEFAULT_COMFY_DIR)
        self.comfy_dir.setToolTip("ComfyUI's folder: its LoRA folders are searched for the chosen LoRA files, and "
                                  "its copies of the clips are removed after they were taken over")
        drow = QHBoxLayout()
        drow.addWidget(self.comfy_dir, 1)
        self.b_rescan = QPushButton("Rescan LoRAs")
        self.b_rescan.setToolTip("Look for new LoRAs in ComfyUI's LoRA folders (also done when you come back to "
                                 "this window)")
        self.b_rescan.clicked.connect(lambda: self.scan_loras(report=True))
        drow.addWidget(self.b_rescan)
        a.addRow("Folder", drow)
        self.comfy_cleanup = QCheckBox("Clean up ComfyUI's copies after each clip")
        self.comfy_cleanup.setChecked(True)
        self.comfy_cleanup.setToolTip("ComfyUI keeps every clip and start image in output/video-chain and "
                                      "input/video-chain; the project has its own copy, so they are removed after "
                                      "the download (only there, never anything else)")
        a.addRow("", self.comfy_cleanup)
        self.comfy_autostart = QCheckBox("Start ComfyUI when generating")
        self.comfy_autostart.setChecked(True)
        self.comfy_autostart.setToolTip("If ComfyUI is not running when you generate, start it with the command "
                                        "above and begin as soon as it answers")
        a.addRow("", self.comfy_autostart)
        self.triggers_online = QCheckBox("Look up LoRA trigger words on CivitAI")
        self.triggers_online.setChecked(True)
        self.triggers_online.setToolTip("In the background, after each LoRA scan: LoRAs are looked up on CivitAI by their "
                                        "file hash (this tells CivitAI which LoRA files you have, nothing else). "
                                        "Found trigger words are kept in ~/.local/share/video-chain/"
                                        "lora_triggers.json, everything else CivitAI tells (descriptions, example "
                                        "prompts, base model, files, license) next to it in civitai/. Right-click a "
                                        "LoRA for its CivitAI details.")
        self.llm_on = QCheckBox("Prompt refiner: also a llama.cpp server (GGUF language models)")
        self.llm_on.setToolTip("For language models ComfyUI cannot run (GGUF, e.g. Huihui Qwen3-VL-30B-A3B "
                               "abliterated). The app starts llama-server when you refine with such a model and "
                               "stops it before clips are made and when it closes (the model fills a lot of RAM).")
        a.addRow("", self.llm_on)
        self.llm_server = QLineEdit(LLM_DEFAULT_SERVER)
        self.llm_server.setToolTip("The llama-server program (llama.cpp)")
        a.addRow("llama-server", self.llm_server)
        lrow = QHBoxLayout()
        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)
        self.llm_model.setToolTip("The GGUF model file (found in the model folders; a path can be typed too)")
        for f in find_gguf_models():
            self.llm_model.addItem(f.name, str(f))
            self.llm_model.setItemData(self.llm_model.count() - 1, str(f), Qt.ToolTipRole)
        self.llm_port = QSpinBox()
        self.llm_port.setRange(1024, 65535)
        self.llm_port.setValue(LLM_DEFAULT_PORT)
        self.llm_port.setToolTip("The local port of the llama.cpp server")
        lrow.addWidget(self.llm_model, 1)
        lrow.addWidget(label("port", level="tertiary"))
        lrow.addWidget(self.llm_port)
        a.addRow("GGUF model", lrow)
        self.llm_cpu_moe = QCheckBox("Keep the model's experts in RAM (GPU stays free for ComfyUI; slower)")
        self.llm_cpu_moe.setChecked(True)
        self.llm_cpu_moe.setToolTip("Mixture-of-experts models (…-A3B): their expert weights stay in RAM and only "
                                    "attention runs on the GPU, a few GB of VRAM. Off: as much as fits on the GPU "
                                    "(faster, but it competes with ComfyUI for VRAM)")
        a.addRow("", self.llm_cpu_moe)
        self.video_gpu = QCheckBox("Play videos on the GPU (smooth; takes effect after a restart)")
        self.video_gpu.setChecked(True)
        self.video_gpu.setToolTip("On: the preview players decode on the GPU and draw through Qt's GPU video surface: "
                                  "smooth also with big or 2K clips. NVIDIA's driver crashed there once while "
                                  "ComfyUI held nearly all GPU memory; if the app crashes when playing, switch it "
                                  "off. Off: decoded and drawn on the CPU: never crashes, but lags with big clips. "
                                  "(The before / after comparison is always drawn on the CPU.)")
        self.video_gpu.toggled.connect(lambda on: self._video_gpu_changed(on))
        a.addRow("", self.video_gpu)
        self.triggers_online.toggled.connect(lambda on: on and self._scan_triggers())
        a.addRow("", self.triggers_online)
        crow = QHBoxLayout()
        self.civitai_key = QLineEdit()
        self.civitai_key.setEchoMode(QLineEdit.Password)
        self.civitai_key.setToolTip("Your CivitAI API key (civitai.com → Account settings → API Keys). With it CivitAI "
                                    "answers as your account, e.g. with the prompts of example images it holds back "
                                    "otherwise. Kept in the system keyring, never in a settings file; sent only to "
                                    "CivitAI. ($CIVITAI_API_KEY wins when it is set.)")
        b_key = QPushButton("Save key")
        b_key.setAutoDefault(False)
        b_key.clicked.connect(lambda: self._save_civitai_key())
        crow.addWidget(self.civitai_key, 1)
        crow.addWidget(b_key)
        a.addRow("CivitAI API key", crow)
        self._show_civitai_key()
        erow = QHBoxLayout()
        self.eleven_key = QLineEdit()
        self.eleven_key.setEchoMode(QLineEdit.Password)
        self.eleven_key.setToolTip("Your ElevenLabs API key (elevenlabs.io → Developers → API keys), for text to "
                                   "speech with ElevenLabs. Kept in the system keyring, never in a settings file or "
                                   "a project; sent only to ElevenLabs. ($ELEVENLABS_API_KEY wins when it is set.)")
        b_ekey = QPushButton("Save key")
        b_ekey.setAutoDefault(False)
        b_ekey.clicked.connect(lambda: self._save_eleven_key())
        erow.addWidget(self.eleven_key, 1)
        erow.addWidget(b_ekey)
        a.addRow("ElevenLabs API key", erow)
        self._show_eleven_key()
        krow = QHBoxLayout()
        self.keep_loaded = QCheckBox("Keep the model loaded for")
        self.keep_loaded.setChecked(True)
        self.keep_loaded.setToolTip("After generating (or editing images), the model stays in memory, so the next "
                                    "clip or edit starts at once instead of loading for minutes. It is unloaded "
                                    "after this idle time, or with Unload in the status bar. Switching between "
                                    "clips (MiniMax H3) and image edits (Qwen) unloads the other model first.")
        self.idle_minutes = QSpinBox()
        self.idle_minutes.setRange(1, 240)
        self.idle_minutes.setValue(15)
        self.idle_minutes.setSuffix(" min")
        krow.addWidget(self.keep_loaded)
        krow.addWidget(self.idle_minutes)
        krow.addStretch(1)
        a.addRow("", krow)
        self.keep_loaded.toggled.connect(self.idle_minutes.setEnabled)
        prefs_v.addWidget(aw)
        pb = QDialogButtonBox(QDialogButtonBox.Close)
        pb.button(QDialogButtonBox.Close).setAutoDefault(False)     # Enter in a field must not close it
        pb.button(QDialogButtonBox.Close).setDefault(False)
        pb.rejected.connect(self.prefs.reject)
        prefs_v.addWidget(pb)
        for v in (proj_v, cast_v, sound_v):
            v.addStretch(1)
        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)
        cast_scroll, sound_scroll, proj_scroll = scrolled(cast_w), scrolled(sound_w), scrolled(proj_w)
        self.inspector = QTabWidget()
        self.inspector.setDocumentMode(True)
        self.inspector.addTab(proj_scroll, "Project")
        self.proj_scroll = proj_scroll
        self.inspector.setMinimumWidth(380)
        self._panels = (proj_w, cast_w, sound_w)

        # the clips graph (left tab), the film strip (under the viewer)
        gw = QWidget()
        gv = QVBoxLayout(gw)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.setSpacing(0)
        film_w = QWidget()
        film_v = QVBoxLayout(film_w)
        film_v.setContentsMargins(0, 0, 0, 0)
        film_v.setSpacing(0)
        fbar, fbh = section_bar("Film", film_w)
        self.film_info = label("", level="tertiary")
        fbh.addWidget(self.film_info)
        film_v.addWidget(fbar)
        self.storyboard = Storyboard()
        self.storyboard.sceneChosen.connect(self.switch_scene)
        self.storyboard.addWanted.connect(self.add_scene)
        self.storyboard.renameWanted.connect(self.rename_scene)
        self.storyboard.deleteWanted.connect(self.delete_scene)
        self.storyboard.moved.connect(self._scenes_moved)
        self.storyboard.joinWanted.connect(self.set_scene_join)
        self.storyboard.duplicateWanted.connect(self.duplicate_scene)
        self.storyboard.notesWanted.connect(self.scene_notes)
        self.b_film_gen = QPushButton("Generate film")
        self.b_film_gen.setToolTip("Make every scene's missing clips, scene after scene (up to each scene's ending)")
        self.b_film_gen.clicked.connect(self.generate_film)
        self.b_film_play = QPushButton("▶ Play film")
        self.b_film_play.setToolTip("Join all scenes in their order (hard cuts) and play the film")
        self.b_film_play.clicked.connect(lambda: self.join_film("play"))
        self.b_film_export = QPushButton("Export film…")
        self.b_film_export.setToolTip("Save the whole film as one video")
        self.b_film_export.clicked.connect(self.export_film)
        self.b_film_cast = QPushButton("Cast…")
        self.b_film_cast.setToolTip("Who appears in which scene (along the clips the film plays)")
        self.b_film_cast.clicked.connect(self.show_film_cast)
        for b in (self.b_film_cast, self.b_film_gen, self.b_film_play, self.b_film_export):
            fbh.addWidget(b)
        film_v.addWidget(self.storyboard)
        self.film = None
        bar, bh = section_bar("Clips", gw)
        self.tree_info = label("", level="tertiary")
        bh.addWidget(self.tree_info)
        bh.addSpacing(8)
        self.b_continue = icon_button("plus", "Continue: a new clip after the selected one (Ctrl+N)")
        self.b_add_video = icon_button("camera", "Add a real video after the selected clip (Ctrl+Shift+V): a "
                                                 "generated clip before it ends at its first frame, one after it "
                                                 "starts from its last frame")
        self.b_branch = icon_button("split", "Branch: another continuation from the same point (Ctrl+B)")
        self.b_variant = icon_button("clone", "Variant: same prompt, other seed (Ctrl+D)")
        self.b_delete = icon_button("trash", "Delete the selected clip and everything after it (Del)")
        self.b_fit = icon_button("frame", "Fit the view (F)")
        for btn, act in ((self.b_continue, "continue"), (self.b_add_video, "video"), (self.b_branch, "branch"),
                         (self.b_variant, "variant"), (self.b_delete, "delete")):
            btn.clicked.connect(lambda _=False, a=act: self.node_action(a, self.target_id()))
            bh.addWidget(btn)
        bh.addSpacing(6)
        self.b_fit.clicked.connect(lambda: self.graph.fit())
        bh.addWidget(self.b_fit)
        gv.addWidget(bar)
        self.graph = GraphView()
        self.graph.selected.connect(self._graph_clicked)
        self.graph.activated.connect(self._activate)
        self.graph.contextMenu.connect(self._context)
        gv.addWidget(self.graph, 1)
        self.left_tabs.addTab(gw, "Clips")
        self.left_tabs.addTab(cast_scroll, "Cast")
        self.left_tabs.addTab(sound_scroll, "Sounds")
        self.left_tabs.setMinimumWidth(340)

        pw2 = QWidget()
        pv = QVBoxLayout(pw2)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        bar, bh = section_bar("Selected path", pw2)
        self.path_info = label("", level="tertiary")
        bh.addWidget(self.path_info)
        pv.addWidget(bar)
        self.cast_strip = CastStrip()
        self.cast_strip.clipClicked.connect(self._show_card_of)
        self.cast_strip.hide()
        pv.addWidget(self.cast_strip)
        self.timeline = Timeline()
        self.timeline.clicked.connect(self.focus_clip)
        self.timeline.activated.connect(self._play_node)
        self.timeline.menu.connect(lambda nid, pos: self._context(nid, pos))
        pv.addWidget(self.timeline)
        pv.addStretch(1)
        # the inspector's Clip tab: the card of the clip in focus (one at a time; the path's other cards wait in
        # the stack), with ‹ › to step along the path
        clip_tab = QWidget()
        ctv = QVBoxLayout(clip_tab)
        ctv.setContentsMargins(8, 6, 8, 6)
        nav = QHBoxLayout()
        self.b_prev_clip = QPushButton("‹")
        self.b_prev_clip.setToolTip("The clip before it on the path (Alt+Left)")
        self.b_prev_clip.clicked.connect(lambda: self._step_focus(-1))
        self.b_next_clip = QPushButton("›")
        self.b_next_clip.setToolTip("The clip after it on the path (Alt+Right)")
        self.b_next_clip.clicked.connect(lambda: self._step_focus(1))
        self.focus_info = label("", level="secondary")
        for b in (self.b_prev_clip, self.b_next_clip):
            b.setFixedWidth(32)
        nav.addWidget(self.b_prev_clip)
        nav.addWidget(self.focus_info, 1)
        nav.addWidget(self.b_next_clip)
        ctv.addLayout(nav)
        self.card_stack = _CurrentStack()
        self.path_hint = label("", level="tertiary")
        self.path_hint.setWordWrap(True)
        self.path_hint.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.card_stack.addWidget(self.path_hint)
        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        card_scroll.setFrameShape(QFrame.NoFrame)
        card_scroll.setWidget(self.card_stack)
        ctv.addWidget(card_scroll, 1)
        self.inspector.insertTab(0, clip_tab, "Clip")
        self.inspector.setCurrentIndex(0)
        self.focus: str | None = None

        # the viewer (the biggest area), under it the film and the selected path / the log
        rp = QWidget()
        rv = QVBoxLayout(rp)
        rv.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview()
        rv.addWidget(self.preview, 1)
        prow = QHBoxLayout()
        prow.setContentsMargins(8, 0, 8, 8)
        self.b_play_clip = QPushButton("▶ Clip")
        self.b_play_clip.setToolTip("Play the clip shown in the inspector (Space in the timeline: double-click)")
        self.b_play_clip.clicked.connect(lambda: self.focus and self._play_node(self.focus))
        self.b_play_path = QPushButton("▶ Path")
        self.b_play_path.setToolTip("Join the clips of the selected path and play them")
        self.b_play_path.clicked.connect(lambda: self.join_selected("play"))
        self.b_play_film = QPushButton("▶ Film")
        self.b_play_film.setToolTip("Join all scenes of the film and play it")
        self.b_play_film.clicked.connect(lambda: self.join_film("play"))
        self.b_export = QPushButton("Export path…")
        self.b_export.setToolTip("Save the selected path as one video")
        self.b_export.clicked.connect(lambda: self.export_path())          # (clicked passes checked=False)
        self.b_export_clip = QPushButton("Export clip…")
        self.b_export_clip.setToolTip("Save the clip in the inspector on its own (as made, or upscaled)")
        self.b_export_clip.clicked.connect(lambda: self.focus and self.export_path(self.focus))
        prow.addWidget(label("Play", level="tertiary"))
        for b in (self.b_play_clip, self.b_play_path, self.b_play_film):
            prow.addWidget(b)
        prow.addSpacing(12)
        prow.addWidget(self.b_export)
        prow.addWidget(self.b_export_clip)
        prow.addStretch(1)
        rv.addLayout(prow)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("type", "mono")
        self.log.setMaximumBlockCount(5000)
        self.tabs = QTabWidget()                            # under the viewer: the selected path, the log
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(pw2, "Selected path")
        self.tabs.addTab(self.log, "Log")
        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(0)
        bv.addWidget(film_w)
        bv.addWidget(self.tabs, 1)
        center = QSplitter(Qt.Vertical)
        center.setChildrenCollapsible(False)
        center.addWidget(rp)
        center.addWidget(bottom)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        self.center = center
        split.addWidget(self.left_tabs)
        split.addWidget(center)
        split.addWidget(self.inspector)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1300, 420])
        self.viewer = rp

        sb = self.statusBar()
        self.b_gen = QPushButton("Generate path")
        self.b_gen.setProperty("variant", "primary")
        self.b_gen.setToolTip("Make every clip on the selected path that is not made yet (Ctrl+Enter)")
        self.b_gen.clicked.connect(self.generate_path)
        self.b_gen_all = QPushButton("Generate all")
        self.b_gen_all.setToolTip("Make every clip of every branch that is not made yet (Ctrl+Shift+Enter)")
        self.b_gen_all.clicked.connect(self.generate_all)
        self.b_stop = QPushButton("Stop")
        self.b_stop.setProperty("variant", "danger")
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setRange(0, 1000)
        self.status = label("Open or create a project folder.", role="statusText")
        self.gpu = label("GPU …", role="statusText")
        tb = self.addToolBar("Run")                # making clips: at the top, like an editor's render buttons
        tb.setObjectName("runToolbar")
        tb.setMovable(False)
        for w in (self.b_gen, self.b_gen_all, self.b_stop):
            tb.addWidget(w)
        tb.addSeparator()
        tb.addWidget(self.progress)
        self.run_toolbar = tb
        sb.addWidget(self.status, 1)
        self.model_label = label("Model: not loaded", role="statusText")
        self.b_unload = QPushButton("Unload")
        self.b_unload.setToolTip("Free the GPU and RAM now (the next Generate loads the model again)")
        self.b_unload.clicked.connect(self.unload)
        self.b_unload.hide()
        self.comfy_label = label("", role="statusText")
        self.b_prefs = icon_button("gear", "ComfyUI and preferences (Ctrl+,)")
        self.b_prefs.clicked.connect(self.show_prefs)
        sb.addPermanentWidget(self.comfy_label)
        sb.addPermanentWidget(self.b_prefs)
        sb.addPermanentWidget(self.model_label)
        sb.addPermanentWidget(self.b_unload)
        sb.addPermanentWidget(self.gpu)
        self._model_timer = QTimer(self, interval=20000, timeout=self._update_model_label)
        self._model_timer.start()

        self.start_tile.clicked.connect(lambda _t: self._choose_image())
        self.start_tile.dropped.connect(lambda f: self.set_image(f, confirm=True))
        for sig in (self.frames.currentIndexChanged, self.resolution.currentIndexChanged,
                    self.aspect.currentIndexChanged,
                    self.steps.valueChanged, self.audio.toggled, self.keep_rep.toggled,
                    self.speed.currentIndexChanged, self.veda.toggled,
                    self.turbo_lora.currentIndexChanged, self.turbo_steps.valueChanged,
                    self.turbo_strength.valueChanged, self.clip_cfg.valueChanged):
            sig.connect(self._options_changed)
        for keys, fn in (("Ctrl+N", lambda: self.node_action("continue", self.target_id())),
                         ("Ctrl+Shift+V", lambda: self.node_action("video", self.target_id())),
                         ("Ctrl+B", lambda: self.node_action("branch", self.target_id())),
                         ("Ctrl+D", lambda: self.node_action("variant", self.target_id())),
                         ("Ctrl+Return", self.generate_path), ("Ctrl+Shift+Return", self.generate_all),
                         ("Alt+Left", lambda: self._step_focus(-1)), ("Alt+Right", lambda: self._step_focus(1))):
            QShortcut(QKeySequence(keys), self, activated=fn)
        for w in (*self._panels, self.prefs):
            guard_wheel(w)
        QShortcut(QKeySequence("Delete"), self.graph,
                  activated=lambda: self.node_action("delete", self.target_id()))
        QShortcut(QKeySequence("F"), self.graph, activated=self.graph.fit)
        self._update_buttons()

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        q = self.qs
        self.keep_loaded.setChecked(q.value("keepLoaded", True, type=bool))
        self.comfy_url.setText(q.value("comfyUrl", DEFAULT_COMFY_URL, type=str))
        self.comfy_launcher.setText(q.value("comfyLauncher", DEFAULT_COMFY_LAUNCHER, type=str))
        self.comfy_dir.setText(q.value("comfyDir", DEFAULT_COMFY_DIR, type=str))
        self.comfy_cleanup.setChecked(q.value("comfyCleanup", True, type=bool))
        self.comfy_autostart.setChecked(q.value("comfyAutoStart", True, type=bool))
        self.triggers_online.setChecked(q.value("loraTriggersOnline", True, type=bool))
        self.llm_on.setChecked(q.value("llmServerOn", self.llm_model.count() > 0, type=bool))  # on when one is found
        self.llm_server.setText(q.value("llmServer", LLM_DEFAULT_SERVER, type=str))
        saved = q.value("llmModel", "", type=str)
        if saved:
            i = self.llm_model.findData(saved)
            if i < 0:
                self.llm_model.addItem(Path(saved).name, saved)
                i = self.llm_model.count() - 1
            self.llm_model.setCurrentIndex(i)
        self.llm_port.setValue(q.value("llmPort", LLM_DEFAULT_PORT, type=int))
        self.llm_cpu_moe.setChecked(q.value("llmCpuMoe", True, type=bool))
        self.video_gpu.blockSignals(True)
        self.video_gpu.setChecked(q.value("videoGpu", True, type=bool))
        self.video_gpu.blockSignals(False)
        self.idle_minutes.setValue(q.value("idleMinutes", 15, type=int))
        if g := q.value("geometry"):
            self.restoreGeometry(g)
        self._restore_layout()
        self._fill_recent()
        proj = q.value("project", "", type=str)
        if proj and Path(proj).is_dir():
            self.open_project(Path(proj), quiet=True)

    def save_settings(self):
        q = self.qs
        q.setValue("keepLoaded", self.keep_loaded.isChecked())
        q.setValue("comfyUrl", self.comfy_url.text().strip())
        q.setValue("comfyLauncher", self.comfy_launcher.text().strip())
        q.setValue("comfyDir", self.comfy_dir.text().strip())
        q.setValue("comfyCleanup", self.comfy_cleanup.isChecked())
        q.setValue("comfyAutoStart", self.comfy_autostart.isChecked())
        q.setValue("loraTriggersOnline", self.triggers_online.isChecked())
        q.setValue("videoGpu", self.video_gpu.isChecked())
        q.setValue("llmServerOn", self.llm_on.isChecked())
        q.setValue("llmServer", self.llm_server.text().strip())
        q.setValue("llmModel", self.llm_model_path())
        q.setValue("llmPort", self.llm_port.value())
        q.setValue("llmCpuMoe", self.llm_cpu_moe.isChecked())
        q.setValue("idleMinutes", self.idle_minutes.value())
        q.setValue("project", str(self.project) if self.project else "")
        if self.tree is not None:
            q.setValue("lastOptions", json.dumps(self.tree.options))
        q.setValue("geometry", self.saveGeometry())
        q.setValue("layout", json.dumps({"v": LAYOUT_VERSION, "split": self.split.sizes(),
                                         "center": self.center.sizes(), "left": self.act_left.isChecked(),
                                         "left_w": (self.split.sizes()[0] if self.act_left.isChecked() else 0)
                                         or getattr(self, "_left_w", 0)}))
        q.sync()

    # ------------------------------------------------------------------ project
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        self.act_new = m.addAction("New project…", self.new_project)
        self.act_new.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self.act_open = m.addAction("Open project…", self._choose_project)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.recent_file_menu = m.addMenu("Recent projects")
        m.addSeparator()
        self.act_save = m.addAction("Save", self.save_project)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_as = m.addAction("Save as…", self.save_project_as)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        m.addSeparator()
        self.act_close = m.addAction("Close project", self.close_project)
        m.addAction("Show project folder", self._open_dir)
        self.act_ai_edit = m.addAction("Edit an image with AI…", lambda: self.edit_image(start=True, mode="edit"))
        self.act_ai_create = m.addAction("Create a new image with AI…",
                                         lambda: self.edit_image(start=True, mode="create"))
        m.addSeparator()
        m.addAction("Quit", self.close).setShortcut(QKeySequence("Ctrl+Q"))
        v = self.menuBar().addMenu("&View")
        for k, name in enumerate(("Clips", "Cast", "Sounds")):
            a = v.addAction(f"{name} (left)", lambda _=False, k=k: self._show_left_tab(k))
            a.setShortcut(QKeySequence(f"Alt+{k + 1}"))
        v.addAction("Log", lambda: self.tabs.setCurrentWidget(self.log)).setShortcut(QKeySequence("Alt+L"))
        v.addSeparator()
        self.act_left = v.addAction("Left panel (clips, cast, sounds)")
        self.act_left.setCheckable(True)
        self.act_left.setChecked(True)
        self.act_left.setShortcut(QKeySequence("Ctrl+Alt+B"))
        self.act_left.toggled.connect(lambda on: self._show_left(on))   # (the panel is built later, in _build)
        for name, key, tip in (("edit", "Ctrl+Alt+1", "Layout: Edit (big viewer)"),
                               ("write", "Ctrl+Alt+2", "Layout: Write (wide prompt)"),
                               ("watch", "Ctrl+Alt+3", "Layout: Watch (viewer and timeline only)")):
            v.addAction(tip, lambda _=False, n=name: self.apply_layout(n)).setShortcut(QKeySequence(key))
        v.addAction("Reset layout", lambda: self.apply_layout("edit"))
        st = self.menuBar().addMenu("&Settings")
        st.addAction("ComfyUI and preferences…", self.show_prefs).setShortcut(QKeySequence("Ctrl+,"))

    def apply_layout(self, name: str):
        """A layout preset, as shares of the window: 'edit' (the default: a big viewer), 'write' (a wide inspector
        for the prompt), 'watch' (no left panel, a narrow inspector)."""
        left, mid, right, view = LAYOUTS.get(name, LAYOUTS["edit"])
        w = max(self.split.width(), 1200)
        if left > 0:
            self._left_w = int(w * left)
        self.act_left.setChecked(left > 0)              # (shows / hides the left panel)
        self.split.setSizes([int(w * left) if left > 0 else 0, int(w * mid), int(w * right)])
        h = max(self.center.height(), 600)
        self.center.setSizes([int(h * view), int(h * (1 - view))])

    def _show_left(self, on: bool):
        """Show / hide the left panel; shown again it gets its last width back (taken from the viewer)."""
        sizes = self.split.sizes()
        if not on and sizes[0] > 0:
            self._left_w = sizes[0]
        self.left_tabs.setVisible(on)
        if on:
            want = max(getattr(self, "_left_w", 0), self.left_tabs.minimumWidth())
            sizes = self.split.sizes()
            if sizes[0] < want:
                give = want - sizes[0]
                self.split.setSizes([want, max(sizes[1] - give, 1), sizes[2]])

    def _show_left_tab(self, k: int):
        self.act_left.setChecked(True)                  # (a hidden panel is shown first)
        self.left_tabs.setCurrentIndex(k)

    def _restore_layout(self):
        """The layout of the last session; the default one if there is none, it was saved by an older version or
        it is damaged."""
        try:
            d = json.loads(self.qs.value("layout", "", type=str) or "{}")
            split, center = d["split"], d["center"]
            ok = (d.get("v") == LAYOUT_VERSION and len(split) == 3 and len(center) == 2
                  and all(isinstance(x, int) and x >= 0 for x in split + center)
                  and split[1] > 0 and split[2] > 0 and all(x > 0 for x in center))
            left_w = int(d.get("left_w") or 0)
        except (TypeError, ValueError, KeyError, AttributeError):
            ok = False
        if not ok:
            self.apply_layout("edit")
            return
        if left_w > 0:
            self._left_w = left_w
        elif split[0] > 0:
            self._left_w = split[0]
        self.split.setSizes(split)
        self.act_left.setChecked(bool(d.get("left", True)))
        self.center.setSizes(center)

    def show_prefs(self):
        """ComfyUI's address, launcher, folder and the model / trigger word settings (not modal: the status bar
        and the window stay usable)."""
        self.prefs.resize(max(self.prefs.width(), 620), self.prefs.sizeHint().height())
        self.prefs.show()
        self.prefs.raise_()
        self.prefs.activateWindow()

    # ---- recent projects
    def recent_projects(self) -> list[str]:
        try:
            items = json.loads(self.qs.value("recentProjects", "[]", type=str))
        except ValueError:
            items = []
        return [p for p in items if isinstance(p, str)]

    def _remember_project(self, folder: Path):
        items = [str(folder)] + [p for p in self.recent_projects() if p != str(folder)]
        self.qs.setValue("recentProjects", json.dumps(items[:MAX_RECENT]))
        self._fill_recent()

    def _fill_recent(self):
        for menu in (self.recent_menu, self.recent_file_menu):
            menu.clear()
            items = [p for p in self.recent_projects() if (Path(p) / T.TREE_FILE).exists() or
                     (Path(p) / "chain.json").exists()]
            for p in items:
                act = menu.addAction(f"{Path(p).name}   —   {Path(p).parent}")
                act.triggered.connect(lambda _=False, f=p: self.open_project(Path(f)))
            if not items:
                menu.addAction("no recent projects").setEnabled(False)
        self.b_recent.setEnabled(bool(self.recent_projects()))

    def _update_title(self):
        if self.project is None:
            self.setWindowTitle("Promptwood")
            return
        scene = self.film.scene_of(self.project) if self.film else None
        name = self.film.root.name if self.film else self.project.name
        many = self.film is not None and len(self.film.scenes) > 1
        self.setWindowTitle(f"{name} › {scene.title} — Promptwood" if many and scene else f"{name} — Promptwood")

    # ---- the whole film: play, export, generate
    def join_film(self, purpose: str = "play", out: Path | None = None):
        if self.film is None or self.running:
            return
        if self._joins:
            self.status.setText("A join is still running; wait for it.")
            return
        if not self._save_tree():
            return
        self._update_continues()
        film, bridge = F.Film.load(self.film.root), self.bridge
        target = Path(out) if out else self.film.root / "paths" / "film.mp4"
        root = str(self.film.root)
        self.status.setText("Checking and joining the film …")
        self._joins += 1                           # (a run waits for it, a second join is refused)
        self._update_buttons()

        def work():
            try:
                bridge.film.emit((F.join_film(film, target), root), None, purpose)
            except Exception as e:  # noqa: BLE001
                bridge.film.emit((None, root), str(e), purpose)
        threading.Thread(target=work, daemon=True).start()

    def _film_joined(self, result, error, purpose):
        self._joins = max(0, self._joins - 1)
        self._update_buttons()
        path, root = result
        if self.film is None or str(self.film.root) != root:
            return                                 # another film was opened meanwhile
        if error:
            self.status.setText(f"The film cannot be joined yet: {error}.")
            return
        try:
            secs = chain.probe(Path(path))["frames"] / 24
        except Exception:  # noqa: BLE001
            secs = 0.0
        if purpose == "play":
            self.preview.open(Path(path), f"Film · {len(self.film.scenes) if self.film else 1} scenes · "
                                          f"{fmt_time(secs)}")
            self.status.setText(f"Film joined: {fmt_time(secs)}.")
        else:
            self.status.setText(f"Exported the film: {Path(path).name} ({fmt_time(secs)}).")

    def export_film(self):
        if self.film is None or self.running:
            return
        start = str(self.film.root / f"{self.film.root.name}-film.mp4")
        f, _ = QFileDialog.getSaveFileName(self, "Export the film", start, "Video (*.mp4)")
        if f:
            self.join_film("export", Path(f if f.lower().endswith(".mp4") else f + ".mp4"))

    def generate_film(self):
        """Make every scene's missing clips (up to each scene's ending), one scene after the other."""
        if self.film is None or self.running:
            return
        if not self._save_tree():
            return
        self._update_continues()
        todo, empty = [], []
        for sc in self.film.scenes:
            d = self.film.path(sc.id)
            try:
                t = T.load(d)
            except (OSError, ValueError) as e:
                self.status.setText(f"Scene “{sc.title}” cannot be read: {e}")
                return
            take = self.film.take_of(sc.id, t)
            if take is None:
                empty.append(sc.title)
                continue
            ch = T.Checker(d, t)
            if any(ch.status(n.id) != "done" for n in t.path_to(take)):
                todo.append(sc.id)
        if empty:
            self.status.setText(f"Scene “{empty[0]}” has no clip yet: write its first clip, or remove the scene.")
            return
        if not todo:
            self.status.setText("Every scene is made: ▶ Play film.")
            return
        ids = [sc.id for sc in self.film.scenes]          # from the first scene with work on, each one checked when
        self._film_queue = ids[ids.index(todo[0]):]       # its turn comes (a continuing one may need it by then)
        self._film_total = len(self._film_queue)
        self._film_next()

    def _film_next(self):
        if not self._film_queue:
            self._film_queue = None
            self.status.setText("The film is made: ▶ Play film.")
            self._show_storyboard()
            return
        if self.film is not None and len(self.film.scenes) > 1:
            try:
                self.film.update_continues()       # the scenes after a new end start from it
            except (OSError, ValueError):
                pass
        sid = self._film_queue.pop(0)
        done = self._film_total - len(self._film_queue)
        d = self.film.path(sid) if self.film else None
        try:
            t = T.load(d)
            take = self.film.take_of(sid, t)
            ch = T.Checker(d, t)
            ready = take is not None and all(ch.status(n.id) == "done" for n in t.path_to(take))
        except (OSError, ValueError, StopIteration):
            ready = False
        if ready:                                  # made already: the next one
            QTimer.singleShot(0, self._film_next)
            return
        self.switch_scene(sid)
        if self.film is None or self.film.path(sid) != self.project:
            self._film_queue = None                # could not open the scene (the status says why)
            return
        take = self.film.take_of(sid, self.tree)
        self._start([take])
        if self._job is None and self._waiting_comfy is None:
            self._film_queue = None                # not started (the status says why)
        else:
            self.status.setText(f"Generating the film: scene {done} of {self._film_total} "
                                f"(“{self.film.scene(sid).title}”) … " + self.status.text())

    # ---- the film's scenes (film.py)
    def _show_storyboard(self):
        if self.film is None:
            self.storyboard.show_scenes([], None)
            self.film_info.setText("")
            return
        scenes = []
        for sc in self.film.scenes:
            if self.project is not None and self.film.path(sc.id) == self.project:
                summ = F.summary(self.film, sc.id, self.tree, self.status_of)
            else:
                summ = F.summary(self.film, sc.id)
            scenes.append((sc.id, sc.title, summ))
        self.storyboard.show_scenes(scenes, self.film.current)
        cur = self.film.scene_of(self.project) if self.project else None
        self.film_info.setText(f"› {cur.title}" if cur and len(self.film.scenes) > 1 else "")
        self._update_title()

    def _update_scene_card(self):
        """The open scene's card follows the work on it (its path, states)."""
        if self.film is None or self.project is None or self.tree is None:
            return
        cur = self.film.scene_of(self.project)
        if cur is not None and cur.take and cur.take not in self.tree.nodes:
            try:
                self.film.set_take(cur.id, None)
                self.status.setText(f"The clip “{cur.title}” ended at is gone: the film plays its selected path "
                                    "now (right-click a clip → End the scene here).")
            except OSError:
                pass
        if cur is not None:
            self.storyboard.update_card(cur.id, cur.title, F.summary(self.film, cur.id, self.tree, self.status_of))

    def _sync_shared(self):
        """The cast, sounds and size settings are the film's: written into the other scenes when they change."""
        if self.film is None or len(self.film.scenes) < 2 or self.tree is None:
            return
        now = json.dumps(self.film.shared(self.tree.options), sort_keys=True)
        if now == getattr(self, "_shared_sent", None):
            return
        try:
            self.film.sync_shared(self.project, self.tree.options)
            self._shared_sent = now
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not pass the cast / sound settings on to the other scenes: {e}")

    def switch_scene(self, sid: str):
        if self.film is None or self.project is None:
            return
        if self.running:
            self.status.setText("Wait until generating has finished.")
            self._show_storyboard()
            return
        if self.film.path(sid) == self.project:
            return
        if not self.open_project(self.film.root, scene=sid):
            self._show_storyboard()                # back to the scene still open

    def add_scene(self):
        if self.film is None or self.project is None or self.running:
            return
        title, ok = QInputDialog.getText(self, "New scene", "Title of the new scene:",
                                         text=f"Scene {len(self.film.scenes) + 1}")
        if not ok:
            return
        if not self._may_leave_project():
            return
        cur = self.film.scene_of(self.project)
        try:
            scene = self.film.add_scene(title, after=cur.id if cur else None, like=cur.id if cur else None)
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not add the scene: {e}")
            return
        if self.open_project(self.film.root, scene=scene.id):
            self.status.setText(f"Scene “{scene.title}” added: choose its start image (Start image), or begin "
                                "with a real video. The cast and sounds are the film's.")

    def rename_scene(self, sid: str):
        if self.film is None:
            return
        title, ok = QInputDialog.getText(self, "Rename scene", "Title:", text=self.film.scene(sid).title)
        if not ok:
            return
        try:
            self.film.rename(sid, title)
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not rename the scene: {e}")
            return
        self._show_storyboard()

    def delete_scene(self, sid: str):
        if self.film is None or self.running:
            return
        if self._joins or self._converts or self._copying:
            self.status.setText("Wait until the joins, conversions and copies have finished.")
            return
        sc = self.film.scene(sid)
        if sc.dir == ".":
            self.status.setText("The film folder's own scene cannot be deleted; move it if it should not be first.")
            return
        if len(self.film.scenes) < 2:
            return
        n = F.summary(self.film, sid)["clips"]
        if QMessageBox.question(self, "Delete scene", f"Delete the scene “{sc.title}” with its {n} clip"
                                f"{'s' if n != 1 else ''} and their files? This cannot be undone.") != QMessageBox.Yes:
            return
        if self.film.path(sid) == self.project:    # leave it first
            other = next(s for s in self.film.scenes if s.id != sid)
            self.switch_scene(other.id)
            if self.project == self.film.path(sid):
                return
        try:
            self.film.remove(sid)
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not delete the scene: {e}")
            return
        self._show_storyboard()
        self.status.setText(f"Scene “{sc.title}” deleted.")

    def _update_continues(self) -> list[str]:
        """Bring continuing scenes to the current end of the scene before them; the open scene is read again
        when its start image changed."""
        if self.film is None or len(self.film.scenes) < 2 or self.film.problem:
            return []
        try:
            updated = self.film.update_continues()
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not update the scenes that continue another: {e}")
            return []
        cur = self.film.scene_of(self.project) if self.project else None
        if cur is not None and cur.id in updated:
            self.open_project(self.film.root, scene=cur.id)
        return updated

    def duplicate_scene(self, sid: str):
        if self.film is None:
            return
        if self.running or self._joins or self._converts or self._copying:
            self.status.setText("Wait until generating, joins and copies have finished.")
            return
        if not self._save_tree():
            return
        root, bridge = self.film.root, self.bridge
        self._copying = str(root)                  # (no scene switch, no other copy meanwhile)
        self._update_buttons()
        self.status.setText(f"Copying “{self.film.scene(sid).title}” with its clips …")

        def work():
            try:
                sc = F.Film.load(root).duplicate(sid)
                bridge.duplicated.emit((sc.id, sc.title), None, str(root))
            except Exception as e:  # noqa: BLE001
                bridge.duplicated.emit(None, str(e), str(root))
        threading.Thread(target=work, daemon=True).start()

    def _scene_duplicated(self, result, error, root):
        self._copying = None
        self._update_buttons()
        if self.film is None or str(self.film.root) != root:
            return
        if error:
            self.status.setText(f"Could not duplicate the scene: {error}")
            return
        current = self.film.current
        try:
            self.film = F.Film.load(self.film.root)
            self.film.current = current
        except (OSError, ValueError):
            pass
        self._show_storyboard()
        self.status.setText(f"“{result[1]}” added: change it freely, the original stays as it is.")

    def scene_notes(self, sid: str):
        if self.film is None:
            return
        text, ok = QInputDialog.getMultiLineText(self, "Scene notes", f"Notes for “{self.film.scene(sid).title}” "
                                                 "(e.g. what happens, a line of script):", self.film.scene(sid).notes)
        if not ok:
            return
        try:
            self.film.set_notes(sid, text)
        except OSError as e:
            self.status.setText(f"Could not save the notes: {e}")
            return
        self._show_storyboard()

    def show_film_cast(self):
        """A table: the film's characters by scene (face ●, voice ♪), along the clips the film plays."""
        if self.film is None:
            return
        if not self._save_tree():
            return
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
        names, rows = F.cast_by_scene(self.film)
        dlg = QDialog(self)
        dlg.setWindowTitle("Cast in the film")
        v = QVBoxLayout(dlg)
        if not names:
            v.addWidget(label("The film has no characters yet (the Cast tab, Alt+2).", level="secondary"))
        table = QTableWidget(len(names), len(rows))
        table.setHorizontalHeaderLabels([f"{k}  {title}" for k, (title, _m) in enumerate(rows, start=1)])
        table.setVerticalHeaderLabels([f"@{n}" for n in names])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        sym = {"face": "●", "voice": "♪", "face+voice": "●♪"}
        for c, (_title, marks) in enumerate(rows):
            for r, n in enumerate(names):
                item = QTableWidgetItem(sym.get(marks.get(n, ""), ""))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        v.addWidget(table)
        v.addWidget(label("● face   ♪ speaks with its voice", level="tertiary"))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        dlg.resize(min(900, 160 + 120 * len(rows)), min(600, 140 + 30 * len(names)))
        self._film_cast_dialog = dlg               # (tests look at it)
        if not getattr(self, "_dialogs_modeless", False):
            dlg.exec()
            dlg.deleteLater()

    def set_scene_join(self, sid: str, join: str):
        """How a scene follows the one before: a cut, a crossfade, or continuing from its last frame."""
        if self.film is None or self.running:
            return
        if not self._save_tree():
            return
        try:
            self.film.set_join(sid, join)
            updated = self.film.update_continues() if join == "continue" else []
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not change the join: {e}")
            return
        sc = self.film.scene(sid)
        if self.film.path(sid) == self.project and sid in updated:
            self.open_project(self.film.root, scene=sid)   # its start image changed under the open window
        self._show_storyboard()
        self.status.setText({"cut": f"“{sc.title}” follows with a hard cut.",
                             "fade": f"“{sc.title}” fades in over the scene before (½ s).",
                             "continue": f"“{sc.title}” starts from the last frame of the scene before"
                                         + ("." if updated else " (once that one is made)."),
                             }[join])

    def _scenes_moved(self, order: list[str]):
        if self.film is None:
            return
        reset = []
        try:
            reset = self.film.reorder(order)
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not move the scene: {e}")
        self._show_storyboard()
        if reset:
            self.status.setText("Moved. " + ", ".join(f"“{self.film.scene(i).title}”" for i in reset) +
                                " now start" + ("s" if len(reset) == 1 else "") + " with a cut: the scene before "
                                "changed (right-click → Join with the scene before → Continue to continue it).")

    # ---- new / open / save / close
    def new_project(self):
        if self.running:
            return
        last = self.recent_projects()
        location = str(Path(last[0]).parent) if last else str(Path.home())
        dlg = NewProjectDialog(self, location)
        if dlg.exec() != QDialog.Accepted:
            return
        self.create_project(dlg.folder(), dlg.image.text().strip())

    def create_project(self, folder: Path, image: str = "") -> bool:
        """Create a new project folder (it must not be a project yet) and open it; optionally with a start image."""
        folder = Path(folder).expanduser()
        if (folder / T.TREE_FILE).exists():
            self.status.setText(f"{folder.name} is already a project.")
            return False
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.status.setText(f"Could not create {folder}: {e}")
            return False
        if not self.open_project(folder):
            return False
        if image:
            self.set_image(image)
        self.status.setText(f"New project {folder.name}." + ("" if image else " Choose a start image."))
        return True

    def _choose_project(self):
        if self.running:
            return
        last = self.recent_projects()
        start = str(self.project.parent) if self.project else (str(Path(last[0]).parent) if last else "")
        f = QFileDialog.getExistingDirectory(self, "Open project folder", start)
        if not f:
            return
        folder = Path(f)
        if not (folder / T.TREE_FILE).exists() and not (folder / "chain.json").exists():
            if QMessageBox.question(self, "Open project", f"{folder.name} is not a Promptwood project yet.\n\n"
                                    "Create a new project in this folder?") != QMessageBox.Yes:
                return
        self.open_project(folder)

    def save_project(self):
        if self.project is None or self.tree is None:
            self.status.setText("No project open.")
            return
        if self.running:
            self.status.setText("The project is saved by the generator while it runs.")
            return
        self._save_timer.stop()
        if self._write_tree():
            self.status.setText(f"Saved {self.project.name} ({time.strftime('%H:%M:%S')}).")

    def save_project_as(self):
        """Copy the whole project (clips included) to a new folder and continue there."""
        if self.project is None or self.running:
            return
        root = self.film.root if self.film else self.project
        d = QFileDialog.getExistingDirectory(self, "Save the project as (choose the folder to put the copy in)",
                                             str(root.parent))
        if not d:
            return
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save as", "Name of the copy:", text=f"{root.name}-copy")
        if not ok or not name.strip():
            return
        self.copy_project_to(Path(d) / name.strip())

    def copy_project_to(self, target: Path):
        if self.project is None or self.running:
            return
        target = Path(target).expanduser()
        if target.exists() and any(target.iterdir()):
            self.status.setText(f"{target} exists and is not empty.")
            return
        self._save_timer.stop()
        if not self._write_tree():
            return
        source, bridge = (self.film.root if self.film else self.project), self.bridge
        scene_rel = os.path.relpath(self.project, source)       # the scene to continue in, inside the copy
        self._copy_scene = scene_rel
        cur = self.film.scene_of(self.project) if self.film else None
        self._copy_scene_id = cur.id if cur else None
        existed = target.exists()
        self.status.setText(f"Copying the project to {target} …")
        self.act_save_as.setEnabled(False)
        self._copying = str(source)


        def work():
            try:
                skip = shutil.ignore_patterns("*.part", "*.part.*", "video-chain-*")

                def ignore(d, names):
                    out = set(skip(d, names))
                    if Path(d).parent.name == F.SCENES_DIR and T.REFS_DIR in names:
                        out.add(T.REFS_DIR)           # a scene's link to the film's refs/: made again below
                    return out
                shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)
                for sc in F.Film.load(target).scenes:  # the copy remembers where its start images came from
                    d = (target / sc.dir).resolve()
                    if sc.dir != "." and not (d / T.REFS_DIR).exists():
                        (target / T.REFS_DIR).mkdir(exist_ok=True)
                        os.symlink(os.path.relpath(target / T.REFS_DIR, d), d / T.REFS_DIR, target_is_directory=True)
                    if (d / T.TREE_FILE).is_file():
                        T.save(d, T.load(d))
                bridge.copied.emit(str(target), "")
            except Exception as e:  # noqa: BLE001
                if not existed:                       # a half copy would block the next try
                    shutil.rmtree(target, ignore_errors=True)
                bridge.copied.emit(str(target), str(e))
        threading.Thread(target=work, daemon=True).start()

    def _copied(self, target: str, error: str):
        source, self._copying = self._copying, None
        self.act_save_as.setEnabled(True)
        self._update_buttons()
        if error:
            self.status.setText(f"Could not copy the project: {error}")
            return
        here = str(self.film.root if self.film else self.project) if self.project is not None else None
        if here != source:
            self.status.setText(f"Copied the project to {target}.")
            return
        if self.open_project((Path(target) / getattr(self, "_copy_scene", ".")).resolve(),
                             scene=getattr(self, "_copy_scene_id", None)):
            self.status.setText(f"Saved as {Path(target).name}; you are working in the copy now.")

    def close_project(self):
        if self.running or self.project is None:
            return
        if not self._may_leave_project():
            return
        self._waiting_comfy = self._waiting_refine = None      # a wait for ComfyUI belongs to this project
        self._film_queue = None
        self.preview.stop()
        self.film = None
        self.storyboard.show_scenes([], None)
        self.film_info.setText("")
        self.project = self.tree = None
        self.project_edit.clear()
        self.qs.setValue("project", "")
        self.focus = self._path_end = None
        self.timeline.show_path([], None)
        self.graph.set_focus(None)
        for c in self.cards:
            self.card_stack.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        self.start_tile.clear_frame("Click or drop an image")
        self.size_info.setText("")
        self.refs.clear()
        self.graph.show_nodes([])
        self.path_info.setText("")
        self.tree_info.setText("")
        self._update_title()
        self._update_buttons()
        self.status.setText("No project open. New… or Open… under Project, or the File menu.")

    def open_project(self, folder: Path, quiet: bool = False, scene: str | None = None) -> bool:
        if self.running:
            return False
        if self.project is not None and not self._may_leave_project():
            return False
        if scene is None:
            self._film_queue = None                # (Generate film opens its scenes with scene=…)
        self._waiting_comfy = self._waiting_refine = None
        folder = Path(folder).expanduser().resolve()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            root = F.root_of(folder)
            film = F.Film.load(root)               # the film this folder belongs to, and which scene opens
            if len(film.scenes) > 1 and not film.problem:
                try:
                    film.update_continues()        # scenes that continue the one before: its current end
                except (OSError, ValueError):
                    pass
            if scene is not None and any(x.id == scene for x in film.scenes):
                film.current = scene
                folder = film.path(scene)
            elif folder == root:
                folder = film.path(film.current)
            else:
                film.current = (film.scene_of(folder) or film.scenes[0]).id
            tree = T.load(folder)
        except (OSError, ValueError) as e:
            msg = f"Could not open the project: {e}"
            self.status.setText(msg)
            if not quiet:
                QMessageBox.warning(self, "Promptwood", msg + "\n\nThe files were not changed.")
            return False
        if not (folder / T.TREE_FILE).exists() and not tree.nodes:
            try:                                 # a new project starts with the last used settings
                last = json.loads(self.qs.value("lastOptions", "{}", type=str))
                tree.options.update({k: v for k, v in last.items()
                                     if k in T.DEFAULT_OPTIONS and k not in ("identity", "cast", "style", "audio_refs")})
            except ValueError:
                pass
        self.project, self.tree = folder, tree
        self.film = film
        self.focus = self._path_end = None         # (ids like n3 exist in every scene: no focus carried over)
        synced = ""
        if len(film.scenes) > 1 and film.shared_state and film.shared(tree.options) != film.shared_state:
            tree.options.update(json.loads(json.dumps(film.shared_state)))   # it missed a change of the film's cast
            synced = " Its cast and sound settings were brought up to the film's."
        self._shared_sent = json.dumps(film.shared(tree.options), sort_keys=True)
        self.project_edit.setText(str(root))
        self.project_edit.setToolTip(str(folder))
        self.preview.stop()
        self.live, self.progress_of = {}, {}
        self._options_to_widgets()
        self._show_start_image()
        self._show_references()
        self._show_sounds()
        self._show_tracks()
        if not tree.nodes:
            tree.selected = tree.add(None, "").id
        if tree.selected is None:
            tree.selected = START
        self._tree_stamp = self._stamp()
        self._write_tree()
        T.ensure_handovers(folder, tree)          # e.g. deleted by hand: the clips after them stay done
        self.qs.setValue("project", str(root))
        self._remember_project(root)
        try:
            film.save()                            # which scene was open (only once there are several)
        except OSError:
            pass
        self._show_storyboard()
        self._update_title()
        self._warm_lora_hashes()
        self.refresh(rebuild_path=True)
        QTimer.singleShot(0, self.graph.fit)
        self.status.setText(f"Project {folder.name}: {len(tree.nodes)} clip{'s' if len(tree.nodes) != 1 else ''}."
                            + synced + (f" Note: {film.problem}; the film's scene list is not changed until it is "
                                        "fixed." if film.problem else ""))
        return True

    def _open_dir(self):
        if self.project and self.project.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project)))

    def _options_to_widgets(self):
        o = self.tree.options
        self._loading = True
        try:
            for k in range(self.frames.count() - 1, len(FRAME_CHOICES) - 1, -1):   # an old value added earlier
                self.frames.removeItem(k)
            fr = int(o.get("frames", 73))
            if fr in FRAME_CHOICES:
                self.frames.setCurrentIndex(FRAME_CHOICES.index(fr))
            else:
                select_frames(self.frames, fr)
            i = self.resolution.findData(int(o.get("resolution", 512)))
            self.resolution.setCurrentIndex(i if i >= 0 else 1)
            self.aspect.setCurrentIndex(max(0, self.aspect.findData(o.get("aspect", "16:9"))))
            self.steps.setValue(int(o.get("steps", 28)))
            self.audio.setChecked(bool(o.get("audio", True)))
            ident = o.get("identity") or {}
            self.ref_size.setCurrentIndex(max(0, self.ref_size.findData(ident.get("size", "match"))))
            self.ref_hint.setChecked(bool(ident.get("hint", True)))
            self.dialogue_lang.setCurrentText(ident.get("language") or "English")
            topt = TTS.options(o)
            self.tts_provider.setCurrentIndex(max(0, self.tts_provider.findData(topt["provider"])))
            if self.eleven_model.findData(topt["eleven_model"]) < 0:
                self.eleven_model.addItem(topt["eleven_model"], topt["eleven_model"])
            self.eleven_model.setCurrentIndex(self.eleven_model.findData(topt["eleven_model"]))
            self._tts_form.setRowVisible(self._eleven_row, topt["provider"] == "elevenlabs")
            turbo = o.get("turbo") or {}
            self.speed.setCurrentIndex(self.speed.findData(
                "turbo" if turbo.get("on") else "hyperflow" if (o.get("hyperflow") or {}).get("on")
                else "pika" if (o.get("pika") or {}).get("on") else "normal"))
            self.turbo_steps.setValue(int(turbo.get("steps", 6)))
            self.turbo_strength.setValue(float(turbo.get("strength", 1.0)))
            self._cfg_mode = self.speed.currentData()
            self.clip_cfg.setValue(float(self._mode_options(o, self._cfg_mode).get("cfg", 1.0)))
            self._cfg_hint()
            self._fill_turbo(turbo.get("lora"))
            self.keep_rep.setChecked(bool(o.get("keep_repeated", False)))
            self.veda.setChecked(bool((o.get("veda") or {"on": True}).get("on", True)))
            style = o.get("style") or {}
            self.style_prefix.setText(style.get("prefix") or "")
            self.style_suffix.setText(style.get("suffix") or "")
            self.lora_triggers.setChecked(bool(o.get("lora_triggers", True)))
        finally:
            self._loading = False

    @staticmethod
    def _mode_options(options: dict, mode: str) -> dict:
        """Where a speed mode keeps its settings (its CFG): the project options for Normal."""
        return options if mode == "normal" else (options.get(mode) or {})

    def _cfg_hint(self):
        v = self.clip_cfg.value()
        name = {"normal": "Normal", "turbo": "Turbo", "hyperflow": "HyperFlow", "pika": "Pika"}.get(
            self.speed.currentData(), "")
        self.cfg_note.setText(f"for {name}: off (one pass per step)" if abs(v - 1) < 1e-6 else
                              f"for {name}: two passes per step (about twice as slow)")

    def _options_changed(self, *_):
        if self._loading or self.tree is None or self.running:
            return
        before = self._done_clips()
        mode = self.speed.currentData()
        if mode != self._cfg_mode:                  # another speed mode: its own CFG into the field (not this one's)
            self._cfg_mode = mode
            self.clip_cfg.blockSignals(True)
            self.clip_cfg.setValue(float(self._mode_options(self.tree.options, mode).get("cfg", 1.0)))
            self.clip_cfg.blockSignals(False)
        cfg = round(self.clip_cfg.value(), 2)
        self._cfg_hint()
        self.tree.options.update({
            "frames": self.frames.currentData(), "resolution": self.resolution.currentData(),
            "aspect": self.aspect.currentData(),
            "steps": self.steps.value(), "audio": self.audio.isChecked(), "keep_repeated": self.keep_rep.isChecked(),
            "turbo": {**(self.tree.options.get("turbo") or {}),
                      "on": mode == "turbo", "lora": self.turbo_lora.currentData(),
                      "steps": self.turbo_steps.value(), "strength": round(self.turbo_strength.value(), 2)},
            "hyperflow": {**(self.tree.options.get("hyperflow") or {"strength": 1.0}), "on": mode == "hyperflow"},
            "pika": {**(self.tree.options.get("pika") or {"strength": 1.0}), "on": mode == "pika"},
            "veda": {"on": self.veda.isChecked()}})
        if mode == "normal":                        # the CFG belongs to the speed mode it was set for
            self.tree.options["cfg"] = cfg
        else:
            self.tree.options[mode]["cfg"] = cfg
        self._update_size_info()
        self._show_references()
        self._turbo_widgets()
        self._refresh_card_choices()
        self._changed()
        if not self._report_remade(before, "Project settings changed"):
            self.status.setText("Project settings changed.")

    def _style_changed(self):
        """The style text (before / after every prompt) was edited: it counts like a project setting."""
        if self._loading or self.tree is None or self.running:
            return
        new = {"prefix": self.style_prefix.text().strip(), "suffix": self.style_suffix.text().strip()}
        old = self.tree.options.get("style") or {}
        if {k: (old.get(k) or "").strip() for k in new} == new:
            return
        before = self._done_clips()
        self.tree.options["style"] = new
        self._changed()
        self._update_cards()
        if not self._report_remade(before, "Style text changed"):
            self.status.setText("Style text changed.")

    def _lora_triggers_changed(self, on: bool):
        if self._loading or self.tree is None or self.running:
            return
        before = self._done_clips()
        self.tree.options["lora_triggers"] = bool(on)
        self._changed()
        self._update_cards()
        if not self._report_remade(before, "Trigger words " + ("on" if on else "off")):
            self.status.setText("LoRA trigger words are " + ("put before the prompts." if on else "not used."))

    # ---- LoRA trigger words (triggers.py), looked up in the background
    def _trigger_paths(self) -> list[str]:
        """The LoRAs to know the trigger words of: those found in ComfyUI's folders, and those the clips use
        (from anywhere)."""
        paths = [p for p, _n in self.h3_loras + self.qwen_loras]
        for items in self.image_loras.values():
            paths += [p for p, _n in items]
        if self.tree is not None:
            for n in self.tree.nodes.values():
                paths += [p for p, _s in T.node_loras(n)]
        return list(dict.fromkeys(p for p in paths if p))

    def _scan_triggers(self):
        """Read the trigger words of LoRAs not known yet (their files; CivitAI if allowed), in the background."""
        if getattr(self, "_trigger_scan", None) is not None or not self._trigger_paths():
            return
        paths, online, bridge = self._trigger_paths(), self.triggers_online.isChecked(), self.bridge
        self._trigger_scan = threading.Event()
        self._trigger_scan_online = online
        stop = self._trigger_scan

        def work():
            counts = {}
            try:
                from .. import triggers
                counts = triggers.db().scan(paths, online=online, stop=stop.is_set)
            except Exception:  # noqa: BLE001
                counts = {}
            try:
                bridge.triggers.emit(counts)
            except RuntimeError:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _video_gpu_changed(self, on: bool):
        self.qs.setValue("videoGpu", on)           # at once: read at the next start, also after a crash
        self.qs.sync()
        self.status.setText("Videos play on the " + ("GPU" if on else "CPU") + " after the app is restarted.")

    def llm_model_path(self) -> str:
        i = self.llm_model.currentIndex()
        typed = self.llm_model.currentText().strip()
        data = self.llm_model.itemData(i) if i >= 0 else None
        return data if data and self.llm_model.itemText(i) == typed else os.path.expanduser(typed)

    def llama_server(self):
        """The llama.cpp server of the refiner (one per app; renewed when its settings change), None when off."""
        if not self.llm_on.isChecked() or not self.llm_model_path():
            return None
        from ..llm_server import LlamaServer
        from ..llm_server import find_mmproj
        key = (self.llm_server.text().strip() or LLM_DEFAULT_SERVER, self.llm_model_path(), self.llm_port.value(),
               self.llm_cpu_moe.isChecked(), find_mmproj(self.llm_model_path()))
        cur = getattr(self, "_llama", None)
        if cur is None or (cur.server, cur.model, cur.port, cur.cpu_moe, cur.mmproj) != key:
            if cur is not None:
                cur.stop(wait=False)
            self._llama = LlamaServer(*key, log_dir=Path.home() / ".cache" / "video-chain")
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        self._llama.before_launch = lambda llama=self._llama: self._room_for_llama(llama, url)
        return self._llama

    def _room_for_llama(self, llama, url: str, settle: float = 15.0) -> None:
        """(start()'s thread, just before the llama.cpp server starts) Too little free VRAM for it: ComfyUI is asked
        to unload its models (the clip / image model loads again when it is needed). While clips are generated it
        refuses instead (their run keeps its VRAM)."""
        from ..llm_server import LlmServerError, vram_needed_mb
        self._make_room(vram_needed_mb(llama.model, llama.cpu_moe, llama.mmproj), url, "the llama.cpp model",
                        LlmServerError, "refine after the run, or choose a ComfyUI model", settle)

    def _make_room(self, need: int, url: str, what: str, error=RuntimeError, later: str = "try again after the run",
                   settle: float = 15.0) -> None:
        """(a worker thread, before a model of this app starts) Too little free VRAM for need MB: ComfyUI is asked to
        unload its models; while clips are generated: error instead (their run keeps its VRAM)."""
        from ..llm_server import free_vram_mb
        free = free_vram_mb()
        if free is None or free >= need:
            return
        gb = lambda mb: f"{mb / 1024:.1f} GB"   # noqa: E731
        if self.running:
            raise error(f"only {gb(free)} of VRAM is free while clips are generated and {what} needs about "
                        f"{gb(need)}; {later}")
        try:
            req = urllib.request.Request(url + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:  # noqa: BLE001  (ComfyUI is not up: the VRAM is used by something else)
            return
        end = time.time() + settle
        now = free
        while time.time() < end:
            time.sleep(0.5)
            now = free_vram_mb()
            if now is None or now >= need:
                break
        self.bridge.llm_room.emit(
            f"ComfyUI's models were unloaded for {what} (it needs about {gb(need)} of VRAM, "
            f"{gb(free)} was free" + (f", now {gb(now)}" if now is not None else "") +
            "); the clip model loads again before the next clips.")

    def _llm_room_made(self, msg: str):
        self._comfy_model = None                    # (nothing of this app is loaded in ComfyUI any more)
        self._qwen_timer.stop()
        if self.worker_alive and not self.running:
            self._quit_worker(free=False)           # its model is gone; a new worker starts for the next clips
        self._update_model_label()
        self.log.appendPlainText(msg)
        self.status.setText(msg)

    def stop_llama(self, why: str = "") -> bool:
        """Stop the refiner's llama.cpp server if the app started it (its RAM / VRAM back for ComfyUI)."""
        cur = getattr(self, "_llama", None)
        stopped = bool(cur is not None and cur.stop(wait=False))     # (the window does not wait for it)
        if stopped and why:
            self.log.appendPlainText(f"Stopped the llama.cpp server ({why}).")
        return stopped

    def _show_civitai_key(self):
        from .. import triggers
        has = bool(triggers.api_key())
        self.civitai_key.clear()
        self.civitai_key.setPlaceholderText("a key is kept (type a new one, or save empty to remove it)" if has else
                                            "none: paste your key to get everything CivitAI holds back")

    def _save_civitai_key(self):
        from .. import triggers
        key = self.civitai_key.text().strip()
        try:
            triggers.set_api_key(key)
        except RuntimeError as e:
            QMessageBox.warning(self, "CivitAI API key", f"Could not keep the key: {e}")
            return
        self._show_civitai_key()
        if key:
            self.status.setText("CivitAI API key kept in the system keyring; LoRAs are looked up once more with it.")
            if self.triggers_online.isChecked():
                self._scan_triggers()
        else:
            self.status.setText("CivitAI API key removed.")

    def _show_eleven_key(self):
        has = bool(TTS.eleven_key())
        self.eleven_key.clear()
        self.eleven_key.setPlaceholderText("a key is kept (type a new one, or save empty to remove it)" if has else
                                           "none: paste your key for text to speech with ElevenLabs")

    def _save_eleven_key(self):
        key = self.eleven_key.text().strip()
        try:
            TTS.set_eleven_key(key)
        except RuntimeError as e:
            QMessageBox.warning(self, "ElevenLabs API key", f"Could not keep the key: {e}")
            return
        self._eleven_voices = None
        self._show_eleven_key()
        self.status.setText("ElevenLabs API key kept in the system keyring." if key else "ElevenLabs API key removed.")

    def _tts_changed(self, *_):
        if self._loading or self.tree is None:
            return
        provider = self.tts_provider.currentData()
        self._tts_form.setRowVisible(self._eleven_row, provider == "elevenlabs")
        new = {"provider": provider, "eleven_model": self.eleven_model.currentData() or
               TTS.DEFAULT_OPTIONS["eleven_model"]}
        if TTS.options(self.tree) != new:
            self.tree.options["tts"] = new
            self._changed()                         # (a spoken track is made with it; clips do not depend on it)
        if provider == "elevenlabs" and not TTS.eleven_key():
            self.status.setText("ElevenLabs needs your API key: Settings → ElevenLabs API key.")

    def _load_eleven_models(self):
        def work():
            try:
                self.bridge.eleven_models.emit(TTS.ElevenLabs().models(), "")
            except Exception as e:  # noqa: BLE001
                self.bridge.eleven_models.emit(None, str(e))
        self.status.setText("Asking ElevenLabs for its models…")
        threading.Thread(target=work, daemon=True).start()

    def _fill_eleven_models(self, models, err: str):
        if models is None:
            self.status.setText(f"Could not ask ElevenLabs: {err}")
            return
        cur = self.eleven_model.currentData()
        self.eleven_model.blockSignals(True)
        self.eleven_model.clear()
        for m in models:
            self.eleven_model.addItem(f"{m['name']} ({m['id']})", m["id"])
        if self.eleven_model.findData(cur) < 0:
            self.eleven_model.addItem(cur, cur)
        self.eleven_model.setCurrentIndex(self.eleven_model.findData(cur))
        self.eleven_model.blockSignals(False)
        self.status.setText(f"ElevenLabs offers {len(models)} text to speech models.")

    # ---- text to speech
    def local_tts(self) -> "TTS.LocalTTS":
        """The app's Qwen3-TTS worker (started on demand; stop_tts unloads it)."""
        if self._tts_local is None:
            d = Path.home() / ".cache" / "video-chain"
            d.mkdir(parents=True, exist_ok=True)
            self._tts_local = TTS.LocalTTS(log=d / "tts-worker.log")
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        self._tts_local.before_launch = lambda _w: self._make_room(TTS.LOCAL_VRAM_MB, url, "Qwen3-TTS",
                                                                   TTS.TtsError, "speak the lines after the run")
        return self._tts_local

    def stop_tts(self, why: str = "") -> bool:
        stopped = bool(self._tts_local is not None and self._tts_local.stop())
        if stopped and why:
            self.log.appendPlainText(f"Stopped Qwen3-TTS ({why}).")
        return stopped

    def _tts_engine(self):
        return TTS.engine(self.tree, self.local_tts())

    def speak_clip(self, nid: str) -> bool:
        """Speak a clip's lines with text to speech (in a thread); _lines_spoken makes them its sound."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return False
        if self._tts_busy:
            self.status.setText("Text to speech is busy; wait until it is done.")
            return False
        node = self.tree.nodes[nid]
        if node.frozen:
            self._frozen_block(node, "speak")
            return False
        eng = self._tts_engine()
        bad = TTS.line_problems(self.tree, node, eng.name) + eng.problems(
            transcribe=eng.name == "local" and TTS.needs_transcript(self.tree, node))
        if bad:
            self.status.setText(f"Cannot speak the lines of clip #{nid[1:]}: " + "; ".join(bad) + ".")
            return False
        snap, project = copy.deepcopy(self.tree), self.project
        self._tts_busy = True
        self._update_buttons()
        self.status.setText(f"Speaking the lines of clip #{nid[1:]} ({TTS.PROVIDERS[eng.name]})…")

        def work():
            try:
                r = TTS.render_lines(project, snap, nid, eng, log=self.bridge.tts_log.emit)
                self.bridge.spoken.emit(nid, r, "", str(project))
            except Exception as e:  # noqa: BLE001
                self.bridge.spoken.emit(nid, None, str(e), str(project))
            finally:
                if eng.name == "local":
                    eng.stop()                      # (unloaded right after: its VRAM back for ComfyUI)
        threading.Thread(target=work, daemon=True).start()
        return True

    def _lines_spoken(self, nid: str, r, err: str, project: str):
        self._tts_busy = False
        self._update_buttons()
        if self.project is None or str(self.project) != project or self.tree is None or nid not in self.tree.nodes:
            if r:                                   # (another project now, or the clip is gone: not used)
                (Path(project) / r["file"]).unlink(missing_ok=True)
            return
        if r is None:
            self.status.setText(f"Could not speak the lines of clip #{nid[1:]}: {err}")
            return
        node = self.tree.nodes[nid]
        if node.frozen:
            (self.project / r["file"]).unlink(missing_ok=True)
            self._frozen_block(node, "speak")
            return
        try:
            done = TTS.apply_lines(self.project, self.tree, nid, r)
        except (TTS.TtsError, OSError) as e:
            (self.project / r["file"]).unlink(missing_ok=True)
            self.status.setText(f"Could not use the spoken lines of clip #{nid[1:]}: {e}")
            return
        self._write_tree()
        self.refresh(rebuild_path=True)
        msg = (f"Clip #{nid[1:]}: {done['lines']} line{'s' if done['lines'] != 1 else ''} spoken ({done['secs']:.1f} s)"
               f" and made its sound (track “{T.track_of(self.tree, done['track'])['name']}”); the picture is made "
               "to it, lips in sync.")
        if done["frames"]:
            msg += f" The clip is now {done['frames'] / 24:.1f} s long, so the lines fit."
        if r.get("prompt") != node.prompt:
            msg += " (The prompt was changed meanwhile: speak again if its lines changed.)"
        self.status.setText(msg)

    def _voice_with_tts(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        if self._tts_busy:
            self.status.setText("Text to speech is busy; wait until it is done.")
            return
        opts = TTS.options(self.tree)
        desc = c.get("desc") or ""
        dlg = VoiceTtsDialog(self, c["name"], desc, TTS.language(self.tree), opts["provider"], opts["eleven_model"],
                             play=lambda path, title: self.preview.open(Path(path), title),
                             local=lambda w=self.local_tts(): w, eleven=TTS.ElevenLabs,
                             voices_cache=self._eleven_voices)
        self._tts_busy = True
        try:
            ok = run_voice_dialog(dlg)
            self._eleven_voices = getattr(dlg, "voices", None) or self._eleven_voices
            if ok and dlg.made is not None and dlg.wav.is_file():
                before = self._done_clips()
                try:
                    entry = TTS.use_sample(self.project, self.tree, c["name"], dlg.wav, dlg.made["label"],
                                           dlg.made["text"], dlg.made["eleven"])
                except (ValueError, OSError, RuntimeError) as e:
                    self.status.setText(f"Could not set the voice: {e}")
                    return
                self._show_references(select=c["name"])
                self._write_tree()
                self._changed()
                msg = f"Voice of @{c['name']} set with text to speech ({entry['secs']:.1f} s)."
                self.status.setText(msg)
                if self._report_remade(before, f"Voice of @{c['name']} set"):
                    self.status.setText(msg + " " + self.status.text())
        finally:
            self._tts_busy = False
            self.preview.stop()                     # (it may still play the sample, whose folder goes now)
            dlg.cleanup()
            self.stop_tts("the voice window was closed")

    def _triggers_found(self, counts: dict):
        self._trigger_scan = None
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.loras.refresh_triggers()
        changed = counts.get("new") or counts.get("looked")
        if self.tree is not None and changed and not self.running:
            self.refresh()                         # clips record their LoRAs' words
        if self.triggers_online.isChecked() and not getattr(self, "_trigger_scan_online", True):
            QTimer.singleShot(0, self._scan_triggers)     # switched on while an offline scan ran
        if counts.get("key_rejected"):
            self.status.setText("CivitAI rejected your API key: LoRAs are looked up without it. Check or renew the "
                                "key in the settings (CivitAI → Account settings → API Keys).")
        elif counts.get("looked"):
            self.status.setText(f"LoRA trigger words: looked up {counts['looked']} LoRA"
                                f"{'s' if counts['looked'] != 1 else ''} on CivitAI, {counts.get('found', 0)} with "
                                "trigger words" + (f", {counts['failed']} could not be asked (offline?)"
                                                   if counts.get("failed") else "") + ".")

    # ---- the cast (characters with face pictures, @Name in prompts)
    def _identity_changed(self, *_):
        if self._loading or self.tree is None or self.running:
            return
        ident = dict(self.tree.options.get("identity") or {})
        lang = self.dialogue_lang.currentText().strip()
        if not lang:                                # an emptied field: back to the language in use
            lang = ident.get("language") or "English"
            self.dialogue_lang.setCurrentText(lang)
        new = {"size": self.ref_size.currentData(), "hint": self.ref_hint.isChecked(), "language": lang}
        if all(ident.get(k) == v for k, v in new.items()):
            return
        before = self._done_clips()
        ident.update(new)
        self.tree.options["identity"] = ident
        self._show_references()
        self._changed()
        self._report_remade(before, "Face settings changed")

    def _cast_names(self) -> list[str]:
        return [c["name"] for c in T.cast_of(self.tree)] if self.tree is not None else []

    def _selected_character(self) -> dict | None:
        item = self.cast_list.currentItem()
        return T.character(self.tree, item.data(Qt.UserRole)) if item is not None and self.tree else None

    def _mentioning(self, name: str) -> list[str]:
        """Clips whose prompt (with the style text) mentions @name."""
        if self.tree is None:
            return []
        low = name.lower()
        return [n.id for n in self.tree.nodes.values() if not n.is_video and any(
            c["name"].lower() == low for c in T.mentions(self.tree.options,
                                                        T.full_prompt(self.tree.options, n.prompt))[0])]

    def _cast_edit_ok(self) -> bool:
        if self.project is None or self.tree is None:
            self.status.setText("Open or create a project folder first.")
            return False
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return False
        return True

    def _new_character(self):
        if not self._cast_edit_ok():
            return
        name, ok = QInputDialog.getText(self, "New character", "Name (you mention it in prompts as @Name):",
                                        text=T.next_character_name(self.tree))
        if not ok:
            return
        try:
            c = T.add_character(self.project, self.tree, name)
        except ValueError as e:
            self.status.setText(f"Could not add the character: {e}")
            return
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        self.status.setText(f"@{c['name']} added. Now add a face: Image…, Start image or Selected clip.")

    def _rename_character(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        new, ok = QInputDialog.getText(self, "Rename character", f"New name for @{c['name']}:", text=c["name"])
        if not ok or new.strip() == c["name"]:
            return
        try:
            T.check_name(self.tree, new, keep=c["name"])
        except ValueError as e:
            self.status.setText(f"Could not rename: {e}")
            return
        users = self._mentioning(c["name"])
        made = [n for n in users if self.tree.nodes[n].made]
        if made and QMessageBox.question(
                self, "Rename character",
                f"{len(users)} clip{'s mention' if len(users) != 1 else ' mentions'} @{c['name']}; the prompts "
                f"change to @{new.strip()}. The name is part of what the model reads, so {len(made)} made "
                f"clip{'s' if len(made) != 1 else ''} will count as CHANGED.") != QMessageBox.Yes:
            return
        before = self._done_clips()
        try:
            T.rename_character(self.tree, c["name"], new)
        except ValueError as e:
            self.status.setText(f"Could not rename: {e}")
            return
        self._show_references(select=new.strip())
        self._write_tree()
        style, self._loading = self.tree.options.get("style") or {}, True
        try:
            self.style_prefix.setText(style.get("prefix") or "")
            self.style_suffix.setText(style.get("suffix") or "")
        finally:
            self._loading = False
        self._rebuild_path()
        self._changed()
        if not self._report_remade(before, "Character renamed"):
            self.status.setText(f"Renamed to @{new.strip()}.")

    def _remove_character(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        users = self._mentioning(c["name"])
        text = (f"Remove @{c['name']} and its {len(c['images'])} face picture{'s' if len(c['images']) != 1 else ''}"
                + (" and its voice?" if c.get("voice") else "?"))
        if users:
            text += (f" {len(users)} clip{'s mention' if len(users) != 1 else ' mentions'} @{c['name']}; "
                     "they cannot be made until their prompts are changed.")
        if QMessageBox.question(self, "Remove character", text) != QMessageBox.Yes:
            return
        before = self._done_clips()
        T.remove_character(self.project, self.tree, c["name"])
        self._pix.clear()
        self._show_references()
        self._write_tree()
        self._changed()
        if not self._report_remade(before, "Character removed"):
            self.status.setText(f"@{c['name']} removed.")

    def _character_desc_changed(self):
        c = self._selected_character()
        text = self.char_desc.text().strip()
        if c is None or self._loading or text == c.get("desc", "") or not self._cast_edit_ok():
            return
        before = self._done_clips()
        T.update_character(self.tree, c["name"], desc=text)
        self._write_tree()
        self._changed()
        if not self._report_remade(before, f"Description of @{c['name']} changed"):
            self.status.setText(f"Description of @{c['name']} saved (used by clips that mention it).")

    def _character_lead_changed(self, on: bool):
        c = self._selected_character()
        if c is None or self._loading or bool(c.get("lead")) == on or not self._cast_edit_ok():
            return
        before = self._done_clips()
        T.update_character(self.tree, c["name"], lead=on)
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        if not self._report_remade(before, "Lead changed"):
            self.status.setText(f"@{c['name']} is the lead now." if on else "No lead: clips that mention nobody "
                                                                               "are made without a face.")

    def _done_clips(self) -> set[str]:
        return {nid for nid, st in self.status_of.items() if st == "done"}

    def _report_remade(self, before: set[str], what: str) -> int:
        """Say how many made clips a change turned CHANGED (they are made again with their paths), and how many
        are done again (a setting set back)."""
        if self.tree is None:
            return 0
        try:
            checker = T.Checker(self.project, self.tree)
            now = {n for n in self.tree.nodes if checker.status(n) == "done"}
        except Exception:  # noqa: BLE001
            return 0
        lost = [n for n in before if n in self.tree.nodes and n not in now]
        back = [n for n in now if n not in before and n in self.status_of]   # e.g. a setting set back
        parts = []
        if lost:
            parts.append(f"{len(lost)} made clip{'s are' if len(lost) != 1 else ' is'} now CHANGED and will be made "
                         "again with their paths")
        if back:
            parts.append(f"{len(back)} clip{'s are' if len(back) != 1 else ' is'} DONE again")
        if parts:
            self.status.setText(f"{what}: " + "; ".join(parts) + ".")
        return len(lost) + len(back)

    def _reference_from_selected(self):
        """The selected video's first frame, or a generated clip's last frame."""
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if nid else None
        if node is None or not node.made:
            self.status.setText("Select a converted video or a made clip first.")
            return
        self.reference_from_node(nid, "first" if node.is_video else "last")

    def _add_reference(self):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        f = pick_image(self, "Reference image (best a close-up of the face)", self.qs, None, self._picker_places(),
                       self.project_images())
        if f:
            self.add_reference(f)

    def _start_image_there(self) -> bool:
        return bool(self.project and (self.project / T.START_FILE).is_file())

    def _reference_from_start(self):
        if self.project is not None and self._start_image_there():
            self.add_reference(str(self.project / T.START_FILE))

    def reference_from_node(self, nid: str, which: str) -> bool:
        """A clip's first (videos only) or last frame as a face reference."""
        node = self.tree.nodes.get(nid) if self.tree else None
        rel = (node.made or {}).get("first" if which == "first" else "frame") if node else None
        if not rel:
            self.status.setText("That clip is not made yet.")
            return False
        if self.add_reference(str(self.project / rel)):
            if "CHANGED" not in self.status.text():                            # keep the CHANGED count
                self.status.setText(f"The {which} frame of #{nid[1:]}: {self.status.text()}")
            return True
        return False

    def face_detector(self) -> Path | None:
        from .. import faces as F
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        return F.find_detector(Path(comfy_dir)) if comfy_dir else None

    def crop_face(self, path: str, title: str, crop=None):
        """The crop window: the square (x, y, side), None for the whole picture, or False (cancelled)."""
        try:
            dlg = FaceCropDialog(QApplication.activeModalWidget() or self, Path(path), self.face_detector(), title,
                                 crop)                # over the video / image window it was opened from
        except ValueError as e:
            self.status.setText(str(e))
            return False
        try:
            return dlg.crop() if dlg.exec() == QDialog.Accepted else False
        finally:
            dlg.deleteLater()

    def create_reference(self, dialog: bool = True):
        """The Create image window for a face picture of the selected character (its "Add as face reference")."""
        c = self._selected_character()
        return self.edit_image(mode="create", dialog=dialog, character=c["name"] if c else None)

    def add_reference(self, path: str, crop="ask") -> bool:
        """A face picture for the selected character (none selected: the lead, or a new character). crop: "ask"
        opens the crop window, else the square (x, y, side) or None (the whole picture)."""
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return False
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return False
        if QImage(path).isNull():
            self.status.setText(f"Not an image: {Path(path).name}")
            return False
        target = self._selected_character()
        owner = target or T.lead_of(self.tree) or (T.cast_of(self.tree) or [None])[0]
        if owner is not None and len(owner["images"]) >= T.MAX_PICTURES_PER_CHARACTER:
            self.status.setText(f"@{owner['name']} has {T.MAX_PICTURES_PER_CHARACTER} face pictures already; remove "
                                "one first (or select another character).")
            return False
        if len(T.all_pictures(self.tree)) >= T.MAX_CAST_PICTURES:
            self.status.setText(f"The cast has {T.MAX_CAST_PICTURES} face pictures, the most the model takes; "
                                "remove one first.")
            return False
        if crop == "ask":
            who = f"@{target['name']}" if target else (f"@{T.lead_of(self.tree)['name']}" if T.lead_of(self.tree)
                                                       else "a new character")
            crop = self.crop_face(path, f"Face picture of {who}")
            if crop is False:
                self.status.setText("Face not added.")
                return False
        before = self._done_clips()
        known = set(self._cast_names())
        try:
            rel = T.add_reference(self.project, Path(path), self.tree, target["name"] if target else None, crop=crop)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not add the face: {e}")
            return False
        owner = next(c["name"] for c in T.cast_of(self.tree) if rel in c["images"])
        self._show_references(select=owner)
        self._write_tree()
        self._changed()
        self.status.setText(f"Face added to @{owner}." if owner in known else
                            f"New character @{owner} with this face (rename it with Rename…).")
        self._report_remade(before, f"Face added to @{owner}")
        return True

    def _recrop_reference(self):
        item = self.refs.currentItem()
        if item is None or not self._cast_edit_ok():
            return
        rel = item.data(Qt.UserRole)
        full = self.project / T.source_of(rel)
        crop = self.crop_face(str(full if full.is_file() else self.project / rel), "Crop the face picture again")
        if crop is False:
            return
        before = self._done_clips()
        try:
            T.recrop_picture(self.project, self.tree, rel, crop)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not crop: {e}")
            return
        self._pix.clear()
        c = self._selected_character()
        self._show_references(select=c["name"] if c else None)
        self._write_tree()
        self._changed()
        if not self._report_remade(before, "Face picture cropped"):
            self.status.setText("Face picture cropped.")

    def _remove_reference(self):
        item = self.refs.currentItem()
        if item is None or not self._cast_edit_ok():
            return
        rel = item.data(Qt.UserRole)
        before = self._done_clips()
        made = [n for n in before if n in self.tree.nodes and not self.tree.nodes[n].is_video
                and rel in T.clip_cast(self.tree.options, self.tree.nodes[n].prompt)["images"]]
        if made and QMessageBox.question(
                self, "Remove face picture",
                f"Remove this face picture? {len(made)} made clip{'s use' if len(made) != 1 else ' uses'} it and "
                f"will count as CHANGED.") != QMessageBox.Yes:
            return
        T.remove_picture(self.project, self.tree, rel)
        self._pix.clear()
        c = self._selected_character()
        self._show_references(select=c["name"] if c else None)
        self._write_tree()
        self._changed()
        self._report_remade(before, "Face picture removed")

    def _show_references(self, select: str | None = None):
        """The cast list (keeps the selected character), then the selected one's fields and pictures."""
        cur = select or (self.cast_list.currentItem().data(Qt.UserRole) if self.cast_list.currentItem() else None)
        cast = T.cast_of(self.tree) if self.tree else []
        self.cast_list.blockSignals(True)
        self.cast_list.clear()
        row = 0
        for k, c in enumerate(cast):
            pm = self._pixmap(self.project / c["images"][0]) if self.project and c["images"] else None
            n = len(c["images"])
            text = (f"@{c['name']}   {n} face{'s' if n != 1 else ''}" if n else f"@{c['name']}   no face yet") + \
                ("   · voice" if c.get("voice") else "") + ("   · lead" if c.get("lead") else "")
            item = QListWidgetItem(QIcon(pm) if pm else QIcon(), text)
            item.setData(Qt.UserRole, c["name"])
            item.setToolTip((c.get("desc") or "") + (f" ({len(self._mentioning(c['name']))} clips mention it)"))
            self.cast_list.addItem(item)
            if cur and c["name"].lower() == cur.lower():
                row = k
        if cast:
            self.cast_list.setCurrentRow(row)
        self.cast_list.blockSignals(False)
        self._show_character()
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.mentions.rehighlight()
        self._update_cards() if self.cards else None
        lead = T.lead_of(self.tree) if self.tree else None
        if not cast:
            text = ("No characters: each clip only continues from the last frame of the clip before, so a face can "
                    "drift. Add a face (Image…, Start image, Selected clip, or a video's Edit range…) or New… "
                    "for a named character.")
        else:
            text = (f"{len(cast)} character{'s' if len(cast) != 1 else ''}. Write @Name in a prompt (typing @ "
                    "offers the names): the clip is made with exactly those faces. "
                    + (f"Clips that mention nobody get @{lead['name']}'s face. " if lead and lead["images"] else
                       "Clips that mention nobody get no face. ")
                    + "A character with a voice speaks its lines in quotes with it (\"...\" in the sentence of "
                    "its @Name). LoRAs can act weaker in clips with faces.")
        self.ident_info.setText(text)

    def _show_character(self):
        c = self._selected_character()
        self._loading, was = True, self._loading
        try:
            self.char_desc.setText(c.get("desc", "") if c else "")
            self.char_lead.setChecked(bool(c and c.get("lead")))
        finally:
            self._loading = was
        v = (c or {}).get("voice")
        el = ((c or {}).get("tts") or {}).get("eleven")
        long = f"; cloned from {T.clone_secs(v):.1f} s" if v and v.get("ref") else ""
        self.char_voice.setText((f"Voice: {v.get('name') or Path(v['file']).name} "
                                 f"({float(v.get('secs') or 0):.1f} s{long})"
                                 if v else "Voice: none") + (f" · ElevenLabs: {el['name']}" if el else ""))
        self.refs.clear()
        for k, rel in enumerate(c["images"] if c else [], start=1):
            pm = self._pixmap(self.project / rel) if self.project else None
            item = QListWidgetItem(QIcon(pm) if pm else QIcon(), f"{c['name']} {k}")
            item.setData(Qt.UserRole, rel)
            item.setToolTip(rel)
            self.refs.addItem(item)
        self._update_ref_start_button()
        self._update_cast_buttons()

    def _update_cast_buttons(self):
        """The cast buttons that need a selected character, voice or face picture (and no run)."""
        c = self._selected_character()
        idle = self.project is not None and not self.running
        v = bool(c and c.get("voice"))
        for w in (self.b_char_rename, self.b_char_del, self.char_desc, self.char_lead, self.b_voice_file,
                  self.b_voice_tts):
            w.setEnabled(idle and c is not None)
        self.b_voice_del.setEnabled(idle and v)
        self.b_voice_clone.setEnabled(idle and v)
        self.b_voice_play.setEnabled(v)
        pic = self.refs.currentItem() is not None
        self.b_ref_crop.setEnabled(idle and pic)
        self.b_ref_del.setEnabled(idle and pic)

    # ---- a character's voice
    def _clone_voice_at_eleven(self):
        """Make an instant ElevenLabs voice clone of the selected character's voice sample (in a thread);
        _eleven_cloned makes it the character's ElevenLabs voice."""
        c = self._selected_character()
        if c is None or not c.get("voice") or not self._cast_edit_ok():
            return
        if self._tts_busy:
            self.status.setText("Text to speech is busy; wait until it is done.")
            return
        if not TTS.eleven_key():
            self.status.setText("ElevenLabs needs your API key: Settings → ElevenLabs API key.")
            return
        name, old = c["name"], (c.get("tts") or {}).get("eleven")
        text = (f"Upload the voice sample of @{name} ({T.clone_secs(c['voice']):.1f} s) to your ElevenLabs "
                "account and make an instant voice clone of it?\n\nThe sample leaves this computer and the new voice "
                "takes a voice slot of your plan (instant voice cloning needs a paid plan). Use a voice only with "
                "the speaker's consent.")
        if old:
            text += (f"\n\n@{name} speaks with the ElevenLabs voice “{old['name']}” now; the clone replaces it for "
                     "this character (the old voice stays in your account).")
        if QMessageBox.question(self, "Clone the voice at ElevenLabs", text) != QMessageBox.Yes:
            return
        snap, project = copy.deepcopy(self.tree), self.project
        self._tts_busy = True
        self._update_buttons()
        self.status.setText(f"Cloning the voice of @{name} at ElevenLabs…")

        def work():
            try:
                self.bridge.eleven_cloned.emit(name, TTS.clone_sample(project, snap, name, TTS.ElevenLabs()), "",
                                               str(project))
            except Exception as e:  # noqa: BLE001
                self.bridge.eleven_cloned.emit(name, None, str(e), str(project))
        threading.Thread(target=work, daemon=True).start()

    def _eleven_cloned(self, name: str, voice, err: str, project: str):
        self._tts_busy = False
        self._update_buttons()
        if voice is None:
            self.status.setText(f"Could not clone the voice of @{name} at ElevenLabs: {err}")
            return
        self._eleven_voices = None                  # (the account has one more voice now)
        if (self.project is None or str(self.project) != project or self.tree is None
                or not TTS.set_eleven(self.tree, name, voice)):
            self.status.setText(f"The voice “{voice['name']}” was made in your ElevenLabs account, but @{name} or its "
                                "voice sample changed meanwhile, so it is not used (TTS… can pick it).")
            return
        self._show_character()
        self._write_tree()
        msg = f"@{name} has the ElevenLabs voice “{voice['name']}” now, cloned from its voice sample."
        if TTS.options(self.tree)["provider"] != "elevenlabs":
            msg += " Its lines are spoken with it when ElevenLabs is the text to speech provider."
        self.status.setText(msg)

    def _voice_from_file(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        f = pick_image(self, f"Voice of @{c['name']} (one speaker, 2-{T.VOICE_MAX_SECS:g} s are used)", self.qs,
                       places=self._picker_places(), collections=self.project_media("media"), kind="media")
        if not f:
            return
        mine = not self._tts_busy                   # (the worker that removes music is the text to speech one)
        dlg = self._voice_dialog(f, c["name"], isolate=mine)
        self._tts_busy = self._tts_busy or mine
        try:
            if not dlg.file:
                self.status.setText(f"Could not set the voice: {dlg.info_label.text()}")
                return
            if dlg.exec() == QDialog.Accepted and dlg.file:
                if dlg.cleaned is not None:
                    self.set_character_voice(str(dlg.cleaned), label=f"{Path(dlg.file).name} (music removed)")
                else:
                    self.set_character_voice(dlg.file, start=dlg.start, end=dlg.end)
        finally:
            dlg.cleanup()
            dlg.deleteLater()
            if mine:
                self._tts_busy = False
                self.stop_tts("the voice window was closed")

    def _voice_dialog(self, file: str, name: str, isolate: bool = True):
        """The window the part of a file is chosen in that becomes a character's voice (isolate: it can take the
        background music out with the app's text to speech worker)."""
        from .audio_editor import AudioRangeDialog
        fn, why = None, "Text to speech is busy; choose the voice again when it is done to remove music."
        if isolate:
            w = self.local_tts()
            bad = w.problems(isolate=True)
            fn, why = (None, "Not set up: " + "; ".join(bad)) if bad else (w.isolate, "")
        return AudioRangeDialog(self, file=file, settings=self.qs, title=f"Voice of @{name}: choose the part",
                                voice=True, player=os.environ.get("QT_QPA_PLATFORM") != "offscreen",
                                isolate=fn, no_isolate=why)

    def _voice_from_selected(self):
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if nid else None
        if node is None or not (node.made or {}).get("video"):
            self.status.setText("Select a made clip or a converted video first.")
            return
        what = f"{'video' if node.is_video else 'clip'} #{nid[1:]}"
        self.set_character_voice(str(self.project / node.made["video"]), label=f"sound of {what}")

    def set_character_voice(self, path: str, label: str | None = None, start: float = 0.0,
                            end: float | None = None) -> bool:
        """The voice of the selected character, from an audio or video file (the part from start to end)."""
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return False
        before = self._done_clips()
        try:
            entry = T.set_voice(self.project, self.tree, c["name"], Path(path), start, label=label, end=end)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not set the voice: {e}")
            return False
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        msg = (f"Voice of @{c['name']} set ({entry['secs']:.1f} s"
               + (f", the first {T.VOICE_MAX_SECS:g} s" if entry.get("cut") else "")
               + f"). Clips use it where @{c['name']} says a line in quotes."
               + (f" Text to speech clones it from {entry['ref']['secs']:.1f} s." if entry.get("ref") else ""))
        self.status.setText(msg)
        if self._report_remade(before, f"Voice of @{c['name']} set"):
            self.status.setText(msg + " " + self.status.text())
        return True

    def _remove_voice(self):
        c = self._selected_character()
        if c is None or not c.get("voice") or not self._cast_edit_ok():
            return
        before = self._done_clips()
        T.remove_voice(self.project, self.tree, c["name"])
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        if not self._report_remade(before, f"Voice of @{c['name']} removed"):
            self.status.setText(f"Voice of @{c['name']} removed.")

    def _play_voice(self):
        c = self._selected_character()
        if c is not None and c.get("voice") and self.project is not None:
            self.preview.open(self.project / c["voice"]["file"], f"Voice of @{c['name']}")

    # ---- sound references (<Audio 1> ...)
    def _sound_users(self, upto: int = 1) -> list[str]:
        """Clips whose prompt names <Audio upto> or a later one (they get the sounds up to that one)."""
        if self.tree is None:
            return []
        return [n.id for n in self.tree.nodes.values() if not n.is_video
                and T.audio_mentioned(T.full_prompt(self.tree.options, n.prompt)) >= upto]

    def _add_sound(self):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        f = pick_image(self, "Sound reference (a voice, a soundtrack, a beat: 2-15 s)", self.qs,
                       places=self._picker_places(), collections=self.project_media("media"), kind="media")
        if f:
            self.add_sound(f)

    def _sound_from_selected(self):
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if nid else None
        if node is None or not (node.made or {}).get("video"):
            self.status.setText("Select a made clip or a converted video first.")
            return
        what = f"{'Video' if node.is_video else 'Clip'} #{nid[1:]}"
        self.add_sound(str(self.project / node.made["video"]), name=f"sound of {what.lower()}")

    def add_sound(self, path: str, name: str | None = None) -> bool:
        if self.project is None or self.tree is None or self.running:
            return False
        before = self._done_clips()
        try:
            entry = T.add_audio_reference(self.project, Path(path), self.tree, name=name)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not add the sound: {e}")
            return False
        n = len(T.audio_refs(self.tree))
        self._show_sounds(select=n - 1)
        self._write_tree()
        self._changed()
        msg = (f"Sound reference <Audio {n}> added ({entry['secs']:.1f} s"
               + (", the first part only: all sounds together may have 15 s" if entry.get("cut") else "")
               + f"). Name it in a clip's prompt as <Audio {n}>.")
        self.status.setText(msg)
        if self._report_remade(before, "Sound reference added"):
            self.status.setText(msg + " " + self.status.text())
        return True

    def _remove_sound(self):
        row = self.sounds.currentRow()
        refs = T.audio_refs(self.tree) if self.tree else []
        if row < 0 or row >= len(refs) or self.project is None or self.running:
            return
        users = self._sound_users(row + 1)
        later = row + 1 < len(refs)
        if users:
            text = (f"Remove <Audio {row + 1}>? {len(users)} clip prompt{'s name' if len(users) != 1 else ' names'} "
                    f"it or a later sound; made ones count as CHANGED.")
            if later:
                text += (f" The sounds after it move up (<Audio {row + 2}> becomes <Audio {row + 1}>), so check "
                         "those prompts.")
            if QMessageBox.question(self, "Remove sound reference", text) != QMessageBox.Yes:
                return
        before = self._done_clips()
        T.remove_audio_reference(self.project, refs[row]["file"], self.tree)
        self._show_sounds()
        self._write_tree()
        self._changed()
        self._report_remade(before, "Sound reference removed")

    def _play_sound(self):
        row = self.sounds.currentRow()
        refs = T.audio_refs(self.tree) if self.tree else []
        if 0 <= row < len(refs) and self.project is not None:
            self.preview.open(self.project / refs[row]["file"], f"<Audio {row + 1}>: {refs[row].get('name', '')}")

    # ---- audio tracks (imported sound the clips play, carried on from clip to clip)
    def _track_dialog(self, file: str = "", start=None, end=None, name: str = "", title: str = "Audio track"):
        from .audio_editor import AudioRangeDialog
        return AudioRangeDialog(self, file=file, start=start, end=end, name=name, settings=self.qs, title=title,
                                player=os.environ.get("QT_QPA_PLATFORM") != "offscreen")

    def import_track(self, file: str = "", name: str = "") -> str | None:
        """Choose a sound file, cut it, add it as an audio track. Returns its id."""
        if self.project is None or self.tree is None or self.running:
            return None
        dlg = self._track_dialog(file, name=name, title="Import an audio track")
        if not file:
            dlg._choose()
            if not dlg.file:
                dlg.deleteLater()
                return None
        try:
            if dlg.exec() != QDialog.Accepted or not dlg.file:
                return None
            return self.add_track(dlg.file, dlg.start, dlg.end, dlg.name.text().strip())
        finally:
            dlg.deleteLater()

    def add_track(self, file: str, start: float, end: float | None, name: str = "") -> str | None:
        if self.project is None or self.tree is None or self.running:
            return None
        try:
            entry = T.add_track(self.project, self.tree, Path(file), start, end, name or None)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not import the sound: {e}")
            return None
        self._show_tracks(select=entry["id"])
        self._write_tree()
        self._changed()
        self.status.setText(f"Audio track “{entry['name']}” added ({entry['secs']:.1f} s). Choose it in a clip's "
                            "Sound setting; the clips after it that play it too carry it on.")
        return entry["id"]

    def _track_from_selected(self):
        node = self.tree.nodes.get(self.target_id()) if self.tree else None
        if node is None or not (node.made or {}).get("video"):
            self.status.setText("Select a made clip or a converted video first.")
            return
        what = f"{'video' if node.is_video else 'clip'} #{node.id[1:]}"
        self.import_track(str(self.project / node.made["video"]), name=f"sound of {what}")

    def _selected_track(self) -> dict | None:
        item = self.track_list.currentItem()
        return T.track_of(self.tree, item.data(Qt.UserRole)) if item is not None and self.tree else None

    def _recut_track(self):
        tr = self._selected_track()
        if tr is None or self.project is None or self.running:
            return
        src = tr.get("source") or ""
        if src and Path(src).is_file():
            dlg = self._track_dialog(src, tr.get("start"), tr.get("end"), tr["name"], title="Cut the audio track")
        else:                                           # the original is gone: cut the track's own sound
            dlg = self._track_dialog(str(self.project / tr["file"]), 0.0, float(tr["secs"]), tr["name"],
                                     title="Cut the audio track")
        try:
            if dlg.exec() == QDialog.Accepted and dlg.file:
                self.recut_track(tr["id"], dlg.file, dlg.start, dlg.end, dlg.name.text().strip())
        finally:
            dlg.deleteLater()

    def recut_track(self, tid: str, file: str, start: float, end: float | None, name: str = "") -> bool:
        if self.project is None or self.tree is None or self.running:
            return False
        before = self._done_clips()
        tr = T.track_of(self.tree, tid)
        if tr is None:
            return False

        def near(a, b) -> bool:                        # (the dialog rounds to ms and clamps to the file's length)
            return b is not None and abs(float(a) - float(b)) < 0.02
        own = Path(file) == self.project / tr["file"]   # its original was gone: the track's own sound was shown
        same = end is not None and ((str(file) == tr.get("source") and near(start, tr.get("start"))
                                     and near(end, tr.get("end")))
                                    or (own and near(start, 0.0) and near(end, tr["secs"])))
        if same:                                       # only the name: the sound and the clips stay as they are
            tr["name"] = name or tr["name"]
        else:
            try:
                T.recut_track(self.project, self.tree, tid, Path(file), start, end, name or None)
            except (ValueError, OSError, RuntimeError) as e:
                self.status.setText(f"Could not cut the audio track: {e}")
                return False
        self._show_tracks(select=tid)
        if self._write_tree():
            self._drop_unused_tracks()
        self._changed()
        self.status.setText(f"Audio track “{T.track_of(self.tree, tid)['name']}” "
                            + ("renamed." if same else "cut again."))
        self._report_remade(before, "Audio track cut again")
        return True

    def _remove_track(self):
        tr = self._selected_track()
        if tr is None or self.project is None or self.running:
            return
        users = self._track_users_everywhere(tr["id"])
        if users and QMessageBox.question(
                self, "Remove audio track",
                f"Remove “{tr['name']}”? {users} clip{'s play' if users != 1 else ' plays'} it"
                + (" (in all scenes)" if self.film is not None and len(self.film.scenes) > 1 else "")
                + "; set another sound for them before they are made again.") != QMessageBox.Yes:
            return
        T.remove_track(self.project, self.tree, tr["id"])
        self._show_tracks()
        if self._write_tree():
            self._drop_unused_tracks()
        self._changed()

    def _scene_dirs(self) -> list[Path]:
        if self.film is not None:
            return [self.film.path(sc.id) for sc in self.film.scenes]
        return [self.project] if self.project is not None else []

    def _track_users_everywhere(self, tid: str) -> int:
        """Clips that play a track in the open scene (as it is now) and in the film's other scenes (as saved)."""
        n = len(T.track_users(self.tree, tid))
        for d in self._scene_dirs():
            if Path(d) == Path(self.project):
                continue
            try:
                n += len(T.track_users(T.load(Path(d)), tid))
            except Exception:  # noqa: BLE001  (a scene that cannot be read: not counted)
                pass
        return n

    def _drop_unused_tracks(self):
        """Delete cut track files no scene lists any more (after the project is saved: never one still in use)."""
        if self.project is None:
            return
        for f in T.unused_track_files(self.project, self._scene_dirs(), self.tree):
            try:
                f.unlink()
            except OSError:
                pass

    def _play_track(self):
        tr = self._selected_track()
        if tr is not None and self.project is not None:
            self.preview.open(self.project / tr["file"], f"Audio track: {tr['name']}")

    def _show_tracks(self, select: str | None = None):
        cur = select or (self.track_list.currentItem().data(Qt.UserRole) if self.track_list.currentItem() else None)
        trs = T.tracks(self.tree) if self.tree else []
        sig = ([(tr["id"], tr["name"], tr["file"], len(T.track_users(self.tree, tr["id"]))) for tr in trs], cur)
        if sig == getattr(self, "_tracks_shown", None) and select is None:
            self._update_track_buttons()
            return
        self._tracks_shown = sig
        self.track_list.blockSignals(True)
        try:
            self._fill_tracks(trs, cur)
        finally:
            self.track_list.blockSignals(False)
        self.track_info.setText(
            "Music, speech or a background for the film: import a sound file (or the sound of a video) and cut it. "
            "A clip whose Sound is “Audio track: …” plays it exactly; the clips after it that play the same track "
            "carry it on where the one before stopped (the picture is made to match, lips in sync)."
            if not trs else
            "A clip plays a track via its Sound setting; the clips after it with the same track carry it on. "
            "Shared by all scenes of the film.")
        self._update_track_buttons()

    def _fill_tracks(self, trs: list[dict], cur: str | None):
        self.track_list.clear()
        for tr in trs:
            users = len(T.track_users(self.tree, tr["id"]))
            item = QListWidgetItem(f"{tr['name']}   {float(tr['secs']):.1f} s"
                                   + (f"   · {users} clip{'s' if users != 1 else ''}" if users else ""))
            item.setData(Qt.UserRole, tr["id"])
            item.setToolTip(f"{tr.get('source') or tr['file']}\n{float(tr.get('start') or 0):.2f}–"
                            f"{float(tr.get('end') or 0):.2f} s of the file")
            self.track_list.addItem(item)
            if tr["id"] == cur:
                self.track_list.setCurrentItem(item)
        if trs and self.track_list.currentRow() < 0:
            self.track_list.setCurrentRow(0)

    def _update_track_buttons(self):
        idle = self.project is not None and self.tree is not None and not self.running
        sel = self._selected_track() is not None if self.tree else False
        self.b_track_add.setEnabled(idle)
        node = self.tree.nodes.get(self.target_id()) if self.tree else None
        has = node is not None and bool((node.made or {}).get("video"))
        self.b_track_clip.setEnabled(idle and has)
        self.b_track_clip.setToolTip(f"Import the sound of {'video' if node.is_video else 'clip'} #{node.id[1:]} "
                                     "and cut it" if has else "Select a made clip or a converted video")
        for b in (self.b_track_cut, self.b_track_del):
            b.setEnabled(idle and sel)
        self.b_track_play.setEnabled(sel)

    def _show_sounds(self, select: int | None = None):
        self.sounds.clear()
        refs = T.audio_refs(self.tree) if self.tree else []
        for k, a in enumerate(refs, start=1):
            item = QListWidgetItem(f"<Audio {k}>   {a.get('name') or Path(a['file']).name}   "
                                   f"{float(a.get('secs') or 0):.1f} s")
            item.setToolTip(a["file"])
            self.sounds.addItem(item)
        if refs:
            self.sounds.setCurrentRow(select if select is not None and 0 <= select < len(refs) else 0)
        self._show_sound_info()
        self._update_sound_buttons()

    def _show_sound_info(self):
        refs = T.audio_refs(self.tree) if self.tree else []
        total = sum(float(a.get("secs") or 0) for a in refs)
        if not refs:
            text = ("A voice, music or a beat for clips: add a sound (2-15 s), then name it in a clip's prompt, e.g. "
                    "\"<Audio 1> is the complete soundtrack\". Only clips that name a sound use it (with the "
                    "reference model).")
        else:
            users = self._sound_users(1)
            text = (f"{total:.1f} of 15 s used, {len(users)} clip{'s use' if len(users) != 1 else ' uses'} them. "
                    "\"<Audio 1> is the complete soundtrack\" copies it (measured: works with a plain prompt). A voice "
                    "only took effect in MiniMax's long reference format (\"<Audio 1> is the voice-timbre reference "
                    "for <Subject 1> (S1)\" …); a plain \"in the voice of <Audio 1>\" did not in the test.")
        self.sound_info.setText(text)

    def _update_sound_buttons(self):
        refs = T.audio_refs(self.tree) if self.tree else []
        idle = self.project is not None and not self.running
        total = sum(float(a.get("secs") or 0) for a in refs)
        room = len(refs) < T.MAX_AUDIO_REFS and total <= T.AUDIO_REF_TOTAL - T.AUDIO_REF_MIN
        self.b_sound_add.setEnabled(idle and room)
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if self.tree and nid else None
        has = node is not None and bool((node.made or {}).get("video"))
        self.b_sound_clip.setEnabled(idle and room and has)
        self.b_sound_clip.setToolTip(
            (f"Add the sound of {'video' if node.is_video else 'clip'} #{nid[1:]} (its first seconds), e.g. to keep "
             "a voice from an earlier clip" if has else "Select a made clip or a converted video")
            if room else "There are already 3 sounds or 15 s; remove one first")
        sel = 0 <= self.sounds.currentRow() < len(refs)
        self.b_sound_del.setEnabled(idle and sel)
        self.b_sound_play.setEnabled(sel)

    def _changed(self):
        """Something the clips depend on changed: save soon and update the states."""
        self._save_timer.start()
        self._refresh_timer.start()

    def _save_tree(self) -> bool:
        """Save the open project now; False if it could not be saved (changed outside, disk error)."""
        if self.project is not None and self.tree is not None and not self.running:
            return bool(self._write_tree())
        return True

    def _may_leave_project(self) -> bool:
        """Save before the project is closed or replaced; if that fails, ask before its edits are dropped."""
        self._save_timer.stop()
        if self._save_tree():
            return True
        why = self.status.text()
        return QMessageBox.question(
            self, "Unsaved changes", f"The project could not be saved: {why}\n\nClose it anyway and lose the "
            "changes since the last save?") == QMessageBox.Yes

    def _stamp(self) -> tuple | None:
        try:
            st = (self.project / T.TREE_FILE).stat()
            return st.st_mtime_ns, st.st_size
        except (OSError, TypeError):
            return None

    def _write_tree(self) -> bool:
        """Save tree.json unless something else (another window, a command-line run) changed it since this window
        read it; then nothing is overwritten and the project has to be reopened."""
        if self.project is None or self.tree is None:
            return False
        now = self._stamp()
        if now is not None and self._tree_stamp is not None and now != self._tree_stamp:
            self.status.setText("tree.json was changed outside this window (another window or a command-line "
                                "run). Nothing was saved; reopen the project to continue.")
            return False
        try:
            T.save(self.project, self.tree)
        except OSError as e:
            self.status.setText(f"Could not save the project: {e}")
            return False
        self._tree_stamp = self._stamp()
        self._sync_shared()
        return True

    # ------------------------------------------------------------------ start image
    def _choose_image(self):
        if self.running:
            return
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            self._choose_project()
            if self.project is None:
                return
        start = str(Path(self.tree.image).parent) if self.tree and self.tree.image and \
            Path(self.tree.image).parent.is_dir() else None
        f = pick_image(self, "Start image", self.qs, start, self._picker_places(), self.project_images())
        if f:
            self.set_image(f, confirm=True)

    def _picker_places(self) -> list[tuple[str, str]]:
        """Extra places in the image picker: the open project (its clips' last frames are there)."""
        if self.project is None:
            return []
        out = [(f"Project {self.project.name}", str(self.project))]
        if (self.project / "nodes").is_dir():
            out.append(("Project clips (last frames)", str(self.project / "nodes")))
        if (self.project / EDITS_DIR).is_dir():
            out.append(("Project AI edits", str(self.project / EDITS_DIR)))
        return out

    def project_media(self, kind: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """The open project's videos ('video') or sounds and videos ('media'), for the file browser:
        [(name, [(path, caption)])], kinds without files left out."""
        if self.project is None or self.tree is None:
            return []
        videos, seen = [], set()
        for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id)):
            if not n.is_video:
                continue
            src = (swap.replaced(n) or {}).get("original") or n.source or {}
            f = src.get("file")
            if f and f not in seen and Path(f).is_file():
                seen.add(f)
                videos.append((f, f"Video #{n.id[1:]} · {Path(f).name}"))
        out = [("Real videos of this project", videos)]
        if kind == "media":
            out.insert(0, ("Audio tracks", [(str(self.project / t["file"]), f"Track \u201c{t['name']}\u201d")
                                            for t in T.tracks(self.tree) if (self.project / t["file"]).is_file()]))
        return [(name, items) for name, items in out if items]

    def project_images(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """The open project's pictures by kind, for the image picker: [(name, [(path, caption)])] (kinds without
        pictures are left out; "All project images" first)."""
        if self.project is None or self.tree is None:
            return []
        p = self.project
        start = [(str(p / T.START_FILE), "Start image")] if (p / T.START_FILE).is_file() else []
        clips, videos, ends = [], [], []
        for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id)):
            m = n.made or {}
            if n.is_video:
                for key, what in (("first", "first frame"), ("frame", "last frame")):
                    if m.get(key) and (p / m[key]).is_file():
                        videos.append((str(p / m[key]), f"Video #{n.id[1:]} · {what}"))
                continue
            if m.get("frame") and (p / m["frame"]).is_file():
                clips.append((str(p / m["frame"]), f"Clip #{n.id[1:]} · last frame"))
            h = T.handover_file(p, self.tree, n) if m else None
            if h is not None and h.is_file() and str(h) != str(p / m.get("frame", "")):
                clips.append((str(h), f"Clip #{n.id[1:]} · hand-over frame"))
            if n.end_image and (p / n.end_image).is_file():
                ends.append((str(p / n.end_image), f"End image of clip #{n.id[1:]}"))
        edits = []
        for png, meta in project_edits(p):
            words = " ".join(str(meta.get("prompt") or "").split())
            what = "New image" if meta.get("mode") == "create" else "Edit"
            edits.append((str(png), f"{what} · {words[:48]}{'…' if len(words) > 48 else ''}" if words else what))
        faces = []
        for c in T.cast_of(self.tree):
            for k, rel in enumerate(c["images"], start=1):
                faces.append((str(p / rel), f"@{c['name']}" + (f" · face {k}" if len(c["images"]) > 1 else "")))
        kinds = [("Start image", start), ("Image edits & new images", edits), ("Cast faces", faces),
                 ("Clip frames", clips), ("Video frames", videos), ("End images", ends)]
        kinds = [(name, items) for name, items in kinds if items]
        everything = [x for _n, items in kinds for x in items]
        return ([("All project images", everything)] + kinds) if everything else []

    def made_from_start(self) -> list[str]:
        """Made generated clips that start (directly or along their path) from the start image."""
        if self.tree is None:
            return []
        roots = [n for n in self.tree.children(None) if not n.is_video]
        ids = [r.id for r in roots] + [d.id for r in roots for d in self.tree.descendants(r.id)]
        return [i for i in ids if self.status_of.get(i) == "done"]

    def _start_frozen(self) -> bool:
        """A frozen first clip starts from the start image: it stays (with a word in the status bar)."""
        roots = [n for n in (self.tree.nodes.values() if self.tree else []) if n.frozen and n.parent is None
                 and not n.is_video]
        if roots:
            self.status.setText(f"Clip #{roots[0].id[1:]} is frozen and starts from the start image: unfreeze it "
                                "first to change the start image.")
        return bool(roots)

    def clear_image(self, confirm: bool = False) -> bool:
        """No start image any more. confirm: ask first when made clips depend on it."""
        if self.running or self.project is None or not (self.project / T.START_FILE).is_file():
            return False
        if self._start_frozen():
            return False
        made = [i for i, st in self.status_of.items() if st == "done"] if self.made_from_start() else []
        if confirm and made and QMessageBox.question(
                self, "Remove the start image",
                f"Remove the start image?\n\n{len(made)} made clip{'s depend' if len(made) != 1 else ' depends'} on it "
                "(it starts from it, or it sets the video size); they will count as CHANGED and be made again with "
                "their paths. The first clip then starts from its prompt.") != QMessageBox.Yes:
            return False
        before = self._done_clips()
        T.clear_start_image(self.project, self.tree)
        self._pix.clear()
        self._write_tree()
        self._show_start_image()
        self.refresh()
        if not self._report_remade(before, "Start image removed"):
            self.status.setText("Start image removed: the first clip starts from its prompt.")
        return True

    def set_image(self, path: str, confirm: bool = False) -> bool:
        """Use an image as the start image. confirm: ask first when made clips start from the current one."""
        if self.running:
            return False
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return False
        if QImage(path).isNull():
            self.status.setText(f"Not an image: {Path(path).name}")
            return False
        if self._start_frozen():
            return False
        made = self.made_from_start()
        try:                                        # another aspect ratio: another video size for every clip
            base = T.Checker(self.project, self.tree).base()
            size = chain.canvas_size(*chain.image_size(Path(path)), int(self.tree.options.get("resolution", 512)))
            if base is not None and size != (base["width"], base["height"]):
                made = [i for i, st in self.status_of.items() if st == "done"]
        except Exception:  # noqa: BLE001
            pass
        if confirm and made and QMessageBox.question(
                self, "Replace the start image",
                f"Use {Path(path).name} as the start image?\n\n{len(made)} made clip{'s depend' if len(made) != 1 else ' depends'}"
                " on the current one (it starts from it, or it sets the video size); they will count as CHANGED and "
                "be made again with their paths.") != QMessageBox.Yes:
            return False
        before = self._done_clips()
        try:
            T.set_start_image(self.project, Path(path), self.tree)
        except (OSError, RuntimeError) as e:
            self.status.setText(f"Could not use {Path(path).name}: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self._show_start_image()
        self.refresh()
        if not self._report_remade(before, "New start image"):
            self.status.setText(f"Start image: {Path(path).name}.")
        return True

    def _update_ref_start_button(self):
        """Enable the 'Add: Start image / Selected clip' buttons and say exactly which frame they take."""
        self._update_sound_buttons()
        self._update_track_buttons()
        start = self._start_image_there()
        self.b_ref_start.setEnabled(start and not self.running)
        self.b_ref_start.setToolTip("Add the start image as a face reference" if start else
                                    "The project has no start image")
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if self.tree and nid else None
        made = node is not None and bool(node.made)
        self.b_ref_clip.setEnabled(made and not self.running)
        if made:
            what = (f"the first frame of video #{nid[1:]}" if node.is_video else f"the last frame of clip #{nid[1:]}")
            tip = f"Add {what} as a face reference"
        else:
            tip = "Select a converted video or a made clip"
        self.b_ref_clip.setToolTip(tip + ". For any frame of a video: Edit range… → Add as face picture…")
        has_sound = node is not None and bool((node.made or {}).get("video"))
        self.b_voice_clip.setEnabled(has_sound and self._selected_character() is not None and not self.running)
        self.b_voice_clip.setToolTip(
            f"Set the voice from the sound of {'video' if node.is_video else 'clip'} #{nid[1:]} (its first "
            f"{T.VOICE_MAX_SECS:g} s)" if has_sound else "Select a made clip or a converted video with the voice")

    def _show_start_image(self):
        self._update_ref_start_button()
        start = self.project / T.START_FILE if self.project else None
        if start and start.is_file():
            self.start_tile.set_image(QImage(str(start)))
            self.start_tile.setToolTip(self.tree.image or str(start))
        else:
            self.start_tile.clear_frame("Click or drop an image")
        self.b_start_clear.setEnabled(bool(start and start.is_file()) and not self.running)
        self._update_size_info()

    def _update_size_info(self):
        start = self.project / T.START_FILE if self.project else None
        if not start or not start.is_file():
            dims = None
            try:
                dims = T.size_source(self.project, self.tree) if self.project else None
            except Exception:  # noqa: BLE001
                pass
            origin = T.size_origin(self.project, self.tree) if self.project else "aspect"
            self.aspect.setEnabled(origin == "aspect" and not self.running)
            if dims and origin == "aspect":
                vw, vh = chain.canvas_size(*dims, self.resolution.currentData())
                self._canvas = (vw, vh)
                self.size_info.setText(f"No start image: {self.aspect.currentData()} → {vw}×{vh} (Aspect); the first "
                                       "clip starts from its prompt (and the cast's face pictures)."
                                       + self._veda_note(vw, vh))
            elif dims:
                vw, vh = chain.canvas_size(*dims, self.resolution.currentData())
                self._canvas = (vw, vh)
                self.size_info.setText(f"No start image: the size follows the first video ({dims[0]}×{dims[1]} → "
                                       f"{vw}×{vh}); a first generated clip starts without a start frame."
                                       + self._veda_note(vw, vh))
            else:
                self.size_info.setText("Choose a start image, or begin with a real video (Clips → camera button).")
            return
        self.aspect.setEnabled(False)                  # (the start image's format applies)
        img = QImage(str(start))
        w, h = img.width(), img.height()
        if img.isNull() or w < 1 or h < 1:
            self.size_info.setText(f"The start image ({T.START_FILE}) cannot be read: choose it again.")
            return
        vw, vh = chain.canvas_size(w, h, self.resolution.currentData())
        r = (w / h) / (vw / vh)
        note = "" if 0.97 < r < 1.03 else " (ratio outside 2:5 … 5:2: its edges are cut off to fit)"
        if w < vw or h < vh:
            note += f" · smaller than the video: the first frame starts soft (best: at least {vw}×{vh})"
        name = Path(self.tree.image).name if self.tree and self.tree.image else T.START_FILE
        self._canvas = (vw, vh)
        self.size_info.setText(f"{name}: {w}×{h} → {vw}×{vh} video{note}" + self._veda_note(vw, vh))

    def _veda_note(self, vw: int, vh: int) -> str:
        """Where Veda is used at this size (for the size line)."""
        if not self.veda.isChecked():
            return ""
        secs = chain.veda_from_seconds(vw, vh)
        return (f" · ⚡ Veda for clips from {secs:.1f} s" if secs is not None else
                " · Veda not used at this size (too small to gain)")

    def veda_used(self, n: T.Node) -> bool:
        """This generated clip is made with Veda sparse attention (see tree.veda_for)."""
        canvas = getattr(self, "_canvas", None)
        if n.is_video or canvas is None or self.tree is None:
            return False
        frames = int((n.overrides or {}).get("frames", self.tree.options.get("frames", 73)))
        return T.veda_for(self.tree.options, canvas[0], canvas[1], frames)

    # ------------------------------------------------------------------ tree view
    def selected_id(self) -> str | None:
        return self.tree.selected if self.tree else None

    def selected_path(self) -> list[T.Node]:
        sel = self.selected_id()
        return self.tree.path_to(sel) if self.tree and sel and sel != START and sel in self.tree.nodes else []

    def _pixmap(self, path: Path | None) -> QPixmap | None:
        if path is None or not path.is_file():
            return None
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key not in self._pix:
            self._pix[key] = QPixmap.fromImage(QImage(str(path)))
        return self._pix[key]

    def _node_frame(self, n: T.Node) -> Path | None:
        """The frame shown for a node: where the clips after it start (its hand-over frame, else its last)."""
        if n.made and self.project:
            h = T.handover_file(self.project, self.tree, n)
            return h if h is not None and h.is_file() else self.project / n.made["frame"]
        return None

    def state(self, nid: str) -> str:
        return self.live.get(nid) or self.status_of.get(nid, "new")

    def refresh(self, rebuild_path: bool = False):
        if self.tree is None:
            self.graph.show_nodes([])
            self._update_buttons()
            return
        if self.tree.options.get("audio_refs"):
            self._show_sound_info()                # how many clips name a sound (prompts change)
        checker = T.Checker(self.project, self.tree)
        try:
            self.status_of = {nid: checker.status(nid) for nid in self.tree.nodes}
            self.why_of = {nid: checker.why(nid) for nid, st in self.status_of.items() if st == "changed"}
        except Exception as e:  # noqa: BLE001  (e.g. an unreadable start image)
            self.status_of = {nid: "new" for nid in self.tree.nodes}
            self.why_of = {}
            self.status.setText(f"Could not check the clips: {e}")
        sel = self.selected_id()
        on_path = {n.id for n in self.selected_path()}
        start_pm = self._pixmap(self.project / T.START_FILE)
        infos = [NodeInfo(START, None, "Start", "", Path(self.tree.image).name if self.tree.image else "",
                          "start", start_pm, on_path=True, selected=sel == START)]
        for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id)):
            text = n.prompt
            if n.is_video:
                text = (f"{Path(n.source['file']).name} · {fmt_time(n.source['start'])}–"
                        f"{fmt_time(n.source['end'])}")
            infos.append(NodeInfo(n.id, n.parent or START,
                                  f"{'Video' if n.is_video else 'Clip'} {self.tree.depth(n.id)}", f"#{n.id[1:]}",
                                  text, self.state(n.id), self._pixmap(self._node_frame(n)),
                                  self.progress_of.get(n.id, 0.0), n.id in on_path, n.id == sel,
                                  self.badge(n), n.kind))
        self.graph.show_nodes(infos)
        self._show_tracks()                        # (how many clips play each track)
        if rebuild_path or [c.nid for c in self.cards] != [n.id for n in self.selected_path()]:
            self._rebuild_path()
        else:
            self._update_cards()
        made = sum(1 for s in self.status_of.values() if s == "done")
        branches = len(self.tree.leaves())
        videos = sum(1 for n in self.tree.nodes.values() if n.is_video)
        self.tree_info.setText(f"{len(self.tree.nodes)} clips" + (f" ({videos} real)" if videos else "") +
                               f" · {branches} branch{'es' if branches != 1 else ''} · {made} made")
        self._update_scene_card()
        self._update_buttons()

    def _frozen_note(self, n: T.Node) -> str:
        try:
            return T.frozen_note(self.tree, n, T.Checker(self.project, self.tree).base())
        except Exception:  # noqa: BLE001  (an unreadable start image: no note)
            return ""

    def badge(self, n: T.Node) -> str:
        """Short text for a clip's own settings, shown on its node ('' = all project defaults)."""
        if n.is_video:
            return ("🔒 frozen · " if n.frozen else "") + f"real video · {T.clip_length(self.tree, n) / 24:.1f} s"
        ov, parts = n.overrides or {}, self._own_parts(n)
        if ov.get("guide"):
            parts.append(f"carry-over {ov['guide']}")
        if T.end_image_used(self.tree, n):
            parts.append("end image")
        if n.keyframes:
            parts.append(f"{len(n.keyframes)} keyframe{'s' if len(n.keyframes) != 1 else ''}")
        if (n.made or {}).get("cuts"):
            parts.append("⚠ cuts")
        if n.frozen:
            parts.insert(0, "🔒 frozen")
        k = T.handover_index(self.tree, n)
        if k is not None:
            parts.append(f"hands over at {k / 24:.1f} s")
        if self.veda_used(n):
            parts.append("⚡ Veda")
        return " · ".join(parts)

    def _faces_text(self, use: dict) -> str:
        """Which faces (and voices) a clip is made with, for its card ('' without a cast); with spoken lines, who
        says each one (as the app reads it: the voice it gets)."""
        if not T.cast_of(self.tree):
            return ""
        text = self._cast_line(use)
        if use["mode"] == "voices":
            said = [(f"@{d['name']}" if d["name"] else "nobody")
                    + f": \u201c{d['line'][:24]}{'…' if len(d['line']) > 24 else ''}\u201d" for d in use["lines"]]
            text += "\nLines: " + " · ".join(said)
        return text

    def _cast_line(self, use: dict) -> str:
        if use["mode"] == "voices":
            faces = [f"@{c['name']}" for c in use["characters"] if c["images"]]
            voices = [f"@{c['name']}" for c in use["characters"]
                      if c.get("voice") and c["voice"]["file"] in use["voices"]]
            return (("Faces: " + ", ".join(faces) + " · ") if faces else "") + "Voices: " + ", ".join(voices)
        if use["mode"] == "mention":
            return "Faces: " + ", ".join(f"@{c['name']}" for c in use["characters"])
        if use["mode"] == "lead":
            return f"Faces: @{use['characters'][0]['name']} (the lead; the prompt mentions nobody)"
        if use["mode"] == "tags":
            return f"Faces: all {len(use['images'])} pictures of the cast, in cast order (<Picture N> written)"
        return "Faces: none (the prompt mentions nobody and there is no lead)"

    def _source_sound_text(self, n: T.Node) -> tuple[bool, str, int, bool]:
        """For the clip card's 'Sound' row: (a real video comes before this clip, where the clip's sound comes from,
        the longest carry-over that fits, there is a clip before it whose sound can be repeated)."""
        def t(x: float) -> str:
            return f"{int(x // 60)}:{x % 60:04.1f}"
        video = T.source_video(self.tree, n)
        repeat = n.parent is not None
        tid = T.track_id(n)
        if tid is not None:
            span = T.track_sound_span(self.tree, n)
            tr = T.track_of(self.tree, tid)
            if span is None or tr is None:
                return video is not None, "This audio track was removed: choose another sound.", 0, repeat
            end = span["start"] + span["secs"] - span["pad"]
            prev = self.tree.nodes.get(n.parent) if n.parent else None
            how = (f"carries it on from clip #{span['from'][1:]}" if span["from"] else
                   "from its beginning" + (f": {'video' if prev.is_video else 'clip'} #{prev.id[1:]} before it "
                                           "does not play it" if prev is not None else ""))
            past = (f"; the track ends at {t(float(tr['secs']))}, silence after it" if end > float(tr["secs"]) + 0.05
                    else "")
            if span["start"] >= float(tr["secs"]) - 0.05:
                return video is not None, (f"“{tr['name']}” has ended before this clip ({t(float(tr['secs']))}): "
                                           "choose another sound or cut the track longer."), 0, repeat
            return video is not None, (f"Plays {t(span['start'])}–{t(min(end, float(tr['secs'])))} of “{tr['name']}”"
                                       f" ({how}{past}), exactly as it is. Write in the prompt what happens; the "
                                       "sound comes from the track."), 0, repeat
        if (n.overrides or {}).get("source_sound") == "previous":
            span = T.previous_sound_span(self.tree, n)
            if span is None:
                return video is not None, "This is the first clip: there is no clip before it to repeat.", 0, False
            p = self.tree.nodes[span["video"]]
            what = (f"video #{p.id[1:]} ({Path(span['file']).name} {t(span['start'])}–"
                    f"{t(span['start'] + span['loop'])})" if p.is_video else f"clip #{p.id[1:]}")
            loop = (f", looped to fill this clip's {span['secs']:.1f} s" if span["secs"] > span["loop"] + 0.05 else "")
            later = "" if p.is_video or p.made else " (it is made first)"
            return video is not None, (f"Repeats the sound of {what}{later}: {span['loop']:.1f} s as the film plays "
                                       f"it{loop}, exactly as it is. Write in the prompt what happens; the sound "
                                       f"comes from the clip before."), 0, True
        if video is None:
            return False, "There is no real video before this clip any more: choose another sound.", 0, repeat
        parent = self.tree.nodes.get(n.parent) if n.parent else None
        have = 0
        if parent is not None:
            pn = T.made_frames(parent) or T.clip_length(self.tree, parent)
            h = T.handover_index(self.tree, parent, pn)
            have = h + 1 if h is not None else pn
        fits = [g for g in T.GUIDE_CHOICES if g <= have and g < T.clip_length(self.tree, n)]
        span = T.source_sound_span(self.tree, n)
        return True, (f"Uses {t(span['start'])}–{t(span['start'] + span['secs'])} of "
                      f"{Path(span['file']).name} (video #{video.id[1:]}): the sound right after its range, exactly as "
                      "recorded. Write in the prompt what happens; the words come from the video."), \
            max(fits, default=0), repeat

    def _own_parts(self, n: T.Node) -> list[str]:
        """A clip's own LoRA / strength / length / steps in a few words."""
        ov, parts = n.overrides or {}, []
        over = " (laid over)" if ov.get("source_sound") and ov.get("sound_over") else ""
        for p, st in T.node_loras(n):
            parts.append(f"{lora_label(p)[:18]} ×{st:g}")
        if "frames" in ov:
            parts.append(f"{ov['frames'] / 24:.1f} s")
        if "steps" in ov:
            parts.append(f"{ov['steps']} steps")
        if CAM.describe(ov.get("camera")):
            parts.append("🎥 " + CAM.describe(ov.get("camera")))
        if ov.get("source_sound") == "previous":
            parts.append("repeats the sound before" + over)
        elif T.track_id(n) is not None:
            tr = T.track_of(self.tree, T.track_id(n)) if self.tree else None
            parts.append((f"track “{tr['name'][:16]}”" if tr else "removed track") + over)
        elif ov.get("source_sound"):
            parts.append("video's sound" + over)
        return parts

    def _rebuild_path(self):
        for c in self.cards:
            self.card_stack.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        path = self.selected_path()
        self.path_hint.setText("The start image is selected. Select a clip to see its path, or press Continue "
                               "(Ctrl+N) to add a clip after the start image.")
        for k, n in enumerate(path):
            if n.is_video:
                card = VideoCard(n)
                card.action.connect(self.node_action)
                card.play.connect(self._play_node)
                self._fit_card(card)
                self.card_stack.addWidget(card)
                self.cards.append(card)
                card.set_editable(not self.running)
                continue
            card = PromptCard(n, names=self._cast_names)
            card.edited.connect(self._card_edited)
            card.seedChanged.connect(self._seed_changed)
            card.new_seed = self._new_seed
            card.overridesChanged.connect(self._overrides_changed)
            card.chooseLora.connect(self._choose_node_lora)
            card.lora_picked.connect(self._lora_picked)
            card.loras.file_dir = self._lora_dir()
            card.settings_toggle.toggled.connect(self._show_card_settings)
            card.set_choices(int(self.tree.options.get("frames", 73)), int(self.tree.options.get("steps", 28)),
                             self.known_loras())
            card.action.connect(self.node_action)
            card.keyframe.connect(self.keyframe_action)
            card.play.connect(self._play_node)
            self._fit_card(card)
            self.card_stack.addWidget(card)
            self.cards.append(card)
            card.set_editable(not self.running)
        ids = [c.nid for c in self.cards]
        end = ids[-1] if ids else None
        if self.focus not in ids or end != getattr(self, "_path_end", None):
            self.focus = end                       # a new path (or a new clip at its end): where one goes on
            if end is not None:
                self.inspector.setCurrentIndex(0)
        self._path_end = end
        self._update_cards()
        self._show_focus()

    @staticmethod
    def _fit_card(card):
        """A card in the inspector takes its width (it had a fixed width in a row)."""
        card.setMinimumWidth(300)
        card.setMaximumWidth(16777215)

    def _show_focus(self):
        card = next((c for c in self.cards if c.nid == self.focus), None)
        self.card_stack.setCurrentWidget(card if card is not None else self.path_hint)
        self.card_stack.updateGeometry()
        ids = [c.nid for c in self.cards]
        k = ids.index(self.focus) if self.focus in ids else -1
        self.b_prev_clip.setEnabled(k > 0)
        self.b_next_clip.setEnabled(0 <= k < len(ids) - 1)
        if k >= 0 and self.tree is not None and self.focus in self.tree.nodes:
            n = self.tree.nodes[self.focus]
            what = "Video" if n.is_video else "Clip"
            self.focus_info.setText(f"{what} #{n.id[1:]} · {k + 1} of {len(ids)} on the path")
        else:
            self.focus_info.setText("")
        self.timeline.set_focus(self.focus)
        self.graph.set_focus(self.focus)
        self._update_buttons()                     # (the actions follow the clip in focus)

    def focus_clip(self, nid: str):
        """Show a clip of the path in the inspector (the path stays as it is)."""
        if any(c.nid == nid for c in self.cards):
            self.focus = nid
            self._show_focus()
            self.inspector.setCurrentIndex(0)

    def _step_focus(self, d: int):
        ids = [c.nid for c in self.cards]
        if self.focus in ids:
            k = max(0, min(len(ids) - 1, ids.index(self.focus) + d))
            self.focus_clip(ids[k])

    def target_id(self):
        """The clip the toolbar / shortcut actions work on: the one shown in the inspector, else the selected."""
        if self.focus and self.tree is not None and self.focus in self.tree.nodes:
            return self.focus
        return self.selected_id()

    def _graph_clicked(self, nid):
        """A click in the graph: a clip of the shown path is shown in the inspector; another one is selected (its
        path is shown, it in focus)."""
        if nid != START and any(c.nid == nid for c in self.cards):
            self.focus_clip(nid)
            return
        self.select(nid)
        if nid != START:
            self.focus_clip(nid)
        else:
            self.show_start_image_settings()

    def show_start_image_settings(self):
        """The Project tab of the inspector, scrolled to the start image (a click on the Start node)."""
        self.inspector.setCurrentWidget(self.proj_scroll)
        QTimer.singleShot(0, lambda: self.proj_scroll.ensureWidgetVisible(self.start_tile, 0, 40))

    def _show_card_of(self, nid: str):
        self.focus_clip(nid)
        card = next((c for c in self.cards if c.nid == nid), None)
        if isinstance(card, PromptCard):
            card.edit.setFocus()

    def _update_timeline(self, path_nodes: list, cuts: dict):
        """The path as blocks as long as its clips are in the joined video; the carry-over hatched."""
        blocks = []
        for n in path_nodes:
            frames = T.clip_length(self.tree, n)
            a, b = cuts.get(n.id, (0, 0))
            secs = max(1, frames - a - b) / 24
            lead = int((n.overrides or {}).get("guide") or 0) / 24 if not n.is_video and n.parent else 0.0
            pm = self._pixmap(self._node_frame(n))
            if pm is not None:                      # scaled once per picture (the timeline is redrawn often)
                cache = self.__dict__.setdefault("_tl_thumbs", {})
                key = pm.cacheKey()
                if key not in cache:
                    if len(cache) > 400:
                        cache.clear()
                    cache[key] = pm.scaledToHeight(48, Qt.SmoothTransformation)
                pm = cache[key]
            what = (Path(n.source["file"]).name if n.is_video else (n.prompt or "(no prompt yet)"))
            tip = (f"{'Video' if n.is_video else 'Clip'} #{n.id[1:]} · {secs:.1f} s · {self.state(n.id)}\n"
                   f"{what[:160]}")
            blocks.append(Block(n.id, f"#{n.id[1:]}", secs, self.state(n.id), n.is_video, pm, lead, tip))
        self.timeline.show_path(blocks, self.focus)

    def _show_card_settings(self, on: bool):
        """Open / close the own settings on every card at once."""
        for c in self.cards:
            if isinstance(c, PromptCard):
                c._show_settings(on)

    def _update_cards(self):
        path_nodes = [self.tree.nodes[c.nid] for c in self.cards if c.nid in self.tree.nodes]
        self.cast_strip.show_path(self.tree.options, path_nodes)
        cuts = dict(zip([n.id for n in path_nodes],
                        T.path_cuts(self.tree, path_nodes, bool(self.tree.options.get("keep_repeated")),
                                    planned=True)))
        self._update_timeline(path_nodes, cuts)
        t = 0.0
        for k, card in enumerate(self.cards):
            n = self.tree.nodes.get(card.nid)
            if n is None:
                continue
            frames = T.clip_length(self.tree, n)
            cut = cuts.get(n.id, (0, 0))
            length = max(0, frames - cut[0] - cut[1]) / 24
            card.title.setText(f"{'Video' if n.is_video else 'Clip'} {k + 1}")
            span = f"{fmt_time(t)} – {fmt_time(t + length)} in this path"
            vc = self.tree.video_child(n.id)
            if vc is not None:
                span += f" · ends at video #{vc.id[1:]}"
            card.span.setText(span)
            t += length
            st = self.state(n.id)
            if n.is_video:
                card.set_source(n)
                step = ("will be converted again" if st == "changed" else
                        "" if Path(n.source["file"]).is_file() or st == "done" else "the video file is missing")
            else:
                why = self.why_of.get(n.id) if st == "changed" else ""
                step = f"Will be made again: {why}." if why else ("will be made again" if st == "changed" else "")
                card.set_revert(T.changed_inputs(self.tree, n.id) if st == "changed" else [])
                card.set_guide_useful(n.parent is not None or vc is not None)
                k = T.handover_index(self.tree, n)
                total = T.made_frames(n) or T.clip_length(self.tree, n)
                if vc is not None:
                    card.set_handover(f"first frame of video #{vc.id[1:]}", False, False,
                                      "A clip that leads into a real video hands over at the video's start")
                elif k is not None:
                    card.set_handover(f"frame {k + 1} of {total} ({k / 24:.2f} s)", True, True,
                                      f"The clips after it start from this frame; the last {total - 1 - k} frames "
                                      "are left out of the path")
                else:
                    card.set_handover("last frame", bool(n.made), n.end_at is not None,
                                      "" if n.made else "Make the clip first, then choose an earlier frame if its "
                                                        "last one is not good")
                card.set_compare(len(self.compare_candidates(n.id)) - 1 if n.made else 0)
                card.words.setText(card.words.text().split(" + ")[0] +
                                   (" + style" if T.full_prompt(self.tree.options, "x") != "x" else ""))
                use = T.clip_cast(self.tree.options, n.prompt)
                words = T.lora_trigger_words([p for p, _s in T.node_loras(n)]) \
                    if self.tree.options.get("lora_triggers", True) and T.node_loras(n) else []
                sounds = [None] * (T.audio_mentioned(T.full_prompt(self.tree.options, n.prompt))
                                   + len(use.get("voices") or []))
                cont = (n.overrides or {}).get("source_sound") \
                    if T.source_sound_span(self.tree, n) is not None and not (n.overrides or {}).get("sound_over") \
                    else None
                sent = T.model_prompt(self.tree.options, n, {"lora_triggers": words, "audio_refs": sounds,
                                                             "source_sound": cont,
                                                             "camera": CAM.clean((n.overrides or {}).get("camera"))})
                card.edit.setToolTip("Sent to the model: " + sent if sent != n.prompt.strip() else "")
                card.set_faces(self._faces_text(use))
                card.set_tracks([(tr["id"], tr["name"]) for tr in T.tracks(self.tree)])
                avail, info, max_guide, rep_ok = self._source_sound_text(n)
                if info and (n.overrides or {}).get("sound_over") and (n.overrides or {}).get("source_sound"):
                    info = (info.replace(" exactly as it is.", " exactly as it is,").replace(" exactly as recorded.",
                                                                                            " exactly as recorded,")
                            + " Laid over: the clip is made without it (full motion, no lip sync) and gets it "
                              "afterwards; it may be shorter than 2 s.")
                card.set_source_sound(avail, info, max_guide, rep_ok)
                card.set_summary(" · ".join(self._own_parts(n)))
                card.set_veda(self.veda_used(n))
                end = self.project / n.end_image if n.end_image else None
                if vc is not None:
                    card.set_end(end, vc.id, self._pixmap(self.project / vc.made["first"]) if vc.made else None)
                else:
                    card.set_end(end, None, self._pixmap(end) if end is not None and end.is_file() else None)
                card.set_keyframes(T.clip_length(self.tree, n) / chain.FPS,
                                   [{"at": k["at"], "pixmap": self._pixmap(self.project / k["image"]),
                                     "note": k["note"], "ok": (self.project / k["image"]).is_file()}
                                    for k in n.keyframes],
                                   [c / chain.FPS for c in (n.made or {}).get("cuts") or []],
                                   bool(n.made and (self.project / n.made["video"]).is_file()))
            card.set_state(st, self._pixmap(self._node_frame(n)), step)
            why = self.why_of.get(n.id) if st == "changed" else ""
            card.tag.setToolTip(f"Changed: {why}. It is made again with the path." if why else STATE_TIP.get(st, ""))
            frozen = n.frozen and st == "done"
            if frozen:
                note = self._frozen_note(n)
                card.tag.setText("FROZEN")
                card.tag.setToolTip("Kept as made, whatever the project's settings say; not made again")
                card.step.setText("Frozen: kept as made" + (f" ({note})" if note else "") +
                                  ". Unfreeze it (lock button) to edit it or make it again.")
            card.set_frozen(frozen)
            if st == "generating" and n.id in self.progress_of and not n.is_video:
                card.bar.setValue(round(100 * self.progress_of[n.id]))
        path = self.selected_path()
        if path:
            todo = sum(1 for n in path if self.state(n.id) != "done")
            self.path_info.setText(f"{len(path)} clips · {fmt_time(t)}" + (f" · {todo} to make" if todo
                                                                           else " · all made"))
        else:
            self.path_info.setText("")

    def _update_buttons(self):
        has = self.tree is not None
        sel = self.target_id() if has else None     # the actions work on the clip in the inspector
        is_clip = bool(has and sel and sel != START and sel in self.tree.nodes)
        idle = not self.running
        self.b_continue.setEnabled(has and idle)
        self.b_add_video.setEnabled(has and idle)
        for w in (self.act_save, self.act_save_as, self.act_close):
            w.setEnabled(has and idle)
        for w in (self.act_new, self.act_open, self.b_new, self.b_open, self.b_recent, self.recent_file_menu):
            w.setEnabled(idle)
        for b in (self.b_branch, self.b_variant, self.b_delete):
            b.setEnabled(is_clip and idle)
        self.b_variant.setEnabled(is_clip and idle and sel in self.tree.nodes and not self.tree.nodes[sel].is_video)
        path = self.selected_path()
        all_done = bool(path) and all(self.state(n.id) == "done" for n in path)
        free = idle and not self._joins and not self._converts   # a join reads clips a new run would replace
        self.b_gen.setEnabled(free and bool(path))
        self.b_gen_all.setEnabled(free and has and bool(self.tree.nodes))
        self.b_delete.setEnabled(is_clip and free)
        num = f" #{sel[1:]}" if is_clip else ""
        self.b_branch.setToolTip(f"Branch: another continuation from the same point as clip{num} (Ctrl+B)")
        self.b_variant.setToolTip(f"Variant of clip{num}: same prompt, other seed (Ctrl+D)")
        self.b_delete.setToolTip(f"Delete clip{num} (the one in the inspector) and everything after it (Del)")
        self.b_play_path.setEnabled(all_done and idle)
        self.b_export.setEnabled(all_done and idle)
        fn = self.tree.nodes.get(self.focus) if has and self.focus else None
        self.b_play_clip.setEnabled(bool(fn and fn.made))
        self.b_export_clip.setEnabled(bool(fn and fn.made) and idle and self.state(fn.id) == "done")
        self.b_export_clip.setToolTip(f"Save clip #{fn.id[1:]} on its own (as made, or upscaled)" if fn and fn.made
                                      else "The clip in the inspector is not made yet" if fn else
                                      "Show a clip in the inspector first")
        self.b_play_clip.setToolTip(f"Play clip #{fn.id[1:]}, the one in the inspector" if fn and fn.made else
                                    "The clip in the inspector is not made yet" if fn else
                                    "Show a clip in the inspector first")
        self.b_play_film.setToolTip("Join the scenes of the film and play it (all scenes must be made)")
        self.b_play_film.setEnabled(self.film is not None and idle and not (self._copying or self._joins
                                                                          or self._converts))
        todo = sum(1 for n in path if self.state(n.id) != "done")
        why = ("Select a clip first: its path is played" if not path else
               f"Not ready: {todo} clip{'s' if todo != 1 else ''} on this path {'are' if todo != 1 else 'is'} new or "
               "changed. Generate path first" if todo else
               "Wait until generating has finished" if not idle else "")
        self.b_play_path.setToolTip(why or "Join the clips of the selected path and play them")
        self.b_export.setToolTip(why or "Save the selected path as one video")
        self.b_stop.setEnabled(not idle or self._waiting_comfy is not None)
        self.b_stop.setToolTip("Stop generating; finished clips are kept" if not idle else
                               "Do not generate when ComfyUI has started" if self._waiting_comfy is not None else
                               "Stop generating")
        self._update_comfy_label()
        for w in (self.start_tile, self.frames, self.resolution, self.steps,
                  self.audio, self.keep_rep, self.b_project,
                  self.comfy_url, self.ref_size, self.ref_hint, self.b_ref_add, self.b_ref_create,
                  self.b_ref_start,
                  self.b_ref_del, self.b_ref_crop, self.b_char_new, self.b_char_rename, self.b_char_del, self.char_desc,
                  self.char_lead, self.b_voice_file, self.b_voice_del, self.dialogue_lang, self.speed, self.style_prefix,
                  self.style_suffix, self.lora_triggers, self.veda, self.b_ai_edit, self.b_ai_create, self.storyboard,
                  self.b_film_gen, self.b_film_play, self.b_film_export, self.b_film_cast):
            w.setEnabled(idle)
        self.aspect.setEnabled(idle and self.project is not None and T.size_origin(self.project, self.tree) == "aspect")
        self.b_start_clear.setEnabled(idle and self.project is not None and (self.project / T.START_FILE).is_file())
        busy = bool(self._copying or self._joins or self._converts)
        self.storyboard.setEnabled(idle and not busy)  # a scene switch would pull the files from under them
        self.b_film_play.setEnabled(idle and not busy)
        self.b_film_export.setEnabled(idle and not busy)
        self.b_film_gen.setEnabled(idle and not busy)
        self._turbo_widgets()                      # after the loop: turbo locks the steps fields again
        self._update_ref_start_button()            # after the loop too: they depend on the selection
        self._update_cast_buttons()

    def select(self, nid):
        if self.tree is None:
            return
        self.tree.selected = nid if (nid == START or nid in self.tree.nodes) else START
        self._save_timer.start()
        self.refresh()

    def _card_edited(self, nid: str):
        card = next((c for c in self.cards if c.nid == nid and isinstance(c, PromptCard)), None)
        if card is None or self.tree is None or nid not in self.tree.nodes:
            return
        if self.tree.nodes[nid].frozen:
            return
        node, text = self.tree.nodes[nid], card.text()
        made = (node.made or {}).get("prompt")
        # the card keeps words, not spacing: a made prompt that differs only in spaces / line breaks stays as made
        node.prompt = made if isinstance(made, str) and " ".join(made.split()) == text else text
        self._changed()

    def known_loras(self) -> list[tuple[str, str]]:
        """(path, label) of the LoRAs to offer: every H3 LoRA in ComfyUI's LoRA folders, then LoRAs from
        elsewhere that this project uses or that were chosen recently."""
        stems = [Path(name).stem for _p, name in self.h3_loras]
        out = []
        for (p, name), stem in zip(self.h3_loras, stems):
            folder = str(Path(name).parent)
            # just the file name; the folder only where two LoRAs share a name
            out.append((p, f"{stem} ({folder})" if stems.count(stem) > 1 and folder != "." else stem))
        listed = {p for p, _ in out} | {os.path.realpath(p) for p, _ in out}
        try:
            recent = json.loads(self.qs.value("recentLoras", "[]", type=str))
        except ValueError:
            recent = []
        cands = ([p for n in self.tree.nodes.values() for p, _st in T.node_loras(n)] if self.tree else []) + recent
        for lo in cands:
            if lo and lo != T.NO_LORA and lo not in listed and os.path.realpath(lo) not in listed \
                    and Path(lo).is_file():
                out.append((lo, lora_label(lo)))
                listed.add(lo)
        return out

    def _fill_turbo(self, current: str | None):
        """The turbo LoRA list; without a choice yet the recommended v4 ema (or the first) is selected."""
        self.turbo_lora.blockSignals(True)
        try:
            self.turbo_lora.clear()
            known = [(p, Path(n).stem) for p, n in self.turbo_loras]
            if not current and known:
                current = next((p for p, n in known if "v4" in n and "ema" in n), known[0][0])
                if self.tree is not None and not self.running:   # store what the list shows
                    turbo = dict(self.tree.options.get("turbo") or {})
                    turbo["lora"] = current
                    self.tree.options["turbo"] = turbo
                    self._changed()
            if known or current:
                fill_lora_combo(self.turbo_lora, known, current)
            else:
                self.turbo_lora.addItem("no turbo LoRA found", None)
        finally:
            self.turbo_lora.blockSignals(False)
        self._turbo_widgets()

    def _turbo_widgets(self):
        on = self.speed.currentData() == "turbo"
        hyper = self.speed.currentData() in ("hyperflow", "pika")
        idle = not self.running
        for w in (self.turbo_lora, self.turbo_steps, self.turbo_strength):
            w.setEnabled(on and idle)
        self._clips_form.setRowVisible(self.turbo_lora, on)      # the turbo settings only matter in turbo mode
        self._clips_form.setRowVisible(self._turbo_row, on)
        locked = on or hyper                                      # their own steps for every clip
        self.steps.setEnabled(idle and not locked)
        self.steps.setToolTip("Turbo mode: every clip uses the turbo steps" if on else
                              "HyperFlow / Pika H3 Flex: every clip uses their own steps" if hyper else STEPS_TIP)
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.steps_locked = locked
                card.steps.setEnabled(idle and not locked and not card.edit.isReadOnly())

    def _warm_lora_hashes(self):
        """Hash the LoRAs this project uses in the background, so the clip states need not wait for it."""
        if self.tree is None:
            return
        o = self.tree.options
        files = [(o.get("turbo") or {}).get("lora")] + \
            [p for n in self.tree.nodes.values() for p, _st in T.node_loras(n)]
        files = [f for f in dict.fromkeys(files) if f]
        def work():
            for f in files:
                chain.file_signature(f)               # cached by (path, size, date) for the clip states
        threading.Thread(target=work, daemon=True).start()

    def scan_loras(self, report: bool = False):
        """Find the H3 LoRAs in ComfyUI's LoRA folders (header reads only), in the background. report: say in the
        status bar how many were found."""
        comfy_dir, bridge = Path(os.path.expanduser(self.comfy_dir.text().strip() or DEFAULT_COMFY_DIR)), self.bridge
        self._scanned_at = time.monotonic()
        self._lora_scan_gen = getattr(self, "_lora_scan_gen", 0) + 1
        gen = self._lora_scan_gen

        def work():
            try:
                from ..comfy import comfy_lora_roots, find_h3_loras, find_qwen_loras, is_turbo_lora
                roots = comfy_lora_roots(comfy_dir)
                found = [(p, n, "turbo" if is_turbo_lora(Path(p)) else "h3") for p, n in find_h3_loras(roots)]
                found += [(p, n, "qwen") for p, n in find_qwen_loras(roots)]
                from ..imagegen import find_image_loras
                for fam, items in find_image_loras(roots).items():
                    found += [(p, n, fam) for p, n in items]
            except Exception:  # noqa: BLE001
                return                           # keep the lists found before (a folder may be offline)
            if gen != getattr(self, "_lora_scan_gen", gen):
                return                           # a newer scan was started (e.g. another ComfyUI folder)
            try:
                bridge.loras.emit(found, report)
            except RuntimeError:                 # (the window was closed meanwhile)
                pass
        threading.Thread(target=work, daemon=True).start()

    def _loras_found(self, found: list, report: bool = False):
        self.h3_loras = [(p, n) for p, n, kind in found if kind == "h3"]
        self.turbo_loras = [(p, n) for p, n, kind in found if kind == "turbo"]
        self.qwen_loras = [(p, n) for p, n, kind in found if kind == "qwen"]     # for the image edit window
        self.image_loras = {fam: [(p, n) for p, n, kind in found if kind == fam]
                            for fam in ("flux1", "sdxl", "krea2", "qwen21", "moss")}
        self._scan_triggers()
        if report:
            self.status.setText(f"{len(self.h3_loras)} LoRA{'s' if len(self.h3_loras) != 1 else ''} and "
                                f"{len(self.turbo_loras)} turbo LoRA{'s' if len(self.turbo_loras) != 1 else ''} in "
                                "ComfyUI's LoRA folders.")
        if self.tree is not None:
            self._fill_turbo((self.tree.options.get("turbo") or {}).get("lora"))
        if self.tree is None:
            return
        self._refresh_card_choices()

    def _refresh_card_choices(self):
        """The cards' lists show the project's LoRA, length and steps as 'default': update them in place."""
        if self.tree is None:
            return
        known = self.known_loras()
        for card in [c for c in self.cards if isinstance(c, PromptCard)]:
            card.set_choices(int(self.tree.options.get("frames", 73)), int(self.tree.options.get("steps", 28)), known)

    def _lora_dir(self) -> str:
        """Where to start looking for LoRA files: ComfyUI's first LoRA folder (the ones it can load)."""
        try:
            from ..comfy import comfy_lora_roots
            roots = comfy_lora_roots(Path(os.path.expanduser(self.comfy_dir.text().strip() or DEFAULT_COMFY_DIR)))
            extra = [r for r in roots if "models/loras" not in r.as_posix()]
            return str((extra or roots)[0]) if roots else ""
        except Exception:  # noqa: BLE001
            return ""

    def _remember_lora(self, path: str):
        try:
            recent = json.loads(self.qs.value("recentLoras", "[]", type=str))
        except ValueError:
            recent = []
        recent = [path] + [r for r in recent if r != path]
        self.qs.setValue("recentLoras", json.dumps(recent[:12]))

    def _overrides_changed(self, nid: str, ov: dict):
        if self.tree and nid in self.tree.nodes and not self.tree.nodes[nid].frozen:
            self.tree.nodes[nid].overrides = dict(ov)
            self._changed()

    def _lora_picked(self, nid: str, path: str):
        """A LoRA chosen on a card: remembered (a file from elsewhere shows up in the lists later); a warning if
        ComfyUI cannot load it."""
        listed = {os.path.realpath(p) for p, _n in self.h3_loras}
        if os.path.realpath(path) not in listed:
            self._remember_lora(path)
        if self.in_comfy_loras(path) is False:
            self.status.setText(f"{Path(path).name} is not in ComfyUI's LoRA folders, so ComfyUI cannot load it: copy "
                                f"or link it into {self._lora_dir() or 'its models/loras folder'} (then Rescan LoRAs).")

    def _choose_node_lora(self, nid: str):
        """Add a LoRA file from any folder to a clip's stack."""
        if self.tree is None or nid not in self.tree.nodes:
            return
        stack = T.node_loras(self.tree.nodes[nid])
        start = str(Path(stack[-1][0]).parent) if stack else self._lora_dir()
        f, _ = QFileDialog.getOpenFileName(self, "LoRA for this clip", start, "LoRA (*.safetensors)")
        if f:
            self.add_node_lora(nid, f)

    def add_node_lora(self, nid: str, path: str, strength: float = 1.0):
        """Put a LoRA file at the end of a clip's stack."""
        if self.tree is None or self.running or nid not in self.tree.nodes or self.tree.nodes[nid].frozen:
            return
        ov = dict(self.tree.nodes[nid].overrides)
        p = lora_path(path)
        ov["loras"] = list(ov.get("loras") or []) + [{"path": p, "strength": float(strength)}]
        self._remember_lora(p)
        self.tree.nodes[nid].overrides = ov
        self._changed()
        self.refresh(rebuild_path=True)
        if self.in_comfy_loras(p) is False:
            self.status.setText(f"{Path(p).name} is not in ComfyUI's LoRA folders, so ComfyUI cannot load it: copy or "
                                f"link it into {self._lora_dir() or 'its models/loras folder'} (then Rescan LoRAs).")
        else:
            self.status.setText(f"Clip #{nid[1:]}: added {Path(p).stem}.")

    def in_comfy_loras(self, path: str) -> bool | None:
        """Whether ComfyUI can load this LoRA file: it lies in one of ComfyUI's LoRA folders (links followed), or
        is a LoRA found there. None: ComfyUI's folder is not known here."""
        try:
            from ..comfy import comfy_lora_roots
            roots = comfy_lora_roots(Path(os.path.expanduser(self.comfy_dir.text().strip() or DEFAULT_COMFY_DIR)))
        except Exception:  # noqa: BLE001
            roots = []
        if not roots:
            return None
        real = os.path.realpath(path)
        for r in roots:
            for top in {str(r), os.path.realpath(r)}:
                if os.path.abspath(path).startswith(top.rstrip(os.sep) + os.sep) or \
                        real.startswith(top.rstrip(os.sep) + os.sep):
                    return True
        return any(os.path.realpath(q) == real for q, _n in self.h3_loras + self.turbo_loras)

    def _seed_changed(self, nid: str, seed: int):
        if self.tree and nid in self.tree.nodes and not self.tree.nodes[nid].frozen:
            self.tree.nodes[nid].seed = int(seed)
            self._changed()

    def _new_seed(self) -> int:
        return self.tree.new_seed()

    def node_action(self, action: str, nid: str | None):
        if self.tree is None or self.running:
            return
        if action in ("take", "take_clear") and self.film is not None:
            cur = self.film.scene_of(self.project)
            if cur is not None and nid:
                try:
                    self.film.set_take(cur.id, nid if action == "take" else None)
                except OSError as e:
                    self.status.setText(f"Could not save the scene's ending: {e}")
                    return
                self._update_continues()               # a scene continuing this one starts from the new end
                self._show_storyboard()
                self.status.setText(f"The film plays “{cur.title}” up to clip #{nid[1:]}." if action == "take"
                                    else f"The film plays “{cur.title}” up to its selected clip.")
            return
        node = self.tree.nodes.get(nid) if nid and nid != START else None
        if action in ("freeze_toggle", "freeze", "unfreeze") and node is not None:
            if action == "unfreeze" or action == "freeze_toggle" and node.frozen:
                self.unfreeze_clip(node.id)
            else:
                self.freeze_clip(node.id)
            return
        if node is not None and self._frozen_block(node, action):
            return
        if action in ("variant", "variants") and node is not None and node.is_video:
            self.status.setText("Variants are for generated clips; a real video stays as it is.")
            return
        if action == "video":
            self._ask_video(node.id if node else None)
            return
        if action == "swap" and node is not None and node.is_video:
            self.replace_person(node.id)
            return
        if action == "swap_undo" and node is not None and node.is_video:
            self.undo_replacement(node.id)
            return
        if action == "edit" and node is not None and node.is_video:
            self.edit_video(node.id)
            return
        if action == "split" and node is not None and node.is_video:
            self.split_video(node.id)
            return
        if action == "end_image" and node is not None and not node.is_video:
            self.choose_end_image(node.id)
            return
        if action == "end_video" and node is not None and not node.is_video:
            self.end_from_video(node.id)
            return
        if action == "end_ai" and node is not None and not node.is_video:
            self.edit_image(node.id)
            return
        if action == "edit_ai" and node is not None:
            self.edit_image(node.id, from_node=True)
            return
        if action == "end_clear" and node is not None and not node.is_video:
            self.set_end_image(node.id, None)
            return
        if action == "revert" and node is not None and not node.is_video:
            self.revert_clip(node.id)
            return
        if action == "speak" and node is not None and not node.is_video:
            self.speak_clip(node.id)
            return
        if action == "refine" and node is not None and not node.is_video:
            card = next((c for c in self.cards if c.nid == node.id and isinstance(c, PromptCard)), None)
            if card is not None:
                ck = T.Checker(self.project, self.tree)
                start = ck.start_frame(node)
                end = T.end_image_used(self.tree, node)
                pics = [("The frame the clip starts from", str(start), True)] if start and start.is_file() else []
                for k in node.keyframes:
                    if (self.project / k["image"]).is_file():
                        pics.append((f"The picture the clip passes through at {k['at']:.1f} s",
                                     str(self.project / k["image"]), True))
                if end and (self.project / end).is_file():
                    pics.append(("The image the clip should end at", str(self.project / end), True))
                new = self.refine_text(card.edit.toPlainText(), "video", pictures=pics)
                if new:
                    replace_text(card.edit, new)          # Ctrl+Z in the prompt takes it back
                    self.status.setText(f"Clip #{node.id[1:]}: prompt refined (Ctrl+Z in the prompt undoes it).")
            return
        if action == "handover" and node is not None and not node.is_video:
            self.choose_handover(node.id)
            return
        if action == "handover_clear" and node is not None and not node.is_video:
            self.set_handover(node.id, None)
            return
        if action == "compare" and node is not None:
            self.compare_variants(node.id)
            return
        if action == "continue":
            new = self.tree.add(node.id if node else None, "")
        elif action == "variants" and node is not None:
            dlg = VariantsDialog(self, node)
            if dlg.exec() == QDialog.Accepted:
                self.create_variants(node.id, dlg.count.value(), dlg.generate.isChecked(),
                                     strength_steps=dlg.strength_steps(), keep_seed=dlg.keep_seed())
            return
        elif action in ("branch", "variant") and node is not None:
            new = self.tree.add(node.parent, node.prompt if action == "variant" else "",
                                self._new_seed() if action == "variant" else None)
            if action == "variant":                # the same clip with another seed: its own settings too
                new.overrides = copy.deepcopy(node.overrides)
                self._copy_video_child(node.id, new.id)
        elif action == "delete" and node is not None:
            if self._busy_files():
                return
            below = len(self.tree.descendants(node.id))
            made = node.made is not None or any(d.made for d in self.tree.descendants(node.id))
            msg = f"Delete {'video' if node.is_video else 'clip'} #{node.id[1:]}" + (f" and the {below} clip{'s' if below != 1 else ''} after it"
                                                   if below else "") + "?"
            if made:
                msg += "\nTheir video files are deleted too."
            if QMessageBox.question(self, "Delete", msg) != QMessageBox.Yes:
                return
            self.tree.remove(node.id, self.project)
            if self.tree.selected is None:
                self.tree.selected = START
            self._write_tree()
            self.refresh()
            return
        else:
            return
        self.tree.selected = new.id
        self._write_tree()
        self.refresh()
        self.graph.reveal(new.id)
        if self.cards and self.cards[-1].nid == new.id and isinstance(self.cards[-1], PromptCard):
            self.cards[-1].edit.setFocus()

    # what a frozen clip / video cannot do: it would change the clip as made (or what it ends at)
    FROZEN_LOCKED = {"end_image", "end_video", "end_ai", "end_clear", "revert", "refine", "speak", "handover",
                     "handover_clear", "edit", "split", "swap", "swap_undo", "video"}

    def _frozen_block(self, node: T.Node, action: str) -> bool:
        """True (with a word in the status bar) when the action would change a frozen clip."""
        if not node.frozen or action not in self.FROZEN_LOCKED or action == "video" and node.is_video:
            return False                           # (a video may follow a frozen video: nothing of it changes)
        what = "Video" if node.is_video else "Clip"
        self.status.setText(f"{what} #{node.id[1:]} is frozen (kept as made): unfreeze it first (the lock button on "
                            "its card, or right-click → Unfreeze).")
        return True

    def freeze_clip(self, nid: str, confirm: bool = True) -> list[str]:
        """Keep a made clip as it is, with every clip before it (tree.freeze). Returns the ids frozen now."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        what = "video" if self.tree.nodes[nid].is_video else "clip"
        why = T.freeze_problem(self.project, self.tree, nid)
        if why:
            self.status.setText(f"Cannot freeze {what} #{nid[1:]}: {why}.")
            return []
        edits = T.freeze_edits(self.tree, nid)
        if confirm and edits and QMessageBox.question(
                self, "Freeze",
                f"Freeze {what} #{nid[1:]} and the clips before it?\n\nThese edits are not made yet and are undone "
                "(back to what the clips were made with):\n" +
                "\n".join(f"#{i[1:]}: {', '.join(e)}" for i, e in edits.items())) != QMessageBox.Yes:
            return []
        try:
            ids = T.freeze(self.project, self.tree, nid)
        except (ValueError, OSError) as e:
            self.status.setText(f"Cannot freeze {what} #{nid[1:]}: {e}.")
            return []
        self._pix.clear()
        self._write_tree()
        self.refresh(rebuild_path=True)
        more = [i for i in ids if i != nid]
        self.status.setText(f"{what.capitalize()} #{nid[1:]} frozen" +
                            (f", with {len(more)} before it" if more else "") +
                            ": kept as made whatever the project's settings say; the clips after it follow them.")
        return ids

    def unfreeze_clip(self, nid: str) -> list[str]:
        """A frozen clip (and the frozen clips after it) follows the settings again. Returns the ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        ids = T.unfreeze(self.tree, nid)
        if not ids:
            return []
        self._write_tree()
        self.refresh(rebuild_path=True)
        changed = [i for i in ids if self.status_of.get(i) == "changed"]
        more = [i for i in ids if i != nid]
        text = (f"#{nid[1:]} unfrozen" + (f", with {', '.join('#' + i[1:] for i in more)}" if more else ""))
        if changed:
            text += (f"; {len(changed)} of them count{'s' if len(changed) == 1 else ''} as CHANGED now (the settings "
                     "differ): Generate makes them again. Freeze again to keep them.")
        self.status.setText(text + ".")
        return ids

    def _busy_files(self, what: str = "delete") -> bool:
        """A path is being joined or a video converted (they read / write clip files): no deleting (or changing
        hand-over frames) meanwhile."""
        if self._joins:
            self.status.setText(f"Wait until the path is joined, then {what}.")
        elif self._converts:
            self.status.setText(f"Wait until the video is converted, then {what}.")
        return bool(self._joins or self._converts)

    def _copy_video_child(self, source: str, target: str) -> None:
        """A variant of a clip that leads into a real video leads into its own copy of that video; one with an
        end image gets its own copy of the image."""
        vc = self.tree.video_child(source)
        if vc is not None and self.tree.can_add_video(target):
            T.copy_video(self.project, self.tree, vc, target)
        try:
            T.copy_end_image(self.project, self.tree, source, target)
            T.copy_keyframes(self.project, self.tree, source, target)
        except (OSError, RuntimeError, ValueError) as e:
            self.status.setText(f"Could not copy the end image or the keyframes: {e}")

    def revert_clip(self, nid: str) -> list[str]:
        """Undo the edits of a made clip (prompt, seed, own settings, end image) so it counts as done again."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        done = T.revert(self.project, self.tree, nid)
        if not done:
            return []
        self._pix.clear()
        self._write_tree()
        self.refresh(rebuild_path=True)
        why = self.why_of.get(nid)
        self.status.setText(f"Clip #{nid[1:]}: {' and '.join(done)} back as made" +
                            (f"; it is still changed: {why}." if why else "; it is done again."))
        return done

    # ---- prompt refiner
    def refine_text(self, text: str, kind: str, pictures: list | None = None) -> str | list[str] | None:
        """Open the prompt refiner for a prompt ('video' / 'image' / 'edit'); the text to use, or None."""
        if not text.strip():
            self.status.setText("Write a prompt first; the refiner improves it.")
            return None
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL)
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None

        llama = self.llama_server()

        def connect():
            from ..comfy import ComfyPromptRefiner
            from ..llm_server import CombinedRefiner
            if llama is None:
                return ComfyPromptRefiner(url, log=lambda _m: None, comfy_dir=comfy_dir, cleanup=False)
            try:
                comfy, err = ComfyPromptRefiner(url, log=lambda _m: None, comfy_dir=comfy_dir, cleanup=False), ""
            except Exception as e:  # noqa: BLE001  (ComfyUI down: the llama.cpp model still refines)
                comfy, err = None, str(e)
            return CombinedRefiner(comfy, llama, err)
        cast = [{"name": c["name"], "desc": c.get("desc", ""), "voice": bool(c.get("voice"))}
                for c in T.cast_of(self.tree)] if self.tree is not None and kind == "video" else []
        dlg = RefineDialog(self, text, kind, connect, self.qs, start_comfy=self.start_comfy, cast=cast,
                           pictures=pictures or [])
        try:
            if dlg.exec() != QDialog.Accepted:
                return None
            ideas = dlg.chosen_ideas() if hasattr(dlg, "chosen_ideas") else None
            return ideas if ideas else dlg.chosen()       # (a list: all ideas, for the image window's variations)
        finally:
            # the llama.cpp model goes with the window: it slows image / clip generation down while it is loaded
            self.stop_llama("the prompt refiner was closed")
            if dlg.refined_any and not self.running and self._comfy_model not in ("qwen",):
                # a language model is in ComfyUI now: unloaded before the next clips, idle time as usual
                self._comfy_model = "llm"
                if self.keep_loaded.isChecked():
                    self._qwen_timer.start(self.idle_minutes.value() * 60_000)
                self._update_model_label()
            dlg.deleteLater()

    # ---- AI image edit
    def edit_image(self, nid: str | None = None, from_node: bool = False, start: bool = False,
                   dialog: bool = True, mode: str = "auto", character: str | None = None) -> ImageEditDialog | None:
        """The image edit dialog. nid: the clip whose end image a result can become (its frame is offered first);
        start: the start image first; character: opened for a face picture of this character (its name on the
        face button). Returns the dialog (dialog=False: not shown, for tests)."""
        if self.project is None or self.tree is None:
            self.status.setText("Open or create a project folder first.")
            return None
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return None
        if self._waiting_comfy is not None:
            self.status.setText("Clips are generated as soon as ComfyUI runs; wait for them (or Stop) first.")
            return None
        node = self.tree.nodes.get(nid) if nid else None
        sources: list[tuple[str, str]] = []

        def add(text, path):
            if path is not None and Path(path).is_file() and str(path) not in [p for _t, p in sources]:
                sources.append((text, str(path)))
        if node is not None and node.made:
            f = self._node_frame(node)
            add(("First frame of video" if node.is_video else
                 "Hand-over frame of clip" if T.handover_index(self.tree, node) is not None else
                 "Last frame of clip") + f" #{node.id[1:]}",
                self.project / node.made["first"] if node.is_video else f)
            if node.is_video:
                add(f"Last frame of video #{node.id[1:]}", self.project / node.made["frame"])
        if node is not None and node.end_image:
            add(f"End image of clip #{node.id[1:]}", self.project / node.end_image)
        add("Start image", self.project / T.START_FILE)
        for n in self.selected_path():
            if n.made and n.id != nid:
                add(f"{'Video' if n.is_video else 'Clip'} #{n.id[1:]}: its last frame",
                    self._node_frame(n) if not n.is_video else self.project / n.made["frame"])
        for c in T.cast_of(self.tree):
            for k, rel in enumerate(c["images"], start=1):
                add(f"Face of @{c['name']}" + (f" ({k})" if len(c["images"]) > 1 else ""), self.project / rel)
        first = None
        if start:
            first = str(self.project / T.START_FILE) if (self.project / T.START_FILE).is_file() else None
        target = nid if node is not None and not node.is_video and self.tree.video_child(nid) is None else None
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL)
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        cleanup = self.comfy_cleanup.isChecked()

        switch = {"free": False, "loaded": False}

        def before_edit():                          # GUI thread, before each Generate
            self._qwen_timer.stop()
            self.stop_llama("its memory is needed for the image")
            # the clip model is loaded: unload it (once: after the first image of this window it is gone)
            switch["free"] = not switch["loaded"] and self._comfy_model in ("h3", "h3-hyperflow")
            if self.worker_alive and not self.running:
                self._quit_worker(free=False)               # its model goes now; it is started again for clips

        def connect(progress):                      # worker thread
            from ..comfy import ComfyImageEditor
            ed = ComfyImageEditor(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            if switch["free"]:
                ed.free()
                switch["free"] = False
            switch["loaded"] = True
            return ed

        def connect_create(progress):               # worker thread: the image creator (any image family)
            from ..imagegen import ComfyImageCreator
            ed = ComfyImageCreator(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            if switch["free"]:
                ed.free()
                switch["free"] = False
            if progress is not None:                # (None: only listing the models)
                switch["loaded"] = True             # an image model is loaded (unloaded before the next clips)
            return ed

        def use(kind: str, png: str):
            if self.running:
                self.status.setText("Wait until generating has finished.")
                return False
            if kind == "ref":                       # one more face reference; the dialog stays open
                if not self.add_reference(png):
                    return False
                c = self._selected_character()
                record_use(png, {"kind": "ref", "file": c["images"][-1]})
                return (f"Added as a face of @{c['name']} (the Cast tab, Alt+2); mention it in prompts as "
                        f"@{c['name']}." + (f" {self.status.text()}" if "CHANGED" in self.status.text() else ""))
            ok = self.set_image(png, confirm=True) if kind == "start" else \
                bool(target) and self.set_end_image(target, png)
            if ok:
                record_use(png, {"kind": "start", "sha1": chain.file_signature(self.project / T.START_FILE)}
                           if kind == "start" else {"kind": "end", "file": self.tree.nodes[target].end_image})
                self.status.setText(("New start image from the AI edit" if kind == "start" else
                                     f"Clip #{target[1:]} now ends at the AI-edited image") +
                                    (f" — {self.status.text()}" if "CHANGED" in self.status.text() else "."))
            return bool(ok)
        stems = [Path(n).stem for _p, n in self.qwen_loras]
        known = [(p, f"{Path(n).stem} ({Path(n).parent})" if stems.count(Path(n).stem) > 1 and str(Path(n).parent) != "."
                  else Path(n).stem) for p, n in self.qwen_loras]
        def labelled(items):
            st = [Path(n).stem for _p, n in items]
            return [(p, f"{Path(n).stem} ({Path(n).parent})" if st.count(Path(n).stem) > 1 and str(Path(n).parent) != "."
                     else Path(n).stem) for p, n in items]
        family_loras = {fam: labelled(items) for fam, items in self.image_loras.items()}
        family_loras["qwen"] = known
        aspect = None
        if (self.project / T.START_FILE).is_file():
            img = QImage(str(self.project / T.START_FILE))
            aspect = img.width() / img.height() if img.height() else None
        dlg = ImageEditDialog(self, self.project, sources, first, target, connect, use, self.qs,
                              self._picker_places(), start_comfy=self.start_comfy, on_start=before_edit,
                              known_loras=known, lora_dir=self._lora_dir(),
                              uses=lambda meta: uses_of(self.project, self.tree, meta),
                              refine=self.refine_text, connect_create=connect_create, family_loras=family_loras,
                              start_aspect=aspect, mode=mode, check_lora=self.in_comfy_loras,
                              collections=self.project_images,
                              use_triggers=bool(self.tree.options.get("lora_triggers", True)))
        self._edit_switch = switch
        if character:
            dlg.b_use_ref.setText(f"Add as face of @{character}")
        if dialog:
            dlg.exec()
            dlg.deleteLater()
            self.edits_closed()
        return dlg

    def edits_closed(self):
        """The edit dialog is closed: the image model stays loaded for the idle time ('Keep the model loaded'), else
        it is unloaded now."""
        sw = getattr(self, "_edit_switch", None) or {}
        if sw.get("loaded"):                      # an image job connected (also when it finished after closing)
            self._comfy_model = "qwen"
        if self._comfy_model != "qwen" or self.running:
            return
        if self.keep_loaded.isChecked():
            self._qwen_timer.start(self.idle_minutes.value() * 60_000)
        else:
            self._free_comfy()
        self._update_model_label()

    # ---- hand-over frame
    def choose_handover(self, nid: str) -> None:
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return
        if self.tree.video_child(nid) is not None:
            self.status.setText(f"Clip #{nid[1:]} ends at the video after it; it hands over there.")
            return
        if not node.made or not (self.project / node.made["video"]).is_file():
            self.status.setText(f"Make clip #{nid[1:]} first; then choose the frame it hands over at.")
            return
        # a child with a motion guide needs that many frames up to the hand-over frame
        need = max([int((c.overrides or {}).get("guide") or 0) for c in self.tree.children(nid)
                    if not c.is_video] + [1])
        dlg = HandoverDialog(self, self.project / node.made["video"], f"Hand-over frame of clip #{nid[1:]}",
                             T.handover_index(self.tree, node), min_frame=max(1, need - 1))
        try:
            if dlg.exec() != QDialog.Accepted:
                return
            frame = dlg.frame()
        finally:
            dlg.deleteLater()                       # it holds every decoded frame
        self.set_handover(nid, frame)

    def set_handover(self, nid: str, frame: int | None) -> bool:
        """The clips after a made clip start from this frame of it (None: its last frame, as usual)."""
        if self.tree is None or self.running or nid not in self.tree.nodes or \
                self._frozen_block(self.tree.nodes[nid], "handover") or \
                self._busy_files("change the hand-over frame"):
            return False
        node = self.tree.nodes[nid]
        if node.is_video:
            return False
        if node.made and not node.made.get("frames") and (self.project / node.made["video"]).is_file():
            try:                                    # an older record: the real count, once
                node.made["frames"] = chain.probe(self.project / node.made["video"])["frames"]
            except (OSError, RuntimeError):
                pass
        n = T.made_frames(node) or T.clip_length(self.tree, node)
        frame = None if frame is None or int(frame) >= n - 1 else max(1, int(frame))
        if frame == node.end_at:
            return False
        before = self._done_clips()
        node.end_at = frame
        try:
            T.ensure_handover(self.project, self.tree, node)
        except (OSError, RuntimeError) as e:
            node.end_at = None
            self.status.setText(f"Could not read that frame of clip #{nid[1:]}: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self.refresh()
        what = (f"Clip #{nid[1:]} hands over at frame {frame + 1} ({frame / 24:.2f} s)" if frame is not None
                else f"Clip #{nid[1:]} hands over at its last frame")
        if not self._report_remade(before, what):
            self.status.setText(what + ".")
        return True

    # ---- compare
    def compare_candidates(self, nid: str) -> list[str]:
        """The made clips from the same point as nid (nid first), in number order."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None:
            return []
        sibs = [n for n in self.tree.children(node.parent) if n.id != nid and n.made and not n.is_video
                and self.status_of.get(n.id) == "done"]
        own = [nid] if node.made else []
        return own + [n.id for n in sorted(sibs, key=lambda n: self.tree.number(n.id))]

    def compare_variants(self, nid: str, player: bool = True) -> CompareDialog | None:
        """Play nid and the other made clips from the same point side by side; 'Keep this one' deletes the others
        (and what follows them)."""
        if self.tree is None or self.running:
            return None
        ids = self.compare_candidates(nid)[:9]
        if len(ids) < 2:
            self.status.setText("Nothing to compare: make variants of this clip first (Variant / Variants…).")
            return None
        items = []
        for i in ids:
            n = self.tree.nodes[i]
            detail = " · ".join([f"seed {n.seed}"] + self._own_parts(n) +
                                ([] if n.prompt == self.tree.nodes[nid].prompt else [f"“{n.prompt[:60]}”"]))
            items.append((i, f"#{i[1:]}", detail, self.project / n.made["video"]))
        self.preview.stop()

        def keep(kid: str) -> bool:
            others = [i for i in ids if i != kid and i in self.tree.nodes]
            return bool(self.keep_only(kid, others))
        dlg = CompareDialog(self, items, keep, player=player,
                            title=f"Compare {len(ids)} clips from the same point as #{nid[1:]}")
        if player:
            dlg.exec()
            dlg.deleteLater()                       # its video players
        return dlg

    def keep_only(self, nid: str, others: list[str], confirm: bool = True) -> list[str]:
        """Delete these other clips (with what follows them); nid stays and is selected."""
        if self.tree is None or self.running or not others or self._busy_files():
            return []
        doomed = others + [d.id for r in others for d in self.tree.descendants(r)]
        made = sum(1 for d in doomed if self.tree.nodes[d].made)
        if confirm:
            msg = (f"Keep clip #{nid[1:]} and delete {len(doomed)} clip{'s' if len(doomed) != 1 else ''} "
                   f"({', '.join('#' + o[1:] for o in others)}" + (" and what follows them" if len(doomed) >
                                                                  len(others) else "") + ")?")
            if made:
                msg += f"\n{made} of them are made; their video files are deleted too."
            if QMessageBox.question(self, "Keep this one", msg) != QMessageBox.Yes:
                return []
        gone = []
        for r in others:
            if r in self.tree.nodes:
                gone += self.tree.remove(r, self.project)
        self.tree.selected = nid
        self._write_tree()
        self.refresh()
        self.status.setText(f"Kept clip #{nid[1:]}; deleted {len(gone)} clip{'s' if len(gone) != 1 else ''}.")
        return gone

    # ---- end images
    def choose_end_image(self, nid: str) -> None:
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return
        if self.tree.video_child(nid) is not None:
            self.status.setText(f"Clip #{nid[1:]} ends at the first frame of the video after it.")
            return
        start = str(Path(self.project / node.end_image).parent) if node.end_image else None
        f = pick_image(self, f"End image of clip #{nid[1:]} (the frame it steers its end to)", self.qs, start,
                       self._picker_places(), self.project_images())
        if f:
            self.set_end_image(nid, f)

    def end_from_video(self, nid: str, video: str | None = None, dialog: bool = True):
        """End image from a video: choose a video (the project's real videos and clips first, or any file), then
        one of its frames; that frame at the video's full size becomes the clip's end image."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return None
        if self.tree.video_child(nid) is not None:
            self.status.setText(f"Clip #{nid[1:]} ends at the first frame of the video after it.")
            return None
        if video is None:
            clips = [(str(self.project / n.made["video"]), f"Clip #{n.id[1:]} · {n.prompt[:40]}")
                     for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id))
                     if not n.is_video and n.made and (self.project / n.made["video"]).is_file() and n.id != nid]
            cols = self.project_media("video") + ([("Clips of this project", clips)] if clips else [])
            video = choose_video(self, self.qs, places=self._picker_places(), collections=cols)
            if not video:
                return None
        try:
            dlg = VideoFrameDialog(self, Path(video), f"End image of clip #{nid[1:]}: a frame of {Path(video).name}")
        except (RuntimeError, OSError) as e:
            self.status.setText(f"Cannot read {Path(video).name}: {e}")
            return None
        if not dialog:
            return dlg
        try:
            if dlg.exec() != QDialog.Accepted:
                return None
            import tempfile
            with tempfile.TemporaryDirectory(prefix="video-chain-end-") as tmp:
                png = Path(tmp) / f"{Path(video).stem}_f{dlg.slider.value()}.png"
                if not dlg.frame_file(png):
                    self.status.setText(f"Could not read that frame of {Path(video).name}.")
                    return None
                if self.set_end_image(nid, str(png)):
                    self.status.setText(f"Clip #{nid[1:]} now ends at frame {dlg.slider.value() + 1} "
                                        f"({dlg.at():.2f} s) of {Path(video).name}.")
        finally:
            dlg.deleteLater()
        return None

    def set_end_image(self, nid: str, image: str | None) -> bool:
        """Give a generated clip its own end image (None: remove it); the clip then counts as changed."""
        if self.tree is None or self.running or nid not in self.tree.nodes or \
                self._frozen_block(self.tree.nodes[nid], "end_image"):
            return False
        try:
            if image is None:
                if not self.tree.nodes[nid].end_image:
                    return False
                T.clear_end_image(self.project, self.tree, nid)
            else:
                if QImage(image).isNull():
                    self.status.setText(f"Not an image: {Path(image).name}")
                    return False
                T.set_end_image(self.project, self.tree, nid, Path(image))
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not set the end image: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self.refresh()
        self.status.setText(f"Clip #{nid[1:]} " + ("now ends at its end image." if image else "ends freely now."))
        return True

    # ---- middle keyframes
    def keyframe_action(self, act: str, nid: str, data: dict | None = None) -> None:
        """A keyframe action of a clip card: add / add_video / add_moment (data "at"), move (data "index", "at"),
        change / video / moment / note / remove (data "index")."""
        if self.tree is not None and nid in self.tree.nodes and self.tree.nodes[nid].frozen:
            self._frozen_block(self.tree.nodes[nid], "end_image")
            return
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return
        data = data or {}
        i, at = data.get("index"), data.get("at")
        if i is not None:
            if not 0 <= i < len(node.keyframes):
                return
            if at is None:
                at = node.keyframes[i]["at"]
        if at is None:
            return
        if act == "move":
            self._set_keyframe(nid, None, at, index=i)
        elif act in ("add", "change"):
            f = pick_image(self, f"Keyframe of clip #{nid[1:]} at {at:.2f} s (a picture the clip passes through)",
                           self.qs, None, self._picker_places(), self.project_images())
            if f:
                self._set_keyframe(nid, f, at, index=i)
        elif act in ("add_video", "video", "add_moment", "moment"):
            self._keyframe_from_video(nid, at, i, own=act in ("add_moment", "moment"))
        elif act == "note":
            text, ok = QInputDialog.getText(
                self, f"Keyframe at {at:.2f} s", "What happens at this moment (added to the prompt with its time, "
                "e.g. “she turns to the camera”; empty: nothing):", text=node.keyframes[i]["note"])
            if ok:
                self._set_keyframe(nid, None, at, note=text.strip(), index=i)
        elif act == "remove":
            T.remove_keyframe(self.project, self.tree, nid, i)
            self._write_tree()
            self.refresh()
            self.status.setText(f"Clip #{nid[1:]}: keyframe at {at:.2f} s removed.")

    def _keyframe_from_video(self, nid: str, at: float, index: int | None, own: bool = False,
                             video: str | None = None, dialog: bool = True):
        """A keyframe from a frame of a video. own: this clip's frame at that moment ("keep this moment": its time
        follows the frame chosen); else any video (the project's first), the keyframe stays at its time."""
        node = self.tree.nodes[nid]
        if own:
            if not node.made or not (self.project / node.made["video"]).is_file():
                self.status.setText(f"Clip #{nid[1:]} is not made yet: there is no moment of it to keep.")
                return None
            video = str(self.project / node.made["video"])
        elif video is None:
            clips = [(str(self.project / n.made["video"]), f"Clip #{n.id[1:]} · {n.prompt[:40]}")
                     for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id))
                     if not n.is_video and n.made and (self.project / n.made["video"]).is_file()]
            cols = self.project_media("video") + ([("Clips of this project", clips)] if clips else [])
            video = choose_video(self, self.qs, places=self._picker_places(), collections=cols)
            if not video:
                return None
        title = (f"Keep a moment of clip #{nid[1:]}: the frame to keep (it is pinned at its time)" if own else
                 f"Keyframe of clip #{nid[1:]} at {at:.2f} s: a frame of {Path(video).name}")
        try:
            dlg = VideoFrameDialog(self, Path(video), title)
        except (RuntimeError, OSError) as e:
            self.status.setText(f"Cannot read {Path(video).name}: {e}")
            return None
        if own:
            dlg.slider.setValue(min(dlg.frames - 2, max(1, T.keyframe_frame(at))))
        if not dialog:
            return dlg
        try:
            if dlg.exec() != QDialog.Accepted:
                return None
            return self._keyframe_from_dialog(nid, dlg, at, index, own)
        finally:
            dlg.deleteLater()

    def _keyframe_from_dialog(self, nid: str, dlg, at: float, index: int | None, own: bool) -> bool:
        import tempfile
        k = dlg.slider.value()
        if own:                                          # (never its first or last frame)
            k = min(dlg.frames - 2, max(1, k))
            dlg.slider.setValue(k)
            at = k / chain.FPS
        with tempfile.TemporaryDirectory(prefix="video-chain-key-") as tmp:
            png = Path(tmp) / f"key_f{k}.png"
            if not dlg.frame_file(png):
                self.status.setText("Could not read that frame.")
                return False
            return self._set_keyframe(nid, str(png), at, index=index)

    def _set_keyframe(self, nid: str, image: str | None, at: float, note: str | None = None,
                      index: int | None = None) -> bool:
        """Add or change a keyframe (image None: keeps its picture); the clip then counts as changed."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return False
        if image is not None and QImage(image).isNull():
            self.status.setText(f"Not an image: {Path(image).name}")
            return False
        frames = T.clip_length(self.tree, self.tree.nodes[nid])
        at = min((frames - 2) / chain.FPS, max(1 / chain.FPS, round(float(at) * chain.FPS) / chain.FPS))
        try:
            T.set_keyframe(self.project, self.tree, nid, Path(image) if image else None, at, note, index)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not set the keyframe: {e}")
            return False
        self._write_tree()
        self.refresh()
        msg = f"Clip #{nid[1:]}: keyframe at {at:.2f} s" + (" set." if index is None or image else " changed.")
        canvas = self._canvas_for()
        if image is not None and canvas:
            img = QImage(image)
            if abs(img.width() / img.height() - canvas[0] / canvas[1]) > 0.05 * canvas[0] / canvas[1]:
                msg += (f" Its shape ({img.width()}×{img.height()}) differs from the clip's ({canvas[0]}×{canvas[1]}):"
                        " H3 crops it to the middle.")
        self.status.setText(msg + " Keep it reachable by motion from what comes before it, or H3 cuts to it.")
        return True

    # ---- real videos
    def _canvas_for(self, file: str | None = None) -> tuple[int, int] | None:
        """The video size clips are made in; without a start image or video yet, the one this file would give."""
        try:
            base = T.Checker(self.project, self.tree).base()
            if base is not None:
                return base["width"], base["height"]
            if file:
                info = chain.probe_source(Path(file))
                return chain.canvas_size(info["width"], info["height"], int(self.tree.options.get("resolution", 512)))
        except Exception:  # noqa: BLE001
            pass
        return None

    def _ask_video(self, parent: str | None):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        if not self.tree.can_add_video(parent):
            vc = self.tree.video_child(parent)
            self.status.setText(f"Clip #{parent[1:]} already leads into video #{vc.id[1:]}: a generated clip can "
                                "end at one video only. Branch from the clip before it for another one.")
            return
        f = choose_video(self, self.qs, places=self._picker_places(), collections=self.project_media("video"))
        if not f:
            return
        dlg = VideoRangeDialog(self, f, canvas=self._canvas_for(f), settings=self.qs)
        if dlg.exec() != QDialog.Accepted or not dlg.file:
            return
        self.add_video(parent, dlg.file, dlg.start, dlg.end)

    def add_video(self, parent: str | None, file: str, start: float, end: float) -> str | None:
        """Put a range of a real video after parent (None: at the start) and convert it right away."""
        if self.tree is None or self.running:
            return None
        try:
            node = self.tree.add_video(parent, os.path.abspath(file), start, end)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self._remember_video(file)
        self.tree.selected = node.id
        self._write_tree()
        self._show_start_image()                  # the size may follow this video now
        self.refresh()
        self.graph.reveal(node.id)
        self.convert_video(node.id)
        return node.id

    def edit_video(self, nid: str):
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running or self._frozen_block(node, "edit"):
            return
        src = node.source
        f = src["file"]
        if not Path(f).is_file():
            self.status.setText(f"{Path(f).name} is not there any more; choose the file again.")
        dlg = VideoRangeDialog(self, f if Path(f).is_file() else "", src["start"], src["end"],
                               canvas=self._canvas_for(f), settings=self.qs, on_reference=self.add_reference)
        if dlg.exec() != QDialog.Accepted or not dlg.file:
            return
        self.set_video_range(nid, dlg.file, dlg.start, dlg.end)

    def set_video_range(self, nid: str, file: str, start: float, end: float) -> None:
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return
        node = self.tree.nodes[nid]
        new = {"file": os.path.abspath(file), "start": round(float(start), 4), "end": round(float(end), 4)}
        if new["end"] <= new["start"]:
            self.status.setText("The end of the range must be after its start.")
            return
        if new == node.source:
            return
        before = self._done_clips() - {nid}
        node.source = new
        self._remember_video(file)
        self._write_tree()
        self._show_start_image()
        self.refresh(rebuild_path=True)
        self.convert_video(nid)
        if not self._report_remade(before, f"Video #{nid[1:]} has a new range (converting)"):
            self.status.setText(f"Video #{nid[1:]} has a new range: converting.")

    def _remember_video(self, file: str):
        self.qs.setValue("videoDir", str(Path(file).parent))

    def convert_video(self, nid: str) -> None:
        """Convert a video node's range in the background (the generator does it too if this is not done)."""
        if self.tree is None or nid not in self.tree.nodes:
            return
        node = self.tree.nodes[nid]
        try:
            checker = T.Checker(self.project, self.tree)
            base = checker.base()
            if base is None or checker.status(nid) == "done" or not Path(node.source["file"]).is_file():
                return
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Could not convert the video: {e}")
            return
        token = os.urandom(6).hex()
        self._converts[nid] = token
        self.live[nid] = "converting"
        work_node, project, bridge = copy.deepcopy(node), self.project, self.bridge
        self.status.setText(f"Converting video #{nid[1:]} …")

        def work():
            try:
                made = T.make_video(project, work_node, base)
                bridge.converted.emit(nid, made, "", token, str(project))
            except Exception as e:  # noqa: BLE001
                bridge.converted.emit(nid, None, str(e), token, str(project))
        threading.Thread(target=work, daemon=True).start()
        self.refresh()

    def _converted(self, nid: str, made, error: str, token: str, project: str):
        def discard():
            for k in ("video", "frame", "first"):
                if made and made.get(k):
                    T._unlink(Path(project) / made[k])
        if self._converts.get(nid) != token:         # a newer conversion of this node is running
            discard()
            return
        self._converts.pop(nid, None)
        if self.live.get(nid) == "converting":
            self.live.pop(nid, None)
        node = self.tree.nodes.get(nid) if self.tree is not None else None
        if project != str(self.project) or node is None or self.running:
            discard()
            self._update_buttons()
            return
        if error:
            self.status.setText(f"Could not convert video #{nid[1:]}: {error}")
            self.refresh()
            return
        key = made.get("key") or {}
        if {k: key.get(k) for k in ("file", "start", "end")} != {k: (node.source or {}).get(k)
                                                               for k in ("file", "start", "end")}:   # changed meanwhile
            discard()
            self.refresh()
            return
        old = T.clip_files(self.project, node)
        node.made = made
        if self._write_tree() and not self._joins:  # a running join may still read them; else the next run does
            for f in old:
                T._unlink(f)
        self._pix.clear()
        self.status.setText(f"Video #{nid[1:]} is ready ({made['frames']} frames).")
        self.refresh()

    def _activate(self, nid: str):
        """Double-click on a node: a real video opens its range editor, a generated clip plays."""
        if self.tree is not None and nid in self.tree.nodes and self.tree.nodes[nid].is_video and not self.running:
            QTimer.singleShot(0, lambda: self.edit_video(nid))
        else:
            self._play_node(nid)

    def keep_branch(self, nid: str, whole_path: bool = False, confirm: bool = True) -> list[str]:
        """Delete the other branches at this clip's point (its siblings), or along its whole path; the clip and
        what follows it stay. Returns the deleted ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        roots = self.tree.others_than_path(nid) if whole_path else [n.id for n in self.tree.siblings(nid)]
        if not roots or self._busy_files():
            return []
        doomed = [r for r in roots] + [d.id for r in roots for d in self.tree.descendants(r)]
        made = sum(1 for d in doomed if self.tree.nodes[d].made)
        if confirm:
            where = "along the path to" if whole_path else "next to"
            msg = (f"Delete {len(doomed)} clip{'s' if len(doomed) != 1 else ''} in "
                   f"{len(roots)} other branch{'es' if len(roots) != 1 else ''} {where} clip #{nid[1:]}?")
            if made:
                msg += f"\n{made} of them are made; their video files are deleted too."
            if QMessageBox.question(self, "Keep this branch", msg) != QMessageBox.Yes:
                return []
        gone = []
        for r in roots:
            if r in self.tree.nodes:
                gone += self.tree.remove(r, self.project)
        self.tree.selected = nid
        self._write_tree()
        self.refresh()
        self.status.setText(f"Kept clip #{nid[1:]}; deleted {len(gone)} clip{'s' if len(gone) != 1 else ''} "
                            "in the other branches.")
        return gone

    def create_variants(self, nid: str, count: int, generate: bool = True, strength_steps: list[float] | None = None,
                        keep_seed: bool = False) -> list[str]:
        """count new branches next to nid: same prompt and clip settings, each its own new seed; then (optionally)
        generate them in one run. strength_steps: per LoRA of the clip's stack, variant k gets its strength + k ×
        step; keep_seed: the clip's seed instead of new ones. Returns the new node ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        node = self.tree.nodes[nid]
        ids = []
        if node.is_video:
            return []
        steps = list(strength_steps or [])
        stack = list((node.overrides or {}).get("loras") or [])
        loras = [e for e in stack if e.get("path")]
        for k in range(1, max(1, int(count)) + 1):
            new = self.tree.add(node.parent, node.prompt, node.seed if keep_seed else self._new_seed())
            new.overrides = copy.deepcopy(node.overrides)
            if any(steps) and loras:
                new.overrides["loras"] = variant_strengths(stack, steps, k)   # (every entry kept)
            self._copy_video_child(nid, new.id)
            ids.append(new.id)
        self._write_tree()
        self.refresh()
        self.graph.reveal(ids[-1])
        sweep = " with stepped LoRA strengths" if any(steps) and loras else ""
        self.status.setText(f"{len(ids)} variant{'s' if len(ids) != 1 else ''} of clip #{nid[1:]} added{sweep}; once "
                            "made, compare them side by side with the Compare button on the card.")
        if generate:
            self._start(ids)
        return ids

    def _context(self, nid: str, pos):
        if self.running or self.tree is None:
            return
        m = QMenu(self)
        acts = {m.addAction("Continue from here"): "continue"}
        node = self.tree.nodes.get(nid)
        vid = m.addAction("Add a real video after this…")
        vid.setEnabled(self.tree.can_add_video(None if nid == START else nid))
        acts[vid] = "video"
        play = gen = delete = keep_here = keep_path = ref_first = ref_last = export_one = None
        if nid != START:
            if node is not None and node.is_video:
                acts[m.addAction("Edit range…")] = "edit"
                split = m.addAction("Split here and insert a clip…")
                split.setEnabled(bool(node.made) and self.state(nid) == "done")
                split.setToolTip("Put a generated clip into the middle of this video: the video ends at a frame you "
                                 "choose, the clip follows, then the rest of the video")
                acts[split] = "split"
                acts[m.addAction("Replace a person…")] = "swap"
                if swap.replaced(node):
                    acts[m.addAction("Back to the original video (undo the replacement)")] = "swap_undo"
            acts[m.addAction("Branch (another continuation)")] = "branch"
            if node is not None and not node.is_video:
                acts[m.addAction("Variant (same prompt, other seed)")] = "variant"
                acts[m.addAction("Variants… (several, generated at once)")] = "variants"
                vc = self.tree.video_child(nid)
                end = m.addAction("End image…" if vc is None else f"End image… (ends at video #{vc.id[1:]})")
                end.setEnabled(vc is None)
                acts[end] = "end_image"
                endv = m.addAction("End image from a video frame…")
                endv.setEnabled(vc is None)
                acts[endv] = "end_video"
                if node.end_image:
                    acts[m.addAction("Clear end image")] = "end_clear"
                if T.changed_inputs(self.tree, nid):
                    acts[m.addAction("Undo changes (back to the made clip)")] = "revert"
                hand = m.addAction("Hand-over frame…")
                hand.setEnabled(bool(node.made) and vc is None)
                acts[hand] = "handover"
                if T.handover_index(self.tree, node) is not None:
                    acts[m.addAction("Hand over at the last frame")] = "handover_clear"
                sp = m.addAction("Speak the lines with text to speech" if T.spoken_lines(node.prompt) else
                                 "Speak the lines with text to speech (write a line in quotes first)")
                sp.setEnabled(not self._tts_busy and bool(T.spoken_lines(node.prompt)))
                acts[sp] = "speak"
                ai = m.addAction("Edit an image with AI…")
                acts[ai] = "edit_ai"
                cmp_ = m.addAction("Compare with its variants…")
                cmp_.setEnabled(len(self.compare_candidates(nid)) > 1)
                acts[cmp_] = "compare"
            if node is not None:
                if node.frozen:
                    acts[m.addAction("Unfreeze (edit it, follow the settings again)")] = "unfreeze"
                else:
                    fr = m.addAction("Freeze (keep it and the clips before it as made)")
                    fr.setEnabled(self.state(nid) == "done")
                    fr.setToolTip(FREEZE_TIP)
                    acts[fr] = "freeze"
                for a_, what in acts.items():
                    if node.frozen and what in self.FROZEN_LOCKED and not (what == "video" and node.is_video):
                        a_.setEnabled(False)
            m.addSeparator()
            play = m.addAction("Play this video" if node is not None and node.is_video else "Play this clip")
            play.setEnabled(bool(self.tree.nodes[nid].made))
            export_one = m.addAction("Export this video…" if node is not None and node.is_video else
                                     "Export this clip…")
            export_one.setEnabled(bool(self.tree.nodes[nid].made) and self.state(nid) == "done")
            if node is not None and node.is_video:
                ref_first = m.addAction("Add first frame as face reference")
                ref_first.setEnabled(bool(node.made))
            ref_last = m.addAction("Add last frame as face reference")
            ref_last.setEnabled(bool(node is not None and node.made))
            gen = m.addAction("Generate the path to here")
            if self.film is not None:
                cur = self.film.scene_of(self.project)
                if cur is not None and cur.take == nid:
                    acts[m.addAction("Clear the scene's ending (the film plays its selected path)")] = "take_clear"
                else:
                    acts[m.addAction("End the scene here (the film plays up to this clip)")] = "take"
            m.addSeparator()
            keep_here = m.addAction("Keep this branch, delete the other branches here")
            keep_here.setEnabled(bool(self.tree.siblings(nid)))
            keep_path = m.addAction("Keep only this path, delete all other branches")
            keep_path.setEnabled(bool(self.tree.others_than_path(nid)))
            delete = m.addAction("Delete (with everything after it)")
        chosen = self._exec_menu(m, pos.toPoint() if hasattr(pos, "toPoint") else pos)
        if chosen is None:
            return
        # run it later: this is still inside the node's mouse handler, and the action may rebuild the graph
        # (deleting that very node)
        if chosen in acts:
            act = acts[chosen]
            QTimer.singleShot(0, lambda: self.node_action(act, nid))
        elif chosen is play:
            QTimer.singleShot(0, lambda: self._play_node(nid))
        elif chosen is export_one:
            QTimer.singleShot(0, lambda: self.export_path(nid))
        elif chosen is not None and chosen in (ref_first, ref_last):
            which = "first" if chosen is ref_first else "last"
            QTimer.singleShot(0, lambda: self.reference_from_node(nid, which))
        elif chosen is gen:
            QTimer.singleShot(0, self.generate_path)
        elif chosen is delete:
            QTimer.singleShot(0, lambda: self.node_action("delete", nid))
        elif chosen is keep_here:
            QTimer.singleShot(0, lambda: self.keep_branch(nid, whole_path=False))
        elif chosen is keep_path:
            QTimer.singleShot(0, lambda: self.keep_branch(nid, whole_path=True))

    def _exec_menu(self, menu: QMenu, pos):
        return menu.exec(pos)

    def find_lora(self, key: str) -> str | None:
        """A LoRA file whose name contains key: among the LoRAs found (H3 and turbo ones), else straight in ComfyUI's
        LoRA folders (a file added after the last scan)."""
        key = key.lower()
        for p, name in self.h3_loras + self.turbo_loras:
            if key in name.lower():
                return p
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        miss = getattr(self, "_lora_misses", {})
        if time.monotonic() - miss.get((key, comfy_dir), -1e9) < 30:
            return None                              # (looked for a moment ago: not there)
        try:
            from ..comfy import comfy_lora_roots
            for root in comfy_lora_roots(comfy_dir):
                for f in sorted(Path(root).rglob("*.safetensors")):
                    if key in f.name.lower():
                        return str(f)
        except Exception:  # noqa: BLE001  (no ComfyUI folder set, a folder offline)
            pass
        self._lora_misses = {**miss, (key, comfy_dir): time.monotonic()}
        return None

    # ---- replace a person in a real video (Wan 2.2 Animate)
    def replace_person(self, nid: str, dialog: bool = True):
        """The swap window for a video node (its converted clip). dialog=False: returned without exec (tests)."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running:
            return None
        try:
            done = T.Checker(self.project, self.tree).status(nid) == "done"
        except Exception:  # noqa: BLE001
            done = False
        if not done or not node.made or not (self.project / node.made["video"]).is_file():
            self.status.setText(f"Video #{nid[1:]} is not converted yet: generate its path (or wait for the "
                                "conversion), then replace the person.")
            return None
        pictures = []
        for c in T.cast_of(self.tree):
            for k, rel in enumerate(c["images"], start=1):
                if (self.project / rel).is_file():
                    pictures.append((f"Face of @{c['name']}" + (f" ({k})" if len(c["images"]) > 1 else ""),
                                     str(self.project / rel)))
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL)
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        cleanup = self.comfy_cleanup.isChecked()
        switch = {"free": False, "loaded": False}

        def before():                               # GUI thread: the clip model makes room
            self._qwen_timer.stop()
            switch["free"] = not switch["loaded"] and self._comfy_model in ("h3", "h3-hyperflow")
            if self.worker_alive and not self.running:
                self._quit_worker(free=False)

        def connect(progress):                      # worker thread
            from ..animate import ComfyAnimator
            an = ComfyAnimator(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            switch["loaded"] = True                 # another model in ComfyUI now (unloaded after the window)
            if switch["free"]:
                an.free()
                switch["free"] = False
            return an

        def describe_face(picture):                 # worker thread: the face details for the head swap
            from ..comfy import ComfyPromptRefiner
            switch["loaded"] = True
            return ComfyPromptRefiner(url, log=lambda _m: None, comfy_dir=comfy_dir, cleanup=cleanup).describe_face(
                picture)

        def connect_h3(progress):                   # worker thread: the clip generator (H3 head swap)
            from ..comfy import ComfyGenerator
            gen = ComfyGenerator(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            if switch["free"]:
                gen.free()
                switch["free"] = False
            switch["loaded"] = True
            return gen

        dlg = SwapDialog(self, self.project, nid, self.project / node.made["video"], self.face_detector(), pictures,
                         connect, self.qs, on_start=before,
                         source=node.source, h3={
                             "lora": self.find_lora("head_swap"),
                             "turbo": self.find_lora(swap.H3_SWAP_TURBO),
                             "settings": T.current_settings(self.project, self.tree.options, self.tree),
                             "connect": connect_h3, "describe": describe_face},
                         groups=[[str(self.project / rel) for rel in c["images"]] for c in T.cast_of(self.tree)],
                         pick=lambda title: pick_image(
                             self, title, self.qs, None, self._picker_places(), self.project_images()))
        self._edit_switch = switch                  # (as the image window: the model is unloaded after it)
        if not dialog:
            return dlg
        try:
            if dlg.exec() == QDialog.Accepted and dlg.result is not None:
                self.use_replacement(nid, *dlg.result)
        finally:
            self.edits_closed()
        return None

    def use_replacement(self, nid: str, file: Path, meta: dict, frames: int) -> bool:
        """The video node plays the clip with the replaced person (converted again; the clips after it change)."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running:
            return False
        before = self._done_clips() - {nid}
        if not Path(file).is_file():
            self.status.setText(f"The made clip is gone ({Path(file).name}).")
            return False
        swap.apply(node, file, frames, {k: v for k, v in meta.items() if k in (
            "reference", "head_size", "size", "gen_size", "seed", "prompt", "model", "made", "method", "details",
            "lora")}, fps=chain.FPS)
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.convert_video(nid)
        if not self._report_remade(before, f"Video #{nid[1:]}: person replaced"):
            self.status.setText(f"Video #{nid[1:]} plays the clip with the replaced person now (its menu goes back "
                                "to the original).")
        return True

    def split_video(self, nid: str, at: float | None = None) -> str | None:
        """Insert a generated clip into a real video: choose the frame (at: the moment in its file, else asked),
        the video ends there, a new clip follows, then the rest of the video. Returns the new clip's id."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running:
            return None
        if not node.made or self.state(nid) != "done":
            self.status.setText(f"Video #{nid[1:]} is not converted yet; wait for it, then split it.")
            return None
        if at is None:
            clip = self.project / node.made["video"]
            frames = int(node.made.get("frames") or chain.probe(clip)["frames"])
            key = node.made.get("key") or {}
            size = (int(key.get("width") or 0) or 672, int(key.get("height") or 0) or 384)
            dlg = SplitDialog(self, clip, frames, size, node.source["start"], node.source["end"],
                              f"Split video #{nid[1:]}")
            try:
                if dlg.exec() != QDialog.Accepted:
                    return None
                at = dlg.at()
            finally:
                dlg.deleteLater()
        before = self._done_clips() - {nid}
        try:
            clip_node, rest = self.tree.split_video(nid, at)
        except ValueError as e:
            self.status.setText(f"Could not split video #{nid[1:]}: {e}")
            return None
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.select(clip_node.id)
        self.convert_video(nid)
        self.convert_video(rest.id)
        if not self._report_remade(before, f"Video #{nid[1:]} split"):
            self.status.setText(f"Video #{nid[1:]} split at {at:.2f} s: write the prompt of the new clip "
                                f"#{clip_node.id[1:]}; it ends at the first frame of video #{rest.id[1:]}.")
        return clip_node.id

    def undo_replacement(self, nid: str) -> bool:
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or self.running or not swap.restore(node):
            return False
        before = self._done_clips() - {nid}
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.convert_video(nid)
        if not self._report_remade(before, f"Video #{nid[1:]}: back to the original"):
            self.status.setText(f"Video #{nid[1:]} plays the original video again.")
        return True

    def _play_node(self, nid: str):
        if self.tree is None or nid not in self.tree.nodes:
            return
        n = self.tree.nodes[nid]
        if n.made and (self.project / n.made["video"]).is_file():
            what = Path(n.source["file"]).name if n.is_video else n.prompt[:60]
            self.preview.open(self.project / n.made["video"],
                              f"{'Video' if n.is_video else 'Clip'} #{nid[1:]} · {what}")

    # ------------------------------------------------------------------ paths
    def join_selected(self, purpose: str, out: Path | None = None, prefix: str = ""):
        path = self.selected_path()
        if not path or self.tree is None:
            return
        nid, project = path[-1].id, self.project
        tree = T.Tree.from_json(self.tree.to_json())
        pre = f"{prefix} " if prefix else ""
        self.status.setText(f"{pre}Joining the path …")
        bridge, lock = self.bridge, self._join_lock
        self._joins += 1
        self._update_buttons()

        def work():
            with lock:
                try:
                    joined = T.join_path(project, tree, nid, out)
                    secs = chain.probe(joined)["frames"] / 24           # not in the GUI thread (network shares)
                    bridge.joined.emit((joined, secs), None, purpose, pre, str(project))
                except Exception as e:  # noqa: BLE001
                    bridge.joined.emit(None, str(e), purpose, pre, str(project))
        threading.Thread(target=work, daemon=True).start()

    def _joined(self, path, error, purpose, pre, project):
        self._joins = max(0, self._joins - 1)
        self._update_buttons()
        if project != str(self.project):            # another project was opened meanwhile
            return
        if error:
            self.status.setText(f"{pre}Could not join the path: {error}")
            return
        path, secs = path
        if purpose == "play":
            self.preview.open(path, f"Path to clip #{Path(path).stem[1:]} · {fmt_time(secs)}")
            self.status.setText(f"{pre}Path joined: {fmt_time(secs)}.")
        else:
            self.status.setText(f"{pre}Exported {Path(path).name} ({fmt_time(secs)}).")

    def export_path(self, single: str | None = None):
        """The export dialog for the selected path, or (single: a clip id) for that clip on its own."""
        if single is not None:
            node = self.tree.nodes.get(single) if self.tree else None
            if node is None or not node.made or self.state(single) != "done":
                self.status.setText("Make the clip first, then export it.")
                return
            path = [node]
        else:
            path = self.selected_path()
        if not path:
            return
        default = str(self.project / f"{self.project.name}-{path[-1].id}{'-clip' if single else ''}.mp4")
        base = T.Checker(self.project, self.tree).base()
        strength = self.qs.value("refineStrength", "natural", type=str)
        method = self.qs.value("exportUpscaler", "seedvr2_7b", type=str)
        denoise = self.qs.value("exportDenoise", False, type=bool)

        def ready(size, how, strength_, denoise_=False):   # every generated clip has a kept copy made this way
            def kept(n):
                f = T.refined_file(self.project, n, size)
                if not (n.made and f.is_file()):
                    return False
                if n.is_video:
                    return True
                rec = T.refine_record(f)
                return (rec.get("method") == how and bool(rec.get("denoise")) == bool(denoise_)) if how != "refine" \
                    else ("method" not in rec and rec.get("strength") == strength_)
            return all(kept(n) for n in path)
        swapped = sum(1 for n in path if T.swapped_video(n))
        dlg = ExportDialog(self, default, (base["width"], base["height"]) if base else None, ready, swapped,
                           strength, method, denoise)
        if single is not None:
            dlg.setWindowTitle(f"Export {'video' if path[0].is_video else 'clip'} #{single[1:]}")
        try:
            if dlg.exec() != QDialog.Accepted:
                return
            out, size, fast = dlg.target(), dlg.size.currentData(), dlg.fast.isChecked()
            refine_swapped = bool(swapped) and dlg.swapped.isChecked()
            strength, method = dlg.strength.currentData(), dlg.method.currentData()
            denoise = dlg.denoise.isChecked()
        finally:
            dlg.deleteLater()
        if size != "made":
            self.qs.setValue("refineStrength", strength)
            self.qs.setValue("exportUpscaler", method)
            self.qs.setValue("exportDenoise", denoise)
        self.export(out, size, fast, refine_swapped, strength, method, single, denoise and method != "refine")

    def export(self, out: Path, size: str = "made", fast_attention: bool = False,
               refine_swapped: bool = False, strength: str = "natural", method: str = "seedvr2_7b",
               single: str | None = None, denoise: bool = False) -> None:
        """Export the selected path (single: that clip alone) to out: as made (joined / copied here), or upscaled at
        '1080p' / '2k' (a generator job)."""
        if single is not None:
            node = self.tree.nodes.get(single) if self.tree else None
            if node is None or not node.made:
                return
            if size == "made":
                out = Path(out)
                try:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.project / node.made["video"], out)
                except OSError as e:
                    self.status.setText(f"Could not export clip #{single[1:]}: {e}")
                    return
                self.status.setText(f"Exported {out.name} (clip #{single[1:]}, as made).")
                return
            target = single
        else:
            path = self.selected_path()
            if not path:
                return
            if size == "made":
                self.join_selected("export", out)
                return
            target = path[-1].id
        self._start([target], refine={"size": size, "out": str(out), "fast_attention": bool(fast_attention),
                                      "refine_swapped": bool(refine_swapped), "strength": strength,
                                      "method": method, **({"single": True} if single is not None else {}),
                                      **({"denoise": True} if denoise else {})})

    # ------------------------------------------------------------------ GPU
    def _poll_gpu(self):
        if not shutil.which("nvidia-smi"):
            self.gpu.setText("GPU: nvidia-smi not found")
            return
        if self._gpu_proc is not None:
            return
        p = QProcess(self)
        self._gpu_proc = p

        def done(*_):
            if self._gpu_proc is not p:          # window closing: the query was abandoned
                return
            try:
                out = bytes(p.readAllStandardOutput()).decode(errors="replace").strip().splitlines()
            except RuntimeError:                 # Qt is destroying the window (and with it the query)
                return
            self._gpu_proc = None
            p.deleteLater()
            try:
                used, total = (float(x) for x in out[0].split(","))
            except (IndexError, ValueError):
                self.gpu.setText("GPU: ?")
                return
            free = (total - used) / 1024
            self.gpu.setText(f"GPU {free:.1f} GB free of {total / 1024:.0f}")
            self.gpu.setProperty("severity", "warning" if free < 4 else "")
            repolish(self.gpu)

        p.finished.connect(done)
        p.start("nvidia-smi", ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])

    # ------------------------------------------------------------------ running
    # The generator runs as a worker process: it loads the model once and then takes jobs (JSON lines on stdin).
    # With "keep loaded" it waits for the next job and quits by itself after the idle time; without, it is told
    # to quit after each job. Stop ends the worker (a clip cannot be interrupted cleanly).

    @property
    def running(self) -> bool:
        """A generation job is in progress (editing is locked)."""
        return self._job is not None

    @property
    def worker_alive(self) -> bool:
        return self.proc is not None

    def generate_path(self):
        path = self.selected_path()
        if not path:
            self.status.setText("Select the last clip of the path you want to make.")
            return
        self._start([path[-1].id])

    def generate_all(self):
        if self.tree is not None:
            self._start([n.id for n in self.tree.leaves()])

    def _worker_key(self) -> tuple:
        """What a running worker is bound to (which ComfyUI, its folder, cleanup); a change needs a new worker."""
        return ((self.comfy_url.text().strip() or DEFAULT_COMFY_URL), self.comfy_dir.text().strip(),
                self.comfy_cleanup.isChecked())

    def _start(self, targets: list[str], refine: dict | None = None):
        if self.running or self.tree is None:
            return
        self._style_changed()                      # a style edit not finished with Enter / leaving the field
        if self._joins:
            self.status.setText("Wait until the path is joined.")
            return
        if self._copying:
            self.status.setText("Wait until the project is copied (Save as).")
            return
        if self._converts:
            self.status.setText("Wait until the video is converted.")
            return
        if self._tts_busy:
            self.status.setText("Wait until text to speech is done.")
            return
        problems = []
        turbo = self.tree.options.get("turbo") or {}
        if turbo.get("on"):
            if not turbo.get("lora"):
                problems.append("choose a turbo LoRA (Clips → Turbo LoRA) or switch Speed to Normal")
        nodes = {n.id: n for t in targets for n in self.tree.path_to(t)}
        empty = [n for n in nodes.values() if not n.is_video and not n.prompt.strip()]
        if empty:
            problems.append(f"write the prompt of clip #{empty[0].id[1:]}")
        try:
            checker = T.Checker(self.project, self.tree)
            if checker.base() is None:
                problems.append("choose a start image (or begin with a real video)")
            else:
                todo = T.plan(self.tree, checker, targets)
                if refine and todo:
                    problems.append(f"generate the path first ({len(todo)} clip{'s are' if len(todo) != 1 else ' is'} "
                                    "new or changed)")
                gone = [n for n in todo if self.tree.nodes[n].is_video
                        and not Path(self.tree.nodes[n].source["file"]).is_file()]
                if gone:
                    problems.append(f"the file of video #{gone[0][1:]} is missing (Edit range… to choose it)")
                problem = T.guide_problem(self.tree, checker, todo)
                if problem:
                    problems.append(problem)
                no_end = T.missing_end_images(self.project, self.tree, todo)
                if no_end:
                    problems.append(f"the end image of clip #{no_end[0][1:]} is missing (choose it again)")
                sounds = T.audio_problems(self.project, self.tree, todo)
                if sounds:
                    problems.append(sounds[0] + " (Sound reference)")
                cont = T.source_sound_problems(self.tree, todo, self.project)
                if cont:
                    problems.append(cont[0] + " (the clip's “Sound” setting)")
                faces = T.cast_problems(self.project, self.tree, todo)
                if faces:
                    problems.append(faces[0] + " (Cast)")
            if checker.base() is not None:
                for lo in T.loras_needed(self.tree, checker, list(nodes)):
                    if not Path(lo).is_file():
                        problems.append(f"the LoRA file does not exist ({Path(lo).name})")
                        break
        except Exception as e:  # noqa: BLE001
            problems.append(f"could not check the settings ({e})")
        if problems:
            self.status.setText("To start: " + ", ".join(problems) + ".")
            if empty and len(targets) > 1:
                self.select(empty[0].id)
            return
        if self.comfy_up is False and not self.worker_alive and self._autostart(targets, refine):
            return
        self._waiting_comfy = None
        self._save_timer.stop()
        if not self._write_tree():
            return
        self.save_settings()
        self._job = {"targets": targets, "sent": False, "refine": refine}
        self._run_started, self._clip_started, self._clip_note = time.monotonic(), time.monotonic(), ""
        self._exported = None
        self._error, self._stopping, self._phase = "", False, "starting"
        self._targets, self._plan = targets, []
        self._made_now: set[str] = set()           # the clips really made (an export reuses kept refined copies)
        self.live, self.progress_of = {}, {}
        self.log.clear()
        self.preview.stop()
        for c in self.cards:
            c.set_editable(False)
        self._update_buttons()
        self.progress.setRange(0, 0)
        if (self.worker_alive and self._worker_idle and not self._worker_leaving
                and self._worker_bound == self._worker_key()):
            self._send_job()
        elif self.worker_alive and self._worker_leaving:  # it just said bye (idle timeout): start a new one
            self.status.setText("Starting the generator again …")
            self._replace_worker = True
            self._quit_worker()
        elif self.worker_alive:                    # bound to another ComfyUI address or folder: replace it
            self.status.setText("ComfyUI settings changed: connecting again …")
            self._replace_worker = True
            self._quit_worker()
        else:
            self._start_worker()

    def _program(self) -> tuple[str, list[str]]:
        if self.command:
            return self.command[0], list(self.command[1:])
        return sys.executable, ["-m", "video_chain"]

    def _start_worker(self):
        program, pre = self._program()
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read_out)
        self.proc.readyReadStandardError.connect(self._read_err)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._proc_error)
        self._buf, self._ebuf = b"", ""
        self._worker_idle, self._worker_bound, self._replace_worker = False, self._worker_key(), False
        self._worker_leaving = False
        minutes = self.idle_minutes.value() if self.keep_loaded.isChecked() else 0
        url, comfy_dir, cleanup = self._worker_bound
        args = pre + ["--serve", "--events", "--idle-minutes", f"{minutes:g}", "--comfy-url", url]
        if comfy_dir:
            args += ["--comfy-dir", os.path.expanduser(comfy_dir)]
        if not cleanup:
            args.append("--keep-comfy-copies")
        self.log.appendPlainText("$ " + " ".join([program, *args]))
        self.status.setText("Starting …")
        self.proc.start(program, args)
        self._update_model_label()

    def _send_job(self):
        if self._job is None or self.proc is None:
            return
        self._job["sent"] = True
        self._worker_idle = False
        self.stop_llama("its RAM and VRAM are needed for the clips")
        cmd = {"cmd": "run", "tree": str(self.project), "targets": self._job["targets"],
               "free_first": self._comfy_model in ("qwen", "llm")}          # an image edit model is loaded: unload it
        if self._comfy_model in ("h3", "h3-hyperflow"):             # earlier workers: was HyperFlow last?
            cmd["last_hyperflow"] = self._comfy_model == "h3-hyperflow"
        if self._job.get("refine"):                                # export refined: re-render the path's clips
            cmd.update({"cmd": "refine", "target": self._job["targets"][0], **self._job["refine"]})
        self._qwen_timer.stop()
        hyper = bool((self.tree.options.get("hyperflow") or {}).get("on")) and not self._job.get("refine") \
            if self.tree is not None else False
        self._comfy_model = "h3-hyperflow" if hyper else "h3"
        self.proc.write((json.dumps(cmd) + "\n").encode())
        self._update_model_label()

    def _quit_worker(self, free: bool = True):
        """Ask an idle worker to quit (free: and unload the model); killed if it does not go."""
        if self.proc is None:
            return
        self._quit_frees = free
        self._worker_leaving = True                       # no new job for it (a new worker starts instead)
        if self.proc.state() == QProcess.Running:
            self.proc.write(b'{"cmd": "quit", "free": true}\n' if free else b'{"cmd": "quit"}\n')
        self._kill_timer.start()

    def unload(self):
        if self.worker_alive and not self.running:
            self.status.setText("Unloading the model …")
            self._quit_worker()
        elif self._comfy_model in ("qwen", "llm") and not self.running:
            self.status.setText("Unloading the image model …")
            self._free_comfy()

    def _free_comfy(self):
        """Ask ComfyUI to unload its models (in the background)."""
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        self._comfy_model = None
        self._qwen_timer.stop()

        def work():
            try:
                req = urllib.request.Request(url + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                             headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=10).read()
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=work, daemon=True).start()
        self._update_model_label()

    def _qwen_idle(self):
        if self._comfy_model in ("qwen", "llm") and not self.running:
            self._free_comfy()
            self.status.setText("The image model was unloaded after the idle time.")

    def stop(self):
        if self._waiting_comfy is not None and not self.running:   # generating once ComfyUI runs: not any more
            self._waiting_comfy = None
            self._film_queue = None
            self.status.setText("Cancelled: nothing is generated when ComfyUI is up (ComfyUI itself keeps "
                                "starting).")
            self._update_buttons()
            return
        if not self.running or self.proc is None:
            return
        self._stopping = True
        self.status.setText("Stopping …")
        self.proc.terminate()
        self._kill_timer.start()

    def _kill(self):
        if self.proc is not None and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def _read_out(self):
        p = self.proc
        if p is None:
            return
        self._buf += bytes(p.readAllStandardOutput())     # bytes: a character split between reads stays whole
        *lines, self._buf = self._buf.split(b"\n")
        for line in lines:
            self._safe_line(line.decode(errors="replace"))

    def _safe_line(self, line: str):
        """One line of the worker's output; a broken event must not lose the lines after it."""
        try:
            self._line(line)
        except Exception as e:  # noqa: BLE001
            self.log.appendPlainText(f"(could not handle the worker's message: {type(e).__name__}: {e})")

    def _read_err(self):
        p = self.proc
        if p is None:
            return
        self._ebuf += bytes(p.readAllStandardError()).decode(errors="replace")
        *lines, self._ebuf = self._ebuf.split("\n")
        for line in lines:
            line = line.split("\r")[-1].rstrip()
            if line:
                self.log.appendPlainText(line)

    def _line(self, line: str):
        line = line.split("\r")[-1].rstrip()
        at = line.find("@@ {")                    # an event can follow text printed without a newline
        if at >= 0:
            try:
                ev = json.loads(line[at + 3:])
            except ValueError:
                ev = None
            if ev is not None:
                if line[:at].strip():
                    self.log.appendPlainText(line[:at].rstrip())
                self._event(ev)
                return
        if line:
            self.log.appendPlainText(line)

    def _overall(self, k: int, frac: float):
        if self._plan:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * ((k - 1) + frac) / len(self._plan)))

    def _reload_made(self):
        """Take what the generator recorded (made) from tree.json; nothing else changes during a run."""
        try:
            fresh = T.load(self.project)
        except (OSError, ValueError):
            return
        for nid, n in fresh.nodes.items():
            if nid in self.tree.nodes:
                self.tree.nodes[nid].made = n.made
                self.tree.nodes[nid].end_at = n.end_at   # a clip made again hands over at its last frame again
        self._tree_stamp = self._stamp()            # the generator wrote it; this window is up to date

    def _event(self, ev: dict):
        e = ev.get("e")
        if e == "loading":
            self._worker_idle = False
            self.status.setText("Connecting to ComfyUI …")
            self.progress.setRange(0, 0)
            self._model_state = "loading"
            self._update_model_label()
        elif e == "loaded":
            self.status.setText("Connected to ComfyUI.")
            self._model_state = "loaded"
        elif e == "idle":
            self._worker_idle = True
            self._idle_since = time.monotonic()
            self._idle_timeout = ev.get("timeout")
            if self._job is not None and not self._job["sent"]:
                self._send_job()
            elif self._job is not None:
                self._job_finished()
            self._update_model_label()
        elif e == "bye":
            self._worker_idle = False
            self._worker_leaving = True
            if self._job is not None and self._job.get("sent") and self._phase != "done":
                self._replace_worker = True               # our job crossed its goodbye: a new worker takes it
            if self._comfy_model in ("h3", "h3-hyperflow") and (ev.get("reason") == "idle" or self._quit_frees):
                self._comfy_model = None                       # it unloaded the clip model
            if ev.get("reason") == "idle" and not self.running:
                self.status.setText("The model was unloaded after the idle time.")
        elif e == "start":
            self.status.setText(f"{ev.get('nodes', 0)} clips to make at {ev['width']}×{ev['height']}, "
                                f"{ev['frames']} frames each.")
        elif e == "plan":
            self._plan = list(ev.get("make", []))
        elif e == "node":
            nid, st = ev["id"], ev["state"]
            if st == "generating":
                self._made_now.add(nid)
                self.live[nid] = "generating"
                self.progress_of[nid] = 0.0
                self._clip_started, self._clip_note = time.monotonic(), ""
                k = self._plan.index(nid) + 1 if nid in self._plan else 1
                self._overall(k, 0)
                node = self.tree.nodes.get(nid) if self.tree else None
                fast = node is not None and self.veda_used(node) and not (self._job or {}).get("refine")
                self.status.setText(f"Clip #{nid[1:]} ({k} of {len(self._plan)}) …"
                                    + (" with ⚡ Veda fast attention (a large clip)" if fast else ""))
            else:
                self.live.pop(nid, None)
                self.progress_of.pop(nid, None)
                if nid in self._plan:
                    self._overall(self._plan.index(nid) + 1, 1)
                self._reload_made()
            self.refresh()
            if st == "generating":
                self.graph.reveal(nid)
        elif e == "step":
            nid = ev.get("id")
            if nid and ev.get("n"):
                frac = ev["i"] / ev["n"]
                self.progress_of[nid] = frac
                k = self._plan.index(nid) + 1 if nid in self._plan else 1
                self._overall(k, frac)
                upscaling = bool((self._job or {}).get("refine"))
                now = time.monotonic()
                left = time_left(now - getattr(self, "_clip_started", now), frac)
                n_plan = max(1, len(self._plan))
                left_all = time_left(now - getattr(self, "_run_started", now), ((k - 1) + frac) / n_plan)
                what = (getattr(self, "_clip_note", "") or "upscaling") if upscaling else \
                    f"step {ev['i']} / {ev['n']}"
                self.status.setText(f"Clip #{nid[1:]} ({k} of {len(self._plan)}): {what} · {round(100 * frac)} %"
                                    + (f" · {fmt_left(left)} left" if left is not None else "")
                                    + (f" (all clips: {fmt_left(left_all)})" if left_all is not None and n_plan > 1
                                       and k < n_plan else ""))
                item = self.graph.items_by_id.get(nid)
                if item is not None:
                    item.info.progress = frac
                    item.update()
                card = next((c for c in self.cards if c.nid == nid), None)
                if card is not None:
                    card.set_progress(ev["i"], ev["n"], f"{round(100 * frac)} %" + (
                        f" · {fmt_left(left)} left" if left is not None else "") if upscaling else None)
        elif e == "note":                              # what the generator is doing now (e.g. a SeedVR2 piece)
            self._clip_note = str(ev.get("text") or "")
        elif e == "done":
            self._phase = "done"
        elif e == "exported":
            self._exported = ev.get("file")
        elif e == "joining":
            self.status.setText("Joining the refined clips …")
        elif e == "error":
            self._error = ev.get("msg", "error")

    def _job_finished(self, code: int = 0):
        """End of a job: the worker reported idle again (code 0), or the worker ended (its exit code)."""
        ok = self._phase == "done" and code == 0 and not self._error
        refine = (self._job or {}).get("refine")         # an export at a refined size
        for nid in [n for n, s in self.live.items() if s == "generating"]:
            self.live[nid] = "stopped" if self._stopping else "failed"
        if ok:
            self.live.clear()
        self.progress_of.clear()
        self._reload_made()
        self._job = None
        for c in self.cards:
            c.set_editable(True)
        self.progress.setRange(0, 1000)
        self.refresh()
        if ok and refine:
            self.progress.setValue(1000)
            out = Path(getattr(self, "_exported", None) or refine["out"])
            n = len(self._made_now)                # the export lists every clip, but reuses kept copies
            how = {"seedvr2_7b": "SeedVR2 7B", "seedvr2_3b": "SeedVR2 3B"}.get(refine.get("method"), "refined")
            self.status.setText(f"Exported {out.name} ({refine['size']}, {how}"
                                + (f"; {n} clip{'s' if n != 1 else ''} upscaled now" if n else
                                   "; every clip was upscaled already") + ").")
        elif ok:
            self.progress.setValue(1000)
            made = len(self._plan)
            msg = f"Done: {made} clip{'s' if made != 1 else ''} made." if made else "Everything was already made."
            self.status.setText(msg)
            path = self.selected_path()
            if len(self._targets) == 1 and path and path[-1].id == self._targets[0] and self._film_queue is None:
                self.join_selected("play", prefix=msg)
        elif self._stopping:
            self.progress.setValue(0)
            self.status.setText("Stopped. Finished clips are kept.")
        else:
            self.progress.setValue(0)
            self.status.setText(f"Failed: {self._error or f'the generator exited with code {code}'}")
            self.tabs.setCurrentWidget(self.log)
        self._stopping = False
        if self.worker_alive and not self.keep_loaded.isChecked():
            self._quit_worker()
        self._save_timer.start()                   # what was chosen during the run (selection …) is saved now
        if ok and self.film is not None and len(self.film.scenes) > 1 and not self.film.problem \
                and self._film_queue is None:
            QTimer.singleShot(0, self._update_continues)   # the next scene starts from this one's (new) end
        if self._film_queue is not None:
            if ok and not refine:
                QTimer.singleShot(0, self._film_next)
            else:
                self._film_queue = None            # stopped or failed: the film stops there
        self._update_buttons()
        self._update_model_label()

    def _proc_error(self, err):
        if err == QProcess.FailedToStart:
            # Qt sends no 'finished' for a process that never started: clean up here
            self._error = f"could not start {self._program()[0]}"
            p = self.proc
            QTimer.singleShot(0, lambda: self.proc is p and self._finished(-1, None))

    def _finished(self, code: int, _status=None):
        self._kill_timer.stop()
        p, self.proc = self.proc, None
        if p is not None:
            rest = self._buf + bytes(p.readAllStandardOutput())
            for line in rest.split(b"\n"):
                self._safe_line(line.decode(errors="replace"))
            err = bytes(p.readAllStandardError()).decode(errors="replace")
            for line in (self._ebuf + err).split("\n"):
                line = line.split("\r")[-1].rstrip()
                if line:
                    self.log.appendPlainText(line)
            p.deleteLater()
        self._buf, self._ebuf = b"", ""
        self._worker_idle = False
        self._model_state = ""
        if self._job is not None and self._replace_worker and not self._stopping:
            self._job["sent"] = False
            self._start_worker()
        elif self._job is not None:
            refine = self._job.get("refine")                # an export stays an export when it waits for ComfyUI
            film_rest = list(self._film_queue) if self._film_queue is not None else None
            self._job_finished(code if code else 1)
            if code == 4:                                   # COMFY_DOWN
                self.comfy_up = False
                if self._autostart(self._targets, refine):
                    if film_rest is not None:               # Generate film goes on once ComfyUI runs
                        self._film_queue = film_rest
                else:
                    self.status.setText(f"ComfyUI is not running at {self.comfy_url.text().strip()}. "
                                        "Start it (Settings → ComfyUI and preferences → Start ComfyUI) and generate again.")
                self._update_comfy_label()
        self._update_model_label()
        self._update_buttons()

    # ---- ComfyUI
    def _poll_comfy(self):
        if self._comfy_polling:
            return
        self._comfy_polling = True
        url, bridge = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/"), self.bridge

        def check():
            try:
                with urllib.request.urlopen(url + "/system_stats", timeout=1.5) as r:
                    ok = r.status == 200
            except Exception:  # noqa: BLE001
                ok = False
            bridge.comfy.emit(ok)
        threading.Thread(target=check, daemon=True).start()

    def _autostart(self, targets: list[str], refine: dict | None = None) -> bool:
        """ComfyUI is down: start it (if allowed) and generate targets once it answers. False: not started."""
        if not self.comfy_autostart.isChecked():
            return False
        if not (self._comfy_starting and time.monotonic() - self._comfy_starting < 180):
            if not self.start_comfy():
                return False
        self._waiting_comfy = list(targets)
        self._waiting_refine = refine                 # an export waits too, not a plain generate
        self.status.setText("Starting ComfyUI … generating begins as soon as it runs (Stop cancels that).")
        self._update_buttons()
        return True

    def _comfy_state(self, up: bool):
        self._comfy_polling = False
        was, self.comfy_up = self.comfy_up, up
        if up and was is False and self._comfy_starting:
            self._comfy_starting = 0.0
            self.status.setText("ComfyUI is running.")
        if self._waiting_comfy is not None:
            if up and QApplication.activeModalWidget() is not None:
                pass                                        # a dialog is open (it may change the project): later
            elif up:
                targets, self._waiting_comfy = self._waiting_comfy, None
                if self.tree is not None and all(t in self.tree.nodes for t in targets):
                    refine = getattr(self, "_waiting_refine", None)
                    QTimer.singleShot(0, lambda: self._start(targets, refine))
                else:
                    self._film_queue = None
            elif not (self._comfy_starting and time.monotonic() - self._comfy_starting < 180):
                self._waiting_comfy = None
                self._film_queue = None
                self.status.setText("ComfyUI did not start within 3 minutes; see its own log.")
            self._update_buttons()
        self._update_comfy_label()

    def _update_comfy_label(self):
        if self.comfy_up:
            text, sev = "ComfyUI: running", ""
        elif self._comfy_starting and time.monotonic() - self._comfy_starting < 180:
            text, sev = "ComfyUI: starting …", "warning"
        elif self.comfy_up is False:
            text, sev = "ComfyUI: not running", "warning"
        else:
            text, sev = "ComfyUI: …", ""
        self.comfy_label.setText(text)
        self.comfy_label.setProperty("severity", sev)
        repolish(self.comfy_label)
        self.b_start_comfy.setEnabled(not self.comfy_up)

    def start_comfy(self) -> bool:
        """Start ComfyUI in the background with the user's launcher (it runs on its own, not tied to this app)."""
        cmd = os.path.expanduser(self.comfy_launcher.text().strip() or DEFAULT_COMFY_LAUNCHER)
        if not Path(cmd).is_file():
            self.status.setText(f"ComfyUI launcher not found: {cmd}")
            return False
        ok = QProcess.startDetached(cmd, [])
        ok = ok[0] if isinstance(ok, tuple) else ok
        if not ok:
            self.status.setText(f"Could not start {cmd}")
            return False
        self._comfy_starting = time.monotonic()
        self.status.setText("Starting ComfyUI … (the clips can be generated once it runs)")
        self._update_comfy_label()
        return True

    def _update_model_label(self):
        if not self.worker_alive and self._comfy_model in ("qwen", "llm"):
            left = self._qwen_timer.remainingTime() if self._qwen_timer.isActive() else -1
            text, sev = (("Model: language model (prompt refiner) loaded in ComfyUI" if self._comfy_model == "llm"
                          else "Model: image model loaded in ComfyUI") +
                         (f" · unloads in {max(1, round(left / 60000))} min" if left >= 0 else "")), ""
        elif not self.worker_alive:
            text, sev = "Model: not loaded", ""
        elif self.running and self._model_state == "loading":
            text, sev = "Model: loading …", "warning"
        elif self.running or not self._worker_idle:
            text, sev = "Model: working", ""
        elif self._idle_timeout:
            left = max(0.0, self._idle_timeout - (time.monotonic() - self._idle_since))
            text, sev = f"Model: loaded in ComfyUI · unloads in {max(1, round(left / 60))} min", ""
        else:
            text, sev = "Model: loaded", ""
        self.model_label.setText(text)
        self.model_label.setProperty("severity", sev)
        repolish(self.model_label)
        self.b_unload.setVisible((self.worker_alive and not self.running and self._worker_idle) or
                                 (not self.worker_alive and self._comfy_model in ("qwen", "llm")))

    def changeEvent(self, e):
        """Back in this window (e.g. after training a LoRA): look for new LoRAs, at most every 30 s."""
        super().changeEvent(e)
        if e.type() == e.Type.ActivationChange and self.isActiveWindow() and \
                time.monotonic() - getattr(self, "_scanned_at", 0.0) > 30:
            self.scan_loras()

    def closeEvent(self, e):
        if self.running:
            if QMessageBox.question(self, "Promptwood", "Clips are being generated. Stop and quit?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
        if self.project is not None and not self.running and not self._may_leave_project():
            e.ignore()                                # the user keeps the window to save elsewhere / reopen
            return
        if getattr(self, "_trigger_scan", None) is not None:
            self._trigger_scan.set()                  # the trigger word lookup stops at the next file
        self.stop_llama()                             # never leave the language model server behind
        self.stop_tts()
        if self.proc is not None:                    # never leave the generator (and the model) behind
            p, self.proc = self.proc, None
            p.finished.disconnect()
            if p.state() == QProcess.Running:
                p.write(b'{"cmd": "quit", "free": true}\n')     # ComfyUI: give the memory back too
                p.closeWriteChannel()
                if not p.waitForFinished(3000):
                    p.terminate()
                    if not p.waitForFinished(5000):
                        p.kill()
                        p.waitForFinished(3000)
            self._job = None
        if self._comfy_model in ("qwen", "llm"):              # nor the image edit model
            try:
                url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
                urllib.request.urlopen(urllib.request.Request(
                    url + "/free", data=b'{"unload_models": true, "free_memory": true}',
                    headers={"Content-Type": "application/json"}, method="POST"), timeout=3).read()
            except Exception:  # noqa: BLE001
                pass
            self._comfy_model = None
        self._qwen_timer.stop()
        self._model_timer.stop()
        self._gpu_timer.stop()
        self._comfy_timer.stop()
        if self._gpu_proc is not None:
            gp, self._gpu_proc = self._gpu_proc, None
            gp.finished.disconnect()
            gp.kill()
            gp.waitForFinished(1000)
        self._save_timer.stop()
        self._save_tree()
        self.preview.stop()
        self.save_settings()
        super().closeEvent(e)
