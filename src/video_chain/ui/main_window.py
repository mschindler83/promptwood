"""Promptwood window: a start image and a tree of clips. Every clip continues from the last frame of the clip
before it; any clip can be continued in several ways (branches). A path from the start to a clip is one video.

The project folder holds tree.json, start.png and the clips (see tree.py). Generation runs in a separate process
(the video-chain launcher with ai-toolkit's Python) that reports '@@ {json}' events; Stop ends only that process.
"""
from __future__ import annotations

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
from PySide6.QtGui import QDesktopServices, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout,
                               QFrame, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import chain
from .. import tree as T
from ..comfy import REF_HINT
from .graph import START, GraphView, NodeInfo
from .image_picker import pick_image
from .theme import icon, repolish, theme
from .widgets import FrameTile, icon_button, label, section_bar

PROJECT_DIR = Path(__file__).resolve().parents[3]
LAUNCHER = PROJECT_DIR / "video-chain"
DEFAULT_AITK = "/path/to/ai/ai-toolkit"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_COMFY_LAUNCHER = str(Path.home() / "bin" / "comfyui")
DEFAULT_COMFY_DIR = str(Path.home() / "ComfyUI")
BACKENDS = (("comfyui", "ComfyUI (fast, keeps the model loaded)"),
            ("ai-toolkit", "ai-toolkit (the trainer's sampler)"))
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
STATE_TEXT = {"done": "DONE", "reused": "DONE", "changed": "CHANGED", "new": "NOT MADE", "generating": "GENERATING",
              "failed": "FAILED", "stopped": "STOPPED"}
STATE_STYLE = {"done": "done", "reused": "done", "generating": "target", "failed": "failed", "stopped": "missing",
               "changed": "missing", "new": ""}


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:04.1f}"


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


class _Bridge(QObject):
    joined = Signal(object, object, str, str, str)  # path or None, error or None, purpose, status prefix, project
    comfy = Signal(bool)                       # ComfyUI reachable
    loras = Signal(list)                       # H3 LoRAs found in ComfyUI's LoRA folders
    copied = Signal(str, str)                  # Save as: target folder, error ('' = done)


# ----------------------------------------------------------------------------- path card

STRENGTH_DEFAULT = -0.05          # spin box minimum = "default"
STEPS_DEFAULT = 3


def lora_path(p: str) -> str:
    """How a chosen LoRA file is stored: absolute, links kept as they are (a link in ComfyUI's LoRA folder keeps
    its own name)."""
    return os.path.abspath(os.path.expanduser(p))


def lora_label(path: str | None) -> str:
    return Path(path).stem if path else "none"


def fill_lora_combo(combo: QComboBox, known: list[tuple[str, str]], current: str | None) -> None:
    """Append the offered LoRAs to a combo box and select current (also when it is stored as the target of a
    listed link, or is not in the list at all)."""
    for path, text in known:
        combo.addItem(text, path)
        combo.setItemData(combo.count() - 1, path, Qt.ToolTipRole)
    if not current or current == T.NO_LORA:
        combo.setCurrentIndex(1 if current == T.NO_LORA and combo.findData(T.NO_LORA) >= 0 else 0)
        return
    i = combo.findData(current)
    if i < 0:
        real = os.path.realpath(current)
        i = next((k for k in range(combo.count()) if isinstance(combo.itemData(k), str)
                  and combo.itemData(k) != T.NO_LORA and os.path.realpath(combo.itemData(k)) == real), -1)
    if i < 0:
        combo.addItem(lora_label(current) + ("" if Path(current).is_file() else " (missing)"), current)
        i = combo.count() - 1
    combo.setCurrentIndex(i)


class LoraStack(QWidget):
    """Further LoRAs stacked after the first one: rows of (LoRA, strength, remove) and '+ LoRA'.
    items() -> [{"path", "strength"}] in order (rows without a chosen LoRA are left out)."""
    changed = Signal()

    def __init__(self, parent=None, add_text: str = "+ LoRA"):
        super().__init__(parent)
        self.known: list[tuple[str, str]] = []
        self.rows: list[tuple[QWidget, QComboBox, QDoubleSpinBox]] = []
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(4)
        self.b_add = QPushButton(add_text)
        self.b_add.setToolTip("Stack another LoRA on top (applied after the ones above, in this order; "
                              "ComfyUI generator only)")
        self.b_add.clicked.connect(lambda: (self._add_row(None, 1.0), self.changed.emit()))
        self.v.addWidget(self.b_add, 0, Qt.AlignLeft)
        self._enabled = True

    def set_items(self, known: list[tuple[str, str]], items: list[dict]):
        self.known = list(known)
        for w, _c, _s in self.rows:
            self.v.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self.rows = []
        for it in items or []:
            self._add_row(it.get("path"), float(it.get("strength", 1.0)))

    def _add_row(self, path: str | None, strength: float):
        w = QWidget(self)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        combo = QComboBox(w)
        combo.addItem("choose a LoRA …", None)
        fill_lora_combo(combo, self.known, path)
        spin = QDoubleSpinBox(w)
        spin.setRange(0.0, 2.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(strength)
        spin.setToolTip("Strength of this LoRA")
        rm = icon_button("x", "Remove this LoRA", w)
        h.addWidget(combo, 1)
        h.addWidget(spin)
        h.addWidget(rm)
        self.v.insertWidget(len(self.rows), w)
        row = (w, combo, spin)
        self.rows.append(row)
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        spin.valueChanged.connect(lambda *_: self.changed.emit())
        rm.clicked.connect(lambda: self._remove(row))
        for x in (combo, spin, rm):
            x.setEnabled(self._enabled)

    def _remove(self, row):
        if row in self.rows:
            self.rows.remove(row)
            self.v.removeWidget(row[0])
            row[0].setParent(None)
            row[0].deleteLater()
            self.changed.emit()

    def items(self) -> list[dict]:
        out = []
        for _w, combo, spin in self.rows:
            path = combo.currentData()
            if path and path != T.NO_LORA:
                out.append({"path": lora_path(path), "strength": round(spin.value(), 2)})
        return out

    def setEnabled(self, on: bool):          # noqa: N802  (Qt naming)
        self._enabled = on
        self.b_add.setEnabled(on)
        for _w, combo, spin in self.rows:
            combo.setEnabled(on)
            spin.setEnabled(on)


class PromptCard(QFrame):
    """One clip of the selected path: prompt, seed, own settings (LoRA, strength, length, steps), status."""
    edited = Signal(str)
    seedChanged = Signal(str, int)
    overridesChanged = Signal(str, dict)
    chooseLora = Signal(str)
    action = Signal(str, str)                  # action, node id
    play = Signal(str)

    def __init__(self, node: T.Node, parent=None):
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
                             ("branch", "split", "Branch: another continuation from the same point (Ctrl+B)"),
                             ("delete", "trash", "Delete this clip and everything after it")):
            b = icon_button(ic, tip)
            b.clicked.connect(lambda _=False, a=act: self.action.emit(a, self.nid))
            head.addWidget(b)
            self.buttons.append(b)
        v.addLayout(head)
        self.span = label("", level="tertiary")
        v.addWidget(self.span)
        self.thumb = FrameTile(312, 176)
        self.thumb.setToolTip("Last frame of this clip; click to play the clip")
        self.thumb.clicked.connect(lambda _t: self.play.emit(self.nid))
        v.addWidget(self.thumb)
        self.edit = QPlainTextEdit(node.prompt)
        self.edit.setPlaceholderText("What happens in this clip? It starts from the last frame of the clip before.")
        self.edit.setTabChangesFocus(True)
        self.edit.setMinimumHeight(96)
        v.addWidget(self.edit, 1)
        foot = QHBoxLayout()
        self.words = label("", level="tertiary")
        foot.addWidget(self.words)
        foot.addStretch(1)
        foot.addWidget(label("Seed", level="tertiary"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(node.seed)
        self.seed.setToolTip("Seed of this clip; another seed gives another version")
        foot.addWidget(self.seed)
        v.addLayout(foot)

        # own settings of this clip; "default" = the project's
        self.ov = dict(node.overrides or {})
        self.steps_locked = False                 # turbo mode: the turbo steps apply to every clip
        grid = QFormLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        lrow = QHBoxLayout()
        self.lora = QComboBox()
        self.lora.setToolTip("LoRA of this clip: the project's, none, or its own")
        self.b_lora = icon_button("fopen", "Choose a LoRA file for this clip")
        self.b_lora.clicked.connect(lambda: self.chooseLora.emit(self.nid))
        lrow.addWidget(self.lora, 1)
        lrow.addWidget(self.b_lora)
        grid.addRow("LoRA", lrow)
        srow = QHBoxLayout()
        self.strength = QDoubleSpinBox()
        self.strength.setRange(STRENGTH_DEFAULT, 2.0)
        self.strength.setSingleStep(0.05)
        self.strength.setDecimals(2)
        self.strength.setSpecialValueText("default")
        self.strength.setToolTip("LoRA strength of this clip (lowest value = the project's)")
        self.length = QComboBox()
        self.length.setToolTip("Length of this clip")
        self.steps = QSpinBox()
        self.steps.setRange(STEPS_DEFAULT, 80)
        self.steps.setSpecialValueText("default")
        self.steps.setToolTip("Sampling steps of this clip (lowest value = the project's)")
        srow.addWidget(self.strength)
        srow.addWidget(self.length, 1)
        srow.addWidget(self.steps)
        grid.addRow("Strength", srow)
        self.extras = LoraStack(self)
        self.extras.setToolTip("LoRAs stacked on top of the one above (with 'project default' also on top of the "
                               "project's own extra LoRAs)")
        grid.addRow("More", self.extras)
        v.addLayout(grid)
        self.extras.changed.connect(self._ov_changed)
        self.lora.currentIndexChanged.connect(self._ov_changed)
        self.strength.valueChanged.connect(self._ov_changed)
        self.length.currentIndexChanged.connect(self._ov_changed)
        self.steps.valueChanged.connect(self._ov_changed)

        self.step = label("", level="tertiary")
        v.addWidget(self.step)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.hide()
        v.addWidget(self.bar)
        self.edit.textChanged.connect(self._text)
        self.seed.valueChanged.connect(lambda s: self.seedChanged.emit(self.nid, s))
        self._words()
        self._restyle()

    def set_choices(self, project_label: str, project_frames: int, project_steps: int,
                    known: list[tuple[str, str]]):
        """Fill the setting widgets (without emitting changes). known: (path, label) of the LoRAs to offer."""
        widgets = (self.lora, self.strength, self.length, self.steps)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.lora.clear()
            self.lora.addItem(f"project default ({project_label})", None)
            self.lora.addItem("no LoRA", T.NO_LORA)
            fill_lora_combo(self.lora, known, self.ov.get("lora"))
            self.extras.blockSignals(True)
            self.extras.set_items(known, self.ov.get("extra_loras") or [])
            self.extras.blockSignals(False)
            self.strength.setValue(float(self.ov.get("strength", STRENGTH_DEFAULT)))
            self.length.clear()
            self.length.addItem(f"default ({project_frames / 24:.1f} s)", None)
            for n in FRAME_CHOICES:
                self.length.addItem(frames_text(n), n)
            select_frames(self.length, self.ov.get("frames"))
            self.steps.setValue(int(self.ov.get("steps", STEPS_DEFAULT)))
            self.steps.setToolTip(f"Sampling steps of this clip (lowest value = the project's: {project_steps})")
            self.steps.setEnabled(not self.edit.isReadOnly() and not self.steps_locked)
            if self.steps_locked:
                self.steps.setToolTip("Turbo mode: every clip uses the turbo steps (Clips → Speed)")
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._enable_strength()

    def _enable_strength(self):
        self.strength.setEnabled(not self.edit.isReadOnly() and self.lora.currentData() != T.NO_LORA)

    def _ov_changed(self, *_):
        ov = {}
        if self.lora.currentData() is not None:
            ov["lora"] = self.lora.currentData()
        if self.strength.value() > STRENGTH_DEFAULT + 1e-6:
            ov["strength"] = round(self.strength.value(), 2)
        if self.length.currentData() is not None:
            ov["frames"] = self.length.currentData()
        if self.steps.value() > STEPS_DEFAULT:
            ov["steps"] = self.steps.value()
        extras = self.extras.items()
        if extras:
            ov["extra_loras"] = extras
        self.ov = ov
        self._enable_strength()
        self.overridesChanged.emit(self.nid, ov)

    def text(self) -> str:
        return " ".join(self.edit.toPlainText().split())

    def _text(self):
        self._words()
        self.edited.emit(self.nid)

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

    def set_progress(self, i: int, n: int):
        if n:
            self.bar.setValue(round(100 * i / n))
            self.step.setText(f"step {i} / {n}")

    def set_editable(self, on: bool):
        self.edit.setReadOnly(not on)
        for w in (self.seed, self.lora, self.b_lora, self.length, self.steps, self.extras, *self.buttons):
            w.setEnabled(on)
        self.steps.setEnabled(on and not self.steps_locked)
        self._enable_strength()

    def _restyle(self):
        t = theme()
        border = t.accent.name() if self.state == "generating" else t.lineSubtle.name()
        self.setStyleSheet(f"QFrame#promptCard {{ background: {t.bgRaised.name()}; border: 1px solid {border};"
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
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except ImportError:
            self.placeholder.setText("Video playback is not available (QtMultimedia missing).")
            return False
        self.video = QVideoWidget()
        self.video.setMinimumHeight(200)
        self.stack.addWidget(self.video)
        self.audio = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
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
        self.command = command                  # tests replace the launcher with a fake backend
        self.project: Path | None = None
        self.tree: T.Tree | None = None
        self.status_of: dict[str, str] = {}
        self.live: dict[str, str] = {}          # states during a run (generating / failed / stopped)
        self.progress_of: dict[str, float] = {}
        self.cards: list[PromptCard] = []
        self._pix: dict[tuple, QPixmap] = {}
        self.proc: QProcess | None = None
        self._buf = self._ebuf = ""
        self._stopping = False
        self._phase = ""
        self._error = ""
        self._plan: list[str] = []
        self._targets: list[str] = []
        self._loading = False
        self._tree_stamp: tuple | None = None   # tree.json as this window last read or wrote it
        self._job: dict | None = None           # the generation in progress
        self._worker_idle = False
        self._worker_bound: tuple | None = None
        self._worker_loras: list[str] = []
        self._replace_worker = self._restart = False
        self._restart_loras: list[str] = []
        self._idle_since, self._idle_timeout = 0.0, None
        self._model_state = ""
        self._joins = 0                         # path joins still running
        self._kill_timer = QTimer(self, singleShot=True, interval=8000, timeout=self._kill)
        self.bridge = _Bridge()
        self.bridge.joined.connect(self._joined)
        self.bridge.comfy.connect(self._comfy_state)
        self.bridge.loras.connect(self._loras_found)
        self.bridge.copied.connect(self._copied)
        self.h3_loras: list[tuple[str, str]] = []
        self.turbo_loras: list[tuple[str, str]] = []
        self.comfy_up: bool | None = None
        self._comfy_polling = False
        self._comfy_starting = 0.0
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

        ls = QScrollArea()
        ls.setWidgetResizable(True)
        ls.setFrameShape(QFrame.NoFrame)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        bar, _ = section_bar("Project", left)
        lv.addWidget(bar)
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
        lv.addWidget(pw)

        bar, _ = section_bar("Start image", left)
        lv.addWidget(bar)
        sw = QWidget()
        sv = QVBoxLayout(sw)
        sv.setContentsMargins(10, 8, 10, 8)
        self.start_tile = StartTile(320, 180)
        sv.addWidget(self.start_tile, 0, Qt.AlignHCenter)
        self.size_info = label("", level="tertiary")
        self.size_info.setWordWrap(True)
        sv.addWidget(self.size_info)
        lv.addWidget(sw)

        bar, _ = section_bar("Identity (face)", left)
        lv.addWidget(bar)
        iw = QWidget()
        iv = QVBoxLayout(iw)
        iv.setContentsMargins(10, 6, 10, 8)
        iv.setSpacing(6)
        self.ident_mode = QComboBox()
        self.ident_mode.addItem("Off", "off")
        self.ident_mode.addItem("Reference images (H3 ref2va)", "reference")
        self.ident_mode.setToolTip("With reference images every clip sees the same original face (<Picture 1> ...), "
                                   "not only the last frame of the clip before, so the face does not drift along "
                                   "the chain. Uses H3's reference-to-video model (ComfyUI generator only).")
        iv.addWidget(self.ident_mode)
        self.refs = QListWidget()
        self.refs.setViewMode(QListWidget.IconMode)
        self.refs.setIconSize(QSize(72, 72))
        self.refs.setFixedHeight(100)
        self.refs.setMovement(QListWidget.Static)
        self.refs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.refs.setToolTip("Reference images: best a sharp close-up of the face; <Picture 1> is the first")
        iv.addWidget(self.refs)
        rrow = QHBoxLayout()
        self.b_ref_add = QPushButton("Add…")
        self.b_ref_add.clicked.connect(self._add_reference)
        self.b_ref_start = QPushButton("Use start image")
        self.b_ref_start.clicked.connect(lambda: self.add_reference(str(self.project / T.START_FILE))
                                         if self.project else None)
        self.b_ref_del = QPushButton("Remove")
        self.b_ref_del.clicked.connect(self._remove_reference)
        for b in (self.b_ref_add, self.b_ref_start, self.b_ref_del):
            rrow.addWidget(b)
        iv.addLayout(rrow)
        irow = QFormLayout()
        irow.setContentsMargins(0, 0, 0, 0)
        self.ref_size = QComboBox()
        self.ref_size.addItem("match (fast)", "match")
        self.ref_size.addItem("max 2048 px (best likeness, slower)", "max")
        irow.addRow("Size", self.ref_size)
        iv.addLayout(irow)
        self.ref_hint = QCheckBox("Add a <Picture 1> hint to prompts")
        self.ref_hint.setChecked(True)
        self.ref_hint.setToolTip(f"Prompts that do not mention <Picture ...> get: \"{REF_HINT}\"")
        iv.addWidget(self.ref_hint)
        self.ident_info = label("", level="tertiary")
        self.ident_info.setWordWrap(True)
        iv.addWidget(self.ident_info)
        lv.addWidget(iw)
        self.ident_mode.currentIndexChanged.connect(self._identity_changed)
        self.ref_size.currentIndexChanged.connect(self._identity_changed)
        self.ref_hint.toggled.connect(self._identity_changed)

        bar, _ = section_bar("LoRA", left)
        lv.addWidget(bar)
        fw = QWidget()
        f = QFormLayout(fw)
        f.setContentsMargins(10, 6, 10, 8)
        lrow = QHBoxLayout()
        self.lora = QComboBox()
        self.lora.setToolTip("The LoRA every clip uses unless it has its own; the list shows every H3 LoRA in "
                             "ComfyUI's LoRA folders")
        self.lora.addItem("none (base model)", None)
        self.b_lora = icon_button("fopen", "Choose a LoRA file (.safetensors)")
        self.b_lora.clicked.connect(self._choose_lora)
        lrow.addWidget(self.lora, 1)
        lrow.addWidget(self.b_lora)
        f.addRow("File", lrow)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 2.0)
        self.strength.setSingleStep(0.05)
        self.strength.setValue(1.0)
        f.addRow("Strength", self.strength)
        self.extra_loras = LoraStack(fw)
        self.extra_loras.setToolTip("LoRAs stacked on top of the project LoRA for every clip on the project "
                                    "default (ComfyUI generator only)")
        f.addRow("More", self.extra_loras)
        self.extra_loras.changed.connect(self._options_changed)
        lv.addWidget(fw)

        bar, _ = section_bar("Clips", left)
        lv.addWidget(bar)
        cw = QWidget()
        c = QFormLayout(cw)
        c.setContentsMargins(10, 6, 10, 8)
        self.frames = QComboBox()
        for n in FRAME_CHOICES:
            self.frames.addItem(frames_text(n), n)
        self.frames.setCurrentIndex(FRAME_CHOICES.index(73))
        self.frames.setToolTip("Length of every clip (a clip can have its own). Longer clips mean fewer hand-overs "
                               "between clips, so the face drifts less; they take longer and need more memory "
                               "(with ai-toolkit beyond 124 frames possibly too much for 24 GB)")
        c.addRow("Length", self.frames)
        self.resolution = QComboBox()
        for r, txt in ((384, "384 (fast draft)"), (512, "512 (like the training)"), (640, "640"), (768, "768")):
            self.resolution.addItem(txt, r)
        self.resolution.setCurrentIndex(1)
        self.resolution.setToolTip("About this many pixels squared, in the start image's aspect ratio")
        c.addRow("Size", self.resolution)
        self.steps = QSpinBox()
        self.steps.setRange(4, 80)
        self.steps.setValue(28)
        self.steps.setToolTip("Sampling steps per clip. The trainer's samples use 20, ai-toolkit's H3 default is 28")
        c.addRow("Steps", self.steps)
        self.speed = QComboBox()
        self.speed.addItem("Normal", "normal")
        self.speed.addItem("Turbo (turbo LoRA, 4-8 steps)", "turbo")
        self.speed.setToolTip("Turbo: a step-distilled LoRA with its own sampler renders every clip in 4-8 steps "
                              "instead of ~20 (needs ComfyUI and the ComfyUI-MiniMax-H3-Turbo nodes)")
        c.addRow("Speed", self.speed)
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
        self.turbo_strength.setRange(0.5, 1.5)
        self.turbo_strength.setSingleStep(0.05)
        self.turbo_strength.setValue(1.0)
        self.turbo_strength.setToolTip("Keep 1.0; smear/ghosting -> a bit higher (1.05-1.2), over-sharp grain -> "
                                       "a bit lower (0.8-0.95)")
        trow.addWidget(self.turbo_steps)
        trow.addWidget(label("strength", level="tertiary"))
        trow.addWidget(self.turbo_strength)
        trow.addStretch(1)
        c.addRow("", trow)
        self.backend = QComboBox()
        for key, text in BACKENDS:
            self.backend.addItem(text, key)
        self.backend.setToolTip("Who renders the clips. ComfyUI loads the model quickly and keeps it; ai-toolkit "
                                "uses the trainer's sampler (like the training samples). Clips made by the other "
                                "one count as changed.")
        c.addRow("Generator", self.backend)
        self.audio = QCheckBox("Generate sound")
        self.audio.setChecked(True)
        c.addRow("", self.audio)
        self.keep_rep = QCheckBox("Keep repeated frames in paths")
        self.keep_rep.setToolTip("Each clip starts with the last frame of the clip before; by default that repeated "
                                 "frame is left out when a path is joined")
        c.addRow("", self.keep_rep)
        lv.addWidget(cw)

        bar, _ = section_bar("Advanced", left)
        lv.addWidget(bar)
        aw = QWidget()
        a = QFormLayout(aw)
        a.setContentsMargins(10, 6, 10, 8)
        self.offload = QSpinBox()
        self.offload.setRange(0, 100)
        self.offload.setValue(90)
        self.offload.setSuffix(" % of layers in RAM")
        a.addRow("Offloading", self.offload)
        self.min_free = QDoubleSpinBox()
        self.min_free.setRange(0, 24)
        self.min_free.setValue(18)
        self.min_free.setSuffix(" GB")
        self.min_free.setToolTip("Refuse to start with less free GPU memory (protects a running training)")
        a.addRow("Min. free GPU", self.min_free)
        self.aitk = QLineEdit(DEFAULT_AITK)
        a.addRow("ai-toolkit", self.aitk)
        self.comfy_url = QLineEdit(DEFAULT_COMFY_URL)
        self.comfy_url.setToolTip("Address of the ComfyUI that renders the clips")
        a.addRow("ComfyUI", self.comfy_url)
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
        a.addRow("ComfyUI folder", self.comfy_dir)
        self.comfy_cleanup = QCheckBox("Delete ComfyUI's copies of the clips")
        self.comfy_cleanup.setChecked(True)
        self.comfy_cleanup.setToolTip("ComfyUI keeps every clip and start image in output/video-chain and "
                                      "input/video-chain; the project has its own copy, so they are removed after "
                                      "the download (only there, never anything else)")
        a.addRow("", self.comfy_cleanup)
        krow = QHBoxLayout()
        self.keep_loaded = QCheckBox("Keep the model loaded for")
        self.keep_loaded.setChecked(True)
        self.keep_loaded.setToolTip("After generating, the model stays in memory, so the next clip starts at once "
                                    "instead of loading for minutes. It is unloaded after this idle time, or with "
                                    "Unload in the status bar.")
        self.idle_minutes = QSpinBox()
        self.idle_minutes.setRange(1, 240)
        self.idle_minutes.setValue(15)
        self.idle_minutes.setSuffix(" min")
        krow.addWidget(self.keep_loaded)
        krow.addWidget(self.idle_minutes)
        krow.addStretch(1)
        a.addRow("", krow)
        self.keep_loaded.toggled.connect(self.idle_minutes.setEnabled)
        lv.addWidget(aw)
        lv.addStretch(1)
        ls.setWidget(left)
        ls.setMinimumWidth(350)
        split.addWidget(ls)

        # middle: graph over the path
        mid = QSplitter(Qt.Vertical)
        mid.setChildrenCollapsible(False)
        gw = QWidget()
        gv = QVBoxLayout(gw)
        gv.setContentsMargins(0, 0, 0, 0)
        gv.setSpacing(0)
        bar, bh = section_bar("Clips", gw)
        self.tree_info = label("", level="tertiary")
        bh.addWidget(self.tree_info)
        bh.addSpacing(8)
        self.b_continue = icon_button("plus", "Continue: a new clip after the selected one (Ctrl+N)")
        self.b_branch = icon_button("split", "Branch: another continuation from the same point (Ctrl+B)")
        self.b_variant = icon_button("clone", "Variant: same prompt, other seed (Ctrl+D)")
        self.b_delete = icon_button("trash", "Delete the selected clip and everything after it (Del)")
        self.b_fit = icon_button("frame", "Fit the view (F)")
        for btn, act in ((self.b_continue, "continue"), (self.b_branch, "branch"), (self.b_variant, "variant"),
                         (self.b_delete, "delete")):
            btn.clicked.connect(lambda _=False, a=act: self.node_action(a, self.selected_id()))
            bh.addWidget(btn)
        bh.addSpacing(6)
        self.b_fit.clicked.connect(lambda: self.graph.fit())
        bh.addWidget(self.b_fit)
        gv.addWidget(bar)
        self.graph = GraphView()
        self.graph.selected.connect(self.select)
        self.graph.activated.connect(self._play_node)
        self.graph.contextMenu.connect(self._context)
        gv.addWidget(self.graph, 1)
        mid.addWidget(gw)

        pw2 = QWidget()
        pv = QVBoxLayout(pw2)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        bar, bh = section_bar("Selected path", pw2)
        self.path_info = label("", level="tertiary")
        bh.addWidget(self.path_info)
        pv.addWidget(bar)
        self.path_area = QScrollArea()
        self.path_area.setWidgetResizable(True)
        self.path_area.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        self.path_box = QHBoxLayout(holder)
        self.path_box.setContentsMargins(10, 10, 10, 10)
        self.path_box.setSpacing(10)
        self.path_hint = label("", level="tertiary")
        self.path_box.addWidget(self.path_hint)
        self.path_box.addStretch(1)
        self.path_area.setWidget(holder)
        pv.addWidget(self.path_area, 1)
        mid.addWidget(pw2)
        mid.setSizes([460, 420])
        split.addWidget(mid)

        # right: preview + log
        self.tabs = QTabWidget()
        rp = QWidget()
        rv = QVBoxLayout(rp)
        rv.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview()
        rv.addWidget(self.preview, 1)
        prow = QHBoxLayout()
        prow.setContentsMargins(8, 0, 8, 8)
        self.b_play_path = QPushButton("Play path")
        self.b_play_path.setToolTip("Join the clips of the selected path and play them")
        self.b_play_path.clicked.connect(lambda: self.join_selected("play"))
        self.b_export = QPushButton("Export path…")
        self.b_export.setToolTip("Save the selected path as one video")
        self.b_export.clicked.connect(self.export_path)
        prow.addWidget(self.b_play_path)
        prow.addWidget(self.b_export)
        prow.addStretch(1)
        rv.addLayout(prow)
        self.tabs.addTab(rp, "Preview")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("type", "mono")
        self.log.setMaximumBlockCount(5000)
        self.tabs.addTab(self.log, "Log")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 760, 440])

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
        for w in (self.b_gen, self.b_gen_all, self.b_stop, self.progress):
            sb.addWidget(w)
        sb.addWidget(self.status, 1)
        self.model_label = label("Model: not loaded", role="statusText")
        self.b_unload = QPushButton("Unload")
        self.b_unload.setToolTip("Free the GPU and RAM now (the next Generate loads the model again)")
        self.b_unload.clicked.connect(self.unload)
        self.b_unload.hide()
        self.comfy_label = label("", role="statusText")
        sb.addPermanentWidget(self.comfy_label)
        sb.addPermanentWidget(self.model_label)
        sb.addPermanentWidget(self.b_unload)
        sb.addPermanentWidget(self.gpu)
        self._model_timer = QTimer(self, interval=20000, timeout=self._update_model_label)
        self._model_timer.start()

        self.start_tile.clicked.connect(lambda _t: self._choose_image())
        self.start_tile.dropped.connect(self.set_image)
        for sig in (self.strength.valueChanged, self.frames.currentIndexChanged, self.resolution.currentIndexChanged,
                    self.steps.valueChanged, self.audio.toggled, self.keep_rep.toggled,
                    self.backend.currentIndexChanged, self.speed.currentIndexChanged,
                    self.turbo_lora.currentIndexChanged, self.turbo_steps.valueChanged,
                    self.turbo_strength.valueChanged):
            sig.connect(self._options_changed)
        self.lora.currentIndexChanged.connect(self._options_changed)
        for keys, fn in (("Ctrl+N", lambda: self.node_action("continue", self.selected_id())),
                         ("Ctrl+B", lambda: self.node_action("branch", self.selected_id())),
                         ("Ctrl+D", lambda: self.node_action("variant", self.selected_id())),
                         ("Ctrl+Return", self.generate_path), ("Ctrl+Shift+Return", self.generate_all)):
            QShortcut(QKeySequence(keys), self, activated=fn)
        QShortcut(QKeySequence("Delete"), self.graph,
                  activated=lambda: self.node_action("delete", self.selected_id()))
        QShortcut(QKeySequence("F"), self.graph, activated=self.graph.fit)
        self._update_buttons()

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        q = self.qs
        self.offload.setValue(q.value("offload", 90, type=int))
        self.keep_loaded.setChecked(q.value("keepLoaded", True, type=bool))
        self.comfy_url.setText(q.value("comfyUrl", DEFAULT_COMFY_URL, type=str))
        self.comfy_launcher.setText(q.value("comfyLauncher", DEFAULT_COMFY_LAUNCHER, type=str))
        self.comfy_dir.setText(q.value("comfyDir", DEFAULT_COMFY_DIR, type=str))
        self.comfy_cleanup.setChecked(q.value("comfyCleanup", True, type=bool))
        self.idle_minutes.setValue(q.value("idleMinutes", 15, type=int))
        self.min_free.setValue(q.value("minFree", 18.0, type=float))
        self.aitk.setText(q.value("aitk", DEFAULT_AITK, type=str))
        if g := q.value("geometry"):
            self.restoreGeometry(g)
        self._fill_recent()
        proj = q.value("project", "", type=str)
        if proj and Path(proj).is_dir():
            self.open_project(Path(proj), quiet=True)

    def save_settings(self):
        q = self.qs
        q.setValue("offload", self.offload.value())
        q.setValue("keepLoaded", self.keep_loaded.isChecked())
        q.setValue("comfyUrl", self.comfy_url.text().strip())
        q.setValue("comfyLauncher", self.comfy_launcher.text().strip())
        q.setValue("comfyDir", self.comfy_dir.text().strip())
        q.setValue("comfyCleanup", self.comfy_cleanup.isChecked())
        q.setValue("idleMinutes", self.idle_minutes.value())
        q.setValue("minFree", self.min_free.value())
        q.setValue("aitk", self.aitk.text().strip())
        q.setValue("project", str(self.project) if self.project else "")
        if self.tree is not None:
            q.setValue("lastOptions", json.dumps(self.tree.options))
        q.setValue("geometry", self.saveGeometry())
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
        m.addSeparator()
        m.addAction("Quit", self.close).setShortcut(QKeySequence("Ctrl+Q"))

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
        self.setWindowTitle(f"{self.project.name} — Promptwood" if self.project else "Promptwood")

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
        d = QFileDialog.getExistingDirectory(self, "Save the project as (choose the folder to put the copy in)",
                                             str(self.project.parent))
        if not d:
            return
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save as", "Name of the copy:", text=f"{self.project.name}-copy")
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
        source, bridge = self.project, self.bridge
        self.status.setText(f"Copying the project to {target} …")
        self.act_save_as.setEnabled(False)

        def work():
            try:
                shutil.copytree(source, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("*.part", "*.part.*", "video-chain-*"))
                tree = T.load(target)                 # the copy remembers where its start image came from
                T.save(target, tree)
                bridge.copied.emit(str(target), "")
            except Exception as e:  # noqa: BLE001
                bridge.copied.emit(str(target), str(e))
        threading.Thread(target=work, daemon=True).start()

    def _copied(self, target: str, error: str):
        self.act_save_as.setEnabled(True)
        if error:
            self.status.setText(f"Could not copy the project: {error}")
            return
        if self.open_project(Path(target)):
            self.status.setText(f"Saved as {Path(target).name}; you are working in the copy now.")

    def close_project(self):
        if self.running or self.project is None:
            return
        self._save_timer.stop()
        self._save_tree()
        self.preview.stop()
        self.project = self.tree = None
        self.project_edit.clear()
        self.qs.setValue("project", "")
        for c in self.cards:
            self.path_box.removeWidget(c)
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

    def open_project(self, folder: Path, quiet: bool = False) -> bool:
        if self.running:
            return False
        self._save_timer.stop()
        self._save_tree()
        folder = Path(folder).expanduser().resolve()
        try:
            folder.mkdir(parents=True, exist_ok=True)
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
                tree.options.update({k: v for k, v in last.items() if k in T.DEFAULT_OPTIONS})
            except ValueError:
                pass
        self.project, self.tree = folder, tree
        self.project_edit.setText(str(folder))
        self.project_edit.setToolTip(str(folder))
        self.preview.stop()
        self.live, self.progress_of = {}, {}
        self._options_to_widgets()
        self._show_start_image()
        self._show_references()
        if not tree.nodes:
            tree.selected = tree.add(None, "").id
        if tree.selected is None:
            tree.selected = START
        self._tree_stamp = self._stamp()
        self._write_tree()
        self.qs.setValue("project", str(folder))
        self._remember_project(folder)
        self._update_title()
        self.refresh(rebuild_path=True)
        QTimer.singleShot(0, self.graph.fit)
        self.status.setText(f"Project {folder.name}: {len(tree.nodes)} clip{'s' if len(tree.nodes) != 1 else ''}.")
        return True

    def _open_dir(self):
        if self.project and self.project.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project)))

    def _options_to_widgets(self):
        o = self.tree.options
        self._loading = True
        try:
            self._fill_project_lora(o.get("lora"))
            self.extra_loras.blockSignals(True)
            self.extra_loras.set_items(self.known_loras(), o.get("extra_loras") or [])
            self.extra_loras.blockSignals(False)
            self.strength.setValue(float(o.get("strength", 1.0)))
            for k in range(self.frames.count() - 1, len(FRAME_CHOICES) - 1, -1):   # an old value added earlier
                self.frames.removeItem(k)
            fr = int(o.get("frames", 73))
            if fr in FRAME_CHOICES:
                self.frames.setCurrentIndex(FRAME_CHOICES.index(fr))
            else:
                select_frames(self.frames, fr)
            i = self.resolution.findData(int(o.get("resolution", 512)))
            self.resolution.setCurrentIndex(i if i >= 0 else 1)
            self.steps.setValue(int(o.get("steps", 28)))
            self.audio.setChecked(bool(o.get("audio", True)))
            i = self.backend.findData(o.get("backend", "comfyui"))
            self.backend.setCurrentIndex(i if i >= 0 else 0)
            ident = o.get("identity") or {}
            self.ident_mode.setCurrentIndex(max(0, self.ident_mode.findData(ident.get("mode", "off"))))
            self.ref_size.setCurrentIndex(max(0, self.ref_size.findData(ident.get("size", "match"))))
            self.ref_hint.setChecked(bool(ident.get("hint", True)))
            turbo = o.get("turbo") or {}
            self.speed.setCurrentIndex(1 if turbo.get("on") else 0)
            self.turbo_steps.setValue(int(turbo.get("steps", 6)))
            self.turbo_strength.setValue(float(turbo.get("strength", 1.0)))
            self._fill_turbo(turbo.get("lora"))
            self.keep_rep.setChecked(bool(o.get("keep_repeated", False)))
        finally:
            self._loading = False

    def _options_changed(self, *_):
        if self._loading or self.tree is None or self.running:
            return
        lora = self.lora.currentData()
        self.tree.options.update({
            "lora": lora_path(lora) if lora else None, "strength": self.strength.value(),
            "frames": self.frames.currentData(), "resolution": self.resolution.currentData(),
            "steps": self.steps.value(), "audio": self.audio.isChecked(), "keep_repeated": self.keep_rep.isChecked(),
            "backend": self.backend.currentData(), "extra_loras": self.extra_loras.items(),
            "turbo": {"on": self.speed.currentData() == "turbo", "lora": self.turbo_lora.currentData(),
                      "steps": self.turbo_steps.value(), "strength": round(self.turbo_strength.value(), 2)}})
        self._update_size_info()
        self._show_references()
        self._turbo_widgets()
        self._refresh_card_choices()
        self._changed()

    # ---- identity references
    def _identity_changed(self, *_):
        if self._loading or self.tree is None or self.running:
            return
        ident = dict(self.tree.options.get("identity") or {})
        ident.update({"mode": self.ident_mode.currentData(), "size": self.ref_size.currentData(),
                      "hint": self.ref_hint.isChecked()})
        self.tree.options["identity"] = ident
        self._show_references()
        self._changed()

    def _add_reference(self):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        f = pick_image(self, "Reference image (best a close-up of the face)", self.qs, None, self._picker_places())
        if f:
            self.add_reference(f)

    def add_reference(self, path: str):
        if self.project is None or self.running:
            return
        if QImage(path).isNull():
            self.status.setText(f"Not an image: {Path(path).name}")
            return
        try:
            T.add_reference(self.project, Path(path), self.tree)
        except (ValueError, OSError, RuntimeError) as e:
            self.status.setText(f"Could not add the reference: {e}")
            return
        if self.ident_mode.currentData() == "off":      # adding one means using them
            self.ident_mode.setCurrentIndex(self.ident_mode.findData("reference"))
        self._show_references()
        self._write_tree()
        self._changed()

    def _remove_reference(self):
        item = self.refs.currentItem()
        if item is None or self.project is None or self.running:
            return
        T.remove_reference(self.project, item.data(Qt.UserRole), self.tree)
        self._pix.clear()
        self._show_references()
        self._write_tree()
        self._changed()

    def _show_references(self):
        self.refs.clear()
        ident = (self.tree.options.get("identity") or {}) if self.tree else {}
        for k, rel in enumerate(ident.get("images") or [], start=1):
            pm = self._pixmap(self.project / rel) if self.project else None
            item = QListWidgetItem(QIcon(pm) if pm else QIcon(), f"<Picture {k}>")
            item.setData(Qt.UserRole, rel)
            item.setToolTip(rel)
            self.refs.addItem(item)
        on = ident.get("mode") == "reference"
        n = len(ident.get("images") or [])
        if not on:
            text = "Off: every clip only continues from the last frame of the clip before."
        elif not n:
            text = "Add a reference image (best a sharp close-up of the face)."
        elif self.current_backend() != "comfyui":
            text = "Reference images need the ComfyUI generator (Clips → Generator)."
        else:
            text = (f"{n} reference image{'s' if n != 1 else ''}: every clip sees the same face. Uses H3's ref2va "
                    "model; LoRAs trained on the normal model may act weaker there.")
        self.ident_info.setText(text)

    def _changed(self):
        """Something the clips depend on changed: save soon and update the states."""
        self._save_timer.start()
        self._refresh_timer.start()

    def _save_tree(self):
        if self.project is not None and self.tree is not None and not self.running:
            self._write_tree()

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
        T.save(self.project, self.tree)
        self._tree_stamp = self._stamp()
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
        f = pick_image(self, "Start image", self.qs, start, self._picker_places())
        if f:
            self.set_image(f)

    def _picker_places(self) -> list[tuple[str, str]]:
        """Extra places in the image picker: the open project (its clips' last frames are there)."""
        if self.project is None:
            return []
        out = [(f"Project {self.project.name}", str(self.project))]
        if (self.project / "nodes").is_dir():
            out.append(("Project clips (last frames)", str(self.project / "nodes")))
        return out

    def set_image(self, path: str):
        if self.running:
            return
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        if QImage(path).isNull():
            self.status.setText(f"Not an image: {Path(path).name}")
            return
        try:
            T.set_start_image(self.project, Path(path), self.tree)
        except (OSError, RuntimeError) as e:
            self.status.setText(f"Could not use {Path(path).name}: {e}")
            return
        self._pix.clear()
        self._write_tree()
        self._show_start_image()
        self.refresh()

    def _show_start_image(self):
        start = self.project / T.START_FILE if self.project else None
        if start and start.is_file():
            self.start_tile.set_image(QImage(str(start)))
            self.start_tile.setToolTip(self.tree.image or str(start))
        else:
            self.start_tile.clear_frame("Click or drop an image")
        self._update_size_info()

    def _update_size_info(self):
        start = self.project / T.START_FILE if self.project else None
        if not start or not start.is_file():
            self.size_info.setText("")
            return
        img = QImage(str(start))
        w, h = img.width(), img.height()
        vw, vh = chain.canvas_size(w, h, self.resolution.currentData())
        r = (w / h) / (vw / vh)
        note = "" if 0.97 < r < 1.03 else " (ratio outside 2:5 … 5:2, it gets stretched)"
        name = Path(self.tree.image).name if self.tree and self.tree.image else T.START_FILE
        self.size_info.setText(f"{name}: {w}×{h} → {vw}×{vh} video{note}")

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
        if n.made and self.project:
            return self.project / n.made["frame"]
        return None

    def state(self, nid: str) -> str:
        return self.live.get(nid) or self.status_of.get(nid, "new")

    def refresh(self, rebuild_path: bool = False):
        if self.tree is None:
            self.graph.show_nodes([])
            self._update_buttons()
            return
        checker = T.Checker(self.project, self.tree)
        try:
            self.status_of = {nid: checker.status(nid) for nid in self.tree.nodes}
        except Exception as e:  # noqa: BLE001  (e.g. an unreadable start image)
            self.status_of = {nid: "new" for nid in self.tree.nodes}
            self.status.setText(f"Could not check the clips: {e}")
        sel = self.selected_id()
        on_path = {n.id for n in self.selected_path()}
        start_pm = self._pixmap(self.project / T.START_FILE)
        infos = [NodeInfo(START, None, "Start", "", Path(self.tree.image).name if self.tree.image else "",
                          "start", start_pm, on_path=True, selected=sel == START)]
        for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id)):
            infos.append(NodeInfo(n.id, n.parent or START, f"Clip {self.tree.depth(n.id)}", f"#{n.id[1:]}",
                                  n.prompt, self.state(n.id), self._pixmap(self._node_frame(n)),
                                  self.progress_of.get(n.id, 0.0), n.id in on_path, n.id == sel,
                                  self.badge(n)))
        self.graph.show_nodes(infos)
        if rebuild_path or [c.nid for c in self.cards] != [n.id for n in self.selected_path()]:
            self._rebuild_path()
        else:
            self._update_cards()
        made = sum(1 for s in self.status_of.values() if s == "done")
        branches = len(self.tree.leaves())
        self.tree_info.setText(f"{len(self.tree.nodes)} clips · {branches} branch{'es' if branches != 1 else ''}"
                               f" · {made} made")
        self._update_buttons()

    def badge(self, n: T.Node) -> str:
        """Short text for a clip's own settings, shown on its node ('' = all project defaults)."""
        ov, parts = n.overrides or {}, []
        if "lora" in ov:
            parts.append("no LoRA" if ov["lora"] == T.NO_LORA else lora_label(ov["lora"])[:18])
        if "strength" in ov and ov.get("lora") != T.NO_LORA:
            parts.append(f"×{ov['strength']:g}")
        for e in ov.get("extra_loras") or []:
            parts.append(f"+ {lora_label(e['path'])[:14]} ×{e.get('strength', 1.0):g}")
        if "frames" in ov:
            parts.append(f"{ov['frames'] / 24:.1f} s")
        if "steps" in ov:
            parts.append(f"{ov['steps']} steps")
        return " · ".join(parts)

    def _rebuild_path(self):
        for c in self.cards:
            self.path_box.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        path = self.selected_path()
        self.path_hint.setVisible(not path)
        self.path_hint.setText("The start image is selected. Select a clip to see its path, or press Continue "
                               "(Ctrl+N) to add a clip after the start image.")
        for k, n in enumerate(path):
            card = PromptCard(n)
            card.edited.connect(self._card_edited)
            card.seedChanged.connect(self._seed_changed)
            card.overridesChanged.connect(self._overrides_changed)
            card.chooseLora.connect(self._choose_node_lora)
            card.set_choices(self._project_lora_label(), int(self.tree.options.get("frames", 73)),
                             int(self.tree.options.get("steps", 28)), self.known_loras())
            card.action.connect(self.node_action)
            card.play.connect(self._play_node)
            self.path_box.insertWidget(k, card)
            self.cards.append(card)
            card.set_editable(not self.running)
        self._update_cards()
        if self.cards:
            last = self.cards[-1]
            QTimer.singleShot(0, lambda: self.path_area.ensureWidgetVisible(last))

    def _update_cards(self):
        drop = not self.tree.options.get("keep_repeated")
        t = 0.0
        for k, card in enumerate(self.cards):
            n = self.tree.nodes.get(card.nid)
            if n is None:
                continue
            frames = int(n.overrides.get("frames", self.tree.options.get("frames", 73)))
            length = (frames - (1 if drop and k > 0 else 0)) / 24
            card.title.setText(f"Clip {k + 1}")
            card.span.setText(f"{fmt_time(t)} – {fmt_time(t + length)} in this path")
            t += length
            st = self.state(n.id)
            card.set_state(st, self._pixmap(self._node_frame(n)), "will be made again" if st == "changed" else "")
            if st == "generating" and n.id in self.progress_of:
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
        sel = self.selected_id() if has else None
        is_clip = bool(has and sel and sel != START)
        idle = not self.running
        self.b_continue.setEnabled(has and idle)
        for w in (self.act_save, self.act_save_as, self.act_close):
            w.setEnabled(has and idle)
        for w in (self.act_new, self.act_open, self.b_new, self.b_open, self.b_recent, self.recent_file_menu):
            w.setEnabled(idle)
        for b in (self.b_branch, self.b_variant, self.b_delete):
            b.setEnabled(is_clip and idle)
        path = self.selected_path()
        all_done = bool(path) and all(self.state(n.id) == "done" for n in path)
        free = idle and not self._joins           # a join reads clips a new run would replace
        self.b_gen.setEnabled(free and bool(path))
        self.b_gen_all.setEnabled(free and has and bool(self.tree.nodes))
        self.b_delete.setEnabled(is_clip and free)
        self.b_play_path.setEnabled(all_done and idle)
        self.b_export.setEnabled(all_done and idle)
        self.b_stop.setEnabled(not idle)
        self._update_comfy_label()
        self.extra_loras.setEnabled(idle)
        for w in (self.start_tile, self.lora, self.b_lora, self.strength, self.frames, self.resolution, self.steps,
                  self.audio, self.keep_rep, self.offload, self.min_free, self.aitk, self.b_project, self.backend,
                  self.comfy_url, self.ident_mode, self.ref_size, self.ref_hint, self.b_ref_add, self.b_ref_start,
                  self.b_ref_del, self.speed):
            w.setEnabled(idle)
        self._turbo_widgets()                      # after the loop: turbo locks the steps fields again

    def select(self, nid):
        if self.tree is None:
            return
        self.tree.selected = nid if (nid == START or nid in self.tree.nodes) else START
        self._save_timer.start()
        self.refresh()

    def _card_edited(self, nid: str):
        card = next((c for c in self.cards if c.nid == nid), None)
        if card is None or self.tree is None or nid not in self.tree.nodes:
            return
        self.tree.nodes[nid].prompt = card.text()
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
        cands = ([self.tree.options.get("lora")] if self.tree else []) + \
            ([n.overrides.get("lora") for n in self.tree.nodes.values()] if self.tree else []) + recent
        if self.tree:
            cands += [e.get("path") for e in self.tree.options.get("extra_loras") or []]
            cands += [e.get("path") for n in self.tree.nodes.values() for e in n.overrides.get("extra_loras") or []]
        for lo in cands:
            if lo and lo != T.NO_LORA and lo not in listed and os.path.realpath(lo) not in listed \
                    and Path(lo).is_file():
                out.append((lo, lora_label(lo)))
                listed.add(lo)
        return out

    def _project_lora_label(self) -> str:
        lo = self.tree.options.get("lora") if self.tree else None
        if not lo:
            return "none"
        real = os.path.realpath(lo)
        return next((text for p, text in self.known_loras() if p == lo or os.path.realpath(p) == real),
                    lora_label(lo))

    def _fill_turbo(self, current: str | None):
        """The turbo LoRA list; without a choice yet the recommended v4 ema (or the first) is selected."""
        self.turbo_lora.blockSignals(True)
        try:
            self.turbo_lora.clear()
            known = [(p, Path(n).stem) for p, n in self.turbo_loras]
            if not current and known:
                current = next((p for p, n in known if "v4" in n and "ema" in n), known[0][0])
            if known or current:
                fill_lora_combo(self.turbo_lora, known, current)
            else:
                self.turbo_lora.addItem("no turbo LoRA found", None)
        finally:
            self.turbo_lora.blockSignals(False)
        self._turbo_widgets()

    def _turbo_widgets(self):
        on = self.speed.currentData() == "turbo"
        idle = not self.running
        for w in (self.turbo_lora, self.turbo_steps, self.turbo_strength):
            w.setEnabled(on and idle)
        self.steps.setEnabled(idle and not on)
        self.steps.setToolTip("Turbo mode: every clip uses the turbo steps" if on else
                              "Sampling steps per clip. The trainer's samples use 20, ai-toolkit's H3 default is 28")
        for card in self.cards:
            card.steps_locked = on
            card.steps.setEnabled(idle and not on and not card.edit.isReadOnly())

    def _fill_project_lora(self, current: str | None):
        self.lora.blockSignals(True)
        try:
            self.lora.clear()
            self.lora.addItem("none (base model)", None)
            fill_lora_combo(self.lora, self.known_loras(), current)
        finally:
            self.lora.blockSignals(False)

    def scan_loras(self):
        """Find the H3 LoRAs in ComfyUI's LoRA folders (header reads only), in the background."""
        comfy_dir, bridge = Path(os.path.expanduser(self.comfy_dir.text().strip() or DEFAULT_COMFY_DIR)), self.bridge

        def work():
            try:
                from ..comfy import comfy_lora_roots, find_h3_loras
                from ..comfy import is_turbo_lora
                found = [(p, n, is_turbo_lora(Path(p))) for p, n in find_h3_loras(comfy_lora_roots(comfy_dir))]
            except Exception:  # noqa: BLE001
                found = []
            bridge.loras.emit(found)
        threading.Thread(target=work, daemon=True).start()

    def _loras_found(self, found: list):
        self.h3_loras = [(p, n) for p, n, *t in found if not (t and t[0])]
        self.turbo_loras = [(p, n) for p, n, *t in found if t and t[0]]
        if self.tree is not None:
            self._fill_turbo((self.tree.options.get("turbo") or {}).get("lora"))
        if self.tree is None:
            return
        self._fill_project_lora(self.tree.options.get("lora"))
        self.extra_loras.blockSignals(True)             # the list of LoRAs changed: rebuild the project's rows
        self.extra_loras.set_items(self.known_loras(), self.tree.options.get("extra_loras") or [])
        self.extra_loras.blockSignals(False)
        self._refresh_card_choices()

    def _refresh_card_choices(self):
        """The cards' lists show the project's LoRA, length and steps as 'default': update them in place."""
        if self.tree is None:
            return
        known = self.known_loras()
        for card in self.cards:
            card.set_choices(self._project_lora_label(), int(self.tree.options.get("frames", 73)),
                             int(self.tree.options.get("steps", 28)), known)

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
        if self.tree and nid in self.tree.nodes:
            self.tree.nodes[nid].overrides = dict(ov)
            self._changed()

    def _choose_node_lora(self, nid: str):
        card = next((c for c in self.cards if c.nid == nid), None)
        if card is None:
            return
        cur = card.lora.currentData()
        start = str(Path(cur).parent) if cur and cur != T.NO_LORA else (
            str(Path(self.tree.options["lora"]).parent) if self.tree.options.get("lora") else self._lora_dir())
        f, _ = QFileDialog.getOpenFileName(self, "LoRA for this clip", start, "LoRA (*.safetensors)")
        if f:
            self.set_node_lora(nid, f)

    def set_node_lora(self, nid: str, path: str | None):
        """Give one clip its own LoRA (a file), no LoRA (T.NO_LORA) or the project's (None)."""
        if self.tree is None or nid not in self.tree.nodes:
            return
        ov = dict(self.tree.nodes[nid].overrides)
        if path is None:
            ov.pop("lora", None)
        else:
            ov["lora"] = path if path == T.NO_LORA else lora_path(path)
            if path != T.NO_LORA:
                self._remember_lora(ov["lora"])
        self.tree.nodes[nid].overrides = ov
        self._changed()
        self.refresh(rebuild_path=True)

    def _seed_changed(self, nid: str, seed: int):
        if self.tree and nid in self.tree.nodes:
            self.tree.nodes[nid].seed = int(seed)
            self._changed()

    def _new_seed(self) -> int:
        used = {n.seed for n in self.tree.nodes.values()}
        while True:
            s = random.randint(1, 2_000_000_000)
            if s not in used:
                return s

    def node_action(self, action: str, nid: str | None):
        if self.tree is None or self.running:
            return
        node = self.tree.nodes.get(nid) if nid and nid != START else None
        if action == "continue":
            new = self.tree.add(node.id if node else None, "")
        elif action in ("branch", "variant") and node is not None:
            new = self.tree.add(node.parent, node.prompt if action == "variant" else "",
                                self._new_seed() if action == "variant" else None)
        elif action == "delete" and node is not None:
            below = len(self.tree.descendants(node.id))
            made = node.made is not None or any(d.made for d in self.tree.descendants(node.id))
            msg = f"Delete clip #{node.id[1:]}" + (f" and the {below} clip{'s' if below != 1 else ''} after it"
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
        if self.cards and self.cards[-1].nid == new.id:
            self.cards[-1].edit.setFocus()

    def _context(self, nid: str, pos):
        if self.running or self.tree is None:
            return
        m = QMenu(self)
        acts = {m.addAction("Continue from here"): "continue"}
        play = gen = delete = None
        if nid != START:
            acts[m.addAction("Branch (another continuation)")] = "branch"
            acts[m.addAction("Variant (same prompt, other seed)")] = "variant"
            m.addSeparator()
            play = m.addAction("Play this clip")
            play.setEnabled(bool(self.tree.nodes[nid].made))
            gen = m.addAction("Generate the path to here")
            m.addSeparator()
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
        elif chosen is gen:
            QTimer.singleShot(0, self.generate_path)
        elif chosen is delete:
            QTimer.singleShot(0, lambda: self.node_action("delete", nid))

    def _exec_menu(self, menu: QMenu, pos):
        return menu.exec(pos)

    def _play_node(self, nid: str):
        if self.tree is None or nid not in self.tree.nodes:
            return
        n = self.tree.nodes[nid]
        if n.made and (self.project / n.made["video"]).is_file():
            self.tabs.setCurrentIndex(0)
            self.preview.open(self.project / n.made["video"], f"Clip #{nid[1:]} · {n.prompt[:60]}")

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
                    bridge.joined.emit(T.join_path(project, tree, nid, out), None, purpose, pre, str(project))
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
        secs = chain.probe(path)["frames"] / 24
        if purpose == "play":
            self.tabs.setCurrentIndex(0)
            self.preview.open(path, f"Path to clip #{Path(path).stem[1:]} · {fmt_time(secs)}")
            self.status.setText(f"{pre}Path joined: {fmt_time(secs)}.")
        else:
            self.status.setText(f"{pre}Exported {Path(path).name} ({fmt_time(secs)}).")

    def export_path(self):
        path = self.selected_path()
        if not path:
            return
        default = str(self.project / f"{self.project.name}-{path[-1].id}.mp4")
        f, _ = QFileDialog.getSaveFileName(self, "Export path", default, "Video (*.mp4)")
        if f:
            self.join_selected("export", Path(f if f.endswith(".mp4") else f + ".mp4"))

    def _choose_lora(self):
        cur = self.lora.currentData()
        start = str(Path(cur).parent) if cur else self._lora_dir()
        f, _ = QFileDialog.getOpenFileName(self, "LoRA", start, "LoRA (*.safetensors)")
        if f:
            self._remember_lora(lora_path(f))
            self._fill_project_lora(lora_path(f))
            self._options_changed()

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
            self.gpu.setProperty("severity", "warning" if free < self.min_free.value() else "")
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
        """What a loaded worker is bound to; a different key needs a new worker."""
        return (self.tree.options.get("model", T.DEFAULT_OPTIONS["model"]), bool(self.tree.options.get("audio", True)),
                self.offload.value(), self.aitk.text().strip() or DEFAULT_AITK, self.current_backend(),
                self.comfy_url.text().strip() or DEFAULT_COMFY_URL, self.comfy_dir.text().strip(),
                self.comfy_cleanup.isChecked())

    def current_backend(self) -> str:
        return (self.tree.options.get("backend") if self.tree else None) or "comfyui"

    def project_loras(self) -> list[str]:
        """LoRA files of this project (its default and every clip's own), prepared when a worker loads."""
        out = []
        for lo in [self.tree.options.get("lora")] + [n.overrides.get("lora") for n in self.tree.nodes.values()]:
            if lo and lo != T.NO_LORA and lo not in out and Path(lo).is_file():
                out.append(lo)
        return out

    def _start(self, targets: list[str]):
        if self.running or self.tree is None:
            return
        if self._joins:
            self.status.setText("Wait until the path is joined.")
            return
        problems = []
        if not (self.project / T.START_FILE).is_file():
            problems.append("choose a start image")
        ident = self.tree.options.get("identity") or {}
        if ident.get("mode") == "reference":
            if not ident.get("images"):
                problems.append("add a reference image (Identity) or switch it off")
            elif self.current_backend() != "comfyui":
                problems.append("reference images need the ComfyUI generator")
        turbo = self.tree.options.get("turbo") or {}
        if turbo.get("on"):
            if not turbo.get("lora"):
                problems.append("choose a turbo LoRA (Clips → Turbo LoRA) or switch Speed to Normal")
            elif self.current_backend() != "comfyui":
                problems.append("turbo mode needs the ComfyUI generator")
        nodes = {n.id: n for t in targets for n in self.tree.path_to(t)}
        empty = [n for n in nodes.values() if not n.prompt.strip()]
        if empty:
            problems.append(f"write the prompt of clip #{empty[0].id[1:]}")
        try:
            checker = T.Checker(self.project, self.tree)
            if checker.base() is not None and self.current_backend() != "comfyui":
                if any(checker.settings(nid).get("extra_loras") for nid in nodes):
                    problems.append("several LoRAs on one clip need the ComfyUI generator")
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
        self._save_timer.stop()
        if not self._write_tree():
            return
        self.save_settings()
        self._job = {"targets": targets, "sent": False, "restarted": False}
        self._error, self._stopping, self._phase = "", False, "starting"
        self._targets, self._plan = targets, []
        self.live, self.progress_of = {}, {}
        self.log.clear()
        self.preview.stop()
        for c in self.cards:
            c.set_editable(False)
        self._update_buttons()
        self.progress.setRange(0, 0)
        if self.worker_alive and self._worker_idle and self._worker_bound == self._worker_key():
            self._send_job()
        elif self.worker_alive:                    # loaded for other settings: replace it
            self.status.setText("Settings changed: loading the model again …")
            self._replace_worker = True
            self._quit_worker()
        else:
            self._start_worker(self.project_loras())

    def _program(self) -> tuple[str, list[str]]:
        if self.command:
            return self.command[0], list(self.command[1:])
        if self.current_backend() == "comfyui":            # no torch needed: this app's own Python
            return sys.executable, ["-m", "video_chain"]
        return str(LAUNCHER), []

    def _start_worker(self, loras: list[str]):
        program, pre = self._program()
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("AITK", self.aitk.text().strip() or DEFAULT_AITK)
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read_out)
        self.proc.readyReadStandardError.connect(self._read_err)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._proc_error)
        self._buf = self._ebuf = ""
        self._worker_idle, self._worker_bound, self._replace_worker, self._restart = False, self._worker_key(), \
            False, False
        self._worker_loras = list(loras)
        minutes = self.idle_minutes.value() if self.keep_loaded.isChecked() else 0
        args = pre + ["--serve", "--events", "--offload", f"{self.offload.value() / 100:g}",
                      "--min-free-gb", f"{self.min_free.value():g}", "--idle-minutes", f"{minutes:g}",
                      "--model", self._worker_bound[0], "--loras", ",".join(loras),
                      "--backend", self._worker_bound[4], "--comfy-url", self._worker_bound[5]]
        if self._worker_bound[6]:
            args += ["--comfy-dir", os.path.expanduser(self._worker_bound[6])]
        if not self._worker_bound[7]:
            args.append("--keep-comfy-copies")
        if not self._worker_bound[1]:
            args.append("--no-audio")
        self.log.appendPlainText("$ " + " ".join([program, *args]))
        self.status.setText("Starting …")
        self.proc.start(program, args)
        self._update_model_label()

    def _send_job(self):
        if self._job is None or self.proc is None:
            return
        self._job["sent"] = True
        self._worker_idle = False
        cmd = {"cmd": "run", "tree": str(self.project), "targets": self._job["targets"]}
        self.proc.write((json.dumps(cmd) + "\n").encode())
        self._update_model_label()

    def _quit_worker(self):
        """Ask an idle worker to quit (frees the model); killed if it does not go."""
        if self.proc is None:
            return
        if self.proc.state() == QProcess.Running:
            self.proc.write(b'{"cmd": "quit", "free": true}\n')     # ComfyUI: unload its models too
        self._kill_timer.start()

    def unload(self):
        if self.worker_alive and not self.running:
            self.status.setText("Unloading the model …")
            self._quit_worker()

    def stop(self):
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
        self._buf += bytes(p.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._line(line)

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
        self._tree_stamp = self._stamp()            # the generator wrote it; this window is up to date

    def _event(self, ev: dict):
        e = ev.get("e")
        if e == "loading":
            self._worker_idle = False
            self.status.setText("Loading MiniMax-H3 … (a few minutes)")
            self.progress.setRange(0, 0)
            self._model_state = "loading"
            self._update_model_label()
        elif e == "loaded":
            self.status.setText("Model loaded.")
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
            if ev.get("reason") == "idle" and not self.running:
                self.status.setText("The model was unloaded after the idle time.")
        elif e == "restart":
            self._restart = True
            self._restart_loras = list(dict.fromkeys(self._worker_loras + list(ev.get("loras") or [])))
        elif e == "start":
            self.status.setText(f"{ev.get('nodes', 0)} clips to make at {ev['width']}×{ev['height']}, "
                                f"{ev['frames']} frames each.")
        elif e == "plan":
            self._plan = list(ev.get("make", []))
        elif e == "node":
            nid, st = ev["id"], ev["state"]
            if st == "generating":
                self.live[nid] = "generating"
                self.progress_of[nid] = 0.0
                k = self._plan.index(nid) + 1 if nid in self._plan else 1
                self._overall(k, 0)
                self.status.setText(f"Clip #{nid[1:]} ({k} of {len(self._plan)}) …")
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
                self.status.setText(f"Clip #{nid[1:]} ({k} of {len(self._plan)}): step {ev['i']} / {ev['n']}")
                item = self.graph.items_by_id.get(nid)
                if item is not None:
                    item.info.progress = frac
                    item.update()
                card = next((c for c in self.cards if c.nid == nid), None)
                if card is not None:
                    card.set_progress(ev["i"], ev["n"])
        elif e == "done":
            self._phase = "done"
        elif e == "error":
            self._error = ev.get("msg", "error")

    def _job_finished(self, code: int = 0):
        """End of a job: the worker reported idle again (code 0), or the worker ended (its exit code)."""
        ok = self._phase == "done" and code == 0 and not self._error
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
        if ok:
            self.progress.setValue(1000)
            made = len(self._plan)
            msg = f"Done: {made} clip{'s' if made != 1 else ''} made." if made else "Everything was already made."
            self.status.setText(msg)
            path = self.selected_path()
            if len(self._targets) == 1 and path and path[-1].id == self._targets[0]:
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
            rest = bytes(p.readAllStandardOutput()).decode(errors="replace")
            for line in (self._buf + rest).split("\n"):
                self._line(line)
            err = bytes(p.readAllStandardError()).decode(errors="replace")
            for line in (self._ebuf + err).split("\n"):
                line = line.split("\r")[-1].rstrip()
                if line:
                    self.log.appendPlainText(line)
            p.deleteLater()
        self._buf = self._ebuf = ""
        self._worker_idle = False
        self._model_state = ""
        if self._job is not None and self._restart and not self._job["restarted"] and not self._stopping:
            # the job needs a worker loaded differently (a LoRA of a higher rank, other sound or model)
            self._job["restarted"], self._job["sent"] = True, False
            self.status.setText("Loading the model again for this job …")
            self._start_worker(self._restart_loras)
        elif self._job is not None and self._replace_worker:
            self._job["sent"] = False
            self._start_worker(self.project_loras())
        elif self._job is not None:
            self._job_finished(code if code else 1)
            if code == 4:                                   # COMFY_DOWN
                self.status.setText(f"ComfyUI is not running at {self.comfy_url.text().strip()}. "
                                    "Start it (Advanced → Start ComfyUI) and generate again.")
                self.comfy_up = False
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

    def _comfy_state(self, up: bool):
        self._comfy_polling = False
        was, self.comfy_up = self.comfy_up, up
        if up and was is False and self._comfy_starting:
            self._comfy_starting = 0.0
            self.status.setText("ComfyUI is running.")
        self._update_comfy_label()

    def _update_comfy_label(self):
        if self.tree is not None and self.current_backend() != "comfyui":
            self.comfy_label.setText("")
            return
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

    def start_comfy(self):
        """Start ComfyUI in the background with the user's launcher (it runs on its own, not tied to this app)."""
        cmd = os.path.expanduser(self.comfy_launcher.text().strip() or DEFAULT_COMFY_LAUNCHER)
        if not Path(cmd).is_file():
            self.status.setText(f"ComfyUI launcher not found: {cmd}")
            return
        ok = QProcess.startDetached(cmd, [])
        ok = ok[0] if isinstance(ok, tuple) else ok
        if not ok:
            self.status.setText(f"Could not start {cmd}")
            return
        self._comfy_starting = time.monotonic()
        self.status.setText("Starting ComfyUI … (the clips can be generated once it runs)")
        self._update_comfy_label()

    def _update_model_label(self):
        if not self.worker_alive:
            text, sev = "Model: not loaded", ""
        elif self.running and self._model_state == "loading":
            text, sev = "Model: loading …", "warning"
        elif self.running or not self._worker_idle:
            text, sev = "Model: working", ""
        elif self._idle_timeout:
            left = max(0.0, self._idle_timeout - (time.monotonic() - self._idle_since))
            where = " in ComfyUI" if self._worker_bound and self._worker_bound[4] == "comfyui" else ""
            text, sev = f"Model: loaded{where} · unloads in {max(1, round(left / 60))} min", ""
        else:
            text, sev = "Model: loaded", ""
        self.model_label.setText(text)
        self.model_label.setProperty("severity", sev)
        repolish(self.model_label)
        self.b_unload.setVisible(self.worker_alive and not self.running and self._worker_idle)

    def closeEvent(self, e):
        if self.running:
            if QMessageBox.question(self, "Promptwood", "Clips are being generated. Stop and quit?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
        if self.proc is not None:                    # never leave the generator (and the model) behind
            p, self.proc = self.proc, None
            p.finished.disconnect()
            if p.state() == QProcess.Running:
                p.write(b'{"cmd": "quit"}\n')
                p.closeWriteChannel()
                if not p.waitForFinished(3000):
                    p.terminate()
                    if not p.waitForFinished(5000):
                        p.kill()
                        p.waitForFinished(3000)
            self._job = None
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
