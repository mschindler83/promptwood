"""Promptwood window: start image + prompt cards -> chained MiniMax-H3 clips.

Generation runs in a separate process (the video-chain launcher with ai-toolkit's Python) that reports progress as
'@@ {json}' lines; this window only edits, starts, stops and shows. Stopping ends only that process.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
                               QVBoxLayout, QWidget)

from .. import chain
from .theme import icon, repolish, theme
from .widgets import FrameTile, icon_button, label, section_bar

PROJECT_DIR = Path(__file__).resolve().parents[3]
LAUNCHER = PROJECT_DIR / "video-chain"
DEFAULT_AITK = "/path/to/ai/ai-toolkit"
FRAME_CHOICES = [39, 56, 73, 90, 107, 124]
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
STATE_TEXT = {"": "", "waiting": "WAITING", "generating": "GENERATING", "done": "DONE", "reused": "REUSED",
              "failed": "FAILED", "stopped": "STOPPED", "changed": "CHANGED"}
STATE_STYLE = {"done": "done", "reused": "done", "generating": "target", "failed": "failed", "stopped": "missing",
               "changed": "missing", "waiting": ""}


def qimage(path: Path | str) -> QImage:
    img = QImage(str(path))
    return img


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:04.1f}"


# ----------------------------------------------------------------------------- prompt card

class PromptCard(QFrame):
    changed = Signal()
    moveUp = Signal(object)
    moveDown = Signal(object)
    remove = Signal(object)
    play = Signal(object)

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("promptCard")
        self.setAttribute(Qt.WA_StyledBackground)
        self.video: Path | None = None
        self.state = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 8, 10)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = label("Clip 1", type_="heading")
        self.span = label("", level="tertiary")
        self.tag = label("", role="statusTag")
        self.tag.hide()
        head.addWidget(self.title)
        head.addWidget(self.span)
        head.addWidget(self.tag)
        head.addStretch(1)
        self.b_up = icon_button("chu", "Move up")
        self.b_down = icon_button("chd", "Move down")
        self.b_del = icon_button("trash", "Remove this prompt")
        for b in (self.b_up, self.b_down, self.b_del):
            head.addWidget(b)
        v.addLayout(head)
        body = QHBoxLayout()
        body.setSpacing(8)
        self.edit = QPlainTextEdit(text)
        self.edit.setPlaceholderText("Describe what happens in this clip. It starts from the last frame of the clip "
                                     "before (clip 1: from the start image).")
        self.edit.setTabChangesFocus(True)
        self.edit.setMinimumHeight(86)
        self.edit.setMaximumHeight(140)
        body.addWidget(self.edit, 1)
        self.thumb = FrameTile(144, 81)
        self.thumb.setToolTip("Last frame of this clip; click to play the clip")
        self.thumb.clear_frame("not made yet")
        body.addWidget(self.thumb, 0, Qt.AlignTop)
        v.addLayout(body)
        foot = QHBoxLayout()
        self.words = label("", level="tertiary")
        self.step = label("", level="tertiary")
        foot.addWidget(self.words)
        foot.addStretch(1)
        foot.addWidget(self.step)
        v.addLayout(foot)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.hide()
        v.addWidget(self.bar)

        self.edit.textChanged.connect(self._text_changed)
        self.b_up.clicked.connect(lambda: self.moveUp.emit(self))
        self.b_down.clicked.connect(lambda: self.moveDown.emit(self))
        self.b_del.clicked.connect(lambda: self.remove.emit(self))
        self.thumb.clicked.connect(lambda _t: self.play.emit(self))
        self._update_words()
        self._restyle()

    def text(self) -> str:
        return " ".join(self.edit.toPlainText().split())

    def _text_changed(self):
        self._update_words()
        self.changed.emit()

    def _update_words(self):
        n = len(self.text().split())
        self.words.setText(f"{n} words" if n else "empty")

    def set_number(self, k: int, start: float, end: float, first: bool, last: bool):
        self.title.setText(f"Clip {k}")
        self.span.setText(f"{fmt_time(start)} – {fmt_time(end)}")
        self.b_up.setEnabled(not first)
        self.b_down.setEnabled(not last)

    def set_state(self, state: str, step: str = ""):
        self.state = state
        self.tag.setVisible(bool(STATE_TEXT.get(state)))
        self.tag.setText(STATE_TEXT.get(state, state.upper()))
        self.tag.setProperty("state", STATE_STYLE.get(state, ""))
        repolish(self.tag)
        self.step.setText(step)
        self.bar.setVisible(state == "generating")
        if state != "generating":
            self.bar.setValue(0)
        self._restyle()

    def set_progress(self, i: int, n: int):
        if n:
            self.bar.setValue(round(100 * i / n))
            self.step.setText(f"step {i} / {n}")

    def set_result(self, video: Path | None, frame: Path | None):
        self.video = video if video and video.exists() else None
        if frame and frame.exists():
            self.thumb.set_image(qimage(frame))
        else:
            self.thumb.clear_frame("not made yet")
        self.thumb.setCursor(Qt.PointingHandCursor if self.video else Qt.ArrowCursor)

    def set_editable(self, on: bool):
        self.edit.setReadOnly(not on)
        for b in (self.b_up, self.b_down, self.b_del):
            b.setVisible(on)

    def _restyle(self):
        t = theme()
        border = t.accent.name() if self.state == "generating" else t.lineSubtle.name()
        self.setStyleSheet(
            f"QFrame#promptCard {{ background: {t.bgRaised.name()}; border: 1px solid {border};"
            f" border-radius: 6px; }}"
            f"QFrame#promptCard QLabel {{ background: transparent; }}")


# ----------------------------------------------------------------------------- start image tile

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


# ----------------------------------------------------------------------------- preview

class Preview(QWidget):
    """Video player for clips and the chain. The media player is created on first use."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = None
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.stack = QStackedWidget()
        self.placeholder = label("Nothing to play yet.\nClick a clip's picture, or play the whole chain.",
                                 level="tertiary")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setMinimumHeight(220)
        self.stack.addWidget(self.placeholder)
        v.addWidget(self.stack, 1)
        self.name = label("", level="secondary")
        v.addWidget(self.name)
        row = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause (Space)")
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
        self.video.setMinimumHeight(220)
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
        self.qs = settings or QSettings()
        self.command = command          # tests replace the launcher with a fake backend
        self.cards: list[PromptCard] = []
        self.proc: QProcess | None = None
        self._buf = ""
        self._stopping = False
        self._run_out: Path | None = None
        self._clips_total = 0
        self._phase = ""
        self._error = ""
        self._image_size: tuple[int, int] | None = None
        self._build()
        self._load_settings()
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._poll_gpu)
        self._gpu_timer.start(4000)
        self._gpu_proc: QProcess | None = None
        QTimer.singleShot(0, self._poll_gpu)

    # ---- layout
    def _build(self):
        self.resize(1480, 900)
        split = QSplitter(Qt.Horizontal, self)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        self.split = split

        # left: settings
        ls = QScrollArea()
        ls.setWidgetResizable(True)
        ls.setFrameShape(QFrame.NoFrame)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        bar, _ = section_bar("Start image", left)
        lv.addWidget(bar)
        sw = QWidget()
        sv = QVBoxLayout(sw)
        sv.setContentsMargins(10, 8, 10, 8)
        self.start_tile = StartTile(320, 180)
        sv.addWidget(self.start_tile, 0, Qt.AlignHCenter)
        self.image_path = label("", level="secondary")
        self.image_path.setWordWrap(True)
        self.size_info = label("", level="tertiary")
        self.size_info.setWordWrap(True)
        sv.addWidget(self.image_path)
        sv.addWidget(self.size_info)
        lv.addWidget(sw)

        bar, _ = section_bar("LoRA", left)
        lv.addWidget(bar)
        fw = QWidget()
        f = QFormLayout(fw)
        f.setContentsMargins(10, 6, 10, 8)
        lrow = QHBoxLayout()
        self.lora = QLineEdit()
        self.lora.setPlaceholderText("none: base model")
        b = icon_button("fopen", "Choose a LoRA (.safetensors)")
        b.clicked.connect(self._choose_lora)
        lrow.addWidget(self.lora, 1)
        lrow.addWidget(b)
        f.addRow("File", lrow)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 2.0)
        self.strength.setSingleStep(0.05)
        self.strength.setDecimals(2)
        self.strength.setValue(1.0)
        f.addRow("Strength", self.strength)
        lv.addWidget(fw)

        bar, _ = section_bar("Clips", left)
        lv.addWidget(bar)
        cw = QWidget()
        c = QFormLayout(cw)
        c.setContentsMargins(10, 6, 10, 8)
        self.frames = QComboBox()
        for n in FRAME_CHOICES:
            self.frames.addItem(f"{n} frames ({n / 24:.1f} s)", n)
        self.frames.setCurrentIndex(FRAME_CHOICES.index(73))
        self.frames.setToolTip("Length of every clip. Longer clips need more GPU memory and time")
        c.addRow("Length", self.frames)
        self.resolution = QComboBox()
        for r, txt in ((384, "384 (fast draft)"), (512, "512 (like the training)"), (640, "640"), (768, "768")):
            self.resolution.addItem(txt, r)
        self.resolution.setCurrentIndex(1)
        self.resolution.setToolTip("About this many pixels squared, in the start image's aspect ratio. "
                                   "Larger sizes need more GPU memory")
        c.addRow("Size", self.resolution)
        self.steps = QSpinBox()
        self.steps.setRange(4, 80)
        self.steps.setValue(28)
        self.steps.setToolTip("Sampling steps per clip. The trainer's samples use 20, ai-toolkit's H3 default is 28")
        c.addRow("Steps", self.steps)
        srow = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(42)
        self.same_seed = QCheckBox("same for all")
        self.same_seed.setToolTip("Off: clip k uses seed + k - 1")
        srow.addWidget(self.seed, 1)
        srow.addWidget(self.same_seed)
        c.addRow("Seed", srow)
        self.audio = QCheckBox("Generate sound")
        self.audio.setChecked(True)
        c.addRow("", self.audio)
        self.keep_rep = QCheckBox("Keep repeated frames in the chain")
        self.keep_rep.setToolTip("Each clip starts with the last frame of the clip before; by default that repeated "
                                 "frame is left out of chain.mp4")
        c.addRow("", self.keep_rep)
        lv.addWidget(cw)

        bar, _ = section_bar("Output", left)
        lv.addWidget(bar)
        ow = QWidget()
        o = QFormLayout(ow)
        o.setContentsMargins(10, 6, 10, 8)
        orow = QHBoxLayout()
        self.out = QLineEdit()
        self.out.setPlaceholderText("folder for the clips and chain.mp4")
        b = icon_button("fopen", "Choose the output folder")
        b.clicked.connect(self._choose_out)
        self.b_open_out = icon_button("folder", "Open the output folder")
        self.b_open_out.clicked.connect(self._open_out)
        orow.addWidget(self.out, 1)
        orow.addWidget(b)
        orow.addWidget(self.b_open_out)
        o.addRow("Folder", orow)
        self.reuse = QCheckBox("Reuse unchanged clips")
        self.reuse.setChecked(True)
        self.reuse.setToolTip("Clips whose prompt, seed, start image and settings did not change are not generated "
                              "again")
        o.addRow("", self.reuse)
        self.b_load_chain = QPushButton("Load prompts from this folder")
        self.b_load_chain.setToolTip("Take the prompts and start image of the chain saved in this folder")
        self.b_load_chain.clicked.connect(self._load_chain_from_out)
        o.addRow("", self.b_load_chain)
        lv.addWidget(ow)

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
        lv.addWidget(aw)
        lv.addStretch(1)
        ls.setWidget(left)
        ls.setMinimumWidth(360)
        split.addWidget(ls)

        # middle: prompts
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar, bh = section_bar("Prompts", mid)
        self.total = label("", level="tertiary")
        bh.addWidget(self.total)
        bh.addSpacing(8)
        b_open = icon_button("import", "Load prompts from a text file (one per line)")
        b_open.clicked.connect(self._import_prompts)
        b_save = icon_button("save", "Save the prompts to a text file")
        b_save.clicked.connect(self._export_prompts)
        self.b_add = icon_button("plus", "Add a prompt (Ctrl+N)")
        self.b_add.clicked.connect(lambda: self.add_card("", focus=True))
        for b in (b_open, b_save, self.b_add):
            bh.addWidget(b)
        mv.addWidget(bar)
        self.cards_area = QScrollArea()
        self.cards_area.setWidgetResizable(True)
        self.cards_area.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        self.cards_box = QVBoxLayout(holder)
        self.cards_box.setContentsMargins(10, 10, 10, 10)
        self.cards_box.setSpacing(10)
        self.b_add_big = QPushButton("Add prompt")
        self.b_add_big.setIcon(icon("plus"))
        self.b_add_big.clicked.connect(lambda: self.add_card("", focus=True))
        self.cards_box.addWidget(self.b_add_big)
        self.cards_box.addStretch(1)
        self.cards_area.setWidget(holder)
        mv.addWidget(self.cards_area, 1)
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
        self.b_chain = QPushButton("Play chain")
        self.b_chain.setEnabled(False)
        self.b_chain.clicked.connect(self.play_chain)
        prow.addWidget(self.b_chain)
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
        split.setSizes([380, 640, 460])

        # bottom: run bar
        sb = self.statusBar()
        self.b_gen = QPushButton("Generate")
        self.b_gen.setProperty("variant", "primary")
        self.b_gen.setMinimumWidth(130)
        self.b_gen.clicked.connect(self.generate)
        self.b_stop = QPushButton("Stop")
        self.b_stop.setProperty("variant", "danger")
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(240)
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.status = label("Choose a start image and write the prompts.", role="statusText")
        self.gpu = label("GPU …", role="statusText")
        sb.addWidget(self.b_gen)
        sb.addWidget(self.b_stop)
        sb.addWidget(self.progress)
        sb.addWidget(self.status, 1)
        sb.addPermanentWidget(self.gpu)

        self.start_tile.clicked.connect(lambda _t: self._choose_image())
        self.start_tile.dropped.connect(self.set_image)
        self.resolution.currentIndexChanged.connect(self._update_size_info)
        self.frames.currentIndexChanged.connect(self._renumber)
        self.keep_rep.toggled.connect(self._renumber)
        self.out.editingFinished.connect(self._out_changed)
        for sig in (self.lora.editingFinished, self.strength.valueChanged, self.frames.currentIndexChanged,
                    self.steps.valueChanged, self.seed.valueChanged, self.same_seed.toggled,
                    self.resolution.currentIndexChanged, self.audio.toggled, self.reuse.toggled):
            sig.connect(self._mark_changed)
        QShortcut(QKeySequence("Ctrl+N"), self, activated=lambda: self.add_card("", focus=True))
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.generate)

    # ---- settings
    def _load_settings(self):
        q = self.qs
        self.lora.setText(q.value("lora", "", type=str))
        self.strength.setValue(q.value("strength", 1.0, type=float))
        fr = q.value("frames", 73, type=int)
        self.frames.setCurrentIndex(FRAME_CHOICES.index(fr) if fr in FRAME_CHOICES else 2)
        i = self.resolution.findData(q.value("resolution", 512, type=int))
        self.resolution.setCurrentIndex(i if i >= 0 else 1)
        self.steps.setValue(q.value("steps", 28, type=int))
        self.seed.setValue(q.value("seed", 42, type=int))
        self.same_seed.setChecked(q.value("sameSeed", False, type=bool))
        self.audio.setChecked(q.value("audio", True, type=bool))
        self.keep_rep.setChecked(q.value("keepRepeated", False, type=bool))
        self.out.setText(q.value("out", "", type=str))
        self.reuse.setChecked(q.value("reuse", True, type=bool))
        self.offload.setValue(q.value("offload", 90, type=int))
        self.min_free.setValue(q.value("minFree", 18.0, type=float))
        self.aitk.setText(q.value("aitk", DEFAULT_AITK, type=str))
        try:
            prompts = json.loads(q.value("prompts", "[]", type=str))
        except ValueError:
            prompts = []
        for p in prompts or [""]:
            self.add_card(p)
        img = q.value("image", "", type=str)
        if img and Path(img).is_file():
            self.set_image(img)
        if g := q.value("geometry"):
            self.restoreGeometry(g)
        self._out_changed()

    def save_settings(self):
        q = self.qs
        for k, v in (("lora", self.lora.text().strip()), ("strength", self.strength.value()),
                     ("frames", self.frames.currentData()), ("resolution", self.resolution.currentData()),
                     ("steps", self.steps.value()), ("seed", self.seed.value()),
                     ("sameSeed", self.same_seed.isChecked()), ("audio", self.audio.isChecked()),
                     ("keepRepeated", self.keep_rep.isChecked()), ("out", self.out.text().strip()),
                     ("reuse", self.reuse.isChecked()), ("offload", self.offload.value()),
                     ("minFree", self.min_free.value()), ("aitk", self.aitk.text().strip()),
                     ("image", self.image_file or ""),
                     ("prompts", json.dumps([c.edit.toPlainText() for c in self.cards], ensure_ascii=False))):
            q.setValue(k, v)
        q.setValue("geometry", self.saveGeometry())
        q.sync()

    # ---- start image
    image_file: str | None = None

    def _choose_image(self):
        if self.running:
            return
        start = str(Path(self.image_file).parent) if self.image_file else ""
        f, _ = QFileDialog.getOpenFileName(self, "Start image", start, IMAGE_FILTER)
        if f:
            self.set_image(f)

    def set_image(self, path: str):
        if self.running:
            return
        img = qimage(path)
        if img.isNull():
            self.status.setText(f"Not an image: {Path(path).name}")
            return
        self.image_file = str(Path(path).resolve())
        self._image_size = (img.width(), img.height())
        self.start_tile.set_image(img)
        self.image_path.setText(Path(path).name)
        self.image_path.setToolTip(self.image_file)
        self._update_size_info()
        self._mark_changed()

    def video_size(self) -> tuple[int, int] | None:
        if not self._image_size:
            return None
        return chain.canvas_size(*self._image_size, self.resolution.currentData())

    def _update_size_info(self):
        if not self._image_size:
            self.size_info.setText("")
            return
        w, h = self._image_size
        vw, vh = self.video_size()
        r = (w / h) / (vw / vh)
        note = "" if 0.97 < r < 1.03 else " (image ratio outside 2:5 … 5:2, it gets stretched)"
        self.size_info.setText(f"{w}×{h} image → {vw}×{vh} video{note}")

    # ---- cards
    def add_card(self, text: str = "", focus: bool = False, index: int | None = None) -> PromptCard:
        card = PromptCard(text)
        card.changed.connect(self._card_changed)
        card.moveUp.connect(lambda c: self._move(c, -1))
        card.moveDown.connect(lambda c: self._move(c, 1))
        card.remove.connect(self._remove)
        card.play.connect(self._play_card)
        index = len(self.cards) if index is None else index
        self.cards.insert(index, card)
        self.cards_box.insertWidget(index, card)
        self._renumber()
        if focus:
            card.edit.setFocus()
            QTimer.singleShot(0, lambda: self.cards_area.ensureWidgetVisible(card))
        return card

    def _remove(self, card: PromptCard):
        if self.running:
            return
        if card.text() and QMessageBox.question(self, "Remove prompt", f"Remove {card.title.text()}?") \
                != QMessageBox.Yes:
            return
        self.cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        if not self.cards:
            self.add_card("")
        self._renumber()
        self._mark_changed()

    def _move(self, card: PromptCard, d: int):
        i = self.cards.index(card)
        j = i + d
        if self.running or not 0 <= j < len(self.cards):
            return
        self.cards[i], self.cards[j] = self.cards[j], self.cards[i]
        self.cards_box.removeWidget(card)
        self.cards_box.insertWidget(j, card)
        self._renumber()
        self._mark_changed()

    def _renumber(self, *_):
        frames = self.frames.currentData() or 73
        drop = not self.keep_rep.isChecked()
        t = 0.0
        for k, c in enumerate(self.cards):
            n = frames - (1 if (drop and k > 0) else 0)
            c.set_number(k + 1, t, t + n / 24, k == 0, k == len(self.cards) - 1)
            t += n / 24
        filled = sum(1 for c in self.cards if c.text())
        self.total.setText(f"{filled} clip{'s' if filled != 1 else ''} · {fmt_time(t if filled else 0)}")

    def _card_changed(self):
        self._renumber()
        self._mark_changed()

    def prompts(self) -> list[str]:
        return [c.text() for c in self.cards if c.text()]

    def expected_settings(self) -> dict | None:
        """The settings record the generator will write for the current inputs (see chain.settings_dict)."""
        size = self.video_size()
        if size is None:
            return None
        lora = self.lora.text().strip()
        return chain.settings_dict(model="Comfy-Org/MiniMax-H3",
                                   lora=str(Path(lora).resolve()) if lora else None,
                                   strength=self.strength.value(), frames=self.frames.currentData(),
                                   steps=self.steps.value(), width=size[0], height=size[1],
                                   audio=self.audio.isChecked())

    def expected_seed(self, k: int) -> int:
        return self.seed.value() if self.same_seed.isChecked() else self.seed.value() + k - 1

    def _mark_changed(self, *_):
        """Compare the cards with the last chain in the output folder: which clips would be reused, which redone.
        Mirrors run_chain's rule: same prompt, seed, settings and start image (a changed clip changes all after)."""
        if self.running:
            return
        out = Path(self.out.text().strip()).expanduser() if self.out.text().strip() else None
        rec = chain.load_record(out) if out else None
        clips = {c["num"]: c for c in (rec or {}).get("clips", [])}
        still_same = bool(rec and self.image_file and rec.get("image") == self.image_file
                          and rec.get("settings") == self.expected_settings() and self.reuse.isChecked())
        for k, card in enumerate(self.cards, start=1):
            old = clips.get(k)
            if (old and out and still_same and old.get("prompt") == card.text()
                    and old.get("seed") == self.expected_seed(k) and (out / old["video"]).exists()):
                card.set_result(out / old["video"], out / old["last_frame"])
                card.set_state("done")
            else:
                still_same = False           # every clip after a changed one starts from a new frame
                if old and out and (out / old["video"]).exists():
                    card.set_result(out / old["video"], out / old["last_frame"])
                    card.set_state("changed", "will be made again")
                else:
                    card.set_result(None, None)
                    card.set_state("")
        chain_file = out / "chain.mp4" if out else None
        self.b_chain.setEnabled(bool(chain_file and chain_file.exists()))

    def _play_card(self, card: PromptCard):
        if card.video:
            self.tabs.setCurrentIndex(0)
            self.preview.open(card.video, f"{card.title.text()} · {card.video.name}")

    def play_chain(self):
        out = Path(self.out.text().strip())
        if (out / "chain.mp4").exists():
            self.tabs.setCurrentIndex(0)
            self.preview.open(out / "chain.mp4", "Whole chain · chain.mp4")

    def _import_prompts(self):
        if self.running:
            return
        f, _ = QFileDialog.getOpenFileName(self, "Load prompts", "", "Text (*.txt);;All files (*)")
        if not f:
            return
        prompts = chain.read_prompts(Path(f))
        if not prompts:
            self.status.setText("No prompts in that file.")
            return
        self.set_prompts(prompts)

    def set_prompts(self, prompts: list[str]):
        for c in list(self.cards):
            self.cards.remove(c)
            c.setParent(None)
            c.deleteLater()
        for p in prompts or [""]:
            self.add_card(p)
        self._mark_changed()

    def _export_prompts(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save prompts", "prompts.txt", "Text (*.txt)")
        if f:
            Path(f).write_text("\n".join(self.prompts()) + "\n", encoding="utf-8")
            self.status.setText(f"Saved {len(self.prompts())} prompts to {Path(f).name}.")

    # ---- output
    def _choose_out(self):
        f = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text().strip())
        if f:
            self.out.setText(f)
            self._out_changed()

    def _out_changed(self):
        out = self.out.text().strip()
        rec = chain.load_record(Path(out)) if out else None
        self.b_load_chain.setVisible(bool(rec))
        self.b_open_out.setEnabled(bool(out) and Path(out).is_dir())
        self._mark_changed()

    def _open_out(self):
        out = self.out.text().strip()
        if out and Path(out).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    def _load_chain_from_out(self):
        out = Path(self.out.text().strip())
        rec = chain.load_record(out)
        if not rec:
            return
        prompts = [c["prompt"] for c in rec.get("clips", [])]
        img = rec.get("image")
        if img and Path(img).is_file():
            self.set_image(img)
        elif (out / "start.png").exists():
            self.set_image(str(out / "start.png"))
        self.set_prompts(prompts)
        s = rec.get("settings") or {}
        if s.get("frames") in FRAME_CHOICES:
            self.frames.setCurrentIndex(FRAME_CHOICES.index(s["frames"]))
        if s.get("steps"):
            self.steps.setValue(int(s["steps"]))
        if "lora" in s:
            self.lora.setText(s.get("lora") or "")
        if s.get("strength") is not None:
            self.strength.setValue(float(s["strength"]))
        if "audio" in s:
            self.audio.setChecked(bool(s["audio"]))
        self._mark_changed()
        self.status.setText(f"Loaded {len(prompts)} prompts from {out.name}.")

    def _choose_lora(self):
        start = str(Path(self.lora.text()).parent) if self.lora.text().strip() else ""
        f, _ = QFileDialog.getOpenFileName(self, "LoRA", start, "LoRA (*.safetensors)")
        if f:
            self.lora.setText(f)

    # ---- GPU
    def _poll_gpu(self):
        if self._gpu_proc is not None or not shutil.which("nvidia-smi"):
            if not shutil.which("nvidia-smi"):
                self.gpu.setText("GPU: nvidia-smi not found")
            return
        p = QProcess(self)
        self._gpu_proc = p

        def done(*_):
            out = bytes(p.readAllStandardOutput()).decode(errors="replace").strip().splitlines()
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

    # ---- running
    @property
    def running(self) -> bool:
        return self.proc is not None

    def _args(self, prompts_file: Path, out: Path) -> list[str]:
        args = [self.image_file, str(prompts_file), "--out", str(out), "--events",
                "--frames", str(self.frames.currentData()), "--steps", str(self.steps.value()),
                "--seed", str(self.seed.value()), "--resolution", str(self.resolution.currentData()),
                "--offload", f"{self.offload.value() / 100:g}", "--min-free-gb", f"{self.min_free.value():g}"]
        if self.lora.text().strip():
            args += ["--lora", self.lora.text().strip(), "--strength", f"{self.strength.value():g}"]
        if self.same_seed.isChecked():
            args.append("--same-seed")
        if not self.audio.isChecked():
            args.append("--no-audio")
        if self.keep_rep.isChecked():
            args.append("--keep-repeated-frames")
        if not self.reuse.isChecked():
            args.append("--redo")
        return args

    def generate(self):
        if self.running:
            return
        prompts = self.prompts()
        out_text = self.out.text().strip()
        problems = []
        if not self.image_file or not Path(self.image_file).is_file():
            problems.append("choose a start image")
        if not prompts:
            problems.append("write at least one prompt")
        if not out_text:
            problems.append("choose an output folder")
        lora = self.lora.text().strip()
        if lora and not Path(lora).is_file():
            problems.append(f"the LoRA file does not exist ({Path(lora).name})")
        if problems:
            self.status.setText("To start: " + ", ".join(problems) + ".")
            return
        if len(prompts) != len(self.cards):        # drop empty cards so card k = clip k
            self.set_prompts(prompts)
        out = Path(out_text).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        prompts_file = out / "prompts.txt"
        prompts_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        self.save_settings()

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
        self._buf, self._ebuf, self._error, self._stopping = "", "", "", False
        self._run_out, self._clips_total, self._phase = out, len(prompts), "starting"
        self.log.clear()
        self.preview.stop()
        for c in self.cards:
            c.set_state("waiting")
            c.set_editable(False)
        self._set_running(True)
        self.progress.setRange(0, 0)
        self.status.setText("Starting …")
        args = pre + self._args(prompts_file, out)
        self.log.appendPlainText("$ " + " ".join([program, *args]))
        self.proc.start(program, args)

    def _program(self) -> tuple[str, list[str]]:
        if self.command:
            return self.command[0], list(self.command[1:])
        return str(LAUNCHER), []

    def _set_running(self, on: bool):
        self.b_gen.setEnabled(not on)
        self.b_stop.setEnabled(on)
        for w in (self.start_tile, self.lora, self.strength, self.frames, self.resolution, self.steps, self.seed,
                  self.same_seed, self.audio, self.keep_rep, self.out, self.reuse, self.offload, self.min_free,
                  self.aitk, self.b_add, self.b_add_big, self.b_load_chain):
            w.setEnabled(not on)

    def stop(self):
        if not self.running:
            return
        self._stopping = True
        self.status.setText("Stopping …")
        self.proc.terminate()
        p = self.proc
        QTimer.singleShot(8000, lambda: p.state() != QProcess.NotRunning and p.kill())

    def _read_out(self):
        self._buf += bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._line(line)

    def _read_err(self):
        self._ebuf += bytes(self.proc.readAllStandardError()).decode(errors="replace")
        *lines, self._ebuf = self._ebuf.split("\n")
        for line in lines:
            line = line.split("\r")[-1].rstrip()
            if line:
                self.log.appendPlainText(line)

    def _line(self, line: str):
        line = line.split("\r")[-1].rstrip()
        if line.startswith("@@ "):
            try:
                self._event(json.loads(line[3:]))
                return
            except ValueError:
                pass
        if line:
            self.log.appendPlainText(line)

    def _card(self, k: int) -> PromptCard | None:
        return self.cards[k - 1] if 1 <= k <= len(self.cards) else None

    def _overall(self, k: int, frac: float):
        if self._clips_total:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * ((k - 1) + frac) / self._clips_total))

    def _event(self, ev: dict):
        e = ev.get("e")
        if e == "start":
            self.status.setText(f"{ev['clips']} clips at {ev['width']}×{ev['height']}, {ev['frames']} frames each.")
        elif e == "loading":
            self._phase = "loading"
            self.status.setText("Loading MiniMax-H3 … (a few minutes)")
            self.progress.setRange(0, 0)
        elif e == "loaded":
            self.status.setText("Model loaded.")
        elif e == "clip":
            k, card = ev["k"], self._card(ev["k"])
            if card is None:
                return
            st = ev["state"]
            if st == "generating":
                card.set_state("generating", "starting …")
                self._overall(k, 0)
                self.status.setText(f"Clip {k} of {ev['n']} …")
                self.cards_area.ensureWidgetVisible(card)
            else:
                card.set_state(st)
                card.set_result(Path(ev["video"]), Path(ev["frame"]))
                self._overall(k, 1)
        elif e == "step":
            card = self._card(ev.get("k", 0))
            if card is not None and ev.get("n"):
                card.set_progress(ev["i"], ev["n"])
                self._overall(ev["k"], ev["i"] / ev["n"])
                self.status.setText(f"Clip {ev['k']} of {self._clips_total}: step {ev['i']} / {ev['n']}")
        elif e == "joining":
            self.status.setText("Joining the clips …")
        elif e == "done":
            self._phase = "done"
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
        elif e == "error":
            self._error = ev.get("msg", "error")

    def _proc_error(self, err):
        if err == QProcess.FailedToStart:
            self._error = f"could not start {self._program()[0]}"

    def _finished(self, code: int, _status):
        p, self.proc = self.proc, None
        if p is not None:
            rest = bytes(p.readAllStandardOutput()).decode(errors="replace")
            for line in (self._buf + rest).split("\n"):
                self._line(line)
            p.deleteLater()
        self._buf = ""
        self._set_running(False)
        for c in self.cards:
            c.set_editable(True)
            if c.state == "generating":
                c.set_state("stopped" if self._stopping else "failed")
            elif c.state == "waiting":
                c.set_state("")
        self.progress.setRange(0, 1000)
        out = self._run_out
        if self._phase == "done" and code == 0:
            secs = chain.probe(out / "chain.mp4")["frames"] / 24 if (out / "chain.mp4").exists() else 0
            self.status.setText(f"Done: {len(self.cards)} clips, chain.mp4 ({fmt_time(secs)}).")
            self.b_chain.setEnabled(True)
            self.play_chain()
        elif self._stopping:
            self.progress.setValue(0)
            self.status.setText("Stopped. Finished clips are kept and reused next time.")
        else:
            self.progress.setValue(0)
            msg = self._error or f"the generator exited with code {code}"
            self.status.setText(f"Failed: {msg}")
            self.tabs.setCurrentWidget(self.log)
        self._stopping = False

    def closeEvent(self, e):
        if self.running:
            if QMessageBox.question(self, "Promptwood", "A chain is being generated. Stop it and quit?") \
                    != QMessageBox.Yes:
                e.ignore()
                return
            self.stop()
            if self.proc is not None:
                self.proc.waitForFinished(10000)
        self.preview.stop()
        self.save_settings()
        super().closeEvent(e)
