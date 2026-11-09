"""Dialogs for made clips: choose the frame a clip hands over at (HandoverDialog), and play sibling clips side by
side to pick one (CompareDialog)."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
                               QPushButton, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from .. import chain
from .theme import icon
from .widgets import FrameTile, VideoView, icon_button, label

FPS = chain.FPS
PREVIEW_W = 480                    # frames are decoded once at this width, so scrubbing is instant


def fmt_s(frame: int) -> str:
    return f"{frame / FPS:.2f} s"


class _Decoded(QObject):
    done = Signal(object, str)        # list of QImage, error


def decode_frames(video: Path, width: int = PREVIEW_W) -> list[QImage]:
    """Every frame of a (short) clip as a small QImage, in order."""
    info = chain.probe(video)
    w = max(2, min(width, info["width"]) // 2 * 2)
    h = max(2, round(info["height"] * w / info["width"] / 2) * 2)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf", f"scale={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg failed")
    size = w * h * 3
    return [QImage(r.stdout[i:i + size], w, h, 3 * w, QImage.Format_RGB888).copy()
            for i in range(0, len(r.stdout) - size + 1, size)]


class HandoverDialog(QDialog):
    """Pick the frame of a made clip that the clips after it start from. frame() is the choice (None: the last)."""

    def __init__(self, parent, video: Path, title: str, current: int | None = None, min_frame: int = 1):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 640)
        self.frames: list[QImage] = []
        self.min_frame = max(1, int(min_frame))
        self._current = current
        v = QVBoxLayout(self)
        info = label("The clips after this one start from the frame you choose here; the frames after it are left "
                     "out when the path is played or exported. This clip stays as it is (nothing is generated again "
                     "for it). Use it when the last frame is blurred, mid-blink or has drifted.", level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        self.tile = FrameTile(720, 405)
        self.tile.clear_frame("reading the frames …")
        v.addWidget(self.tile, 1, Qt.AlignHCenter)
        row = QHBoxLayout()
        self.b_prev = icon_button("skipb", "One frame back (Left)")
        self.b_next = icon_button("skipf", "One frame on (Right)")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.where = label("", level="secondary", type_="mono")
        self.b_prev.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.b_next.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        row.addWidget(self.b_prev)
        row.addWidget(self.slider, 1)
        row.addWidget(self.b_next)
        row.addWidget(self.where)
        v.addLayout(row)
        self.summary = label("", level="tertiary")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Hand over here")
        self.b_last = self.buttons.addButton("Use the last frame", QDialogButtonBox.ResetRole)
        self.b_last.setToolTip("Back to normal: the clips after it start from its last frame")
        self.b_last.clicked.connect(lambda: self.slider.setValue(self.slider.maximum()))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        v.addWidget(self.buttons)
        for key, k in (("Left", -1), ("Right", 1)):
            QShortcut(QKeySequence(key), self, activated=lambda k=k: self.slider.setValue(self.slider.value() + k))
        self.slider.valueChanged.connect(self._show)
        self._sig = _Decoded()
        self._sig.done.connect(self._decoded)
        sig = self._sig

        def work():
            try:
                frames, err = decode_frames(Path(video)), ""
            except Exception as e:  # noqa: BLE001
                frames, err = [], str(e)
            try:
                sig.done.emit(frames, err)
            except RuntimeError:                       # the dialog is gone
                pass
        threading.Thread(target=work, daemon=True).start()

    def _decoded(self, frames: list, error: str):
        if error or not frames:
            self.tile.clear_frame(f"Could not read the clip: {error or 'no frames'}")
            return
        self.frames = frames
        n = len(frames)
        self.slider.setRange(min(self.min_frame, n - 1), n - 1)
        self.slider.setEnabled(True)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        start = n - 1 if self._current is None else min(int(self._current), n - 1)
        self.slider.setValue(start)
        self._show(self.slider.value())

    def _show(self, k: int):
        if not self.frames:
            return
        n = len(self.frames)
        self.tile.set_image(self.frames[k])
        self.where.setText(f"frame {k + 1} / {n} · {fmt_s(k)}")
        if k >= n - 1:
            self.summary.setText("Last frame: the clips after it start where the clip ends (as usual).")
        else:
            self.summary.setText(f"The last {n - 1 - k} frame{'s' if n - 1 - k != 1 else ''} "
                                 f"({(n - 1 - k) / FPS:.2f} s) are left out; the clips after it start from this one "
                                 "and are made again.")

    def done(self, r):
        choice = self.frame()
        self.frames = []                              # let go of the decoded frames (tens of MB)
        self._choice = choice
        super().done(r)

    def frame(self) -> int | None:
        """The chosen frame, None for the last one."""
        if not self.frames and hasattr(self, "_choice"):
            return self._choice
        if not self.frames or self.slider.value() >= len(self.frames) - 1:
            return None
        return int(self.slider.value())


class CompareDialog(QDialog):
    """Sibling clips (variants) side by side, played together; 'Keep this one' deletes the others.
    items: [(node id, title, detail, video file)]; on_keep(nid) -> bool (True: kept, close)."""

    def __init__(self, parent, items: list[tuple[str, str, str, Path]], on_keep, player: bool = True,
                 title: str = "Compare"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1280, 820)
        self.items, self.on_keep = items, on_keep
        self.players: list = []
        self.kept: str | None = None
        v = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(10)
        cols = 2 if len(items) <= 4 else 3
        self.keep_buttons: dict[str, QPushButton] = {}
        self.audio_outs: list = []
        has_player = False
        if player:
            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
                has_player = True
            except ImportError:
                has_player = False
        for k, (nid, name, detail, video) in enumerate(items):
            box = QWidget()
            bv = QVBoxLayout(box)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.setSpacing(4)
            if has_player:
                view = VideoView()                 # (painted frames: no native GPU window, see VideoView)
                view.setMinimumSize(360, 220)
                out = QAudioOutput(self)
                out.setMuted(k != 0)
                p = QMediaPlayer(self)
                p.setAudioOutput(out)
                p.setVideoOutput(view.sink)
                p.setSource(QUrl.fromLocalFile(str(video)))
                p.mediaStatusChanged.connect(lambda st, p=p: self._ended(p, st))
                self.players.append(p)
                self.audio_outs.append(out)
                bv.addWidget(view, 1)
            else:
                stack = QStackedWidget()
                stack.addWidget(label(Path(video).name, level="tertiary"))
                bv.addWidget(stack, 1)
            head = QHBoxLayout()
            head.addWidget(label(name, type_="heading"))
            d = label(detail, level="tertiary")
            d.setWordWrap(True)
            head.addWidget(d, 1)
            keep = QPushButton("Keep this one")
            keep.setToolTip("Keep this clip and what follows it; delete the other clips shown here (with "
                            "everything after them)")
            keep.clicked.connect(lambda _=False, nid=nid: self._keep(nid))
            self.keep_buttons[nid] = keep
            head.addWidget(keep)
            bv.addLayout(head)
            grid.addWidget(box, k // cols, k % cols)
        v.addLayout(grid, 1)
        ctl = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause all (Space)")
        self.b_play.clicked.connect(self.toggle)
        self.b_restart = QPushButton("From the start")
        self.b_restart.clicked.connect(self.restart)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.seek)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        self.sound = QComboBox()
        for nid, name, _d, _v in items:
            self.sound.addItem(f"Sound: {name}", nid)
        self.sound.addItem("No sound", None)
        self.sound.currentIndexChanged.connect(self._sound)
        for w in (self.b_play, self.b_restart):
            ctl.addWidget(w)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.loop)
        ctl.addWidget(self.sound)
        v.addLayout(ctl)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        v.addWidget(close)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle)
        for w in (self.b_play, self.b_restart, self.slider, self.loop, self.sound):
            w.setEnabled(bool(self.players))
        if self.players:
            for p in self.players:                   # the longest clip sets the slider's range
                p.durationChanged.connect(lambda _d: self.slider.setRange(
                    0, max(int(q.duration()) for q in self.players)))
            self.players[0].positionChanged.connect(
                lambda ms: None if self.slider.isSliderDown() else self.slider.setValue(int(ms)))
            QTimer.singleShot(200, self.restart)

    def _playing(self) -> bool:
        from PySide6.QtMultimedia import QMediaPlayer
        return any(p.playbackState() == QMediaPlayer.PlayingState for p in self.players)

    def toggle(self):
        if not self.players:
            return
        playing = self._playing()
        for p in self.players:
            p.pause() if playing else p.play()
        self.b_play.setIcon(icon("play" if playing else "pause"))

    def restart(self):
        for p in self.players:
            p.setPosition(0)
            p.play()
        if self.players:
            self.b_play.setIcon(icon("pause"))

    def seek(self, ms: int):
        for p in self.players:
            p.setPosition(int(ms))

    def _ended(self, p, status):
        from PySide6.QtMultimedia import QMediaPlayer
        over = (QMediaPlayer.EndOfMedia, QMediaPlayer.InvalidMedia, QMediaPlayer.NoMedia)
        if status not in over[:2] or not self.loop.isChecked():
            return
        # the clips may differ in length: start all again once the longest one has ended (one that cannot be
        # played does not hold the others up)
        if all(q.mediaStatus() in over or q is p for q in self.players):
            self.restart()

    def _sound(self, *_):
        nid = self.sound.currentData()
        for (inid, *_r), out in zip(self.items, self.audio_outs):
            out.setMuted(inid != nid)

    def _keep(self, nid: str):
        for p in self.players:                       # the files of the others are deleted: let go of them
            p.stop()
            p.setSource(QUrl())
        if self.on_keep(nid):
            self.kept = nid
            self.accept()
        else:
            for p, (_n, _t, _d, video) in zip(self.players, self.items):
                p.setSource(QUrl.fromLocalFile(str(video)))
            self.restart()

    def done(self, r):
        for p in self.players:
            p.stop()
            p.setSource(QUrl())
        super().done(r)
