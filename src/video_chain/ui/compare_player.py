"""A frame-exact player for two clips of the same length: the original and the result side by side under a
draggable before / after divider (left of it the original, right of it the result).

Both clips are decoded once, frame by frame (ffmpeg, the clip's own frames: no frame is dropped or doubled), into
memory at the size they are shown at; stepping and playing then only paint stored frames, so every frame can be
inspected exactly. The stored size keeps the memory below MEMORY_BUDGET."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from . import shortcuts as S
from .workers import post, run_in_background
from .. import chain, media
from .theme import icon, theme
from .widgets import FrameStepper, icon_button, label, shortcut

MEMORY_BUDGET = 900 * 2**20               # bytes for the stored frames of both clips (RGB, 3 bytes a pixel)
MAX_SIZE = (1280, 720)                    # never stored larger than this (the view is smaller)


def store_size(width: int, height: int, frames: int, clips: int = 2) -> tuple[int, int]:
    """The size frames are stored at: the clip's own size, made smaller to fit MAX_SIZE and MEMORY_BUDGET (even)."""
    k = min(1.0, MAX_SIZE[0] / max(1, width), MAX_SIZE[1] / max(1, height))
    per = MEMORY_BUDGET / max(1, frames * clips * 3)               # pixels a frame may have
    k = min(k, (per / max(1, width * height)) ** 0.5)
    return max(2, int(width * k) // 2 * 2), max(2, int(height * k) // 2 * 2)


def decode(clip: Path, size: tuple[int, int], progress=None, cancelled=None) -> list[bytes]:
    """Every frame of the clip as RGB bytes at size (in the clip's own frame order and count)."""
    frames: list[bytes] = []
    for buf in media.iter_frames(clip, size, "rgb24", flags="area", passthrough=True, cancelled=cancelled):
        frames.append(buf)
        if progress is not None:
            progress(len(frames))
    return frames


class CompareView(QWidget):
    """Paints a frame (before), or two (before | after) split at a divider the mouse moves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.before = QImage()
        self.after: QImage | None = None
        self.split = 0.5                                 # the divider, as a share of the picture's width
        self.text = ""
        self.setMouseTracking(True)

    def sizeHint(self):
        return QSize(640, 360)

    def show_frames(self, before: QImage, after: QImage | None = None) -> None:
        self.before, self.after, self.text = before, after, ""
        self.setCursor(Qt.CursorShape.SplitHCursor if after is not None else Qt.CursorShape.ArrowCursor)
        self.update()

    def show_text(self, text: str) -> None:
        self.before, self.after, self.text = QImage(), None, text
        self.update()

    def picture_rect(self) -> QRectF:
        img = self.before
        if img.isNull():
            return QRectF()
        k = min(self.width() / img.width(), self.height() / img.height())
        w, h = img.width() * k, img.height() * k
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), theme().bgSunken)
        if self.before.isNull():
            p.setPen(theme().textTertiary)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.text)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = self.picture_rect()
        if self.after is None or self.after.isNull():
            p.drawImage(r, self.before)
            return
        x = r.left() + r.width() * self.split
        iw = self.before.width()
        cut = self.split * iw                            # the divider in the stored frame's pixels
        p.drawImage(QRectF(r.left(), r.top(), x - r.left(), r.height()), self.before,
                    QRectF(0, 0, cut, self.before.height()))
        p.drawImage(QRectF(x, r.top(), r.right() - x, r.height()), self.after,
                    QRectF(cut * self.after.width() / iw, 0, self.after.width() * (1 - self.split),
                           self.after.height()))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(theme().overlayInk, 2))
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.setBrush(theme().overlayInk)
        p.drawEllipse(QPointF(x, r.center().y()), 7, 7)
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        for text, left in (("Before", True), ("After", False)):
            w = p.fontMetrics().horizontalAdvance(text) + 12
            box = QRectF(r.left() + 8 if left else r.right() - 8 - w, r.top() + 8, w, 20)
            if (left and box.right() > x - 4) or (not left and box.left() < x + 4):
                continue                                 # (no label over the divider)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme().hudBg)
            p.drawRoundedRect(box, 4, 4)
            p.setPen(theme().overlayInk)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _move_split(self, x: float):
        r = self.picture_rect()
        if self.after is None or r.width() <= 0:
            return
        self.split = min(1.0, max(0.0, (x - r.left()) / r.width()))
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._move_split(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._move_split(e.position().x())


class ComparePlayer(QWidget):
    """The original clip (and the result, once there is one) with play / pause, a frame and a second back and on,
    a frame slider and the before / after divider. overlay(k, image) may draw onto the original's frame k (e.g. the
    head area; the image is at the stored size, scale() maps clip pixels to it)."""

    frameChanged = Signal(int)

    def __init__(self, parent=None, fps: float = chain.FPS):
        super().__init__(parent)
        self.fps = fps
        self.overlay: Callable[[int, QImage], Any] | None = None
        self.size_ = (0, 0)                              # the stored frame size
        self.clip_size = (0, 0)
        self.frames: list[bytes] = []
        self.result: list[bytes] | None = None
        self._gen = [0, 0]                               # the current load of the clip / the result
        self._cancel = threading.Event()                 # stops the clip's decode (and the result's, with the clip)
        self._result_cancel = threading.Event()          # stops the running decode of the result
        self._events = [self._cancel, self._result_cancel]
        events = self._events                            # (a list, so the destroyed hook sees the current ones)
        def stop_loading(*_) -> None:
            for e in events:
                e.set()
        self.destroyed.connect(stop_loading)
        self._t0 = 0.0                                   # when playing started (monotonic) and at which frame
        self._k0 = 0
        self._shown_k = -1
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.view = CompareView()
        v.addWidget(self.view, 1)
        self.stepper = FrameStepper(self, fps, context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.stepper.stepped.connect(self.pause)         # (a step by hand stops the playing)
        self.slider, self.time = self.stepper.slider, self.stepper.label
        self.b_back, self.b_prev = self.stepper.b_back_s, self.stepper.b_back1       # (second / frame back and on)
        self.b_next, self.b_fwd = self.stepper.b_fwd1, self.stepper.b_fwd_s
        self.slider.setEnabled(False)
        self.stepper.valueChanged.connect(self._show)
        v.addWidget(self.stepper)
        row = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause (Space)")
        self.b_play.clicked.connect(self.toggle)
        row.addWidget(self.b_play)
        row.addSpacing(8)
        self.hint = label("", level="tertiary")
        row.addWidget(self.hint, 1)
        v.addLayout(row)
        self._controls(False)
        first, last = S.sequences("compare_first_last")
        for key, fn in ((S.sequence("play"), self.toggle), (first, lambda: self.seek(0)),
                        (last, lambda: self.seek(self.count() - 1))):
            shortcut(key, self, fn, context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _controls(self, on: bool):
        self.b_play.setEnabled(on)
        self.stepper.setEnabled(on and self.count() > 1)

    # ---- loading
    def load(self, clip: Path, frames: int, size: tuple[int, int]) -> None:
        """The original clip (frames: its frame count, size: its pixel size)."""
        self.pause()
        self._new_events(clip=True)
        self.frames, self.result = [], None
        self.clip_size = (int(size[0]), int(size[1]))
        self.size_ = store_size(size[0], size[1], frames)
        self._controls(False)
        self.view.show_text("Reading the clip …")
        self._decode(Path(clip), 0)

    def load_result(self, clip: Path) -> None:
        """The result (the same frames as the original): shown right of the divider."""
        self.pause()
        self.result = None
        self._new_events(clip=False)                     # an older result decode still running is stopped
        self.hint.setText("Reading the result …")
        self._decode(Path(clip), 1)

    def _new_events(self, clip: bool) -> None:
        """Stop the running decode(s) this load replaces: the clip's (and with it the result's), or the result's."""
        for ev in ((self._cancel, self._result_cancel) if clip else (self._result_cancel,)):
            ev.set()
        if clip:
            self._cancel = threading.Event()
        self._result_cancel = threading.Event()
        self._events[:] = [self._cancel, self._result_cancel]

    def _decode(self, clip: Path, which: int):
        self._gen[which] += 1
        if not which:
            self._gen[1] += 1                            # a result of the clip before is not shown
        tid, size = self._gen[which], self.size_
        own = self._result_cancel if which else self._cancel
        parent = self._cancel

        def cancel() -> bool:                            # (a result also stops with its clip)
            return own.is_set() or parent.is_set()

        def progress(k: int) -> None:
            if k % 12 == 0:
                post(self, lambda: self._progress(tid, which, k))

        def work():
            fr = decode(clip, size, progress, cancel)
            return None if cancel() else fr             # (a cancelled decode answers nothing)

        def done(fr):
            if fr is not None:
                self._loaded(tid, fr, "", which)
        run_in_background(work, on_done=done, on_error=lambda e: self._loaded(tid, None, str(e), which), owner=self)

    def _progress(self, tid: int, which: int, k: int):
        if self._gen[which] != tid:
            return
        if which:
            self.hint.setText(f"Reading the result: frame {k}")
        elif not self.frames:
            self.view.show_text(f"Reading the clip: frame {k}")

    def _loaded(self, tid: int, frames, error: str, which: int):
        if self._gen[which] != tid:
            return                                      # an older clip
        if frames is None or not frames:
            if which:
                self.hint.setText(f"Cannot show the result: {error or 'no frames'}")
            else:
                self.view.show_text(f"Cannot show the clip: {error or 'no frames'}")
            return
        if which:
            self.result = frames
            n = len(self.frames)
            self.hint.setText("Drag over the picture to move the divider" + (
                f" (the result has {len(frames)} frames, the clip {n})" if len(frames) != n else ""))
        else:
            self.frames = frames
            self.slider.blockSignals(True)
            self.stepper.set_range(0, max(0, len(frames) - 1))
            self.slider.setValue(min(self.slider.value(), len(frames) - 1))
            self.slider.blockSignals(False)
            self._controls(True)
        self._show(self.slider.value())

    # ---- frames
    def count(self) -> int:
        return len(self.frames)

    def scale(self) -> float:
        """Clip pixels -> stored pixels."""
        return self.size_[0] / max(1, self.clip_size[0])

    def image(self, k: int, after: bool = False) -> QImage:
        src = self.result if after else self.frames
        if src is None or not 0 <= k < len(src):
            return QImage()
        w, h = self.size_
        return QImage(src[k], w, h, w * 3, QImage.Format.Format_RGB888).copy()

    def position(self) -> int:
        return self.slider.value()

    def _show(self, k: int):
        if not self.frames:
            return
        before = self.image(k)
        if self.overlay is not None:
            before = before.convertToFormat(QImage.Format.Format_ARGB32)
            self.overlay(k, before)
        after = self.image(k, True) if self.result is not None else None
        self.view.show_frames(before, after if after is not None and not after.isNull() else None)
        self.frameChanged.emit(k)

    def refresh(self):
        self._show(self.slider.value())

    def seek(self, k: int):
        if self.frames:
            self.slider.setValue(max(0, min(self.count() - 1, int(k))))

    def step(self, d: int):
        self.pause()
        self.seek(self.position() + d)

    # ---- playing
    def playing(self) -> bool:
        return self.timer.isActive()

    def play(self, from_start: bool = False):
        if not self.frames:
            return
        if from_start or self.position() >= self.count() - 1:
            self.seek(0)
        self._t0, self._k0 = time.monotonic(), self.position()
        self._shown_k = self.position()
        self.timer.start(max(1, int(1000 / self.fps)))
        self.b_play.setIcon(icon("pause"))

    def pause(self):
        self.timer.stop()
        self.b_play.setIcon(icon("play"))

    def toggle(self):
        self.pause() if self.playing() else self.play()

    def _tick(self):
        """The frame the clock says (so a slow tick catches up instead of playing slower)."""
        if self.position() != self._shown_k:             # moved by hand meanwhile: carry on from there
            self._t0, self._k0 = time.monotonic(), self.position()
        k = self._k0 + int((time.monotonic() - self._t0) * self.fps)
        last = self.count() - 1
        if self.position() >= last or k >= last:
            self.seek(last)
            self._shown_k = self.position()
            self.pause()
            return
        self.seek(max(k, self.position()))
        self._shown_k = self.position()

    def stop_loading(self):
        self.pause()
        self._cancel.set()
