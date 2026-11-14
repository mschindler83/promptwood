"""Dialogs for made clips: choose the frame a clip hands over at (HandoverDialog), and play sibling clips side by
side to pick one (CompareDialog)."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from .. import chain, media
from .theme import icon
from .widgets import FrameTile, VideoView, icon_button, label

FPS = chain.FPS
PREVIEW_W = 480                    # frames are decoded once at this width, so scrubbing is instant


def fmt_s(frame: int) -> str:
    return f"{frame / FPS:.2f} s"


class _Decoded(QObject):
    done = Signal(object, str)        # list of QImage, error


def decode_frames(video: Path, width: int = PREVIEW_W, on_start=None) -> list[QImage]:
    """Every frame of a (short) clip as a small QImage, in order. on_start(process) is given the ffmpeg process
    as soon as it runs (so a closing dialog can end it)."""
    info = chain.probe(video)
    w = max(2, min(width, info["width"]) // 2 * 2)
    h = max(2, round(info["height"] * w / info["width"] / 2) * 2)
    return [QImage(raw, w, h, 3 * w, QImage.Format_RGB888).copy()
            for raw in media.iter_frames(video, (w, h), "rgb24", flags=None, on_start=on_start, timeout=600)]


class HandoverDialog(QDialog):
    """Pick the frame of a made clip that the clips after it start from. frame() is the choice (None: the last)."""

    def __init__(self, parent, video: Path, title: str, current: int | None = None, min_frame: int = 1):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 640)
        self.frames: list[QImage] = []
        self.min_frame = max(1, int(min_frame))
        self._current = current
        self._proc = None                              # the ffmpeg that reads the frames (ended when closing)
        self._choice: int | None = None
        self._decided = False
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
                frames, err = decode_frames(Path(video), on_start=lambda pr: setattr(self, "_proc", pr)), ""
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
        self._choice, self._decided = choice, True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()                    # (a long clip is not decoded on for nobody)
        super().done(r)

    def frame(self) -> int | None:
        """The chosen frame, None for the last one."""
        if not self.frames and self._decided:
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
                bv.addWidget(label(Path(video).name, level="tertiary"), 1)
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
        self._syncing = False                         # (the slider follows the players: that is not a seek)
        self.slider.valueChanged.connect(self._slider_changed)
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
                self._follow)
            QTimer.singleShot(200, self.restart)

    def _follow(self, ms: int):
        if not self.slider.isSliderDown():
            self._syncing = True
            try:
                self.slider.setValue(int(ms))
            finally:
                self._syncing = False

    def _slider_changed(self, ms: int):
        """The user moved the slider (drag, click in the groove, keys): all players go there."""
        if not self._syncing:
            self.seek(ms)

    def _playing(self) -> bool:
        from PySide6.QtMultimedia import QMediaPlayer
        return any(p.playbackState() == QMediaPlayer.PlayingState for p in self.players)

    def toggle(self):
        if not self.players:
            return
        playing = self._playing()
        for p in self.players:
            if playing:
                p.pause()
            else:
                p.play()
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
        kept = False
        try:
            kept = bool(self.on_keep(nid))
        finally:
            if not kept:                             # (not kept, or it failed: the clips can be played again)
                for p, (_n, _t, _d, video) in zip(self.players, self.items):
                    p.setSource(QUrl.fromLocalFile(str(video)))
                self.restart()
        if kept:
            self.kept = nid
            self.accept()

    def done(self, r):
        for p in self.players:
            p.stop()
            p.setSource(QUrl())
        super().done(r)


class SplitDialog(QDialog):
    """Choose where to split a real video and insert a generated clip: the video (as converted for the chain)
    frame by frame; at() is the moment in the video's file (start + frame / 24 s)."""

    def __init__(self, parent, clip: Path, frames: int, size: tuple[int, int], start: float, end: float,
                 title: str):
        from .compare_player import ComparePlayer
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 640)
        self.start, self.end, self.n = float(start), float(end), int(frames)
        v = QVBoxLayout(self)
        info = label("The video ends at the frame you choose; a new clip follows it (it starts from that frame), "
                     "then the rest of the video, whose first frame the clip ends at. Clips after the video come "
                     "after its rest.", level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        self.player = ComparePlayer(self)
        self.player.load(Path(clip), self.n, size)
        self.player.frameChanged.connect(self._show)
        v.addWidget(self.player, 1)
        self.summary = label("", level="tertiary")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Split here and insert a clip")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

    def at(self) -> float:
        """The moment in the video's file the chosen frame shows."""
        return round(self.start + self.player.position() / chain.FPS, 4)

    def _show(self, k: int):
        ok = 0 < k < self.n - 1
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        t = self.at()
        self.summary.setText(
            f"Part 1: {self.start:.2f}–{t:.2f} s · the new clip · part 2: {t:.2f}–{self.end:.2f} s" if ok else
            "Choose a frame inside the video (not its first or last).")

    def done(self, r):
        self.player.stop_loading()
        super().done(r)


def grab_frame(video: Path, at: float, out: Path, width: int | None = None) -> bool:
    """The frame of a video shown at `at` seconds as a PNG (full size, or scaled to width), accurately seeked
    (ffmpeg decodes from the keyframe before it). False if there is none."""
    vf = ["-vf", f"scale={int(width)}:-2:flags=lanczos"] if width else []
    try:
        media.ffmpeg(["-y", "-ss", f"{max(0.0, at):.6f}", "-i", video, "-frames:v", "1", *vf, out], timeout=60)
    except (OSError, media.MediaError):
        return False
    return Path(out).is_file() and Path(out).stat().st_size > 0


class _Grabbed(QObject):
    ready = Signal(int, QImage)                 # request number, frame


class VideoFrameDialog(QDialog):
    """Choose one frame of any video (any length: only the frame shown is decoded): a slider over the whole
    video, one frame / one second back and on. frame_file() writes the chosen frame at the video's full size."""

    def __init__(self, parent, video: Path, title: str, info: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 640)
        self.video = Path(video)
        info = info or chain.probe_source(self.video)
        self.fps = float(info.get("fps") or FPS)
        self.duration = float(info.get("duration") or 0.0)
        self.frames = max(1, int(round(self.duration * self.fps)))
        if int(info.get("frames") or 0) > 0:           # the video stream's own count (a longer sound track
            self.frames = min(self.frames, int(info["frames"]))   # must not add positions without a frame)
        self._req = 0
        self._closed = False
        self._grabbing = self._again = False        # one ffmpeg at a time; the newest position after it
        self._sig = _Grabbed()
        self._sig.ready.connect(self._ready)
        self._tmp = media.scratch("frame")
        v = QVBoxLayout(self)
        self.tile = FrameTile(800, 450)
        self.tile.clear_frame("reading the frame …")
        v.addWidget(self.tile, 1, Qt.AlignHCenter)
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.frames - 1)
        buttons = []
        for text, tip, step in (("«", "One second back (Shift+Left)", -round(self.fps)),
                                ("‹", "One frame back (Left)", -1), ("›", "One frame on (Right)", 1),
                                ("»", "One second on (Shift+Right)", round(self.fps))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedWidth(34)
            b.clicked.connect(lambda _c=False, s=step: self.slider.setValue(self.slider.value() + s))
            buttons.append(b)
        for b in buttons[:2]:
            row.addWidget(b)
        row.addWidget(self.slider, 1)
        for b in buttons[2:]:
            row.addWidget(b)
        self.where = label("", level="secondary", type_="mono")
        row.addWidget(self.where)
        v.addLayout(row)
        self.note = label(f"{self.video.name} · {info.get('width', '?')}×{info.get('height', '?')} · "
                          f"{self.duration:.1f} s · {self.fps:.3g} fps", level="tertiary")
        v.addWidget(self.note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Use this frame")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        for key, step in (("Left", -1), ("Right", 1), ("Shift+Left", -round(self.fps)),
                          ("Shift+Right", round(self.fps))):
            QShortcut(QKeySequence(key), self, activated=lambda s=step: self.slider.setValue(self.slider.value() + s))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)                 # (dragging the slider: only the frame it stops at is read)
        self._timer.timeout.connect(self._grab)
        self.slider.valueChanged.connect(self._moved)
        self._moved(0)                               # (the first frame: read when the timer fires)

    def at(self) -> float:
        """The chosen frame's time in the video (its own frame rate)."""
        return self.slider.value() / self.fps

    def _seek(self) -> float:
        """Where to seek for exactly the chosen frame: a quarter frame before it (ffmpeg gives the first frame at
        or after the time; exactly k / fps can round to just after frame k and give the next one)."""
        return max(0.0, (self.slider.value() - 0.25) / self.fps)

    def _moved(self, k: int):
        self.where.setText(f"frame {k + 1} / {self.frames} · {k / self.fps:.2f} s")
        self._timer.start()

    def _grab(self):
        if self._grabbing:                           # a decode is running: the newest position comes after it
            self._again = True
            return
        self._grabbing = True
        self._req += 1
        req, at, sig = self._req, self._seek(), self._sig
        out = Path(self._tmp.name) / f"p{req}.png"

        def work():
            img = QImage()
            try:
                if grab_frame(self.video, at, out, width=960):
                    img = QImage(str(out))
            finally:                                # (always answered: a stuck "grabbing" would freeze the dialog)
                try:
                    sig.ready.emit(req, img)
                except RuntimeError:                # (the dialog is gone)
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _ready(self, req: int, img: QImage):
        self._grabbing = False
        if self._closed:
            return
        if self._again:                              # moved on meanwhile: read the position it is at now
            self._again = False
            self._grab()
            return
        if req != self._req:
            return                                  # an older position
        if img.isNull():
            self.tile.clear_frame("no frame here")
        else:
            self.tile.set_image(img)

    def frame_file(self, out: Path) -> bool:
        """The chosen frame at the video's full size into out (PNG)."""
        return grab_frame(self.video, self._seek(), out)

    def done(self, r):
        self._closed = True                         # (no further grabs after it closed)
        self._req += 1
        super().done(r)
