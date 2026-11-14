"""Dialogs for made clips: choose the frame a clip hands over at (HandoverDialog), and play sibling clips side by
side to pick one (CompareDialog)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

from . import shortcuts as S
from .audio_devices import follow_default_output
from .workers import run_in_background
from .. import chain, media
from .theme import icon
from .dialogs import add_action, button_row
from .widgets import fit_size, FrameStepper, FrameTile, icon_button, label, remember_geometry, VideoView, shortcut

log = logging.getLogger(__name__)

FPS = chain.FPS
PREVIEW_W = 480                    # frames are decoded once at this width, so scrubbing is instant


def decode_frames(video: Path, width: int = PREVIEW_W, on_start=None) -> list[QImage]:
    """Every frame of a (short) clip as a small QImage, in order. on_start(process) is given the ffmpeg process
    as soon as it runs (so a closing dialog can end it)."""
    info = chain.probe(video)
    w = max(2, min(width, info["width"]) // 2 * 2)
    h = max(2, round(info["height"] * w / info["width"] / 2) * 2)
    return [QImage(raw, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            for raw in media.iter_frames(video, (w, h), "rgb24", flags=None, on_start=on_start, timeout=600)]


class HandoverDialog(QDialog):
    """Pick the frame of a made clip that the clips after it start from. frame() is the choice (None: the last)."""

    def __init__(self, parent, video: Path, title: str, current: int | None = None, min_frame: int = 1,
                 settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        fit_size(self, 760, 640)
        remember_geometry(self, settings, "handover")
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
        v.addWidget(self.tile, 1, Qt.AlignmentFlag.AlignHCenter)
        self.stepper = FrameStepper(self, FPS)
        self.slider, self.where = self.stepper.slider, self.stepper.label
        self.b_prev, self.b_next = self.stepper.b_back1, self.stepper.b_fwd1
        self.stepper.setEnabled(False)
        v.addWidget(self.stepper)
        self.summary = label("", level="tertiary")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        self.buttons, ok = button_row(self, "Hand over here", enabled=False)
        self.b_last = add_action(self.buttons, "Use the last frame")
        self.b_last.setToolTip("Back to normal: the clips after it start from its last frame")
        self.b_last.clicked.connect(lambda: self.slider.setValue(self.slider.maximum()))
        v.addWidget(self.buttons)
        self.slider.valueChanged.connect(self._show)
        run_in_background(lambda: decode_frames(Path(video), on_start=lambda pr: setattr(self, "_proc", pr)),
                          on_done=lambda frames: self._decoded(frames, ""),
                          on_error=lambda e: self._decoded([], str(e)), owner=self)

    def _decoded(self, frames: list, error: str):
        if error or not frames:
            self.tile.clear_frame(f"Could not read the clip: {error or 'no frames'}")
            return
        self.frames = frames
        n = len(frames)
        self.stepper.set_range(min(self.min_frame, n - 1), n - 1)
        self.stepper.setEnabled(True)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        start = n - 1 if self._current is None else min(int(self._current), n - 1)
        self.slider.setValue(start)
        self._show(self.slider.value())

    def _show(self, k: int):
        if not self.frames:
            return
        n = len(self.frames)
        self.tile.set_image(self.frames[k])
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
                 title: str = "Compare", settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        fit_size(self, 1280, 820)
        remember_geometry(self, settings, "compare")
        self.items, self.on_keep = items, on_keep
        self.players: list = []
        self.kept: str | None = None
        v = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(10)
        cols = 2 if len(items) <= 4 else 3
        self.keep_buttons: dict[str, QPushButton] = {}
        self.errors: dict[str, QLabel] = {}
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
                follow_default_output(out)
                out.setMuted(k != 0)
                p = QMediaPlayer(self)
                p.setAudioOutput(out)
                p.setVideoOutput(view.sink)
                p.setSource(QUrl.fromLocalFile(str(video)))
                p.mediaStatusChanged.connect(lambda st, p=p: self._ended(p, st))
                p.errorOccurred.connect(lambda _e, msg="", nid=nid: self._player_error(nid, msg))
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
            err = label("", severity="error")             # (why this clip cannot be played: filled by its player)
            err.setWordWrap(True)
            err.hide()
            self.errors[nid] = err
            bv.addWidget(err)
            grid.addWidget(box, k // cols, k % cols)
        v.addLayout(grid, 1)
        ctl = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause all (Space)")
        self.b_play.clicked.connect(self.toggle)
        self.b_restart = QPushButton("From the start")
        self.b_restart.clicked.connect(self.restart)
        self.slider = QSlider(Qt.Orientation.Horizontal)
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
        close, _none = button_row(self)
        v.addWidget(close)
        shortcut(S.sequence("play"), self, self.toggle)
        for control in (self.b_play, self.b_restart, self.slider, self.loop, self.sound):
            control.setEnabled(bool(self.players))
        if self.players:
            for p in self.players:                   # the longest clip sets the slider's range
                p.durationChanged.connect(lambda _d: self.slider.setRange(
                    0, max(int(q.duration()) for q in self.players)))
            self.players[0].positionChanged.connect(
                self._follow)
            QTimer.singleShot(200, self.restart)

    def _player_error(self, nid: str, message: str) -> None:
        """A clip that cannot be played (no codec, unreadable file, no audio device): say so under it."""
        lab = self.errors.get(nid)
        if lab is not None:
            lab.setText(f"Cannot play this clip: {message or 'the player reported an error'}")
            lab.show()

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
        return any(p.playbackState() == QMediaPlayer.PlaybackState.PlayingState for p in self.players)

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
        over = (QMediaPlayer.MediaStatus.EndOfMedia, QMediaPlayer.MediaStatus.InvalidMedia,
                QMediaPlayer.MediaStatus.NoMedia)
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
                 title: str, settings: QSettings | None = None):
        from .compare_player import ComparePlayer
        super().__init__(parent)
        self.setWindowTitle(title)
        fit_size(self, 820, 640)
        remember_geometry(self, settings, "split")
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
        self.buttons, _ok = button_row(self, "Split here and insert a clip", enabled=False)
        v.addWidget(self.buttons)

    def at(self) -> float:
        """The moment in the video's file the chosen frame shows."""
        return round(self.start + self.player.position() / chain.FPS, 4)

    def _show(self, k: int):
        ok = 0 < k < self.n - 1
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
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


class VideoFrameDialog(QDialog):
    """Choose one frame of any video (any length: only the frame shown is decoded): a slider over the whole
    video, one frame / one second back and on. frame_file() writes the chosen frame at the video's full size."""

    def __init__(self, parent, video: Path, title: str, info: dict | None = None, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        fit_size(self, 860, 640)
        remember_geometry(self, settings, "frame")
        self.video = Path(video)
        self._req = 0
        self._closed = False
        self.ready = False                          # (the video is probed: for a file the user picked that is a worker's job)
        self._when_ready: list[Callable[[], None]] = []
        self._grabbing = self._again = False        # one ffmpeg at a time; the newest position after it
        self._tmp = media.scratch("frame")
        self.fps, self.duration, self.frames = float(FPS), 0.0, 1
        self._v = QVBoxLayout(self)
        self.note = label("Reading the video …", level="tertiary")
        self.tile = FrameTile(800, 450)
        self.tile.clear_frame("reading the video …")
        self._v.addWidget(self.tile, 1, Qt.AlignmentFlag.AlignHCenter)
        self._v.addWidget(self.note)
        self.buttons, self._ok = button_row(self, "Use this frame", enabled=False)
        self._v.addWidget(self.buttons)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)                 # (dragging the slider: only the frame it stops at is read)
        self._timer.timeout.connect(self._grab)
        self.slider = QSlider(Qt.Orientation.Horizontal)     # (replaced by the stepper's once the video is known)
        self.where = QLabel()
        if info:
            self._setup(info)
        else:
            run_in_background(lambda: chain.probe_source(self.video), on_done=self._probed,
                              on_error=self._unreadable, owner=self)

    def _probed(self, info: dict) -> None:
        if not self._closed:
            self._setup(info)

    def _unreadable(self, error: BaseException) -> None:
        log.warning("could not read %s: %s", self.video, error)
        self.tile.clear_frame("no frame here")
        self.note.setText(f"{self.video.name} could not be read: is it a video?")

    def _setup(self, info: dict) -> None:
        """The frames of the video, once its frame rate and length are known."""
        self.fps = float(info.get("fps") or FPS)
        self.duration = float(info.get("duration") or 0.0)
        self.frames = max(1, int(round(self.duration * self.fps)))
        if int(info.get("frames") or 0) > 0:           # the video stream's own count (a longer sound track
            self.frames = min(self.frames, int(info["frames"]))   # must not add positions without a frame)
        self.tile.clear_frame("reading the frame …")
        self.stepper = FrameStepper(self, self.fps)
        self.slider, self.where = self.stepper.slider, self.stepper.label
        self.stepper.set_range(0, self.frames - 1)
        self.stepper.setEnabled(True)
        self._v.insertWidget(1, self.stepper)
        self.note.setText(f"{self.video.name} · {info.get('width', '?')}×{info.get('height', '?')} · "
                          f"{self.duration:.1f} s · {self.fps:.3g} fps")
        self._ok.setEnabled(True)
        self.ready = True
        self.slider.valueChanged.connect(self._moved)
        self._moved(0)                               # (the first frame: read when the timer fires)
        waiting, self._when_ready = self._when_ready, []
        for call in waiting:
            call()

    def after_ready(self, call: Callable[[], None]) -> None:
        """Run call once the video is known (at once if it is): `frames` and `slider` are the placeholders before."""
        if self.ready:
            call()
        else:
            self._when_ready.append(call)

    def at(self) -> float:
        """The chosen frame's time in the video (its own frame rate)."""
        return self.slider.value() / self.fps

    def _seek(self) -> float:
        """Where to seek for exactly the chosen frame: a quarter frame before it (ffmpeg gives the first frame at
        or after the time; exactly k / fps can round to just after frame k and give the next one)."""
        return max(0.0, (self.slider.value() - 0.25) / self.fps)

    def _moved(self, k: int):
        self._timer.start()

    def _grab(self):
        if self._grabbing:                           # a decode is running: the newest position comes after it
            self._again = True
            return
        self._grabbing = True
        self._req += 1
        req, at = self._req, self._seek()
        out = Path(self._tmp.name) / f"p{req}.png"

        def work():
            return QImage(str(out)) if grab_frame(self.video, at, out, width=960) else QImage()
        # (always answered: a stuck "grabbing" would freeze the dialog)
        run_in_background(work, on_done=lambda img: self._ready(req, img),
                          on_error=lambda _e: self._ready(req, QImage()), owner=self)

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
