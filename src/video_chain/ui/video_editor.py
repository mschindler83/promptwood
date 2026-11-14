"""Video range editor: choose the part of a real video that goes into the chain.

A player with a timeline (drag the two handles or set them at the playhead), frame steps in the source's own frame
rate, a quarter turn either way for a video filmed sideways, and the first and last frame of the range as the
chain will see them (24 fps, turned, cut to the chain's size): the first frame is where a generated clip before the
video ends, the last one where a clip after it starts.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLineEdit,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from .. import chain, media
from .theme import icon, theme, with_alpha
from .dialogs import button_row
from .widgets import fit_size, format_time, FrameTile, guard_wheel, icon_button, label, safe_emit, VideoView

FPS = chain.FPS


def fmt(t: float) -> str:
    return format_time(t, 3)


def choose_video(parent, settings: QSettings | None, start: str | None = None,
                 places: list | None = None, collections: list | None = None) -> str | None:
    """The app's own file browser for videos (frames as thumbnails; it remembers its folder, 'videoDir')."""
    from .image_picker import pick_image, scratch_settings
    return pick_image(parent, "Real video", settings if settings is not None else scratch_settings(), start,
                      places=places, collections=collections, kind="video")


class RangeBar(QWidget):
    """Timeline: the whole video, the chosen range between two handles, the playhead. Click or drag outside the
    handles to move the playhead."""
    seek = Signal(float)
    rangeChanged = Signal(float, float)

    PAD = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(42)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Range of the video: start, end and playhead")
        self.setToolTip("Drag the two handles to set the range. With the keyboard: Ctrl+Left / Ctrl+Right move the "
                        "start, Ctrl+Shift+Left / Right the end, one frame at a time.")
        self.duration, self.start, self.end, self.pos_ = 1.0, 0.0, 1.0, 0.0
        self._drag: str | None = None

    def set_values(self, duration: float, start: float, end: float):
        self.duration = max(duration, 1e-3)
        self.start, self.end = start, end
        self.update()

    def set_pos(self, t: float):
        self.pos_ = t
        self.update()

    def _x(self, t: float) -> float:
        return self.PAD + (self.width() - 2 * self.PAD) * min(1.0, max(0.0, t / self.duration))

    def _t(self, x: float) -> float:
        return min(self.duration, max(0.0, (x - self.PAD) / max(1, self.width() - 2 * self.PAD) * self.duration))

    def paintEvent(self, _e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        track = QRectF(self.PAD, h / 2 - 5, self.width() - 2 * self.PAD, 10)
        p.setPen(Qt.NoPen)
        p.setBrush(t.bgSunken)
        p.drawRoundedRect(track, 4, 4)
        sel = QRectF(QPointF(self._x(self.start), track.top()), QPointF(self._x(self.end), track.bottom()))
        p.setBrush(with_alpha(t.accent, 0.55))
        p.drawRect(sel)
        for x in (self._x(self.start), self._x(self.end)):
            p.setBrush(t.accent)
            p.drawRoundedRect(QRectF(x - 4, 4, 8, h - 8), 3, 3)
        p.setPen(QPen(t.textPrimary, 2))
        x = self._x(self.pos_)
        p.drawLine(QPointF(x, 2), QPointF(x, h - 2))

    def _hit(self, x: float) -> str | None:
        near = [name for name, t in (("start", self.start), ("end", self.end)) if abs(x - self._x(t)) <= 7]
        if len(near) == 2:                      # the handles overlap: the side of the click decides
            return "start" if x < (self._x(self.start) + self._x(self.end)) / 2 else "end"
        return near[0] if near else None

    def keyPressEvent(self, e):
        step = 1 / FPS
        sign = {Qt.Key_Left: -1, Qt.Key_Right: 1}.get(e.key())
        if sign and e.modifiers() & Qt.ControlModifier:
            if e.modifiers() & Qt.ShiftModifier:
                self.end = min(self.duration, max(self.start + step, self.end + sign * step))
                self.seek.emit(self.end)
            else:
                self.start = max(0.0, min(self.end - step, self.start + sign * step))
                self.seek.emit(self.start)
            self.rangeChanged.emit(self.start, self.end)
            self.update()
            e.accept()
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        x = e.position().x()
        self._drag = self._hit(x) or "pos"
        self._move(x)

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if self._drag:
            self._move(x)
        else:
            self.setCursor(Qt.SizeHorCursor if self._hit(x) else Qt.PointingHandCursor)

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def _move(self, x: float):
        t = self._t(x)
        if self._drag == "start":
            self.start = min(t, self.end - 1 / FPS)
            self.rangeChanged.emit(self.start, self.end)
            self.seek.emit(self.start)
        elif self._drag == "end":
            self.end = max(t, self.start + 1 / FPS)
            self.rangeChanged.emit(self.start, self.end)
            self.seek.emit(self.end)
        else:
            self.seek.emit(t)
        self.update()


class RangePlayerMixin:
    """What the audio and the video range dialogs share: the media player (and what happens when it fails), play /
    pause, play the range, the playhead, the start / end boxes. The dialog has: player, bar, time, b_play,
    b_play_sel, info_label, start / end, _stop_at, _pos and _total() (the file's length, 0 when none); it
    implements _update_enabled(), and _before_play() when something else must stop before the player plays."""

    SPIN_STEP = 0.1                                     # the step of the start / end boxes (seconds)
    player_error = ""                                   # why the player cannot play this file ("" = it can)

    def _spin(self, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(3)
        s.setSingleStep(self.SPIN_STEP)
        s.setSuffix(" s")
        s.setToolTip(tip)
        s.setMinimumWidth(110)
        return s

    def _total(self) -> float:
        return 0.0

    def _before_play(self) -> None:
        pass

    def _can_play(self) -> bool:
        return self.player is not None and not self.player_error

    def _attach_player(self, player) -> None:
        player.positionChanged.connect(self._player_pos)
        player.playbackStateChanged.connect(self._player_state)
        player.errorOccurred.connect(self._on_player_error)

    def _on_player_error(self, _error, message: str = "") -> None:
        """A file the player cannot play (no codec, unreadable, no audio device): say so and switch Play off."""
        self.player_error = message or "the player reported an error"
        self.info_label.setText(f"Cannot play {Path(self.file).name}: {self.player_error}")
        self._update_enabled()

    def toggle(self):
        if not self._can_play():
            return
        from PySide6.QtMultimedia import QMediaPlayer
        self._before_play()
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self._stop_at = None
            self.player.play()

    def play_range(self):
        if not self._can_play():
            return
        self._before_play()
        self.seek(self.start)
        self._stop_at = self.end
        self.player.play()

    def _player_pos(self, ms: int):
        self._pos = ms / 1000
        self.bar.set_pos(self._pos)
        if self._total() > 0:
            self.time.setText(f"{fmt(self._pos)} / {fmt(self._total())}")
        if self._stop_at is not None and self._pos >= self._stop_at:
            self._stop_at = None
            self.player.pause()

    def _player_state(self, st):
        from PySide6.QtMultimedia import QMediaPlayer
        self.b_play.setIcon(icon("pause" if st == QMediaPlayer.PlayingState else "play"))


class _Probed(QObject):
    ready = Signal(int, object, str)        # generation, the source's info (None), error


class _Grabbed(QObject):
    ready = Signal(int, str)                # generation, error ("" = the frame file is there)


class _Frames(QObject):
    ready = Signal(int, str, QImage)        # generation, "first" | "last", frame


class VideoRangeDialog(RangePlayerMixin, QDialog):
    """Pick a video file, a range of it and its rotation. file / start / end / rotate hold the result.

    canvas: the chain's video size; canvas_for(file, rotate) -> size or None, when given, works it out again for
    the file and rotation chosen (the size may follow this video)."""

    SPIN_STEP = 1 / FPS

    def __init__(self, parent=None, file: str = "", start: float | None = None, end: float | None = None,
                 canvas: tuple[int, int] | None = None, settings: QSettings | None = None, player: bool = True,
                 on_reference=None, rotate: int = 0, canvas_for=None):
        super().__init__(parent)
        self.setWindowTitle("Real video")
        fit_size(self, 980, 760)
        self.settings, self.canvas, self.want_player = settings, canvas, player
        self.canvas_for = canvas_for
        self.rotate = chain.rotation(rotate)
        self.on_reference = on_reference              # callable(png path) -> bool: crop and add a face picture
        self.file, self.info = "", None
        self.start, self.end = 0.0, 0.0
        self.player = None
        self.video = None
        self._closed = False
        self._gen = 0
        self._probed = _Probed()
        self._probed.ready.connect(self._probe_done)
        self._grabbed = _Grabbed()
        self._grabbed.ready.connect(self._grab_done)
        self._load_gen = 0
        self.loading = False                          # a file is being read (probe or frame grab) off the UI thread
        self._pending: tuple = ()
        self._frames = _Frames()
        self._frames.ready.connect(self._frame_ready)
        self._frame_timer = QTimer(self, singleShot=True, interval=300, timeout=self._update_frames)
        self._stop_at: float | None = None

        v = QVBoxLayout(self)
        frow = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        b = QPushButton("Choose…")
        b.clicked.connect(self._choose)
        frow.addWidget(self.path, 1)
        frow.addWidget(b)
        v.addLayout(frow)
        self.info_label = label("", level="tertiary")
        v.addWidget(self.info_label)

        self.stack = QStackedWidget()
        self.placeholder = label("Choose a video.", level="tertiary")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setMinimumHeight(260)
        self.stack.addWidget(self.placeholder)
        v.addWidget(self.stack, 1)

        self.bar = RangeBar()
        self.bar.seek.connect(self.seek)
        self.bar.rangeChanged.connect(lambda s, e: self.set_range(s, e, from_bar=True))
        v.addWidget(self.bar)

        crow = QHBoxLayout()
        self.b_prev = icon_button("skipb", "One frame back (Left)")
        self.b_play = icon_button("play", "Play / pause (Space)")
        self.b_next = icon_button("skipf", "One frame on (Right)")
        self.b_back1 = QPushButton("−1 s")
        self.b_fwd1 = QPushButton("+1 s")
        self.b_prev.clicked.connect(lambda: self.step_frames(-1))
        self.b_next.clicked.connect(lambda: self.step_frames(1))
        self.b_play.clicked.connect(self.toggle)
        self.b_back1.clicked.connect(lambda: self.seek(self.position() - 1))
        self.b_fwd1.clicked.connect(lambda: self.seek(self.position() + 1))
        self.time = label("0:00.000", level="secondary", type_="mono")
        for w in (self.b_prev, self.b_play, self.b_next, self.b_back1, self.b_fwd1):
            crow.addWidget(w)
        crow.addSpacing(8)
        crow.addWidget(self.time)
        crow.addStretch(1)
        self.b_rot_left = QPushButton("↺ 90°")
        self.b_rot_left.setToolTip("Turn the video a quarter turn to the left (Shift+R); the chain gets it turned")
        self.b_rot_left.clicked.connect(lambda: self.set_rotation(self.rotate - 90))
        self.b_rot_right = QPushButton("↻ 90°")
        self.b_rot_right.setToolTip("Turn the video a quarter turn to the right (R); the chain gets it turned")
        self.b_rot_right.clicked.connect(lambda: self.set_rotation(self.rotate + 90))
        crow.addWidget(self.b_rot_left)
        crow.addWidget(self.b_rot_right)
        crow.addSpacing(16)
        self.b_play_sel = QPushButton("Play range")
        self.b_play_sel.setToolTip("Play from the start to the end of the range")
        self.b_play_sel.clicked.connect(self.play_range)
        crow.addWidget(self.b_play_sel)
        v.addLayout(crow)

        rrow = QHBoxLayout()
        self.start_spin = self._spin("Start of the range (seconds in the video)")
        self.end_spin = self._spin("End of the range (seconds in the video)")
        b_set_start = QPushButton("Start here")
        b_set_start.setToolTip("Set the start of the range at the playhead ([)")
        b_set_start.clicked.connect(lambda: self.set_range(self.position(), self.end))
        b_set_end = QPushButton("End here")
        b_set_end.setToolTip("Set the end of the range at the playhead (])")
        b_set_end.clicked.connect(lambda: self.set_range(self.start, self.position(), moved="end"))
        rrow.addWidget(label("Start", level="tertiary"))
        rrow.addWidget(self.start_spin)
        rrow.addWidget(b_set_start)
        rrow.addSpacing(16)
        rrow.addWidget(label("End", level="tertiary"))
        rrow.addWidget(self.end_spin)
        rrow.addWidget(b_set_end)
        rrow.addStretch(1)
        self.b_ref = QPushButton("Add as face picture…")
        self.b_ref.setToolTip("Crop a face from the frame at the playhead (the face is found for you, or set the "
                              "square by hand) and add it to the character selected in Cast & references; clips that "
                              "mention that character get this face")
        self.b_ref.clicked.connect(self.reference_here_async)
        self.face = True                              # set_reference_kind: a person's face, else a whole picture
        self.b_ref.setVisible(on_reference is not None)
        rrow.addWidget(self.b_ref)
        v.addLayout(rrow)
        self.keys_hint = label("Keys: Space play / pause · ← → one frame · [ start here · ] end here · "
                               "R turn right · Shift+R turn left", level="tertiary")
        v.addWidget(self.keys_hint)
        self.start_spin.valueChanged.connect(lambda x: self.set_range(x, self.end, moved="start"))
        self.end_spin.valueChanged.connect(lambda x: self.set_range(self.start, x, moved="end"))
        guard_wheel(self.start_spin, self.end_spin)
        self.summary = label("", level="secondary")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)

        tiles = QHBoxLayout()
        self.first_tile, self.last_tile = FrameTile(300, 170), FrameTile(300, 170)
        for title, tile, tip in (("First frame", self.first_tile, "Where a generated clip before the video ends"),
                                 ("Last frame", self.last_tile, "Where a generated clip after the video starts")):
            col = QVBoxLayout()
            col.addWidget(label(title, level="tertiary"))
            tile.setToolTip(tip + " (as the chain sees it: 24 fps, cut to the video size)")
            col.addWidget(tile)
            tiles.addLayout(col)
        tiles.addStretch(1)
        v.addLayout(tiles)

        self.buttons, _ok = button_row(self, "Use this range")
        v.addWidget(self.buttons)
        for key, fn in (("Space", self.toggle), ("Left", lambda: self.step_frames(-1)),
                        ("Right", lambda: self.step_frames(1)), ("[", lambda: self.set_range(self.position(), self.end)),
                        ("]", lambda: self.set_range(self.start, self.position(), moved="end")),
                        ("R", lambda: self.set_rotation(self.rotate + 90)),
                        ("Shift+R", lambda: self.set_rotation(self.rotate - 90))):
            from PySide6.QtGui import QKeySequence, QShortcut
            QShortcut(QKeySequence(key), self, activated=fn)
        self._pos = 0.0
        self._keep = None if file else (start, end)   # a missing file: its range (and turn) is kept for the one
                                                      # chosen again
        if file:
            self.load(file, start, end)
        self._update_enabled()

    # ---- file
    def _choose(self):
        f = choose_video(self, self.settings, str(Path(self.file).parent) if self.file else None)
        if f:
            keep = self._keep or (None, None)
            if self._keep is None and f != self.file:
                self.rotate = 0                     # another video: not turned until it is turned
            self.load_async(f, *keep)

    def _set_loading(self, on: bool, text: str = "") -> None:
        self.loading = on
        if on:
            self.info_label.setText(text)
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            self.b_ref.setEnabled(False)
        else:
            self._update_enabled()

    def load_async(self, file: str, start: float | None = None, end: float | None = None) -> None:
        """load(), with ffprobe run off the UI thread: the dialog shows "Reading …" and cannot be accepted meanwhile.
        An answer that arrives after another load began or the dialog closed is dropped."""
        self._load_gen += 1
        gen, sig = self._load_gen, self._probed
        self._pending = (str(file), start, end)
        self._set_loading(True, f"Reading {Path(file).name} …")

        def work():
            try:
                info, err = chain.probe_source(Path(file)), ""
            except (RuntimeError, OSError, ValueError) as e:
                info, err = None, str(e)
            safe_emit(sig.ready, gen, info, err)
        threading.Thread(target=work, daemon=True).start()

    def _probe_done(self, gen: int, info, err: str) -> None:
        if gen != self._load_gen or self._closed:
            return
        file, start, end = self._pending
        self.loading = False
        if info is None:
            self.info_label.setText(f"Cannot read {Path(file).name}: {err}")
            self._update_enabled()
            return
        if self._load_info(file, info, start, end):
            self._keep = None                      # (a file that cannot be read keeps the saved range)

    def load(self, file: str, start: float | None = None, end: float | None = None) -> bool:
        try:
            info = chain.probe_source(Path(file))
        except (RuntimeError, OSError, ValueError) as e:
            self.info_label.setText(f"Cannot read {Path(file).name}: {e}")
            return False
        return self._load_info(file, info, start, end)

    def _load_info(self, file: str, info: dict, start: float | None, end: float | None) -> bool:
        if info["duration"] <= 0:
            self.info_label.setText(f"{Path(file).name} has no known length.")
            return False
        self.file, self.info = str(file), info
        self.path.setText(self.file)
        self._show_info()
        for s in (self.start_spin, self.end_spin):
            s.blockSignals(True)
            s.setRange(0.0, info["duration"])
            s.blockSignals(False)
        if start is not None and end is not None and float(end) > info["duration"] + 0.05:
            start = end = None                      # the range does not fit this file: all of it
        st = 0.0 if start is None else float(start)
        en = info["duration"] if end is None else float(end)
        self._open_player()
        self._apply_rotation()
        self.set_range(st, en)
        self.seek(st)
        self._update_enabled()
        return True

    def _open_player(self):
        if not self.want_player:
            return
        if self.player is None:
            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            except ImportError:
                self.placeholder.setText("Video playback is not available (QtMultimedia missing); "
                                         "the frames below show the range.")
                return
            try:
                self.video = VideoView(turnable=True)   # (GPU or CPU as set; either can be turned, see VideoView)
            except (ImportError, RuntimeError):         # (a broken Qt Quick set-up: the CPU view still works)
                self.video = VideoView(turnable=True, gpu=False)
            self.stack.addWidget(self.video)
            self.audio = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video.sink)
            self._attach_player(self.player)
        self.player_error = ""
        self.player.setSource(QUrl.fromLocalFile(self.file))
        self.stack.setCurrentWidget(self.video)

    # ---- rotation
    def set_rotation(self, degrees: int) -> None:
        """Turn the video clockwise by degrees (0/90/180/270): the player, the frames and the chain's clip."""
        degrees = int(degrees) % 360
        if self.info is None or degrees == self.rotate:
            return
        self.rotate = chain.rotation(degrees)
        self._apply_rotation()
        self.set_range(self.start, self.end)       # (the summary and the first / last frame again)

    def _apply_rotation(self):
        if self.canvas_for is not None and self.file:
            try:
                self.canvas = self.canvas_for(self.file, self.rotate) or self.canvas
            except Exception:  # noqa: BLE001  (the size stays as it was)
                pass
        if self.player is not None:
            self.video.set_rotation(self.rotate)
        self._show_info()

    def _show_info(self):
        info = self.info
        if info is None:
            return
        w, h = chain.rotated_size(info["width"], info["height"], self.rotate)
        turned = f" · turned {self.rotate}° right" if self.rotate else ""
        self.info_label.setText(f"{w}×{h}{turned} · {info['fps']:.3f} fps · {fmt(info['duration'])}"
                                f" · {'with sound' if info['audio'] else 'no sound'}")

    def _update_enabled(self):
        on = self.info is not None
        for w in (self.bar, self.b_prev, self.b_play, self.b_next, self.b_back1, self.b_fwd1, self.b_play_sel,
                  self.start_spin, self.end_spin, self.b_ref, self.b_rot_left, self.b_rot_right):
            w.setEnabled(on)
        self.b_play.setEnabled(on and self._can_play())
        self.b_play_sel.setEnabled(on and self._can_play())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(on and self.frames() >= 1)

    # ---- face picture
    def set_reference_kind(self, name: str, kind: str) -> None:
        """The Cast's selected entry is a place, object or style (not "person"): its pictures are whole frames."""
        self.face = kind == "person"
        if not self.face:
            self.b_ref.setText(f"Add as picture of @{name}")
            self.b_ref.setToolTip(f"Add the whole frame at the playhead as a picture of @{name} (a {kind}); clips "
                                  "that mention it get this picture")

    def reference_here_async(self) -> None:
        """reference_here with the frame read by ffmpeg off the UI thread (the button's action)."""
        if self.info is None or self.on_reference is None or self.loading:
            return
        t = self.position()
        ctx = media.scratch("ref")
        d = ctx.__enter__()
        png = Path(d) / "frame.png"
        turn = chain.rotate_filter(self.rotate)
        cmd = ["-y", "-ss", f"{t:.4f}", "-i", self.file] + (["-vf", turn] if turn else []) + \
              ["-frames:v", "1", "-update", "1", png]
        self._load_gen += 1
        gen, sig = self._load_gen, self._grabbed
        self._pending = (ctx, png, t)
        self._set_loading(True, "Reading the frame …")

        def work():
            try:
                media.ffmpeg(cmd, timeout=60)
            except (OSError, media.MediaError):
                pass
            safe_emit(sig.ready, gen, "" if png.is_file() else "missing")
        threading.Thread(target=work, daemon=True).start()

    def _grab_done(self, gen: int, err: str) -> None:
        ctx, png, t = self._pending
        if gen != self._load_gen or self._closed:
            ctx.__exit__(None, None, None)
            return
        self.loading = False
        try:
            if err:
                self.summary.setText(f"Could not read the frame at {fmt(t)}.")
                ok = False
            else:
                ok = bool(self.on_reference(str(png)))
                what = "face picture" if self.face else "picture"
                where = "(see the Cast & references tab, Alt+2)"
                self.summary.setText((f"The face from the frame at {fmt(t)} was added {where}." if self.face
                                      else f"The frame at {fmt(t)} was added as a picture {where}.") if ok
                                     else f"The frame at {fmt(t)} was not added as a {what}.")
        finally:
            ctx.__exit__(None, None, None)
            self._update_enabled()

    def reference_here(self) -> bool:
        """The frame at the playhead at the source's size (turned as set), handed to on_reference (which opens the
        crop window)."""
        if self.info is None or self.on_reference is None:
            return False
        t = self.position()
        with media.scratch("ref") as d:
            png = Path(d) / "frame.png"
            try:
                turn = chain.rotate_filter(self.rotate)
                media.ffmpeg(["-y", "-ss", f"{t:.4f}", "-i", self.file] + (["-vf", turn] if turn else [])
                             + ["-frames:v", "1", "-update", "1", png], timeout=60)
            except (OSError, media.MediaError):
                pass
            if not png.is_file():
                self.summary.setText(f"Could not read the frame at {fmt(t)}.")
                return False
            ok = bool(self.on_reference(str(png)))
        what = "face picture" if self.face else "picture"
        where = "(see the Cast & references tab, Alt+2)"
        self.summary.setText((f"The face from the frame at {fmt(t)} was added {where}." if self.face
                              else f"The frame at {fmt(t)} was added as a picture {where}.") if ok
                             else f"The frame at {fmt(t)} was not added as a {what}.")
        return ok

    # ---- range
    def frames(self) -> int:
        return chain.range_frames(self.start, self.end)

    def set_range(self, start: float, end: float, from_bar: bool = False, moved: str = "start"):
        """moved: the value the user set; if the range gets shorter than one frame, the other one gives way."""
        if self.info is None:
            return
        dur = self.info["duration"]
        start = min(max(0.0, float(start)), dur)
        end = min(max(0.0, float(end)), dur)
        if end - start < 1 / FPS:                    # at least one frame of the chain
            if moved == "end":
                start = max(0.0, end - 1 / FPS)
                end = start + 1 / FPS
            else:
                end = min(dur, start + 1 / FPS)
                start = end - 1 / FPS
        self.start, self.end = round(start, 4), round(end, 4)
        if not from_bar:
            self.bar.set_values(dur, self.start, self.end)
        else:
            self.bar.duration = dur
        for spin, val in ((self.start_spin, self.start), (self.end_spin, self.end)):
            if abs(spin.value() - val) > 5e-4:
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
        n = self.frames()
        size = f", cut to {self.canvas[0]}×{self.canvas[1]} at the centre" if self.canvas else ""
        turned = f", turned {self.rotate}° right" if self.rotate else ""
        self.summary.setText(f"Range {fmt(self.start)} – {fmt(self.end)}: {self.end - self.start:.3f} s → {n} frames "
                             f"at {FPS} fps{turned}{size}." + ("" if self.info["audio"] else " No sound."))
        self._update_enabled()
        self._frame_timer.start()

    def _update_frames(self):
        if self.info is None or self._closed:
            return
        self._gen += 1
        gen, file, sig = self._gen, self.file, self._frames
        n = self.frames()
        start = self.start
        w, h = self.canvas or (self.info["width"], self.info["height"])
        turn = chain.rotate_filter(self.rotate)
        size = (f",{turn}" if turn else "") + \
            (f",scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}" if self.canvas else "")

        def work():
            # the frames exactly as the conversion picks them (chain.convert_range): seek to the start, then the
            # 24 fps filter, then frame number 0 / n-1 of its output, turned and cut as the conversion does it
            # (decodes the range: exact, a bit slower)
            for which, idx in (("first", 0), ("last", max(0, n - 1))):
                if gen != self._gen:                      # the range moved on: skip the stale frame
                    return
                img = self._decode(["ffmpeg", "-v", "error", "-ss", f"{float(start):.6f}", "-i", file,
                                    "-map", "0:v:0", "-vf", f"fps={FPS},select=eq(n\\,{idx}){size}",
                                    "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"], gen)
                if img is None:                           # the range moved on (or the dialog closed): stopped
                    return
                if not safe_emit(sig.ready, gen, which, img):
                    return
        threading.Thread(target=work, daemon=True).start()

    def _decode(self, cmd: list[str], gen: int) -> QImage | None:
        """Run a frame decode; killed as soon as the range moves on or the dialog closes (None then)."""
        import time
        try:
            proc = media.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        except (OSError, media.MediaError):
            return QImage()
        end = time.monotonic() + 120
        while True:
            try:
                out, _ = proc.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if gen != self._gen or time.monotonic() > end:
                    proc.kill()
                    proc.communicate()
                    return None if gen != self._gen else QImage()
        return QImage.fromData(out, "PNG") if proc.returncode == 0 else QImage()

    def _frame_ready(self, gen: int, which: str, img: QImage):
        if gen != self._gen:
            return
        (self.first_tile if which == "first" else self.last_tile).set_image(img)

    # ---- playback
    def position(self) -> float:
        return self._pos

    def seek(self, t: float):
        if self.info is None:
            return
        self._stop_at = None                        # a seek ends "play range"
        self._pos = min(max(0.0, t), self.info["duration"])
        self.bar.set_pos(self._pos)
        self.time.setText(f"{fmt(self._pos)} / {fmt(self.info['duration'])}")
        if self.player is not None:
            self.player.setPosition(round(self._pos * 1000))

    def step_frames(self, k: int):
        if self.info is None:
            return
        if self.player is not None:
            self.player.pause()
        fps = self.info["fps"] or FPS
        frame = round(self._pos * fps) + k
        self.seek(frame / fps)

    def _total(self) -> float:
        return self.info["duration"] if self.info else 0.0

    def done(self, r):                                  # stop playback and the file handle with the dialog
        self._closed = True
        self._load_gen += 1                             # a probe or frame grab still running is dropped
        self._frame_timer.stop()                        # (a decode must not start for a dialog nobody sees)
        self._gen += 1                                  # running frame decodes stop
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
        super().done(r)
