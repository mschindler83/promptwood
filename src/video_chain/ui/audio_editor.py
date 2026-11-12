"""Audio track editor: choose a sound file (MP3, WAV, OGG, FLAC … or a video) and cut it to the part the clips play.

The whole sound as a waveform with the chosen range between two handles; play it, set start / end at the playhead or
type them in. The cut part becomes an audio track of the project: clips that play it carry it on one after another.
"""
from __future__ import annotations

import array
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLineEdit, QPushButton, QVBoxLayout)

from .. import chain
from ..tree import AUDIO_REF_MIN, VOICE_MAX_SECS, VOICE_REF_MAX_SECS
from .theme import icon, theme, with_alpha
from .video_editor import RangeBar, fmt
from .widgets import guard_wheel, icon_button, label

PEAKS = 1200                                    # waveform columns


def waveform(path: str | Path, duration: float, columns: int = PEAKS, rate: int = 4000,
             cancelled=lambda: False) -> list[float]:
    """The loudest sample of each of about `columns` equal parts of a file's sound (0..1), read as a stream (long
    files need little memory); [] if it cannot be read or `cancelled()` turns true."""
    import math
    step = max(1, math.ceil(max(duration, 0.001) * rate / columns))     # samples per column
    try:
        proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
                                 "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return []
    out, rest = [], b""
    try:
        while True:
            if cancelled():
                return []
            data = proc.stdout.read(step * 4 * 64)
            if not data:
                break
            data = rest + data
            n = len(data) // (step * 4) * (step * 4)
            a = array.array("f")
            a.frombytes(data[:n])
            rest = data[n:]
            for k in range(0, len(a), step):
                chunk = a[k:k + step]
                out.append(min(1.0, max(max(chunk), -min(chunk))))
        if len(rest) >= 4:
            a = array.array("f")
            a.frombytes(rest[: len(rest) // 4 * 4])
            out.append(min(1.0, max(max(a), -min(a))))
        return out
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


class WaveBar(RangeBar):
    """The range bar with the sound's waveform behind it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.peaks: list[float] = []

    def set_peaks(self, peaks: list[float]):
        self.peaks = peaks
        self.update()

    def paintEvent(self, e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h, w = self.height(), self.width() - 2 * self.PAD
        p.fillRect(QRectF(self.PAD, 4, w, h - 8), t.bgSunken)
        x0, x1 = self._x(self.start), self._x(self.end)
        p.fillRect(QRectF(x0, 4, x1 - x0, h - 8), with_alpha(t.accent, 0.18))
        if self.peaks:
            n = len(self.peaks)
            mid = h / 2
            for k, v in enumerate(self.peaks):
                x = self.PAD + w * k / n
                inside = x0 <= x <= x1
                p.setPen(QPen(t.accent if inside else t.textTertiary, max(1.0, w / n)))
                p.drawLine(QPointF(x, mid - v * (mid - 8)), QPointF(x, mid + v * (mid - 8)))
        for x in (x0, x1):
            p.setPen(Qt.NoPen)
            p.setBrush(t.accent)
            p.drawRoundedRect(QRectF(x - 3, 2, 6, h - 4), 3, 3)
        p.setPen(QPen(t.textPrimary, 2))
        x = self._x(self.pos_)
        p.drawLine(QPointF(x, 2), QPointF(x, h - 2))


class _Peaks(QObject):
    ready = Signal(int, list)


class AudioRangeDialog(QDialog):
    """Choose a sound file and the part of it that becomes an audio track. file / start / end / name: the result.
    voice: the part becomes a character's voice instead (no name; a video's picture is shown while it plays, to
    see who speaks; a new file starts with its first VOICE_REF_MAX_SECS chosen)."""

    def __init__(self, parent=None, file: str = "", start: float | None = None, end: float | None = None,
                 name: str = "", settings: QSettings | None = None, player: bool = True, title: str = "Audio track",
                 voice: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(920, 640 if voice else 420)
        self.settings, self.want_player, self.voice = settings, player, voice
        self.video = None
        self.file, self.duration = "", 0.0
        self.start, self.end = 0.0, 0.0
        self.player = None
        self._pos = 0.0
        self._stop_at: float | None = None
        self._gen = 0
        self._peaks = _Peaks()
        self._peaks.ready.connect(self._peaks_ready)
        self._named = bool(name)                       # a name typed (or given): a new file does not replace it

        v = QVBoxLayout(self)
        frow = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("MP3, WAV, OGG, FLAC … or a video")
        b = QPushButton("Choose…")
        b.clicked.connect(self._choose)
        frow.addWidget(self.path, 1)
        frow.addWidget(b)
        v.addLayout(frow)
        form = QFormLayout()
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("e.g. Interview, Song, Rain")
        self.name.textEdited.connect(lambda _t: setattr(self, "_named", True))
        form.addRow("Name", self.name)
        v.addLayout(form)
        if voice:
            form.setRowVisible(0, False)
        self.info_label = label("", level="tertiary")
        v.addWidget(self.info_label)
        if voice and player:
            from .widgets import VideoView
            self.video = VideoView()                # (a video's picture: who speaks when)
            self.video.setMinimumHeight(220)
            self.video.hide()
            v.addWidget(self.video, 3)

        self.bar = WaveBar()
        self.bar.seek.connect(self.seek)
        self.bar.rangeChanged.connect(lambda s, e: self.set_range(s, e, from_bar=True))
        v.addWidget(self.bar, 1)

        crow = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause (Space)")
        self.b_play.clicked.connect(self.toggle)
        self.b_back = QPushButton("−1 s")
        self.b_fwd = QPushButton("+1 s")
        self.b_back.clicked.connect(lambda: self.seek(self.position() - 1))
        self.b_fwd.clicked.connect(lambda: self.seek(self.position() + 1))
        self.time = label("0:00.000", level="secondary", type_="mono")
        for w in (self.b_play, self.b_back, self.b_fwd):
            crow.addWidget(w)
        crow.addSpacing(8)
        crow.addWidget(self.time)
        crow.addStretch(1)
        self.b_play_sel = QPushButton("Play range")
        self.b_play_sel.setToolTip("Play from the start to the end of the range")
        self.b_play_sel.clicked.connect(self.play_range)
        crow.addWidget(self.b_play_sel)
        v.addLayout(crow)

        rrow = QHBoxLayout()
        what = "voice" if voice else "track"
        self.start_spin, self.end_spin = self._spin(f"Where the {what} starts (seconds in the file)"), \
            self._spin(f"Where the {what} ends (seconds in the file)")
        b_s = QPushButton("Start here")
        b_s.setToolTip("Set the start at the playhead ([)")
        b_s.clicked.connect(lambda: self.set_range(self.position(), self.end))
        b_e = QPushButton("End here")
        b_e.setToolTip("Set the end at the playhead (])")
        b_e.clicked.connect(lambda: self.set_range(self.start, self.position(), moved="end"))
        rrow.addWidget(label("Start", level="tertiary"))
        rrow.addWidget(self.start_spin)
        rrow.addWidget(b_s)
        rrow.addSpacing(16)
        rrow.addWidget(label("End", level="tertiary"))
        rrow.addWidget(self.end_spin)
        rrow.addWidget(b_e)
        rrow.addStretch(1)
        v.addLayout(rrow)
        self.start_spin.valueChanged.connect(lambda x: self.set_range(x, self.end, moved="start"))
        self.end_spin.valueChanged.connect(lambda x: self.set_range(self.start, x, moved="end"))
        guard_wheel(self.start_spin, self.end_spin)
        self.summary = label("", level="secondary")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Use this part")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        for key, fn in (("Space", self.toggle), ("[", lambda: self.set_range(self.position(), self.end)),
                        ("]", lambda: self.set_range(self.start, self.position(), moved="end"))):
            QShortcut(QKeySequence(key), self, activated=fn)
        if file:
            self.load(file, start, end)
        self._update_enabled()

    def _spin(self, tip: str) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(3)
        s.setSingleStep(0.1)
        s.setSuffix(" s")
        s.setToolTip(tip)
        s.setMinimumWidth(110)
        return s

    # ---- file
    def _choose(self):
        folder = str(Path(self.file).parent) if self.file else (
            self.settings.value("soundDir", "", type=str) if self.settings is not None else "")
        from .image_picker import pick_image
        f = pick_image(self, "A recording of the voice (audio or video)" if self.voice else "Sound for an audio track",
                       self.settings, folder or None, kind="media")
        if f:
            self.load(f)

    def load(self, file: str, start: float | None = None, end: float | None = None) -> bool:
        try:
            dur = chain.audio_seconds(Path(file))
        except (RuntimeError, OSError, ValueError) as e:
            self.info_label.setText(f"Cannot read {Path(file).name}: {e}")
            return False
        if not dur or dur <= 0:
            self.info_label.setText(f"{Path(file).name} has no sound.")
            return False
        self.file, self.duration = str(file), float(dur)
        self.path.setText(self.file)
        if not self._named:
            self.name.setText(Path(file).stem)
        self.info_label.setText(f"{fmt(dur)} of sound")
        for s in (self.start_spin, self.end_spin):
            s.blockSignals(True)
            s.setRange(0.0, dur)
            s.blockSignals(False)
        if start is not None and end is not None and float(end) > dur + 0.05:
            start = end = None                         # the range does not fit this file: all of it
        self.bar.set_peaks([])
        self._open_player()
        whole = min(dur, VOICE_REF_MAX_SECS) if self.voice else dur
        self.set_range(0.0 if start is None else float(start), whole if end is None else float(end))
        self.seek(self.start)
        self._gen += 1
        gen, sig, f, dur = self._gen, self._peaks, self.file, self.duration

        def work():
            peaks = waveform(f, dur, cancelled=lambda: gen != self._gen)
            if gen != self._gen:
                return
            try:
                sig.ready.emit(gen, peaks)
            except RuntimeError:                       # the dialog is gone
                pass
        threading.Thread(target=work, daemon=True).start()
        self._update_enabled()
        return True

    def _peaks_ready(self, gen: int, peaks: list):
        if gen == self._gen:
            self.bar.set_peaks(peaks)

    def _open_player(self):
        if not self.want_player:
            return
        if self.player is None:
            try:
                from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            except ImportError:
                return
            self.audio = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio)
            if self.video is not None:
                self.player.setVideoOutput(self.video.sink)
                self.player.hasVideoChanged.connect(self.video.setVisible)
            self.player.positionChanged.connect(self._player_pos)
            self.player.playbackStateChanged.connect(self._player_state)
        self.player.setSource(QUrl.fromLocalFile(self.file))

    def _update_enabled(self):
        on = self.duration > 0
        for w in (self.bar, self.b_back, self.b_fwd, self.start_spin, self.end_spin):
            w.setEnabled(on)
        self.b_play.setEnabled(on and self.player is not None)
        self.b_play_sel.setEnabled(on and self.player is not None)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(on and self.end - self.start >= AUDIO_REF_MIN - 1e-6)

    # ---- range
    def set_range(self, start: float, end: float, from_bar: bool = False, moved: str = "start"):
        """moved: the value the user set; if the range gets shorter than 0.1 s, the other one gives way."""
        if self.duration <= 0:
            return
        dur = self.duration
        start = min(max(0.0, float(start)), dur)
        end = min(max(0.0, float(end)), dur)
        if end - start < 0.1:
            if moved == "end":
                start = max(0.0, end - 0.1)
                end = start + 0.1
            else:
                end = min(dur, start + 0.1)
                start = end - 0.1
        was = (self.start, self.end)
        self.start, self.end = round(start, 3), round(end, 3)
        if self.video is not None and not self._playing():      # the picture at the edge that was moved
            if abs(self.start - was[0]) > 5e-4:
                self.seek(self.start)
            elif abs(self.end - was[1]) > 5e-4:
                self.seek(self.end)
        if not from_bar:
            self.bar.set_values(dur, self.start, self.end)
        else:
            self.bar.duration = dur
        for spin, val in ((self.start_spin, self.start), (self.end_spin, self.end)):
            if abs(spin.value() - val) > 5e-4:
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
        secs = self.end - self.start
        short = f" At least {AUDIO_REF_MIN:.0f} s are needed." if secs < AUDIO_REF_MIN else ""
        if self.voice:
            first = min(secs, VOICE_MAX_SECS)
            cloned = "all of it" if secs <= VOICE_REF_MAX_SECS else f"its first {VOICE_REF_MAX_SECS:g} s"
            self.summary.setText(
                f"Voice {fmt(self.start)} – {fmt(self.end)}: {secs:.2f} s.{short} Choose a part where only this "
                f"speaker talks, without music or echo. The clip model hears its first {first:.1f} s; text to speech "
                f"clones the voice from {cloned}.")
        else:
            self.summary.setText(f"Track {fmt(self.start)} – {fmt(self.end)}: {secs:.2f} s.{short} Clips that play "
                                 "it carry it on one after another; past its end they are silent.")
        self._update_enabled()

    # ---- playback
    def position(self) -> float:
        return self._pos

    def seek(self, t: float):
        if self.duration <= 0:
            return
        self._pos = min(max(0.0, t), self.duration)
        self.bar.set_pos(self._pos)
        self.time.setText(f"{fmt(self._pos)} / {fmt(self.duration)}")
        if self.player is not None:
            self.player.setPosition(round(self._pos * 1000))
            if self.video is not None:
                from PySide6.QtMultimedia import QMediaPlayer
                if self.player.playbackState() == QMediaPlayer.StoppedState:
                    self.player.pause()             # (a stopped player shows no picture: paused, it shows the frame)

    def _playing(self) -> bool:
        if self.player is None:
            return False
        from PySide6.QtMultimedia import QMediaPlayer
        return self.player.playbackState() == QMediaPlayer.PlayingState

    def toggle(self):
        if self.player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self._stop_at = None
            self.player.play()

    def play_range(self):
        if self.player is None:
            return
        self.seek(self.start)
        self._stop_at = self.end
        self.player.play()

    def _player_pos(self, ms: int):
        self._pos = ms / 1000
        self.bar.set_pos(self._pos)
        self.time.setText(f"{fmt(self._pos)} / {fmt(self.duration)}")
        if self._stop_at is not None and self._pos >= self._stop_at:
            self._stop_at = None
            self.player.pause()

    def _player_state(self, st):
        from PySide6.QtMultimedia import QMediaPlayer
        self.b_play.setIcon(icon("pause" if st == QMediaPlayer.PlayingState else "play"))

    def done(self, r):                                  # stop playback and the file handle with the dialog
        self._gen += 1
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
        super().done(r)
