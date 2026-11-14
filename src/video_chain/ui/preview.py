"""The viewer: the start picture tile and the preview player."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import QHBoxLayout, QSlider, QStackedWidget, QVBoxLayout, QWidget

from .theme import icon
from .widgets import FrameTile, VideoView, format_time, icon_button, label

if TYPE_CHECKING:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
else:
    try:                                        # (the player is optional: Preview says so when it is missing)
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    except ImportError:                         # pragma: no cover
        QAudioOutput = QMediaPlayer = None


class StartTile(FrameTile):
    dropped = Signal(str)

    def __init__(self, w, h, parent=None):
        super().__init__(w, h, parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
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
    """Video player for clips and paths. The media player is created on first use.
    opened: a file is played now (its path); moved: where it is (seconds), -1 when it stopped or was closed."""
    opened = Signal(str)
    moved = Signal(float)
    failed = Signal(str)              # a file is missing or cannot be played (what to tell the user)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player: QMediaPlayer | None = None
        self.video: VideoView | None = None
        self.audio = None
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.stack = QStackedWidget()
        self.placeholder = label("Nothing to play yet.\nDouble-click a clip, or play the selected path.",
                                 level="tertiary")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setMinimumHeight(200)
        self.stack.addWidget(self.placeholder)
        v.addWidget(self.stack, 1)
        self.name = label("", level="secondary")
        v.addWidget(self.name)
        row = QHBoxLayout()
        self.b_play = icon_button("play", "Play / pause")
        self.b_play.setEnabled(False)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Position in the video")
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
            if QMediaPlayer is None or QAudioOutput is None:
                raise ImportError
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
        self.player.errorOccurred.connect(self._error)
        return True

    def _error(self, _code, text: str):
        if self.player is None or self.player.source().isEmpty():
            return
        self._fail(f"Cannot play {self.player.source().fileName()}: {text}")

    def _fail(self, text: str):
        """Show why nothing plays (the placeholder says it) and tell the window (status line, log)."""
        if self.player is not None:
            self.player.stop()
        self.name.setText(text)
        self.stack.setCurrentWidget(self.placeholder)
        self.b_play.setEnabled(False)
        self.slider.setEnabled(False)
        self.failed.emit(text)

    def open(self, path: Path, title: str, autoplay: bool = True):
        if not path:
            return
        if not Path(path).exists():
            self._fail(f"File not found: {path}")
            return
        if not self._ensure_player():
            return
        assert self.player is not None and self.video is not None       # (_ensure_player)
        self.player.stop()
        self.player.setSource(QUrl())
        self.opened.emit(str(path))
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
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop(self):
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
            if self.video is not None:
                self.video.clear()
            self.stack.setCurrentWidget(self.placeholder)
            self.name.setText("")
            self.slider.setValue(0)
            self.time.setText("0:00.0 / 0:00.0")
            self.b_play.setEnabled(False)
            self.slider.setEnabled(False)
            self.moved.emit(-1.0)

    def _pos(self, ms):
        assert self.player is not None
        if self.player.source().isEmpty():
            return
        self.moved.emit(ms / 1000)
        if not self.slider.isSliderDown():
            self.slider.setValue(int(ms))
        self.time.setText(f"{format_time(ms / 1000, 1)} / {format_time(self.player.duration() / 1000, 1)}")

    def _state(self, st):
        self.b_play.setIcon(icon("pause" if st == QMediaPlayer.PlaybackState.PlayingState else "play"))
        if st == QMediaPlayer.PlaybackState.StoppedState:
            self.moved.emit(-1.0)                   # (played to its end, or stopped)
