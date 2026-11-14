"""Small themed widgets shared by the windows (same conventions as another app / the design)."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QHBoxLayout, QLabel, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from .theme import icon, theme


def format_time(seconds: float, precision: int = 0) -> str:
    """m:ss (precision 0) or m:ss.mmm (precision 3). Rounded first, then split: 59.6 s is 1:00, never 0:60."""
    q = 10 ** precision
    total = round(max(0.0, float(seconds)) * q)
    whole, frac = divmod(total, q)
    m, s = divmod(whole, 60)
    return f"{m}:{s:02d}" + (f".{frac:0{precision}d}" if precision else "")


def frame_image(frame) -> QImage:
    """A decoded video frame as an RGB image, converted on the CPU (YUV420P / NV12, limited or full range, BT.709
    when the frame says so, else BT.601 as Qt does): QVideoFrame.toImage() may use a GPU (OpenGL through the EGL driver that crashed).
    Turned by the stream's rotation (a phone video's flag, in the frame's format), as toImage() and ffmpeg turn it.
    Other formats fall back to toImage()."""
    from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat
    try:
        import numpy as np
    except ImportError:                            # (numpy is optional for the app)
        return frame.toImage()
    fmt = frame.pixelFormat()
    if fmt not in (QVideoFrameFormat.Format_YUV420P, QVideoFrameFormat.Format_NV12) or \
            not frame.map(QVideoFrame.ReadOnly):
        return frame.toImage()
    try:
        w, h = frame.width(), frame.height()

        def plane(i, rows, cols, ch=1):
            line = frame.bytesPerLine(i)
            a = np.frombuffer(frame.bits(i), np.uint8, count=line * rows).reshape(rows, line)
            return a[:, :cols * ch].reshape(rows, cols, ch) if ch > 1 else a[:, :cols]
        hw, hh = (w + 1) // 2, (h + 1) // 2
        y = plane(0, h, w).astype(np.float32)
        if fmt == QVideoFrameFormat.Format_NV12:
            uv = plane(1, hh, hw, 2).astype(np.float32)
            u, v = uv[..., 0], uv[..., 1]
        else:
            u, v = plane(1, hh, hw).astype(np.float32), plane(2, hh, hw).astype(np.float32)
    finally:
        frame.unmap()
    u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:h, :w] - 128.0
    v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:h, :w] - 128.0
    sf = frame.surfaceFormat()
    if sf.colorRange() == QVideoFrameFormat.ColorRange_Full:
        k, cs = 1.0, 1.0
    else:                                          # limited (video) range: 16-235, chroma 16-240
        y, k, cs = y - 16.0, 255.0 / 219.0, 255.0 / 224.0
    if sf.colorSpace() == QVideoFrameFormat.ColorSpace_BT709:
        cr, cgu, cgv, cb = 1.5748, 0.187324, 0.468124, 1.8556
    else:                                          # BT.601: also when the clip does not say (as Qt converts it)
        cr, cgu, cgv, cb = 1.402, 0.344136, 0.714136, 1.772
    y, u, v = y * k, u * cs, v * cs
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = y + cr * v
    rgb[..., 1] = y - cgu * u - cgv * v
    rgb[..., 2] = y + cb * u
    out = np.ascontiguousarray(rgb.clip(0, 255).astype(np.uint8))
    img = QImage(out.data, w, h, w * 3, QImage.Format_RGB888).copy()
    turn = sf.rotation().value if hasattr(sf, "rotation") else 0
    return img.transformed(QTransform().rotate(turn)) if turn else img


GPU_VIDEO = False          # set at start-up from the setting (app.py): players draw through Qt's GPU video surface


def set_gpu_video(on: bool) -> None:
    global GPU_VIDEO
    GPU_VIDEO = bool(on)


class VideoView(QWidget):
    """Shows a QMediaPlayer's video. Use: player.setVideoOutput(view.sink).

    GPU (the "Play videos on the GPU" setting): a QVideoWidget inside, with hardware decoding: smooth with big
    clips, but it renders through its own EGL/OpenGL window, and NVIDIA's EGL driver crashed there once (segfault
    in libnvidia-eglcore while the first frame was drawn) when ComfyUI held nearly all GPU memory after a long clip.
    CPU: frames come through a QVideoSink and are painted like an image (kept in proportion, black around): never
    crashes, but decoding, colour conversion and smooth scaling of every frame on the CPU lag with large clips.

    turnable: set_rotation turns the picture clockwise on top of what the player shows. CPU: the painted image is
    turned. GPU: Qt Quick's VideoOutput (in a QQuickWidget) instead of the QVideoWidget, turned with its own
    orientation property; the frames stay on the GPU (never converted here: converting hardware-decoded frames is
    what crashed). (Not the frames' own rotation setting: QVideoWidget then turns the picture's shape, but not the
    picture, which comes out stretched.)"""

    QML = (b"import QtQuick\nimport QtMultimedia\n"
           b"Rectangle { color: 'black'\n"
           b"  VideoOutput { objectName: 'out'; anchors.fill: parent; fillMode: VideoOutput.PreserveAspectFit } }\n")

    def __init__(self, parent=None, gpu: bool | None = None, turnable: bool = False):
        super().__init__(parent)
        self.gpu = GPU_VIDEO if gpu is None else bool(gpu)
        self.turnable = bool(turnable)
        self._img = QImage()
        self._raw = QImage()                       # (CPU: the frame before the turn)
        self.rotation = 0
        self._out = None                           # (GPU, turnable: the VideoOutput item)
        self._shown: tuple | None = None           # (the frame, the size) -> the frame scaled to fit, for painting
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self.gpu:
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            if self.turnable:
                self.widget = self._quick_output()
                self.sink = self._out.property("videoSink")
            else:
                from PySide6.QtMultimediaWidgets import QVideoWidget
                self.widget = QVideoWidget(self)
                self.widget.setAspectRatioMode(Qt.KeepAspectRatio)
                self.sink = self.widget.videoSink()
            lay.addWidget(self.widget)
            return
        from PySide6.QtMultimedia import QVideoSink
        self.widget = None
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self._frame)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def _frame(self, frame):
        if not frame.isValid():                    # (the end of a clip: its last frame stays, as in QVideoWidget)
            return
        self._raw = frame_image(frame)
        self._turn()

    def _turn(self):
        self._img = self._raw.transformed(QTransform().rotate(self.rotation)) if self.rotation and \
            not self._raw.isNull() else self._raw
        self.update()

    def _quick_output(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtQml import QQmlComponent
        from PySide6.QtQuickWidgets import QQuickWidget
        view = QQuickWidget(self)
        view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        view.setClearColor(QColor(0, 0, 0))
        comp = QQmlComponent(view.engine())
        comp.setData(self.QML, QUrl())
        root = comp.create()
        if root is None:
            raise RuntimeError("Qt Quick video output: " + "; ".join(e.toString() for e in comp.errors()))
        view.setContent(QUrl(), comp, root)
        self._root, self._comp = root, comp        # (kept alive with the view)
        self._out = root.findChild(QObject, "out")
        return view

    def set_rotation(self, degrees: int) -> None:
        """Turn the picture clockwise by 0, 90, 180 or 270 degrees (a view made with turnable=True)."""
        if not self.turnable:
            raise RuntimeError("this video view was not made turnable")
        self.rotation = int(degrees) % 360
        if self.gpu:
            self._out.setProperty("orientation", self.rotation)      # (clockwise, as Qt's documentation says)
        else:
            self._turn()

    def image(self) -> QImage:
        """The frame shown (GPU: converted only when asked)."""
        if self.gpu:
            frame = self.sink.videoFrame()
            return frame.toImage() if frame.isValid() else QImage()
        return self._img

    def clear(self):
        self._img = self._raw = QImage()
        if self.gpu:
            from PySide6.QtMultimedia import QVideoFrame
            self.sink.setVideoFrame(QVideoFrame())
        self.update()

    def sizeHint(self):
        return QSize(640, 360)

    def scaled_frame(self) -> QImage:
        """The frame scaled to fit the view with Qt's smooth (area-averaging) scaling, once per frame and size:
        a painter's bilinear scaling skips most source pixels when it shrinks a lot (a 2K portrait into a small
        preview), and fine detail then shimmers as grain and moving artifacts."""
        dpr = self.devicePixelRatioF()
        target = self._img.size().scaled(self.size() * dpr, Qt.KeepAspectRatio)
        key = (self._img.cacheKey(), target.width(), target.height())
        if self._shown is None or self._shown[0] != key:
            img = self._img if target == self._img.size() else \
                self._img.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img.setDevicePixelRatio(dpr)
            self._shown = (key, img)
        return self._shown[1]

    def paintEvent(self, _e):                     # noqa: N802  (Qt naming)
        if self.gpu:
            return                                 # (the video widget draws)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if not self._img.isNull():
            img = self.scaled_frame()
            w, h = round(img.width() / img.devicePixelRatio()), round(img.height() / img.devicePixelRatio())
            p.drawImage(QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h), img)
        p.end()


def label(text: str = "", *, level: str | None = None, type_: str | None = None, role: str | None = None,
          severity: str | None = None, parent=None) -> QLabel:
    lab = QLabel(text, parent)
    for k, v in (("level", level), ("type", type_), ("role", role), ("severity", severity)):
        if v:
            lab.setProperty(k, v)
    return lab


class _WheelGuard(QObject):
    """The mouse wheel over a list or number field that does not have the keyboard focus scrolls the panel
    around it instead of changing the value (scrolling through the settings must not change them)."""

    @staticmethod
    def has_focus(w: QWidget) -> bool:
        """The field or a child of it (an editable combo's line edit) has the keyboard focus."""
        f = QApplication.focusWidget()
        return w.hasFocus() or (f is not None and w.isAncestorOf(f))

    def eventFilter(self, obj, e):                # noqa: N802  (Qt naming)
        if e.type() == QEvent.Wheel and isinstance(obj, QWidget) and not self.has_focus(obj):
            parent = obj.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent, e)       # Qt passes it on up to the scroll area
            return True
        return False


_GUARD: _WheelGuard | None = None


def guard_wheel(*roots: QWidget) -> None:
    """Give every combo box and spin box in these widgets (and the widgets themselves) the wheel guard; the
    wheel changes a value only after the field was clicked or tabbed into."""
    global _GUARD
    if _GUARD is None:
        _GUARD = _WheelGuard(QApplication.instance())
    for root in roots:
        if root is None:
            continue
        found = [root] + root.findChildren(QComboBox) + root.findChildren(QAbstractSpinBox)
        for w in found:
            if isinstance(w, (QComboBox, QAbstractSpinBox)) and not w.property("wheelGuard"):
                w.setProperty("wheelGuard", True)
                w.setFocusPolicy(Qt.StrongFocus)
                w.installEventFilter(_GUARD)


def badge(text: str, severity: str | None = None) -> QLabel:
    b = label(text, role="badge", severity=severity)
    b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return b


def icon_button(name: str, tip: str, parent=None, checkable: bool = False) -> QToolButton:
    b = QToolButton(parent)
    b.setProperty("role", "iconButton")
    b.setIcon(icon(name))
    b.setIconSize(QSize(16, 16))
    b.setToolTip(tip)
    b.setAutoRaise(True)
    b.setCheckable(checkable)
    return b


def section_bar(title: str, parent=None) -> tuple[QWidget, QHBoxLayout]:
    w = QWidget(parent)
    w.setProperty("role", "sectionBar")
    w.setAttribute(Qt.WA_StyledBackground)
    w.setFixedHeight(theme().sectionHeader)
    h = QHBoxLayout(w)
    h.setContentsMargins(8, 0, 4, 0)
    h.setSpacing(2)
    h.addWidget(label(title, role="sectionTitle"))
    h.addStretch(1)
    return w, h


class FrameTile(QLabel):
    """Fixed-size image tile; FrameService puts frames into it. Click emits the tile's time."""

    clicked = Signal(float)

    def __init__(self, w: int, h: int, parent=None):
        super().__init__(parent)
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("frameTile")
        self.key = ""
        self.t = 0.0
        self._img = QImage()
        self.clear_frame("")

    def clear_frame(self, text: str = "…") -> None:
        self._img = QImage()
        self.setPixmap(QPixmap())
        self.setText(text)

    def has_image(self) -> bool:
        return not self._img.isNull()

    def set_image(self, img: QImage) -> None:
        if img.isNull():
            self.clear_frame("no frame")
            return
        self._img = img
        dpr = self.devicePixelRatioF()
        pm = QPixmap.fromImage(img).scaled(self.size() * dpr, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self.setText("")
        self.setPixmap(pm)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.t)
        super().mousePressEvent(e)
