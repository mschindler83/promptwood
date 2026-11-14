"""Crop a face picture: a square around the face, found by a detector when there is one. Drag the square to move
it, its corner (or the mouse wheel) to resize it; click another found face to crop to that one."""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .. import chain
from .. import faces as F
from .theme import theme
from .widgets import label

MIN_SIDE = 32
SMALL_SIDE = 160           # below this the face picture is soft for the model


class CropView(QWidget):
    """The picture with the square crop (x, y, side in picture pixels) and the found faces."""
    changed = Signal()

    def __init__(self, image: QImage, faces: list, crop: tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.image, self.faces, self.crop = image, faces, crop
        self.setMinimumSize(560, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._shown: QImage | None = None           # the picture scaled to the view (a big photo drags smoothly)
        self._drag: tuple[str, QPointF, tuple[int, int, int]] | None = None

    # picture <-> widget coordinates
    def set_content(self, image: QImage, faces: list, crop: tuple[int, int, int]) -> None:
        """The picture arrives after the view was made (it is read and searched in a worker)."""
        self.image, self.faces, self.crop = image, faces, crop
        self._shown = None
        self.update()
        self.changed.emit()

    def _frame(self) -> tuple[float, float, float]:
        w, h = self.image.width(), self.image.height()
        if not w or not h:
            return 1.0, 0.0, 0.0
        s = min(self.width() / w, self.height() / h)
        return s, (self.width() - w * s) / 2, (self.height() - h * s) / 2

    def _to_widget(self, x, y, side_w, side_h) -> QRectF:
        s, ox, oy = self._frame()
        return QRectF(ox + x * s, oy + y * s, side_w * s, side_h * s)

    def _to_image(self, p: QPointF) -> tuple[float, float]:
        s, ox, oy = self._frame()
        return (p.x() - ox) / s, (p.y() - oy) / s

    def set_crop(self, crop: tuple[int, int, int]) -> None:
        w, h = self.image.width(), self.image.height()
        if not w or not h:
            return
        x, y, side = crop
        side = int(max(min(side, w, h), min(MIN_SIDE, w, h)))
        x = int(min(max(x, 0), w - side))
        y = int(min(max(y, 0), h - side))
        if (x, y, side) != self.crop:
            self.crop = (x, y, side)
            self.update()
            self.changed.emit()

    def _ensure_shown(self) -> None:
        """The picture scaled to the view in device pixels (sharp on a 150 % or 200 % display)."""
        s = self._frame()[0]
        dpr = self.devicePixelRatioF()
        size = QSize(round(self.image.width() * s * dpr), round(self.image.height() * s * dpr))
        if self._shown is None or self._shown.size() != size or self._shown.devicePixelRatio() != dpr:
            self._shown = self.image.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self._shown.setDevicePixelRatio(dpr)

    def paintEvent(self, _e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.image.isNull():
            return
        s, ox, oy = self._frame()
        target = QRectF(ox, oy, self.image.width() * s, self.image.height() * s)
        self._ensure_shown()
        p.drawImage(target.topLeft(), self._shown)
        x, y, side = self.crop
        box = self._to_widget(x, y, side, side)
        shade = QPainterPath()
        shade.addRect(QRectF(ox, oy, self.image.width() * s, self.image.height() * s))
        inner = QPainterPath()
        inner.addRect(box)
        p.fillPath(shade.subtracted(inner), QColor(0, 0, 0, 140))
        for f in self.faces:
            p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.DashLine))
            p.drawRect(self._to_widget(f[0], f[1], f[2] - f[0], f[3] - f[1]))
        p.setPen(QPen(t.accent, 2))
        p.drawRect(box)
        p.fillRect(QRectF(box.right() - 6, box.bottom() - 6, 12, 12), t.accent)

    def _at_corner(self, pos: QPointF) -> bool:
        x, y, side = self.crop
        box = self._to_widget(x, y, side, side)
        return abs(pos.x() - box.right()) < 12 and abs(pos.y() - box.bottom()) < 12

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self.image.isNull():
            return
        pos = e.position()
        ix, iy = self._to_image(pos)
        x, y, side = self.crop
        if self._at_corner(pos):
            self._drag = ("size", pos, self.crop)
        elif x <= ix <= x + side and y <= iy <= y + side:
            self._drag = ("move", pos, self.crop)
        else:                                       # a click on a found face: crop to it
            hit = next((f for f in self.faces if f[0] <= ix <= f[2] and f[1] <= iy <= f[3]), None)
            if hit is not None:
                self.set_crop(F.face_square(hit, self.image.width(), self.image.height()))

    def mouseMoveEvent(self, e):
        if self.image.isNull():
            return
        pos = e.position()
        if self._drag is None:
            x, y, side = self.crop
            inside = self._to_widget(x, y, side, side).contains(pos)
            self.setCursor(Qt.SizeFDiagCursor if self._at_corner(pos) else Qt.SizeAllCursor if inside
                           else Qt.ArrowCursor)
            return
        kind, start, (x, y, side) = self._drag
        s = self._frame()[0]
        dx, dy = (pos.x() - start.x()) / s, (pos.y() - start.y()) / s
        if kind == "move":
            self.set_crop((round(x + dx), round(y + dy), side))
        else:                                       # the corner grows the square; its top-left stays
            limit = min(self.image.width() - x, self.image.height() - y)
            self.set_crop((x, y, min(round(side + (dx + dy) / 2), limit)))

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def wheelEvent(self, e):
        if self.image.isNull():
            return
        x, y, side = self.crop
        w, h = self.image.width(), self.image.height()
        new = round(side * (0.93 if e.angleDelta().y() > 0 else 1.07))
        new = int(max(min(new, w, h), min(MIN_SIDE, w, h)))
        if new == side:                             # at the smallest / largest size: nothing moves
            return
        self.set_crop((round(x + (side - new) / 2), round(y + (side - new) / 2), new))
        e.accept()

    def keyPressEvent(self, e):
        """Arrow keys move the square (Shift: 10 px), + and - resize it (the mouse is not needed)."""
        if self.image.isNull():
            return super().keyPressEvent(e)
        x, y, side = self.crop
        step = 10 if e.modifiers() & Qt.ShiftModifier else 1
        moves = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}
        if e.key() in moves:
            dx, dy = moves[e.key()]
            self.set_crop((x + dx, y + dy, side))
        elif e.key() in (Qt.Key_Plus, Qt.Key_Minus, Qt.Key_Equal):
            new = round(side * (1.07 if e.key() != Qt.Key_Minus else 0.93))
            new = max(new, side + 1) if e.key() != Qt.Key_Minus else min(new, side - 1)
            new = int(max(min(new, self.image.width(), self.image.height()),
                          min(MIN_SIDE, self.image.width(), self.image.height())))
            self.set_crop((round(x + (side - new) / 2), round(y + (side - new) / 2), new))
        else:
            return super().keyPressEvent(e)
        e.accept()


class _Loaded(QObject):
    done = Signal()


class FaceCropDialog(QDialog):
    """crop(): the square (x, y, side) to use, or None for the whole picture; exec() Rejected: cancelled.
    The picture is made upright (ffmpeg) and searched for faces in a worker: the window opens at once ("Looking for
    the face …") and "Use this crop" waits until it is ready."""

    def __init__(self, parent, image_path: Path, model: Path | None, title: str = "Crop the face",
                 crop: tuple[int, int, int] | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 640)
        # one PNG for the view, the detector and the crop: ffmpeg turns a rotated (EXIF) photo upright, as it
        # does when the face picture is written, so the square lands where it is shown
        self._tmp = tempfile.TemporaryDirectory(prefix="video-chain-crop-")
        self._given = crop
        self._closed = False
        self._result: tuple | None = None            # what the worker found, until the window takes it
        self._finished = threading.Event()
        self._loaded = _Loaded()
        self._loaded.done.connect(self._apply)
        self.loading = True
        self.faces: list = []
        self.why: str | None = None
        self._note = ""
        self.whole = False
        v = QVBoxLayout(self)
        self.view = CropView(QImage(), [], crop or (0, 0, 0))
        self.view.changed.connect(self._info)
        v.addWidget(self.view, 1)
        self.info = label("", level="secondary")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        row = QHBoxLayout()
        self.b_face = QPushButton("Crop to face")
        self.b_face.setToolTip("A square around the (largest) face found: its hair and chin, not the scene")
        self.b_face.setEnabled(False)
        self.b_face.clicked.connect(self._crop_to_face)
        self.b_whole = QPushButton("Whole picture")
        self.b_whole.setToolTip("Use the picture as it is, uncropped")
        self.b_whole.clicked.connect(self._use_whole)
        row.addWidget(self.b_face)
        row.addWidget(self.b_whole)
        row.addStretch(1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Use this crop")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        row.addWidget(self.buttons)
        v.addLayout(row)
        self._set_busy(True)
        self.info.setText("Looking for the face …")
        threading.Thread(target=self._work, args=(Path(image_path), model), daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self.loading = busy
        self.b_whole.setEnabled(not busy)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not busy)
        self.setCursor(Qt.BusyCursor if busy else Qt.ArrowCursor)

    def _work(self, image_path: Path, model: Path | None) -> None:
        """Worker thread: the upright picture, then the faces. A window closed meanwhile just drops the result."""
        try:
            png = Path(self._tmp.name) / "picture.png"
            note = ""
            try:
                chain.to_png(image_path, png)
            except Exception:  # noqa: BLE001
                png = image_path
                note = "the picture could not be turned upright (a rotated photo may show sideways)"
            image = QImage(str(png))
            faces: list = []
            why = None
            if image.isNull():
                why = f"not an image: {image_path.name}"
            else:
                why = F.available(model)
                if why is None and not self._closed:
                    try:
                        faces = F.detect(png, model)
                    except Exception as e:  # noqa: BLE001
                        why = f"the face detector failed: {e}"
            self._result = (image, faces, why, note)
        except Exception as e:  # noqa: BLE001  (the temporary folder removed by a close, ...)
            self._result = (QImage(), [], f"the picture could not be read: {e}", "")
        self._finished.set()
        try:
            self._loaded.done.emit()
        except RuntimeError:
            pass

    def _apply(self) -> None:
        if self._result is None or self._closed:
            return
        image, self.faces, why, self._note = self._result
        self._result = None
        if image.isNull():
            self.view.set_content(image, [], (0, 0, 0))
            self.info.setText(why or "not an image")
            self.setCursor(Qt.ArrowCursor)
            self.loading = False
            return
        self.why = why
        w, h = image.width(), image.height()
        start = self._given or (F.face_square(self.faces[0], w, h) if self.faces else F.centre_square(w, h))
        self.view.set_content(image, self.faces, start)
        self.view.set_crop(start)
        self.b_face.setEnabled(bool(self.faces))
        self._set_busy(False)
        self._info()

    def _crop_to_face(self) -> None:
        self.view.set_crop(F.face_square(self.faces[0], self.view.image.width(), self.view.image.height()))

    def _info(self):
        if self.loading or self.view.image.isNull():
            return
        side = self.view.crop[2]
        n = len(self.faces)
        note = f"Note: {self._note}. " if self._note else ""
        if self.why:
            text = f"No face detection: {self.why}. Set the square by hand."
        elif n == 0:
            text = "No face found. Set the square around the face by hand."
        elif n == 1:
            text = f"Face found ({self.faces[0][4]:.0%} sure)."
        else:
            text = f"{n} faces found (dashed): click one to crop to it."
        text += (f" Crop: {side} × {side} px. Drag the square, its corner or use the wheel to change it; a tight "
                 "crop of the face keeps the model from copying the picture's scene or pose.")
        if side < SMALL_SIDE:
            text += f" Small: below {SMALL_SIDE} px the face picture is soft; a bigger source works better."
        self.info.setText(note + text)

    def done(self, r):
        self._closed = True
        super().done(r)
        self._tmp.cleanup()

    def _use_whole(self):
        self.whole = True
        self.accept()

    def crop(self) -> tuple[int, int, int] | None:
        if self.loading and not self._closed:             # asked before the picture is ready (not by the buttons)
            self._finished.wait(60)
            self._apply()
        if self.whole:
            return None
        x, y, side = self.view.crop
        return int(x), int(y), int(side)
