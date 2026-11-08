"""Crop a face picture: a square around the face, found by a detector when there is one. Drag the square to move
it, its corner (or the mouse wheel) to resize it; click another found face to crop to that one."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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
        self._shown: QImage | None = None           # the picture scaled to the view (a big photo drags smoothly)
        self._drag: tuple[str, QPointF, tuple[int, int, int]] | None = None

    # picture <-> widget coordinates
    def _frame(self) -> tuple[float, float, float]:
        w, h = self.image.width(), self.image.height()
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
        x, y, side = crop
        side = int(max(min(side, w, h), min(MIN_SIDE, w, h)))
        x = int(min(max(x, 0), w - side))
        y = int(min(max(y, 0), h - side))
        if (x, y, side) != self.crop:
            self.crop = (x, y, side)
            self.update()
            self.changed.emit()

    def paintEvent(self, _e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s, ox, oy = self._frame()
        target = QRectF(ox, oy, self.image.width() * s, self.image.height() * s)
        if self._shown is None or self._shown.size() != target.size().toSize():
            self._shown = self.image.scaled(target.size().toSize(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
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
        if e.button() != Qt.LeftButton:
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
        pos = e.position()
        if self._drag is None:
            self.setCursor(Qt.SizeFDiagCursor if self._at_corner(pos) else Qt.ArrowCursor)
            return
        kind, start, (x, y, side) = self._drag
        s = self._frame()[0]
        dx, dy = (pos.x() - start.x()) / s, (pos.y() - start.y()) / s
        if kind == "move":
            self.set_crop((round(x + dx), round(y + dy), side))
        else:                                       # the corner grows the square; its top-left stays
            limit = min(self.image.width() - x, self.image.height() - y)
            self.set_crop((x, y, min(round(side + max(dx, dy)), limit)))

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def wheelEvent(self, e):
        x, y, side = self.crop
        w, h = self.image.width(), self.image.height()
        new = round(side * (0.93 if e.angleDelta().y() > 0 else 1.07))
        new = int(max(min(new, w, h), min(MIN_SIDE, w, h)))
        if new == side:                             # at the smallest / largest size: nothing moves
            return
        self.set_crop((round(x + (side - new) / 2), round(y + (side - new) / 2), new))


class FaceCropDialog(QDialog):
    """crop(): the square (x, y, side) to use, or None for the whole picture; exec() Rejected: cancelled."""

    def __init__(self, parent, image_path: Path, model: Path | None, title: str = "Crop the face",
                 crop: tuple[int, int, int] | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 640)
        # one PNG for the view, the detector and the crop: ffmpeg turns a rotated (EXIF) photo upright, as it
        # does when the face picture is written, so the square lands where it is shown
        self._tmp = tempfile.TemporaryDirectory(prefix="video-chain-crop-")
        png = Path(self._tmp.name) / "picture.png"
        try:
            chain.to_png(Path(image_path), png)
        except Exception:  # noqa: BLE001
            png = Path(image_path)
        image = QImage(str(png))
        if image.isNull():
            self._tmp.cleanup()
            raise ValueError(f"not an image: {Path(image_path).name}")
        w, h = image.width(), image.height()
        why = F.available(model)
        self.faces: list = []
        if why is None:
            try:
                self.faces = F.detect(png, model)
            except Exception as e:  # noqa: BLE001
                why = f"the face detector failed: {e}"
        self.why = why
        self.whole = False
        start = crop or (F.face_square(self.faces[0], w, h) if self.faces else F.centre_square(w, h))
        v = QVBoxLayout(self)
        self.view = CropView(image, self.faces, start)
        self.view.set_crop(start)
        self.view.changed.connect(self._info)
        v.addWidget(self.view, 1)
        self.info = label("", level="secondary")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        row = QHBoxLayout()
        self.b_face = QPushButton("Crop to face")
        self.b_face.setToolTip("A square around the (largest) face found: its hair and chin, not the scene")
        self.b_face.setEnabled(bool(self.faces))
        self.b_face.clicked.connect(lambda: self.view.set_crop(F.face_square(self.faces[0], w, h)))
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
        self._info()

    def _info(self):
        side = self.view.crop[2]
        n = len(self.faces)
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
        self.info.setText(text)

    def done(self, r):
        super().done(r)
        self._tmp.cleanup()

    def _use_whole(self):
        self.whole = True
        self.accept()

    def crop(self) -> tuple[int, int, int] | None:
        return None if self.whole else tuple(int(v) for v in self.view.crop)
