"""Paint the part of an image an edit draws anew (inpainting): white in the mask = drawn anew, black = kept.

The image is shown as large as fits, the mask over it in translucent red. Left mouse paints, right mouse (or the
Erase tool) takes it away; Ctrl + wheel changes the brush. The mask is kept at the image's own pixel size.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox, QHBoxLayout, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from .widgets import label

MASK_COLOUR = QColor(255, 40, 40, 120)


def empty_mask(size) -> QImage:
    m = QImage(size, QImage.Format_Grayscale8)
    m.fill(0)
    return m


def mask_area(mask: QImage | None) -> float:
    """The share of the image that is painted (0..1), on a coarse grid (fast)."""
    if mask is None or mask.isNull():
        return 0.0
    small = mask.scaled(128, 128, Qt.IgnoreAspectRatio, Qt.FastTransformation).convertToFormat(
        QImage.Format_Grayscale8)
    bits = small.constBits()
    row = small.bytesPerLine()
    n = sum(1 for y in range(small.height()) for x in range(small.width()) if bits[y * row + x] >= 128)
    return n / (small.width() * small.height())


def overlay(image: QImage, mask: QImage | None) -> QImage:
    """The image with the mask over it in translucent red (for previews)."""
    out = image.convertToFormat(QImage.Format_ARGB32)
    if mask is None or mask.isNull() or mask_area(mask) == 0:
        return out
    tint = QImage(out.size(), QImage.Format_ARGB32)
    tint.fill(MASK_COLOUR)
    alpha = mask.scaled(out.size()).convertToFormat(QImage.Format_Alpha8)
    p = QPainter(tint)
    p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    p.drawImage(0, 0, alpha)
    p.end()
    p = QPainter(out)
    p.drawImage(0, 0, tint)
    p.end()
    return out


class MaskCanvas(QWidget):
    """The image and its mask; paints with a round brush in image pixels."""
    changed = Signal()

    def __init__(self, image: QImage, mask: QImage | None = None, parent=None):
        super().__init__(parent)
        self.image = image
        self.mask = mask.copy() if mask is not None and mask.size() == image.size() else empty_mask(image.size())
        self.mask = self.mask.convertToFormat(QImage.Format_Grayscale8)
        self.brush = max(4, min(image.width(), image.height()) // 20)      # diameter in image pixels
        self.erase = False
        self._undo: list[QImage] = []
        self._last: QPointF | None = None
        self._stroke_erase = False
        self._cursor: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(480, 320)
        self.setCursor(Qt.CrossCursor)

    # ---- geometry
    def _frame(self) -> QRectF:
        w, h = self.width(), self.height()
        s = min(w / self.image.width(), h / self.image.height())
        iw, ih = self.image.width() * s, self.image.height() * s
        return QRectF((w - iw) / 2, (h - ih) / 2, iw, ih)

    def _to_image(self, pos: QPointF) -> QPointF:
        f = self._frame()
        return QPointF((pos.x() - f.x()) / f.width() * self.image.width(),
                       (pos.y() - f.y()) / f.height() * self.image.height())

    # ---- drawing
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 22))
        f = self._frame()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(f, overlay(self.image, self.mask))
        if self._cursor is not None:
            r = self.brush / 2 * f.width() / self.image.width()
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.5, Qt.DashLine if self.erase else Qt.SolidLine))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self._cursor, r, r)

    def _dab(self, a: QPointF, b: QPointF, erase: bool):
        p = QPainter(self.mask)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0) if erase else QColor(255, 255, 255), self.brush, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawLine(a, b)
        p.end()

    def mousePressEvent(self, e):
        if e.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self._undo.append(self.mask.copy())
        del self._undo[:-30]
        self._stroke_erase = self.erase or e.button() == Qt.RightButton
        self._last = self._to_image(e.position())
        self._dab(self._last, self._last, self._stroke_erase)
        self.update()

    def mouseMoveEvent(self, e):
        self._cursor = e.position()
        if self._last is not None:
            pt = self._to_image(e.position())
            self._dab(self._last, pt, self._stroke_erase)
            self._last = pt
        self.update()

    def mouseReleaseEvent(self, _e):
        if self._last is not None:
            self._last = None
            self.changed.emit()

    def leaveEvent(self, _e):
        self._cursor = None
        self.update()

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            k = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
            self.set_brush(round(self.brush * k))
            e.accept()
        else:
            super().wheelEvent(e)

    # ---- tools
    def set_brush(self, d: int):
        self.brush = max(2, min(int(d), max(self.image.width(), self.image.height())))
        self.update()
        self.changed.emit()

    def clear(self):
        self._undo.append(self.mask.copy())
        self.mask.fill(0)
        self.update()
        self.changed.emit()

    def invert(self):
        self._undo.append(self.mask.copy())
        self.mask.invertPixels()
        self.update()
        self.changed.emit()

    def undo(self):
        if self._undo:
            self.mask = self._undo.pop()
            self.update()
            self.changed.emit()


class MaskPainterDialog(QDialog):
    """Paint a mask over image (a QImage at its own size). .mask: the result (Format_Grayscale8)."""

    def __init__(self, parent, image: QImage, mask: QImage | None = None):
        super().__init__(parent)
        self.setWindowTitle("Paint the part to draw anew")
        self.resize(1100, 800)
        v = QVBoxLayout(self)
        v.addWidget(label("Paint over what should change (red). The rest stays exactly as it is. Left mouse "
                          "paints, right mouse erases; Ctrl + wheel: brush size; Ctrl+Z: undo.", level="secondary"))
        self.canvas = MaskCanvas(image, mask)
        v.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.b_paint, self.b_erase = QPushButton("Paint"), QPushButton("Erase")
        tools = QButtonGroup(self)
        for b in (self.b_paint, self.b_erase):
            b.setCheckable(True)
            tools.addButton(b)
            row.addWidget(b)
        self.b_paint.setChecked(True)
        self.b_paint.toggled.connect(lambda on: setattr(self.canvas, "erase", not on))
        row.addSpacing(12)
        row.addWidget(label("Brush", level="tertiary"))
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(2, max(8, min(image.width(), image.height()) // 2))
        self.size.setValue(self.canvas.brush)
        self.size.setMinimumWidth(180)
        self.size.valueChanged.connect(self.canvas.set_brush)
        row.addWidget(self.size)
        self.size_text = label("", level="tertiary")
        row.addWidget(self.size_text)
        row.addSpacing(12)
        for text, fn in (("Undo", self.canvas.undo), ("Clear", self.canvas.clear), ("Invert", self.canvas.invert)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.area = label("", level="tertiary")
        row.addWidget(self.area)
        v.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Use this mask")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)
        QShortcut(QKeySequence.Undo, self, activated=self.canvas.undo)
        self.canvas.changed.connect(self._show_area)
        self._show_area()

    def _show_area(self):
        self.size_text.setText(f"{self.canvas.brush} px")
        a = mask_area(self.canvas.mask)
        self.area.setText(f"{a * 100:.0f} % of the image drawn anew" if a else "nothing painted yet")
        if self.size.value() != self.canvas.brush:
            self.size.blockSignals(True)
            self.size.setValue(min(self.size.maximum(), self.canvas.brush))
            self.size.blockSignals(False)

    @property
    def mask(self) -> QImage:
        return self.canvas.mask
