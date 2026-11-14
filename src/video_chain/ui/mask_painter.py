"""Paint the part of an image an edit draws anew (inpainting): white in the mask = drawn anew, black = kept.

The image is shown as large as fits, the mask over it in translucent red. Left mouse paints, right mouse (or the
Erase tool) takes it away; Ctrl + wheel changes the brush. The mask is kept at the image's own pixel size.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QDialog, QHBoxLayout, QPushButton, 
                               QSlider, QVBoxLayout, QWidget)

from .dialogs import button_row, confirm
from .widgets import fit_size, label, smooth_scaled, shortcut

MASK_COLOUR = QColor(255, 40, 40, 120)
UNDO_STEPS = 30                 # steps kept; each holds only the area it changed (Clear and Invert: the whole mask)
UNDO_BYTES = 96 * 1024 * 1024    # and never more than this much memory


def empty_mask(size) -> QImage:
    m = QImage(size, QImage.Format.Format_Grayscale8)
    m.fill(0)
    return m


_HALF = bytes(0 if v < 128 else 1 for v in range(256))     # pixel value -> painted (white half) or not


def mask_area(mask: QImage | None) -> float:
    """The share of the image that is painted (0..1), every pixel counted (a dot or a thin line is not 0)."""
    if mask is None or mask.isNull():
        return 0.0
    m = mask.convertToFormat(QImage.Format.Format_Grayscale8)
    data, row, w = bytes(m.constBits()), m.bytesPerLine(), m.width()
    n = sum(data[y * row:y * row + w].translate(_HALF).count(1) for y in range(m.height()))
    return n / (w * m.height())


def tinted(mask: QImage, size) -> QImage:
    """The mask as translucent red at size (white -> red, black -> clear)."""
    alpha = mask.scaled(size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation).convertToFormat(
        QImage.Format.Format_Grayscale8)
    tint = QImage(alpha.size(), QImage.Format.Format_ARGB32)
    tint.fill(QColor(MASK_COLOUR.red(), MASK_COLOUR.green(), MASK_COLOUR.blue()))
    tint.setAlphaChannel(alpha)                    # (a grayscale picture: its values become the alpha)
    return tint


def overlay(image: QImage, mask: QImage | None) -> QImage:
    """The image with the mask over it in translucent red (for previews)."""
    out = image.convertToFormat(QImage.Format.Format_ARGB32)
    if mask is None or mask.isNull() or mask_area(mask) == 0:
        return out
    p = QPainter(out)
    p.setOpacity(MASK_COLOUR.alphaF())
    p.drawImage(0, 0, tinted(mask, out.size()))
    p.end()
    return out


class MaskCanvas(QWidget):
    """The image and its mask; paints with a round brush in image pixels."""
    changed = Signal()
    # (the white-is-drawn picture; the name shadows QWidget.mask(), which this app never calls)
    mask: QImage  # type: ignore[assignment]

    def __init__(self, image: QImage, mask: QImage | None = None, parent=None):
        super().__init__(parent)
        self.image = image
        self.mask = mask.copy() if mask is not None and mask.size() == image.size() else empty_mask(image.size())
        self.mask = self.mask.convertToFormat(QImage.Format.Format_Grayscale8)
        self.brush = max(4, min(image.width(), image.height()) // 20)      # diameter in image pixels
        self.erase = False
        self._undo: list[tuple[QRect, QImage]] = []     # (area, the mask's pixels there before the step)
        self._redo: list[tuple[QRect, QImage]] = []
        self._dirty: QRect | None = None            # the area the stroke in progress touched
        self._last: QPointF | None = None
        self._stroke_erase = False
        self._cursor: QPointF | None = None
        self._before: QImage | None = None
        self._stroke_button = Qt.MouseButton.NoButton
        self._shown: tuple | None = None           # (frame size, display picture) of the image
        self._tint: QImage | None = None           # the mask in red at the frame's size (redone after a change)
        self.setMouseTracking(True)
        self.setMinimumSize(480, 320)
        self.setCursor(Qt.CursorShape.CrossCursor)

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
    def _ensure_display(self) -> None:
        """The image and the red mask at the frame's size. Strokes paint into the red picture themselves (`_dab`), so
        it is made anew only when the size changes or the mask is replaced (undo, clear, invert)."""
        size = self._frame().size().toSize()
        dpr = self.devicePixelRatioF()
        dev = QSize(round(size.width() * dpr), round(size.height() * dpr))     # (device pixels: sharp on HiDPI)
        if self._shown is None or self._shown[0] != (size, dpr):   # the image at the frame's size, once per size
            shown = QImage(smooth_scaled(self.image, dev, Qt.AspectRatioMode.IgnoreAspectRatio))
            shown.setDevicePixelRatio(dpr)
            self._shown = ((size, dpr), shown)
            self._tint = None
        if self._tint is None or self._tint.size() != dev:
            self._tint = tinted(self.mask, dev)
            self._tint.setDevicePixelRatio(dpr)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 22))
        f = self._frame()
        self._ensure_display()
        assert self._shown is not None and self._tint is not None      # (made by _ensure_display)
        p.drawImage(f.topLeft(), self._shown[1])
        p.setOpacity(MASK_COLOUR.alphaF())
        p.drawImage(f.topLeft(), self._tint)
        p.setOpacity(1.0)
        if self._cursor is not None:
            r = self.brush / 2 * f.width() / self.image.width()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.5,
                          Qt.PenStyle.DashLine if self.erase else Qt.PenStyle.SolidLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(self._cursor, r, r)

    def _dab(self, a: QPointF, b: QPointF, erase: bool):
        p = QPainter(self.mask)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0) if erase else QColor(255, 255, 255), self.brush, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawLine(a, b)
        p.end()
        pad = int(self.brush / 2) + 2
        box = QRect(int(min(a.x(), b.x())) - pad, int(min(a.y(), b.y())) - pad,
                    int(abs(a.x() - b.x())) + 2 * pad + 1, int(abs(a.y() - b.y())) + 2 * pad + 1)
        box = box.intersected(self.mask.rect())
        self._dirty = box if self._dirty is None else self._dirty.united(box)
        if self._tint is not None:                   # the same line into the red picture (no full re-scale)
            k = self._tint.width() / self._tint.devicePixelRatio() / self.image.width()    # (logical units)
            t = QPainter(self._tint)
            t.setRenderHint(QPainter.RenderHint.Antialiasing)
            t.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear if erase else QPainter.CompositionMode.CompositionMode_Source)
            t.setPen(QPen(QColor(MASK_COLOUR.red(), MASK_COLOUR.green(), MASK_COLOUR.blue()), self.brush * k,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            t.drawLine(QPointF(a.x() * k, a.y() * k), QPointF(b.x() * k, b.y() * k))
            t.end()

    def mousePressEvent(self, e):
        if e.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) or self._last is not None:
            return                                     # (a second button during a stroke: ignored)
        self._stroke_button = e.button()
        self._begin_stroke(self._to_image(e.position()), self.erase or e.button() == Qt.MouseButton.RightButton)
        self.update()

    def _begin_stroke(self, pos: QPointF, erase: bool) -> None:
        self._before = self.mask.copy()               # (kept only for the area the stroke touches, see below)
        self._dirty = None
        self._stroke_erase = erase
        self._last = pos
        self._dab(pos, pos, erase)

    def _stroke_to(self, pt: QPointF) -> None:
        assert self._last is not None                  # (a stroke is in progress)
        self._dab(self._last, pt, self._stroke_erase)
        self._last = pt

    def _finish_stroke(self) -> None:
        self._last = None
        if self._before is not None and self._dirty is not None:
            old, now = self._before.copy(self._dirty), self.mask.copy(self._dirty)
            if old != now:                            # (a stroke that changed nothing: no undo)
                self._push_undo(self._dirty, old)
        self._before, self._dirty = None, None
        self.changed.emit()

    def mouseMoveEvent(self, e):
        self._cursor = e.position()
        if self._last is not None:
            self._stroke_to(self._to_image(e.position()))
        self.update()

    def mouseReleaseEvent(self, e):
        if self._last is not None and e.button() == self._stroke_button:
            self._finish_stroke()

    def _push_undo(self, rect: QRect, before: QImage) -> None:
        """One undo step; every step goes through here (the same caps), and a new step ends the redo line."""
        self._undo.append((rect, before))
        self._redo.clear()
        del self._undo[:-UNDO_STEPS]
        while len(self._undo) > 1 and sum(img.sizeInBytes() for _r, img in self._undo) > UNDO_BYTES:
            del self._undo[0]

    def leaveEvent(self, _e):
        self._cursor = None
        self.update()

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if e.angleDelta().y() > 0:
                self.set_brush(max(self.brush + 1, round(self.brush * 1.15)))   # (a 2 px brush grows too)
            else:
                self.set_brush(min(self.brush - 1, round(self.brush / 1.15)))
            e.accept()
        else:
            super().wheelEvent(e)

    # ---- tools
    def set_brush(self, d: int):
        self.brush = max(2, min(int(d), max(self.image.width(), self.image.height())))
        self.update()
        self.changed.emit()

    def clear(self):
        if mask_area(self.mask) == 0:
            return
        self._push_undo(self.mask.rect(), self.mask.copy())
        self.mask.fill(0)
        self._tint = None
        self.update()
        self.changed.emit()

    def invert(self):
        self._push_undo(self.mask.rect(), self.mask.copy())
        self.mask.invertPixels()
        self._tint = None
        self.update()
        self.changed.emit()

    def _swap(self, take: list, give: list) -> None:
        """Put the last step of `take` back and keep what it replaced in `give` (undo and redo are the same move)."""
        if not take:
            return
        rect, img = take.pop()
        give.append((rect, self.mask.copy(rect)))
        p = QPainter(self.mask)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(rect.topLeft(), img)
        p.end()
        self._tint = None
        self.update()
        self.changed.emit()

    def undo(self):
        self._swap(self._undo, self._redo)

    def redo(self):
        self._swap(self._redo, self._undo)


class MaskPainterDialog(QDialog):
    """Paint a mask over image (a QImage at its own size). .mask: the result (Format_Grayscale8)."""
    size: QSlider  # type: ignore[assignment]   # (the brush size; the name shadows QWidget.size(), never called here)

    def __init__(self, parent, image: QImage, mask: QImage | None = None):
        super().__init__(parent)
        self.setWindowTitle("Paint the part to draw anew")
        fit_size(self, 1100, 800)
        v = QVBoxLayout(self)
        v.addWidget(label("Paint over what should change (red). The rest stays exactly as it is. Left mouse "
                          "paints, right mouse erases; Ctrl + wheel or [ and ]: brush size; B / E: paint / erase; Ctrl+Z: undo, "
                          "Ctrl+Shift+Z: redo.", level="secondary"))
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
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(2, max(8, min(image.width(), image.height()) // 2))
        self.size.setValue(self.canvas.brush)
        self.size.setMinimumWidth(180)
        self.size.valueChanged.connect(self.canvas.set_brush)
        row.addWidget(self.size)
        self.size_text = label("", level="tertiary")
        row.addWidget(self.size_text)
        row.addSpacing(12)
        for text, fn in (("Undo", self.canvas.undo), ("Redo", self.canvas.redo), ("Clear", self.canvas.clear),
                         ("Invert", self.canvas.invert)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.area = label("", level="tertiary")
        row.addWidget(self.area)
        v.addLayout(row)
        buttons, _ok = button_row(self, "Use this mask")
        v.addWidget(buttons)
        shortcut(QKeySequence.StandardKey.Undo, self, self.canvas.undo)
        for key in (QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")):
            shortcut(key, self, self.canvas.redo)
        shortcut(QKeySequence("["), self, lambda: self.canvas.set_brush(round(self.canvas.brush / 1.15) - 1))
        shortcut(QKeySequence("]"), self, lambda: self.canvas.set_brush(round(self.canvas.brush * 1.15) + 1))
        shortcut(QKeySequence("B"), self, self.b_paint.click)
        shortcut(QKeySequence("E"), self, self.b_erase.click)
        self._start = self.canvas.mask.copy()            # (to tell, when closing, whether work would be lost)
        self.canvas.changed.connect(self._show_area)
        self._show_area()

    def reject(self):
        """Cancel, Esc, the window's close button: painted work is only dropped after asking."""
        if self.canvas.mask != self._start and not confirm(
                self, "Mask", "Discard the changes to the mask?"):
            return
        super().reject()

    def _show_area(self):
        self.size_text.setText(f"{self.canvas.brush} px")
        a = mask_area(self.canvas.mask)
        self.area.setText((f"{a * 100:.0f} %" if a >= 0.01 else "under 1 %") + " of the image drawn anew"
                          if a else "nothing painted yet")
        if self.size.value() != self.canvas.brush:
            self.size.blockSignals(True)
            self.size.setValue(min(self.size.maximum(), self.canvas.brush))
            self.size.blockSignals(False)

    @property
    def mask(self) -> QImage:  # type: ignore[override]   # (shadows QWidget.mask(), which this app never uses)
        return self.canvas.mask
