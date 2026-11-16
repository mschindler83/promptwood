"""The cut of a start image whose shape is not the clips' shape: shown on the picture (cut_image) and set in a
window (StartCropDialog). The frame always has the clips' shape: drag it to move it, drag a corner or turn the mouse
wheel to make it smaller or bigger. Only what is inside the frame goes into the clips (tree/crops.py)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .. import tree as T
from .widgets import fit_size, label, remember_geometry

SHADE = QColor(0, 0, 0, 150)          # over the parts of the picture that are cut away
FRAME = QColor(255, 255, 255)
HANDLE = 12                           # pixels: a corner is grabbed this close to it


def shape_text(w: int, h: int) -> str:
    """'9:16' for a size of a listed shape, else the ratio ('1.78:1')."""
    for key, (a, b) in T.ASPECTS.items():
        if abs(w / h - a / b) <= T.crops.MISMATCH * (a / b) / 2:
            return key
    return f"{w / h:.2f}:1"


def cut_image(img: QImage, width: int, height: int, crop=None) -> QImage:
    """The picture with the parts a canvas of width x height cuts away shaded and the cut framed (img itself when
    nothing is cut away)."""
    if img.isNull() or not T.mismatched(img.width(), img.height(), width, height):
        return img
    x, y, w, h = T.crop_box(img.width(), img.height(), width, height, crop)
    out = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(out)
    keep = QRectF(x, y, w, h)
    for r in (QRectF(0, 0, out.width(), y), QRectF(0, y + h, out.width(), out.height() - y - h),
              QRectF(0, y, x, h), QRectF(x + w, y, out.width() - x - w, h)):
        if r.width() > 0 and r.height() > 0:
            p.fillRect(r, SHADE)
    pen = QPen(FRAME)
    pen.setWidthF(max(2.0, out.width() / 160))
    p.setPen(pen)
    p.drawRect(keep)
    p.end()
    return out


class CutView(QWidget):
    """The picture with the frame of the cut (crop: {"cx", "cy", "zoom"}, as tree.crops stores it)."""
    changed = Signal()

    def __init__(self, image: QImage, width: int, height: int, crop: dict | None, parent=None):
        super().__init__(parent)
        self.image, self.canvas = image, (int(width), int(height))
        c = T.crops.clean(crop) or {"cx": 0.5, "cy": 0.5, "zoom": 1.0}
        self.crop = {"cx": c["cx"], "cy": c["cy"], "zoom": c["zoom"]}
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._drag: tuple[str, QPointF, dict] | None = None

    def box(self) -> tuple[int, int, int, int]:
        """The cut in the picture's pixels (as it is used)."""
        return T.crop_box(self.image.width(), self.image.height(), *self.canvas, self.crop)

    def set_crop(self, cx: float, cy: float, zoom: float) -> None:
        """The frame moved / resized; the centre is kept where the frame really is (inside the picture)."""
        self.crop = {"cx": cx, "cy": cy, "zoom": min(1.0, max(T.crops.MIN_ZOOM, zoom))}
        x, y, w, h = self.box()
        iw, ih = max(1, self.image.width()), max(1, self.image.height())
        self.crop.update(cx=(x + w / 2) / iw, cy=(y + h / 2) / ih)
        self.update()
        self.changed.emit()

    # picture <-> widget coordinates
    def _frame(self) -> tuple[float, float, float]:
        w, h = self.image.width(), self.image.height()
        if not w or not h:
            return 1.0, 0.0, 0.0
        s = min(self.width() / w, self.height() / h)
        return s, (self.width() - w * s) / 2, (self.height() - h * s) / 2

    def _box_rect(self) -> QRectF:
        s, ox, oy = self._frame()
        x, y, w, h = self.box()
        return QRectF(ox + x * s, oy + y * s, w * s, h * s)

    def paintEvent(self, e):                       # noqa: N802  (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        s, ox, oy = self._frame()
        pic = QRectF(ox, oy, self.image.width() * s, self.image.height() * s)
        p.drawImage(pic, self.image)
        r = self._box_rect()
        for part in (QRectF(pic.left(), pic.top(), pic.width(), r.top() - pic.top()),
                     QRectF(pic.left(), r.bottom(), pic.width(), pic.bottom() - r.bottom()),
                     QRectF(pic.left(), r.top(), r.left() - pic.left(), r.height()),
                     QRectF(r.right(), r.top(), pic.right() - r.right(), r.height())):
            if part.width() > 0 and part.height() > 0:
                p.fillRect(part, SHADE)
        pen = QPen(FRAME)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(r)
        for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
            p.fillRect(QRectF(c.x() - 4, c.y() - 4, 8, 8), FRAME)
        p.end()

    def _corner(self, pos: QPointF) -> bool:
        r = self._box_rect()
        return any(abs(pos.x() - c.x()) <= HANDLE and abs(pos.y() - c.y()) <= HANDLE
                   for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()))

    def mousePressEvent(self, e):                  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pos = e.position()
        if self._corner(pos):
            self._drag = ("size", pos, dict(self.crop))
        elif self._box_rect().contains(pos):
            self._drag = ("move", pos, dict(self.crop))

    def mouseMoveEvent(self, e):                   # noqa: N802
        pos = e.position()
        if self._drag is None:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor if self._corner(pos) else
                           Qt.CursorShape.SizeAllCursor if self._box_rect().contains(pos) else
                           Qt.CursorShape.ArrowCursor)
            return
        kind, start, was = self._drag
        s, _ox, _oy = self._frame()
        iw, ih = max(1, self.image.width()), max(1, self.image.height())
        if kind == "move":
            self.set_crop(was["cx"] + (pos.x() - start.x()) / s / iw, was["cy"] + (pos.y() - start.y()) / s / ih,
                          was["zoom"])
        else:                                      # a corner: the frame grows / shrinks around its centre
            r = self._box_rect()
            centre = r.center()
            old = max(1.0, abs(start.x() - centre.x()))
            self.set_crop(was["cx"], was["cy"], was["zoom"] * abs(pos.x() - centre.x()) / old)

    def mouseReleaseEvent(self, e):                # noqa: N802
        self._drag = None

    def wheelEvent(self, e):                       # noqa: N802
        steps = e.angleDelta().y() / 120
        if steps:
            self.set_crop(self.crop["cx"], self.crop["cy"], self.crop["zoom"] * (0.95 ** steps))


class StartCropDialog(QDialog):
    """Where the clips' frame lies on a start image. crop(): {"cx", "cy", "zoom"} to store (tree.set_crop).
    title / note: what the window is about (e.g. 'Start image 1 of 2: clip #ab12'); cancel_text: the button that
    closes it without a cut (at a start: 'Don't start')."""

    def __init__(self, parent, picture: Path, width: int, height: int, crop: dict | None = None,
                 title: str = "The start image's cut", note: str = "", cancel_text: str = "Cancel",
                 settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        fit_size(self, 860, 640)
        remember_geometry(self, settings, "startCut")
        img = QImage(str(picture))
        self.view = CutView(img, width, height, crop)
        v = QVBoxLayout(self)
        pw, ph = img.width(), img.height()
        info = label((note + "\n" if note else "") +
                     f"The clips are {width}×{height} ({shape_text(width, height)}); this picture is {pw}×{ph} "
                     f"({shape_text(pw, ph) if pw and ph else '?'}). Only the part inside the frame is used: drag "
                     "the frame to move it, drag a corner or turn the mouse wheel to make it smaller (closer) or "
                     "bigger.", level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        v.addWidget(self.view, 1)
        self.size_note = label("", level="tertiary")
        v.addWidget(self.size_note)
        row = QHBoxLayout()
        self.b_centre = QPushButton("Centre")
        self.b_centre.setToolTip("Move the frame to the middle of the picture (its size stays)")
        self.b_centre.clicked.connect(lambda: self.view.set_crop(0.5, 0.5, self.view.crop["zoom"]))
        self.b_largest = QPushButton("Largest")
        self.b_largest.setToolTip("Make the frame as big as the picture allows (its centre stays)")
        self.b_largest.clicked.connect(lambda: self.view.set_crop(self.view.crop["cx"], self.view.crop["cy"], 1.0))
        row.addWidget(self.b_centre)
        row.addWidget(self.b_largest)
        row.addStretch(1)
        self.b_cancel = QPushButton(cancel_text)
        self.b_cancel.clicked.connect(self.reject)
        self.b_use = QPushButton("Use this cut")
        self.b_use.setDefault(True)
        self.b_use.clicked.connect(self.accept)
        row.addWidget(self.b_cancel)
        row.addWidget(self.b_use)
        v.addLayout(row)
        self.view.changed.connect(self._note)
        self._note()

    def _note(self):
        x, y, w, h = self.view.box()
        cw = self.view.canvas[0]
        soft = " · smaller than the clips: the first frame starts soft" if w < cw else ""
        self.size_note.setText(f"Cut: {w}×{h} at {x}, {y}{soft}")

    def crop(self) -> dict:
        return dict(self.view.crop)
