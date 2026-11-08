"""The selected path as a timeline: one block per clip, as wide as the clip is long in the joined video, with its
state colour, a thumbnail and its number; carry-over frames (the part repeated from the clip before) hatched, real
videos in their own colour. Click a block to show that clip in the inspector, double-click to play it, right-click
for its menu."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QScrollArea, QSizePolicy, QWidget

from .graph import status_colour
from .theme import theme, with_alpha

RULER = 16
MIN_BLOCK = 46


@dataclass
class Block:
    nid: str
    label: str              # "#5"
    secs: float             # its length in the joined path
    state: str              # done / changed / new / generating ...
    is_video: bool
    pixmap: QPixmap | None
    lead_secs: float = 0.0  # carry-over from the clip before (hatched)
    tip: str = ""


class _Lane(QWidget):
    clicked = Signal(str)
    activated = Signal(str)
    menu = Signal(str, QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blocks: list[Block] = []
        self.focus: str | None = None
        self.px_per_sec = 40.0
        self._rects: list[tuple[QRectF, Block]] = []
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(RULER + 64)

    def set_blocks(self, blocks: list[Block], focus: str | None, view_width: int):
        self.blocks, self.focus = blocks, focus
        total = sum(b.secs for b in blocks) or 1.0
        avail = max(200, view_width - 16)
        self.px_per_sec = max(avail / total, 1.0)
        width = int(sum(max(MIN_BLOCK, b.secs * self.px_per_sec) for b in blocks)) + 16
        self.setMinimumWidth(min(width, 100000))
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.minimumWidth(), RULER + 64)

    def paintEvent(self, _e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), t.bgSunken)
        h = self.height()
        x = 8.0
        self._rects = []
        # ruler: one pass over the blocks (they have a minimum width, so time is mapped block by block); ticks
        # spaced to be readable at this zoom
        p.setPen(QPen(t.textTertiary, 1))
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 2))
        p.setFont(small)
        pos, px = [], 8.0
        for b in self.blocks:
            w = max(MIN_BLOCK, b.secs * self.px_per_sec)
            pos.append((px, w, b))
            px += w
        total = sum(b.secs for b in self.blocks)
        gap = (px - 8.0) / max(total, 0.001)            # pixels per second on average
        label_step = next((st for st in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if st * gap >= 44), 1200)
        tick_step = next((st for st in (1, 2, 5, 10, 30, 60, 300) if st * gap >= 6), 600)
        sec, i, acc = 0, 0, 0.0
        while pos and sec <= total + 0.01:
            while i < len(pos) - 1 and acc + pos[i][2].secs < sec:
                acc += pos[i][2].secs
                i += 1
            bx, bw, b = pos[i]
            xx = bx + (sec - acc) / max(b.secs, 0.001) * bw
            major = sec % label_step == 0
            p.drawLine(QPointF(xx, RULER - (6 if major else 3)), QPointF(xx, RULER))
            if major:
                p.drawText(QPointF(xx + 2, RULER - 5), f"{sec // 60}:{sec % 60:02d}")
            sec += tick_step
        font = QFont(self.font())
        font.setBold(True)
        for bx, bw, b in pos:
            r = QRectF(bx + 1, RULER + 4, bw - 2, h - RULER - 8)
            self._rects.append((r, b))
            base = t.colour("bgRaised") or QColor(60, 60, 60)
            p.setPen(Qt.NoPen)
            p.setBrush(with_alpha(QColor(90, 140, 200), 0.35) if b.is_video else base)
            p.drawRoundedRect(r, 4, 4)
            if b.pixmap is not None and not b.pixmap.isNull() and r.width() > 30:
                th = r.height() - 18
                tw = min(r.width() - 6, th * b.pixmap.width() / max(1, b.pixmap.height()))
                p.drawPixmap(QRectF(r.left() + 3, r.top() + 15, tw, th), b.pixmap, QRectF(b.pixmap.rect()))
            if b.lead_secs > 0:                     # the carry-over: repeated from the clip before
                lw = min(r.width() / 2, b.lead_secs * self.px_per_sec)
                p.setBrush(QBrush(with_alpha(t.textTertiary, 0.5), Qt.BDiagPattern))
                p.drawRect(QRectF(r.left(), r.top(), lw, r.height()))
            p.setBrush(status_colour(b.state))      # its state, as in the graph
            p.drawRect(QRectF(r.left(), r.bottom() - 4, r.width(), 4))
            p.setFont(font)
            p.setPen(t.textPrimary)
            p.drawText(QRectF(r.left() + 4, r.top() + 1, r.width() - 8, 14), Qt.AlignLeft | Qt.AlignVCenter,
                       ("▶ " if b.is_video else "") + b.label)
            if b.nid == self.focus:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(t.accent, 2))
                p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 5, 5)

    def _at(self, pos: QPointF) -> Block | None:
        return next((b for r, b in self._rects if r.contains(pos)), None)

    def mousePressEvent(self, e):
        b = self._at(e.position())
        if b is None:
            return
        if e.button() == Qt.LeftButton:
            self.clicked.emit(b.nid)
        elif e.button() == Qt.RightButton:
            self.menu.emit(b.nid, e.globalPosition())

    def mouseDoubleClickEvent(self, e):
        b = self._at(e.position())
        if b is not None and e.button() == Qt.LeftButton:   # (the first press already focused it)
            self.activated.emit(b.nid)

    def mouseMoveEvent(self, e):
        b = self._at(e.position())
        self.setToolTip(b.tip if b else "")


class Timeline(QScrollArea):
    """clicked(nid): show that clip; activated(nid): play it; menu(nid, global pos): its menu."""
    clicked = Signal(str)
    activated = Signal(str)
    menu = Signal(str, QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lane = _Lane()
        self.setWidget(self.lane)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(RULER + 64 + 14)
        self.lane.clicked.connect(self.clicked)
        self.lane.activated.connect(self.activated)
        self.lane.menu.connect(self.menu)
        self._blocks: list[Block] = []
        self._focus: str | None = None

    def show_path(self, blocks: list[Block], focus: str | None):
        self._blocks, self._focus = blocks, focus
        self.lane.set_blocks(blocks, focus, self.viewport().width())

    def set_focus(self, nid: str | None):
        self._focus = nid
        self.lane.focus = nid
        self.lane.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.lane.set_blocks(self._blocks, self._focus, self.viewport().width())
