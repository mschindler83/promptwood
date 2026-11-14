"""The selected path as a timeline: one block per clip, as wide as the clip is long in the joined video, with its
state colour, a thumbnail and its number; carry-over frames (the part repeated from the clip before) hatched, real
videos in their own colour. Click a block to show that clip in the inspector, double-click to play it, right-click
for its menu."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QScrollArea, QSizePolicy, QWidget

from .graph import status_colour
from .theme import theme, with_alpha
from .widgets import format_time, small_font

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
        self.playing: str | None = None
        self._rects: list[tuple[QRectF, Block]] = []
        self._pos: list[tuple[float, float, Block]] = []      # (x, width, block): the layout, made in _layout()
        self._thumbs: dict[str, tuple] = {}                   # block id -> ((picture key, size, scale), scaled)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(RULER + 64)

    def set_blocks(self, blocks: list[Block], focus: str | None, view_width: int):
        self.blocks, self.focus = blocks, focus
        total = sum(b.secs for b in blocks) or 1.0
        avail = max(200, view_width - 16)
        self.px_per_sec = max(avail / total, 1.0)
        width = int(sum(max(MIN_BLOCK, b.secs * self.px_per_sec) for b in blocks)) + 16
        self.setMinimumWidth(min(width, 100000))
        self._layout()
        self.updateGeometry()
        self.update()

    def _layout(self) -> None:
        """Where each block is (hit-testing and painting use the same layout; it needs no paint to exist)."""
        h = self.height() if self.height() > 0 else RULER + 64
        self._pos, px = [], 8.0
        for b in self.blocks:
            w = max(MIN_BLOCK, b.secs * self.px_per_sec)
            self._pos.append((px, w, b))
            px += w
        self._rects = [(QRectF(bx + 1, RULER + 4, bw - 2, h - RULER - 8), b) for bx, bw, b in self._pos]

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout()

    def block_rect(self, nid: str) -> QRectF | None:
        return next((r for r, b in self._rects if b.nid == nid), None)

    def thumb(self, b: Block, w: int, h: int, dpr: float) -> QPixmap | None:
        """The block's picture scaled to w x h (logical pixels) at the screen's scale, once, not on every paint."""
        pm = b.pixmap
        if pm is None or pm.isNull():
            return None
        key = (pm.cacheKey(), w, h, dpr)
        hit = self._thumbs.get(b.nid)
        if hit is None or hit[0] != key:
            sc = pm.scaled(max(1, int(w * dpr)), max(1, int(h * dpr)), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            sc.setDevicePixelRatio(dpr)
            hit = self._thumbs[b.nid] = (key, sc)
        return hit[1]

    def sizeHint(self) -> QSize:
        return QSize(self.minimumWidth(), RULER + 64)

    def paintEvent(self, _e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), t.bgSunken)
        # ruler: one pass over the blocks (they have a minimum width, so time is mapped block by block); ticks
        # spaced to be readable at this zoom
        p.setPen(QPen(t.textTertiary, 1))
        small = QFont(self.font())
        small = small_font(small, 0.8)
        p.setFont(small)
        pos = self._pos
        px = pos[-1][0] + pos[-1][1] if pos else 8.0
        total = sum(b.secs for b in self.blocks)
        gap = (px - 8.0) / max(total, 0.001)            # pixels per second on average
        tick_step = next((st for st in (1, 2, 5, 10, 30, 60, 300) if st * gap >= 6), 600)
        label_step = next((st for st in (1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600)
                           if st % tick_step == 0 and st * gap >= 44), 3600)   # labels fall on ticks
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
                p.drawText(QPointF(xx + 2, RULER - 5), format_time(sec))
            sec += tick_step
        font = QFont(self.font())
        font.setBold(True)
        for (bx, bw, b), (r, _b) in zip(pos, self._rects):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(t.accent, 0.35) if b.is_video else t.bgRaised)
            p.drawRoundedRect(r, 4, 4)
            if b.pixmap is not None and not b.pixmap.isNull() and r.width() > 30:
                th = r.height() - 18
                tw = min(r.width() - 6, th * b.pixmap.width() / max(1, b.pixmap.height()))
                pm = self.thumb(b, max(1, round(tw)), max(1, round(th)), self.devicePixelRatioF())
                p.drawPixmap(QRectF(r.left() + 3, r.top() + 15, tw, th), pm, QRectF(pm.rect()))
            if b.lead_secs > 0:                     # the carry-over: repeated from the clip before
                lw = min(r.width() / 2, b.lead_secs * self.px_per_sec)
                p.setBrush(QBrush(with_alpha(t.textTertiary, 0.5), Qt.BrushStyle.BDiagPattern))
                p.drawRect(QRectF(r.left(), r.top(), lw, r.height()))
            p.setBrush(status_colour(b.state))      # its state, as in the graph
            p.drawRect(QRectF(r.left(), r.bottom() - 4, r.width(), 4))
            p.setFont(font)
            p.setPen(t.textPrimary)
            p.drawText(QRectF(r.left() + 4, r.top() + 1, r.width() - 8, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       ("▶ " if b.is_video else "") + b.label)
            if b.nid == self.focus:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(t.accent, 2))
                p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 5, 5)
            if b.nid == self.playing:               # the played path is at this clip
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(with_alpha(t.accent, 0.3))
                p.drawRoundedRect(r, 4, 4)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(t.accent, 3))
                p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 5, 5)

    def _at(self, pos: QPointF) -> Block | None:
        return next((b for r, b in self._rects if r.contains(pos)), None)

    def mousePressEvent(self, e):
        b = self._at(e.position())
        if b is None:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(b.nid)
        elif e.button() == Qt.MouseButton.RightButton:
            self.clicked.emit(b.nid)                # the clip is selected first, as in the graph
            self.menu.emit(b.nid, e.globalPosition())

    def mouseDoubleClickEvent(self, e):
        b = self._at(e.position())
        if b is not None and e.button() == Qt.MouseButton.LeftButton:   # (the first press already focused it)
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
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(RULER + 64 + 14)
        self.lane.clicked.connect(self.clicked)
        self.lane.activated.connect(self.activated)
        self.lane.menu.connect(self.menu)
        self._blocks: list[Block] = []
        self._focus: str | None = None

    def show_path(self, blocks: list[Block], focus: str | None):
        self._blocks, self._focus = blocks, focus
        self.lane._thumbs = {k: v for k, v in self.lane._thumbs.items() if k in {b.nid for b in blocks}}
        self.lane.set_blocks(blocks, focus, self.viewport().width())

    def set_focus(self, nid: str | None):
        self._focus = nid
        self.lane.focus = nid
        self.lane.update()

    def set_playing(self, nid: str | None):
        """Mark the clip the played path is at (None: nothing is played)."""
        if self.lane.playing != nid:
            self.lane.playing = nid
            self.lane.update()
        r = self.lane.block_rect(nid) if nid is not None else None
        if r is not None:                           # the played clip stays in view
            self.ensureVisible(round(r.center().x()), round(r.center().y()), round(r.width() / 2) + 20, 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.lane.set_blocks(self._blocks, self._focus, self.viewport().width())
