"""A clip's middle keyframes on its timeline: thumbnails at their moments, a click adds one, a drag moves one, a
right-click changes, annotates or removes one. Hard cuts of the made clip show as red marks."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from .theme import theme

FPS = 24
THUMB_W, THUMB_H = 40, 23


class KeyframeStrip(QWidget):
    addAt = Signal(float)                 # seconds: the empty spot clicked
    moved = Signal(int, float)            # keyframe index, its new time
    wanted = Signal(str, int)             # "change" / "video" / "moment" / "note" / "remove", keyframe index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(THUMB_H + 18)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.duration = 3.0
        self.items: list[dict] = []       # {"at", "pixmap", "note", "ok"}
        self.cuts: list[float] = []
        self.made = False                 # the clip exists (its own moments can be kept)
        self._drag: int | None = None
        self._drag_at = 0.0
        self._press = None
        self._hover: int | None = None

    def sizeHint(self):
        return self.minimumSizeHint().expandedTo(self.minimumSize())

    # ---- data
    def set_data(self, duration: float, items: list[dict], cuts: list[float] = (), made: bool = False):
        self.duration = max(0.25, float(duration))
        self.items, self.cuts, self.made = list(items), list(cuts), made
        self._drag = None
        self.update()
        tips = [f"{it['at']:.2f} s" + (f": {it['note']}" if it.get("note") else "") +
                ("" if it.get("ok", True) else " (picture missing)") for it in self.items]
        self.setToolTip("Middle keyframes: pictures the clip passes through at these moments.\n"
                        "Click an empty spot to add one there; drag one to move it; right-click it to change it.\n"
                        "Keep each one reachable by motion from what comes before it (the same place and light): "
                        "a picture it cannot move to makes H3 cut to it (marked red) or slowly morph the whole scene (not marked)." + ("\n\n" + "\n".join(tips) if tips else ""))

    # ---- geometry
    def _track(self) -> QRectF:
        m = THUMB_W / 2 + 2
        return QRectF(m, 4, max(10.0, self.width() - 2 * m), THUMB_H)

    def x_of(self, at: float) -> float:
        tr = self._track()
        return tr.left() + tr.width() * min(1.0, max(0.0, at / self.duration))

    def at_of(self, x: float) -> float:
        tr = self._track()
        at = (x - tr.left()) / tr.width() * self.duration
        lo, hi = 1 / FPS, self.duration - 2 / FPS             # never the first or the last frame
        return round(min(hi, max(lo, at)) * FPS) / FPS

    def _rect(self, i: int) -> QRectF:
        at = self._drag_at if i == self._drag else self.items[i]["at"]
        x = self.x_of(at)
        return QRectF(x - THUMB_W / 2, self._track().top(), THUMB_W, THUMB_H)

    def index_at(self, pos) -> int | None:
        for i in reversed(range(len(self.items))):
            if self._rect(i).adjusted(-2, -2, 2, 2).contains(QPointF(pos)):
                return i
        return None

    # ---- painting
    def paintEvent(self, _e):
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tr = self._track()
        mid = tr.center().y()
        p.setPen(Qt.NoPen)
        p.setBrush(t.bgSunken)
        p.drawRoundedRect(QRectF(tr.left(), mid - 3, tr.width(), 6), 3, 3)
        p.setPen(QPen(t.textTertiary, 1))
        f = p.font()
        f.setPointSizeF(max(6.5, f.pointSizeF() * 0.8))
        p.setFont(f)
        step = 1.0 if self.duration <= 12 else 2.0
        s = 0.0
        while s <= self.duration + 1e-6:
            x = self.x_of(s)
            p.drawLine(QPointF(x, tr.bottom() + 1), QPointF(x, tr.bottom() + 4))
            p.drawText(QRectF(x - 14, tr.bottom() + 3, 28, 12), Qt.AlignCenter, f"{s:g}s")
            s += step
        for c in self.cuts:                                     # hard cuts in the made clip
            x = self.x_of(c)
            p.setPen(QPen(QColor("#e5534b"), 2))
            p.drawLine(QPointF(x, tr.top()), QPointF(x, tr.bottom()))
        order = [i for i in range(len(self.items)) if i != self._drag] + \
            ([self._drag] if self._drag is not None else [])
        for i in order:
            it = self.items[i]
            r = self._rect(i)
            path = QPainterPath()
            path.addRoundedRect(r, 3, 3)
            pm: QPixmap | None = it.get("pixmap")
            if pm is not None and not pm.isNull():
                p.save()
                p.setClipPath(path)
                sc = pm.scaled(int(r.width()), int(r.height()), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                p.drawPixmap(int(r.center().x() - sc.width() / 2), int(r.center().y() - sc.height() / 2), sc)
                p.restore()
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(t.bgRaised)
                p.drawPath(path)
            ok = it.get("ok", True)
            border = QColor("#e5534b") if not ok else t.accent if i in (self._hover, self._drag) else t.lineStrong
            p.setPen(QPen(border, 2 if i in (self._hover, self._drag) or not ok else 1))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            if it.get("note"):
                p.setPen(Qt.NoPen)
                p.setBrush(t.accent)
                p.drawEllipse(QPointF(r.right() - 4, r.top() + 4), 3, 3)
        if self._drag is not None:
            p.setPen(t.textPrimary)
            r = self._rect(self._drag)
            p.drawText(QRectF(r.left() - 10, tr.bottom() + 3, r.width() + 20, 12), Qt.AlignCenter,
                       f"{self._drag_at:.2f}s")
        p.end()

    # ---- mouse
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.isEnabled():
            self._press = e.position()
            i = self.index_at(e.position())
            if i is not None:
                self._drag, self._drag_at = i, self.items[i]["at"]
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None and self._press is not None and abs(e.position().x() - self._press.x()) > 3:
            self._drag_at = self.at_of(e.position().x())
            self.update()
        else:
            h = self.index_at(e.position())
            if h != self._hover:
                self._hover = h
                self.setCursor(Qt.OpenHandCursor if h is not None else Qt.PointingHandCursor)
                self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self._hover = None
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or self._press is None:
            return super().mouseReleaseEvent(e)
        drag, press = self._drag, self._press
        self._drag, self._press = None, None
        moved = abs(e.position().x() - press.x()) > 3
        if drag is not None:
            at = self.at_of(e.position().x())
            if moved and abs(at - self.items[drag]["at"]) > 1e-6:
                self.moved.emit(drag, at)
        elif not moved and self.isEnabled():
            self.addAt.emit(self.at_of(e.position().x()))
        self.update()

    def mouseDoubleClickEvent(self, e):
        i = self.index_at(e.position())
        if i is not None and self.isEnabled():
            self.wanted.emit("change", i)

    def contextMenuEvent(self, e):
        i = self.index_at(e.pos())
        if i is None or not self.isEnabled():
            return
        m = QMenu(self)
        acts = {m.addAction("Another picture…"): "change", m.addAction("A frame of a video…"): "video"}
        if self.made:
            acts[m.addAction("This clip's frame at this moment (keep it)…")] = "moment"
        m.addSeparator()
        acts[m.addAction("What happens here (note)…")] = "note"
        m.addSeparator()
        acts[m.addAction("Remove")] = "remove"
        a = m.exec(e.globalPos())
        if a in acts:
            self.wanted.emit(acts[a], i)
