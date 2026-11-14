"""Node view of the clip tree: the start image on the left, every clip a node, lines to its continuations.
Clicking a node selects it (the path from the start to it is highlighted); double-click plays its clip (a real
video: opens its range editor)."""
from __future__ import annotations

import html
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsObject, QGraphicsPathItem, QGraphicsScene, QGraphicsView

from .theme import theme, with_alpha
from .widgets import small_font

W, H = 216, 178
THUMB_H = 114
GAP_X, GAP_Y = 64, 24
START = "__start__"
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.2, 2.5, 1.15
TIP_CHARS = 300
TIP_HINT = "Double-click: play. Right-click: more."


@dataclass
class NodeInfo:
    id: str
    parent: str | None            # START for clips right after the start image, None for the start node itself
    title: str
    sub: str
    prompt: str
    status: str                   # start | done | changed | new | generating | failed | stopped
    pixmap: QPixmap | None = None
    progress: float = 0.0
    on_path: bool = False
    selected: bool = False
    badge: str = ""               # the clip's own settings (LoRA, strength, length, steps)
    kind: str = "ai"              # ai | video (a range of a real video)


STATUS_TEXT = {"done": "done", "changed": "changed", "new": "new", "generating": "generating",
               "failed": "failed", "stopped": "stopped", "reused": "done", "start": "start image",
               "converting": "converting"}


def status_colour(status: str) -> QColor:
    t = theme()
    return {"done": t.positive, "reused": t.positive, "changed": t.warning, "generating": t.accent,
            "converting": t.accent, "failed": t.negative, "stopped": t.warning}.get(status, t.textTertiary)


def node_tooltip(info: NodeInfo) -> str:
    """The prompt (cut short, as plain text) and what a click, a double-click and a right-click do."""
    text = (info.prompt or info.title).strip()
    if len(text) > TIP_CHARS:
        text = text[:TIP_CHARS].rstrip() + "…"
    return f"<p>{html.escape(text).replace(chr(10), '<br>')}</p><p><i>{TIP_HINT}</i></p>"


class NodeItem(QGraphicsObject):
    clicked = Signal(str)
    doubleClicked = Signal(str)
    contextMenu = Signal(str, object)

    def __init__(self, info: NodeInfo):
        super().__init__()
        self.info = info
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = False
        self.focused = False                  # shown in the inspector
        self.playing = False                  # the clip the played path is at
        self.setToolTip(node_tooltip(info))
        self._thumb: tuple | None = None      # (picture key, scale, size) -> the scaled picture

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, W + 4, H + 4)

    def set_info(self, info: NodeInfo):
        self.info = info
        self.setToolTip(node_tooltip(info))
        self.update()

    def thumb_pixmap(self, dpr: float) -> QPixmap | None:
        """The node's picture scaled to its thumbnail box at the screen's scale, once per picture and scale (not on
        every paint)."""
        pm = self.info.pixmap
        if pm is None or pm.isNull():
            return None
        key = (pm.cacheKey(), dpr)
        if self._thumb is None or self._thumb[0] != key:
            box = QRectF(8, 8, W - 16, THUMB_H)
            scaled = pm.scaled(int(box.width() * dpr), int(box.height() * dpr), Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            scaled.setDevicePixelRatio(dpr)
            self._thumb = (key, scaled)
        return self._thumb[1]

    def paint(self, p: QPainter, _opt, _w=None):
        t, i = theme(), self.info
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, W, H)
        if i.selected:
            pen = QPen(t.accent, 2)
        elif i.on_path:
            pen = QPen(with_alpha(t.accent, 0.55), 1.5)
        elif self._hover:
            pen = QPen(t.lineStrong, 1)
        else:
            pen = QPen(t.lineSubtle, 1)
        p.setPen(pen)
        p.setBrush(t.bgRaised)
        p.drawRoundedRect(r, 8, 8)
        if self.playing:                      # the played path is at this clip: a wide ring (and a play mark)
            p.setPen(QPen(t.accent, 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 9, 9)
        if self.focused:                      # the clip in the inspector: a second, outer ring
            p.setPen(QPen(with_alpha(t.accent, 0.8), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(-1.5, -1.5, 1.5, 1.5), 9, 9)

        thumb = QRectF(8, 8, W - 16, THUMB_H)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(t.bgSunken)
        p.drawRoundedRect(thumb, 5, 5)
        dpr = p.device().devicePixelRatioF() if p.device() is not None else 1.0
        pm = self.thumb_pixmap(dpr)
        if pm is not None:
            pw, ph = pm.width() / dpr, pm.height() / dpr
            target = QRectF(thumb.center().x() - pw / 2, thumb.center().y() - ph / 2, pw, ph)
            if i.status in ("changed", "stopped", "failed"):
                p.setOpacity(0.45)
            p.drawPixmap(target, pm, QRectF(pm.rect()))
            p.setOpacity(1.0)
        else:
            p.setPen(t.textTertiary)
            p.drawText(thumb, Qt.AlignmentFlag.AlignCenter, "no start image" if i.status == "start" else
                       "not converted yet" if i.kind == "video" else "not made yet")
        if i.badge:
            base_font = QFont(p.font())
            bf = small_font(base_font, 0.82)
            p.setFont(bf)
            bfm = QFontMetrics(bf)
            text = bfm.elidedText(i.badge, Qt.TextElideMode.ElideRight, int(thumb.width() - 16))
            bw, bh = bfm.horizontalAdvance(text) + 10, bfm.height() + 2
            pill = QRectF(thumb.x() + 5, thumb.y() + 5, bw, bh)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(t.accent, 0.85))
            p.drawRoundedRect(pill, bh / 2, bh / 2)
            p.setPen(t.onAccent)
            p.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
            p.setFont(base_font)

        f = QFont(p.font())
        f.setBold(True)
        p.setFont(f)
        if i.kind == "video":                   # a strip of film holes: this is real footage, not generated
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(with_alpha(t.textPrimary, 0.55))
            for k in range(int((thumb.width() - 8) // 12)):      # (bottom edge only: the badge is at the top)
                p.drawRoundedRect(QRectF(thumb.x() + 6 + 12 * k, thumb.bottom() - 6, 6, 4), 1, 1)
        p.setPen(t.textPrimary)
        y = 8 + THUMB_H + 8
        p.setBrush(status_colour(i.status))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(14, y + 8), 4, 4)
        p.setPen(t.textPrimary)
        p.drawText(QRectF(24, y, W - 32, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, i.title)
        f.setBold(False)
        p.setFont(f)
        p.setPen(t.textTertiary)
        p.drawText(QRectF(24, y, W - 32, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{i.sub}  {STATUS_TEXT.get(i.status, i.status)}".strip())
        p.setPen(t.textSecondary)
        fm = QFontMetrics(f)
        text = i.prompt.strip() or ("(empty prompt)" if i.status != "start" else "")
        p.drawText(QRectF(10, y + 20, W - 20, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(text, Qt.TextElideMode.ElideRight, W - 20))
        if self.playing:
            mark = QRectF(thumb.right() - 27, thumb.y() + 5, 22, 22)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(t.accent)
            p.drawEllipse(mark)
            c = mark.center()
            p.setBrush(t.onAccent)
            p.drawPolygon([QPointF(c.x() - 3.5, c.y() - 5.5), QPointF(c.x() - 3.5, c.y() + 5.5),
                           QPointF(c.x() + 5.5, c.y())])
        if i.status == "generating":
            bar = QRectF(8, H - 7, W - 16, 3)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(t.bgOverlay)
            p.drawRoundedRect(bar, 1.5, 1.5)
            p.setBrush(t.accent)
            p.drawRoundedRect(QRectF(bar.x(), bar.y(), bar.width() * max(0.02, min(1.0, i.progress)), 3), 1.5, 1.5)

    def hoverEnterEvent(self, e):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, e):
        self._hover = False
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.info.id)
            e.accept()
        elif e.button() == Qt.MouseButton.RightButton:
            self.clicked.emit(self.info.id)
            self.contextMenu.emit(self.info.id, e.screenPos())
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit(self.info.id)
            e.accept()


class GraphView(QGraphicsView):
    selected = Signal(object)          # node id, or START
    activated = Signal(str)
    contextMenu = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(theme().bgSunken))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.items_by_id: dict[str, NodeItem] = {}
        self.edges: list[tuple[str, str, QGraphicsPathItem]] = []
        self._structure: tuple = ()
        self._focus: str | None = None
        self._playing: str | None = None
        self._parent_of: dict[str, str | None] = {}
        self._kids_of: dict[str, list[str]] = {}
        # the keyboard reaches the branches too (see keyPressEvent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_playing(self, nid: str | None) -> None:
        """Mark the clip the played path is at (None: nothing is played), and bring it into view."""
        if nid == self._playing:
            return
        self._playing = nid
        for k, item in self.items_by_id.items():
            if item.playing != (k == nid):
                item.playing = k == nid
                item.update()
        if nid is not None:
            self.reveal(nid)

    def set_focus(self, nid: str | None) -> None:
        """Mark the clip shown in the inspector."""
        self._focus = nid
        for k, item in self.items_by_id.items():
            if item.focused != (k == nid):
                item.focused = k == nid
                item.update()

    # ---- content
    def show_nodes(self, infos: list[NodeInfo]) -> None:
        """Rebuild when the structure changed, else update the nodes in place (keeps scroll and zoom)."""
        structure = tuple((i.id, i.parent) for i in infos)
        if structure != self._structure:
            self._rebuild(infos)
            self._structure = structure
        else:
            for i in infos:
                self.items_by_id[i.id].set_info(i)
        self._style_edges()

    def _rebuild(self, infos: list[NodeInfo]) -> None:
        sc = self.scene()
        sc.clear()
        self.items_by_id, self.edges = {}, []
        by_parent: dict[str | None, list[NodeInfo]] = {}
        for i in infos:
            by_parent.setdefault(i.parent, []).append(i)
        pos: dict[str, QPointF] = {}
        ids = {i.id for i in infos}
        self._parent_of = {i.id: i.parent for i in infos}
        self._kids_of = {i.id: [k.id for k in by_parent.get(i.id, [])] for i in infos}
        # a node whose parent is not in the list is a root of its own (never stacked on the start node)
        roots = [i for i in infos if i.parent is None or i.parent not in ids]
        row = 0
        for root in roots:                      # iterative post-order: a long chain must not hit the recursion limit
            ys: dict[str, float] = {}
            stack: list[tuple[str, int, bool]] = [(root.id, 0, False)]
            while stack:
                nid, depth, done = stack.pop()
                kids = [k.id for k in by_parent.get(nid, [])]
                if not done and kids:
                    stack.append((nid, depth, True))
                    stack.extend((k, depth + 1, False) for k in reversed(kids))
                    continue
                if kids:
                    y = (ys[kids[0]] + ys[kids[-1]]) / 2
                else:
                    y = row * (H + GAP_Y)
                    row += 1
                ys[nid] = y
                pos[nid] = QPointF(depth * (W + GAP_X), y)
        for i in infos:
            item = NodeItem(i)
            item.setPos(pos.get(i.id, QPointF(0, 0)))
            item.setZValue(1)
            item.clicked.connect(self._clicked)
            item.doubleClicked.connect(self.activated)
            item.contextMenu.connect(self.contextMenu)
            sc.addItem(item)
            self.items_by_id[i.id] = item
            item.focused = i.id == self._focus
            item.playing = i.id == self._playing
        for i in infos:
            if i.parent is not None and i.parent in pos:
                a, b = pos[i.parent], pos[i.id]
                start, end = QPointF(a.x() + W, a.y() + H / 2), QPointF(b.x(), b.y() + H / 2)
                path = QPainterPath(start)
                dx = (end.x() - start.x()) / 2
                path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(end.x() - dx, end.y()), end)
                edge = QGraphicsPathItem(path)
                edge.setZValue(0)
                sc.addItem(edge)
                self.edges.append((i.parent, i.id, edge))
        r = sc.itemsBoundingRect().adjusted(-40, -40, 40, 40)
        sc.setSceneRect(r)

    def _style_edges(self):
        t = theme()
        for parent, child, edge in self.edges:
            on = self.items_by_id[child].info.on_path
            edge.setPen(QPen(t.accent if on else t.lineStrong, 2.2 if on else 1.4))

    def _clicked(self, nid: str):
        self.selected.emit(nid)

    # ---- keyboard: Up / Down the siblings, Left the parent, Right the first continuation, Enter plays, Menu
    def _order(self) -> list[str]:
        return list(self.items_by_id)

    def keyPressEvent(self, e):
        key = e.key()
        cur = self._focus if self._focus in self.items_by_id else None
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right,
                   Qt.Key.Key_Home) and self.items_by_id:
            if cur is None:
                self._go(self._order()[0])
            elif key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                par = self._parent_of.get(cur)
                sibs = self._kids_of.get(par, []) if par in self._kids_of else \
                    [k for k in self._order() if self._parent_of.get(k) == par]
                k = sibs.index(cur) + (1 if key == Qt.Key.Key_Down else -1) if cur in sibs else -1
                if 0 <= k < len(sibs):
                    self._go(sibs[k])
            elif key == Qt.Key.Key_Left:
                par = self._parent_of.get(cur)
                if par in self.items_by_id:
                    self._go(par)
            elif key == Qt.Key.Key_Right:
                kids = self._kids_of.get(cur, [])
                if kids:
                    self._go(kids[0])
            else:
                self._go(self._order()[0])
            e.accept()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and cur is not None:
            self.activated.emit(cur)
            e.accept()
        elif key == Qt.Key.Key_Menu and cur is not None:
            item = self.items_by_id[cur]
            self.contextMenu.emit(cur,
                                  self.viewport().mapToGlobal(self.mapFromScene(item.sceneBoundingRect().center())))
            e.accept()
        else:
            super().keyPressEvent(e)

    def _go(self, nid: str):
        self.selected.emit(nid)
        self.set_focus(nid)
        self.reveal(nid)

    # ---- view
    def zoom_by(self, f: float) -> None:
        """Zoom by the factor, kept between ZOOM_MIN and ZOOM_MAX."""
        now = self.transform().m11()
        f = min(ZOOM_MAX, max(ZOOM_MIN, now * f)) / now
        self.scale(f, f)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(ZOOM_STEP ** (e.angleDelta().y() / 120))
            e.accept()
        else:
            super().wheelEvent(e)

    def fit(self):
        r = self.scene().itemsBoundingRect().adjusted(-30, -30, 30, 30)
        if r.isEmpty():
            return
        self.resetTransform()
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self.centerOn(r.center())

    def reveal(self, nid: str):
        item = self.items_by_id.get(nid)
        if item is not None:
            self.ensureVisible(item, 40, 40)
