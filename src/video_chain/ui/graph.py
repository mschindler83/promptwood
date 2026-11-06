"""Node view of the clip tree: the start image on the left, every clip a node, lines to its continuations.
Clicking a node selects it (the path from the start to it is highlighted); double-click plays its clip."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsObject, QGraphicsPathItem, QGraphicsScene, QGraphicsView

from .theme import theme, with_alpha

W, H = 216, 178
THUMB_H = 114
GAP_X, GAP_Y = 64, 24
START = "__start__"


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


STATUS_TEXT = {"done": "done", "changed": "changed", "new": "not made", "generating": "generating",
               "failed": "failed", "stopped": "stopped", "reused": "done", "start": "start image"}


def status_colour(status: str) -> QColor:
    t = theme()
    return {"done": t.positive, "reused": t.positive, "changed": t.warning, "generating": t.accent,
            "failed": t.negative, "stopped": t.warning}.get(status, t.textTertiary)


class NodeItem(QGraphicsObject):
    clicked = Signal(str)
    doubleClicked = Signal(str)
    contextMenu = Signal(str, object)

    def __init__(self, info: NodeInfo):
        super().__init__()
        self.info = info
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False
        self.setToolTip(info.prompt or info.title)

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, W + 4, H + 4)

    def set_info(self, info: NodeInfo):
        self.info = info
        self.setToolTip(info.prompt or info.title)
        self.update()

    def paint(self, p: QPainter, _opt, _w=None):
        t, i = theme(), self.info
        p.setRenderHint(QPainter.Antialiasing)
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

        thumb = QRectF(8, 8, W - 16, THUMB_H)
        p.setPen(Qt.NoPen)
        p.setBrush(t.bgSunken)
        p.drawRoundedRect(thumb, 5, 5)
        if i.pixmap is not None and not i.pixmap.isNull():
            pm = i.pixmap.scaled(int(thumb.width() * 2), int(thumb.height() * 2), Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
            pm.setDevicePixelRatio(2)
            pw, ph = pm.width() / 2, pm.height() / 2
            target = QRectF(thumb.center().x() - pw / 2, thumb.center().y() - ph / 2, pw, ph)
            if i.status in ("changed", "stopped", "failed"):
                p.setOpacity(0.45)
            p.drawPixmap(target, pm, QRectF(pm.rect()))
            p.setOpacity(1.0)
        else:
            p.setPen(t.textTertiary)
            p.drawText(thumb, Qt.AlignCenter, "not made yet" if i.status != "start" else "no start image")

        f = QFont(p.font())
        f.setBold(True)
        p.setFont(f)
        p.setPen(t.textPrimary)
        y = 8 + THUMB_H + 8
        p.setBrush(status_colour(i.status))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(14, y + 8), 4, 4)
        p.setPen(t.textPrimary)
        p.drawText(QRectF(24, y, W - 32, 16), Qt.AlignLeft | Qt.AlignVCenter, i.title)
        f.setBold(False)
        p.setFont(f)
        p.setPen(t.textTertiary)
        p.drawText(QRectF(24, y, W - 32, 16), Qt.AlignRight | Qt.AlignVCenter,
                   f"{i.sub}  {STATUS_TEXT.get(i.status, i.status)}".strip())
        p.setPen(t.textSecondary)
        fm = QFontMetrics(f)
        text = i.prompt.strip() or ("(empty prompt)" if i.status != "start" else "")
        p.drawText(QRectF(10, y + 20, W - 20, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(text, Qt.ElideRight, W - 20))
        if i.status == "generating":
            bar = QRectF(8, H - 7, W - 16, 3)
            p.setPen(Qt.NoPen)
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
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.info.id)
            e.accept()
        elif e.button() == Qt.RightButton:
            self.clicked.emit(self.info.id)
            self.contextMenu.emit(self.info.id, e.screenPos())
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.doubleClicked.emit(self.info.id)
            e.accept()


class GraphView(QGraphicsView):
    selected = Signal(object)          # node id, or START, or None
    activated = Signal(str)
    contextMenu = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(theme().bgSunken))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.items_by_id: dict[str, NodeItem] = {}
        self.edges: list[tuple[str, str, QGraphicsPathItem]] = []
        self._structure: tuple = ()

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
        row = [0]

        def place(info: NodeInfo, depth: int) -> float:
            kids = by_parent.get(info.id, [])
            if kids:
                ys = [place(k, depth + 1) for k in kids]
                y = (ys[0] + ys[-1]) / 2
            else:
                y = row[0] * (H + GAP_Y)
                row[0] += 1
            pos[info.id] = QPointF(depth * (W + GAP_X), y)
            return y

        for root in by_parent.get(None, []):
            place(root, 0)
        for i in infos:
            item = NodeItem(i)
            item.setPos(pos.get(i.id, QPointF(0, 0)))
            item.setZValue(1)
            item.clicked.connect(self._clicked)
            item.doubleClicked.connect(self.activated)
            item.contextMenu.connect(self.contextMenu)
            sc.addItem(item)
            self.items_by_id[i.id] = item
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

    def mousePressEvent(self, e):
        super().mousePressEvent(e)

    # ---- view
    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            f = 1.15 ** (e.angleDelta().y() / 120)
            s = self.transform().m11() * f
            if 0.2 <= s <= 2.5:
                self.scale(f, f)
            e.accept()
        else:
            super().wheelEvent(e)

    def fit(self):
        r = self.scene().itemsBoundingRect().adjusted(-30, -30, 30, 30)
        if r.isEmpty():
            return
        self.resetTransform()
        self.fitInView(r, Qt.KeepAspectRatio)
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self.centerOn(r.center())

    def reveal(self, nid: str):
        item = self.items_by_id.get(nid)
        if item is not None:
            self.ensureVisible(item, 40, 40)
