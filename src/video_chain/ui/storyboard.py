"""The film's scenes as a strip of cards above the clip graph: click a card to work on that scene, drag to reorder,
right-click to rename or delete, '+ Scene' to add one."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QListWidget, QListWidgetItem, QMenu, QPushButton, QWidget

from .widgets import label

THUMB = QSize(96, 54)


def state_text(s: dict) -> str:
    """'0:24 · done' / '0:18 · 1 changed, 2 new' / 'empty' for a scene card."""
    n = s.get("done", 0) + s.get("changed", 0) + s.get("new", 0)
    secs = float(s.get("secs") or 0)
    length = f"{int(secs // 60)}:{int(round(secs % 60)):02d}"
    if not n:
        return "no clip chosen" if s.get("clips") else "empty"
    if s.get("done", 0) == n:
        return f"{length} · done"
    parts = [f"{s[k]} {k}" for k in ("changed", "new") if s.get(k)]
    return f"{length} · " + ", ".join(parts)


class Storyboard(QWidget):
    """sceneChosen(id), addWanted(), renameWanted(id), deleteWanted(id), moved(ids in their new order)."""
    sceneChosen = Signal(str)
    addWanted = Signal()
    renameWanted = Signal(str)
    deleteWanted = Signal(str)
    moved = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(6)
        self.list = QListWidget()
        self.list.setFlow(QListWidget.LeftToRight)
        self.list.setWrapping(False)
        self.list.setViewMode(QListWidget.ListMode)
        self.list.setIconSize(THUMB)
        self.list.setFixedHeight(THUMB.height() + 26)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setSpacing(4)
        self.list.setToolTip("The scenes of the film, in the order they are played. Click a scene to work on it, "
                             "drag to reorder, right-click to rename or delete")
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.currentItemChanged.connect(self._chosen)
        self.list.model().rowsMoved.connect(lambda *_: self.moved.emit(self.order()))
        h.addWidget(self.list, 1)
        side = QWidget()
        sv = QHBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        self.b_add = QPushButton("+ Scene")
        self.b_add.setToolTip("Add a scene after the current one: its own start image and clips; the cast, sounds "
                              "and video size are shared by the whole film")
        self.b_add.clicked.connect(self.addWanted)
        self.total = label("", level="tertiary")
        sv.addWidget(self.total)
        sv.addWidget(self.b_add)
        h.addWidget(side)
        self._loading = False

    def show_scenes(self, scenes: list[tuple[str, str, dict]], current: str | None) -> None:
        """scenes: (id, title, summary) in film order."""
        self._loading = True
        try:
            self.list.clear()
            total = 0.0
            for k, (sid, title, s) in enumerate(scenes, start=1):
                pm = QPixmap()
                if s.get("thumb") and Path(s["thumb"]).is_file():
                    img = QImage(str(s["thumb"]))
                    if not img.isNull():
                        pm = QPixmap.fromImage(img.scaled(THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                item = QListWidgetItem(QIcon(pm) if not pm.isNull() else QIcon(), f"{k}  {title}\n{state_text(s)}")
                item.setData(Qt.UserRole, sid)
                item.setToolTip(f"Scene {k}: {title}. {s.get('clips', 0)} clips; its selected path is what the "
                                "film plays. Click to work on it, drag to move it")
                item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
                self.list.addItem(item)
                if sid == current:
                    self.list.setCurrentItem(item)
                total += float(s.get("secs") or 0)
            n = len(scenes)
            self.total.setText(f"{n} scene{'s' if n != 1 else ''} · {int(total // 60)}:{int(round(total % 60)):02d}")
        finally:
            self._loading = False

    def update_card(self, sid: str, title: str, s: dict) -> None:
        """One scene's card (the open scene, while it is being worked on)."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == sid:
                item.setText(f"{i + 1}  {title}\n{state_text(s)}")

    def order(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def _chosen(self, item, _prev=None):
        if not self._loading and item is not None:
            self.sceneChosen.emit(item.data(Qt.UserRole))

    def _menu(self, pos):
        item = self.list.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        menu = QMenu(self)
        a_rename = menu.addAction("Rename…")
        a_delete = menu.addAction("Delete scene…")
        chosen = menu.exec(self.list.mapToGlobal(pos))
        if chosen is a_rename:
            self.renameWanted.emit(sid)
        elif chosen is a_delete:
            self.deleteWanted.emit(sid)

    def setEnabled(self, on: bool):          # noqa: N802  (Qt naming): no switching while clips are made
        self.list.setEnabled(on)
        self.b_add.setEnabled(on)
