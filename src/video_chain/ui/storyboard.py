"""The film's scenes as a strip of cards above the clip graph: click a card to work on that scene, drag to reorder,
right-click to rename or delete, '+ Scene' to add one."""
from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QListWidget, QListWidgetItem, QMenu, QPushButton, QWidget

from .widgets import format_time, label

THUMB = QSize(96, 54)


def state_text(s: dict) -> str:
    """'0:24 · done' / '0:18 · 1 changed, 2 new' / 'empty' for a scene card."""
    n = s.get("done", 0) + s.get("changed", 0) + s.get("new", 0)
    secs = float(s.get("secs") or 0)
    length = format_time(secs)
    if not n:
        return "no clip chosen" if s.get("clips") else "empty"
    end = f" · ends at #{s['take'][1:]}" if s.get("fixed") and s.get("take") else ""
    if s.get("done", 0) == n:
        return f"{length} · done{end}"
    parts = [f"{s[k]} {k}" for k in ("changed", "new") if s.get(k)]
    return f"{length} · " + ", ".join(parts) + end


JOIN_MARK = {"fade": "  ⤳ crossfade", "continue": "  → continues"}


def card_text(k: int, title: str, s: dict) -> str:
    return f"{k}  {title}{JOIN_MARK.get(s.get('join') or '', '')}\n{state_text(s)}"


class Storyboard(QWidget):
    """sceneChosen(id), addWanted(), renameWanted(id), deleteWanted(id), moved(ids in their new order)."""
    sceneChosen = Signal(str)
    joinWanted = Signal(str, str)                 # scene id, "cut" / "fade" / "continue"
    duplicateWanted = Signal(str)
    notesWanted = Signal(str)
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
        self.list.itemClicked.connect(self._chosen)            # a click (not a press: dragging must work)
        self.list.itemActivated.connect(self._chosen)          # Enter
        self.list.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self._emit_moved))
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
        self._current: str | None = None          # the open scene (a click on it changes nothing)
        self._thumbs: dict[str, tuple[tuple, QPixmap]] = {}      # path -> ((mtime, size), the scaled picture)

    def _thumb(self, path: str) -> QPixmap:
        """The scene's start picture scaled to a card, read from disk once per file version."""
        try:
            st = Path(path).stat()
        except OSError:
            return QPixmap()
        key = (st.st_mtime_ns, st.st_size)
        hit = self._thumbs.get(path)
        if hit is None or hit[0] != key:
            img = QImage(path)
            pm = QPixmap() if img.isNull() else QPixmap.fromImage(
                img.scaled(THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            hit = self._thumbs[path] = (key, pm)
        return hit[1]

    def show_scenes(self, scenes: list[tuple[str, str, dict]], current: str | None) -> None:
        """scenes: (id, title, summary) in film order."""
        self._loading = True
        self._current = current
        self._thumbs = {p: v for p, v in self._thumbs.items()
                        if p in {str(s.get("thumb")) for _i, _t, s in scenes}}
        try:
            self.list.clear()
            total = 0.0
            for k, (sid, title, s) in enumerate(scenes, start=1):
                pm = self._thumb(str(s["thumb"])) if s.get("thumb") else QPixmap()
                item = QListWidgetItem(QIcon(pm) if not pm.isNull() else QIcon(), card_text(k, title, s))
                item.setData(Qt.UserRole, sid)
                item.setData(Qt.UserRole + 1, s.get("join"))
                notes = html.escape(s["notes"]).replace("\n", "<br>") if s.get("notes") else ""
                item.setToolTip("<p>" + (f"{notes}<br><br>" if notes else "") +
                                f"Scene {k}: {html.escape(title)}. {s.get('clips', 0)} clips; the film plays it up to "
                                + (f"clip #{s['take'][1:]}" if s.get("fixed") else "its selected clip") +
                                ". Click to work on it, drag to move it, right-click for more</p>")
                item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
                self.list.addItem(item)
                if sid == current:
                    self.list.setCurrentItem(item)
                total += float(s.get("secs") or 0)
            n = len(scenes)
            self.total.setText(f"{n} scene{'s' if n != 1 else ''} · {format_time(total)}")
        finally:
            self._loading = False

    def update_card(self, sid: str, title: str, s: dict) -> None:
        """One scene's card (the open scene, while it is being worked on)."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == sid:
                item.setText(card_text(i + 1, title, s))

    def order(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def _chosen(self, item, _prev=None):
        """A click and Enter both come here, and a double-click sends several: the open scene is not chosen again."""
        if not self._loading and item is not None:
            sid = item.data(Qt.UserRole)
            if sid == self._current:
                return
            self._current = sid
            QTimer.singleShot(0, lambda: self.sceneChosen.emit(sid))   # after the click is handled

    def _emit_moved(self):
        """After a drop (not inside it): the list is rebuilt by the handler."""
        self.moved.emit(self.order())

    def _menu(self, pos):
        item = self.list.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        menu = QMenu(self)
        a_rename = menu.addAction("Rename…")
        a_notes = menu.addAction("Notes…")
        a_dup = menu.addAction("Duplicate (a copy with its clips)")
        joins = {}
        if self.list.row(item) > 0:
            sub = menu.addMenu("Join with the scene before")
            cur = item.data(Qt.UserRole + 1) or "cut"
            for key, text in (("cut", "Cut (hard cut)"), ("fade", "Crossfade (½ s, picture and sound)"),
                              ("continue", "Continue (starts from its last frame)")):
                a = sub.addAction(text)
                a.setCheckable(True)
                a.setChecked(key == cur)
                joins[a] = key
        menu.addSeparator()
        a_delete = menu.addAction("Delete scene…")
        chosen = menu.exec(self.list.mapToGlobal(pos))
        if chosen is a_rename:
            self.renameWanted.emit(sid)
        elif chosen is a_notes:
            self.notesWanted.emit(sid)
        elif chosen is a_dup:
            self.duplicateWanted.emit(sid)
        elif chosen is a_delete:
            self.deleteWanted.emit(sid)
        elif chosen in joins:
            self.joinWanted.emit(sid, joins[chosen])

    def setEnabled(self, on: bool):          # noqa: N802  (Qt naming): no switching while clips are made
        super().setEnabled(on)
