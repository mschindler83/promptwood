"""Who appears in which clip of the selected path: one row per character, one column per clip.
● face, ◆ picture (a place, object or style), ♪ speaks with its voice, ○ the lead's face (the prompt mentions
nobody), · mentioned without a face / picture."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from .. import tree as T


def cast_cells(options: dict, nodes: list) -> dict[tuple[str, str], tuple[str, str]]:
    """{(character name, node id): (mark, tooltip)} for the clips of a path (videos have none)."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for n in nodes:
        if n.is_video:
            continue
        use = T.node_cast(options, n)
        mentioned = {c["name"] for c in T.mentions(options, T.full_prompt(options, n.prompt))[0]}
        talking = {c["name"] for c in use["characters"]
                   if c.get("voice") and c["voice"]["file"] in use.get("voices", [])}
        shown = {c["name"] for c in use["characters"] if c["images"]} if use["mode"] in ("voices", "mention", "tags") \
            else set()
        lead = use["characters"][0]["name"] if use["mode"] == "lead" and use["characters"] else None
        for c in T.cast_of(options):
            name, what = c["name"], []
            if name == lead:
                mark, what = "○", ["the lead's face (the prompt mentions nobody)"]
            else:
                person = T.kind_of(c) == "person"
                mark = (("●" if person else "◆") if name in shown else "") + ("♪" if name in talking else "")
                if name in shown:
                    what.append("face" if person else f"picture ({T.kind_of(c)})")
                if name in talking:
                    what.append("speaks with its voice")
                if not mark and name in mentioned:
                    mark, what = "·", [f"mentioned, but no {'face picture' if person else 'picture'} is sent"]
            if mark:
                out[(name, n.id)] = (mark, f"@{name} in #{n.id[1:]}: {', '.join(what)}")
    return out


class CastStrip(QTableWidget):
    """The cast along the selected path; clicking a clip's column shows its card."""
    clipClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setMinimumSectionSize(34)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().sectionClicked.connect(self._clicked)
        self.cellClicked.connect(lambda _r, c: self._clicked(c))
        self.setToolTip("Who and what appears in which clip: ● face, ◆ picture (a place, object or style), ♪ speaks "
                        "with its voice, ○ the lead (the prompt mentions nobody), · mentioned without a face or "
                        "picture. Click a clip to show its card.")
        self._ids: list[str] = []

    def show_path(self, options: dict, nodes: list) -> None:
        cast = T.cast_of(options)
        show = bool(cast) and bool(nodes)
        self.setVisible(show)
        if not show:                                # (not isVisible(): False too before the window is shown)
            return
        cells = cast_cells(options, nodes)
        self._ids = [n.id for n in nodes]
        self.clear()
        self.setRowCount(len(cast))
        self.setColumnCount(len(nodes))
        self.setHorizontalHeaderLabels([("▶" if n.is_video else "#") + n.id[1:] for n in nodes])
        self.setVerticalHeaderLabels([f"@{c['name']}" for c in cast])
        for r, c in enumerate(cast):
            for k, n in enumerate(nodes):
                mark, tip = cells.get((c["name"], n.id), ("", ""))
                item = QTableWidgetItem(mark)
                item.setTextAlignment(Qt.AlignCenter)
                item.setToolTip(tip or (f"video #{n.id[1:]} (real footage)" if n.is_video else
                                        f"@{c['name']} is not in #{n.id[1:]}"))
                self.setItem(r, k, item)
        h = self.horizontalHeader().height() + sum(self.rowHeight(r) for r in range(len(cast))) + 4
        self.setFixedHeight(min(h, 24 + 20 * 5))

    def _clicked(self, col: int) -> None:
        if 0 <= col < len(self._ids):
            self.clipClicked.emit(self._ids[col])
