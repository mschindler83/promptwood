"""The cast library windows (Cast → Cast library…): the people of a cast kept for every project
(video_chain.cast_library). The project's people are saved as one cast; a saved cast is brought into the project
(BringInDialog: which people, and for a name the project already has: replace it or add under another name)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QInputDialog, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .. import cast_library as CL
from .. import tree as T
from . import dialogs as D
from .ui_logic import plural
from .widgets import fit_size, label, remember_geometry

if TYPE_CHECKING:
    from .main_window import MainWindow

ADD, REPLACE = "add", "replace"


def person_text(p: dict) -> str:
    """@Anna   2 faces · voice · lead"""
    n = int(p.get("faces") or 0)
    return (f"@{p['name']}   " + (plural(n, "face") if n else "no face") + ("   · voice" if p.get("voice") else "")
            + ("   · lead" if p.get("lead") else ""))


def cast_text(c: dict) -> str:
    people = c.get("people") or []
    voices = sum(1 for p in people if p.get("voice"))
    names = ", ".join("@" + p["name"] for p in people)
    return f"{c['name']}   ({names}" + (f"; {plural(voices, 'voice')}" if voices else "") + ")"


def free_name(tree: T.Tree, name: str, taken: set[str] | None = None) -> str:
    """name_2, name_3 … : the first one neither in the cast nor in taken (lower case)."""
    taken = taken or set()
    k = 2
    while T.character(tree, f"{name}_{k}") is not None or f"{name}_{k}".lower() in taken:
        k += 1
    return f"{name}_{k}"


class BringInDialog(QDialog):
    """Which people of a saved cast come into the project. A person whose name the project has is not ticked:
    ticked, it replaces the project's one or is added under the name in its field."""

    def __init__(self, parent, saved: dict, tree: T.Tree, settings=None):
        super().__init__(parent)
        self.saved = saved
        self.setWindowTitle(f"Bring in the cast “{saved['name']}”")
        fit_size(self, 640, 420)
        remember_geometry(self, settings, "castBringIn")
        v = QVBoxLayout(self)
        info = label("Tick the people to bring into this project; their face pictures, descriptions and voices "
                     "are copied into it. For a name the project already has, choose: replace yours (its pictures "
                     "and voice go to old/) or add the saved one under another name.", level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setColumnStretch(0, 1)
        self.rows: list[tuple[dict, QCheckBox, QComboBox | None, QLineEdit | None]] = []
        taken: set[str] = set()
        for r, p in enumerate(saved.get("people") or []):
            mine = T.character(tree, p["name"])
            box = QCheckBox(person_text(p))
            box.setChecked(mine is None)
            grid.addWidget(box, r, 0)
            how = edit = None
            if mine is not None:
                box.setText(person_text(p) + f"   · @{mine['name']} is already in this project")
                how = QComboBox()
                how.addItem("Add as", ADD)
                how.addItem(f"Replace @{mine['name']}", REPLACE)
                edit = QLineEdit(free_name(tree, p["name"], taken))
                taken.add(edit.text().lower())
                edit.setToolTip("The name the saved person gets in this project (@Name in the prompts)")
                how.currentIndexChanged.connect(lambda _i, h=how, e=edit: e.setVisible(h.currentData() == ADD))
                grid.addWidget(how, r, 1)
                grid.addWidget(edit, r, 2)
            self.rows.append((p, box, how, edit))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        v.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.clicked.connect(self.reject)
        self.b_ok = QPushButton("Bring in")
        self.b_ok.setDefault(True)
        self.b_ok.clicked.connect(self.accept)
        row.addWidget(self.b_cancel)
        row.addWidget(self.b_ok)
        v.addLayout(row)

    def choices(self) -> list[CL.Choice]:
        out = []
        for p, box, how, edit in self.rows:
            if not box.isChecked():
                continue
            if how is None or edit is None:
                out.append(CL.Choice(p["name"], p["name"]))
            elif how.currentData() == REPLACE:
                out.append(CL.Choice(p["name"], p["name"], replace=True))
            else:
                out.append(CL.Choice(p["name"], edit.text().strip()))
        return out


class CastLibraryDialog(QDialog):
    def __init__(self, win: MainWindow, lib: CL.CastLibrary | None = None):
        super().__init__(win)
        self.win, self.lib = win, lib or CL.library()
        self.setWindowTitle("Cast library")
        fit_size(self, 640, 420)
        remember_geometry(self, win.qs, "castLibrary")
        v = QVBoxLayout(self)
        info = label("Casts kept for all your projects: their people with face pictures, descriptions and voices "
                     "(places, objects and styles are not kept). Bringing one in copies the people into the "
                     f"project; deleting or renaming a cast here changes no project. They are kept in {self.lib.dir}.",
                     level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        self.list = QListWidget()
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.itemDoubleClicked.connect(lambda _i: self.bring_in())
        v.addWidget(self.list, 1)
        self.empty = label("No casts saved yet: Save this cast keeps the people of this project.", level="tertiary")
        v.addWidget(self.empty)
        row = QHBoxLayout()
        self.b_save = QPushButton("Save this cast…")
        self.b_save.setToolTip("Keep the people of this project's cast in the library")
        self.b_save.clicked.connect(lambda: self.save())
        self.b_rename = QPushButton("Rename…")
        self.b_rename.clicked.connect(lambda: self.rename())
        self.b_delete = QPushButton("Delete")
        self.b_delete.setToolTip("Take the cast out of the library (projects that use its people keep their copy)")
        self.b_delete.clicked.connect(lambda: self.delete())
        for b in (self.b_save, self.b_rename, self.b_delete):
            row.addWidget(b)
        row.addStretch(1)
        self.b_bring = QPushButton("Bring in…")
        self.b_bring.setDefault(True)
        self.b_bring.setToolTip("Choose people of the selected cast to bring into this project")
        self.b_bring.clicked.connect(lambda: self.bring_in())
        self.b_close = QPushButton("Close")
        self.b_close.clicked.connect(self.reject)
        row.addWidget(self.b_bring)
        row.addWidget(self.b_close)
        v.addLayout(row)
        self.fill()

    def fill(self, select: str | None = None) -> None:
        keep = select or self.selected_id()
        self.list.clear()
        for c in self.lib.list():
            it = QListWidgetItem(cast_text(c))
            it.setData(Qt.ItemDataRole.UserRole, c["id"])
            self.list.addItem(it)
            if c["id"] == keep:
                self.list.setCurrentItem(it)
        self.list.setVisible(self.list.count() > 0)
        self.empty.setVisible(self.list.count() == 0)
        self._update_buttons()

    def selected_id(self) -> str | None:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def _update_buttons(self) -> None:
        picked = self.selected_id() is not None
        has_people = self.win.tree is not None and bool(CL.people_of(self.win.tree))
        self.b_save.setEnabled(has_people)
        self.b_bring.setEnabled(picked and self.win.project is not None and not self.win.running)
        self.b_rename.setEnabled(picked)
        self.b_delete.setEnabled(picked)

    # ---- the buttons (the arguments: what a test or a caller gives instead of asking)
    def save(self, name: str | None = None) -> dict | None:
        if self.win.tree is None or self.win.project is None:
            return None
        if name is None:
            name, ok = QInputDialog.getText(self, "Save this cast", "Name of the cast in the library:",
                                            text=self.win.project.name)
            if not ok:
                return None
        try:
            entry = self.lib.save(self.win.project, self.win.tree, name)
        except (ValueError, OSError) as e:
            D.failed(self, "Cast library", "save the cast", e)
            return None
        self.fill(select=entry["id"])
        self.win.status.setText(f"The cast is in the cast library as “{entry['name']}” "
                                f"({plural(len(entry['people']), 'person', 'people')}).")
        return entry

    def bring_in(self, choices: list[CL.Choice] | None = None) -> BringInDialog | bool:
        """Bring people of the selected cast in: choose them in BringInDialog (choices: given instead). Without
        modal dialogs the BringInDialog is made and handed back (its accept brings them in)."""
        cid = self.selected_id()
        saved = self.lib.get(cid) if cid else None
        if saved is None or self.win.tree is None or not self.b_bring.isEnabled():
            return False
        if choices is not None:
            return self._bring(saved, choices)
        dlg = BringInDialog(self, saved, self.win.tree, self.win.qs)
        dlg.accepted.connect(lambda: self._bring(saved, dlg.choices()))
        if not self.win.modal_dialogs:
            return dlg
        return dlg.exec() == QDialog.DialogCode.Accepted

    def _bring(self, saved: dict, choices: list[CL.Choice]) -> bool:
        if not choices:
            self.win.status.setText("Nobody was ticked: nothing brought in.")
            return False
        if self.win.bring_in_cast(saved["id"], choices, self.lib):
            self.accept()
            return True
        return False

    def rename(self, new: str | None = None) -> bool:
        cid = self.selected_id()
        c = self.lib.get(cid) if cid else None
        if c is None:
            return False
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename the cast", "New name:", text=c["name"])
            if not ok:
                return False
        try:
            self.lib.rename(c["id"], new)
        except (ValueError, OSError) as e:
            D.failed(self, "Cast library", "rename the cast", e)
            return False
        self.fill(select=c["id"])
        return True

    def delete(self, confirm: bool = True) -> bool:
        cid = self.selected_id()
        c = self.lib.get(cid) if cid else None
        if c is None:
            return False
        if confirm and not D.confirm(self, "Delete the cast", f"Delete “{c['name']}” from the cast library? "
                                     "Projects that use its people keep their own copy."):
            return False
        try:
            self.lib.delete(c["id"])
        except OSError as e:
            D.failed(self, "Cast library", "delete the cast", e)
            return False
        self.fill()
        return True
