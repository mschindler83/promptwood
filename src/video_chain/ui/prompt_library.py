"""The prompt library window (a clip card's Library button, the image window's Library button): prompts kept for
every project (video_chain.prompt_library), in folders per kind. The prompt of the field it was opened from is saved
from here, and a saved one is put into that field: at the cursor (Insert) or instead of its text (Replace)."""
from __future__ import annotations

import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QInputDialog,
                               QLineEdit, QPlainTextEdit, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import prompt_library as PL
from . import dialogs as D
from .problems import insert_at_cursor
from .text_edit import replace_text
from .ui_logic import plural
from .widgets import fit_size, label, remember_geometry

ROLE_ID = Qt.ItemDataRole.UserRole                 # a prompt's id (a folder item: None)
ROLE_FOLDER = Qt.ItemDataRole.UserRole + 1         # a folder item's folder name


class SaveDialog(QDialog):
    """The name and folder a prompt is saved under."""

    def __init__(self, parent, folders: list[str], name: str = "", folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Save the prompt")
        form = QFormLayout(self)
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("e.g. Slow push in, golden hour")
        self.folder = QComboBox()
        self.folder.setEditable(True)
        self.folder.addItem("")
        self.folder.addItems(folders)
        self.folder.setCurrentText(folder)
        self.folder.setToolTip("Choose a folder, type a new one, or leave it empty")
        form.addRow("Name", self.name)
        form.addRow("Folder", self.folder)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)


class PromptLibraryDialog(QDialog):
    """target: the prompt field it was opened from (None: only browsing); kind: what it shows first."""

    def __init__(self, parent, kind: str, target: QPlainTextEdit | None = None, lib: PL.PromptLibrary | None = None,
                 settings=None):
        super().__init__(parent)
        self.lib, self.target = lib or PL.library(), target
        self._gone = False                            # the field it was opened from was deleted
        self._closed: set[str] | None = None          # the folders the user collapsed (None: not shown yet)
        self.setWindowTitle("Prompt library")
        fit_size(self, 820, 520)
        remember_geometry(self, settings, "promptLibrary")
        v = QVBoxLayout(self)
        info = label("Prompts kept for all your projects, in folders. Save the prompt you are writing, or put a "
                     f"saved one into it. They are kept in {self.lib.dir}.", level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        top = QHBoxLayout()
        self.kind = QComboBox()
        for key, text in PL.KINDS.items():
            self.kind.addItem(f"{text} prompts", key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        self.kind.setToolTip("Which prompts are shown (a saved prompt can be used in any prompt field)")
        self.kind.currentIndexChanged.connect(lambda *_: self.fill())
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search names and text")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda *_: self.fill())
        top.addWidget(self.kind)
        top.addWidget(self.search, 1)
        v.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._selected)
        self.tree.itemDoubleClicked.connect(lambda it, _c: self._double(it))
        split.addWidget(self.tree)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.where = label("", level="tertiary")
        rv.addWidget(self.where)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Choose a prompt to see it")
        rv.addWidget(self.preview, 1)
        split.addWidget(right)
        split.setSizes([300, 480])
        v.addWidget(split, 1)
        self.empty = label("No prompts saved here yet: Save this prompt keeps the one you are writing.",
                           level="tertiary")
        v.addWidget(self.empty)
        row = QHBoxLayout()
        self.b_save = QPushButton("Save this prompt…")
        self.b_save.setToolTip("Keep the prompt you are writing in the library")
        self.b_save.clicked.connect(lambda: self.save())
        self.b_update = QPushButton("Update")
        self.b_update.setToolTip("Replace the selected saved prompt's text with the prompt you are writing")
        self.b_update.clicked.connect(lambda: self.update_selected())
        self.b_rename = QPushButton("Rename…")
        self.b_rename.clicked.connect(lambda: self.rename())
        self.b_move = QPushButton("Move…")
        self.b_move.setToolTip("Put the selected prompt into another folder")
        self.b_move.clicked.connect(lambda: self.move_to())
        self.b_delete = QPushButton("Delete")
        self.b_delete.clicked.connect(lambda: self.delete())
        for b in (self.b_save, self.b_update, self.b_rename, self.b_move, self.b_delete):
            row.addWidget(b)
        row.addStretch(1)
        self.b_insert = QPushButton("Insert at cursor")
        self.b_insert.setToolTip("Put the selected prompt into your prompt where the cursor is")
        self.b_insert.clicked.connect(lambda: self.use("insert"))
        self.b_replace = QPushButton("Replace")
        self.b_replace.setToolTip("Put the selected prompt in place of your prompt's text (Ctrl+Z there takes it "
                                  "back)")
        self.b_replace.clicked.connect(lambda: self.use("replace"))
        self.b_close = QPushButton("Close")
        self.b_close.clicked.connect(self.reject)
        for b in (self.b_insert, self.b_replace, self.b_close):
            row.addWidget(b)
        v.addLayout(row)
        if target is not None:                         # (a deleted field disconnects by itself)
            target.textChanged.connect(self._selected)
        self.fill()

    # ---- what is shown
    def current_kind(self) -> str:
        return str(self.kind.currentData())

    def _field(self) -> QPlainTextEdit | None:
        """The prompt field, None when there is none or it is gone (a clip card rebuilt meanwhile)."""
        if self.target is not None and not shiboken6.isValid(self.target):
            self.target, self._gone = None, True
        return self.target

    def _text(self) -> str:
        f = self._field()
        return f.toPlainText().strip() if f is not None else ""

    def _editable(self) -> bool:
        f = self._field()
        return f is not None and not f.isReadOnly()

    def fill(self, select: str | None = None) -> None:
        keep = select or self.selected_id()
        keep_folder = None if select else self.selected_folder()
        if self._closed is not None:                   # (what the user collapsed stays so)
            shown = {str(it.data(0, ROLE_FOLDER)).lower(): it.isExpanded() for it in self._folder_items()}
            self._closed = {f for f in self._closed if shown.get(f) is not True} | {f for f, e in shown.items()
                                                                                     if not e}
        else:
            self._closed = set()
        words = self.search.text().lower().split()
        prompts = [p for p in self.lib.list(self.current_kind())
                   if all(w in (p["name"] + " " + p["text"] + " " + p["folder"]).lower() for w in words)]
        prompts.sort(key=lambda p: (not p["folder"], p["folder"].lower(), p["name"].lower()))   # folders first
        self.tree.clear()
        folders: dict[str, QTreeWidgetItem] = {}
        pick = None
        for p in prompts:
            parent = None
            if p["folder"]:
                parent = folders.get(p["folder"].lower())
                if parent is None:
                    parent = QTreeWidgetItem([p["folder"]])
                    parent.setData(0, ROLE_FOLDER, p["folder"])
                    f = parent.font(0)
                    f.setBold(True)
                    parent.setFont(0, f)
                    folders[p["folder"].lower()] = parent
                    self.tree.addTopLevelItem(parent)
            it = QTreeWidgetItem([p["name"]])
            it.setData(0, ROLE_ID, p["id"])
            it.setToolTip(0, p["text"][:400])
            if parent is None:
                self.tree.addTopLevelItem(it)
            else:
                parent.addChild(it)
            if p["id"] == keep:
                pick = it
        for name, it in folders.items():
            it.setText(0, f"{it.data(0, ROLE_FOLDER)}   ({it.childCount()})")
            it.setExpanded(bool(words) or name not in self._closed or pick is not None and pick.parent() is it)
            if pick is None and keep_folder is not None and name == keep_folder.lower():
                pick = it
        if pick is not None:
            self.tree.setCurrentItem(pick)
        self.tree.setVisible(bool(prompts) or bool(words))
        self.empty.setVisible(not prompts and not words)
        self._selected()

    def _folder_items(self) -> list[QTreeWidgetItem]:
        items = [self.tree.topLevelItem(k) for k in range(self.tree.topLevelItemCount())]
        return [it for it in items if it is not None and it.data(0, ROLE_FOLDER)]

    def selected_id(self) -> str | None:
        it = self.tree.currentItem()
        return it.data(0, ROLE_ID) if it is not None else None

    def selected_folder(self) -> str | None:
        it = self.tree.currentItem()
        return it.data(0, ROLE_FOLDER) if it is not None and not it.data(0, ROLE_ID) else None

    def select(self, pid: str) -> bool:
        """Select a prompt by id (it is shown: the search is cleared when it hides it)."""
        self.fill(select=pid)
        if self.selected_id() != pid and self.search.text():
            self.search.clear()
            self.fill(select=pid)
        return self.selected_id() == pid

    def _selected(self) -> None:
        pid, folder = self.selected_id(), self.selected_folder()
        p = self.lib.get(pid) if pid else None
        self.preview.setPlainText(p["text"] if p else "")
        self.where.setText((f"In “{p['folder']}” · " if p and p["folder"] else "") +
                           (f"saved {p.get('updated') or p.get('saved') or ''}" if p else
                            f"Folder “{folder}”" if folder else ""))
        has_text, editable = bool(self._text()), self._editable()
        self.b_save.setEnabled(has_text)
        self.b_update.setEnabled(p is not None and has_text and p["text"] != self._text())
        self.b_rename.setEnabled(p is not None or folder is not None)
        self.b_rename.setToolTip("Rename the selected folder" if folder else "Rename the selected prompt")
        self.b_move.setEnabled(p is not None)
        self.b_delete.setEnabled(p is not None or folder is not None)
        self.b_delete.setToolTip("Delete the selected folder and its prompts" if folder else
                                 "Delete the selected prompt from the library")
        self.b_insert.setEnabled(p is not None and editable)
        self.b_replace.setEnabled(p is not None and editable)
        why = ("" if editable else "That prompt is not shown any more: open the library from it again"
               if self._gone else "Open the library from a prompt to put a saved one into it" if self.target is None
               else "That prompt cannot be changed now")
        for b in (self.b_insert, self.b_replace):
            if why:
                b.setToolTip(why)

    def _double(self, it: QTreeWidgetItem) -> None:
        """A double-click on a prompt: into an empty field it replaces, else it goes in at the cursor."""
        if it.data(0, ROLE_ID) and self._editable():
            self.use("insert" if self._text() else "replace")

    # ---- the buttons (the arguments: what a test or a caller gives instead of asking)
    def save(self, name: str | None = None, folder: str | None = None, replace: bool | None = None) -> dict | None:
        text, kind = self._text(), self.current_kind()
        if not text:
            return None
        if name is None or folder is None:
            sel = self.lib.get(self.selected_id() or "")
            dlg = SaveDialog(self, self.lib.folders(kind), folder=self.selected_folder() or (sel or {}).get(
                "folder", ""))
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            name, folder = dlg.name.text(), dlg.folder.currentText()
        old = self.lib.find(kind, folder, name)
        if old is not None and replace is None:
            replace = D.confirm(self, "Save the prompt", f"There is already “{old['name']}” here. Replace its "
                                "text with this prompt?")
        if old is not None and not replace:
            return None
        try:
            entry = self.lib.save(kind, name, text, folder=folder, replace=bool(replace))
        except (ValueError, OSError) as e:
            D.failed(self, "Prompt library", "save the prompt", e)
            return None
        self.select(entry["id"])
        return entry

    def update_selected(self, confirm: bool = True) -> bool:
        pid, text = self.selected_id(), self._text()
        p = self.lib.get(pid) if pid else None
        if p is None or not text:
            return False
        if confirm and not D.confirm(self, "Update the prompt", f"Replace the text of “{p['name']}” with the "
                                     "prompt you are writing?"):
            return False
        try:
            self.lib.update_text(p["id"], text)
        except (ValueError, OSError) as e:
            D.failed(self, "Prompt library", "update the prompt", e)
            return False
        self.fill(select=p["id"])
        return True

    def rename(self, new: str | None = None) -> bool:
        pid, folder = self.selected_id(), self.selected_folder()
        p = self.lib.get(pid) if pid else None
        if p is None and folder is None:
            return False
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename", "New name:", text=p["name"] if p else folder or "")
            if not ok:
                return False
        try:
            if p is not None:
                self.lib.rename(p["id"], new)
            else:
                self.lib.rename_folder(self.current_kind(), str(folder), new)
        except (ValueError, OSError) as e:
            D.failed(self, "Prompt library", "rename it", e)
            return False
        self.fill(select=p["id"] if p else None)
        return True

    def move_to(self, folder: str | None = None) -> bool:
        pid = self.selected_id()
        p = self.lib.get(pid) if pid else None
        if p is None:
            return False
        if folder is None:
            folders = [""] + self.lib.folders(self.current_kind())
            folder, ok = QInputDialog.getItem(self, "Move the prompt", "Into the folder (type a new one, or leave "
                                              "it empty for none):", folders,
                                              max(0, folders.index(p["folder"]) if p["folder"] in folders else 0),
                                              True)
            if not ok:
                return False
        try:
            self.lib.move(p["id"], folder)
        except (ValueError, OSError) as e:
            D.failed(self, "Prompt library", "move the prompt", e)
            return False
        self.fill(select=p["id"])
        return True

    def delete(self, confirm: bool = True) -> bool:
        pid, folder = self.selected_id(), self.selected_folder()
        p = self.lib.get(pid) if pid else None
        try:
            if p is not None:
                if confirm and not D.confirm(self, "Delete the prompt", f"Delete “{p['name']}” from the library?"):
                    return False
                self.lib.delete(p["id"])
            elif folder is not None:
                n = len([q for q in self.lib.list(self.current_kind()) if q["folder"].lower() == folder.lower()])
                if confirm and not D.confirm(self, "Delete the folder", f"Delete the folder “{folder}” and its "
                                             f"{plural(n, 'prompt')} from the library?"):
                    return False
                self.lib.delete_folder(self.current_kind(), folder)
            else:
                return False
        except (ValueError, OSError) as e:
            D.failed(self, "Prompt library", "delete it", e)
            return False
        self.fill()
        return True

    def use(self, how: str) -> bool:
        """Put the selected prompt into the field: "insert" at the cursor, "replace" its text."""
        pid = self.selected_id()
        p = self.lib.get(pid) if pid else None
        if p is None or not self._editable():
            return False
        field = self._field()
        assert field is not None
        if how == "replace":
            replace_text(field, p["text"])
        else:
            insert_at_cursor(field, p["text"])          # (spaced from the words around it)
        self.accept()
        return True
