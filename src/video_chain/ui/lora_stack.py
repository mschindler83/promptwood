"""A list of LoRAs with a strength each (rows of LoRA, strength, remove; '+ LoRA'), the LoRA picker (search, full
names, other files), and the helpers to fill a LoRA combo box. Used by the clip cards (H3 LoRAs) and the image edit window (Qwen LoRAs)."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from .. import tree as T
from .widgets import guard_wheel, icon_button, label


LORA_MIN, LORA_MAX = -10.0, 10.0       # strengths (ComfyUI allows -100..100 for LoRAs, -10..10 for the turbo LoRA)


def lora_path(p: str) -> str:
    """How a chosen LoRA file is stored: absolute, links kept as they are (a link in ComfyUI's LoRA folder keeps
    its own name)."""
    return os.path.abspath(os.path.expanduser(p))


def lora_label(path: str | None) -> str:
    return Path(path).stem if path else "none"


def fill_lora_combo(combo: QComboBox, known: list[tuple[str, str]], current: str | None) -> None:
    """Append the offered LoRAs to a combo box and select current (also when it is stored as the target of a
    listed link, or is not in the list at all)."""
    for path, text in known:
        combo.addItem(text, path)
        combo.setItemData(combo.count() - 1, path, Qt.ToolTipRole)
    if not current or current == T.NO_LORA:
        combo.setCurrentIndex(1 if current == T.NO_LORA and combo.findData(T.NO_LORA) >= 0 else 0)
        return
    i = combo.findData(current)
    if i < 0:
        real = os.path.realpath(current)
        i = next((k for k in range(combo.count()) if isinstance(combo.itemData(k), str)
                  and combo.itemData(k) != T.NO_LORA and os.path.realpath(combo.itemData(k)) == real), -1)
    if i < 0:
        combo.addItem(lora_label(current) + ("" if Path(current).is_file() else " (missing)"), current)
        i = combo.count() - 1
    combo.setCurrentIndex(i)


def _home(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home + os.sep) else path


def matches(query: str, *texts: str) -> bool:
    """Every word of the query is in one of the texts (case does not matter; '-', '_' and spaces count the same)."""
    norm = lambda t: t.lower().replace("_", " ").replace("-", " ")   # noqa: E731
    hay = " ".join(norm(t) for t in texts)
    return all(w in hay for w in norm(query).split())


class LoraPickerDialog(QDialog):
    """Choose a LoRA: a search field over the full names and folders, a list with the whole name of each LoRA
    (its folder below it), 'Other file…' for a file from anywhere. chosen() -> the path, or None."""

    def __init__(self, parent, known: list[tuple[str, str]], current: str | None = None,
                 used: set[str] | None = None, title: str = "Choose a LoRA", file_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 620)
        self.known, self.file_dir, self._chosen = list(known), file_dir, None
        used = {os.path.realpath(u) for u in (used or set())}
        v = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search LoRAs by name or folder … (several words: all must match)")
        self.search.setClearButtonEnabled(True)
        v.addWidget(self.search)
        self.count = label("", level="tertiary")
        v.addWidget(self.count)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(False)
        self.list.setAlternatingRowColors(True)
        v.addWidget(self.list, 1)
        for path, text in sorted(self.known, key=lambda x: Path(x[0]).stem.lower()):
            name = Path(path).stem
            mark = "   · in use" if os.path.realpath(path) in used else ""
            it = QListWidgetItem(f"{name}{mark}\n{_home(str(Path(path).parent))}")
            it.setData(Qt.UserRole, path)
            it.setData(Qt.UserRole + 1, (name, str(Path(path).parent), text))
            it.setToolTip(path)
            self.list.addItem(it)
            if current and os.path.realpath(path) == os.path.realpath(current):
                self.list.setCurrentItem(it)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Choose")
        self.b_other = self.buttons.addButton("Other file…", QDialogButtonBox.ResetRole)
        self.b_other.setToolTip("A .safetensors file from any folder (ComfyUI must find it in its LoRA folders to "
                                "use it)")
        self.b_other.clicked.connect(self._other)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._accept)
        self.search.installEventFilter(self)
        self.list.itemDoubleClicked.connect(lambda _it: self._accept())
        self.list.currentItemChanged.connect(lambda *_: self._update())
        self._filter("")
        self.search.setFocus()

    def eventFilter(self, obj, e):                 # noqa: N802  (Qt naming): Down from the search goes to the list
        from PySide6.QtCore import QEvent
        if obj is self.search and e.type() == QEvent.KeyPress and e.key() == Qt.Key_Down:
            self.list.setFocus()
            if self.list.currentItem() is None or self.list.currentItem().isHidden():
                self._select_first()
            return True
        return False

    def visible_paths(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if not self.list.item(i).isHidden()]

    def _select_first(self):
        for i in range(self.list.count()):
            if not self.list.item(i).isHidden():
                self.list.setCurrentRow(i)
                return

    def _filter(self, text: str):
        shown = 0
        for i in range(self.list.count()):
            it = self.list.item(i)
            ok = matches(text, *it.data(Qt.UserRole + 1))
            it.setHidden(not ok)
            shown += ok
        if self.list.currentItem() is None or self.list.currentItem().isHidden():
            self._select_first()
        total = self.list.count()
        self.count.setText(f"{shown} of {total} LoRA{'s' if total != 1 else ''}" if text.strip() else
                           f"{total} LoRA{'s' if total != 1 else ''}" if total else
                           "No LoRAs found in ComfyUI's LoRA folders (Other file… for one from elsewhere)")
        self._update()

    def _update(self):
        it = self.list.currentItem()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(it is not None and not it.isHidden())

    def _accept(self):
        it = self.list.currentItem()
        if it is not None and not it.isHidden():
            self._chosen = it.data(Qt.UserRole)
            self.accept()

    def _other(self):
        from PySide6.QtWidgets import QFileDialog
        f, _ = QFileDialog.getOpenFileName(self, "LoRA file", self.file_dir, "LoRA (*.safetensors)")
        if f:
            self._chosen = lora_path(f)
            self.accept()

    def chosen(self) -> str | None:
        return self._chosen


def pick_lora(parent, known, current=None, used=None, file_dir: str = "") -> str | None:
    """Show the LoRA picker; the chosen path or None. (Tests replace LoraStack.pick.)"""
    dlg = LoraPickerDialog(parent, known, current, used, file_dir=file_dir)
    try:
        return dlg.chosen() if dlg.exec() == QDialog.Accepted else None
    finally:
        dlg.deleteLater()


class LoraStack(QWidget):
    """A list of LoRAs: rows of (LoRA, strength, remove) and '+ LoRA'; a row's LoRA button and '+ LoRA' open the
    LoRA picker (search, full names, other files). items() -> [{"path", "strength"}] in order. None by default."""
    changed = Signal()
    fileWanted = Signal()                          # kept for callers that add a file themselves
    picked = Signal(str)                           # a LoRA was chosen in the picker (e.g. a file from elsewhere)
    pick = staticmethod(pick_lora)

    def __init__(self, parent=None, add_text: str = "+ LoRA"):
        super().__init__(parent)
        self.known: list[tuple[str, str]] = []
        self.file_dir = ""
        self.rows: list[tuple[QWidget, QPushButton, QDoubleSpinBox]] = []
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(4)
        brow = QHBoxLayout()
        brow.setSpacing(4)
        self.none = label("none", level="tertiary")
        self.b_add = QPushButton(add_text)
        self.b_add.setToolTip("Add a LoRA: search by name, or take a file from another folder; several are applied "
                              "in this order")
        self.b_add.clicked.connect(self.add_from_picker)
        self.b_add.setAutoDefault(False)               # Enter in a text field of the window must not open it
        brow.addWidget(self.none)
        brow.addWidget(self.b_add)
        brow.addStretch(1)
        self.v.addLayout(brow)
        self._enabled = True

    def set_items(self, known: list[tuple[str, str]], items: list[dict]):
        self.known = list(known)
        for w, _b, _s in self.rows:
            self.v.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self.rows = []
        for it in items or []:
            if it.get("path"):
                self._add_row(it["path"], float(it.get("strength", 1.0)))
        self.none.setVisible(not self.rows)

    def _used(self) -> set[str]:
        return {b.property("path") for _w, b, _s in self.rows}

    def add_from_picker(self):
        path = self.pick(self.window(), self.known, None, self._used(), self.file_dir)
        if path:
            self._add_row(lora_path(path), 1.0)
            self.changed.emit()
            self.picked.emit(lora_path(path))

    def _choose_again(self, button: QPushButton):
        path = self.pick(self.window(), self.known, button.property("path"), self._used(), self.file_dir)
        if path and path != button.property("path"):
            self._set_button(button, lora_path(path))
            self.changed.emit()
            self.picked.emit(lora_path(path))

    def _set_button(self, button: QPushButton, path: str):
        button.setProperty("path", path)
        missing = "" if Path(path).is_file() else " (missing)"
        button.setText(Path(path).stem + missing)
        button.setToolTip(f"{path}{missing}\nClick to choose another LoRA")

    def _add_row(self, path: str, strength: float):
        w = QWidget(self)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        button = QPushButton(w)
        button.setAutoDefault(False)                   # Enter in a text field must not open the picker
        button.setStyleSheet("text-align: left; padding-left: 6px;")
        button.setMinimumWidth(60)
        from PySide6.QtWidgets import QSizePolicy
        button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)        # long names shorten, the row stays
        self._set_button(button, path)
        button.clicked.connect(lambda: self._choose_again(button))
        spin = QDoubleSpinBox(w)
        spin.setRange(LORA_MIN, LORA_MAX)              # some LoRAs need negative strengths (e.g. sliders)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(strength)
        spin.setToolTip(f"Strength of this LoRA ({LORA_MIN:g} … {LORA_MAX:g}; negative values reverse its effect, "
                        "e.g. for slider LoRAs)")
        rm = icon_button("x", "Remove this LoRA", w)
        h.addWidget(button, 1)
        h.addWidget(spin)
        h.addWidget(rm)
        self.v.insertWidget(len(self.rows), w)
        row = (w, button, spin)
        self.rows.append(row)
        spin.valueChanged.connect(lambda *_: self.changed.emit())
        rm.clicked.connect(lambda: self._remove(row))
        for x in (button, spin, rm):
            x.setEnabled(self._enabled)
        guard_wheel(spin)
        self.none.hide()

    def _remove(self, row):
        if row in self.rows:
            self.rows.remove(row)
            self.v.removeWidget(row[0])
            row[0].setParent(None)
            row[0].deleteLater()
            self.none.setVisible(not self.rows)
            self.changed.emit()

    def items(self) -> list[dict]:
        return [{"path": b.property("path"), "strength": round(spin.value(), 2)} for _w, b, spin in self.rows]

    def setEnabled(self, on: bool):          # noqa: N802  (Qt naming)
        self._enabled = on
        self.b_add.setEnabled(on)
        for _w, b, spin in self.rows:
            b.setEnabled(on)
            spin.setEnabled(on)
