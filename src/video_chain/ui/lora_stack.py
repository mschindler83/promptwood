"""A list of LoRAs with a strength each (rows of LoRA, strength, remove; '+ LoRA'), the LoRA picker (search, full
names, other files), and the helpers to fill a LoRA combo box. Used by the clip cards (H3 LoRAs) and the image edit window (Qwen LoRAs)."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from .. import tree as T
from .widgets import guard_wheel, icon_button, label


LORA_MIN, LORA_MAX = -10.0, 10.0       # strengths (ComfyUI allows -100..100 for LoRAs, -10..10 for the turbo LoRA)


def lora_path(p: str) -> str:
    """How a chosen LoRA file is stored: absolute, links kept as they are (a link in ComfyUI's LoRA folder keeps
    its own name)."""
    return os.path.abspath(os.path.expanduser(p))


def clean_words(text: str) -> list[str]:
    """Trigger words typed comma-separated: trimmed, empty ones dropped."""
    return [w.strip() for w in text.split(",") if w.strip()]


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
    filled = Signal()                              # set_items put a new list in (e.g. the remembered LoRAs)
    pick = staticmethod(pick_lora)

    def __init__(self, parent=None, add_text: str = "+ LoRA"):
        super().__init__(parent)
        self.known: list[tuple[str, str]] = []
        self.file_dir = ""
        self.rows: list[tuple[QWidget, QPushButton, QDoubleSpinBox]] = []
        self.use_prompt = None                     # set by the owner: put a text into the prompt these LoRAs are for
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(4)
        brow = QHBoxLayout()
        brow.setSpacing(4)
        self.none = label("none", level="tertiary")
        self.b_add = QPushButton(add_text)
        self.b_add.setToolTip("Add a LoRA (an add-on that teaches the model a look, a person or a motion): search "
                              "by name, or take a file from another folder; several are applied in this order")
        self.b_add.clicked.connect(self.add_from_picker)
        self.b_add.setAutoDefault(False)               # Enter in a text field of the window must not open it
        brow.addWidget(self.none)
        brow.addWidget(self.b_add)
        brow.addStretch(1)
        self.v.addLayout(brow)
        self._enabled = True

    def set_items(self, known: list[tuple[str, str]], items: list[dict]):
        self.known = list(known)
        wanted = [(it["path"], round(float(it.get("strength", 1.0)), 2)) for it in items or [] if it.get("path")]
        if wanted and wanted == [(b.property("path"), round(s.value(), 2)) for _w, b, s in self.rows]:
            self.filled.emit()                         # (the same list: the rows, and the reading of their files, stay)
            return
        for w, _b, _s in self.rows:
            self.v.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self.rows = []
        for it in items or []:
            if it.get("path"):
                self._add_row(it["path"], float(it.get("strength", 1.0)))
        self.none.setVisible(not self.rows)
        self.filled.emit()

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
        if path and lora_path(path) != lora_path(button.property("path") or ""):
            self._set_button(button, lora_path(path))
            self.changed.emit()
            self.picked.emit(lora_path(path))

    def _set_button(self, button: QPushButton, path: str):
        button.setProperty("path", path)
        missing = "" if Path(path).is_file() else " (missing)"
        button.setText(Path(path).stem + missing)
        button.setToolTip(f"{path}{missing}\nClick to choose another LoRA; right-click for its trigger words and CivitAI details")
        trig = getattr(button.parentWidget(), "trigger_label", None)
        if trig is not None:
            self._show_trigger(trig, path)

    @staticmethod
    def _show_trigger(lab, path: str):
        """The LoRA's trigger words under its row (triggers.py), with where they come from."""
        try:
            from .. import triggers
            info = triggers.db().info(path) if Path(path).is_file() else None
        except Exception:  # noqa: BLE001
            info = None
        if info is None:
            text, tip = "trigger words: not looked up yet", "Looked up in the background (file, then CivitAI)"
        elif info.get("suggested"):
            text = (f"trigger words: none used; its training tags suggest {', '.join(info['suggested'])} "
                    "(right-click to use them)")
            tip = "A guess from the tags on its training images: not put into prompts unless you take it"
        elif not info["words"]:
            text = "trigger words: none" + (" (CivitAI lists none)" if info["source"] == "civitai" else
                                            " (set by you)" if info["source"] == "user" else "")
            tip = info.get("url") or "Neither the file nor CivitAI names any"
        else:
            src = {"user": "set by you", "file": "from the file", "civitai": "from CivitAI"}.get(info["source"], "")
            text = f"trigger words: {', '.join(info['words'])}  ({src})"
            tip = (f"{info.get('model') or ''}  {info.get('url') or ''}".strip() or src)
        if info and info.get("base_model"):
            text += f"  · base model {info['base_model']}"
        lab.setText(text)
        lab.setToolTip(tip + "\nRight-click the LoRA to change them or to see all that is known about it "
                       "(example prompts with their pictures, description)")

    def strengths(self) -> dict[str, float]:
        return {b.property("path"): round(s.value(), 2) for _w, b, s in self.rows}

    def set_strength(self, path: str, value: float) -> bool:
        """Set a LoRA's strength as if typed (the owner hears `changed`); False when it is not in the stack or
        the stack cannot be changed now."""
        for _w, b, spin in self.rows:
            if b.property("path") == path and self._enabled:
                spin.setValue(max(LORA_MIN, min(LORA_MAX, float(value))))
                return True
        return False

    def window_args(self) -> dict:
        """What the LoRA window may do for this stack: set a strength, put an example into its prompt."""
        on = self._enabled
        return {"strengths": self.strengths(), "set_strength": self.set_strength if on else None,
                "use_prompt": self.use_prompt if on else None}

    def refresh_triggers(self):
        for w, b, _s in self.rows:
            if getattr(w, "trigger_label", None) is not None:
                self._show_trigger(w.trigger_label, b.property("path"))

    def _trigger_menu(self, button: QPushButton, pos):
        from PySide6.QtWidgets import QInputDialog, QMenu
        from .. import triggers
        path = button.property("path")
        if not path or not Path(path).is_file():
            return
        info = triggers.db().info(path) or {}
        menu = QMenu(self)
        a_take = menu.addAction(f"Use the suggested words ({', '.join(info['suggested'])})") \
            if info.get("suggested") else None
        a_edit = menu.addAction("Change the trigger words…")
        a_reset = menu.addAction("Use the found trigger words again")
        menu.addSeparator()
        a_details = menu.addAction("About this LoRA (example prompts with pictures, what the app knows)…")
        chosen = menu.exec(button.mapToGlobal(pos))
        if chosen is not None and chosen is a_details:
            from .lora_window import LoraWindow
            LoraWindow(self.window(), self.known, current=path, **self.window_args()).exec()
            self.refresh_triggers()
            return
        try:
            if chosen is not None and chosen is a_take:
                triggers.db().set_user(path, list(info["suggested"]))
            elif chosen is a_edit:
                words, ok = QInputDialog.getText(
                    self, "Trigger words", f"Trigger words of {Path(path).stem} (comma-separated; empty: none):",
                    text=", ".join(triggers.db().words(path) or info.get("suggested") or []))
                if not ok:
                    return
                triggers.db().set_user(path, clean_words(words))
            elif chosen is a_reset:
                triggers.db().set_user(path, None)
            else:
                return
        except OSError as e:
            QMessageBox.warning(self, "Trigger words", f"Could not save the trigger words: {e}")
            return
        self.refresh_triggers()
        self.changed.emit()

    def _add_row(self, path: str, strength: float):
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        outer.addLayout(h)
        w.trigger_label = label("", level="tertiary")
        w.trigger_label.setWordWrap(True)
        outer.addWidget(w.trigger_label)
        button = QPushButton(w)
        button.setAutoDefault(False)                   # Enter in a text field must not open the picker
        button.setStyleSheet("text-align: left; padding-left: 6px;")
        button.setMinimumWidth(60)
        from PySide6.QtWidgets import QSizePolicy
        button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)        # long names shorten, the row stays
        self._set_button(button, path)
        button.clicked.connect(lambda: self._choose_again(button))
        button.setContextMenuPolicy(Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(lambda pos: self._trigger_menu(button, pos))
        spin = QDoubleSpinBox(w)
        spin.setRange(LORA_MIN, LORA_MAX)              # some LoRAs need negative strengths (e.g. sliders)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(strength)
        spin.setToolTip(f"Strength of this LoRA ({LORA_MIN:g} … {LORA_MAX:g}; negative values reverse its effect, "
                        "e.g. for slider LoRAs)")
        rm = icon_button("x", "Remove this LoRA", w)
        w.remove_button = rm                           # (setEnabled reaches it through the row's widget)
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
        for w, b, spin in self.rows:
            b.setEnabled(on)
            spin.setEnabled(on)
            w.remove_button.setEnabled(on)
