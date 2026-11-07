"""A list of LoRAs with a strength each (rows of LoRA, strength, remove; '+ LoRA'; a file button), and the helpers
to fill a LoRA combo box. Used by the clip cards (H3 LoRAs) and the image edit window (Qwen LoRAs)."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .. import tree as T
from .widgets import guard_wheel, icon_button, label


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


class LoraStack(QWidget):
    """A clip's LoRAs: rows of (LoRA, strength, remove), '+ LoRA' and a file button for a LoRA from elsewhere.
    items() -> [{"path", "strength"}] in order (rows without a chosen LoRA are left out). None by default."""
    changed = Signal()
    fileWanted = Signal()

    def __init__(self, parent=None, add_text: str = "+ LoRA"):
        super().__init__(parent)
        self.known: list[tuple[str, str]] = []
        self.rows: list[tuple[QWidget, QComboBox, QDoubleSpinBox]] = []
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(4)
        brow = QHBoxLayout()
        brow.setSpacing(4)
        self.none = label("none", level="tertiary")
        self.b_add = QPushButton(add_text)
        self.b_add.setToolTip("Add a LoRA (from ComfyUI's LoRA folders); several are applied in this order")
        self.b_add.clicked.connect(lambda: (self._add_row(None, 1.0), self.changed.emit()))
        self.b_file = icon_button("fopen", "Add a LoRA file from another folder (.safetensors)")
        self.b_file.clicked.connect(self.fileWanted.emit)
        brow.addWidget(self.none)
        brow.addWidget(self.b_add)
        brow.addWidget(self.b_file)
        brow.addStretch(1)
        self.v.addLayout(brow)
        self._enabled = True

    def set_items(self, known: list[tuple[str, str]], items: list[dict]):
        self.known = list(known)
        empty = sum(1 for _w, c, _s in self.rows if not c.currentData())   # '+ LoRA' rows not chosen yet stay
        for w, _c, _s in self.rows:
            self.v.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self.rows = []
        for it in items or []:
            self._add_row(it.get("path"), float(it.get("strength", 1.0)))
        for _ in range(empty):
            self._add_row(None, 1.0)
        self.none.setVisible(not self.rows)

    def _add_row(self, path: str | None, strength: float):
        w = QWidget(self)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        combo = QComboBox(w)
        combo.addItem("choose a LoRA …", None)
        fill_lora_combo(combo, self.known, path)
        combo.setProperty("stored", path or "")   # the path as stored (the list may show a link to the same file)
        spin = QDoubleSpinBox(w)
        spin.setRange(0.0, 2.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(strength)
        spin.setToolTip("Strength of this LoRA")
        rm = icon_button("x", "Remove this LoRA", w)
        h.addWidget(combo, 1)
        h.addWidget(spin)
        h.addWidget(rm)
        self.v.insertWidget(len(self.rows), w)
        row = (w, combo, spin)
        self.rows.append(row)
        combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        spin.valueChanged.connect(lambda *_: self.changed.emit())
        rm.clicked.connect(lambda: self._remove(row))
        for x in (combo, spin, rm):
            x.setEnabled(self._enabled)
        guard_wheel(combo, spin)
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
        out = []
        for _w, combo, spin in self.rows:
            path = combo.currentData()
            if path and path != T.NO_LORA:
                stored = combo.property("stored") or ""
                if stored and stored != path and os.path.realpath(stored) == os.path.realpath(path):
                    path = stored                 # the same file: keep its path, so the clip stays done
                out.append({"path": lora_path(path), "strength": round(spin.value(), 2)})
        return out

    def setEnabled(self, on: bool):          # noqa: N802  (Qt naming)
        self._enabled = on
        self.b_add.setEnabled(on)
        self.b_file.setEnabled(on)
        for _w, combo, spin in self.rows:
            combo.setEnabled(on)
            spin.setEnabled(on)
