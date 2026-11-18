"""The image window's Random LoRAs: images from one prompt, each with a random choice of the image model's LoRAs (the
pinned ones always, at most a number in all) at random strengths, a fixed number of them or until Stop (continuous).
The choosing itself is lora_random; this is the window that sets it up: a table of every LoRA of the model (use it or
leave it out, pin it, its strength range), with a search and one range for all LoRAs shown."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout,
                               QHBoxLayout, QHeaderView, QLineEdit, QPushButton, QSpinBox, QStyledItemDelegate,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import lora_random as R
from .lora_stack import LORA_MAX, LORA_MIN, lora_label
from .widgets import fit_size, guard_wheel, label, label_for

USE, PIN, NAME, LOW, HIGH = range(5)


def _strength_box(value: float, tip: str, parent=None) -> QDoubleSpinBox:
    box = QDoubleSpinBox(parent)
    box.setRange(LORA_MIN, LORA_MAX)
    box.setDecimals(R.DECIMALS)
    box.setSingleStep(0.05)
    box.setValue(value)
    box.setToolTip(tip)
    return box


class _StrengthDelegate(QStyledItemDelegate):
    """The From / To cells are edited with a strength box (negative strengths too)."""

    def createEditor(self, parent, option, index):      # noqa: N802  (Qt naming)
        return _strength_box(float(index.data(Qt.ItemDataRole.EditRole) or 0.0), "", parent)

    def setModelData(self, editor, model, index):       # noqa: N802
        editor.interpretText()
        model.setData(index, round(editor.value(), R.DECIMALS), Qt.ItemDataRole.EditRole)


def _check_item(on: bool, tip: str) -> QTableWidgetItem:
    it = QTableWidgetItem()
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
    it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
    it.setToolTip(tip)
    return it


def _number_item(value: float) -> QTableWidgetItem:
    it = QTableWidgetItem()
    it.setData(Qt.ItemDataRole.EditRole, float(value))
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
    return it


class RandomLorasDialog(QDialog):
    """Every LoRA of the image model: used or left out, pinned, its strength range; how many LoRAs an image may
    have, how many images, the same seed or a new one each. Generate N images or Continuous closes it with what to
    make (pool(), most, n, same_seed, continuous)."""

    def __init__(self, parent, set_up: dict, seed: int, labels: dict[str, str] | None = None, model: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Random LoRAs")
        self.continuous = False
        self._filling = False
        labels = labels or {}
        v = QVBoxLayout(self)
        k = len(set_up["choices"])
        intro = label(f"Every image has the same prompt and a random choice of the {k} LoRA{'s' if k != 1 else ''} "
                      f"ComfyUI has{' for ' + model if model else ''}, each at a random strength from its range. "
                      "Pinned LoRAs are in every image; LoRAs switched off under Use are never chosen. An image gets "
                      "at least one LoRA and at most the number below, pinned ones included, often fewer. The LoRAs "
                      "in the image window's list are not used for these images.", level="secondary")
        intro.setWordWrap(True)
        v.addWidget(intro)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the LoRAs")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _t: self._filter())
        top.addWidget(self.search, 1)
        self.all_low = _strength_box(0.5, "The lowest strength for every LoRA shown")
        self.all_high = _strength_box(1.0, "The highest strength for every LoRA shown")
        self.b_all = QPushButton("Set for the LoRAs shown")
        self.b_all.setToolTip("Give every LoRA the table shows now (all, or those the search finds) this range")
        self.b_all.clicked.connect(lambda: self.set_range_shown(self.all_low.value(), self.all_high.value()))
        top.addWidget(label("Range", level="tertiary"))
        top.addWidget(self.all_low)
        top.addWidget(label("to", level="tertiary"))
        top.addWidget(self.all_high)
        top.addWidget(self.b_all)
        v.addLayout(top)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Use", "Pin", "LoRA", "Strength from", "to"])
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        delegate = _StrengthDelegate(self.table)
        self.table.setItemDelegateForColumn(LOW, delegate)
        self.table.setItemDelegateForColumn(HIGH, delegate)
        head = self.table.horizontalHeader()
        for col in (USE, PIN, LOW, HIGH):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(NAME, QHeaderView.ResizeMode.Stretch)
        self.paths: list[str] = []
        self._filling = True
        for c in set_up["choices"]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            name = labels.get(c.path) or lora_label(c.path)
            self.table.setItem(row, USE, _check_item(c.on, f"Off: {name} is never chosen (unless it is pinned)"))
            self.table.setItem(row, PIN, _check_item(c.pinned, f"Pinned: every image has {name}"))
            it = QTableWidgetItem(name)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it.setToolTip(c.path)
            self.table.setItem(row, NAME, it)
            self.table.setItem(row, LOW, _number_item(c.low))
            self.table.setItem(row, HIGH, _number_item(c.high))
            self.paths.append(c.path)
        self._filling = False
        self.table.itemChanged.connect(self._item_changed)
        v.addWidget(self.table, 1)
        opts = QGridLayout()
        self.most = QSpinBox()
        self.most.setRange(1, R.MAX_MOST)
        self.most.setValue(set_up["most"])
        self.most.setToolTip("How many LoRAs an image may have at most, pinned ones included; each image gets a "
                             "random number from 1 (or the number pinned) up to this")
        self.most.valueChanged.connect(lambda _v: self._check())
        self.most_note = label("", level="tertiary")
        self.n = QSpinBox()
        self.n.setRange(1, R.MAX_N)
        self.n.setValue(set_up["n"])
        self.n.setToolTip("How many images Generate makes (Continuous makes images until you press Stop)")
        self.n.valueChanged.connect(lambda _v: self._check())
        self.same_seed = QCheckBox(f"The same seed for every image (the window's, {seed})")
        self.same_seed.setChecked(set_up["same_seed"])
        self.same_seed.setToolTip("On: only the LoRAs make the difference, so they can be compared. Off: every image "
                                  "gets a new random seed. The seed and the LoRAs are kept with each image either way")
        opts.addWidget(label_for("LoRAs per image, at most", self.most, level="tertiary"), 0, 0)
        opts.addWidget(self.most, 0, 1)
        opts.addWidget(self.most_note, 0, 2)
        opts.addWidget(label_for("Images", self.n, level="tertiary"), 1, 0)
        opts.addWidget(self.n, 1, 1)
        opts.addWidget(self.same_seed, 2, 0, 1, 3)
        opts.setColumnStretch(3, 1)
        v.addLayout(opts)
        self.problem = label("", severity="error")
        self.problem.setWordWrap(True)
        v.addWidget(self.problem)
        box = QDialogButtonBox()
        self.b_make = QPushButton()
        self.b_make.setProperty("variant", "primary")
        self.b_make.clicked.connect(lambda: self._go(False))
        self.b_continuous = QPushButton("Continuous")
        self.b_continuous.setToolTip("Make images with random LoRAs one after another until you press Stop in the "
                                     "image window")
        self.b_continuous.clicked.connect(lambda: self._go(True))
        box.addButton(self.b_make, QDialogButtonBox.ButtonRole.AcceptRole)
        box.addButton(self.b_continuous, QDialogButtonBox.ButtonRole.AcceptRole)
        box.addButton(QDialogButtonBox.StandardButton.Cancel)
        box.rejected.connect(self.reject)
        v.addWidget(box)
        guard_wheel(self)
        fit_size(self, 760, 640)
        self._check()

    # ---- the table
    def row_of(self, path: str) -> int:
        return self.paths.index(path)

    def _checked(self, row: int, col: int) -> bool:
        item = self.table.item(row, col)
        return item is not None and item.checkState() == Qt.CheckState.Checked

    def _value(self, row: int, col: int) -> float:
        item = self.table.item(row, col)
        try:
            return round(float(item.data(Qt.ItemDataRole.EditRole)), R.DECIMALS) if item is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def set_checked(self, path: str, col: int, on: bool) -> None:
        item = self.table.item(self.row_of(path), col)
        if item is not None:
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def set_range(self, path: str, low: float, high: float) -> None:
        row = self.row_of(path)
        for col, value in ((LOW, low), (HIGH, high)):
            item = self.table.item(row, col)
            if item is not None:
                item.setData(Qt.ItemDataRole.EditRole, round(float(value), R.DECIMALS))

    def set_range_shown(self, low: float, high: float) -> None:
        """Every LoRA the table shows (all, or those the search finds) gets this range."""
        for row, path in enumerate(self.paths):
            if not self.table.isRowHidden(row):
                self.set_range(path, low, high)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        if item.column() == PIN and item.checkState() == Qt.CheckState.Checked:
            use = self.table.item(item.row(), USE)
            if use is not None and use.checkState() != Qt.CheckState.Checked:
                use.setCheckState(Qt.CheckState.Checked)       # (a pinned LoRA is used)
        self._check()

    def _filter(self) -> None:
        words = self.search.text().lower().split()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, NAME)
            text = f"{name.text()} {name.toolTip()}".lower() if name is not None else ""
            self.table.setRowHidden(row, not all(w in text for w in words))

    # ---- what it makes
    def pool(self) -> list[R.Choice]:
        return [R.Choice(path, self._value(row, LOW), self._value(row, HIGH), self._checked(row, PIN),
                         self._checked(row, USE)) for row, path in enumerate(self.paths)]

    def problems(self) -> list[str]:
        return R.problems(self.pool(), self.most.value())

    def _check(self) -> None:
        pool = self.pool()
        why = R.problems(pool, self.most.value())
        self.problem.setText(why[0] if why else "")
        self.problem.setVisible(bool(why))
        k = sum(1 for c in pool if c.usable)
        pinned = sum(1 for c in pool if c.pinned)
        self.most_note.setText(f"({k} LoRA{'s' if k != 1 else ''} to choose from"
                               + (f", {pinned} pinned)" if pinned else ")"))
        n = self.n.value()
        self.b_make.setText(f"Generate {n} image{'s' if n != 1 else ''}")
        for b in (self.b_make, self.b_continuous):
            b.setEnabled(not why)

    def _go(self, continuous: bool) -> None:
        if self.problems():
            return
        self.continuous = continuous
        self.accept()

    def saved(self) -> dict:
        """What is remembered for next time (lora_random.to_saved)."""
        return R.to_saved(self.pool(), most=self.most.value(), n=self.n.value(), same_seed=self.same_seed.isChecked())
