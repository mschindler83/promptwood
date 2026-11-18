"""The image window's Random LoRAs: images from one prompt, each with a random choice of the list's LoRAs (the pinned
ones always, at most a number in all) at random strengths, a fixed number of them or until Stop (continuous).
The choosing itself is lora_random; this is the window that sets it up."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QPushButton,
                               QSpinBox, QVBoxLayout)

from .. import lora_random as R
from .lora_stack import LORA_MAX, LORA_MIN, lora_label
from .widgets import guard_wheel, label, label_for


def _strength_box(value: float, tip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(LORA_MIN, LORA_MAX)
    box.setDecimals(R.DECIMALS)
    box.setSingleStep(0.05)
    box.setValue(value)
    box.setToolTip(tip)
    return box


class RandomLorasDialog(QDialog):
    """Pin and range per LoRA of the list, how many LoRAs an image may have, how many images, the same seed or a new
    one each. Generate N images or Continuous closes it with what to make (choice())."""

    def __init__(self, parent, set_up: dict, seed: int):
        super().__init__(parent)
        self.setWindowTitle("Random LoRAs")
        self.continuous = False
        v = QVBoxLayout(self)
        intro = label("Every image has the same prompt and a random choice of these LoRAs, each at a random strength "
                      "from its range. Pinned LoRAs are in every image. An image gets at least one LoRA and at most "
                      "the number below, pinned ones included, often fewer.", level="secondary")
        intro.setWordWrap(True)
        v.addWidget(intro)
        grid = QGridLayout()
        for col, text in enumerate(("Pin", "LoRA", "Strength from", "to")):
            grid.addWidget(label(text, level="tertiary"), 0, col)
        self.rows: list[tuple[str, QCheckBox, QDoubleSpinBox, QDoubleSpinBox]] = []
        for i, c in enumerate(set_up["choices"], start=1):
            name = lora_label(c.path)
            pin = QCheckBox()
            pin.setChecked(c.pinned)
            pin.setToolTip(f"Pinned: every image has {name}")
            pin.setAccessibleName(f"Pin {name}")
            low = _strength_box(c.low, f"The lowest strength {name} gets")
            high = _strength_box(c.high, f"The highest strength {name} gets (the same as the lowest: always that one)")
            low.setAccessibleName(f"{name} strength from")
            high.setAccessibleName(f"{name} strength to")
            grid.addWidget(pin, i, 0)
            grid.addWidget(label(name), i, 1)
            grid.addWidget(low, i, 2)
            grid.addWidget(high, i, 3)
            pin.toggled.connect(lambda _on: self._check())
            self.rows.append((c.path, pin, low, high))
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)
        opts = QGridLayout()
        self.most = QSpinBox()
        self.most.setRange(1, max(1, len(self.rows)))
        self.most.setValue(min(set_up["most"], self.most.maximum()))
        self.most.setToolTip("How many LoRAs an image may have at most, pinned ones included; each image gets a "
                             "random number from 1 (or the number pinned) up to this")
        self.most.valueChanged.connect(lambda _v: self._check())
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
        opts.addWidget(label_for("Images", self.n, level="tertiary"), 1, 0)
        opts.addWidget(self.n, 1, 1)
        opts.addWidget(self.same_seed, 2, 0, 1, 2)
        opts.setColumnStretch(2, 1)
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
        self._check()

    def pool(self) -> list[R.Choice]:
        return [R.Choice(path, round(low.value(), R.DECIMALS), round(high.value(), R.DECIMALS), pin.isChecked())
                for path, pin, low, high in self.rows]

    def problems(self) -> list[str]:
        return R.problems(self.pool(), self.most.value())

    def _check(self) -> None:
        why = self.problems()
        self.problem.setText(why[0] if why else "")
        self.problem.setVisible(bool(why))
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
