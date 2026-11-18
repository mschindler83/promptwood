"""Which scenes Play film / Export film… use: from one scene to another (both included)."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout

from .widgets import label


class SceneRangeDialog(QDialog):
    """From scene / To scene, numbered as in the storyboard; the last one never comes before the first."""

    def __init__(self, parent, titles: list[str], first: int, last: int, action: str):
        super().__init__(parent)
        self.setWindowTitle(f"{action} the film")
        v = QVBoxLayout(self)
        v.addWidget(label("Which scenes? The whole film, or from one scene to another (both included)."))
        form = QFormLayout()
        self.first, self.last = QComboBox(), QComboBox()
        for box in (self.first, self.last):
            box.addItems([f"{k} · {t}" for k, t in enumerate(titles, start=1)])
        self.first.setCurrentIndex(max(0, min(first, len(titles)) - 1))
        self.last.setCurrentIndex(max(0, min(last, len(titles)) - 1))
        self.first.currentIndexChanged.connect(self._first_changed)
        self.last.currentIndexChanged.connect(self._last_changed)
        form.addRow("From scene", self.first)
        form.addRow("To scene", self.last)
        v.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText(action)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _first_changed(self, k: int):
        if self.last.currentIndex() < k:
            self.last.setCurrentIndex(k)

    def _last_changed(self, k: int):
        if self.first.currentIndex() > k:
            self.first.setCurrentIndex(k)

    def value(self) -> tuple[int, int]:
        return self.first.currentIndex() + 1, self.last.currentIndex() + 1


def pick_range(parent, titles: list[str], first: int, last: int, action: str) -> tuple[int, int] | None:
    """(first, last) scene numbers (from 1), or None when cancelled."""
    dlg = SceneRangeDialog(parent, titles, first, last, action)
    try:
        return dlg.value() if dlg.exec() == QDialog.DialogCode.Accepted else None
    finally:
        dlg.deleteLater()
