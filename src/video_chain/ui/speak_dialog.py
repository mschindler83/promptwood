"""Speak a clip's lines with text to speech: several takes to choose from, and how they are spoken (steady or lively,
slower or faster). The takes are listened to; "Use this take" makes one the clip's sound."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout)

from .. import tts
from .widgets import guard_wheel, label


class _Signals(QObject):
    take = Signal(object, str)                 # a rendered take (tts.render_lines) or None, error
    done = Signal()


class SpeakLinesDialog(QDialog):
    """lines: [{"name", "line"}]; speak: the settings to start with (tts.speak_options); play(path, title): how a
    take is listened to; render(take number, speak) -> a take (tts.render_lines; called in a thread), file(take) ->
    its path. chosen: the take to use (after "Use this take"); unused(): the takes to delete."""

    def __init__(self, parent, clip: str, lines: list[dict], provider: str, speak: dict, play, render, file):
        super().__init__(parent)
        self.setWindowTitle(f"Speak the lines of clip {clip}")
        self.clip, self.provider = clip, provider
        self._play, self._render, self._file = play, render, file
        self.takes: list[dict] = []
        self.chosen: dict | None = None
        self.busy = False
        self._stop = False
        self._close_when_done = False
        self._next = 0
        self._sig = _Signals(self)
        self._sig.take.connect(self._take_done)
        self._sig.done.connect(self._all_done)
        v = QVBoxLayout(self)
        self.lines = QPlainTextEdit("\n".join(f"{d['name']}: “{d['line']}”" for d in lines))
        self.lines.setReadOnly(True)
        self.lines.setFixedHeight(min(120, 30 + 22 * len(lines)))
        self.lines.setToolTip("The lines as the clip's prompt has them (change them there)")
        v.addWidget(self.lines)
        form = QFormLayout()
        self.n = QSpinBox()
        self.n.setRange(1, 8)
        self.n.setValue(speak["takes"])
        self.n.setToolTip("How many takes are made in one go: the same lines, spoken a little differently each time")
        form.addRow("Takes", self.n)
        self.delivery = QComboBox()
        for k, text in tts.DELIVERIES.items():
            self.delivery.addItem(text, k)
        self.delivery.setCurrentIndex(max(0, self.delivery.findData(speak["delivery"])))
        self.delivery.setToolTip("How even or how varied the lines are spoken. Lively takes differ more from each "
                                 "other and slip more often (a wrong stress, a swallowed word): make a few more.")
        form.addRow("Delivery", self.delivery)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(*tts.SPEED_RANGE)
        self.speed.setSingleStep(0.05)
        self.speed.setDecimals(2)
        self.speed.setSuffix(" ×")
        self.speed.setValue(speak["speed"])
        self.speed.setToolTip("Slower or faster than the voice speaks by itself (the pitch stays)")
        form.addRow("Speed", self.speed)
        guard_wheel(self.n, self.delivery, self.speed)
        v.addLayout(form)
        hint = label("The tone (calm, angry, whispering …) comes from the voice sample: a voice set from a part "
                     "where the speaker sounds like that speaks its lines like that. Punctuation is heard too "
                     "(! ? …)." if provider == "local" else
                     "The tone comes from the ElevenLabs voice and from the punctuation of the line (! ? …).",
                     level="tertiary")
        hint.setWordWrap(True)
        v.addWidget(hint)
        brow = QHBoxLayout()
        self.b_make = QPushButton("Make the takes")
        self.b_make.setDefault(True)
        self.b_make.clicked.connect(self.make)
        self.b_play = QPushButton("Listen")
        self.b_play.clicked.connect(self.listen)
        brow.addWidget(self.b_make)
        brow.addWidget(self.b_play)
        brow.addStretch(1)
        v.addLayout(brow)
        self.list = QListWidget()
        self.list.setMinimumHeight(140)
        self.list.currentRowChanged.connect(lambda _r: self._enable())
        self.list.itemDoubleClicked.connect(lambda _i: self.listen())
        v.addWidget(self.list, 1)
        self.info = label(f"Spoken by {tts.PROVIDERS[provider]}. Make a few takes, listen, and use the best one.",
                          level="tertiary")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.b_use = self.buttons.addButton("Use this take", QDialogButtonBox.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        self._enable()
        self.resize(560, 520)

    def speak_values(self) -> dict:
        """The settings as they are set now (to keep in the project)."""
        return tts.speak_options(given={"takes": self.n.value(), "delivery": self.delivery.currentData(),
                                        "speed": self.speed.value()})

    def selected(self) -> dict | None:
        row = self.list.currentRow()
        return self.takes[row] if 0 <= row < len(self.takes) else None

    def _enable(self):
        self.b_make.setEnabled(not self.busy)
        self.b_make.setText("Make more takes" if self.takes else "Make the takes")
        for w in (self.n, self.delivery, self.speed):
            w.setEnabled(not self.busy)
        self.b_play.setEnabled(self.selected() is not None)
        self.b_use.setEnabled(not self.busy and self.selected() is not None)

    def make(self):
        if self.busy:
            return
        how, n, first = self.speak_values(), self.n.value(), self._next
        self._next += n
        self.busy, self._stop = True, False
        self._enable()
        self.info.setText(f"Speaking take 1 of {n}…" + (" (the model loads first)" if self.provider == "local"
                                                         and not self.takes else ""))
        self._left = n

        def work():
            try:
                for k in range(n):
                    if self._stop:
                        break
                    try:
                        self._sig.take.emit(self._render(first + k, how), "")
                    except Exception as e:  # noqa: BLE001
                        self._sig.take.emit(None, str(e))
                        break
            finally:
                self._sig.done.emit()
        threading.Thread(target=work, daemon=True).start()

    def _take_done(self, r, err: str):
        if r is None:
            self.info.setText(f"Could not speak the lines: {err}")
            self._failed = True
            return
        self.takes.append(r)
        how = r.get("speak") or {}
        text = f"Take {len(self.takes)}   ·   {r['secs']:.1f} s   ·   {how.get('delivery', 'natural')}"
        if abs(float(how.get("speed", 1.0)) - 1.0) >= 0.005:
            text += f", {how['speed']:g} ×"
        self.list.addItem(QListWidgetItem(text))
        self._left -= 1
        if self._left > 0 and not self._stop:
            self.info.setText(f"Speaking take {self.n.value() - self._left + 1} of {self.n.value()}…")
        if self.list.currentRow() < 0:              # the first take: selected and played while the others are made
            self.list.setCurrentRow(0)
            self.listen()
        self._enable()

    def _all_done(self):
        self.busy = False
        if self._close_when_done:
            super().reject()
            return
        if not getattr(self, "_failed", False):
            n = len(self.takes)
            self.info.setText(f"{n} take{'s' if n != 1 else ''}. Listen (double-click), then use the best one, or "
                              "change the settings and make more.")
        self._failed = False
        self._enable()

    def listen(self):
        r = self.selected()
        if r is not None:
            self._play(self._file(r), f"Lines of clip {self.clip}, take {self.list.currentRow() + 1}")

    def unused(self) -> list[dict]:
        return [r for r in self.takes if r is not self.chosen]

    def accept(self):
        if not self.busy and self.selected() is not None:
            self.chosen = self.selected()
            super().accept()

    def reject(self):
        if self.busy:                               # (its thread still writes a take: closed when that is done)
            self._stop = self._close_when_done = True
            self.info.setText("Stopping after this take…")
            return
        super().reject()


def run_dialog(dlg: SpeakLinesDialog) -> bool:
    return dlg.exec() == QDialog.Accepted
