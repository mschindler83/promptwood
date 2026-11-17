"""Speak a clip's lines with text to speech: several takes to choose from, and how they are spoken (steady or lively,
slower or faster). The words of the lines can be changed to try other sentences; with an engine that hears [tags]
(Fish S2 Pro, ElevenLabs v3) a tone can be put into a line ([angry], [laughs] ...). The takes are listened to; "Use
this take" makes one the clip's sound (and its words the clip's lines)."""
from __future__ import annotations


from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout)

from .workers import post, run_in_background
from .. import tts
from .dialogs import button_row
from .widgets import fit_size, guard_wheel, label


class SpeakLinesDialog(QDialog):
    """lines: [{"name", "line"}]; speak: the settings to start with (tts.speak_options); play(path, title): how a
    take is listened to; render(take number, speak, line texts) -> a take (tts.render_lines; called in a thread),
    file(take) -> its path. chosen: the take to use (after "Use this take"; its "prompt" has the words it speaks);
    unused(): the takes to delete. tags: the engine hears [tags] in a line (a tone can be put in)."""

    def __init__(self, parent, clip: str, lines: list[dict], provider: str, speak: dict, play, render, file,
                 tags: bool = False):
        super().__init__(parent)
        self.setWindowTitle(f"Speak the lines of clip {clip}")
        self.clip, self.provider = clip, provider
        self._play, self._render, self._file = play, render, file
        self.takes: list[dict] = []
        self.chosen: dict | None = None
        self.busy = False
        self._stop = False
        self._close_when_done = False
        self._failed = False                       # a take could not be made (this round)
        self._left = 0                             # takes still to come in this round
        self._next = 0
        v = QVBoxLayout(self)
        self.original = [d["line"] for d in lines]
        self.line_edits: list[QLineEdit] = []
        self._target: QLineEdit | None = None       # the line a tone goes into (the one last clicked or typed in)
        lform = QFormLayout()
        for d in lines:
            e = QLineEdit(d["line"])
            e.setToolTip(f"What {d['name']} says. Change it to try another sentence: the take you use writes its "
                         "words into the clip's prompt." + (" A tone in square brackets says how it is spoken "
                                                            "([angry] at the start, [laughs] in the middle)."
                                                            if tags else ""))
            e.textChanged.connect(lambda _t: self._enable())
            e.cursorPositionChanged.connect(lambda _a, _b, e=e: self._aim(e))
            self.line_edits.append(e)
            lform.addRow(d["name"], e)
        v.addLayout(lform)
        self.tone = self.b_tone = None
        if tags:
            trow = QHBoxLayout()
            self.tone = QComboBox()
            self.tone.setEditable(True)
            self.tone.addItems(tts.TONE_TAGS)
            self.tone.setToolTip("How the line is spoken. Pick one or type your own words: any short description "
                                 "works ([tired, half asleep], [proud]); several go together ([whispering, scared]).")
            self.b_tone = QPushButton("Put into the line")
            self.b_tone.setToolTip("Writes the tone as [tag] into the line you clicked last, where its cursor is "
                                   "(the first line, at its start, before you click one)")
            self.b_tone.clicked.connect(self.put_tone)
            trow.addWidget(self.tone, 1)
            trow.addWidget(self.b_tone)
            v.addLayout(trow)
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
        hint = label("Say how a line is spoken in square brackets: [angry] at its start, [laughs] or "
                     "[whispering] in the middle. Any words work; the clip's picture never sees them. A lively "
                     "voice sample gives the tones more room than a very calm one." if tags else
                     "The tone (calm, angry, whispering …) comes from the voice sample: a voice set from a range "
                     "where the speaker sounds like that speaks its lines like that. Punctuation is heard too "
                     "(! ? …). [Tags] in a line are left out (Fish S2 Pro hears them)." if provider == "local" else
                     "The tone comes from the ElevenLabs voice and from the punctuation of the line (! ? …); its "
                     "v3 models also hear [tags] in a line.",
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
        self.buttons, self.b_use = button_row(self, "Use this take")
        v.addWidget(self.buttons)
        self._enable()
        fit_size(self, 560, 520)

    def speak_values(self) -> dict:
        """The settings as they are set now (to keep in the project)."""
        return tts.speak_options(given={"takes": self.n.value(), "delivery": self.delivery.currentData(),
                                        "speed": self.speed.value()})

    def line_texts(self) -> list[str]:
        """The lines as they are typed now."""
        return [tts.clean_line(e.text()) for e in self.line_edits]

    def _aim(self, e: QLineEdit):
        if e.hasFocus():
            self._target = e

    def put_tone(self):
        """The chosen tone as [tag] into the line clicked last, at its cursor (else the first line, at its start)."""
        word = (self.tone.currentText() if self.tone is not None else "").strip().strip("[]").strip()
        if not word or not self.line_edits or self.busy:
            return
        e = self._target or self.line_edits[0]
        text, at = e.text(), (e.cursorPosition() if e is self._target else 0)
        before, after = text[:at].rstrip(), text[at:].lstrip()
        tag = f"[{word}]"
        new = " ".join(x for x in (before, tag, after) if x)
        e.setText(new)
        e.setCursorPosition(len(" ".join(x for x in (before, tag) if x)))
        self._target = e

    def selected(self) -> dict | None:
        row = self.list.currentRow()
        return self.takes[row] if 0 <= row < len(self.takes) else None

    def _enable(self):
        if self.busy:                               # the same button stops the round (the takes made stay)
            self.b_make.setEnabled(not self._stop)
            self.b_make.setText("Stop")
        else:
            self.b_make.setEnabled(all(self.line_texts()))
            self.b_make.setText("Make more takes" if self.takes else "Make the takes")
        for w in (self.n, self.delivery, self.speed, *self.line_edits, *[x for x in (self.tone, self.b_tone) if x]):
            w.setEnabled(not self.busy)
        self.b_play.setEnabled(self.selected() is not None)
        self.b_use.setEnabled(not self.busy and self.selected() is not None)

    def make(self):
        if self.busy:                               # the button says "Stop" now
            self._stop = True
            self.info.setText("Stopping after this take…")
            self._enable()
            return
        if not all(self.line_texts()):
            return
        how, n, first, said = self.speak_values(), self.n.value(), self._next, self.line_texts()
        self._next += n
        self.busy, self._stop = True, False
        self._enable()
        self.info.setText(f"Speaking take 1 of {n}…" + (" (the model loads first)" if self.provider in ("local", "fish")
                                                         and not self.takes else ""))
        self._left = n
        self._failed = False

        def work():
            for k in range(n):                      # (every take is handed over as it is made: post keeps the order)
                if self._stop:
                    break
                try:
                    take = self._render(first + k, how, said)
                except Exception as e:  # noqa: BLE001
                    post(self, self._take_done, None, str(e))
                    break
                post(self, self._take_done, take, "")
        run_in_background(work, on_done=lambda _r: self._all_done(), on_error=lambda _e: self._all_done(),
                          owner=self)

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
        said = [d["line"] for d in r.get("lines") or []]
        if said != self.original:                   # other words than the prompt has: shown with the take
            words = " / ".join(said)
            text += f"   ·   “{words[:48]}{'…' if len(words) > 48 else ''}”"
        item = QListWidgetItem(text)
        item.setToolTip("\n".join(f"{d['name']}: “{d['line']}”" for d in r.get("lines") or []))
        self.list.addItem(item)
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
        if not self._failed:
            n = len(self.takes)
            self.info.setText(f"{n} take{'s' if n != 1 else ''}. Listen (double-click), then use the best one, or "
                              "change the words or settings and make more.")
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
    return dlg.exec() == QDialog.DialogCode.Accepted
