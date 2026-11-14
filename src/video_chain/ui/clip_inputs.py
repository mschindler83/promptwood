"""What a clip is made with, on its card: a row of chips in the order the model gets the inputs (start frame, end
frame, the subjects with their pictures, the clip before it continues, the sounds and voices, the model and prompt
format) and the "Model prompt"
window with the exact text the model reads. Both show tree.clip_inputs; nothing here works anything out itself."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QLayout, QPlainTextEdit, QPushButton,
                               QToolButton, QVBoxLayout, QWidget)

from .widgets import label

KIND_NAMES = {"person": "Person", "place": "Place", "object": "Object", "style": "Style"}
FORMAT_NAMES = {"h3": "official", "classic": "classic"}


class FlowLayout(QLayout):
    """Lays out widgets left to right and wraps them into new lines (for the chips; as in another app)."""

    def __init__(self, parent=None, spacing: int = 4):
        super().__init__(parent)
        self._items = []
        self._sp = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):                       # noqa: N802  (Qt naming)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):                           # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):                           # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):                 # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):                   # noqa: N802
        return True

    def heightForWidth(self, w):                   # noqa: N802
        return self._layout(QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):                   # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):                            # noqa: N802
        return self.minimumSize()

    def minimumSize(self):                         # noqa: N802
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def clear(self):
        while self._items:
            w = self._items.pop().widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    def _layout(self, rect, test):
        x, y, line = rect.x(), rect.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            if x + hint.width() > rect.right() and line > 0:
                x, y, line = rect.x(), y + line + self._sp, 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._sp
            line = max(line, hint.height())
        return y + line - rect.y()


def _tags(word: str, numbers: list[int]) -> str:
    tags = [f"<{word} {n}>" for n in numbers]
    return tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]


def _frame_chip(which: str, d: dict) -> tuple[str, str, str] | None:
    """The start / end frame's chip: (text, tooltip, variant) or None (the clip has none)."""
    if d["given"] == "guide":
        return (f"{which.capitalize()}: carry-over",
                f"The clip's {'first' if which == 'start' else 'last'} frames are pinned from the "
                f"{'clip before' if which == 'start' else 'video after it'} (Carry-over), instead of one "
                f"{which} frame.", "neutral")
    if d["given"] != "frame":
        return None
    name = Path(d["file"]).name if d.get("file") else ""
    pinned = ("The clip starts exactly from this frame" if which == "start" else
              "The clip ends exactly on this frame") + (f" ({name})" if name else "") + "."
    if d.get("picture"):
        return (f"{which.capitalize()} frame · Picture {d['picture']}",
                f"{pinned[:-1]}; it is also sent as reference <Picture {d['picture']}>, so the prompt can name it.",
                "")
    return f"{which.capitalize()} frame", pinned, ""


def chip_specs(info: dict, lead_off: str | None = None) -> list[dict]:
    """The chips of a clip from tree.clip_inputs, in the order the model gets the inputs: [{"text", "tip",
    "variant": "" | "neutral", "lead": None | True | False}] (lead: the checkable chip of the implicit lead).
    lead_off: the lead's name when the clip's lead switch is off and it would get the lead otherwise."""
    out = []
    for which in ("start", "end"):
        chip = _frame_chip(which, info[which])
        if chip:
            out.append({"text": chip[0], "tip": chip[1], "variant": chip[2], "lead": None})
    for s in info["subjects"]:
        n, kind = len(s["pictures"]), s["kind"]
        pics = _tags("Picture", s["pictures"]) if n else ""
        if s.get("lead"):
            out.append({"text": f"@{s['name']} · lead", "variant": "", "lead": True,
                        "tip": f"The prompt mentions nobody, so the lead @{s['name']} is used"
                               + (f" (face picture{'s' if n != 1 else ''} {pics})" if n else "")
                               + ". Uncheck to make this clip without the lead."})
            continue
        if kind == "person":
            text = f"@{s['name']} · face" + (f" ×{n}" if n > 1 else "") if n else f"@{s['name']} · no face"
            tip = (f"@{s['name']}'s face picture{'s go' if n != 1 else ' goes'} to the model as {pics}." if n else
                   f"@{s['name']} is mentioned but has no face picture: only the description is used.")
        else:
            text = f"@{s['name']} · {kind}" + (f" ×{n}" if n > 1 else "") if n else f"@{s['name']} · no picture"
            tip = (f"The picture{'s' if n != 1 else ''} of @{s['name']} (a {kind}) go{'' if n != 1 else 'es'} to the "
                   f"model as {pics}." if n else
                   f"@{s['name']} (a {kind}) is mentioned but has no picture: only the description is used.")
        out.append({"text": text, "tip": tip, "variant": "", "lead": None})
    if lead_off:
        out.append({"text": f"@{lead_off} · lead off", "variant": "neutral", "lead": False,
                    "tip": f"The lead is switched off for this clip: it is made without @{lead_off}. Check to use "
                           "the lead again."})
    for vid in info.get("videos") or []:            # (the node's order: pictures, videos, then the sounds)
        out.append({"text": f"Continues clip before · Video {vid['n']}", "variant": "", "lead": None,
                    "tip": f"The {vid['what']} (its last {vid['frames'] / 24:.1f} s, {vid['frames']} frames, "
                           f"pictures only) goes to the model as <Video {vid['n']}>, and the prompt tells it that "
                           "this clip continues it: motion, camera and scene carry on (Continue)."})
    for snd in info["sounds"]:
        name = Path(snd["file"]).name if snd.get("file") else ""
        if snd["what"] == "soundtrack":
            text, tip = "♪ soundtrack", (f"The clip's whole sound is given (its Sound setting) as <Audio {snd['n']}>; "
                                         "the picture is made to it.")
        elif snd["what"].startswith("voice of "):
            who = snd["what"][len("voice of "):]
            text, tip = f"♪ {snd['what']}", (f"{who}'s voice sample goes to the model as <Audio {snd['n']}>, so "
                                             f"{who}'s lines are spoken with it.")
        else:
            text, tip = f"♪ sound {snd['n']}", (f"The sound reference the prompt names as <Audio {snd['n']}>"
                                                 + (f" ({name})." if name else " (it is missing)."))
        out.append({"text": text, "tip": tip, "variant": "", "lead": None})
    ss = (info.get("settings") or {}).get("source_sound")
    if isinstance(ss, dict) and ss.get("overlay"):     # not an input of the model, but the clip's sound all the same
        out.append({"text": "♪ mixed in" if ss.get("mix") else "♪ laid over", "variant": "neutral", "lead": None,
                    "tip": "The clip's Sound setting is mixed with the sound the model makes, after the clip is made; "
                           "the model does not hear it." if ss.get("mix") else
                           "The clip's Sound setting is put into the clip after it is made, instead of the model's "
                           "sound; the model does not hear it."})
    model = info["model"]
    out.append({"text": model_chip_text(info), "variant": "neutral", "lead": None, "model": True,
                "tip": ("Made with MiniMax H3's reference model (ref2va): it gets the pictures, videos and sounds "
                        "above. "
                        if model == "ref2va" else
                        "Made with MiniMax H3's first / last frame model (fl2va): no reference pictures or sounds. ")
                + (f"The prompt is written in MiniMax H3's {FORMAT_NAMES['h3']} format." if info["format"] == "h3"
                   else "The prompt is written in the app's classic format.") + " (Project → Prompt format)"})
    return out


def model_chip_text(info: dict) -> str:
    """How the clip is made, in plain words (the model and the prompt format are in the chip's tooltip)."""
    if info["model"] == "ref2va":
        return ("With reference pictures" if info["pictures"] else
                "With a reference video" if info.get("videos") else "With reference sounds")
    given = [w for w in ("start", "end") if info[w]["given"] in ("frame", "guide")]
    return {("start",): "From the start frame", ("end",): "To the end frame",
            ("start", "end"): "From the start to the end frame"}.get(tuple(given), "From the prompt only")


class ChipRow(QWidget):
    """The chips of one clip and its "Model prompt" button. leadToggled(on): the lead chip was clicked."""
    leadToggled = Signal(bool)
    promptWanted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, spacing=4)
        self._specs: list[dict] | None = None
        self.chips: list[QToolButton] = []
        self.model_chip: QToolButton | None = None      # how the clip is made (the Simple view hides it)
        self.b_prompt = None
        self.setVisible(False)

    def set_specs(self, specs: list[dict] | None):
        """Show these chips (None: none, hidden); rebuilt only when they differ from the ones shown."""
        if specs == self._specs:
            return
        self._specs = specs
        self.flow.clear()
        self.chips = []
        self.model_chip = self.b_prompt = None
        if not specs:
            self.setVisible(False)
            return
        for s in specs:
            b = QToolButton(self)
            b.setProperty("role", "chip")
            if s["variant"]:
                b.setProperty("variant", s["variant"])
            b.setText(s["text"])
            b.setToolTip(s["tip"])
            b.setFocusPolicy(Qt.NoFocus)
            if s["lead"] is not None:
                b.setCheckable(True)
                b.setChecked(bool(s["lead"]))
                b.clicked.connect(lambda on: self.leadToggled.emit(bool(on)))
            self.flow.addWidget(b)
            self.chips.append(b)
            if s.get("model"):
                self.model_chip = b
        self.b_prompt = QToolButton(self)
        self.b_prompt.setText("Model prompt")
        self.b_prompt.setToolTip("The exact text the model gets for this clip, and its pictures and sounds in order")
        self.b_prompt.clicked.connect(self.promptWanted.emit)
        self.flow.addWidget(self.b_prompt)
        self.setVisible(True)
        self.updateGeometry()

    def texts(self) -> list[str]:
        return [b.text() for b in self.chips]


def inputs_list(info: dict) -> str:
    """The numbered pictures, videos and sounds in the order the model gets them, as text (file names only)."""
    lines = [f"<Picture {p['n']}> — {p['what']} ({Path(str(p['file'])).name if p.get('file') else 'missing'})"
             for p in info["pictures"]]
    for which in ("start", "end"):
        d = info[which]
        if d["given"] == "frame" and not d.get("picture"):
            name = Path(d["file"]).name if d.get("file") else "not made yet"
            lines.append(f"{which.capitalize()} frame ({name}): pinned as the {'first' if which == 'start' else 'last'}"
                         " frame, not numbered")
        elif d["given"] == "guide":
            lines.append(f"{which.capitalize()}: carry-over frames, not numbered")
    lines += [f"<Video {v['n']}> — {v['what']} ({v['frames']} frames)" for v in info.get("videos") or []]
    lines += [f"<Audio {s['n']}> — {s['what']} ("
              + (Path(s["file"]).name if s.get("file") else "cut when the clip is made") + ")" for s in info["sounds"]]
    return "\n".join(lines) if lines else "No pictures or sounds besides the prompt."


class ModelPromptDialog(QDialog):
    """Read-only: the exact prompt of a clip (selectable, Copy) and the numbered pictures / sounds."""

    def __init__(self, parent, nid: str, info: dict):
        super().__init__(parent)
        self.setWindowTitle(f"Model prompt · clip #{nid[1:]}")
        v = QVBoxLayout(self)
        v.addWidget(label(f"{info['model']} · {'official H3 format' if info['format'] == 'h3' else 'classic format'}"
                          f" · {info['seconds']:.2f} s", level="secondary"))
        self.text = QPlainTextEdit(info["prompt"])
        self.text.setProperty("type", "mono")
        self.text.setReadOnly(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        v.addWidget(self.text, 1)
        v.addWidget(label("Pictures and sounds, in the order the model gets them", level="tertiary"))
        self.inputs = label(inputs_list(info), level="secondary")
        self.inputs.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.inputs.setTextFormat(Qt.PlainText)
        self.inputs.setWordWrap(True)
        v.addWidget(self.inputs)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.b_copy = QPushButton("Copy")
        self.b_copy.setToolTip("Copy the prompt to the clipboard")
        self.b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        buttons.addButton(self.b_copy, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)
        self.resize(720, 560)
