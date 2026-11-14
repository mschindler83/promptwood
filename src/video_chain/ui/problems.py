"""Problems where they are: what blocks a clip (or what went wrong when it was made), on its card, each as one plain
sentence with a button where the app can fix it. And the prompt helpers under the card's prompt (@ person or place,
camera move, examples)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from .. import camera as CAM
from .theme import icon, repolish
from .widgets import label

# short example prompts (generic, harmless): a spoken line in quotes, a #camera move, a described sound
EXAMPLES = (
    "A woman in a red coat walks along a rainy street at night; the neon signs shine in the puddles.",
    "The man looks up from his book, smiles and says \"I knew you would come back.\"",
    "A small boat drifts across a calm lake at sunrise, mist rising from the water, as #push-in-slow.",
    "A dog runs through the autumn leaves in a park. The leaves rustle under its paws and birds sing in the trees.",
    "The girl blows out the candles on her birthday cake, and everyone at the table claps and cheers.",
)


def insert_at_cursor(edit: QPlainTextEdit, text: str) -> None:
    """Put text in at the cursor, with a space before / after where words would run together (one undo step)."""
    cur = edit.textCursor()
    doc = edit.document()
    pos = cur.selectionStart()
    before = doc.characterAt(pos - 1) if pos > 0 else ""
    end = cur.selectionEnd()
    after = doc.characterAt(end) if end < doc.characterCount() - 1 else ""
    pre = " " if before and not before.isspace() else ""
    post = "" if after and (after.isspace() or after in ".,;:!?'\")") else " "
    cur.beginEditBlock()
    cur.insertText(pre + text + post)
    cur.endEditBlock()
    edit.setTextCursor(cur)
    edit.setFocus()


def add_example(edit: QPlainTextEdit, text: str) -> None:
    """An example prompt: it replaces an empty prompt, else it is added as a new sentence at the end (one undo step,
    Ctrl+Z takes it back)."""
    old = edit.toPlainText()
    cur = QTextCursor(edit.document())
    cur.beginEditBlock()
    if not old.strip():
        cur.select(QTextCursor.Document)
        cur.insertText(text)
    else:
        cur.movePosition(QTextCursor.End)
        tail = old.rstrip()
        cur.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(old) - len(tail))   # (trailing spaces go)
        cur.insertText(("" if tail[-1] in ".!?\"'" else ".") + " " + text)
    cur.endEditBlock()
    edit.setTextCursor(cur)
    edit.setFocus()


def camera_menu_items() -> list[tuple[str, list[tuple[str, str]]]]:
    """The camera moves for the Camera button: (move in plain words, [(variant in plain words, token)])."""
    out = []
    for slug, (text, _w, mods) in CAM.MOVES.items():
        variants = [(CAM.label(t), t) for t in CAM.tokens() if CAM.parse(t) and CAM.parse(t)[0] == slug]
        out.append((text, variants if mods else [(text, slug)]))
    return out


class ProblemBox(QFrame):
    """One compact, marked area on a card: each problem as a plain sentence (the technical detail in its tooltip)
    and a button where the app can do it. set_items([{"text", "tip", "action", "button", "error", "more": [(action,
    button)]}]); wanted(action) when a button is pressed."""
    wanted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "notice")
        self.setProperty("severity", "warning")
        self.setAttribute(Qt.WA_StyledBackground)
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(8, 6, 6, 6)
        self.v.setSpacing(4)
        self.items: list[dict] = []
        self.rows: list[tuple[QLabel, list[QPushButton]]] = []
        self.hide()

    def set_items(self, items: list[dict]) -> None:
        items = list(items or [])
        if items == self.items:
            return
        self.items = items
        while self.v.count():
            it = self.v.takeAt(0)
            lay = it.layout()
            if lay is not None:
                while lay.count():
                    w = lay.takeAt(0).widget()
                    if w is not None:
                        w.setParent(None)
                        w.deleteLater()
        self.rows = []
        error = any(p.get("error") for p in items)
        self.setProperty("severity", "error" if error else "warning")
        repolish(self)
        for p in items:
            h = QHBoxLayout()
            h.setSpacing(6)
            mark = QLabel()
            mark.setPixmap(icon("err" if p.get("error") else "warn").pixmap(14, 14))
            text = label(p["text"], severity="error" if p.get("error") else "warning")
            text.setWordWrap(True)
            text.setToolTip(p.get("tip", ""))
            h.addWidget(mark, 0, Qt.AlignTop)
            h.addWidget(text, 1)
            self.v.addLayout(h)
            bs = []
            brow = QHBoxLayout()                       # (the buttons under the sentence: the card is narrow)
            brow.setSpacing(6)
            brow.addStretch(1)
            for act, name in list(p.get("more", [])) + [(p.get("action"), p.get("button"))]:
                if act and name:
                    b = QPushButton(name)
                    b.setAutoDefault(False)
                    b.clicked.connect(lambda _=False, a=act: self.wanted.emit(a))
                    brow.addWidget(b)
                    bs.append(b)
            if bs:
                self.v.addLayout(brow)
            self.rows.append((text, bs))
        self.setVisible(bool(items))

    def buttons(self) -> dict[str, QPushButton]:
        """Button text -> button (for tests and tools)."""
        return {b.text(): b for b in self.findChildren(QPushButton)}
