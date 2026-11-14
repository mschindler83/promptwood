"""The guided start in the Clips tab: a short guide while a project has no made clip (three steps, each with one
button that runs the app's own action and a tick from the real state), the engine banner while the video engine is
not reachable, and the one-time line after the first made clip. Not a wizard: everything else stays usable.

The widgets only show; MainWindow decides what is done (set_state) and connects the buttons to its handlers."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .theme import icon, repolish
from . import shortcuts as K
from .widgets import label


def _button(text: str, tip: str = "", primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setAutoDefault(False)
    if tip:
        b.setToolTip(tip)
    if primary:
        b.setProperty("variant", "primary")
    return b


class _Step(QWidget):
    """One step: its number until it is done, then a tick; the text and its button(s)."""

    def __init__(self, number: int, text: str, tip: str, buttons: list[QPushButton]):
        super().__init__()
        self.number = number
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.mark = QLabel()
        self.mark.setFixedWidth(18)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text = label(text)
        self.text.setWordWrap(True)
        self.text.setToolTip(tip)
        h.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        h.addWidget(self.text, 1)
        for b in buttons:
            h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        self.buttons = buttons
        self.done = None
        self.set_done(False)

    def set_done(self, on: bool):
        on = bool(on)
        if on == self.done:
            return
        self.done = on
        if on:
            self.mark.setText("")
            self.mark.setPixmap(icon("check").pixmap(14, 14))
            self.mark.setToolTip("Done")
        else:
            self.mark.setPixmap(QPixmap())
            self.mark.setText(str(self.number) if self.number else "")      # (not by its dot alone: a number says "step 2")
            self.mark.setToolTip("Still to do")
        self.text.setProperty("level", "secondary" if on else "")
        repolish(self.text)


class GuidePanel(QFrame):
    """Getting started: step 0 without a project, else the three steps and the Generate hint. hidden(): Hide guide
    was pressed (the window stores it; View → Getting started brings the guide back)."""
    hideWanted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "notice")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 8, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label("Getting started", type_="heading"))
        head.addStretch(1)
        self.b_hide = _button("Hide", "Hide this guide (View → Getting started shows it again)")
        self.b_hide.setProperty("variant", "ghost")
        self.b_hide.clicked.connect(self.hideWanted)
        head.addWidget(self.b_hide)
        v.addLayout(head)
        # step 0: no project open
        self.b_new = _button("New project…", K.tip("Make a new project folder", "new_project"), primary=True)
        self.b_open = _button("Open…", K.tip("Open a project folder", "open"))
        self.step0 = _Step(0, "Create or open a project.", "A project is a folder: it keeps the clips, the pictures and "
                           "the sounds of one video", [self.b_new, self.b_open])
        v.addWidget(self.step0)
        # the three steps of a project
        self.b_picture = _button("Choose a picture…", "Choose the picture the first clip starts from")
        self.b_text_only = _button("Text only", "Begin without a picture: the first clip is made from its "
                                                "description alone (choose its Aspect in the Project tab on the right)")
        self.step1 = _Step(1, "Choose a start image, or start from text only.", "The first clip starts from this "
                           "picture; every clip after it continues from the one before",
                           [self.b_picture, self.b_text_only])
        self.b_cast = _button("Add a person or place…", "Open Cast & references: add a person (with face pictures and a voice) or "
                                                      "a place, then name it as @Name in a description")
        self.b_skip_cast = _button("Skip", "Go on without people or places (you can add them any time)")
        self.step2 = _Step(2, "Add people or places (optional).", "With pictures of a person or a place the clips "
                           "keep the same face and look", [self.b_cast, self.b_skip_cast])
        self.b_write = _button("Go to the description", "Go to the first clip's description")
        self.step3 = _Step(3, "Describe what happens in the first clip.", "A few sentences: who does what, what "
                           "is said (in quotes), what the camera does, what is heard", [self.b_write])
        for s in (self.step1, self.step2, self.step3):
            v.addWidget(s)
        grow = QHBoxLayout()
        self.gen_hint = label(f"Now press Generate path ({K.keys('generate')}) to make the clip.", severity="positive")
        self.gen_hint.setWordWrap(True)
        self.b_show_gen = _button("Show me", "Put the keyboard focus on Generate path; Enter or Space presses it")
        grow.addWidget(self.gen_hint, 1)
        grow.addWidget(self.b_show_gen, 0, Qt.AlignmentFlag.AlignTop)
        self.gen_row = QWidget()
        grow.setContentsMargins(0, 0, 0, 0)
        self.gen_row.setLayout(grow)
        v.addWidget(self.gen_row)
        self.set_state(project=False)

    def set_state(self, project: bool, picture: bool = False, cast: bool = False, prompt: bool = False) -> bool:
        """Tick the steps from the real state; True when everything needed is done (Generate is next)."""
        self.step0.setVisible(not project)
        self.step0.set_done(project)
        for s, on in ((self.step1, picture), (self.step2, cast), (self.step3, prompt)):
            s.setVisible(project)
            s.set_done(on)
        ready = bool(project and picture and prompt)       # (people and places are optional)
        self.gen_row.setVisible(ready)
        return ready


class EngineBanner(QFrame):
    """'The video engine (ComfyUI) is not running' with one button: start it (the app has a start command) or open
    the preferences (it has none)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "notice")
        self.setProperty("severity", "warning")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 4, 6, 4)
        h.setSpacing(6)
        self.text = label("The video engine (ComfyUI) is not running.")
        self.text.setWordWrap(True)
        self.text.setToolTip("ComfyUI is the program that makes the clips. Without it nothing can be generated")
        self.button = _button("Start the video engine")
        h.addWidget(self.text, 1)
        h.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.can_start = True

    def set_state(self, starting: bool, can_start: bool):
        self.can_start = bool(can_start)
        self.text.setText("The video engine (ComfyUI) is starting …" if starting else
                          "The video engine (ComfyUI) is not running.")
        self.button.setText("Start the video engine" if can_start else "Open preferences")
        self.button.setToolTip("Start ComfyUI in the background with the start command from the preferences"
                               if can_start else "No start command for ComfyUI is set: set it (or the address of a "
                                                 "running ComfyUI) in the preferences")
        self.button.setEnabled(not starting)


class NextHint(QFrame):
    """Shown once, after the first made clip: how to go on."""
    closeWanted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "notice")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 4, 4, 4)
        h.setSpacing(6)
        self.text = label("Add the next clip: it continues from this one's last frame. To make it better, refine the "
                          "prompt, add LoRAs or a sound in the clip's settings; when the path is done, use Export "
                          "path. Simple / Advanced (top right) shows fewer or more settings.")
        self.text.setWordWrap(True)
        self.b_next = _button("Add the next clip", K.tip("A new clip after the selected one", "continue"), primary=True)
        self.b_close = _button("Hide", "Do not show this again")
        self.b_close.clicked.connect(self.closeWanted)
        h.addWidget(self.text, 1)
        h.addWidget(self.b_next, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(self.b_close, 0, Qt.AlignmentFlag.AlignVCenter)
