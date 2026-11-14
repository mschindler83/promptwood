"""Prompt refiner: a language model in ComfyUI rewrites a prompt (a video clip's, an image's, an edit instruction).
The window shows the original and the suggestion side by side; the suggestion can be edited, refined again (another
seed, a hint like "more cinematic"), and taken over. Which model refines is chosen here and remembered."""
from __future__ import annotations

import html
import random
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPixmap, QShortcut, QTextCursor
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLineEdit, QPlainTextEdit,
                               QPushButton, QSpinBox, QTextBrowser, QToolButton, QVBoxLayout)

from ..tree import refine_problems
from .theme import repolish
from .widgets import guard_wheel, icon_button, label

KIND_TEXT = {"video": "video clip prompt", "video_h3": "video clip prompt", "image": "image prompt",
             "edit": "edit instruction"}


def refine_button(tip: str = "Refine this prompt with a language model (ComfyUI)") -> "QPushButton":
    b = icon_button("gen", tip)
    return b


def replace_text(edit: QPlainTextEdit, text: str) -> None:
    """Replace a text field's content as one step Ctrl+Z takes back."""
    cur = edit.textCursor()
    cur.beginEditBlock()
    cur.select(QTextCursor.Document)
    cur.insertText(text)
    cur.endEditBlock()
    edit.setTextCursor(cur)


class _Sig(QObject):
    models = Signal(object, str)          # [model dicts], error
    result = Signal(str, str)             # text, error
    ideas = Signal(object, str)           # [idea dicts], error


class RefineDialog(QDialog):
    """connect() -> a refiner (called in a thread): ComfyUI's, or CombinedRefiner with llama.cpp models.
    ComfyUI's models refine once per press (with an optional hint). llama.cpp models hold a conversation: each
    message changes the current version, every version stays and can be continued from, and pictures (the clip's
    start frame, the image to edit, …) can be shown to a model that sees. chosen() -> the text to use, or None.
    pictures: [(title, path, on)] offered at the start. seconds, start_image: the clip's length and the frame it
    starts from (kind "video_h3": the refiner fits the action to the length and anchors on that frame)."""

    def __init__(self, parent, text: str, kind: str, connect, settings=None, start_comfy=None, cast=None,
                 pictures=None, seconds: float | None = None, start_image=None):
        super().__init__(parent)
        self.cast = cast or []                    # the project's characters: the refiner keeps @Name, writes lines
        self.seconds, self.start_image = seconds, (str(start_image) if start_image else None)
        self.setWindowTitle(f"Refine the {KIND_TEXT.get(kind, 'prompt')}")
        self.resize(1100, 680)
        self.kind, self.connect, self.qs, self.start_comfy = kind, connect, settings, start_comfy
        self.refiner = None
        self.refined_any = False                  # a refine ran (its model is loaded in ComfyUI now)
        self._busy = False
        self._chosen: str | None = None
        self.turns: list[tuple[str, str]] = []   # the conversation: (what the user asked, the version it gave)
        self.base = -1                           # the version the next message changes (-1: the original)
        self._pending = ""
        self._base_before: int | None = None     # (Another try: the base to go back to if it fails)
        self._edited: dict[int, bool] = {}       # versions the user edited by hand
        self.pictures = [{"title": t, "path": str(p), "on": bool(on)} for t, p, on in (pictures or [])
                         if p and Path(p).is_file()]
        self._sig = _Sig()
        self._sig.models.connect(self._models)
        self._sig.result.connect(self._result)
        self._sig.ideas.connect(self._ideas_ready)
        self.ideas: list[dict] = []
        self._chosen_ideas: list[str] | None = None
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(label("Model", level="tertiary"))
        self.model = QComboBox()
        self.model.addItem("looking for language models …", None)
        self.model.setEnabled(False)
        self.model.setToolTip("The language model that rewrites the prompt. Uncensored ones follow your wording "
                              "without softening it; llama.cpp models hold a conversation and can see pictures")
        self.model.currentIndexChanged.connect(self._remember)
        self.model.currentIndexChanged.connect(lambda *_: self._mode_changed())
        top.addWidget(self.model, 1)
        self.caps = label("", level="tertiary")
        top.addWidget(self.caps)
        self.b_mode_refine = QPushButton("Refine")
        self.b_mode_ideas = QPushButton("Ideas from bullet points")
        self.b_mode_refine.setToolTip("Improve this prompt, step by step in a conversation")
        self.b_mode_ideas.setToolTip("Write bullet points; the model makes several clearly different image ideas from "
                                     "them (each uses every bullet point)")
        for b in (self.b_mode_refine, self.b_mode_ideas):
            b.setCheckable(True)
            b.setAutoDefault(False)
            top.addWidget(b)
        self.b_mode_refine.setChecked(True)
        self.b_mode_refine.clicked.connect(lambda: self.set_ideas_mode(False))
        self.b_mode_ideas.clicked.connect(lambda: self.set_ideas_mode(True))
        v.addLayout(top)
        row = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        self.original_title = label("Your prompt", level="tertiary")
        left.addWidget(self.original_title)
        self._texts = {"refine": text, "ideas": text}     # each mode keeps its own text: prompt / bullet points
        self._shown_mode = "refine"
        self.original = QPlainTextEdit(text)
        self.original.setReadOnly(True)
        left.addWidget(self.original, 1)
        self.pics_title = label("Pictures the model sees (click to switch on / off)", level="tertiary")
        left.addWidget(self.pics_title)
        self.pics_row = QHBoxLayout()
        self.pics_row.setSpacing(6)
        left.addLayout(self.pics_row)
        right.addWidget(label("Conversation (click a version to go back to it)", level="tertiary"))
        self.talk = QTextBrowser()
        self.talk.setOpenLinks(False)
        self.talk.anchorClicked.connect(self._version_clicked)
        self.talk.setMinimumHeight(140)
        right.addWidget(self.talk, 1)
        self.out_title = label("Refined (you can edit it)", level="tertiary")
        right.addWidget(self.out_title)
        self.out = QPlainTextEdit()
        self.out.setPlaceholderText("Press Refine")
        right.addWidget(self.out, 1)
        self.ideas_title = label("Ideas", level="tertiary")
        right.addWidget(self.ideas_title)
        self.ideas_view = QTextBrowser()
        self.ideas_view.setOpenLinks(False)
        self.ideas_view.anchorClicked.connect(self._idea_link)
        right.addWidget(self.ideas_view, 3)
        row.addLayout(left, 2)
        row.addLayout(right, 3)
        v.addLayout(row, 1)
        hrow = QHBoxLayout()
        self.hint = QLineEdit()
        self.hint.returnPressed.connect(lambda: self.refine())
        hrow.addWidget(self.hint, 1)
        self.count = QSpinBox()
        self.count.setRange(2, 8)
        self.count.setValue(5)
        self.count.setSuffix(" ideas")
        self.count.setToolTip("How many different ideas (one answer: the model keeps them apart)")
        self.count.valueChanged.connect(lambda v: self.ideas_mode() and self.b_refine.setText(f"Make {v} ideas"))
        hrow.addWidget(self.count)
        self.b_all = QPushButton("Generate all")
        self.b_all.setToolTip("Put all ideas into the image window's Variations as {idea}: one Generate makes one "
                              "image per idea")
        self.b_all.clicked.connect(lambda: self._use_all())
        self.b_again = QPushButton("Another try")
        self.b_again.setToolTip("Ask the last message again (another answer to it)")
        self.b_again.clicked.connect(lambda: self.again())
        hrow.addWidget(self.b_again)
        hrow.addWidget(self.b_all)
        self.b_refine = QPushButton("Refine")
        self.b_refine.setProperty("variant", "primary")
        self.b_refine.clicked.connect(lambda: self.refine())     # (clicked would pass checked=False as ask)
        self.b_refine.setEnabled(False)
        hrow.addWidget(self.b_refine)
        v.addLayout(hrow)
        self.state = label("", level="secondary")
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Use this")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.b_start = self.buttons.addButton("Start ComfyUI", QDialogButtonBox.ResetRole)
        self.b_start.hide()
        self.b_start.clicked.connect(self._start_comfy)
        self.buttons.accepted.connect(self._use)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)
        self.out.textChanged.connect(lambda: self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            bool(self.out.toPlainText().strip()) and not self._busy))
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=lambda: self.refine())
        guard_wheel(self)
        self._show_pictures()
        self._mode_changed()
        self._load_models()

    # ---- what the chosen model can do
    def current_model(self) -> dict:
        return next((m for m in (self.refiner.models if self.refiner else [])
                     if m["file"] == self.model.currentData()), {})

    def chatting(self) -> bool:
        return bool(self.current_model().get("chat"))

    def seeing(self) -> bool:
        return bool(self.current_model().get("vision"))

    def ideas_mode(self) -> bool:
        return self.b_mode_ideas.isChecked() and self.ideas_possible()

    def ideas_possible(self) -> bool:
        return self.kind == "image" and self.chatting()

    def set_ideas_mode(self, on: bool):
        if self._busy:                              # (not while an answer is on its way)
            self.b_mode_ideas.setChecked(self._shown_mode == "ideas")
            self.b_mode_refine.setChecked(self._shown_mode != "ideas")
            return
        self.b_mode_ideas.setChecked(bool(on))
        self.b_mode_refine.setChecked(not on)
        self._mode_changed()

    def _swap_text(self, mode: str):
        """The left field shows this mode's text (the prompt for Refine, the bullet points for Ideas)."""
        if mode != self._shown_mode:
            self._texts[self._shown_mode] = self.original.toPlainText()
            self._shown_mode = mode
            self.original.setPlainText(self._texts[mode])

    def _mode_changed(self):
        chat, see = self.chatting(), self.seeing()
        ideas = self.ideas_mode()
        for b in (self.b_mode_refine, self.b_mode_ideas):
            b.setVisible(self.ideas_possible())
        self._swap_text("ideas" if ideas else "refine")
        self.original.setReadOnly(not ideas or self._busy)
        self.original_title.setText("Your bullet points (one per line; you can edit them here)" if ideas
                                    else "Your prompt")
        for w in (self.ideas_title, self.ideas_view, self.count):
            w.setVisible(ideas)
        self.b_all.setVisible(ideas)
        self.b_all.setEnabled(ideas and bool(self.ideas) and not self._busy)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not ideas and not self._busy
                                                            and bool(self.out.toPlainText().strip()))
        self.buttons.button(QDialogButtonBox.Ok).setToolTip("In Ideas mode: Use this on an idea, or Generate all"
                                                            if ideas else "")
        self._render_ideas()
        self.caps.setText(" · ".join(x for x in ("conversation" if chat else "", "sees pictures" if see else "") if x)
                          or ("one refine per press" if self.model.currentData() else ""))
        self.talk.setVisible(chat and not ideas)
        self.out.setVisible(not ideas)
        self.out_title.setVisible(not ideas)
        self.b_again.setVisible(chat and not ideas)
        self.b_refine.setText(f"Make {self.count.value()} ideas" if ideas else
                              "Send" if chat and self.turns else "Refine")
        self.b_refine.setToolTip("Make the ideas (Ctrl+Enter)" if ideas else
                                 "Send this message: the model changes the current version (Ctrl+Enter)" if chat
                                 else "Ask the model (again: another try) (Ctrl+Enter)")
        self.hint.setPlaceholderText(
            "Optional, for all ideas: e.g. “surreal”, “35mm film photo”, “always at night”" if ideas else
            "Tell it what to change, e.g. “darker mood”, “keep her coat, camera behind her”, “use what you see in "
            "the picture” (empty: just improve it)" if chat else
            "Optional: how should it change? e.g. “more cinematic”, “shorter”, “add rain sounds”")
        self.out_title.setText("Current version (you can edit it; the next message continues from your edit)"
                               if chat else "Refined (you can edit it)")
        show_pics = chat                              # (only a conversation model can be shown pictures)
        self.pics_title.setVisible(show_pics)
        for i in range(self.pics_row.count()):
            w = self.pics_row.itemAt(i).widget()
            if w is not None:
                w.setVisible(show_pics)
                w.setEnabled(see)
        self.pics_title.setText("Pictures the model sees (click to switch on / off)" if see else
                                "Pictures (this model cannot see them)")
        self._update_buttons()

    # ---- pictures
    def _show_pictures(self):
        while self.pics_row.count():
            it = self.pics_row.takeAt(0)
            w = it.widget()                        # (after setParent(None) the item no longer knows it)
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for i, pic in enumerate(self.pictures):
            b = QToolButton()
            b.setCheckable(True)
            b.setChecked(pic["on"])
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            img = QImage(pic["path"])
            if not img.isNull():
                b.setIcon(QIcon(QPixmap.fromImage(img.scaled(96, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))))
                b.setIconSize(QSize(96, 64))
            b.setText(pic["title"] if len(pic["title"]) <= 22 else pic["title"][:20] + "…")
            b.setToolTip(f"{pic['title']}\n{pic['path']}\nClick: the model sees it (on) or not (off); right-click: "
                         "remove")
            b.toggled.connect(lambda on, i=i: self.pictures[i].update(on=on))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, i=i: self._remove_picture(i))
            self.pics_row.addWidget(b)
        add = QToolButton()
        add.setText("+ Picture")
        add.setToolTip("Show the model another picture (e.g. a reference for the look)")
        add.clicked.connect(self._add_picture)
        self.pics_row.addWidget(add)
        self.pics_row.addStretch(1)

    def _add_picture(self):
        from .image_picker import pick_image
        path = pick_image(self, "A picture for the model to see", self.qs)
        if path:
            self.pictures.append({"title": Path(path).stem, "path": path, "on": True})
            self._show_pictures()
            self._mode_changed()

    def _remove_picture(self, i: int):
        if 0 <= i < len(self.pictures):
            del self.pictures[i]
            self._show_pictures()
            self._mode_changed()

    def chosen_pictures(self) -> list[tuple[str, str]]:
        return [(p["title"], p["path"]) for p in self.pictures if p["on"]] if self.seeing() else []

    # ---- models
    def _load_models(self):
        self.state.setText("Connecting …")
        sig, connect = self._sig, self.connect

        def work():
            try:
                self.refiner = connect()
                sig.models.emit(self.refiner.models, "")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.models.emit([], str(e))
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _models(self, models: list, error: str):
        if error and getattr(self, "_retries", 0) > 0:        # ComfyUI is starting: ask again
            self._retries -= 1
            QTimer.singleShot(5000, self._load_models)
            return
        if error:
            down = "not reachable" in error
            self.state.setText(("ComfyUI is not running. " if down else "") + f"Cannot refine: {error}")
            self.b_start.setVisible(down and self.start_comfy is not None)
            self.model.setItemText(0, "no language model")
            return
        self.model.blockSignals(True)
        self.model.clear()
        for m in models:
            self.model.addItem(m["label"] + f"  —  {Path(m['file'].split(':', 1)[-1]).name}", m["file"])
            self.model.setItemData(self.model.count() - 1, m["file"], Qt.ToolTipRole)
        saved = self.qs.value("refinerModel", "", type=str) if self.qs is not None else ""
        i = self.model.findData(saved)
        self.model.setCurrentIndex(i if i >= 0 else 0)
        self.model.blockSignals(False)
        self.model.setEnabled(True)
        self.b_refine.setEnabled(True)
        self._mode_changed()
        self.state.setText(("Write what should change and press Send (Ctrl+Enter), or just Send to improve it. "
                            if self.chatting() else "Choose a model and press Refine (Ctrl+Enter). ")
                           + "The first answer loads the model, which takes a moment.")
        self.b_refine.setFocus()

    def _remember(self, *_):
        if self.qs is not None and self.model.currentData():
            self.qs.setValue("refinerModel", self.model.currentData())

    def _start_comfy(self):
        if self.start_comfy and self.start_comfy():
            self.b_start.hide()
            self.state.setText("Starting ComfyUI … the models are listed as soon as it answers.")
            self._retries = 36                     # up to 3 minutes
            QTimer.singleShot(5000, self._load_models)

    # ---- the conversation
    def _capture_edit(self):
        """The user's own edits of the current version count as that version (before switching or sending)."""
        if not self.chatting() or not (0 <= self.base < len(self.turns)):
            return
        edited = self.out.toPlainText().strip()
        if edited and " ".join(edited.split()) != " ".join(self.turns[self.base][1].split()):
            self.turns[self.base] = (self.turns[self.base][0], edited)
            self._edited[self.base] = True

    def _version_clicked(self, url):
        s = url.toString()
        if not s.startswith("v:") or self._busy:
            return
        k = int(s[2:])
        if not 0 <= k < len(self.turns):
            return
        self._capture_edit()
        self.base = k
        self.out.setPlainText(self.turns[k][1])
        self._render()
        self._update_buttons()
        self.state.setText(f"Back at version {k + 1}: the next message changes this one (the later versions stay "
                           "listed until you send).")

    def _render(self):
        if not self.turns:
            self.talk.setHtml("<p style='color:gray'>Your prompt is the start. Each message changes the current "
                              "version; every version stays here.</p>")
            return
        rows = []
        for k, (ask, version) in enumerate(self.turns):
            cur = k == self.base
            short = html.escape(version if len(version) <= 220 else version[:218] + "…")
            rows.append(f"<p style='margin:4px 0 0 0'><b>You:</b> {html.escape(ask) if ask else '<i>improve it</i>'}</p>"
                        f"<p style='margin:0 0 6px 12px'><a href='v:{k}' style='text-decoration:none'>"
                        f"<b>{'▶ ' if cur else ''}Version {k + 1}</b></a>"
                        f"{' <i>(edited by you)</i>' if self._edited.get(k) else ''}<br>"
                        f"<span style='color:{'#ddd' if cur else 'gray'}'>{short}</span></p>")
        self.talk.setHtml("".join(rows))
        self.talk.verticalScrollBar().setValue(self.talk.verticalScrollBar().maximum())

    def _update_buttons(self):
        idle = not self._busy and self.refiner is not None
        self.b_all.setEnabled(idle and bool(self.ideas))
        self.b_refine.setEnabled(idle)
        self.b_again.setEnabled(idle and self.chatting() and bool(self.turns) and self.base >= 0)

    def again(self):
        """The message that led to the current version, asked again (another answer replaces it; the version
        before it stays as it was)."""
        if self._busy or self.base < 0 or self.refiner is None:
            return
        ask = self.turns[self.base][0]
        self._base_before = self.base
        self.base -= 1
        self.refine(ask, capture=False)

    def refine(self, ask: str | None = None, capture: bool = True):
        if self._busy or self.refiner is None or not self.original.toPlainText().strip():
            if self._base_before is not None:          # (Another try that could not start: back where it was)
                self.base = self._base_before
            self._base_before = None
            return
        if self.ideas_mode():
            return self.make_ideas()
        chat = self.chatting()
        if chat and capture:
            self._capture_edit()
        self._busy = True
        self._update_buttons()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.state.setText("Thinking …" if chat else "Refining …")
        if not str(self.model.currentData() or "").startswith("llama:"):
            self.refined_any = True                # a ComfyUI model is being loaded (unloaded before clips)
        self.state.setProperty("severity", "")
        repolish(self.state)
        text, kind, model = self.original.toPlainText(), self.kind, self.model.currentData()
        ask = self.hint.text().strip() if ask is None else ask
        self._pending = ask
        sig, refiner, seed, cast = self._sig, self.refiner, random.randint(0, 2**31 - 1), self.cast
        turns, pics = list(self.turns[:self.base + 1]), self.chosen_pictures()
        clip = self._clip()

        def work():
            try:
                if chat:
                    out = refiner.chat(text, kind, turns, ask, pictures=pics, cast=cast or None, seed=seed, **clip)
                else:
                    out = refiner.refine_prompt(text, kind, model, seed, ask, **({"cast": cast} if cast else {}),
                                                **clip)
                sig.result.emit(out, "")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.result.emit("", str(e))
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _clip(self) -> dict:
        """What a "video_h3" refine is told about the clip (nothing for the other kinds). A conversation shows the
        start frame only when it is among the chosen pictures (else the model is told it cannot see it); ComfyUI's
        refiner shows it to its vision models."""
        return {"seconds": self.seconds, "start_image": self.start_image} if self.kind == "video_h3" else {}

    def _result(self, text: str, error: str):
        self._busy = False
        if error:
            if self._base_before is not None:          # Another try failed: back where it was
                self.base = self._base_before
                self._render()
            self._base_before = None
            self.state.setText(f"Could not refine: {error}")
            self.state.setProperty("severity", "")
            repolish(self.state)
        else:
            self._base_before = None
            if not str(self.model.currentData() or "").startswith("llama:"):
                self.refined_any = True
            if self.chatting():
                self.turns = self.turns[:self.base + 1] + [(self._pending, text)]
                self.base = len(self.turns) - 1
                self._edited = {k: v for k, v in self._edited.items() if k < self.base}
                self.hint.clear()
                self._render()
            self.out.setPlainText(text)
            same = " ".join(text.lower().strip(" .").split()) == \
                " ".join(self.original.toPlainText().lower().strip(" .").split())
            lost = [] if same else refine_problems(self.original.toPlainText(), text)
            nxt = "Send another message to change it further, " if self.chatting() else "Refine for another try, "
            self.state.setText("The model gave the prompt back unchanged. " + nxt + "or choose another model."
                               if same else
                               ("Check this: " + "; ".join(lost) + ". " + nxt + "or edit it." if lost
                                else f"{len(text.split())} words. Edit it if you like, " + nxt.lower()
                                + "or Use this."))
            self.state.setProperty("severity", "warning" if lost else "")
            repolish(self.state)
        self._mode_changed()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.out.toPlainText().strip())
                                                            and not self.ideas_mode())

    # ---- ideas from bullet points (image prompts)
    def make_ideas(self):
        if self._busy or self.refiner is None or not self.original.toPlainText().strip():
            return
        self._busy = True
        self._update_buttons()
        self._mode_changed()
        n, wish, bullets = self.count.value(), self.hint.text().strip(), self.original.toPlainText()
        self._asked_bullets = bullets
        self.state.setText(f"Making {n} ideas … (about {4 * n} s)")
        self.state.setProperty("severity", "")
        repolish(self.state)
        sig, refiner, seed, cast, pics = self._sig, self.refiner, random.randint(0, 2**31 - 1), self.cast, \
            self.chosen_pictures()

        def work():
            try:
                sig.ideas.emit(refiner.ideas(bullets, n, wish, pictures=pics, cast=cast or None, seed=seed), "")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.ideas.emit([], str(e))
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _ideas_ready(self, ideas: list, error: str):
        self._busy = False
        if error:
            self.state.setText(f"Could not make ideas: {error}")
        else:
            from ..llm_server import missing_bullets
            bullets = getattr(self, "_asked_bullets", self.original.toPlainText())   # (what was asked)
            for d in ideas:
                d["missing"] = missing_bullets(bullets, d["prompt"])
            self.ideas = ideas
            short = sum(1 for d in ideas if d["missing"])
            self.state.setText(f"{len(ideas)} ideas. Use one, refine one further, or Generate all (one image each)."
                               + (f" {short} may leave a bullet point out (marked)." if short else ""))
            self.state.setProperty("severity", "warning" if short else "")
            repolish(self.state)
        self._update_buttons()
        self._mode_changed()

    def _render_ideas(self):
        if not self.ideas_view.isVisibleTo(self):
            return
        if not self.ideas:
            self.ideas_view.setHtml("<p style='color:gray'>Write bullet points on the left, e.g.<br>"
                                    "- @Anna in a red coat<br>- an old lighthouse<br>- heavy rain<br>"
                                    f"then press Make {self.count.value()} ideas.</p>")
            return
        rows = []
        for k, d in enumerate(self.ideas):
            miss = ("<br><span style='color:#e0a040'>may leave out: "
                    + html.escape("; ".join(d["missing"])) + "</span>") if d.get("missing") else ""
            rows.append(f"<p style='margin:0 0 10px 0'><b>{k + 1}. {html.escape(d['title'])}</b>"
                        f" <span style='color:gray'>— {html.escape(d['angle'])}</span><br>"
                        f"{html.escape(d['prompt'])}{miss}<br>"
                        f"<a href='use:{k}'>Use this</a> &nbsp; <a href='more:{k}'>Refine further</a></p>")
        self.ideas_view.setHtml("".join(rows))

    def _idea_link(self, url):
        if self._busy:
            return
        s = url.toString()
        try:
            kind, k = s.split(":", 1)
            d = self.ideas[int(k)]
        except (ValueError, IndexError):
            return
        if kind == "use":
            self._chosen = d["prompt"]
            self.accept()
        elif kind == "more":                       # a conversation about this idea: it is the prompt now
            self._texts["refine"] = d["prompt"]
            self.turns, self.base, self._edited = [], -1, {}
            self.set_ideas_mode(False)
            self.out.setPlainText(d["prompt"])
            self._render()
            self.state.setText(f"Idea {int(k) + 1} is the prompt now: tell the model what to change (your bullet "
                               "points stay in Ideas).")

    def _use_all(self):
        if self.ideas and not self._busy:
            self._chosen_ideas = [d["prompt"] for d in self.ideas]
            self.accept()

    def chosen_ideas(self) -> list[str] | None:
        """All ideas, when the user chose Generate all (the caller makes them variations)."""
        return self._chosen_ideas

    def _use(self):
        text = " ".join(self.out.toPlainText().split())
        if text:
            self._chosen = text
            self.accept()

    def chosen(self) -> str | None:
        return self._chosen

    def done(self, r):
        if self._busy and self.refiner is not None and hasattr(self.refiner, "interrupt"):
            threading.Thread(target=self.refiner.interrupt, daemon=True).start()
        super().done(r)
