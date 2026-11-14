"""Prompt refiner: a language model in ComfyUI rewrites a prompt (a video clip's, an image's, an edit instruction).
The window shows the original and the suggestion side by side; the suggestion can be edited, refined again (another
seed, a hint like "more cinematic"), and taken over. Which model refines is chosen here and remembered."""
from __future__ import annotations

import html
import logging
import random
from collections.abc import Callable
from typing import Any
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLineEdit, QPlainTextEdit,
                               QPushButton, QSpinBox, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from . import shortcuts as S
from .workers import post, run_in_background
from ..tree import refine_problems
from .lora_details import themed
from .theme import repolish, theme
from .dialogs import add_action, button_row
from .text_edit import replace_text  # noqa: F401  (main_window and image_edit import it from here)
from .widgets import fit_size, guard_wheel, label, remember_geometry, shortcut
from .qsvalue import qs_value

log = logging.getLogger(__name__)

LIKE_EXAMPLE = ("Rewrite my prompt in the manner of this example prompt, which is known to work with the LoRA: take "
                "its wording, its structure and its level of detail. Keep what my prompt is about (my subjects, my "
                "cast, the place and what happens) and take nothing of the example's content. The example: "
                "\"{example}\"")
LIKE_EXAMPLE_LABEL = "like an example prompt of the LoRA"


def shown_ask(ask: str) -> str:
    """What the conversation shows for a message: the long fixed instruction for "write like this example" as a short label."""
    return LIKE_EXAMPLE_LABEL if ask.startswith(LIKE_EXAMPLE.split("{example}")[0]) else ask


def guidance_or_empty(loras: list, prompt: str) -> str:
    """What the language model is told about the LoRAs (nothing when their data cannot be read: a plain refine)."""
    from .. import lora_notes
    try:
        return lora_notes.guidance(loras, prompt=prompt)
    except Exception:  # noqa: BLE001
        log.warning("what the LoRAs' notes say could not be read", exc_info=True)
        return ""


KIND_TEXT = {"video": "video clip prompt", "video_h3": "video clip prompt", "image": "image prompt",
             "edit": "edit instruction"}


SETUP_HINT = ("More models: set up a local language model (llama.cpp, GGUF files) in the main window under "
              "Preferences, Prompt refiner.")


class RefineDialog(QDialog):
    """connect() -> a refiner (called in a thread): ComfyUI's, or CombinedRefiner with llama.cpp models.
    ComfyUI's models refine once per press (with an optional hint). llama.cpp models hold a conversation: each
    message changes the current version, every version stays and can be continued from, and pictures (the clip's
    start frame, the image to edit, …) can be shown to a model that sees. chosen() -> the text to use, or None.
    pictures: [(title, path, on)] offered at the start. seconds, start_image: the clip's length and the frame it
    starts from (kind "video_h3": the refiner fits the action to the length and anchors on that frame).
    loras: the paths of the LoRAs the result is made with; the model is told what is known of them (lora_notes:
    their creators' words and prompts that work with them), and writes a note on each once."""
    connect: Callable   # (shadows QObject.connect, which this dialog never uses)

    def __init__(self, parent, text: str, kind: str, connect, settings=None, start_comfy=None, cast=None,
                 pictures=None, seconds: float | None = None, start_image=None, loras=None):
        super().__init__(parent)
        self.loras = [str(p) for p in (loras or []) if p]
        self.cast = cast or []                    # the project's characters: the refiner keeps @Name, writes lines
        self.seconds, self.start_image = seconds, (str(start_image) if start_image else None)
        self.setWindowTitle(f"Refine the {KIND_TEXT.get(kind, 'prompt')}")
        fit_size(self, 1100, 680)
        self.kind, self.connect, self.qs, self.start_comfy = kind, connect, settings, start_comfy
        remember_geometry(self, settings, "promptRefiner")
        self.refiner: Any = None                  # (the connected refiner)
        self._connecting = False
        self._retries = 0                         # (more tries to connect: ComfyUI is starting)
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
        self.ideas: list[dict] = []
        self._chosen_ideas: list[str] | None = None
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        model_label = label("Model", level="tertiary")
        top.addWidget(model_label)
        self.model = QComboBox()
        model_label.setBuddy(self.model)
        self.model.setAccessibleName("Model")
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
        self.use_loras = QCheckBox("Write for the LoRAs")
        self.use_loras.setChecked(qs_value(self.qs, "refinerLoraKnowledge", True, bool) if self.qs is not None
                                  else True)
        self.use_loras.toggled.connect(self._remember_loras)
        self.b_loras = QToolButton()
        self.b_loras.setProperty("role", "textButton")
        self.b_loras.setText("What the model is told…")
        self.b_loras.setToolTip("Open the LoRA window: word for word what the language model is told about these "
                                "LoRAs, their example prompts with the pictures they made, and everything else "
                                "that is known of them")
        self.b_loras.setAutoRaise(True)
        self.b_loras.setCursor(Qt.CursorShape.PointingHandCursor)
        self.b_loras.clicked.connect(lambda: self.show_loras())
        self._lora_row = QWidget()
        lrow = QHBoxLayout(self._lora_row)
        lrow.setContentsMargins(0, 0, 0, 0)
        lrow.addWidget(self.use_loras)
        lrow.addWidget(self.b_loras)
        lrow.addStretch(1)
        self._lora_row.hide()                     # (shown when the prompt has LoRAs)
        self._told = ""                           # the LoRA part of the last instruction, word for word
        v.addWidget(self._lora_row)
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
        self.count.valueChanged.connect(self._ideas_count_changed)
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
        self.b_stop = QPushButton("Stop")
        self.b_stop.setToolTip("Stop the model's answer that is being made")
        self.b_stop.clicked.connect(lambda: self.stop())
        self.b_stop.hide()
        hrow.addWidget(self.b_stop)
        v.addLayout(hrow)
        self.state = label("", level="secondary")
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        self.buttons, _ok = button_row(self, "Use this", enabled=False, accept=self._use)
        self.b_start = add_action(self.buttons, "Start ComfyUI")
        self.b_start.hide()
        self.b_start.clicked.connect(self._start_comfy)
        v.addWidget(self.buttons)
        self.out.textChanged.connect(lambda: self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self.out.toPlainText().strip()) and not self._busy))
        shortcut(S.sequence("refine"), self, lambda: self.refine())
        guard_wheel(self)
        self._show_pictures()
        self._mode_changed()
        self._load_models()

    # ---- what the chosen model can do
    def current_model(self) -> dict:
        models: list[dict] = self.refiner.models if self.refiner else []
        return next((m for m in models
                     if m["file"] == self.model.currentData()), {})

    def chatting(self) -> bool:
        return bool(self.current_model().get("chat"))

    def seeing(self) -> bool:
        return bool(self.current_model().get("vision"))

    def _ideas_count_changed(self, v: int) -> None:
        if self.ideas_mode():
            self.b_refine.setText(f"Make {v} ideas")

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
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not ideas and not self._busy
                                                            and bool(self.out.toPlainText().strip()))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setToolTip("In Ideas mode: Use this on an idea, or Generate all"
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
            item = self.pics_row.itemAt(i)
            pic = item.widget() if item is not None else None
            if pic is not None:
                pic.setVisible(show_pics)
                pic.setEnabled(see)
        self.pics_title.setText("Pictures the model sees (click to switch on / off)" if see else
                                "Pictures (this model cannot see them)")
        self._update_buttons()

    # ---- pictures
    def _show_pictures(self):
        while self.pics_row.count():
            it = self.pics_row.takeAt(0)
            # (after setParent(None) the item no longer knows it)
            w = it.widget() if it is not None else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for i, pic in enumerate(self.pictures):
            b = QToolButton()
            b.setCheckable(True)
            b.setChecked(pic["on"])
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            img = QImage(pic["path"])
            if not img.isNull():
                b.setIcon(QIcon(QPixmap.fromImage(img.scaled(96, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation))))
                b.setIconSize(QSize(96, 64))
            b.setText(pic["title"] if len(pic["title"]) <= 22 else pic["title"][:20] + "…")
            b.setToolTip(f"{pic['title']}\n{pic['path']}\nClick: the model sees it (on) or not (off); right-click: "
                         "remove")
            b.toggled.connect(lambda on, i=i: self.pictures[i].update(on=on))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
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
    def _connected(self, refiner):
        self.refiner = refiner

    def _load_models(self):
        if self._connecting:
            return
        self._connecting = True
        self.state.setText("Connecting …")
        connect, loras = self.connect, list(self.loras)

        def work():
            if loras:
                self._tell_loras(loras)
            refiner = connect()
            post(self, lambda: self._connected(refiner))        # (queued before the list: set in the GUI thread)
            return refiner.models
        run_in_background(work, on_done=lambda models: self._models(models, ""),
                          on_error=lambda e: self._models([], str(e)), owner=self)

    def _tell_loras(self, loras: list) -> None:
        """(a worker thread) What is known of the LoRAs, to the dialog; nothing known: a plain refine."""
        try:
            from .. import lora_notes
            summary = lora_notes.summary(loras)
        except Exception:  # noqa: BLE001  (a LoRA's data cannot be read)
            log.warning("what is known of the LoRAs could not be read", exc_info=True)
            return
        post(self, lambda: self._loras_known(summary))

    def _models(self, models: list, error: str):
        self._connecting = False
        if error and self._retries > 0:        # ComfyUI is starting: ask again
            self._retries -= 1
            QTimer.singleShot(5000, self._load_models)
            return
        if error:
            down = "not reachable" in error
            self._say(("ComfyUI is not running. " if down else "") + f"Cannot refine: {error}", "error")
            self.b_start.setVisible(down and self.start_comfy is not None)
            self.model.setItemText(0, "no language model")
            return
        has_local = any(str(m.get("file", "")).startswith("llama:") for m in models)
        self.model.blockSignals(True)
        self.model.clear()
        for m in models:
            self.model.addItem(m["label"] + f"  —  {Path(m['file'].split(':', 1)[-1]).name}", m["file"])
            self.model.setItemData(self.model.count() - 1, m["file"], Qt.ItemDataRole.ToolTipRole)
        saved = qs_value(self.qs, "refinerModel", "", str) if self.qs is not None else ""
        i = self.model.findData(saved)
        self.model.setCurrentIndex(i if i >= 0 else 0)
        self.model.blockSignals(False)
        self.model.setEnabled(True)
        self.b_refine.setEnabled(True)
        self._mode_changed()
        self.state.setText(("Write what should change and press Send (Ctrl+Enter), or just Send to improve it. "
                            if self.chatting() else "Choose a model and press Refine (Ctrl+Enter). ")
                           + "The first answer loads the model, which takes a moment."
                           + ("" if has_local else " " + SETUP_HINT))
        self.model.setToolTip(self.model.toolTip().split("\n\n" + SETUP_HINT)[0]
                              + ("" if has_local else "\n\n" + SETUP_HINT))
        self.b_refine.setFocus()

    # ---- the LoRAs the result is made with
    def _loras_known(self, info: dict):
        """Say what is known of the stack's LoRAs (lora_notes.summary; called once their files are read): each
        with its number of example prompts and whether it has a note. Writing for them is offered when something
        is known of one."""
        each = info.get("loras") or []
        if not each:
            return
        from ..lora_notes import known_text
        names = info.get("known") or []
        parts = [f"{k['name']} ({known_text(k)})" for k in each[:3]]
        shown = ", ".join(parts) + (f" and {len(each) - 3} more" if len(each) > 3 else "")
        self.use_loras.setEnabled(bool(names))
        if names:
            self.use_loras.setText(f"Write for the LoRA{'s' if len(each) > 1 else ''}: {shown}")
            tip = ("The model is told what is known of these LoRAs: their trigger words, what their creators say "
                   "and prompts that work with them (from their CivitAI galleries; the most liked one and those "
                   "nearest to your prompt), and takes their wording where it fits your prompt. It never takes "
                   "the content of those prompts. “What the model is told…” shows it word for word.")
            if info.get("missing"):
                tip += (f"\n\nThe first answer takes longer: the model reads up on {info['missing']} LoRA"
                        f"{'s' if info['missing'] > 1 else ''} first and keeps a note on each (once per LoRA).")
            if info.get("settings"):
                tip += "\n\nSettings its example pictures were made with:\n" + "\n".join(
                    f"{n}: {t}" for n, t in info["settings"])
        else:
            self.use_loras.setText(f"LoRA{'s' if len(each) > 1 else ''}: {shown}")
            tip = ("Nothing is known of this prompt's LoRAs that helps writing for them (no trigger words, no "
                   "description, no example prompts), so the model is told nothing about them. They are looked "
                   "up on CivitAI in the background when that is on in the settings.")
        self.use_loras.setToolTip(tip)
        self._lora_row.show()

    def _was_told(self, text: str):
        self._told = text

    def show_loras(self):
        """The LoRA window for this prompt's LoRAs: what the model was told at the last refine (before the first
        one: what it will be told, without the notes still to write), and the examples behind it."""
        from .lora_window import LoraWindow
        on = self.use_loras.isEnabled() and self.use_loras.isChecked()
        told, title = self._told, "What the language model was told"
        if not told and on:
            told = guidance_or_empty(self.loras, self.original.toPlainText())
            title = "What the language model will be told"
        wanted: list[str] = []                     # (an example to rewrite like: asked once the window is closed)
        idle = not self._busy and self.refiner is not None and not self.ideas_mode()
        LoraWindow(self, self.loras, current=self.loras[0] if self.loras else None, settings=self.qs,
                   prompt=self.original.toPlainText(), told=told if on or self._told else "",
                   told_title=title, use_prompt=self._use_example if idle else None,
                   adapt=wanted.append if idle else None).exec()
        self._refresh_loras()
        if wanted:
            self.refine(LIKE_EXAMPLE.format(example=wanted[0]))

    def _use_example(self, text: str):
        """An example prompt from the LoRA window as the version to go on from (it can be edited, changed further
        by the model, or taken with Use this)."""
        if self.chatting():                        # a version of the conversation: the next message goes on from it
            self._capture_edit()
            self.turns = self.turns[:self.base + 1] + [("start from an example prompt of the LoRA", text)]
            self.base = len(self.turns) - 1
            self._edited = {k: v for k, v in self._edited.items() if k < self.base}
            self._render()
        self.out.setPlainText(text)
        self._mode_changed()
        self.state.setText("The example is in the field on the right: edit it, have it changed further, or press "
                           "Use this.")
        self.state.setProperty("severity", "")
        repolish(self.state)

    def _refresh_loras(self):
        """Read again what is known of the LoRAs (a note was written or forgotten)."""
        loras = list(self.loras)
        if not loras:
            return
        run_in_background(lambda: self._tell_loras(loras), on_done=lambda _r: None, owner=self)

    def _remember_loras(self, on: bool):
        if self.qs is not None:
            self.qs.setValue("refinerLoraKnowledge", bool(on))

    def _lora_guide(self, text: str = ""):
        """-> a function for the worker thread: the LoRA part of the instruction ("" when off or nothing is
        known), with the example prompts nearest to `text`. The model first writes the notes that are missing
        (lora_notes.make_notes). What it is told is kept for "What the model is told…"."""
        paths = list(self.loras) if (not self._lora_row.isHidden() and self.use_loras.isEnabled()
                                     and self.use_loras.isChecked()) else []
        refiner, model = self.refiner, self.model.currentData()

        def guide() -> str:
            if not paths:
                post(self, lambda: self._was_told(""))
                return ""
            from .. import lora_notes
            try:
                if hasattr(refiner, "complete"):
                    def said(name, i, n):
                        post(self, lambda: self._progress(f"Reading up on the LoRA {name} ({i} of {n}; once per LoRA) …"))
                    lora_notes.make_notes(paths, lambda system, user, n: refiner.complete(
                        system, user, n, model_file=model), progress=said)
                    post(self, lambda: self._progress(""))
                told = lora_notes.guidance(paths, prompt=text)
            except Exception:  # noqa: BLE001  (the LoRAs' data cannot be read: a plain refine)
                log.warning("the LoRAs' notes could not be read", exc_info=True)
                told = ""
            post(self, lambda: self._was_told(told))
            if told:
                self._tell_loras(paths)                                # (a note may be new: the counts say so)
            return told
        return guide

    def _say(self, text: str, severity: str = ""):
        """The line under the buttons; severity "" (plain), "warning" or "error" (the previous one is cleared)."""
        self.state.setText(text)
        self.state.setProperty("severity", severity)
        repolish(self.state)

    def _progress(self, text: str):
        if self._busy:
            self.state.setText(text or ("Thinking …" if self.chatting() else "Refining …"))

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
            self.talk.setHtml(themed("<p style='color:gray'>Your prompt is the start. Each message changes the current "
                              "version; every version stays here.</p>"))
            return
        rows = []
        for k, (ask, version) in enumerate(self.turns):
            cur = k == self.base
            short = html.escape(version if len(version) <= 220 else version[:218] + "…")
            rows.append(f"<p style='margin:4px 0 0 0'><b>You:</b> {html.escape(shown_ask(ask)) if ask else '<i>improve it</i>'}</p>"
                        f"<p style='margin:0 0 6px 12px'><a href='v:{k}' style='text-decoration:none'>"
                        f"<b>{'▶ ' if cur else ''}Version {k + 1}</b></a>"
                        f"{' <i>(edited by you)</i>' if self._edited.get(k) else ''}<br>"
                        f"<span style='color:{theme().textPrimary.name() if cur else theme().textTertiary.name()}'>{short}</span></p>")
        self.talk.setHtml(themed("".join(rows)))
        self.talk.verticalScrollBar().setValue(self.talk.verticalScrollBar().maximum())

    def _update_buttons(self):
        idle = not self._busy and self.refiner is not None
        self.b_all.setEnabled(idle and bool(self.ideas))
        self.b_refine.setEnabled(idle)
        self.b_again.setEnabled(idle and self.chatting() and bool(self.turns) and self.base >= 0)
        self.b_stop.setVisible(self._busy and hasattr(self.refiner, "interrupt"))
        if not self._busy:
            self.b_stop.setEnabled(True)

    def stop(self):
        """Ask the refiner to give up the answer being made (the dialog then shows the error that follows)."""
        if self._busy and hasattr(self.refiner, "interrupt"):
            self.b_stop.setEnabled(False)
            run_in_background(self.refiner.interrupt, on_done=lambda _r: None)

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
            if not self._busy and self.refiner is not None:
                self._say("There is no prompt to refine: write it on the left first.")
            return
        if self.ideas_mode():
            return self.make_ideas()
        chat = self.chatting()
        if chat and capture:
            self._capture_edit()
        self._busy = True
        self._update_buttons()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._say("Thinking …" if chat else "Refining …")
        if not str(self.model.currentData() or "").startswith("llama:"):
            self.refined_any = True                # a ComfyUI model is being loaded (unloaded before clips)
        text, kind, model = self.original.toPlainText(), self.kind, self.model.currentData()
        ask = self.hint.text().strip() if ask is None else ask
        self._pending = ask
        refiner, seed, cast = self.refiner, random.randint(0, 2**31 - 1), self.cast
        turns, pics = list(self.turns[:self.base + 1]), self.chosen_pictures()
        clip, guide = self._clip(), self._lora_guide(text)

        def work():
            loras = guide()
            more = {**clip, **({"loras": loras} if loras else {})}
            if chat:
                return refiner.chat(text, kind, turns, ask, pictures=pics, cast=cast or None, seed=seed, **more)
            return refiner.refine_prompt(text, kind, model, seed, ask, **({"cast": cast} if cast else {}), **more)
        run_in_background(work, on_done=lambda out: self._result(out, ""),
                          on_error=lambda e: self._result("", str(e)), owner=self)

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
            self._say(f"Could not refine: {error}", "error")
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
            self._say("The model gave the prompt back unchanged. " + nxt + "or choose another model." if same else
                      ("Check this: " + "; ".join(lost) + ". " + nxt + "or edit it." if lost
                       else f"{len(text.split())} words. Edit it if you like, " + nxt.lower() + "or Use this."),
                      "warning" if lost else "")
        self._mode_changed()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.out.toPlainText().strip())
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
        self._say(f"Making {n} ideas … (about {4 * n} s)")
        refiner, seed, cast, pics = self.refiner, random.randint(0, 2**31 - 1), self.cast, self.chosen_pictures()
        guide = self._lora_guide(bullets)

        def work():
            loras = guide()
            return refiner.ideas(bullets, n, wish, pictures=pics, cast=cast or None, seed=seed,
                                 **({"loras": loras} if loras else {}))
        run_in_background(work, on_done=lambda ideas: self._ideas_ready(ideas, ""),
                          on_error=lambda e: self._ideas_ready([], str(e)), owner=self)

    def _ideas_ready(self, ideas: list, error: str):
        self._busy = False
        if error:
            self._say(f"Could not make ideas: {error}", "error")
        else:
            from ..llm_server import missing_bullets
            bullets = getattr(self, "_asked_bullets", self.original.toPlainText())   # (what was asked)
            for d in ideas:
                d["missing"] = missing_bullets(bullets, d["prompt"])
            self.ideas = ideas
            short = sum(1 for d in ideas if d["missing"])
            self._say(f"{len(ideas)} ideas. Use one, refine one further, or Generate all (one image each)."
                      + (f" {short} may leave a bullet point out (marked)." if short else ""),
                      "warning" if short else "")
        self._update_buttons()
        self._mode_changed()

    def _render_ideas(self):
        if not self.ideas_view.isVisibleTo(self):
            return
        if not self.ideas:
            self.ideas_view.setHtml(themed("<p style='color:gray'>Write bullet points on the left, e.g.<br>"
                                           "- @Anna in a red coat<br>- an old lighthouse<br>- heavy rain<br>"
                                           f"then press Make {self.count.value()} ideas.</p>"))
            return
        rows = []
        for k, d in enumerate(self.ideas):
            miss = ("<br><span style='color:" + (theme().colour("warning") or theme().accent).name() + "'>may leave out: "
                    + html.escape("; ".join(d["missing"])) + "</span>") if d.get("missing") else ""
            rows.append(f"<p style='margin:0 0 10px 0'><b>{k + 1}. {html.escape(d['title'])}</b>"
                        f" <span style='color:gray'>— {html.escape(d['angle'])}</span><br>"
                        f"{html.escape(d['prompt'])}{miss}<br>"
                        f"<a href='use:{k}'>Use this</a> &nbsp; <a href='more:{k}'>Refine further</a></p>")
        self.ideas_view.setHtml(themed("".join(rows)))

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
        # (line breaks are kept: only runs of spaces inside a line and the blank ends are tidied)
        text = "\n".join(" ".join(line.split()) for line in self.out.toPlainText().strip().splitlines()).strip()
        if text:
            self._chosen = text
            self.accept()

    def chosen(self) -> str | None:
        return self._chosen

    def done(self, r):
        if self._busy and self.refiner is not None and hasattr(self.refiner, "interrupt"):
            run_in_background(self.refiner.interrupt, on_done=lambda _r: None)
        super().done(r)
