"""Prompt refiner: a language model in ComfyUI rewrites a prompt (a video clip's, an image's, an edit instruction).
The window shows the original and the suggestion side by side; the suggestion can be edited, refined again (another
seed, a hint like "more cinematic"), and taken over. Which model refines is chosen here and remembered."""
from __future__ import annotations

import random
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLineEdit, QPlainTextEdit,
                               QPushButton, QVBoxLayout)

from .widgets import guard_wheel, icon_button, label

KIND_TEXT = {"video": "video clip prompt", "image": "image prompt", "edit": "edit instruction"}


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


class RefineDialog(QDialog):
    """connect() -> a ComfyPromptRefiner (called in a thread). chosen() -> the text to use, or None."""

    def __init__(self, parent, text: str, kind: str, connect, settings=None, start_comfy=None):
        super().__init__(parent)
        self.setWindowTitle(f"Refine the {KIND_TEXT.get(kind, 'prompt')}")
        self.resize(900, 560)
        self.kind, self.connect, self.qs, self.start_comfy = kind, connect, settings, start_comfy
        self.refiner = None
        self.refined_any = False                  # a refine ran (its model is loaded in ComfyUI now)
        self._busy = False
        self._chosen: str | None = None
        self._sig = _Sig()
        self._sig.models.connect(self._models)
        self._sig.result.connect(self._result)
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(label("Model", level="tertiary"))
        self.model = QComboBox()
        self.model.addItem("looking for language models in ComfyUI …", None)
        self.model.setEnabled(False)
        self.model.setToolTip("The language model that rewrites the prompt (it runs in ComfyUI). Uncensored ones "
                              "follow your wording without softening it")
        self.model.currentIndexChanged.connect(self._remember)
        top.addWidget(self.model, 1)
        v.addLayout(top)
        row = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        left.addWidget(label("Your prompt", level="tertiary"))
        self.original = QPlainTextEdit(text)
        self.original.setReadOnly(True)
        left.addWidget(self.original, 1)
        right.addWidget(label("Refined (you can edit it)", level="tertiary"))
        self.out = QPlainTextEdit()
        self.out.setPlaceholderText("Press Refine")
        right.addWidget(self.out, 1)
        row.addLayout(left, 1)
        row.addLayout(right, 1)
        v.addLayout(row, 1)
        hrow = QHBoxLayout()
        self.hint = QLineEdit()
        self.hint.setPlaceholderText("Optional: how should it change? e.g. “more cinematic”, “shorter”, "
                                     "“add rain sounds”")
        self.hint.returnPressed.connect(self.refine)
        hrow.addWidget(self.hint, 1)
        self.b_refine = QPushButton("Refine")
        self.b_refine.setProperty("variant", "primary")
        self.b_refine.setToolTip("Ask the model (again: another try) (Ctrl+Enter)")
        self.b_refine.clicked.connect(self.refine)
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
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.refine)
        guard_wheel(self)
        self._load_models()

    # ---- models
    def _load_models(self):
        self.state.setText("Connecting to ComfyUI …")
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
            self.model.addItem(m["label"] + f"  —  {m['file']}", m["file"])
        saved = self.qs.value("refinerModel", "", type=str) if self.qs is not None else ""
        i = self.model.findData(saved)
        self.model.setCurrentIndex(i if i >= 0 else 0)
        self.model.blockSignals(False)
        self.model.setEnabled(True)
        self.b_refine.setEnabled(True)
        self.state.setText("Choose a model and press Refine (Ctrl+Enter). The first refine loads the model, which "
                           "takes a moment.")
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

    # ---- refining
    def refine(self):
        if self._busy or self.refiner is None or not self.original.toPlainText().strip():
            return
        self._busy = True
        self.b_refine.setEnabled(False)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.state.setText("Refining …")
        text, kind, model, hint = (self.original.toPlainText(), self.kind, self.model.currentData(),
                                   self.hint.text())
        sig, refiner, seed = self._sig, self.refiner, random.randint(0, 2**31 - 1)

        def work():
            try:
                sig.result.emit(refiner.refine_prompt(text, kind, model, seed, hint), "")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.result.emit("", str(e))
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _result(self, text: str, error: str):
        self._busy = False
        self.b_refine.setEnabled(True)
        if error:
            self.state.setText(f"Could not refine: {error}")
        else:
            self.refined_any = True
            self.out.setPlainText(text)
            same = " ".join(text.lower().strip(" .").split()) == \
                " ".join(self.original.toPlainText().lower().strip(" .").split())
            self.state.setText("The model gave the prompt back unchanged. Refine for another try, or choose "
                               "another model (Qwen3-VL 4B refined most reliably in tests)." if same else
                               f"{len(text.split())} words. Edit it if you like, Refine for another try, or "
                               "Use this.")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.out.toPlainText().strip()))

    def _use(self):
        text = " ".join(self.out.toPlainText().split())
        if text:
            self._chosen = text
            self.accept()

    def chosen(self) -> str | None:
        return self._chosen

    def done(self, r):
        if self._busy and self.refiner is not None:
            threading.Thread(target=self.refiner.interrupt, daemon=True).start()
        super().done(r)
