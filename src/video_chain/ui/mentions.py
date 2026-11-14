"""@mentions of the cast in prompt editors: a list of names while typing @..., known names highlighted, unknown
ones marked. #camera tokens likewise (camera.py: #push-in-slow ...): offered while typing #, in plain words ("Push in,
slow"; what is typed still matches the token), known ones highlighted."""
from __future__ import annotations

import re
from typing import Callable

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import (QFont, QStandardItem, QStandardItemModel, QSyntaxHighlighter, QTextCharFormat,
                           QTextCursor)
from PySide6.QtWidgets import QCompleter, QPlainTextEdit

from .. import camera as CAM
from .. import tree as T
from .theme import theme

_TYPING = re.compile(r"(?<![\w@])@(\w[\w-]*)?$")      # an @mention being typed right before the cursor
_TYPING_CAM = re.compile(r"(?<![\w#&/])#([A-Za-z][\w-]*)?$")   # a #camera token being typed
INSERT = Qt.UserRole + 1                               # the list's role with the text that is put in (its match too)


class MentionHighlighter(QSyntaxHighlighter):
    def __init__(self, doc, names: Callable[[], list[str]]):
        super().__init__(doc)
        self.names = names

    def highlightBlock(self, text: str):
        known = {n.lower() for n in self.names()}
        t = theme()
        for m in T.MENTION.finditer(text):
            f = QTextCharFormat()
            if m.group(1).lower() in known:
                f.setForeground(t.accent)
                f.setFontWeight(QFont.DemiBold)
            else:
                f.setForeground(t.colour("warning") or t.accent)
                f.setFontUnderline(True)
                f.setUnderlineStyle(QTextCharFormat.WaveUnderline)
            self.setFormat(m.start(), m.end() - m.start(), f)
        for m in CAM.TOKEN.finditer(text):           # known camera tokens (other #words stay plain)
            if CAM.is_token(m.group(1)):
                f = QTextCharFormat()
                f.setForeground(t.accent)
                f.setFontItalic(True)
                self.setFormat(m.start(), m.end() - m.start(), f)


class MentionHelper(QObject):
    """Typing @ (and the first letters of a name) in the editor offers the cast's names; Enter / Tab / a click
    puts the whole name in. Typing # offers the camera moves, shown in plain words, and puts the token in."""

    def __init__(self, edit: QPlainTextEdit, names: Callable[[], list[str]]):
        super().__init__(edit)
        self.edit, self.names = edit, names
        self._kind = "@"                           # what is being typed: "@" a cast name, "#" a camera token
        self.highlighter = MentionHighlighter(edit.document(), names)
        self.model = QStandardItemModel(self)
        self.completer = QCompleter(self.model, self)
        self.completer.setCompletionRole(INSERT)          # matched and put in: the name / the token
        self.completer.setWidget(edit)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.activated.connect(self._insert)
        edit.textChanged.connect(self._update)
        edit.installEventFilter(self)

    def _typed(self) -> re.Match | None:
        """An @name or #camera token being typed right before the cursor (its kind in self._kind)."""
        cur = self.edit.textCursor()
        block = cur.block().text()[:cur.positionInBlock()]
        m = _TYPING.search(block)
        self._kind = "@"
        if m is None:
            m = _TYPING_CAM.search(block)
            self._kind = "#"
        return m

    def _update(self):
        popup = self.completer.popup()
        m = self._typed() if self.edit.hasFocus() else None
        names = (self.names() if self._kind == "@" else CAM.tokens()) if m is not None else []
        if m is None or not names:
            popup.hide()
            return
        prefix = m.group(1) or ""
        self._fill(names)
        self.completer.setCompletionPrefix(prefix)
        n = self.completer.completionCount()
        if n == 0 or (n == 1 and self.completer.currentCompletion().lower() == prefix.lower()):
            popup.hide()                            # nothing to offer, or the name is already complete
            return
        popup.setCurrentIndex(self.completer.completionModel().index(0, 0))
        rect = self.edit.cursorRect()
        rect.setWidth(max(160, popup.sizeHintForColumn(0) + popup.verticalScrollBar().sizeHint().width() + 16))
        self.completer.complete(rect)

    def _fill(self, names: list[str]):
        """The list: cast names as they are, camera tokens in plain words (the token in the tooltip)."""
        self.model.clear()
        for name in names:
            words = CAM.label(name) if self._kind == "#" else None
            it = QStandardItem(words or name)
            it.setData(name, INSERT)
            if words:
                it.setToolTip(f"#{name}")
            self.model.appendRow(it)

    def offered(self) -> list[tuple[str, str]]:
        """What the list offers now: (shown text, put-in text) (for tests and tools)."""
        cm = self.completer.completionModel()
        return [(cm.index(r, 0).data(), cm.index(r, 0).data(INSERT)) for r in range(cm.rowCount())]

    def _insert(self, name: str):
        m = self._typed()
        if m is None:
            return
        cur = self.edit.textCursor()
        cur.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(m.group(1) or ""))
        doc = self.edit.document()
        at_end = cur.selectionEnd() >= doc.characterCount() - 1
        nxt = "" if at_end else doc.characterAt(cur.selectionEnd())
        cur.insertText(name + ("" if nxt and (nxt.isspace() or nxt in ".,;:!?'\")") else " "))
        self.edit.setTextCursor(cur)
        self.completer.popup().hide()

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.KeyPress and self.completer.popup().isVisible():
            if ev.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Tab):
                idx = self.completer.popup().currentIndex()
                if idx.isValid():                  # (nothing chosen: the key does what it does without the list)
                    self._insert(idx.data(INSERT))
                    return True
                return False
            if ev.key() == Qt.Key_Escape:
                self.completer.popup().hide()
                return True
        return False

    def rehighlight(self):
        """After the cast changed. Qt reports new highlighting as a text change, so the editor's signals are held
        back: the card would otherwise write its (maybe outdated) text back into the clip."""
        was = self.edit.blockSignals(True)
        try:
            self.highlighter.rehighlight()
        finally:
            self.edit.blockSignals(was)
