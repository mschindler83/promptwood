"""Putting text into a text field: at the cursor (with a space only where words would run together) or in place of
everything (one undo step, Ctrl+Z takes it back). Shared by the prompt helpers, the @ / # completer and the refiner."""
from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

CLOSERS = ".,;:!?'\")"          # no space goes before these


def insert_spaced(edit: QPlainTextEdit, text: str, cursor: QTextCursor | None = None, *, space_before: bool = True,
                  focus: bool = True) -> None:
    """Put text in place of cursor's selection (default: the field's cursor), with a space after it unless a space
    or a closing mark follows, and before it (space_before) unless the text before is a space. One undo step."""
    cur = cursor if cursor is not None else edit.textCursor()
    doc = edit.document()
    pos, end = cur.selectionStart(), cur.selectionEnd()
    before = doc.characterAt(pos - 1) if pos > 0 else ""
    after = doc.characterAt(end) if end < doc.characterCount() - 1 else ""
    pre = " " if space_before and before and not before.isspace() else ""
    post = "" if after and (after.isspace() or after in CLOSERS) else " "
    cur.beginEditBlock()
    cur.insertText(pre + text + post)
    cur.endEditBlock()
    edit.setTextCursor(cur)
    if focus:
        edit.setFocus()


def replace_text(edit: QPlainTextEdit, text: str) -> None:
    """Replace a text field's content as one step Ctrl+Z takes back."""
    cur = edit.textCursor()
    cur.beginEditBlock()
    cur.select(QTextCursor.Document)
    cur.insertText(text)
    cur.endEditBlock()
    edit.setTextCursor(cur)
