"""Fixes in the LoRA window, the LoRA stack and the refine window (UIL-1, UIL-2, UIL-3, UIL-6). Made-up data."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QToolButton

from test_lora_notes import db_with, details, lora

app = QApplication.instance() or QApplication([])


def window(tmp_path, **more):
    from video_chain.ui.lora_window import LoraWindow
    db, p = db_with(tmp_path, details(7))
    other = str(lora(tmp_path / "loras" / "other.safetensors", {"y": "2"}))
    db.scan([other], lookup=lambda sha: None, gap=0)
    w = LoraWindow(None, [p, other], current=p, db=db, fetch=lambda item: None, **more)
    w.flush()
    return w, db, p, other


def test_lora_stack_disabled_disables_the_remove_buttons():
    from video_chain.ui.lora_stack import LoraStack
    stack = LoraStack()
    stack.set_items([], [{"path": "/x/a.safetensors", "strength": 1.0}])
    rm = [b for b in stack.findChildren(QToolButton) if b.toolTip() == "Remove this LoRA"]
    assert rm and all(b.isEnabled() for b in rm)
    stack.setEnabled(False)
    assert not any(b.isEnabled() for b in rm)
    stack.setEnabled(True)
    assert all(b.isEnabled() for b in rm)


def test_chosen_prompt_keeps_its_line_breaks(tmp_path):
    from video_chain.ui.prompt_refiner import RefineDialog

    class Fake:
        models = [{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": False}]
    dlg = RefineDialog(None, "a cat", "video", lambda: Fake())
    dlg.out.setPlainText("  A cat   sits.\nShe says:  hello  \n\nThen leaves. ")
    dlg._use()
    assert dlg.chosen() == "A cat sits.\nShe says: hello\n\nThen leaves."


def test_unsaved_note_is_not_lost_on_switching(tmp_path):
    w, db, p, other = window(tmp_path)
    asked = []
    answer = ["cancel"]
    w._ask_unsaved = lambda: (asked.append(1), answer[0])[1]
    w.note_edit.setPlainText("my own note")
    w._select(other)
    w.flush()
    assert asked and w.path == p and w.note_edit.toPlainText() == "my own note"       # cancelled: stays
    assert w.list.currentItem().data(0x0100) == p
    answer[0] = "discard"
    w._select(other)
    w.flush()
    assert w.path == other and w.note_edit.toPlainText() == "" and db.note(p) is None
    w._select(p)
    w.flush()
    w.note_edit.setPlainText("kept note")
    answer[0] = "save"
    w._select(other)
    w.flush()
    assert w.path == other and db.note(p)["note"] == "kept note"
    w.done(0)


def test_strength_link_keeps_unsaved_edits(tmp_path):
    from PySide6.QtCore import QUrl
    w, db, p, other = window(tmp_path, set_strength=lambda path, v: True)
    w.strengths[p] = 1.0
    w.show_lora(p)
    w.note_edit.setPlainText("typing …")
    w._link(QUrl("strength:0.80"))
    assert w.note_edit.toPlainText() == "typing …"
    assert "at 0.80" in w.how.toPlainText()
    w._ask_unsaved = lambda: "discard"
    w.done(0)


def test_closing_asks_about_unsaved_edits(tmp_path):
    w, db, p, other = window(tmp_path)
    w.remarks_edit.setPlainText("remember this")
    w._ask_unsaved = lambda: "cancel"
    w.done(0)
    assert not w._closed.is_set()
    w._ask_unsaved = lambda: "save"
    w.done(0)
    assert w._closed.is_set() and db.note(p)["remarks"] == "remember this"


def test_find_and_selection_are_debounced(tmp_path, monkeypatch):
    w, db, p, other = window(tmp_path)
    shows, refills = [], []
    orig_show, orig_refill = w.show_lora, w._refill
    monkeypatch.setattr(w, "show_lora", lambda path: (shows.append(path), orig_show(path))[1])
    monkeypatch.setattr(w, "_refill", lambda: (refills.append(1), orig_refill())[1])
    for text in ("f", "fo", "fox"):
        w.find.setText(text)
    assert not refills
    w.flush()
    assert len(refills) == 1 and w.find.text() == "fox"
    w._select(other)
    w._select(p)
    w._select(other)
    assert not shows
    w.flush()
    assert shows == [other]
    w.done(0)


def test_kept_pictures_are_looked_up_once(tmp_path, monkeypatch):
    from video_chain import lora_pictures
    w, db, p, other = window(tmp_path)
    calls = []
    real = lora_pictures.cached
    monkeypatch.setattr(lora_pictures, "cached", lambda item, d: (calls.append(1), real(item, d))[1])
    item = {"id": 1}
    w._kept(item)
    w._kept(item)
    assert len(calls) == 1
    w.done(0)
