"""Fixes in the LoRA window, the LoRA stack and the refine window (UIL-1, UIL-2, UIL-3, UIL-6). Made-up data."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QToolButton

from waiting import show_lora, wait_loaded

from test_lora_notes import db_with, details, lora

app = QApplication.instance() or QApplication([])


def window(tmp_path, **more):
    from video_chain.ui.lora_window import LoraWindow
    d = details(7)
    for i, it in enumerate(d["examples"]):
        if isinstance(it, dict):
            it.update({"id": 100 + i, "url": f"https://image.civitai.com/k/u/original=true/{i}.jpeg"})
    db, p = db_with(tmp_path, d)
    other = str(lora(tmp_path / "loras" / "other.safetensors", {"y": "2"}))
    db.scan([other], lookup=lambda sha: None, gap=0)
    w = LoraWindow(None, [p, other], current=p, db=db, fetch=lambda item: None, **more)
    w.flush()
    wait_loaded(app, w)
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
    show_lora(app, w, p)
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


def test_forgetting_a_note_and_deleting_pictures_ask_first(tmp_path, monkeypatch):
    from video_chain import lora_notes as LN
    w, db, p, other = window(tmp_path)
    assert LN.save_user(db, p, note="Does: mine.")
    show_lora(app, w, p)
    asked = []
    answer = [False]
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[1]), QMessageBox.Yes if answer[0]
                                                                  else QMessageBox.No)[1])
    w.b_rewrite.click()
    assert asked and db.note(p)["note"] == "Does: mine."               # "no": the note stays
    answer[0] = True
    w.b_rewrite.click()
    assert db.note(p) is None or not db.note(p).get("note")
    asked.clear()
    answer[0] = False
    w.b_forget.click()
    assert asked
    w.done(0)


def _refine_dialog(models):
    from video_chain.ui.prompt_refiner import RefineDialog

    class Fake:
        pass
    Fake.models = models
    dlg = RefineDialog(None, "a cat", "image", lambda: Fake())
    end = time.time() + 5
    while not dlg.model.isEnabled() and time.time() < end:
        app.processEvents()
    app.processEvents()
    return dlg


def test_refiner_points_to_the_local_model_setup_when_there_is_none():
    only_comfy = _refine_dialog([{"file": "comfy/q.safetensors", "label": "Q", "chat": False, "vision": False}])
    assert "Preferences" in only_comfy.state.text() and "Prompt refiner" in only_comfy.state.text()
    assert "Preferences" in only_comfy.model.toolTip()
    only_comfy.reject()
    with_llama = _refine_dialog([{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": False}])
    assert "Preferences" not in with_llama.state.text()
    with_llama.reject()


def test_chip_row_uses_the_widget_flow_and_chips_are_reachable_by_tab():
    from PySide6.QtCore import QObject, Qt
    from PySide6.QtWidgets import QLayout
    from video_chain.ui.clip_inputs import ChipRow
    row = ChipRow()
    specs = [{"text": "@ann · lead", "tip": "t", "variant": "", "lead": True},
             {"text": "Start frame", "tip": "t", "variant": "", "lead": None}]
    row.set_specs(specs)
    assert not any(type(c).__module__.startswith("video_chain") for c in row.findChildren(QLayout)), "no Python QLayout"
    assert row.chips[0].focusPolicy() & Qt.TabFocus          # (the lead chip can be switched from the keyboard)
    assert row.b_prompt.focusPolicy() & Qt.TabFocus
    assert row.chips[1].focusPolicy() == Qt.NoFocus
    row.set_specs([specs[1]])
    assert row.texts() == ["Start frame"]
    row.resize(300, 40)
    row.set_specs(specs)
    assert row.texts() == ["@ann · lead", "Start frame"]


def test_refiner_failures_are_shown_as_errors_and_a_stale_warning_is_cleared():
    dlg = _refine_dialog([{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": False}])
    dlg.state.setProperty("severity", "warning")
    dlg._result("", "server went away")
    assert dlg.state.property("severity") == "error" and "server went away" in dlg.state.text()
    dlg._ideas_ready([], "no answer")
    assert dlg.state.property("severity") == "error"
    dlg._result("A fine prompt of a cat sitting.", "")
    assert dlg.state.property("severity") in ("", None)
    dlg.reject()


def test_refiner_says_why_it_cannot_start_and_connects_only_once():
    from video_chain.ui.prompt_refiner import RefineDialog
    dlg = _refine_dialog([{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": False}])
    dlg.original.setPlainText("   ")
    dlg.state.setText("")
    dlg.refine()
    assert dlg.state.text() and "prompt" in dlg.state.text().lower()
    dlg.reject()
    calls = []

    class Fake:
        models = []
    dlg = RefineDialog(None, "a cat", "image", lambda: (calls.append(1), time.sleep(0.2), Fake())[2])
    dlg._load_models()                     # (the retry timer and the Start button can both ask)
    end = time.time() + 5
    while dlg.model.isEnabled() is False and time.time() < end and not dlg.refiner:
        app.processEvents()
    app.processEvents()
    assert len(calls) == 1
    dlg.reject()


def test_a_running_refine_can_be_stopped():
    dlg = _refine_dialog([{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": False}])
    stopped = []
    dlg.refiner.interrupt = lambda: stopped.append(1)
    assert dlg.b_stop.isHidden()
    dlg._busy = True
    dlg._update_buttons()
    assert not dlg.b_stop.isHidden() and dlg.b_stop.isEnabled()
    dlg.b_stop.click()
    end = time.time() + 3
    while not stopped and time.time() < end:
        app.processEvents()
    assert stopped
    dlg._busy = False
    dlg._update_buttons()
    assert dlg.b_stop.isHidden()
    dlg.reject()


def test_mention_popup_keys_only_swallowed_when_something_is_inserted():
    from PySide6.QtCore import QEvent, QModelIndex
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QPlainTextEdit
    from video_chain.ui.mentions import MentionHelper
    edit = QPlainTextEdit()
    helper = MentionHelper(edit, lambda: ["Anna", "Anton"])
    popup = helper.completer.popup()
    helper._fill(["Anna", "Anton"])
    popup.show()
    popup.setCurrentIndex(QModelIndex())                     # (nothing chosen)
    assert popup.isVisible()

    def press(k, mod=Qt.NoModifier):
        return helper.eventFilter(edit, QKeyEvent(QEvent.KeyPress, k, mod))
    assert press(Qt.Key_Return) is False and press(Qt.Key_Tab) is False
    assert press(Qt.Key_Backtab, Qt.ShiftModifier) is False
    popup.hide()


def test_trigger_words_are_trimmed_and_the_stack_is_not_rebuilt_for_the_same_items():
    from video_chain.ui.lora_stack import LoraStack, clean_words
    assert clean_words(" one, two ,, three  ,") == ["one", "two", "three"]
    stack = LoraStack()
    items = [{"path": "/x/a.safetensors", "strength": 1.0}]
    stack.set_items([], items)
    first = stack.rows[0][0]
    stack.set_items([], [dict(i) for i in items])
    assert stack.rows[0][0] is first                         # (same list: the rows stay)
    stack.set_items([], [{"path": "/x/a.safetensors", "strength": 0.5}])
    assert stack.rows[0][0] is not first and stack.rows[0][2].value() == 0.5


def test_camera_menu_parses_each_token_once(monkeypatch):
    from video_chain import camera as CAM
    from video_chain.ui import problems
    calls = []
    real = CAM.parse
    monkeypatch.setattr(CAM, "parse", lambda t: (calls.append(t), real(t))[1])
    problems.camera_menu_items()
    assert len(calls) <= 2 * len(CAM.tokens())             # (one here, one in label(): was ~40x)


def test_problem_box_rebuild_leaves_no_old_rows():
    from video_chain.ui.problems import ProblemBox
    box = ProblemBox()
    box.set_items([{"text": "a", "action": "x", "button": "Fix"}, {"text": "b", "error": True}])
    box.set_items([{"text": "c"}])
    app.processEvents()
    assert [t.text() for t, _b in box.rows] == ["c"] and box.v.count() == 1
    assert len(box.findChildren(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel)) == 2    # mark + text


def test_description_html_drops_remote_loaders():
    from video_chain.ui.lora_details import clean_html
    out = clean_html('<link rel="stylesheet" href="http://x/y.css"><meta http-equiv="refresh" content="0"><p>hi</p>')
    assert "<link" not in out and "<meta" not in out and "<p>hi</p>" in out


def test_rich_text_uses_the_theme_colours_not_fixed_grey():
    from video_chain.ui.lora_details import details_html, themed
    from video_chain.ui.theme import theme
    dim = theme().textTertiary.name()
    assert themed("<p style='color:gray'>x</p>") == f"<p style='color:{dim}'>x</p>"
    out = details_html({"version": {"id": 1, "name": "v", "trainedWords": [], "files": [
        {"name": "f", "hashes": {"SHA256": "ab"}}]}, "model": {"name": "m"}}, [{"prompt": "p", "negative": "n"}])
    assert "color:gray" not in out
    w, db, p, other = window(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    assert "color:gray" not in w.how.toHtml() and "gray" not in w._how_html(p, *w._ctx)
    w.done(0)


def test_picture_loader_threads_start_only_when_a_picture_is_wanted(tmp_path):
    from PySide6.QtCore import QSettings
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    qs.setValue("loraPictures", "off")
    w, db, p, other = window(tmp_path, settings=qs)
    assert w._loaders == []
    w.pictures.setCurrentIndex(w.pictures.findData("all"))
    assert len(w._loaders) > 0
    w.done(0)


def test_mention_typed_tells_its_kind_without_a_side_effect():
    from PySide6.QtWidgets import QPlainTextEdit
    from video_chain.ui.mentions import MentionHelper
    edit = QPlainTextEdit()
    helper = MentionHelper(edit, lambda: ["Anna"])
    edit.setPlainText("hello @An")
    c = edit.textCursor()
    c.movePosition(c.MoveOperation.End)
    edit.setTextCursor(c)
    kind, m = helper._typed()
    assert kind == "@" and m.group(1) == "An"
