"""Shared UI building blocks (XC-14/16/17/18/19/20/21, UX-10/12/17/24, UIL-23/30/31, UMD-29/30). Made-up data, offscreen."""
import ast
import inspect
import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QFont, QImage, QKeySequence
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox, QPlainTextEdit

app = QApplication.instance() or QApplication([])

UI = Path(__file__).resolve().parent.parent / "src" / "video_chain" / "ui"


# ---------------------------------------------------------------- questions and messages (XC-16)
def test_confirm_defaults_to_no_and_returns_a_bool(monkeypatch):
    from video_chain.ui import dialogs
    seen = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (seen.append(a), QMessageBox.Yes)[1])
    assert dialogs.confirm(None, "Delete", "Delete it?") is True
    parent, title, text, buttons, default = seen[0]
    assert (title, text) == ("Delete", "Delete it?")
    assert default == QMessageBox.No and buttons == QMessageBox.Yes | QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    assert dialogs.confirm(None, "Delete", "Delete it?") is False


def test_tell_warn_and_failed_use_the_message_boxes(monkeypatch):
    from video_chain.ui import dialogs
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(("i", a[1], a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: said.append(("w", a[1], a[2])))
    dialogs.tell(None, "Rename", "Use letters")
    dialogs.warn(None, "Wildcard file", "Not valid")
    dialogs.failed(None, "Preset", "save the preset", OSError("disk full"))
    assert said == [("i", "Rename", "Use letters"), ("w", "Wildcard file", "Not valid"),
                    ("w", "Preset", "Could not save the preset: disk full")]


def test_no_local_copies_of_the_question_helpers():
    assert not hasattr(__import__("video_chain.ui.variations_panel", fromlist=["x"]), "_ask")
    from video_chain.ui import lora_window
    assert not hasattr(lora_window.LoraWindow, "_confirm")


# ---------------------------------------------------------------- the button row (XC-17, UX-25, UIL-41)
def test_button_row_wires_the_dialog_and_returns_the_ok_button():
    from video_chain.ui import dialogs
    dlg = QDialog()
    box, ok = dialogs.button_row(dlg, "Use this crop")
    assert ok.text() == "Use this crop" and box.button(QDialogButtonBox.Cancel).text() == "Cancel"
    assert box.button(QDialogButtonBox.Ok) is ok
    box.accepted.emit()
    assert dlg.result() == QDialog.Accepted
    dlg2 = QDialog()
    box2, _ = dialogs.button_row(dlg2, "Use this crop")
    box2.rejected.emit()
    assert dlg2.result() == QDialog.Rejected


def test_button_row_without_an_action_is_a_close_row():
    from video_chain.ui import dialogs
    dlg = QDialog()
    box, ok = dialogs.button_row(dlg)
    assert ok is None and box.button(QDialogButtonBox.Close) is not None and box.button(QDialogButtonBox.Cancel) is None


def test_extra_buttons_are_action_buttons_and_a_destructive_one_says_so():
    from video_chain.ui import dialogs
    box, _ = dialogs.button_row(QDialog(), "Use this")
    other = dialogs.add_action(box, "Other file…")
    gone = dialogs.add_action(box, "Delete the kept pictures", destructive=True)
    assert box.buttonRole(other) == QDialogButtonBox.ActionRole
    assert box.buttonRole(gone) == QDialogButtonBox.DestructiveRole


# ---------------------------------------------------------------- safe_emit (UIL-23)
class _Sig(QObject):
    ping = Signal(int)


def test_safe_emit_ignores_a_deleted_signal_source():
    import shiboken6
    from video_chain.ui.widgets import safe_emit
    got = []
    s = _Sig()
    s.ping.connect(got.append)
    assert safe_emit(s.ping, 3) is True and got == [3]
    bound = s.ping
    shiboken6.delete(s)
    with pytest.raises(RuntimeError):
        bound.emit(4)
    assert safe_emit(bound, 4) is False


def test_the_ui_files_have_no_local_emit_copies():
    for name in ("prompt_refiner", "tts_dialog"):
        text = (UI / f"{name}.py").read_text()
        assert "def _emit(" not in text and "def _emit_safe(" not in text, name
    for name in ("clip_dialogs", "lora_window", "speak_dialog", "swap_dialog", "image_edit", "audio_editor",
                 "video_editor", "image_picker", "results_gallery"):
        text = (UI / f"{name}.py").read_text()
        assert not re.search(r"\.emit\([^\n]*\)\n\s*except RuntimeError", text), name


# ---------------------------------------------------------------- FlowBox lives in widgets (XC-20, UIL-8)
def test_flowbox_is_defined_in_widgets_and_imported_from_there():
    from video_chain.ui import clip_inputs, results_gallery, variations_panel, widgets
    assert inspect.getmodule(widgets.FlowBox) is widgets
    assert "class FlowBox" not in (UI / "results_gallery.py").read_text()
    assert clip_inputs.FlowBox is widgets.FlowBox and results_gallery.FlowBox is widgets.FlowBox
    assert variations_panel.FlowBox is widgets.FlowBox


# ---------------------------------------------------------------- time (XC-18)
def test_one_time_formatter_in_the_ui_files():
    for name in ("clip_dialogs", "audio_editor", "video_editor", "timeline", "storyboard", "compare_player",
                 "swap_dialog", "keyframe_strip", "results_gallery"):
        text = (UI / f"{name}.py").read_text()
        assert not re.search(r"^def (fmt_s|fmt_time|fmt_left|_fmt)\(", text, re.M), name


# ---------------------------------------------------------------- one smooth scaler (XC-19)
def test_smooth_scaled_is_cached_per_image_and_size():
    from video_chain.ui.widgets import smooth_scaled
    img = QImage(200, 100, QImage.Format_RGB32)
    img.fill(0x336699)
    a = smooth_scaled(img, QSize(50, 50))
    assert (a.width(), a.height()) == (50, 25)
    assert smooth_scaled(img, QSize(50, 50)) is a
    assert smooth_scaled(img, QSize(40, 40)) is not a
    assert smooth_scaled(img, img.size()) is img
    img.fill(0x112233)                               # (changed pixels: the cache key moved on)
    assert smooth_scaled(img, QSize(50, 50)) is not a


def test_the_ui_files_use_the_shared_scaler():
    for name in ("face_crop", "mask_painter", "results_gallery"):
        text = (UI / f"{name}.py").read_text()
        assert "smooth_scaled" in text, name


# ---------------------------------------------------------------- LoRA names (XC-21)
def test_lora_names_come_from_one_helper():
    from video_chain.ui.lora_stack import lora_label
    assert lora_label("a/b/c.safetensors") == "c" and lora_label(None) == "none"
    for name in ("lora_stack", "lora_window"):
        stems = re.findall(r"Path\([^)]*\)\.stem", (UI / f"{name}.py").read_text())
        assert len(stems) <= (1 if name == "lora_stack" else 0), (name, stems)


# ---------------------------------------------------------------- settings (XC-14, TST-4)
def test_the_picker_never_falls_back_to_a_default_qsettings():
    from video_chain.ui import image_picker
    for fn in (image_picker.ImagePickerDialog.__init__, image_picker.pick_image):
        p = inspect.signature(fn).parameters["settings"]
        assert p.default is inspect.Parameter.empty, fn
    assert "QSettings()" not in (UI / "image_picker.py").read_text()


# ---------------------------------------------------------------- accessible names (UX-10, MWA-31)
def test_icon_buttons_have_an_accessible_name_from_the_tooltip():
    from video_chain.ui.widgets import icon_button
    b = icon_button("plus", "Continue: a new clip after the selected one (Ctrl+N)")
    assert b.accessibleName() == "Continue"
    assert b.toolTip().startswith("Continue: a new clip")
    assert icon_button("plus", "Remove").accessibleName() == "Remove"


def test_labelled_fields_get_a_buddy():
    from PySide6.QtWidgets import QLineEdit
    from video_chain.ui.widgets import label_for
    e = QLineEdit()
    lab = label_for("Name", e)
    assert lab.buddy() is e and e.accessibleName() == "Name"


# ---------------------------------------------------------------- sizes (UX-12)
def test_capped_size_stays_inside_the_screen():
    from video_chain.ui.widgets import capped_size
    assert capped_size(1240, 820, QSize(1366, 728)) == QSize(1229, 655)
    assert capped_size(800, 600, QSize(1920, 1080)) == QSize(800, 600)


def test_fit_size_resizes_a_widget_within_the_available_screen():
    from video_chain.ui import widgets
    dlg = QDialog()
    widgets.fit_size(dlg, 100000, 100000)
    avail = QApplication.primaryScreen().availableGeometry()
    assert dlg.width() <= avail.width() and dlg.height() <= avail.height()


def test_big_dialogs_use_fit_size():
    for name in ("image_edit", "image_picker", "prompt_refiner", "lora_window", "swap_dialog", "video_editor",
                 "audio_editor", "variations_panel", "face_crop", "mask_painter", "clip_dialogs"):
        text = (UI / f"{name}.py").read_text()
        assert not re.search(r"self\.resize\(\d", text), name


# ---------------------------------------------------------------- font sizes (UX-24)
def test_small_font_is_never_below_the_themes_small_size():
    from video_chain.ui.theme import theme
    from video_chain.ui.widgets import small_font
    f = small_font(QFont(), 0.3)
    assert f.pixelSize() >= theme().font.small


# ---------------------------------------------------------------- shortcuts (UMD-30, UX-17)
def test_the_shortcut_table_is_valid_and_has_no_double_chords():
    from video_chain.ui import shortcuts
    seen = {}
    for s in shortcuts.SHORTCUTS:
        assert all(QKeySequence(k).toString() for k in s.keys.split(" / ")), s
        assert s.text and s.where, s
        if s.where == "main window":
            assert s.keys not in seen, (s.keys, seen.get(s.keys))
            seen[s.keys] = s.id
    assert shortcuts.keys("generate") == "Ctrl+Return" and shortcuts.keys("new_project") == "Ctrl+Shift+N"
    assert shortcuts.tip("Open a project folder", "open") == "Open a project folder (Ctrl+O)"
    html = shortcuts.help_html()
    assert "Ctrl+Shift+N" in html and "Generate" in html


def test_the_guide_takes_its_shortcuts_from_the_table():
    from video_chain.ui import shortcuts
    from video_chain.ui.guide import GuidePanel, NextHint
    g = GuidePanel()
    assert shortcuts.keys("new_project") in g.b_new.toolTip() and shortcuts.keys("open") in g.b_open.toolTip()
    assert shortcuts.keys("generate") in g.gen_hint.text()
    assert shortcuts.keys("continue") in NextHint().b_next.toolTip()


# ---------------------------------------------------------------- the guide (UMD-29)
def test_guide_steps_show_their_number_until_done_and_name_their_buttons():
    from video_chain.ui.guide import GuidePanel, NextHint
    g = GuidePanel()
    g.set_state(project=True, picture=False, cast=False, prompt=False)
    assert g.step1.mark.text() == "1" and g.step3.mark.text() == "3"
    g.set_state(project=True, picture=True, cast=False, prompt=False)
    assert g.step1.mark.text() == "" and g.step1.mark.pixmap() is not None and g.step3.mark.text() == "3"
    assert g.b_picture.text() == "Choose a picture…" and g.b_cast.text() == "Add a person or place…"
    assert g.b_write.text() == "Go to the description"
    assert g.b_hide.text() == NextHint().b_close.text() == "Hide"


# ---------------------------------------------------------------- the frame stepper (UIL-30)
def test_frame_stepper_steps_by_one_frame_and_by_a_second():
    from video_chain.ui.widgets import FrameStepper
    host = QDialog()
    st = FrameStepper(host, fps=10)
    st.set_range(0, 99)
    seen = []
    st.valueChanged.connect(seen.append)
    st.setValue(5)
    st.b_fwd1.click()
    assert st.value() == 6
    st.b_back1.click()
    st.b_back1.click()
    assert st.value() == 4
    st.b_fwd_s.click()
    assert st.value() == 14
    st.b_back_s.click()
    st.b_back_s.click()
    assert st.value() == 0                           # (kept inside the range)
    assert seen[-1] == 0
    assert st.label.text() == "frame 1 / 100 · 0.00 s"
    for b in (st.b_back_s, st.b_back1, st.b_fwd1, st.b_fwd_s):
        assert b.accessibleName() and b.toolTip()


def test_the_frame_stepper_has_the_keys_of_every_dialog():
    from video_chain.ui.widgets import FrameStepper
    host = QDialog()
    st = FrameStepper(host, fps=10)
    st.set_range(0, 99)
    keys = {sc.key().toString() for sc in host.findChildren(__import__("PySide6.QtGui", fromlist=["x"]).QShortcut)}
    assert {"Left", "Right", "Shift+Left", "Shift+Right"} <= keys


# ---------------------------------------------------------------- text at the cursor (UIL-31)
def test_insert_spaced_adds_spaces_only_where_words_would_run_together():
    from video_chain.ui.text_edit import insert_spaced
    e = QPlainTextEdit()
    e.setPlainText("a dog and a cat")
    cur = e.textCursor()
    cur.setPosition(5)
    e.setTextCursor(cur)
    insert_spaced(e, "@Ann")
    assert e.toPlainText() == "a dog @Ann and a cat"
    e.setPlainText("hello.")
    cur = e.textCursor()
    cur.setPosition(5)
    e.setTextCursor(cur)
    insert_spaced(e, "world")
    assert e.toPlainText() == "hello world."
    e.undo()
    assert e.toPlainText() == "hello."


def test_replace_text_is_one_undo_step_and_is_shared():
    from video_chain.ui import prompt_refiner, text_edit
    assert prompt_refiner.replace_text is text_edit.replace_text
    e = QPlainTextEdit()
    e.setPlainText("one")
    text_edit.replace_text(e, "two")
    assert e.toPlainText() == "two"
    e.undo()
    assert e.toPlainText() == "one"


# ---------------------------------------------------------------- one word per thing (UX-14 .. 16, UMD-39)
REJECTED = [(r"\bstart pictures?\b", "start image"), (r"\bend pictures?\b", "end image"),
            (r"\bUse this part\b", "Use this range"), (r"H3 cuts to it", "the video model"),
            (r"H3's last-frame", "the video model's last-frame input"),
            (r"\b[Vv]ariations? of (this|the) clip\b", "variant")]


def _shown_strings(path: Path):
    """String constants that are not docstrings or comments (they can reach the user)."""
    tree = ast.parse(path.read_text())
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body and \
                isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
            docs.add(id(n.body[0].value))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            yield n.lineno, n.value


def test_the_ui_files_use_one_word_per_thing():
    bad = []
    for path in sorted(UI.glob("*.py")):
        if path.name in ("main_window.py", "widgets.py"):       # (main_window: see DEFERRED_main_window_ui.md)
            continue
        for line, s in _shown_strings(path):
            for pat, use in REJECTED:
                if re.search(pat, s):
                    bad.append(f"{path.name}:{line} {pat} -> {use}")
    assert not bad, "\n".join(bad)


def test_the_terms_are_written_down():
    from video_chain.ui import widgets
    assert widgets.TERMS["start image"] and "variant" in widgets.TERMS


# ---------------------------------------------------------------- fonts, colours (UX-22, UX-23, UX-24)
def test_painted_labels_do_not_shrink_below_the_themes_small_size():
    for name in ("graph", "keyframe_strip", "timeline", "results_gallery"):
        text = (UI / f"{name}.py").read_text()
        assert not re.search(r"max\(\s*[67]\.[05]\s*,|setPixelSize\(10\)", text), name


def test_error_red_comes_from_the_theme():
    assert "QColor(255, 60, 60)" not in (UI / "swap_dialog.py").read_text()


def test_a_variable_in_the_prompt_is_not_marked_by_colour_alone():
    from video_chain import variations as V
    from video_chain.ui import variations_panel as VP
    p = VP.VariationsPanel()
    p.set_templates("a {x} cat")
    p.on.setChecked(True)
    p.variable("x").values = [V.Value("red")]
    p._rebuild()
    from PySide6.QtWidgets import QPlainTextEdit
    e = QPlainTextEdit("a {x} cat")
    h = VP.PlaceholderHighlighter(e.document(), p)
    got = []
    h.setFormat = lambda start, n, f: got.append(f)
    h.highlightBlock("a {x} cat")
    assert got and got[0].underlineStyle() == got[0].UnderlineStyle.SingleUnderline
    assert "{x}" in got[0].toolTip()


def test_the_shortcuts_dialog_lists_the_table_and_only_closes():
    from video_chain.ui import shortcuts
    dlg = shortcuts.shortcuts_dialog()
    from PySide6.QtWidgets import QTextBrowser
    assert "Ctrl+Shift+N" in dlg.findChild(QTextBrowser).toPlainText()
    assert dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Close) is not None
