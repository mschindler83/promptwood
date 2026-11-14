"""Final wave A (main window): menu paths in hints, named constants, no dead getattr."""
import re
from pathlib import Path

UI = Path(__file__).resolve().parent.parent / "src" / "video_chain" / "ui"


def test_hints_only_name_menu_entries_that_exist():
    """Settings has one entry, Preferences; every 'Settings → ...' hint must continue with it (UX-5)."""
    bad = []
    for f in UI.glob("*.py"):
        for m in re.finditer(r"Settings → ([^\n\"]{0,40})", f.read_text(encoding="utf-8")):
            if not m.group(1).startswith("Preferences") and "API Keys" not in m.group(1):
                bad.append(f"{f.name}: {m.group(0)}")
    assert not bad


def test_note_failure_uses_the_named_engine_down_code():
    src = (UI / "mw_inspector.py").read_text(encoding="utf-8")
    assert "code == 4" not in src


def test_prompt_card_does_not_guess_its_own_attributes():
    assert 'getattr(self, "ov"' not in (UI / "clip_cards.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ MWA-14: widget defaults come from T.DEFAULT_OPTIONS
import copy  # noqa: E402
import os  # noqa: E402

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from ui_helpers import FAKE, app, settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402


def _other_defaults():
    d = copy.deepcopy(T.DEFAULT_OPTIONS)
    d.update({"frames": 90, "steps": 30, "resolution": 768, "aspect": "9:16", "audio": False, "keep_repeated": True,
              "lora_triggers": False})
    d["turbo"] = {**d["turbo"], "steps": 7, "strength": 1.1}
    d["veda"] = {"on": False}
    return d


def _check_widgets(w):
    assert w.frames.currentData() == 90
    assert w.steps.value() == 30
    assert w.resolution.currentData() == 768
    assert w.aspect.currentData() == "9:16"
    assert not w.audio.isChecked()
    assert w.keep_rep.isChecked()
    assert not w.lora_triggers.isChecked()
    assert w.turbo_steps.value() == 7
    assert abs(w.turbo_strength.value() - 1.1) < 1e-6
    assert not w.veda.isChecked()


def test_widgets_start_with_the_default_options(app, tmp_path, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(T, "DEFAULT_OPTIONS", _other_defaults())
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        _check_widgets(w)
    finally:
        w.close()


def test_options_without_a_value_show_the_default_options(app, tmp_path, monkeypatch, image=None):
    from PIL import Image
    from video_chain.ui.main_window import MainWindow
    img = tmp_path / "s.jpg"
    Image.new("RGB", (64, 36)).save(img)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.create_project(tmp_path / "p", str(img))
        monkeypatch.setattr(T, "DEFAULT_OPTIONS", _other_defaults())
        w.tree.options.clear()
        w._options_to_widgets()
        _check_widgets(w)
    finally:
        w.close()


# ------------------------------------------------------------------ MWB-9: typing in a spin box does the heavy work once
def test_spin_box_ticks_are_settled_once_with_the_last_value(app, tmp_path, monkeypatch):
    from PIL import Image
    from video_chain.ui.main_window import MainWindow
    img = tmp_path / "s.jpg"
    Image.new("RGB", (64, 36)).save(img)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.create_project(tmp_path / "p", str(img))
        calls = []
        monkeypatch.setattr(w, "_refresh_card_choices", lambda: calls.append("cards"))
        monkeypatch.setattr(w, "_show_references", lambda *a, **k: calls.append("refs"))
        for v in (31, 32, 33, 34, 35):
            w.steps.setValue(v)
        assert w.tree.options["steps"] == 35            # the value is never lost
        assert calls == []                              # ... the heavy chain waits for a pause
        w._options_timer.timeout.emit()
        assert calls.count("cards") == 1 and calls.count("refs") == 1
        assert w.tree.options["steps"] == 35
        w.frames.setCurrentIndex(w.frames.currentIndex() + 1)      # a combo box is not typing: at once
        assert calls.count("cards") == 2
    finally:
        w.close()


def test_leaving_the_project_settles_a_pending_option_change(app, tmp_path):
    from PIL import Image
    from video_chain.ui.main_window import MainWindow
    img = tmp_path / "s.jpg"
    Image.new("RGB", (64, 36)).save(img)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.create_project(tmp_path / "p", str(img))
        w.steps.setValue(41)
        assert w._options_timer.isActive()
        assert w._may_leave_project()
        assert not w._options_timer.isActive()
    finally:
        w.close()


# ------------------------------------------------------------------ MWB-14 / MWB-15: problems with a kind
def test_cast_problems_are_classified_in_one_place():
    from video_chain.ui.ui_logic import CastProblem, ProblemKind, classify_cast_problem
    assert classify_cast_problem("clip #2 mentions @Anan, but there is no such character — did you mean @Anna?") == \
        CastProblem(ProblemKind.CAST_UNKNOWN, "Anan", "Anna")
    assert classify_cast_problem("clip #2 mentions @Bob, but there is no such character") == \
        CastProblem(ProblemKind.CAST_UNKNOWN, "Bob", "")
    assert classify_cast_problem("@Anna (clip #2) has no picture yet") == CastProblem(ProblemKind.CAST_NO_PICTURE, "Anna", "")
    assert classify_cast_problem("@Anna (clip #2) has no face picture or voice yet").kind is ProblemKind.CAST_NO_PICTURE
    assert classify_cast_problem("something else").kind is ProblemKind.CAST


def test_clip_problems_carry_a_kind_and_make_the_card_items(app, tmp_path):
    from PIL import Image
    from video_chain.ui.main_window import MainWindow
    from video_chain.ui.ui_logic import Problem, ProblemKind
    img = tmp_path / "s.jpg"
    Image.new("RGB", (64, 36)).save(img)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.create_project(tmp_path / "p", str(img))
        w.tree.options["turbo"] = {**w.tree.options["turbo"], "on": True, "lora": None}
        ps = w.clip_problems(w.tree.selected)
        assert ps and isinstance(ps[0], Problem) and ps[0].kind is ProblemKind.TURBO
        item = ps[0].item()
        assert item["action"] == "turbo" and item["button"] == "Show" and item["text"] == ps[0].text
    finally:
        w.close()


def test_a_check_that_fails_is_said_not_swallowed(app, tmp_path, monkeypatch, caplog):
    from PIL import Image
    from video_chain.ui.main_window import MainWindow
    img = tmp_path / "s.jpg"
    Image.new("RGB", (64, 36)).save(img)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.create_project(tmp_path / "p", str(img))
        monkeypatch.setattr(T, "guide_problem", lambda *a, **k: 1 / 0)
        ps = w.clip_problems(w.tree.selected)
        assert any("could not be checked" in p.text for p in ps)      # the card says it, not only the log
    finally:
        w.close()
