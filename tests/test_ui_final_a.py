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


# ------------------------------------------------------------------ MWC-8: worker events are checked once, at the boundary
import logging  # noqa: E402

import pytest  # noqa: E402


@pytest.mark.parametrize("ev", [
    {"e": "start", "nodes": 2},                                   # no size
    {"e": "start", "width": "wide", "height": 2, "frames": 3},    # a text where a number belongs
    {"e": "step", "id": "n1", "i": 1},                            # no n
    {"e": "node", "state": "generating"},
    {"e": "plan", "make": "n1"},
    ["not", "an", "event"],
])
def test_a_malformed_worker_event_is_ignored_with_one_warning(app, tmp_path, caplog, ev):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        before = w.status.text()
        with caplog.at_level(logging.WARNING):
            w._event(ev)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "worker event" in warnings[0].getMessage()
        assert w.status.text() == before and not w.live
    finally:
        w.close()


def test_a_good_worker_event_reaches_its_handler(app, tmp_path):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        w._event({"e": "start", "nodes": 2, "width": 640, "height": 360, "frames": 73})
        assert w.status.text() == "2 clips to make at 640×360, 73 frames each."
        w._event({"e": "somethingnew", "x": 1})                  # an unknown kind is not an error
    finally:
        w.close()


# ------------------------------------------------------------------ MWC-17: no test seams in production code
def test_production_code_does_not_read_the_test_environment():
    hits = [f.name for f in UI.glob("*.py") if "QT_QPA_PLATFORM" in f.read_text(encoding="utf-8")]
    assert hits == []


def test_dialog_policy_is_a_constructor_argument(app, tmp_path):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE, modal_dialogs=False, audio_player=False)
    v = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.modal_dialogs is False and w.audio_player is False
        assert v.modal_dialogs is True and v.audio_player is True
        assert w._track_dialog().want_player is False and v._track_dialog().want_player is True
    finally:
        w.close()
        v.close()


# ------------------------------------------------------------------ MWC-10 / MWC-11: one registry of node actions
def test_every_node_action_the_window_names_is_in_the_registry():
    from video_chain.ui.mw_nodes import NODE_ACTIONS
    names = set()
    for f in UI.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        names |= set(re.findall(r'node_action\(\s*"(\w+)"', src))
        names |= set(re.findall(r'\badd\(f?"[^"\n]*", "(\w+)"', src)) if f.name == "mw_nodes.py" else set()
    names -= {"x"}
    assert names and names <= set(NODE_ACTIONS), names - set(NODE_ACTIONS)
    assert {"continue", "branch", "variant", "variants", "delete", "freeze", "refine", "take"} <= set(NODE_ACTIONS)


def test_the_context_menu_has_one_dispatch(app):
    src = (UI / "mw_nodes.py").read_text(encoding="utf-8")
    ctx = src[src.index("    def _context("):]
    assert "elif chosen is" not in ctx and "chosen is play" not in ctx


# ------------------------------------------------------------------ MWC-13: which model ComfyUI holds is one object
def test_comfy_service_knows_which_model_is_loaded():
    from video_chain.ui.generator_state import ComfyModel, ComfyService
    c = ComfyService()
    assert not c.holds_clip() and not c.holds_image()
    c.model = ComfyModel.H3_HYPERFLOW
    assert c.holds_clip() and not c.holds_image()
    c.language_model_loaded()                       # a refined prompt: the language model is there now
    assert c.model is ComfyModel.LLM and c.holds_image()
    c.model = ComfyModel.QWEN
    c.language_model_loaded()                       # the image model stays the one to unload
    assert c.model is ComfyModel.QWEN
    c.forget()
    assert c.model is None


def test_window_keeps_the_model_in_its_comfy_service(app, tmp_path):
    from video_chain.ui.generator_state import ComfyModel
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        w.comfy.model = ComfyModel.QWEN
        assert w._comfy_model == "qwen"                # (the old name still reads it)
        w._comfy_model = "h3"
        assert w.comfy.model is ComfyModel.H3
    finally:
        w.close()


def test_refine_text_asks_the_dialog_directly():
    assert "hasattr(dlg" not in (UI / "mw_ends.py").read_text(encoding="utf-8")
    assert "isinstance(new, str)" in (UI / "mw_nodes.py").read_text(encoding="utf-8")      # a clip prompt is one text


# ------------------------------------------------------------------ MWC-29: a stale background answer is dropped (Latest)
def test_only_the_newest_lora_scan_counts(app, tmp_path, monkeypatch):
    import threading
    import time

    from video_chain import comfy as CMF
    from video_chain.ui.main_window import MainWindow
    first_go = threading.Event()
    calls = []

    def find(roots):
        n = len(calls)
        calls.append(n)
        if n == 0:
            first_go.wait(10)                        # the first scan answers last
            return [("/x/old.safetensors", "old")]
        return [("/x/new.safetensors", "new")]
    monkeypatch.setattr(CMF, "find_h3_loras", find)
    monkeypatch.setattr(CMF, "comfy_lora_roots", lambda d: [])
    monkeypatch.setattr(CMF, "find_qwen_loras", lambda roots: [])
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        calls.clear()
        w.scan_loras()
        w.scan_loras()
        deadline = time.time() + 10
        while time.time() < deadline and w.h3_loras != [("/x/new.safetensors", "new")]:
            app.processEvents()
            time.sleep(0.01)
        assert w.h3_loras == [("/x/new.safetensors", "new")]
        first_go.set()
        for _ in range(50):
            app.processEvents()
            time.sleep(0.01)
        assert w.h3_loras == [("/x/new.safetensors", "new")]
    finally:
        first_go.set()
        w.close()


# ------------------------------------------------------------------ MWA-3: the model folders are searched in the background
def test_gguf_models_are_found_off_the_gui_thread(app, tmp_path, monkeypatch):
    import threading
    import time
    from pathlib import Path

    from video_chain.ui import ui_logic as UL
    from video_chain.ui.main_window import MainWindow
    threads = []

    def fake(roots=None):
        threads.append(threading.current_thread())
        return [Path("/models/a-Q4.gguf"), Path("/models/b-Q4.gguf")]
    monkeypatch.setattr(UL, "find_gguf_models", fake)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        assert w.llm_model.count() == 0 or threads            # (nothing was searched on the way up)
        deadline = time.time() + 10
        while time.time() < deadline and w.llm_model.count() < 2:
            app.processEvents()
            time.sleep(0.01)
        assert [w.llm_model.itemText(i) for i in range(2)] == ["a-Q4.gguf", "b-Q4.gguf"]
        assert threads and all(t is not threading.main_thread() for t in threads)
        assert w.llm_on.isChecked()                            # a model was found and nothing was saved: on
    finally:
        w.close()


# ------------------------------------------------------------------ MWB-36: what a change said about CHANGED clips
from PySide6.QtWidgets import QMessageBox  # noqa: E402
from ui_helpers import crop_accepted, flush, image, new_project, run_and_wait, silent_player, win  # noqa: E402,F401


def test_remade_note_holds_what_a_change_said_about_changed_clips(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    win.steps.setValue(20)
    win._options_settled()
    assert "CHANGED" in win._remade_note and "2 made clips" in win._remade_note
    win.steps.setValue(28)                                  # set back: the clips are done again, none is CHANGED
    win._options_settled()
    assert win._remade_note == ""


# ------------------------------------------------------------------ MWB-11: refresh walks the selected path once
def test_refresh_asks_for_the_selected_path_once(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a", "b"))
    calls = []
    orig = win.selected_path
    import sys

    def counted():
        if sys._getframe(1).f_code.co_name == "refresh":          # (what refresh itself asks, not its helpers)
            calls.append(1)
        return orig()
    monkeypatch.setattr(win, "selected_path", counted)
    win.refresh()
    assert len(calls) == 1


# ------------------------------------------------------------------ MWB-39: one place for the speed modes
def test_speed_modes_are_named_in_one_place():
    from video_chain.ui import ui_logic as UL
    assert list(UL.SPEED_NAMES) == ["normal", "turbo", "hyperflow", "pika"]
    for mode in ("turbo", "hyperflow", "pika"):
        assert UL.speed_mode({mode: {"on": True}}) == mode
    assert "options.get(\"hyperflow\")" not in (UI / "mw_generator.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ MWB-27: the llama server's settings have names
def test_llama_config_names_what_decides_a_new_server(app, tmp_path):
    from video_chain.ui.mw_llama import LlamaConfig
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        w.llm_on.setChecked(True)
        w.llm_model.setEditText("/m/x.gguf")
        w.llm_port.setValue(8123)
        cfg = w.llama_config()
        assert isinstance(cfg, LlamaConfig) and cfg.model == "/m/x.gguf" and cfg.port == 8123
        srv = w.llama_server()
        assert srv is not None and srv is w.llama_server()          # the same settings: the same server
        w.llm_port.setValue(8124)
        srv2 = w.llama_server()
        assert srv2 is not srv and srv2 is not None and srv2.port == 8124
    finally:
        w.close()


# ------------------------------------------------------------------ MWB-31: text to speech busy flag with a session
def test_tts_session_marks_busy_and_never_leaves_it_stuck(app, tmp_path):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        with w._tts_session():                                   # handed to a worker: its answer frees it
            assert w._tts_busy
        assert w._tts_busy
        w._tts_busy = False
        with pytest.raises(ValueError):
            with w._tts_session():
                raise ValueError("never started")
        assert not w._tts_busy                                   # (it never started: not busy for ever)
    finally:
        w.close()


# ------------------------------------------------------------------ UX-18: a Clip menu and Help -> About
def _menu(win, title):
    return next(a.menu() for a in win.menuBar().actions() if a.text().replace("&", "") == title)


def test_clip_menu_offers_the_clip_actions(app, win, tmp_path, image, monkeypatch):
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert titles.index("Clip") == titles.index("View") + 1
    clip = _menu(win, "Clip")
    clip.aboutToShow.emit()
    assert not any(a.isEnabled() for a in clip.actions() if not a.isSeparator())      # no project: nothing to act on
    ids = new_project(win, tmp_path, image, ("a",))
    clip.aboutToShow.emit()
    by = {a.text().split("\t")[0]: a for a in clip.actions() if not a.isSeparator()}
    assert by["Continue from here"].isEnabled() and by["Branch (another continuation)"].isEnabled()
    assert "Ctrl+N" in by["Continue from here"].text()                                 # the key is shown, not bound twice
    got = []
    monkeypatch.setattr(win, "node_action", lambda action, nid: got.append((action, nid)))
    by["Branch (another continuation)"].trigger()
    assert got == [("branch", win.target_id())] and got[0][1] == ids[0]


def test_help_has_about(app, win, monkeypatch):
    help_ = _menu(win, "Help")
    texts = [a.text() for a in help_.actions()]
    assert "About Promptwood" in texts
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, text: shown.append((title, text)))
    next(a for a in help_.actions() if a.text() == "About Promptwood").trigger()
    assert shown and "Promptwood" in shown[0][1]


# ------------------------------------------------------------------ UX-30: Generate is greyed, with the reason, not a status line later
@pytest.mark.parametrize("flag, word", [("_tts_busy", "text to speech"), ("_copying", "copied"),
                                        ("_converts", "converted"), ("_joins", "joined")])
def test_generate_is_disabled_with_its_reason_while_something_else_works(app, win, tmp_path, image, flag, word):
    new_project(win, tmp_path, image, ("a",))
    flush(app, win)
    assert win.b_gen.isEnabled() and win.b_gen_all.isEnabled()
    old = getattr(win, flag)
    setattr(win, flag, True if isinstance(old, bool) else 1)
    try:
        win._update_buttons()
        assert not win.b_gen.isEnabled() and not win.b_gen_all.isEnabled()
        assert word in win.b_gen.toolTip() and word in win.b_gen_all.toolTip()
    finally:
        setattr(win, flag, old)
        win._update_buttons()
    assert win.b_gen.isEnabled() and "Make every clip" in win.b_gen.toolTip()


def test_the_context_menu_is_shown_greyed_while_generating(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    seen = {}
    monkeypatch.setattr(win, "_exec_menu", lambda m, pos: seen.update(
        {a.text(): a.isEnabled() for a in m.actions() if not a.isSeparator()}))
    monkeypatch.setattr(type(win), "running", property(lambda self: True))
    win._context(ids[0], win.pos())
    assert seen and not any(seen.values())


# ------------------------------------------------------------------ MWA-31: the progress bars and the preview slider have names
def test_bars_and_the_preview_slider_have_accessible_names(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    assert win.progress.accessibleName() == "Progress of the run"
    assert win.preview.slider.accessibleName() == "Position in the video"
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.bar.accessibleName() == "Progress of this clip"
