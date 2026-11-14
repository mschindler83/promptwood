"""main_window.py, first part: shortcuts, the dialogs, Preview, the clip cards, the guide (review findings MWA-*, UX-*)."""
import json
import os
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox  # noqa: E402

from video_chain import tree as T  # noqa: E402
from test_ui import FAKE, app, settings  # noqa: E402,F401  (the module's QApplication / own settings file)


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    w = MainWindow(settings(tmp_path), command=FAKE)
    yield w
    w.close()


def _shortcut(win, key):
    return [s for s in win.findChildren(QShortcut) if s.key() == QKeySequence(key)]


# ------------------------------------------------------------------ MWA-1 / UX-1
@pytest.mark.parametrize("key", ["Delete", "F"])
def test_plain_key_shortcuts_only_act_in_the_graph(win, key):
    found = _shortcut(win, key)
    assert found
    assert all(s.context() == Qt.WidgetWithChildrenShortcut for s in found)
    assert win.graph.focusPolicy() != Qt.NoFocus


# ------------------------------------------------------------------ MWA-27
def test_state_exists_before_the_ui_is_built(app, tmp_path, monkeypatch):
    from video_chain.ui import main_window as MW
    seen = {}
    orig_menu, orig_load = MW.MainWindow._build_menu, MW.MainWindow._load_settings

    def menu(self):
        seen["menu"] = (hasattr(self, "qs"), hasattr(self, "view"))
        return orig_menu(self)

    def load(self):
        seen["load"] = (hasattr(self, "_gpu_proc"), hasattr(self, "_gpu_timer"))
        return orig_load(self)
    monkeypatch.setattr(MW.MainWindow, "_build_menu", menu)
    monkeypatch.setattr(MW.MainWindow, "_load_settings", load)
    w = MW.MainWindow(settings(tmp_path), command=FAKE)
    w.close()
    assert seen["menu"] == (True, True)
    assert seen["load"] == (True, True)


# ------------------------------------------------------------------ MWA-2 / 21 / 22
def test_preview_says_when_the_file_is_missing(app, tmp_path):
    from video_chain.ui.main_window import Preview
    p = Preview()
    got = []
    p.failed.connect(got.append)
    p.open(tmp_path / "gone.mp4", "Clip #1")
    assert "not found" in p.name.text().lower()
    assert got and "gone.mp4" in got[0]
    assert p.stack.currentWidget() is p.placeholder


def test_preview_stop_resets_the_controls(app):
    from video_chain.ui.main_window import Preview
    p = Preview()
    assert p.video is None and p.audio is None
    p._ensure_player()
    p.b_play.setEnabled(True)
    p.slider.setEnabled(True)
    p.slider.setValue(5)
    p.time.setText("0:03.0 / 0:04.0")
    p.stop()
    assert not p.b_play.isEnabled() and not p.slider.isEnabled()
    assert p.time.text().startswith("0:00.0")
    assert p.slider.value() == 0


# ------------------------------------------------------------------ MWA-8 / 9
def _variants(loras):
    from video_chain.ui.main_window import VariantsDialog
    n = T.Node("n1", None, "a cat", overrides={"loras": loras})
    return VariantsDialog(None, n)


def test_variants_keep_seed_without_steps_says_why(app):
    d = _variants([{"path": "a.safetensors", "strength": 1.0}])
    d.advanced.setChecked(True)
    d.same_seed.setChecked(True)
    assert not d.keep_seed()
    assert "step" in d.preview.text().lower()
    d.steps[0].setValue(0.2)
    assert d.keep_seed()
    assert "step" not in d.preview.text().lower().split("strengths")[0]


# ------------------------------------------------------------------ MWA-10
def test_new_project_rejects_dot_names_and_files(app, tmp_path):
    from video_chain.ui.main_window import NewProjectDialog
    d = NewProjectDialog(None, str(tmp_path))
    d.name.setText("..")
    assert d.folder() is None
    d.name.setText(".")
    assert d.folder() is None
    (tmp_path / "afile").write_text("x")
    d.name.setText("afile")
    d._check()                                   # (used to raise NotADirectoryError)
    assert "file" in d.target.text()
    assert not d.buttons.button(QDialogButtonBox.Ok).isEnabled()


# ------------------------------------------------------------------ MWA-11
def test_export_target_keeps_an_upper_case_extension_and_needs_a_name(app, tmp_path):
    from video_chain.ui.main_window import ExportDialog
    d = ExportDialog(None, str(tmp_path / "x.mp4"), (832, 480), {})
    d.file.setText(str(tmp_path / "Film.MP4"))
    assert d.target().name == "Film.MP4"
    d.file.setText(str(tmp_path / "Film"))
    assert d.target().name == "Film.mp4"
    d.file.setText("")
    assert not d.ok.isEnabled()
    d.file.setText(str(tmp_path / "Film"))
    assert d.ok.isEnabled()


# ------------------------------------------------------------------ MWA-12
def test_export_ready_is_asked_once_per_size_and_change(app, tmp_path):
    from video_chain.ui.main_window import ExportDialog
    calls = []
    d = ExportDialog(None, str(tmp_path / "x.mp4"), (832, 480),
                     lambda size, how, st, dn=False: calls.append(size) or False)
    calls.clear()
    d._update()
    assert sorted(calls) == ["1080p", "2k"]


# ------------------------------------------------------------------ UX-3
def test_export_cloud_tip_names_the_advanced_view_in_simple(app, tmp_path):
    from video_chain.ui.main_window import ExportDialog
    d = ExportDialog(None, str(tmp_path / "x.mp4"), (832, 480), {}, cloud=True, simple=True)
    assert "Advanced view" in d.cloud.toolTip()
    d2 = ExportDialog(None, str(tmp_path / "x.mp4"), (832, 480), {}, cloud=True, simple=False)
    assert "Advanced view" not in d2.cloud.toolTip()


# ------------------------------------------------------------------ MWA-3
def test_find_gguf_models_lists_each_snapshot_folder_once(tmp_path, monkeypatch):
    from video_chain.ui import main_window as MW
    snap = tmp_path / "models--a--b" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for n in ("m-instruct-Q4.gguf", "m-instruct-Q8.gguf", "plain.gguf", "mmproj-m.gguf"):
        (snap / n).write_bytes(b"x")
    calls = []
    orig = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pat: calls.append(pat) or orig(self, pat))
    found = MW.find_gguf_models([tmp_path])
    assert sorted(f.name for f in found) == ["m-instruct-Q4.gguf", "m-instruct-Q8.gguf", "plain.gguf"]
    assert len(calls) <= 2        # the roots' glob and one look at the snapshot folder (not 3 per file)


# ------------------------------------------------------------------ MWA-19 / 32 / 17 / 18
def test_guide_combo_does_not_collect_odd_items(app):
    from video_chain.ui.main_window import PromptCard
    card = PromptCard(T.Node("n1", None, "x", overrides={"guide": 7}))
    card.set_choices(73, 28, [])
    card.ov["guide"] = 9
    card.set_choices(73, 28, [])
    odd = [card.guide.itemText(i) for i in range(card.guide.count()) if "frames" in card.guide.itemText(i)
           and card.guide.itemData(i) not in (None,)]
    assert sum(1 for t in odd if t.startswith("7 ")) == 0
    assert sum(1 for t in odd if t.startswith("9 ")) == 1


def test_card_does_not_restyle_when_nothing_changed(app):
    from video_chain.ui.main_window import PromptCard, VideoCard
    for card in (PromptCard(T.Node("n1", None, "x")),
                 VideoCard(T.Node("n2", None, "", kind=T.VIDEO, source={"file": "/x.mp4", "start": 0, "end": 1}))):
        card.set_state("done", None)
        n = []
        orig = card.setStyleSheet
        card.setStyleSheet = lambda s, orig=orig: (n.append(1), orig(s))
        card.set_state("done", None)
        assert not n
        card.set_state("generating", None)
        assert n


def test_key_menu_is_released_and_time_is_exact(app):
    import shiboken6
    from video_chain.ui.main_window import PromptCard
    card = PromptCard(T.Node("n1", None, "x"))
    card.set_keyframes(4.0, [], [], False)
    made, got = [], []
    card.keyframe.connect(lambda act, nid, d: got.append((act, d)))

    def pick(m):
        made.append(m)
        return m.actions()[0]
    card._exec_menu = pick
    card._add_key_menu(None)
    assert got == [("add", {"at": 2.0})]
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(made[0])


# ------------------------------------------------------------------ UX-31 / UX-19 / 20 / 21
def test_log_tab_stays_in_the_simple_view(app, tmp_path):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path, view="simple"), command=FAKE)
    try:
        assert w.tabs.isTabVisible(w.tabs.indexOf(w.log))
    finally:
        w.close()


def test_hide_guide_says_where_it_went(win):
    win.show_guide(False)
    assert "Getting started" in win.status.text()


def test_guide_wording_uses_the_controls_name():
    from video_chain.ui.guide import GuidePanel
    g = GuidePanel()
    assert "Aspect" in g.b_text_only.toolTip()
    assert "picture format" not in g.b_text_only.toolTip()


def test_guide_has_a_second_stage_after_the_first_clip(win, tmp_path):
    from video_chain.ui.guide import NextHint
    h = NextHint()
    assert any(w in h.text.text().lower() for w in ("lora", "refine", "export"))


# ------------------------------------------------------------------ MWA-26 / 25
def test_guide_choices_survive_a_path_spelled_differently(win, tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    win.project = p
    win.tree = None
    win._guide_mark("no_cast")
    win.project = tmp_path / "proj" / ".." / "proj"
    assert "no_cast" in win._guide_choices()


def test_guide_choices_stored_before_the_resolved_key_are_still_found(win, tmp_path):
    p = tmp_path / "proj" / ".." / "proj"
    (tmp_path / "proj").mkdir()
    win.qs.setValue("guideChoices", json.dumps({str(p): ["text_only"]}))     # (as stored by earlier versions)
    win.project, win.tree = p, None
    assert "text_only" in win._guide_choices()
    win._guide_mark("no_cast")                       # a new mark moves the old entry to the resolved key
    d = json.loads(win.qs.value("guideChoices", type=str))
    assert d == {str(p.resolve()): ["no_cast", "text_only"]}


# ------------------------------------------------------------------ MWA-7
def test_enter_in_an_api_key_field_saves_it(win, monkeypatch):
    saved = []
    for name, field in (("_save_civitai_key", "civitai_key"), ("_save_eleven_key", "eleven_key"),
                        ("_save_runpod_key", "runpod_key")):
        if hasattr(win, field):
            monkeypatch.setattr(type(win), name, lambda self, n=name: saved.append(n))
            getattr(win, field).returnPressed.emit()
    assert saved
