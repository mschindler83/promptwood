"""Plugins (plugins.py, ui/plugin_host.py): found by entry point or VIDEO_CHAIN_PLUGINS, never able to stop the app,
and what one can add to the window. The plugins here are made up in the test's folder."""
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from video_chain.ui.generator_state import Job  # noqa: E402
import waiting as W
from video_chain import plugins

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_backend.py")]

GOOD = '''
calls = []

def setup(app):
    from PySide6.QtWidgets import QLabel
    calls.append(("setup", type(app).__name__))
    app.add_menu_action("Count the clips", lambda: app.status(f"{len(app.tree.nodes)} clips"), shortcut="Ctrl+Alt+K")
    app.add_menu_action("Break", lambda: 1 / 0)
    app.add_clip_action("Mark", lambda nid: calls.append(("mark", nid)), when=lambda node: not node.is_video,
                        tip="mark this clip")
    app.add_clip_action("Never", lambda nid: calls.append(("never", nid)), when=lambda node: False)
    app.add_clip_action("Bad condition", lambda nid: None, when=lambda node: node.nothing_like_this)
    calls.append(("panel", app.add_panel("My panel", QLabel("hello"))))
    for event in ("project_opened", "clip_made", "run_finished"):
        app.on(event, lambda *a, event=event: calls.append((event, *a)))
    app.on("run_finished", lambda code: 1 / 0)
    s = app.settings("good")
    s.setValue("seen", int(s.value("seen", 0, type=int)) + 1)

def generator(gen, args):
    return ("wrapped", gen, args)
'''


@pytest.fixture
def made_up(tmp_path, monkeypatch):
    """Three plugins as modules in the test's folder: a good one, one that cannot be imported, one whose setup
    fails."""
    d = tmp_path / "plugin_modules"
    d.mkdir()
    (d / "plug_good.py").write_text(GOOD)
    (d / "plug_broken.py").write_text("raise RuntimeError('no such model')\n")
    (d / "plug_bad_setup.py").write_text("def setup(app):\n    app.add_panel('x', None, where='nowhere')\n"
                                         "def generator(gen, args):\n    raise ValueError('no')\n")
    monkeypatch.syspath_prepend(str(d))
    for name in ("plug_good", "plug_broken", "plug_bad_setup"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv(plugins.OFF, raising=False)
    # only the made-up plugins: none that happens to be installed next to the app
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda group: [])
    plugins.reset()
    return d


def test_plugins_are_found_and_one_that_fails_is_left_out(made_up, monkeypatch):
    assert plugins.discover() == []                                  # none installed, none named
    point = types.SimpleNamespace(name="good", value="plug_good:setup",
                                  dist=types.SimpleNamespace(name="my-private-features", version="1.2"))
    twice = types.SimpleNamespace(name="zz-again", value="plug_good", dist=None)
    monkeypatch.setattr(plugins.metadata, "entry_points", lambda group: [twice, point] if group == plugins.GROUP else [])
    monkeypatch.setenv(plugins.ENV, "plug_broken, plug_good,plug_bad_setup,")
    found = plugins.discover()
    assert [(p.name, p.target, p.source) for p in found] == [
        ("good", "plug_good", "package my-private-features 1.2"),     # an installed package's entry point
        ("plug_broken", "plug_broken", plugins.ENV), ("plug_bad_setup", "plug_bad_setup", plugins.ENV)]
    loaded = plugins.load()
    assert [p.loaded for p in loaded] == [True, False, True] and plugins.load() is loaded     # once per process
    assert "could not be imported: RuntimeError: no such model" in loaded[1].error and "Traceback" in loaded[1].trace
    # the generation process: a plugin may give another generator; one that fails is skipped
    said = []
    gen = plugins.wrap_generator("comfy", "args", log=said.append)
    assert gen == ("wrapped", "comfy", "args")
    assert said == ["Plugin plug_bad_setup: generator() failed: ValueError: no"]
    # switched off: nothing is found, nothing runs
    monkeypatch.setenv(plugins.OFF, "1")
    plugins.reset()
    assert plugins.discover() == [] and plugins.load() == [] and plugins.wrap_generator("comfy", "args") == "comfy"


def wait(app, cond, timeout=60):
    W.wait(app, cond, timeout, raising=True)


def test_what_a_plugin_adds_to_the_window(made_up, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    app = QApplication.instance() or QApplication([])
    from video_chain.ui.main_window import MainWindow, Preview
    monkeypatch.setattr(Preview, "open", lambda self, path, title, autoplay=True: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setenv(plugins.ENV, "plug_good,plug_broken,plug_bad_setup")
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    for k, v in (("viewMode", "advanced"), ("comfyDir", str(tmp_path / "no-comfyui")), ("comfyAutoStart", False),
                 ("thumbCacheDir", str(tmp_path / "thumbs")), ("loraTriggersOnline", False)):
        qs.setValue(k, v)
    win = MainWindow(qs, command=FAKE)                               # (two broken plugins: the window still opens)
    try:
        import plug_good
        calls = plug_good.calls
        host = win.plugins
        assert calls[0] == ("setup", "PluginHost") and [p.name for p in host.plugins if not host.error(p)] == [
            "plug_good"]
        bad = {p.name: host.error(p) for p in host.plugins if host.error(p)}
        assert set(bad) == {"plug_broken", "plug_bad_setup"} and "setup() failed: ValueError" in bad["plug_bad_setup"]
        wait(app, lambda: "plug_broken, plug_bad_setup" in win.status.text())      # (said once the window is up)
        assert "About plugins" in win.status.text()
        about = host.about_text()
        assert "plug_good  (VIDEO_CHAIN_PLUGINS, module plug_good)\n    in use" in about
        assert "not in use: could not be imported: RuntimeError: no such model" in about
        # the Plugins menu and the panel
        menus = [a.text() for a in win.menuBar().actions()]
        assert menus[-1] == "&Plugins"
        entries = {a.text(): a for a in host.menu().actions() if not a.isSeparator()}
        assert list(entries) == ["Count the clips", "Break", "About plugins…"]
        assert entries["Count the clips"].shortcut().toString() == "Ctrl+Alt+K"
        index = dict(calls)["panel"]
        assert win.left_tabs.tabText(index) == "My panel" and isinstance(win.left_tabs.widget(index), QLabel)
        with pytest.raises(ValueError):
            host.on("no_such_event", print)
        # a project: the plugin hears it; its menu action works on the open project; a failing action only says so
        image = tmp_path / "start.png"
        Image.new("RGB", (192, 108), (200, 200, 200)).save(image)
        win.open_project(tmp_path / "proj")
        assert calls[-1] == ("project_opened", tmp_path / "proj")
        win.set_image(str(image))
        nid = win.selected_id()
        win.tree.nodes[nid].prompt = "she waves"
        win._write_tree()
        entries["Count the clips"].trigger()
        assert win.status.text() == "1 clips" and host.tree is win.tree and host.project == win.project
        win.tree.nodes[nid].prompt = "she waves twice"                # a plugin changed the project: saved and shown
        host.changed()
        from video_chain import tree as T
        assert T.load(win.project).nodes[nid].prompt == "she waves twice"
        assert win.graph.items_by_id[nid].info.prompt == "she waves twice"
        entries["Break"].trigger()
        assert "“Break” failed: ZeroDivisionError" in win.status.text()
        # the clip's right-click menu: the plugin's action for this clip (not the one its condition rules out)
        shown = {}

        def choose(menu, pos):
            shown["texts"] = [a.text() for a in menu.actions()]
            return next(a for a in menu.actions() if a.text() == "Mark")
        monkeypatch.setattr(win, "_exec_menu", choose)
        win._context(nid, win.mapToGlobal(win.rect().center()))
        wait(app, lambda: ("mark", nid) in calls)
        assert "Mark" in shown["texts"] and "Never" not in shown["texts"] and "Bad condition" not in shown["texts"]
        # a run: each made clip and the end are heard; the listener that fails is switched off, the app goes on
        win.generate_path()
        wait(app, lambda: any(c[0] == "run_finished" for c in calls))
        assert ("clip_made", nid) in calls and ("run_finished", 0) in calls and not win.running
        assert "run_finished listener failed and was switched off (plug_good)" in win.status.text()
        assert len(host._heard["run_finished"]) == 1
        good = next(p for p in host.plugins if p.name == "plug_good")     # still in use; the failure is listed
        assert good.loaded and not good.problems and host.state(good).problems == [
            "the action “Break” failed: ZeroDivisionError: division by zero",
            "the condition of the clip action “Bad condition” failed: AttributeError: 'Node' object has no "
            "attribute 'nothing_like_this'",
            "its run_finished listener failed: ZeroDivisionError: division by zero"]
        assert "its run_finished listener failed" in host.about_text()
        # its own settings, apart from the app's
        assert qs.value("plugins/good/seen", 0, type=int) == 1
        # the next start opens the last project by itself: the window comes up and the plugin hears it too
        calls.clear()
        qs.setValue("project", str(win.project))
        qs.sync()
        again = MainWindow(qs, command=FAKE)
        try:
            assert again.project == win.project and ("project_opened", win.project) in calls
        finally:
            again.close()
    finally:
        win.close()


def test_without_plugins_the_window_has_no_plugins_menu(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    app = QApplication.instance() or QApplication([])
    from video_chain.ui.main_window import MainWindow
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    for k, v in (("viewMode", "advanced"), ("comfyDir", str(tmp_path / "no-comfyui")), ("comfyAutoStart", False),
                 ("thumbCacheDir", str(tmp_path / "thumbs")), ("loraTriggersOnline", False)):
        qs.setValue(k, v)
    (tmp_path / "last").mkdir()
    qs.setValue("project", str(tmp_path / "last"))                   # (a last project is opened at the start)
    win = MainWindow(qs, command=FAKE)
    try:
        assert win.project == tmp_path / "last"
        assert win.plugins.plugins == [] and "&Plugins" not in [a.text() for a in win.menuBar().actions()]
        assert win.plugins.about_text() == "No plugins are installed."
        win.plugins.emit("run_finished", 0)                          # (nobody listens: nothing happens)
    finally:
        win.close()


# ---- what a plugin that fails may leave behind, and what the host tells about it
def window(folder, monkeypatch, names, project=None):
    """A main window (offscreen, its own settings file in folder) with the made-up plugins `names`."""
    from PySide6.QtCore import QSettings
    QApplication.instance() or QApplication([])
    from video_chain.ui.main_window import MainWindow, Preview
    monkeypatch.setattr(Preview, "open", lambda self, path, title, autoplay=True: None)
    monkeypatch.setenv(plugins.ENV, names)
    folder.mkdir(exist_ok=True)
    qs = QSettings(str(folder / "s.ini"), QSettings.IniFormat)
    for k, v in (("viewMode", "advanced"), ("comfyDir", str(folder / "no-comfyui")), ("comfyAutoStart", False),
                 ("thumbCacheDir", str(folder / "thumbs")), ("loraTriggersOnline", False)):
        qs.setValue(k, v)
    if project is not None:
        qs.setValue("project", str(project))
    return MainWindow(qs, command=FAKE)


def plugin(made_up, monkeypatch, name, text):
    (made_up / f"{name}.py").write_text(text)
    monkeypatch.delitem(sys.modules, name, raising=False)


def with_clip(win, tmp_path):
    image = tmp_path / "start.png"
    Image.new("RGB", (192, 108), (200, 200, 200)).save(image)
    win.open_project(tmp_path / "proj")
    win.set_image(str(image))
    return win.selected_id()


HALF = '''
def setup(app):
    from PySide6.QtWidgets import QLabel
    app.add_menu_action("Half", lambda: None)
    app.add_clip_action("Half clip", lambda nid: None)
    app.add_panel("Half panel", QLabel("x"))
    app.on("run_finished", lambda code: None)
    raise RuntimeError("half way")
'''


def test_a_failing_setup_leaves_nothing_behind(made_up, tmp_path, monkeypatch):
    plugin(made_up, monkeypatch, "plug_half", HALF)
    win = window(tmp_path, monkeypatch, "plug_half")
    try:
        host = win.plugins
        assert "setup() failed: RuntimeError: half way" in host.error(host.plugins[0])
        assert [a.text() for a in host.menu().actions() if not a.isSeparator()] == ["About plugins…"]
        assert host._clip_actions == [] and host._heard["run_finished"] == []
        assert all(win.left_tabs.tabText(i) != "Half panel" for i in range(win.left_tabs.count()))
    finally:
        win.close()


LATER = '''
import sys

def setup(app):
    app.add_menu_action("Quit now", lambda: sys.exit(3))
    app.add_menu_action("Fail", lambda: 1 / 0)
    app.add_clip_action("Odd", lambda nid: None, when=lambda node: node.nothing_like_this)
    app.on("run_finished", lambda code: sys.exit(4))
'''


def test_later_failures_are_listed_and_the_plugin_stays_in_use(made_up, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMenu
    plugin(made_up, monkeypatch, "plug_later", LATER)
    win = window(tmp_path, monkeypatch, "plug_later")
    try:
        host = win.plugins
        p = host.plugins[0]
        nid = with_clip(win, tmp_path)
        entries = {a.text(): a for a in host.menu().actions() if not a.isSeparator()}
        entries["Fail"].trigger()
        entries["Quit now"].trigger()                                # (sys.exit in an action does not end the app)
        assert "“Quit now” failed: SystemExit: 3" in win.status.text()
        for _ in range(2):                                           # (a menu built twice: the failure is noted once)
            host.context_actions(QMenu(), nid)
        host.emit("run_finished", 0)
        assert p.loaded and not host.error(p)
        assert host.state(p).problems == [
            "the action “Fail” failed: ZeroDivisionError: division by zero",
            "the action “Quit now” failed: SystemExit: 3",
            "the condition of the clip action “Odd” failed: AttributeError: 'Node' object has no attribute "
            "'nothing_like_this'",
            "its run_finished listener failed: SystemExit: 4"]
        assert "in use\n    the action “Fail” failed" in host.about_text()
        assert p.problems == []                                      # (the plugin object is shared: not touched)
    finally:
        win.close()


def test_setup_that_exits_and_a_generator_that_exits_do_not_stop_anything(made_up, tmp_path, monkeypatch):
    plugin(made_up, monkeypatch, "plug_exit", "import sys\ndef setup(app):\n    sys.exit(2)\n"
           "def generator(gen, args):\n    sys.exit(5)\n")
    win = window(tmp_path, monkeypatch, "plug_exit")
    try:
        assert "setup() failed: SystemExit: 2" in win.plugins.error(win.plugins.plugins[0])
        said = []                                                    # (setup() never runs in the generation process:
        assert plugins.wrap_generator("comfy", "args", log=said.append) == "comfy"     # its generator() is asked
        assert said == ["Plugin plug_exit: generator() failed: SystemExit: 5"]         # all the same)
    finally:
        win.close()


def test_a_module_that_breaks_when_asked_for_a_name_does_not_stop_the_start(made_up, tmp_path, monkeypatch):
    plugin(made_up, monkeypatch, "plug_getattr", "def __getattr__(name):\n    raise RuntimeError('no ' + name)\n")
    win = window(tmp_path, monkeypatch, "plug_getattr")
    try:
        assert "setup() failed: RuntimeError: no setup" in win.plugins.error(win.plugins.plugins[0])
        assert plugins.wrap_generator("comfy", "args", log=lambda m: None) == "comfy"
    finally:
        win.close()


def test_plugins_that_cannot_be_loaded_at_all_leave_the_app_running(made_up, tmp_path, monkeypatch):
    def boom(again=False):
        raise RuntimeError("no metadata")
    monkeypatch.setattr(plugins, "load", boom)
    win = window(tmp_path, monkeypatch, "plug_good")
    try:
        assert win.plugins.plugins == [] and win.plugins.about_text() == "No plugins are installed."
        assert plugins.wrap_generator("comfy", "args") == "comfy"
    finally:
        win.close()


def test_what_failed_is_told_after_the_last_project_is_open(made_up, tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    (tmp_path / "last").mkdir()
    win = window(tmp_path, monkeypatch, "plug_broken", project=tmp_path / "last")
    try:
        assert win.project == tmp_path / "last"
        wait(app, lambda: "Plugin not in use: plug_broken" in win.status.text())
    finally:
        win.close()


def test_events_come_after_the_app_and_clips_of_an_export_are_not_new(made_up, tmp_path, monkeypatch):
    plugin(made_up, monkeypatch, "plug_events", "seen = []\n\ndef setup(app):\n"
           "    app.on('run_event', lambda ev: seen.append(('run_event', app.window._clip_note)))\n"
           "    app.on('clip_made', lambda nid: seen.append(('clip_made', nid)))\n")
    win = window(tmp_path, monkeypatch, "plug_events")
    try:
        import plug_events
        win._event({"e": "note", "text": "step one"})
        assert plug_events.seen == [("run_event", "step one")]       # (the app had handled the event)
        host = win.plugins
        host.emit("run_event", {"e": "node", "id": "n1", "state": "done"})
        win._job = Job(["n2"], {"width": 640}, sent=True)     # an export at another size
        host.emit("run_event", {"e": "node", "id": "n2", "state": "done"})
        win._job = None
        assert [x for x in plug_events.seen if x[0] == "clip_made"] == [("clip_made", "n1")]
    finally:
        win._job = None
        win.close()


def test_changed_saves_only_when_the_app_would(made_up, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)   # (no dialog may wait)
    win = window(tmp_path, monkeypatch, "plug_good")
    try:
        host = win.plugins
        nid = with_clip(win, tmp_path)
        refreshed = []
        monkeypatch.setattr(win, "refresh", lambda **k: refreshed.append(k))
        assert host.changed() is True and refreshed == [{"rebuild_path": True}]
        win._job = Job([nid], sent=True)  # a generation runs: the app does not save
        saved = []
        monkeypatch.setattr(win, "_write_tree", lambda: saved.append(1) or True)
        assert host.changed() is False and saved == [] and len(refreshed) == 1
        assert "while clips are being generated" in win.status.text()
        win._job = None
        monkeypatch.setattr(win, "_write_tree", lambda: False)       # (changed outside the window)
        assert host.changed() is False and len(refreshed) == 1
    finally:
        win._job = None
        win.close()


def test_every_window_keeps_its_own_notes(made_up, tmp_path, monkeypatch):
    plugin(made_up, monkeypatch, "plug_second", "n = []\n\ndef setup(app):\n    n.append(1)\n"
           "    if len(n) == 2:\n        raise ValueError('the second window')\n")
    first = window(tmp_path / "one", monkeypatch, "plug_second")
    second = None
    try:
        second = window(tmp_path / "two", monkeypatch, "plug_second")
        assert "not in use: setup() failed: ValueError: the second window" in second.plugins.about_text()
        assert "plug_second  (VIDEO_CHAIN_PLUGINS, module plug_second)\n    in use" in first.plugins.about_text()
        assert first.plugins.plugins[0] is second.plugins.plugins[0] and first.plugins.plugins[0].loaded
        assert not first.plugins.plugins[0].error
    finally:
        first.close()
        if second is not None:
            second.close()


def test_the_about_box_shows_what_a_plugin_says_as_plain_text(made_up, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    seen = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: seen.append(self.textFormat()))
    win = window(tmp_path, monkeypatch, "plug_broken")
    try:
        win.plugins.show_about()
        assert seen == [Qt.PlainText]
    finally:
        win.close()


def test_a_settings_name_is_a_plain_name(made_up, tmp_path, monkeypatch):
    win = window(tmp_path, monkeypatch, "plug_good")
    try:
        for bad in ("", "a/b", "a\\b"):
            with pytest.raises(ValueError):
                win.plugins.settings(bad)
        assert win.plugins.settings("fine").group() == "plugins/fine"
    finally:
        win.close()
