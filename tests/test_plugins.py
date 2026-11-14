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
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


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
        assert calls[0] == ("setup", "PluginHost") and [p.name for p in host.plugins if p.loaded] == ["plug_good"]
        bad = {p.name: p.error for p in host.plugins if p.error}
        assert set(bad) == {"plug_broken", "plug_bad_setup"} and "setup() failed: ValueError" in bad["plug_bad_setup"]
        assert "plug_broken, plug_bad_setup" in win.status.text() and "About plugins" in win.status.text()
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
        # its own settings, apart from the app's
        assert qs.value("plugins/good/seen", 0, type=int) == 1
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
    win = MainWindow(qs, command=FAKE)
    try:
        assert win.plugins.plugins == [] and "&Plugins" not in [a.text() for a in win.menuBar().actions()]
        assert win.plugins.about_text() == "No plugins are installed."
        win.plugins.emit("run_finished", 0)                          # (nobody listens: nothing happens)
    finally:
        win.close()
