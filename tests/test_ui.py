"""GUI tests: offscreen, own QSettings file, the fake backend instead of ai-toolkit."""
import os
import sys
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from video_chain import chain  # noqa: E402
from video_chain import tree as T  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_backend.py")]


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from video_chain.ui.theme import ThemeManager
    ThemeManager.instance().apply(a)
    return a


def settings(tmp_path, name="s.ini"):
    from PySide6.QtCore import QSettings
    return QSettings(str(tmp_path / name), QSettings.IniFormat)


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "start.jpg"
    Image.new("RGB", (192, 108), (200, 200, 200)).save(p)
    return p


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from video_chain.ui.main_window import MainWindow, Preview
    played = []
    monkeypatch.setattr(Preview, "open", lambda self, path, title, autoplay=True: played.append(Path(path)))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w = MainWindow(settings(tmp_path), command=FAKE)
    w.played = played
    w.show()
    yield w
    if w.running:
        w.stop()
        w.proc and w.proc.waitForFinished(10000)
    w.close()


def wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out")


def flush(app, win):
    """Run the debounced save/refresh now."""
    for t in (win._save_timer, win._refresh_timer):
        if t.isActive():
            t.stop()
            t.timeout.emit()
    app.processEvents()


def type_prompt(win, nid, text):
    card = next(c for c in win.cards if c.nid == nid)
    card.edit.setPlainText(text)


def new_project(win, tmp_path, image, prompts=("one", "two")):
    win.open_project(tmp_path / "proj")
    win.set_image(str(image))
    ids = [win.selected_id()]
    type_prompt(win, ids[0], prompts[0])
    for p in prompts[1:]:
        win.node_action("continue", win.selected_id())
        ids.append(win.selected_id())
        type_prompt(win, ids[-1], p)
    return ids


def run_and_wait(app, win, fn):
    fn()
    assert win.running
    wait(app, lambda: not win.running)


def test_new_project_and_tree_actions(app, win, tmp_path, image):
    win.open_project(tmp_path / "proj")
    assert list(win.tree.nodes) == ["n1"] and win.selected_id() == "n1"
    assert set(win.graph.items_by_id) == {"__start__", "n1"} and len(win.cards) == 1
    type_prompt(win, "n1", "first")
    win.node_action("continue", "n1")
    assert win.selected_id() == "n2" and win.tree.nodes["n2"].parent == "n1"
    assert [c.nid for c in win.cards] == ["n1", "n2"]
    type_prompt(win, "n2", "second")
    win.node_action("branch", "n2")                                    # another continuation of n1
    assert win.tree.nodes["n3"].parent == "n1" and win.tree.nodes["n3"].prompt == ""
    assert [c.nid for c in win.cards] == ["n1", "n3"]
    win.node_action("variant", "n2")                                   # same prompt, other seed
    n4 = win.tree.nodes["n4"]
    assert n4.parent == "n1" and n4.prompt == "second" and n4.seed != win.tree.nodes["n2"].seed
    assert len(win.tree.leaves()) == 3 and "4 clips · 3 branches" in win.tree_info.text()
    win.select("n2")
    assert [c.nid for c in win.cards] == ["n1", "n2"]
    assert win.graph.items_by_id["n2"].info.selected and win.graph.items_by_id["n1"].info.on_path
    assert not win.graph.items_by_id["n4"].info.on_path
    win.node_action("delete", "n1")                                    # removes the whole tree below
    assert win.tree.nodes == {} and win.selected_id() == "__start__"
    win.node_action("continue", "__start__")
    assert win.tree.nodes["n5"].parent is None


def test_generate_path_then_branch_reuses(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("one", "two", "three"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: win.played)                                      # the joined path is played
    assert win.status.text().startswith("Done: 3 clips made. Path joined: 0:02.0")
    assert [win.state(i) for i in ids] == ["done"] * 3
    assert win.played[-1] == tmp_path / "proj" / "paths" / f"{ids[-1]}.mp4"
    assert chain.probe(win.played[-1])["frames"] == 17 + 16 + 16
    # branch after clip 1: only the new clip is made
    win.node_action("branch", ids[1])
    b = win.selected_id()
    type_prompt(win, b, "two, other way")
    flush(app, win)
    assert win.state(b) == "new" and win.state(ids[0]) == "done"
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: "Path joined" in win.status.text())
    assert win.status.text().startswith("Done: 1 clip made") and win.state(b) == "done"
    assert "generating" in win.log.toPlainText() and win.log.toPlainText().count("generating") == 1
    # editing clip 2 changes only clip 2 and 3 of the first branch
    win.select(ids[1])
    type_prompt(win, ids[1], "two, edited")
    flush(app, win)
    assert [win.state(i) for i in ids] == ["done", "changed", "changed"] and win.state(b) == "done"
    assert not win.b_play_path.isEnabled()


def test_generate_all(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    win.node_action("branch", ids[1])
    type_prompt(win, win.selected_id(), "b2")
    flush(app, win)
    run_and_wait(app, win, win.generate_all)
    assert all(win.state(n) == "done" for n in win.tree.nodes)
    assert win.status.text().startswith("Done: 3 clips made")


def test_settings_change_marks_everything(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    win.steps.setValue(20)
    flush(app, win)
    assert [win.state(i) for i in ids] == ["changed", "changed"]
    win.steps.setValue(28)
    flush(app, win)
    assert [win.state(i) for i in ids] == ["done", "done"]


def test_project_is_restored(app, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w = MainWindow(settings(tmp_path, "r.ini"), command=FAKE)
    ids = new_project(w, tmp_path, image, ("x", "y"))
    w.node_action("branch", ids[1])
    w.steps.setValue(20)
    w.close()
    w2 = MainWindow(settings(tmp_path, "r.ini"), command=FAKE)
    assert w2.project == (tmp_path / "proj").resolve() and len(w2.tree.nodes) == 3
    assert w2.selected_id() == "n3" and w2.steps.value() == 20
    assert [n.prompt for n in w2.tree.path_to("n2")] == ["x", "y"]
    w2.close()


def test_new_project_takes_the_last_settings(app, win, tmp_path, image):
    win.open_project(tmp_path / "p1")
    win.frames.setCurrentIndex(5)
    win.save_settings()
    win.open_project(tmp_path / "p2")
    assert win.frames.currentData() == 124 and win.tree.options["frames"] == 124


def test_old_linear_chain_opens_as_tree(app, win, tmp_path, image):
    from fakes import FakeGenerator
    out = tmp_path / "old"
    s = chain.settings_dict(model="Comfy-Org/MiniMax-H3", lora=None, strength=1.0, frames=73, steps=28,
                            width=672, height=384, audio=True)
    chain.run_chain(FakeGenerator(frames=17), image, ["one", "two"], out, settings=s, log=lambda _m: None)
    win.open_project(out)
    assert [n.prompt for n in win.selected_path()] == ["one", "two"]
    assert [win.state(n.id) for n in win.selected_path()] == ["done", "done"]
    assert (out / T.TREE_FILE).exists()


def test_failure_and_stop(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b", "c"))
    monkeypatch.setenv("FAKE_FAIL", "1")
    run_and_wait(app, win, win.generate_path)
    assert win.status.text().startswith("Failed:") and "fake failure in clip 2" in win.status.text()
    assert [win.state(i) for i in ids] == ["done", "failed", "new"]
    assert win.tabs.currentWidget() is win.log
    monkeypatch.delenv("FAKE_FAIL")
    monkeypatch.setenv("FAKE_DELAY", "0.5")
    win.generate_path()
    wait(app, lambda: win.state(ids[1]) == "generating")
    win.stop()
    wait(app, lambda: not win.running, timeout=20)
    assert win.status.text().startswith("Stopped") and win.state(ids[1]) == "stopped"
    assert win.state(ids[0]) == "done" and win.b_gen.isEnabled()


def test_start_explains_what_is_missing(app, win, tmp_path, image):
    win.open_project(tmp_path / "proj")
    win.generate_path()
    assert not win.running and "start image" in win.status.text() and "prompt of clip #1" in win.status.text()


def test_step_progress_and_export(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    ids = new_project(win, tmp_path, image, ("only",))
    seen = []
    card = win.cards[0]
    orig = card.set_progress
    card.set_progress = lambda i, n: (seen.append((i, n)), orig(i, n))
    run_and_wait(app, win, win.generate_path)
    assert seen[-1] == (4, 4)
    target = tmp_path / "export.mp4"
    win.join_selected("export", target)
    wait(app, lambda: target.exists() and "Exported" in win.status.text())
    assert chain.probe(target)["frames"] == 17
