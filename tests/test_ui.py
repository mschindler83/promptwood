"""GUI tests: offscreen, own QSettings file, a fake worker instead of ComfyUI."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

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
    q = QSettings(str(tmp_path / name), QSettings.IniFormat)
    if q.value("comfyDir") is None:                  # never the real ComfyUI folder in tests
        q.setValue("comfyDir", str(tmp_path / "no-comfyui"))
    if q.value("thumbCacheDir") is None:             # nor the real thumbnail cache
        q.setValue("thumbCacheDir", str(tmp_path / "thumb-cache"))
    if q.value("comfyAutoStart") is None:            # never start the real ComfyUI from a test
        q.setValue("comfyAutoStart", False)
    if q.value("loraTriggersOnline") is None:        # never ask CivitAI from a test
        q.setValue("loraTriggersOnline", False)
    return q


@pytest.fixture(autouse=True)
def no_runpod_key(monkeypatch):
    """Never the real RunPod key from the system keyring in a test (a test that needs one sets its own)."""
    from video_chain import runpod as RP
    monkeypatch.setattr(RP, "api_key", lambda: "")


@pytest.fixture(autouse=True)
def silent_player(monkeypatch):
    """No test plays anything: the test clips have a 440 Hz tone, and offscreen only hides the picture, not the
    sound (windows made without the win fixture played their joined paths aloud)."""
    from PySide6.QtMultimedia import QMediaPlayer
    monkeypatch.setattr(QMediaPlayer, "play", lambda self: None)     # Preview.open itself still runs


@pytest.fixture(autouse=True)
def crop_accepted(monkeypatch):
    """The face crop window is taken as it opens (its default square); no modal window waits in a test."""
    from video_chain.ui.face_crop import FaceCropDialog
    monkeypatch.setattr(FaceCropDialog, "exec", lambda self: FaceCropDialog.Accepted)


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
    wait(app, lambda: not win._joins, timeout=30)           # a path still being joined blocks Generate
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
    # while the path plays, the clip it is at is marked in the graph and the timeline
    file, spans = win._played_path
    assert file == str(win.played[-1]) and [n for n, _a, _b in spans] == ids
    assert [(round(a * 24), round(b * 24)) for _n, a, b in spans] == [(0, 17), (17, 33), (33, 49)]
    marked = lambda: ([k for k, it in win.graph.items_by_id.items() if it.playing], win.timeline.lane.playing)  # noqa: E731
    assert marked() == ([], None)
    for secs, nid in ((0.0, ids[0]), (0.7, ids[0]), (17 / 24, ids[1]), (1.5, ids[2]), (99.0, ids[2])):
        win.preview.moved.emit(secs)
        assert marked() == ([nid], nid), secs
    win.refresh(rebuild_path=True)                                     # (the mark stays when the graph is redrawn)
    assert marked() == ([ids[2]], ids[2])
    win.preview.moved.emit(-1.0)                                       # stopped, or played to its end
    assert marked() == ([], None)
    win.preview.moved.emit(0.2)
    win.preview.opened.emit(str(tmp_path / "other.mp4"))               # something else is played: no mark
    win.preview.moved.emit(0.5)
    assert marked() == ([], None) and win._played_path is None
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
    win.frames.setCurrentIndex(win.frames.findData(124))
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
    win.unload()                                   # the loaded worker still has the failing generator
    wait(app, lambda: not win.worker_alive, timeout=15)
    monkeypatch.delenv("FAKE_FAIL")
    monkeypatch.setenv("FAKE_DELAY", "0.5")
    win.generate_path()
    wait(app, lambda: win.state(ids[1]) == "generating")
    win.stop()
    wait(app, lambda: not win.running, timeout=20)
    assert win.status.text().startswith("Stopped") and win.state(ids[1]) == "stopped"
    assert win.state(ids[0]) == "done" and win.b_gen.isEnabled()


def test_stopping_an_export_leaves_the_clips_as_they_are(app, win, tmp_path, image, monkeypatch):
    """An export works on upscaled copies only: stopped or failed in the middle of a clip, every clip of the path
    stays "done" (it was marked "stopped" and blocked Play path and Export path until the project was reopened),
    and the path can be exported again at once."""
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    assert [win.state(i) for i in ids] == ["done", "done"] and win.b_export.isEnabled()
    monkeypatch.setenv("FAKE_DELAY", "0.5")                           # (each fake upscale takes 5 s)
    win.unload()
    wait(app, lambda: not win.worker_alive, timeout=15)
    win.export(tmp_path / "x.mp4", "2k")
    wait(app, lambda: any(win.live.get(i) == "generating" for i in ids), timeout=30)
    win.stop()
    wait(app, lambda: not win.running, timeout=20)
    assert win.status.text().startswith("Stopped")
    assert [win.state(i) for i in ids] == ["done", "done"]
    assert win.b_export.isEnabled() and win.b_play_path.isEnabled()
    assert not (tmp_path / "x.mp4").exists()


def test_start_explains_what_is_missing(app, win, tmp_path, image):
    win.open_project(tmp_path / "proj")
    win.generate_path()
    assert not win.running and "start image" not in win.status.text() and "prompt of clip #1" in win.status.text()


def test_step_progress_and_export(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    ids = new_project(win, tmp_path, image, ("only",))
    seen = []
    card = win.cards[0]
    orig = card.set_progress
    card.set_progress = lambda i, n, text=None: (seen.append((i, n)), orig(i, n, text))
    run_and_wait(app, win, win.generate_path)
    assert seen[-1] == (4, 4)
    target = tmp_path / "export.mp4"
    win.join_selected("export", target)
    wait(app, lambda: target.exists() and "Exported" in win.status.text())
    assert chain.probe(target)["frames"] == 17


def test_clip_settings_per_node(app, win, tmp_path, image):
    lora = tmp_path / "nose.safetensors"
    lora.write_bytes(b"x")
    other = tmp_path / "smile.safetensors"
    other.write_bytes(b"y")
    ids = new_project(win, tmp_path, image, ("a", "b", "c"))
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.loras.items() == [] and card.loras.none.isVisibleTo(card)         # no LoRA by default
    win.add_node_lora(ids[1], str(lora))
    win.add_node_lora(ids[1], str(other))
    card = next(c for c in win.cards if c.nid == ids[1])
    assert [r[1].property("path") for r in card.loras.rows] == [str(lora.resolve()), str(other.resolve())]
    card.loras.rows[0][2].setValue(0.8)                                          # each LoRA its own strength
    card.loras.rows[1][2].setValue(0.4)
    card.length.setCurrentIndex(card.length.findData(124))
    card.steps.setValue(20)
    flush(app, win)
    assert win.tree.nodes[ids[1]].overrides == {"loras": [{"path": str(lora.resolve()), "strength": 0.8},
                                                          {"path": str(other.resolve()), "strength": 0.4}],
                                                "frames": 124, "steps": 20}
    assert win.badge(win.tree.nodes[ids[1]]) == "nose ×0.8 · smile ×0.4 · 5.2 s · 20 steps"
    assert win.graph.items_by_id[ids[1]].info.badge.startswith("glow")
    assert win.graph.items_by_id[ids[0]].info.badge == ""
    # time spans follow the clip's own length: clip 2 is 124 frames (minus the repeated one)
    assert win.cards[2].span.text().startswith(f"0:{(73 + 123) / 24:04.1f}")
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[1]].made["settings"]
    assert made["lora"] == str(lora.resolve()) and made["strength"] == 0.8 and made["frames"] == 124
    assert made["extra_loras"][0]["strength"] == 0.4
    assert [win.state(i) for i in ids] == ["done", "done", "done"]
    win.add_node_lora(ids[2], str(lora))                                # a LoRA on clip 3: made again
    flush(app, win)
    assert [win.state(i) for i in ids] == ["done", "done", "changed"]
    card3 = next(c for c in win.cards if c.nid == ids[2])
    card3.loras._remove(card3.loras.rows[0])                            # removed again: done again
    flush(app, win)
    assert win.tree.nodes[ids[2]].overrides == {} and win.state(ids[2]) == "done"
    # the chosen files are offered on every card
    card3 = next(c for c in win.cards if c.nid == ids[2])
    assert str(lora.resolve()) in [p for p, _t in card3.loras.known]


def test_missing_node_lora_stops_the_start(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    win.tree.nodes[ids[0]].overrides = {"loras": [{"path": str(tmp_path / "gone.safetensors"), "strength": 1}]}
    win.generate_path()
    assert not win.running and "LoRA file does not exist (gone.safetensors)" in win.status.text()


def test_launcher_that_cannot_start(app, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w = MainWindow(settings(tmp_path, "f.ini"), command=[str(tmp_path / "no-such-launcher")])
    new_project(w, tmp_path, image, ("a",))
    w.generate_path()
    wait(app, lambda: not w.running, timeout=10)
    assert w.status.text().startswith("Failed: could not start") and w.b_gen.isEnabled()
    assert not w.cards[0].edit.isReadOnly()
    w.close()


def test_damaged_project_does_not_break_the_window(app, win, tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / T.TREE_FILE).write_text("{ broken")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    assert win.open_project(bad) is False
    assert warned and "damaged" in warned[0] and (bad / T.TREE_FILE).read_text() == "{ broken"
    assert win.project is None or win.project != bad.resolve()


def test_outside_change_is_not_overwritten(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    flush(app, win)
    f = tmp_path / "proj" / T.TREE_FILE
    other = T.load(tmp_path / "proj")
    other.nodes[ids[0]].prompt = "changed elsewhere"
    time.sleep(0.01)
    T.save(tmp_path / "proj", other)                                  # e.g. a second window
    type_prompt(win, ids[0], "edited here")
    flush(app, win)
    assert T.load(tmp_path / "proj").nodes[ids[0]].prompt == "changed elsewhere"
    assert "changed outside this window" in win.status.text()
    assert f.exists()


def test_event_after_unterminated_text(app, win):
    seen = []
    win._event = seen.append
    win._line('some progress text @@ {"e": "done", "made": []}')
    assert seen == [{"e": "done", "made": []}] and "some progress text" in win.log.toPlainText()


def test_context_menu_action_runs_after_the_menu(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    offered = []

    def pick_delete(menu, _pos):
        offered.extend(a.text() for a in menu.actions() if a.text())
        return [a for a in menu.actions() if a.text().startswith("Delete")][0]
    monkeypatch.setattr(win, "_exec_menu", pick_delete)
    win._context(ids[1], win.graph.mapToGlobal(win.graph.rect().center()))
    assert offered[:4] == ["Continue from here", "Add a real video after this…", "Branch (another continuation)",
                           "Variant (same prompt, other seed)"]
    assert ids[1] in win.tree.nodes                                    # not inside the node's handler
    wait(app, lambda: ids[1] not in win.tree.nodes, timeout=5)


def loads(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_model_stays_loaded_between_runs(app, win, tmp_path, image, monkeypatch):
    log = tmp_path / "loads.jsonl"
    monkeypatch.setenv("FAKE_LOADS", str(log))
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert win.worker_alive and win.model_label.text() == "Model: loaded in ComfyUI · unloads in 15 min"
    assert win.b_unload.isVisibleTo(win)
    win.node_action("continue", ids[0])
    type_prompt(win, win.selected_id(), "b")
    flush(app, win)
    run_and_wait(app, win, win.generate_path)
    assert win.status.text().startswith("Done: 1 clip made") and len(loads(log)) == 1     # loaded once
    win.unload()
    wait(app, lambda: not win.worker_alive, timeout=15)
    assert win.model_label.text() == "Model: not loaded" and not win.b_unload.isVisibleTo(win)


def test_without_keep_loaded_the_worker_quits(app, win, tmp_path, image):
    win.keep_loaded.setChecked(False)
    new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win.worker_alive, timeout=15)
    assert win.status.text().startswith("Done: 1 clip made")


def test_one_worker_until_comfyui_settings_change(app, win, tmp_path, image, monkeypatch):
    """Sound, LoRAs and sizes are per clip in ComfyUI: no new worker for them; another ComfyUI address needs one."""
    log = tmp_path / "loads.jsonl"
    monkeypatch.setenv("FAKE_LOADS", str(log))
    lora = tmp_path / "x.safetensors"
    lora.write_bytes(b"x")
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    win.audio.setChecked(False)
    win.add_node_lora(ids[0], str(lora))
    flush(app, win)
    wait(app, lambda: not win._joins, timeout=30)
    run_and_wait(app, win, win.generate_path)
    assert win.state(ids[0]) == "done" and len(loads(log)) == 1
    win.comfy_url.setText("http://127.0.0.1:8199")
    wait(app, lambda: not win._joins, timeout=30)
    win.node_action("continue", ids[0])
    type_prompt(win, win.selected_id(), "b")
    flush(app, win)
    run_and_wait(app, win, win.generate_path)
    got = loads(log)
    assert len(got) == 2 and got[1]["url"] == "http://127.0.0.1:8199"


def test_comfyui_is_started_when_generating(app, win, tmp_path, image, monkeypatch):
    started = []
    monkeypatch.setattr(win, "start_comfy", lambda: started.append(1) or setattr(
        win, "_comfy_starting", time.monotonic()) or True)
    ids = new_project(win, tmp_path, image, ("a",))
    win.comfy_autostart.setChecked(True)
    win._comfy_state(False)
    win.generate_path()
    assert started == [1] and not win.running and win._waiting_comfy == [ids[0]]
    assert "generating begins as soon as it runs" in win.status.text()
    win.generate_path()                                       # pressed again while it starts: no second start
    assert started == [1]
    win._comfy_state(True)                                    # ComfyUI answers: the generation begins
    wait(app, lambda: win.running or win.state(ids[0]) == "done", timeout=10)
    wait(app, lambda: not win.running)
    assert win.state(ids[0]) == "done" and win._waiting_comfy is None


def test_worker_quits_after_the_idle_time(tmp_path):
    import subprocess
    p = subprocess.Popen([*FAKE, "--serve", "--events", "--idle-minutes", "0.01"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)      # stdin stays open: only the timeout ends it
    events, start = [], time.time()
    for line in p.stdout:
        if line.startswith("@@ "):
            events.append(json.loads(line[3:]))
        if events and events[-1]["e"] == "bye":
            break
    p.wait(timeout=10)
    p.stdin.close()
    assert [e["e"] for e in events] == ["loading", "loaded", "idle", "bye"]
    assert events[-1]["reason"] == "idle" and p.returncode == 0 and time.time() - start < 20


def test_unload_frees_comfyui(app, win, tmp_path, image, monkeypatch):
    frees = tmp_path / "frees.txt"
    monkeypatch.setenv("FAKE_FREES", str(frees))
    new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    win.unload()
    wait(app, lambda: not win.worker_alive, timeout=15)
    assert frees.read_text().split() == ["free"]


def test_lora_missing_in_comfyui(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setenv("FAKE_COMFY_MISSING", "own_lora")
    lora = tmp_path / "own_lora.safetensors"
    lora.write_bytes(b"x")
    ids = new_project(win, tmp_path, image, ("a",))
    win.add_node_lora(ids[0], str(lora))
    flush(app, win)
    run_and_wait(app, win, win.generate_path)
    assert "own_lora.safetensors is not in ComfyUI's LoRA folders" in win.status.text()
    assert win.state(ids[0]) == "new"


def test_comfyui_not_running(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setenv("FAKE_COMFY_DOWN", "1")
    new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert "ComfyUI is not running" in win.status.text() and win.comfy_label.text() == "ComfyUI: not running"
    assert win.b_start_comfy.isEnabled()


def write_lora(path, names):
    import numpy as np
    from safetensors.numpy import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({n: np.zeros((2, 2), np.float32) for n in names}, str(path))


H3_KEYS = ["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.0.mlp.fc1.lora_A.weight"]


def test_lora_lists_offer_the_h3_loras(app, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "loras").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    write_lora(comfy / "models" / "loras" / "MiniMax H3" / "smoke.safetensors", H3_KEYS)
    write_lora(comfy / "models" / "loras" / "wan" / "other.safetensors", ["blocks.0.self_attn.q.lora_A.weight"])
    q = settings(tmp_path, "l.ini")
    q.setValue("comfyDir", str(comfy))
    w = MainWindow(q, command=FAKE)
    ids = new_project(w, tmp_path, image, ("a",))
    wait(app, lambda: w.h3_loras, timeout=10)
    assert [n for _p, n in w.h3_loras] == ["MiniMax H3/smoke.safetensors"]          # the Wan LoRA is left out
    card = w.cards[0]
    from video_chain.ui.lora_stack import LoraStack
    offered = []
    monkeypatch.setattr(LoraStack, "pick", staticmethod(
        lambda parent, known, current, used, file_dir: offered.append([t for _p, t in known]) or known[0][0]))
    card.loras.b_add.click()                                        # '+ LoRA' opens the picker
    assert offered == [["smoke"]]
    flush(app, w)
    smoke = str(comfy / "models" / "loras" / "MiniMax H3" / "smoke.safetensors")
    assert w.tree.nodes[ids[0]].overrides == {"loras": [{"path": smoke, "strength": 1.0}]}
    w.close()


def test_identity_references_in_the_window(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    face = tmp_path / "face.png"
    Image.new("RGB", (80, 100), (190, 150, 120)).save(face)
    win.add_reference(str(face))                                               # no character yet: Person1, lead
    flush(app, win)
    assert "2 made clips are now CHANGED" in win.status.text()
    assert win.cast_list.count() == 1 and win.cast_list.item(0).text().startswith("@Person1")
    assert win.char_lead.isChecked() and win.refs.count() == 1 and win.refs.item(0).text() == "Person1 1"
    assert (tmp_path / "proj" / "refs" / "ref1.png").is_file()
    assert [win.state(i) for i in ids] == ["changed", "changed"]               # every clip gets the lead's face
    card = win.cards[0]
    assert card.faces.text() == "Faces: @Person1 (the lead; the prompt mentions nobody)"
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[0]].made["settings"]
    assert made["refs"][0]["file"] == "refs/ref1.png" and made["ref_size"] == "match"
    win.ref_size.setCurrentIndex(win.ref_size.findData("max"))
    flush(app, win)
    assert win.state(ids[0]) == "changed"
    wait(app, lambda: not win._joins, timeout=30)                              # the auto-play join of the run
    # a second character, mentioned in clip 2 only
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Ben", True))
    win._new_character()
    assert win._selected_character()["name"] == "Ben" and "@Ben added" in win.status.text()
    other = tmp_path / "ben.png"
    Image.new("RGB", (80, 100), (60, 50, 40)).save(other)
    win.add_reference(str(other))
    assert win.status.text().startswith("Face added to @Ben")
    assert T.character(win.tree, "Ben")["images"] == ["refs/ref2.png"] and not win.char_lead.isChecked()
    proj = tmp_path / "proj"
    assert Image.open(proj / "refs/ref2.png").size == (80, 80)           # no face found: the middle square
    assert Image.open(proj / "refs/ref2_full.png").size == (80, 100)     # the full picture is kept
    monkeypatch.setattr(win, "crop_face", lambda path, title, crop=None: (10, 20, 40))
    win.refs.setCurrentRow(0)
    win._recrop_reference()
    assert Image.open(proj / "refs/ref2.png").size == (40, 40) and "cropped" in win.status.text()
    monkeypatch.setattr(win, "crop_face", lambda path, title, crop=None: False)     # cancelled: nothing added
    assert not win.add_reference(str(other)) and len(T.character(win.tree, "Ben")["images"]) == 1
    monkeypatch.delattr(win, "crop_face")                                     # the window's own again
    # Create…: the Create image window, its face button for the selected character
    assert win.b_ref_create.isEnabled()
    dlg = win.create_reference(dialog=False)
    assert dlg.mode() == "create" and dlg.b_use_ref.text() == "Add as face of @Ben"
    made = tmp_path / "made.png"
    Image.new("RGB", (90, 90), (20, 90, 160)).save(made)
    monkeypatch.setattr(win, "crop_face", lambda path, title, crop=None: None)
    assert dlg.on_use("ref", str(made)).startswith("Added as a face of @Ben")
    assert len(T.character(win.tree, "Ben")["images"]) == 2
    monkeypatch.delattr(win, "crop_face")
    win.refs.setCurrentRow(1)
    win._remove_reference()                                                  # (one face again for the rest)
    assert len(T.character(win.tree, "Ben")["images"]) == 1
    dlg.reject()
    dlg.deleteLater()
    win.edits_closed()
    type_prompt(win, ids[1], "@Bem waves")
    flush(app, win)
    win.select(ids[1])
    win.generate_path()
    assert not win.running and "did you mean @Ben? (Cast)" in win.status.text()
    type_prompt(win, ids[1], "@Ben waves")
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.faces.text() == "Faces: @Ben" and "Ben (the person in <Picture 1>)" in card.edit.toolTip()
    # typing @B offers the names; the chosen one goes in whole
    card.edit.setPlainText("Then @B")
    card.edit.moveCursor(QTextCursor.End)
    card.mentions._insert("Ben")
    assert card.edit.toPlainText() == "Then @Ben "
    # rename: every @mention follows
    type_prompt(win, ids[1], "@Ben waves")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Tom", True))
    win._rename_character()
    assert win.tree.nodes[ids[1]].prompt == "@Tom waves" and T.character(win.tree, "Ben") is None
    # a voice for Tom: used where Tom says a line
    import subprocess
    speech = tmp_path / "tom.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=9",
                    str(speech)], check=True)
    assert not win.b_voice_play.isEnabled() and win.char_voice.text() == "Voice: none"
    assert win.set_character_voice(str(speech))
    assert win.status.text().startswith("Voice of @Tom set (7.5 s, the first 7.5 s)")
    assert "Text to speech clones it from 9.0 s." in win.status.text()
    # Voice…: the file is chosen, then the part of it (a video here: 40 s, the first 30 s chosen to begin with)
    from PySide6.QtWidgets import QDialog, QDialogButtonBox
    from video_chain.ui import main_window as M
    from video_chain.ui.audio_editor import AudioRangeDialog
    film = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=40", "-f", "lavfi",
                    "-i", "color=gray:s=64x48:d=40", "-shortest", str(film)], check=True)
    monkeypatch.setattr(M, "pick_image", lambda *a, **k: str(film))
    seen = {}

    def choose(dlg):
        ok = dlg.buttons.button(QDialogButtonBox.Ok)
        seen.update(title=dlg.windowTitle(), first=(dlg.start, dlg.end), name=dlg.name.isVisibleTo(dlg),
                    text=dlg.summary.text())
        dlg.set_range(12.0, 13.0)
        seen["short"] = ok.isEnabled()
        dlg.set_range(12.0, 24.5)
        seen["text2"] = dlg.summary.text()
        return QDialog.Accepted if ok.isEnabled() else QDialog.Rejected
    monkeypatch.setattr(AudioRangeDialog, "exec", choose)
    win._voice_from_file()
    assert seen["title"] == "Voice of @Tom: choose the part" and seen["first"] == (0.0, 30.0) and not seen["name"]
    assert "clones the voice from all of it" in seen["text"] and not seen["short"]
    assert "Voice 0:12.000 – 0:24.500: 12.50 s." in seen["text2"] and "first 7.5 s" in seen["text2"]
    assert win.char_voice.text() == "Voice: talk.mp4 (7.5 s; cloned from 12.5 s)"
    monkeypatch.setattr(AudioRangeDialog, "exec", lambda dlg: QDialog.Rejected)       # cancelled: nothing changes
    win._voice_from_file()
    assert win.char_voice.text() == "Voice: talk.mp4 (7.5 s; cloned from 12.5 s)"
    # Remove music: the chosen part is cleaned by the text to speech worker (a fake here) and used instead
    import shutil

    class Sep:
        calls, stopped = [], 0

        def problems(self, **_k):
            return []

        def isolate(self, audio, out):
            Sep.calls.append(round(chain.audio_seconds(Path(audio)), 1))
            shutil.copy(audio, out)
            return 1.0

        def stop(self):
            Sep.stopped += 1
            return True
    win._tts_local = Sep()

    def clean(dlg):
        ok = dlg.buttons.button(QDialogButtonBox.Ok)
        seen.update(can=dlg.b_clean.isEnabled(), before=dlg.cleaned, use=dlg.use_clean.isEnabled())
        dlg.set_range(3.0, 38.0)
        dlg.remove_music()
        seen["busy"] = (dlg.clean_busy, ok.isEnabled(), dlg.b_clean.isEnabled())
        dlg.done(0)                                          # (closing waits for it)
        seen["open"] = dlg.clean_busy
        wait(app, lambda: not dlg.clean_busy, timeout=30)
        seen.update(info=dlg.clean_info.text(), checked=dlg.use_clean.isChecked(), file=dlg.cleaned)
        dlg.set_range(4.0, 38.0)                             # another part: the cleaned sound is not of it
        seen.update(stale=dlg.cleaned, info2=dlg.clean_info.text())
        dlg.set_range(3.0, 20.0)
        dlg.remove_music()
        wait(app, lambda: not dlg.clean_busy, timeout=30)
        seen["exists"] = dlg.cleaned is not None and dlg.cleaned.is_file()
        seen["dlg"] = dlg
        return QDialog.Accepted
    monkeypatch.setattr(AudioRangeDialog, "exec", clean)
    win._voice_from_file()
    assert seen["can"] and seen["before"] is None and not seen["use"]
    assert seen["busy"] == (True, False, False) and seen["open"]
    assert "Music removed from 0:03.000 – 0:33.000" in seen["info"] and seen["checked"] and seen["file"] is not None
    assert seen["stale"] is None and "remove the music again" in seen["info2"] and seen["exists"]
    assert Sep.calls == [30.0, 17.0] and Sep.stopped == 1    # (30 s at most; the worker is stopped afterwards)
    assert win.char_voice.text() == "Voice: talk.mp4 (music removed) (7.5 s; cloned from 17.0 s)"
    assert seen.pop("dlg")._tmp is None and not win._tts_busy             # its files are gone
    # unticked: the sound as it is; without the package or model the button says why
    def as_it_is(dlg):
        dlg.set_range(3.0, 20.0)
        dlg.remove_music()
        wait(app, lambda: not dlg.clean_busy, timeout=30)
        dlg.use_clean.setChecked(False)
        return QDialog.Accepted
    monkeypatch.setattr(AudioRangeDialog, "exec", as_it_is)
    win._voice_from_file()
    assert win.char_voice.text() == "Voice: talk.mp4 (7.5 s; cloned from 17.0 s)"
    # From clip: the same window on the selected clip's video
    shutil.copy(film, win.project / "made.mp4")
    node = win.tree.nodes[ids[1]]
    was_made, was_sel = node.made, win.tree.selected
    node.made, win.tree.selected = {**(node.made or {}), "video": "made.mp4"}, ids[1]

    def from_clip(dlg):
        seen.update(file=Path(dlg.file).name, can=dlg.b_clean.isEnabled())
        dlg.set_range(1.0, 11.0)
        if seen["remove"]:
            dlg.remove_music()
            wait(app, lambda: not dlg.clean_busy, timeout=30)
        return QDialog.Accepted
    monkeypatch.setattr(AudioRangeDialog, "exec", from_clip)
    for seen["remove"], name in ((False, f"sound of clip #{ids[1][1:]}"),
                                 (True, f"sound of clip #{ids[1][1:]} (music removed)")):
        win._voice_from_selected()
        assert seen["file"] == "made.mp4" and seen["can"]
        assert win.char_voice.text() == f"Voice: {name} (7.5 s; cloned from 10.0 s)"
    assert Sep.calls[-1] == 10.0 and not win._tts_busy
    node.made, win.tree.selected = was_made, was_sel
    (win.project / "made.mp4").unlink()
    Sep.problems = lambda self, **_k: ["removing music needs the audio-separator package"]
    monkeypatch.setattr(AudioRangeDialog, "exec", lambda dlg: (seen.update(
        can=dlg.b_clean.isEnabled(), tip=dlg.b_clean.toolTip()), QDialog.Rejected)[1])
    win._voice_from_file()
    assert not seen["can"] and seen["tip"].startswith("Not set up: removing music needs")
    win._tts_local = None
    mute = tmp_path / "mute.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=64x48:d=3", str(mute)], check=True)
    monkeypatch.setattr(M, "pick_image", lambda *a, **k: str(mute))
    win._voice_from_file()
    assert "has no sound" in win.status.text()
    assert win.set_character_voice(str(speech))
    assert win.char_voice.text() == "Voice: tom.wav (7.5 s; cloned from 9.0 s)"
    assert "· voice" in win.cast_list.currentItem().text()
    type_prompt(win, ids[1], '@Tom waves and says "Hello there."')
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.faces.text() == "Faces: @Tom · Voices: @Tom\nLines: @Tom: \u201cHello there.\u201d" and "<d>[English] Hello there.</d>" in card.edit.toolTip()
    # the overview: who appears in which clip of the path
    strip = win.cast_strip
    rows = [strip.verticalHeaderItem(r).text() for r in range(strip.rowCount())]
    cols = [c.nid for c in win.cards]
    assert strip.isVisibleTo(win) and rows == ["@Person1", "@Tom"] and strip.columnCount() == len(cols)
    assert strip.item(rows.index("@Tom"), cols.index(ids[1])).text() == "●♪"
    assert strip.item(rows.index("@Person1"), cols.index(ids[0])).text() == "○"           # the lead, nobody named
    assert strip.item(rows.index("@Tom"), cols.index(ids[0])).text() == ""
    assert "speaks with its voice" in strip.item(rows.index("@Tom"), cols.index(ids[1])).toolTip()
    strip._clicked(cols.index(ids[1]))                                                   # shows that card
    win.dialogue_lang.setCurrentText("Ger")                                  # typing: nothing changes yet
    assert win.tree.options["identity"]["language"] == "English"
    win.dialogue_lang.setCurrentText("German")
    win.dialogue_lang.lineEdit().editingFinished.emit()
    assert win.tree.options["identity"]["language"] == "German"
    win._remove_voice()
    assert win.char_voice.text() == "Voice: none" and not (tmp_path / "proj" / "refs" / "voice1.wav").exists()
    # removing the lead's face: clips that mention nobody are made without a face
    win.cast_list.setCurrentRow(0)
    win.refs.setCurrentRow(0)
    win._remove_reference()
    flush(app, win)
    assert win.refs.count() == 0 and not (tmp_path / "proj" / "refs" / "ref1.png").exists()
    assert win.cards[0].faces.text() == "Faces: none (the prompt mentions nobody and there is no lead)"


def test_sound_references_in_the_window(app, win, tmp_path, image, monkeypatch):
    import subprocess
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    wav = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=4", str(wav)],
                   check=True)
    assert win.add_sound(str(wav))
    flush(app, win)
    assert win.sounds.count() == 1 and win.sounds.item(0).text().startswith("<Audio 1>   voice.wav   4.0 s")
    assert "<Audio 1> added" in win.status.text() and [win.state(i) for i in ids] == ["done", "done"]
    assert "4.0 of 15 s used, 0 clips use them" in win.sound_info.text() and win.b_sound_del.isEnabled()
    type_prompt(win, ids[1], "b, in the voice of <Audio 2>")
    flush(app, win)
    assert win.state(ids[1]) == "changed"
    win.generate_path()
    assert not win.running and "names <Audio 2>, but there is only 1 sound reference" in win.status.text()
    type_prompt(win, ids[1], "b, in the voice of <Audio 1>")
    flush(app, win)
    assert "1 clip uses them" in win.sound_info.text()
    win.tree.selected = ids[0]
    assert win.b_sound_clip.isEnabled()
    win._sound_from_selected()                          # the test clips are 0.7 s: too short, and it says so
    flush(app, win)
    assert win.sounds.count() == 1 and "needs at least 2 s" in win.status.text()
    win.select(ids[1])
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[1]].made["settings"]
    assert [a["file"] for a in made["audio_refs"]] == ["refs/audio1.wav"]
    wait(app, lambda: not win._joins, timeout=30)
    win.open_project(tmp_path / "proj")                                         # kept with the project
    assert win.sounds.count() == 1 and win.state(ids[1]) == "done"
    win.sounds.setCurrentRow(0)
    win._remove_sound()
    flush(app, win)
    assert win.sounds.count() == 0 and not (tmp_path / "proj" / "refs" / "audio1.wav").exists()
    assert win.state(ids[1]) == "changed" and win.state(ids[0]) == "done"       # only the clip that named it


def test_stacked_loras_in_the_window(app, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    comfy = tmp_path / "ComfyUI"
    (comfy / "main.py").parent.mkdir(parents=True, exist_ok=True)
    (comfy / "main.py").write_text("")
    for n in ("glow", "motion", "style"):
        write_lora(comfy / "models" / "loras" / "MiniMax H3" / f"{n}.safetensors", H3_KEYS)
    q = settings(tmp_path, "st.ini")
    q.setValue("comfyDir", str(comfy))
    w = MainWindow(q, command=FAKE)
    ids = new_project(w, tmp_path, image, ("a", "b"))
    wait(app, lambda: len(w.h3_loras) == 3, timeout=10)
    path = {Path(p).stem: p for p, _n in w.h3_loras}
    # clip 2: nose, motion and style stacked on the card, each with its strength
    card = next(c for c in w.cards if c.nid == ids[1])
    for name, st in (("glow", 1.0), ("motion", 0.5), ("style", 0.3)):
        card.loras._add_row(path[name], st)
    card.loras.changed.emit()
    flush(app, w)
    assert w.tree.nodes[ids[1]].overrides["loras"] == [{"path": path["glow"], "strength": 1.0},
                                                       {"path": path["motion"], "strength": 0.5},
                                                       {"path": path["style"], "strength": 0.3}]
    assert w.badge(w.tree.nodes[ids[1]]) == "nose ×1 · motion ×0.5 · style ×0.3"
    ch = T.Checker(w.project, w.tree)
    s2 = ch.settings(ids[1])
    assert s2["lora"] == path["glow"] and [e["lora"] for e in s2["extra_loras"]] == [path["motion"], path["style"]]
    assert ch.settings(ids[0])["lora"] is None                                      # clip 1: none
    run_and_wait(app, w, w.generate_path)
    made = T.load(w.project).nodes[ids[1]].made["settings"]
    assert [e["strength"] for e in made["extra_loras"]] == [0.5, 0.3]
    # removing a row takes the LoRA off again
    card = next(c for c in w.cards if c.nid == ids[1])
    card.loras._remove(card.loras.rows[2])
    flush(app, w)
    assert len(w.tree.nodes[ids[1]].overrides["loras"]) == 2 and w.state(ids[1]) == "changed"
    w.close()


def test_lora_names_are_short_unless_they_clash(app, win):
    win.h3_loras = [("/l/MiniMax H3/a.safetensors", "MiniMax H3/a.safetensors"),
                    ("/l/MiniMax H3/b.safetensors", "MiniMax H3/b.safetensors"),
                    ("/l/old/b.safetensors", "old/b.safetensors")]
    labels = [text for _p, text in win.known_loras()]
    assert labels == ["a", "b (MiniMax H3)", "b (old)"]


def test_clip_lengths_3_to_15_seconds(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    values = [win.frames.itemData(i) for i in range(win.frames.count())]
    assert values[0] == 73 and values[-1] == 362 and all(v % 17 == 5 for v in values) and 39 not in values
    card = win.cards[0]
    assert [card.length.itemData(i) for i in range(card.length.count())][1:] == values
    win.frames.setCurrentIndex(win.frames.findData(362))
    flush(app, win)
    assert win.tree.options["frames"] == 362 and win.cards[0].span.text().startswith("0:00.0 – 0:15.1")
    # a project saved with 39 frames keeps showing them (not silently another length)
    t = T.load(win.project)
    t.options["frames"] = 39
    t.nodes[ids[0]].overrides = {"frames": 56}
    T.save(win.project, t)
    win.open_project(win.project)
    assert win.frames.currentData() == 39 and "no longer offered" in win.frames.currentText()
    assert win.cards[0].length.currentData() == 56
    win.open_project(tmp_path / "other")
    assert win.frames.count() == len(values)                                # the extra entry is gone again


def test_turbo_in_the_window(app, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "loras").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    d = comfy / "models" / "loras" / "MiniMax H3"
    write_lora(d / "nose.safetensors", H3_KEYS)
    write_lora(d / "minimax_h3_turbo_v4_step600.safetensors", H3_KEYS)
    write_lora(d / "minimax_h3_turbo_v4_step600_ema.safetensors", H3_KEYS)
    q = settings(tmp_path, "tu.ini")
    q.setValue("comfyDir", str(comfy))
    w = MainWindow(q, command=FAKE)
    ids = new_project(w, tmp_path, image, ("a",))
    wait(app, lambda: w.turbo_loras, timeout=10)
    assert [Path(p).name for p, _n in w.h3_loras] == ["nose.safetensors"]          # turbo kept out of the lists
    assert [t for _p, t in w.cards[0].loras.known] == ["glow"]
    assert w.turbo_lora.currentText() == "minimax_h3_turbo_v4_step600_ema"         # the recommended one
    run_and_wait(app, w, w.generate_path)
    w.speed.setCurrentIndex(w.speed.findData("turbo"))
    flush(app, w)
    assert w.tree.options["turbo"]["on"] and w.tree.options["turbo"]["lora"].endswith("_ema.safetensors")
    w.clip_cfg.setValue(0.9)                                           # the turbo mode's CFG (below 1 too)
    flush(app, w)
    assert w.tree.options["turbo"]["cfg"] == 0.9 and "two passes" in w.cfg_note.text()
    w.speed.setCurrentIndex(w.speed.findData("normal"))                # another mode: its own CFG (1)
    flush(app, w)
    assert w.clip_cfg.value() == 1.0 and w.tree.options["turbo"]["cfg"] == 0.9
    w.clip_cfg.setValue(1.3)
    flush(app, w)
    assert w.tree.options["cfg"] == 1.3
    w.speed.setCurrentIndex(w.speed.findData("turbo"))                 # back: the turbo value again
    flush(app, w)
    assert w.clip_cfg.value() == 0.9
    assert not w.steps.isEnabled() and not w.cards[0].steps.isEnabled() and w.state(ids[0]) == "changed"
    wait(app, lambda: not w._joins, timeout=30)
    run_and_wait(app, w, w.generate_path)
    made = T.load(w.project).nodes[ids[0]].made["settings"]
    assert made["steps"] == 6 and made["turbo"]["lora"].endswith("_ema.safetensors")
    w.close()


def test_new_open_recent_save_close(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox

    from video_chain.ui.main_window import NewProjectDialog
    assert win.create_project(tmp_path / "projects" / "first", str(image))
    assert win.windowTitle() == "first — Promptwood" and (tmp_path / "projects" / "first" / T.START_FILE).is_file()
    type_prompt(win, win.selected_id(), "hello")
    win.save_project()
    assert "Saved first" in win.status.text()
    assert T.load(tmp_path / "projects" / "first").nodes[win.selected_id()].prompt == "hello"
    assert not win.create_project(tmp_path / "projects" / "first")            # already a project
    win.create_project(tmp_path / "projects" / "second")
    assert win.recent_projects()[:2] == [str((tmp_path / "projects" / "second").resolve()),
                                         str((tmp_path / "projects" / "first").resolve())]
    texts = [a.text() for a in win.recent_file_menu.actions()]
    assert texts[0].startswith("second") and texts[1].startswith("first")
    win.recent_file_menu.actions()[1].trigger()                                 # reopen 'first' from the list
    assert win.project.name == "first"
    assert T.load(win.project).nodes[win.selected_id()].prompt == "hello"
    # dialog: name + location -> folder, refuses existing projects
    dlg = NewProjectDialog(win, str(tmp_path / "projects"))
    dlg.name.setText("first")
    assert "already a project" in dlg.target.text() and not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dlg.name.setText("third")
    assert dlg.folder() == tmp_path / "projects" / "third" and dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dlg.deleteLater()
    # save as: a full copy, and the window continues in it
    win.copy_project_to(tmp_path / "projects" / "first-copy")
    wait(app, lambda: win.project and win.project.name == "first-copy", timeout=20)
    assert (win.project / T.START_FILE).is_file() and win.windowTitle().startswith("first-copy")
    assert T.load(win.project).nodes[win.selected_id()].prompt == "hello"
    win.close_project()
    assert win.project is None and win.windowTitle() == "Promptwood" and not win.act_save.isEnabled()
    assert win.graph.items_by_id == {}


def test_image_picker(app, tmp_path):
    from PySide6.QtCore import Qt

    from video_chain.ui.image_picker import ImagePickerDialog
    d = tmp_path / "pics"
    (d / "sub").mkdir(parents=True)
    Image.new("RGB", (800, 600), (200, 0, 0)).save(d / "b_big.jpg")
    Image.new("RGB", (64, 64), (0, 200, 0)).save(d / "a_small.png")
    Image.new("RGB", (64, 64), (0, 0, 200)).save(d / "sub" / "inner.webp")
    (d / "notes.txt").write_text("not an image")
    q = settings(tmp_path, "pick.ini")
    dlg = ImagePickerDialog(None, "Start image", str(d), q, cache_dir=tmp_path / "thumbs",
                            desktop_thumbs=tmp_path / "desktop-thumbs")
    names = [dlg.grid.item(i).text() for i in range(dlg.grid.count())]
    assert names == ["sub", "a_small.png", "b_big.jpg"]                         # folders first, no txt
    big = dlg.grid.item(2)
    wait(app, lambda: big.data(Qt.UserRole + 1) is not None, timeout=10)       # thumbnail made in the background
    assert big.data(Qt.UserRole + 1) == "320×240"                              # read at reduced size
    dlg.grid.setCurrentItem(big)
    assert "800×600 px" in dlg.info.text() and dlg.buttons.button(dlg.buttons.StandardButton.Open).isEnabled()
    dlg._activated(dlg.grid.item(0))                                            # into the subfolder
    assert dlg.folder == (d / "sub").resolve() and dlg.grid.item(0).text() == "inner.webp"
    dlg.go_up()
    assert dlg.folder == d.resolve()
    dlg.sort.setCurrentIndex(dlg.sort.findData("date"))
    dlg._activated(dlg.grid.item(next(i for i in range(dlg.grid.count()) if dlg.grid.item(i).text() == "a_small.png")))
    assert dlg.result() == dlg.DialogCode.Accepted and dlg.chosen() == str((d / "a_small.png").resolve())
    assert q.value("pickerDir") == str(d.resolve())                             # remembered for next time
    dlg2 = ImagePickerDialog(None, "again", None, q, cache_dir=tmp_path / "thumbs", desktop_thumbs=None)
    assert dlg2.folder == d.resolve()
    assert any(dlg2.places.item(i).data(Qt.UserRole) == str(d.resolve()) for i in range(dlg2.places.count()))
    dlg2.reject()


def test_create_variants(app, win, tmp_path, image):
    lora = tmp_path / "x.safetensors"
    lora.write_bytes(b"x")
    ids = new_project(win, tmp_path, image, ("a", "b"))
    win.tree.nodes[ids[1]].overrides = {"loras": [{"path": str(lora), "strength": 0.7}], "frames": 90}
    new = win.create_variants(ids[1], 3, generate=True)
    assert len(new) == 3 and win.running
    nodes = [win.tree.nodes[n] for n in new]
    assert all(n.parent == ids[0] and n.prompt == "b" for n in nodes)
    assert all(n.overrides == {"loras": [{"path": str(lora), "strength": 0.7}], "frames": 90} for n in nodes)
    seeds = {n.seed for n in nodes} | {win.tree.nodes[ids[1]].seed}
    assert len(seeds) == 4                                                    # every variant its own seed
    wait(app, lambda: not win.running)
    assert all(win.state(n) == "done" for n in new) and win.state(ids[0]) == "done"
    assert win.state(ids[1]) == "new"                                         # the original was not asked for
    assert len(win.tree.leaves()) == 4
    only = win.create_variants(ids[0], 2, generate=False)
    assert not win.running and all(win.state(n) == "new" for n in only)



def test_variants_with_stepped_lora_strengths(app, win, tmp_path, image):
    """Advanced variants: each LoRA's strength moves by its step per variant (within the range), optionally with
    the clip's own seed so only the strengths differ."""
    from video_chain.ui.main_window import VariantsDialog
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    ids = new_project(win, tmp_path, image, ("one",))
    node = win.tree.nodes[ids[0]]
    node.overrides = {"loras": [{"path": str(a), "strength": 0.7}, {"path": str(b), "strength": 9.8}]}
    dlg = VariantsDialog(win, node)
    assert not dlg.advanced.isChecked() and dlg.strength_steps() == [0.0, 0.0] and not dlg.keep_seed()
    dlg.advanced.setChecked(True)
    dlg.count.setValue(3)
    dlg.steps[0].setValue(-0.1)
    dlg.steps[1].setValue(0.15)
    dlg.same_seed.setChecked(True)
    assert dlg.strength_steps() == [-0.1, 0.15] and dlg.keep_seed()
    assert "#1: ×0.6, ×9.95" in dlg.preview.text() and "#3: ×0.4, ×10" in dlg.preview.text()   # capped at 10
    assert dlg.ok.isEnabled()
    dlg.steps[0].setValue(0.0)                                  # only b moves, and it is capped: #3 = #2
    assert "#3 repeats" in dlg.preview.text() and not dlg.ok.isEnabled()
    dlg.same_seed.setChecked(False)                             # other seeds: allowed (only a warning)
    assert dlg.ok.isEnabled()
    dlg.advanced.setChecked(False)
    assert dlg.ok.isEnabled() and dlg.preview.text() == ""
    dlg.done(0)
    empty = VariantsDialog(win, win.tree.add(None, "no loras"))
    assert not empty.advanced.isEnabled()
    empty.done(0)
    new = win.create_variants(ids[0], 3, generate=False, strength_steps=[-0.1, 0.15], keep_seed=True)
    got = [[e["strength"] for e in win.tree.nodes[n].overrides["loras"]] for n in new]
    assert got == [[0.6, 9.95], [0.5, 10.0], [0.4, 10.0]]
    assert all(win.tree.nodes[n].seed == node.seed for n in new)
    assert node.overrides["loras"][0]["strength"] == 0.7                     # the clip itself is unchanged
    plain = win.create_variants(ids[0], 1, generate=False, strength_steps=[0.0, 0.0])
    assert win.tree.nodes[plain[0]].overrides == node.overrides and win.tree.nodes[plain[0]].seed != node.seed


def test_keep_this_branch(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))                       # n1 -> n2
    variants = win.create_variants(ids[1], 2, generate=False)                 # n1 -> n2 | v1 | v2
    win.node_action("continue", variants[0])                                  # v1 -> v1a
    after_v1 = win.selected_id()
    win.node_action("branch", ids[0])                                         # another branch at the root's level
    other_root_level = win.selected_id()
    gone = win.keep_branch(ids[1], whole_path=False, confirm=False)           # keep n2, drop its siblings
    assert set(gone) == {variants[0], variants[1], after_v1}                   # with what follows them
    assert ids[1] in win.tree.nodes and other_root_level in win.tree.nodes     # other levels untouched
    assert win.selected_id() == ids[1]
    # whole path: also the branches next to the clips before
    win.node_action("branch", ids[0])
    root_sibling = win.selected_id()
    win.node_action("continue", ids[1])
    kept_child = win.selected_id()
    assert win.tree.others_than_path(ids[1])
    win.keep_branch(ids[1], whole_path=True, confirm=False)
    assert set(win.tree.nodes) == {ids[0], ids[1], kept_child} and root_sibling not in win.tree.nodes


def test_review_fixes_in_the_window(app, win, tmp_path, image, monkeypatch):
    # a new project does not take over the references of the last one
    new_project(win, tmp_path, image, ("a",))
    face = tmp_path / "face.png"
    Image.new("RGB", (40, 40), (1, 2, 3)).save(face)
    win.add_reference(str(face))
    win.save_settings()
    win.create_project(tmp_path / "fresh")
    assert T.cast_of(win.tree) == [] and win.refs.count() == 0 and win.cast_list.count() == 0
    # an auto-selected turbo LoRA is stored, so turbo really runs with it
    win.turbo_loras = [("/l/minimax_h3_turbo_v4_step600_ema.safetensors", "minimax_h3_turbo_v4_step600_ema.safetensors")]
    win.tree.options["turbo"] = {"on": True, "lora": None, "steps": 6, "strength": 1.0}
    win._fill_turbo(None)
    assert win.tree.options["turbo"]["lora"] == "/l/minimax_h3_turbo_v4_step600_ema.safetensors"
    # a failed save is shown, not swallowed
    monkeypatch.setattr(T, "save", lambda *a, **k: (_ for _ in ()).throw(OSError(5, "Input/output error")))
    assert win._write_tree() is False and "Could not save the project" in win.status.text()


def test_thumbnails_are_cached_on_disk_and_desktop_ones_used(app, tmp_path):
    import hashlib
    import urllib.parse

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    from video_chain.ui import image_picker as IP
    d = tmp_path / "share"
    d.mkdir()
    for k in range(5):
        Image.new("RGB", (900, 600), (k * 40, 0, 0)).save(d / f"img{k}.jpg")
    cache, desktop = tmp_path / "thumbs", tmp_path / "desk"
    q = settings(tmp_path, "c.ini")

    def thumbs_of(dlg):
        wait(app, lambda: all(dlg.grid.item(i).data(Qt.UserRole + 1) for i in range(dlg.grid.count())), timeout=10)
        return dict(dlg.stats)
    first = IP.ImagePickerDialog(None, "x", str(d), q, cache_dir=cache, desktop_thumbs=desktop)
    assert thumbs_of(first)["read"] == 5                                   # read once over the "network"
    first.reject()
    second = IP.ImagePickerDialog(None, "x", str(d), q, cache_dir=cache, desktop_thumbs=desktop)
    st = thumbs_of(second)
    assert st["read"] == 0 and st["disk"] == 5                             # then from the local disk
    second.reject()
    # a changed image is read again
    Image.new("RGB", (900, 600), (0, 0, 255)).save(d / "img0.jpg")
    os.utime(d / "img0.jpg", (time.time() + 5, time.time() + 5))
    third = IP.ImagePickerDialog(None, "x", str(d), q, cache_dir=cache, desktop_thumbs=desktop)
    st = thumbs_of(third)
    assert st["read"] == 1 and st["disk"] == 4
    third.reject()
    # a fresh image with a matching desktop thumbnail is not read at all
    new = d / "fresh.jpg"
    Image.new("RGB", (900, 600), (0, 255, 0)).save(new)
    uri = "file://" + urllib.parse.quote(str(new.resolve()))
    thumb = QImage(128, 85, QImage.Format_RGB32)
    thumb.fill(0x00ff00)
    thumb.setText("Thumb::URI", uri)
    thumb.setText("Thumb::MTime", str(int(new.stat().st_mtime)))
    (desktop / "large").mkdir(parents=True)
    thumb.save(str(desktop / "large" / (hashlib.md5(uri.encode()).hexdigest() + ".png")))
    fourth = IP.ImagePickerDialog(None, "x", str(d), q, cache_dir=cache, desktop_thumbs=desktop)
    st = thumbs_of(fourth)
    assert st["desktop"] == 1 and st["read"] == 0
    fourth.reject()
    # pruning keeps the newest
    IP.prune_cache(cache, keep=2)
    assert len([f for sub in cache.iterdir() for f in sub.iterdir()]) == 2


# ----------------------------------------------------------------------------- real videos

def make_real_video(path, secs=2.0):
    """A synthetic 'real' video: red, then blue; 30 fps, 16:9, with sound."""
    import subprocess
    half = secs / 2
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=red:s=320x180:r=30:d={half}",
                    "-f", "lavfi", "-i", f"color=blue:s=320x180:r=30:d={half}", "-f", "lavfi", "-i",
                    f"sine=f=500:d={secs}", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                    "-map", "2:a", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_video_first_then_generated_clip(app, win, tmp_path):
    from video_chain.ui.main_window import PromptCard, VideoCard
    src = make_real_video(tmp_path / "real.mp4")
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.5)
    assert v and win.live.get(v) == "converting" and not win.b_gen.isEnabled()
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done" and win.tree.nodes[v].made["frames"] == 36
    assert isinstance(win.cards[0], VideoCard) and "real.mp4" in win.cards[0].file.text()
    assert "follows the first video" in win.size_info.text()
    assert win.graph.items_by_id[v].info.kind == "video"
    # no start image: the reference button takes the selected video's first frame
    assert not win.b_ref_start.isEnabled() and "no start image" in win.b_ref_start.toolTip()
    win.select(v)
    assert win.b_ref_clip.isEnabled() and f"first frame of video #{v[1:]}" in win.b_ref_clip.toolTip()
    win.b_ref_clip.click()
    win.reference_from_node(v, "last")
    assert win.refs.count() == 2 and win.char_lead.isChecked()                 # both: faces of the lead
    assert "last frame of" in win.status.text()
    win.char_lead.setChecked(False)                                            # clips without a face again
    win.node_action("continue", v)
    a = win.selected_id()
    assert isinstance(win.cards[1], PromptCard)
    type_prompt(win, a, "after the real video")
    run_and_wait(app, win, win.generate_path)
    assert win.state(a) == "done", win.status.text()
    wait(app, lambda: win.played)
    assert chain.probe(win.played[-1])["frames"] == 36 + 16
    # a later start, same last frame: converted again, but the clip after it stays (it starts from that frame)
    win.set_video_range(v, str(src), 0.5, 1.5)
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done" and win.state(a) == "done"
    assert win.tree.nodes[v].made["frames"] == 24
    # another last frame (still red): the clip after it must be made again
    win.set_video_range(v, str(src), 0.0, 0.8)
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done" and win.state(a) == "changed"


def test_clip_into_video_and_variants(app, win, tmp_path, image):
    src = make_real_video(tmp_path / "real.mp4")
    ids = new_project(win, tmp_path, image, ("walks toward the door",))
    a = ids[0]
    v = win.add_video(a, str(src), 1.0, 2.0)
    wait(app, lambda: not win._converts)
    assert win.add_video(a, str(src), 0.0, 1.0) is None                 # one video per generated clip
    assert "already leads into" in win.status.text()
    assert "ends at video" in win.cards[0].span.text()
    run_and_wait(app, win, win.generate_path)                             # path start -> a -> v
    assert win.state(a) == "done" and win.state(v) == "done", win.status.text()
    assert win.tree.nodes[a].made["end_key"] == win.tree.nodes[v].made["key"]
    wait(app, lambda: win.played)
    assert chain.probe(win.played[-1])["frames"] == 16 + 24              # the clip's repeated last frame is cut
    new = win.create_variants(a, 2, generate=False)
    for nid in new:
        vc = win.tree.video_child(nid)
        assert vc is not None and vc.id != v and vc.source == win.tree.nodes[v].source
        assert win.state(vc.id) == "done" and vc.made["video"] != win.tree.nodes[v].made["video"]
    win.node_action("variant", v)                                         # not for real videos
    assert "Variants are for generated clips" in win.status.text()


def test_end_image_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    card = win.cards[0]
    assert card.end_text.text() == "free" and card.b_end.isEnabled() and not card.b_end_clear.isEnabled()
    end = tmp_path / "end.png"
    Image.new("RGB", (40, 30), (255, 255, 0)).save(end)
    assert win.set_end_image(ids[0], str(end))
    card = win.cards[0]
    assert card.end_text.text() == "own end image" and card.b_end_clear.isEnabled()
    assert "end image" in win.badge(win.tree.nodes[ids[0]])
    run_and_wait(app, win, win.generate_path)
    assert win.state(ids[0]) == "done", win.status.text()
    new = win.create_variants(ids[0], 1, generate=False)                    # a variant gets its own copy
    assert win.tree.nodes[new[0]].end_image not in (None, win.tree.nodes[ids[0]].end_image)
    win.node_action("end_clear", ids[0])
    flush(app, win)
    assert win.tree.nodes[ids[0]].end_image is None and win.state(ids[0]) == "changed"
    assert win.cards[0].end_text.text() == "free"


def test_undo_changes_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("walks in", "sits down"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    type_prompt(win, ids[0], "walks in slowly")
    flush(app, win)
    card = win.cards[0]
    assert win.state(ids[0]) == "changed" and win.state(ids[1]) == "changed"
    assert card.b_revert.isVisibleTo(card) and "prompt changed" in card.step.text()
    assert not win.cards[1].b_revert.isVisibleTo(win.cards[1])      # only changed because of the clip before
    card.b_revert.click()
    flush(app, win)
    assert win.cards[0].edit.toPlainText() == "walks in"
    assert [win.state(i) for i in ids] == ["done", "done"] and "done again" in win.status.text()
    assert T.load(tmp_path / "proj").nodes[ids[0]].prompt == "walks in"


def test_clearer_controls(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    card = win.cards[0]
    assert card.tag.text() == "NEW" and "Not made yet" in card.tag.toolTip()
    assert not win.b_play_path.isEnabled() and "2 clips on this path are new or changed" in win.b_play_path.toolTip()
    assert card.settings_toggle.text() == "Own settings: project defaults"
    card.settings_toggle.click()
    assert card.own.isVisibleTo(card) and PromptCard_open(win)
    assert win.cards[1].own.isVisibleTo(win.cards[1])                    # every card opens together
    card.length.setCurrentIndex(card.length.findData(90))
    flush(app, win)
    assert win.cards[0].settings_toggle.text() == "Own settings: 3.8 s" and win.cards[0].own.isVisibleTo(win.cards[0])
    win.cards[0].settings_toggle.click()
    assert not win.cards[1].own.isVisibleTo(win.cards[1]) and not PromptCard_open(win)
    assert not win._clips_form.isRowVisible(win.turbo_lora)              # turbo settings only in turbo mode
    win.speed.setCurrentIndex(win.speed.findData("turbo"))
    assert win._clips_form.isRowVisible(win.turbo_lora)
    win.speed.setCurrentIndex(win.speed.findData("normal"))
    run_and_wait(app, win, win.generate_path)
    assert win.b_play_path.toolTip() == "Join the clips of the selected path and play them"


def PromptCard_open(win):
    from video_chain.ui.main_window import PromptCard
    return PromptCard.settings_open


def test_motion_guide_choice_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("x", "y"))
    card = win.cards[1]
    card.guide.setCurrentIndex(card.guide.findData(22))
    assert win.tree.nodes[ids[1]].overrides["guide"] == 22
    assert "carry-over 22" in win.badge(win.tree.nodes[ids[1]])
    flush(app, win)
    assert win.cards[1].span.text().startswith("0:03.0 – 0:05.2")          # 22 guide frames left out


def test_video_range_dialog(app, tmp_path):
    from video_chain.ui.video_editor import VideoRangeDialog
    src = make_real_video(tmp_path / "real.mp4", secs=2.0)
    from PySide6.QtGui import QImage
    got = []
    dlg = VideoRangeDialog(None, str(src), canvas=(672, 384), player=False,
                           on_reference=lambda p: got.append(QImage(p).size().toTuple()) or True)
    assert dlg.b_ref.isEnabled() and dlg.reference_here()
    assert got == [(dlg.info["width"], dlg.info["height"])]  # the whole frame, not cut to the chain's size
    assert "was added (see the Cast" in dlg.summary.text()
    assert dlg.info["fps"] == 30 and abs(dlg.end - 2.0) < 0.05 and dlg.frames() == 48
    dlg.set_range(0.5, 1.0)
    assert dlg.frames() == 12 and "12 frames" in dlg.summary.text() and "672×384" in dlg.summary.text()
    dlg.set_range(1.0, 0.2)                                   # never shorter than one frame
    assert dlg.end - dlg.start >= 1 / 24 - 1e-6
    dlg.end_spin.setValue(0.1)                                # moving the end below the start moves the start
    assert dlg.end == 0.1 and dlg.start < 0.1
    dlg.seek(1.5)
    dlg.step_frames(3)
    assert abs(dlg.position() - 1.6) < 1e-6
    dlg.set_range(0.0, 1.0)
    dlg._update_frames()
    wait(app, lambda: dlg.first_tile.has_image() and dlg.last_tile.has_image(), timeout=30)
    from PySide6.QtWidgets import QDialogButtonBox
    assert dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dlg.close()


def test_video_range_dialog_rotation(app, tmp_path):
    """A quarter turn in the range window: the size it reports, the chain's size worked out again, the face
    picture and the first / last frame are turned; R and Shift+R turn it back."""
    from video_chain.ui.video_editor import VideoRangeDialog
    src = make_real_video(tmp_path / "real.mp4", secs=2.0)              # 320x180
    asked, got = [], []

    def canvas_for(file, rot):
        asked.append(rot)
        w, h = chain.rotated_size(320, 180, rot)
        return chain.canvas_size(w, h, 512)
    dlg = VideoRangeDialog(None, str(src), canvas=(672, 384), player=False, canvas_for=canvas_for,
                           on_reference=lambda p: got.append(QImage(p).size().toTuple()) or True)
    from PySide6.QtGui import QImage
    assert dlg.rotate == 0 and dlg.b_rot_right.isEnabled() and dlg.info_label.text().startswith("320×180 ·")
    dlg.b_rot_right.click()
    assert dlg.rotate == 90 and asked[-1] == 90 and dlg.canvas == (384, 672)
    assert dlg.info_label.text().startswith("180×320 · turned 90° right")
    assert "turned 90° right, cut to 384×672" in dlg.summary.text()
    assert dlg.reference_here() and got == [(180, 320)]
    dlg._update_frames()
    wait(app, lambda: dlg.first_tile.has_image() and dlg.last_tile.has_image(), timeout=30)
    dlg.b_rot_left.click()
    dlg.b_rot_left.click()
    assert dlg.rotate == 270 and dlg.canvas == (384, 672)
    dlg.set_rotation(360)
    assert dlg.rotate == 0 and dlg.canvas == (672, 384) and "turned" not in dlg.summary.text()
    dlg.close()
    again = VideoRangeDialog(None, str(src), 0.0, 1.0, canvas=(384, 672), player=False, rotate=180)
    assert again.rotate == 180 and "turned 180° right" in again.info_label.text()
    again.close()


def test_video_view_rotation(app, tmp_path):
    """The preview turns on top of what the player gives; a phone video's own rotation flag is applied as ffmpeg
    applies it (red left / blue right, flagged 90° counter-clockwise: blue on top)."""
    import subprocess

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtMultimedia import QMediaPlayer, QVideoFrame, QVideoSink

    from video_chain.ui.widgets import VideoView, frame_image
    cpu = VideoView(gpu=False)
    cpu._raw = QImage(320, 180, QImage.Format_RGB888)
    cpu.set_rotation(90)
    assert cpu.image().size().toTuple() == (180, 320)
    cpu.set_rotation(180)
    assert cpu.image().size().toTuple() == (320, 180)
    with pytest.raises(RuntimeError):
        VideoView(gpu=True).set_rotation(90)
    split, phone = tmp_path / "split.mp4", tmp_path / "phone.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=32x36:r=30:d=0.5", "-f", "lavfi",
                    "-i", "color=blue:s=32x36:r=30:d=0.5", "-filter_complex", "[0:v][1:v]hstack", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(split)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(split), "-c", "copy",
                    str(phone)], check=True)
    sink, player, got = QVideoSink(), QMediaPlayer(), []
    player.setVideoOutput(sink)
    sink.videoFrameChanged.connect(lambda f: f.isValid() and not got and got.append(QVideoFrame(f)))
    player.setSource(QUrl.fromLocalFile(str(phone)))
    player.pause()                                  # (the first frame, without playing: the tests never play)
    wait(app, lambda: got, timeout=15)
    player.stop()
    img = frame_image(got[0])
    assert img.size().toTuple() == (36, 64) == got[0].toImage().size().toTuple()
    assert img.pixelColor(18, 3).blue() > 200 and img.pixelColor(18, 60).red() > 200


def test_rotated_real_video_in_window(app, win, tmp_path):
    """A video added turned: the clips follow its portrait size; turning it back converts it again, unturned."""
    src = make_real_video(tmp_path / "real.mp4")
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.0, rotate=90)
    wait(app, lambda: not win._converts)
    node = win.tree.nodes[v]
    assert win.state(v) == "done" and node.source["rotate"] == 90 and node.made["key"]["rotate"] == 90
    info = chain.probe(win.project / node.made["video"])
    assert info["height"] > info["width"]
    assert "turned 90° right" in win.cards[0].range.text()
    assert win._canvas_for(str(src), 0, v)[0] > win._canvas_for(str(src), 0, v)[1]   # as if it were not turned
    win.set_video_range(v, str(src), 0.0, 1.0)                 # (no rotation given: kept)
    assert win.state(v) == "done" and win.tree.nodes[v].source["rotate"] == 90
    win.set_video_range(v, str(src), 0.0, 1.0, 0)
    wait(app, lambda: not win._converts)
    node = win.tree.nodes[v]
    assert "rotate" not in node.source and win.state(v) == "done"
    info = chain.probe(win.project / node.made["video"])
    assert info["width"] > info["height"]


def test_picker_cells_show_whole_thumbnails(app, tmp_path):
    """Every cell has room for the whole thumbnail (the first item, a folder icon, used to set a tiny size for
    all of them), also after zooming."""
    from video_chain.ui.image_picker import ImagePickerDialog
    d = tmp_path / "pics"
    (d / "sub").mkdir(parents=True)
    Image.new("RGB", (1280, 720), (200, 0, 0)).save(d / "wide.png")
    Image.new("RGB", (720, 1280), (0, 200, 0)).save(d / "tall.png")
    dlg = ImagePickerDialog(None, "x", str(d), settings(tmp_path, "cells.ini"), cache_dir=None, desktop_thumbs=None)
    dlg.show()
    for px in (150, 240):
        dlg.zoom.setValue(px)
        app.processEvents()
        for i in range(dlg.grid.count()):
            r = dlg.grid.visualItemRect(dlg.grid.item(i))
            assert r.width() >= px and r.height() >= px + dlg.grid.fontMetrics().height(), (px, r)
    dlg.reject()


def test_wheel_does_not_change_unfocused_fields(app, win, tmp_path, image):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    ids = new_project(win, tmp_path, image, ("a",))
    card = win.cards[0]

    def wheel(w):
        e = QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, -120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(w, e)
    for field, value in ((win.resolution, lambda: win.resolution.currentIndex()), (card.seed, card.seed.value),
                         (card.length, card.length.currentIndex)):
        before = value()
        wheel(field)
        assert value() == before, field
    flush(app, win)
    assert win.state(ids[0]) == "new" and win.tree.nodes[ids[0]].seed == card.seed.value()
    card.seed.setFocus()
    app.processEvents()
    if card.seed.hasFocus():                                     # focused (clicked into): the wheel works
        before = card.seed.value()
        wheel(card.seed)
        assert card.seed.value() != before


def test_project_changes_are_confirmed_and_counted(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image)
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
    other = tmp_path / "other.png"
    Image.new("RGB", (192, 108), (10, 10, 10)).save(other)
    before = (win.project / T.START_FILE).read_bytes()
    win.start_tile.dropped.emit(str(other))                      # dropped by accident: asked, kept
    assert asked and "2 made clips depend" in asked[0] and (win.project / T.START_FILE).read_bytes() == before
    assert [win.state(i) for i in ids] == ["done", "done"]
    win.resolution.setCurrentIndex(win.resolution.findData(384))
    assert "2 made clips are now CHANGED" in win.status.text()
    flush(app, win)
    win.resolution.setCurrentIndex(win.resolution.findData(512))
    assert "2 clips are DONE again" in win.status.text()
    flush(app, win)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    win.start_tile.dropped.emit(str(other))
    assert (win.project / T.START_FILE).read_bytes() != before
    assert "New start image: 2 made clips are now CHANGED" in win.status.text()


def test_lora_stack_keeps_paths_and_picks_from_the_picker(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    real = tmp_path / "loras" / "nose.safetensors"
    write_lora(real, H3_KEYS)
    link = tmp_path / "comfy-loras" / "nose.safetensors"
    link.parent.mkdir()
    link.symlink_to(real)
    win.h3_loras = [(str(link), "nose.safetensors")]               # ComfyUI lists the link
    node = win.tree.nodes[ids[1]]
    node.overrides = {"loras": [{"path": str(real), "strength": 0.8}]}   # the clip stores the file itself
    win.refresh(rebuild_path=True)
    card = win.cards[1]
    assert card.loras.rows[0][1].property("path") == str(real) and card.loras.rows[0][1].text() == "glow"
    card.guide.setCurrentIndex(card.guide.findData(5))             # touching another setting …
    assert node.overrides["loras"] == [{"path": str(real), "strength": 0.8}]   # … keeps the stored path
    from video_chain.ui.lora_stack import LoraStack
    monkeypatch.setattr(LoraStack, "pick", staticmethod(lambda parent, known, current, used, file_dir: str(link)))
    card.loras.b_add.click()                                        # '+ LoRA': chosen in the picker
    assert [it["path"] for it in node.overrides["loras"]] == [str(real), str(link)]
    win.frames.setCurrentIndex(win.frames.findData(90))             # the cards' lists are filled again
    assert len(win.cards[1].loras.rows) == 2


def test_lora_outside_comfyui_and_rescan(app, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "loras").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    q = settings(tmp_path, "rescan.ini")
    q.setValue("comfyDir", str(comfy))
    w = MainWindow(q, command=FAKE)
    ids = new_project(w, tmp_path, image, ("a",))
    wait(app, lambda: w._scanned_at > 0, timeout=10)
    outside = tmp_path / "elsewhere" / "mine.safetensors"
    write_lora(outside, H3_KEYS)
    w.add_node_lora(ids[0], str(outside))
    assert "not in ComfyUI's LoRA folders" in w.status.text()
    inside = comfy / "models" / "loras" / "new.safetensors"
    write_lora(inside, H3_KEYS)
    assert w.in_comfy_loras(str(inside)) is True and w.in_comfy_loras(str(outside)) is False
    w.b_rescan.click()                                               # a freshly trained LoRA shows up
    wait(app, lambda: any(Path(p).name == "new.safetensors" for p, _n in w.h3_loras), timeout=10)
    assert "1 LoRA and 0 turbo LoRAs" in w.status.text()
    w.close()


def test_waiting_for_comfyui_can_be_cancelled(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    win.comfy_up = False
    win.comfy_autostart.setChecked(True)
    monkeypatch.setattr(win, "start_comfy", lambda: True)
    win.generate_path()
    assert win._waiting_comfy == [ids[0]] and not win.running and win.b_stop.isEnabled()
    win.stop()
    assert win._waiting_comfy is None and not win.b_stop.isEnabled() and "Cancelled" in win.status.text()
    win._comfy_state(True)                                            # ComfyUI comes up: nothing starts
    app.processEvents()
    assert not win.running


def test_no_delete_while_files_are_busy_and_small_card_fixes(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image)
    win._joins = 1                                                    # a path is being joined
    win.node_action("delete", ids[1])
    assert ids[1] in win.tree.nodes and "Wait until the path is joined" in win.status.text()
    win._joins = 0
    win.refresh()
    # carry-over has nothing to take from on the first clip
    assert not win.cards[0].guide.isEnabled() and win.cards[1].guide.isEnabled()
    # a made prompt that differs only in spacing stays as made
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    n = win.tree.nodes[ids[0]]
    n.made["prompt"] = n.prompt = "one\ntwo  three"
    win.refresh(rebuild_path=True)
    win.cards[0].edit.setPlainText("one two three")
    assert n.prompt == "one\ntwo  three"
    win.cards[0].edit.setPlainText("one two four")
    assert n.prompt == "one two four"


def test_hand_over_frame_in_the_window(app, win, tmp_path, image):
    from video_chain.ui.clip_dialogs import HandoverDialog
    ids = new_project(win, tmp_path, image, ("a", "b"))
    card = win.cards[0]
    assert not card.b_hand.isEnabled()                                  # not made yet
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    a = win.tree.nodes[ids[0]]
    dlg = HandoverDialog(None, win.project / a.made["video"], "x", None)
    wait(app, lambda: dlg.frames, timeout=20)
    assert len(dlg.frames) == 17 and dlg.frame() is None               # starts at the last frame
    dlg.slider.setValue(9)
    assert dlg.frame() == 9 and "7 frames" in dlg.summary.text()
    dlg.reject()
    assert win.cards[0].b_hand.isEnabled()
    wait(app, lambda: not win._joins, timeout=30)                       # the path played after generating
    assert win.set_handover(ids[0], 9)
    assert a.end_at == 9 and [win.state(i) for i in ids] == ["done", "changed"]
    assert "1 made clip is now CHANGED" in win.status.text()
    assert win.cards[0].hand_text.text() == "frame 10 of 17 (0.38 s)" and "hands over at 0.4 s" in win.badge(a)
    assert win._node_frame(a).name.endswith("_f9.png")
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    assert [win.state(i) for i in ids] == ["done", "done"]
    wait(app, lambda: not win._joins, timeout=30)
    win.node_action("handover_clear", ids[0])                           # back to the last frame
    assert a.end_at is None and win.state(ids[1]) == "changed"


def test_compare_variants_and_keep_one(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    new = win.create_variants(ids[1], 2, generate=True)
    wait(app, lambda: not win.running)
    run_and_wait(app, win, win.generate_all)
    flush(app, win)
    wait(app, lambda: not win._joins, timeout=30)
    assert win.compare_candidates(ids[1]) == [ids[1]] + new
    win.select(ids[1])
    assert win.cards[1].b_compare.isEnabled() and not win.cards[0].b_compare.isEnabled()
    dlg = win.compare_variants(ids[1], player=False)
    assert len(dlg.keep_buttons) == 3 and "seed" in dlg.items[1][2]
    dlg.keep_buttons[new[0]].click()                                    # keep the first variant
    assert dlg.kept == new[0] and set(win.tree.nodes) == {ids[0], new[0]}
    assert win.selected_id() == new[0]
    assert win.compare_variants(new[0], player=False) is None and "Nothing to compare" in win.status.text()


def test_style_text_in_the_window(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("walks in",))
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    win.style_prefix.setText("cinematic,")
    win.style_prefix.editingFinished.emit()
    assert win.tree.options["style"] == {"prefix": "cinematic,", "suffix": ""}
    assert "1 made clip is now CHANGED" in win.status.text()
    flush(app, win)
    assert win.cards[0].words.text().endswith("+ style")
    assert win.cards[0].edit.toolTip() == "Sent to the model: cinematic, walks in"
    win.style_suffix.setText("slow motion")                             # not finished with Enter: taken at Generate
    run_and_wait(app, win, win.generate_path)
    made = T.load(win.project).nodes[ids[0]].made["settings"]
    assert made["prompt_prefix"] == "cinematic," and made["prompt_suffix"] == "slow motion"
    win.close_project()
    win.open_project(tmp_path / "proj2")                                 # a new project starts without it
    assert win.style_prefix.text() == "" and (win.tree.options.get("style") or {}).get("prefix", "") == ""


def test_ai_image_edit_for_start_and_end_images(app, tmp_path, image, monkeypatch):
    from PySide6.QtCore import Qt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "edit.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        ids = new_project(w, tmp_path, image, ("a", "b"))
        run_and_wait(app, w, w.generate_path)
        flush(app, w)
        dlg = w.edit_image(ids[1], dialog=False)
        texts = [dlg.src_combo.itemText(i) for i in range(dlg.src_combo.count())]
        assert texts[0] == f"Last frame of clip #{ids[1][1:]}" and "Start image" in texts and texts[-2:] == ["Choose file…", "Choose folder…"]
        assert dlg.source.endswith(".png") and not dlg.b_gen.isEnabled()          # no prompt yet
        assert dlg.b_use_end.text() == f"Use as end image of clip #{ids[1][1:]}"
        dlg.prompt.setPlainText("she smiles at the camera")
        dlg.tries.setValue(2)
        lora = tmp_path / "loras" / "MiniMax H3" / "nose.safetensors"     # ComfyUI (fake) lists it by this name
        lora.parent.mkdir(parents=True)
        lora.write_bytes(b"x")
        dlg.add_lora_file(str(lora), 0.7)
        assert dlg.loras.items() == [{"path": str(lora), "strength": 0.7}]
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.state.text().startswith("Done") and dlg.results.count() == 2
        png, meta = dlg.selected()
        assert meta["loras"] == [{"path": str(lora), "strength": 0.7}] and "nose ×0.7" in dlg.res_info.text()
        assert server.prompts[-1]["lora0"]["inputs"]["lora_name"] == "MiniMax H3/glow.safetensors"
        size = QImage_size(meta["source"])
        assert QImage_size(png) == size and meta["prompt"] == "she smiles at the camera"   # back to the source size
        assert Path(meta["source_copy"]).is_file()
        dlg.b_before.setChecked(True)
        assert dlg.res_info.text().startswith("ORIGINAL")
        dlg.b_use_ref.click()                                       # one more face reference, the dialog stays
        refs = T.all_pictures(w.tree)
        assert len(refs) == 1 and (w.project / refs[0]).is_file() and dlg.used is None
        assert "@Person1" in dlg.state.text() and w.refs.count() == 1
        assert T.lead_of(w.tree)["name"] == "Person1"                # the first face: the lead
        dlg.b_use_end.click()                                       # the clip's end image
        assert dlg.used == ("end", png) and w.tree.nodes[ids[1]].end_image and w.state(ids[1]) == "changed"
        # from the start image section: a result as the new start image; earlier results are listed
        dlg2 = w.edit_image(start=True, dialog=False)
        assert dlg2.results.count() == 2 and dlg2.source == str(w.project / T.START_FILE)
        assert dlg2.loras.items() == [{"path": str(lora), "strength": 0.7}]      # remembered for next time
        dlg2.loras.set_items(dlg2._known(), [])
        dlg2._prompt_again()                                        # back to the selected result's prompt and LoRAs
        assert dlg2.loras.items() == [{"path": str(lora), "strength": 0.7}]
        # bad results can be deleted; one that is in use (here: the end image of clip 2, a face reference) not
        dlg2.results.setCurrentRow(next(i for i in range(dlg2.results.count())
                                        if dlg2.results.item(i).data(Qt.UserRole)[0] == png))
        assert not dlg2.b_delete.isEnabled() and dlg2.res_info.text().startswith(
            f"IN USE: face of @Person1, the end image of clip #{ids[1][1:]} · ")
        other = next(i for i in range(dlg2.results.count()) if dlg2.results.item(i).data(Qt.UserRole)[0] != png)
        dlg2.results.setCurrentRow(other)
        bad = dlg2.selected()[0]
        assert dlg2.b_delete.isEnabled()
        dlg2.b_delete.click()
        assert dlg2.results.count() == 1 and not Path(bad).exists() and not dlg2.b_undo.isHidden()
        dlg2.b_undo.click()                                         # undo: it is back
        assert dlg2.results.count() == 2 and Path(bad).is_file()
        dlg2.results.setCurrentRow(next(i for i in range(dlg2.results.count())
                                        if dlg2.results.item(i).data(Qt.UserRole)[0] == bad))
        dlg2.b_delete.click()
        dlg2.results.setCurrentRow(0)
        assert dlg2.b_use_end.isEnabled() is False
        before = (w.project / T.START_FILE).read_bytes()
        dlg2.b_use_start.click()
        assert (w.project / T.START_FILE).read_bytes() != before and w.state(ids[0]) == "changed"
        assert not (w.project / "edits" / ".deleted").exists() and not Path(bad).exists()   # gone for good on close
        dlg4 = w.edit_image(start=True, dialog=False)
        assert dlg4.results.count() == 1 and "the start image" in dlg4.res_info.text() and not dlg4.b_delete.isEnabled()
        dlg4.reject()
        assert any(text == "Project AI edits" for text, _p in w._picker_places())
        server.close()                                              # ComfyUI gone: a clear message
        dlg3 = w.edit_image(start=True, dialog=False)
        dlg3.prompt.setPlainText("x")
        dlg3.generate()
        wait(app, lambda: not dlg3._busy, timeout=60)
        assert "ComfyUI is not running" in dlg3.state.text() and not dlg3.b_start_comfy.isHidden()
        w.close()
    finally:
        try:
            server.close()
        except Exception:  # noqa: BLE001
            pass


def QImage_size(path):
    from PySide6.QtGui import QImage
    img = QImage(str(path))
    return img.width() // 2 * 2, img.height() // 2 * 2


def test_switching_between_clips_and_image_edits_unloads_only_then(app, tmp_path, image, monkeypatch):
    """ComfyUI is asked to unload only when the model changes: clips (H3) -> image edit (Qwen) frees H3 once;
    more edits reuse Qwen; clips again free Qwen once; more clips reuse H3."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    worker_frees = tmp_path / "frees.txt"
    monkeypatch.setenv("FAKE_FREES", str(worker_frees))
    server = FakeComfy(tmp_path / "comfy")

    def comfy_frees():
        return sum(1 for c in server.calls if c[0] == "BODY" and c[1] == "/free")

    def worker_free_count():
        return len(worker_frees.read_text().split()) if worker_frees.exists() else 0
    try:
        q = settings(tmp_path, "switch.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        ids = new_project(w, tmp_path, image, ("a",))
        run_and_wait(app, w, w.generate_path)
        assert w._comfy_model == "h3" and w.worker_alive
        dlg = w.edit_image(ids[0], dialog=False)
        dlg.prompt.setPlainText("night")
        dlg.tries.setValue(1)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert comfy_frees() == 1                                        # H3 unloaded once, before Qwen
        wait(app, lambda: not w.worker_alive, timeout=20)                # the worker left without unloading again
        assert worker_free_count() == 0
        dlg.generate()                                                    # another edit: Qwen stays
        wait(app, lambda: not dlg._busy, timeout=60)
        assert comfy_frees() == 1
        dlg.b_use_end.click()
        w.edits_closed()
        assert w._qwen_timer.isActive() and "image model loaded" in w.model_label.text()
        assert not w.b_unload.isHidden()
        wait(app, lambda: not w._joins, timeout=30)
        run_and_wait(app, w, w.generate_path)                            # clips again: Qwen unloaded once
        assert worker_free_count() == 1 and w._comfy_model == "h3" and not w._qwen_timer.isActive()
        w.tree.nodes[ids[0]].prompt = "a2"
        wait(app, lambda: not w._joins, timeout=30)
        run_and_wait(app, w, w.generate_path)                            # more clips: nothing unloaded
        assert worker_free_count() == 1
        w.close()
    finally:
        server.close()


def test_export_refined_and_hyperflow_speed(app, win, tmp_path, image):
    from video_chain.ui.main_window import ExportDialog
    ids = new_project(win, tmp_path, image, ("a", "b"))
    win.speed.setCurrentIndex(win.speed.findData("hyperflow"))           # HyperFlow: 8 steps for every clip
    assert win.tree.options["hyperflow"]["on"] and not win.steps.isEnabled()
    assert T.Checker(win.project, win.tree).settings(ids[0])["steps"] == 8
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384), {"2k": False, "1080p": True})
    texts = [dlg.size.itemText(i) for i in range(dlg.size.count())]
    assert texts[0].startswith("As made (672×384)") and "1888×1088 · 2.8× the draft · already upscaled" in texts[1] \
        and "2528×1440 · 3.8× the draft" in texts[2]
    assert dlg.scale_note.isHidden()                                         # (as made: no enlargement)
    assert dlg.swapped.isHidden()                                            # no video with a replaced person
    assert dlg.method.currentData() == "seedvr2_7b" and not dlg.method.isEnabled()      # SeedVR2 7B by default
    assert [dlg.method.itemData(i) for i in range(dlg.method.count())] == ["seedvr2_7b", "seedvr2_3b", "refine"]
    assert dlg.strength.currentData() == "natural" and not dlg.strength.isEnabled()
    assert [dlg.strength.itemData(i) for i in range(dlg.strength.count())] == ["soft", "natural", "crisp"]
    dlg.size.setCurrentIndex(2)
    assert dlg.method.isEnabled() and not dlg.strength.isEnabled() and not dlg.fast.isEnabled()   # (refine only)
    assert dlg.denoise.isEnabled() and not dlg.denoise.isChecked()                # SeedVR2 only, off by default
    dlg.method.setCurrentIndex(dlg.method.findData("refine"))
    assert dlg.strength.isEnabled() and dlg.fast.isEnabled() and not dlg.denoise.isEnabled()
    dlg.reject()
    asked = []
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384),
                       lambda size, how, st, dn=False: asked.append(dn) or dn, denoise=True)
    assert dlg.denoise.isChecked() and "already upscaled" in dlg.size.itemText(2)   # kept copies per choice
    dlg.denoise.setChecked(False)
    assert "already upscaled" not in dlg.size.itemText(2) and asked[-1] is False
    dlg.reject()
    seen = []
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384),
                       lambda size, how, st, dn=False: seen.append((size, how, st)) or how == "seedvr2_3b", swapped=2)
    assert "already upscaled" not in dlg.size.itemText(2) and ("2k", "seedvr2_7b", "natural") in seen
    dlg.method.setCurrentIndex(dlg.method.findData("seedvr2_3b"))         # kept copies are per method
    assert "already upscaled" in dlg.size.itemText(2)
    assert not dlg.swapped.isHidden() and dlg.swapped.isChecked() and not dlg.swapped.isEnabled()
    assert dlg.swapped.text() == "Upscale the videos with a replaced person too"
    dlg.size.setCurrentIndex(2)
    assert dlg.swapped.isEnabled()
    dlg.reject()
    out = tmp_path / "export-2k.mp4"
    win.export(out, "2k")                                                    # SeedVR2 7B
    assert win.running
    wait(app, lambda: not win.running, timeout=120)
    info = chain.probe(out)
    base = T.Checker(win.project, win.tree).base()
    from video_chain.comfy import refine_size
    assert (info["width"], info["height"]) == refine_size(base["width"], base["height"], "2k")
    assert win.status.text().startswith(f"Exported {out.name} (2k, SeedVR2 7B; 2 clips upscaled now"), \
        win.status.text()
    win.export(out, "2k")                                                    # again: the kept copies are reused
    wait(app, lambda: not win.running, timeout=120)
    assert win.status.text() == f"Exported {out.name} (2k, SeedVR2 7B; every clip was upscaled already)."
    win.export(out, "2k", method="refine")                                   # the refine: its own copies
    wait(app, lambda: not win.running, timeout=120)
    assert win.status.text().startswith(f"Exported {out.name} (2k, refined; 2 clips upscaled now")
    assert {win.state(n) for n in ids} == {"done"}


def test_export_on_runpod(app, win, tmp_path, image, monkeypatch):
    """“Upscale on RunPod”: only for SeedVR2 at a higher size and with a key; the export gets its own worker bound
    to the pod settings (GPU, volume, fallback, SeedVR2 size), the local ComfyUI is neither needed nor started,
    the pod's preparation shows in the status bar, and the worker (with its pod) ends after the export."""
    from video_chain import runpod as RP
    from video_chain.ui.main_window import ExportDialog
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384), {}, cloud=None, cloud_on=True)
    assert not dlg.cloud.isEnabled() and not dlg.cloud.isChecked() and "API key" in dlg.cloud.toolTip()
    dlg.reject()
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384), {}, cloud=True, cloud_on=True)
    assert dlg.cloud.isChecked() and not dlg.cloud.isEnabled() and not dlg.on_cloud()   # as made: nothing to do
    dlg.size.setCurrentIndex(2)
    assert dlg.cloud.isEnabled() and dlg.on_cloud()
    dlg.method.setCurrentIndex(dlg.method.findData("refine"))                # the refine needs the H3 models
    assert not dlg.cloud.isEnabled() and not dlg.on_cloud()
    dlg.reject()

    loads = tmp_path / "loads.jsonl"
    monkeypatch.setenv("FAKE_LOADS", str(loads))
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)                                # the clip, by the local worker
    wait(app, lambda: not win._joins, timeout=30)
    out = tmp_path / "cloud-2k.mp4"
    win.export(out, "2k", cloud=True)
    assert not win.running and "RunPod API key" in win.status.text()        # no key: not started
    monkeypatch.setattr(RP, "api_key", lambda: "k-123")
    win.runpod_gpu.setCurrentIndex(win.runpod_gpu.findData("NVIDIA L40S"))
    win.runpod_volume.setText("vol9")
    win.runpod_volume.editingFinished.emit()
    win.runpod_fallback.setChecked(False)
    win.comfy_up = False                                                     # no local ComfyUI: not needed
    monkeypatch.setenv("FAKE_LOAD_DELAY", "0.5")                             # (the pod "prepares" a moment)
    win.export(out, "2k", method="seedvr2_3b", cloud=True)
    assert win.running and win._worker_key()[0] == "runpod"
    seen = []
    wait(app, lambda: seen.append(win.status.text()) or not win.running, timeout=120)
    assert any(t.startswith("RunPod: downloading") for t in seen)
    assert win.status.text().startswith(f"Exported {out.name} (2k, SeedVR2 3B"), win.status.text()
    assert chain.probe(out)["frames"] > 0
    cloud = [json.loads(x) for x in loads.read_text().splitlines() if "cloud" in x]
    assert cloud == [{"cloud": "runpod", "gpu": "NVIDIA L40S", "models": "3b", "volume": "vol9",
                      "fallback": False, "idle": 0, "without_volume": True}]
    wait(app, lambda: not win.worker_alive, timeout=30)                      # its pod is not left waiting
    assert not win._comfy_starting                                            # the local ComfyUI was not started
    assert {win.state(n) for n in ids} == {"done"}


def test_pika_speed_saved_and_steps_locked(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    win.speed.setCurrentIndex(win.speed.findData("pika"))
    o = win.tree.options
    assert o["pika"]["on"] and not o["hyperflow"]["on"] and not o["turbo"]["on"] and not win.steps.isEnabled()
    assert T.Checker(win.project, win.tree).settings(ids[0])["steps"] == 6
    win.open_project(tmp_path / "proj")                                   # saved with the project
    assert win.speed.currentData() == "pika" and not win.steps.isEnabled()
    win.speed.setCurrentIndex(win.speed.findData("normal"))
    assert not win.tree.options["pika"]["on"] and win.steps.isEnabled()
    assert "pika" not in T.Checker(win.project, win.tree).settings(ids[0])


def test_stop_looks_disabled_when_nothing_runs(app, win):
    """The danger style must not hide that Stop is disabled (it looked active after a finished export)."""
    from video_chain.ui.theme import ThemeManager
    qss = ThemeManager.instance().qss
    assert 'QPushButton[variant="danger"]:disabled' in qss
    assert not win.running and not win.b_stop.isEnabled()


def test_lora_picker_search_and_full_names(app, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from video_chain.ui.lora_stack import LoraPickerDialog, LoraStack, matches
    names = ["MiniMax H3/smoke_exhale_v2_very_long_name_that_used_to_be_cut_off", "MiniMax H3/glow_smoke",
             "other/smile-strong"]
    known = []
    for n in names:
        f = tmp_path / "loras" / f"{n}.safetensors"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
        known.append((str(f), Path(n).name))
    assert matches("smoke exh", "smoke_exhale_v2") and not matches("smoke smile", "glow_smoke")
    dlg = LoraPickerDialog(None, known, used={known[1][0]})
    texts = [dlg.list.item(i).text() for i in range(dlg.list.count())]
    assert texts[0].startswith("glow_smoke   · in use\n")                                 # marked, sorted by name
    assert any(t.startswith(Path(names[0]).name + "\n") for t in texts)                    # the whole name
    dlg.search.setText("smoke")
    assert len(dlg.visible_paths()) == 2 and dlg.count.text() == "2 of 3 LoRAs"
    dlg.search.setText("smoke exhale")
    assert dlg.visible_paths() == [known[0][0]]
    dlg.search.setText("other")                                                            # the folder counts too
    assert dlg.visible_paths() == [known[2][0]]
    QTest.keyClick(dlg.search, Qt.Key_Return)                                              # Enter takes the match
    assert dlg.chosen() == known[2][0] and dlg.result() == dlg.DialogCode.Accepted
    dlg2 = LoraPickerDialog(None, known, current=known[0][0])
    assert dlg2.list.currentItem().data(Qt.UserRole) == known[0][0]                       # the row's LoRA selected
    dlg2.search.setText("zzz")
    assert dlg2.visible_paths() == [] and not dlg2.buttons.button(dlg2.buttons.StandardButton.Ok).isEnabled()
    dlg2.reject()
    stack = LoraStack()
    stack.set_items(known, [{"path": known[0][0], "strength": 0.5}])
    assert stack.rows[0][1].text() == Path(names[0]).name and known[0][0] in stack.rows[0][1].toolTip()
    stack.pick = lambda parent, kn, current, used, file_dir: known[1][0]                  # click the row: swap
    stack.rows[0][1].click()
    assert stack.items() == [{"path": known[1][0], "strength": 0.5}]


def test_prompt_refiner_on_the_card(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a man walks",))
    seen, seen_pics = {}, []

    class FakeDialog:
        def __init__(self, parent, text, kind, connect, settings=None, start_comfy=None, cast=None, pictures=None):
            seen.update(text=text, kind=kind, cast=cast)
            seen_pics.extend(pictures or [])
            self.refined_any = True

        def exec(self):
            return M.QDialog.Accepted

        def chosen(self):
            return "A man walks slowly towards the camera; footsteps on gravel."

        def deleteLater(self):
            pass
    monkeypatch.setattr(M, "RefineDialog", FakeDialog)
    card = win.cards[0]
    card.b_refine.click()
    assert seen == {"text": "a man walks", "kind": "video", "cast": []}
    assert [t for t, _p, on in seen_pics] == ["The frame the clip starts from"]   # (the start image; no end image)
    assert win.tree.nodes[ids[0]].prompt == "A man walks slowly towards the camera; footsteps on gravel."
    card.edit.undo()                                              # one Ctrl+Z takes the refined text back
    assert card.edit.toPlainText() == "a man walks"
    assert win._comfy_model == "llm" and "language model" in win.model_label.text()   # unloaded before clips


def test_create_a_new_image_in_the_window(app, tmp_path, image, monkeypatch):
    from PySide6.QtCore import Qt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "create.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        sdxl_lora = tmp_path / "loras" / "MiniMax H3" / "nose.safetensors"     # the fake lists it by this name
        sdxl_lora.parent.mkdir(parents=True)
        sdxl_lora.write_bytes(b"x")
        w.image_loras = {"sdxl": [(str(sdxl_lora), "anime/style.safetensors")], "flux1": [], "krea2": []}
        dlg = w.edit_image(start=True, dialog=False, mode="create")
        assert dlg.mode() == "create" and not dlg.edit_box.isVisibleTo(dlg) and dlg.create_box.isVisibleTo(dlg)
        wait(app, lambda: dlg.families, timeout=20)
        labels = [dlg.family.itemText(i) for i in range(dlg.family.count())]
        assert labels[0] == "Krea-2 Turbo" and "FLUX.3 Image Klein  (not available)" in labels
        k3 = dlg.family.findData("flux3")
        assert not dlg.family.model().item(k3).isEnabled() and "licence" in dlg.family.itemData(k3, Qt.ToolTipRole)
        dlg.family.setCurrentIndex(dlg.family.findData("sdxl"))
        dlg.family.activated.emit(dlg.family.currentIndex())          # chosen by the user: remembered
        assert [t for _p, t in dlg._known()] == ["style"] and dlg.negative.isVisibleTo(dlg) and dlg.tags.isVisibleTo(dlg)
        assert dlg.prompt_kind() == "image" and "×" in dlg.size_text.text()
        dlg.loras.set_items(dlg._known(), [{"path": str(sdxl_lora), "strength": 0.8}])
        dlg.prompt.setPlainText("1girl, red coat, rainy street")
        dlg.tries.setValue(1)
        assert dlg.b_gen.isEnabled()
        dlg.seed.setValue(4242)                                         # a chosen seed, kept for another try
        dlg.fresh_seed.setChecked(False)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.state.text().startswith("Done") and dlg.results.count() == 1
        png, meta = dlg.selected()
        assert meta["seed"] == 4242 and dlg.seed.value() == 4242                # (kept: "new seed" is off)
        assert dlg._seeds(3) == [4242, 4243, 4244] and dlg.seed.value() == 4242    # several results: +1, +2
        dlg.fresh_seed.setChecked(True)
        assert dlg._seeds(1) == [4242] and dlg.seed.value() != 4242               # a fresh one afterwards
        before = dlg.seed.value()
        dlg.b_seed.click()                                                       # one click: a new seed
        assert dlg.seed.value() != before
        assert meta["mode"] == "create" and meta["family"] == "sdxl" and meta["loras"][0]["strength"] == 0.8
        from PySide6.QtTest import QTest
        dlg.show()
        dlg.negative.setFocus()
        QTest.keyClick(dlg.negative, Qt.Key_Return)                     # Enter in a field presses no button
        assert dlg.mode() == "create"
        dlg.quality.setFocus()
        QTest.keyClick(dlg.quality, Qt.Key_Delete)                      # Delete only in the results list
        assert dlg.results.count() == 1
        dlg.b_use_ref.click()                                           # in use at once (record read again)
        assert not dlg.b_delete.isEnabled() and "IN USE: face of @Person1" in dlg.res_info.text()
        assert server.created["lora0"]["class_type"] == "LoraLoader" and "Illustrious" in dlg.res_info.text()
        dlg.use_refs.setChecked(True)                                   # from a picture: needs one first
        assert not dlg.b_gen.isEnabled() and dlg.ref_strength.isVisibleTo(dlg)
        dlg.cpics[0].set(str(w.project / T.START_FILE))
        assert dlg.b_gen.isEnabled()
        dlg.b_use_start.click()                                         # the new image as the start image
        assert dlg.used == ("start", png)
        dlg2 = w.edit_image(start=True, dialog=False)                  # remembered: create mode, the family
        assert dlg2.mode() == "create"
        wait(app, lambda: dlg2.families, timeout=20)
        assert dlg2.family.currentData() == "sdxl" and dlg2.loras.items()[0]["strength"] == 0.8
        dlg2.m_edit.click()
        assert dlg2.mode() == "edit" and dlg2.edit_box.isVisibleTo(dlg2) and dlg2.prompt_kind() == "edit"
        dlg2.reject()
        w.close()
    finally:
        server.close()


def test_lora_strengths_from_minus_10_to_10(app, win, tmp_path, image):
    lora = tmp_path / "slider.safetensors"
    lora.write_bytes(b"x")
    ids = new_project(win, tmp_path, image, ("a",))
    win.add_node_lora(ids[0], str(lora))
    spin = win.cards[0].loras.rows[0][2]
    assert (spin.minimum(), spin.maximum()) == (-10.0, 10.0)
    spin.setValue(-2.5)                                          # a slider LoRA the other way round
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["loras"][0]["strength"] == -2.5
    assert "slider ×-2.5" in win.badge(win.tree.nodes[ids[0]])
    assert T.Checker(win.project, win.tree).settings(ids[0])["strength"] == -2.5
    assert (win.turbo_strength.minimum(), win.turbo_strength.maximum()) == (-10.0, 10.0)


def test_image_window_lora_rows_fit_and_can_be_removed(app, tmp_path):
    """The left panel must fit its view in both modes: an unwrapped hint pushed the strength and the remove button
    of a LoRA row out of sight, so a LoRA could not be removed."""
    from PySide6.QtCore import QSettings

    from video_chain.ui.image_edit import ImageEditDialog
    lora = tmp_path / "x.safetensors"
    lora.write_bytes(b"x")
    q = QSettings(str(tmp_path / "fit.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                          known_loras=[(str(lora), "x")], connect_create=lambda p: None)
    dlg.show()
    for mode in ("edit", "create"):
        dlg.set_mode(mode)
        dlg.loras.set_items(dlg._known(), [{"path": str(lora), "strength": 1.0}])
        app.processEvents()
        view = dlg.left_scroll.viewport().width()
        assert dlg.left_scroll.widget().width() <= view, mode
        row, _b, _spin = dlg.loras.rows[0]
        rm = row.findChildren(type(dlg.b_refine))[0]                  # the remove button (a tool button)
        assert rm.mapTo(dlg.left_scroll.widget(), rm.rect().topRight()).x() <= view, mode
        rm.click()
        assert dlg.loras.items() == [], mode
    dlg.reject()


def test_image_window_own_sampling(app, tmp_path):
    """Create: steps / cfg / sampler / scheduler, each "auto (the model's own)"; remembered per family, warned
    about when they fight the model, sent with Generate and taken back with a result's settings."""
    from PySide6.QtCore import QSettings

    from video_chain import imagegen as G
    from video_chain.ui.image_edit import ImageEditDialog
    q = QSettings(str(tmp_path / "samp.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                          connect_create=lambda p: None)
    files = G.pick_files(["krea2/krea2_turbo_fp8_scaled.safetensors"], ["illustrij_v20.safetensors"],
                         ["qwen3vl_4b_fp8_scaled.safetensors"], ["qwen_image_vae.safetensors"], [])
    files["sampling"] = {"samplers": ["euler", "heun", "dpmpp_2m"], "schedulers": ["simple", "karras", "beta"]}
    q.setValue("createFamily", "krea2")
    dlg._families_found(files, "")
    dlg.set_mode("create")
    assert dlg.family.currentData() == "krea2" and dlg.sampling() == {}
    dlg.tries.setValue(25)
    assert dlg.tries.value() == 20                                  # up to 20 results per Generate
    assert dlg.steps.specialValueText() == "auto (8)" and dlg.cfg.specialValueText() == "auto (1)"
    assert dlg.sampler.itemText(0) == "auto (euler)" and dlg.sampler.count() == 4
    assert not dlg.negative.isVisibleTo(dlg)                   # Krea-2 at CFG 1: a negative does nothing
    dlg.cfg.setValue(4.0)
    assert dlg.negative.isVisibleTo(dlg) and "counts because CFG is 4" in dlg.negative.placeholderText()
    dlg.steps.setValue(3)
    dlg.scheduler.setCurrentIndex(dlg.scheduler.findData("karras"))
    assert dlg.sampling() == {"steps": 3, "cfg": 4.0, "scheduler": "karras"}
    assert "twice as slow" in dlg.sampling_note.text() and "far below" in dlg.sampling_note.text()
    dlg.family.setCurrentIndex(dlg.family.findData("sdxl"))
    assert dlg.sampling() == {} and dlg.sampler.itemText(0).startswith("auto (euler")      # its own settings
    dlg.family.setCurrentIndex(dlg.family.findData("krea2"))
    assert dlg.sampling() == {"steps": 3, "cfg": 4.0, "scheduler": "karras"}              # remembered per family
    dlg.set_sampling({"sampler": "dpmpp_2m"})
    assert dlg.sampling() == {"sampler": "dpmpp_2m"} and "Own settings in use" in dlg.sampling_note.text()
    dlg.set_sampling({})
    assert dlg.sampling() == {} and not dlg.sampling_note.text()
    dlg.reject()


def test_face_crop_dialog(app, tmp_path):
    from video_chain.ui.face_crop import FaceCropDialog
    p = tmp_path / "pic.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(p)
    dlg = FaceCropDialog(None, p, None)                       # no detector: the middle square, set by hand
    assert dlg.crop() == (50, 0, 200) and "No face detection" in dlg.info.text()
    assert not dlg.b_face.isEnabled()
    dlg.view.set_crop((280, -5, 100))                         # kept inside the picture
    assert dlg.crop() == (200, 0, 100) and "100 × 100 px" in dlg.info.text() and "Small" in dlg.info.text()
    dlg.b_whole.click()
    assert dlg.crop() is None
    dlg.deleteLater()


def test_qwen_image_21_in_the_image_window(app, tmp_path, image, monkeypatch):
    """Qwen-Image 2.1 next to Qwen-Image-Edit 2511 for edits (the model can be chosen), and in Create's list."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "q21.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        dlg = w.edit_image(start=True, dialog=False)
        wait(app, lambda: dlg.families, timeout=20)
        models = [dlg.edit_model.itemText(i) for i in range(dlg.edit_model.count())]
        assert models == ["Qwen-Image-Edit 2511", "Qwen-Image 2.1", "MOSS-Image Edit"]
        assert dlg.edit_model.currentData() == "2511"
        assert [dlg.quality.itemData(i) for i in range(dlg.quality.count())] == ["fast", "standard", "best"]
        dlg.edit_model.setCurrentIndex(1)
        dlg.edit_model.activated.emit(1)                            # the user's choice is remembered
        assert [dlg.quality.itemText(i) for i in range(dlg.quality.count())] == ["Fast (Turbo, 8 steps)",
                                                                                 "Best (25 steps)"]
        assert q.value("editModel") == "qwen21" and "Qwen-Image 2.1" in dlg.lora_text.text()
        dlg.prompt.setPlainText("she smiles")
        dlg.tries.setValue(1)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        png, meta = dlg.selected()
        assert meta["edit_model"] == "qwen21" and meta["steps"] == 8 and "Qwen-Image 2.1 Fast" in dlg.res_info.text()
        assert server.edited["text"]["class_type"] == "TextEncodeQwenImage21"
        dlg.edit_model.setCurrentIndex(2)                           # MOSS-Image Edit
        assert [dlg.quality.itemText(i) for i in range(dlg.quality.count())] == ["Standard (30 steps)"]
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.selected()[1]["edit_model"] == "moss" and "MOSS-Image Edit" in dlg.res_info.text()
        dlg.results.setCurrentRow(dlg.results.count() - 1)          # the 2.1 result (oldest last)
        dlg.edit_model.setCurrentIndex(0)
        dlg._prompt_again()                                         # back to the model the result was made with
        assert dlg.edit_model.currentData() == "qwen21" and dlg.quality.currentData() == "fast"
        # Create: Qwen-Image 2.1 in the model list
        dlg.set_mode("create")
        k = dlg.family.findData("qwen21")
        assert dlg.family.itemText(k) == "Qwen-Image 2.1" and dlg.family.model().item(k).isEnabled()
        dlg.family.setCurrentIndex(k)
        assert dlg.model_file.count() == 1 and "Fast (Turbo, 8 steps)" in [dlg.quality.itemText(i)
                                                                         for i in range(dlg.quality.count())]
        k = dlg.family.findData("moss")
        assert dlg.family.itemText(k) == "MOSS-Image" and dlg.family.model().item(k).isEnabled()
        dlg.done(0)
        # without the 2.1 and MOSS files: offered, but not available, with what is missing
        server.qwen21 = server.moss = False
        dlg = w.edit_image(start=True, dialog=False)
        wait(app, lambda: dlg.families, timeout=20)
        k = dlg.edit_model.findData("qwen21")
        assert not dlg.edit_model.model().item(k).isEnabled() and dlg.edit_model.currentData() == "2511"
        assert "not available" in dlg.edit_model.itemText(k)
        dlg.done(0)
        w.close()
    finally:
        server.close()


def test_lora_trigger_words_in_the_image_window(app, tmp_path, image, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain import triggers as TR
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "trig.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        lora = tmp_path / "loras" / "MiniMax H3" / "nose.safetensors"           # ComfyUI (fake) lists this name
        lora.parent.mkdir(parents=True)
        save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(lora),
                  metadata={"modelspec.trigger_phrase": "glowx"})
        dlg = w.edit_image(start=True, dialog=False)
        dlg.add_lora_file(str(lora), 1.0)
        row = dlg.loras.rows[0][0]
        assert row.trigger_label.text() == "trigger words: not looked up yet"
        TR.db().scan([str(lora)], online=False)
        dlg.loras.refresh_triggers()
        assert row.trigger_label.text() == "trigger words: glowx  (from the file)"
        dlg.prompt.setPlainText("she smiles")
        dlg.tries.setValue(1)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        meta = dlg.selected()[1]
        assert server.prompts[-1]["pos"]["inputs"]["prompt"] == "glowx, she smiles"
        assert meta["prompt"] == "she smiles" and meta["trigger_words"] == ["glowx"]
        assert "she smiles" not in dlg.res_info.text()             # (a long prompt pushed over the picture)
        assert "Prompt: she smiles" in dlg.res_info.toolTip() and "Trigger words added: glowx" in dlg.view.toolTip()
        dlg.done(0)
        w.lora_triggers.setChecked(False)                                       # the project's choice
        dlg = w.edit_image(start=True, dialog=False)
        assert dlg._trigger_words() == []
        dlg.done(0)
        w.close()
    finally:
        server.close()


def test_one_image_per_lora_in_the_image_window(app, tmp_path, image, monkeypatch):
    """One image per LoRA: each result is made with every LoRA alone in turn, same seed, prompt and settings;
    each image gets only its own LoRA's trigger words."""
    import numpy as np
    from safetensors.numpy import save_file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain import triggers as TR
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy", loras=("MiniMax H3/glow.safetensors", "MiniMax H3/rain.safetensors"))
    try:
        q = settings(tmp_path, "each.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        files = []
        for name in ("glow", "rain"):
            f = tmp_path / "loras" / "MiniMax H3" / f"{name}.safetensors"
            f.parent.mkdir(parents=True, exist_ok=True)
            save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(f),
                      metadata={"modelspec.trigger_phrase": f"{name}x"})
            files.append(str(f))
        TR.db().scan(files, online=False)
        dlg = w.edit_image(start=True, dialog=False)
        dlg.add_lora_file(files[0], 0.7)
        dlg.add_lora_file(files[1], 1.2)
        dlg.prompt.setPlainText("she smiles")
        dlg.tries.setValue(2)
        assert dlg.b_gen.text() == "Generate"
        dlg.lora_each.setChecked(True)
        assert dlg.b_gen.text() == "Generate 4 images"                 # 2 results × 2 LoRAs
        dlg.fresh_seed.setChecked(False)
        dlg.seed.setValue(500)
        n0 = len(server.prompts)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=90)
        assert dlg.state.text().startswith("Done"), dlg.state.text()
        sent = [wf["pos"]["inputs"]["prompt"] for wf in server.prompts[n0:]]
        assert sent == ["glowx, she smiles", "rainx, she smiles", "glowx, she smiles", "rainx, she smiles"]
        metas = sorted((json.loads(f.read_text()) for f in (w.project / "edits").glob("edit_*.json")),
                       key=lambda m: (m["seed"], m["loras"][0]["path"]))
        assert len(metas) == 4
        got = [(m["seed"], Path(m["loras"][0]["path"]).stem, m["loras"][0]["strength"], len(m["loras"]),
                m["trigger_words"], m["prompt"]) for m in metas]
        assert got == [(500, "rain", 1.2, 1, ["rainx"], "she smiles"), (500, "glow", 0.7, 1, ["glowx"], "she smiles"),
                       (501, "rain", 1.2, 1, ["rainx"], "she smiles"), (501, "glow", 0.7, 1, ["glowx"], "she smiles")]
        dlg.done(0)
        dlg = w.edit_image(start=True, dialog=False)                   # remembered
        assert dlg.lora_each.isChecked()
        dlg.lora_each.setChecked(False)
        assert len(dlg._lora_runs()) == 1 and len(dlg._lora_runs()[0]) == 2   # off: all LoRAs together
        dlg.done(0)
        w.close()
    finally:
        server.close()


def test_edit_every_image_of_a_folder(app, tmp_path, image, monkeypatch):
    """A folder as the image to edit: each of its images is edited in turn with the same prompt, seed and settings
    (here also one image per LoRA), at its own size; an unreadable file is skipped; no mask with a folder."""
    import subprocess

    import numpy as np
    from safetensors.numpy import save_file
    from PySide6.QtGui import QImage
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy", loras=("MiniMax H3/glow.safetensors", "MiniMax H3/rain.safetensors"))
    try:
        q = settings(tmp_path, "folder.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        folder = tmp_path / "pics"
        (folder / "sub").mkdir(parents=True)
        for name, size in (("img10.png", "320x240"), ("img2.jpg", "240x320"), ("img1.png", "256x256"),
                           ("sub/deep.png", "64x64")):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s={size}", "-frames:v", "1",
                            str(folder / name)], check=True)
        (folder / "broken.png").write_bytes(b"not a picture")
        (folder / "notes.txt").write_text("x")
        files = []
        for name in ("glow", "rain"):
            f = tmp_path / "loras" / "MiniMax H3" / f"{name}.safetensors"
            f.parent.mkdir(parents=True, exist_ok=True)
            save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(f))
            files.append(str(f))
        dlg = w.edit_image(start=True, dialog=False)
        assert [Path(f).name for f in dlg.folder_images(folder)] == ["broken.png", "img1.png", "img2.jpg", "img10.png"]
        assert dlg.set_folder(str(folder))
        assert dlg.batch and Path(dlg.source).name == "img1.png"        # (broken.png is not shown)
        assert dlg.src_combo.currentData() == "\x00folder:" + str(folder) and "4 images" in dlg.src_combo.currentText()
        assert not dlg.b_mask.isEnabled()                              # a mask belongs to one picture
        dlg.add_lora_file(files[0], 1.0)
        dlg.add_lora_file(files[1], 1.0)
        dlg.lora_each.setChecked(True)
        dlg.prompt.setPlainText("she smiles")
        dlg.tries.setValue(1)
        dlg.fresh_seed.setChecked(False)
        dlg.seed.setValue(77)
        assert dlg.b_gen.text() == "Generate 8 images"                 # 4 files × 2 LoRAs
        dlg.keep_size.setChecked(True)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=120)
        assert dlg.state.text().startswith("Done") and "Skipped (not readable as an image): broken.png" in dlg.state.text()
        metas = [json.loads(f.read_text()) for f in (w.project / "edits").glob("edit_*.json")]
        got = sorted((Path(m["source"]).name, Path(m["loras"][0]["path"]).stem, m["seed"], tuple(m["size"]))
                     for m in metas)
        assert got == [("img1.png", "rain", 77, (256, 256)), ("img1.png", "glow", 77, (256, 256)),
                       ("img10.png", "rain", 77, (320, 240)), ("img10.png", "glow", 77, (320, 240)),
                       ("img2.jpg", "rain", 77, (240, 320)), ("img2.jpg", "glow", 77, (240, 320))]
        for m in metas:                                                # each result: its own original, its size
            assert Path(m["source_copy"]).is_file() and m["folder"] == str(folder)
        pngs = sorted((w.project / "edits").glob("edit_*.png"))
        assert len(pngs) == 6 and len(list((w.project / "edits" / "sources").iterdir())) == 3
        assert {(QImage(str(f)).width(), QImage(str(f)).height()) for f in pngs} == {(256, 256), (320, 240), (240, 320)}
        dlg.set_source(str(folder / "img2.jpg"))                       # one image again: no folder
        assert not dlg.batch and dlg.b_mask.isEnabled() and dlg.b_gen.text() == "Generate 2 images"
        dlg.lora_each.setChecked(False)
        assert dlg.b_gen.text() == "Generate"
        assert not dlg.set_folder(str(folder / "sub" / "missing")) and "No images" in dlg.state.text()
        dlg.done(0)
        w.close()
    finally:
        server.close()


def test_continue_the_real_videos_sound_on_the_card(app, win, tmp_path):
    """The card's sound choice appears only after a real video; switching it on also carries the picture over."""
    from video_chain.ui.main_window import PromptCard
    src = make_real_video(tmp_path / "real.mp4", secs=8.0)
    win.open_project(tmp_path / "proj")
    first = win.cards[0]
    assert isinstance(first, PromptCard) and not first.src_sound.isVisibleTo(first)   # no real video before it
    assert first.b_speak.isVisibleTo(first)                  # (the row itself stays: Speak the lines)
    v = win.add_video(first.nid, str(src), 0.0, 1.5)
    wait(app, lambda: not win._converts)
    win.node_action("continue", v)
    a = win.selected_id()
    card = next(c for c in win.cards if c.nid == a)
    assert card.sound_box.isVisibleTo(card) and card.src_sound.currentData() is None and not card.guide.currentData()
    card.src_sound.setCurrentIndex(card.src_sound.findData(True))
    flush(app, win)
    ov = win.tree.nodes[a].overrides
    assert ov["source_sound"] is True and ov["guide"] == 22    # the longest carry-over the 36-frame video allows
    card = next(c for c in win.cards if c.nid == a)
    assert card.src_sound_info.text().startswith("Uses 0:") and "real.mp4" in card.src_sound_info.text()
    assert "video's sound" in card.settings_toggle.text()
    assert T.SOURCE_SOUND_TEXT.format(k=1) in card.edit.toolTip()
    card.src_sound.setCurrentIndex(0)                                                   # optional: off again
    flush(app, win)
    assert "source_sound" not in win.tree.nodes[a].overrides
    card = next(c for c in win.cards if c.nid == a)                         # or the sound of the clip before, again
    card.src_sound.setCurrentIndex(card.src_sound.findData("previous"))
    flush(app, win)
    assert win.tree.nodes[a].overrides["source_sound"] == "previous"
    card = next(c for c in win.cards if c.nid == a)
    assert card.src_sound_info.text().startswith(f"Repeats the sound of video #{v[1:]}")
    assert "repeats the sound before" in card.settings_toggle.text()
    win.node_action("continue", a)                    # after a generated clip: repeat it, or go on with the video
    b = win.selected_id()
    card = next(c for c in win.cards if c.nid == b)
    model = card.src_sound.model()
    assert card.sound_box.isVisibleTo(card)
    assert model.item(card.src_sound.findData("previous")).isEnabled()
    assert model.item(card.src_sound.findData(True)).isEnabled()           # (the real video is still before it)


def test_scenes_in_the_window(app, win, tmp_path, image, monkeypatch):
    """Scenes: the storyboard shows them; a new one opens; switching keeps each scene's clips; the cast is shared."""
    ids = new_project(win, tmp_path, image, ("kitchen clip",))
    root = win.project
    assert win.storyboard.list.count() == 1 and win.windowTitle().startswith(root.name + " — ")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Street", True))
    win.add_scene()
    assert win.project == (root / "scenes" / "s2").resolve() and win.storyboard.list.count() == 2
    assert "› Street" in win.windowTitle() and win.film_info.text() == "› Street"
    assert win.tree.nodes and all(n.prompt == "" for n in win.tree.nodes.values())    # its own (empty) clips
    win.add_reference(str(image))                           # a face in scene 2: the film's cast
    flush(app, win)
    win._save_tree()
    win.storyboard.sceneChosen.emit("s1")                   # back to scene 1 (the film folder)
    assert win.project == root and [n.prompt for n in win.tree.nodes.values()] == ["kitchen clip"]
    assert T.cast_of(win.tree) and (root / T.cast_of(win.tree)[0]["images"][0]).is_file()
    assert win.qs.value("project") == str(root)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Night street", True))
    win.rename_scene("s2")
    win._scenes_moved(["s2", "s1"])
    assert [win.storyboard.list.item(i).text().split("\n")[0] for i in range(2)] == ["1  Night street", "2  Scene 1"]
    win.storyboard.sceneChosen.emit("s2")
    copy = tmp_path / "film-copy"
    win.copy_project_to(copy)                               # Save as: the whole film, the same scene open
    wait(app, lambda: win._copying is None, timeout=30)
    assert win.project == (copy / "scenes" / "s2").resolve() and (copy / "scenes" / "s2" / T.REFS_DIR).is_symlink()
    assert (copy / "film.json").is_file() and win.storyboard.list.count() == 2
    win.open_project(root)
    win.storyboard.sceneChosen.emit("s1")
    win.open_project(root)                                  # the film opens the scene worked on last
    assert win.project == root
    win.delete_scene("s1")                                  # the film folder's scene stays
    assert "cannot be deleted" in win.status.text()
    win.storyboard.sceneChosen.emit("s2")
    win.delete_scene("s2")                                  # the open scene: the window moves to another first
    assert win.project == root and win.storyboard.list.count() == 1 and not (root / "scenes" / "s2").exists()


def test_generate_and_play_the_film(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("kitchen a", "kitchen b"))
    root = win.project
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Street", True))
    win.add_scene()
    win.set_image(str(image))
    first = next(iter(win.tree.nodes))
    type_prompt(win, first, "street a")
    flush(app, win)
    win.storyboard.sceneChosen.emit("s1")
    win.node_action("take", ids[0])                         # scene 1 ends at its first clip in the film
    assert win.film.scene("s1").take == ids[0] and "ends at #" in win.storyboard.list.item(0).text()
    win.storyboard.joinWanted.emit("s2", "fade")           # the street fades in over the kitchen
    assert win.film.scene("s2").join == "fade" and "crossfade" in win.storyboard.list.item(1).text()
    win.generate_film()
    wait(app, lambda: win._film_queue is None and not win.running, timeout=120)
    assert win.status.text().startswith("The film is made"), win.status.text()
    assert win.state(ids[1]) != "done"                      # after scene 1's ending: not part of the film
    win.join_film("play")
    wait(app, lambda: win.status.text().startswith("Film joined"), timeout=60)
    assert win.played and win.played[-1] == root / "paths" / "film.mp4"


def test_scene_duplicate_notes_and_film_cast(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a clip",))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Two", True))
    win.add_scene()
    win.storyboard.duplicateWanted.emit("s1")
    wait(app, lambda: win._copying is None, timeout=30)          # copied in the background
    assert win.storyboard.list.count() == 3 and "(copy)" in win.storyboard.list.item(1).text()
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *a, **k: ("She opens the door.", True))
    win.storyboard.notesWanted.emit("s1")
    assert win.film.scene("s1").notes == "She opens the door."
    assert "She opens the door." in win.storyboard.list.item(0).toolTip()
    win._dialogs_modeless = True
    win.show_film_cast()
    assert win._film_cast_dialog.windowTitle() == "Cast in the film"
    win._film_cast_dialog.deleteLater()


def test_inspector_and_timeline(app, win, tmp_path, image):
    """The editor layout: the path as a timeline; the clip in focus in the inspector; clicking a clip of the path
    shows it without changing the path; a new clip gets the focus."""
    ids = new_project(win, tmp_path, image, ("one", "two", "three"))
    assert [b.nid for b in win.timeline._blocks] == ids and win.focus == ids[-1]
    assert win.card_stack.currentWidget().nid == ids[-1] and "3 of 3" in win.focus_info.text()
    win.timeline.clicked.emit(ids[0])                      # a block: shown, the path stays
    assert win.card_stack.currentWidget().nid == ids[0] and [c.nid for c in win.cards] == ids
    assert not win.b_prev_clip.isEnabled() and win.b_next_clip.isEnabled()
    win._step_focus(1)
    assert win.focus == ids[1]
    win._graph_clicked(ids[2])                             # a clip of the path in the graph: focus only
    assert win.focus == ids[2] and win.selected_id() == ids[2]
    win._graph_clicked(ids[0])
    assert win.focus == ids[0] and [c.nid for c in win.cards] == ids    # (the path is not cut short)
    win.node_action("continue", ids[2])                    # a new clip: in focus at once, to write its prompt
    assert win.focus == win.selected_id() not in ids and win.card_stack.currentWidget().nid == win.focus
    assert win.timeline._blocks[-1].nid == win.focus and win.inspector.tabText(0) == "Clip"


def test_layout_presets_and_saved_layout(app, win):
    """View presets size the panels; the layout is saved with the settings and restored; an unknown one is reset."""
    from video_chain.ui.main_window import LAYOUT_VERSION
    win.resize(1600, 900)
    win.apply_layout("watch")
    assert not win.act_left.isChecked() and win.left_tabs.isHidden()
    win.apply_layout("write")
    assert win.act_left.isChecked() and not win.left_tabs.isHidden()
    sizes = win.split.sizes()
    assert sizes[2] > sizes[0]                       # a wide inspector for writing
    win.save_settings()
    d = json.loads(win.qs.value("layout", type=str))
    assert d["v"] == LAYOUT_VERSION and d["left"] is True and len(d["split"]) == 3
    win.apply_layout("watch")
    win._restore_layout()
    assert win.act_left.isChecked() and win.split.sizes()[2] > win.split.sizes()[0]
    win.qs.setValue("layout", "not json")
    win._restore_layout()                            # falls back to the default layout
    assert win.act_left.isChecked()
    for bad in ("[]", "5", '{"v": 1, "split": ["a", 1, 1], "center": [1, 1]}',
                '{"v": 1, "split": [0, 0, 0], "center": [1, 1]}'):
        win.apply_layout("watch")
        win.qs.setValue("layout", bad)
        win._restore_layout()
        assert win.act_left.isChecked(), bad
    # hidden, saved, restored and shown again: the left panel gets its width back
    win.show()
    app.processEvents()
    win.apply_layout("edit")
    app.processEvents()
    win.act_left.setChecked(False)
    win.save_settings()
    win._restore_layout()
    assert not win.act_left.isChecked()
    win.act_left.setChecked(True)
    app.processEvents()
    assert win.split.sizes()[0] >= win.left_tabs.minimumWidth()
    win.act_left.setChecked(False)
    win._show_left_tab(1)                            # a tab chosen from the menu shows the panel
    assert win.act_left.isChecked() and win.left_tabs.currentIndex() == 1
    win.hide()


def test_audio_tracks_in_the_window(app, win, tmp_path, image):
    """Import and cut an audio track; a clip plays it, the next one carries it on; renamed / removed."""
    import subprocess
    from PySide6.QtWidgets import QDialogButtonBox
    from video_chain.ui.audio_editor import AudioRangeDialog
    src = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=500:d=12", str(src)], check=True)
    dlg = AudioRangeDialog(None, file=str(src), start=2.0, end=3.0, player=False)
    assert abs(dlg.duration - 12.0) < 0.1 and dlg.start == 2.0 and dlg.name.text() == "song"
    assert not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()                  # under 2 s
    dlg.set_range(2.0, 9.0)
    assert dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    wait(app, lambda: bool(dlg.bar.peaks))                                    # the waveform, made in the background
    dlg.done(0)
    ids = new_project(win, tmp_path, image, ("one", "two"))
    tid = win.add_track(str(src), 2.0, 9.0, "Song")
    assert tid and win.track_list.count() == 1 and "Song" in win.track_list.item(0).text()
    flush(app, win)
    first = next(c for c in win.cards if c.nid == ids[0])
    k = first.src_sound.findData(f"track:{tid}")
    assert k >= 0 and first.sound_box.isVisibleTo(first)                       # offered on every clip
    first.src_sound.setCurrentIndex(k)
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["source_sound"] == f"track:{tid}"
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.src_sound.currentData() == f"track:{tid}" and "from its beginning" in first.src_sound_info.text()
    assert first.sound_over.isVisibleTo(first) and not first.sound_over.isChecked()
    first.sound_over.setChecked(True)                                         # full motion, the sound laid over
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides.get("sound_over") is True
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.sound_over.isChecked() and "laid over" in first.settings_toggle.text()
    assert "<Audio" not in first.edit.toolTip()
    assert "Laid over" in first.src_sound_info.text()
    first.set_editable(False)                                                 # (during a run)
    assert not first.sound_over.isEnabled()
    first.set_editable(True)
    first.src_sound.setCurrentIndex(0)                                        # no sound: the box goes off too
    assert not first.sound_over.isChecked()
    first.src_sound.setCurrentIndex(first.src_sound.findData(f"track:{tid}"))
    flush(app, win)
    assert "sound_over" not in win.tree.nodes[ids[0]].overrides
    second = next(c for c in win.cards if c.nid == ids[1])
    second.src_sound.setCurrentIndex(second.src_sound.findData(f"track:{tid}"))
    flush(app, win)
    second = next(c for c in win.cards if c.nid == ids[1])
    assert f"carries it on from clip #{ids[0][1:]}" in second.src_sound_info.text()
    assert T.TRACK_SOUND_TEXT.format(k=1) in second.edit.toolTip()
    assert "1 clip" not in win.track_list.item(0).text() and "2 clips" in win.track_list.item(0).text()
    old_file = T.track_of(win.tree, tid)["file"]
    assert win.recut_track(tid, str(src), 2.0, 9.0, "Song 2")                # only the name
    assert T.track_of(win.tree, tid)["name"] == "Song 2" and T.track_of(win.tree, tid)["file"] == old_file
    assert win.recut_track(tid, str(src), 1.0, 9.0)                           # another part: the old file goes
    assert not (win.project / old_file).exists() and (win.project / T.track_of(win.tree, tid)["file"]).exists()
    win.track_list.setCurrentRow(0)
    monkey = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        win._remove_track()
    finally:
        QMessageBox.question = monkey
    flush(app, win)
    assert win.track_list.count() == 0
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.src_sound.currentText() == "Audio track: (removed)" and "was removed" in first.src_sound_info.text()


def test_start_node_opens_the_project_tab(app, win, tmp_path, image):
    """A click on the Start node shows the start image settings (the inspector's Project tab)."""
    from video_chain.ui.graph import START
    ids = new_project(win, tmp_path, image, ("one",))
    assert win.inspector.currentIndex() == 0
    win._graph_clicked(START)
    assert win.inspector.currentWidget() is win.proj_scroll
    win._graph_clicked(ids[0])                               # a clip: back to the Clip tab
    assert win.inspector.currentIndex() == 0


def test_paint_a_mask_and_inpaint(app, tmp_path, image, monkeypatch):
    """Paint a mask in the edit window: only with Qwen-Image 2.1; the edit sends it (inpaint), the result keeps the
    source's size, the mask is kept with it and comes back with Prompt again; a new source drops it."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QImage
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy
    from video_chain.ui.main_window import MainWindow
    from video_chain.ui.mask_painter import MaskPainterDialog, mask_area
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "inpaint.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        dlg = w.edit_image(start=True, dialog=False)
        wait(app, lambda: dlg.families, timeout=20)
        assert dlg.source and dlg.mask is None and dlg.b_mask.isEnabled() and not dlg.b_mask_clear.isVisible()
        from video_chain.ui.image_edit import upright
        src = upright(dlg.source)
        painter = MaskPainterDialog(dlg, src)
        c = painter.canvas
        assert c.mask.size() == src.size() and mask_area(c.mask) == 0
        c.set_brush(20)
        c._undo.append(c.mask.copy())
        c._dab(QPointF(10, 10), QPointF(src.width() / 2, src.height() / 2), False)
        painted = mask_area(c.mask)
        assert 0 < painted < 0.5
        c.invert()
        assert mask_area(c.mask) > 0.5
        c.undo()
        assert abs(mask_area(c.mask) - painted) < 1e-9
        from video_chain.ui.mask_painter import overlay
        shown = overlay(src, c.mask)                               # red only where painted
        inside, outside = shown.pixelColor(5, 5), shown.pixelColor(src.width() - 3, src.height() - 3)
        assert inside != src.pixelColor(5, 5) and outside == src.pixelColor(src.width() - 3, src.height() - 3)
        dot = QImage(src.size(), QImage.Format_Grayscale8)
        dot.fill(0)
        dot.setPixel(src.width() - 1, src.height() - 1, 0xFFFFFFFF)  # one pixel: still a mask
        assert mask_area(dot) > 0
        dlg.set_mask(c.mask)
        painter.done(0)
        dlg.edit_model.setCurrentIndex(dlg.edit_model.findData("2511"))
        dlg.prompt.setPlainText("a red scarf")
        dlg.tries.setValue(1)
        assert not dlg.b_gen.isEnabled() and "Qwen-Image 2.1" in dlg.mask_info.text()
        dlg.edit_model.setCurrentIndex(dlg.edit_model.findData("qwen21"))
        assert dlg.b_gen.isEnabled() and "drawn anew" in dlg.mask_info.text()
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        png, meta = dlg.selected()
        assert meta["inpaint"]["patch"].startswith("Qwen-Image-2.1-Fun") and server.edited["inpaint"]
        m = QImage(meta["mask"])
        assert m.size() == src.size() and abs(mask_area(m) - painted) < 0.02
        dlg.set_mask(None)
        assert dlg.mask is None and not dlg.b_mask_clear.isVisible()
        dlg._prompt_again()                                        # the mask of that result comes back
        assert dlg.mask is not None and abs(mask_area(dlg.mask) - painted) < 0.02
        other = tmp_path / "other.png"
        QImage(64, 48, QImage.Format_RGB32).save(str(other))
        dlg.set_source(str(other))                                 # another image: no mask
        assert dlg.mask is None
        dlg.done(0)
        w.close()
    finally:
        server.close()


def test_replace_a_person_in_a_real_video(app, tmp_path, image, monkeypatch):
    """Replace a person: the head found and shown, the new person's picture, Generate (ComfyUI), Use this: the video
    plays the new clip (converted again); its menu goes back to the original."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy
    from video_chain import swap
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "swap.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        src = make_real_video(tmp_path / "real.mp4", secs=2.0)
        w.open_project(tmp_path / "proj")
        v = w.add_video(None, str(src), 0.0, 1.5)
        wait(app, lambda: not w._converts)
        assert w.state(v) == "done"
        node = w.tree.nodes[v]
        n = swap.probe(w.project / node.made["video"])[2]

        def heads(video, model, scale=1.0, progress=None, cancelled=lambda: False):
            tr = [(100, 40, 140, 90)] * n
            return {"boxes": swap.head_boxes(tr, *swap.probe(video)[:2], scale), "found": n, "frames": n,
                    "size": swap.probe(video)[:2], "fps": 24.0, "track": tr}
        monkeypatch.setattr(swap, "find_heads", heads)
        monkeypatch.setattr(w, "face_detector", lambda: tmp_path / "det.onnx")
        dlg = w.replace_person(v, dialog=False)
        wait(app, lambda: dlg.heads is not None, timeout=20)
        assert f"found in {n} of {n} frames" in dlg.found_text.text() and not dlg.b_gen.isEnabled()
        dlg.head_size.setValue(1.3)                                    # larger head: the boxes grow
        assert dlg.boxes[0][2] - dlg.boxes[0][0] > 40 * 1.3
        dlg.set_reference(str(image))
        assert dlg.b_gen.isEnabled()
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.result is not None, dlg.state.text()
        out, meta, frames = dlg.result
        assert frames == n and out.is_file() and swap.probe(out)[2] == n and dlg.b_use.isEnabled()
        assert server.animated["ref"]["inputs"]["image"] and "dw_pose" in server.animated
        assert "stored for the next try" in dlg.state.text(), dlg.state.text()
        assert len(list((w.project / swap.POSE_CACHE_DIR).glob("*/info.json"))) == 1
        # the result in this window: before | after, frame by frame (not in the main window's preview)
        p = dlg.player
        wait(app, lambda: p.result is not None, timeout=20)
        assert p.count() == n and len(p.result) == n and p.view.after is not None
        assert not dlg.show_head.isChecked()                           # the head area off for comparing
        dlg.b_play.click()
        assert p.playing() and p.position() == 0
        p.step(1)                                                      # a step pauses
        assert not p.playing() and p.position() == 1 and p.time.text().startswith(f"frame 2 / {n}")
        p.b_fwd.click()
        assert p.position() == 25
        p.b_back.click()
        p.b_prev.click()
        assert p.position() == 0
        dlg.generate()                                                 # again: DWPose is not run again
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.result is not None and "dw_pose" not in server.animated, dlg.state.text()
        assert "stored motion used: face, pose" in dlg.state.text()
        out, meta, frames = dlg.result
        # H3 head swap: the whole clip as the guide, the picture as <Picture 1>, the LoRA, its prompt; the result
        # cut to the clip's frames with the clip's own sound
        import subprocess

        from PySide6.QtGui import QImage
        calls, seen_ref = [], []

        class Gen:
            def generate(self, prompt, start, out_video, out_last, seed, settings=None):
                calls.append((prompt, start, settings))
                ref_img = QImage(settings["ref_paths"][0])                 # letterboxed to 256 x 256 on black
                assert (ref_img.width(), ref_img.height()) == (256, 256)
                seen_ref.append((ref_img.width(), ref_img.height(), ref_img.pixelColor(128, 3).black() == 255
                                 and ref_img.pixelColor(128, 3).lightness() == 0))
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                                f"color=green:s={settings['width']}x{settings['height']}:r=24",
                                "-frames:v", str(settings["frames"]), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                str(out_video)], check=True)
            def interrupt(self):
                pass
        dlg.h3 = {"lora": "/loras/MiniMax H3/minimax_h3_head_swap_v1.0_r32.safetensors",
                  "turbo": "/loras/MiniMax H3/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",
                  "settings": {"steps": 6, "turbo": {"lora": "/t.safetensors"}, "pika": {"strength": 1.0}},
                  "connect": lambda progress: Gen()}
        # a 16:9 photo: the head square found in it is what the H3 head swap gets (not the whole wide picture)
        from video_chain import faces as F
        wide = tmp_path / "wide.png"
        wide_img = QImage(1600, 900, QImage.Format_RGB32)
        wide_img.fill(0xC86432)                                        # (no black anywhere: bars would show)
        wide_img.save(str(wide))
        monkeypatch.setattr(F, "available", lambda model: None)
        monkeypatch.setattr(F, "detect", lambda png, model: [(700.0, 300.0, 820.0, 450.0, 0.9)])
        dlg.set_reference(str(wide))
        assert dlg.ref_crop is not None and dlg.ref_crop[2] == round(150 * swap.HEAD_CROP_MARGIN)
        dlg.method.setCurrentIndex(dlg.method.findData("h3"))
        assert dlg.head_crop.isVisibleTo(dlg) and dlg.head_crop.isChecked() and "head square" in dlg.crop_info.text()
        from video_chain.ui.swap_dialog import letterbox
        sq = tmp_path / "sq.png"
        assert letterbox(str(wide), sq, 256, dlg.ref_crop) and QImage(str(sq)).size().width() == 256
        dlg.head_crop.setChecked(False)
        assert "whole picture" in dlg.crop_info.text()
        dlg.head_crop.setChecked(True)
        dlg.h3["describe"] = lambda pic: "Male, oval face, short black hair"   # (the vision model)
        dlg.method.setCurrentIndex(dlg.method.findData("h3"))
        dlg.prompt.setText("")                                         # empty: described automatically
        assert dlg.b_gen.isEnabled() and "face" in dlg.prompt.placeholderText()
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.result is not None, dlg.state.text()
        prompt, start, st = calls[-1]
        assert start is None and st["guides"] == [{"video": str(dlg.clip), "frame_idx": 0, "audio": True,
                                                    "at": "start"}]
        assert st["frames"] == swap.h3_length(n) >= n and st["lora"].endswith("dmd_ref2va_8step_turbo_pruned.safetensors")
        assert st["extra_loras"] == [{"lora": dlg.h3["lora"], "strength": 1.0}]       # the turbo first, then the swap
        assert st["recipe"] == {"sampler": "euler", "scheduler": "beta", "shift": [12.0, 3.0]}
        assert st["ref_paths"][0].endswith("_head_ref.png") and st["steps"] == 8 and not st["ref_hint"]
        assert seen_ref and seen_ref[-1] == (256, 256, False)              # the head square, no black bars
        assert not Path(st["ref_paths"][0]).exists()                  # (a temporary copy, removed after)
        assert "turbo" not in st and "pika" not in st and "hyperflow" not in st     # (the project's speed modes)
        assert prompt.startswith("head_swap:") and "Male, oval face, short black hair" in prompt
        assert dlg.prompt.text() == "Male, oval face, short black hair"   # shown (and editable) for the next try
        h3_out = dlg.result[0]
        assert swap.probe(h3_out)[2] == n and "H3" in dlg.state.text()
        has_sound = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                    "stream=index", "-of", "csv=p=0", str(h3_out)], capture_output=True,
                                   text=True).stdout.strip()
        assert has_sound                                               # the clip's own sound
        dlg.set_reference(str(wide))                                   # the same picture: its description stays
        assert dlg.prompt.text() == "Male, oval face, short black hair"
        dlg.set_reference(str(image))                                  # another picture: its own description
        assert dlg.prompt.text() == ""
        dlg.h3["turbo"] = None
        dlg._update()
        assert not dlg.b_gen.isEnabled() and "DMD 8-step" in dlg.b_gen.toolTip()
        dlg.h3["lora"] = None
        dlg._update()
        assert not dlg.b_gen.isEnabled() and "head swap LoRA" in dlg.b_gen.toolTip()
        dlg.done(0)
        # without a face detector: Face only and H3 still work (the clip's own frames), the Animate methods not
        monkeypatch.setattr(w, "face_detector", lambda: None)
        nodet = w.replace_person(v, dialog=False)
        assert nodet.heads is not None and nodet.heads["frames"] == n and not nodet.boxes
        nodet.set_reference(str(image))
        nodet.method.setCurrentIndex(nodet.method.findData("face"))
        assert nodet.b_gen.isEnabled()
        nodet.method.setCurrentIndex(nodet.method.findData("animate"))
        assert not nodet.b_gen.isEnabled() and "head marked" in nodet.b_gen.toolTip()
        nodet.done(0)
        assert w.use_replacement(v, out, meta, frames)
        node = w.tree.nodes[v]
        assert swap.replaced(node)["original"]["file"] == str(src) and node.source["file"] == str(out.resolve())
        wait(app, lambda: not w._converts)
        assert w.state(v) == "done", w.status.text()                  # converted again
        assert w.undo_replacement(v) and w.tree.nodes[v].source["file"] == str(src)
        wait(app, lambda: not w._converts)
        w.close()
    finally:
        server.close()


def test_image_picker_shows_the_projects_pictures(app, win, tmp_path, image):
    """The picker's "This project" section: the project's pictures by kind, with captions; the collection used
    last opens again; folders can still be browsed."""
    from PySide6.QtCore import Qt

    from video_chain.ui.image_picker import ImagePickerDialog
    ids = new_project(win, tmp_path, image, ("one", "two"))
    run_and_wait(app, win, win.generate_path)
    edits = win.project / "edits"
    edits.mkdir(exist_ok=True)
    Image.new("RGB", (64, 64), (9, 9, 9)).save(edits / "edit_1.png")
    (edits / "edit_1.json").write_text('{"prompt": "she smiles at the camera", "mode": "edit"}')
    cols = win.project_images()
    names = [n for n, _items in cols]
    assert names[0] == "All project images" and "Start image" in names and "Clip frames" in names
    assert "Image edits & new images" in names
    clips = dict(cols)["Clip frames"]
    assert [c for _p, c in clips] == [f"Clip #{ids[0][1:]} · last frame", f"Clip #{ids[1][1:]} · last frame"]
    assert dict(cols)["Image edits & new images"] == [(str(edits / "edit_1.png"), "Edit · she smiles at the camera")]
    q = settings(tmp_path, "pick2.ini")
    dlg = ImagePickerDialog(None, "Start image", None, q, cache_dir=None, desktop_thumbs=None, collections=cols)
    assert dlg.collection == "All project images" and dlg.path.text() == "This project › All project images"
    places = [dlg.places.item(i).text() for i in range(dlg.places.count())]
    assert places[0] == "This project" and places[1].startswith("All project images (") and "Folders" in places
    dlg.open_collection("Clip frames")
    texts = [dlg.grid.item(i).text() for i in range(dlg.grid.count())]
    assert texts == [c for _p, c in clips]
    dlg.grid.setCurrentRow(1)
    assert dlg.info.text().startswith(f"Clip #{ids[1][1:]} · last frame")
    dlg._activated(dlg.grid.item(1))
    assert dlg.chosen() == clips[1][0] and q.value("pickerPlace") == "col:Clip frames"
    again = ImagePickerDialog(None, "x", None, q, cache_dir=None, desktop_thumbs=None, collections=cols)
    assert again.collection == "Clip frames"                                   # the collection used last
    again.go_up()                                                              # to the folders
    assert again.collection is None and again.grid.count() >= 0
    again.reject()
    plain = ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None)
    assert plain.collection is None and plain.places.item(0).text() == "Home"   # no project: folders only
    plain.reject()
    wait(app, lambda: not win._joins, timeout=30)                              # (the path joined after generating)
    for d in (dlg, again, plain):                                              # as pick_image does
        d._pool.shutdown(wait=True)
        d.deleteLater()
    app.processEvents()


def test_find_lora_in_both_lists_and_on_disk(app, win, tmp_path, monkeypatch):
    """A LoRA is found among the H3 and the turbo LoRAs (a turbo LoRA is listed as one), else in ComfyUI's folders."""
    from video_chain import comfy as C
    win.h3_loras = [("/l/MiniMax H3/minimax_h3_head_swap_v1.0_r32.safetensors", "MiniMax H3/minimax_h3_head_swap_v1.0_r32")]
    win.turbo_loras = [("/l/MiniMax H3/minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors",
                        "MiniMax H3/minimax_h3_dmd_ref2va_8step_turbo_pruned")]
    assert win.find_lora("head_swap").endswith("head_swap_v1.0_r32.safetensors")
    assert win.find_lora("minimax_h3_dmd_ref2va_8step_turbo_pruned").endswith("turbo_pruned.safetensors")
    root = tmp_path / "loras"
    (root / "new").mkdir(parents=True)
    (root / "new" / "Fresh_LoRA.safetensors").write_bytes(b"x")
    monkeypatch.setattr(C, "comfy_lora_roots", lambda comfy_dir: [root])
    assert win.find_lora("fresh_lora") == str(root / "new" / "Fresh_LoRA.safetensors")   # added after the scan
    assert win.find_lora("nothing_like_this") is None


def test_compare_player_frames_and_divider(app, tmp_path):
    """The before / after player: every frame of both clips, exact; the divider follows the mouse."""
    import subprocess

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from video_chain.ui.compare_player import ComparePlayer, store_size
    assert store_size(832, 480, 107) == (832, 480)                       # small clips at their own size
    w, h = store_size(1920, 1080, 2000)
    assert w <= 1280 and 2 * 2000 * w * h * 3 <= 900 * 2**20 and abs(w / h - 16 / 9) < 0.02
    clips = []
    for name, colour in (("before", "red"), ("after", "blue")):
        f = tmp_path / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color={colour}:s=64x36:r=24:d=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f)], check=True)
        clips.append(f)
    p = ComparePlayer()
    p.resize(640, 420)
    p.load(clips[0], 24, (64, 36))
    wait(app, lambda: p.count() == 24, timeout=20)
    assert p.b_play.isEnabled() and p.view.after is None
    p.load_result(clips[1])
    wait(app, lambda: p.result is not None, timeout=20)
    assert len(p.result) == 24
    before, after = p.image(5), p.image(5, True)
    assert before.pixelColor(32, 18).red() > 200 and after.pixelColor(32, 18).blue() > 200
    p.show()
    r = p.view.picture_rect()
    QTest.mouseClick(p.view, Qt.LeftButton, pos=QPoint(int(r.left() + r.width() * 0.25), int(r.center().y())))
    assert abs(p.view.split - 0.25) < 0.02
    p.seek(23)
    p.play()                                                             # at the end: from the start again
    assert p.position() == 0 and p.playing()
    wait(app, lambda: not p.playing(), timeout=10)                       # plays through and stops at the end
    assert p.position() == 23
    p.load(clips[1], 24, (64, 36))                                       # another clip: the result is gone
    assert p.result is None and p.count() == 0
    wait(app, lambda: p.count() == 24, timeout=20)
    p.stop_loading()
    p.close()
    p.deleteLater()


def test_new_clips_get_random_seeds_and_the_seed_button(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b", "c"))
    seeds = [win.tree.nodes[i].seed for i in ids]
    assert len(set(seeds)) == 3 and 42 not in seeds and all(1 <= s <= T.SEED_MAX for s in seeds)
    card = next(c for c in win.cards if c.nid == ids[1])
    before = card.seed.value()
    card.b_seed.click()                                                  # a new random seed, saved with the clip
    assert card.seed.value() != before and win.tree.nodes[ids[1]].seed == card.seed.value()
    assert card.seed.value() not in (seeds[0], seeds[2])
    card.set_editable(False)
    assert not card.b_seed.isEnabled()
    card.set_editable(True)


def test_veda_setting_and_marks(app, win, tmp_path, image):
    """Veda is on by default, used only for large clips, and every clip that gets it says so."""
    ids = new_project(win, tmp_path, image, ("a",))
    assert win.veda.isChecked() and "Veda not used at this size" in win.size_info.text()     # 672x384: never
    node = win.tree.nodes[ids[0]]
    assert not win.veda_used(node) and "Veda" not in win.badge(node)
    win.resolution.setCurrentIndex(win.resolution.findData(768))                            # 1024x576
    assert "⚡ Veda for clips from" in win.size_info.text()
    node.overrides = {"frames": 241}                                                        # 10 s: large enough
    win.refresh()
    assert win.veda_used(node) and "⚡ Veda" in win.badge(node)
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.veda_mark.isVisibleTo(card) and "Veda" not in card.settings_toggle.text()
    assert T.Checker(win.project, win.tree).settings(ids[0])["veda"] is True
    win.veda.setChecked(False)                                                              # off: saved, no mark
    assert win.tree.options["veda"] == {"on": False} and not win.veda_used(node)
    win.refresh()
    card = next(c for c in win.cards if c.nid == ids[0])
    assert not card.veda_mark.isVisibleTo(card)
    assert "Veda" not in win.size_info.text() and "veda" not in T.Checker(win.project, win.tree).settings(ids[0])


def test_preview_paints_frames_without_a_native_video_window(app, tmp_path, monkeypatch):
    """The players show frames through a QVideoSink painted in a normal widget: QVideoWidget's own GPU window
    crashed in NVIDIA's EGL driver (segfault) while ComfyUI held the GPU memory."""
    import subprocess

    from video_chain.ui.main_window import Preview
    from video_chain.ui.widgets import VideoView
    monkeypatch.undo()                                                   # (this test plays: a clip without sound,
    clip = tmp_path / "c.mp4"                                            # the player muted as well)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=320x180:r=24:d=2", "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    p = Preview()
    p.resize(640, 420)
    p.show()
    p._ensure_player()
    p.audio.setMuted(True)
    p.open(clip, "clip")
    assert isinstance(p.video, VideoView)
    wait(app, lambda: not p.video.image().isNull(), timeout=20)
    assert p.video.image().width() == 320
    p.video.repaint()                                                    # (paints the frame scaled into the view)
    wait(app, lambda: p.player.position() >= 1900, timeout=20)
    for _ in range(50):
        app.processEvents()
    assert not p.video.image().isNull()                                  # the end: the last frame stays
    p.stop()
    assert p.video.image().isNull()
    p.close()
    p.deleteLater()
    import video_chain.ui as ui
    for f in Path(ui.__file__).parent.glob("*.py"):                      # no QVideoWidget anywhere in the app
        assert "QVideoWidget()" not in f.read_text(encoding="utf-8"), f.name


def test_export_one_clip(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    assert not win.b_export_clip.isEnabled()                               # not made yet
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    win.select(ids[0])
    flush(app, win)
    assert win.focus == ids[0] and win.b_export_clip.isEnabled()
    node = win.tree.nodes[ids[0]]
    out = tmp_path / "clip-a.mp4"
    win.export(out, "made", single=ids[0])                                 # as made: the clip itself
    assert out.read_bytes() == (win.project / node.made["video"]).read_bytes()
    assert win.status.text() == f"Exported clip-a.mp4 (clip #{ids[0][1:]}, as made)."
    up = tmp_path / "clip-a-2k.mp4"
    win.export(up, "2k", single=ids[0])                                    # upscaled: that clip only
    assert win.running
    wait(app, lambda: not win.running, timeout=120)
    assert chain.probe(up)["frames"] == chain.probe(win.project / node.made["video"])["frames"]
    assert win.status.text().startswith(f"Exported {up.name} (2k, SeedVR2 7B; 1 clip upscaled now"), win.status.text()
    # the dialog: titled for the clip, its default file named after it
    from video_chain.ui import main_window as M
    seen = {}

    def fake_exec(dlg):
        seen.update(title=dlg.windowTitle(), file=dlg.file.text())
        return QDialog.Rejected
    from PySide6.QtWidgets import QDialog
    monkeypatch.setattr(M.ExportDialog, "exec", fake_exec)
    win.export_path(ids[1])
    assert seen["title"] == f"Export clip #{ids[1][1:]}" and seen["file"].endswith(f"-{ids[1]}-clip.mp4")
    # the clip's menu: "Export this clip…" opens the same dialog for that clip
    called = []
    monkeypatch.setattr(win, "export_path", lambda nid=None: called.append(nid))
    monkeypatch.setattr(win, "_exec_menu", lambda m, pos: next(a for a in m.actions() if a.text() == "Export this clip…"))
    from PySide6.QtCore import QPoint
    win._context(ids[1], QPoint(0, 0))
    wait(app, lambda: called, timeout=5)
    assert called == [ids[1]]


def test_progress_says_what_runs_and_how_long_it_still_takes(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import fmt_left, time_left
    assert fmt_left(4) == "a few seconds" and fmt_left(42) == "about 40 s" and fmt_left(185) == "about 3 min"
    assert fmt_left(3900) == "about 1 h 05 min"
    assert time_left(100, 0.01) is None and time_left(1, 0.5) is None and time_left(60, 0.25) == 180
    ids = new_project(win, tmp_path, image, ("a", "b"))
    clock = [1000.0]
    monkeypatch.setattr("video_chain.ui.main_window.time.monotonic", lambda: clock[0])
    win._job, win._plan = {"refine": {"size": "2k"}}, list(ids)            # (an export of 2 clips)
    win._run_started = clock[0]
    win._event({"e": "node", "id": ids[0], "state": "generating"})
    win._event({"e": "note", "id": ids[0], "text": "SeedVR2 piece 2 of 5"})
    clock[0] += 60
    win._event({"e": "step", "id": ids[0], "i": 250, "n": 1000})          # a quarter of clip 1 after 60 s
    text = win.status.text()
    assert text.startswith(f"Clip #{ids[0][1:]} (1 of 2): SeedVR2 piece 2 of 5 · 25 %"), text
    assert "about 3 min left" in text and "(all clips: about 7 min)" in text, text
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.step.text() == "25 % · about 3 min left"
    win._job = None


def test_video_view_shrinks_frames_without_aliasing(app):
    """A 2K portrait in a small preview: shrunk by area averaging, so fine detail does not turn into shimmering
    grain (a painter's bilinear scaling skips most source pixels)."""
    import numpy as np
    from PySide6.QtGui import QImage

    from video_chain.ui.widgets import VideoView
    board = (np.indices((2112, 1696)).sum(axis=0) % 2 * 255).astype(np.uint8)       # 1-px checkerboard
    rgb = np.ascontiguousarray(np.repeat(board[..., None], 3, axis=2))
    img = QImage(rgb.data, 1696, 2112, 1696 * 3, QImage.Format_RGB888).copy()
    v = VideoView()
    v.resize(200, 250)
    v._img = img
    small = v.scaled_frame()
    assert small.height() <= 250 * v.devicePixelRatioF() and small.width() < 1696
    g = small.convertToFormat(QImage.Format_Grayscale8)
    a = np.frombuffer(g.constBits(), np.uint8).reshape(g.height(), g.bytesPerLine())[:, :g.width()].astype(float)
    assert abs(a.mean() - 127.5) < 6 and a.std() < 8                                  # an even grey, no false pattern
    assert v.scaled_frame() is small                                                    # (once per frame and size)
    v.deleteLater()


def test_camera_row_and_hash_tokens(app, win, tmp_path, image):
    from PySide6.QtGui import QTextCursor
    ids = new_project(win, tmp_path, image, ("She waits",))
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.cam_move.currentData() is None and not card.cam_amp.isEnabled()
    card.cam_move.setCurrentIndex(card.cam_move.findData("push-in"))
    assert card.cam_amp.isEnabled() and card.cam_speed.isEnabled()
    card.cam_amp.setCurrentIndex(card.cam_amp.findData("small"))
    card.cam_speed.setCurrentIndex(card.cam_speed.findData("slow"))
    flush(app, win)
    node = win.tree.nodes[ids[0]]
    assert node.overrides["camera"] == {"move": "push-in", "amplitude": "small", "speed": "slow"}
    assert "🎥 push in · small · slow" in win.badge(node)
    card = next(c for c in win.cards if c.nid == ids[0])
    assert "camera pushes in with small amplitude at slow speed" in card.edit.toolTip()   # (sent to the model)
    card.cam_move.setCurrentIndex(card.cam_move.findData("static"))                     # no amplitude / speed
    assert not card.cam_amp.isEnabled() and not card.cam_speed.isEnabled()
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["camera"] == {"move": "static"}
    # typing # offers the camera tokens; the chosen one goes in whole
    card = next(c for c in win.cards if c.nid == ids[0])
    card.edit.setFocus()
    card.edit.setPlainText("She waits, #pan-ri")
    card.edit.moveCursor(QTextCursor.End)
    assert card.mentions._typed() is not None and card.mentions._kind == "#"
    card.mentions._insert("pan-right-slow")
    assert card.edit.toPlainText() == "She waits, #pan-right-slow "
    card.edit.setPlainText("Hi @Ann")                                                   # (@ still the cast)
    card.edit.moveCursor(QTextCursor.End)
    assert card.mentions._typed() is not None and card.mentions._kind == "@"


def test_export_shows_the_enlargement_and_size_1024(app, win, tmp_path, image):
    from video_chain.ui.main_window import ExportDialog
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 832), {})       # a portrait draft at size 768
    assert "1280×1600 · 1.9× the draft" in dlg.size.itemText(1) and "1696×2112 · 2.5× the draft" in dlg.size.itemText(2)
    dlg.size.setCurrentIndex(1)
    assert dlg.scale_note.isHidden()                                         # 1.9×: no note
    dlg.size.setCurrentIndex(2)
    assert not dlg.scale_note.isHidden() and "2.5× the draft" in dlg.scale_note.text()
    assert "1080p stays within 2× here" in dlg.scale_note.text() and "H3's native size" in dlg.scale_note.text()
    dlg.reject()
    win.open_project(tmp_path / "proj")
    from video_chain import chain
    k = win.resolution.findData(chain.H3_NATIVE)
    assert k >= 0 and "H3's native size" in win.resolution.itemText(k)


def test_buttons_open_their_windows(app, win, tmp_path, image, monkeypatch):
    """Real clicks (a button's clicked passes checked=False: it must not land in an optional parameter)."""
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    opened = []
    monkeypatch.setattr(M.ExportDialog, "exec", lambda dlg: opened.append(dlg.windowTitle()) or 0)
    win.b_export.click()                                                    # Export path…: the path's dialog
    assert opened == ["Export path"], (opened, win.status.text())
    win.select(ids[0])
    flush(app, win)
    win.b_export_clip.click()
    assert opened[-1] == f"Export clip #{ids[0][1:]}"
    seen = []
    monkeypatch.setattr(win, "edit_image", lambda **kw: seen.append(kw))
    win.b_ref_create.click()                                                # Cast → Create…: the window opens
    assert seen and seen[0]["mode"] == "create" and seen[0]["dialog"] is True


def test_a_project_without_a_start_image(app, win, tmp_path, image):
    """No start image nor video: the Aspect setting gives the size, the first clip starts from its text."""
    win.open_project(tmp_path / "proj")
    nid = win.selected_id()
    type_prompt(win, nid, "a lighthouse at dusk")
    flush(app, win)
    assert win.aspect.isEnabled() and win.aspect.currentData() == "16:9"
    assert "16:9 → 672×384 (Aspect)" in win.size_info.text() and "starts from its prompt" in win.size_info.text()
    win.aspect.setCurrentIndex(win.aspect.findData("9:16"))
    assert win.tree.options["aspect"] == "9:16" and "9:16 → 384×672" in win.size_info.text()
    run_and_wait(app, win, win.generate_path)
    assert win.state(nid) == "done" and win.tree.nodes[nid].made["settings"]["width"] == 384
    win.aspect.setCurrentIndex(win.aspect.findData("1:1"))                   # another format: made again
    flush(app, win)
    assert win.state(nid) == "changed"
    win.set_image(str(image))                                                 # a start image: its format
    assert not win.aspect.isEnabled() and "(Aspect)" not in win.size_info.text()


def test_remove_the_start_image(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    assert win.b_start_clear.isEnabled()
    run_and_wait(app, win, win.generate_path)
    flush(app, win)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
    win.b_start_clear.click()                                                # made clips depend on it: asked first
    assert asked and "1 made clip depends" in asked[0] and (win.project / T.START_FILE).is_file()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    win.b_start_clear.click()
    flush(app, win)
    assert not (win.project / T.START_FILE).is_file() and win.tree.image is None
    assert not list(win.project.glob("start_*x*.png")) and not win.b_start_clear.isEnabled()
    assert win.aspect.isEnabled() and "(Aspect)" in win.size_info.text() and win.state(ids[0]) == "changed"
    assert win.status.text().startswith("Start image removed")


def test_file_browser_for_videos_and_sounds(app, win, tmp_path, image):
    """The app's own browser also for videos (a frame as thumbnail, length and size) and sounds (their icon)."""
    import subprocess

    from PIL import Image as PImage

    from PySide6.QtCore import Qt

    from video_chain.ui.image_picker import ImagePickerDialog
    d = tmp_path / "media"
    d.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=96x54:r=24:d=2", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(d / "clip.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    str(d / "voice.wav")], check=True)
    PImage.new("RGB", (40, 30), (0, 0, 255)).save(d / "pic.png")
    q = settings(tmp_path, "picker.ini")

    def names(dlg):
        return sorted(dlg.grid.item(i).text() for i in range(dlg.grid.count()))
    vids = ImagePickerDialog(None, "Real video", str(d), q, cache_dir=None, desktop_thumbs=None, kind="video")
    assert names(vids) == ["clip.mp4"] and "(1 videos)" in vids.windowTitle()
    item = vids.grid.item(0)
    wait(app, lambda: item.data(Qt.UserRole + 1) == "96×54", timeout=20)          # a frame became the thumbnail
    vids.grid.setCurrentItem(item)
    assert "0:02.0" in vids.info.text() and "96×54" in vids.info.text() and "24 fps" in vids.info.text()
    vids._accept()
    assert vids.chosen() == str(d / "clip.mp4") and q.value("videoDir", type=str) == str(d)   # (its own folder)
    vids.deleteLater()
    media = ImagePickerDialog(None, "Voice", str(d), q, cache_dir=None, desktop_thumbs=None, kind="media")
    assert names(media) == ["clip.mp4", "voice.wav"]
    media.grid.setCurrentItem(next(media.grid.item(i) for i in range(media.grid.count())
                                   if media.grid.item(i).text() == "voice.wav"))
    assert "0:01.0" in media.info.text() and "sound" in media.info.text()
    media.deleteLater()
    # the project's own real videos are offered first
    win.open_project(tmp_path / "proj")
    win.set_image(str(image))
    v = win.add_video(None, str(d / "clip.mp4"), 0.0, 1.0)
    cols = win.project_media("video")
    assert cols == [("Real videos of this project", [(str(d / "clip.mp4"), f"Video #{v[1:]} · clip.mp4")])]


def test_file_browser_favorites(app, tmp_path):
    """Folders as favorites: the star adds / removes the open folder; shared by every kind; right-click removes."""
    from PySide6.QtCore import Qt

    from video_chain.ui.image_picker import ImagePickerDialog
    q = settings(tmp_path, "fav.ini")
    a, b = tmp_path / "shots", tmp_path / "music"
    a.mkdir()
    b.mkdir()
    dlg = ImagePickerDialog(None, "Choose", str(a), q, cache_dir=None, desktop_thumbs=None)

    def places(d):
        return [(d.places.item(i).text(), d.places.item(i).data(Qt.UserRole)) for i in range(d.places.count())]
    assert "Favorites" not in [t for t, _d in places(dlg)] and dlg.b_fav.toolTip().startswith("Add")
    dlg.b_fav.click()                                                          # the open folder: a favorite
    assert dlg.favorites() == [str(a)] and dlg.b_fav.toolTip().startswith("Remove")
    texts = [t for t, _d in places(dlg)]
    assert texts.index("Favorites") < texts.index("shots") < texts.index("Folders")
    dlg.open_folder(b)
    dlg.b_fav.click()
    assert dlg.favorites() == [str(a), str(b)]
    dlg.deleteLater()
    vids = ImagePickerDialog(None, "Video", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="video")
    assert vids.favorites() == [str(a), str(b)]                                 # shared by every kind
    row = next(i for i in range(vids.places.count()) if vids.places.item(i).data(Qt.UserRole) == str(a))
    vids._place_clicked(vids.places.item(row))
    assert vids.folder == a.resolve()
    pos = vids.places.visualItemRect(vids.places.item(row)).center()
    vids._exec_menu = lambda m, p: m.actions()[0]                               # right-click → Remove from favorites
    vids._places_menu(pos)
    assert vids.favorites() == [str(b)]
    vids.deleteLater()


def test_split_a_video_and_insert_a_clip_in_the_window(app, win, tmp_path, monkeypatch):
    from PySide6.QtCore import QPoint

    from video_chain.ui import main_window as M
    src = make_real_video(tmp_path / "real.mp4", secs=4.0)
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")                                            # (a new project's empty first clip)
    v = win.add_video(None, str(src), 0.0, 3.0)
    wait(app, lambda: not win._converts)
    assert win.state(v) == "done"
    # the menu entry opens the dialog; the dialog's frame becomes the moment in the file
    seen = {}

    def fake_exec(dlg):
        wait(app, lambda: dlg.player.count() > 0, timeout=20)
        dlg.player.seek(48)                                                    # frame 48 = 2.0 s into the range
        seen["text"] = dlg.summary.text()
        seen["at"] = dlg.at()
        return QDialog.Accepted
    from PySide6.QtWidgets import QDialog
    monkeypatch.setattr(M.SplitDialog, "exec", fake_exec)
    monkeypatch.setattr(win, "_exec_menu", lambda m, pos: next(a for a in m.actions()
                                                                if a.text() == "Split here and insert a clip…"))
    win._context(v, QPoint(0, 0))
    wait(app, lambda: len(win.tree.nodes) == 3, timeout=10)
    assert seen["at"] == 2.0 and "Part 1: 0.00–2.00 s" in seen["text"]
    clip = win.selected_id()
    rest = win.tree.video_child(clip)
    assert win.tree.nodes[v].source["end"] == 2.0 and rest.source == {"file": str(src), "start": 2.0, "end": 3.0}
    assert "write the prompt of the new clip" in win.status.text()
    wait(app, lambda: not win._converts, timeout=30)
    assert win.state(v) == "done" and win.state(rest.id) == "done" and win.state(clip) == "new"


def test_end_image_from_a_video_frame(app, win, tmp_path, image, monkeypatch):
    """A frame of any video as a clip's end image: exactly the chosen frame, at the video's full size."""
    import subprocess

    from PIL import Image as PImage
    from PySide6.QtWidgets import QDialog

    from video_chain.ui import main_window as M
    from video_chain.ui.clip_dialogs import VideoFrameDialog
    vid = tmp_path / "ramp.mp4"                                             # frame k has the brightness 5k
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=128x72:r=24:d=2",
                    "-vf", "geq=lum='5*N':cb=128:cr=128", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
                    str(vid)], check=True)
    dlg = VideoFrameDialog(None, vid, "frame")
    assert dlg.frames == 48 and dlg.fps == 24
    def exact(k):                                                          # ffmpeg's decode of exactly frame k
        ref = tmp_path / f"ref{k}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",
                        str(ref)], check=True)
        return PImage.open(ref).convert("L").getpixel((64, 36))
    for k in (0, 1, 13, 24, 47):
        dlg.slider.setValue(k)
        out = tmp_path / f"f{k}.png"
        assert dlg.frame_file(out)
        im = PImage.open(out).convert("L")
        assert im.size == (128, 72) and im.getpixel((64, 36)) == exact(k), (k, im.getpixel((64, 36)), exact(k))
    dlg.deleteLater()
    ids = new_project(win, tmp_path, image, ("a",))
    card = next(c for c in win.cards if c.nid == ids[0])
    assert card.b_end_video.isEnabled()

    def fake_exec(d):
        d.slider.setValue(20)
        return QDialog.Accepted
    monkeypatch.setattr(M.VideoFrameDialog, "exec", fake_exec)
    win.end_from_video(ids[0], video=str(vid))
    end = win.tree.nodes[ids[0]].end_image
    assert end and PImage.open(win.project / end).convert("L").getpixel((64, 36)) == exact(20)
    assert "ends at frame 21 (0.83 s) of ramp.mp4" in win.status.text()


def test_civitai_api_key_field(app, win):
    """The settings field keeps the key in the (test's in-memory) keyring, never shows it back, and removes it
    when saved empty."""
    from video_chain import triggers
    win.triggers_online.setChecked(False)
    win.civitai_key.setText(" abc123 ")
    win._save_civitai_key()
    assert triggers.api_key() == "abc123" and win.civitai_key.text() == ""
    assert "kept" in win.civitai_key.placeholderText() and "keyring" in win.status.text()
    win._save_civitai_key()                                   # empty: removed
    assert triggers.api_key() == "" and "none" in win.civitai_key.placeholderText()


def test_lora_presets_in_the_image_window(app, tmp_path, monkeypatch):
    """Named presets per LoRA (prompt, negative, quality, sampling, strength): save, choose, change, rename,
    delete; with two LoRAs the preset names say whose they are."""
    from PySide6.QtCore import QSettings

    from video_chain import imagegen as G
    from video_chain import presets
    from video_chain.ui import image_edit as IE
    a, b = tmp_path / "walk.safetensors", tmp_path / "run.safetensors"
    for f in (a, b):
        f.write_bytes(b"x")
    q = QSettings(str(tmp_path / "pre.ini"), QSettings.IniFormat)
    q.setValue("createLoras_sdxl", json.dumps([{"path": str(a), "strength": 0.5}]))      # remembered from before
    dlg = IE.ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                             connect_create=lambda p: None)
    files = G.pick_files(["krea2/krea2_turbo_fp8_scaled.safetensors"], ["illustrij_v20.safetensors"],
                         ["qwen3vl_4b_fp8_scaled.safetensors"], ["qwen_image_vae.safetensors"], [])
    q.setValue("createFamily", "sdxl")
    dlg._families_found(files, "")
    dlg.set_mode("create")
    assert dlg.loras.items() == [{"path": str(a), "strength": 0.5}]
    assert dlg.b_preset_new.isEnabled(), "the remembered LoRA's presets can be saved at once"
    dlg.loras.set_items([], [])
    assert not dlg.b_preset_new.isEnabled() and "add a LoRA" in dlg.preset.itemText(0)
    dlg.loras.set_items([], [{"path": str(a), "strength": 0.8}])    # (as the remembered LoRAs come in: no changed)
    assert dlg.b_preset_new.isEnabled() and not dlg.b_preset_delete.isEnabled()
    dlg.prompt.setPlainText("a woman walks, w4lk, very long prompt " * 20)
    dlg.negative.setText("blurry")
    dlg.set_sampling({"steps": 30, "cfg": 5.5, "sampler": "dpmpp_2m", "scheduler": "karras"})
    assert dlg.save_preset_as("  Slow   walk ")
    p = presets.store().get(str(a), "Slow walk")
    assert p["prompt"].startswith("a woman walks") and p["negative"] == "blurry" and p["strength"] == 0.8
    assert p["sampling"] == {"steps": 30, "cfg": 5.5, "sampler": "dpmpp_2m", "scheduler": "karras"}
    assert dlg.preset.currentText() == "Slow walk" and dlg.b_preset_delete.isEnabled()
    dlg.prompt.setPlainText("something else")                        # choosing the preset puts it back
    dlg.set_sampling({})
    dlg.loras.set_items([], [{"path": str(a), "strength": 0.3}])
    dlg.apply_preset()
    assert dlg.prompt.toPlainText().startswith("a woman walks") and dlg.sampling()["steps"] == 30
    assert dlg.loras.items()[0]["strength"] == 0.8 and dlg.preset.currentText() == "Slow walk"
    dlg.cfg.setValue(7.0)                                             # edit: saved into the chosen preset
    assert dlg.update_preset() and presets.store().get(str(a), "Slow walk")["sampling"]["cfg"] == 7.0
    assert dlg.rename_preset("Walk, cfg 7") and dlg.preset.currentText() == "Walk, cfg 7"
    assert presets.store().get(str(a), "Slow walk") is None
    dlg.loras.set_items([], [{"path": str(a), "strength": 0.8}, {"path": str(b), "strength": 1.0}])
    assert dlg.save_preset_as("Sprint", lora=str(b))
    texts = [dlg.preset.itemText(i) for i in range(1, dlg.preset.count())]
    assert texts == ["walk: Walk, cfg 7", "run: Sprint"]
    monkeypatch.setattr(IE.QMessageBox, "question", lambda *a, **k: IE.QMessageBox.No)
    assert not dlg.save_preset_as("Sprint", lora=str(b))                    # same name: replace only if confirmed
    assert not dlg.delete_preset()
    monkeypatch.setattr(IE.QMessageBox, "question", lambda *a, **k: IE.QMessageBox.Yes)
    assert dlg.delete_preset() and presets.store().list(str(b)) == []
    assert [dlg.preset.itemText(i) for i in range(1, dlg.preset.count())] == ["walk: Walk, cfg 7"]
    dlg.reject()


def test_video_view_gpu_or_cpu(app, win):
    """The player view draws through Qt's video widget (GPU) or paints frames itself (CPU); the setting is kept
    at once (read at the next start, also after a crash)."""
    from video_chain.ui.widgets import VideoView
    gpu, cpu = VideoView(gpu=True), VideoView(gpu=False)
    assert gpu.widget is not None and gpu.sink is gpu.widget.videoSink() and gpu.image().isNull()
    assert cpu.widget is None and cpu.sink is not None
    gpu.clear()
    gpu.deleteLater()
    cpu.deleteLater()
    win.video_gpu.setChecked(False)
    assert win.qs.value("videoGpu", True, type=bool) is False and "CPU" in win.status.text()
    win.video_gpu.setChecked(True)
    assert win.qs.value("videoGpu", False, type=bool) is True


def test_results_gallery_groups_filters_and_favourites(app, tmp_path):
    """Gallery: results grouped by Generate run (old ones without a run id by prompt and time), filters (★, in
    use, model, prompt search), ★ by key / star corner, Single <-> Gallery, the info card and its seed."""
    import time as _t

    from PySide6.QtCore import QSettings, Qt

    from video_chain.ui import results_gallery as RG
    from video_chain.ui.image_edit import ImageEditDialog
    ed = tmp_path / "edits"
    ed.mkdir()
    now = _t.time()

    def result(name, age, **meta):
        png = ed / f"{name}.png"
        Image.new("RGB", (64, 48), (90, 80, 70)).save(png)
        png.with_suffix(".json").write_text(json.dumps(meta))
        os.utime(png, (now - age, now - age))
        return str(png)
    a1 = result("a1", 10, mode="create", family="krea2", prompt="a red fox in snow", seed=11, run="r1", steps=8, cfg=1.0)
    a2 = result("a2", 12, mode="create", family="krea2", prompt="a red fox in snow", seed=12, run="r1")
    b1 = result("b1", 4000, mode="create", family="sdxl", prompt="1girl, neon alley", seed=21)    # no run id
    b2 = result("b2", 4100, mode="create", family="sdxl", prompt="1girl, neon alley", seed=22)
    c1 = result("c1", 90000, prompt="she smiles", seed=31, uses=[{"kind": "start", "sha1": "x"}])  # an edit
    groups = RG.run_groups([(p, json.loads(Path(p).with_suffix(".json").read_text()), now - age)
                            for p, age in ((a1, 10), (a2, 12), (b1, 4000), (b2, 4100), (c1, 90000))])
    assert [len(g["items"]) for g in groups] == [2, 2, 1]
    assert groups[0]["title"].startswith("Today") and "Krea-2 Turbo" in groups[0]["title"] and "2 results" in groups[0]["title"]
    assert "Edit: " in groups[2]["title"]

    q = QSettings(str(tmp_path / "g.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                          connect_create=lambda p: None)
    dlg.uses = lambda meta: ["the start image"] if meta.get("uses") else []
    dlg._load_results()
    assert dlg.results.count() == 5 and dlg.f_count.text() == "5 results" and dlg.selected()[0] == a1
    assert len(dlg.gallery.headers) == 3 and set(dlg.gallery.thumbs) == {a1, a2, b1, b2, c1}
    assert dlg.gallery.thumbs[c1].in_use and not dlg.gallery.thumbs[a1].in_use
    assert "Krea-2 Turbo" in dlg.res_info.text() and "seed 11" in dlg.res_info.text() and "8 steps" in dlg.res_info.text()
    assert "Prompt: a red fox in snow" in dlg.res_info.toolTip()
    deadline = _t.time() + 5                                         # thumbnails come from the background thread
    while _t.time() < deadline and len(dlg._thumbs) < 5:
        app.processEvents()
    assert len(dlg._thumbs) == 5 and not dlg.gallery.thumbs[a1].pix.isNull()

    dlg.set_view("gallery")
    assert dlg.view_mode() == "gallery" and q.value("resultsView") == "gallery" and dlg.thumb_size.isVisibleTo(dlg)
    dlg.gallery.clicked.emit(b2)                                     # a click selects
    assert dlg.selected()[0] == b2 and dlg.gallery.thumbs[b2].selected
    assert dlg.toggle_favorite() is True                             # F / the button: ★ kept in the record
    assert json.loads(Path(b2).with_suffix(".json").read_text())["favorite"] is True
    from PySide6.QtTest import QTest
    dlg.show()
    th = dlg.gallery.thumbs[a2]
    QTest.mouseClick(th, Qt.LeftButton, pos=th.rect().topRight() + __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(-8, 8))
    for _ in range(5):                                               # a real click on the star corner (crashed)
        app.processEvents()
    assert json.loads(Path(a2).with_suffix(".json").read_text()).get("favorite") is True
    assert a2 in dlg.gallery.thumbs and dlg.gallery.thumbs[a2].meta.get("favorite")   # updated in place
    dlg.f_show.setCurrentIndex(dlg.f_show.findData("fav"))
    assert {dlg.results.item(i).data(Qt.UserRole)[0] for i in range(dlg.results.count())} == {b2, a2}
    assert dlg.f_count.text() == "2 of 5"
    dlg.f_show.setCurrentIndex(dlg.f_show.findData("used"))
    assert dlg.results.count() == 1 and dlg.selected()[0] == c1
    dlg.f_show.setCurrentIndex(0)
    dlg.f_model.setCurrentIndex(dlg.f_model.findData("create:sdxl"))
    assert dlg.results.count() == 2
    dlg.f_model.setCurrentIndex(0)
    dlg.f_search.setText("FOX snow")
    dlg._search_timer.timeout.emit()
    assert dlg.results.count() == 2 and dlg.f_count.text() == "2 of 5"
    dlg.f_search.clear()
    dlg._search_timer.timeout.emit()
    dlg.gallery.opened.emit(b1)                                      # double-click: large, in Single
    assert dlg.view_mode() == "single" and dlg.selected()[0] == b1 and dlg.view.has_image()
    dlg.res_info.use_seed.emit(21)                                   # the seed chip
    assert dlg.seed.value() == 21 and not dlg.fresh_seed.isChecked()
    dlg.res_info.b_prompt.setChecked(True)
    assert dlg.res_info.prompt_box.isVisibleTo(dlg) and "1girl, neon alley" in dlg.res_info.prompt_box.toPlainText()
    dlg.reject()


def test_prompt_variations_in_the_create_window(app, tmp_path, image, monkeypatch):
    """Variations: {name} in the prompt gets a card; values as pills; Generate makes one image per combination
    (the prompts substituted, negative too), the next Generate continues; values are shown with the result,
    "Use its prompt" gives the template back; renamed in the prompt; kept in presets and the settings."""
    from PySide6.QtCore import Qt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_comfy import FakeComfy

    from video_chain import presets
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "var.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        dlg = w.edit_image(start=True, dialog=False, mode="create")
        wait(app, lambda: dlg.families, timeout=20)
        dlg.family.setCurrentIndex(dlg.family.findData("sdxl"))
        dlg.show()                                                           # (offscreen; the negative is sent when shown)
        panel = dlg.variations
        assert not panel.active() and dlg.b_gen.text() == "Generate"
        dlg.prompt.setPlainText("a {animal} in the {place}, {?mood|{mood} light}")
        dlg.negative.setText("blurry, {animal} toy")
        dlg._templates_changed()
        assert panel.names() == ["animal", "place", "mood"]                  # cards for the placeholders
        panel.on.setChecked(True)
        cards = [panel.cards_box.itemAt(i).widget() for i in range(panel.cards_box.count())]
        cards[0].add.setText("cat, dog")                                     # comma list at once
        cards[0].add.returnPressed.emit()
        app.processEvents()
        cards = [panel.cards_box.itemAt(i).widget() for i in range(panel.cards_box.count())]
        cards[1].add_values(["forest"])
        app.processEvents()
        cards = [panel.cards_box.itemAt(i).widget() for i in range(panel.cards_box.count())]
        cards[2].add_values(["warm", ""])
        app.processEvents()
        assert [len(v.values) for v in panel.vars] == [2, 1, 2]
        cards = [panel.cards_box.itemAt(i).widget() for i in range(panel.cards_box.count())]
        assert [c.var.name for c in cards] == ["animal", "place", "mood"]          # the cards, once each
        assert [len(c.pills.items) for c in cards] == [2, 1, 2]
        assert "2 × 1 × 2 = 4 combinations" in panel.summary.text() and panel.active()
        assert dlg.b_gen.text() == "Generate 4 images" and not dlg.tries.isEnabled()
        assert "The next Generate makes 4 images" in panel.preview.toPlainText()
        panel.next.setChecked(True)
        panel.count.setValue(3)
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.results.count() == 3
        made = sorted((m["variation"]["step"], m["prompt"]) for _p, m in (dlg.results.item(i).data(Qt.UserRole)
                                                                          for i in range(3)))
        assert made == [(0, "a cat in the forest, warm light"), (1, "a dog in the forest, warm light"),
                        (2, "a cat in the forest")]
        assert server.created["neg"]["inputs"]["text"] == "blurry, cat toy"     # the negative varies too
        assert "Next: #4 of 4" in panel.position.text()                        # continues where it stopped
        dlg.results.setCurrentRow(0)
        meta = dlg.selected()[1]
        assert meta["template"].startswith("a {animal}") and "animal: cat" in dlg.res_info.text()
        dlg.prompt.setPlainText("something else")
        dlg._prompt_again()
        assert dlg.prompt.toPlainText() == "a {animal} in the {place}, {?mood|{mood} light}"
        panel._remove("rename:place:where")                                   # renamed in the prompt too
        assert "{where}" in dlg.prompt.toPlainText() and panel.variable("where") is not None
        lora = tmp_path / "w.safetensors"
        lora.write_bytes(b"x")
        dlg.loras.set_items([], [{"path": str(lora), "strength": 1.0}])
        assert dlg.save_preset_as("Animals", lora=str(lora))
        assert presets.store().get(str(lora), "Animals")["variations"]["vars"][0]["name"] == "animal"
        st = json.loads(q.value("promptVariations"))
        assert st["on"] and st["vars"][0]["values"][1]["text"] == "dog"
        dlg.reject()
        w.close()
    finally:
        server.close()


def test_variations_panel_review_cases(app, monkeypatch):
    """Double-click + Cancel keeps pill and preview in step; the preview builds only what it shows; the position
    key knows the repeat; done() stores under the planned key."""
    from video_chain import variations as V
    from video_chain.ui import variations_panel as VP
    p = VP.VariationsPanel()
    p.set_templates("{a} {b}")
    p.on.setChecked(True)
    p.variable("a").values = [V.Value(str(k)) for k in range(300)]
    p.variable("b").values = [V.Value(str(k)) for k in range(300)]
    p._rebuild()
    assert p.amount() == 90000 and "… and 89900 more" in p.preview.toPlainText()
    key1 = p.position_key()
    p.repeat.setValue(3)
    assert p.position_key() != key1                                    # repeat is part of the position
    p.done(4, key1)
    assert p.positions[key1] == 5 and p.start() == 0
    small = VP.VariationsPanel()
    small.set_templates("{c}")
    small.on.setChecked(True)
    small.variable("c").values = [V.Value("x"), V.Value("y")]
    small._rebuild()
    card = small.cards_box.itemAt(0).widget()
    pill = card.pills.items[0]
    monkeypatch.setattr(VP.QInputDialog, "getText", lambda *a, **k: ("", False))
    pill.value.on = False                                              # (the first click of the double-click)
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    ev = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(5, 5), QPointF(5, 5), Qt.LeftButton, Qt.LeftButton,
                     Qt.NoModifier)
    pill.mouseDoubleClickEvent(ev)                                     # then Cancel
    app.processEvents()
    assert small.variable("c").values[0].on and small.amount() == 2 and "makes 2 images" in small.preview.toPlainText()

    def nested(*a, **k):                                               # (a modal dialog runs its own event loop)
        for _ in range(3):
            app.processEvents()
        return ("z", True)
    monkeypatch.setattr(VP.QInputDialog, "getText", nested)
    card = small.cards_box.itemAt(0).widget()
    pill = card.pills.items[1]
    pill.value.on = False
    pill.mouseDoubleClickEvent(ev)                                     # OK with a new text: no error, shown
    app.processEvents()
    assert [v.text for v in small.variable("c").values] == ["x", "z"] and "z" in small.preview.toPlainText()


def test_all_ideas_become_variations(app, tmp_path):
    from PySide6.QtCore import QSettings

    from video_chain.ui.image_edit import ImageEditDialog
    q = QSettings(str(tmp_path / "i.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                          connect_create=lambda p: None, refine=lambda text, kind, pictures=None: ["a cat", "a dog"])
    dlg.set_mode("create")
    dlg.prompt.setPlainText("- an animal")
    dlg._refine_prompt()
    assert dlg.prompt.toPlainText() == "{idea}" and dlg.variations.active()
    assert [v.text for v in dlg.variations.variable("idea").values] == ["a cat", "a dog"] and dlg.variations.amount() == 2
    assert dlg.b_gen.text() == "Generate 2 images"
    dlg.prompt.undo()
    assert dlg.prompt.toPlainText() == "- an animal"                  # Ctrl+Z: the bullet points back
    dlg.reject()


def test_middle_keyframes_on_the_card(app, win, tmp_path, image, monkeypatch):
    """The keyframe strip: add (picture; the strip's menu), move, note, keep a moment of the made clip, remove;
    the clip counts as changed and the made clip's hard cuts show as a warning."""
    from PIL import Image as PImage
    from PySide6.QtWidgets import QInputDialog

    from video_chain import tree as T
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a",))
    nid = ids[0]
    card = next(c for c in win.cards if c.nid == nid)
    assert card.keys.isEnabled() and card.keys.items == [] and not card.keys.made
    pic = tmp_path / "key.png"
    PImage.new("RGB", (192, 108), (0, 200, 0)).save(pic)
    monkeypatch.setattr(M, "pick_image", lambda *a, **k: str(pic))
    monkeypatch.setattr(M.PromptCard, "_exec_menu", lambda self, m: m.actions()[0])     # "A picture at …"
    card.keys.addAt.emit(1.0)                                     # a click on the strip at 1 s
    node = win.tree.nodes[nid]
    assert [k["at"] for k in node.keyframes] == [1.0] and "keyframe at 1.00 s set" in win.status.text()
    card = next(c for c in win.cards if c.nid == nid)
    assert len(card.keys.items) == 1 and card.keys.items[0]["ok"] and "1 keyframe" in win.badge(node)
    win.keyframe_action("move", nid, {"index": 0, "at": 1.5})
    assert node.keyframes[0]["at"] == 1.5
    win.keyframe_action("move", nid, {"index": 0, "at": 99.0})    # never past the clip's second-to-last frame
    assert round(node.keyframes[0]["at"] * 24) == T.clip_length(win.tree, node) - 2
    win.keyframe_action("move", nid, {"index": 0, "at": 1.5})
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("she turns", True)))
    win.keyframe_action("note", nid, {"index": 0})
    assert node.keyframes[0]["note"] == "she turns"
    run_and_wait(app, win, win.generate_path)
    node = win.tree.nodes[nid]
    assert node.made["keyframes"] == node.keyframes and T.Checker(win.project, win.tree).status(nid) == "done"
    card = next(c for c in win.cards if c.nid == nid)
    assert card.keys.made
    if node.made.get("cuts"):                                     # (the fake clip's coloured last frame is a jump)
        assert card.key_warn.isVisibleTo(card) and "cuts at" in card.key_warn.text()
    # keep a moment of the made clip: its frame, pinned at its time
    dlg = win._keyframe_from_video(nid, 0.5, None, own=True, dialog=False)
    assert dlg is not None and dlg.slider.value() == 12
    dlg.slider.setValue(6)
    assert win._keyframe_from_dialog(nid, dlg, 0.5, None, True)
    dlg.deleteLater()
    assert [k["at"] for k in node.keyframes] == [0.25, 1.5]
    assert T.changed_inputs(win.tree, nid) == ["keyframes"]
    win.keyframe_action("remove", nid, {"index": 0})
    assert [k["at"] for k in node.keyframes] == [1.5] and "removed" in win.status.text()
    win.node_action("revert", nid)
    assert node.keyframes == node.made["keyframes"]



def test_freeze_a_clip_on_its_card(app, win, tmp_path, image):
    """The lock button freezes a made clip and the ones before it: FROZEN, locked inputs, done after a size change;
    the edit actions are refused; the start image stays; clicking again unfreezes it (and the ones after it)."""
    ids = new_project(win, tmp_path, image, ("a", "b"))
    a, b = ids
    card = next(c for c in win.cards if c.nid == b)
    assert not card.b_freeze.isEnabled()                    # not made yet
    win.node_action("freeze_toggle", b)
    assert "not made yet" in win.status.text() and not win.tree.nodes[b].frozen
    run_and_wait(app, win, win.generate_path)
    card = next(c for c in win.cards if c.nid == b)
    assert card.b_freeze.isEnabled()
    card.b_freeze.click()
    assert win.tree.nodes[a].frozen and win.tree.nodes[b].frozen and "frozen, with 1 before it" in win.status.text()
    card = next(c for c in win.cards if c.nid == b)
    assert card.frozen and card.b_freeze.isChecked() and card.tag.text() == "FROZEN"
    assert card.edit.isReadOnly() and not card.seed.isEnabled() and not card.keys.isEnabled()
    assert not card.b_end.isEnabled() and card.buttons[0].isEnabled()          # a variant can still be made
    assert "🔒 frozen" in win.badge(win.tree.nodes[b])
    win.resolution.setCurrentIndex(win.resolution.findData(384))   # another size: still done
    win.refresh()
    assert win.state(a) == win.state(b) == "done"
    win.node_action("handover", b)
    assert "is frozen" in win.status.text() and win.tree.nodes[b].end_at is None
    win.keyframe_action("add", b, {"at": 1.0})
    assert win.tree.nodes[b].keyframes == []
    assert not win.set_image(str(image)) and "unfreeze it first" in win.status.text()
    win.node_action("freeze_toggle", a)                     # unfreezing a takes b along
    assert not win.tree.nodes[a].frozen and not win.tree.nodes[b].frozen
    card = next(c for c in win.cards if c.nid == b)
    assert not card.frozen and not card.edit.isReadOnly()
    assert win.state(a) == "changed" and "CHANGED" in win.status.text()


def tr_lines(win, nid):
    """The lines the clip's spoken track was made of."""
    from video_chain import tree as T
    return [d["line"] for d in T.track_of(win.tree, T.track_id(win.tree.nodes[nid]))["tts"]["lines"]]


def test_text_to_speech_in_the_app(app, win, tmp_path, image, monkeypatch):
    """Speak the lines on a card (a fake engine) makes them the clip's sound; a voice made in the TTS window becomes
    the character's voice; the provider and the ElevenLabs key are kept (the key in the keyring only)."""
    import subprocess

    from video_chain import tree as T
    from video_chain import tts as TTS
    from video_chain.ui import main_window as M
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    def tone(path, secs):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}",
                        str(path)], check=True)

    class Fake:
        name = "local"
        stopped = 0

        def problems(self, **_k):
            return []

        def transcribe(self, audio, lang=None):
            return "sample words"

        def clone(self, text, lang, ref, ref_text, out, seed=None, temperature=None):
            tone(out, 1.5)

        def design(self, text, lang, instruct, out, seed=None):
            tone(out, 4.0)

        def stop(self):
            Fake.stopped += 1
            return True
    fake = Fake()
    ids = new_project(win, tmp_path, image, ("@Anna waves.",))
    nid = ids[0]
    T.add_character(win.project, win.tree, "Anna", desc="young woman")
    win.refresh(rebuild_path=True)
    card = next(c for c in win.cards if c.nid == nid)
    assert card.b_speak.isVisibleTo(card) and not card.b_speak.isEnabled()   # always there; no spoken line yet
    assert "in quotes first" in card.b_speak.toolTip()
    type_prompt(win, nid, '@Anna waves and says "Hallo, schön dich zu sehen."')
    assert card.b_speak.isEnabled()                          # as soon as a line is typed
    win.refresh(rebuild_path=True)
    card = next(c for c in win.cards if c.nid == nid)
    assert card.b_speak.isVisibleTo(card) and card.b_speak.isEnabled()
    monkeypatch.setattr(win, "_tts_engine", lambda: fake)
    win.node_action("speak", nid)
    assert "no voice sample" in win.status.text()            # Anna has no voice yet
    # a voice from the TTS window (its sample made by the fake engine)
    win._show_references(select="Anna")
    assert win._selected_character()["name"] == "Anna" and win.b_voice_tts.isEnabled()
    win._tts_local = fake                                    # the app's worker (local_tts gives it its room check)

    def accept(dlg):
        dlg.desc.setPlainText("warm young voice")
        dlg._make()
        wait(app, lambda: dlg.made is not None or "Could not" in dlg.info.text(), timeout=30)
        return dlg.made is not None
    monkeypatch.setattr(M, "run_voice_dialog", accept)
    win._voice_with_tts()
    c = T.character(win.tree, "Anna")
    assert c["voice"]["name"].startswith("designed: warm young voice") and c["voice"]["text"]
    assert "set with text to speech" in win.status.text() and Fake.stopped >= 1
    assert win.played and win.played[-1].name == "sample.wav"           # the sample was played
    # now the lines
    assert win.speak_clip(nid)
    wait(app, lambda: not win._tts_busy, timeout=30)
    node = win.tree.nodes[nid]
    tr = T.track_of(win.tree, T.track_id(node))
    assert tr and tr["tts"]["clip"] == nid and "spoken" in win.status.text()
    assert T.load(win.project).nodes[nid].overrides["source_sound"] == T.TRACK_PREFIX + tr["id"]   # saved
    # the Speak the lines window: takes to choose from, how they are spoken; the one used becomes the sound
    seen = {}

    def choose(dlg):
        seen.update(lines=dlg.line_texts(), start=dlg.speak_values(), use=dlg.b_use.isEnabled())
        dlg.n.setValue(2)
        dlg.delivery.setCurrentIndex(dlg.delivery.findData("lively"))
        dlg.speed.setValue(0.8)
        dlg.make()
        seen["busy"] = (dlg.busy, dlg.b_make.isEnabled())
        dlg.reject()                                         # (while takes are made: it would close afterwards)
        dlg._close_when_done = False
        wait(app, lambda: not dlg.busy, timeout=30)
        dlg.n.setValue(1)
        dlg.line_edits[0].setText("")
        seen["empty"] = dlg.b_make.isEnabled()               # an empty line cannot be spoken
        dlg.line_edits[0].setText('Hallo, "Welt"!')          # other words: tried, and used
        dlg.make()
        wait(app, lambda: not dlg.busy, timeout=30)
        seen.update(rows=[dlg.list.item(k).text() for k in range(dlg.list.count())], info=dlg.info.text(),
                    files=[win.project / r["file"] for r in dlg.takes])
        dlg.list.setCurrentRow(dlg.list.count() - 1)
        dlg.accept()
        return dlg.chosen is not None
    monkeypatch.setattr(M, "run_speak_dialog", choose)
    old_track = T.track_id(win.tree.nodes[nid])
    win.node_action("speak", nid)
    assert seen["lines"] == ["Hallo, schön dich zu sehen."] and not seen["use"] and not seen["empty"]
    assert seen["rows"][-1].endswith("lively, 0.8 ×   ·   “Hallo, 'Welt'!”") and "“" not in seen["rows"][0]
    assert win.tree.nodes[nid].prompt == """@Anna waves and says "Hallo, 'Welt'!\"""" # the take's words: the prompt's
    assert T.load(win.project).nodes[nid].prompt == win.tree.nodes[nid].prompt and tr_lines(win, nid) == ["Hallo, 'Welt'!"]
    assert next(c for c in win.cards if c.nid == nid).text() == win.tree.nodes[nid].prompt
    assert seen["start"] == TTS.SPEAK_DEFAULTS and seen["busy"] == (True, False)
    assert 1 <= len(seen["rows"]) <= 3 and seen["rows"][0].startswith("Take 1   ·   ")
    assert [f.is_file() for f in seen["files"]] == [False] * (len(seen["files"]) - 1) + [True]   # the others went
    assert all((win.project / chain.OLD_DIR / "refs" / f.name).is_file() for f in seen["files"][:-1])   # ... to old/
    tr = T.track_of(win.tree, T.track_id(win.tree.nodes[nid]))
    assert tr["id"] != old_track and win.project / tr["file"] == seen["files"][-1] and "spoken" in win.status.text()
    assert T.load(win.project).options["tts"]["speak"] == {"takes": 1, "delivery": "lively", "speed": 0.8}
    assert win.played[-1] == seen["files"][0] and not win._tts_busy
    monkeypatch.setattr(M, "run_speak_dialog", lambda dlg: (seen.update(again=dlg.speak_values()), False)[1])
    win.node_action("speak", nid)                            # cancelled: nothing changes; the settings were kept
    assert seen["again"]["delivery"] == "lively" and T.track_id(win.tree.nodes[nid]) == tr["id"]
    # a frozen clip's lines are not spoken
    win.tree.nodes[nid].frozen = True
    win.node_action("speak", nid)
    assert "is frozen" in win.status.text()
    win.tree.nodes[nid].frozen = False
    # provider + key
    win.tts_provider.setCurrentIndex(win.tts_provider.findData("elevenlabs"))
    assert TTS.options(win.tree)["provider"] == "elevenlabs" and "API key" in win.status.text()
    assert TTS.speak_options(win.tree)["delivery"] == "lively"               # (kept when the provider changes)
    assert win._tts_form.isRowVisible(win._eleven_row)
    win.eleven_key.setText(" sk_abc ")
    win._save_eleven_key()
    assert TTS.eleven_key() == "sk_abc" and win.eleven_key.text() == "" and "kept" in win.eleven_key.placeholderText()
    win.eleven_key.setText("")
    win._save_eleven_key()
    assert TTS.eleven_key() == ""


def test_clone_a_voice_at_elevenlabs(app, win, tmp_path, image, monkeypatch):
    """Clone uploads the character's voice sample (a fake ElevenLabs) and makes the new voice its ElevenLabs voice;
    nothing is uploaded without a key or when the question is answered with No."""
    import subprocess

    from video_chain import tree as T
    from video_chain import tts as TTS
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    calls = []

    class Fake:
        fail = ""

        def clone(self, name, sample, description=""):
            calls.append((name, Path(sample).name, Path(sample).is_file()))
            if Fake.fail:
                raise TTS.TtsError(Fake.fail)
            return f"clone{len(calls)}"
    monkeypatch.setattr(TTS, "ElevenLabs", Fake)
    new_project(win, tmp_path, image, ('@Anna says "Hallo."',))
    T.add_character(win.project, win.tree, "Anna")
    win._show_references(select="Anna")
    assert not win.b_voice_clone.isEnabled()                 # no voice sample yet
    speech = tmp_path / "anna.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(speech)],
                   check=True)
    assert win.set_character_voice(str(speech)) and win.b_voice_clone.isEnabled()
    asked = []
    answer = [QMessageBox.No]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), answer[0])[1])
    win._clone_voice_at_eleven()
    assert "API key" in win.status.text() and not asked and not calls      # no key: nothing asked or sent
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    win._clone_voice_at_eleven()
    assert "leaves this computer" in asked[-1] and "consent" in asked[-1] and not calls and not win._tts_busy
    answer[0] = QMessageBox.Yes
    Fake.fail = "instant voice cloning needs a paid plan"
    win._clone_voice_at_eleven()
    wait(app, lambda: not win._tts_busy, timeout=30)
    assert "Could not clone" in win.status.text() and "paid plan" in win.status.text()
    assert "tts" not in T.character(win.tree, "Anna")
    Fake.fail = ""
    win._eleven_voices = [{"id": "x", "name": "old list"}]
    win._clone_voice_at_eleven()
    wait(app, lambda: not win._tts_busy, timeout=30)
    assert calls[-1] == ("Anna (video-chain)", "voice1.wav", True)
    assert T.character(win.tree, "Anna")["tts"]["eleven"] == {"id": "clone2", "name": "Anna (video-chain)"}
    assert T.character(T.load(win.project), "Anna")["tts"]["eleven"]["id"] == "clone2"          # saved
    assert win.char_voice.text() == "Voice: anna.wav (3.0 s) · ElevenLabs: Anna (video-chain)"
    assert "cloned from its voice sample" in win.status.text() and "provider" in win.status.text()
    assert win._eleven_voices is None                        # (the account's voices are asked again)
    assert not TTS.line_problems(win.tree, next(iter(win.tree.nodes.values())), "elevenlabs")
    # again: the question names the voice it replaces
    win._clone_voice_at_eleven()
    wait(app, lambda: not win._tts_busy, timeout=30)
    assert "Anna (video-chain)" in asked[-1] and "replaces it" in asked[-1]
    assert T.character(win.tree, "Anna")["tts"]["eleven"]["id"] == "clone3"
