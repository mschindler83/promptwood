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
    assert win.char_voice.text() == "Voice: tom.wav (7.5 s)" and "· voice" in win.cast_list.currentItem().text()
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
        assert texts[0] == f"Last frame of clip #{ids[1][1:]}" and "Start image" in texts and texts[-1] == "Choose file…"
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
    assert texts[0].startswith("As made (672×384)") and "1888×1088 · already refined" in texts[1] \
        and "2528×1440" in texts[2]
    dlg.reject()
    out = tmp_path / "export-2k.mp4"
    win.export(out, "2k")
    assert win.running
    wait(app, lambda: not win.running, timeout=120)
    info = chain.probe(out)
    base = T.Checker(win.project, win.tree).base()
    from video_chain.comfy import refine_size
    assert (info["width"], info["height"]) == refine_size(base["width"], base["height"], "2k")
    assert win.status.text().startswith(f"Exported {out.name} (2k, refined; 2 clips refined now")
    win.export(out, "2k")                                                    # again: the kept copies are reused
    wait(app, lambda: not win.running, timeout=120)
    assert win.status.text() == f"Exported {out.name} (2k, refined; every clip was refined already)."
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
    seen = {}

    class FakeDialog:
        def __init__(self, parent, text, kind, connect, settings=None, start_comfy=None, cast=None):
            seen.update(text=text, kind=kind, cast=cast)
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
        dlg.generate()
        wait(app, lambda: not dlg._busy, timeout=60)
        assert dlg.state.text().startswith("Done") and dlg.results.count() == 1
        png, meta = dlg.selected()
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
        assert "“she smiles” + glowx" in dlg.res_info.text()
        dlg.done(0)
        w.lora_triggers.setChecked(False)                                       # the project's choice
        dlg = w.edit_image(start=True, dialog=False)
        assert dlg._trigger_words() == []
        dlg.done(0)
        w.close()
    finally:
        server.close()


def test_continue_the_real_videos_sound_on_the_card(app, win, tmp_path):
    """The card's 'Sound' row appears only after a real video; switching it on also carries the picture over."""
    from video_chain.ui.main_window import PromptCard
    src = make_real_video(tmp_path / "real.mp4", secs=8.0)
    win.open_project(tmp_path / "proj")
    first = win.cards[0]
    assert isinstance(first, PromptCard) and not first.sound_box.isVisibleTo(first)   # no real video before it
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
