"""GUI tests: offscreen, own QSettings file, the fake backend instead of ai-toolkit."""
import json
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
    q = QSettings(str(tmp_path / name), QSettings.IniFormat)
    if q.value("comfyDir") is None:                  # never the real ComfyUI folder in tests
        q.setValue("comfyDir", str(tmp_path / "no-comfyui"))
    if q.value("thumbCacheDir") is None:             # nor the real thumbnail cache
        q.setValue("thumbCacheDir", str(tmp_path / "thumb-cache"))
    return q


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
    ids = new_project(win, tmp_path, image, ("a", "b", "c"))
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.lora.currentIndex() == 0 and card.strength.text() == "default"
    win.set_node_lora(ids[1], str(lora))
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.lora.currentData() == str(lora.resolve())
    card.strength.setValue(0.8)
    card.length.setCurrentIndex(card.length.findData(124))
    card.steps.setValue(20)
    flush(app, win)
    assert win.tree.nodes[ids[1]].overrides == {"lora": str(lora.resolve()), "strength": 0.8, "frames": 124,
                                                "steps": 20}
    assert win.badge(win.tree.nodes[ids[1]]) == "nose · ×0.8 · 5.2 s · 20 steps"
    assert win.graph.items_by_id[ids[1]].info.badge.startswith("glow")
    assert win.graph.items_by_id[ids[0]].info.badge == ""
    # time spans follow the clip's own length: clip 2 is 124 frames (minus the repeated one)
    assert win.cards[2].span.text().startswith(f"0:{(73 + 123) / 24:04.1f}")
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[1]].made["settings"]
    assert made["lora"] == str(lora.resolve()) and made["strength"] == 0.8 and made["frames"] == 124
    # no LoRA on clip 3: strength is not used there
    card3 = next(c for c in win.cards if c.nid == ids[2])
    card3.lora.setCurrentIndex(card3.lora.findData(T.NO_LORA))
    flush(app, win)
    assert win.tree.nodes[ids[2]].overrides == {"lora": T.NO_LORA} and not card3.strength.isEnabled()
    assert [win.state(i) for i in ids] == ["done", "done", "done"]     # the project has no LoRA: same clip
    win.set_node_lora(ids[2], str(lora))                                # its own LoRA: made again
    flush(app, win)
    assert [win.state(i) for i in ids] == ["done", "done", "changed"]
    card3 = next(c for c in win.cards if c.nid == ids[2])
    card3.lora.setCurrentIndex(0)                                       # back to the project default
    flush(app, win)
    assert win.tree.nodes[ids[2]].overrides == {} and win.state(ids[2]) == "done"
    # the project LoRA list now offers the chosen file on every card
    assert str(lora.resolve()) in [card3.lora.itemData(i) for i in range(card3.lora.count())]


def test_missing_node_lora_stops_the_start(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    win.tree.nodes[ids[0]].overrides = {"lora": str(tmp_path / "gone.safetensors")}
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


def test_bigger_lora_and_sound_change_reload(app, win, tmp_path, image, monkeypatch):
    log = tmp_path / "loads.jsonl"
    monkeypatch.setenv("FAKE_LOADS", str(log))
    monkeypatch.setenv("FAKE_BIG", "big")
    big = tmp_path / "big_rank64.safetensors"
    big.write_bytes(b"x")
    ids = new_project(win, tmp_path, image, ("a",))
    win.backend.setCurrentIndex(win.backend.findData("ai-toolkit"))    # only ai-toolkit prepares LoRA ranks
    flush(app, win)
    run_and_wait(app, win, win.generate_path)
    win.node_action("continue", ids[0])
    n2 = win.selected_id()
    type_prompt(win, n2, "b")
    flush(app, win)
    win.tree.nodes[n2].overrides = {"lora": str(big)}       # added after the worker loaded: does not fit
    flush(app, win)
    win.refresh()
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: win.state(n2) == "done", timeout=20)
    got = loads(log)
    assert len(got) == 2 and got[0]["loras"] == [] and got[1]["loras"] == [str(big)]
    win.audio.setChecked(False)                              # another sound setting: a new worker
    flush(app, win)
    run_and_wait(app, win, win.generate_path)
    got = loads(log)
    assert len(got) == 3 and got[2]["audio"] is False


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
    win.set_node_lora(ids[0], str(lora))
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


def test_backend_change_marks_clips(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert win.state(ids[0]) == "done"
    win.backend.setCurrentIndex(win.backend.findData("ai-toolkit"))
    flush(app, win)
    assert win.state(ids[0]) == "changed" and win.tree.options["backend"] == "ai-toolkit"


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
    items = [card.lora.itemText(i) for i in range(card.lora.count())]
    assert items == ["project default (none)", "no LoRA", "smoke"]
    project = [w.lora.itemText(i) for i in range(w.lora.count())]
    assert project == ["none (base model)", "smoke"]
    w.lora.setCurrentIndex(1)                                                        # project LoRA from the list
    flush(app, w)
    assert w.tree.options["lora"] == str(comfy / "models" / "loras" / "MiniMax H3" / "smoke.safetensors")
    assert w.cards[0].lora.itemText(0) == "project default (smoke)"
    card = w.cards[0]
    card.lora.setCurrentIndex(card.lora.findText("no LoRA"))
    flush(app, w)
    assert w.tree.nodes[ids[0]].overrides == {"lora": T.NO_LORA}
    w.close()


def test_identity_references_in_the_window(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    face = tmp_path / "face.png"
    Image.new("RGB", (80, 100), (190, 150, 120)).save(face)
    win.add_reference(str(face))
    flush(app, win)
    assert win.ident_mode.currentData() == "reference" and win.refs.count() == 1
    assert win.refs.item(0).text() == "<Picture 1>" and (tmp_path / "proj" / "refs" / "ref1.png").is_file()
    assert [win.state(i) for i in ids] == ["changed", "changed"]               # every clip sees the reference now
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[0]].made["settings"]
    assert made["refs"][0]["file"] == "refs/ref1.png" and made["ref_size"] == "match"
    win.ref_size.setCurrentIndex(win.ref_size.findData("max"))
    flush(app, win)
    assert win.state(ids[0]) == "changed"
    win.backend.setCurrentIndex(win.backend.findData("ai-toolkit"))
    flush(app, win)
    wait(app, lambda: not win._joins, timeout=30)                              # the auto-play join of the run
    win.generate_path()
    assert not win.running and "reference images need the ComfyUI generator" in win.status.text()
    win.backend.setCurrentIndex(win.backend.findData("comfyui"))
    win.refs.setCurrentRow(0)
    win._remove_reference()
    flush(app, win)
    assert win.refs.count() == 0 and not (tmp_path / "proj" / "refs" / "ref1.png").exists()
    win.generate_path()
    assert not win.running and "add a reference image" in win.status.text()
    win.ident_mode.setCurrentIndex(win.ident_mode.findData("off"))
    flush(app, win)
    assert [win.state(i) for i in ids] == ["changed", "changed"]               # made with refs, now without


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
    # project: nose + motion stacked
    w.lora.setCurrentIndex(w.lora.findData(path["glow"]))
    w.extra_loras._add_row(path["motion"], 0.5)
    w.extra_loras.changed.emit()
    flush(app, w)
    assert w.tree.options["extra_loras"] == [{"path": path["motion"], "strength": 0.5}]
    # clip 2: style on top of the project's LoRAs
    card = next(c for c in w.cards if c.nid == ids[1])
    card.extras._add_row(path["style"], 0.3)
    card.extras.changed.emit()
    flush(app, w)
    assert w.tree.nodes[ids[1]].overrides["extra_loras"] == [{"path": path["style"], "strength": 0.3}]
    assert w.badge(w.tree.nodes[ids[1]]) == "+ style ×0.3"
    ch = T.Checker(w.project, w.tree)
    s2 = ch.settings(ids[1])
    assert s2["lora"] == path["glow"] and [e["lora"] for e in s2["extra_loras"]] == [path["motion"], path["style"]]
    run_and_wait(app, w, w.generate_path)
    made = T.load(w.project).nodes[ids[1]].made["settings"]
    assert [e["strength"] for e in made["extra_loras"]] == [0.5, 0.3]
    # removing the row takes the LoRA off again
    card = next(c for c in w.cards if c.nid == ids[1])
    card.extras._remove(card.extras.rows[0])
    flush(app, w)
    assert "extra_loras" not in w.tree.nodes[ids[1]].overrides and w.state(ids[1]) == "changed"
    # ai-toolkit can apply only one LoRA
    w.backend.setCurrentIndex(w.backend.findData("ai-toolkit"))
    flush(app, w)
    wait(app, lambda: not w._joins, timeout=30)
    w.generate_path()
    assert not w.running and "several LoRAs on one clip need the ComfyUI generator" in w.status.text()
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
    assert [w.lora.itemText(i) for i in range(w.lora.count())] == ["none (base model)", "glow"]
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
    w.backend.setCurrentIndex(w.backend.findData("ai-toolkit"))
    flush(app, w)
    wait(app, lambda: not w._joins, timeout=30)
    w.generate_path()
    assert not w.running and "turbo mode needs the ComfyUI generator" in w.status.text()
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
    win.tree.nodes[ids[1]].overrides = {"lora": str(lora), "strength": 0.7, "frames": 90}
    new = win.create_variants(ids[1], 3, generate=True)
    assert len(new) == 3 and win.running
    nodes = [win.tree.nodes[n] for n in new]
    assert all(n.parent == ids[0] and n.prompt == "b" for n in nodes)
    assert all(n.overrides == {"lora": str(lora), "strength": 0.7, "frames": 90} for n in nodes)
    seeds = {n.seed for n in nodes} | {win.tree.nodes[ids[1]].seed}
    assert len(seeds) == 4                                                    # every variant its own seed
    wait(app, lambda: not win.running)
    assert all(win.state(n) == "done" for n in new) and win.state(ids[0]) == "done"
    assert win.state(ids[1]) == "new"                                         # the original was not asked for
    assert len(win.tree.leaves()) == 4
    only = win.create_variants(ids[0], 2, generate=False)
    assert not win.running and all(win.state(n) == "new" for n in only)


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
    assert win.tree.options["identity"]["images"] == [] and win.refs.count() == 0
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


def test_ai_toolkit_cannot_end_at_a_video(app, win, tmp_path, image):
    src = make_real_video(tmp_path / "real.mp4")
    ids = new_project(win, tmp_path, image, ("x",))
    win.add_video(ids[0], str(src), 0.0, 1.0)
    wait(app, lambda: not win._converts)
    win.backend.setCurrentIndex(win.backend.findData("ai-toolkit"))
    win.generate_path()
    assert not win.running and "needs the ComfyUI generator" in win.status.text()


def test_motion_guide_choice_on_the_card(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("x", "y"))
    card = win.cards[1]
    card.guide.setCurrentIndex(card.guide.findData(22))
    assert win.tree.nodes[ids[1]].overrides["guide"] == 22
    assert "guide 22" in win.badge(win.tree.nodes[ids[1]])
    flush(app, win)
    assert win.cards[1].span.text().startswith("0:03.0 – 0:05.2")          # 22 guide frames left out


def test_video_range_dialog(app, tmp_path):
    from video_chain.ui.video_editor import VideoRangeDialog
    src = make_real_video(tmp_path / "real.mp4", secs=2.0)
    dlg = VideoRangeDialog(None, str(src), canvas=(672, 384), player=False)
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
