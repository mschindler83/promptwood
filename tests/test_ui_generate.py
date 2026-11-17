"""GUI tests: generate: running, stopping, the worker and ComfyUI, models kept loaded, progress.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import json
import os
import time


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PySide6.QtWidgets import QInputDialog, QMessageBox  # noqa: E402

from video_chain.ui.generator_state import Job, Refine  # noqa: E402
from video_chain import chain  # noqa: E402
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    FAKE, app, crop_accepted, flush, image, loads, new_project, run_and_wait, silent_player, type_prompt, wait,
    win)


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
    wait(app, lambda: not win.running)                  # (the check runs in the background)
    assert not win.running and "start image" not in win.status.text() and "prompt of clip #1" in win.status.text()


def test_clip_settings_per_node(app, win, tmp_path, image):
    lora = tmp_path / "glow.safetensors"
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
    assert win.badge(win.tree.nodes[ids[1]]) == "glow ×0.8 · smile ×0.4 · 5.2 s · 20 steps"
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


def test_outside_change_is_not_overwritten(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    flush(app, win)
    f = tmp_path / "proj" / T.TREE_FILE
    other = T.load(tmp_path / "proj")
    other.nodes[ids[0]].prompt = "changed elsewhere"
    T.save(tmp_path / "proj", other)                                  # e.g. a second window
    st = f.stat()                                                     # (a newer date set outright: no waiting for a
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))  # clock tick, coarse on some drives)
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
    wait(app, lambda: not win.running)                        # (the check of the project runs in the background)
    assert started == [1] and not win.running and win._waiting_comfy == [ids[0]]
    assert "generating begins as soon as it runs" in win.status.text()
    win.generate_path()                                       # pressed again while it starts: no second start
    wait(app, lambda: not win.running)
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


def test_comfyui_not_running(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setenv("FAKE_COMFY_DOWN", "1")
    new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert "ComfyUI is not running" in win.status.text() and win.comfy_label.text() == "Video engine: not running"
    assert win.b_start_comfy.isEnabled()


def test_waiting_for_comfyui_can_be_cancelled(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    win.comfy_up = False
    win.comfy_autostart.setChecked(True)
    monkeypatch.setattr(win, "start_comfy", lambda: True)
    win.generate_path()
    wait(app, lambda: win._waiting_comfy == [ids[0]] and not win.running)      # (Generate checks in the background first)
    assert win.b_stop.isEnabled()
    win.stop()
    assert win._waiting_comfy is None and not win.b_stop.isEnabled() and "Cancelled" in win.status.text()
    win._comfy_state(True)                                            # ComfyUI comes up: nothing starts
    app.processEvents()
    assert not win.running


def test_stop_looks_disabled_when_nothing_runs(app, win):
    """The danger style must not hide that Stop is disabled (it looked active after a finished export)."""
    from video_chain.ui.theme import ThemeManager
    qss = ThemeManager.instance().qss
    assert 'QPushButton[variant="danger"]:disabled' in qss
    assert not win.running and not win.b_stop.isEnabled()


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


def test_progress_says_what_runs_and_how_long_it_still_takes(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui.main_window import fmt_left, time_left
    assert fmt_left(4) == "a few seconds" and fmt_left(42) == "about 40 s" and fmt_left(185) == "about 3 min"
    assert fmt_left(3900) == "about 1 h 05 min"
    assert time_left(100, 0.01) is None and time_left(1, 0.5) is None and time_left(60, 0.25) == 180
    ids = new_project(win, tmp_path, image, ("a", "b"))
    clock = [1000.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    win._job, win._plan = Job(list(ids), Refine(size="2k")), list(ids)            # (an export of 2 clips)
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