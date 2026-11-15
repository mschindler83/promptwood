"""GUI tests: the window and its project: new / open / save, tree actions, scenes and the film, settings.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import os
from pathlib import Path


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QInputDialog, QMessageBox  # noqa: E402

from video_chain import chain  # noqa: E402
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    FAKE, H3_KEYS, app, chain_sig, crop_accepted, flush, image, new_project, run_and_wait, silent_player,
    type_prompt, wait, win, write_lora)


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


def test_damaged_project_does_not_break_the_window(app, win, tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / T.TREE_FILE).write_text("{ broken")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    assert win.open_project(bad) is False
    assert warned and "damaged" in warned[0] and (bad / T.TREE_FILE).read_text() == "{ broken"
    assert win.project is None or win.project != bad.resolve()


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
    dlg = NewProjectDialog(win, str(tmp_path / "projects"), win.qs)
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
    win.modal_dialogs = False
    win.show_film_cast()
    assert win._film_cast_dialog.windowTitle() == "Cast in the film"
    win._film_cast_dialog.deleteLater()


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


def test_plain_words(app, win, tmp_path, image):
    """The words a beginner reads: the tab named like its section, the model chip and the engine's state in plain
    words, the first sentence of the expert tooltips."""
    assert win.left_tabs.tabText(1) == "Cast & references"
    view_menu = next(a.menu() for a in win.menuBar().actions() if a.text() == "&View")
    assert "Cast & references (left)" in [a.text() for a in view_menu.actions()]
    assert win.comfy_label.text().startswith("Video engine")
    assert win.steps.toolTip().startswith("Steps: how many times the model refines")
    assert win.clip_cfg.toolTip().startswith("CFG: how strictly the model follows the prompt")
    ids = new_project(win, tmp_path, image, ("A cat jumps.",))
    flush(app, win)
    card = win.cards[0]
    assert card.loras.toolTip().startswith("LoRAs: add-ons that teach the model a look, a person or a motion")
    assert card.seed.toolTip().startswith("Seed: the number the clip's randomness starts from")
    assert card.chips.model_chip.text() == "From the start frame" and "fl2va" in card.chips.model_chip.toolTip()
    from video_chain.ui.clip_inputs import model_chip_text
    none = {"given": None}
    assert model_chip_text({"model": "fl2va", "start": none, "end": none, "pictures": []}) == "From the prompt only"
    assert model_chip_text({"model": "ref2va", "start": none, "end": none, "pictures": []}) == "With reference sounds"


# ---------------------------------------------------------------- guided start, writing the prompt, problems
def test_guide_steps_tick_with_the_state_and_hide(app, win, tmp_path, image):
    """No project: step 0. A project: the three steps tick from the real state (text only / a start image, a
    cast entry or Skip, the first clip's prompt), then Generate is pointed out; Hide guide is stored, View → Getting
    started brings it back; once a clip is made the guide goes and the "how to go on" line shows once."""
    g = win.guide_panel
    assert g.isVisibleTo(win) and g.step0.isVisibleTo(g) and not g.step1.isVisibleTo(g)
    win.open_project(tmp_path / "proj")
    flush(app, win)
    assert g.isVisibleTo(win) and not g.step0.isVisibleTo(g)
    assert [s.done for s in (g.step1, g.step2, g.step3)] == [False, False, False] and not g.gen_row.isVisibleTo(g)
    g.b_text_only.click()
    assert g.step1.done and win.inspector.currentWidget() is win.proj_scroll
    g.b_skip_cast.click()
    assert g.step2.done
    g.b_write.click()
    assert win.focus == "n1" and win.inspector.currentIndex() == 0
    type_prompt(win, "n1", "A cat jumps.")
    flush(app, win)
    assert g.step3.done and g.gen_row.isVisibleTo(g) and "Generate" in g.gen_hint.text()
    g.b_show_gen.click()
    win.open_project(tmp_path / "proj2")                  # another project: its own steps
    flush(app, win)
    assert not g.step1.done and not g.step2.done
    win.set_image(str(image))
    T.add_character(win.project, win.tree, "Anna")
    win._show_references()
    win._changed()                                        # (as New in the Cast tab does)
    flush(app, win)
    assert g.step1.done and g.step2.done and not g.step3.done
    g.b_hide.click()
    assert not g.isVisibleTo(win) and win.qs.value("guideHidden", type=bool) is True
    view_menu = next(a.menu() for a in win.menuBar().actions() if a.text() == "&Help")
    next(a for a in view_menu.actions() if a.text() == "Getting started").trigger()
    assert g.isVisibleTo(win) and win.qs.value("guideHidden", type=bool) is False and win.left_tabs.currentIndex() == 0
    type_prompt(win, "n1", "A cat jumps.")
    flush(app, win)
    assert not win.next_hint.isVisibleTo(win)
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)          # (the path is joined to play after a run)
    flush(app, win)
    assert not g.isVisibleTo(win) and win.next_hint.isVisibleTo(win)
    win.next_hint.b_next.click()                         # a second clip: the line has done its job, for good
    flush(app, win)
    assert len(win.tree.nodes) == 2 and not win.next_hint.isVisibleTo(win) and win.qs.value("nextHintDone", type=bool)


def test_engine_banner_follows_the_engine_state(app, win, tmp_path, monkeypatch):
    """Not reachable: the banner with "Start the video engine" (a start command exists) or "Open preferences";
    reachable: gone. Fed with the app's own connection state, no network call of the test's."""
    win._comfy_timer.stop()
    started, prefs = [], []
    monkeypatch.setattr(win, "start_comfy", lambda: started.append(1) or True)
    monkeypatch.setattr(win, "show_prefs", lambda: prefs.append(1))
    launcher = tmp_path / "start-comfy.sh"
    launcher.write_text("#!/bin/sh\n")
    win.comfy_launcher.setText(str(launcher))
    win._comfy_state(False)
    b = win.engine_banner
    assert b.isVisibleTo(win) and "not running" in b.text.text() and b.button.text() == "Start the video engine"
    b.button.click()
    assert started == [1] and prefs == []
    win.comfy_launcher.setText(str(tmp_path / "none.sh"))
    win._comfy_state(False)
    assert b.button.text() == "Open preferences"
    b.button.click()
    assert prefs == [1] and started == [1]
    win._comfy_state(True)
    assert not b.isVisibleTo(win)


def test_cloud_upscaling_not_offered_in_simple_without_a_key(app, win, tmp_path, image, monkeypatch):
    """Simple without a RunPod key: the export dialog does not offer the cloud (its tooltip says where it is set
    up); with a key it is offered as before; Advanced shows it greyed as before."""
    from video_chain import runpod as RP
    from video_chain.ui import main_window as MW
    seen = []
    monkeypatch.setattr(MW.ExportDialog, "exec", lambda self: seen.append(self) or 0)
    new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    win.set_simple(True)
    win.export_path()
    assert seen[-1].cloud.isHidden() and "Advanced view" in seen[-1].size.toolTip()
    win.set_simple(False)
    win.export_path()
    assert not seen[-1].cloud.isHidden() and not seen[-1].cloud.isEnabled()
    monkeypatch.setattr(RP, "api_key", lambda: "k-test")
    win.set_simple(True)
    win.export_path()
    assert not seen[-1].cloud.isHidden() and "Advanced view" not in seen[-1].size.toolTip()


def test_more_start_pictures_make_rows(app, win, tmp_path, image, monkeypatch):
    """A first clip, made once per start image: each new clip on its own row of the node view, behind a start
    node that shows its picture; the prompt can be given to the others later; a row's picture can be changed."""
    from video_chain.ui.graph import START
    from video_chain.ui.main_window import PromptCard, row_start, row_start_of
    ids = new_project(win, tmp_path, image, ("she waves", "then she leaves"))
    first, second = ids
    pics = []
    for k, colour in enumerate(((200, 10, 10), (10, 200, 10), (10, 10, 200))):
        f = tmp_path / f"pic{k}.png"
        Image.new("RGB", (108, 192) if k == 0 else (192, 108), colour).save(f)
        pics.append(str(f))
    assert win.add_start_rows(second, pics, generate=False) == []            # only a clip that starts the video
    assert "first generated clip" in win.status.text()
    from video_chain.ui import image_picker as IP
    picked = []                                                              # the app's own picker, several at once
    monkeypatch.setattr(IP, "pick_images", lambda *a, **k: picked.append((a, k)) or pics[:2])
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.Yes)
    win.node_action("start_rows", first)                                      # (asked "make them now?": yes)
    new = [n.id for n in T.row_members(win.tree, first) if n.id != first]
    assert len(new) == 2 and win.running
    assert len(picked) == 1 and picked[0][1]["collections"] is not None       # (with the project's pictures)
    assert "2 start images added" in asked[0] and "cut from its centre" in asked[0] and "Make the 3 clips" in asked[0]
    nodes = [win.tree.nodes[n] for n in new]
    src = win.tree.nodes[first]
    assert all(n.parent is None and n.prompt == "she waves" and n.seed == src.seed and n.start_image for n in nodes)
    wait(app, lambda: not win.running)
    assert all(win.state(n) == "done" for n in new + [first]) and win.state(second) == "new"
    # the node view: the project's start node with its clip, then a start node and a clip per picture
    items = win.graph.items_by_id
    assert set(items) == {START, first, second, new[0], new[1], row_start(new[0]), row_start(new[1])}
    assert items[new[0]].info.parent == row_start(new[0]) and items[first].info.parent == START
    assert items[row_start(new[0])].info.status == "start" and items[row_start(new[0])].info.pixmap is not None
    ys = sorted(items[i].pos().y() for i in (START, row_start(new[0]), row_start(new[1])))
    assert ys[0] < ys[1] < ys[2]                                              # one row under the other
    assert row_start_of(row_start(new[1])) == new[1] and row_start_of(START) is None and row_start_of(first) is None
    win._graph_clicked(row_start(new[1]))                                     # a click on a row's picture: its clip
    assert win.selected_id() == new[1] and win.focus == new[1]
    win._activate(row_start(new[1]))                                          # (a double click plays nothing)
    # the card's More menu offers both actions for a first clip only
    card = next(c for c in win.cards if isinstance(c, PromptCard) and c.nid == new[1])
    assert card.more_actions["start_rows"].isVisible() and card.more_actions["row_apply"].isVisible()
    win.select(second)
    win.focus_clip(second)
    later = next(c for c in win.cards if isinstance(c, PromptCard) and c.nid == second)
    assert not later.more_actions["start_rows"].isVisible() and not later.more_actions["row_apply"].isVisible()
    # a changed prompt goes to the other rows on request: they count as changed
    win.tree.nodes[first].prompt = "she waves and smiles"
    assert win.apply_to_row(first) == new and all(win.tree.nodes[n].prompt == "she waves and smiles" for n in new)
    assert all(win.state(n) == "changed" for n in new) and "given to" in win.status.text()
    assert win.apply_to_row(first) == [] and "already have" in win.status.text()
    # a variant of a row starts from the same picture (its own copy); another picture for one row; back to the
    # project's start image
    win.node_action("variant", new[0])
    variant = max(win.tree.nodes, key=win.tree.number)
    v = win.tree.nodes[variant]
    assert v.start_image and v.start_image != win.tree.nodes[new[0]].start_image and v.row == first
    assert chain_sig(win.project / v.start_image) == chain_sig(win.project / win.tree.nodes[new[0]].start_image)
    before = win.tree.nodes[new[1]].start_image
    assert win.set_row_start(new[1], pics[2]) and win.tree.nodes[new[1]].start_image != before
    assert win.set_row_start(new[1], clear=True) and win.tree.nodes[new[1]].start_image is None
    assert row_start(new[1]) not in win.graph.items_by_id and win.graph.items_by_id[new[1]].info.parent == START
    # reloaded, the rows are as they were
    proj = win.project
    win.open_project(proj)
    assert win.tree.nodes[new[0]].start_image and row_start(new[0]) in win.graph.items_by_id


def test_speak_window_puts_a_tone_into_a_line(app):
    """With an engine that hears [tags] the Speak the lines window offers tones: one goes into the line clicked
    last, where its cursor is (before any click: the first line, at its start); without such an engine: no tones."""
    from video_chain.ui.speak_dialog import SpeakLinesDialog
    lines = [{"name": "Anna", "line": "Lass mich in Ruhe!"}, {"name": "Ben", "line": "Ja genau."}]
    make = lambda tags: SpeakLinesDialog(None, "#1", lines, "fish" if tags else "local",   # noqa: E731
                                         {"takes": 1, "delivery": "natural", "speed": 1.0}, play=lambda *a: None,
                                         render=lambda *a: None, file=lambda r: r, tags=tags)
    dlg = make(True)
    try:
        dlg.tone.setCurrentText("angry")
        dlg.put_tone()
        assert dlg.line_texts() == ["[angry] Lass mich in Ruhe!", "Ja genau."]
        e = dlg.line_edits[1]
        dlg._target = e                                      # (Ben's line clicked, the cursor after "Ja")
        e.setCursorPosition(2)
        dlg.tone.setCurrentText("[laughs]")                  # (typed with its brackets: still one pair)
        dlg.put_tone()
        assert dlg.line_texts()[1] == "Ja [laughs] genau."
        dlg.tone.setCurrentText("tired, half asleep")        # own words
        dlg.put_tone()
        assert dlg.line_texts()[1] == "Ja [laughs] [tired, half asleep] genau."
        assert any("square brackets" in w.text() for w in dlg.findChildren(type(dlg.info)))
    finally:
        dlg.deleteLater()
    dlg = make(False)
    try:
        assert dlg.tone is None and dlg.b_tone is None
        assert any("Fish S2 Pro hears them" in w.text() for w in dlg.findChildren(type(dlg.info)))
    finally:
        dlg.deleteLater()


def test_fish_engine_in_the_window(win, monkeypatch):
    """Fish S2 Pro is a provider; its worker transcribes with the Qwen3-TTS worker; before it starts the llama.cpp
    server goes and ComfyUI is asked for its VRAM; stop_tts stops both workers."""
    from video_chain import tts as TTS
    assert win.tts_provider.findData("fish") >= 0
    win.tree = win.tree or T.Tree()
    win.tree.options["tts"] = {"provider": "fish"}
    eng = win._tts_engine()
    assert eng.name == "fish" and eng.helper is win.local_tts() and win._tts_engine() is eng
    calls = []

    class Llama:
        def stop(self, wait=True, grace=None):
            calls.append(("llama", wait))
            return True
    monkeypatch.setattr(win, "_llama", Llama(), raising=False)
    monkeypatch.setattr(win, "_make_room", lambda need, url, what, error, later: calls.append((need, what, error)))
    eng.before_launch(eng)
    assert calls == [("llama", True), (TTS.FISH_VRAM_MB, "Fish S2 Pro", TTS.TtsError)]
    stops = []
    monkeypatch.setattr(win._tts_local, "stop", lambda: stops.append("qwen") or True)
    monkeypatch.setattr(win._tts_fish, "stop", lambda: stops.append("fish") or True)
    assert win.stop_tts("test") and stops == ["qwen", "fish"]