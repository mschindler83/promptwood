"""GUI tests: loRAs in the windows: lists, the stack, the picker, trigger words, strengths.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import json
import os
import sys
from pathlib import Path


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    FAKE, H3_KEYS, app, crop_accepted, flush, image, new_project, run_and_wait, silent_player, wait, win,
    write_lora)


def test_missing_node_lora_stops_the_start(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    win.tree.nodes[ids[0]].overrides = {"loras": [{"path": str(tmp_path / "gone.safetensors"), "strength": 1}]}
    win.generate_path()
    wait(app, lambda: not win.running)
    assert not win.running and "LoRA file does not exist (gone.safetensors)" in win.status.text()


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
    # clip 2: glow, motion and style stacked on the card, each with its strength
    card = next(c for c in w.cards if c.nid == ids[1])
    for name, st in (("glow", 1.0), ("motion", 0.5), ("style", 0.3)):
        card.loras._add_row(path[name], st)
    card.loras.changed.emit()
    flush(app, w)
    assert w.tree.nodes[ids[1]].overrides["loras"] == [{"path": path["glow"], "strength": 1.0},
                                                       {"path": path["motion"], "strength": 0.5},
                                                       {"path": path["style"], "strength": 0.3}]
    assert w.badge(w.tree.nodes[ids[1]]) == "glow ×1 · motion ×0.5 · style ×0.3"
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


def test_lora_stack_keeps_paths_and_picks_from_the_picker(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    real = tmp_path / "loras" / "glow.safetensors"
    write_lora(real, H3_KEYS)
    link = tmp_path / "comfy-loras" / "glow.safetensors"
    link.parent.mkdir()
    link.symlink_to(real)
    win.h3_loras = [(str(link), "glow.safetensors")]               # ComfyUI lists the link
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
    assert texts[0] == "glow_smoke" and dlg.list.item(0).data(dlg.IN_USE)                  # marked, sorted by name
    assert Path(names[0]).name in texts and not any("MiniMax" in t for t in texts)        # whole name, no folder
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

    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    lora = tmp_path / "x.safetensors"
    lora.write_bytes(b"x")
    q = QSettings(str(tmp_path / "fit.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, services=ImageEditServices(connect=lambda p: None, on_use=lambda k, p: False, settings=q, known_loras=[(str(lora), "x")], connect_create=lambda p: None))
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
        lora = tmp_path / "loras" / "MiniMax H3" / "glow.safetensors"           # ComfyUI (fake) lists this name
        lora.parent.mkdir(parents=True)
        save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(lora),
                  metadata={"modelspec.trigger_phrase": "glowx"})
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
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
        assert got == [(500, "glow", 0.7, 1, ["glowx"], "she smiles"), (500, "rain", 1.2, 1, ["rainx"], "she smiles"),
                       (501, "glow", 0.7, 1, ["glowx"], "she smiles"), (501, "rain", 1.2, 1, ["rainx"], "she smiles")]
        dlg.done(0)
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)                   # remembered
        assert dlg.lora_each.isChecked()
        dlg.lora_each.setChecked(False)
        assert len(dlg._lora_runs()) == 1 and len(dlg._lora_runs()[0]) == 2   # off: all LoRAs together
        dlg.done(0)
        w.close()
    finally:
        server.close()


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
    dlg = IE.ImageEditDialog(None, tmp_path, [], None, None, services=IE.ImageEditServices(connect=lambda p: None, on_use=lambda k, p: False, settings=q, connect_create=lambda p: None))
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
    from PySide6.QtWidgets import QMessageBox as _QMB
    monkeypatch.setattr(_QMB, "question", lambda *a, **k: _QMB.No)
    assert not dlg.save_preset_as("Sprint", lora=str(b))                    # same name: replace only if confirmed
    assert not dlg.delete_preset()
    monkeypatch.setattr(_QMB, "question", lambda *a, **k: _QMB.Yes)
    assert dlg.delete_preset() and presets.store().list(str(b)) == []
    assert [dlg.preset.itemText(i) for i in range(1, dlg.preset.count())] == ["walk: Walk, cfg 7"]
    dlg.reject()

def test_lora_on_off_switch_in_a_clip(app, win, tmp_path, image):
    """Each LoRA row has an on/off box: a LoRA switched off stays in the clip's list with its strength, but the
    clip is made without it (settings, trigger words, the summary); switched on again, the clip is as before."""
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    write_lora(a, H3_KEYS)
    write_lora(b, H3_KEYS)
    ids = new_project(win, tmp_path, image, ("one",))
    node = win.tree.nodes[ids[0]]
    node.overrides = {"loras": [{"path": str(a), "strength": 0.7}, {"path": str(b), "strength": 0.4}]}
    win.refresh(rebuild_path=True)
    before = T.Checker(win.project, win.tree).settings(ids[0])
    stack = win.cards[0].loras
    assert [w.on_box.isChecked() for w, _b, _s in stack.rows] == [True, True]
    stack.rows[1][0].on_box.click()                                 # b off
    assert node.overrides["loras"] == [{"path": str(a), "strength": 0.7},
                                       {"path": str(b), "strength": 0.4, "off": True}]
    assert T.node_loras(node) == [(str(a), 0.7)]
    s = T.Checker(win.project, win.tree).settings(ids[0])
    assert s["lora"] == str(a) and not s.get("extra_loras")
    assert stack.rows[1][1].font().strikeOut() and stack.rows[1][2].isEnabled()   # (its strength stays editable)
    win.refresh(rebuild_path=True)                                  # shown again as it was stored
    stack = win.cards[0].loras
    assert [w.on_box.isChecked() for w, _b, _s in stack.rows] == [True, False]
    stack.rows[0][0].on_box.click()                                 # a off too: the clip is made without LoRAs
    assert T.node_loras(node) == [] and T.Checker(win.project, win.tree).settings(ids[0])["lora"] is None
    stack.rows[0][0].on_box.click()
    stack.rows[1][0].on_box.click()                                 # both on again: the same clip as before
    assert node.overrides["loras"] == [{"path": str(a), "strength": 0.7}, {"path": str(b), "strength": 0.4}]
    assert T.Checker(win.project, win.tree).settings(ids[0]) == before


def test_variants_leave_lora_that_is_off_alone(app, win, tmp_path, image):
    """A strength sweep offers only the LoRAs that are on; the one that is off is kept in every variant as it is."""
    from video_chain.ui.main_window import VariantsDialog
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    ids = new_project(win, tmp_path, image, ("one",))
    node = win.tree.nodes[ids[0]]
    node.overrides = {"loras": [{"path": str(a), "strength": 0.5, "off": True}, {"path": str(b), "strength": 1.0}]}
    dlg = VariantsDialog(win, node)
    assert len(dlg.steps) == 1
    dlg.done(0)
    new = win.create_variants(ids[0], 2, generate=False, strength_steps=[0.25], keep_seed=True)
    got = [win.tree.nodes[n].overrides["loras"] for n in new]
    assert got == [[{"path": str(a), "strength": 0.5, "off": True}, {"path": str(b), "strength": 1.25}],
                   [{"path": str(a), "strength": 0.5, "off": True}, {"path": str(b), "strength": 1.5}]]


def test_lora_on_off_switch_in_the_image_window(app, tmp_path):
    """The image window: a LoRA that is off is not used (runs, trigger words) but is remembered with its switch."""
    from PySide6.QtCore import QSettings

    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    q = QSettings(str(tmp_path / "onoff.ini"), QSettings.IniFormat)
    services = ImageEditServices(connect=lambda p: None, on_use=lambda k, p: False, settings=q,
                                 known_loras=[(str(a), "a"), (str(b), "b")], connect_create=lambda p: None)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, services=services)
    dlg.set_mode("edit")
    dlg.add_lora_file(str(a), 0.6)
    dlg.add_lora_file(str(b), 0.9)
    dlg.loras.rows[0][0].on_box.click()                             # a off
    assert dlg._lora_runs() == [[(str(b), 0.9)]]
    dlg.lora_each.setChecked(True)
    assert dlg._lora_runs() == [[(str(b), 0.9)]]                    # one image per LoRA: only those that are on
    assert dlg.loras.active_items() == [{"path": str(b), "strength": 0.9}]
    dlg.reject()
    dlg = ImageEditDialog(None, tmp_path, [], None, None, services=services)
    dlg.set_mode("edit")
    assert dlg.loras.items() == [{"path": str(a), "strength": 0.6, "off": True}, {"path": str(b), "strength": 0.9}]
    assert not dlg.loras.rows[0][0].on_box.isChecked()
    dlg.reject()
