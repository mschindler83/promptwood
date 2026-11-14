"""GUI tests: the image windows: edit / create, the picker, face crop, masks, replacing a person, the gallery, variations.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import json
import os
import sys
import time
from pathlib import Path


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    FAKE, QImage_size, app, crop_accepted, flush, image, make_real_video, new_project, run_and_wait,
    silent_player, wait, win)


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
        w.modal_dialogs = False
        dlg = w.edit_image(ids[1])
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
        w.modal_dialogs = False
        dlg2 = w.edit_image(start=True)
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
        w.modal_dialogs = False
        dlg4 = w.edit_image(start=True)
        assert dlg4.results.count() == 1 and "the start image" in dlg4.res_info.text() and not dlg4.b_delete.isEnabled()
        dlg4.reject()
        assert any(text == "Project AI edits" for text, _p in w._picker_places())
        server.close()                                              # ComfyUI gone: a clear message
        w.modal_dialogs = False
        dlg3 = w.edit_image(start=True)
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
        w.modal_dialogs = False
        dlg = w.edit_image(ids[0])
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


def test_prompt_refiner_on_the_card(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import prompt_refiner as PR
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a man walks",))
    seen, seen_pics, seen_clip = {}, [], {}

    class FakeDialog:
        def __init__(self, parent, text, kind, connect, settings=None, start_comfy=None, cast=None, pictures=None,
                     **clip):
            seen.update(text=text, kind=kind, cast=cast)
            seen_pics.extend(pictures or [])
            seen_clip.update(clip)
            self.refined_any = True

        def exec(self):
            return M.QDialog.Accepted

        def chosen(self):
            return "A man walks slowly towards the camera; footsteps on gravel."

        def deleteLater(self):
            pass
    monkeypatch.setattr(PR, "RefineDialog", FakeDialog)
    card = win.cards[0]
    card.b_refine.click()
    assert seen == {"text": "a man walks", "kind": "video", "cast": []}
    assert [t for t, _p, on in seen_pics] == ["The frame the clip starts from"]   # (the start image; no end image)
    assert win.tree.nodes[ids[0]].prompt == "A man walks slowly towards the camera; footsteps on gravel."
    card.edit.undo()                                              # one Ctrl+Z takes the refined text back
    assert card.edit.toPlainText() == "a man walks"
    assert win._comfy_model == "llm" and "language model" in win.model_label.text()   # unloaded before clips
    assert seen_clip["seconds"] == win.tree.options.get("frames", 73) / 24 and Path(seen_clip["start_image"]) == \
        Path(seen_pics[0][1])
    win.tree.options["prompt_format"] = "h3"                       # the official format: its own refine
    win.tree.nodes[ids[0]].overrides["frames"] = 121
    next(c for c in win.cards if c.nid == ids[0]).b_refine.click()
    assert seen["kind"] == "video_h3" and seen["cast"] == [] and seen_clip["seconds"] == 121 / 24
    del win.tree.options["prompt_format"]


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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True, mode="create")
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
        w.modal_dialogs = False
        dlg2 = w.edit_image(start=True)                  # remembered: create mode, the family
        assert dlg2.mode() == "create"
        wait(app, lambda: dlg2.families, timeout=20)
        assert dlg2.family.currentData() == "sdxl" and dlg2.loras.items()[0]["strength"] == 0.8
        dlg2.m_edit.click()
        assert dlg2.mode() == "edit" and dlg2.edit_box.isVisibleTo(dlg2) and dlg2.prompt_kind() == "edit"
        dlg2.reject()
        w.close()
    finally:
        server.close()


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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
        wait(app, lambda: dlg.families, timeout=20)
        k = dlg.edit_model.findData("qwen21")
        assert not dlg.edit_model.model().item(k).isEnabled() and dlg.edit_model.currentData() == "2511"
        assert "not available" in dlg.edit_model.itemText(k)
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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
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
        w.modal_dialogs = False
        dlg = w.edit_image(start=True)
        wait(app, lambda: dlg.families, timeout=20)
        assert dlg.source and dlg.mask is None and dlg.b_mask.isEnabled() and not dlg.b_mask_clear.isVisible()
        from video_chain.ui.image_edit import upright
        src = upright(dlg.source)
        painter = MaskPainterDialog(dlg, src)
        c = painter.canvas
        assert c.mask.size() == src.size() and mask_area(c.mask) == 0
        c.set_brush(20)
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
        w.modal_dialogs = False
        dlg = w.replace_person(v)
        wait(app, lambda: dlg.heads is not None, timeout=20)
        assert f"found in {n} of {n} frames" in dlg.found_text.text() and not dlg.b_gen.isEnabled()
        dlg.head_size.setValue(1.3)                                    # larger head: the boxes grow
        assert dlg.boxes[0][2] - dlg.boxes[0][0] > 40 * 1.3
        dlg.set_reference(str(image))
        wait(app, lambda: not dlg.ref_busy, timeout=20)
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
        wait(app, lambda: not dlg.ref_busy, timeout=20)
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
        wait(app, lambda: not dlg.ref_busy, timeout=20)
        assert dlg.prompt.text() == "Male, oval face, short black hair"
        dlg.set_reference(str(image))                                  # another picture: its own description
        wait(app, lambda: not dlg.ref_busy, timeout=20)
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
        w.modal_dialogs = False
        nodet = w.replace_person(v)
        assert nodet.heads is not None and nodet.heads["frames"] == n and not nodet.boxes
        nodet.set_reference(str(image))
        wait(app, lambda: not nodet.ref_busy, timeout=20)
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
    from video_chain import variations as V
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    server = FakeComfy(tmp_path / "comfy")
    try:
        q = settings(tmp_path, "var.ini")
        q.setValue("comfyUrl", server.url)
        w = MainWindow(q, command=FAKE)
        new_project(w, tmp_path, image, ("a",))
        w.modal_dialogs = False
        dlg = w.edit_image(start=True, mode="create")
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
        cards[0].add_values(V.split_values("cat, dog"))                      # a list at once
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
        panel._rename("place", "where")                                  # renamed in the prompt too
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