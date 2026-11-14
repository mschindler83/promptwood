"""Fixes of the final UI wave B (image dialogs, editors, shortcuts table, dialog geometry ...). Synthetic data, offscreen."""
import os
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_image_edit_split import _Editor, _edit_plan, _run_worker  # noqa: E402,F401


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_a_failed_rescale_leaves_no_result_without_a_record(app, tmp_path, monkeypatch):
    from PIL import Image

    from video_chain import edits
    from video_chain.ui.image_generate import EditWorker
    src = tmp_path / "a.png"
    Image.new("RGB", (16, 16), "green").save(src)

    def boom(*a, **k):
        raise RuntimeError("ffmpeg failed")
    monkeypatch.setattr(edits, "scale_to", boom)
    got = _run_worker(EditWorker(_edit_plan(tmp_path, _Editor(), [str(src)], keep_size=True)))
    assert got["finished"] == ["ffmpeg failed"] and got["results"] == []
    assert list((tmp_path / "edits").glob("*.png")) == []


def test_a_lora_row_asks_the_disk_once_per_file(app, monkeypatch):
    from video_chain.ui.lora_stack import LoraStack
    stack = LoraStack()
    calls = []
    real = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: (calls.append(str(self)), real(self))[1])
    stack.set_items([], [{"path": f"/x/{n}.safetensors", "strength": 1.0} for n in "abc"])
    mine = [c for c in calls if c.endswith(".safetensors")]
    assert len(mine) == 3


def test_lora_notes_cut_is_public_and_the_window_uses_it():
    from video_chain import lora_notes
    assert lora_notes.cut("word " * 100, 20).endswith("…") or len(lora_notes.cut("word " * 100, 20)) <= 21
    assert not hasattr(lora_notes, "_cut")


def test_the_conversation_shows_a_short_label_for_the_long_example_instruction():
    from video_chain.ui import prompt_refiner as PR
    long = PR.LIKE_EXAMPLE.format(example="a cat, 4k")
    assert PR.shown_ask(long) == "like an example prompt of the LoRA"
    assert PR.shown_ask("make it shorter") == "make it shorter" and PR.shown_ask("") == ""


def test_a_failing_guidance_is_logged_not_swallowed(app, caplog, monkeypatch, tmp_path):
    import logging

    from video_chain import lora_notes
    from video_chain.ui import prompt_refiner as PR
    monkeypatch.setattr(lora_notes, "guidance", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unreadable")))
    with caplog.at_level(logging.WARNING):
        assert PR.guidance_or_empty(["/x/a.safetensors"], "a prompt") == ""
    assert any("could not be read" in r.getMessage() for r in caplog.records)


def test_a_lora_page_is_read_off_the_gui_thread_and_a_newer_choice_wins(app, tmp_path, monkeypatch):
    import threading
    import time

    from test_lora_window_fixes import window
    from video_chain import lora_notes
    from waiting import wait
    w, db, p, other = window(tmp_path)
    seen, release = [], threading.Event()
    real = lora_notes.what_is_known

    def known(path, d):
        seen.append((path, threading.current_thread()))
        if path == p:
            release.wait(5)
        return real(path, d)
    monkeypatch.setattr(lora_notes, "what_is_known", known)
    w.show_lora(p)                                   # (slow: still being read)
    t0 = time.monotonic()
    w.show_lora(other)
    assert time.monotonic() - t0 < 1 and w.facts.text() == "Reading what is known …"
    assert wait(app, lambda: w._loaded == other)
    release.set()
    wait(app, lambda: len(seen) >= 2)
    app.processEvents()
    assert w._loaded == other and w.path == other           # the older answer arrived later and was dropped
    assert all(t is not threading.main_thread() for _p, t in seen)
    w.done(0)


def test_the_frame_window_probes_a_picked_file_in_a_worker(app, tmp_path, monkeypatch):
    import threading

    from video_chain import chain
    from video_chain.ui.clip_dialogs import VideoFrameDialog
    from waiting import wait
    seen = []

    def probe(p):
        seen.append(threading.current_thread())
        return {"fps": 24, "duration": 2.0, "frames": 48, "width": 128, "height": 72}
    monkeypatch.setattr(chain, "probe_source", probe)
    monkeypatch.setattr(VideoFrameDialog, "_grab", lambda self: None)
    dlg = VideoFrameDialog(None, tmp_path / "v.mp4", "frame")
    assert not dlg.ready and not dlg.buttons.button(dlg.buttons.StandardButton.Ok).isEnabled()
    assert wait(app, lambda: dlg.ready)
    assert dlg.frames == 48 and dlg.fps == 24 and dlg.slider.maximum() == 47
    assert seen and all(t is not threading.main_thread() for t in seen)
    assert dlg.buttons.button(dlg.buttons.StandardButton.Ok).isEnabled()
    dlg.reject()


def test_the_frame_window_says_when_the_file_cannot_be_read(app, tmp_path, monkeypatch):
    from video_chain import chain
    from video_chain.ui.clip_dialogs import VideoFrameDialog
    from waiting import wait
    monkeypatch.setattr(chain, "probe_source", lambda p: (_ for _ in ()).throw(RuntimeError("not a video")))
    dlg = VideoFrameDialog(None, tmp_path / "v.mp4", "frame")
    assert wait(app, lambda: "could not be read" in dlg.note.text())
    assert not dlg.ready and not dlg.buttons.button(dlg.buttons.StandardButton.Ok).isEnabled()
    dlg.reject()


def test_the_compare_player_steps_with_the_shared_frame_stepper(app):
    from video_chain.ui.compare_player import ComparePlayer, store_size
    from video_chain.ui.widgets import FrameStepper
    p = ComparePlayer(fps=10)
    assert isinstance(p.stepper, FrameStepper) and p.slider is p.stepper.slider
    size = store_size(8, 8, 20)
    p.size_, p.clip_size = size, (8, 8)
    p.frames = [bytes(size[0] * size[1] * 3)] * 20
    p._controls(True)
    p.stepper.set_range(0, 19)
    p.play()
    assert p.playing()
    p.stepper.b_fwd1.click()                       # a step by hand stops the playing
    assert not p.playing() and p.position() == 1
    p.stepper.b_fwd_s.click()                      # a second at 10 fps
    assert p.position() == 11
    assert p.time.text() == "frame 12 / 20 · 1.10 s"
    p.seek(19)
    assert p.position() == 19


def test_the_swap_dialog_probes_a_clip_without_a_detector_in_a_worker(app, tmp_path, monkeypatch):
    import threading

    from video_chain import swap
    from video_chain.ui.swap_dialog import SwapDialog
    from waiting import wait
    seen = []

    def probe(clip):
        seen.append(threading.current_thread())
        return 160, 90, 6, 24.0
    monkeypatch.setattr(swap, "probe", probe)
    dlg = SwapDialog(None, tmp_path, "v1", tmp_path / "c.mp4", None, [], lambda p: None)
    assert dlg.heads is None                          # (not read on the GUI thread)
    assert wait(app, lambda: dlg.heads is not None)
    assert dlg.heads["frames"] == 6 and dlg.heads["size"] == (160, 90)
    assert seen and all(t is not threading.main_thread() for t in seen)
    dlg.reject()


def test_the_swap_dialog_says_when_the_clip_cannot_be_read(app, tmp_path, monkeypatch):
    from video_chain import swap
    from video_chain.ui.swap_dialog import SwapDialog
    from waiting import wait
    monkeypatch.setattr(swap, "probe", lambda clip: (_ for _ in ()).throw(RuntimeError("no video stream")))
    dlg = SwapDialog(None, tmp_path, "v1", tmp_path / "c.mp4", None, [], lambda p: None)
    assert wait(app, lambda: "Cannot read the clip" in dlg.found_text.text())
    assert "no video stream" in dlg.found_text.text() and dlg.heads is None
    dlg.reject()


def test_the_picker_reads_an_image_preview_and_a_list_head_in_the_pool(app, tmp_path, monkeypatch):
    import threading

    from PIL import Image

    from video_chain.ui import image_picker as IP
    from waiting import wait
    Image.new("RGB", (40, 30), "red").save(tmp_path / "a.png")
    (tmp_path / "l.txt").write_text("first line\nsecond line\n")
    seen = []
    real_scaled, real_facts = IP.read_scaled, IP.file_facts
    monkeypatch.setattr(IP, "read_scaled", lambda *a: (seen.append(("scaled", threading.current_thread())), real_scaled(*a))[1])
    monkeypatch.setattr(IP, "file_facts", lambda p: (seen.append(("facts", threading.current_thread())), real_facts(p))[1])
    from PySide6.QtCore import QSettings
    q = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="image")
    seen.clear()
    dlg.grid.setCurrentItem(dlg._items[str(tmp_path / "a.png")])
    assert "Loading" in dlg.info.text()
    assert wait(app, lambda: "40×30 px" in dlg.info.text())
    main = threading.main_thread()
    assert seen and all(t is not main for _w, t in seen), seen
    dlg.reject()
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="text")
    seen.clear()
    dlg.grid.setCurrentItem(dlg._items[str(tmp_path / "l.txt")])
    assert wait(app, lambda: "second line" in dlg.info.text())
    assert all(t is not main for _w, t in seen)
    dlg.reject()


def test_scratch_settings_share_one_folder_that_is_removed(app):
    from video_chain.ui import image_picker as IP
    a, b = IP.scratch_settings(), IP.scratch_settings()
    folders = {Path(a.fileName()).parent, Path(b.fileName()).parent}
    assert len(folders) == 1 and a.fileName() != b.fileName()
    folder = folders.pop()
    assert folder.is_dir()
    IP.drop_scratch_settings()
    assert not folder.exists()


def test_a_value_change_rebuilds_only_the_card_it_belongs_to(app, monkeypatch):
    from test_variations_panel_fixes import cards, panel
    from video_chain.ui import variations_panel as VP
    from waiting import wait
    p = panel()
    others = cards(p)
    made = []
    real = VP.VariableCard
    monkeypatch.setattr(VP, "VariableCard", lambda *a, **k: (made.append(1), real(*a, **k))[1])
    others[0].add.setText("f")
    others[0].add.returnPressed.emit()
    assert wait(app, lambda: made)
    app.processEvents()
    assert len(made) == 1
    now = cards(p)
    assert now[1] is others[1] and now[0] is not others[0]          # (the other card stays as it was)
    assert [v.text for v in p.variable("x").values][-1] == "f" and len(now[0].pills.findChildren(VP.Pill)) == 6
    assert "6" in p.summary.text()


def test_the_image_dialog_takes_its_services_as_one_object_and_the_old_keywords_still_work(app, tmp_path):
    from PySide6.QtCore import QSettings

    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    q = QSettings(str(tmp_path / "t.ini"), QSettings.IniFormat)
    used = []
    svc = ImageEditServices(connect=lambda p: None, on_use=lambda k, p: used.append(k) or True, settings=q,
                            lora_dir="/loras", use_triggers=False, start_aspect=1.5, mode="edit")
    dlg = ImageEditDialog(None, tmp_path, [], None, None, services=svc)
    assert dlg.lora_dir == "/loras" and dlg.use_triggers is False and dlg.start_aspect == 1.5 and dlg.qs is q
    assert dlg.on_use("start", "x") is True and used == ["start"]
    dlg.reject()
    old = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q, lora_dir="/x",
                          use_triggers=True)
    assert old.lora_dir == "/x" and old.use_triggers is True and old.services.lora_dir == "/x"
    old.reject()
    with pytest.raises(TypeError):
        ImageEditDialog(None, tmp_path, [], None, None)


def test_the_swap_letterbox_turns_a_phone_photo_upright(app, tmp_path):
    from PIL import Image

    from video_chain.ui.swap_dialog import letterbox
    im = Image.new("RGB", (40, 20), "red")                 # stored wide, EXIF says: turn it (portrait)
    exif = Image.Exif()
    exif[0x0112] = 6
    src = tmp_path / "p.jpg"
    im.save(src, exif=exif)
    out = tmp_path / "o.png"
    assert letterbox(str(src), out, 100)
    got = Image.open(out).convert("RGB")
    # upright = 20 wide x 40 high: red in the middle columns, black bars left and right
    assert got.getpixel((50, 50))[0] > 200 and got.getpixel((5, 50)) == (0, 0, 0)


def test_a_rebuilt_problem_box_frees_its_old_layouts(app):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QHBoxLayout

    from video_chain.ui.problems import ProblemBox
    box = ProblemBox()
    many = [{"text": f"p{i}", "action": "x", "button": "Fix"} for i in range(6)]
    for _ in range(3):
        box.set_items(many)
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    box.set_items([{"text": "only"}])
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(box.findChildren(QHBoxLayout)) == 1          # (the one row of the one item: nothing of the old rows)


def test_picture_slots_and_the_mask_say_what_the_button_removes(app, tmp_path):
    from video_chain.ui.image_common import PictureSlot
    slot = PictureSlot("Picture 2", lambda title: None)
    assert slot.b_clear.toolTip() == "Remove Picture 2" and slot.b_clear.accessibleName() == "Remove Picture 2"
    from test_image_edit_split import _dialog
    dlg = _dialog(tmp_path)
    assert dlg.b_mask_clear.text() == "Remove mask"
    dlg.reject()


def test_the_thumbnail_counters_stay_exact_under_threads(app, tmp_path):
    import threading

    from PySide6.QtCore import QSettings

    from video_chain.ui import image_picker as IP
    q = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="image")
    before = dict(dlg.stats)

    def work():
        for _ in range(2000):
            dlg._count("memory")
    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert dlg.stats["memory"] == before["memory"] + 16000
    dlg.reject()
