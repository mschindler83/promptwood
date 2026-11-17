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


def test_the_image_dialog_takes_its_services_as_one_object(app, tmp_path):
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
    with pytest.raises(TypeError):
        ImageEditDialog(None, tmp_path, [], None, None, services=ImageEditServices())


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


def test_the_image_dialog_stops_a_running_job_with_stop_like_the_other_windows(app, tmp_path):
    from test_image_edit_split import _dialog
    dlg = _dialog(tmp_path)
    assert dlg.b_cancel.text() == "Stop"
    dlg.reject()


def test_the_play_cleaned_player_says_when_it_cannot_play(app, tmp_path):
    from PySide6.QtMultimedia import QMediaPlayer

    from test_ui_media_fixes import make_wav
    from video_chain.ui.audio_editor import AudioRangeDialog
    dlg = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=True)
    dlg.clean = {"file": str(make_wav(tmp_path / "c.wav")), "start": 0.0, "end": 1.0}
    dlg._update_enabled()
    dlg.play_cleaned()
    assert dlg.clean_player is not None and dlg.b_clean_play.isEnabled()
    dlg.clean_player.errorOccurred.emit(QMediaPlayer.Error.ResourceError, "no audio device")
    assert "no audio device" in dlg.clean_info.text() and not dlg.b_clean_play.isEnabled()
    dlg.done(0)


def test_a_compare_player_that_fails_is_named_in_its_clip(app, tmp_path):
    from PySide6.QtMultimedia import QMediaPlayer

    from video_chain.ui.clip_dialogs import CompareDialog
    dlg = CompareDialog(None, [("a", "A", "", tmp_path / "a.mp4"), ("b", "B", "", tmp_path / "b.mp4")],
                        lambda nid: True)
    assert len(dlg.players) == 2
    dlg.players[1].errorOccurred.emit(QMediaPlayer.Error.FormatError, "unsupported format")
    assert "unsupported format" in dlg.errors["b"].text() and not dlg.errors["b"].isHidden()
    assert dlg.errors["a"].isHidden()
    dlg.reject()


def test_the_mask_painter_and_the_gallery_take_their_colours_from_the_theme():
    import re
    ui = Path(__file__).parent.parent / "src" / "video_chain" / "ui"
    literal = re.compile(r"QColor\(\s*\d")
    mask = [ln for ln in (ui / "mask_painter.py").read_text().splitlines() if literal.search(ln)]
    # the mask's own data colours stay literal: the red tint of the drawn area and the black / white of the mask itself
    assert all("MASK_COLOUR" in ln or "erase" in ln for ln in mask), mask
    assert [ln for ln in (ui / "results_gallery.py").read_text().splitlines() if literal.search(ln)] == []


def test_the_audio_editor_and_the_speak_dialog_say_range_not_part():
    import re

    from test_ui_shared import _shown_strings
    ui = Path(__file__).parent.parent / "src" / "video_chain" / "ui"
    bad = [f"{n}:{line}: {s[:50]}" for n in ("audio_editor.py", "speak_dialog.py", "video_editor.py", "tts_dialog.py")
           for line, s in _shown_strings(ui / n) if re.search(r"\bparts?\b", s) and not s.startswith("part")]
    assert not bad, bad


def test_every_key_of_the_dialogs_is_in_the_shortcut_table_and_bound_from_it():
    import re

    from video_chain.ui import shortcuts as S
    listed = {s.keys for s in S.SHORTCUTS}
    for k in ("Delete", "B", "F", "[", "]", "Home / End", "Ctrl+Shift+Z / Ctrl+Y", "Up / Down / Left / Right",
              "Enter", "Menu", "Home"):
        assert any(k in keys.split(" / ") or k == keys for keys in listed), k
    where = {s.id: s.where for s in S.SHORTCUTS}
    assert where["image_delete"] == where["image_before_after"] == where["image_favorite"] == "image dialog"
    assert where["range_set_start"] == where["range_set_end"] == "range editors"
    assert where["compare_first_last"] == "compare player" and where["graph_open"] == "clip graph"
    assert [q.toString() for q in S.sequences("mask_brush")] == ["[", "]"]
    assert [q.toString() for q in S.sequences("mask_redo")] == ["Ctrl+Shift+Z", "Ctrl+Y"]
    ui = Path(__file__).parent.parent / "src" / "video_chain" / "ui"
    for name in ("audio_editor", "video_editor", "image_edit", "mask_painter", "compare_player", "prompt_refiner",
                 "clip_dialogs"):
        text = (ui / f"{name}.py").read_text()
        assert not re.search(r"QKeySequence\(\"", text), f"{name} binds a literal key"


def test_a_dialog_remembers_its_size_and_position_in_its_settings(app, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QDialog

    from video_chain.ui.widgets import remember_geometry
    q = QSettings(str(tmp_path / "g.ini"), QSettings.IniFormat)
    first = QDialog()
    first.resize(500, 410)
    remember_geometry(first, q, "demo")
    first.done(0)
    assert q.value("geometry/demo") is not None
    second = QDialog()
    second.resize(300, 200)
    remember_geometry(second, q, "demo")
    assert (second.width(), second.height()) == (500, 410)
    other = QDialog()
    other.resize(300, 200)
    remember_geometry(other, q, "another")                  # (each dialog has its own entry)
    assert (other.width(), other.height()) == (300, 200)
    remember_geometry(other, None, "x")                      # (no settings: nothing remembered, nothing breaks)
    q.setValue("geometry/broken", "not a geometry")
    broken = QDialog()
    remember_geometry(broken, q, "broken")


def test_the_big_dialogs_remember_their_geometry(app):
    import re
    ui = Path(__file__).parent.parent / "src" / "video_chain" / "ui"
    for name in ("image_edit", "image_picker", "lora_window", "prompt_refiner", "swap_dialog", "audio_editor",
                 "video_editor"):
        assert "remember_geometry(self" in (ui / f"{name}.py").read_text(), name
    assert re.search(r"def remember_geometry", (ui / "widgets.py").read_text())


def test_garbage_is_collected_on_the_gui_thread_not_in_a_worker(app, monkeypatch):
    import gc
    import threading

    from video_chain.ui import workers
    workers.run_in_background(lambda: 1, on_done=lambda r: None)          # (starts the relay and the collector)
    assert workers._gc_timer is not None and not gc.isenabled()
    seen = []
    real = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *a: (seen.append(threading.current_thread()), real(*a))[1])
    workers._gc_timer.timeout.emit()
    assert seen and all(t is threading.main_thread() for t in seen)


def _buddies(widget) -> dict:
    from PySide6.QtWidgets import QLabel
    return {lab.text(): lab.buddy() for lab in widget.findChildren(QLabel) if lab.buddy() is not None}


def test_the_form_labels_of_the_dialogs_are_buddies_of_their_fields(app, tmp_path):
    from PySide6.QtCore import QSettings

    from test_image_edit_split import _dialog
    from test_ui_media_fixes import make_wav
    from video_chain.ui import image_picker as IP
    from video_chain.ui.audio_editor import AudioRangeDialog
    img = _dialog(tmp_path)
    got = _buddies(img)
    for text in ("Model", "Shape", "Quality", "Results", "Seed", "Preset"):
        assert text in got, (text, sorted(got))
    assert got["Seed"] is img.seed and got["Quality"] is img.quality and got["Model"] is img.edit_model
    assert got["Seed"].accessibleName() == "Seed"
    img.reject()
    ad = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=False)
    got = _buddies(ad)
    assert got["Start"] is ad.start_spin and got["End"] is ad.end_spin
    ad.done(0)
    q = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    pk = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="image")
    got = _buddies(pk)
    assert got["Sort"] is pk.sort and got["Size"] is pk.zoom
    pk.reject()
    from video_chain.ui.lora_window import LoraWindow
    lw = LoraWindow(None, [], settings=q)
    assert _buddies(lw)["Pictures"] is lw.pictures
    lw.done(0)


def _next_with_focus(w):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractSpinBox
    n = w.nextInFocusChain()
    while n is not w and not (n.focusPolicy() & Qt.FocusPolicy.TabFocus and n.isEnabledTo(n.window())
                              and not isinstance(n.parentWidget(), QAbstractSpinBox)):
        n = n.nextInFocusChain()
    return n


def test_the_range_fields_are_tabbed_in_the_order_they_are_shown(app, tmp_path):
    from test_ui_media_fixes import make_wav
    from video_chain.ui.audio_editor import AudioRangeDialog
    dlg = AudioRangeDialog(None, str(make_wav(tmp_path / "s.wav")), player=False)
    dlg.show()
    here = []
    w = dlg.start_spin
    for _ in range(4):                            # start field, Start here, end field, End here
        here.append(w)
        w = _next_with_focus(w)
    assert here[0] is dlg.start_spin and here[1].text() == "Start here"
    assert here[2] is dlg.end_spin and here[3].text() == "End here"
    dlg.done(0)


def test_the_graph_the_timeline_and_the_keyframe_strip_describe_themselves(app):
    from test_ui_media_fixes import infos
    from video_chain.ui.graph import GraphView
    from video_chain.ui.keyframe_strip import KeyframeStrip
    from video_chain.ui.timeline import Block, Timeline
    g = GraphView()
    assert g.accessibleName() == "Clip graph"
    nodes = infos()
    g.show_nodes(nodes)
    assert f"{len(nodes) - 1} clip" in g.accessibleDescription()
    first = nodes[1].id
    g.set_focus(first)
    assert nodes[1].title in g.accessibleDescription() and "Focused" in g.accessibleDescription()
    t = Timeline()
    assert t.accessibleName() == "Timeline of the played path"
    t.show_path([Block("n1", "#1", 4.0, "done", True, None), Block("n2", "#2", 6.5, "new", True, None)], "n2")
    assert "2 clips" in t.accessibleDescription() and "10.5 s" in t.accessibleDescription()
    assert "#2" in t.accessibleDescription()
    k = KeyframeStrip()
    assert k.accessibleName() == "Middle keyframes"
    k.set_data(5.0, [{"at": 1.0, "pixmap": None, "note": "", "ok": True}, {"at": 3.5, "pixmap": None, "note": "", "ok": True}])
    assert "2 keyframes" in k.accessibleDescription() and "1.00 s" in k.accessibleDescription()
    k.set_data(5.0, [])
    assert "no keyframes" in k.accessibleDescription()
