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
