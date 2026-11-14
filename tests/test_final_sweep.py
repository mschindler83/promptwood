"""Tests of the last sweep (MWB-24 and the others below)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from ui_helpers import FAKE, app, image, new_project, run_and_wait, settings, wait, win  # noqa: E402,F401


def test_no_shortcut_needs_ctrl_and_alt_together():
    """On a German keyboard AltGr is Ctrl+Alt: Ctrl+Alt+<key> collides with @ { } [ ] \\ ~ | (MWB-24)."""
    from video_chain.ui import shortcuts as S
    for s in S.SHORTCUTS:
        for k in s.keys.split(" / "):
            parts = {p.strip().lower() for p in k.split("+")}
            assert not {"ctrl", "alt"} <= parts, f"{s.id}: {k}"


def test_no_two_shortcuts_of_one_place_share_keys():
    from video_chain.ui import shortcuts as S
    seen: dict[tuple[str, str], str] = {}
    for s in S.SHORTCUTS:
        for k in s.keys.split(" / "):
            assert (s.where, k) not in seen, f"{s.id} and {seen[(s.where, k)]} both use {k}"
            seen[(s.where, k)] = s.id


# ------------------------------------------------------------------ MWC-18: refine_text returns a RefineResult
def test_refine_text_returns_a_result_with_a_text_or_ideas(app, tmp_path, monkeypatch):
    from ui_helpers import FAKE, app, settings  # noqa: F401
    from video_chain.ui import prompt_refiner as PR
    from video_chain.ui.main_window import MainWindow
    answers = {}

    class FakeDialog:
        refined_any = False

        def __init__(self, *a, **k):
            pass

        def exec(self):
            return 1

        def chosen(self):
            return answers["text"]

        def chosen_ideas(self):
            return answers["ideas"]

        def deleteLater(self):
            pass
    monkeypatch.setattr(PR, "RefineDialog", FakeDialog)
    w = MainWindow(settings(tmp_path), command=FAKE)
    try:
        answers.update(text="one text", ideas=None)
        r = w.refine_text("a cat", "image")
        assert isinstance(r, PR.RefineResult) and r.text == "one text" and r.ideas == []
        answers.update(text=None, ideas=["a", "b"])
        r = w.refine_text("a cat", "image")
        assert isinstance(r, PR.RefineResult) and r.ideas == ["a", "b"] and r.text == ""
    finally:
        w.close()


def test_the_image_window_takes_a_refine_result(app, tmp_path):
    from PySide6.QtCore import QSettings

    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    from video_chain.ui.prompt_refiner import RefineResult
    q = QSettings(str(tmp_path / "i.ini"), QSettings.IniFormat)
    answer = {}
    dlg = ImageEditDialog(None, tmp_path, [], None, None, services=ImageEditServices(connect=lambda p: None, on_use=lambda k, p: False, settings=q, connect_create=lambda p: None, refine=lambda text, kind, pictures=None: answer["r"]))
    try:
        dlg.set_mode("create")
        dlg.prompt.setPlainText("- an animal")
        answer["r"] = RefineResult(text="a dog on a beach")
        dlg._refine_prompt()
        assert dlg.prompt.toPlainText() == "a dog on a beach"
        answer["r"] = RefineResult(ideas=["a cat", "a dog"])
        dlg._refine_prompt()
        assert [v.text for v in dlg.variations.variable("idea").values] == ["a cat", "a dog"]
    finally:
        dlg.reject()


# ------------------------------------------------------------------ the frame window is read only when it is ready
def test_keep_a_moment_waits_for_the_probe_of_a_clip_without_a_frame_count(app, win, tmp_path, image):
    """A clip whose record has no frame count is probed in a worker: the position of the 'keep this moment' frame is
    set when the window is ready, not on the placeholder slider that the stepper replaces."""
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    del win.tree.nodes[ids[0]].made["frames"]                  # (so the window cannot be told at once)
    win.modal_dialogs = False
    dlg = win._keyframe_from_video(ids[0], 0.5, None, own=True)
    assert dlg is not None
    try:
        assert wait(app, lambda: dlg.ready)
        assert wait(app, lambda: dlg.slider.value() != 0)
        from video_chain import tree as T
        assert dlg.slider.value() == min(dlg.frames - 2, max(1, T.keyframe_frame(0.5)))
    finally:
        dlg.deleteLater()


def test_a_frame_window_runs_its_callbacks_when_ready(app, tmp_path):
    import subprocess

    from video_chain.ui.clip_dialogs import VideoFrameDialog
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=64x48:r=24:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)], check=True)
    dlg = VideoFrameDialog(None, vid, "frame")
    seen = []
    dlg.after_ready(lambda: seen.append(dlg.frames))
    assert seen == []                                              # (still reading the file)
    assert wait(app, lambda: dlg.ready)
    assert seen == [24]
    dlg.after_ready(lambda: seen.append("now"))                    # ready: at once
    assert seen == [24, "now"]
    dlg.deleteLater()
