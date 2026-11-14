"""Main window: the clip cards / inspector, the destructive confirmations, the closing of the window and the
pre-flight of Generate (code review MWC-*, UX-2, UX-27, TST-20/21)."""
import os
import sys
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PIL import Image  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from test_ui import (FAKE, app, crop_accepted, image, new_project, run_and_wait,  # noqa: E402,F401
                     settings, silent_player, wait, win)
from video_chain import tree as T  # noqa: E402


def _recording_question(monkeypatch, answer=QMessageBox.Yes):
    """Replace QMessageBox.question; the list gets (title, text, default button) of every question."""
    asked = []

    def question(parent, title, text, buttons=None, defaultButton=None):
        asked.append((title, text, defaultButton))
        return answer
    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


# ---------------------------------------------------------------- MWC-3: the LoRA lists once per rebuild
def test_rebuild_path_reads_the_lora_lists_once(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a", "b", "c"))
    calls = {"known": 0, "dir": 0}
    known, lora_dir = win.known_loras, win._lora_dir
    monkeypatch.setattr(win, "known_loras", lambda: (calls.__setitem__("known", calls["known"] + 1), known())[1])
    monkeypatch.setattr(win, "_lora_dir", lambda: (calls.__setitem__("dir", calls["dir"] + 1), lora_dir())[1])
    win._rebuild_path()
    assert len(win.cards) == 3
    assert calls == {"known": 1, "dir": 1}


# ---------------------------------------------------------------- MWC-2: no file looked at twice per update
def test_update_cards_looks_at_each_file_once(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    node = win.tree.nodes[ids[0]]
    (win.project / "ends").mkdir(exist_ok=True)
    Image.new("RGB", (64, 36), (9, 9, 9)).save(win.project / "ends" / "e.png")
    Image.new("RGB", (64, 36), (9, 9, 9)).save(win.project / "ends" / "k.png")
    node.end_image = "ends/e.png"
    node.keyframes = [{"at": 0.5, "image": "ends/k.png", "note": ""}]
    wait(app, lambda: not win._joins, timeout=30)
    watched = {str(win.project / "ends" / "e.png"), str(win.project / "ends" / "k.png"),
               str(win.project / node.made["video"])}
    seen: dict[str, int] = {}
    where: dict = {}
    real = os.stat

    def counting(p, *a, **k):
        if str(p) in watched:
            seen[str(p)] = seen.get(str(p), 0) + 1
            import traceback
            where.setdefault(str(p), []).append("".join(traceback.format_stack(limit=9)[:-3]))
        return real(p, *a, **k)
    monkeypatch.setattr(os, "stat", counting)
    win._update_cards()
    monkeypatch.setattr(os, "stat", real)
    assert seen, "the files were not looked at at all"
    # (the card widget itself looks at the end image once more in PromptCard.set_end, outside this window code)
    assert max(v for k, v in seen.items() if not k.endswith("e.png")) == 1 and seen[str(win.project / "ends" / "e.png")] <= 2


def test_compare_candidates_are_the_same_inside_an_update(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    win.create_variants(ids[0], 2, False)
    run_and_wait(app, win, lambda: win.generate_all())
    wait(app, lambda: not win._joins, timeout=30)
    outside = {i: win.compare_candidates(i) for i in win.tree.nodes}
    win._cmp_memo = {}
    try:
        inside = {i: win.compare_candidates(i) for i in win.tree.nodes}
    finally:
        win._cmp_memo = None
    assert inside == outside and len(outside[ids[0]]) == 3 and outside[ids[0]][0] == ids[0]


# ---------------------------------------------------------------- UX-2 / UX-27: confirmations default to No
@pytest.mark.parametrize("action", ["delete"])
def test_delete_asks_with_no_as_default(app, win, tmp_path, image, monkeypatch, action):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    asked = _recording_question(monkeypatch, QMessageBox.No)
    win.node_action("delete", ids[1])
    assert asked and asked[0][0] == "Delete" and asked[0][2] == QMessageBox.No
    assert ids[1] in win.tree.nodes                     # (No: kept)


def test_keep_only_and_keep_branch_ask_with_no_as_default(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    win.node_action("variant", ids[0])
    other = win.selected_id()
    for nid in (ids[0], other):
        win.tree.nodes[nid].made = {"video": f"clips/{nid}.mp4", "frame": "f.png", "first": "g.png"}
    asked = _recording_question(monkeypatch, QMessageBox.No)
    win.keep_only(ids[0], [other])
    win.keep_branch(ids[0])
    assert [a[0] for a in asked] == ["Keep this one", "Keep this branch"]
    assert all(a[2] == QMessageBox.No for a in asked)


def test_freeze_asks_with_no_as_default(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    win.tree.nodes[ids[0]].prompt = "changed after it was made"
    asked = _recording_question(monkeypatch, QMessageBox.No)
    win.freeze_clip(ids[0])
    assert asked and asked[0][0] == "Freeze" and asked[0][2] == QMessageBox.No
    assert not win.tree.nodes[ids[0]].frozen


def test_quit_while_generating_defaults_to_keep_working(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    monkeypatch.setenv("FAKE_DELAY", "0.5")
    win.generate_path()
    wait(app, lambda: win.worker_alive)
    asked = _recording_question(monkeypatch, QMessageBox.No)
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert not ev.isAccepted() and win.running
    assert asked[0][0] == "Quit while generating?" and asked[0][2] == QMessageBox.No


# ---------------------------------------------------------------- MWC-1 / TST-20 / TST-21: closing the window
SLOW_WORKER = [sys.executable, "-c", "import time\nwhile True: time.sleep(0.05)"]   # (never quits by itself)


def _window_with_stuck_worker(tmp_path):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=SLOW_WORKER)
    w.open_project(tmp_path / "proj")
    w._start_worker()
    wait_for = time.time() + 10
    from PySide6.QtCore import QProcess
    while w.proc.state() != QProcess.Running and time.time() < wait_for:
        w.proc.waitForStarted(200)
    return w


def test_closing_keeps_the_window_alive_while_the_worker_ends(app, tmp_path, monkeypatch):
    w = _window_with_stuck_worker(tmp_path)
    ticks = []
    t = QTimer()
    t.timeout.connect(lambda: ticks.append(1))
    t.start(20)
    proc = w.proc
    w.close()
    t.stop()
    assert len(ticks) >= 20, "the event loop stood still while the generator was ended"
    assert proc.state() == proc.ProcessState.NotRunning


def test_closing_saves_the_project_before_the_worker_is_ended(app, tmp_path, monkeypatch):
    w = _window_with_stuck_worker(tmp_path)
    proc, states = w.proc, []
    real = w._write_tree
    monkeypatch.setattr(w, "_write_tree", lambda: (states.append(proc.state()), real())[1])
    w.close()
    assert states and states[0] == proc.ProcessState.Running


def test_closing_a_run_saves_what_the_generator_recorded(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    monkeypatch.setenv("FAKE_DELAY", "0.3")
    win.generate_path()
    wait(app, lambda: win.state(ids[1]) == "generating")    # (the first clip is recorded in tree.json by now)
    saved = []
    real = T.save
    monkeypatch.setattr(T, "save", lambda *a, **k: (saved.append(1), real(*a, **k))[1])
    asked = _recording_question(monkeypatch, QMessageBox.Yes)
    win.close()
    assert saved, "tree.json was not written: " + win.status.text()
    assert "outside this window" not in win.status.text()
    assert T.load(win.project).nodes[ids[0]].made


# ---------------------------------------------------------------- MWC-4: the pre-flight does not block
def test_start_checks_in_the_background(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    seen = []
    real = T.Checker

    def slow_checker(*a, **k):
        seen.append(__import__("threading").current_thread())
        time.sleep(0.4)
        return real(*a, **k)
    monkeypatch.setattr(T, "Checker", slow_checker)
    t0 = time.monotonic()
    win.generate_path()
    assert time.monotonic() - t0 < 0.3, "Generate waited for the check"
    assert win.running and not win.b_gen.isEnabled()          # (locked while it is checked)
    assert "Checking" in win.status.text()
    wait(app, lambda: win.worker_alive)
    assert any(th is not __import__("threading").main_thread() for th in seen)
    monkeypatch.setattr(T, "Checker", real)
    wait(app, lambda: not win.running)
    assert win.state(ids[0]) == "done"


def test_stop_while_checking_starts_nothing(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    real = T.Checker

    def slow_checker(*a, **k):
        time.sleep(0.4)
        return real(*a, **k)
    monkeypatch.setattr(T, "Checker", slow_checker)
    win.generate_path()
    assert win.running
    win.stop()
    assert not win.running
    time.sleep(0.6)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.02)
    assert not win.running and not win.worker_alive


def test_a_problem_found_in_the_background_is_reported(app, win, tmp_path, image):
    win.open_project(tmp_path / "proj")
    win.generate_path()
    wait(app, lambda: not win.running and "prompt of clip #1" in win.status.text(), timeout=10)
    assert not win.worker_alive


# ---------------------------------------------------------------- MWC-5: a GPU query that cannot start
def test_gpu_poll_goes_on_after_a_query_that_could_not_start(app, win, monkeypatch):
    import shutil

    from PySide6.QtCore import QProcess
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    real = QProcess.start
    monkeypatch.setattr(QProcess, "start", lambda self, prog, args=[]: real(self, "/nonexistent/nvidia-smi", args))
    win._gpu_proc = None
    win._poll_gpu()
    wait(app, lambda: win._gpu_proc is None, timeout=5)
    assert win.gpu.text() == "GPU: ?"


# ---------------------------------------------------------------- MWC-14: LoRA labels
def test_lora_labels_show_the_folder_only_for_equal_names():
    from video_chain.ui.main_window import label_loras
    items = [("/a/x.safetensors", "x.safetensors"), ("/b/one/y.safetensors", "one/y.safetensors"),
             ("/b/two/y.safetensors", "two/y.safetensors")]
    assert label_loras(items) == [("/a/x.safetensors", "x"), ("/b/one/y.safetensors", "y (one)"),
                                  ("/b/two/y.safetensors", "y (two)")]


# ---------------------------------------------------------------- MWC-12: an ignored action says why
def test_actions_while_generating_say_why(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    monkeypatch.setenv("FAKE_DELAY", "0.3")
    win.generate_path()
    wait(app, lambda: win.worker_alive)
    win.status.setText("")
    win.node_action("continue", ids[0])
    assert "Wait until generating has finished" in win.status.text()
    win.status.setText("")
    win._context(ids[0], win.mapToGlobal(win.rect().center()))
    assert "Wait until generating has finished" in win.status.text()
    win.stop()
    wait(app, lambda: not win.running, timeout=20)


# ---------------------------------------------------------------- UX-8: a frozen clip says so where one pressed
def test_frozen_clip_shows_its_notice_at_the_mouse(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtWidgets import QToolTip
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert win.freeze_clip(ids[0], confirm=False)
    shown = []
    monkeypatch.setattr(QToolTip, "showText", lambda pos, text, *a: shown.append(text))
    win.node_action("end_clear", ids[0])
    assert shown and "is frozen" in shown[0] and "is frozen" in win.status.text()


# ---------------------------------------------------------------- MWC-26 / MWC-27: keyframes
def test_keyframe_note_without_an_index_does_nothing(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    ids = new_project(win, tmp_path, image, ("a",))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("x", True))
    win.keyframe_action("note", ids[0], {"at": 1.0})            # (no "index": used to raise TypeError)
    win.keyframe_action("remove", ids[0], {"at": 1.0})


def test_failed_keyframe_removal_is_reported(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    Image.new("RGB", (64, 36), (9, 9, 9)).save(tmp_path / "k.png")
    win.tree.nodes[ids[0]].keyframes = [{"at": 0.5, "image": "k.png", "note": ""}]

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(T, "remove_keyframe", boom)
    win.keyframe_action("remove", ids[0], {"index": 0})
    assert win.status.text().startswith("Could not remove the keyframe") and "disk gone" in win.status.text()


# ---------------------------------------------------------------- MWC-19 / MWC-20: the Sound row's sentence
def test_sound_sentence_ends_with_the_laid_over_part(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    second = win.tree.nodes[ids[1]]
    second.overrides = {"source_sound": "previous"}
    info = win._source_sound_text(second)
    assert info.can_repeat and info.text.endswith("exactly as it is. Write in the prompt what happens; the sound "
                                                  "comes from the clip before.")
    laid = win._source_sound_text(second, win._laid_over_text(False))
    assert laid.text.startswith(info.text[:-len(" Write in the prompt what happens; the sound comes from the clip "
                                                "before.") - 1] + ", Write in the prompt")
    assert laid.text.endswith("it may be shorter than 2 s.") and " Laid over: " in laid.text
    assert " Mixed in: " in win._source_sound_text(second, win._laid_over_text(True)).text
    avail, text, max_guide, can_repeat = info                    # (a plain tuple, too)
    assert not avail and max_guide == 0 and can_repeat


# ---------------------------------------------------------------- MWC-24: a LoRA found is remembered
def test_find_lora_walks_the_folders_once(app, win, tmp_path, monkeypatch):
    from video_chain import comfy as C
    root = tmp_path / "loras"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "head_swap_v1.safetensors").write_bytes(b"x")
    monkeypatch.setattr(C, "comfy_lora_roots", lambda d: [root])
    walks = []
    real = Path.rglob
    monkeypatch.setattr(Path, "rglob", lambda self, pat: (walks.append(1), real(self, pat))[1])
    win.h3_loras, win.turbo_loras = [], []
    first = win.find_lora("head_swap")
    assert first and first.endswith("head_swap_v1.safetensors")
    assert win.find_lora("head_swap") == first and len(walks) == 1
    (root / "sub" / "head_swap_v1.safetensors").unlink()           # gone: looked for again
    assert win.find_lora("head_swap") is None and len(walks) == 2


# ---------------------------------------------------------------- MWC-23: the canvas of a video being edited
def _video_window(app, win, tmp_path):
    from test_ui import make_real_video
    src = make_real_video(tmp_path / "real.mp4")
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.0)
    wait(app, lambda: not win._converts)
    return src, v


def test_canvas_for_does_not_copy_the_whole_tree(app, win, tmp_path, monkeypatch):
    import copy
    src, v = _video_window(app, win, tmp_path)
    monkeypatch.setattr(copy, "deepcopy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("deep copy")))
    w1, h1 = win._canvas_for(str(src), 0, v)
    w2, h2 = win._canvas_for(str(src), 90, v)
    assert (w1, h1) == (w2 and h2, w2) or w1 > h1 and h2 > w2       # (turned: portrait)
    assert win.tree.nodes[v].source["start"] == 0.0 and win.tree.nodes[v].source.get("rotate", 0) == 0   # untouched


# ---------------------------------------------------------------- MWC-32: a missing video file is announced
def test_missing_video_file_is_announced_before_its_window(app, win, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from video_chain.ui.main_window import VideoRangeDialog
    src, v = _video_window(app, win, tmp_path)
    src.unlink()
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: told.append(a[2]))
    monkeypatch.setattr(VideoRangeDialog, "exec", lambda self: QDialog.Rejected)
    win.edit_video(v)
    assert told and "not there any more" in told[0]


# ---------------------------------------------------------------- MWC-37: a conversion that cannot be saved
def test_converted_video_that_cannot_be_saved_is_dropped(app, win, tmp_path, monkeypatch):
    import shutil
    src, v = _video_window(app, win, tmp_path)
    node = win.tree.nodes[v]
    before = dict(node.made)
    new = dict(before)
    for k in ("video", "frame", "first"):
        if before.get(k):
            f = win.project / before[k]
            copy = f.with_name("copy_" + f.name)
            shutil.copy(f, copy)
            new[k] = str(copy.relative_to(win.project))
    win._converts[v] = "tok"
    monkeypatch.setattr(win, "_write_tree", lambda: False)
    win._converted(v, new, "", "tok", str(win.project))
    assert win.tree.nodes[v].made == before                         # the record is as it was
    assert (win.project / before["video"]).is_file()
    assert not (win.project / new["video"]).exists()                # (the new files are put aside)


def test_join_result_without_a_project_is_ignored(app, win):
    win.close_project() if hasattr(win, "close_project") else None
    win.project = None
    win._joins = 1
    win._joined(("x.mp4", 1.0, []), None, "play", "", "None")
    assert win._joins == 0 and "Path joined" not in win.status.text()


# ---------------------------------------------------------------- MWC-25: closing while a join runs
def test_closing_asks_while_a_video_is_being_joined(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    win._joins = 1
    asked = _recording_question(monkeypatch, QMessageBox.No)
    ev = QCloseEvent()
    win.closeEvent(ev)
    win._joins = 0
    assert not ev.isAccepted() and asked[0][0] == "Quit while working?" and asked[0][2] == QMessageBox.No


# ---------------------------------------------------------------- MWC-41: ComfyUI is up while a dialog is open
def test_waiting_for_comfyui_says_it_waits_for_the_dialog(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtWidgets import QApplication
    ids = new_project(win, tmp_path, image, ("a",))
    win._waiting_comfy = [ids[0]]
    win.comfy_up = False
    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: win))
    win._comfy_state(True)
    assert "close the dialog" in win.status.text() and win._waiting_comfy == [ids[0]]
    win._waiting_comfy = None


# ---------------------------------------------------------------- MWC-36: a job that cannot be written
def test_job_that_cannot_be_written_ends_the_generator(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtCore import QProcess
    new_project(win, tmp_path, image, ("a",))

    class Dead:
        def state(self):
            return QProcess.Running

        def write(self, data):
            return -1
    win._job = {"targets": [], "sent": False, "refine": None}
    win.proc = Dead()
    killed = []
    monkeypatch.setattr(win, "_kill", lambda: killed.append(1))
    try:
        win._send_job()
    finally:
        win.proc, win._job = None, None
    assert killed and "Could not send the job" in win.status.text()


def test_staying_after_a_stopped_run_keeps_the_window_working(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    monkeypatch.setenv("FAKE_DELAY", "0.3")
    win.generate_path()
    wait(app, lambda: win.state(ids[1]) == "generating")
    assert win._gpu_timer.isActive() and win._comfy_timer.isActive()
    _recording_question(monkeypatch, QMessageBox.Yes)          # stop and quit ...
    monkeypatch.setattr(win, "_may_leave_project", lambda: False)   # ... but keep the window (to save elsewhere)
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert not ev.isAccepted()
    assert win._gpu_timer.isActive() and win._comfy_timer.isActive(), "the GPU / ComfyUI status stopped updating"
