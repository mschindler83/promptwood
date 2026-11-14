"""Main window behaviour of the second review pass: a project opened from its backup, the refiner setup in the Simple
view, closing, ComfyUI's /free, busy flags, settings, keyboard and drops (code review HIGH items, UX-*, TST-3/4, XC-10/15)."""
import os
import threading
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from conftest import settings as make_settings  # noqa: E402
from test_ui import FAKE, app, image, new_project, settings, silent_player, wait, win  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402


def _window(tmp_path, view="advanced"):          # (one settings file per view)
    from video_chain.ui.main_window import MainWindow
    return MainWindow(make_settings(tmp_path, name=f"{view}.ini", view=view), command=FAKE)


# ---------------------------------------------------------------- the backup was opened
def test_a_project_opened_from_its_backup_says_so(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a",))
    folder = win.project
    T.save(folder, win.tree)                       # twice: the first file is now the backup
    win.close_project()
    (folder / T.TREE_FILE).write_text('{"nodes": [ truncated')
    assert win.open_project(folder, quiet=True)
    assert "backup" in win.status.text().lower() and "tree.json.bak" in win.status.text()
    assert "backup" in win.log.toPlainText().lower()
    assert win._notices


def test_a_good_project_opens_without_a_notice(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a",))
    folder = win.project
    win.close_project()
    assert win.open_project(folder, quiet=True)
    assert "backup" not in win.status.text().lower() and not win._notices


# ---------------------------------------------------------------- UX-4 / UX-3: the Simple view
def test_the_refiner_setup_is_reachable_in_the_simple_view(app, tmp_path):
    w = _window(tmp_path, "simple")
    try:
        assert w.simple
        a = w._comfy_form
        for ctl in (w.llm_on, w.llm_server, w._gguf_row, w.llm_cpu_moe):
            assert w.view.visible((a, ctl)), ctl
    finally:
        w.close()


def test_the_runpod_status_names_the_advanced_view_in_simple(app, tmp_path):
    w = _window(tmp_path, "simple")
    try:
        assert "Advanced view" in w._runpod_key_hint()
    finally:
        w.close()
    w = _window(tmp_path, "advanced")
    try:
        assert "Advanced view" not in w._runpod_key_hint()
        assert "Cloud upscaling" in w._runpod_key_hint()
    finally:
        w.close()


# ---------------------------------------------------------------- closing: a short grace for llama-server
def test_closing_gives_the_language_model_server_a_short_grace(app, win):
    calls = []

    class Fake:
        def stop(self, wait=True, grace=20):
            calls.append((wait, grace))
            return True
    win._llama = Fake()
    win.close()
    assert calls and calls[0][0] is True and calls[0][1] <= 5


def test_a_leftover_server_of_ours_is_stopped_at_start(app, tmp_path, monkeypatch):
    from video_chain.llm_server import LlamaServer
    seen = []
    monkeypatch.setattr(LlamaServer, "stop", lambda self, wait=True, grace=20: seen.append((self.log_dir, grace)) or False)
    w = _window(tmp_path)
    try:
        wait(app, lambda: seen)
        assert seen[0][0] is not None                   # (it knows the pid file's folder)
    finally:
        w.close()


# ---------------------------------------------------------------- ComfyUI /free through comfy.free_models
def test_unloading_comfy_uses_the_shared_function_and_says_when_it_fails(app, win, monkeypatch):
    from video_chain import comfy
    calls = []
    monkeypatch.setattr(comfy, "free_models", lambda url, unload_models=True, timeout=10, opener=None:
                        calls.append((url, timeout)) or False)
    win._comfy_model = "qwen"
    win._free_comfy()
    wait(app, lambda: "Could not unload" in win.status.text())
    assert calls and calls[0][0].startswith("http")
    assert win._comfy_model == "qwen"                    # not reported as unloaded: it still holds the memory


def test_unloading_comfy_clears_the_model_when_it_worked(app, win, monkeypatch):
    from video_chain import comfy
    monkeypatch.setattr(comfy, "free_models", lambda *a, **k: True)
    win._comfy_model = "qwen"
    win._free_comfy()
    wait(app, lambda: win._comfy_model is None)


def test_make_room_and_closing_use_free_models_too(app, win, monkeypatch):
    from video_chain import comfy, llm_server
    calls = []
    monkeypatch.setattr(comfy, "free_models", lambda url, unload_models=True, timeout=10, opener=None:
                        calls.append(timeout) or True)
    monkeypatch.setattr(llm_server, "free_vram_mb", lambda: 100)
    win._make_room(5000, "http://127.0.0.1:1", "a model", settle=0)
    done = threading.Event()
    win._post_free("http://127.0.0.1:1", done)
    assert done.wait(5)
    assert len(calls) == 2


def test_no_hand_built_free_requests_are_left():
    src = (Path(__file__).parent.parent / "src/video_chain/ui/main_window.py").read_text()
    assert '"/free"' not in src and "/free\"" not in src
    assert 'Path("/' + 'run/media' not in src


# ---------------------------------------------------------------- settings are passed, never defaulted
def test_the_window_needs_its_settings():
    import inspect
    from video_chain.ui.main_window import MainWindow, NewProjectDialog
    assert inspect.signature(MainWindow.__init__).parameters["settings"].default is inspect.Parameter.empty
    assert inspect.signature(NewProjectDialog.__init__).parameters["settings"].default is inspect.Parameter.empty


def test_the_new_project_dialog_picks_images_with_the_windows_settings(app, win, tmp_path, monkeypatch):
    from video_chain.ui import image_picker as IP
    from video_chain.ui import main_window as M
    got = []
    monkeypatch.setattr(IP, "pick_image", lambda parent, title, qs, *a, **k: got.append(qs) or None)
    d = M.NewProjectDialog(None, str(tmp_path), win.qs)
    d._choose_image()
    assert got == [win.qs]
    d.deleteLater()


# ---------------------------------------------------------------- media errors are RuntimeErrors
def test_split_video_survives_an_unreadable_clip(app, win, tmp_path, monkeypatch):
    from test_ui import make_real_video
    from video_chain import chain, media
    src = make_real_video(tmp_path / "real.mp4")
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.5)
    wait(app, lambda: not win._converts)
    win.tree.nodes[v].made.pop("frames", None)

    def broken(path):
        raise media.MediaError("ffprobe could not read it")
    monkeypatch.setattr(chain, "probe", broken)
    assert win.split_video(v) is None                # (no dialog, no traceback)
    assert "ffprobe could not read it" in win.status.text()


# ---------------------------------------------------------------- TST-3: one exception never leaves the window busy
class _Boom(BaseException):
    """Not an Exception: what `except Exception` in a worker does not catch."""


def test_a_failing_conversion_does_not_leave_the_window_busy(app, win, tmp_path, monkeypatch):
    from test_ui import make_real_video
    src = make_real_video(tmp_path / "real.mp4")
    monkeypatch.setattr(T, "make_video", lambda *a, **k: (_ for _ in ()).throw(_Boom()))
    win.open_project(tmp_path / "proj")
    win.node_action("delete", "n1")
    v = win.add_video(None, str(src), 0.0, 1.5)
    assert v
    wait(app, lambda: not win._converts)
    assert win.live.get(v) != "converting"


def test_a_failing_remix_does_not_leave_the_window_busy(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    monkeypatch.setattr(T.Checker, "remix_due", lambda self, nid: True)
    monkeypatch.setattr(T, "remix_sounds", lambda *a, **k: (_ for _ in ()).throw(_Boom()))
    win._start_remix()
    assert win._remixing
    wait(app, lambda: not win._remixing)
    assert win._joins == 0


def test_a_failing_join_does_not_leave_the_window_busy(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    wait(app, lambda: not win._joins)
    monkeypatch.setattr(T, "join_path", lambda *a, **k: (_ for _ in ()).throw(_Boom()))
    win.join_selected("play")
    wait(app, lambda: not win._joins)


def test_speaking_does_not_stay_busy_when_it_cannot_start(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    class Eng:
        name = "local"
        def stop(self):
            pass
    monkeypatch.setattr(win, "_speak_engine", lambda nid: Eng())
    monkeypatch.setattr(win, "_update_buttons", lambda: (_ for _ in ()).throw(RuntimeError("broken")))
    with pytest.raises(RuntimeError):
        win.speak_clip(ids[0])
    assert win._tts_busy is False


# ---------------------------------------------------------------- UIM-26: one index per reload
def test_the_uses_callback_builds_its_index_once_per_change(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import image_edit as IE
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a",))
    built = []
    real = IE.uses_index
    monkeypatch.setattr(IE, "uses_index", lambda tree: built.append(1) or real(tree))
    for _ in range(5):
        win._uses({"uses": []})
    assert len(built) == 1
    win.tree.nodes[ids[0]].end_image = "ends/x.png"
    win._changed()                                  # something changed: the next question builds it again
    win._uses({"uses": []})
    assert len(built) == 2


# ---------------------------------------------------------------- MWA-20: a file is looked at once per card update
def test_cards_do_not_look_at_their_files_again_themselves(app, win, tmp_path, image, monkeypatch):
    from PIL import Image
    from test_ui import make_real_video
    ids = new_project(win, tmp_path, image, ("a", "b"))
    (win.project / "ends").mkdir(exist_ok=True)
    Image.new("RGB", (64, 36), (9, 9, 9)).save(win.project / "ends" / "e.png")
    win.tree.nodes[ids[0]].end_image = "ends/e.png"
    src = make_real_video(tmp_path / "real.mp4")
    v = win.add_video(ids[1], str(src), 0.0, 1.0)
    wait(app, lambda: not win._converts)
    watched = {str(win.project / "ends" / "e.png"): 0, str(src): 0}
    real = os.stat

    def counting(p, *a, **k):
        if str(p) in watched:
            watched[str(p)] += 1
        return real(p, *a, **k)
    monkeypatch.setattr(os, "stat", counting)
    win._update_cards()
    monkeypatch.setattr(os, "stat", real)
    assert watched[str(win.project / "ends" / "e.png")] == 1, watched
    assert watched[str(src)] == 1, watched


# ---------------------------------------------------------------- UIL-27: a clip of the project is not probed again
def test_the_frame_window_of_an_own_clip_does_not_probe(app, win, tmp_path, image, monkeypatch):
    from video_chain import chain
    from video_chain.ui import main_window as M
    ids = new_project(win, tmp_path, image, ("a", "b"))
    from test_ui import run_and_wait
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    clip = win.project / win.tree.nodes[ids[0]].made["video"]
    probed = []
    real = chain.probe_source
    monkeypatch.setattr(chain, "probe_source", lambda p: probed.append(p) or real(p))
    monkeypatch.setattr(M.VideoFrameDialog, "exec", lambda self: 0)
    win.end_from_video(ids[1], video=str(clip))
    assert probed == []
    dlg = win.end_from_video(ids[1], video=str(clip), dialog=False)
    assert dlg is not None and dlg.frames == win.tree.nodes[ids[0]].made["frames"]
    dlg.deleteLater()


# ---------------------------------------------------------------- UX-8: a frozen clip's buttons are disabled
def test_a_frozen_clips_card_buttons_are_disabled(app, win, tmp_path, image):
    from video_chain.ui.main_window import PromptCard
    from test_ui import run_and_wait
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    assert win.freeze_clip(ids[0], confirm=False)
    wait(app, lambda: not win._joins, timeout=30)
    card = next(c for c in win.cards if c.nid == ids[0])
    assert isinstance(card, PromptCard) and card.frozen
    for name in ("b_refine", "b_speak", "b_end", "b_end_ai", "b_end_video", "b_end_clear", "b_hand", "b_hand_clear",
                 "b_revert", "b_key"):
        assert not getattr(card, name).isEnabled(), name
    assert card.b_freeze.isEnabled()                    # (the way out stays)


# ---------------------------------------------------------------- UX-11: the graph can be driven from the keyboard
def test_the_graph_has_the_focus_when_the_window_shows(app, tmp_path):
    w = _window(tmp_path)
    try:
        w.show()
        wait(app, lambda: w.focusWidget() is w.graph)
    finally:
        w.close()


def test_opening_a_project_puts_the_focus_on_the_graph(app, win, tmp_path, image):
    from PySide6.QtWidgets import QLineEdit
    win.project_edit.setFocus()
    new_project(win, tmp_path, image, ("a",))
    wait(app, lambda: win.focusWidget() is win.graph)


# ---------------------------------------------------------------- UX-17 / UMD-30: one table of shortcuts
def test_help_shows_the_shortcut_table_and_f1_opens_it(app, win, monkeypatch):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import QDialog, QTextBrowser
    from video_chain.ui import shortcuts as S
    shown = []
    monkeypatch.setattr(QDialog, "exec", lambda self: shown.append(self) or 0)
    monkeypatch.setattr(QDialog, "open", lambda self: shown.append(self))
    monkeypatch.setattr(QDialog, "show", lambda self: shown.append(self))
    act = next(a for a in win.menuBar().findChildren(type(win.act_open)) if a.text().startswith("Keyboard shortcuts"))
    assert act.shortcut() == QKeySequence("F1")
    act.trigger()
    assert shown and shown[0].windowTitle() == "Keyboard shortcuts"
    html = shown[0].findChild(QTextBrowser).toHtml()
    assert "Ctrl+Return" in html and "Alt+L" in html


def test_every_menu_shortcut_comes_from_the_table(app, win):
    from video_chain.ui import shortcuts as S
    table = {S.sequence(s.id).toString() for s in S.SHORTCUTS if " / " not in s.keys}
    seen = []
    for top in win.menuBar().actions():
        menu = top.menu()
        for a in (menu.actions() if menu else []):
            if not a.shortcut().isEmpty():
                seen.append(a.shortcut().toString())
    assert seen and all(k in table for k in seen), [k for k in seen if k not in table]


def test_the_graph_keys_are_in_the_table():
    from video_chain.ui import shortcuts as S
    ids = {s.id for s in S.SHORTCUTS}
    assert {"graph_delete", "graph_fit"} <= ids


# ---------------------------------------------------------------- UX-9: more drop targets than the start tile
def _drop(widget, *paths):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    pos = QPoint(5, 5)
    enter = QDragEnterEvent(pos, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    drop = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, drop)
    return enter.isAccepted()


def test_a_project_folder_dropped_on_the_window_opens(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a",))
    folder = win.project
    win.close_project()
    assert _drop(win, folder)
    assert win.project == folder.resolve()


def test_a_dropped_non_project_says_so(app, win, tmp_path):
    (tmp_path / "plain").mkdir()
    win.open_project(tmp_path / "proj")
    assert _drop(win, tmp_path / "plain")
    assert win.project == (tmp_path / "proj").resolve() and "project" in win.status.text()


def test_pictures_dropped_on_the_references_are_added(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a",))
    added = []
    win.add_reference = lambda path, crop="ask": added.append(path) or True
    assert _drop(win.refs.viewport(), image)
    assert added == [str(image)]


def test_sounds_dropped_on_the_sounds_are_added(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a",))
    added = []
    win.add_sound = lambda path, name=None: added.append(path) or True
    wav = tmp_path / "s.wav"
    wav.write_bytes(b"x")
    assert _drop(win.sounds.viewport(), wav)
    assert added == [str(wav)]


def test_a_picture_dropped_on_the_end_of_a_clip_sets_it(app, win, tmp_path, image):
    ids = new_project(win, tmp_path, image, ("a",))
    card = next(c for c in win.cards if c.nid == ids[0])
    assert _drop(card.end_text, image)
    assert win.tree.nodes[ids[0]].end_image


# ---------------------------------------------------------------- MWC-33: a single clip is exported in the background
def test_exporting_a_single_clip_does_not_block_the_window(app, win, tmp_path, image, monkeypatch):
    from test_ui import run_and_wait
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    gate, started = threading.Event(), threading.Event()
    import shutil
    real = shutil.copyfile

    def slow(src, dst, *a, **k):
        started.set()
        assert gate.wait(20)
        return real(src, dst, *a, **k)
    monkeypatch.setattr(shutil, "copyfile", slow)
    out = tmp_path / "out" / "clip.mp4"
    win.export(out, "made", single=ids[0])              # returns at once
    assert started.wait(10) and not out.exists() and win._joins == 1
    assert "xporting" in win.status.text()
    gate.set()
    wait(app, lambda: "Exported clip.mp4" in win.status.text())
    assert out.is_file() and win._joins == 0
    assert not list(out.parent.glob("*.part*"))


def test_a_failed_single_export_says_why(app, win, tmp_path, image, monkeypatch):
    from test_ui import run_and_wait
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    win.export(blocker / "clip.mp4", "made", single=ids[0])      # (its folder is a file)
    wait(app, lambda: "Could not export" in win.status.text())
    assert win._joins == 0


def test_a_project_of_a_newer_version_is_refused_with_its_reason(app, win, tmp_path, monkeypatch):
    import json
    folder = tmp_path / "newer"
    folder.mkdir()
    (folder / T.TREE_FILE).write_text(json.dumps({"version": T.TREE_VERSION + 5, "nodes": []}))
    told = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text, *a: told.append(text))
    assert win.open_project(folder) is False
    assert told and "newer version" in told[0] and win.project is None


def test_the_comfy_poll_goes_through_net_without_a_proxy(app, win, monkeypatch):
    from video_chain import net
    seen = []

    class Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(net, "urlopen", lambda req, timeout: seen.append(str(req)) or Answer())
    win._comfy_polling = False
    win._poll_comfy()
    wait(app, lambda: win.comfy_up is True)
    assert seen and seen[0].endswith("/system_stats")
