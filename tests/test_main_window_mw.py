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
    from video_chain.ui import main_window as M
    got = []
    monkeypatch.setattr(M, "pick_image", lambda parent, title, qs, *a, **k: got.append(qs) or None)
    d = M.NewProjectDialog(None, str(tmp_path), win.qs)
    d._choose_image()
    assert got == [win.qs]
    d.deleteLater()
