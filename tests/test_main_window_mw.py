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
