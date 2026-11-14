"""Getting started is offered once: in the Help menu, not also in View (UX-18)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from ui_helpers import FAKE, app, settings  # noqa: E402,F401


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from video_chain.ui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    w = MainWindow(settings(tmp_path), command=FAKE)
    yield w
    w.close()


def _menu(win, name):
    return next(a.menu() for a in win.menuBar().actions() if a.text() == name)


def test_getting_started_is_only_in_the_help_menu(win):
    names = {n: [a.text() for a in _menu(win, n).actions()] for n in ("&View", "&Help")}
    assert "Getting started" in names["&Help"] and "Getting started" not in names["&View"]


def test_the_texts_name_the_help_menu(win):
    win.show_guide(False)
    assert "Help → Getting started" in win.status.text()
