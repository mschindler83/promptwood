"""Logging is set up by the entry points; silent catches now leave a warning."""
import logging
import signal
import sys
import threading

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_gui_main_sets_up_logging(qapp, monkeypatch):
    from video_chain import app as A
    from video_chain import logs
    called = []
    monkeypatch.setattr(logs, "setup", lambda *a, **k: called.append((a, k)))
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    A.main(["video-chain"])
    assert called


def test_cli_main_sets_up_logging(monkeypatch):
    from video_chain import __main__ as M
    from video_chain import logs
    called = []
    monkeypatch.setattr(logs, "setup", lambda *a, **k: called.append((a, k)))
    monkeypatch.setattr(M, "_main", lambda a, emit: 0)
    monkeypatch.setattr(M, "parse_args", lambda argv=None: type("A", (), {"events": False})())
    assert M.main([]) == 0
    assert called and called[0][1].get("to_file") is False      # (the GUI keeps the CLI's stderr)


def test_setup_without_file(tmp_path):
    from video_chain import logs
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        logs.setup(to_file=False)
        assert len([h for h in root.handlers if getattr(h, "_video_chain", False)]) == 1
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
