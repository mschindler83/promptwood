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


# ---- a swallowed error leaves a warning in the log
def test_a_failing_note_is_logged(tmp_path, caplog):
    import test_lora_notes as TL
    from video_chain import lora_notes as LN
    db, p = TL.db_with(tmp_path)

    def broken(system, user, n):
        raise RuntimeError("the model is gone")
    with caplog.at_level(logging.WARNING):
        assert LN.make_notes([p], broken, db=db) == 0
    assert any("the model is gone" in r.getMessage() for r in caplog.records)


def test_a_failing_remote_remove_is_logged(tmp_path, caplog):
    from video_chain import runpod as RP
    s = RP.RunpodSession(None, config=tmp_path / "cfg")

    def boom(*a, **k):
        raise OSError("ssh gone")
    s.run = boom
    with caplog.at_level(logging.WARNING):
        s.remove_remote("input", RP.SUBFOLDER, "a.png")
    assert any("ssh gone" in r.getMessage() for r in caplog.records)


def test_a_failing_server_stop_at_exit_is_logged(caplog):
    from video_chain import llm_server as L

    class Bad:
        def _take(self):
            raise RuntimeError("take failed")

        def _kill(self, *a, **k):
            pass
    bad = Bad()
    old = L._LIVE
    L._LIVE = [bad]
    try:
        with caplog.at_level(logging.WARNING):
            L._stop_all()
    finally:
        L._LIVE = old
    assert any("take failed" in r.getMessage() for r in caplog.records)


def test_a_failing_progress_callback_is_logged(tmp_path, caplog):
    from video_chain import triggers as TR
    f = tmp_path / "a.safetensors"
    f.write_bytes(b"x" * 100)
    db = TR.TriggerDB(tmp_path / "db.json")

    def boom(*a):
        raise RuntimeError("window gone")
    with caplog.at_level(logging.WARNING):
        db.scan([str(f)], online=True, progress=boom, lookup=lambda sha: None, gap=0)
    assert any("window gone" in r.getMessage() for r in caplog.records)


def test_an_odd_file_header_is_logged_and_does_not_stop_a_scan(monkeypatch, caplog):
    from video_chain import triggers as TR

    def boom(path):
        raise ValueError("odd header")
    monkeypatch.setattr(TR, "_file_words", boom)
    with caplog.at_level(logging.WARNING):
        assert TR.file_words("/x/a.safetensors") == (None, "")
    assert any("odd header" in r.getMessage() for r in caplog.records)


def test_unreadable_trigger_words_are_logged(monkeypatch, caplog):
    from video_chain import tree as T
    from video_chain import triggers as TR

    def boom(paths):
        raise OSError("database locked")
    monkeypatch.setattr(TR, "stack_words", boom)
    with caplog.at_level(logging.WARNING):
        assert T.lora_trigger_words(["/x/a.safetensors"]) is None
    assert any("database locked" in r.getMessage() for r in caplog.records)
