"""app.py: startup, the exception hooks (an error in a slot or a worker thread is shown, not lost) and the signal
handling (SIGTERM / SIGINT close the window, so its cleanup runs and no llama-server is left behind)."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import signal
import sys
import threading

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from video_chain import app as A


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    names = app.organizationName(), app.applicationName()
    yield app
    app.setOrganizationName(names[0])                   # (main() names the application; not for the next test)
    app.setApplicationName(names[1])


@pytest.fixture
def hooks(monkeypatch):
    """The hooks installed for one test only (sys / threading restored after)."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    shown = []
    A.install_exception_hooks(shown.append)
    return shown


def test_an_uncaught_error_is_logged_and_shown(hooks, capsys):
    try:
        raise ValueError("boom in a slot")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert "ValueError: boom in a slot" in capsys.readouterr().err
    assert len(hooks) == 1 and "boom in a slot" in hooks[0]


def test_an_error_in_a_worker_thread_is_logged_and_shown(hooks, capsys):
    def work():
        raise RuntimeError("worker died")
    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert "RuntimeError: worker died" in capsys.readouterr().err
    assert len(hooks) == 1 and "worker died" in hooks[0]


def test_a_failing_display_does_not_hide_the_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)

    def broken(_msg):
        raise OSError("no display")
    A.install_exception_hooks(broken)
    try:
        raise KeyError("x")
    except KeyError:
        sys.excepthook(*sys.exc_info())
    assert "KeyError" in capsys.readouterr().err


def test_the_error_message_is_shown_non_blocking(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(QMessageBox, "open", lambda self: opened.append(self.text()))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: pytest.fail("a blocking error window"))
    relay = A.ErrorRelay()
    relay.show("trace text")
    qapp.processEvents()
    assert opened and "trace text" in opened[0]


def test_sigterm_closes_the_window_and_sigint_too(qapp, monkeypatch):
    closed = []

    class Win:
        def close(self):
            closed.append(1)
            return True
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    seen = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: seen.__setitem__(sig, fn))
    keep = A.install_signal_handlers(qapp, lambda: Win())
    assert signal.SIGTERM in seen and signal.SIGINT in seen
    seen[signal.SIGTERM](signal.SIGTERM, None)
    assert closed == [1]
    keep.stop()


def test_a_second_signal_quits_even_when_the_window_stays(qapp, monkeypatch):
    quits = []

    class Win:
        def close(self):
            return False                      # (the user keeps it: unsaved project)
    seen = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: seen.__setitem__(sig, fn))
    monkeypatch.setattr(qapp, "quit", lambda: quits.append(1))
    keep = A.install_signal_handlers(qapp, lambda: Win())
    seen[signal.SIGTERM](signal.SIGTERM, None)
    assert quits == []
    seen[signal.SIGTERM](signal.SIGTERM, None)
    assert quits == [1]
    keep.stop()


def test_main_without_ffmpeg_says_so_and_quits(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    assert A.main(["video-chain"]) == 1
    assert shown and "ffmpeg" in shown[0]


def test_main_builds_the_window_and_runs_the_loop(qapp, monkeypatch):
    events = []
    import video_chain.ui.main_window as M

    class FakeWin:
        def __init__(self):
            events.append("made")

        def show(self):
            events.append("shown")
    monkeypatch.setattr(M, "MainWindow", FakeWin)
    monkeypatch.setattr(A.shutil, "which", lambda name: "/bin/true")
    monkeypatch.setattr(QApplication, "exec", lambda self: 7)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    assert A.main(["video-chain"]) == 7
    assert events == ["made", "shown"]
