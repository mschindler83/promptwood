"""ui/workers.py: one helper for background work (XC-5). Made-up work, offscreen."""
import logging
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QWidget

from waiting import wait

app = QApplication.instance() or QApplication([])


def test_the_result_arrives_on_the_gui_thread():
    from video_chain.ui.workers import run_in_background
    seen = []
    owner = QWidget()
    ran_in = []

    def fn():
        ran_in.append(threading.current_thread())
        return 41 + 1
    run_in_background(fn, on_done=lambda r: seen.append((r, threading.current_thread())), owner=owner)
    assert wait(app, lambda: seen)
    assert seen[0][0] == 42
    assert ran_in[0] is not threading.main_thread() and seen[0][1] is threading.main_thread()


def test_an_error_goes_to_on_error_on_the_gui_thread():
    from video_chain.ui.workers import run_in_background
    errors, owner = [], QWidget()

    def boom():
        raise ValueError("no good")
    run_in_background(boom, on_done=lambda r: errors.append("done"),
                      on_error=lambda e: errors.append((type(e), threading.current_thread())), owner=owner)
    assert wait(app, lambda: errors)
    assert errors == [(ValueError, threading.main_thread())]


def test_an_error_without_on_error_is_logged(caplog):
    from video_chain.ui.workers import run_in_background
    def boom():
        raise ValueError("kaputt")
    with caplog.at_level(logging.WARNING):
        owner = QWidget()
        run_in_background(boom, on_done=lambda r: None, owner=owner)
        assert wait(app, lambda: any("kaputt" in r.getMessage() or "kaputt" in str(r.exc_info) for r in caplog.records))


def test_nothing_is_delivered_to_a_deleted_owner():
    from video_chain.ui.workers import run_in_background
    gate, got, errs = threading.Event(), [], []
    owner = QObject()
    run_in_background(lambda: gate.wait(5), on_done=got.append, on_error=errs.append, owner=owner)
    shiboken6.delete(owner)                   # (its C++ object is gone, the Python name is not)
    gate.set()
    done, other = [], QWidget()
    run_in_background(lambda: 1, on_done=done.append, owner=other)
    assert wait(app, lambda: done)
    wait(app, lambda: False, timeout=0.3)
    assert got == [] and errs == []


def test_only_the_latest_answer_counts():
    from video_chain.ui.workers import Latest, run_in_background
    first_go, got = threading.Event(), []
    latest, owner = Latest(), QWidget()
    run_in_background(lambda: (first_go.wait(5), "old")[1], on_done=got.append, owner=owner, latest=latest)
    run_in_background(lambda: "new", on_done=got.append, owner=owner, latest=latest)
    assert wait(app, lambda: got == ["new"])
    first_go.set()
    wait(app, lambda: False, timeout=0.3)
    assert got == ["new"]


def test_post_runs_a_call_on_the_gui_thread_from_a_worker():
    from video_chain.ui.workers import post, run_in_background
    lines, owner = [], QWidget()

    def fn():
        for i in range(3):
            post(owner, lambda i=i: lines.append((i, threading.current_thread())))
        return "ok"
    out = []
    run_in_background(fn, on_done=out.append, owner=owner)
    assert wait(app, lambda: out and len(lines) == 3)
    assert [i for i, _ in lines] == [0, 1, 2] and all(t is threading.main_thread() for _, t in lines)
