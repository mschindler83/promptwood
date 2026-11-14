"""Background work for the UI (XC-5): run a function in a thread and hand its result back on the GUI thread.

The result (or the error) travels through one queued signal. It is dropped when the owner (the window or dialog
the answer is for) was deleted meanwhile, and when a newer answer was asked for (Latest). An error nobody asked
to see is logged, never lost."""
from __future__ import annotations

import gc
import logging
import threading
import weakref
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

log = logging.getLogger(__name__)


class Latest:
    """'Only the latest answer counts': every start takes a ticket, an answer with an older ticket is dropped."""

    def __init__(self) -> None:
        self._n = 0

    def ticket(self) -> int:
        self._n += 1
        return self._n

    def current(self, ticket: int) -> bool:
        return ticket == self._n


class _Relay(QObject):
    call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001 - a slot that fails must not stop the next answer
            log.exception("a background answer failed in its handler")


_relay: _Relay | None = None
_gc_timer: QTimer | None = None
_GC_MS = 1000                              # the cyclic garbage is collected this often (the GUI thread does it)
_gc_runs = 0


def _collect() -> None:
    """Collect cyclic garbage here, on the GUI thread (every 10th time all generations)."""
    global _gc_runs
    _gc_runs += 1
    gc.collect(2 if _gc_runs % 10 == 0 else 1)


def _collect_on_the_gui_thread(parent: _Relay) -> None:
    """Python's automatic collector runs in whichever thread allocates: with workers running, a window that is
    cyclic garbage (its signal connections make cycles) was destroyed in a worker thread and Qt crashed. So the automatic
    collector is off and a timer of the GUI thread collects instead."""
    global _gc_timer
    if _gc_timer is not None:
        return
    if QThread.currentThread() is not parent.thread():     # (first used from a worker: set up on the GUI thread,
        _relay_call(parent, lambda: _collect_on_the_gui_thread(parent))   # a timer made here would never run)
        return
    if _gc_timer is None:
        gc.disable()                                        # (only once the timer that replaces it exists)
        _gc_timer = QTimer(parent)
        _gc_timer.setInterval(_GC_MS)
        _gc_timer.timeout.connect(_collect)
        _gc_timer.start()


def _relay_call(relay: _Relay, fn: Callable[[], None]) -> None:
    """fn on the relay's (the GUI) thread, through its queued signal."""
    relay.call.emit(fn)


def _the_relay() -> _Relay:
    global _relay
    if _relay is None:
        _relay = _Relay()
        app = QApplication.instance()
        if app is not None:
            _relay.moveToThread(app.thread())
            _collect_on_the_gui_thread(_relay)
    return _relay


def _alive(owner: weakref.ref[QObject] | None) -> bool:
    """Owner still there: its Python object (held weakly, so a waiting answer does not keep a window alive) and its
    C++ object."""
    if owner is None:
        return True
    obj = owner()
    return obj is not None and isValid(obj)


def post(owner: QObject | None, fn: Callable[..., None], *args: Any) -> bool:
    """From any thread: call fn(*args) on the GUI thread (at once when already there; not when the owner is gone). True when
    it was handed over."""
    return _post(weakref.ref(owner) if owner is not None else None, lambda: fn(*args))


def _post(ref: weakref.ref[QObject] | None, fn: Callable[[], None]) -> bool:
    if not _alive(ref):
        return False

    def call() -> None:
        if _alive(ref):
            fn()
    try:
        relay = _the_relay()
        if QThread.currentThread() is relay.thread():       # (already on the GUI thread: like a direct connection)
            call()
            return True
        relay.call.emit(call)
    except RuntimeError:                      # (the application is shutting down)
        return False
    return True


def run_in_background(fn: Callable[[], Any], *, on_done: Callable[[Any], None],
                      on_error: Callable[[BaseException], None] | None = None,
                      owner: QObject | None = None, latest: Latest | None = None,
                      name: str | None = None) -> threading.Thread:
    """Run fn() in a daemon thread; on the GUI thread call on_done(result), or on_error(exception) when it raised
    (without on_error the exception is logged). Nothing is delivered when owner was deleted meanwhile or when
    `latest` was asked for a newer answer. Call it from the GUI thread."""
    _the_relay()                              # (made here: its thread is the GUI thread)
    ticket = latest.ticket() if latest is not None else 0
    ref = weakref.ref(owner) if owner is not None else None     # (a waiting answer must not keep the window alive)

    def stale() -> bool:
        return latest is not None and not latest.current(ticket)

    def work() -> None:
        try:
            result = fn()
        except BaseException as e:  # noqa: BLE001 - whatever ends a worker goes to the caller or the log
            if not _alive(ref):
                log.debug("background work failed after its owner was gone: %s", e, exc_info=e)
            def failed(e: BaseException = e) -> None:
                if stale():
                    return
                if on_error is None:
                    log.warning("background work failed: %s: %s", type(e).__name__, e, exc_info=e)
                else:
                    on_error(e)
            _post(ref, failed)
            return

        def done() -> None:
            if not stale():
                on_done(result)
        _post(ref, done)
    t = threading.Thread(target=work, daemon=True, name=name)
    t.start()
    return t
