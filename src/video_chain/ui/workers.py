"""Background work for the UI (XC-5): run a function in a thread and hand its result back on the GUI thread.

The result (or the error) travels through one queued signal. It is dropped when the owner (the window or dialog
the answer is for) was deleted meanwhile, and when a newer answer was asked for (Latest). An error nobody asked
to see is logged, never lost."""
from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
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


def _the_relay() -> _Relay:
    global _relay
    if _relay is None:
        _relay = _Relay()
        app = QApplication.instance()
        if app is not None:
            _relay.moveToThread(app.thread())
    return _relay


def _alive(owner: weakref.ref[QObject] | None) -> bool:
    """Owner still there: its Python object (held weakly, so a waiting answer does not keep a window alive) and its
    C++ object."""
    if owner is None:
        return True
    obj = owner()
    return obj is not None and isValid(obj)


def post(owner: QObject | None, fn: Callable[[], None]) -> bool:
    """From any thread: call fn on the GUI thread (not when the owner is gone). True when it was handed over."""
    return _post(weakref.ref(owner) if owner is not None else None, fn)


def _post(ref: weakref.ref[QObject] | None, fn: Callable[[], None]) -> bool:
    if not _alive(ref):
        return False

    def call() -> None:
        if _alive(ref):
            fn()
    try:
        _the_relay().call.emit(call)
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
        except Exception as e:  # noqa: BLE001 - every failure of a worker goes to the caller or the log
            def failed(e: Exception = e) -> None:
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
