"""One wait for Qt tests: run the event loop until the exact state the test asserts next is there."""
import time


def wait(app, cond, timeout: float = 10.0, *, raising: bool = False) -> bool:
    """Process events until cond() is true. False on a timeout, or AssertionError when raising."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    if cond():
        return True
    if raising:
        raise AssertionError("timed out")
    return False
