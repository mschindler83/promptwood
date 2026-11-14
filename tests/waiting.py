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


def show_lora(app, window, path: str) -> None:
    """LoraWindow.show_lora reads in the background: wait until the page of this LoRA is filled in."""
    window.show_lora(path)
    wait(app, lambda: window._loaded == path, raising=True)


def wait_loaded(app, window) -> None:
    """A LoraWindow shows its first LoRA from a background read: wait until it is filled in."""
    if window.path:
        wait(app, lambda: window._loaded == window.path, raising=True)
