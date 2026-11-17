"""GUI entry point: video-chain-gui"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import traceback
from typing import cast

from . import media

logger = logging.getLogger(__name__)


def _qt_error_relay():
    """The QObject that shows an error in the GUI thread (made when first needed: Qt is imported lazily here)."""
    from PySide6.QtCore import QObject, Qt, Signal, Slot
    from PySide6.QtWidgets import QMessageBox

    class ErrorRelay(QObject):
        message = Signal(str)

        def __init__(self):
            super().__init__()
            self.message.connect(self._open, Qt.ConnectionType.QueuedConnection)   # (any thread may call show)
            self._boxes = []

        def show(self, text: str) -> None:
            self.message.emit(text)

        @Slot(str)
        def _open(self, text: str) -> None:
            box = QMessageBox(QMessageBox.Icon.Warning, "Promptwood", "Something went wrong. The app keeps running; "
                              "the details are below and in the terminal log.")
            box.setDetailedText(text)
            box.setText(box.text() + "\n\n" + text.strip().splitlines()[-1][:300])
            box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._boxes.append(box)
            box.destroyed.connect(lambda _=None, b=box: self._boxes.remove(b) if b in self._boxes else None)
            box.open()                                               # (not modal: the app is not blocked)
    return ErrorRelay


def __getattr__(name):
    if name == "ErrorRelay":
        return _qt_error_relay()
    raise AttributeError(name)


def install_exception_hooks(show) -> None:
    """An error that nothing caught (a slot, the start-up, a worker thread) goes to stderr with its traceback and to
    show(text), which tells the user; the app is not left dying silently (a worker would just stop, a window never
    open). show may be called from any thread and must not raise (if it does, the traceback is still on stderr)."""
    def report(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr, flush=True)
        try:
            show(text)
        except Exception:  # noqa: BLE001  (the log above is what is left)
            traceback.print_exc()

    def main_hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        report(exc_type, exc, tb)

    def thread_hook(args) -> None:
        if args.exc_type is SystemExit:
            return
        report(args.exc_type, args.exc_value, args.exc_traceback)
    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def install_signal_handlers(app, window):
    """SIGTERM / SIGINT close the main window (window(): the window or None), so its cleanup runs (llama-server, the
    voice worker and the generator stop, the project is saved) as on a click on the close button; a second signal, or a
    window that stays (the user is asked about unsaved work), quits anyway. Returns the timer that lets Python see
    the signal while Qt's loop runs (keep it alive)."""
    from PySide6.QtCore import QTimer
    asked: list[int] = []

    def handle(_sig, _frame) -> None:
        w = window()
        if asked or w is None:
            app.quit()
            return
        asked.append(1)
        if w.close():
            app.quit()
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    tick = QTimer(app)
    tick.start(250)                       # (the loop gives Python's signal handlers a turn this often)
    tick.timeout.connect(lambda: None)
    return tick


def main(argv: list[str] | None = None) -> int:
    from . import logs, paths, win
    if win.WINDOWS:                                 # (Linux and macOS: nothing here runs)
        again = win.ensure_utf8("video_chain.app", argv[1:] if argv else None)
        if again is not None:
            return again
        win.hide_consoles()
    logs.setup()
    media.sweep_scratch()                 # (temp folders a killed run left behind, a day or older)

    from PySide6.QtCore import QLoggingCategory, QSettings
    # "Play videos on the GPU" (settings; read before Qt's media backend loads, so a change needs a restart): GPU =
    # hardware decoding and Qt's GPU video surface (smooth, but NVIDIA's EGL driver crashed once with a full GPU);
    # CPU = decoded and painted on the CPU (safe, lags with big clips). A decoder value set by the user is kept.
    gpu = bool(QSettings("video-chain", "Video Chain").value("videoGpu", True, type=bool))
    if not gpu:
        os.environ.setdefault("QT_FFMPEG_DECODING_HW_DEVICE_TYPES", "none")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .ui.main_window import MainWindow
    from .ui.widgets import set_gpu_video
    set_gpu_video(gpu)
    from .ui.theme import ThemeManager

    # harmless Qt messages on Wayland: text input warns when focus leaves a field of an inactive window, and the
    # desktop portal cannot register the app ID because the app has no installed .desktop file
    QLoggingCategory.setFilterRules("qt.qpa.wayland.textinput.warning=false\nqt.qpa.services.warning=false")
    argv = sys.argv if argv is None else argv
    app = cast(QApplication, QApplication.instance() or QApplication(argv))
    app.setOrganizationName("video-chain")
    app.setApplicationName("Video Chain")
    app.setDesktopFileName("video-chain")
    ThemeManager.instance().apply(app)
    paths.use_private_tools()
    try:
        media.require_ffmpeg()
    except media.MediaError:
        from .setup.system import install_hint
        QMessageBox.critical(None, "Promptwood",
                             "ffmpeg and ffprobe are required and were not found.\n\nRun  video-chain setup  in a "
                             f"terminal (it installs what is missing), or:  {install_hint('ffmpeg')}")
        return 1
    relay = _qt_error_relay()()
    install_exception_hooks(relay.show)
    w = MainWindow(QSettings())
    w.show()
    tick = install_signal_handlers(app, lambda: w)
    app.aboutToQuit.connect(lambda: _stop_helpers(w))     # (a quit that skipped the close button: logout, quit())
    code = app.exec()
    tick.stop()
    return code


def _stop_helpers(w) -> None:
    """What must not outlive the app: the language model server and the voice worker (they hold the GPU)."""
    for name in ("stop_llama", "stop_tts"):
        try:
            getattr(w, name)()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
