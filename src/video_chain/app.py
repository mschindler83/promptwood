"""GUI entry point: video-chain-gui"""
from __future__ import annotations

import shutil
import sys


def main(argv: list[str] | None = None) -> int:
    import os

    from . import paths
    from PySide6.QtCore import QLoggingCategory, QSettings
    # "Play videos on the GPU" (settings; read before Qt's media backend loads, so a change needs a restart): GPU =
    # hardware decoding and Qt's GPU video surface (smooth, but NVIDIA's EGL driver crashed once with a full GPU);
    # CPU = decoded and painted on the CPU (safe, lags with big clips). A decoder value set by the user is kept.
    gpu = QSettings("video-chain", "Video Chain").value("videoGpu", True, type=bool)
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
    app = QApplication(argv)
    app.setOrganizationName("video-chain")
    app.setApplicationName("Video Chain")
    app.setDesktopFileName("video-chain")
    ThemeManager.instance().apply(app)
    paths.use_private_tools()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        from .setup.system import install_hint
        QMessageBox.critical(None, "Promptwood",
                             "ffmpeg and ffprobe are required and were not found.\n\nRun  video-chain setup  in a "
                             f"terminal (it installs what is missing), or:  {install_hint('ffmpeg')}")
        return 1
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
