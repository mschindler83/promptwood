"""GUI entry point: video-chain-gui"""
from __future__ import annotations

import shutil
import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import QLoggingCategory
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .ui.main_window import MainWindow
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
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        QMessageBox.critical(None, "Promptwood", "ffmpeg and ffprobe are required (sudo dnf install ffmpeg).")
        return 1
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
