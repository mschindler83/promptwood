"""GUI entry point: video-chain-gui"""
from __future__ import annotations

import shutil
import sys


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from .ui.main_window import LAUNCHER, MainWindow
    from .ui.theme import ThemeManager

    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setOrganizationName("video-chain")
    app.setApplicationName("Video Chain")
    app.setDesktopFileName("video-chain")
    ThemeManager.instance().apply(app)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        QMessageBox.critical(None, "Promptwood", "ffmpeg and ffprobe are required (sudo dnf install ffmpeg).")
        return 1
    if not LAUNCHER.exists():
        QMessageBox.critical(None, "Promptwood", f"The launcher {LAUNCHER} is missing.")
        return 1
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
