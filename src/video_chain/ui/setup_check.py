"""Help > Setup check: what the app needs on this computer, what is there and how to get the rest (the same report
as `promptwood doctor`)."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout


class _Checker(QThread):
    done = Signal(str, str)

    def run(self):
        try:
            from ..setup import doctor
            checks = doctor.check_all()
            self.done.emit(doctor.report(checks), doctor.summary(checks))
        except Exception as e:  # noqa: BLE001
            self.done.emit("", f"The check itself failed: {type(e).__name__}: {e}")


class SetupCheckDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup check")
        self.resize(920, 640)
        lay = QVBoxLayout(self)
        self.head = QLabel("Looking at this computer …")
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.text, 1)
        hint = QLabel("Missing parts are installed in a terminal with the command after each arrow "
                      "(<code>promptwood setup</code> installs everything that is missing; it is safe to run again).")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.again = QPushButton("Check again")
        self.copy = QPushButton("Copy report")
        box.addButton(self.again, QDialogButtonBox.ButtonRole.ActionRole)
        box.addButton(self.copy, QDialogButtonBox.ButtonRole.ActionRole)
        box.rejected.connect(self.reject)
        self.again.clicked.connect(self.check)
        self.copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.text.toPlainText()))
        lay.addWidget(box)
        self._thread: _Checker | None = None
        self.check()

    def check(self):
        if self._thread is not None and self._thread.isRunning():
            return
        self.again.setEnabled(False)
        self.head.setText("Looking at this computer …")
        self._thread = _Checker(self)
        self._thread.done.connect(self._show)
        self._thread.start()

    def _show(self, report: str, summary: str):
        self.text.setPlainText(report)
        self.head.setText(summary)
        self.again.setEnabled(True)

    def done(self, r: int):
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(15000)
        super().done(r)
