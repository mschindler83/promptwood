"""A message box nobody answers must fail the test at once (offscreen it would block the run for ever)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])


def _close_soon():
    """Before the guard existed the box really opened: close it, so this test fails instead of hanging."""
    def close():
        w = QApplication.activeModalWidget()
        if w is not None:
            w.reject()
    QTimer.singleShot(1500, close)


@pytest.mark.parametrize("kind", ["question", "warning", "critical", "information"])
def test_an_unanswered_message_box_fails_the_test(kind):
    _close_soon()
    with pytest.raises(AssertionError, match="answer it in the test"):
        getattr(QMessageBox, kind)(None, "Title", "Text")


def test_a_test_can_still_answer_one(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    assert QMessageBox.question(None, "Title", "Text") == QMessageBox.Yes
