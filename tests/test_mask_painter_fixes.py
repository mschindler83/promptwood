"""Mask painter: cheap strokes (no full re-tint), bounded undo with redo, asks before discarding work."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def canvas(w=400, h=300):
    from video_chain.ui.mask_painter import MaskCanvas
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(0x808080)
    c = MaskCanvas(img)
    c.resize(200, 150)
    return c


def test_a_stroke_does_not_retint_the_whole_mask(app, monkeypatch):
    from video_chain.ui import mask_painter as MP
    c = canvas()
    c._ensure_display()                                   # the first picture of the tint
    calls = []
    real = MP.tinted
    monkeypatch.setattr(MP, "tinted", lambda *a: (calls.append(1), real(*a))[1])
    for k in range(20):                                   # twenty mouse moves
        c._dab(QPointF(10 + k, 10), QPointF(11 + k, 12), False)
        c._ensure_display()
    assert calls == []
    assert c._tint.pixelColor(c._tint.width() // 2, 3).alpha() > 0 or c._tint.pixelColor(10, 6).alpha() > 0
    c._dab(QPointF(10, 10), QPointF(30, 12), True)        # erasing clears the tint too
    c._ensure_display()
    assert calls == []


def test_undo_is_bounded_and_redo_works(app):
    from video_chain.ui.mask_painter import mask_area
    c = canvas()
    for _ in range(45):
        c.invert()                                        # used to grow without a cap
    assert len(c._undo) <= 30
    c = canvas()
    c.set_brush(40)
    c._begin_stroke(QPointF(100, 100), False)
    c._stroke_to(QPointF(200, 100))
    c._finish_stroke()
    painted = mask_area(c.mask)
    assert painted > 0 and len(c._undo) == 1
    rect, img = c._undo[0]
    assert img.width() < c.mask.width()                   # only the stroke's own area is kept, not the image
    c.undo()
    assert mask_area(c.mask) == 0
    c.redo()
    assert abs(mask_area(c.mask) - painted) < 1e-9


def test_closing_a_changed_mask_asks(app, monkeypatch):
    from video_chain.ui.mask_painter import MaskPainterDialog
    img = QImage(80, 60, QImage.Format_RGB32)
    img.fill(0)
    asked, finished = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(1) or QMessageBox.No)
    d = MaskPainterDialog(None, img)
    d.finished.connect(finished.append)
    d.reject()                                            # nothing painted: closes without asking
    assert finished == [0] and not asked
    d = MaskPainterDialog(None, img)
    d.finished.connect(finished.append)
    d.canvas.invert()
    d.reject()
    assert asked == [1] and finished == [0]               # "No": still open
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    d.reject()
    assert finished == [0, 0]
