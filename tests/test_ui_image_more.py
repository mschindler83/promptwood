"""More fixes in the image windows: HiDPI mask canvas, crop view handling, gallery, result view."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_mask_canvas_is_sharp_on_a_hidpi_display(app):
    from video_chain.ui.mask_painter import MaskCanvas
    img = QImage(400, 300, QImage.Format_RGB32)
    img.fill(0x808080)
    c = MaskCanvas(img)
    c.resize(200, 150)
    c.devicePixelRatioF = lambda: 2.0
    c._ensure_display()
    assert c._shown[1].devicePixelRatio() == 2.0 and c._tint.devicePixelRatio() == 2.0
    assert c._tint.width() == c._shown[1].width()
    c._dab(QPointF(100, 100), QPointF(120, 100), False)         # an incremental stroke still lands in the tint
    k = c._tint.width() / 400
    assert c._tint.pixelColor(int(110 * k), int(100 * k)).alpha() > 0


def crop_view(w=400, h=300, crop=(100, 100, 100)):
    from video_chain.ui.face_crop import CropView
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(0x808080)
    v = CropView(img, [], crop)
    v.resize(w, h)                                               # 1 view pixel = 1 picture pixel
    return v


def mouse(kind, x, y, button=Qt.LeftButton):
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, button, Qt.NoModifier)


def test_crop_corner_resize_follows_the_diagonal(app):
    v = crop_view()
    b = v._to_widget(100, 100, 100, 100)
    cx, cy = b.right(), b.bottom()
    v.mousePressEvent(mouse(QEvent.MouseButtonPress, cx, cy))            # the corner
    v.mouseMoveEvent(mouse(QEvent.MouseMove, cx - 20, cy + 10))          # net inward: shrinks (max(dx, dy) grew)
    assert v.crop[2] < 100
    v.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, cx - 20, cy + 10))


def test_crop_keys_and_cursor(app):
    v = crop_view()
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    assert v.crop[0] == 101 or v.crop[0] > 100
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Plus, Qt.NoModifier))
    assert v.crop[2] > 100
    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Minus, Qt.NoModifier))
    c = v._to_widget(100, 100, 100, 100).center()
    v.mouseMoveEvent(mouse(QEvent.MouseMove, c.x(), c.y(), Qt.NoButton))
    assert v.cursor().shape() == Qt.SizeAllCursor                         # over the square: it can be moved
    v.mouseMoveEvent(mouse(QEvent.MouseMove, 10, 10, Qt.NoButton))
    assert v.cursor().shape() == Qt.ArrowCursor


def test_crop_wheel_is_accepted(app):
    v = crop_view()
    e = QWheelEvent(QPointF(150, 150), QPointF(150, 150), QPointF(0, 0).toPoint(), QPointF(0, 120).toPoint(),
                    Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    e.ignore()
    v.wheelEvent(e)
    assert e.isAccepted()
