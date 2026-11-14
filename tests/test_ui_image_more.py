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


def groups(names):
    return [{"title": "g", "prompt": "p", "items": [(n, {"prompt": f"prompt {n}"}, 0.0) for n in names],
             "oldest": 0, "model": "m"}]


def test_gallery_keeps_its_thumb_widgets_on_a_rebuild(app):
    from video_chain.ui.results_gallery import ResultGallery
    g = ResultGallery()
    g.set_groups(groups(["a.png", "b.png"]), set())
    a, b = g.thumbs["a.png"], g.thumbs["b.png"]
    g.set_groups(groups(["c.png", "a.png", "b.png"]), {"a.png"})        # a new result arrives
    assert g.thumbs["a.png"] is a and g.thumbs["b.png"] is b and "c.png" in g.thumbs
    assert a.in_use and not b.in_use
    g.set_groups(groups(["b.png"]), set())                              # a filter hides one
    assert list(g.thumbs) == ["b.png"] and g.thumbs["b.png"] is b


def test_gallery_keys_home_end_page(app):
    from video_chain.ui.results_gallery import ResultGallery
    g = ResultGallery()
    names = [f"{i}.png" for i in range(6)]
    g.set_groups(groups(names), set())
    got = []
    g.clicked.connect(got.append)
    g._selected = "2.png"
    g.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Home, Qt.NoModifier))
    g.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_End, Qt.NoModifier))
    assert got == ["0.png", "5.png"]
    assert g.thumbs["0.png"].accessibleName() == "prompt 0.png"


def test_result_view_is_quick_while_resizing_then_smooth(app):
    from video_chain.ui.results_gallery import ResultView
    v = ResultView()
    img = QImage(800, 600, QImage.Format_RGB32)
    img.fill(0x336699)
    v.resize(400, 300)
    v.set_image(img)
    v._pixmap()
    assert v._scaled[2] is True                                          # at rest: smooth
    v.resize(380, 285)
    v.resizeEvent(None)
    v._pixmap()
    assert v._scaled[2] is False                                         # while the edge is dragged: fast
    v._settled()
    v._pixmap()
    assert v._scaled[2] is True


def test_info_card_colours_come_from_the_theme_when_painted(app):
    from PySide6.QtGui import QColor

    from video_chain.ui.results_gallery import InfoCard
    from video_chain.ui.theme import theme
    c = InfoCard()
    assert c.styleSheet() == ""
    t = theme()
    old = t.bgRaised
    try:
        t.bgRaised = QColor(1, 2, 3)
        assert c.colours()[0] == QColor(1, 2, 3)
    finally:
        t.bgRaised = old


def test_picker_preview_background_comes_from_the_theme(app, tmp_path):
    from PySide6.QtGui import QColor
    from PySide6.QtCore import QSettings

    from video_chain.ui import image_picker as IP
    from video_chain.ui.theme import theme
    q = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None)
    assert dlg.preview.styleSheet() == ""
    t = theme()
    old = t.bgSunken
    try:
        t.bgSunken = QColor(4, 5, 6)
        assert dlg.preview.background() == QColor(4, 5, 6)
    finally:
        t.bgSunken = old
    dlg.deleteLater()
