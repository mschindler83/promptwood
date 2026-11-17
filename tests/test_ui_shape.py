"""GUI tests: the clips' Shape and the cut of a start picture of another shape (seen on the picture, set in a
window, checked before a start)."""
import json
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PySide6.QtWidgets import QDialog  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, flush, image, new_project, run_and_wait, wait, win  # noqa: E402,F401


def _shape(win, key):
    win.shape.setCurrentIndex(win.shape.findData(key))


def test_a_chosen_shape_wins_over_the_start_image(app, win, tmp_path, image):
    """A 16:9 start image and Shape 9:16: the clips are 9:16 (the line says why), the picture is cut, the tile shows
    the cut and Edit cut… is offered."""
    ids = new_project(win, tmp_path, image, ("one",))                       # start image 192x108 (16:9)
    assert "Clips: 672×384 (16:9), like the start image (192×108)" in win.size_info.text()
    assert win.b_cut.isHidden()
    plain = win.start_tile.pixmap().toImage()
    _shape(win, "9:16")
    flush(app, win)
    base = T.Checker(win.project, win.tree).base()
    assert (base["width"], base["height"]) == (384, 672) and win.tree.options["shape"] == "9:16"
    text = win.size_info.text()
    assert "Clips: 384×672 (9:16), the Shape chosen for this project" in text
    assert "The start image (192×108, 16:9) is cut to 9:16 at the centre" in text and not win.b_cut.isHidden()
    shaded = win.start_tile.pixmap().toImage()
    edge, mid = shaded.pixelColor(3, shaded.height() // 2), shaded.pixelColor(shaded.width() // 2,
                                                                                shaded.height() // 2)
    assert edge.lightness() < plain.pixelColor(3, plain.height() // 2).lightness() - 40   # cut away: shaded
    assert abs(mid.lightness() - plain.pixelColor(plain.width() // 2, plain.height() // 2).lightness()) < 10
    assert T.Checker(win.project, win.tree).settings(ids[0]).get("start_crop") is None    # (the centre: as before)


def test_cut_is_set_in_its_window_and_used(app, win, tmp_path, image):
    """Edit cut…: the frame has the clips' shape and stays inside the picture; the cut is stored, looked at (no check
    at the start), recorded with the clip and used for the picture H3 gets."""
    ids = new_project(win, tmp_path, image, ("one",))
    _shape(win, "9:16")
    flush(app, win)
    win.modal_dialogs = False
    dlg = win.edit_start_cut()
    x, y, w, h = dlg.view.box()
    assert (x, y, w, h) == T.crop_box(192, 108, 384, 672) and (w, h) == (62, 108)   # the largest cut, centred
    dlg.view.set_crop(-1.0, 0.5, 1.0)                                         # dragged past the left edge: kept in
    assert dlg.view.box()[0] == 0 and dlg.view.crop["cx"] > 0
    dlg.view.set_crop(0.0, 0.5, 0.5)
    assert dlg.view.box()[2:] == (31, 54) and "smaller than the clips" in dlg.size_note.text()
    dlg.b_largest.click()
    assert dlg.view.box() == (0, 0, 62, 108)
    dlg.accept()
    crop = T.get_crop(win.tree, win.project, win.project / T.START_FILE)
    assert crop["zoom"] == 1.0 and crop["cx"] < 0.2 and crop["seen"] == T.shape_key(384, 672)
    assert "where you set it" in win.size_info.text()
    saved = json.loads((win.project / "tree.json").read_text())["options"]["crops"]
    assert saved["start.png"]["cx"] == crop["cx"]
    run_and_wait(app, win, win.generate_path)
    assert win._cut_dialog is None and win.state(ids[0]) == "done"           # looked at: nothing asked
    made = win.tree.nodes[ids[0]].made["settings"]
    assert made["start_crop"] == [round(crop["cx"], 4), 0.5, 1.0]
    rec = json.loads((win.project / "start_384x672.json").read_text())
    assert rec["crop"] == made["start_crop"]


def test_a_cut_not_looked_at_is_shown_before_the_start(app, win, tmp_path, image, monkeypatch):
    """Generate with a start picture of another shape whose cut was not looked at: its window first; Don't start
    leaves everything as it was, Use this cut starts and is not asked again."""
    from video_chain.ui.start_crop import StartCropDialog
    ids = new_project(win, tmp_path, image, ("one",))
    _shape(win, "1:1")
    flush(app, win)
    shown = []
    monkeypatch.setattr(StartCropDialog, "exec", lambda self: shown.append(self) or QDialog.DialogCode.Rejected)
    win.generate_path()
    wait(app, lambda: not win.running)
    assert len(shown) == 1 and shown[0].b_cancel.text() == "Don't start" and "Not started" in win.status.text()
    assert win.state(ids[0]) == "new" and T.get_crop(win.tree, win.project, win.project / T.START_FILE) is None

    def keep(self):
        self.view.set_crop(0.7, 0.5, 1.0)
        shown.append(self)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(StartCropDialog, "exec", keep)
    run_and_wait(app, win, win.generate_path)
    assert len(shown) == 2 and win.state(ids[0]) == "done"
    assert win.tree.nodes[ids[0]].made["settings"]["start_crop"][0] > 0.5
    _shape(win, "1:1")                                                       # (unchanged) …
    win.tree.nodes[ids[0]].prompt = "two"
    run_and_wait(app, win, win.generate_path)
    assert len(shown) == 2                                                   # … looked at for 1:1: not again
    _shape(win, "4:5")                                                       # another shape: shown again
    run_and_wait(app, win, win.generate_path)
    assert len(shown) == 3


def test_a_clips_own_start_picture_has_its_cut(app, win, tmp_path, image):
    """A first clip with its own start picture: its menu offers the cut of that picture; the node view's start
    picture shows it; a new picture starts at the centre again."""
    from PIL import Image
    ids = new_project(win, tmp_path, image, ("one",))
    own = tmp_path / "tall.png"
    Image.new("RGB", (108, 192), (250, 250, 250)).save(own)
    assert win.set_row_start(ids[0], str(own))
    win.modal_dialogs = False
    dlg = win.edit_start_cut(ids[0])                                         # (the clip's menu: start_cut)
    assert dlg.view.image.size().toTuple() == (108, 192) and "clip #" in dlg.windowTitle()
    dlg.view.set_crop(0.5, 0.2, 1.0)
    dlg.accept()
    pic = win.project / win.tree.nodes[ids[0]].start_image
    assert T.get_crop(win.tree, win.project, pic)["cy"] < 0.5
    assert win.state(ids[0]) == "new"
    other = tmp_path / "tall2.png"
    Image.new("RGB", (108, 192), (5, 5, 5)).save(other)
    assert win.set_row_start(ids[0], str(other))
    assert T.get_crop(win.tree, win.project, win.project / win.tree.nodes[ids[0]].start_image) is None
    assert pic.name not in (win.tree.options.get("crops") or {})
