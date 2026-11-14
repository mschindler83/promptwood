"""The face crop and swap dialogs find faces and run ffmpeg in a worker, not in the constructor (UI thread)."""
import os
import threading
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    assert cond()


@pytest.fixture
def stubbed(monkeypatch):
    """A detector that finds one face, recording which thread ran it; ffmpeg's to_png likewise (a plain copy)."""
    from video_chain import chain
    from video_chain import faces as F
    log = {"detect": [], "png": [], "gate": threading.Event()}
    log["gate"].set()

    def detect(png, model):
        log["gate"].wait(10)
        log["detect"].append(threading.current_thread())
        return [(100, 40, 160, 110, 0.9)]

    def to_png(src, dest, crop=None):
        log["png"].append(threading.current_thread())
        Image.open(src).save(dest)
    monkeypatch.setattr(F, "available", lambda model: None)
    monkeypatch.setattr(F, "detect", detect)
    monkeypatch.setattr(chain, "to_png", to_png)
    return log


def test_crop_dialog_detects_in_a_worker(app, tmp_path, stubbed):
    from video_chain import faces as F
    from video_chain.ui.face_crop import FaceCropDialog
    p = tmp_path / "pic.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(p)
    stubbed["gate"].clear()
    dlg = FaceCropDialog(None, p, tmp_path / "det.onnx")
    assert dlg.loading and not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()       # busy, nothing found yet
    assert "Looking for the face" in dlg.info.text()
    stubbed["gate"].set()
    wait(app, lambda: not dlg.loading)
    assert all(t is not threading.main_thread() for t in stubbed["detect"] + stubbed["png"])
    assert dlg.crop() == F.face_square((100, 40, 160, 110, 0.9), 300, 200)
    assert dlg.b_face.isEnabled() and dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dlg.deleteLater()


def test_crop_dialog_closed_while_looking(app, tmp_path, stubbed):
    from video_chain.ui.face_crop import FaceCropDialog
    p = tmp_path / "pic.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(p)
    stubbed["gate"].clear()
    dlg = FaceCropDialog(None, p, tmp_path / "det.onnx")
    dlg.reject()                                                    # closed before the detector answers
    stubbed["gate"].set()
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    assert dlg.faces == [] and not dlg._tmp._finalizer.alive       # nothing was applied, the folder is gone
    dlg.deleteLater()


def test_crop_dialog_not_an_image(app, tmp_path, monkeypatch):
    from video_chain.ui.face_crop import FaceCropDialog
    p = tmp_path / "x.png"
    p.write_bytes(b"nope")
    dlg = FaceCropDialog(None, p, None)
    wait(app, lambda: not dlg.loading)
    assert "not an image" in dlg.info.text() and not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    dlg.deleteLater()


def _swap_dialog(tmp_path, app):
    import subprocess

    from video_chain.ui.swap_dialog import SwapDialog
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=24", "-frames:v", "6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    dlg = SwapDialog(None, tmp_path, "v1", clip, None, [], lambda p: None)
    return dlg


def test_swap_reference_is_prepared_in_a_worker(app, tmp_path, stubbed):
    from video_chain.ui import swap_dialog as SD
    dlg = _swap_dialog(tmp_path, app)
    pic = tmp_path / "ref.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(pic)
    stubbed["gate"].clear()
    dlg.detector = tmp_path / "det.onnx"
    dlg.set_reference(str(pic))
    assert dlg.ref_busy and dlg.ref is None and not dlg.b_gen.isEnabled()
    assert "Looking" in dlg.crop_info.text() or dlg.ref_tile is not None
    stubbed["gate"].set()
    wait(app, lambda: not dlg.ref_busy)
    assert dlg.ref == str(pic) and dlg.ref_crop is not None
    assert all(t is not threading.main_thread() for t in stubbed["detect"] + stubbed["png"])
    assert SD is not None
    dlg.done(0)


def test_swap_closed_while_reference_is_prepared(app, tmp_path, stubbed):
    dlg = _swap_dialog(tmp_path, app)
    pic = tmp_path / "ref.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(pic)
    stubbed["gate"].clear()
    dlg.detector = tmp_path / "det.onnx"
    dlg.set_reference(str(pic))
    dlg.done(0)
    stubbed["gate"].set()
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    assert dlg.ref is None                                          # the late answer was dropped


def test_head_size_repaints_the_player(app, tmp_path):
    dlg = _swap_dialog(tmp_path, app)
    dlg.heads = {"size": (160, 90), "track": [(60, 20, 100, 70)] * 6, "boxes": [], "frames": 6, "found": 6, "fps": 24}
    seen = []
    dlg.player.refresh = lambda: seen.append(1)
    dlg.head_size.setValue(1.4)
    assert seen
    dlg.done(0)


def test_head_size_without_a_track_does_not_fail(app, tmp_path):
    dlg = _swap_dialog(tmp_path, app)
    dlg.heads = {"size": (160, 90), "track": [], "boxes": [], "frames": 6, "found": 0, "fps": 24}
    dlg.head_size.setValue(1.4)                                    # (used to raise "no face")
    dlg.done(0)


def test_swap_start_hook_failure_is_reported(app, tmp_path):
    dlg = _swap_dialog(tmp_path, app)
    dlg.heads = {"size": (160, 90), "track": [], "boxes": [], "frames": 6, "found": 0, "fps": 24}
    dlg.ref = str(tmp_path / "r.png")
    dlg.method.setCurrentIndex(dlg.method.findData("face"))
    dlg._update()

    def boom():
        raise RuntimeError("no way")
    dlg.on_start = boom
    dlg.generate()
    assert not dlg._busy and "no way" in dlg.state.text()
    dlg.done(0)


def test_swap_finished_with_a_thin_record(app, tmp_path):
    dlg = _swap_dialog(tmp_path, app)
    loaded = []
    dlg.player.load_result = loaded.append
    dlg._finished((tmp_path / "o.mp4", {}, 6), "")                 # a record without gen_size / windows
    assert loaded and "Done" in dlg.state.text()
    dlg.done(0)


def test_swap_close_with_a_result_asks(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    dlg = _swap_dialog(tmp_path, app)
    dlg.result = (tmp_path / "o.mp4", {}, 6)
    finished = []
    dlg.finished.connect(finished.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    dlg.reject()
    assert finished == []                                           # kept open
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg.reject()
    assert finished == [0]


def test_crop_view_picture_follows_pixel_ratio(app):
    from PySide6.QtGui import QImage

    from video_chain.ui.face_crop import CropView
    img = QImage(400, 300, QImage.Format_RGB32)
    img.fill(0x808080)
    v = CropView(img, [], (0, 0, 100))
    v.resize(200, 150)
    v.devicePixelRatioF = lambda: 2.0
    v._ensure_shown()
    assert v._shown.devicePixelRatio() == 2.0 and v._shown.width() == round(400 * v._frame()[0] * 2)


def test_crop_says_when_the_picture_could_not_be_turned(app, tmp_path, monkeypatch):
    from video_chain import chain
    from video_chain.ui.face_crop import FaceCropDialog
    p = tmp_path / "pic.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(p)

    def bad(*a, **k):
        raise RuntimeError("ffmpeg missing")
    monkeypatch.setattr(chain, "to_png", bad)
    dlg = FaceCropDialog(None, p, None)
    wait(app, lambda: not dlg.loading)
    assert "upright" in dlg.info.text()
    dlg.deleteLater()


def test_h3_trim_gets_the_clips_frame_rate(app, tmp_path, monkeypatch):
    """GEN-12: the made video is cut at the clip's own frame rate (a 30 fps clip is not retimed to 24)."""
    from video_chain import swap
    dlg = _swap_dialog(tmp_path, app)
    pic = tmp_path / "ref.png"
    Image.new("RGB", (64, 64), (90, 80, 70)).save(pic)
    dlg.heads = {"size": (160, 90), "track": [], "boxes": [], "frames": 6, "found": 0, "fps": 30.0}
    dlg.ref, dlg.ref_png, dlg.ref_crop = str(pic), pic, None

    class Gen:
        def generate(self, prompt, start, out, png, seed, settings=None):
            out.write_bytes(b"x")
    dlg.h3 = {"lora": "l.safetensors", "turbo": "t.safetensors", "settings": {}, "connect": lambda p: Gen(),
              "describe": None}
    dlg.method.setCurrentIndex(dlg.method.findData("h3"))
    dlg.prompt.setText("a face")
    seen = {}

    def trim(made, clip, out, frames, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")
    monkeypatch.setattr(swap, "trim_with_sound", trim)
    dlg.generate()
    wait(app, lambda: not dlg._busy, timeout=20)
    assert seen.get("fps") == 30.0
    dlg.done(0)
