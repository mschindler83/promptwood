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
