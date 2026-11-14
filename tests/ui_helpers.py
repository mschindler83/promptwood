"""Shared fixtures and helpers of the GUI tests (tests/test_ui_*.py, test_main_window_*.py): offscreen, own QSettings
file, a fake worker instead of ComfyUI."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from conftest import settings  # noqa: E402,F401  (the shared isolated settings file)

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_backend.py")]




@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from video_chain.ui.theme import ThemeManager
    ThemeManager.instance().apply(a)
    return a


@pytest.fixture(autouse=True)
def silent_player(monkeypatch):
    """No test plays anything: the test clips have a 440 Hz tone, and offscreen only hides the picture, not the
    sound (windows made without the win fixture played their joined paths aloud)."""
    from PySide6.QtMultimedia import QMediaPlayer
    monkeypatch.setattr(QMediaPlayer, "play", lambda self: None)     # Preview.open itself still runs


@pytest.fixture(autouse=True)
def crop_accepted(monkeypatch):
    """The face crop window is taken as it opens (its default square); no modal window waits in a test."""
    from video_chain.ui.face_crop import FaceCropDialog
    monkeypatch.setattr(FaceCropDialog, "exec", lambda self: FaceCropDialog.Accepted)


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "start.jpg"
    Image.new("RGB", (192, 108), (200, 200, 200)).save(p)
    return p


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from video_chain.ui.main_window import MainWindow, Preview
    played = []
    monkeypatch.setattr(Preview, "open", lambda self, path, title, autoplay=True: played.append(Path(path)))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w = MainWindow(settings(tmp_path), command=FAKE)
    w.played = played
    w.show()
    yield w
    if w.running:
        w.stop()
        w.proc and w.proc.waitForFinished(10000)
    w.close()
    w.deleteLater()                              # (no window stays alive after its test)


def wait(app, cond, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out")


def wait_for_joined_path(app, win, last: str, timeout=30):
    """Until the joined path that ends at clip `last` is played (one played before it, e.g. of a shorter path or the
    clip just made, does not count: under load it can come first)."""
    want = win.project / "paths" / f"{last}.mp4"
    wait(app, lambda: win.played and win.played[-1] == want, timeout)
    return want


def flush(app, win):
    """Run the debounced save/refresh now."""
    for t in (win._save_timer, win._refresh_timer):
        if t.isActive():
            t.stop()
            t.timeout.emit()
    app.processEvents()


def type_prompt(win, nid, text):
    card = next(c for c in win.cards if c.nid == nid)
    card.edit.setPlainText(text)


def new_project(win, tmp_path, image, prompts=("one", "two")):
    win.open_project(tmp_path / "proj")
    win.set_image(str(image))
    ids = [win.selected_id()]
    type_prompt(win, ids[0], prompts[0])
    for p in prompts[1:]:
        win.node_action("continue", win.selected_id())
        ids.append(win.selected_id())
        type_prompt(win, ids[-1], p)
    return ids


def run_and_wait(app, win, fn):
    wait(app, lambda: not win._joins, timeout=30)           # a path still being joined blocks Generate
    fn()
    assert win.running
    wait(app, lambda: not win.running)


def loads(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def write_lora(path, names):
    import numpy as np
    from safetensors.numpy import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({n: np.zeros((2, 2), np.float32) for n in names}, str(path))


H3_KEYS = ["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.0.mlp.fc1.lora_A.weight"]


# ----------------------------------------------------------------------------- real videos

def make_real_video(path, secs=2.0):
    """A synthetic 'real' video: red, then blue; 30 fps, 16:9, with sound."""
    import subprocess
    half = secs / 2
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=red:s=320x180:r=30:d={half}",
                    "-f", "lavfi", "-i", f"color=blue:s=320x180:r=30:d={half}", "-f", "lavfi", "-i",
                    f"sine=f=500:d={secs}", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                    "-map", "2:a", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def PromptCard_open(win):
    from video_chain.ui.main_window import PromptCard
    return PromptCard.settings_open


def QImage_size(path):
    from PySide6.QtGui import QImage
    img = QImage(str(path))
    return img.width() // 2 * 2, img.height() // 2 * 2


def tr_lines(win, nid):
    """The lines the clip's spoken track was made of."""
    from video_chain import tree as T
    return [d["line"] for d in T.track_of(win.tree, T.track_id(win.tree.nodes[nid]))["tts"]["lines"]]


def _chips(win, nid):
    card = next(c for c in win.cards if c.nid == nid)
    return card, card.chips.texts()


# ----------------------------------------------------------------------------- Simple / Advanced view, card menu

def _row_hidden(form, field):
    lab = form.labelForField(field)
    return field.isHidden() and (lab is None or lab.isHidden())


def _make_sound(path, secs=2.5):
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=400:d={secs}", str(path)],
                   check=True)
    return path


def chain_sig(path):
    from video_chain import chain
    return chain.file_signature(path)
