"""UMD-12: opening a file in the range dialogs probes it off the UI thread (busy state, guarded against closing)."""
import os
import shutil
import subprocess
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from video_chain import chain
from video_chain.ui import audio_editor as AE
from video_chain.ui import video_editor as VE

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg")
app = QApplication.instance() or QApplication([])


def wait(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_video(path, secs=2.0):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=30:duration={secs}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_video_dialog_probes_off_the_ui_thread(tmp_path, monkeypatch):
    f = make_video(tmp_path / "v.mp4")
    seen = []
    real = chain.probe_source
    monkeypatch.setattr(chain, "probe_source", lambda p: (seen.append(threading.current_thread()), real(p))[1])
    d = VE.VideoRangeDialog(None, player=False)
    d.load_async(str(f))
    assert d.loading and not d.buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert wait(lambda: d.info is not None)
    assert seen and all(t is not threading.main_thread() for t in seen)
    assert not d.loading and d.file == str(f) and d.end > 1.5


def test_video_dialog_closed_while_probing_ignores_the_answer(tmp_path):
    f = make_video(tmp_path / "v.mp4")
    d = VE.VideoRangeDialog(None, player=False)
    d.load_async(str(f))
    d.reject()
    time.sleep(0.5)
    app.processEvents()
    assert d.info is None


def test_video_dialog_unreadable_file_says_so(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    d = VE.VideoRangeDialog(None, player=False)
    d.load_async(str(bad))
    assert wait(lambda: not d.loading)
    assert "Cannot read bad.mp4" in d.info_label.text() and d.info is None


def test_audio_dialog_probes_off_the_ui_thread(tmp_path, monkeypatch):
    f = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(f)],
                   check=True)
    seen = []
    real = chain.audio_seconds
    monkeypatch.setattr(chain, "audio_seconds", lambda p: (seen.append(threading.current_thread()), real(p))[1])
    d = AE.AudioRangeDialog(None, player=False)
    d.load_async(str(f))
    assert d.loading
    assert wait(lambda: d.file == str(f))
    assert seen and all(t is not threading.main_thread() for t in seen)
    assert not d.loading


def test_the_frame_for_a_face_picture_is_grabbed_off_the_ui_thread(tmp_path, monkeypatch):
    f = make_video(tmp_path / "v.mp4")
    got, threads = [], []
    d = VE.VideoRangeDialog(None, str(f), player=False, on_reference=lambda png: got.append(png) or True)
    real = VE.media.ffmpeg
    monkeypatch.setattr(VE.media, "ffmpeg", lambda *a, **k: (threads.append(threading.current_thread()), real(*a, **k))[1])
    d.reference_here_async()
    assert d.loading
    assert wait(lambda: bool(got))
    assert threads and all(t is not threading.main_thread() for t in threads)
    assert not d.loading
