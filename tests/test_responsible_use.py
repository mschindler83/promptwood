"""The app says where its use needs the consent of the people in it (faces, heads, voices) and under which licence
it is (Help → About). Offscreen, no models."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
import subprocess  # noqa: E402

from PySide6.QtWidgets import QLabel, QMessageBox  # noqa: E402

from conftest import settings  # noqa: E402,F401
from ui_helpers import app, image, new_project, win  # noqa: E402,F401
from video_chain.ui import responsible as R  # noqa: E402


def _texts(widget) -> str:
    return " ".join(lab.text() for lab in widget.findChildren(QLabel))


def test_the_texts_say_consent_and_the_licence():
    assert "consent" in R.FACES.lower() and "consent" in R.VOICES.lower()
    assert "PolyForm Noncommercial" in R.ABOUT and "commercial" in R.ABOUT.lower() and "consent" in R.ABOUT.lower()


def test_about_names_the_licence_and_responsible_use(app, win, monkeypatch):  # noqa: F811
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, text: shown.append(text))
    win.show_about()
    assert "PolyForm Noncommercial" in shown[0] and "consent" in shown[0].lower()


def test_the_swap_window_asks_for_consent(app, tmp_path):  # noqa: F811
    from video_chain.ui.swap_dialog import SwapDialog
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x96:r=24:d=1",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    dlg = SwapDialog(None, tmp_path, "n1", clip, detector=None, pictures=[], connect=lambda p: None)
    assert R.FACES in _texts(dlg)
    dlg.reject()


def test_the_voice_window_asks_for_consent(app, tmp_path):  # noqa: F811
    from video_chain.ui.audio_editor import AudioRangeDialog
    wav = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3", str(wav)], check=True)
    voice = AudioRangeDialog(None, file=str(wav), voice=True)
    assert R.VOICES in _texts(voice)
    voice.reject()
    track = AudioRangeDialog(None, file=str(wav))                 # an audio track: no voice, no note
    assert R.VOICES not in _texts(track)
    track.reject()
