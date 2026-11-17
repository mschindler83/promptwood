"""The players' audio output follows the default output device when it changes (UMD-1). Offscreen, no sound."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtMultimedia")
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices
from PySide6.QtWidgets import QApplication


def test_the_output_follows_a_new_default_device(monkeypatch):
    from video_chain.ui import audio_devices
    app = QApplication.instance() or QApplication([])
    out = QAudioOutput()
    seen = []
    monkeypatch.setattr(audio_devices, "default_output", lambda: "new default")
    monkeypatch.setattr(out, "setDevice", seen.append, raising=False)
    devs = audio_devices.follow_default_output(out)
    devs.audioOutputsChanged.emit()
    assert seen == ["new default"]
    assert isinstance(devs, QMediaDevices) and app is not None


def test_every_player_follows_the_device():
    import re
    from pathlib import Path
    ui = Path(__file__).resolve().parent.parent / "src" / "video_chain" / "ui"
    for name in ("clip_dialogs", "audio_editor", "preview", "video_editor"):
        src = (ui / f"{name}.py").read_text()
        assert len(re.findall(r"QAudioOutput\(self\)", src)) == len(re.findall(r"follow_default_output\(", src)), name
