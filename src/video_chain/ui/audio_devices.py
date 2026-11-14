"""The players' sound goes to the default output device, also after it changes (headphones plugged in or out).
(QtMultimedia is imported inside the functions: the app starts without it.)"""
from __future__ import annotations

from typing import Any


def default_output() -> Any:
    from PySide6.QtMultimedia import QMediaDevices
    return QMediaDevices.defaultAudioOutput()


def follow_default_output(audio: Any) -> Any:
    """Let the QAudioOutput `audio` move to the new default device whenever the list of outputs changes. The returned
    watcher is a child of the output (it lives as long as it does)."""
    from PySide6.QtMultimedia import QMediaDevices
    devs = QMediaDevices(audio)
    devs.audioOutputsChanged.connect(lambda: audio.setDevice(default_output()))
    return devs
