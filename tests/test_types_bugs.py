"""Real bugs the type checker (mypy) found in the non-UI code."""
from __future__ import annotations

from video_chain import tts


def test_user_systemd_does_not_crash():
    """_user_systemd used shutil without importing it: every FishTTS start with a RAM cap raised NameError."""
    assert isinstance(tts._user_systemd(), bool)
