"""The voice window holds text to speech busy through the shared path (MWB-31). No window, no worker."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from video_chain.ui.mw_tts import TtsMixin


class Host(TtsMixin):
    def __init__(self):
        self._tts_busy = False
        self.seen = []

    def _set_tts_busy(self, on):                # the shared path: the buttons follow it
        self.seen.append(on)
        self._tts_busy = on


def test_a_hold_is_busy_inside_and_free_after_even_on_an_error():
    h = Host()
    with h._tts_hold() as mine:
        assert mine and h._tts_busy
    assert not h._tts_busy and h.seen == [True, False]
    with pytest.raises(ValueError), h._tts_hold():
        raise ValueError
    assert not h._tts_busy


def test_a_hold_while_busy_is_not_mine_and_leaves_it_busy():
    h = Host()
    h._tts_busy = True
    with h._tts_hold() as mine:
        assert not mine
    assert h._tts_busy and h.seen == []


def test_the_voice_window_uses_the_hold():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "src/video_chain/ui/mw_audio.py").read_text()
    assert "_tts_hold()" in src and "_set_tts_busy(" not in src.split("def _choose_voice_part")[1].split("def _voice_dialog")[0]
