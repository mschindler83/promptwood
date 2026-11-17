"""Clip dialogs (UIL-26, UIL-28, UIL-29). Made-up clips, offscreen."""
import os
import shutil
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def make_clip(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=0.5",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg")
def test_decoding_hands_out_its_process_and_closing_ends_it(tmp_path):
    from video_chain.ui.clip_dialogs import HandoverDialog, decode_frames
    clip = make_clip(tmp_path / "c.mp4")
    seen = []
    frames = decode_frames(clip, on_start=seen.append)
    assert len(frames) == 12 and seen and hasattr(seen[0], "terminate")
    killed = []

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            killed.append(1)
    dlg = HandoverDialog(None, clip, "t")
    dlg._proc = Proc()
    dlg.done(0)
    assert killed


class FakePlayer:
    def __init__(self):
        self.pos, self.source = None, "video"

    def setPosition(self, ms):          # noqa: N802
        self.pos = ms

    def stop(self):
        pass

    def setSource(self, url):           # noqa: N802
        self.source = url

    def play(self):
        pass


def test_compare_slider_seeks_on_clicks_and_keeps_players_when_keep_fails(tmp_path):
    from PySide6.QtWidgets import QAbstractSlider
    from video_chain.ui.clip_dialogs import CompareDialog

    def keep(nid):
        raise OSError("file in use")
    dlg = CompareDialog(None, [("a", "A", "", tmp_path / "a.mp4"), ("b", "B", "", tmp_path / "b.mp4")], keep,
                        player=False)
    dlg.players = [FakePlayer(), FakePlayer()]
    dlg.slider.setRange(0, 1000)
    dlg.slider.triggerAction(QAbstractSlider.SliderPageStepAdd)          # (a click in the groove)
    assert dlg.players[0].pos == dlg.slider.value() > 0
    with pytest.raises(OSError):
        dlg._keep("a")
    assert all(p.source.toLocalFile().endswith(".mp4") for p in dlg.players)         # restored, not left empty
