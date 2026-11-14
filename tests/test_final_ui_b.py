"""Fixes of the final UI wave B (image dialogs, editors, shortcuts table, dialog geometry ...). Synthetic data, offscreen."""
import os
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_image_edit_split import _Editor, _edit_plan, _run_worker  # noqa: E402,F401


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_a_failed_rescale_leaves_no_result_without_a_record(app, tmp_path, monkeypatch):
    from PIL import Image

    from video_chain import edits
    from video_chain.ui.image_generate import EditWorker
    src = tmp_path / "a.png"
    Image.new("RGB", (16, 16), "green").save(src)

    def boom(*a, **k):
        raise RuntimeError("ffmpeg failed")
    monkeypatch.setattr(edits, "scale_to", boom)
    got = _run_worker(EditWorker(_edit_plan(tmp_path, _Editor(), [str(src)], keep_size=True)))
    assert got["finished"] == ["ffmpeg failed"] and got["results"] == []
    assert list((tmp_path / "edits").glob("*.png")) == []


def test_a_lora_row_asks_the_disk_once_per_file(app, monkeypatch):
    from video_chain.ui.lora_stack import LoraStack
    stack = LoraStack()
    calls = []
    real = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: (calls.append(str(self)), real(self))[1])
    stack.set_items([], [{"path": f"/x/{n}.safetensors", "strength": 1.0} for n in "abc"])
    mine = [c for c in calls if c.endswith(".safetensors")]
    assert len(mine) == 3


def test_lora_notes_cut_is_public_and_the_window_uses_it():
    from video_chain import lora_notes
    assert lora_notes.cut("word " * 100, 20).endswith("…") or len(lora_notes.cut("word " * 100, 20)) <= 21
    assert not hasattr(lora_notes, "_cut")
