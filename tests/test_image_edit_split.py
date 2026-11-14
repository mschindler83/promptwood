"""The image dialog after its split: edits records without Qt, undo of several deletes, one generate worker."""
import json
import os
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _results(tmp_path, names=("r0", "r1", "r2")):
    from PIL import Image
    edits = tmp_path / "edits"
    edits.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        Image.new("RGB", (16, 16), "red").save(edits / f"{n}.png")
        (edits / f"{n}.json").write_text(json.dumps({"prompt": n}))
        os.utime(edits / f"{n}.png", (time.time() - 50 + i, time.time() - 50 + i))
    return edits


def _dialog(tmp_path, **kw):
    from video_chain.ui.image_edit import ImageEditDialog
    q = QSettings(str(tmp_path / "t.ini"), QSettings.IniFormat)
    return ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q, **kw)


def test_several_deletes_can_each_be_undone(app, tmp_path):
    edits = _results(tmp_path)
    dlg = _dialog(tmp_path)
    first = Path(dlg.selected()[0]).name
    dlg.delete_selected()
    second = Path(dlg.selected()[0]).name
    dlg.delete_selected()
    assert not (edits / first).exists() and not (edits / second).exists()
    assert dlg.b_undo.isVisibleTo(dlg)
    dlg.undo_delete()
    assert (edits / second).exists() and (edits / second).with_suffix(".json").exists()
    assert dlg.b_undo.isVisibleTo(dlg)                  # one more to bring back
    dlg.undo_delete()
    assert (edits / first).exists() and (edits / first).with_suffix(".json").exists()
    assert not dlg.b_undo.isVisibleTo(dlg)
    dlg.reject()


def test_closing_empties_the_trash_for_good(app, tmp_path):
    edits = _results(tmp_path)
    dlg = _dialog(tmp_path)
    dlg.delete_selected()
    dlg.delete_selected()
    assert any((edits / ".deleted").iterdir())
    dlg.reject()
    assert not (edits / ".deleted").exists()
    assert len(list(edits.glob("*.png"))) == 1
