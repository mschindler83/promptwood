"""Fixes in the image windows: picker kinds, kept masks, no per-keystroke disk reads."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_picker_opens_for_every_kind(app, tmp_path):
    from video_chain.ui import image_picker as IP
    q = QSettings(str(tmp_path / "p.ini"), QSettings.IniFormat)
    for kind in IP.KINDS:
        dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind=kind)
        assert dlg.kind == kind
        dlg.deleteLater()


def test_empty_trash_keeps_masks(tmp_path):
    from video_chain.ui.image_edit import EDITS_DIR, ImageEditDialog
    edits = tmp_path / EDITS_DIR
    (edits / "sources").mkdir(parents=True)
    src, mask, junk = (edits / "sources" / n for n in ("src.png", "mask_1.png", "junk.png"))
    for f in (src, mask, junk):
        f.write_bytes(b"x")
        os.utime(f, (time.time() - 100, time.time() - 100))
    (edits / "r.png").write_bytes(b"x")
    (edits / "r.json").write_text(json.dumps({"source_copy": str(src), "mask": str(mask)}))
    ImageEditDialog._empty_trash(SimpleNamespace(project=tmp_path, _trash=None))
    assert src.exists() and mask.exists() and not junk.exists()
