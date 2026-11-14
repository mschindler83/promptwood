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


def test_typing_does_not_read_the_disk(app, tmp_path, monkeypatch):
    """Every typed character used to re-read the selected result's record (twice) from disk."""
    from PIL import Image

    from video_chain.ui.image_edit import EDITS_DIR, ImageEditDialog
    edits = tmp_path / EDITS_DIR
    edits.mkdir()
    Image.new("RGB", (16, 16), "red").save(edits / "r.png")
    (edits / "r.json").write_text(json.dumps({"prompt": "a", "mode": "create"}))
    q = QSettings(str(tmp_path / "t.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                          connect_create=lambda p: None)
    app.processEvents()
    assert dlg.selected() is not None
    reads = []
    orig = Path.read_text

    def counting(self, *a, **k):
        if self.suffix == ".json":
            reads.append(self)
        return orig(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", counting)
    for i in range(5):
        dlg.prompt.setPlainText("x" * (i + 1))
    assert reads == []
    assert dlg.b_gen.isEnabled() is not None
    dlg.reject()


def test_generate_tooltip_resets_when_enabled(app, tmp_path):
    from video_chain.ui.image_edit import ImageEditDialog
    q = QSettings(str(tmp_path / "g.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                          connect_create=lambda p: None)
    dlg.set_mode("create")
    from video_chain.ui.image_edit import FAMILIES
    dlg.families = {f: {"models": ["m"]} for f in FAMILIES}
    dlg.family.clear()
    dlg.family.addItem("x", next(iter(FAMILIES)))
    dlg.use_refs.setChecked(False)
    dlg.prompt.setPlainText("")
    dlg._update_buttons()
    assert dlg.b_gen.toolTip() == "Describe the image"
    dlg.prompt.setPlainText("a cat")
    assert dlg.b_gen.isEnabled() and "Describe" not in dlg.b_gen.toolTip() and "Ctrl+Enter" in dlg.b_gen.toolTip()
    dlg.reject()
