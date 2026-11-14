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


def _dialog(tmp_path, name="d.ini", **kw):
    from video_chain.ui.image_edit import ImageEditDialog
    q = QSettings(str(tmp_path / name), QSettings.IniFormat)
    return ImageEditDialog(None, tmp_path, [], None, None, lambda p: None, lambda k, p: False, q,
                           connect_create=lambda p: None, **kw)


def test_before_after_tooltip_says_toggle(app, tmp_path):
    dlg = _dialog(tmp_path)
    assert "hold" not in dlg.b_before.toolTip().lower()
    dlg.reject()


def test_original_is_shown_upright(app, tmp_path):
    """A phone photo (EXIF-rotated) as the original: shown turned like the result, not sideways."""
    from PIL import Image

    from video_chain.ui.image_edit import EDITS_DIR
    orig = tmp_path / "phone.jpg"
    im = Image.new("RGB", (40, 20), "blue")
    ex = Image.Exif()
    ex[0x0112] = 6                                         # rotated 90 degrees
    im.save(orig, exif=ex)
    edits = tmp_path / EDITS_DIR
    edits.mkdir()
    Image.new("RGB", (20, 40), "red").save(edits / "r.png")
    (edits / "r.json").write_text(json.dumps({"prompt": "a", "source_copy": str(orig)}))
    dlg = _dialog(tmp_path)
    shown = []
    dlg.view.set_image = lambda img: shown.append(img.size())
    dlg.b_before.setChecked(True)
    dlg._show_selected()
    assert (shown[-1].width(), shown[-1].height()) == (20, 40)
    dlg.reject()


def test_close_while_working_asks(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    dlg = _dialog(tmp_path)
    cancelled = []
    dlg.cancel = lambda: cancelled.append(1)
    dlg._busy = True
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    dlg.reject()
    assert not cancelled and not dlg.closed                 # kept working, still open
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dlg.reject()
    assert cancelled and dlg.closed


def test_folder_run_of_a_dozen_asks(app, tmp_path, monkeypatch):
    from PIL import Image
    from PySide6.QtWidgets import QMessageBox
    folder = tmp_path / "pics"
    folder.mkdir()
    for i in range(6):
        Image.new("RGB", (16, 16), "red").save(folder / f"p{i}.png")
    dlg = _dialog(tmp_path)
    assert dlg.set_folder(str(folder))
    dlg.prompt.setPlainText("x")
    dlg.tries.setValue(2)                                   # 6 pictures x 2 = 12 images
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a) or QMessageBox.No)
    dlg.generate()
    assert len(asked) == 1 and "12" in asked[0][2]
    assert not dlg._busy
    dlg.reject()


def test_uses_of_with_a_prebuilt_index(tmp_path):
    from video_chain import tree as T
    from video_chain.ui.image_edit import uses_of, uses_index
    t = T.Tree()
    meta = {"uses": [{"kind": "end", "file": "e.png"}, {"kind": "ref", "file": "r.png"}]}
    idx = uses_index(t)
    assert uses_of(tmp_path, t, meta, idx) == uses_of(tmp_path, t, meta)


def test_result_record_is_written_atomically(app, tmp_path, monkeypatch):
    from PIL import Image

    from video_chain.ui import image_edit as IE

    class Editor:
        def models(self):
            return list(IE.EDIT_MODELS)

        def qualities(self, model):
            return ["fast", "standard", "best"]

        def edit(self, src, prompt, out, seed, quality, refs, loras, model=None, mask=None):
            Image.new("RGB", (16, 16), "red").save(out)
            return {"edit_model": model}

        def interrupt(self):
            pass

    written = []
    from video_chain import edits as ED
    real = ED.write_meta                                       # (the worker writes through video_chain.edits)
    monkeypatch.setattr(ED, "write_meta", lambda f, m: (written.append(Path(f)), real(f, m)))
    src = tmp_path / "s.png"
    Image.new("RGB", (16, 16), "blue").save(src)
    dlg = _dialog(tmp_path)
    dlg.connect_editor = lambda progress: Editor()
    dlg.set_source(str(src))
    dlg.prompt.setPlainText("x")
    dlg.tries.setValue(1)
    dlg.generate()
    end = time.time() + 20
    while dlg._busy and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    assert any(f.parent.name == IE.EDITS_DIR and f.suffix == ".json" and f.stem.startswith("edit_") for f in written)
    dlg.reject()


def test_picker_sort_by_date_survives_an_unreadable_entry(app, tmp_path, monkeypatch):
    from PIL import Image

    from video_chain.ui import image_picker as IP
    for n in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "red").save(tmp_path / n)
    real = os.scandir

    class Entry:
        def __init__(self, e):
            self._e = e
            self.name, self.path = e.name, e.path

        def is_dir(self):
            return self._e.is_dir()

        def is_file(self):
            return self._e.is_file()

        def stat(self):
            if self.name == "a.png":
                raise OSError("gone")
            return self._e.stat()
    monkeypatch.setattr(IP.os, "scandir", lambda p: [Entry(e) for e in real(p)])
    q = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None)
    dlg.sort.setCurrentIndex(dlg.sort.findData("date"))
    dlg.open_folder(tmp_path)
    assert {Path(p).name for p in dlg._items} == {"a.png", "b.png"}
    dlg.deleteLater()


def test_picker_zoom_is_saved_once_it_settles(app, tmp_path):
    from video_chain.ui import image_picker as IP
    writes = []

    class Q(QSettings):
        def setValue(self, k, v):
            if k == "pickerZoom":
                writes.append(v)
            super().setValue(k, v)
    q = Q(str(tmp_path / "z.ini"), QSettings.IniFormat)
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None)
    writes.clear()
    for v in (120, 130, 140, 150, 160):
        dlg.zoom.setValue(v)
    assert writes == []
    end = time.time() + 3
    while not writes and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert writes == [160]
    dlg.deleteLater()


def test_picker_video_preview_runs_off_the_ui_thread(app, tmp_path, monkeypatch):
    import threading

    from PySide6.QtGui import QImage

    from video_chain.ui import image_picker as IP
    (tmp_path / "v.mp4").write_bytes(b"x")
    threads = []
    gate = threading.Event()

    def frame(path, box=IP.THUMB_MAX):
        threads.append(threading.current_thread())
        gate.wait(5)
        img = QImage(8, 8, QImage.Format_RGB32)
        img.fill(0x123456)
        return img

    def info(path):
        threads.append(threading.current_thread())
        return "0:01.0 · fake"
    monkeypatch.setattr(IP, "_video_frame", frame)
    monkeypatch.setattr(IP, "_media_info", info)
    q = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    dlg = IP.ImagePickerDialog(None, "x", str(tmp_path), q, cache_dir=None, desktop_thumbs=None, kind="video")
    item = dlg._items[str(tmp_path / "v.mp4")]
    threads.clear()
    dlg.grid.setCurrentItem(item)
    assert "Loading" in dlg.info.text()                      # answered at once
    gate.set()
    end = time.time() + 5
    while "fake" not in dlg.info.text() and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert "fake" in dlg.info.text()
    assert threads and all(t is not threading.main_thread() for t in threads)
    dlg.deleteLater()


def test_thumbnail_loader_survives_a_failing_decode(app, tmp_path, monkeypatch):
    from PySide6.QtGui import QImage

    from video_chain.ui import results_gallery as RG
    got = []

    def read(png):
        if png.endswith("bad.png"):
            raise MemoryError("boom")
        img = QImage(10, 10, QImage.Format_RGB32)
        img.fill(0x00ff00)
        return img
    monkeypatch.setattr(RG, "read_thumb", read)
    loader = RG.ThumbLoader()
    loader.ready.connect(lambda png, ns, img: got.append((png, img.isNull())))
    loader.request("bad.png", 1)
    time.sleep(0.2)
    loader.request("good.png", 2)
    end = time.time() + 5
    while len(got) < 2 and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    loader.stop()
    assert ("bad.png", True) in got and ("good.png", False) in got


def test_thumbnails_follow_the_widgets_pixel_ratio():
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QPixmap

    from video_chain.ui.results_gallery import badged
    pm = QPixmap(20, 20)
    pm.fill()
    assert badged(pm, QSize(150, 100), "", False, False, dpr=2.0).devicePixelRatio() == 2.0


def test_project_edits_reads_each_record_once(tmp_path, monkeypatch):
    from video_chain.ui import image_edit as IE
    edits = tmp_path / IE.EDITS_DIR
    edits.mkdir()
    for i in range(3):
        (edits / f"r{i}.png").write_bytes(b"x")
        (edits / f"r{i}.json").write_text(json.dumps({"prompt": str(i)}))
    reads = []
    orig = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self), orig(self, *a, **k))[1])
    first = IE.project_edits(tmp_path)
    n = len(reads)
    again = IE.project_edits(tmp_path)
    assert len(reads) == n and sorted(m["prompt"] for _p, m in again) == ["0", "1", "2"]
    again[0][1]["prompt"] = "changed"                     # (a caller's change does not leak into the cache)
    assert sorted(m["prompt"] for _p, m in IE.project_edits(tmp_path)) == ["0", "1", "2"]
    (edits / "r0.json").write_text(json.dumps({"prompt": "new, longer"}))      # a changed record is read again
    assert "new, longer" in [m["prompt"] for _p, m in IE.project_edits(tmp_path)]
    assert first


def test_source_picture_is_decoded_once(app, tmp_path, monkeypatch):
    """Setting a source, painting and setting a mask, and browsing results decoded the same file again and again."""
    from PIL import Image

    from video_chain.ui import image_edit as IE
    src = tmp_path / "s.png"
    Image.new("RGB", (64, 48), "blue").save(src)
    reads = []
    from video_chain.ui import imageio
    real = imageio.decode_upright                              # (upright() lives in ui/imageio.py now)
    monkeypatch.setattr(imageio, "decode_upright", lambda p: (reads.append(p), real(p))[1])
    dlg = _dialog(tmp_path)
    dlg.set_source(str(src))
    from PySide6.QtGui import QImage
    m = QImage(64, 48, QImage.Format_Grayscale8)
    m.fill(255)
    dlg.set_mask(m)
    dlg.set_mask(None)
    dlg.set_mask(m)
    assert len(reads) == 1
    Image.new("RGB", (64, 48), "red").save(src)                # the file changed: read again
    os.utime(src, (time.time() + 5, time.time() + 5))
    dlg.set_source(str(src))
    assert len(reads) == 2
    dlg.reject()


def test_text_builders():
    from video_chain.ui import image_edit as IE
    assert IE.how_made({"mode": "create", "family": "qwen21", "quality": "fast"}).startswith(
        IE.FAMILIES["qwen21"]["label"])
    assert IE.how_made({"edit_model": "qwen21", "quality": "fast"}) == (
        f"{IE.EDIT_MODELS['qwen21']} {IE.FAMILIES['qwen21']['qualities'].get('fast', '')}")
    assert IE.how_made({"edit_model": "moss"}) == f"{IE.EDIT_MODELS['moss']} {IE.MOSS_EDIT_QUALITIES['standard']}"
    assert IE.how_made({"quality": "fast"}) == IE.QUALITY_TEXT.get("fast", "")
    assert "Qwen-Image 2.1 LoRAs" in IE.lora_note("edit", None, "", "qwen21", {"qwen21": [1, 2]}) \
        and "2 found" in IE.lora_note("edit", None, "", "qwen21", {"qwen21": [1, 2]})
    assert "applied before its edit LoRA" in IE.lora_note("edit", None, "", "moss", {})
    assert "Text-to-image Qwen LoRAs" in IE.lora_note("edit", None, "", "2511", {})
    assert "Chroma and FLUX.1" in IE.lora_note("create", {"label": "Chroma"}, "chroma", "", {})
    assert IE.lora_note("create", None, "x", "", {}) == ""


def test_thumbnail_arrival_finds_its_item_directly(app, tmp_path):
    from PIL import Image
    from PySide6.QtGui import QImage

    from video_chain.ui.image_edit import EDITS_DIR
    edits = tmp_path / EDITS_DIR
    edits.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), "red").save(edits / f"r{i}.png")
        (edits / f"r{i}.json").write_text(json.dumps({"prompt": str(i)}))
    dlg = _dialog(tmp_path)
    assert set(dlg._strip_items) == {str(edits / f"r{i}.png") for i in range(3)}
    img = QImage(8, 8, QImage.Format_RGB32)
    img.fill(0xff0000)
    png = str(edits / "r1.png")
    dlg._thumb_ready(png, 5, img)
    assert not dlg._strip_items[png].icon().isNull()
    dlg.reject()
