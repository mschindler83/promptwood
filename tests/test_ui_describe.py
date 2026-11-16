"""GUI tests: Describe writes a place's description from its pictures (a fake llama.cpp model; no GPU)."""
import os
import threading

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, flush, image, new_project, wait, win  # noqa: E402,F401


class FakeLlama:
    model, mmproj = "/m/vision-Q4.gguf", "/m/mmproj-F16.gguf"

    def __init__(self, answer="a small sunny kitchen with white tiles", fail=""):
        self.answer, self.fail, self.asked, self.go = answer, fail, [], threading.Event()
        self.go.set()

    def describe(self, kind, name, pictures, desc=""):
        self.asked.append((kind, name, [os.path.basename(p) for p in pictures], desc))
        self.go.wait(10)
        if self.fail:
            raise RuntimeError(self.fail)
        return self.answer

    def stop(self, wait=True, grace=20):
        return False


def _place(win, tmp_path):
    pic = tmp_path / "k.png"
    Image.new("RGB", (160, 120), (200, 200, 0)).save(pic)
    T.add_character(win.project, win.tree, "Kitchen", pic, kind="place")
    T.add_character(win.project, win.tree, "Anna")
    win._show_references(select="Kitchen")


def test_describe_fills_a_places_description(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("Anna cooks in @Kitchen",))
    _place(win, tmp_path)
    assert not win.b_char_describe.isHidden() and win.b_char_describe.isEnabled()
    monkeypatch.setattr(win, "llama_server", lambda: None)
    assert not win.describe_entry() and "Prompt refiner: also a llama.cpp server" in win.status.text()
    blind = FakeLlama()
    blind.mmproj = None
    monkeypatch.setattr(win, "llama_server", lambda: blind)
    assert not win.describe_entry() and "cannot see pictures" in win.status.text() and not blind.asked
    fake = FakeLlama()
    monkeypatch.setattr(win, "llama_server", lambda: fake)
    fake.go.clear()
    assert win.describe_entry()
    assert not win.b_char_describe.isEnabled() and "Describing @Kitchen" in win.status.text()
    assert not win.describe_entry()                                          # one at a time
    fake.go.set()
    wait(app, lambda: win._describing is None)
    flush(app, win)
    assert fake.asked == [("place", "Kitchen", ["ref1.png"], "")]
    assert T.character(T.load(win.project), "Kitchen")["desc"] == "a small sunny kitchen with white tiles"
    assert win.char_desc.text() == "a small sunny kitchen with white tiles" and win.b_char_describe.isEnabled()
    assert "written from its pictures" in win.status.text()
    win._show_references(select="Anna")                                     # a person: no Describe
    assert win.b_char_describe.isHidden()


def test_describe_asks_before_replacing_and_keeps_an_edit_made_meanwhile(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("one",))
    _place(win, tmp_path)
    T.update_character(win.tree, "Kitchen", desc="mine")
    win._show_references(select="Kitchen")
    fake = FakeLlama()
    monkeypatch.setattr(win, "llama_server", lambda: fake)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), QMessageBox.No)[1])
    assert not win.describe_entry() and "“mine”" in asked[0] and not fake.asked
    fake.go.clear()
    assert win.describe_entry(confirm=False)
    T.update_character(win.tree, "Kitchen", desc="edited while it ran")
    fake.go.set()
    wait(app, lambda: win._describing is None)
    assert T.character(win.tree, "Kitchen")["desc"] == "edited while it ran"
    assert "not replaced" in win.status.text() and "white tiles" in win.status.text()
    fake.fail = "the llama.cpp server could not be asked"
    assert win.describe_entry(confirm=False)
    wait(app, lambda: win._describing is None)
    assert "Could not describe @Kitchen" in win.status.text() and win.b_char_describe.isEnabled()


def test_adding_a_picture_to_a_place_without_description_mentions_describe(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("one",))
    T.add_character(win.project, win.tree, "Garden", kind="place")
    win._show_references(select="Garden")
    pic = tmp_path / "g.png"
    Image.new("RGB", (160, 120), (0, 200, 0)).save(pic)
    assert win.add_reference(str(pic), crop=None)
    assert "Describe can write its description" in win.status.text()
