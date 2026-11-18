"""The cast panel and character sheets: Sheets… makes a person's head and full-body sheet (the sheets window), the
first photo of a person asks whether to make sheets from it or use it as a face picture, and sheet pictures are
shown as such (and not cropped)."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QInputDialog  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, crop_accepted, image, new_project, silent_player, win  # noqa: E402,F401


def _png(path, colour, size=(160, 90)):
    Image.new("RGB", size, colour).save(path)
    return path


def _fake_sheets(monkeypatch, tmp_path, opened, save=True):
    """The sheets window as if one try of each sheet was made (and Save pressed, or closed)."""
    from video_chain.ui.sheets_window import SheetsDialog

    def run(self):
        opened.append({"name": self.name, "source": self.source, "title": self.windowTitle()})
        if not save:
            self.reject()
            return 0
        self._add_result(str(_png(tmp_path / f"h{len(opened)}.png", (0, 200, 0), (300, 90))), "head_sheet")
        self._add_result(str(_png(tmp_path / f"b{len(opened)}.png", (0, 0, 200))), "body_sheet")
        self.accept()
        return 1
    monkeypatch.setattr(SheetsDialog, "exec", run)


def test_sheets_turns_a_person_with_a_face_into_one_with_sheets(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    face = _png(tmp_path / "face.png", (190, 150, 120), (80, 100))
    win.add_reference(str(face))                                     # (asked: a face picture, as before)
    assert not win.b_ref_sheets.isHidden() and win.b_ref_sheets.isEnabled()
    opened = []
    _fake_sheets(monkeypatch, tmp_path, opened)
    win.make_sheets()
    proj = tmp_path / "proj"
    assert opened[0]["name"] == "Person1" and opened[0]["title"] == "Character sheets of @Person1"
    assert opened[0]["source"] == proj / "refs/ref1_full.png"        # (the face's full picture: the photo)
    p1 = T.character(win.tree, "Person1")
    assert [T.role_of(p1, r) for r in p1["images"]] == ["head_sheet", "body_sheet"] and p1["lead"]
    assert Image.open(proj / T.sheet_source(p1)).size == (80, 100)
    assert "Sheets saved to @Person1" in win.status.text()
    assert "head + body sheet" in win.cast_list.item(0).text()
    assert [win.refs.item(i).text() for i in range(2)] == ["Head sheet", "Full-body sheet"]
    win.refs.setCurrentRow(0)
    assert not win.b_ref_crop.isEnabled()                            # (a sheet is never cropped)
    assert T.load(proj).options["cast"][0]["roles"] == p1["roles"]   # (saved)
    win.make_sheets()                                                # again: from the kept photo
    assert opened[1]["source"] == proj / T.sheet_source(p1)


def test_the_first_photo_of_a_person_asks_sheets_or_face(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    photo = _png(tmp_path / "photo.png", (100, 100, 100), (90, 120))
    asked = []
    monkeypatch.setattr(win, "ask_first_photo", lambda who: asked.append(who) or "sheets")
    opened = []
    _fake_sheets(monkeypatch, tmp_path, opened, save=False)
    assert not win.add_reference(str(photo))                         # closed without saving: nobody new
    assert asked == ["a new character"] and opened[0]["source"] == photo and T.cast_of(win.tree) == []
    _fake_sheets(monkeypatch, tmp_path, opened)
    assert win.add_reference(str(photo))
    p1 = T.character(win.tree, "Person1")
    assert p1["lead"] and T.sheet_of(p1, "body_sheet") and win._selected_character()["name"] == "Person1"
    # a person who has pictures: no question, a face picture as before
    monkeypatch.setattr(win, "ask_first_photo", lambda who: (_ for _ in ()).throw(AssertionError("asked")))
    assert win.add_reference(str(_png(tmp_path / "f2.png", (5, 5, 5), (80, 80))))
    assert len(T.character(win.tree, "Person1")["images"]) == 3
    # a new, empty person: asked; "face" is the old crop
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Ben", True))
    win._new_character()
    monkeypatch.setattr(win, "ask_first_photo", lambda who: asked.append(who) or "face")
    assert win.add_reference(str(_png(tmp_path / "ben.png", (9, 9, 9), (80, 100))))
    ben = T.character(win.tree, "Ben")
    assert asked[-1] == "@Ben" and len(ben["images"]) == 1 and not T.role_of(ben, ben["images"][0])
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Cara", True))
    win._new_character()
    monkeypatch.setattr(win, "ask_first_photo", lambda who: None)    # (cancelled)
    assert not win.add_reference(str(photo)) and T.character(win.tree, "Cara")["images"] == []


def test_sheets_needs_a_person_and_asks_for_a_photo_when_it_has_none(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import image_picker as IP
    new_project(win, tmp_path, image, ("a",))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Kitchen", True))
    win._new_character("place")
    assert win.b_ref_sheets.isHidden()
    assert not win.make_sheets() and "person" in win.status.text()
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Dan", True))
    win._new_character()
    photo = _png(tmp_path / "dan.png", (3, 3, 3), (90, 120))
    monkeypatch.setattr(IP, "pick_image", lambda *a, **k: str(photo))
    opened = []
    _fake_sheets(monkeypatch, tmp_path, opened)
    assert win.make_sheets()
    assert opened[0]["source"] == photo and T.sheet_of(T.character(win.tree, "Dan"), "head_sheet")


def test_no_sheets_window_when_two_sheets_do_not_fit(app, win, tmp_path, image, monkeypatch):
    """Review: the limit was only checked at Save, after the GPU work."""
    new_project(win, tmp_path, image, ("a",))
    for k in range(T.MAX_CAST_PICTURES):
        T.add_character(win.project, win.tree, f"O{k}", _png(tmp_path / f"o{k}.png", (k, k, k)), kind="object")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Eve", True))
    opened = []
    _fake_sheets(monkeypatch, tmp_path, opened)
    win._show_references()
    assert not win.make_sheets("Eve", _png(tmp_path / "e.png", (1, 1, 1)), new=True)
    assert opened == [] and "at most 9" in win.status.text()


def test_replacing_plain_faces_with_sheets_is_confirmed(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    new_project(win, tmp_path, image, ("a",))
    win.add_reference(str(_png(tmp_path / "face.png", (190, 150, 120), (80, 100))))
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
    opened = []
    _fake_sheets(monkeypatch, tmp_path, opened)
    assert not win.make_sheets() and opened == [] and "face" in asked[0]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    assert win.make_sheets() and len(opened) == 1
    assert win.make_sheets() and len(opened) == 2                   # (sheets already: nothing plain to lose)


def test_sheets_from_inside_another_image_window_keep_its_model_switch(app, win, tmp_path, image, monkeypatch):
    from PySide6.QtWidgets import QApplication, QWidget
    new_project(win, tmp_path, image, ("a",))
    outer = object()
    win._edit_switch = outer
    monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: QWidget()))
    closed = []
    monkeypatch.setattr(win, "edits_closed", lambda: closed.append(1))
    monkeypatch.setattr(win, "ask_first_photo", lambda who: "sheets")
    opened = []
    _fake_sheets(monkeypatch, tmp_path, opened)
    assert win.add_reference(str(_png(tmp_path / "p.png", (2, 2, 2), (90, 120))))
    assert win._edit_switch is outer and closed == []                # (the image window still runs)
