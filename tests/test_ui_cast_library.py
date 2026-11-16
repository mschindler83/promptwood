"""GUI tests: the cast library window (Cast → Cast library…): save a cast, bring its entries into another
project, with the choice for a name that project already has."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import cast_library as CL  # noqa: E402
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, image, new_project, win  # noqa: E402,F401


def _face(path, colour):
    Image.new("RGB", (80, 100), colour).save(path)
    return path


def test_the_cast_library_window(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("@Anna and @Ben",))
    T.add_character(win.project, win.tree, "Anna", _face(tmp_path / "a.png", (200, 150, 120)), desc="woman")
    T.add_character(win.project, win.tree, "Ben", _face(tmp_path / "b.png", (90, 90, 200)))
    T.add_character(win.project, win.tree, "Kitchen", kind="place")
    win._show_references(select="Anna")
    assert win.b_cast_lib.isEnabled()
    win.modal_dialogs = False
    dlg = win.open_cast_library()
    assert dlg.list.count() == 0 and not dlg.empty.isHidden() and not dlg.b_bring.isEnabled()
    saved = dlg.save("Family")
    assert saved and dlg.list.count() == 1 and "@Anna, @Ben" in dlg.list.item(0).text()
    assert "@Kitchen" in dlg.list.item(0).text() and "3 entries" in win.status.text()
    assert dlg.rename("The family") and dlg.list.item(0).text().startswith("The family")
    dlg.close()

    second = tmp_path / "second"                                             # another project with its own Anna
    new_project(win, second, image, ("one",))
    T.add_character(win.project, win.tree, "Anna", _face(tmp_path / "mine.png", (1, 2, 3)), desc="mine")
    win._show_references(select="Anna")
    dlg = win.open_cast_library()
    dlg.list.setCurrentRow(0)
    pick = dlg.bring_in()
    rows = {p["name"]: (box, how, edit) for p, box, how, edit in pick.rows}
    assert not rows["Anna"][0].isChecked() and rows["Ben"][0].isChecked()   # a name it has: not ticked
    assert rows["Ben"][1] is None and rows["Anna"][2].text() == "Anna_2"
    assert "already in this project" in rows["Anna"][0].text()
    assert rows["Kitchen"][0].isChecked() and "place" in rows["Kitchen"][0].text()
    rows["Kitchen"][0].setChecked(False)
    rows["Anna"][0].setChecked(True)
    assert [(c.name, c.as_name, c.replace) for c in pick.choices()] == [("Anna", "Anna_2", False),
                                                                          ("Ben", "Ben", False)]
    rows["Anna"][1].setCurrentIndex(rows["Anna"][1].findData("replace"))
    assert rows["Anna"][2].isHidden()
    assert pick.choices()[0] == CL.Choice("Anna", "Anna", True)
    pick.accept()
    cast = T.cast_of(T.load(win.project))                                    # (saved with the project)
    assert [c["name"] for c in cast] == ["Anna", "Ben"] and cast[0]["desc"] == "woman" and cast[0]["lead"]
    assert all((win.project / c["images"][0]).is_file() for c in cast)
    assert "From the cast library: @Anna, @Ben" in win.status.text()
    assert win.cast_list.count() == 2 and not dlg.isVisible()

    dlg = win.open_cast_library()                                           # the same again: both names taken
    dlg.list.setCurrentRow(0)
    assert not dlg.bring_in([CL.Choice("Ben", "Ben")]) and "already" in win.status.text()
    assert [c["name"] for c in T.cast_of(win.tree)] == ["Anna", "Ben"]
    assert dlg.delete(confirm=False) and dlg.list.count() == 0


def test_proposed_names_never_meet_another_saved_persons_name(app, win, tmp_path, image):
    """The project has @Anna; the saved cast has Anna and Anna_2: Anna is proposed as Anna_3, not Anna_2."""
    from video_chain.ui.cast_library import BringInDialog
    new_project(win, tmp_path, image, ("one",))
    T.add_character(win.project, win.tree, "Anna", _face(tmp_path / "a.png", (1, 2, 3)))
    saved = {"name": "S", "entries": [{"name": "Anna", "faces": 1}, {"name": "Anna_2", "faces": 1}]}
    pick = BringInDialog(win, saved, win.tree)
    rows = {p["name"]: (box, how, edit) for p, box, how, edit in pick.rows}
    assert rows["Anna"][2].text() == "Anna_3" and rows["Anna_2"][0].isChecked()


def test_a_bring_in_failing_halfway_is_shown_and_saved(app, win, tmp_path, image, monkeypatch):
    """A copy fails after the first person (a full disk): the people brought in so far are shown and saved, and the
    status says what failed."""
    new_project(win, tmp_path, image, ("one",))
    T.add_character(win.project, win.tree, "Anna", _face(tmp_path / "a.png", (200, 150, 120)))
    T.add_character(win.project, win.tree, "Ben", _face(tmp_path / "b.png", (90, 90, 200)))
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(win.project, win.tree, "Two")
    new_project(win, tmp_path / "second", image, ("one",))
    real, calls = T.put_entry, []

    def second_fails(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("No space left on device")
        return real(*a, **k)
    monkeypatch.setattr(T, "put_entry", second_fails)
    assert not win.bring_in_cast(saved["id"], [CL.Choice("Anna", "Anna"), CL.Choice("Ben", "Ben")], lib)
    assert "No space left" in win.status.text() and "@Anna" in win.status.text()
    assert [c["name"] for c in T.cast_of(T.load(win.project))] == ["Anna"] and win.cast_list.count() == 1


def test_replacing_with_another_kind_says_so(app, win, tmp_path, image):
    from video_chain.ui.cast_library import BringInDialog
    new_project(win, tmp_path, image, ("one",))
    T.add_character(win.project, win.tree, "Anna")
    T.add_character(win.project, win.tree, "Home", kind="place")
    saved = {"name": "S", "entries": [{"name": "Anna", "kind": "place", "faces": 0},
                                      {"name": "Home", "kind": "place", "faces": 0}]}
    pick = BringInDialog(win, saved, win.tree)
    rows = {p["name"]: how for p, _box, how, _edit in pick.rows}
    assert rows["Anna"].itemText(1) == "Replace @Anna (a person now: it becomes a place)"
    assert "voice" in rows["Anna"].toolTip() and "lead" in rows["Anna"].toolTip()
    assert rows["Home"].itemText(1) == "Replace @Home"
