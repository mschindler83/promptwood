"""GUI tests: the prompt library window, opened from a clip card and from the image window: save prompts into
folders, find them, put one into the prompt (insert or replace), organize them."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtWidgets import QPlainTextEdit  # noqa: E402

from conftest import settings  # noqa: E402,F401
from ui_helpers import app, image, new_project, win  # noqa: E402,F401
from video_chain import prompt_library as PL  # noqa: E402
from video_chain.ui.prompt_library import PromptLibraryDialog  # noqa: E402


def _names(dlg):
    out = []
    for k in range(dlg.tree.topLevelItemCount()):
        it = dlg.tree.topLevelItem(k)
        out.append(it.text(0))
        out += ["  " + it.child(j).text(0) for j in range(it.childCount())]
    return out


def test_save_find_and_use_prompts(app, tmp_path):  # noqa: F811
    lib = PL.PromptLibrary(tmp_path / "lib")
    field = QPlainTextEdit("the camera pushes in slowly")
    dlg = PromptLibraryDialog(None, "video", field, lib)
    assert not dlg.empty.isHidden() and dlg.b_save.isEnabled() and not dlg.b_insert.isEnabled()
    a = dlg.save("Push in", "Camera")
    assert a and dlg.selected_id() == a["id"] and dlg.preview.toPlainText() == "the camera pushes in slowly"
    field.setPlainText("@Anna walks to the window")
    assert dlg.save("Walk", "") and _names(dlg) == ["Camera   (1)", "  Push in", "Walk"]
    assert dlg.save("walk", "", replace=False) is None                         # (the same name: not replaced)
    assert lib.find("video", "", "Walk")["text"] == "@Anna walks to the window"
    dlg.search.setText("pushes")                                                # search names and text
    assert _names(dlg) == ["Camera   (1)", "  Push in"]
    dlg.search.clear()
    assert dlg.select(a["id"]) and dlg.b_insert.isEnabled() and dlg.b_update.isEnabled()
    field.setPlainText("She smiles.")
    cur = field.textCursor()
    cur.movePosition(QTextCursor.MoveOperation.End)
    field.setTextCursor(cur)
    assert dlg.use("insert") and field.toPlainText().rstrip() == "She smiles. the camera pushes in slowly"
    dlg = PromptLibraryDialog(None, "video", field, lib)
    dlg.select(a["id"])
    assert dlg.use("replace") and field.toPlainText() == "the camera pushes in slowly"
    field.undo()                                                                # Ctrl+Z takes it back
    assert field.toPlainText().rstrip() == "She smiles. the camera pushes in slowly"


def test_organize_the_library(app, tmp_path):  # noqa: F811
    lib = PL.PromptLibrary(tmp_path / "lib")
    lib.save("video", "Push in", "push", folder="Camera")
    lib.save("video", "Pan", "pan", folder="Camera")
    walk = lib.save("video", "Walk", "walk")
    lib.save("edit", "Sheet", "a character sheet", folder="Sheets")
    field = QPlainTextEdit("walks faster")
    dlg = PromptLibraryDialog(None, "video", field, lib)
    assert dlg.select(walk["id"]) and dlg.update_selected(confirm=False)
    assert lib.get(walk["id"])["text"] == "walks faster"
    assert dlg.move_to("Moves") and lib.get(walk["id"])["folder"] == "Moves"
    assert dlg.rename("Stroll") and lib.get(walk["id"])["name"] == "Stroll"
    folder = next(dlg.tree.topLevelItem(k) for k in range(dlg.tree.topLevelItemCount())
                  if dlg.tree.topLevelItem(k).text(0).startswith("Camera"))
    dlg.tree.setCurrentItem(folder)
    assert dlg.selected_folder() == "Camera" and not dlg.b_insert.isEnabled() and not dlg.b_move.isEnabled()
    assert dlg.rename("Shots") and lib.folders("video") == ["Moves", "Shots"]
    folder = next(dlg.tree.topLevelItem(k) for k in range(dlg.tree.topLevelItemCount())
                  if dlg.tree.topLevelItem(k).text(0).startswith("Shots"))
    dlg.tree.setCurrentItem(folder)
    assert dlg.delete(confirm=False) and [p["name"] for p in lib.list("video")] == ["Stroll"]
    dlg.kind.setCurrentIndex(dlg.kind.findData("edit"))                         # the other kinds
    assert _names(dlg) == ["Sheets   (1)", "  Sheet"]
    field.setReadOnly(True)                                                     # (a frozen clip: saved, not changed)
    dlg = PromptLibraryDialog(None, "video", field, lib)
    dlg.select(walk["id"])
    assert dlg.b_save.isEnabled() and not dlg.b_replace.isEnabled() and not dlg.use("replace")


def test_a_clip_card_opens_it(app, win, tmp_path, image):  # noqa: F811
    new_project(win, tmp_path, image, ("@Anna walks to the window",))
    nid = next(n.id for n in win.tree.nodes.values() if n.prompt)
    card = win._card_of(nid, type(next(c for c in win.cards if c.nid == nid)))
    opened = []
    with pytest.MonkeyPatch.context() as mp:     # (its own: monkeypatch.undo would undo conftest's folders too)
        mp.setattr(win, "open_prompt_library", lambda kind, target=None: opened.append((kind, target)))
        card.a_prompts.trigger()                    # (the Prompts menu's last entry)
    assert opened == [("video", card.edit)]
    assert str(PL.library().dir).startswith(str(tmp_path))                   # (never the real library)
    win.modal_dialogs = False
    dlg = win.open_prompt_library("video", card.edit)
    assert dlg.save("Walk", "People") and PL.library().find("video", "People", "Walk")
    card.edit.setPlainText("")
    assert dlg.use("replace") and card.edit.toPlainText() == "@Anna walks to the window"


def test_the_image_window_opens_it(app, tmp_path):  # noqa: F811
    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    q = QSettings(str(tmp_path / "d.ini"), QSettings.IniFormat)
    dlg = ImageEditDialog(None, tmp_path, [], None, None, services=ImageEditServices(
        connect=lambda p: None, on_use=lambda k, p: False, settings=q, connect_create=lambda p: None))
    dlg.set_mode("edit")
    lib_dlg = dlg.open_prompt_library(modal=False)
    assert lib_dlg.current_kind() == "edit" and lib_dlg.target is dlg.prompt
    dlg.set_mode("create")
    assert dlg.open_prompt_library(modal=False).current_kind() == "create"
    dlg.reject()
