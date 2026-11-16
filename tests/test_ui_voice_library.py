"""GUI tests: the voice library window (Cast → Library…): save a character's voice, give it to another character
(also in another project), listen, rename, delete."""
import os
import subprocess

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, image, new_project, win  # noqa: E402,F401


def test_the_voice_library_window(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ('@Anna says "Hallo."',))
    T.add_character(win.project, win.tree, "Anna")
    T.add_character(win.project, win.tree, "Ben")
    win._show_references(select="Anna")
    assert win.b_voice_lib.isEnabled()
    speech = tmp_path / "anna.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
                    str(speech)], check=True)
    assert win.set_character_voice(str(speech))
    win.modal_dialogs = False
    dlg = win.open_voice_library()
    assert dlg.list.count() == 0 and not dlg.empty.isHidden() and dlg.b_save.isEnabled()
    assert dlg.b_save.text() == "Save the voice of @Anna…" and not dlg.b_use.isEnabled()
    saved = dlg.save("Anna calm")
    assert saved and dlg.list.count() == 1 and dlg.selected_id() == saved["id"]
    assert "Anna calm" in dlg.list.item(0).text() and "cloned from" in dlg.list.item(0).text()
    assert "voice library" in win.status.text()
    played = []
    monkeypatch.setattr(win.preview, "open", lambda path, title: played.append((path, title)))
    dlg.play()
    assert played and played[0][0].name == "voice.wav" and "Anna calm" in played[0][1]
    assert dlg.rename("Anna, calm") and "Anna, calm" in dlg.list.item(0).text()
    dlg.close()

    win._show_references(select="Ben")                                       # another character: Use
    dlg = win.open_voice_library()
    assert dlg.b_use.text() == "Use for @Ben" and not dlg.b_save.isEnabled()
    dlg.list.setCurrentRow(0)
    assert dlg.b_use.isEnabled() and dlg.use()
    ben = T.character(T.load(win.project), "Ben")                            # (saved with the project)
    assert ben["voice"]["name"] == "Anna, calm" and ben["voice"]["ref"]
    assert (win.project / ben["voice"]["file"]).is_file()
    assert win.char_voice.text().startswith("Voice: Anna, calm") and "from the voice library" in win.status.text()

    second = tmp_path / "second"                                             # another project: still there
    new_project(win, second, image, ("one",))
    T.add_character(win.project, win.tree, "Cleo")
    win._show_references(select="Cleo")
    dlg = win.open_voice_library()
    assert dlg.list.count() == 1
    dlg.list.setCurrentRow(0)
    assert dlg.use() and T.character(win.tree, "Cleo")["voice"]["name"] == "Anna, calm"
    dlg = win.open_voice_library()
    dlg.list.setCurrentRow(0)
    assert dlg.delete(confirm=False) and dlg.list.count() == 0
    assert (win.project / T.character(win.tree, "Cleo")["voice"]["file"]).is_file()   # the project keeps its copy
