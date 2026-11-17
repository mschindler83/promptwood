"""GUI tests: people, sounds and voices: the cast, audio tracks, text to speech, ElevenLabs.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import os
from pathlib import Path


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtWidgets import QInputDialog, QMessageBox  # noqa: E402

from video_chain import chain  # noqa: E402
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    app, crop_accepted, flush, image, make_real_video, new_project, run_and_wait, silent_player, tr_lines,
    type_prompt, wait, win)


def test_identity_references_in_the_window(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import image_picker as IP
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    face = tmp_path / "face.png"
    Image.new("RGB", (80, 100), (190, 150, 120)).save(face)
    win.add_reference(str(face))                                               # no character yet: Person1, lead
    flush(app, win)
    assert "2 made clips are now CHANGED" in win.status.text()
    assert win.cast_list.count() == 1 and win.cast_list.item(0).text().startswith("@Person1")
    assert win.char_lead.isChecked() and win.refs.count() == 1 and win.refs.item(0).text() == "Person1 1"
    assert (tmp_path / "proj" / "refs" / "ref1.png").is_file()
    assert [win.state(i) for i in ids] == ["changed", "changed"]               # every clip gets the lead's face
    card = win.cards[0]
    assert card.faces.text() == "Faces: @Person1 (the lead; the prompt mentions nobody)"
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[0]].made["settings"]
    assert made["refs"][0]["file"] == "refs/ref1.png" and made["ref_size"] == "match"
    win.ref_size.setCurrentIndex(win.ref_size.findData("max"))
    flush(app, win)
    assert win.state(ids[0]) == "changed"
    wait(app, lambda: not win._joins, timeout=30)                              # the auto-play join of the run
    # a second character, mentioned in clip 2 only
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Ben", True))
    win._new_character()
    assert win._selected_character()["name"] == "Ben" and "@Ben added" in win.status.text()
    other = tmp_path / "ben.png"
    Image.new("RGB", (80, 100), (60, 50, 40)).save(other)
    win.add_reference(str(other))
    assert win.status.text().startswith("Face added to @Ben")
    assert T.character(win.tree, "Ben")["images"] == ["refs/ref2.png"] and not win.char_lead.isChecked()
    proj = tmp_path / "proj"
    assert Image.open(proj / "refs/ref2.png").size == (80, 80)           # no face found: the middle square
    assert Image.open(proj / "refs/ref2_full.png").size == (80, 100)     # the full picture is kept
    monkeypatch.setattr(win, "crop_face", lambda path, title, crop=None: (10, 20, 40))
    win.refs.setCurrentRow(0)
    win._recrop_reference()
    assert Image.open(proj / "refs/ref2.png").size == (40, 40) and "cropped" in win.status.text()
    monkeypatch.setattr(win, "crop_face", lambda path, title, crop=None: False)     # cancelled: nothing added
    assert not win.add_reference(str(other)) and len(T.character(win.tree, "Ben")["images"]) == 1
    monkeypatch.delattr(win, "crop_face")                                     # the window's own again
    # Create…: the Create image window, its face button for the selected character
    assert win.b_ref_create.isEnabled()
    win.modal_dialogs = False
    dlg = win.create_reference()
    assert dlg.mode() == "create" and dlg.b_use_ref.text() == "Add as face of @Ben"
    made = tmp_path / "made.png"
    Image.new("RGB", (90, 90), (20, 90, 160)).save(made)
    monkeypatch.setattr(win, "crop_face", lambda path, title, crop=None: None)
    assert dlg.on_use("ref", str(made)).startswith("Added as a face of @Ben")
    assert len(T.character(win.tree, "Ben")["images"]) == 2
    monkeypatch.delattr(win, "crop_face")
    win.refs.setCurrentRow(1)
    win._remove_reference()                                                  # (one face again for the rest)
    assert len(T.character(win.tree, "Ben")["images"]) == 1
    dlg.reject()
    dlg.deleteLater()
    win.edits_closed()
    type_prompt(win, ids[1], "@Bem waves")
    flush(app, win)
    win.select(ids[1])
    win.generate_path()
    wait(app, lambda: not win.running)
    assert not win.running and "did you mean @Ben? (Cast)" in win.status.text()
    type_prompt(win, ids[1], "@Ben waves")
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.faces.text() == "Faces: @Ben" and "Ben (the person in <Picture 1>)" in card.edit.toolTip()
    # typing @B offers the names; the chosen one goes in whole
    card.edit.setPlainText("Then @B")
    card.edit.moveCursor(QTextCursor.End)
    card.mentions._insert("Ben")
    assert card.edit.toPlainText() == "Then @Ben "
    # rename: every @mention follows
    type_prompt(win, ids[1], "@Ben waves")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Tom", True))
    win._rename_character()
    assert win.tree.nodes[ids[1]].prompt == "@Tom waves" and T.character(win.tree, "Ben") is None
    # a voice for Tom: used where Tom says a line
    import subprocess
    speech = tmp_path / "tom.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=9",
                    str(speech)], check=True)
    assert not win.b_voice_play.isEnabled() and win.char_voice.text() == "Voice: none"
    assert win.set_character_voice(str(speech))
    assert win.status.text().startswith("Voice of @Tom set (7.5 s, the first 7.5 s)")
    assert "Text to speech clones it from 9.0 s." in win.status.text()
    # Voice…: the file is chosen, then the part of it (a video here: 40 s, the first 30 s chosen to begin with)
    from PySide6.QtWidgets import QDialog, QDialogButtonBox
    from video_chain.ui.audio_editor import AudioRangeDialog
    film = tmp_path / "talk.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=40", "-f", "lavfi",
                    "-i", "color=gray:s=64x48:d=40", "-shortest", str(film)], check=True)
    monkeypatch.setattr(IP, "pick_image", lambda *a, **k: str(film))
    seen = {}

    def choose(dlg):
        ok = dlg.buttons.button(QDialogButtonBox.Ok)
        seen.update(title=dlg.windowTitle(), first=(dlg.start, dlg.end), name=dlg.name.isVisibleTo(dlg),
                    text=dlg.summary.text())
        dlg.set_range(12.0, 13.0)
        seen["short"] = ok.isEnabled()
        dlg.set_range(12.0, 24.5)
        seen["text2"] = dlg.summary.text()
        return QDialog.Accepted if ok.isEnabled() else QDialog.Rejected
    monkeypatch.setattr(AudioRangeDialog, "exec", choose)
    win._voice_from_file()
    assert seen["title"] == "Voice of @Tom: choose the part" and seen["first"] == (0.0, 30.0) and not seen["name"]
    assert "clones the voice from all of it" in seen["text"] and not seen["short"]
    assert "Voice 0:12.000 – 0:24.500: 12.50 s." in seen["text2"] and "first 7.5 s" in seen["text2"]
    assert win.char_voice.text() == "Voice: talk.mp4 (7.5 s; cloned from 12.5 s)"
    monkeypatch.setattr(AudioRangeDialog, "exec", lambda dlg: QDialog.Rejected)       # cancelled: nothing changes
    win._voice_from_file()
    assert win.char_voice.text() == "Voice: talk.mp4 (7.5 s; cloned from 12.5 s)"
    # Remove music: the chosen part is cleaned by the text to speech worker (a fake here) and used instead
    import shutil

    class Sep:
        calls, stopped = [], 0

        def problems(self, **_k):
            return []

        def isolate(self, audio, out):
            Sep.calls.append(round(chain.audio_seconds(Path(audio)), 1))
            shutil.copy(audio, out)
            return 1.0

        def stop(self):
            Sep.stopped += 1
            return True
    win._tts_local = Sep()

    def clean(dlg):
        ok = dlg.buttons.button(QDialogButtonBox.Ok)
        seen.update(can=dlg.b_clean.isEnabled(), before=dlg.cleaned, use=dlg.use_clean.isEnabled())
        dlg.set_range(3.0, 38.0)
        dlg.remove_music()
        seen["busy"] = (dlg.clean_busy, ok.isEnabled(), dlg.b_clean.isEnabled())
        dlg.done(0)                                          # (closing waits for it)
        seen["open"] = dlg.clean_busy
        wait(app, lambda: not dlg.clean_busy, timeout=30)
        seen.update(info=dlg.clean_info.text(), checked=dlg.use_clean.isChecked(), file=dlg.cleaned)
        dlg.set_range(4.0, 38.0)                             # another part: the cleaned sound is not of it
        seen.update(stale=dlg.cleaned, info2=dlg.clean_info.text())
        dlg.set_range(3.0, 20.0)
        dlg.remove_music()
        wait(app, lambda: not dlg.clean_busy, timeout=30)
        seen["exists"] = dlg.cleaned is not None and dlg.cleaned.is_file()
        seen["dlg"] = dlg
        return QDialog.Accepted
    monkeypatch.setattr(AudioRangeDialog, "exec", clean)
    win._voice_from_file()
    assert seen["can"] and seen["before"] is None and not seen["use"]
    assert seen["busy"] == (True, False, False) and seen["open"]
    assert "Music removed from 0:03.000 – 0:33.000" in seen["info"] and seen["checked"] and seen["file"] is not None
    assert seen["stale"] is None and "remove the music again" in seen["info2"] and seen["exists"]
    assert Sep.calls == [30.0, 17.0] and Sep.stopped == 1    # (30 s at most; the worker is stopped afterwards)
    assert win.char_voice.text() == "Voice: talk.mp4 (music removed) (7.5 s; cloned from 17.0 s)"
    assert seen.pop("dlg")._tmp is None and not win._tts_busy             # its files are gone
    # unticked: the sound as it is; without the package or model the button says why
    def as_it_is(dlg):
        dlg.set_range(3.0, 20.0)
        dlg.remove_music()
        wait(app, lambda: not dlg.clean_busy, timeout=30)
        dlg.use_clean.setChecked(False)
        return QDialog.Accepted
    monkeypatch.setattr(AudioRangeDialog, "exec", as_it_is)
    win._voice_from_file()
    assert win.char_voice.text() == "Voice: talk.mp4 (7.5 s; cloned from 17.0 s)"
    # From clip: the same window on the selected clip's video
    shutil.copy(film, win.project / "made.mp4")
    node = win.tree.nodes[ids[1]]
    was_made, was_sel = node.made, win.tree.selected
    node.made, win.tree.selected = {**(node.made or {}), "video": "made.mp4"}, ids[1]

    def from_clip(dlg):
        seen.update(file=Path(dlg.file).name, can=dlg.b_clean.isEnabled())
        dlg.set_range(1.0, 11.0)
        if seen["remove"]:
            dlg.remove_music()
            wait(app, lambda: not dlg.clean_busy, timeout=30)
        return QDialog.Accepted
    monkeypatch.setattr(AudioRangeDialog, "exec", from_clip)
    for seen["remove"], name in ((False, f"sound of clip #{ids[1][1:]}"),
                                 (True, f"sound of clip #{ids[1][1:]} (music removed)")):
        win._voice_from_selected()
        assert seen["file"] == "made.mp4" and seen["can"]
        assert win.char_voice.text() == f"Voice: {name} (7.5 s; cloned from 10.0 s)"
    assert Sep.calls[-1] == 10.0 and not win._tts_busy
    node.made, win.tree.selected = was_made, was_sel
    (win.project / "made.mp4").unlink()
    Sep.problems = lambda self, **_k: ["removing music needs the audio-separator package"]
    monkeypatch.setattr(AudioRangeDialog, "exec", lambda dlg: (seen.update(
        can=dlg.b_clean.isEnabled(), tip=dlg.b_clean.toolTip()), QDialog.Rejected)[1])
    win._voice_from_file()
    assert not seen["can"] and seen["tip"].startswith("Not set up: removing music needs")
    win._tts_local = None
    mute = tmp_path / "mute.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=64x48:d=3", str(mute)], check=True)
    monkeypatch.setattr(IP, "pick_image", lambda *a, **k: str(mute))
    win._voice_from_file()
    assert "has no sound" in win.status.text()
    assert win.set_character_voice(str(speech))
    assert win.char_voice.text() == "Voice: tom.wav (7.5 s; cloned from 9.0 s)"
    assert "· voice" in win.cast_list.currentItem().text()
    type_prompt(win, ids[1], '@Tom waves and says "Hello there."')
    flush(app, win)
    card = next(c for c in win.cards if c.nid == ids[1])
    assert card.faces.text() == "Faces: @Tom · Voices: @Tom\nLines: @Tom: \u201cHello there.\u201d" and "<d>[English] Hello there.</d>" in card.edit.toolTip()
    # the overview: who appears in which clip of the path
    strip = win.cast_strip
    rows = [strip.verticalHeaderItem(r).text() for r in range(strip.rowCount())]
    cols = [c.nid for c in win.cards]
    assert strip.isVisibleTo(win) and rows == ["@Person1", "@Tom"] and strip.columnCount() == len(cols)
    assert strip.item(rows.index("@Tom"), cols.index(ids[1])).text() == "●♪"
    assert strip.item(rows.index("@Person1"), cols.index(ids[0])).text() == "○"           # the lead, nobody named
    assert strip.item(rows.index("@Tom"), cols.index(ids[0])).text() == ""
    assert "speaks with its voice" in strip.item(rows.index("@Tom"), cols.index(ids[1])).toolTip()
    strip._clicked(cols.index(ids[1]))                                                   # shows that card
    win.dialogue_lang.setCurrentText("Ger")                                  # typing: nothing changes yet
    assert win.tree.options["identity"]["language"] == "English"
    win.dialogue_lang.setCurrentText("German")
    win.dialogue_lang.lineEdit().editingFinished.emit()
    assert win.tree.options["identity"]["language"] == "German"
    win._remove_voice()
    assert win.char_voice.text() == "Voice: none" and not (tmp_path / "proj" / "refs" / "voice1.wav").exists()
    # removing the lead's face: clips that mention nobody are made without a face
    win.cast_list.setCurrentRow(0)
    win.refs.setCurrentRow(0)
    win._remove_reference()
    flush(app, win)
    assert win.refs.count() == 0 and not (tmp_path / "proj" / "refs" / "ref1.png").exists()
    assert win.cards[0].faces.text() == "Faces: none (the prompt mentions nobody and there is no lead)"


def test_sound_references_in_the_window(app, win, tmp_path, image, monkeypatch):
    import subprocess
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    wav = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=4", str(wav)],
                   check=True)
    assert win.add_sound(str(wav))
    flush(app, win)
    assert win.sounds.count() == 1 and win.sounds.item(0).text().startswith("<Audio 1>   voice.wav   4.0 s")
    assert "<Audio 1> added" in win.status.text() and [win.state(i) for i in ids] == ["done", "done"]
    assert "4.0 of 15 s used, 0 clips use them" in win.sound_info.text() and win.b_sound_del.isEnabled()
    type_prompt(win, ids[1], "b, in the voice of <Audio 2>")
    flush(app, win)
    assert win.state(ids[1]) == "changed"
    win.generate_path()
    wait(app, lambda: not win.running)                                        # (checked in the background)
    assert not win.running and "names <Audio 2>, but there is only 1 sound reference" in win.status.text()
    type_prompt(win, ids[1], "b, in the voice of <Audio 1>")
    flush(app, win)
    assert "1 clip uses them" in win.sound_info.text()
    win.tree.selected = ids[0]
    assert win.b_sound_clip.isEnabled()
    win._sound_from_selected()                          # the test clips are 0.7 s: too short, and it says so
    flush(app, win)
    assert win.sounds.count() == 1 and "needs at least 2 s" in win.status.text()
    win.select(ids[1])
    run_and_wait(app, win, win.generate_path)
    made = T.load(tmp_path / "proj").nodes[ids[1]].made["settings"]
    assert [a["file"] for a in made["audio_refs"]] == ["refs/audio1.wav"]
    wait(app, lambda: not win._joins, timeout=30)
    win.open_project(tmp_path / "proj")                                         # kept with the project
    assert win.sounds.count() == 1 and win.state(ids[1]) == "done"
    win.sounds.setCurrentRow(0)
    win._remove_sound()
    flush(app, win)
    assert win.sounds.count() == 0 and not (tmp_path / "proj" / "refs" / "audio1.wav").exists()
    assert win.state(ids[1]) == "changed" and win.state(ids[0]) == "done"       # only the clip that named it


def test_continue_the_real_videos_sound_on_the_card(app, win, tmp_path):
    """The card's sound choice appears only after a real video; switching it on also carries the picture over."""
    from video_chain.ui.main_window import PromptCard
    src = make_real_video(tmp_path / "real.mp4", secs=8.0)
    win.open_project(tmp_path / "proj")
    first = win.cards[0]
    assert isinstance(first, PromptCard) and not first.src_sound.isVisibleTo(first)   # no real video before it
    assert first.b_speak.isVisibleTo(first)                  # (the row itself stays: Speak the lines)
    v = win.add_video(first.nid, str(src), 0.0, 1.5)
    wait(app, lambda: not win._converts)
    win.node_action("continue", v)
    a = win.selected_id()
    card = next(c for c in win.cards if c.nid == a)
    assert card.sound_box.isVisibleTo(card) and card.src_sound.currentData() is None and not card.guide.currentData()
    card.src_sound.setCurrentIndex(card.src_sound.findData(True))
    flush(app, win)
    ov = win.tree.nodes[a].overrides
    assert ov["source_sound"] is True and ov["guide"] == 22    # the longest carry-over the 36-frame video allows
    card = next(c for c in win.cards if c.nid == a)
    assert card.src_sound_info.text().startswith("Uses 0:") and "real.mp4" in card.src_sound_info.text()
    assert "video's sound" in card.settings_toggle.text()
    assert T.SOURCE_SOUND_TEXT.format(k=1) in card.edit.toolTip()
    card.src_sound.setCurrentIndex(0)                                                   # optional: off again
    flush(app, win)
    assert "source_sound" not in win.tree.nodes[a].overrides
    card = next(c for c in win.cards if c.nid == a)                         # or the sound of the clip before, again
    card.src_sound.setCurrentIndex(card.src_sound.findData("previous"))
    flush(app, win)
    assert win.tree.nodes[a].overrides["source_sound"] == "previous"
    card = next(c for c in win.cards if c.nid == a)
    assert card.src_sound_info.text().startswith(f"Repeats the sound of video #{v[1:]}")
    assert "repeats the sound before" in card.settings_toggle.text()
    win.node_action("continue", a)                    # after a generated clip: repeat it, or go on with the video
    b = win.selected_id()
    card = next(c for c in win.cards if c.nid == b)
    model = card.src_sound.model()
    assert card.sound_box.isVisibleTo(card)
    assert model.item(card.src_sound.findData("previous")).isEnabled()
    assert model.item(card.src_sound.findData(True)).isEnabled()           # (the real video is still before it)


def test_audio_tracks_in_the_window(app, win, tmp_path, image):
    """Import and cut an audio track; a clip plays it, the next one carries it on; renamed / removed."""
    import subprocess
    from PySide6.QtWidgets import QDialogButtonBox
    from video_chain.ui.audio_editor import AudioRangeDialog
    src = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=500:d=12", str(src)], check=True)
    dlg = AudioRangeDialog(None, file=str(src), start=2.0, end=3.0, player=False)
    assert abs(dlg.duration - 12.0) < 0.1 and dlg.start == 2.0 and dlg.name.text() == "song"
    assert not dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()                  # under 2 s
    dlg.set_range(2.0, 9.0)
    assert dlg.buttons.button(QDialogButtonBox.Ok).isEnabled()
    wait(app, lambda: bool(dlg.bar.peaks))                                    # the waveform, made in the background
    dlg.done(0)
    ids = new_project(win, tmp_path, image, ("one", "two"))
    tid = win.add_track(str(src), 2.0, 9.0, "Song")
    assert tid and win.track_list.count() == 1 and "Song" in win.track_list.item(0).text()
    flush(app, win)
    first = next(c for c in win.cards if c.nid == ids[0])
    k = first.src_sound.findData(f"track:{tid}")
    assert k >= 0 and first.sound_box.isVisibleTo(first)                       # offered on every clip
    first.src_sound.setCurrentIndex(k)
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["source_sound"] == f"track:{tid}"
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.src_sound.currentData() == f"track:{tid}" and "from its beginning" in first.src_sound_info.text()
    assert first.sound_over.isVisibleTo(first) and not first.sound_over.isChecked()
    first.sound_over.setChecked(True)                                         # full motion, the sound laid over
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides.get("sound_over") is True
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.sound_over.isChecked() and "laid over" in first.settings_toggle.text()
    assert "<Audio" not in first.edit.toolTip()
    assert "Laid over" in first.src_sound_info.text()
    first.set_editable(False)                                                 # (during a run)
    assert not first.sound_over.isEnabled()
    first.set_editable(True)
    first.src_sound.setCurrentIndex(0)                                        # no sound: the box goes off too
    assert not first.sound_over.isChecked()
    first.src_sound.setCurrentIndex(first.src_sound.findData(f"track:{tid}"))
    flush(app, win)
    assert "sound_over" not in win.tree.nodes[ids[0]].overrides
    second = next(c for c in win.cards if c.nid == ids[1])
    second.src_sound.setCurrentIndex(second.src_sound.findData(f"track:{tid}"))
    flush(app, win)
    second = next(c for c in win.cards if c.nid == ids[1])
    assert f"carries it on from clip #{ids[0][1:]}" in second.src_sound_info.text()
    assert T.TRACK_SOUND_TEXT.format(k=1) in second.edit.toolTip()
    assert "1 clip" not in win.track_list.item(0).text() and "2 clips" in win.track_list.item(0).text()
    old_file = T.track_of(win.tree, tid)["file"]
    assert win.recut_track(tid, str(src), 2.0, 9.0, "Song 2")                # only the name
    assert T.track_of(win.tree, tid)["name"] == "Song 2" and T.track_of(win.tree, tid)["file"] == old_file
    assert win.recut_track(tid, str(src), 1.0, 9.0)                           # another part: the old file goes
    assert not (win.project / old_file).exists() and (win.project / T.track_of(win.tree, tid)["file"]).exists()
    win.track_list.setCurrentRow(0)
    monkey = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        win._remove_track()
    finally:
        QMessageBox.question = monkey
    flush(app, win)
    assert win.track_list.count() == 0
    first = next(c for c in win.cards if c.nid == ids[0])
    assert first.src_sound.currentText() == "Audio track: (removed)" and "was removed" in first.src_sound_info.text()


def test_mix_the_laid_over_sound_with_the_clips_own(app, win, tmp_path, image):
    """'Keep the clip's own sound too' shows only under 'Lay the sound over', stores sound_over "mix" (and a level
    other than 0, an advanced setting), comes back on a rebuilt card, says so in the summary, the sound text and a
    chip; with the project's sound off it cannot be chosen."""
    import subprocess
    src = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=500:d=12", str(src)], check=True)
    ids = new_project(win, tmp_path, image, ("one", "two"))
    tid = win.add_track(str(src), 2.0, 9.0, "Song")
    flush(app, win)
    card = lambda: next(c for c in win.cards if c.nid == ids[0])            # noqa: E731
    card().src_sound.setCurrentIndex(card().src_sound.findData(f"track:{tid}"))
    flush(app, win)
    assert not card().mix_row.isVisibleTo(card())                             # not laid over: no mix choice
    card().sound_over.setChecked(True)
    flush(app, win)
    c = card()
    assert c.mix_row.isVisibleTo(c) and not c.sound_mix.isChecked() and not c.mix_level_box.isVisibleTo(c)
    assert c.sound_mix.isEnabled() and "steps, voices" in c.sound_mix.toolTip()
    assert any(ch.text() == "♪ laid over" for ch in c.chips.chips)
    c.sound_mix.setChecked(True)
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["sound_over"] == "mix"
    assert "sound_mix_level" not in win.tree.nodes[ids[0]].overrides
    c = card()                                                                # rebuilt: still mixed
    assert c.sound_over.isChecked() and c.sound_mix.isChecked() and c.mix_level_box.isVisibleTo(c)
    assert "(mixed with the clip's own sound)" in c.settings_toggle.text() and "Mixed in" in c.src_sound_info.text()
    assert any(ch.text() == "♪ mixed in" for ch in c.chips.chips) and "<Audio" not in c.edit.toolTip()
    assert c.advanced_active() == []
    c.mix_level.setValue(-6)
    flush(app, win)
    assert win.tree.nodes[ids[0]].overrides["sound_mix_level"] == -6
    c = card()
    assert c.mix_level.value() == -6 and "-6 dB" in c.settings_toggle.text()
    assert c.advanced_active() == ["Sound level"]
    win.set_simple(True)
    c = card()
    assert c.sound_mix.isVisibleTo(c) and not c.mix_level_box.isVisibleTo(c) and not c.adv_note.isHidden()
    win.set_simple(False)
    card().sound_over.setChecked(False)                                       # not laid over: the mix goes too
    flush(app, win)
    ov = win.tree.nodes[ids[0]].overrides
    assert "sound_over" not in ov and "sound_mix_level" not in ov and not card().sound_mix.isChecked()
    win.audio.setChecked(False)                                               # the model makes no sound
    card().sound_over.setChecked(True)
    flush(app, win)
    c = card()
    assert c.mix_row.isVisibleTo(c) and not c.sound_mix.isEnabled() and "Generate sound" in c.sound_mix.toolTip()


def test_mixing_a_made_clip_again_needs_no_generation(app, win, tmp_path, image):
    """On a made clip with its sound laid over, switching to the mix (at once) or changing the level (once it
    settles) mixes the sound again in the background from the model's kept sound: no generation, the clip stays
    done, the status line says so; the clip file keeps its name."""
    import subprocess
    src = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=500:d=12", str(src)], check=True)
    ids = new_project(win, tmp_path, image, ("one",))
    tid = win.add_track(str(src), 2.0, 9.0, "Song")
    flush(app, win)
    card = lambda: next(c for c in win.cards if c.nid == ids[0])            # noqa: E731
    card().src_sound.setCurrentIndex(card().src_sound.findData(f"track:{tid}"))
    card().sound_over.setChecked(True)
    flush(app, win)
    run_and_wait(app, win, win.generate_path)
    node = win.tree.nodes[ids[0]]
    video = node.made["video"]
    assert node.made.get("own_sound") and win.state(ids[0]) == "done"
    from video_chain.ui.main_window import MIX_TIP
    assert "no new generation" in MIX_TIP and "made before" in MIX_TIP
    card().sound_mix.setChecked(True)
    assert win._remix_timer.isActive() and win._remix_timer.interval() == 0     # the mode: at once
    wait(app, lambda: "sound mixed again (no new generation)" in win.status.text(), timeout=30)
    assert not win.running and not win._remixing and win.state(ids[0]) == "done"
    node = win.tree.nodes[ids[0]]
    assert node.made["video"] == video and node.made["settings"]["source_sound"]["mix"] is True
    assert node.made["sound_rev"] == 1 and not win.remix_due
    win.status.setText("")
    card().mix_level.setValue(-6)
    assert win._remix_timer.isActive() and win._remix_timer.interval() == 400   # the level: once it settles
    wait(app, lambda: "sound mixed again" in win.status.text(), timeout=30)
    node = win.tree.nodes[ids[0]]
    assert node.made["settings"]["source_sound"]["mix_level"] == -6 and node.made["sound_rev"] == 2
    flush(app, win)
    saved = T.load(win.project).nodes[ids[0]].made
    assert saved["sound_rev"] == 2 and saved["video"] == video
    assert T.Checker(win.project, T.load(win.project)).status(ids[0]) == "done"


def test_text_to_speech_in_the_app(app, win, tmp_path, image, monkeypatch):
    from video_chain.ui import speak_dialog as SPD
    from video_chain.ui import tts_dialog as TD
    """Speak the lines on a card (a fake engine) makes them the clip's sound; a voice made in the TTS window becomes
    the character's voice; the provider and the ElevenLabs key are kept (the key in the keyring only)."""
    import subprocess

    from video_chain import tree as T
    from video_chain import tts as TTS
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    def tone(path, secs):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}",
                        str(path)], check=True)

    class Fake:
        name = "local"
        stopped = 0

        def problems(self, **_k):
            return []

        def transcribe(self, audio, lang=None):
            return "sample words"

        def clone(self, text, lang, ref, ref_text, out, seed=None, temperature=None):
            tone(out, 1.5)

        def design(self, text, lang, instruct, out, seed=None):
            tone(out, 4.0)

        def stop(self):
            Fake.stopped += 1
            return True
    fake = Fake()
    ids = new_project(win, tmp_path, image, ("@Anna waves.",))
    nid = ids[0]
    T.add_character(win.project, win.tree, "Anna", desc="young woman")
    win.refresh(rebuild_path=True)
    card = next(c for c in win.cards if c.nid == nid)
    assert card.b_speak.isVisibleTo(card) and not card.b_speak.isEnabled()   # always there; no spoken line yet
    assert "in quotes first" in card.b_speak.toolTip()
    type_prompt(win, nid, '@Anna waves and says "Hallo, schön dich zu sehen."')
    assert card.b_speak.isEnabled()                          # as soon as a line is typed
    win.refresh(rebuild_path=True)
    card = next(c for c in win.cards if c.nid == nid)
    assert card.b_speak.isVisibleTo(card) and card.b_speak.isEnabled()
    monkeypatch.setattr(win, "_tts_engine", lambda: fake)
    win.node_action("speak", nid)
    assert "no voice sample" in win.status.text()            # Anna has no voice yet
    # a voice from the TTS window (its sample made by the fake engine)
    win._show_references(select="Anna")
    assert win._selected_character()["name"] == "Anna" and win.b_voice_tts.isEnabled()
    win._tts_local = fake                                    # the app's worker (local_tts gives it its room check)

    def accept(dlg):
        dlg.desc.setPlainText("warm young voice")
        dlg._make()
        wait(app, lambda: dlg.made is not None or "Could not" in dlg.info.text(), timeout=30)
        return dlg.made is not None
    monkeypatch.setattr(TD, "run_dialog", accept)
    win._voice_with_tts()
    c = T.character(win.tree, "Anna")
    assert c["voice"]["name"].startswith("designed: warm young voice") and c["voice"]["text"]
    assert "set with text to speech" in win.status.text() and Fake.stopped >= 1
    assert win.played and win.played[-1].name == "sample.wav"           # the sample was played
    # now the lines
    assert win.speak_clip(nid)
    wait(app, lambda: not win._tts_busy, timeout=30)
    node = win.tree.nodes[nid]
    tr = T.track_of(win.tree, T.track_id(node))
    assert tr and tr["tts"]["clip"] == nid and "spoken" in win.status.text()
    assert T.load(win.project).nodes[nid].overrides["source_sound"] == T.TRACK_PREFIX + tr["id"]   # saved
    # the Speak the lines window: takes to choose from, how they are spoken; the one used becomes the sound
    seen = {}

    def choose(dlg):
        seen.update(lines=dlg.line_texts(), start=dlg.speak_values(), use=dlg.b_use.isEnabled())
        dlg.n.setValue(2)
        dlg.delivery.setCurrentIndex(dlg.delivery.findData("lively"))
        dlg.speed.setValue(0.8)
        dlg.make()
        seen["busy"] = (dlg.busy, dlg.b_make.isEnabled(), dlg.b_make.text())      # (the button stops the round)
        dlg.reject()                                         # (while takes are made: it would close afterwards)
        dlg._close_when_done = False
        wait(app, lambda: not dlg.busy, timeout=30)
        dlg.n.setValue(1)
        dlg.line_edits[0].setText("")
        seen["empty"] = dlg.b_make.isEnabled()               # an empty line cannot be spoken
        dlg.line_edits[0].setText('Hallo, "Welt"!')          # other words: tried, and used
        dlg.make()
        wait(app, lambda: not dlg.busy, timeout=30)
        seen.update(rows=[dlg.list.item(k).text() for k in range(dlg.list.count())], info=dlg.info.text(),
                    files=[win.project / r["file"] for r in dlg.takes])
        dlg.list.setCurrentRow(dlg.list.count() - 1)
        dlg.accept()
        return dlg.chosen is not None
    monkeypatch.setattr(SPD, "run_dialog", choose)
    old_track = T.track_id(win.tree.nodes[nid])
    win.node_action("speak", nid)
    assert seen["lines"] == ["Hallo, schön dich zu sehen."] and not seen["use"] and not seen["empty"]
    assert seen["rows"][-1].endswith("lively, 0.8 ×   ·   “Hallo, 'Welt'!”") and "“" not in seen["rows"][0]
    assert win.tree.nodes[nid].prompt == """@Anna waves and says "Hallo, 'Welt'!\"""" # the take's words: the prompt's
    assert T.load(win.project).nodes[nid].prompt == win.tree.nodes[nid].prompt and tr_lines(win, nid) == ["Hallo, 'Welt'!"]
    assert next(c for c in win.cards if c.nid == nid).text() == win.tree.nodes[nid].prompt
    assert seen["start"] == TTS.SPEAK_DEFAULTS and seen["busy"] == (True, True, "Stop")
    assert 1 <= len(seen["rows"]) <= 3 and seen["rows"][0].startswith("Take 1   ·   ")
    assert [f.is_file() for f in seen["files"]] == [False] * (len(seen["files"]) - 1) + [True]   # the others went
    assert all((win.project / chain.OLD_DIR / "refs" / f.name).is_file() for f in seen["files"][:-1])   # ... to old/
    tr = T.track_of(win.tree, T.track_id(win.tree.nodes[nid]))
    assert tr["id"] != old_track and win.project / tr["file"] == seen["files"][-1] and "spoken" in win.status.text()
    assert T.load(win.project).options["tts"]["speak"] == {"takes": 1, "delivery": "lively", "speed": 0.8}
    assert win.played[-1] == seen["files"][0] and not win._tts_busy
    monkeypatch.setattr(SPD, "run_dialog", lambda dlg: (seen.update(again=dlg.speak_values()), False)[1])
    win.node_action("speak", nid)                            # cancelled: nothing changes; the settings were kept
    assert seen["again"]["delivery"] == "lively" and T.track_id(win.tree.nodes[nid]) == tr["id"]
    # a frozen clip's lines are not spoken
    win.tree.nodes[nid].frozen = True
    win.node_action("speak", nid)
    assert "is frozen" in win.status.text()
    win.tree.nodes[nid].frozen = False
    # provider + key
    win.tts_provider.setCurrentIndex(win.tts_provider.findData("elevenlabs"))
    assert TTS.options(win.tree)["provider"] == "elevenlabs" and "API key" in win.status.text()
    assert TTS.speak_options(win.tree)["delivery"] == "lively"               # (kept when the provider changes)
    assert win._tts_form.isRowVisible(win._eleven_row)
    win.eleven_key.setText(" sk_abc ")
    win._save_eleven_key()
    assert TTS.eleven_key() == "sk_abc" and win.eleven_key.text() == "" and "kept" in win.eleven_key.placeholderText()
    win.eleven_key.setText("")
    win._save_eleven_key()
    assert TTS.eleven_key() == ""


def test_clone_a_voice_at_elevenlabs(app, win, tmp_path, image, monkeypatch):
    """Clone uploads the character's voice sample (a fake ElevenLabs) and makes the new voice its ElevenLabs voice;
    nothing is uploaded without a key or when the question is answered with No."""
    import subprocess

    from video_chain import tree as T
    from video_chain import tts as TTS
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    calls = []

    class Fake:
        fail = ""

        def clone(self, name, sample, description=""):
            calls.append((name, Path(sample).name, Path(sample).is_file()))
            if Fake.fail:
                raise TTS.TtsError(Fake.fail)
            return f"clone{len(calls)}"
    monkeypatch.setattr(TTS, "ElevenLabs", Fake)
    new_project(win, tmp_path, image, ('@Anna says "Hallo."',))
    T.add_character(win.project, win.tree, "Anna")
    win._show_references(select="Anna")
    assert not win.b_voice_clone.isEnabled()                 # no voice sample yet
    speech = tmp_path / "anna.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(speech)],
                   check=True)
    assert win.set_character_voice(str(speech)) and win.b_voice_clone.isEnabled()
    asked = []
    answer = [QMessageBox.No]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[2]), answer[0])[1])
    win._clone_voice_at_eleven()
    assert "API key" in win.status.text() and not asked and not calls      # no key: nothing asked or sent
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    win._clone_voice_at_eleven()
    assert "leaves this computer" in asked[-1] and "consent" in asked[-1] and not calls and not win._tts_busy
    answer[0] = QMessageBox.Yes
    Fake.fail = "instant voice cloning needs a paid plan"
    win._clone_voice_at_eleven()
    wait(app, lambda: not win._tts_busy, timeout=30)
    assert "Could not clone" in win.status.text() and "paid plan" in win.status.text()
    assert "tts" not in T.character(win.tree, "Anna")
    Fake.fail = ""
    win._eleven_voices = [{"id": "x", "name": "old list"}]
    win._clone_voice_at_eleven()
    wait(app, lambda: not win._tts_busy, timeout=30)
    assert calls[-1] == ("Anna (video-chain)", "voice1.wav", True)
    assert T.character(win.tree, "Anna")["tts"]["eleven"] == {"id": "clone2", "name": "Anna (video-chain)"}
    assert T.character(T.load(win.project), "Anna")["tts"]["eleven"]["id"] == "clone2"          # saved
    assert win.char_voice.text() == "Voice: anna.wav (3.0 s) · ElevenLabs: Anna (video-chain)"
    assert "cloned from its voice sample" in win.status.text() and "provider" in win.status.text()
    assert win._eleven_voices is None                        # (the account's voices are asked again)
    assert not TTS.line_problems(win.tree, next(iter(win.tree.nodes.values())), "elevenlabs")
    # again: the question names the voice it replaces
    win._clone_voice_at_eleven()
    wait(app, lambda: not win._tts_busy, timeout=30)
    assert "Anna (video-chain)" in asked[-1] and "replaces it" in asked[-1]
    assert T.character(win.tree, "Anna")["tts"]["eleven"]["id"] == "clone3"


def test_cast_kinds_in_the_window(app, win, tmp_path, image, monkeypatch):
    """New ▸ Place…: a place gets whole pictures (no crop window), no voice controls, no lead; its kind can be
    changed, and the core's refusals are shown in the status line."""
    new_project(win, tmp_path, image, ("a",))
    menu = win.b_char_new.menu()
    assert [a.text() for a in menu.actions()] == ["Person…", "Place…", "Object…", "Style…"]
    asked = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (asked.append(a[1]), ("Kitchen", True))[1])
    menu.actions()[1].trigger()
    assert asked == ["New place"] and "@Kitchen added (place)" in win.status.text()
    c = T.character(win.tree, "Kitchen")
    assert T.kind_of(c) == "place" and c["lead"] is False and win.char_kind.currentData() == "place"
    assert win._voice_row.isHidden() and win.char_lead.isHidden()
    assert win.ref_add_label.text() == "Add picture:" and "whole picture" in win.refs.toolTip()
    assert win.b_ref_crop.text() == "Whole picture"

    def no_crop(*_a, **_k):
        raise AssertionError("no crop window for a place")
    monkeypatch.setattr(win, "crop_face", no_crop)
    room = tmp_path / "room.png"
    Image.new("RGB", (120, 90), (90, 90, 200)).save(room)
    assert win.add_reference(str(room))
    rel = T.character(win.tree, "Kitchen")["images"][0]
    assert Image.open(win.project / rel).size == (120, 90)                         # the whole picture
    assert win.status.text().startswith("Picture added to @Kitchen")
    assert win.cast_list.item(0).text() == "@Kitchen   1 picture   · place"
    monkeypatch.delattr(win, "crop_face")
    # a person again: the voice controls come back
    win.char_kind.setCurrentIndex(win.char_kind.findData("person"))
    win._character_kind_changed()
    assert T.kind_of(T.character(win.tree, "Kitchen")) == "person" and not win._voice_row.isHidden()
    assert win.ref_add_label.text() == "Add face:" and win.b_ref_crop.text() == "Crop…"
    import subprocess
    speech = tmp_path / "v.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
                    str(speech)], check=True)
    T.set_voice(win.project, win.tree, "Kitchen", speech)
    win.char_kind.setCurrentIndex(win.char_kind.findData("object"))
    win._character_kind_changed()                                                  # refused: it has a voice
    assert "Could not change @Kitchen" in win.status.text() and "has a voice" in win.status.text()
    assert T.kind_of(T.character(win.tree, "Kitchen")) == "person" and win.char_kind.currentData() == "person"
    assert T.load(win.project).options["cast"][0].get("kind") is None