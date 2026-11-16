"""Voices, sounds and audio tracks."""
from __future__ import annotations

import logging
import copy
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QListWidgetItem

from .workers import run_in_background
from .. import chain, tree as T, tts as TTS
from . import dialogs as D, image_picker as IP
from .ui_logic import AUDIO_REF_RANGE, plural
from .mw_host import MainWindowHost

if TYPE_CHECKING:
    from .main_window import MainWindow

log = logging.getLogger(__name__)


class AudioMixin(MainWindowHost):
    """Voices, sounds and audio tracks."""

    # ---- a character's voice
    def _clone_voice_at_eleven(self):
        """Make an instant ElevenLabs voice clone of the selected character's voice sample (in a thread);
        _eleven_cloned makes it the character's ElevenLabs voice."""
        c = self._selected_character()
        if c is None or not c.get("voice") or not self._cast_edit_ok():
            return
        if self._tts_busy:
            self.status.setText("Text to speech is busy; wait until it is done.")
            return
        if not TTS.eleven_key():
            self.status.setText("ElevenLabs needs your API key: Preferences → ElevenLabs API key.")
            return
        name, old = c["name"], (c.get("tts") or {}).get("eleven")
        text = (f"Upload the voice sample of @{name} ({T.clone_secs(c['voice']):.1f} s) to your ElevenLabs "
                "account and make an instant voice clone of it?\n\nThe sample leaves this computer and the new voice "
                "takes a voice slot of your plan (instant voice cloning needs a paid plan). Use a voice only with "
                "the speaker's consent.")
        if old:
            text += (f"\n\n@{name} speaks with the ElevenLabs voice “{old['name']}” now; the clone replaces it for "
                     "this character (the old voice stays in your account).")
        if not D.confirm(self, "Clone the voice at ElevenLabs", text):
            return
        assert self.tree is not None and self.project is not None
        snap, project = copy.deepcopy(self.tree), self.project
        with self._tts_session():
            self._update_buttons()
            self.status.setText(f"Cloning the voice of @{name} at ElevenLabs…")

        run_in_background(lambda: TTS.clone_sample(project, snap, name, TTS.ElevenLabs()),
                          on_done=lambda v: self._eleven_cloned(name, v, "", str(project)),      # (the answer ends "busy")
                          on_error=lambda e: self._eleven_cloned(name, None, str(e) or type(e).__name__, str(project)),
                          owner=self)

    def _eleven_cloned(self, name: str, voice, err: str, project: str):
        self._set_tts_busy(False)
        self._update_buttons()
        if voice is None:
            self._report_error(f"Could not clone the voice of @{name} at ElevenLabs: {err}")
            return
        self._eleven_voices = None                  # (the account has one more voice now)
        if (self.project is None or str(self.project) != project or self.tree is None
                or not TTS.set_eleven(self.tree, name, voice)):
            self.status.setText(f"The voice “{voice['name']}” was made in your ElevenLabs account, but @{name} or its "
                                "voice sample changed meanwhile, so it is not used (TTS… can pick it).")
            return
        self._show_character()
        self._write_tree()
        msg = f"@{name} has the ElevenLabs voice “{voice['name']}” now, cloned from its voice sample."
        if TTS.options(self.tree)["provider"] != "elevenlabs":
            msg += " Its lines are spoken with it when ElevenLabs is the text to speech provider."
        self._say(msg)

    def _voice_from_file(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        f = IP.pick_image(self, f"Voice of @{c['name']} (one speaker, 2-{T.VOICE_MAX_SECS:g} s are used)", self.qs,
                       places=self._picker_places(), collections=self.project_media("media"), kind="media")
        if f:
            self._choose_voice_part(f)

    def _choose_voice_part(self, f: str, label: str | None = None) -> None:
        """The voice of the selected character from a part of the file f, chosen in the voice window (with or
        without its background music). label: what the voice is called (else the file's name)."""
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        with self._tts_hold() as mine:               # (the worker that removes music is the text to speech one)
            dlg = self._voice_dialog(f, c["name"], isolate=mine)
            try:
                if not dlg.file:
                    self._report_error(f"Could not set the voice: {dlg.info_label.text()}")
                    return
                if dlg.exec() == QDialog.DialogCode.Accepted and dlg.file:
                    name = label if label and dlg.file == str(f) else None     # (another file chosen in the window)
                    if dlg.cleaned is not None:
                        self.set_character_voice(str(dlg.cleaned),
                                                 label=f"{name or Path(dlg.file).name} (music removed)")
                    else:
                        self.set_character_voice(dlg.file, label=name, start=dlg.start, end=dlg.end)
            finally:
                dlg.cleanup()
                dlg.deleteLater()
                if mine:
                    self.stop_tts("the voice window was closed")

    def _voice_dialog(self, file: str, name: str, isolate: bool = True):
        """The window the part of a file is chosen in that becomes a character's voice (isolate: it can take the
        background music out with the app's text to speech worker)."""
        from .audio_editor import AudioRangeDialog      # (lazy: a rarely opened editor with its own heavy parts)
        fn, why = None, "Text to speech is busy; choose the voice again when it is done to remove music."
        if isolate:
            w = self.local_tts()
            bad = w.problems(isolate=True)
            fn, why = (None, "Not set up: " + "; ".join(bad)) if bad else (w.isolate, "")
        return AudioRangeDialog(self, file=file, settings=self.qs, title=f"Voice of @{name}: choose the part",
                                voice=True, player=self.audio_player,
                                isolate=fn, no_isolate=why)

    def _voice_from_selected(self):
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if self.tree and nid else None
        if nid is None or node is None or not (node.made or {}).get("video"):
            self.status.setText("Select a made clip or a converted video first.")
            return
        assert self.project is not None and node.made is not None
        what = f"{'video' if node.is_video else 'clip'} #{T.short_id(nid)}"
        self._choose_voice_part(str(self.project / node.made["video"]), label=f"sound of {what}")

    def set_character_voice(self, path: str, label: str | None = None, start: float = 0.0,
                            end: float | None = None) -> bool:
        """The voice of the selected character, from an audio or video file (the part from start to end)."""
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return False
        before = self._done_clips()
        try:
            assert self.tree is not None
            assert self.project is not None
            entry = T.set_voice(self.project, self.tree, c["name"], Path(path), start, label=label, end=end)
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not set the voice: {e}")
            return False
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        msg = (f"Voice of @{c['name']} set ({entry['secs']:.1f} s"
               + (f", the first {T.VOICE_MAX_SECS:g} s" if entry.get("cut") else "")
               + f"). Clips use it where @{c['name']} says a line in quotes."
               + (f" Text to speech clones it from {entry['ref']['secs']:.1f} s." if entry.get("ref") else ""))
        self._say(msg)
        if self._report_remade(before, f"Voice of @{c['name']} set"):
            self.status.setText(msg + " " + self.status.text())
        return True

    def _remove_voice(self):
        c = self._selected_character()
        if c is None or not c.get("voice") or not self._cast_edit_ok():
            return
        if not D.confirm(self, "Remove voice", f"Remove the voice of @{c['name']}? Its sample (a recording "
                                "or a clone) is taken out of the project."):
            return
        before = self._done_clips()
        try:
            assert self.tree is not None
            assert self.project is not None
            T.remove_voice(self.project, self.tree, c["name"])
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not remove the voice: {e}")
            return
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        if not self._report_remade(before, f"Voice of @{c['name']} removed"):
            self._say(f"Voice of @{c['name']} removed.")

    def open_voice_library(self):
        """The voice library window (for the selected character). Without modal dialogs it is shown and handed
        back."""
        from .voice_library import VoiceLibraryDialog
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return None
        dlg = VoiceLibraryDialog(cast("MainWindow", self))
        if not self.modal_dialogs:
            dlg.show()
            return dlg
        dlg.exec()
        dlg.deleteLater()
        return None

    def use_library_voice(self, vid: str, lib=None) -> bool:
        """A voice of the voice library becomes the selected character's voice (copied into the project)."""
        from .. import voice_library as VL
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return False
        before = self._done_clips()
        try:
            assert self.tree is not None and self.project is not None
            entry = (lib or VL.library()).use(vid, self.project, self.tree, c["name"])
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not use the voice: {e}")
            return False
        self._show_references(select=c["name"])
        self._write_tree()
        self._changed()
        msg = f"@{c['name']} has the voice “{entry['name']}” from the voice library now ({entry['secs']:.1f} s)."
        self._say(msg)
        if self._report_remade(before, f"Voice of @{c['name']} set"):
            self.status.setText(msg + " " + self.status.text())
        return True

    def _play_voice(self):
        c = self._selected_character()
        if c is not None and c.get("voice") and self.project is not None:
            self.preview.open(self.project / c["voice"]["file"], f"Voice of @{c['name']}")

    # ---- sound references (<Audio 1> ...)
    def _sound_users(self, upto: int = 1) -> list[str]:
        """Clips whose prompt names <Audio upto> or a later one (they get the sounds up to that one)."""
        if self.tree is None:
            return []
        return [n.id for n in self.tree.nodes.values() if not n.is_video
                and T.audio_mentioned(T.full_prompt(self.tree.options, n.prompt)) >= upto]

    def _add_sound(self):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        f = IP.pick_image(self, f"Sound reference (a voice, a soundtrack, a beat: {AUDIO_REF_RANGE})", self.qs,
                       places=self._picker_places(), collections=self.project_media("media"), kind="media")
        if f:
            self.add_sound(f)

    def _sound_from_selected(self):
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if self.tree and nid else None
        if nid is None or node is None or not (node.made or {}).get("video"):
            self.status.setText("Select a made clip or a converted video first.")
            return
        assert self.project is not None and node.made is not None
        what = f"{'Video' if node.is_video else 'Clip'} #{T.short_id(nid)}"
        self.add_sound(str(self.project / node.made["video"]), name=f"sound of {what.lower()}")

    def add_sound(self, path: str, name: str | None = None) -> bool:
        if self.project is None or self.tree is None or self.running:
            return False
        before = self._done_clips()
        try:
            entry = T.add_audio_reference(self.project, Path(path), self.tree, name=name)
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not add the sound: {e}")
            return False
        n = len(T.audio_refs(self.tree))
        self._show_sounds(select=n - 1)
        self._write_tree()
        self._changed()
        msg = (f"Sound reference <Audio {n}> added ({entry['secs']:.1f} s"
               + (f", the first part only: all sounds together may have {T.AUDIO_REF_TOTAL:g} s" if entry.get("cut")
                  else "")
               + f"). Name it in a clip's prompt as <Audio {n}>.")
        self._say(msg)
        if self._report_remade(before, "Sound reference added"):
            self.status.setText(msg + " " + self.status.text())
        return True

    def _remove_sound(self):
        row = self.sounds.currentRow()
        refs = T.audio_refs(self.tree) if self.tree else []
        if row < 0 or row >= len(refs) or self.project is None or self.running:
            return
        users = self._sound_users(row + 1)
        later = row + 1 < len(refs)
        if users:
            text = (f"Remove <Audio {row + 1}>? {len(users)} clip prompt{'s name' if len(users) != 1 else ' names'} "
                    f"it or a later sound; made ones count as CHANGED.")
            if later:
                text += (f" The sounds after it move up (<Audio {row + 2}> becomes <Audio {row + 1}>), so check "
                         "those prompts.")
            if not D.confirm(self, "Remove sound reference", text):
                return
        before = self._done_clips()
        try:
            assert self.tree is not None
            T.remove_audio_reference(self.project, refs[row]["file"], self.tree)
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not remove the sound reference: {e}")
            return
        self._show_sounds()
        self._write_tree()
        self._changed()
        self._report_remade(before, "Sound reference removed")

    def _play_sound(self):
        row = self.sounds.currentRow()
        refs = T.audio_refs(self.tree) if self.tree else []
        if 0 <= row < len(refs) and self.project is not None:
            self.preview.open(self.project / refs[row]["file"], f"<Audio {row + 1}>: {refs[row].get('name', '')}")

    # ---- audio tracks (imported sound the clips play, carried on from clip to clip)
    def _track_dialog(self, file: str = "", start=None, end=None, name: str = "", title: str = "Audio track"):
        from .audio_editor import AudioRangeDialog      # (lazy: a rarely opened editor with its own heavy parts)
        return AudioRangeDialog(self, file=file, start=start, end=end, name=name, settings=self.qs, title=title,
                                player=self.audio_player)

    def import_track(self, file: str = "", name: str = "") -> str | None:
        """Choose a sound file, cut it, add it as an audio track. Returns its id."""
        if self.project is None or self.tree is None or self.running:
            return None
        dlg = self._track_dialog(file, name=name, title="Import an audio track")
        if not file:
            dlg.choose()
            if not dlg.file:
                dlg.deleteLater()
                return None
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.file:
                return None
            return self.add_track(dlg.file, dlg.start, dlg.end, dlg.name.text().strip())
        finally:
            dlg.deleteLater()

    def add_track(self, file: str, start: float, end: float | None, name: str = "") -> str | None:
        if self.project is None or self.tree is None or self.running:
            return None
        try:
            entry = T.add_track(self.project, self.tree, Path(file), start, end, name or None)
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not import the sound: {e}")
            return None
        self._show_tracks(select=entry["id"])
        self._write_tree()
        self._changed()
        self._say(f"Audio track “{entry['name']}” added ({entry['secs']:.1f} s). Choose it in a clip's "
                            "Sound setting; the clips after it that play it too carry it on.")
        return entry["id"]

    def _track_from_selected(self):
        node = self.tree.nodes.get(self.target_id()) if self.tree else None
        if node is None or not (node.made or {}).get("video"):
            self.status.setText("Select a made clip or a converted video first.")
            return
        assert self.project is not None and node.made is not None
        what = f"{'video' if node.is_video else 'clip'} #{T.short_id(node.id)}"
        self.import_track(str(self.project / node.made["video"]), name=f"sound of {what}")

    def _selected_track(self) -> T.Track | None:
        item = self.track_list.currentItem()
        return T.track_of(self.tree, item.data(Qt.ItemDataRole.UserRole)) if item is not None and self.tree else None

    def _recut_track(self):
        tr = self._selected_track()
        if tr is None or self.project is None or self.running:
            return
        src = tr.get("source") or ""
        if src and Path(src).is_file():
            dlg = self._track_dialog(src, tr.get("start"), tr.get("end"), tr["name"], title="Cut the audio track")
        else:                                           # the original is gone: cut the track's own sound
            dlg = self._track_dialog(str(self.project / tr["file"]), 0.0, float(tr["secs"]), tr["name"],
                                     title="Cut the audio track")
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.file:
                self.recut_track(tr["id"], dlg.file, dlg.start, dlg.end, dlg.name.text().strip())
        finally:
            dlg.deleteLater()

    def recut_track(self, tid: str, file: str, start: float, end: float | None, name: str = "") -> bool:
        if self.project is None or self.tree is None or self.running:
            return False
        before = self._done_clips()
        tr = T.track_of(self.tree, tid)
        if tr is None:
            return False

        def near(a, b) -> bool:                        # (the dialog rounds to ms and clamps to the file's length)
            return b is not None and abs(float(a) - float(b)) < 0.02
        own = Path(file) == self.project / tr["file"]   # its original was gone: the track's own sound was shown
        same = end is not None and ((str(file) == tr.get("source") and near(start, tr.get("start"))
                                     and near(end, tr.get("end")))
                                    or (own and near(start, 0.0) and near(end, tr["secs"])))
        if same:                                       # only the name: the sound and the clips stay as they are
            tr["name"] = name or tr["name"]
        else:
            try:
                T.recut_track(self.project, self.tree, tid, Path(file), start, end, name or None)
            except (ValueError, OSError, RuntimeError) as e:
                self._report_error(f"Could not cut the audio track: {e}")
                return False
        self._show_tracks(select=tid)
        if self._write_tree():
            self._drop_unused_tracks()
        self._changed()
        self.status.setText(f"Audio track “{(T.track_of(self.tree, tid) or {}).get('name')}” "
                            + ("renamed." if same else "cut again."))
        self._report_remade(before, "Audio track cut again")
        return True

    def _remove_track(self):
        tr = self._selected_track()
        if tr is None or self.project is None or self.running:
            return
        users = self._track_users_everywhere(tr["id"])
        if users and not D.confirm(
                self, "Remove audio track",
                f"Remove “{tr['name']}”? {users} clip{'s play' if users != 1 else ' plays'} it"
                + (" (in all scenes)" if self.film is not None and len(self.film.scenes) > 1 else "")
                + "; set another sound for them before they are made again."):
            return
        try:
            assert self.tree is not None
            T.remove_track(self.tree, tr["id"])
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not remove the audio track: {e}")
            return
        self._show_tracks()
        if self._write_tree():
            self._drop_unused_tracks()
        self._changed()

    def _scene_dirs(self) -> list[Path]:
        if self.film is not None:
            return [self.film.path(sc.id) for sc in self.film.scenes]
        return [self.project] if self.project is not None else []

    def _track_users_everywhere(self, tid: str) -> int:
        """Clips that play a track in the open scene (as it is now) and in the film's other scenes (as saved)."""
        assert self.tree is not None
        n = len(T.track_users(self.tree, tid))
        for d in self._scene_dirs():
            assert self.project is not None
            if Path(d) == Path(self.project):
                continue
            try:
                n += len(T.track_users(T.load(Path(d)), tid))
            except Exception:  # noqa: BLE001  (a scene that cannot be read: not counted)
                log.warning("the tracks of the scene %s could not be counted", d, exc_info=True)
        return n

    def _drop_unused_tracks(self):
        """Cut track files no scene lists any more go to old/ (never deleted; after the project is saved: never one
        still in use)."""
        if self.project is None:
            return
        for f in T.unused_track_files(self.project, self._scene_dirs(), self.tree):
            try:
                chain.keep_aside(self.project, f)
            except OSError as e:
                self._report_error(f"Could not put {Path(f).name} aside: {e}")

    def _play_track(self):
        tr = self._selected_track()
        if tr is not None and self.project is not None:
            self.preview.open(self.project / tr["file"], f"Audio track: {tr['name']}")

    def _show_tracks(self, select: str | None = None):
        cur = select or (self.track_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.track_list.currentItem() else None)
        tree = self.tree
        trs = T.tracks(tree) if tree is not None else []
        sig = ([(tr["id"], tr["name"], tr["file"], len(T.track_users(tree, tr["id"]))) for tr in trs], cur) \
            if tree is not None else ([], cur)
        if sig == self._tracks_shown and select is None:
            self._update_track_buttons()
            return
        self._tracks_shown = sig
        self.track_list.blockSignals(True)
        try:
            self._fill_tracks(trs, cur)
        finally:
            self.track_list.blockSignals(False)
        self.track_info.setText(
            "Music, speech or a background for the film: import a sound file (or the sound of a video) and cut it. "
            "A clip whose Sound is “Audio track: …” plays it exactly; the clips after it that play the same track "
            "carry it on where the one before stopped (the picture is made to match, lips in sync)."
            if not trs else
            "A clip plays a track via its Sound setting; the clips after it with the same track carry it on. "
            "Shared by all scenes of the film.")
        self._update_track_buttons()

    def _fill_tracks(self, trs: list[T.Track], cur: str | None):
        self.track_list.clear()
        for tr in trs:
            assert self.tree is not None
            users = len(T.track_users(self.tree, tr["id"]))
            item = QListWidgetItem(f"{tr['name']}   {float(tr['secs']):.1f} s"
                                   + (f"   · {plural(users, 'clip')}" if users else ""))
            item.setData(Qt.ItemDataRole.UserRole, tr["id"])
            item.setToolTip(f"{tr.get('source') or tr['file']}\n{float(tr.get('start') or 0):.2f}–"
                            f"{float(tr.get('end') or 0):.2f} s of the file")
            self.track_list.addItem(item)
            if tr["id"] == cur:
                self.track_list.setCurrentItem(item)
        if trs and self.track_list.currentRow() < 0:
            self.track_list.setCurrentRow(0)

    def _update_track_buttons(self):
        idle = self.project is not None and self.tree is not None and not self.running
        sel = self._selected_track() is not None if self.tree else False
        self.b_track_add.setEnabled(idle)
        node = self.tree.nodes.get(self.target_id()) if self.tree else None
        has = node is not None and bool((node.made or {}).get("video"))
        self.b_track_clip.setEnabled(idle and has)
        self.b_track_clip.setToolTip(f"Import the sound of {'video' if node.is_video else 'clip'} "
                                     f"#{T.short_id(node.id)} "
                                     "and cut it" if node is not None and has else
                                     "Select a made clip or a converted video")
        for b in (self.b_track_cut, self.b_track_del):
            b.setEnabled(idle and sel)
        self.b_track_play.setEnabled(sel)

    def _show_sounds(self, select: int | None = None):
        self.sounds.clear()
        refs = T.audio_refs(self.tree) if self.tree else []
        for k, a in enumerate(refs, start=1):
            item = QListWidgetItem(f"<Audio {k}>   {a.get('name') or Path(a['file']).name}   "
                                   f"{float(a.get('secs') or 0):.1f} s")
            item.setToolTip(a["file"])
            self.sounds.addItem(item)
        if refs:
            self.sounds.setCurrentRow(select if select is not None and 0 <= select < len(refs) else 0)
        self._show_sound_info()
        self._update_sound_buttons()

    def _show_sound_info(self):
        refs = T.audio_refs(self.tree) if self.tree else []
        total = sum(float(a.get("secs") or 0) for a in refs)
        if not refs:
            text = (f"A voice, music or a beat for clips: add a sound ({AUDIO_REF_RANGE}), then name it in a clip's "
                    "prompt, e.g. \"<Audio 1> is the complete soundtrack\". Only clips that name a sound use it "
                    "(with the reference model).")
        else:
            users = self._sound_users(1)
            text = (f"{total:.1f} of {T.AUDIO_REF_TOTAL:g} s used, {len(users)} "
                    f"clip{'s use' if len(users) != 1 else ' uses'} "
                    "them. \"<Audio 1> is the complete soundtrack\" copies it. A voice only works in the long "
                    "reference format (\"<Audio 1> is the voice-timbre reference for <Subject 1> (S1)\" …), not as a "
                    "plain \"in the voice of <Audio 1>\".")
        self.sound_info.setText(text)

    def _update_sound_buttons(self):
        refs = T.audio_refs(self.tree) if self.tree else []
        idle = self.project is not None and not self.running
        total = sum(float(a.get("secs") or 0) for a in refs)
        room = len(refs) < T.MAX_AUDIO_REFS and total <= T.AUDIO_REF_TOTAL - T.AUDIO_REF_MIN
        self.b_sound_add.setEnabled(idle and room)
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if self.tree and nid else None
        has = node is not None and bool((node.made or {}).get("video"))
        self.b_sound_clip.setEnabled(idle and room and has)
        self.b_sound_clip.setToolTip(
            (f"Add the sound of {'video' if node.is_video else 'clip'} #{T.short_id(nid)} (its first seconds), e.g. "
             "to keep a voice from an earlier clip" if node is not None and nid is not None and has
             else "Select a made clip or a converted video")
            if room else f"There are already {T.MAX_AUDIO_REFS} sounds or {T.AUDIO_REF_TOTAL:g} s; remove one first")
        sel = 0 <= self.sounds.currentRow() < len(refs)
        self.b_sound_del.setEnabled(idle and sel)
        self.b_sound_play.setEnabled(sel)
