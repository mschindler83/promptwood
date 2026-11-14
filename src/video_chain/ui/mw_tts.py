"""Text to speech and speaking a clip's lines."""
from __future__ import annotations

import copy
import threading
from pathlib import Path

from .. import chain, tree as T, tts as TTS
from .speak_dialog import SpeakLinesDialog
from .tts_dialog import VoiceTtsDialog
from .widgets import safe_emit
from . import speak_dialog as SPD, tts_dialog as TD
from .ui_logic import DEFAULT_COMFY_URL


class TtsMixin:
    """Text to speech and speaking a clip's lines."""

    def _tts_changed(self, *_):
        if self._loading or self.tree is None:
            return
        provider = self.tts_provider.currentData()
        self.view.show((self._tts_form, self._eleven_row), provider == "elevenlabs")
        new = {"provider": provider, "eleven_model": self.eleven_model.currentData() or
               TTS.DEFAULT_OPTIONS["eleven_model"]}
        old = TTS.options(self.tree)
        if {k: old[k] for k in new} != new:
            self.tree.options["tts"] = {**(self.tree.options.get("tts") or {}), **new}
            self._changed()                         # (a spoken track is made with it; clips do not depend on it)
        if provider == "elevenlabs" and not TTS.eleven_key():
            self.status.setText("ElevenLabs needs your API key: Preferences → ElevenLabs API key.")

    def _load_eleven_models(self):
        def work():
            try:
                safe_emit(self.bridge.eleven_models, TTS.ElevenLabs().models(), "")
            except Exception as e:  # noqa: BLE001
                safe_emit(self.bridge.eleven_models, None, str(e))
        self.status.setText("Asking ElevenLabs for its models…")
        threading.Thread(target=work, daemon=True).start()

    def _fill_eleven_models(self, models, err: str):
        if models is None:
            self._report_error(f"Could not ask ElevenLabs: {err}")
            return
        cur = self.eleven_model.currentData()
        self.eleven_model.blockSignals(True)
        self.eleven_model.clear()
        for m in models:
            self.eleven_model.addItem(f"{m.get('name', m['id'])} ({m['id']})", m["id"])
        if cur and self.eleven_model.findData(cur) < 0:
            self.eleven_model.addItem(cur, cur)
        self.eleven_model.setCurrentIndex(max(0, self.eleven_model.findData(cur)) if cur else 0)
        self.eleven_model.blockSignals(False)
        self.status.setText(f"ElevenLabs offers {len(models)} text to speech models.")

    # ---- text to speech
    def local_tts(self) -> "TTS.LocalTTS":
        """The app's Qwen3-TTS worker (started on demand; stop_tts unloads it)."""
        if self._tts_local is None:
            d = Path.home() / ".cache" / "video-chain"
            d.mkdir(parents=True, exist_ok=True)
            self._tts_local = TTS.LocalTTS(log=d / "tts-worker.log")
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        self._tts_local.before_launch = lambda _w: self._make_room(TTS.LOCAL_VRAM_MB, url, "Qwen3-TTS",
                                                                   TTS.TtsError, "speak the lines after the run")
        return self._tts_local

    def fish_tts(self) -> "TTS.FishTTS":
        """The app's Fish S2 Pro worker (started on demand; stop_tts unloads it). Before it starts, the refiner's
        llama.cpp server is stopped (its RAM and VRAM) and ComfyUI unloads its models when VRAM is short."""
        if self._tts_fish is None:
            d = Path.home() / ".cache" / "video-chain"
            d.mkdir(parents=True, exist_ok=True)
            self._tts_fish = TTS.FishTTS(log=d / "fish-worker.log", helper=self.local_tts())
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        self._tts_fish.helper = self.local_tts()
        self._tts_fish.before_launch = lambda _w: self._room_for_fish(url)
        return self._tts_fish

    def _room_for_fish(self, url: str) -> None:
        """(a worker thread) Room for Fish S2 Pro: the llama.cpp server goes (it may hold many GB of RAM as well as
        VRAM), then ComfyUI is asked to unload its models (_make_room)."""
        cur = getattr(self, "_llama", None)
        if cur is not None and cur.stop(wait=True):
            self.bridge.tts_log.emit("stopped the llama.cpp server: Fish S2 Pro needs its RAM and VRAM")
        self._make_room(TTS.FISH_VRAM_MB, url, "Fish S2 Pro", TTS.TtsError, "speak the lines after the run")

    def stop_tts(self, why: str = "") -> bool:
        stopped = False
        for w, what in ((self._tts_local, "Qwen3-TTS"), (self._tts_fish, "Fish S2 Pro")):
            if w is not None and w.stop():
                stopped = True
                if why:
                    self.log.appendPlainText(f"Stopped {what} ({why}).")
        return stopped

    def _tts_engine(self):
        return TTS.engine(self.tree, self.local_tts(), self.fish_tts())

    def _speak_engine(self, nid: str):
        """The engine a clip's lines can be spoken with now, or None (the status line says why not)."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return None
        if self._tts_busy:
            self.status.setText("Text to speech is busy; wait until it is done.")
            return None
        node = self.tree.nodes[nid]
        if node.frozen:
            self._frozen_block(node, "speak")
            return None
        eng = self._tts_engine()
        bad = TTS.line_problems(self.tree, node, eng.name) + eng.problems(
            transcribe=eng.name in ("local", "fish") and TTS.needs_transcript(self.tree, node))
        if bad:
            self._report_error(f"Cannot speak the lines of clip #{T.short_id(nid)}: " + "; ".join(bad) + ".")
            return None
        return eng

    def speak_clip_window(self, nid: str) -> bool:
        """The Speak the lines window for a clip: takes are made and listened to, how they are spoken is set; the
        take that is used becomes the clip's sound (_lines_spoken). True: a take was used."""
        eng = self._speak_engine(nid)
        if eng is None:
            return False
        snap, project = copy.deepcopy(self.tree), self.project

        def render(take: int, how: dict, lines: list[str]) -> dict:
            r = TTS.render_lines(project, snap, nid, eng, take=take, speak=how, lines=lines)
            for name, text in (r.get("texts") or {}).items():     # (transcribed once, not for every take)
                TTS._set_voice_fields(snap, name, text=text)
            return r
        dlg = SpeakLinesDialog(
            self, f"#{T.short_id(nid)}", [{"name": d["speaker"]["name"], "line": d["line"]}
                                  for d in TTS.clip_lines(self.tree, self.tree.nodes[nid])],
            eng.name, TTS.speak_options(self.tree),
            tags=TTS.takes_tags(eng.name, TTS.options(self.tree)["eleven_model"]),
            play=lambda path, title: self.preview.open(Path(path), title),
            render=render,
            file=lambda r: project / r["file"])
        self._tts_busy = True
        used = False
        try:
            self._update_buttons()
            ok = SPD.run_dialog(dlg)
            how = dlg.speak_values()
            if TTS.speak_options(self.tree) != how:     # how lines are spoken is kept with the project
                self.tree.options["tts"] = {**(self.tree.options.get("tts") or {}), "speak": how}
                self._changed()
            self.preview.stop()                         # (it may still play a take that goes now)
            for r in dlg.unused():                      # the other takes are never deleted: they go to old/
                chain.keep_aside(project, project / r["file"])
            if ok and dlg.chosen is not None:
                texts = {}                              # (every take's transcripts: each was made once only)
                for r in dlg.takes:
                    texts.update(r.get("texts") or {})
                node = self.tree.nodes.get(nid)
                if node is not None and not node.frozen and dlg.chosen["prompt"] != node.prompt \
                        and dlg.chosen.get("base_prompt") == node.prompt:
                    node.prompt = dlg.chosen["prompt"]  # the words the take speaks become the clip's lines
                    self._changed()
                self._lines_spoken(nid, {**dlg.chosen, "texts": texts}, "", str(project))
                used = True
            elif dlg.takes:                             # nothing used: the transcripts are still kept
                self._keep_transcripts(dlg.takes)
        finally:
            self._tts_busy = False
            self._update_buttons()
            dlg.deleteLater()
            if eng.name in ("local", "fish"):
                self.stop_tts("the lines window was closed")
        return used

    def _keep_transcripts(self, takes: list[dict]) -> None:
        """What the voice samples say (transcribed while takes were made) is kept, also when no take is used."""
        changed = False
        for r in takes:
            files = r.get("voice_files") or {}
            for name, text in (r.get("texts") or {}).items():
                c = T.character(self.tree, name)
                if c is not None and c.get("voice") and files.get(name) == c["voice"]["file"] \
                        and not c["voice"].get("text"):
                    TTS._set_voice_fields(self.tree, name, text=text)
                    changed = True
        if changed:
            self._write_tree()

    def speak_clip(self, nid: str) -> bool:
        """Speak a clip's lines with text to speech right away (one take, in a thread); _lines_spoken makes them
        its sound."""
        eng = self._speak_engine(nid)
        if eng is None:
            return False
        snap, project = copy.deepcopy(self.tree), self.project
        self._tts_busy = True
        try:
            self._update_buttons()
            self.status.setText(f"Speaking the lines of clip #{T.short_id(nid)} ({TTS.PROVIDERS[eng.name]})…")
        except BaseException:
            self._tts_busy = False                  # (it never started: not busy for ever)
            raise

        def work():
            try:
                r = TTS.render_lines(project, snap, nid, eng, log=self.bridge.tts_log.emit)
                safe_emit(self.bridge.spoken, nid, r, "", str(project))
            except BaseException as e:  # noqa: BLE001  (the answer ends "busy")
                safe_emit(self.bridge.spoken, nid, None, str(e) or type(e).__name__, str(project))
            finally:
                if eng.name in ("local", "fish"):
                    eng.stop()                      # (unloaded right after: its VRAM back for ComfyUI)
        threading.Thread(target=work, daemon=True).start()
        return True

    def _lines_spoken(self, nid: str, r, err: str, project: str):
        self._tts_busy = False
        self._update_buttons()
        if self.project is None or str(self.project) != project or self.tree is None or nid not in self.tree.nodes:
            if r:                                   # (another project now, or the clip is gone: not used)
                chain.keep_aside(Path(project), Path(project) / r["file"])
            return
        if r is None:
            self._report_error(f"Could not speak the lines of clip #{T.short_id(nid)}: {err}")
            return
        node = self.tree.nodes[nid]
        if node.frozen:
            chain.keep_aside(self.project, self.project / r["file"])
            self._frozen_block(node, "speak")
            return
        try:
            done = TTS.apply_lines(self.project, self.tree, nid, r)
        except (TTS.TtsError, OSError) as e:
            chain.keep_aside(self.project, self.project / r["file"])
            self._report_error(f"Could not use the spoken lines of clip #{T.short_id(nid)}: {e}")
            return
        self._write_tree()
        self.refresh(rebuild_path=True)
        msg = (f"Clip #{T.short_id(nid)}: {done['lines']} line{'s' if done['lines'] != 1 else ''} spoken ({done['secs']:.1f} s)"
               f" and made its sound (track “{T.track_of(self.tree, done['track'])['name']}”); the picture is made "
               "to it, lips in sync.")
        if done["frames"]:
            msg += f" The clip is now {done['frames'] / chain.FPS:.1f} s long, so the lines fit."
        if r.get("prompt") != node.prompt:
            msg += " (The prompt was changed meanwhile: speak again if its lines changed.)"
        self.status.setText(msg)

    def _voice_with_tts(self):
        c = self._selected_character()
        if c is None or not self._cast_edit_ok():
            return
        if self._tts_busy:
            self.status.setText("Text to speech is busy; wait until it is done.")
            return
        opts = TTS.options(self.tree)
        desc = c.get("desc") or ""
        dlg = VoiceTtsDialog(self, c["name"], desc, TTS.language(self.tree), opts["provider"], opts["eleven_model"],
                             play=lambda path, title: self.preview.open(Path(path), title),
                             local=lambda w=self.local_tts(): w, eleven=TTS.ElevenLabs,
                             voices_cache=self._eleven_voices)
        self._tts_busy = True
        try:
            ok = TD.run_dialog(dlg)
            self._eleven_voices = getattr(dlg, "voices", None) or self._eleven_voices
            if ok and dlg.made is not None and dlg.wav.is_file():
                before = self._done_clips()
                try:
                    entry = TTS.use_sample(self.project, self.tree, c["name"], dlg.wav, dlg.made["label"],
                                           dlg.made["text"], dlg.made["eleven"])
                except (ValueError, OSError, RuntimeError) as e:
                    self._report_error(f"Could not set the voice: {e}")
                    return
                self._show_references(select=c["name"])
                self._write_tree()
                self._changed()
                msg = f"Voice of @{c['name']} set with text to speech ({entry['secs']:.1f} s)."
                self.status.setText(msg)
                if self._report_remade(before, f"Voice of @{c['name']} set"):
                    self.status.setText(msg + " " + self.status.text())
        finally:
            self._tts_busy = False
            self.preview.stop()                     # (it may still play the sample, whose folder goes now)
            dlg.cleanup()
            self.stop_tts("the voice window was closed")
