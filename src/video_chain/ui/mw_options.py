"""The project's options widgets and the LoRA trigger words."""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QTimer

from .workers import post, run_in_background
from . import settings_keys as SK
from .. import chain, tree as T, triggers, tts as TTS
from ..comfy import REF_HINT
from .clip_cards import PromptCard
from .mw_support import select_frames
from .ui_logic import FRAME_CHOICES, SPEED_NAMES, as_dict, looks_distilled, speed_mode
from .mw_host import MainWindowHost


class OptionsMixin(MainWindowHost):
    """The project's options widgets and the LoRA trigger words."""

    @staticmethod
    def _clean_options(options: T.Options):
        """Stored option values that cannot be read (null, text where a number belongs, a wrong shape) go back to the
        defaults, so the option widgets can always be filled."""
        d = as_dict(T.DEFAULT_OPTIONS)
        o = as_dict(options)                        # (keys named by variables)
        for k, kind in (("frames", int), ("resolution", int), ("steps", int), ("seed", int)):
            try:
                kind(o[k])
            except (KeyError, TypeError, ValueError):
                o[k] = d[k]
        for k in ("identity", "style", "turbo", "hyperflow", "pika", "veda", "checkpoint", "two_pass"):
            if k in o and not isinstance(o[k], dict):
                if k in d:
                    o[k] = copy.deepcopy(d[k])
                else:
                    del o[k]
        for k in ("turbo", "hyperflow", "pika"):
            for key, kind2 in (("steps", int), ("strength", float), ("cfg", float)):
                try:
                    if key in (o.get(k) or {}):
                        kind2(o[k][key])
                except (TypeError, ValueError):
                    del o[k][key]
        try:
            float(o.get("cfg", 1.0))
        except (TypeError, ValueError):
            del o["cfg"]

    def _options_to_widgets(self):
        assert self.tree is not None
        o = self.tree.options
        d = as_dict(T.DEFAULT_OPTIONS)              # (what a missing value shows)
        self._loading = True
        try:
            for k in range(self.frames.count() - 1, len(FRAME_CHOICES) - 1, -1):   # an old value added earlier
                self.frames.removeItem(k)
            fr = int(o.get("frames", d["frames"]))
            if fr in FRAME_CHOICES:
                self.frames.setCurrentIndex(FRAME_CHOICES.index(fr))
            else:
                select_frames(self.frames, fr)
            i = self.resolution.findData(int(o.get("resolution", d["resolution"])))
            self.resolution.setCurrentIndex(i if i >= 0 else 1)
            exact = chain.exact_size(o.get("size"))
            self.exact_size.setChecked(exact is not None)
            if exact:
                self.exact_width.setValue(exact[0])
                self.exact_height.setValue(exact[1])
            self._exact_widgets()
            self.shape.setCurrentIndex(max(0, self.shape.findData(self._shape_shown(o))))
            self.steps.setValue(int(o.get("steps", d["steps"])))
            self.audio.setChecked(bool(o.get("audio", d["audio"])))
            ident = o.get("identity") or {}
            self.ref_size.setCurrentIndex(max(0, self.ref_size.findData(ident.get("size", d["identity"]["size"]))))
            self.ref_hint.setChecked(bool(ident.get("hint", d["identity"]["hint"])))
            self.prompt_format.setCurrentIndex(max(0, self.prompt_format.findData("h3" if T.is_h3(o) else "classic")))
            self._update_hint_enabled()
            self.dialogue_lang.setCurrentText(ident.get("language") or d["identity"]["language"])
            topt = TTS.options(o)
            self.tts_provider.setCurrentIndex(max(0, self.tts_provider.findData(topt["provider"])))
            if self.eleven_model.findData(topt["eleven_model"]) < 0:
                self.eleven_model.addItem(topt["eleven_model"], topt["eleven_model"])
            self.eleven_model.setCurrentIndex(self.eleven_model.findData(topt["eleven_model"]))
            self.view.show((self._tts_form, self._eleven_row), topt["provider"] == "elevenlabs")
            turbo = o.get("turbo") or {}
            self.speed.setCurrentIndex(self.speed.findData(speed_mode(as_dict(o))))
            self.turbo_steps.setValue(int(turbo.get("steps", d["turbo"]["steps"])))
            self.turbo_strength.setValue(float(turbo.get("strength", d["turbo"]["strength"])))
            self._cfg_mode = self.speed.currentData()
            self.clip_cfg.setValue(float(self._mode_options(as_dict(o), self._cfg_mode).get("cfg", 1.0)))
            self._cfg_hint()
            self._fill_turbo(turbo.get("lora"))
            self._fill_h3_models()
            self.keep_rep.setChecked(bool(o.get("keep_repeated", d["keep_repeated"])))
            self.veda.setChecked(bool((o.get("veda") or d["veda"]).get("on", d["veda"]["on"])))
            self.two_pass.setChecked(bool((o.get("two_pass") or {}).get("on")))
            style = o.get("style") or {}
            self.style_prefix.setText(style.get("prefix") or "")
            self.style_suffix.setText(style.get("suffix") or "")
            self.lora_triggers.setChecked(bool(o.get("lora_triggers", d["lora_triggers"])))
        finally:
            self._loading = False
        self._update_adv_note()

    def _fill_h3_models(self):
        """The H3 model list: the standard files, the checkpoints found, and the project's own choice (also when
        it was not found: it may be on another machine's ComfyUI); the Distilled switch as the project has it."""
        ck = T.checkpoint_of(self.tree.options) if self.tree is not None else None
        chosen = ck["name"] if ck else ""
        loading, self._loading = self._loading, True
        self.h3_model.blockSignals(True)
        try:
            self.h3_model.clear()
            self.h3_model.addItem("MiniMax H3 (standard)", "")
            for name in self.h3_checkpoints:
                self.h3_model.addItem(Path(name).stem, name)
            if chosen and self.h3_model.findData(chosen) < 0:
                self.h3_model.addItem(f"{Path(chosen).stem} (not found)", chosen)
            self.h3_model.setCurrentIndex(max(0, self.h3_model.findData(chosen)))
            self.h3_distilled.setChecked(bool(ck and ck["distilled"]))
        finally:
            self.h3_model.blockSignals(False)
            self._loading = loading
        self._turbo_widgets()

    def _h3_model_chosen(self, *_):
        """Another H3 model chosen: the Distilled switch takes the guess from its name, then it counts as a
        change of the project's settings."""
        if self._loading:
            return
        name = self.h3_model.currentData() or ""
        self.h3_distilled.blockSignals(True)
        self.h3_distilled.setChecked(bool(name) and looks_distilled(name))
        self.h3_distilled.blockSignals(False)
        self._options_changed()

    @staticmethod
    def _mode_options(options: dict, mode: str) -> dict:
        """Where a speed mode keeps its settings (its CFG): the project options for Normal."""
        return options if mode == "normal" else (options.get(mode) or {})

    def _exact_from_widgets(self) -> tuple[int, int] | None:
        """The exact size chosen (None: off): the nearest size the model takes to what the fields say."""
        if not self.exact_size.isChecked():
            return None
        return chain.snap_size(self.exact_width.value(), self.exact_height.value())

    def _exact_shown(self) -> None:
        """A field was left: it shows the size that is used (while typing it is left alone)."""
        size = self._exact_from_widgets()
        for spin, value in zip((self.exact_width, self.exact_height), size or ()):
            if spin.value() != value:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

    def _exact_widgets(self, idle: bool | None = None) -> None:
        """The fields of the exact size, and Size, which does not count while an exact size is set."""
        idle = (not self.running) if idle is None else idle
        on = self.exact_size.isChecked()
        self.exact_width.setEnabled(on and idle)
        self.exact_height.setEnabled(on and idle)
        self.resolution.setEnabled(idle and not on)

    def _shape_shown(self, o) -> str | None:
        """The Shape box for these options: the shape chosen; a project of an older version without a start image or
        video showed its Aspect setting, which gave its shape: that one (stored as the shape with the next change)."""
        if "shape" in o or self.project is None:
            return T.shape_of(o)
        if T.size_origin(self.project, self.tree) == "aspect":
            a = o.get("aspect")
            return a if a in T.ASPECTS else None
        return None

    def _chosen_size(self, width: int, height: int) -> tuple[int, int]:
        """The clips' size for a source of this size, as the widgets say now."""
        return chain.video_size(width, height, self.resolution.currentData(),
                                list(self._exact_from_widgets() or ()) or None)

    def _cfg_hint(self):
        v = self.clip_cfg.value()
        name = SPEED_NAMES.get(self.speed.currentData(), "")
        self.cfg_note.setText(f"for {name}: off (one pass per step)" if abs(v - 1) < 1e-6 else
                              f"for {name}: two passes per step (about twice as slow)")

    def _options_typing(self, *_):
        """A spin box ticked (typing, arrow keys, the wheel): the value is taken at once, the heavy follow-up work
        (references, card choices, saving, what is remade) waits for a short pause and runs once (MWB-9)."""
        self._options_changed(defer=True)

    def _options_changed(self, *_, defer: bool = False):
        if self._loading or self.tree is None or self.running:
            return
        if self._options_before is None:            # (what was done before the first tick of this burst)
            self._options_before = self._done_clips()
        mode = self.speed.currentData()
        if mode != self._cfg_mode:                  # another speed mode: its own CFG into the field (not this one's)
            self._cfg_mode = mode
            self.clip_cfg.blockSignals(True)
            self.clip_cfg.setValue(float(self._mode_options(as_dict(self.tree.options), mode).get("cfg", 1.0)))
            self.clip_cfg.blockSignals(False)
        cfg = round(self.clip_cfg.value(), 2)
        self._cfg_hint()
        exact = self._exact_from_widgets()
        if exact:
            self.tree.options["size"] = list(exact)
        else:
            self.tree.options.pop("size", None)
        self._exact_widgets()
        self.tree.options.update({
            "frames": self.frames.currentData(), "resolution": self.resolution.currentData(),
            "steps": self.steps.value(), "audio": self.audio.isChecked(), "keep_repeated": self.keep_rep.isChecked(),
            "turbo": {**(self.tree.options.get("turbo") or {}),
                      "lora": self.turbo_lora.currentData(),
                      "steps": self.turbo_steps.value(), "strength": round(self.turbo_strength.value(), 2)},
            "veda": {"on": self.veda.isChecked()}, "two_pass": {"on": self.two_pass.isChecked()},
            "checkpoint": {"name": self.h3_model.currentData() or "",
                           "distilled": bool(self.h3_model.currentData()) and self.h3_distilled.isChecked()}})
        if self.shape.currentData():
            self.tree.options["shape"] = self.shape.currentData()
        else:
            self.tree.options.pop("shape", None)
        T.set_speed_mode(self.tree.options, mode)
        if mode == "normal":                        # the CFG belongs to the speed mode it was set for
            as_dict(self.tree.options)["cfg"] = cfg          # (not in the Options schema: tree/ should list it)
        else:
            as_dict(self.tree.options)[mode]["cfg"] = cfg
        self._update_size_info()
        self._turbo_widgets()
        if defer:
            self._options_timer.start()
        else:
            self._options_settled()

    def _options_settled(self):
        """The burst of option changes is over (or a combo box / check box changed): the follow-up work, once."""
        self._options_timer.stop()
        before, self._options_before = self._options_before, None
        if before is None or self.tree is None:
            return
        self._show_references()
        self._refresh_card_choices()
        self._changed()
        if not self._report_remade(before, "Project settings changed"):
            self.status.setText("Project settings changed.")

    def _style_changed(self):
        """The style text (before / after every prompt) was edited: it counts like a project setting."""
        if self._loading or self.tree is None or self.running:
            return
        new = {"prefix": self.style_prefix.text().strip(), "suffix": self.style_suffix.text().strip()}
        old = self.tree.options.get("style") or {}
        if {k: (old.get(k) or "").strip() for k in new} == new:
            return
        before = self._done_clips()
        self.tree.options["style"] = new
        self._changed()
        self._update_cards()
        if not self._report_remade(before, "Style text changed"):
            self.status.setText("Style text changed.")

    def _prompt_format_changed(self, *_):
        """Official H3 / Classic: how the prompts are written for the model. Made clips are kept (CHANGED)."""
        fmt = self.prompt_format.currentData()
        if self._loading or self.tree is None or fmt == ("h3" if T.is_h3(self.tree) else "classic"):
            return
        if self.running:
            self._loading = True
            try:
                self.prompt_format.setCurrentIndex(self.prompt_format.findData("h3" if T.is_h3(self.tree)
                                                                               else "classic"))
            finally:
                self._loading = False
            self.status.setText("Wait until generating has finished.")
            return
        before = self._done_clips()
        T.set_prompt_format(self.tree, fmt)
        self._update_hint_enabled()
        self._changed()
        self._update_cards()
        name = "Official H3" if fmt == "h3" else "Classic"
        if not self._report_remade(before, f"Prompt format: {name}"):
            self.status.setText(f"Prompt format: {name}.")

    def _update_hint_enabled(self):
        """The "Tie each person to their face" switch is only for the classic format (the official one always
        ties each subject to its pictures)."""
        h3 = self.tree is not None and T.is_h3(self.tree)
        self.ref_hint.setEnabled(not self.running and not h3)
        self.ref_hint.setToolTip("The official H3 format always ties each person (and place, object, style) to its "
                                 "pictures in the prompt; this switch is for the classic format" if h3 else
                                 "With @mentions: \"Anna has exactly the face of the person in <Picture 1>.\" "
                                 "and that the pictures only show who the people are. Without: "
                                 f"\"{REF_HINT}\"")

    def _lora_triggers_changed(self, on: bool):
        if self._loading or self.tree is None or self.running:
            return
        before = self._done_clips()
        self.tree.options["lora_triggers"] = bool(on)
        self._changed()
        self._update_cards()
        if not self._report_remade(before, "Trigger words " + ("on" if on else "off")):
            self.status.setText("LoRA trigger words are " + ("put before the prompts." if on else "not used."))

    # ---- LoRA trigger words (triggers.py), looked up in the background
    def _trigger_paths(self) -> list[str]:
        """The LoRAs to know the trigger words of: those found in ComfyUI's folders, and those the clips use
        (from anywhere)."""
        paths = [p for p, _n in self.h3_loras + self.qwen_loras]
        for items in self.image_loras.values():
            paths += [p for p, _n in items]
        if self.tree is not None:
            for n in self.tree.nodes.values():
                paths += [p for p, _s in T.node_loras(n)]
        return list(dict.fromkeys(p for p in paths if p))

    def _scan_triggers(self):
        """Read the trigger words of LoRAs not known yet (their files; CivitAI if allowed), in the background."""
        paths = self._trigger_paths()
        if self._trigger_scan is not None or not paths:
            return
        online = self.triggers_online.isChecked()
        self._trigger_scan = threading.Event()
        self._trigger_scan_online = online
        stop = self._trigger_scan

        def progress(*a) -> None:
            post(self, lambda: self._trigger_progress(*a))

        def failed(e: BaseException):                 # (the scan ends; what went wrong is in the log)
            logging.getLogger(__name__).warning("trigger word scan failed", exc_info=e)
            self.log.appendPlainText(f"The trigger word scan failed: {type(e).__name__}: {e}")
            self._triggers_found({})
        run_in_background(lambda: triggers.db().scan(paths, online=online, stop=stop.is_set, progress=progress),
                          on_done=self._triggers_found, on_error=failed, owner=self)

    def _video_gpu_changed(self, on: bool):
        self.qs.setValue(SK.VIDEO_GPU, on)           # at once: read at the next start, also after a crash
        self.qs.sync()
        self.status.setText("Videos play on the " + ("GPU" if on else "CPU") + " after the app is restarted.")

    def _civitai_site_changed(self, *_):
        """The CivitAI site asked first and linked to (kept at once; LoRAs neither site was asked about yet are
        looked up with the next scan)."""
        chosen = triggers.set_site(self.civitai_site.currentData())
        self.qs.setValue(SK.CIVITAI_SITE, chosen)
        self.status.setText(f"CivitAI: {chosen.split('//', 1)[1]} is asked first and is where links open.")
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.loras.refresh_triggers()
        if self.triggers_online.isChecked():
            self._scan_triggers()

    def _trigger_progress(self, k: int, n: int, name: str):
        """The background lookup says where it is (one line per LoRA asked; it reads a LoRA's whole gallery)."""
        if not self.running:                       # (a run's own status lines stay)
            self.status.setText(f"CivitAI: reading up on LoRA {k} of {n} ({name}): trigger words, description "
                                "and the prompts of its gallery …")

    def _triggers_found(self, counts: dict):
        self._trigger_scan = None
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.loras.refresh_triggers()
        changed = counts.get("new") or counts.get("looked")
        if self.tree is not None and changed and not self.running:
            self.refresh()                         # clips record their LoRAs' words
        if self.triggers_online.isChecked() and not self._trigger_scan_online:
            QTimer.singleShot(0, self._scan_triggers)     # switched on while an offline scan ran
        if counts.get("key_rejected"):
            self.status.setText("CivitAI rejected your API key: LoRAs are looked up without it. Check or renew the "
                                "key in the settings (CivitAI → Account settings → API Keys).")
        elif counts.get("looked"):
            self.status.setText(f"CivitAI: looked up {counts['looked']} LoRA"
                                f"{'s' if counts['looked'] != 1 else ''} (trigger words, descriptions, gallery "
                                f"prompts), {counts.get('found', 0)} with trigger words"
                                + (f", {counts['failed']} could not be asked (offline?)"
                                   if counts.get("failed") else "") + ".")
