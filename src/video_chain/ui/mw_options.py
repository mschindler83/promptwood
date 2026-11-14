"""The project's options widgets and the LoRA trigger words."""
from __future__ import annotations

import copy
import logging
import threading

from PySide6.QtCore import QTimer

from .. import tree as T, triggers, tts as TTS
from ..comfy import REF_HINT
from .widgets import safe_emit
from .clip_cards import PromptCard
from .mw_support import select_frames
from .ui_logic import FRAME_CHOICES, speed_mode
from .mw_host import MainWindowHost


class OptionsMixin(MainWindowHost):
    """The project's options widgets and the LoRA trigger words."""

    @staticmethod
    def _clean_options(o: dict):
        """Stored option values that cannot be read (null, text where a number belongs, a wrong shape) go back to the
        defaults, so the option widgets can always be filled."""
        d = T.DEFAULT_OPTIONS
        for k, kind in (("frames", int), ("resolution", int), ("steps", int), ("seed", int)):
            try:
                kind(o[k])
            except (KeyError, TypeError, ValueError):
                o[k] = d[k]
        for k in ("identity", "style", "turbo", "hyperflow", "pika", "veda"):
            if k in o and not isinstance(o[k], dict):
                o[k] = copy.deepcopy(d[k])
        for k in ("turbo", "hyperflow", "pika"):
            for key, kind in (("steps", int), ("strength", float), ("cfg", float)):
                try:
                    if key in (o.get(k) or {}):
                        kind(o[k][key])
                except (TypeError, ValueError):
                    del o[k][key]
        try:
            float(o.get("cfg", 1.0))
        except (TypeError, ValueError):
            del o["cfg"]

    def _options_to_widgets(self):
        o = self.tree.options
        self._loading = True
        try:
            for k in range(self.frames.count() - 1, len(FRAME_CHOICES) - 1, -1):   # an old value added earlier
                self.frames.removeItem(k)
            fr = int(o.get("frames", 73))
            if fr in FRAME_CHOICES:
                self.frames.setCurrentIndex(FRAME_CHOICES.index(fr))
            else:
                select_frames(self.frames, fr)
            i = self.resolution.findData(int(o.get("resolution", 512)))
            self.resolution.setCurrentIndex(i if i >= 0 else 1)
            self.aspect.setCurrentIndex(max(0, self.aspect.findData(o.get("aspect", "16:9"))))
            self.steps.setValue(int(o.get("steps", 28)))
            self.audio.setChecked(bool(o.get("audio", True)))
            ident = o.get("identity") or {}
            self.ref_size.setCurrentIndex(max(0, self.ref_size.findData(ident.get("size", "match"))))
            self.ref_hint.setChecked(bool(ident.get("hint", True)))
            self.prompt_format.setCurrentIndex(max(0, self.prompt_format.findData("h3" if T.is_h3(o) else "classic")))
            self._update_hint_enabled()
            self.dialogue_lang.setCurrentText(ident.get("language") or "English")
            topt = TTS.options(o)
            self.tts_provider.setCurrentIndex(max(0, self.tts_provider.findData(topt["provider"])))
            if self.eleven_model.findData(topt["eleven_model"]) < 0:
                self.eleven_model.addItem(topt["eleven_model"], topt["eleven_model"])
            self.eleven_model.setCurrentIndex(self.eleven_model.findData(topt["eleven_model"]))
            self.view.show((self._tts_form, self._eleven_row), topt["provider"] == "elevenlabs")
            turbo = o.get("turbo") or {}
            self.speed.setCurrentIndex(self.speed.findData(speed_mode(o)))
            self.turbo_steps.setValue(int(turbo.get("steps", 6)))
            self.turbo_strength.setValue(float(turbo.get("strength", 1.0)))
            self._cfg_mode = self.speed.currentData()
            self.clip_cfg.setValue(float(self._mode_options(o, self._cfg_mode).get("cfg", 1.0)))
            self._cfg_hint()
            self._fill_turbo(turbo.get("lora"))
            self.keep_rep.setChecked(bool(o.get("keep_repeated", False)))
            self.veda.setChecked(bool((o.get("veda") or {"on": True}).get("on", True)))
            style = o.get("style") or {}
            self.style_prefix.setText(style.get("prefix") or "")
            self.style_suffix.setText(style.get("suffix") or "")
            self.lora_triggers.setChecked(bool(o.get("lora_triggers", True)))
        finally:
            self._loading = False
        self._update_adv_note()

    @staticmethod
    def _mode_options(options: dict, mode: str) -> dict:
        """Where a speed mode keeps its settings (its CFG): the project options for Normal."""
        return options if mode == "normal" else (options.get(mode) or {})

    def _cfg_hint(self):
        v = self.clip_cfg.value()
        name = {"normal": "Normal", "turbo": "Turbo", "hyperflow": "HyperFlow", "pika": "Pika"}.get(
            self.speed.currentData(), "")
        self.cfg_note.setText(f"for {name}: off (one pass per step)" if abs(v - 1) < 1e-6 else
                              f"for {name}: two passes per step (about twice as slow)")

    def _options_changed(self, *_):
        if self._loading or self.tree is None or self.running:
            return
        before = self._done_clips()
        mode = self.speed.currentData()
        if mode != self._cfg_mode:                  # another speed mode: its own CFG into the field (not this one's)
            self._cfg_mode = mode
            self.clip_cfg.blockSignals(True)
            self.clip_cfg.setValue(float(self._mode_options(self.tree.options, mode).get("cfg", 1.0)))
            self.clip_cfg.blockSignals(False)
        cfg = round(self.clip_cfg.value(), 2)
        self._cfg_hint()
        self.tree.options.update({
            "frames": self.frames.currentData(), "resolution": self.resolution.currentData(),
            "aspect": self.aspect.currentData(),
            "steps": self.steps.value(), "audio": self.audio.isChecked(), "keep_repeated": self.keep_rep.isChecked(),
            "turbo": {**(self.tree.options.get("turbo") or {}),
                      "on": mode == "turbo", "lora": self.turbo_lora.currentData(),
                      "steps": self.turbo_steps.value(), "strength": round(self.turbo_strength.value(), 2)},
            "hyperflow": {**(self.tree.options.get("hyperflow") or {"strength": 1.0}), "on": mode == "hyperflow"},
            "pika": {**(self.tree.options.get("pika") or {"strength": 1.0}), "on": mode == "pika"},
            "veda": {"on": self.veda.isChecked()}})
        if mode == "normal":                        # the CFG belongs to the speed mode it was set for
            self.tree.options["cfg"] = cfg
        else:
            self.tree.options[mode]["cfg"] = cfg
        self._update_size_info()
        self._show_references()
        self._turbo_widgets()
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
        online, bridge = self.triggers_online.isChecked(), self.bridge
        self._trigger_scan = threading.Event()
        self._trigger_scan_online = online
        stop = self._trigger_scan

        def work():
            counts = {}
            try:
                counts = triggers.db().scan(paths, online=online, stop=stop.is_set,
                                            progress=lambda *a: safe_emit(bridge.trigger_progress, *a))
            except Exception as e:  # noqa: BLE001  (the scan ends; what went wrong is in the log)
                logging.getLogger(__name__).warning("trigger word scan failed", exc_info=True)
                safe_emit(bridge.log, f"The trigger word scan failed: {type(e).__name__}: {e}")
                counts = {}
            safe_emit(bridge.triggers, counts)
        threading.Thread(target=work, daemon=True).start()

    def _video_gpu_changed(self, on: bool):
        self.qs.setValue("videoGpu", on)           # at once: read at the next start, also after a crash
        self.qs.sync()
        self.status.setText("Videos play on the " + ("GPU" if on else "CPU") + " after the app is restarted.")

    def _civitai_site_changed(self, *_):
        """The CivitAI site asked first and linked to (kept at once; LoRAs neither site was asked about yet are
        looked up with the next scan)."""
        chosen = triggers.set_site(self.civitai_site.currentData())
        self.qs.setValue("civitaiSite", chosen)
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
