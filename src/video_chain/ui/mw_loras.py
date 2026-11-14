"""LoRAs: the list, the scan, the pickers."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog

from .. import chain, comfy as CMF, imagegen, tree as T
from .lora_stack import fill_lora_combo, lora_label, lora_path
from .widgets import safe_emit
from . import ui_logic as UL
from .clip_cards import PromptCard
from .ui_logic import MAX_RECENT_LORAS, REMIX_DELAY_MS, STEPS_TIP, label_loras, plural
from .mw_host import MainWindowHost
from .qsvalue import qs_value


class LorasMixin(MainWindowHost):
    """LoRAs: the list, the scan, the pickers."""

    def show_loras(self, current: str | None = None):
        """The LoRA window: every LoRA the app offers, with what is known of each (its gallery's prompts with their
        pictures, the prompt refiner's note, the settings its examples were made with)."""
        from .lora_window import LoraWindow      # (lazy: a rarely opened window)
        card = self._card_of(self.focus, PromptCard)
        more = card.loras.window_args() if card is not None else {}    # (the clip in focus: its LoRAs, its prompt)
        LoraWindow(self, self.known_loras(), current=current, settings=self.qs, **more).exec()
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.loras.refresh_triggers()

    def known_loras(self) -> list[tuple[str, str]]:
        """(path, label) of the LoRAs to offer: every H3 LoRA in ComfyUI's LoRA folders, then LoRAs from
        elsewhere that this project uses or that were chosen recently."""
        out = label_loras(self.h3_loras)
        listed = {p for p, _ in out} | {os.path.realpath(p) for p, _ in out}
        try:
            recent = json.loads(qs_value(self.qs, "recentLoras", "[]", str))
        except ValueError:
            recent = []
        cands = ([p for n in self.tree.nodes.values() for p, _st in T.node_loras(n)] if self.tree else []) + recent
        for lo in cands:
            if lo and lo != T.NO_LORA and lo not in listed and os.path.realpath(lo) not in listed \
                    and Path(lo).is_file():
                out.append((lo, lora_label(lo)))
                listed.add(lo)
        return out

    def _fill_turbo(self, current: str | None):
        """The turbo LoRA list; without a choice yet the recommended v4 ema (or the first) is selected."""
        self.turbo_lora.blockSignals(True)
        try:
            self.turbo_lora.clear()
            known = [(p, Path(n).stem) for p, n in self.turbo_loras]
            if not current and known:
                current = next((p for p, n in known if "v4" in n and "ema" in n), known[0][0])
                if self.tree is not None and not self.running:   # store what the list shows
                    turbo = dict(self.tree.options.get("turbo") or {})
                    turbo["lora"] = current
                    self.tree.options["turbo"] = turbo
                    self._changed()
            if known or current:
                fill_lora_combo(self.turbo_lora, known, current)
            else:
                self.turbo_lora.addItem("no turbo LoRA found", None)
        finally:
            self.turbo_lora.blockSignals(False)
        self._turbo_widgets()

    def _turbo_widgets(self):
        on = self.speed.currentData() == "turbo"
        hyper = self.speed.currentData() in ("hyperflow", "pika")
        idle = not self.running
        for w in (self.turbo_lora, self.turbo_steps, self.turbo_strength):
            w.setEnabled(on and idle)
        self.view.show((self._clips_form, self.turbo_lora), on)  # the turbo settings only matter in turbo mode
        self.view.show((self._clips_form, self._turbo_row), on)
        locked = on or hyper                                      # their own steps for every clip
        self.steps.setEnabled(idle and not locked)
        self.steps.setToolTip("Turbo mode: every clip uses the turbo steps" if on else
                              "HyperFlow / Pika H3 Flex: every clip uses their own steps" if hyper else STEPS_TIP)
        for card in self.cards:
            if isinstance(card, PromptCard):
                card.steps_locked = locked
                card.steps.setEnabled(idle and not locked and not card.edit.isReadOnly())

    def _warm_lora_hashes(self):
        """Hash the LoRAs this project uses in the background, so the clip states need not wait for it."""
        if self.tree is None:
            return
        o = self.tree.options
        files = [(o.get("turbo") or {}).get("lora")] + \
            [p for n in self.tree.nodes.values() for p, _st in T.node_loras(n)]
        files = [f for f in dict.fromkeys(files) if f]
        def work():
            for f in files:
                chain.file_signature(f)               # cached by (path, size, date) for the clip states
        threading.Thread(target=work, daemon=True).start()

    def scan_loras(self, report: bool = False):
        """Find the H3 LoRAs in ComfyUI's LoRA folders (header reads only), in the background. report: say in the
        status bar how many were found."""
        comfy_dir, bridge = Path(os.path.expanduser(self.comfy_dir.text().strip() or UL.DEFAULT_COMFY_DIR)), self.bridge
        self._scanned_at = time.monotonic()
        self._lora_scan_gen = self._lora_scan_gen + 1
        gen = self._lora_scan_gen

        def work():
            try:
                roots = CMF.comfy_lora_roots(comfy_dir)
                found = [(p, n, "turbo" if CMF.is_turbo_lora(Path(p)) else "h3") for p, n in CMF.find_h3_loras(roots)]
                found += [(p, n, "qwen") for p, n in CMF.find_qwen_loras(roots)]
                for fam, items in imagegen.find_image_loras(roots).items():
                    found += [(p, n, fam) for p, n in items]
            except Exception:  # noqa: BLE001
                return                           # keep the lists found before (a folder may be offline)
            if gen != self._lora_scan_gen:
                return                           # a newer scan was started (e.g. another ComfyUI folder)
            try:
                safe_emit(bridge.loras, found, report)
            except RuntimeError:                 # (the window was closed meanwhile)
                pass
        threading.Thread(target=work, daemon=True).start()

    def _loras_found(self, found: list, report: bool = False):
        self.h3_loras = [(p, n) for p, n, kind in found if kind == "h3"]
        self.turbo_loras = [(p, n) for p, n, kind in found if kind == "turbo"]
        self.qwen_loras = [(p, n) for p, n, kind in found if kind == "qwen"]     # for the image edit window
        self.image_loras = {fam: [(p, n) for p, n, kind in found if kind == fam]
                            for fam in ("flux1", "sdxl", "krea2", "qwen21", "moss")}
        self._scan_triggers()
        if report:
            self.status.setText(f"{plural(len(self.h3_loras), 'LoRA')} and "
                                f"{plural(len(self.turbo_loras), 'turbo LoRA')} in "
                                "ComfyUI's LoRA folders.")
        if self.tree is not None:
            self._fill_turbo((self.tree.options.get("turbo") or {}).get("lora"))
        if self.tree is None:
            return
        self._refresh_card_choices()

    def _refresh_card_choices(self):
        """The cards' lists show the project's LoRA, length and steps as 'default': update them in place."""
        if self.tree is None:
            return
        known = self.known_loras()
        for card in [c for c in self.cards if isinstance(c, PromptCard)]:
            card.set_choices(int(self.tree.options.get("frames", 73)), int(self.tree.options.get("steps", 28)), known)

    def _lora_dir(self) -> str:
        """Where to start looking for LoRA files: ComfyUI's first LoRA folder (the ones it can load)."""
        try:
            roots = CMF.comfy_lora_roots(Path(os.path.expanduser(self.comfy_dir.text().strip()
                                                                 or UL.DEFAULT_COMFY_DIR)))
            extra = [r for r in roots if "models/loras" not in r.as_posix()]
            return str((extra or roots)[0]) if roots else ""
        except Exception:  # noqa: BLE001
            return ""

    def _remember_lora(self, path: str):
        try:
            recent = json.loads(qs_value(self.qs, "recentLoras", "[]", str))
        except ValueError:
            recent = []
        recent = [path] + [r for r in recent if r != path]
        self.qs.setValue("recentLoras", json.dumps(recent[:MAX_RECENT_LORAS]))

    def _overrides_changed(self, nid: str, ov: dict):
        if self.tree and nid in self.tree.nodes and not self.tree.nodes[nid].frozen:
            before = self.tree.nodes[nid].overrides or {}
            self.tree.nodes[nid].overrides = dict(ov)
            self._failures.pop(nid, None)
            if self.tree.nodes[nid].made and any(before.get(k) != ov.get(k) for k in ("sound_over",
                                                                                       "sound_mix_level")):
                # (a made clip's sound may only need mixing again: the mode at once, the level once it settles)
                self._remix_stuck = set()
                self._remix_timer.start(0 if before.get("sound_over") != ov.get("sound_over") else REMIX_DELAY_MS)
            self._changed()

    def _lora_picked(self, nid: str, path: str):
        """A LoRA chosen on a card: remembered (a file from elsewhere shows up in the lists later); a warning if
        ComfyUI cannot load it."""
        listed = {os.path.realpath(p) for p, _n in self.h3_loras}
        if os.path.realpath(path) not in listed:
            self._remember_lora(path)
        if self.in_comfy_loras(path) is False:
            self.status.setText(f"{Path(path).name} is not in ComfyUI's LoRA folders, so ComfyUI cannot load it: copy "
                                f"or link it into {self._lora_dir() or 'its models/loras folder'} (then Rescan LoRAs).")

    def _choose_node_lora(self, nid: str):
        """Add a LoRA file from any folder to a clip's stack."""
        if self.tree is None or nid not in self.tree.nodes:
            return
        stack = T.node_loras(self.tree.nodes[nid])
        start = str(Path(stack[-1][0]).parent) if stack else self._lora_dir()
        f, _ = QFileDialog.getOpenFileName(self, "LoRA for this clip", start, "LoRA (*.safetensors)")
        if f:
            self.add_node_lora(nid, f)

    def add_node_lora(self, nid: str, path: str, strength: float = 1.0):
        """Put a LoRA file at the end of a clip's stack."""
        if self.tree is None or self.running or nid not in self.tree.nodes or self.tree.nodes[nid].frozen:
            return
        ov: dict[str, Any] = dict(self.tree.nodes[nid].overrides)
        p = lora_path(path)
        ov["loras"] = list(ov.get("loras") or []) + [{"path": p, "strength": float(strength)}]
        self._remember_lora(p)
        self.tree.nodes[nid].overrides = ov
        self._changed()
        self.refresh(rebuild_path=True)
        if self.in_comfy_loras(p) is False:
            self.status.setText(f"{Path(p).name} is not in ComfyUI's LoRA folders, so ComfyUI cannot load it: copy or "
                                f"link it into {self._lora_dir() or 'its models/loras folder'} (then Rescan LoRAs).")
        else:
            self.status.setText(f"Clip #{T.short_id(nid)}: added {Path(p).stem}.")

    def in_comfy_loras(self, path: str) -> bool | None:
        """Whether ComfyUI can load this LoRA file: it lies in one of ComfyUI's LoRA folders (links followed), or
        is a LoRA found there. None: ComfyUI's folder is not known here."""
        try:
            roots = CMF.comfy_lora_roots(Path(os.path.expanduser(self.comfy_dir.text().strip()
                                                                 or UL.DEFAULT_COMFY_DIR)))
        except Exception:  # noqa: BLE001
            roots = []
        if not roots:
            return None
        real = os.path.realpath(path)
        for r in roots:
            for top in {str(r), os.path.realpath(r)}:
                if os.path.abspath(path).startswith(top.rstrip(os.sep) + os.sep) or \
                        real.startswith(top.rstrip(os.sep) + os.sep):
                    return True
        return any(os.path.realpath(q) == real for q, _n in self.h3_loras + self.turbo_loras)

    def _seed_changed(self, nid: str, seed: int):
        if self.tree and nid in self.tree.nodes and not self.tree.nodes[nid].frozen:
            self.tree.nodes[nid].seed = int(seed)
            self._changed()

    def _new_seed(self) -> int:
        assert self.tree is not None
        return self.tree.new_seed()

    def _wait_note(self) -> bool:
        """True (and the status says why) while generating locks the project."""
        if self.running:
            self.status.setText("Wait until generating has finished (Stop cancels it).")
        return self.running

    def find_lora(self, key: str) -> str | None:
        """A LoRA file whose name contains key: among the LoRAs found (H3 and turbo ones), else straight in ComfyUI's
        LoRA folders (a file added after the last scan)."""
        key = key.lower()
        for p, name in self.h3_loras + self.turbo_loras:
            if key in name.lower():
                return p
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        miss = self._lora_misses
        now = time.monotonic()
        hit = self._lora_hits.get((key, comfy_dir))
        if hit is not None and now - hit[1] < 30 and Path(hit[0]).is_file():
            return hit[0]                            # (found a moment ago: no new walk through the folders)
        if now - miss.get((key, comfy_dir), -1e9) < 30:
            return None                              # (looked for a moment ago: not there)
        try:
            for root in CMF.comfy_lora_roots(comfy_dir):
                for f in sorted(Path(root).rglob("*.safetensors")):
                    if key in f.name.lower():
                        self._lora_hits = {**self._lora_hits, (key, comfy_dir): (str(f), now)}
                        return str(f)
        except Exception:  # noqa: BLE001  (no ComfyUI folder set, a folder offline)
            pass
        self._lora_misses = {**miss, (key, comfy_dir): time.monotonic()}
        return None
