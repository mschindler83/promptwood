"""The image dialog's LoRAs: which are offered for the mode / model, what is remembered per mode and family, trigger
words and the one-image-per-LoRA runs.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..edits import LORA_FAMILY
from ..triggers import stack_words
from .lora_stack import lora_label


class LorasMixin:
    # ---- loras
    def _trigger_words(self, loras: list | None = None) -> list[str]:
        """The trigger words of the chosen LoRAs, or of `loras` [(path, strength)] (put before the prompt when the
        project wants them)."""
        if not self.use_triggers:
            return []
        return stack_words([it["path"] for it in self.loras.items()] if loras is None else [p for p, _s in loras])

    def _lora_runs(self) -> list[list[tuple[str, float]]]:
        """The LoRAs of each image in turn: one list with all of them, or (One image per LoRA) each alone."""
        loras = [(it["path"], it["strength"]) for it in self.loras.items()]
        if self.lora_each.isChecked() and loras:
            return [[lo] for lo in loras]
        return [loras]

    def _lora_key(self) -> str:
        if self.mode() == "create":
            return f"createLoras_{self.family.currentData()}"
        return "editLoras" if self.edit_model.currentData() == "2511" else f"editLoras_{self.edit_model.currentData()}"

    def _known(self) -> list[tuple[str, str]]:
        """The LoRAs ComfyUI has for this mode (edit: Qwen; create: the chosen family), then LoRA files chosen here
        earlier (from other folders)."""
        if self.mode() == "create":
            fam = self.family.currentData()
            out = list(self.family_loras.get(LORA_FAMILY.get(fam, fam), []))
        elif self.edit_model.currentData() in ("qwen21", "moss"):
            out = list(self.family_loras.get(self.edit_model.currentData(), []))   # their own (found by contents)
        else:
            out = list(self.known_loras)
        listed = {p for p, _ in out}
        for p in self._recent_files():
            if p not in listed and Path(p).is_file():
                out.append((p, lora_label(p)))
        return out

    def _recent_key(self) -> str:
        return self._lora_key().replace("Loras", "LoraFiles")       # per mode / family

    def _recent_files(self) -> list[str]:
        try:
            return [p for p in json.loads(self.qs.value(self._recent_key(), "[]", type=str)) if isinstance(p, str)]
        except (ValueError, AttributeError):
            return []

    def _remembered_loras(self) -> list[dict]:
        try:
            items = json.loads(self.qs.value(self._lora_key(), "[]", type=str)) if self.qs is not None else []
        except ValueError:
            items = []
        return [it for it in items if isinstance(it, dict) and it.get("path") and Path(it["path"]).is_file()]

    def _remember_loras(self):
        if self.qs is not None:
            self.qs.setValue(self._lora_key(), json.dumps(self.loras.items()))

    def _lora_picked(self, path: str):
        """A file from another folder chosen in the picker: offered again next time (in this mode / family); a
        warning if ComfyUI cannot load it."""
        fam = self.family.currentData()
        listed = {p for p, _ in (self.family_loras.get(LORA_FAMILY.get(fam, fam), []) if self.mode() == "create"
                                 else self.known_loras)}
        if path not in listed and self.qs is not None:
            self.qs.setValue(self._recent_key(), json.dumps([path] + [r for r in self._recent_files() if r != path][:11]))
        if self.check_lora is not None and self.check_lora(path) is False:
            self.state.setText(f"{Path(path).name} is not in ComfyUI's LoRA folders, so ComfyUI cannot load it: "
                               "copy or link it there (then Rescan LoRAs in the main window).")

    def add_lora_file(self, path: str, strength: float = 1.0):
        """A LoRA file from any folder at the end of the list (ComfyUI must find it in its LoRA folders)."""
        p = os.path.abspath(path)
        if self.qs is not None:
            self.qs.setValue(self._recent_key(), json.dumps([p] + [r for r in self._recent_files() if r != p][:11]))
        self.loras.set_items(self._known(), self.loras.items() + [{"path": p, "strength": float(strength)}])
        self._remember_loras()
