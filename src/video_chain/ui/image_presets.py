"""The image dialog's LoRA presets (Create): named prompt + negative + quality + sampling + strength, several per LoRA.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QInputDialog

from .. import presets
from ..imagegen import FAMILIES
from .dialogs import confirm as ask, failed
from .lora_stack import lora_label
from .widgets import icon_button, label


class PresetsMixin:
    # ---- presets
    def _build_presets(self, cv):
        row = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.setToolTip("Presets of the LoRAs above: choosing one puts its prompt, negative prompt, quality, "
                               "sampling settings and LoRA strength into the fields")
        self.preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset.setMinimumContentsLength(12)
        self.preset.activated.connect(lambda _i: self.apply_preset())
        self.preset.currentIndexChanged.connect(lambda *_: self._preset_buttons())
        self.b_preset_new = icon_button("plus", "Save the prompt and these settings as a new preset of a LoRA")
        self.b_preset_new.clicked.connect(lambda: self.save_preset_as())
        self.b_preset_save = icon_button("save", "Save the current prompt and settings into the chosen preset")
        self.b_preset_save.clicked.connect(lambda: self.update_preset())
        self.b_preset_rename = icon_button("pen", "Rename the chosen preset")
        self.b_preset_rename.clicked.connect(lambda: self.rename_preset())
        self.b_preset_delete = icon_button("trash", "Delete the chosen preset")
        self.b_preset_delete.clicked.connect(lambda: self.delete_preset())
        row.addWidget(label("Preset", level="tertiary"))
        row.addWidget(self.preset, 1)
        for b in (self.b_preset_new, self.b_preset_save, self.b_preset_rename, self.b_preset_delete):
            row.addWidget(b)
        cv.addLayout(row)
        self._fill_presets()

    def _preset_loras(self) -> list[str]:
        return [it["path"] for it in self.loras.items() if it.get("path")]

    def _fill_presets(self, select: tuple[str, str] | None = None):
        if not self._built:
            return
        cur = "\n".join(select) if select else self.preset.currentData()
        self.preset.blockSignals(True)
        self.preset.clear()
        loras = self._preset_loras()
        found = [(lo, p) for lo in loras for p in presets.store().list(lo)]
        self.preset.addItem("— choose a preset —" if found else
                            "no presets for these LoRAs (+ saves one)" if loras else "add a LoRA for its presets", None)
        for lo, p in found:
            text = p["name"] if len(loras) == 1 else f"{lora_label(lo)}: {p['name']}"
            self.preset.addItem(text, f"{lo}\n{p['name']}")         # (a string: findData cannot match tuples)
            self.preset.setItemData(self.preset.count() - 1, (p.get("prompt") or "")[:300], Qt.ToolTipRole)
        i = self.preset.findData(cur) if cur else -1
        self.preset.setCurrentIndex(max(0, i))
        self.preset.blockSignals(False)
        self._preset_buttons()

    def _chosen_preset(self) -> tuple[str, str] | None:
        data = self.preset.currentData()
        return tuple(data.rsplit("\n", 1)) if data else None

    def _preset_buttons(self):
        chosen = self.preset.currentData() is not None
        has = bool(self._preset_loras())
        self.b_preset_new.setEnabled(has)
        self.b_preset_new.setToolTip("Save the prompt and these settings as a new preset of a LoRA" if has else
                                     "Presets belong to a LoRA: add a LoRA above first")
        for b, tip in ((self.b_preset_save, "Save the current prompt and settings into the chosen preset"),
                       (self.b_preset_rename, "Rename the chosen preset"),
                       (self.b_preset_delete, "Delete the chosen preset")):
            b.setToolTip(tip if chosen else tip + " (choose a preset first)")
        for b in (self.b_preset_save, self.b_preset_rename, self.b_preset_delete):
            b.setEnabled(chosen)

    def _preset_values(self, lora: str) -> dict:
        strength = next((it["strength"] for it in self.loras.items()
                         if os.path.realpath(it["path"]) == os.path.realpath(lora)), 1.0)
        var = {k: v for k, v in self.variations.state().items() if k != "positions"}
        return {"variations": var, "prompt": self.prompt.toPlainText().strip(), "negative": self.negative.text().strip(),
                "quality": self.quality.currentData(), "sampling": self.sampling(), "strength": strength,
                "family": self.family.currentData()}

    def apply_preset(self):
        """The chosen preset into the fields: prompt, negative, quality, sampling, its LoRA's strength."""
        data = self._chosen_preset()
        if not data:
            return
        p = presets.store().get(*data)
        if p is None:
            self._fill_presets()
            return
        if p.get("prompt") is not None:
            self.prompt.setPlainText(p["prompt"])
        if p.get("negative") is not None:
            self.negative.setText(p["negative"])
        other = p.get("family") and p["family"] != self.family.currentData()
        if p.get("variations") is not None:
            self.variations.set_state(p["variations"])
        q = self.quality.findData(p.get("quality")) if not other else -1
        if q >= 0:
            self.quality.setCurrentIndex(q)
        if not other:                               # (sampling settings of another model would not fit this one)
            self.set_sampling(p.get("sampling") or {})
        if p.get("strength") is not None:
            items = [dict(it, strength=float(p["strength"])) if os.path.realpath(it["path"]) ==
                     os.path.realpath(data[0]) else it for it in self.loras.items()]
            self.loras.set_items(self._known(), items)
            self._remember_loras()
            self._fill_presets(data)
        self.state.setText(f"Preset “{p['name']}” of {lora_label(data[0])} is in the fields."
                           + (f" It was made with {FAMILIES.get(p['family'], {}).get('label', p['family'])}: its "
                              "quality and sampling settings were left as they are." if other else ""))

    def save_preset_as(self, name: str | None = None, lora: str | None = None) -> bool:
        """A new preset from the fields (asks for its name, and for the LoRA when there are several)."""
        loras = self._preset_loras()
        if not loras:
            return False
        if lora is None:
            if len(loras) == 1:
                lora = loras[0]
            else:
                labels = [lora_label(lo) for lo in loras]
                cur = self._chosen_preset()
                start = loras.index(cur[0]) if cur and cur[0] in loras else 0
                pick, ok = QInputDialog.getItem(self, "New preset", "The preset belongs to the LoRA:", labels, start,
                                                False)
                if not ok:
                    return False
                lora = loras[labels.index(pick)]
        if name is None:
            name, ok = QInputDialog.getText(self, "New preset", f"Name of the new preset of {lora_label(lora)}:")
            if not ok:
                return False
        name = " ".join(str(name).split())
        if not name:
            return False
        if presets.store().get(lora, name) is not None and not ask(
                self, "New preset", f"{lora_label(lora)} already has a preset “{name}”. Replace it?"):
            return False
        try:
            presets.store().save(lora, name, self._preset_values(lora))
        except (OSError, ValueError) as e:
            failed(self, "New preset", "save the preset", e)
            return False
        self._fill_presets((lora, name))
        self.state.setText(f"Saved the preset “{name}” of {lora_label(lora)}.")
        return True

    def update_preset(self) -> bool:
        """The chosen preset takes the current prompt and settings."""
        data = self._chosen_preset()
        if not data:
            return False
        try:
            presets.store().save(data[0], data[1], self._preset_values(data[0]))
        except (OSError, ValueError) as e:
            failed(self, "Preset", "save the preset", e)
            return False
        self._fill_presets(data)
        self.state.setText(f"Saved the changes to the preset “{data[1]}”.")
        return True

    def rename_preset(self, new: str | None = None) -> bool:
        data = self._chosen_preset()
        if not data:
            return False
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename preset", "New name:", text=data[1])
            if not ok:
                return False
        new = " ".join(str(new).split())
        if not new or new == data[1]:
            return False
        try:
            presets.store().rename(data[0], data[1], new)
        except (OSError, ValueError) as e:
            failed(self, "Rename preset", "rename it", e)
            return False
        self._fill_presets((data[0], new))
        return True

    def delete_preset(self, confirm: bool = True) -> bool:
        data = self._chosen_preset()
        if not data:
            return False
        if confirm and not ask(self, "Delete preset", f"Delete the preset “{data[1]}” of "
                                                f"{lora_label(data[0])}?"):
            return False
        try:
            presets.store().delete(*data)
        except OSError as e:
            failed(self, "Delete preset", "delete it", e)
            return False
        self._fill_presets()
        self.state.setText(f"Deleted the preset “{data[1]}”.")
        return True
