"""The image dialog's modes (edit / create) and the models ComfyUI has: qualities, families, model files.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer

from ..comfy import EDIT_MODELS
from ..edits import MOSS_EDIT_QUALITIES, QUALITY_TEXT, lora_note
from ..imagegen import FAMILIES, size_for
from .image_common import CREATE_HINT, PROMPT_HINT
from .widgets import safe_emit


class ModelsMixin:
    # ---- models
    def _mode_clicked(self, i: int):
        mode = "create" if i == 1 else "edit"
        if self.qs is not None:
            self.qs.setValue("imageMode", mode)       # remembered only when the user chooses it
        self.set_mode(mode)

    def mode(self) -> str:
        return "create" if self.m_create.isChecked() else "edit"

    def set_mode(self, mode: str):
        create = mode == "create" and self.connect_create is not None
        (self.m_create if create else self.m_edit).setChecked(True)

        self.setWindowTitle("Create a new image with AI" if create else "Edit an image with AI")
        for w in (self.edit_box, self.edit_pics, self.keep_size):
            w.setVisible(not create)
        self.create_box.setVisible(create)
        self.prompt_title.setText("2  Describe the image" if create else "2  What should change?")
        self.prompt.setPlaceholderText(CREATE_HINT if create else PROMPT_HINT)
        self.b_refine.setToolTip("Refine this prompt with a language model")
        self.quality.blockSignals(True)
        self.quality.clear()
        if create:
            fam = self.family.currentData()
            files = self.families.get(fam) or {}
            need = {"qwen21": {"fast": "turbo", "best": "base"}, "chroma": {"fast": "flash", "best": "hd"}}.get(fam, {})
            for key, text in (FAMILIES[fam]["qualities"].items() if fam in FAMILIES else []):
                if "models" in files and key in need and not files.get(need[key]):
                    continue                       # its model file is not installed
                self.quality.addItem(text, key)
        elif self.edit_model.currentData() in ("qwen21", "moss"):
            qs = FAMILIES["qwen21"]["qualities"] if self.edit_model.currentData() == "qwen21" else MOSS_EDIT_QUALITIES
            for key, text in qs.items():
                self.quality.addItem(text, key)
        else:
            for q in ("fast", "standard", "best"):
                self.quality.addItem(QUALITY_TEXT[q], q)
        saved = self.qs.value(self._quality_key(), "", type=str) if self.qs is not None else ""
        i = self.quality.findData(saved)
        self.quality.setCurrentIndex(i if i >= 0 else 0)
        self.quality.blockSignals(False)
        self.loras.set_items(self._known(), self._remembered_loras())
        self._create_widgets()
        self._variations_changed()
        if create and self._families_error:
            self.state.setText(f"Cannot list the models: {self._families_error}")
            self.b_start_comfy.setVisible("not reachable" in self._families_error and self.start_comfy is not None)
        self._update_buttons()

    def _quality_key(self) -> str:
        if self.mode() == "create":
            return f"createQuality_{self.family.currentData()}"
        return "editQuality" if self.edit_model.currentData() == "2511" else f"editQuality_{self.edit_model.currentData()}"

    def _edit_model_changed(self, *_):
        if self.mode() == "edit":
            self.set_mode("edit")                # its qualities and LoRAs
        self._show_mask()
        self._update_buttons()

    def _show_edit_models(self):
        """Qwen-Image 2.1 and MOSS-Image can be chosen only when ComfyUI has them (known from Create's model list;
        MOSS also needs its edit LoRA)."""
        if not self.families:                      # not listed (yet, or ComfyUI is gone): all offered again
            for i in range(self.edit_model.count()):
                self.edit_model.model().item(i).setEnabled(True)
                self.edit_model.setItemText(i, EDIT_MODELS[self.edit_model.itemData(i)])
            return
        for key in ("2511", "qwen21", "moss"):
            info = self.families.get({"2511": "edit2511"}.get(key, key)) or {}
            k = self.edit_model.findData(key)
            ok = "models" in info and (key != "moss" or bool(info.get("edit_lora")))
            why = info.get("missing") or ("MOSS-Image's edit LoRA (moss_image_edit_lora_rank_64_bf16) is missing"
                                          if key == "moss" else "")
            self.edit_model.model().item(k).setEnabled(ok)
            self.edit_model.setItemText(k, EDIT_MODELS[key] + ("" if ok else "  (not available)"))
            self.edit_model.setItemData(k, EDIT_MODELS[key] if ok else why, Qt.ToolTipRole)
        cur = self.edit_model.currentIndex()
        if not self.edit_model.model().item(cur).isEnabled():   # the chosen one is not there: the first one that is
            ok = [i for i in range(self.edit_model.count()) if self.edit_model.model().item(i).isEnabled()]
            if ok:
                self.edit_model.setCurrentIndex(ok[0])

    def _remember_quality(self):
        if self.qs is not None and self.quality.currentData():
            self.qs.setValue(self._quality_key(), self.quality.currentData())

    def _load_families(self):
        sig, connect = self._sig, self.connect_create

        def work():
            try:
                sig.families.emit(connect(None).files, "")
            except Exception as e:  # noqa: BLE001
                safe_emit(sig.families, {}, str(e))
        threading.Thread(target=work, daemon=True).start()

    def _families_found(self, files: dict, error: str):
        before = self.family.currentData()
        kept = self.loras.items() if self.mode() == "create" and before not in FAMILIES else []
        self.families = files
        self._families_error = error
        if error and self._retry_families > 0:           # ComfyUI is starting: ask again in a few seconds
            self._retry_families -= 1
            QTimer.singleShot(5000, self._load_families)
        elif not error and self._retry_families >= 0 and self._was_down:
            self.state.setText("ComfyUI is running: the models are listed.")
        self._was_down = bool(error)
        self.family.blockSignals(True)
        self.family.clear()
        if error:
            self.family.addItem("ComfyUI is not reachable", None)
            if self.mode() == "create":
                self.state.setText(f"Cannot list the models: {error}")
                self.b_start_comfy.setVisible("not reachable" in error and self.start_comfy is not None)
        for key, f in FAMILIES.items():
            info = files.get(key) or {}
            self.family.addItem(f["label"] + ("" if "models" in info else "  (not available)"), key)
            k = self.family.count() - 1
            self.family.setItemData(k, info.get("missing", f["label"]), Qt.ToolTipRole)
            if "models" not in info:
                self.family.model().item(k).setEnabled(False)
        saved = self.qs.value("createFamily", "", type=str) if self.qs is not None else ""
        avail = [i for i in range(self.family.count()) if self.family.model().item(i).isEnabled()
                 and self.family.itemData(i)]
        i = self.family.findData(saved)
        self.family.setCurrentIndex(i if i in avail else (avail[0] if avail else 0))
        self.family.blockSignals(False)
        self._show_edit_models()
        self._family_changed()
        if kept and not self.loras.items():         # chosen before the list came: keep them
            self.loras.set_items(self._known(), kept)
            self._remember_loras()

    def _family_changed(self, *_):
        fam = self.family.currentData()
        self.model_file.clear()
        for m in (self.families.get(fam) or {}).get("models") or []:
            self.model_file.addItem(Path(m).stem, m)
            self.model_file.setItemData(self.model_file.count() - 1, m, Qt.ToolTipRole)
        if self.mode() == "create":
            self.set_mode("create")              # qualities, LoRAs of this family

    def _create_widgets(self):
        fam = self.family.currentData()
        f = FAMILIES.get(fam)
        create = self.mode() == "create"
        refs = create and self.use_refs.isChecked() and f is not None
        n = f["refs"] if f else 0
        moss_no_edit = fam == "moss" and "models" in (self.families.get("moss") or {}) \
            and not (self.families.get("moss") or {}).get("edit_lora")
        if moss_no_edit:
            n = 0                                  # pictures go through MOSS's edit LoRA
        for k, slot in enumerate(self.cpics):
            slot.setVisible(refs and k < n)
        self.ref_text.setVisible(refs)
        self.ref_text.setText("MOSS-Image needs its edit LoRA (moss_image_edit_lora_rank_64_bf16) for pictures"
                              if moss_no_edit else f["ref_text"] if f else "")
        img2img = refs and fam in ("flux1", "sdxl", "chroma")
        self.ref_strength.setVisible(img2img)
        self._sampling_defaults()
        self.ref_strength_label.setVisible(img2img)
        self._show_negative()
        self.tags.setVisible(create and fam == "sdxl")
        if f:
            w, h = size_for(fam, self.aspect.currentData() or 1.0)
            self.size_text.setText(f"{w}×{h}")
        self.lora_text.setText(lora_note("create" if create else "edit", f, fam, self.edit_model.currentData(),
                                         self.family_loras))
        self.b_gen.setToolTip("Make the image (Ctrl+Enter)" if create else
                              "Edit the image with this prompt (Ctrl+Enter)")
        self._update_buttons()
