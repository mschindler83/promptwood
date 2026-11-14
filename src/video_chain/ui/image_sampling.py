"""The image dialog's sampling controls (Create): steps, CFG, sampler, scheduler (each "auto" = what the model uses by
itself), the negative prompt's visibility and the note about odd settings.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QSpinBox

from ..imagegen import FAMILIES, SAMPLERS, SCHEDULERS, default_sampling
from .widgets import guard_wheel, icon_button, label
from .image_host import ImageEditHost
from .qsvalue import qs_value



class SamplingMixin(ImageEditHost):
    # ---- sampling
    def _build_sampling(self, cv):
        row = QHBoxLayout()
        self.steps = QSpinBox()
        self.steps.setRange(0, 200)
        self.steps.setToolTip("Sampling steps (auto: the model's own for this quality). More steps rarely help a "
                              "distilled / Turbo / Lightning model and cost time in proportion")
        self.cfg = QDoubleSpinBox()
        self.cfg.setRange(0.0, 30.0)
        self.cfg.setSingleStep(0.1)
        self.cfg.setDecimals(1)
        self.cfg.setToolTip("CFG (auto: the model's own). Above 1 the negative prompt counts and each step runs "
                            "twice (about twice as slow); distilled / Turbo / Lightning models want 1")
        self.sampler = QComboBox()
        self.sampler.setToolTip("The sampler (auto: the model's own)")
        self.scheduler = QComboBox()
        self.scheduler.setToolTip("The scheduler: how the noise levels are spread over the steps (auto: the "
                                  "model's own; Qwen-Image 2.1 Turbo uses a fixed list of its own, any choice here "
                                  "replaces it)")
        b_reset = icon_button("rotate", "Back to the model's own settings (all auto)")
        b_reset.clicked.connect(lambda: self.set_sampling({}))
        for spin in (self.steps, self.cfg):
            guard_wheel(spin)
        grid = QGridLayout()                       # two rows, so the panel stays narrow
        for k, (w, text) in enumerate(((self.steps, "Steps"), (self.cfg, "CFG"), (self.sampler, "Sampler"),
                                       (self.scheduler, "Scheduler"))):
            grid.addWidget(label(text, level="tertiary"), k // 2, (k % 2) * 2)
            grid.addWidget(w, k // 2, (k % 2) * 2 + 1)
            w.setMinimumWidth(60)
            if isinstance(w, QComboBox):
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(8)
        grid.addWidget(b_reset, 0, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        row.addLayout(grid, 1)
        cv.addLayout(row)
        self.sampling_note = label("", level="tertiary")
        self.sampling_note.setWordWrap(True)
        cv.addWidget(self.sampling_note)
        self.steps.valueChanged.connect(lambda *_: self._sampling_changed())
        self.cfg.valueChanged.connect(lambda *_: self._sampling_changed())
        self.sampler.currentIndexChanged.connect(lambda *_: self._sampling_changed())
        self.scheduler.currentIndexChanged.connect(lambda *_: self._sampling_changed())
        self.model_file.currentIndexChanged.connect(lambda *_: self._sampling_defaults())
        self._fill_sampling_lists()

    def _fill_sampling_lists(self):
        lists = (self.families or {}).get("sampling") or {}
        for combo, names in ((self.sampler, lists.get("samplers") or SAMPLERS),
                             (self.scheduler, lists.get("schedulers") or SCHEDULERS)):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("auto", None)
            for n in names:
                combo.addItem(n, n)
            i = combo.findData(cur)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)

    def sampling(self) -> dict:
        """The chosen sampling settings; a missing key: the model's own."""
        out: dict[str, Any] = {}
        if self.steps.value():
            out["steps"] = self.steps.value()
        if self.cfg.value():
            out["cfg"] = round(self.cfg.value(), 2)
        if self.sampler.currentData():
            out["sampler"] = self.sampler.currentData()
        if self.scheduler.currentData():
            out["scheduler"] = self.scheduler.currentData()
        return out

    def set_sampling(self, sp: dict):
        self._sampling_loading = True
        try:
            self.steps.setValue(int(sp.get("steps") or 0))
            self.cfg.setValue(float(sp.get("cfg") or 0))
            for combo, key in ((self.sampler, "sampler"), (self.scheduler, "scheduler")):
                name = sp.get(key)
                if name and combo.findData(name) < 0:
                    combo.addItem(name, name)          # (one this ComfyUI does not list: kept, ComfyUI says no)
                combo.setCurrentIndex(max(0, combo.findData(name)) if name else 0)
        finally:
            self._sampling_loading = False
        self._sampling_changed()

    def _sampling_key(self) -> str:
        return f"createSampling_{self.family.currentData()}"

    def _sampling_defaults(self):
        """The model's own settings shown as "auto (…)"; the family's remembered settings when it changed."""
        if not self._built:
            return
        fam = self.family.currentData()
        if self.sampler.count() <= 1 or (self.families or {}).get("sampling"):
            self._fill_sampling_lists()
        if fam != self._sampling_family:
            self._sampling_family = fam
            saved = {}
            if self.qs is not None and fam:
                try:
                    saved = json.loads(qs_value(self.qs, self._sampling_key(), "{}", str) or "{}")
                except ValueError:
                    saved = {}
            self.set_sampling(saved if isinstance(saved, dict) else {})
        refs = sum(1 for s in self.cpics[:FAMILIES.get(fam, {}).get("refs", 0)] if s.path) \
            if self.use_refs.isChecked() else 0
        auto = default_sampling(fam, (self.families or {}).get(fam) or {}, self.model_file.currentData(),
                                self.quality.currentData() or "", refs, self.ref_strength.value()) if fam in FAMILIES \
            else None
        self._sampling_auto = auto or {}
        a = self._sampling_auto
        self.steps.setSpecialValueText(f"auto ({a['steps']})" if a else "auto")
        self.cfg.setSpecialValueText(f"auto ({a['cfg']:g})" if a else "auto")
        self.sampler.setItemText(0, f"auto ({a['sampler']})" if a else "auto")
        self.scheduler.setItemText(0, f"auto ({a['scheduler']})" if a else "auto")
        self._sampling_note()
        self._show_negative()                       # (a model's own CFG above 1 shows the negative prompt)

    def _sampling_changed(self):
        self._show_negative()
        if self._sampling_loading:
            return
        if self.qs is not None and self.family.currentData():
            self.qs.setValue(self._sampling_key(), json.dumps(self.sampling()))
        self._sampling_note()

    def _show_negative(self):
        """The negative prompt: for families that use one, and for the others (made for CFG 1, where a negative
        prompt does nothing) as soon as CFG is set above 1."""
        if not self._built:
            return
        f = FAMILIES.get(self.family.currentData()) if self.mode() == "create" else None
        own = bool(f and f["negative"])
        cfg_on = (self.cfg.value() or self._sampling_auto.get("cfg", 1)) > 1
        was = self.negative.isVisibleTo(self)
        self.negative.setVisible(bool(f) and (own or cfg_on))
        if was != self.negative.isVisibleTo(self):
            self._var_timer.start()                 # (variations count the negative only while it is used)
        cfg = self.cfg.value() or self._sampling_auto.get("cfg", 1)
        self.negative.setPlaceholderText("Avoid (optional negative prompt; empty: the model's usual one)" if own else
                                         f"Avoid (optional negative prompt; counts because CFG is {cfg:g})"
                                         if cfg_on else "")

    def _sampling_note(self):
        sp, a = self.sampling(), self._sampling_auto
        notes = []
        if a and sp.get("cfg") and sp["cfg"] > 1 and a.get("cfg", 1) <= 1:
            notes.append(f"CFG {sp['cfg']:g} on a model made for CFG 1 (distilled / Turbo / Lightning): about twice "
                         "as slow, colours often burn")
        if a and sp.get("steps") and a.get("steps") and sp["steps"] < a["steps"] * 0.6:
            notes.append(f"{sp['steps']} steps is far below the model's own {a['steps']}: expect unfinished, "
                         "noisy images")
        if a and sp.get("scheduler") and a.get("scheduler") == "manual":
            notes.append("replaces Qwen-Image 2.1 Turbo's own sigma list")
        self.sampling_note.setText(("Own settings: " + "; ".join(notes) + ".") if notes else
                                   "Own settings in use (each auto field uses the model's own)." if sp else "")
        self.sampling_note.setVisible(bool(self.sampling_note.text()))
