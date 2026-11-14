"""The dialogs of the project menu: new project, variants of a clip, export of a path."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLineEdit, QSpinBox)

from .. import tree as T
from ..comfy import refine_size
from .lora_stack import LORA_MAX, LORA_MIN, lora_label
from .widgets import guard_wheel, icon_button, label, remember_geometry
from . import image_picker as IP
from .ui_logic import repeated_variants, variant_strengths


class NewProjectDialog(QDialog):
    """Name + location (+ an optional start image) for a new project."""

    def __init__(self, parent, location: str, settings: QSettings, image: str = ""):
        super().__init__(parent)
        self.qs = settings                 # (the image picker keeps its folders and cache in it)
        self.setWindowTitle("New project")
        self.setMinimumWidth(520)
        remember_geometry(self, settings, "new")
        f = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. smoke-scene-1")
        f.addRow("Name", self.name)
        lrow = QHBoxLayout()
        self.location = QLineEdit(location)
        b = icon_button("fopen", "Choose the folder the project is created in")
        b.clicked.connect(self._choose_location)
        lrow.addWidget(self.location, 1)
        lrow.addWidget(b)
        f.addRow("Location", lrow)
        irow = QHBoxLayout()
        self.image = QLineEdit(image)
        self.image.setPlaceholderText("optional: start image")
        b2 = icon_button("fopen", "Choose the start image")
        b2.clicked.connect(self._choose_image)
        irow.addWidget(self.image, 1)
        irow.addWidget(b2)
        f.addRow("Start image", irow)
        self.target = label("", level="tertiary")
        self.target.setWordWrap(True)
        f.addRow("", self.target)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        f.addRow(self.buttons)
        for w in (self.name, self.location):
            w.textChanged.connect(self._check)
        self._check()

    def folder(self) -> Path | None:
        name = "".join(ch for ch in self.name.text().strip() if ch not in '/\\:*?"<>|').strip()
        loc = self.location.text().strip()
        if not name or name.startswith(".") or name.endswith(".") or not loc:    # ("." / ".." are no names)
            return None
        return Path(os.path.expanduser(loc)) / name

    def _check(self):
        f = self.folder()
        problem = ""
        if f is None:
            problem = "Enter a name (not starting or ending with a dot) and a location."
        elif (f / T.TREE_FILE).exists():
            problem = f"{f} is already a project (use Open)."
        elif f.is_file():
            problem = f"{f} is a file."
        else:
            try:
                if f.is_dir() and any(f.iterdir()):
                    problem = f"{f} exists and is not empty."
            except OSError as e:
                problem = f"{f} cannot be read ({e.strerror or e})."
        self.target.setText(problem or f"Creates {f}")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not problem)

    def _choose_location(self):
        d = QFileDialog.getExistingDirectory(self, "Location", self.location.text().strip())
        if d:
            self.location.setText(d)

    def _choose_image(self):
        start = str(Path(self.image.text()).parent) if self.image.text().strip() else None
        f = IP.pick_image(self, "Start image", self.qs, start)
        if f:
            self.image.setText(f)


class VariantsDialog(QDialog):
    """How many variants of a clip, and whether to generate them right away. Advanced: move each LoRA's strength
    by a step per variant (a strength sweep), and keep the clip's seed so only the strengths differ."""

    def __init__(self, parent, node, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Create variants")
        remember_geometry(self, settings, "variants")
        self.stack = list((node.overrides or {}).get("loras") or [])
        self.loras = [e for e in self.stack if e.get("path")]
        f = QFormLayout(self)
        text = node.prompt if len(node.prompt) < 90 else node.prompt[:87] + "…"
        info = label(f"Clip #{T.short_id(node.id)}: “{text}”\nEach variant is a new branch from the same point with "
                     "the same prompt and settings and another seed.", level="secondary")
        info.setWordWrap(True)
        f.addRow(info)
        self.count = QSpinBox()
        self.count.setRange(1, 20)
        self.count.setValue(3)
        f.addRow("Variants", self.count)
        self.generate = QCheckBox("Generate them now")
        self.generate.setChecked(True)
        f.addRow("", self.generate)

        self.advanced = QGroupBox("Advanced: change the LoRA strengths per variant")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        self.advanced.setToolTip("Variant 1 gets each strength plus its step, variant 2 plus twice the step … "
                                 "(a negative step lowers it), e.g. to find the strength that works best")
        av = QFormLayout(self.advanced)
        self.steps: list[QDoubleSpinBox] = []
        if not self.loras:
            av.addRow(label("This clip has no LoRAs.", level="tertiary"))
        for e in self.loras:
            spin = QDoubleSpinBox()
            spin.setRange(-5.0, 5.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setPrefix("± ")
            spin.setSuffix(" per variant")
            spin.setMinimumWidth(170)
            spin.setToolTip(f"Now {float(e.get('strength', 1.0)):g}; variant k gets {float(e.get('strength', 1.0)):g} "
                            "+ k × this step")
            spin.valueChanged.connect(self._preview)
            guard_wheel(spin)
            self.steps.append(spin)
            av.addRow(f"{lora_label(e['path'])[:32]} (×{float(e.get('strength', 1.0)):g})", spin)
        self.same_seed = QCheckBox("Keep the clip's seed (only the strengths differ)")
        self.same_seed.setToolTip("Off: every variant gets a new seed too. On: the same seed as the clip, so the "
                                  "variants differ only by their LoRA strengths")
        self.same_seed.setEnabled(bool(self.loras))
        self.same_seed.toggled.connect(self._preview)
        av.addRow("", self.same_seed)
        self.preview = label("", level="tertiary")
        self.preview.setWordWrap(True)
        av.addRow(self.preview)
        f.addRow(self.advanced)
        self.advanced.setEnabled(bool(self.loras))
        if not self.loras:
            self.advanced.setToolTip("This clip has no LoRAs of its own (add them in its settings first)")
        self.advanced.toggled.connect(self._preview)
        self.count.valueChanged.connect(self._preview)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        f.addRow(buttons)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._preview()

    def strength_steps(self) -> list[float]:
        """The step per variant of each LoRA (all 0 when the advanced settings are off)."""
        on = self.advanced.isChecked()
        return [s.value() if on else 0.0 for s in self.steps]

    def keep_seed(self) -> bool:
        return self.advanced.isChecked() and self.same_seed.isChecked() and any(self.strength_steps())

    def _preview(self, *_):
        steps = self.strength_steps()
        if not self.advanced.isChecked() or not any(steps):
            # (the seed option does nothing until a step is set: say so instead of looking active)
            self.preview.setText("Set a step above for at least one LoRA: until then the seed is not kept."
                                 if self.advanced.isChecked() and self.same_seed.isChecked() else "")
            self.ok.setEnabled(True)
            return
        rows = []
        for k in range(1, self.count.value() + 1):
            st = variant_strengths(self.loras, steps, k)
            rows.append(f"#{k}: " + ", ".join(f"×{float(e.get('strength', 1.0)):g}" for e, s in zip(st, steps) if s))
        text = "Strengths: " + " · ".join(rows)
        same = repeated_variants(self.loras, steps, self.count.value())
        if same:
            text += (f"\nVariant{'s' if len(same) > 1 else ''} {', '.join(f'#{k}' for k in same)} "
                     f"repeat{'s' if len(same) == 1 else ''} strengths already there (the range ends at "
                     f"{LORA_MIN:g}…{LORA_MAX:g})" + (": with the same seed that is the same clip again." if
                                                       self.same_seed.isChecked() else "."))
        self.preview.setText(text)
        self.ok.setEnabled(not (same and self.same_seed.isChecked()))


@dataclass
class ExportRequest:
    """What the export dialog asked for."""
    out: Path
    size: str                # "made", "1080p" or "2k"
    fast: bool               # Sol-Attn for the refine
    swapped: bool            # upscale videos with a replaced person too
    strength: str
    method: str              # "seedvr2_7b", "seedvr2_3b" or "refine"
    denoise: bool
    cloud: bool


class ExportDialog(QDialog):
    """Where to save a path, and at which size: as made (quick: the clips are joined), or at 1080p / 2K, upscaled in
    ComfyUI by SeedVR2 (restores the clips as made) or re-rendered by the Open-H3 refine (draws detail anew); every
    clip once, the copies are kept for the next export. ready(size, method, strength, denoise) -> True when every
    clip has a kept copy for that choice (a dict {size: bool} works too)."""
    # (these names shadow QWidget methods, which this class never uses)
    size: QComboBox  # type: ignore[assignment]

    METHODS = (("seedvr2_7b", "SeedVR2 7B — restores the clips as made, clean (default)"),
               ("seedvr2_3b", "SeedVR2 3B — crisper, a little harder"),
               ("refine", "Open-H3 refine — draws detail anew"))

    def __init__(self, parent, default: str, made_size: tuple[int, int] | None, refined,
                 swapped: int = 0, strength: str = "natural", method: str = "seedvr2_7b", denoise: bool = False,
                 cloud: bool | None = None, cloud_on: bool = False, simple: bool = False,
                 settings: QSettings | None = None):
        """cloud: None without a RunPod API key (the option is shown greyed), else whether it can be used;
        cloud_on: its last choice. simple: the Simple view, where cloud upscaling is offered only once it is set up
        (its settings are in the Advanced view)."""
        super().__init__(parent)
        self.setWindowTitle("Export path")
        self.cloud_ok = cloud
        self.setMinimumWidth(620)
        remember_geometry(self, settings, "export")
        self.ready = refined if callable(refined) else (lambda size, _m, _s, _d=False: bool(refined.get(size)))
        self.made_size = made_size
        f = QFormLayout(self)
        row = QHBoxLayout()
        self.file = QLineEdit(default)
        b = icon_button("fopen", "Choose where to save the video")
        b.clicked.connect(self._choose)
        row.addWidget(self.file, 1)
        row.addWidget(b)
        f.addRow("File", row)
        self.size = QComboBox()
        w, h = made_size or (0, 0)
        self.size.addItem(f"As made ({w}×{h}) — quick" if made_size else "As made — quick", "made")
        for key in ("1080p", "2k"):
            self.size.addItem(key, key)
        self.size.setToolTip("As made: the clips are joined as they are. 1080p / 2K: every clip of the path is "
                             "upscaled in ComfyUI (see Upscaler); the copies are kept, so the next export of the "
                             "same clips is quick. Real videos are converted again from their file at that size")
        f.addRow("Size", self.size)
        self.scale_note = label("", level="tertiary")
        self.scale_note.setWordWrap(True)
        f.addRow("", self.scale_note)
        self.method = QComboBox()
        for key, text in self.METHODS:
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(0, self.method.findData(method)))
        self.method.setToolTip(
            "SeedVR2 (video restoration, ComfyUI's own nodes): sharpens and cleans every frame of the clip as it was "
            "made — faces, eyes and colours stay. Measured on an RTX 4090 for a 1.6 s face clip at 1080p: about 33 s, "
            "it changed the clip least of all; 7B was the cleanest, 3B crisper but harder. Long clips are split in "
            "time to fit the GPU memory.\nOpen-H3 refine: re-renders each clip with its prompt and LoRAs after a "
            "latent upscale (3 steps); it invents more new detail but can change faces (see Refine). About 1.7 min "
            "(1080p) and 4-5 min (2K) per 3 s clip")
        f.addRow("Upscaler", self.method)
        self.denoise = QCheckBox("Light denoise before SeedVR2")
        self.denoise.setChecked(bool(denoise))
        self.denoise.setToolTip("Calms the draft's fine noise first (mostly the noise that changes from frame to "
                                "frame, little blur), so SeedVR2 does not sharpen it into grain or a painted texture "
                                "on smooth areas (walls, long hair, skin). Upscaled copies are kept per choice")
        f.addRow("", self.denoise)
        self.cloud = QCheckBox("Upscale on RunPod (a cloud GPU, terminated afterwards)")
        self.cloud.setChecked(bool(cloud_on and cloud))
        self.cloud.setToolTip(
            "SeedVR2 on a RunPod Secure Cloud GPU instead of this computer's: a pod is made for this export (about "
            "1-2 min with a network volume, longer the first time: ComfyUI and the weights are fetched), the clips "
            "go there and back through an SSH tunnel and only ever sit in the pod's RAM, and the pod is terminated "
            "when the export ends (it also removes itself 20 min after a lost connection). RunPod bills the pod by "
            "the second. Settings → Preferences → Cloud upscaling (RunPod)" if cloud is not None else
            "Needs a RunPod API key: Settings → Preferences → Cloud upscaling (RunPod)")
        if simple:
            self.cloud.setToolTip(self.cloud.toolTip() + " (Simple view: switch to the Advanced view first, "
                                  "View → Advanced view, then Settings → Preferences)")
        f.addRow("", self.cloud)
        if simple and cloud is None:               # not set up: not offered; where to set it up is in the tooltip
            f.setRowVisible(self.cloud, False)
            self.size.setToolTip(self.size.toolTip() + ". Upscaling on a cloud GPU (RunPod) can be set up in the "
                                 "Advanced view (View → Advanced view, then Settings → Preferences)")
        self.strength = QComboBox()
        for key, text in (("soft", "Soft — calm, little new detail"),
                          ("natural", "Natural — clean, keeps the person (default)"),
                          ("crisp", "Crisp — sharper hair and texture, busier")):
            self.strength.addItem(text, key)
        self.strength.setCurrentIndex(max(0, self.strength.findData(strength)))
        self.strength.setToolTip("How much the refine draws anew. Natural keeps faces, eyes and hair as made and "
                                 "adds clean detail; Crisp adds more texture (hair, knit) and some grain; Soft "
                                 "stays closest to the clip but is softer. (The release's own setting repainted "
                                 "faces: closed eyes drawn half open, eye and hair colour changed.) Refined copies "
                                 "are kept per strength")
        f.addRow("Refine", self.strength)
        self.fast = QCheckBox("Faster refine with Sol-Attn (about 2× faster at 2K, very slightly different)")
        self.fast.setToolTip("Block-sparse attention in the refine pass. Measured on an RTX 4090 at 2K: 120 s instead "
                             "of 267 s per 3 s clip, PSNR 33 dB against exact attention (side by side practically the "
                             "same; skin texture very slightly smoother). Off: exact attention")
        f.addRow("", self.fast)
        self.swapped_count = swapped
        self.swapped = QCheckBox(f"Upscale the video{'s' if swapped != 1 else ''} with a replaced person too")
        self.swapped.setToolTip("A video whose person was replaced was made at the generation size. On: it is "
                                "upscaled like a generated clip (SeedVR2 restores it; the refine re-renders it with "
                                "the new person's face description), its recorded sound kept. Off: it is only "
                                "scaled up from its made size")
        self.swapped.setChecked(True)
        self.swapped.setVisible(bool(swapped))
        f.addRow("", self.swapped)
        note = label("Needs ComfyUI with SeedVR2 (its 7B / 3B model and VAE from Comfy-Org/SeedVR2) or, for the "
                     "refine, the H3 2K nodes, the latent upscaler and the refiner LoRA. Generating and the preview "
                     "stay at the normal size.", level="tertiary")
        note.setWordWrap(True)
        f.addRow("", note)
        for w_ in (self.size, self.method, self.strength):
            w_.currentIndexChanged.connect(lambda *_: self._update())
        self.denoise.toggled.connect(lambda *_: self._update())
        self._update()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        f.addRow(buttons)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.file.textChanged.connect(self._check_file)
        self._check_file()
        guard_wheel(self)

    def _check_file(self, *_):
        """Export needs a file name (an empty one would make a hidden ".mp4")."""
        self.ok.setEnabled(bool(self.file.text().strip()))

    def _ready(self, size: str) -> bool:
        how, st = self.method.currentData(), self.strength.currentData()
        return bool(self.ready(size, how, st, self.denoise.isChecked() and how != "refine"))

    def _update(self):
        up = self.size.currentData() != "made"
        refine = self.method.currentData() == "refine"
        self.method.setEnabled(up)
        self.strength.setEnabled(up and refine)
        self.fast.setEnabled(up and refine)
        self.denoise.setEnabled(up and not refine)
        self.cloud.setEnabled(bool(up and not refine and self.cloud_ok))
        self.swapped.setEnabled(up)
        w, h = self.made_size or (0, 0)
        for i in range(1, self.size.count()):
            key = self.size.itemData(i)
            text = {"1080p": "1080p", "2k": "2K"}[key]
            if self.made_size:
                rw, rh = refine_size(w, h, key)
                text += f" — {rw}×{rh} · {rw / max(1, w):.1f}× the draft"
            if self._ready(key):
                text += " · already upscaled, quick"
            self.size.setItemText(i, text)
        size = self.size.currentData()
        factor = refine_size(w, h, size)[0] / w if self.made_size and size != "made" else 0.0
        calmer = self.made_size and size == "2k" and refine_size(w, h, "1080p")[0] / w <= 2.05
        self.scale_note.setText(
            f"{factor:.1f}× the draft: above about 2× SeedVR2 invents texture that shimmers in motion (measured: "
            "+29 % in a pan at 2.5× against 1.9×). " + ("1080p stays within 2× here. " if calmer else "")
            + "Larger drafts help: Clips → Size \u201c768p — H3's native size\u201d." if factor > 2.05 else "")
        self.scale_note.setVisible(factor > 2.05)

    def _choose(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export path", self.file.text(), "Video (*.mp4)")
        if f:
            self.file.setText(f)

    def on_cloud(self) -> bool:
        """The export runs on RunPod (only SeedVR2, only at a higher size, only with a key)."""
        return bool(self.cloud.isEnabled() and self.cloud.isChecked())

    def target(self) -> Path:
        f = self.file.text().strip()
        return Path(os.path.expanduser(f if f.lower().endswith(".mp4") else f + ".mp4"))

    def request(self) -> ExportRequest:
        """The dialog's answers in one piece."""
        return ExportRequest(self.target(), self.size.currentData(), self.fast.isChecked(),
                             bool(self.swapped_count) and self.swapped.isChecked(), self.strength.currentData(),
                             self.method.currentData(), self.denoise.isChecked(), self.on_cloud())
