"""Edit an image by a prompt (Qwen Image Edit in ComfyUI) and use the result as the start image or as a clip's end
image. Every result is kept in the project (edits/<name>.png + .json with its prompt, the image it was made from in
edits/sources/), so earlier tries stay at hand.

1. the image to edit (start image, a clip's frame, an end image, a face reference, any file, an earlier result)
2. what should change (the prompt), optionally up to two more pictures (Picture 2 / 3, e.g. a face)
3. Generate: one or more tries (other seeds); 4. pick a result, compare it with the original, use it.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QSlider, QStackedWidget, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from .image_picker import pick_image
from ..comfy import EDIT_MODELS
from ..triggers import stack_words, with_triggers
from ..imagegen import FAMILIES, SAMPLERS, SCHEDULERS, default_sampling, size_for
from .lora_stack import LoraStack, lora_label
from .variations_panel import PlaceholderHighlighter, VariationsPanel
from .results_gallery import (InfoCard, ResultGallery, ResultView, ThumbLoader, badged, model_key, model_name,
                              run_groups)
from .mask_painter import MaskPainterDialog, mask_area, overlay
from .widgets import FrameTile, guard_wheel, icon_button, label

MAX_RESULTS = 20                           # results per Generate (made one after another, Cancel stops)
SEED_MAX = 2**31 - 1                      # (the seeds ComfyUI's image samplers take here)

EDITS_DIR = "edits"
MOSS_EDIT_QUALITIES = {"standard": "Standard (30 steps)"}
LORA_FAMILY = {"chroma": "flux1"}         # Chroma LoRAs have FLUX.1's layer names (and FLUX.1 LoRAs often work)
QUALITY_TEXT = {"fast": "Fast (Lightning, 4 steps)", "standard": "Standard (20 steps)", "best": "Best (40 steps, slow)"}
PROMPT_HINT = ("Describe the change, e.g. “she smiles and looks into the camera”, “night, lit by a street lamp”, "
               "“replace the red jacket with a black leather jacket”. Say what must stay: “keep her face, pose and "
               "the background”.")


CREATE_HINT = ("Describe the image: who or what, doing what, where; light, mood, camera and style, e.g. “a woman "
               "in a red coat on a rainy Paris street at night, neon reflections, 35 mm photo”. With reference "
               "pictures, say what to take from them.")


def scale_to(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Scale an image to exactly this size (the result keeps the size of the image it was made from)."""
    tmp = dst.with_name(dst.stem + ".part.png")
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf",
                            f"scale={size[0]}:{size[1]}:flags=lanczos", "-frames:v", "1", str(tmp)],
                           capture_output=True, timeout=120)
        if r.returncode != 0 or not tmp.is_file():
            raise RuntimeError(r.stderr.decode(errors="replace").strip()[-300:] or "ffmpeg failed")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def upright(path) -> QImage:
    """An image as ComfyUI's LoadImage sees it: turned by its EXIF orientation (phone photos)."""
    from PySide6.QtGui import QImageReader
    r = QImageReader(str(path))
    r.setAutoTransform(True)
    return r.read()


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def project_edits(project: Path) -> list[tuple[Path, dict]]:
    """The edit results kept in a project, newest first, with what they were made from."""
    d = Path(project) / EDITS_DIR
    out = []
    for png in d.glob("*.png") if d.is_dir() else []:
        if ".part" in png.name:
            continue
        try:
            meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        out.append((png, meta))
    return sorted(out, key=lambda x: _mtime(x[0]), reverse=True)


def record_use(png: str | Path, use: dict) -> None:
    """Note in a result's record which project copy it became ({"kind": "start", "sha1"} / {"kind": "end",
    "file"} / {"kind": "ref", "file"}), so the window knows it is in use."""
    meta_file = Path(png).with_suffix(".json")
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    uses = [u for u in meta.get("uses") or [] if u != use]
    meta["uses"] = uses + [use]
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def uses_of(project: Path, tree, meta: dict) -> list[str]:
    """Where a result is used now, in words: the start image (still that picture), a clip's end image (set now or
    made with), a face reference. Empty: it can go."""
    from .. import chain
    from .. import tree as T
    out = []
    owner = {r: c["name"] for c in (T.cast_of(tree) if tree is not None else []) for r in c["images"]}
    for u in meta.get("uses") or []:
        kind = u.get("kind")
        if kind == "start" and u.get("sha1") and chain.file_signature(Path(project) / T.START_FILE) == u["sha1"]:
            out.append("the start image")
        elif kind == "end" and tree is not None:
            for n in tree.nodes.values():
                if u.get("file") and u["file"] in (n.end_image, (n.made or {}).get("end_image")):
                    out.append(f"the end image of clip #{n.id[1:]}")
        elif kind == "ref" and u.get("file") in owner:
            out.append(f"face of @{owner[u['file']]}")
    return list(dict.fromkeys(out))


class _Sig(QObject):
    families = Signal(object, str)              # what ComfyUI has of each image family, error
    progress = Signal(int, int, int, int)       # try k, of n, step i, of steps
    result = Signal(str, object)                # png, meta
    finished = Signal(str)                      # error ('' = all done or cancelled)


class PictureSlot(QWidget):
    """A small optional picture (Picture 2 / 3) with choose / remove."""
    changed = Signal()

    def __init__(self, title: str, chooser, parent=None):
        super().__init__(parent)
        self.path: str | None = None
        self.chooser = chooser
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.tile = FrameTile(120, 90)
        self.tile.clear_frame(title)
        self.tile.setCursor(Qt.PointingHandCursor)
        self.tile.setToolTip(f"{title}: click to choose a picture (e.g. a face or a piece of clothing)")
        self.tile.clicked.connect(lambda _t: self.choose())
        self.title = title
        row = QHBoxLayout()
        row.addWidget(label(title, level="tertiary"))
        row.addStretch(1)
        self.b_clear = icon_button("x", f"No {title}")
        self.b_clear.clicked.connect(lambda: self.set(None))
        self.b_clear.setEnabled(False)
        row.addWidget(self.b_clear)
        v.addWidget(self.tile)
        v.addLayout(row)

    def choose(self):
        f = self.chooser(f"{self.title} (optional)")
        if f:
            self.set(f)

    def set(self, path: str | None):
        self.path = path
        if path:
            self.tile.set_image(QImage(path))
            self.tile.setToolTip(path)
        else:
            self.tile.clear_frame(self.title)
        self.b_clear.setEnabled(bool(path))
        self.changed.emit()


class ImageEditDialog(QDialog):
    """sources: [(label, path)] offered as the image to edit; end_target: the clip a result can become the end
    image of (None: only the start image). connect(progress) -> a ComfyImageEditor (called in a thread);
    on_use(kind 'start' | 'end', png) -> bool; start_comfy() -> bool."""

    def __init__(self, parent, project: Path, sources: list[tuple[str, str]], source: str | None,
                 end_target: str | None, connect, on_use, settings=None, places=None, start_comfy=None,
                 on_start=None, known_loras: list[tuple[str, str]] | None = None, lora_dir: str = "",
                 uses=None, refine=None, connect_create=None, family_loras=None,
                 start_aspect: float | None = None, mode: str = "edit", check_lora=None,
                 use_triggers: bool = True, collections=None):
        super().__init__(parent)
        self.setWindowTitle("Edit an image with AI")
        self.resize(1240, 820)
        self.project, self.connect_editor, self.on_use = Path(project), connect, on_use
        self.end_target, self.qs, self.places, self.start_comfy = end_target, settings, places or [], start_comfy
        self.on_start = on_start                         # called (GUI thread) before each Generate
        self.uses = uses or (lambda _meta: [])           # uses(meta) -> where a result is used (words)
        self.refine = refine                             # refine(text, kind) -> new text or None
        self.connect_create = connect_create             # connect_create(progress) -> a ComfyImageCreator
        self.start_aspect = start_aspect
        self.check_lora = check_lora                     # check_lora(path) -> False if ComfyUI cannot load it
        self.use_triggers = use_triggers                 # the LoRAs' trigger words before the prompt (project)
        self.collections = collections                   # () -> the project's pictures by kind (image picker)
        self.families: dict = {}                         # what ComfyUI has of each family (from the creator)
        self._families_error = ""
        self._retry_families = 0                         # retries left while ComfyUI starts
        self._trash: Path | None = None                  # the result deleted last (Undo delete brings it back)
        self._thumbs: dict[str, tuple[int, QPixmap]] = {}  # result thumbnails: png -> (its mtime_ns, pixmap)
        self._requested: set[tuple[str, int]] = set()
        self._entries: list[tuple[str, dict, float]] = []
        self._loader = ThumbLoader(self)
        self._loader.ready.connect(self._thumb_ready)
        self.closed = False
        self.source: str | None = None
        self.editor = None
        self._busy = False
        self._cancel = False
        self.used: tuple[str, str] | None = None        # (kind, png) once a result was used
        self._sig = _Sig()
        self._sig.progress.connect(self._progress)
        self._sig.result.connect(self._result)
        self._sig.finished.connect(self._finished)
        self._sig.families.connect(self._families_found)

        top = QHBoxLayout(self)
        left = QVBoxLayout()
        left.setSpacing(6)
        # mode: edit an image (Qwen Image Edit) or create a new one (four model families)
        mrow = QHBoxLayout()
        self.m_edit = QPushButton("Edit an image")
        self.m_create = QPushButton("Create a new image")
        self.modes = QButtonGroup(self)
        for k, b in enumerate((self.m_edit, self.m_create)):
            b.setCheckable(True)
            self.modes.addButton(b, k)
            mrow.addWidget(b)
        self.m_edit.setToolTip("Change an existing image by a prompt (Qwen Image Edit)")
        self.m_create.setToolTip("Make a new image from a prompt, or from reference pictures (Krea-2, Qwen-Image, "
                                 "FLUX, Illustrious)")
        self.m_edit.setChecked(True)
        left.addLayout(mrow)

        # ---- edit: the image to edit
        self.edit_box = QWidget()
        ev = QVBoxLayout(self.edit_box)
        ev.setContentsMargins(0, 0, 0, 0)
        ev.addWidget(label("1  Image to edit", type_="heading"))
        self.src_tile = FrameTile(380, 214)
        self.src_tile.clear_frame("Choose the image to edit")
        self.src_tile.setCursor(Qt.PointingHandCursor)
        self.src_tile.clicked.connect(lambda _t: self._choose_source())
        ev.addWidget(self.src_tile)
        self.src_combo = QComboBox()
        self.src_combo.setToolTip("Images of this project; or Choose file… for any image")
        for text, path in sources:
            self.src_combo.addItem(text, path)
        self.src_combo.addItem("Choose file…", None)
        self.src_combo.activated.connect(self._source_picked)
        ev.addWidget(self.src_combo)
        self.src_info = label("", level="tertiary")
        self.src_info.setWordWrap(True)
        ev.addWidget(self.src_info)
        erow = QHBoxLayout()
        self.edit_model = QComboBox()
        for key, text in EDIT_MODELS.items():
            self.edit_model.addItem(text, key)
        self.edit_model.setToolTip("Qwen-Image-Edit 2511: the proven edit model. Qwen-Image 2.1 (2026): newer and "
                                   "smaller, one model for creating and editing; research licence (non-commercial). "
                                   "MOSS-Image Edit (2026, 5B, Apache 2.0): MOSS-Image with its edit LoRA")
        saved = settings.value("editModel", "", type=str) if settings is not None else ""
        self.edit_model.setCurrentIndex(max(0, self.edit_model.findData(saved)))
        self.edit_model.currentIndexChanged.connect(self._edit_model_changed)
        self.edit_model.activated.connect(lambda _i: self.qs is not None and
                                          self.qs.setValue("editModel", self.edit_model.currentData()))
        erow.addWidget(label("Model", level="tertiary"))
        erow.addWidget(self.edit_model, 1)
        ev.addLayout(erow)
        mrow2 = QHBoxLayout()
        self.b_mask = QPushButton("Paint mask…")
        self.b_mask.setToolTip("Paint the part that should change; the rest stays exactly as it is (inpainting "
                               "with Qwen-Image 2.1 and its Fun ControlNet)")
        self.b_mask.clicked.connect(self.paint_mask)
        self.b_mask_clear = QPushButton("No mask")
        self.b_mask_clear.setToolTip("Edit the whole image again")
        self.b_mask_clear.clicked.connect(lambda: self.set_mask(None))
        self.mask_info = label("", level="tertiary")
        self.mask_info.setWordWrap(True)
        mrow2.addWidget(self.b_mask)
        mrow2.addWidget(self.b_mask_clear)
        mrow2.addWidget(self.mask_info, 1)
        ev.addLayout(mrow2)
        self.mask: QImage | None = None              # white = drawn anew, at the source's (upright) size
        left.addWidget(self.edit_box)

        # ---- create: model, references, size
        self.create_box = QWidget()
        cv = QVBoxLayout(self.create_box)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(label("1  Model", type_="heading"))
        crow = QHBoxLayout()
        self.family = QComboBox()
        self.family.setToolTip("The model family; unavailable ones say what is missing")
        self.family.addItem("looking for models in ComfyUI …", None)
        self.family.currentIndexChanged.connect(self._family_changed)
        self.family.activated.connect(lambda _i: self.qs is not None and self.family.currentData() and
                                      self.qs.setValue("createFamily", self.family.currentData()))
        self.model_file = QComboBox()
        self.model_file.setToolTip("The model file (the recommended one first; fine-tunes of the family too)")
        crow.addWidget(self.family, 1)
        crow.addWidget(self.model_file, 1)
        cv.addLayout(crow)
        self.use_refs = QCheckBox("From reference pictures")
        self.use_refs.toggled.connect(lambda _on: self._create_widgets())
        cv.addWidget(self.use_refs)
        self.ref_text = label("", level="tertiary")
        self.ref_text.setWordWrap(True)
        cv.addWidget(self.ref_text)
        rrow = QHBoxLayout()
        self.cpics = [PictureSlot(f"Picture {k}", self._pick) for k in (1, 2, 3)]
        for slot in self.cpics:
            rrow.addWidget(slot)
            slot.changed.connect(self._update_buttons)
        rrow.addStretch(1)
        cv.addLayout(rrow)
        srow2 = QHBoxLayout()
        self.ref_strength = QDoubleSpinBox()
        self.ref_strength.setRange(0.2, 1.0)
        self.ref_strength.setSingleStep(0.05)
        self.ref_strength.setValue(0.65)
        self.ref_strength.setToolTip("How far the new image may move away from the picture: low keeps it close, "
                                     "1.0 ignores it")
        self.ref_strength_label = label("Change", level="tertiary")
        srow2.addWidget(self.ref_strength_label)
        srow2.addWidget(self.ref_strength)
        srow2.addStretch(1)
        cv.addLayout(srow2)
        arow = QHBoxLayout()
        self.aspect = QComboBox()
        for text, value in (("16:9 landscape", 16 / 9), ("9:16 portrait", 9 / 16), ("1:1 square", 1.0),
                            ("4:3", 4 / 3), ("3:4", 3 / 4), ("3:2", 3 / 2), ("2:3", 2 / 3)):
            self.aspect.addItem(text, value)
        if self.start_aspect:
            self.aspect.insertItem(0, "like the start image", self.start_aspect)
        self.aspect.setCurrentIndex(0)
        self.aspect.currentIndexChanged.connect(lambda *_: self._create_widgets())
        self.size_text = label("", level="tertiary")
        arow.addWidget(label("Shape", level="tertiary"))
        arow.addWidget(self.aspect, 1)
        arow.addWidget(self.size_text)
        cv.addLayout(arow)
        self._build_sampling(cv)
        self._build_presets(cv)
        left.addWidget(self.create_box)

        prow0 = QHBoxLayout()
        self.prompt_title = label("2  What should change?", type_="heading")
        prow0.addWidget(self.prompt_title)
        prow0.addStretch(1)
        self.b_refine = icon_button("gen", "Refine this prompt with a language model")
        self.b_refine.clicked.connect(self._refine_prompt)
        self.b_refine.setVisible(refine is not None)
        prow0.addWidget(self.b_refine)
        left.addLayout(prow0)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(PROMPT_HINT)
        self.prompt.setMinimumHeight(110)
        self.prompt.textChanged.connect(self._update_buttons)
        left.addWidget(self.prompt, 1)
        self.negative = QLineEdit()
        self.negative.setPlaceholderText("Avoid (optional negative prompt; empty: the model's usual one)")
        left.addWidget(self.negative)
        self.variations = VariationsPanel(self, pick_file=lambda: self._pick_list())
        left.addWidget(self.variations)
        self._highlight = PlaceholderHighlighter(self.prompt.document(), self.variations)
        self._var_timer = QTimer(self)
        self._var_timer.setSingleShot(True)
        self._var_timer.setInterval(150)
        self._var_timer.timeout.connect(self._templates_changed)
        self.prompt.textChanged.connect(self._var_timer.start)
        self.negative.textChanged.connect(lambda *_: self._var_timer.start())
        self.variations.changed.connect(self._variations_changed)
        self.variations.insert.connect(self._insert_placeholder)
        self.variations.rename.connect(self._rename_placeholder)
        self.edit_pics = QWidget()
        epv = QVBoxLayout(self.edit_pics)
        epv.setContentsMargins(0, 0, 0, 0)
        more = label("More pictures (optional): name them in the prompt, e.g. “give her the face from Picture 2”. "
                     "The image to edit is Picture 1.", level="tertiary")
        more.setWordWrap(True)                     # unwrapped it made the panel wider than its view
        epv.addWidget(more)
        prow = QHBoxLayout()
        self.pic2 = PictureSlot("Picture 2", self._pick)
        self.pic3 = PictureSlot("Picture 3", self._pick)
        prow.addWidget(self.pic2)
        prow.addWidget(self.pic3)
        prow.addStretch(1)
        epv.addLayout(prow)
        left.addWidget(self.edit_pics)
        self.lora_text = label("", level="tertiary")
        self.lora_text.setWordWrap(True)
        left.addWidget(self.lora_text)
        self.known_loras, self.lora_dir = list(known_loras or []), lora_dir
        self.family_loras = family_loras or {}
        self.loras = LoraStack(self)
        self.loras.file_dir = lora_dir
        self.loras.picked.connect(self._lora_picked)
        self.loras.changed.connect(self._remember_loras)
        self.loras.changed.connect(lambda: self._fill_presets())
        self.loras.filled.connect(lambda: self._fill_presets())      # (also the LoRAs remembered for this family)
        left.addWidget(self.loras)

        opts = QGridLayout()
        self.quality = QComboBox()
        self.quality.setToolTip("Faster settings use the model's speed LoRA or distilled version; Best samples "
                                "longer and can follow the prompt more closely")
        self.quality.currentIndexChanged.connect(lambda *_: self._remember_quality())
        self.quality.currentIndexChanged.connect(lambda *_: self._sampling_defaults())
        self.tries = QSpinBox()
        self.tries.setRange(1, MAX_RESULTS)
        self.tries.setValue(2)
        self.tries.setToolTip(f"How many results at once (1 … {MAX_RESULTS}; each with the next seed), to pick the "
                              "best one. They are made one after another and each appears when it is ready; "
                              "Cancel stops after the current one")
        self.keep_size = QCheckBox("Keep the size of the image to edit")
        self.keep_size.setChecked(True)
        self.keep_size.setToolTip("The model works at about 1 megapixel in its own aspect steps; this scales the "
                                  "result back to the exact size of the image it was made from, so it fits the "
                                  "chain's video size")
        self.tags = QCheckBox("Add the usual quality tags (masterpiece, best quality …)")
        self.tags.setChecked(True)
        self.tags.setToolTip("Illustrious models are trained on such tags; they are put before your prompt")
        opts.addWidget(label("Quality", level="tertiary"), 0, 0)
        opts.addWidget(self.quality, 0, 1)
        opts.addWidget(label("Results", level="tertiary"), 0, 2)
        opts.addWidget(self.tries, 0, 3)
        srow = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(1, SEED_MAX)
        self.seed.setValue(random.randint(1, SEED_MAX))
        self.seed.setToolTip("The seed of the next result (more results at once: this seed, +1, +2 …); the same "
                             "seed and prompt give the same image again")
        self.b_seed = icon_button("rotate", "New random seed")
        self.b_seed.clicked.connect(lambda: self.new_seed())
        self.fresh_seed = QCheckBox("New seed after each Generate")
        self.fresh_seed.setChecked(True)
        self.fresh_seed.setToolTip("On: Generate again gives new images. Off: the seed stays, so a changed prompt can "
                                   "be compared with the same seed")
        srow.addWidget(self.seed, 1)
        srow.addWidget(self.b_seed)
        srow.addWidget(self.fresh_seed)
        opts.addWidget(label("Seed", level="tertiary"), 1, 0)
        opts.addLayout(srow, 1, 1, 1, 3)
        opts.addWidget(self.keep_size, 2, 0, 1, 4)
        opts.addWidget(self.tags, 3, 0, 1, 4)
        left.addLayout(opts)
        brow = QHBoxLayout()
        self.b_gen = QPushButton("Generate")
        self.b_gen.setProperty("variant", "primary")
        self.b_gen.setToolTip("Edit the image with this prompt (Ctrl+Enter)")
        self.b_gen.clicked.connect(self.generate)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.clicked.connect(self.cancel)
        self.b_cancel.setEnabled(False)
        self.b_start_comfy = QPushButton("Start ComfyUI")
        self.b_start_comfy.hide()
        self.b_start_comfy.clicked.connect(self._start_comfy)
        brow.addWidget(self.b_gen)
        brow.addWidget(self.b_cancel)
        brow.addWidget(self.b_start_comfy)
        brow.addStretch(1)
        left.addLayout(brow)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.hide()
        left.addWidget(self.bar)
        self.state = label("", level="secondary")
        self.state.setWordWrap(True)
        left.addWidget(self.state)
        lw = QWidget()
        lw.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(lw)
        scroll.setFixedWidth(430)
        self.left_scroll = scroll
        top.addWidget(scroll)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        top.addWidget(line)

        right = QVBoxLayout()
        head = QHBoxLayout()
        head.addWidget(label("3  Results", type_="heading"))
        head.addSpacing(10)
        self.b_single = QPushButton("Single")
        self.b_gallery = QPushButton("Gallery")
        self.views = QButtonGroup(self)
        for k, (b, tip) in enumerate(((self.b_single, "One result large, the others in a row below"),
                                      (self.b_gallery, "All results as a grid, grouped by Generate run "
                                                       "(double-click or Enter: large; Esc: back)"))):
            b.setCheckable(True)
            b.setToolTip(tip)
            self.views.addButton(b, k)
            head.addWidget(b)
        self.b_single.setChecked(True)
        self.views.idClicked.connect(lambda i: self.set_view("gallery" if i else "single"))
        self.thumb_size = QSlider(Qt.Horizontal)
        self.thumb_size.setRange(120, 360)
        self.thumb_size.setValue(200)
        self.thumb_size.setFixedWidth(120)
        self.thumb_size.setToolTip("Thumbnail size")
        self.thumb_size.valueChanged.connect(self._thumb_size_changed)
        head.addSpacing(6)
        head.addWidget(self.thumb_size)
        head.addStretch(1)
        self.b_fav = QPushButton("★ Favourite")
        self.b_fav.setCheckable(True)
        self.b_fav.setToolTip("Mark this result as a favourite (F); the filter can show only favourites")
        self.b_fav.clicked.connect(lambda: self.toggle_favorite())
        head.addWidget(self.b_fav)
        self.b_before = QPushButton("Show the original")
        self.b_before.setCheckable(True)
        self.b_before.setToolTip("Switch between the result and the image it was made from (or hold B)")
        self.b_before.toggled.connect(lambda _on: self._show_selected())
        head.addWidget(self.b_before)
        right.addLayout(head)
        frow = QHBoxLayout()
        self.f_show = QComboBox()
        for text, key in (("All results", "all"), ("★ Favourites", "fav"), ("In use", "used")):
            self.f_show.addItem(text, key)
        self.f_show.setToolTip("Which results are shown")
        self.f_model = QComboBox()
        self.f_model.addItem("All models", None)
        self.f_model.setToolTip("Only the results of one model")
        self.f_search = QLineEdit()
        self.f_search.setPlaceholderText("Search the prompts …")
        self.f_search.setClearButtonEnabled(True)
        self.f_count = label("", level="tertiary")
        for w in (self.f_show, self.f_model):
            w.currentIndexChanged.connect(lambda *_: self._load_results(keep=True))
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(lambda: self._load_results(keep=True))
        self.f_search.textChanged.connect(lambda *_: self._search_timer.start())
        frow.addWidget(self.f_show)
        frow.addWidget(self.f_model)
        frow.addWidget(self.f_search, 1)
        frow.addWidget(self.f_count)
        right.addLayout(frow)
        self.stack = QStackedWidget()
        single = QWidget()
        sv = QVBoxLayout(single)
        sv.setContentsMargins(0, 0, 0, 0)
        self.view = ResultView()
        self.view.clear_frame("Results appear here")
        sv.addWidget(self.view, 1)
        self.results = QListWidget()
        self.results.setViewMode(QListWidget.IconMode)
        self.results.setFlow(QListWidget.LeftToRight)
        self.results.setWrapping(False)
        self.results.setIconSize(QSize(150, 100))
        self.results.setGridSize(QSize(158, 108))
        self.results.setFixedHeight(130)
        self.results.setMovement(QListWidget.Static)
        self.results.setToolTip("The results shown (newest first); Gallery shows them all, grouped by run")
        self.results.setStyleSheet("QListWidget { background: transparent; border: none; } "
                                   "QListWidget::item:selected, QListWidget::item:hover { background: transparent; }")
        self.results.currentItemChanged.connect(self._strip_selection)
        self.results.currentItemChanged.connect(lambda *_: self._show_selected())
        sv.addWidget(self.results)
        self.stack.addWidget(single)
        self.gallery = ResultGallery()
        self.gallery.clicked.connect(self._gallery_clicked)
        self.gallery.opened.connect(self._gallery_opened)
        self.gallery.starred.connect(lambda png: self.toggle_favorite(png))
        self.gallery.back.connect(lambda: self.set_view("single"))
        self.stack.addWidget(self.gallery)
        right.addWidget(self.stack, 1)
        self.res_info = InfoCard()
        self.res_info.use_seed.connect(self._use_seed)
        right.addWidget(self.res_info)
        arow = QHBoxLayout()
        self.b_use_start = QPushButton("Use as start image")
        self.b_use_start.setToolTip("The first frame of the chain: the clips right after the start begin with it")
        self.b_use_start.clicked.connect(lambda: self._use("start"))
        self.b_use_end = QPushButton(f"Use as end image of clip #{end_target[1:]}" if end_target else
                                     "Use as end image")
        self.b_use_end.setToolTip("The frame this clip steers its end to (H3's last-frame input)" if end_target else
                                  "Open this from a clip (its End row or its menu) to make its end image")
        self.b_use_end.clicked.connect(lambda: self._use("end"))
        self.b_further = QPushButton("Edit this one further")
        self.b_further.setToolTip("Take this result as the image to edit (several small steps often work better "
                                  "than one big change)")
        self.b_further.clicked.connect(self._edit_further)
        self.b_use_ref = QPushButton("Add as face reference")
        self.b_use_ref.setToolTip("Add this result to the project's face pictures (the Cast tab, Alt+2), e.g. a face or an "
                                  "outfit made here; the dialog stays open, so you can add several")
        self.b_use_ref.clicked.connect(self._add_reference)
        self.b_again = QPushButton("Use its prompt and LoRAs")
        self.b_again.setToolTip("Put the prompt and the LoRAs this result was made with into the fields")
        self.b_again.clicked.connect(self._prompt_again)
        self.b_delete = QPushButton("Delete")
        self.b_delete.setProperty("variant", "danger")
        self.b_delete.clicked.connect(self.delete_selected)
        self.b_undo = QPushButton("Undo delete")
        self.b_undo.setToolTip("Bring the result deleted last back")
        self.b_undo.clicked.connect(self.undo_delete)
        self.b_undo.hide()
        for b in (self.b_use_start, self.b_use_end, self.b_use_ref, self.b_further, self.b_again):
            arow.addWidget(b)
        arow.addStretch(1)
        arow.addWidget(self.b_undo)
        arow.addWidget(self.b_delete)
        right.addLayout(arow)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        right.addWidget(close)
        top.addLayout(right, 1)

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.generate)
        for key, fn in (("Delete", self.delete_selected), ("B", self.b_before.toggle),
                        ("F", lambda: self.toggle_favorite())):
            for w in (self.results, self.gallery):
                sc = QShortcut(QKeySequence(key), w, activated=fn)
                sc.setContext(Qt.WidgetWithChildrenShortcut)  # only while the results have the focus
        for b in self.findChildren(QPushButton):              # Enter in a field must not press a button
            b.setAutoDefault(False)
            b.setDefault(False)
        guard_wheel(self)
        if self.qs is not None:
            self.thumb_size.setValue(self.qs.value("resultsThumb", 200, type=int))
            try:
                self.variations.set_state(json.loads(self.qs.value("promptVariations", "{}", type=str) or "{}"))
            except ValueError:
                pass
        self._load_results()
        self.set_view(self.qs.value("resultsView", "single", type=str) if self.qs is not None else "single",
                      remember=False)
        if source:
            i = self.src_combo.findData(source)
            if i >= 0:
                self.src_combo.setCurrentIndex(i)
            self.set_source(source)
        elif sources:
            self.set_source(sources[0][1])
        self.modes.idClicked.connect(self._mode_clicked)
        self.m_create.setEnabled(connect_create is not None)
        saved = self.qs.value("imageMode", "", type=str) if self.qs is not None else ""
        if mode != "auto" and self.qs is not None:
            self.qs.setValue("imageMode", mode)       # opened from its Edit / Create button: the user's choice
        self.set_mode(mode if mode != "auto" else (saved or "edit"))
        if connect_create is not None:
            self._load_families()
        self._update_buttons()

    # ---- modes
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
                try:
                    sig.families.emit({}, str(e))
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _families_found(self, files: dict, error: str):
        before = self.family.currentData()
        kept = self.loras.items() if self.mode() == "create" and before not in FAMILIES else []
        self.families = files
        self._families_error = error
        if error and self._retry_families > 0:           # ComfyUI is starting: ask again in a few seconds
            self._retry_families -= 1
            QTimer.singleShot(5000, self._load_families)
        elif not error and self._retry_families >= 0 and getattr(self, "_was_down", False):
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
        if create:
            found = len(self.family_loras.get(LORA_FAMILY.get(fam, fam), [])) if f else 0
            self.lora_text.setText(
                f"LoRAs (optional): Chroma and FLUX.1 LoRAs (the same layer names, so they are listed together; "
                f"FLUX.1 LoRAs often work on Chroma, not always; {found} found), applied in this order."
                if fam == "chroma" else
                f"LoRAs (optional): {f['label']} LoRAs from ComfyUI's LoRA folders (found by their contents; "
                f"{found} found), applied in this order." if f else "")
        else:
            self.lora_text.setText(
                f"LoRAs (optional): Qwen-Image 2.1 LoRAs from ComfyUI's LoRA folders (found by their contents; "
                f"{len(self.family_loras.get('qwen21', []))} found), applied in this order."
                if self.edit_model.currentData() == "qwen21" else
                f"LoRAs (optional): MOSS-Image LoRAs from ComfyUI's LoRA folders (found by their contents; "
                f"{len(self.family_loras.get('moss', []))} found), applied before its edit LoRA."
                if self.edit_model.currentData() == "moss" else
                "LoRAs (optional): Qwen Image / Image Edit LoRAs from ComfyUI's LoRA folders, applied in this order. "
                "Text-to-image Qwen LoRAs often work for edits too, not always.")
        self.b_gen.setToolTip("Make the image (Ctrl+Enter)" if create else
                              "Edit the image with this prompt (Ctrl+Enter)")
        self._update_buttons()

    # ---- source and pictures
    def _pick(self, title: str) -> str | None:
        return pick_image(self, title, self.qs, None, self.places,
                          self.collections() if self.collections is not None else None)

    def _choose_source(self):
        f = self._pick("Image to edit")
        if f:
            self.set_source(f)

    def _source_picked(self, i: int):
        path = self.src_combo.itemData(i)
        if path is None:
            self._choose_source()
        else:
            self.set_source(path)

    def set_source(self, path: str):
        img = upright(path)
        if img.isNull():
            self.state.setText(f"Not an image: {Path(path).name}")
            return
        changed = str(path) != self.source
        self.source = str(path)
        if changed or (self.mask is not None and self.mask.size() != img.size()):
            self.mask = None                       # a mask belongs to its image
        self.src_tile.set_image(overlay(img, self.mask) if self.mask is not None else img)
        self.src_tile.setToolTip(str(path))
        self._show_mask()
        i = self.src_combo.findData(str(path))
        self.src_info.setText(f"{self.src_combo.itemText(i) + ' · ' if i >= 0 else ''}{Path(path).name} · "
                              f"{img.width()}×{img.height()}")
        self._update_buttons()

    # ---- mask (inpainting)
    def paint_mask(self):
        if not self.source:
            return
        img = upright(self.source)
        dlg = MaskPainterDialog(self, img, self.mask)
        try:
            if dlg.exec() == QDialog.Accepted:
                self.set_mask(dlg.mask if mask_area(dlg.mask) > 0 else None)
        finally:
            dlg.deleteLater()

    def set_mask(self, mask: QImage | None):
        """The painted mask for the image to edit (None: the whole image is edited)."""
        self.mask = mask.copy() if mask is not None else None
        if self.source:
            img = upright(self.source)
            self.src_tile.set_image(overlay(img, self.mask) if self.mask is not None else img)
        self._show_mask()
        self._update_buttons()

    def _mask_problem(self) -> str:
        """Why the painted mask cannot be used now ("" if it can, or there is none)."""
        if self.mask is None:
            return ""
        if self.edit_model.currentData() != "qwen21":
            return "A painted mask works with Qwen-Image 2.1: choose it as the model (or remove the mask)."
        return ""

    def _show_mask(self):
        on = self.mask is not None
        self.b_mask.setText("Change mask…" if on else "Paint mask…")
        self.b_mask.setEnabled(bool(self.source) and not self._busy)
        self.b_mask_clear.setVisible(on)
        why = self._mask_problem()
        a = mask_area(self.mask) if on else 0.0
        self.mask_info.setText(why or (f"Only the red part ({f'{a * 100:.0f} %' if a >= 0.01 else 'under 1 %'}) is "
                                       "drawn anew; the rest stays exactly as it is. Say what it should show, e.g. “She tilts her head back "
                                       "and looks up at the ceiling” (a short “She is looking up” only moves the "
                                       "eyes)." if on else ""))

    # ---- results
    def _load_results(self, select: str | None = None, keep: bool = False):
        """The results with the filters applied: the strip below the large view and the gallery (grouped by
        Generate run); thumbnails are decoded in the background (each file once)."""
        cur = self.selected()
        if select is None and (keep or cur is not None):
            select = cur[0] if cur else None
        entries = []
        for png, meta in project_edits(self.project):
            try:
                entries.append((str(png), meta, png.stat().st_mtime, png.stat().st_mtime_ns))
            except OSError:
                continue
        self._entries = [(p, m, t) for p, m, t, _ns in entries]
        used = {p for p, m, _t in self._entries if self.uses(m)}
        self._in_use = used
        models = sorted({model_key(m) for _p, m, _t in self._entries}, key=model_name)
        want = self.f_model.currentData()
        self.f_model.blockSignals(True)
        self.f_model.clear()
        self.f_model.addItem("All models", None)
        for k in models:
            self.f_model.addItem(model_name(k), k)
        self.f_model.setCurrentIndex(max(0, self.f_model.findData(want)))
        self.f_model.blockSignals(False)
        show, model = self.f_show.currentData(), self.f_model.currentData()
        words = self.f_search.text().lower().split()

        def keep_it(png, meta):
            if show == "fav" and not meta.get("favorite"):
                return False
            if show == "used" and png not in used:
                return False
            if model and model_key(meta) != model:
                return False
            text = " ".join(str(meta.get(k) or "") for k in ("prompt", "negative")).lower()
            return all(w in text for w in words)
        shown = [(p, m, t, ns) for p, m, t, ns in entries if keep_it(p, m)]
        self.results.blockSignals(True)
        self.results.clear()
        for png, meta, _t, ns in shown:
            it = QListWidgetItem(self._strip_icon(png, meta), "")
            it.setData(Qt.UserRole, (png, meta))
            it.setToolTip(meta.get("prompt", Path(png).name))
            self.results.addItem(it)
            if self._thumbs.get(png, (None,))[0] != ns and (png, ns) not in self._requested:
                self._requested.add((png, ns))
                self._loader.request(png, ns)
        self.results.blockSignals(False)
        self.gallery.set_groups(run_groups([(p, m, t) for p, m, t, _ns in shown]), used)
        for png, (_ns, pix) in self._thumbs.items():
            self.gallery.set_thumb(png, pix)
        n, total = len(shown), len(entries)
        self.f_count.setText(f"{n} result{'s' if n != 1 else ''}" if n == total else f"{n} of {total}")
        if self.results.count():
            k = next((i for i in range(self.results.count())
                      if self.results.item(i).data(Qt.UserRole)[0] == select), 0)
            self.results.setCurrentRow(k)
        self._show_selected()

    def _strip_icon(self, png: str, meta: dict, selected: bool = False) -> QIcon:
        pix = self._thumbs.get(png, (None, QPixmap()))[1]
        return QIcon(badged(pix, QSize(150, 100), "", bool(meta.get("favorite")), png in self._in_use, selected))

    def _strip_selection(self, cur, prev):
        for it, on in ((prev, False), (cur, True)):
            if it is not None:
                png, meta = it.data(Qt.UserRole)
                it.setIcon(self._strip_icon(png, meta, on))

    def _thumb_ready(self, png: str, ns: int, img: QImage):
        self._requested.discard((png, ns))
        if img.isNull():
            return
        pix = QPixmap.fromImage(img)
        self._thumbs[png] = (ns, pix)
        for i in range(self.results.count()):
            it = self.results.item(i)
            if it.data(Qt.UserRole)[0] == png:
                it.setIcon(self._strip_icon(png, it.data(Qt.UserRole)[1], it is self.results.currentItem()))
                break
        self.gallery.set_thumb(png, pix)

    # ---- views, favourites
    def set_view(self, view: str, remember: bool = True):
        gallery = view == "gallery"
        self.stack.setCurrentIndex(1 if gallery else 0)
        (self.b_gallery if gallery else self.b_single).setChecked(True)
        self.thumb_size.setVisible(gallery)
        if remember and self.qs is not None:
            self.qs.setValue("resultsView", "gallery" if gallery else "single")
        sel = self.selected()
        if gallery:
            self.gallery.set_selected(sel[0] if sel else None)
            self.gallery.setFocus()
        else:
            self.results.setFocus()
            it = self.results.currentItem()
            if it is not None:
                self.results.scrollToItem(it)

    def view_mode(self) -> str:
        return "gallery" if self.stack.currentIndex() == 1 else "single"

    def _thumb_size_changed(self, px: int):
        self.gallery.set_thumb_size(px)
        if self.qs is not None:
            self.qs.setValue("resultsThumb", px)

    def _select_png(self, png: str):
        for i in range(self.results.count()):
            if self.results.item(i).data(Qt.UserRole)[0] == png:
                self.results.setCurrentRow(i)
                return

    def _gallery_clicked(self, png: str):
        self._select_png(png)
        self.gallery.setFocus()

    def _gallery_opened(self, png: str):
        self._select_png(png)
        self.set_view("single")

    def toggle_favorite(self, png: str | None = None) -> bool:
        """★ a result (the selected one if none is given) or take its star away; kept in its record."""
        sel = self.selected()
        png = png or (sel[0] if sel else None)
        if not png:
            return False
        f = Path(png).with_suffix(".json")
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        meta["favorite"] = not meta.get("favorite")
        if not meta["favorite"]:
            meta.pop("favorite")
        try:
            f.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            self.state.setText(f"Could not mark it: {e}")
            return False
        self._load_results(select=png if self.f_show.currentData() != "fav" or meta.get("favorite") else None,
                           keep=True)
        if self.view_mode() == "gallery":
            self.gallery.setFocus()
        return bool(meta.get("favorite"))

    def _use_seed(self, seed: int):
        self.seed.setValue(int(seed))
        self.fresh_seed.setChecked(False)
        self.state.setText(f"Seed {seed} is set for the next Generate (New seed after each Generate is off).")

    def selected(self) -> tuple[str, dict] | None:
        """The selected result and its record, read again (a use is recorded while the window is open)."""
        it = self.results.currentItem()
        if it is None:
            return None
        png, meta = it.data(Qt.UserRole)
        try:
            meta = json.loads(Path(png).with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        return png, meta

    def _show_selected(self):
        sel = self.selected()
        if sel is None:
            self.view.clear_frame("Results appear here")
            self.res_info.set(None)
            self.view.setToolTip("")
            self.gallery.set_selected(None)
            self.b_fav.setChecked(False)
        else:
            png, meta = sel
            orig = meta.get("source_copy") or meta.get("source")
            show = orig if self.b_before.isChecked() and orig and Path(orig).is_file() else png
            img = QImage(show)
            self.view.set_image(img)
            how = (f"{FAMILIES.get(meta.get('family'), {}).get('label', '')} "
                   f"{FAMILIES.get(meta.get('family'), {}).get('qualities', {}).get(meta.get('quality'), '')}"
                   if meta.get("mode") == "create" else
                   f"{EDIT_MODELS['qwen21']} {FAMILIES['qwen21']['qualities'].get(meta.get('quality'), '')}"
                   if meta.get("edit_model") == "qwen21" else
                   f"{EDIT_MODELS['moss']} {MOSS_EDIT_QUALITIES['standard']}" if meta.get("edit_model") == "moss"
                   else QUALITY_TEXT.get(meta.get("quality"), ""))
            smp = " · ".join(x for x in (f"{meta['steps']} steps" if meta.get("steps") else "",
                                         f"CFG {meta['cfg']:g}" if isinstance(meta.get("cfg"), (int, float)) else "",
                                         meta.get("sampler") or "", meta.get("scheduler") or "") if x)
            w, h = meta.get("width"), meta.get("height")
            if not (w and h) and not img.isNull():
                w, h = img.width(), img.height()
            self.res_info.set({
                "original": show != png, "where": self.uses(meta), "model": how.strip(), "sampling": smp,
                "size": f"{w} × {h}" if w and h else "",
                "loras": [f"{lora_label(lo['path'])} ×{lo['strength']:g}" for lo in meta.get("loras") or []],
                "seed": meta.get("seed"),
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(_mtime(Path(png)))),
                "values": (meta.get("variation") or {}).get("values") or {},
                "prompt": meta.get("prompt", ""), "trigger_words": meta.get("trigger_words") or [],
                "negative": meta.get("negative") or ""})
            self.view.setToolTip(self.res_info.toolTip())
            self.gallery.set_selected(png, scroll=self.view_mode() == "gallery")
            self.b_fav.setChecked(bool(meta.get("favorite")))
        self._update_buttons()

    def _update_buttons(self):
        has = self.selected() is not None
        idle = not self._busy
        text = bool(self.prompt.toPlainText().strip())
        if self.mode() == "create":
            fam = self.family.currentData()
            refs_ok = not self.use_refs.isChecked() or any(s.path for s in self.cpics[:FAMILIES.get(fam, {}).get("refs", 0)])
            ok = idle and text and fam in FAMILIES and "models" in (self.families.get(fam) or {}) and refs_ok
            why = ("Choose an available model" if fam not in FAMILIES or "models" not in (self.families.get(fam) or {})
                   else "Add a reference picture" if not refs_ok else "Describe the image")
        else:
            ok = idle and bool(self.source) and text and not self._mask_problem()
            why = self._mask_problem() or "Choose the image to edit and write what should change"
        self.b_gen.setEnabled(ok)
        self.b_mask.setEnabled(bool(self.source) and idle)
        if not ok:
            self.b_gen.setToolTip(why if idle else "Working …")
        self.b_cancel.setEnabled(not idle)
        for b in (self.b_use_start, self.b_use_ref, self.b_further, self.b_again, self.b_before, self.b_fav):
            b.setEnabled(has)
        self.b_use_end.setEnabled(has and self.end_target is not None)
        sel = self.selected()
        where = self.uses(sel[1]) if sel is not None else []
        self.b_delete.setEnabled(has and idle and not where)
        self.b_delete.setToolTip("Delete this result (Del); Undo delete brings it back" if not where else
                                 f"In use as {', '.join(where)}: it stays (the project keeps its own copy)")

    # ---- generating
    def generate(self):
        if self.mode() == "create":
            return self.generate_new()
        if self._busy or not self.source or not self.prompt.toPlainText().strip():
            return
        self._busy, self._cancel = True, False
        prompt = " ".join(self.prompt.toPlainText().split())
        source, quality, n = self.source, self.quality.currentData(), self.tries.value()
        model = self.edit_model.currentData()
        jobs = self._variation_jobs()
        if jobs is not None:
            if not jobs:
                self._busy = False
                return
            n = len(jobs)
        if self._mask_problem():
            self._busy = False
            self.state.setText(self._mask_problem())
            return
        mask = self.mask.copy() if self.mask is not None else None
        words = self._trigger_words()
        sent = with_triggers(prompt, words)
        refs = [p for p in (self.pic2.path, self.pic3.path) if p]
        loras = [(it["path"], it["strength"]) for it in self.loras.items()]
        self._remember_loras()
        size = None
        if self.keep_size.isChecked() and mask is None:     # (an inpaint comes back at the source's own size)
            img = upright(source)                 # the size the edit comes back at (a phone photo turned)
            size = (img.width() // 2 * 2, img.height() // 2 * 2)
        edits = self.project / EDITS_DIR
        sig, connect = self._sig, self.connect_editor
        if not self._run_on_start():
            return
        seeds = self._seeds(n)                         # (after the checks: a refused Generate keeps the seed)
        self.bar.setValue(0)
        self.bar.show()
        self.b_start_comfy.hide()
        self.state.setText("Connecting to ComfyUI … (the first edit loads the model, which takes a while; if a "
                           "clip is being generated, the edit waits for it)")
        self._update_buttons()

        run = uuid.uuid4().hex[:12]                  # this Generate's results belong together

        def work():
            kept = mask_file = None
            made = 0
            try:
                edits.mkdir(parents=True, exist_ok=True)
                cur = {"k": 1}
                self.editor = connect(lambda i, s: sig.progress.emit(cur["k"], n, i, s))
                if model not in self.editor.models():
                    raise RuntimeError(f"{EDIT_MODELS[model]} is not available in ComfyUI")
                if quality == "fast" and "fast" not in self.editor.qualities(model):
                    sig.progress.emit(0, n, 0, 0)
                kept = edits / "sources" / (f"source_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
                                            f"{Path(source).suffix.lower() or '.png'}")
                kept.parent.mkdir(exist_ok=True)
                kept.write_bytes(Path(source).read_bytes())         # the original stays with its results
                if mask is not None:                                 # the mask too (white = drawn anew)
                    mask_file = kept.with_name(kept.stem.replace("source_", "mask_", 1) + ".png")
                    if not mask.save(str(mask_file), "PNG"):
                        raise RuntimeError("could not save the mask")
                for k in range(1, n + 1):
                    if self._cancel:
                        break
                    cur["k"] = k
                    seed = seeds[k - 1]
                    stem = f"edit_{time.strftime('%Y%m%d-%H%M%S')}_{seed}"
                    out = edits / f"{stem}.png"
                    job = jobs[k - 1] if jobs else None
                    meta = self.editor.edit(Path(source), with_triggers(job.prompt, words) if job else sent, out, seed,
                                            quality, [Path(r) for r in refs], loras, model=model, mask=mask_file)
                    made += 1
                    if size is not None:
                        scale_to(out, out, size)
                    meta.update({"run": run, "prompt": job.prompt if job else prompt, "trigger_words": words,
                                 "seed": seed, "source": source,
                                 "source_copy": str(kept), **({"mask": str(mask_file)} if mask_file else {}),
                                 "refs": refs, "size": list(size) if size else None,
                                 **(self._variation_meta(job, prompt, job.key) if job else {})})
                    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")
                    sig.result.emit(str(out), meta)
                sig.finished.emit("")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.finished.emit("cancelled" if self._cancel else str(e))
                except RuntimeError:
                    pass
            finally:
                if not made:                     # no result refers to them: the copies of source and mask go
                    for f in (kept, mask_file):
                        if f is not None:
                            Path(f).unlink(missing_ok=True)
        threading.Thread(target=work, daemon=True).start()

    # ---- sampling (Create): steps, cfg, sampler, scheduler; each "auto" = what the model uses by itself
    def _build_sampling(self, cv):
        self._sampling_family, self._sampling_loading = None, False
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
        for w in (self.steps, self.cfg):
            guard_wheel(w)
        grid = QGridLayout()                       # two rows, so the panel stays narrow
        for k, (w, text) in enumerate(((self.steps, "Steps"), (self.cfg, "CFG"), (self.sampler, "Sampler"),
                                       (self.scheduler, "Scheduler"))):
            grid.addWidget(label(text, level="tertiary"), k // 2, (k % 2) * 2)
            grid.addWidget(w, k // 2, (k % 2) * 2 + 1)
            w.setMinimumWidth(60)
            if isinstance(w, QComboBox):
                w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
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
        self._sampling_auto: dict = {}
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
        out = {}
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
        if not hasattr(self, "steps") or not hasattr(self, "quality"):
            return
        fam = self.family.currentData()
        if self.sampler.count() <= 1 or (self.families or {}).get("sampling"):
            self._fill_sampling_lists()
        if fam != self._sampling_family:
            self._sampling_family = fam
            saved = {}
            if self.qs is not None and fam:
                try:
                    saved = json.loads(self.qs.value(self._sampling_key(), "{}", type=str) or "{}")
                except ValueError:
                    saved = {}
            self.set_sampling(saved if isinstance(saved, dict) else {})
        refs = sum(1 for s in self.cpics if s.path) if self.use_refs.isChecked() else 0
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
        if not hasattr(self, "negative"):
            return
        f = FAMILIES.get(self.family.currentData()) if self.mode() == "create" else None
        own = bool(f and f["negative"])
        cfg_on = hasattr(self, "cfg") and self.cfg.value() > 1
        self.negative.setVisible(bool(f) and (own or cfg_on))
        self.negative.setPlaceholderText("Avoid (optional negative prompt; empty: the model's usual one)" if own else
                                         f"Avoid (optional negative prompt; counts because CFG is {self.cfg.value():g})"
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

    # ---- LoRA presets (Create): named prompt + negative + quality + sampling + strength, several per LoRA
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
        return [it["path"] for it in self.loras.items() if it.get("path")] if hasattr(self, "loras") else []

    def _fill_presets(self, select: tuple[str, str] | None = None):
        if not hasattr(self, "preset"):
            return
        from .. import presets
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
        return tuple(data.split("\n", 1)) if data else None

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
        from .. import presets
        p = presets.store().get(*data)
        if p is None:
            self._fill_presets()
            return
        if p.get("prompt") is not None:
            self.prompt.setPlainText(p["prompt"])
        if p.get("negative") is not None:
            self.negative.setText(p["negative"])
        if p.get("variations") is not None:
            self.variations.set_state(p["variations"])
        q = self.quality.findData(p.get("quality"))
        if q >= 0:
            self.quality.setCurrentIndex(q)
        self.set_sampling(p.get("sampling") or {})
        if p.get("strength") is not None:
            items = [dict(it, strength=float(p["strength"])) if os.path.realpath(it["path"]) ==
                     os.path.realpath(data[0]) else it for it in self.loras.items()]
            self.loras.set_items(self._known(), items)
            self._remember_loras()
            self._fill_presets(data)
        self.state.setText(f"Preset “{p['name']}” of {lora_label(data[0])} is in the fields.")

    def save_preset_as(self, name: str | None = None, lora: str | None = None) -> bool:
        """A new preset from the fields (asks for its name, and for the LoRA when there are several)."""
        from .. import presets
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
        if presets.store().get(lora, name) is not None and QMessageBox.question(
                self, "New preset", f"{lora_label(lora)} already has a preset “{name}”. Replace it?") \
                != QMessageBox.Yes:
            return False
        try:
            presets.store().save(lora, name, self._preset_values(lora))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "New preset", f"Could not save the preset: {e}")
            return False
        self._fill_presets((lora, name))
        self.state.setText(f"Saved the preset “{name}” of {lora_label(lora)}.")
        return True

    def update_preset(self) -> bool:
        """The chosen preset takes the current prompt and settings."""
        from .. import presets
        data = self._chosen_preset()
        if not data:
            return False
        try:
            presets.store().save(data[0], data[1], self._preset_values(data[0]))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Preset", f"Could not save the preset: {e}")
            return False
        self._fill_presets(data)
        self.state.setText(f"Saved the changes to the preset “{data[1]}”.")
        return True

    def rename_preset(self, new: str | None = None) -> bool:
        from .. import presets
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
            QMessageBox.warning(self, "Rename preset", f"Could not rename it: {e}")
            return False
        self._fill_presets((data[0], new))
        return True

    def delete_preset(self, confirm: bool = True) -> bool:
        from .. import presets
        data = self._chosen_preset()
        if not data:
            return False
        if confirm and QMessageBox.question(self, "Delete preset", f"Delete the preset “{data[1]}” of "
                                            f"{lora_label(data[0])}?") != QMessageBox.Yes:
            return False
        try:
            presets.store().delete(*data)
        except OSError as e:
            QMessageBox.warning(self, "Delete preset", f"Could not delete it: {e}")
            return False
        self._fill_presets()
        self.state.setText(f"Deleted the preset “{data[1]}”.")
        return True

    # ---- prompt variations (variations_panel.py)
    def _negative_used(self) -> bool:
        """The negative prompt goes with the next Generate (create mode, its field shown for this model)."""
        return self.mode() == "create" and self.negative.isVisibleTo(self)

    def _templates_changed(self):
        self.variations.set_templates(self.prompt.toPlainText(),
                                      self.negative.text() if self._negative_used() else "")

    def _variations_changed(self):
        if self.qs is not None:
            st = self.variations.state()
            self.qs.setValue("promptVariations", json.dumps(st, ensure_ascii=False))
        if not getattr(self, "_rehighlighting", False):
            self._rehighlighting = True
            try:
                self._highlight.rehighlight()
            finally:
                self._rehighlighting = False
        active = self.variations.active()
        if hasattr(self, "tries"):
            self.tries.setEnabled(not active)
            self.tries.setToolTip("Variations decide how many images are made (Make … below the prompt)" if active
                                  else f"How many results at once (1 … {MAX_RESULTS}; each with the next seed), to "
                                       "pick the best one. They are made one after another and each appears when it "
                                       "is ready; Cancel stops after the current one")
        if hasattr(self, "b_gen"):
            n = self.variations.amount() if active else 0
            self.b_gen.setText(f"Generate {n} image{'s' if n != 1 else ''}" if active else "Generate")

    def _insert_placeholder(self, name: str):
        cur = self.prompt.textCursor()
        before = self.prompt.toPlainText()[:cur.position()]
        cur.insertText(("" if not before or before[-1].isspace() else " ") + "{" + name + "}")
        self.prompt.setTextCursor(cur)
        self.prompt.setFocus()

    def _rename_placeholder(self, old: str, new: str):
        import re as _re
        pat = _re.compile(r"\{(\??)" + _re.escape(old) + r"([|}])")
        from .prompt_refiner import replace_text
        replace_text(self.prompt, pat.sub(lambda m: "{" + m.group(1) + new + m.group(2), self.prompt.toPlainText()))
        self.negative.setText(pat.sub(lambda m: "{" + m.group(1) + new + m.group(2), self.negative.text()))

    def _pick_list(self) -> str | None:
        return pick_image(self, "Choose a wildcard list (.txt, one value per line)", self.qs, kind="text")

    def _variation_jobs(self) -> list | None:
        """The next Generate's prompts when variations are active (None: not active; [] after a refusal)."""
        if not self.variations.active():
            return None
        self._templates_changed()
        many = self.variations.too_many()
        from ..variations import MAX_IMAGES
        if many and QMessageBox.question(self, "Variations", f"Every combination makes {many:,} images"
                                         + (f" (one Generate makes at most {MAX_IMAGES:,})" if many > MAX_IMAGES
                                            else "") + ". Make them all?") != QMessageBox.Yes:
            return []
        jobs = self.variations.jobs()
        key = self.variations.position_key()
        for j in jobs:                                     # (as the plain prompt: one line, single spaces)
            j.prompt, j.negative, j.key = " ".join(j.prompt.split()), " ".join(j.negative.split()), key
        return jobs

    def _variation_meta(self, job, prompt: str, key: str) -> dict:
        return {"template": prompt, "variation": {"values": job.combo, "step": job.step, "index": job.index,
                                                  "key": key}}

    def new_seed(self):
        """A fresh random seed in the field."""
        self.seed.setValue(random.randint(1, SEED_MAX))

    def _seeds(self, n: int) -> list[int]:
        """The seeds of the next n results (the field's, +1 ...); a fresh one in the field afterwards if wanted."""
        first = self.seed.value()
        seeds = [(first - 1 + k) % SEED_MAX + 1 for k in range(n)]
        if self.fresh_seed.isChecked():
            self.new_seed()
        return seeds

    def generate_new(self):
        """Create mode: new images from the prompt (and reference pictures) with the chosen model family."""
        if self._busy or not self.b_gen.isEnabled():
            return
        self._busy, self._cancel = True, False
        fam, model = self.family.currentData(), self.model_file.currentData()
        prompt = " ".join(self.prompt.toPlainText().split())
        negative = " ".join(self.negative.text().split()) if self._negative_used() else ""
        quality, n = self.quality.currentData(), self.tries.value()
        jobs = self._variation_jobs()
        if jobs is not None:
            if not jobs:
                self._busy = False
                return
            n = len(jobs)
        words = self._trigger_words()
        sent = with_triggers(prompt, words)
        refs = [s.path for s in self.cpics[:FAMILIES[fam]["refs"]] if s.path] if self.use_refs.isChecked() else []
        loras = [(it["path"], it["strength"]) for it in self.loras.items()]
        self._remember_loras()                    # what it was made with is offered next time
        w, h = size_for(fam, self.aspect.currentData() or 1.0)
        strength, tags = self.ref_strength.value(), self.tags.isChecked()
        sampling = self.sampling()
        edits = self.project / EDITS_DIR
        sig, connect = self._sig, self.connect_create
        if not self._run_on_start():
            return
        seeds = self._seeds(n)                         # (after the checks: a refused Generate keeps the seed)
        self.bar.setValue(0)
        self.bar.show()
        self.b_start_comfy.hide()
        self.state.setText(f"Connecting to ComfyUI … (the first image loads {FAMILIES[fam]['label']}, which takes a "
                           "while)")
        self._update_buttons()

        run = uuid.uuid4().hex[:12]                  # this Generate's results belong together

        def work():
            try:
                edits.mkdir(parents=True, exist_ok=True)
                cur = {"k": 1}
                self.editor = connect(lambda i, st: sig.progress.emit(cur["k"], n, i, st))
                kept = None
                if refs:                              # the first picture is shown as the "original"
                    kept = edits / "sources" / (f"source_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
                                                f"{Path(refs[0]).suffix.lower() or '.png'}")
                    kept.parent.mkdir(exist_ok=True)
                    kept.write_bytes(Path(refs[0]).read_bytes())
                for k in range(1, n + 1):
                    if self._cancel:
                        break
                    cur["k"] = k
                    seed = seeds[k - 1]
                    out = edits / f"new_{time.strftime('%Y%m%d-%H%M%S')}_{seed}.png"
                    job = jobs[k - 1] if jobs else None
                    neg_k = (" ".join(job.negative.split()) if negative else "") if job else negative
                    meta = self.editor.create(fam, model, with_triggers(job.prompt, words) if job else sent, out, seed,
                                              quality, w, h, neg_k, loras, [Path(r) for r in refs], strength, tags,
                                              sampling)
                    meta.update({"run": run, "prompt": job.prompt if job else prompt, "trigger_words": words,
                                 "seed": seed, "refs": refs, **(self._variation_meta(job, prompt, job.key) if job else {}),
                                 "tags": tags,
                                 "source_copy": str(kept) if kept else None})
                    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")
                    sig.result.emit(str(out), meta)
                sig.finished.emit("")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.finished.emit("cancelled" if self._cancel else str(e))
                except RuntimeError:
                    pass
        threading.Thread(target=work, daemon=True).start()

    def _run_on_start(self) -> bool:
        try:
            if self.on_start is not None:
                self.on_start()
            return True
        except Exception as e:  # noqa: BLE001
            self._busy = False
            self.state.setText(f"Could not start: {e}")
            self._update_buttons()
            return False

    def cancel(self):
        if not self._busy:
            return
        self._cancel = True
        self.state.setText("Cancelling …")
        ed = self.editor
        if ed is not None:
            threading.Thread(target=ed.interrupt, daemon=True).start()

    def _progress(self, k: int, n: int, i: int, steps: int):
        if k == 0:
            self.state.setText("No Lightning LoRA for this model in ComfyUI: using Standard (20 steps).")
            return
        frac = (k - 1 + (i / steps if steps else 0)) / max(1, n)
        self.bar.setValue(round(1000 * frac))
        self.state.setText(f"Result {k} of {n}: step {i} / {steps}")

    def _result(self, png: str, meta: dict):
        if isinstance(meta.get("variation"), dict) and meta["variation"].get("step") is not None:
            self.variations.done(int(meta["variation"]["step"]), meta["variation"].get("key"))   # continues after
        self._load_results(select=png)
        self.b_before.setChecked(False)

    def _finished(self, error: str):
        self._busy = False
        self.bar.hide()
        if error == "cancelled":
            self.state.setText("Cancelled.")
        elif error:
            down = "not reachable" in error
            self.state.setText(("ComfyUI is not running. " if down else "") + f"Could not edit the image: {error}")
            self.b_start_comfy.setVisible(down and self.start_comfy is not None)
        else:
            self.state.setText("Done. Pick a result below, compare it with the original (B), then use it.")
        self._update_buttons()

    def _start_comfy(self):
        if self.start_comfy and self.start_comfy():
            self.state.setText("Starting ComfyUI … the models are listed as soon as it answers; then press "
                               "Generate.")
            self.b_start_comfy.hide()
            if self.connect_create is not None:
                self._retry_families = 36              # up to 3 minutes
                QTimer.singleShot(5000, self._load_families)

    # ---- using a result
    def _use(self, kind: str):
        sel = self.selected()
        if sel is None:
            return
        if self.on_use(kind, sel[0]):
            self.used = (kind, sel[0])
            self.accept()

    # ---- deleting bad results
    def delete_selected(self):
        """Delete the selected result if it is not in use; it can be brought back until the window closes."""
        sel = self.selected()
        if sel is None or self._busy or self.uses(sel[1]):
            return
        png = Path(sel[0])
        self._empty_trash()
        trash = self.project / EDITS_DIR / ".deleted"
        trash.mkdir(exist_ok=True)
        row = self.results.currentRow()
        for f in (png, png.with_suffix(".json")):
            if f.exists():
                f.replace(trash / f.name)
        self._trash = trash / png.name
        self._load_results()
        if self.results.count():
            self.results.setCurrentRow(min(row, self.results.count() - 1))
        self.b_undo.show()
        self.state.setText(f"Deleted {png.name}.")

    def undo_delete(self):
        if self._trash is None:
            return
        back = self.project / EDITS_DIR / self._trash.name
        for f in (self._trash, self._trash.with_suffix(".json")):
            if f.exists():
                f.replace(back.with_suffix(f.suffix))
        self._trash = None
        self.b_undo.hide()
        self._load_results(select=str(back))
        self.state.setText(f"{back.name} is back.")

    def _empty_trash(self, keep_sources: bool = False):
        """Delete for good what was deleted before, and originals no result needs any more."""
        trash = self.project / EDITS_DIR / ".deleted"
        if trash.is_dir():
            for f in trash.iterdir():
                f.unlink(missing_ok=True)
            trash.rmdir()
        self._trash = None
        if keep_sources:
            return
        needed = {m.get("source_copy") for _p, m in project_edits(self.project)}
        for f in (self.project / EDITS_DIR / "sources").glob("*") if (self.project / EDITS_DIR / "sources").is_dir() \
                else []:
            if str(f) not in needed and f.stat().st_mtime < time.time() - 5:
                f.unlink(missing_ok=True)

    def prompt_kind(self) -> str:
        return "image" if self.mode() == "create" else "edit"

    def _refine_prompt(self):
        if self.refine is None:
            return
        import inspect
        try:
            takes = "pictures" in inspect.signature(self.refine).parameters
        except (TypeError, ValueError):
            takes = False
        new = self.refine(self.prompt.toPlainText(), self.prompt_kind(),
                          **({"pictures": self._refine_pictures()} if takes else {}))
        if new:
            from .prompt_refiner import replace_text
            replace_text(self.prompt, new)

    def _refine_pictures(self) -> list[tuple[str, str, bool]]:
        """What the refiner's model may see: the image to edit and the extra pictures (edit), the reference
        pictures (create), and the selected result (off at first)."""
        out = []
        if self.mode() == "edit":
            if self.source:
                out.append(("The image to edit (Picture 1)", self.source, True))
            for k, p in ((2, self.pic2.path), (3, self.pic3.path)):
                if p:
                    out.append((f"Picture {k}", p, True))
        elif self.use_refs.isChecked():
            out += [(f"Reference picture {k + 1}", s.path, True) for k, s in enumerate(self.cpics) if s.path]
        sel = self.selected()
        if sel is not None:
            out.append(("A result made here", sel[0], False))
        return out

    def _add_reference(self):
        """The selected result as one more face reference; the dialog stays open."""
        sel = self.selected()
        if sel is None:
            return
        msg = self.on_use("ref", sel[0])
        self.state.setText(msg if isinstance(msg, str) and msg else
                           "Added as a face reference." if msg else "Could not add it as a face reference "
                                                                   "(see the status bar).")
        self._show_selected()                      # now in use: shown, and it cannot be deleted

    def _edit_further(self):
        sel = self.selected()
        if sel is not None:
            if self.mode() != "edit":
                self.set_mode("edit")
            if self.src_combo.findData(sel[0]) < 0:
                self.src_combo.insertItem(self.src_combo.count() - 1, f"Result: {sel[1].get('prompt', '')[:40]}",
                                          sel[0])
            self.src_combo.setCurrentIndex(self.src_combo.findData(sel[0]))
            self.set_source(sel[0])
            self.prompt.clear()
            self.prompt.setFocus()

    def _prompt_again(self):
        """Back to what the selected result was made with: prompt and LoRAs; for a new image also its model,
        quality, negative prompt and tags."""
        sel = self.selected()
        if sel is None:
            return
        meta = sel[1]
        if meta.get("mode") == "create" and self.connect_create is not None:
            if self.mode() != "create":
                self.set_mode("create")
            i = self.family.findData(meta.get("family"))
            if i >= 0 and self.family.model().item(i).isEnabled():
                self.family.setCurrentIndex(i)
            j = self.model_file.findData(meta.get("chosen_model") or meta.get("model"))
            if j >= 0:
                self.model_file.setCurrentIndex(j)
            q = self.quality.findData(meta.get("quality"))
            if q >= 0:
                self.quality.setCurrentIndex(q)
            self.negative.setText(meta.get("negative") or "")
            self.set_sampling(meta.get("sampling_set") or {})
            if "tags" in meta:
                self.tags.setChecked(bool(meta["tags"]))
        elif meta.get("mode") != "create":
            if self.mode() != "edit":
                self.set_mode("edit")
            i = self.edit_model.findData(meta.get("edit_model") or "2511")
            if i >= 0 and self.edit_model.model().item(i).isEnabled():
                self.edit_model.setCurrentIndex(i)
            q = self.quality.findData(meta.get("quality"))
            if q >= 0:
                self.quality.setCurrentIndex(q)
            self.set_mask(None)                   # (a result made without a mask: none)
            m = meta.get("mask")                  # an inpaint of the image shown: its mask again
            if m and Path(m).is_file() and meta.get("source") == self.source:
                img = QImage(m).convertToFormat(QImage.Format_Grayscale8)
                if not img.isNull() and img.size() == upright(self.source).size():
                    self.set_mask(img)
        if meta.get("template"):                  # made from variations: the prompt with its placeholders
            self.prompt.setPlainText(meta["template"])
            self.variations.on.setChecked(True)
        else:
            self.prompt.setPlainText(meta.get("prompt", ""))
        self.loras.set_items(self._known(), meta.get("loras") or [])
        self._remember_loras()

    # ---- LoRAs
    def _trigger_words(self) -> list[str]:
        """The trigger words of the chosen LoRAs (put before the prompt when the project wants them)."""
        if not self.use_triggers:
            return []
        return stack_words([it["path"] for it in self.loras.items()])

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
        import os
        p = os.path.abspath(path)
        if self.qs is not None:
            self.qs.setValue(self._recent_key(), json.dumps([p] + [r for r in self._recent_files() if r != p][:11]))
        self.loras.set_items(self._known(), self.loras.items() + [{"path": p, "strength": float(strength)}])
        self._remember_loras()

    def done(self, r):
        busy = self._busy
        if busy:
            self.cancel()
        self._empty_trash(keep_sources=busy)       # a running job may still write its result with its source
        self._loader.stop()
        self.closed = True
        super().done(r)
