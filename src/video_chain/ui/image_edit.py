"""Edit an image by a prompt (Qwen Image Edit in ComfyUI) and use the result as the start image or as a clip's end
image. Every result is kept in the project (edits/<name>.png + .json with its prompt, the image it was made from in
edits/sources/), so earlier tries stay at hand.

1. the image to edit (start image, a clip's frame, an end image, a face reference, any file, an earlier result)
2. what should change (the prompt), optionally up to two more pictures (Picture 2 / 3, e.g. a face)
3. Generate: one or more tries (other seeds); 4. pick a result, compare it with the original, use it.

The dialog is one window made of parts in sibling modules: image_models (modes, model families), image_source
(image to edit, folder, mask), image_prompt (variations, seeds, refiner), image_loras, image_sampling,
image_presets, image_results (strip, gallery, favourites, delete); image_generate runs a Generate in a thread and
`video_chain.edits` holds the records and files without Qt.
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from PySide6.QtCore import QSettings, QSize, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from ..comfy import EDIT_MODELS
# (main_window and the tests import these from here: they moved to video_chain/edits.py)
from ..edits import (EDITS_DIR, MOSS_EDIT_QUALITIES, QUALITY_TEXT, ResultRecord, how_made, lora_note,  # noqa: F401
                     project_edits, record_use, scale_to, uses_index, uses_of, write_meta)
from ..imagegen import FAMILIES, size_for
from . import shortcuts as S
from .dialogs import button_row, confirm as ask
from .image_common import (ASK_ABOVE, CREATE_HINT, FOLDER_ITEM, FOLDER_PREFIX, MAX_RESULTS, PROMPT_HINT,  # noqa: F401
                           SEED_MAX, PictureSlot)
from .image_generate import CreatePlan, EditPlan, CreateWorker, EditWorker, GenerateWorker
from .image_loras import LorasMixin
from .image_models import ModelsMixin
from .image_presets import PresetsMixin
from .image_prompt import PromptMixin
from .image_results import ResultsMixin
from .image_sampling import SamplingMixin
from .image_source import SourceMixin
from .imageio import decode_upright as _decode_upright, upright  # noqa: F401  (tests / main_window import from here)
from .lora_stack import LoraStack
from .results_gallery import InfoCard, ResultGallery, ResultView, ThumbLoader
from .variations_panel import PlaceholderHighlighter, VariationsPanel
from .widgets import (FrameTile, fit_size, guard_wheel, icon_button, label, remember_geometry,
                      shortcut, combo_item, label_for)
from .qsvalue import qs_value


@dataclass
class ImageEditServices:
    """What the image dialog gets from the app: the calls it makes out (connect to ComfyUI, use a result, start
    ComfyUI, refine a prompt ...) and the app's data it shows (LoRAs, places, the start image's aspect, options).
    connect and on_use are required; every other field has a harmless default."""
    connect: Callable[[Callable | None], object] | None = None       # connect(progress) -> a ComfyImageEditor
    on_use: Callable[[str, str], bool] | None = None                 # on_use(kind 'start' | 'end', png) -> bool
    settings: QSettings | None = None
    places: list[tuple[str, str]] | None = None
    start_comfy: Callable[[], bool] | None = None
    on_start: Callable[[], None] | None = None                      # called (GUI thread) before each Generate
    known_loras: list[tuple[str, str]] | None = None
    lora_dir: str = ""
    uses: Callable[[ResultRecord], list[str]] | None = None          # where a result is used (words)
    refine: Callable[..., str | list[str] | None] | None = None      # refine(text, kind) -> new text or None
    connect_create: Callable[[Callable | None], object] | None = None
    family_loras: dict[str, list] | None = field(default=None)
    start_aspect: float | None = None
    mode: str = "edit"
    check_lora: Callable[[str | None], bool | None] | None = None    # False if ComfyUI cannot load the LoRA
    use_triggers: bool = True
    collections: Callable[[], list] | None = None                   # () -> the project's pictures by kind


class ImageEditDialog(ModelsMixin, SourceMixin, ResultsMixin, PromptMixin, LorasMixin, SamplingMixin, PresetsMixin,
                      QDialog):
    """sources: [(label, path)] offered as the image to edit; end_target: the clip a result can become the end
    image of (None: only the start image). connect(progress) -> a ComfyImageEditor (called in a thread);
    on_use(kind 'start' | 'end', png) -> bool; start_comfy() -> bool."""

    def __init__(self, parent, project: Path, sources: list[tuple[str, str]], source: str | None,
                 end_target: str | None, connect: Callable[[Callable | None], object] | None = None,
                 on_use: Callable[[str, str], bool] | None = None, settings: QSettings | None = None,
                 places: list[tuple[str, str]] | None = None, start_comfy: Callable[[], bool] | None = None,
                 on_start: Callable[[], None] | None = None, known_loras: list[tuple[str, str]] | None = None,
                 lora_dir: str = "", uses: Callable[[ResultRecord], list[str]] | None = None,
                 refine: Callable[..., str | list[str] | None] | None = None,
                 connect_create: Callable[[Callable | None], object] | None = None,
                 family_loras: dict[str, list] | None = None, start_aspect: float | None = None, mode: str = "edit",
                 check_lora: Callable[[str | None], bool | None] | None = None, use_triggers: bool = True,
                 collections: Callable[[], list] | None = None, *, services: ImageEditServices | None = None):
        """What the dialog is for (project, the images to offer, the one chosen, the clip it ends) is given as
        arguments; everything it calls out to or is told by the app is one ImageEditServices (`services`). The
        single keyword arguments after end_target are the older way to give the same and still work."""
        super().__init__(parent)
        if services is None:
            services = ImageEditServices(
                connect=connect, on_use=on_use, settings=settings, places=places, start_comfy=start_comfy,
                on_start=on_start, known_loras=known_loras, lora_dir=lora_dir, uses=uses, refine=refine,
                connect_create=connect_create, family_loras=family_loras, start_aspect=start_aspect, mode=mode,
                check_lora=check_lora, use_triggers=use_triggers, collections=collections)
        if services.connect is None or services.on_use is None:
            raise TypeError("ImageEditDialog needs connect and on_use (in services or as arguments)")
        self.services = services
        self.setWindowTitle("Edit an image with AI")
        fit_size(self, 1240, 820)
        remember_geometry(self, services.settings, "imageEdit")
        self._init_state(project, end_target, services)
        known_loras, lora_dir, family_loras, mode = (services.known_loras, services.lora_dir, services.family_loras,
                                                     services.mode)
        top = QHBoxLayout(self)
        left = QVBoxLayout()
        left.setSpacing(6)
        self._build_mode_row(left)
        self._build_edit_box(left, sources)
        self._build_create_box(left)
        self._build_prompt_box(left)
        self._build_lora_box(left, known_loras, lora_dir, family_loras)
        self._build_options(left)
        self._build_actions(left)
        top.addWidget(self._scrolled(left))
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        top.addWidget(line)
        top.addLayout(self._build_results_panel(), 1)
        self._built = True                               # (every widget the parts above talk to exists now)
        self._fill_presets()
        self._add_shortcuts()
        self._restore(source, sources, mode)

    # ---- building the window
    def _init_state(self, project: Path, end_target: str | None, svc: ImageEditServices):
        connect, on_use = cast("Callable[[Callable | None], object]", svc.connect), cast("Callable[[str, str], bool]", svc.on_use)
        settings, places, start_comfy, on_start = svc.settings, svc.places, svc.start_comfy, svc.on_start
        known_loras, lora_dir, uses, refine = svc.known_loras, svc.lora_dir, svc.uses, svc.refine
        connect_create, family_loras, start_aspect = svc.connect_create, svc.family_loras, svc.start_aspect
        check_lora, use_triggers, collections = svc.check_lora, svc.use_triggers, svc.collections
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
        self._was_down = False                           # ComfyUI was not reachable at the last listing
        self._retry_families = 0                         # retries left while ComfyUI starts
        self._trash: list[Path] = []                     # the results deleted (Undo delete brings back the last)
        self._thumbs: dict[str, tuple[int, QPixmap]] = {}  # result thumbnails: png -> (its mtime_ns, pixmap)
        self._requested: set[tuple[str, int]] = set()
        self._failed: set[tuple[str, int]] = set()
        self._entries: list[tuple[str, ResultRecord, float]] = []
        self._in_use: set[str] = set()                   # the results that are used somewhere
        self._shown: list[tuple[str, ResultRecord, float, int]] = []   # the strip's results: png, record, mtime, ns
        self._strip_items: dict[str, QListWidgetItem] = {}
        self._loader = ThumbLoader(self)
        self._loader.ready.connect(self._thumb_ready)
        self._built = False                              # True once every widget exists (parts talk to each other)
        self._rehighlighting = False
        self._sampling_family: str | None = None
        self._sampling_loading = False
        self._sampling_auto: dict = {}                   # the model's own settings, shown as "auto (…)"
        self.mask = None                                 # white = drawn anew, at the source's (upright) size
        self.known_loras, self.lora_dir = list(known_loras or []), lora_dir
        self.family_loras = family_loras or {}
        self.closed = False
        self.source: str | None = None
        self.batch: list[str] = []                       # a folder chosen: its images (each edited in turn)
        self._batch_dir = ""
        self._batch_names: list[str] = []                # (the images of the running Generate, for its progress)
        self._per_image = 1
        self._skipped: list[str] = []
        self._worker: GenerateWorker | None = None       # the running Generate
        self._busy = False
        self.used: tuple[str, str] | None = None        # (kind, png) once a result was used
    def _build_mode_row(self, left: QVBoxLayout):
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

    def _build_edit_box(self, left: QVBoxLayout, sources: list[tuple[str, str]]):
        # ---- edit: the image to edit
        self.edit_box = QWidget()
        ev = QVBoxLayout(self.edit_box)
        ev.setContentsMargins(0, 0, 0, 0)
        ev.addWidget(label("1  Image to edit", type_="heading"))
        self.src_tile = FrameTile(380, 214)
        self.src_tile.clear_frame("Choose the image to edit")
        self.src_tile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.src_tile.clicked.connect(lambda _t: self._choose_source())
        ev.addWidget(self.src_tile)
        self.src_combo = QComboBox()
        self.src_combo.setToolTip("Images of this project; Choose file… for any image; Choose folder… to edit "
                                  "every image of a folder in turn with the same settings")
        for text, path in sources:
            self.src_combo.addItem(text, path)
        self.src_combo.addItem("Choose file…", None)
        self.src_combo.addItem("Choose folder…", FOLDER_ITEM)
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
        saved = qs_value(self.qs, "editModel", "", str) if self.qs is not None else ""
        self.edit_model.setCurrentIndex(max(0, self.edit_model.findData(saved)))
        self.edit_model.currentIndexChanged.connect(self._edit_model_changed)
        self.edit_model.activated.connect(lambda _i: self._remember("editModel", self.edit_model.currentData()))
        erow.addWidget(label_for("Model", self.edit_model, level="tertiary"))
        erow.addWidget(self.edit_model, 1)
        ev.addLayout(erow)
        mrow2 = QHBoxLayout()
        self.b_mask = QPushButton("Paint mask…")
        self.b_mask.setToolTip("Paint the part that should change; the rest stays exactly as it is (inpainting "
                               "with Qwen-Image 2.1 and its Fun ControlNet)")
        self.b_mask.clicked.connect(self.paint_mask)
        self.b_mask_clear = QPushButton("Remove mask")
        self.b_mask_clear.setToolTip("Edit the whole image again")
        self.b_mask_clear.clicked.connect(lambda: self.set_mask(None))
        self.mask_info = label("", level="tertiary")
        self.mask_info.setWordWrap(True)
        mrow2.addWidget(self.b_mask)
        mrow2.addWidget(self.b_mask_clear)
        mrow2.addWidget(self.mask_info, 1)
        ev.addLayout(mrow2)
        self.mask = None                              # white = drawn anew, at the source's (upright) size
        left.addWidget(self.edit_box)

    def _build_create_box(self, left: QVBoxLayout):
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
        self.family.activated.connect(lambda _i: self._remember("createFamily", self.family.currentData(),
                                                                needs_value=True))
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
        arow.addWidget(label_for("Shape", self.aspect, level="tertiary"))
        arow.addWidget(self.aspect, 1)
        arow.addWidget(self.size_text)
        cv.addLayout(arow)
        self._build_sampling(cv)
        self._build_presets(cv)
        left.addWidget(self.create_box)

    def _build_prompt_box(self, left: QVBoxLayout):
        prow0 = QHBoxLayout()
        self.prompt_title = label("2  What should change?", type_="heading")
        prow0.addWidget(self.prompt_title)
        prow0.addStretch(1)
        self.b_refine = icon_button("gen", "Refine this prompt with a language model")
        self.b_refine.clicked.connect(self._refine_prompt)
        self.b_refine.setVisible(self.refine is not None)
        prow0.addWidget(self.b_refine)
        left.addLayout(prow0)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(PROMPT_HINT)
        self.prompt.setMinimumHeight(110)
        self.prompt.textChanged.connect(self._update_generate_button)   # (typing changes nothing else)
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

    def _build_lora_box(self, left: QVBoxLayout, known_loras, lora_dir, family_loras):
        self.lora_text = label("", level="tertiary")
        self.lora_text.setWordWrap(True)
        left.addWidget(self.lora_text)
        self.known_loras, self.lora_dir = list(known_loras or []), lora_dir
        self.family_loras = family_loras or {}
        self.loras = LoraStack(self)
        self.loras.file_dir = lora_dir
        self.loras.picked.connect(self._lora_picked)
        self.loras.use_prompt = self._use_example      # (the LoRA window: an example prompt into this one)
        self.loras.changed.connect(self._remember_loras)
        self.loras.changed.connect(lambda: self._fill_presets())
        self.loras.filled.connect(lambda: self._fill_presets())      # (also the LoRAs remembered for this family)
        left.addWidget(self.loras)
        self.lora_each = QCheckBox("One image per LoRA (each alone, in turn)")
        self.lora_each.setToolTip(
            "<p>On: Generate makes one image for each LoRA in the list, with only that LoRA (at its strength) and "
            "its trigger words. The seed, prompt and every other setting stay the same, so the LoRAs can be "
            "compared side by side.<br>With more results (or variations), each one is made with every LoRA in "
            "turn.</p>")
        self.lora_each.setChecked(self.qs is not None and qs_value(self.qs, "loraEach", False, bool))
        self.lora_each.toggled.connect(lambda on: self._remember("loraEach", on))
        self.lora_each.toggled.connect(lambda _on: self._variations_changed())
        self.loras.changed.connect(lambda: self._variations_changed())
        self.loras.filled.connect(lambda: self._variations_changed())
        left.addWidget(self.lora_each)

    def _build_options(self, left: QVBoxLayout):
        opts = QGridLayout()
        self.quality = QComboBox()
        self.quality.setToolTip("Faster settings use the model's speed LoRA or distilled version; Best samples "
                                "longer and can follow the prompt more closely")
        self.quality.currentIndexChanged.connect(lambda *_: self._remember_quality())
        self.quality.currentIndexChanged.connect(lambda *_: self._sampling_defaults())
        self.tries = QSpinBox()
        self.tries.setRange(1, MAX_RESULTS)
        self.tries.setValue(2)
        self.tries.valueChanged.connect(lambda _v: self._variations_changed())
        self.tries.setToolTip(f"How many results at once (1 … {MAX_RESULTS}; each with the next seed), to pick the "
                              "best one. They are made one after another and each appears when it is ready; "
                              "Stop ends it after the current one")
        self.keep_size = QCheckBox("Keep the size of the image to edit")
        self.keep_size.setChecked(True)
        self.keep_size.setToolTip("The model works at about 1 megapixel in its own aspect steps; this scales the "
                                  "result back to the exact size of the image it was made from, so it fits the "
                                  "chain's video size")
        self.tags = QCheckBox("Add the usual quality tags (masterpiece, best quality …)")
        self.tags.setChecked(True)
        self.tags.setToolTip("Illustrious models are trained on such tags; they are put before your prompt")
        opts.addWidget(label_for("Quality", self.quality, level="tertiary"), 0, 0)
        opts.addWidget(self.quality, 0, 1)
        opts.addWidget(label_for("Results", self.tries, level="tertiary"), 0, 2)
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
        opts.addWidget(label_for("Seed", self.seed, level="tertiary"), 1, 0)
        opts.addLayout(srow, 1, 1, 1, 3)
        opts.addWidget(self.keep_size, 2, 0, 1, 4)
        opts.addWidget(self.tags, 3, 0, 1, 4)
        left.addLayout(opts)

    def _build_actions(self, left: QVBoxLayout):
        brow = QHBoxLayout()
        self.b_gen = QPushButton("Generate")
        self.b_gen.setProperty("variant", "primary")
        self.b_gen.setToolTip("Edit the image with this prompt (Ctrl+Enter)")
        self.b_gen.clicked.connect(self.generate)
        self.b_cancel = QPushButton("Stop")
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

    def _scrolled(self, left: QVBoxLayout) -> QScrollArea:
        lw = QWidget()
        lw.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(lw)
        scroll.setFixedWidth(430)
        self.left_scroll = scroll
        return scroll

    def _build_results_panel(self):
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
        self.thumb_size = QSlider(Qt.Orientation.Horizontal)
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
        self.b_before.setToolTip("Switch between the result and the image it was made from (press B)")
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
        self.results.setViewMode(QListWidget.ViewMode.IconMode)
        self.results.setFlow(QListWidget.Flow.LeftToRight)
        self.results.setWrapping(False)
        self.results.setIconSize(QSize(150, 100))
        self.results.setGridSize(QSize(158, 108))
        self.results.setFixedHeight(130)
        self.results.setMovement(QListWidget.Movement.Static)
        self.results.setToolTip("The results shown (newest first); Gallery shows them all, grouped by run")
        self.results.setProperty("role", "resultStrip")      # (the theme styles it: transparent, no frame)
        self.results.currentItemChanged.connect(self._strip_selection)
        self.results.currentItemChanged.connect(lambda *_: self._show_selected())
        sv.addWidget(self.results)
        self.stack.addWidget(single)
        self.gallery = ResultGallery()
        self.gallery.clicked.connect(self._gallery_clicked)
        self.gallery.opened.connect(self._gallery_opened)
        # (later: the star's thumbnail is still handling its click; a rebuild now would delete it under it)
        self.gallery.starred.connect(lambda png: QTimer.singleShot(0, lambda: self.toggle_favorite(png)))
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
        self.b_use_end = QPushButton(f"Use as end image of clip #{self.end_target[1:]}" if self.end_target else
                                     "Use as end image")
        self.b_use_end.setToolTip("The frame this clip steers its end to (the video model's last-frame input)" if self.end_target else
                                  "Open this from a clip (its End row or its menu) to make its end image")
        self.b_use_end.clicked.connect(lambda: self._use("end"))
        self.b_further = QPushButton("Edit this one further")
        self.b_further.setToolTip("Take this result as the image to edit (several small steps often work better "
                                  "than one big change)")
        self.b_further.clicked.connect(self._edit_further)
        self.b_use_ref = QPushButton("Add as face reference")
        self.b_use_ref.setToolTip("Add this result to the project's face pictures (the Cast & references tab, "
                                  "Alt+2), e.g. a face or an outfit made here; the dialog stays open, so you can add "
                                  "several")
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
        close, _none = button_row(self)
        right.addWidget(close)
        return right

    def _add_shortcuts(self):
        shortcut(S.sequence("generate_image"), self, self.generate)
        for key, fn in ((S.sequence("image_delete"), self.delete_selected),
                        (S.sequence("image_before_after"), self.b_before.toggle),
                        (S.sequence("image_favorite"), lambda: self.toggle_favorite())):
            for w in (self.results, self.gallery):
                sc = shortcut(key, w, fn)
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)  # only while the results have the focus
        for b in self.findChildren(QPushButton):              # Enter in a field must not press a button
            b.setAutoDefault(False)
            b.setDefault(False)
        guard_wheel(self)

    def _restore(self, source: str | None, sources: list[tuple[str, str]], mode: str):
        """What the user had last time: sizes, variations, view, the image to edit, the mode; then the first look."""
        if self.qs is not None:
            self.thumb_size.setValue(qs_value(self.qs, "resultsThumb", 200, int))
            try:
                self.variations.set_state(json.loads(qs_value(self.qs, "promptVariations", "{}", str) or "{}"))
            except ValueError:
                pass
        self._load_results()
        self.set_view(qs_value(self.qs, "resultsView", "single", str) if self.qs is not None else "single",
                      remember=False)
        if source:
            i = self.src_combo.findData(source)
            if i >= 0:
                self.src_combo.setCurrentIndex(i)
            self.set_source(source)
        elif sources:
            self.set_source(sources[0][1])
        self.modes.idClicked.connect(self._mode_clicked)
        self.m_create.setEnabled(self.connect_create is not None)
        saved = qs_value(self.qs, "imageMode", "", str) if self.qs is not None else ""
        if mode != "auto" and self.qs is not None:
            self.qs.setValue("imageMode", mode)       # opened from its Edit / Create button: the user's choice
        self.set_mode(mode if mode != "auto" else (saved or "edit"))
        if self.connect_create is not None:
            self._load_families()
        self._update_buttons()

    # ---- buttons
    def _update_generate_button(self):
        """Generate and Paint mask: what the prompt, source and model decide (no disk reads: used while typing)."""
        idle = not self._busy
        text = bool(self.prompt.toPlainText().strip())
        if self.mode() == "create":
            fam = self.family.currentData()
            refs_ok = not self.use_refs.isChecked() or any(s.path for s in self.cpics[:FAMILIES.get(fam,
                                                                                                    {}).get("refs", 0)])
            ok = idle and text and fam in FAMILIES and "models" in (self.families.get(fam) or {}) and refs_ok
            why = ("Choose an available model" if fam not in FAMILIES or "models" not in (self.families.get(fam) or {})
                   else "Add a reference picture" if not refs_ok else "Describe the image")
        else:
            ok = idle and bool(self.source) and text and not self._mask_problem()
            why = self._mask_problem() or "Choose the image to edit and write what should change"
        self.b_gen.setEnabled(ok)
        self.b_mask.setEnabled(bool(self.source) and idle and not self.batch)
        self.b_gen.setToolTip(("Make the image (Ctrl+Enter)" if self.mode() == "create" else
                               "Edit the image with this prompt (Ctrl+Enter)") if ok else
                              why if idle else "Working …")
        self.b_cancel.setEnabled(not idle)

    def _update_buttons(self):
        self._update_generate_button()
        idle = not self._busy
        sel = self.selected()
        has = sel is not None
        for b in (self.b_use_start, self.b_use_ref, self.b_further, self.b_again, self.b_before, self.b_fav):
            b.setEnabled(has)
        self.b_use_end.setEnabled(has and self.end_target is not None)
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
        self._busy = True
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
        mask = self.mask.copy() if self.mask is not None and not self.batch else None
        runs = self._lora_runs()                     # (One image per LoRA: each result with every LoRA in turn)
        sources = list(self.batch) or [source]       # (a folder: each of its images in turn, with all of it)
        base_n, per = n, n * len(runs)
        n = per * len(sources)
        if len(sources) > 1 and n > ASK_ABOVE and not ask(
                self, "Edit a folder", f"This makes {n:,} images: {len(sources)} pictures × {per} each. "
                "They are made one after another (Stop ends it after the current one). Start?"):
            self._busy = False
            return
        run_words = [self._trigger_words(r) for r in runs]
        refs = [p for p in (self.pic2.path, self.pic3.path) if p]
        self._remember_loras()
        keep_size = self.keep_size.isChecked() and mask is None   # (an inpaint comes back at the source's own size)
        self._batch_names = [Path(f).name for f in sources] if len(sources) > 1 else []
        self._per_image = per
        plan = EditPlan(self.project / EDITS_DIR, self.connect_editor, prompt, quality, n, [], runs, run_words, refs,
                        jobs, model=model, sources=sources, per=per, mask=mask, keep_size=keep_size)
        self._start_worker(EditWorker(plan), base_n,
                           "Connecting to ComfyUI … (the first edit loads the model, which takes a while; if a "
                           "clip is being generated, the edit waits for it)")

    def generate_new(self):
        """Create mode: new images from the prompt (and reference pictures) with the chosen model family."""
        if self._busy or not self.b_gen.isEnabled():
            return
        self._busy = True
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
        runs = self._lora_runs()                     # (One image per LoRA: each result with every LoRA in turn)
        base_n, n = n, n * len(runs)
        run_words = [self._trigger_words(r) for r in runs]
        refs = [s.path for s in self.cpics[:FAMILIES[fam]["refs"]] if s.path] if self.use_refs.isChecked() else []
        self._remember_loras()                    # what it was made with is offered next time
        w, h = size_for(fam, self.aspect.currentData() or 1.0)
        assert self.connect_create is not None
        plan = CreatePlan(self.project / EDITS_DIR, self.connect_create, prompt, quality, n, [], runs, run_words,
                          refs, jobs, family=fam, model_file=model, negative=negative, width=w, height=h,
                          strength=self.ref_strength.value(), tags=self.tags.isChecked(), sampling=self.sampling())
        self._start_worker(CreateWorker(plan), base_n,
                           f"Connecting to ComfyUI … (the first image loads {FAMILIES[fam]['label']}, which takes a "
                           "while)")

    def _start_worker(self, worker: GenerateWorker, base_n: int, text: str):
        """Run a Generate: on_start first (a refusal ends it), then the seeds (a refused Generate keeps the seed)."""
        if not self._run_on_start():
            return
        worker.plan.seeds = self._seeds(base_n)
        self._worker, self._skipped = worker, worker.skipped
        worker.progress.connect(self._progress)
        worker.result.connect(self._result)
        worker.finished.connect(self._finished)
        self.bar.setValue(0)
        self.bar.show()
        self.b_start_comfy.hide()
        self.state.setText(text)
        self._update_buttons()
        worker.start()

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
        self.state.setText("Stopping …")
        if self._worker is not None:
            self._worker.cancel()

    def _progress(self, k: int, n: int, i: int, steps: int):
        if k == 0:
            self.state.setText("No Lightning LoRA for this model in ComfyUI: using Standard (20 steps).")
            return
        frac = (k - 1 + (i / steps if steps else 0)) / max(1, n)
        self.bar.setValue(round(1000 * frac))
        names = self._batch_names if self._busy else []
        pic = (f" · picture {(k - 1) // self._per_image + 1} of {len(names)} "
               f"({names[(k - 1) // self._per_image]})" if names and (k - 1) // self._per_image < len(names) else "")
        self.state.setText(f"Result {k} of {n}{pic}: step {i} / {steps}")

    def _result(self, png: str, meta: ResultRecord):
        if isinstance(meta.get("variation"), dict) and meta["variation"].get("step") is not None:
            self.variations.done(int(meta["variation"]["step"]), meta["variation"].get("key"))   # continues after
        self._result_added(png, meta)
        self.b_before.setChecked(False)

    def _finished(self, error: str):
        self._busy = False
        self._worker = None
        self.bar.hide()
        if error == "cancelled":
            self.state.setText("Stopped.")
        elif error:
            down = "not reachable" in error
            self.state.setText(("ComfyUI is not running. " if down else "") + f"Could not edit the image: {error}")
            self.b_start_comfy.setVisible(down and self.start_comfy is not None)
        else:
            self.state.setText("Done. Pick a result below, compare it with the original (B), then use it.")
        if self._skipped:
            sk = self._skipped
            self.state.setText(self.state.text() + f" Skipped (not readable as an image): {', '.join(sk[:5])}"
                               + (f" and {len(sk) - 5} more" if len(sk) > 5 else "") + ".")
        self._batch_names = []
        self._update_buttons()

    def _remember(self, key: str, value: object, needs_value: bool = False) -> None:
        """Keep a choice in the settings (when there are settings; needs_value: and the choice is not empty)."""
        if self.qs is not None and not (needs_value and not value):
            self.qs.setValue(key, value)

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
                self.src_combo.insertItem(self.src_combo.findData(None), f"Result: {sel[1].get('prompt', '')[:40]}",
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
            if i >= 0 and combo_item(self.family, i).isEnabled():
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
            if i >= 0 and combo_item(self.edit_model, i).isEnabled():
                self.edit_model.setCurrentIndex(i)
            q = self.quality.findData(meta.get("quality"))
            if q >= 0:
                self.quality.setCurrentIndex(q)
            self.set_mask(None)                   # (a result made without a mask: none)
            m = meta.get("mask")                  # an inpaint of the image shown: its mask again
            if m and Path(m).is_file() and meta.get("source") == self.source:
                img = QImage(m).convertToFormat(QImage.Format.Format_Grayscale8)
                if not img.isNull() and img.size() == upright(self.source).size():
                    self.set_mask(img)
        if meta.get("template"):                  # made from variations: the prompt with its placeholders
            self.prompt.setPlainText(meta["template"])
            self.variations.on.setChecked(True)
        else:
            self.prompt.setPlainText(meta.get("prompt", ""))
        self.loras.set_items(self._known(), meta.get("loras") or [])
        self._remember_loras()

    def reject(self):
        """Close, Esc: a running job is only stopped after asking (the rest of its images would be lost)."""
        if self._busy and not ask(
                self, "Edit an image", "Images are being made. Stop and close?"):
            return
        super().reject()

    def done(self, r):
        busy = self._busy
        if busy:
            self.cancel()
        self._empty_trash(keep_sources=busy)       # a running job may still write its result with its source
        self._loader.stop()
        self.closed = True
        super().done(r)
