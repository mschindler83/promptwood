"""Edit an image by a prompt (Qwen Image Edit in ComfyUI) and use the result as the start image or as a clip's end
image. Every result is kept in the project (edits/<name>.png + .json with its prompt, the image it was made from in
edits/sources/), so earlier tries stay at hand.

1. the image to edit (start image, a clip's frame, an end image, a face reference, any file, an earlier result)
2. what should change (the prompt), optionally up to two more pictures (Picture 2 / 3, e.g. a face)
3. Generate: one or more tries (other seeds); 4. pick a result, compare it with the original, use it.
"""
from __future__ import annotations

import json
import random
import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFrame, QGridLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from .image_picker import pick_image
from ..imagegen import FAMILIES, size_for
from .lora_stack import LoraStack, lora_label
from .widgets import FrameTile, guard_wheel, icon_button, label

EDITS_DIR = "edits"
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
    return sorted(out, key=lambda x: x[0].stat().st_mtime, reverse=True)


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
                 start_aspect: float | None = None, mode: str = "edit", check_lora=None):
        super().__init__(parent)
        self.setWindowTitle("Edit an image with AI (Qwen Image Edit)")
        self.resize(1240, 820)
        self.project, self.connect_editor, self.on_use = Path(project), connect, on_use
        self.end_target, self.qs, self.places, self.start_comfy = end_target, settings, places or [], start_comfy
        self.on_start = on_start                         # called (GUI thread) before each Generate
        self.uses = uses or (lambda _meta: [])           # uses(meta) -> where a result is used (words)
        self.refine = refine                             # refine(text, kind) -> new text or None
        self.connect_create = connect_create             # connect_create(progress) -> a ComfyImageCreator
        self.start_aspect = start_aspect
        self.check_lora = check_lora                     # check_lora(path) -> False if ComfyUI cannot load it
        self.families: dict = {}                         # what ComfyUI has of each family (from the creator)
        self._families_error = ""
        self._retry_families = 0                         # retries left while ComfyUI starts
        self._trash: Path | None = None                  # the result deleted last (Undo delete brings it back)
        self._icons: dict[tuple, QIcon] = {}             # thumbnails of the results by (path, date)
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
        left.addWidget(self.loras)

        opts = QGridLayout()
        self.quality = QComboBox()
        self.quality.setToolTip("Faster settings use the model's speed LoRA or distilled version; Best samples "
                                "longer and can follow the prompt more closely")
        self.quality.currentIndexChanged.connect(lambda *_: self._remember_quality())
        self.tries = QSpinBox()
        self.tries.setRange(1, 4)
        self.tries.setValue(2)
        self.tries.setToolTip("How many results at once (each with another seed), to pick the best one")
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
        opts.addWidget(self.keep_size, 1, 0, 1, 4)
        opts.addWidget(self.tags, 2, 0, 1, 4)
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
        head.addStretch(1)
        self.b_before = QPushButton("Show the original")
        self.b_before.setCheckable(True)
        self.b_before.setToolTip("Switch between the result and the image it was made from (or hold B)")
        self.b_before.toggled.connect(lambda _on: self._show_selected())
        head.addWidget(self.b_before)
        right.addLayout(head)
        self.view = FrameTile(780, 440)
        self.view.clear_frame("Results appear here")
        right.addWidget(self.view, 1, Qt.AlignHCenter)
        self.res_info = label("", level="secondary")
        self.res_info.setWordWrap(True)
        right.addWidget(self.res_info)
        self.results = QListWidget()
        self.results.setViewMode(QListWidget.IconMode)
        self.results.setFlow(QListWidget.LeftToRight)
        self.results.setWrapping(False)
        self.results.setIconSize(QSize(150, 100))
        self.results.setGridSize(QSize(162, 112))
        self.results.setFixedHeight(130)
        self.results.setMovement(QListWidget.Static)
        self.results.setToolTip("All edits kept in this project (newest first)")
        self.results.currentItemChanged.connect(lambda *_: self._show_selected())
        right.addWidget(self.results)
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
        self.b_use_ref.setToolTip("Add this result to the project's face references (left panel), e.g. a face or an "
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
        for key, fn in (("Delete", self.delete_selected), ("B", self.b_before.toggle)):
            sc = QShortcut(QKeySequence(key), self.results, activated=fn)
            sc.setContext(Qt.WidgetWithChildrenShortcut)     # only while the results list has the focus
        for b in self.findChildren(QPushButton):              # Enter in a field must not press a button
            b.setAutoDefault(False)
            b.setDefault(False)
        guard_wheel(self)
        self._load_results()
        if source:
            i = self.src_combo.findData(source)
            if i >= 0:
                self.src_combo.setCurrentIndex(i)
            self.set_source(source)
        elif sources:
            self.set_source(sources[0][1])
        self.modes.idClicked.connect(lambda i: self.set_mode("create" if i == 1 else "edit"))
        self.m_create.setEnabled(connect_create is not None)
        saved = self.qs.value("imageMode", "", type=str) if self.qs is not None else ""
        self.set_mode(mode if mode != "auto" else (saved or "edit"))
        if connect_create is not None:
            self._load_families()
        self._update_buttons()

    # ---- modes
    def mode(self) -> str:
        return "create" if self.m_create.isChecked() else "edit"

    def set_mode(self, mode: str):
        create = mode == "create" and self.connect_create is not None
        (self.m_create if create else self.m_edit).setChecked(True)
        if self.qs is not None:
            self.qs.setValue("imageMode", "create" if create else "edit")
        self.setWindowTitle("Create a new image with AI" if create else "Edit an image with AI (Qwen Image Edit)")
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
            for key, text in (FAMILIES[fam]["qualities"].items() if fam in FAMILIES else []):
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
        return f"createQuality_{self.family.currentData()}" if self.mode() == "create" else "editQuality"

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
        self._family_changed()

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
        for k, slot in enumerate(self.cpics):
            slot.setVisible(refs and k < n)
        self.ref_text.setVisible(refs)
        self.ref_text.setText(f["ref_text"] if f else "")
        img2img = refs and fam in ("flux1", "sdxl")
        self.ref_strength.setVisible(img2img)
        self.ref_strength_label.setVisible(img2img)
        self.negative.setVisible(create and bool(f and f["negative"]))
        self.tags.setVisible(create and fam == "sdxl")
        if f:
            w, h = size_for(fam, self.aspect.currentData() or 1.0)
            self.size_text.setText(f"{w}×{h}")
        if create:
            found = len(self.family_loras.get(fam, [])) if f else 0
            self.lora_text.setText(f"LoRAs (optional): {f['label']} LoRAs from ComfyUI's LoRA folders (found by "
                                   f"their contents; {found} found), applied in this order." if f else "")
        else:
            self.lora_text.setText("LoRAs (optional): Qwen Image / Image Edit LoRAs from ComfyUI's LoRA folders, "
                                   "applied in this order. Text-to-image Qwen LoRAs often work for edits too, not "
                                   "always.")
        self.b_gen.setToolTip("Make the image (Ctrl+Enter)" if create else
                              "Edit the image with this prompt (Ctrl+Enter)")
        self._update_buttons()

    # ---- source and pictures
    def _pick(self, title: str) -> str | None:
        return pick_image(self, title, self.qs, None, self.places)

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
        img = QImage(path)
        if img.isNull():
            self.state.setText(f"Not an image: {Path(path).name}")
            return
        self.source = str(path)
        self.src_tile.set_image(img)
        self.src_tile.setToolTip(str(path))
        i = self.src_combo.findData(str(path))
        self.src_info.setText(f"{self.src_combo.itemText(i) + ' · ' if i >= 0 else ''}{Path(path).name} · "
                              f"{img.width()}×{img.height()}")
        self._update_buttons()

    # ---- results
    def _load_results(self, select: str | None = None):
        self.results.blockSignals(True)
        self.results.clear()
        for png, meta in project_edits(self.project):
            try:
                key = (str(png), png.stat().st_mtime_ns)
            except OSError:
                continue
            if key not in self._icons:                 # each result is decoded once, not on every update
                self._icons[key] = QIcon(QPixmap.fromImage(QImage(str(png)).scaled(
                    300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            it = QListWidgetItem(self._icons[key], "")
            it.setData(Qt.UserRole, (str(png), meta))
            it.setToolTip(meta.get("prompt", png.name))
            self.results.addItem(it)
        self.results.blockSignals(False)
        if self.results.count():
            k = next((i for i in range(self.results.count())
                      if self.results.item(i).data(Qt.UserRole)[0] == select), 0)
            self.results.setCurrentRow(k)
        self._show_selected()

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
            self.res_info.setText("")
        else:
            png, meta = sel
            orig = meta.get("source_copy") or meta.get("source")
            show = orig if self.b_before.isChecked() and orig and Path(orig).is_file() else png
            self.view.set_image(QImage(show))
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(Path(png).stat().st_mtime))
            used = " · ".join(f"{lora_label(lo['path'])} ×{lo['strength']:g}" for lo in meta.get("loras") or [])
            where = self.uses(meta)
            how = (f"{FAMILIES.get(meta.get('family'), {}).get('label', '')} "
                   f"{FAMILIES.get(meta.get('family'), {}).get('qualities', {}).get(meta.get('quality'), '')}"
                   if meta.get("mode") == "create" else QUALITY_TEXT.get(meta.get("quality"), ""))
            self.res_info.setText(("ORIGINAL · " if show != png else "") +
                                  (f"IN USE: {', '.join(where)} · " if where else "") +
                                  f"“{meta.get('prompt', '')}” · {how} · "
                                  + (f"{used} · " if used else "") + f"seed {meta.get('seed', '?')} · {when}")
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
            ok = idle and bool(self.source) and text
            why = "Choose the image to edit and write what should change"
        self.b_gen.setEnabled(ok)
        if not ok:
            self.b_gen.setToolTip(why if idle else "Working …")
        self.b_cancel.setEnabled(not idle)
        for b in (self.b_use_start, self.b_use_ref, self.b_further, self.b_again, self.b_before):
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
        refs = [p for p in (self.pic2.path, self.pic3.path) if p]
        loras = [(it["path"], it["strength"]) for it in self.loras.items()]
        self._remember_loras()
        size = None
        if self.keep_size.isChecked():
            img = QImage(source)
            size = (img.width() // 2 * 2, img.height() // 2 * 2)
        edits = self.project / EDITS_DIR
        sig, connect = self._sig, self.connect_editor
        if not self._run_on_start():
            return
        self.bar.setValue(0)
        self.bar.show()
        self.b_start_comfy.hide()
        self.state.setText("Connecting to ComfyUI … (the first edit loads the model, which takes a while; if a "
                           "clip is being generated, the edit waits for it)")
        self._update_buttons()

        def work():
            try:
                edits.mkdir(parents=True, exist_ok=True)
                cur = {"k": 1}
                self.editor = connect(lambda i, s: sig.progress.emit(cur["k"], n, i, s))
                if quality == "fast" and "fast" not in self.editor.qualities():
                    sig.progress.emit(0, n, 0, 0)
                kept = edits / "sources" / f"source_{time.strftime('%Y%m%d-%H%M%S')}{Path(source).suffix.lower() or '.png'}"
                kept.parent.mkdir(exist_ok=True)
                kept.write_bytes(Path(source).read_bytes())         # the original stays with its results
                for k in range(1, n + 1):
                    if self._cancel:
                        break
                    cur["k"] = k
                    seed = random.randint(1, 2**31 - 1)
                    stem = f"edit_{time.strftime('%Y%m%d-%H%M%S')}_{seed}"
                    out = edits / f"{stem}.png"
                    meta = self.editor.edit(Path(source), prompt, out, seed, quality, [Path(r) for r in refs], loras)
                    if size is not None:
                        scale_to(out, out, size)
                    meta.update({"prompt": prompt, "seed": seed, "source": source, "source_copy": str(kept),
                                 "refs": refs, "size": list(size) if size else None})
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

    def generate_new(self):
        """Create mode: new images from the prompt (and reference pictures) with the chosen model family."""
        if self._busy or not self.b_gen.isEnabled():
            return
        self._busy, self._cancel = True, False
        fam, model = self.family.currentData(), self.model_file.currentData()
        prompt = " ".join(self.prompt.toPlainText().split())
        negative = " ".join(self.negative.text().split()) if self.negative.isVisible() else ""
        quality, n = self.quality.currentData(), self.tries.value()
        refs = [s.path for s in self.cpics[:FAMILIES[fam]["refs"]] if s.path] if self.use_refs.isChecked() else []
        loras = [(it["path"], it["strength"]) for it in self.loras.items()]
        self._remember_loras()                    # what it was made with is offered next time
        w, h = size_for(fam, self.aspect.currentData() or 1.0)
        strength, tags = self.ref_strength.value(), self.tags.isChecked()
        edits = self.project / EDITS_DIR
        sig, connect = self._sig, self.connect_create
        if not self._run_on_start():
            return
        self.bar.setValue(0)
        self.bar.show()
        self.b_start_comfy.hide()
        self.state.setText(f"Connecting to ComfyUI … (the first image loads {FAMILIES[fam]['label']}, which takes a "
                           "while)")
        self._update_buttons()

        def work():
            try:
                edits.mkdir(parents=True, exist_ok=True)
                cur = {"k": 1}
                self.editor = connect(lambda i, st: sig.progress.emit(cur["k"], n, i, st))
                kept = None
                if refs:                              # the first picture is shown as the "original"
                    kept = edits / "sources" / f"source_{time.strftime('%Y%m%d-%H%M%S')}{Path(refs[0]).suffix.lower() or '.png'}"
                    kept.parent.mkdir(exist_ok=True)
                    kept.write_bytes(Path(refs[0]).read_bytes())
                for k in range(1, n + 1):
                    if self._cancel:
                        break
                    cur["k"] = k
                    seed = random.randint(1, 2**31 - 1)
                    out = edits / f"new_{time.strftime('%Y%m%d-%H%M%S')}_{seed}.png"
                    meta = self.editor.create(fam, model, prompt, out, seed, quality, w, h, negative, loras,
                                              [Path(r) for r in refs], strength, tags)
                    meta.update({"prompt": prompt, "seed": seed, "refs": refs, "tags": tags,
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
        new = self.refine(self.prompt.toPlainText(), self.prompt_kind())
        if new:
            from .prompt_refiner import replace_text
            replace_text(self.prompt, new)

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
            j = self.model_file.findData(meta.get("model"))
            if j >= 0:
                self.model_file.setCurrentIndex(j)
            q = self.quality.findData(meta.get("quality"))
            if q >= 0:
                self.quality.setCurrentIndex(q)
            self.negative.setText(meta.get("negative") or "")
            if "tags" in meta:
                self.tags.setChecked(bool(meta["tags"]))
        elif meta.get("mode") != "create" and self.mode() != "edit":
            self.set_mode("edit")
        self.prompt.setPlainText(meta.get("prompt", ""))
        self.loras.set_items(self._known(), meta.get("loras") or [])
        self._remember_loras()

    # ---- LoRAs
    def _lora_key(self) -> str:
        return f"createLoras_{self.family.currentData()}" if self.mode() == "create" else "editLoras"

    def _known(self) -> list[tuple[str, str]]:
        """The LoRAs ComfyUI has for this mode (edit: Qwen; create: the chosen family), then LoRA files chosen here
        earlier (from other folders)."""
        if self.mode() == "create":
            fam = self.family.currentData()
            out = list(self.family_loras.get("qwen" if fam == "qwen" else fam, []))
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
        listed = {p for p, _ in (self.family_loras.get(self.family.currentData(), []) if self.mode() == "create"
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
        self.closed = True
        super().done(r)
