"""Replace a person's head in a real video with another person (Wan 2.2 Animate in ComfyUI, see swap.py).

The clip with the head that is drawn anew (found by the face detector, followed through the clip): check it with
the frame slider, make it larger or smaller; choose the new person's picture; Generate. The result is the clip with
the new head, at its own size, frames and sound; "Use this" makes the video node play it (it can go back)."""
from __future__ import annotations

import random
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLineEdit,
                               QProgressBar, QPushButton, QVBoxLayout)

from .. import chain, media, swap
from .. import tree as T
from ..animate import DEFAULT_PROMPT, SIZES, gen_size, window_count
from .compare_player import ComparePlayer
from .dialogs import confirm
from .theme import theme
from .widgets import fit_size, FrameTile, guard_wheel, label, safe_emit
from .qsvalue import qs_value

METHODS = (("animate+lock", "Head + face lock (Animate, then face swap)"), ("animate", "Head only (Animate)"),
           ("face", "Face only (face swap, the clip's own motion)"),
           ("h3", "Head swap with H3 (head swap LoRA, the clip as guide)"))


def letterbox(src: str, dest, size: int, crop: tuple[int, int, int] | None = None) -> bool:
    """The picture (cut to the square crop (x, y, side) first, when given) fitted into a size x size square, centred
    on black (the head swap LoRA's reference format)."""
    img = QImage(src)
    if img.isNull():
        return False
    if crop is not None:
        x, y, side = crop
        img = img.copy(int(x), int(y), int(side), int(side))
    fit = img.convertToFormat(QImage.Format.Format_RGB32).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    out = QImage(size, size, QImage.Format.Format_RGB32)
    out.fill(QColor(0, 0, 0))
    p = QPainter(out)
    p.drawImage((size - fit.width()) // 2, (size - fit.height()) // 2, fit)
    p.end()
    return out.save(str(dest), "PNG")  # type: ignore[call-overload]  # (stubs: bytes, runtime: str)


class _Sig(QObject):
    found = Signal(object, str)             # find_heads result (or None), error
    progress = Signal(str, int, int)        # what, done, of
    finished = Signal(object, str)          # (out path, meta) or None, error
    details = Signal(str)                   # the face details written by the vision model
    reference = Signal(int, str, object, object)   # picture generation, path, upright PNG (None: unreadable), crop


class SwapDialog(QDialog):
    """clip: the converted clip of the video node; pictures: [(label, path)] offered as the new person;
    connect(progress) -> ComfyAnimator (worker thread); on_start(): before Generate (GUI thread);
    .result: (path, meta, frames) of the clip to use; it is played and compared with the clip in the window."""

    def __init__(self, parent, project: Path, nid: str, clip: Path, detector: Path | None,
                 pictures: list[tuple[str, str]], connect: Callable, settings: QSettings | None = None,
                 on_start: Callable[[], None] | None = None,
                 source: dict | None = None, pick: Callable[[str], str | None] | None = None, groups: list[list[str]] | None = None,
                 h3: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Replace a person in video #{T.short_id(nid)}")
        fit_size(self, 1180, 720)
        self.project, self.nid, self.clip, self.detector = Path(project), nid, Path(clip), detector
        self.connect, self.qs, self.on_start = connect, settings, on_start
        self.source = dict(source or {})          # the video's file and range (the DWPose cache is bound to it)
        self.pick = pick                          # pick(title) -> an image path or None (the app's image picker)
        self.groups = [list(g) for g in groups or []]   # the pictures of each Cast character (the face pass uses all)
        # H3 head swap: {"lora": the head swap LoRA's path or None, "settings": the project's clip settings,
        # "connect": progress -> a ComfyGenerator}
        self.h3 = dict(h3 or {})
        self.heads: dict | None = None
        self.boxes: list[tuple] = []
        self.result: tuple[Path, dict, int] | None = None
        self.animator = None
        self._busy = False
        self._cancel = False
        self._tmp = media.scratch("swap")
        self._sig = _Sig()
        self._sig.found.connect(self._found)
        self._sig.progress.connect(self._progress)
        self._sig.finished.connect(self._finished)
        self._sig.details.connect(self._auto_details_made)
        self._sig.reference.connect(self._reference_ready)
        self._ref_gen = 0                              # which picture the worker is preparing (older answers drop)
        self.ref_busy = False
        self._closed = False
        self._auto_details = ""                        # the face details the vision model wrote (for this picture)

        top = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(label("The head that is drawn anew", type_="heading"))
        self.player = ComparePlayer(self)               # the clip; with a result: before | after, frame by frame
        self.player.overlay = self._draw_head
        self.player.view.show_text("Looking for the head …")
        self.slider = self.player.slider
        left.addWidget(self.player, 1)
        hrow = QHBoxLayout()
        self.show_head = QCheckBox("Show the head area")
        self.show_head.setChecked(True)
        self.show_head.setToolTip("The head area that is drawn anew, on the original (left of the divider)")
        self.show_head.toggled.connect(lambda *_: self.player.refresh())
        hrow.addWidget(self.show_head)
        hrow.addSpacing(12)
        self.head_size = QDoubleSpinBox()
        self.head_size.setRange(0.7, 2.0)
        self.head_size.setSingleStep(0.05)
        self.head_size.setValue(float(qs_value(settings, "swapHeadSize", 1.0, float)) if settings else 1.0)
        self.head_size.setToolTip("The head area around the found face: larger takes in more hair, the ears and "
                                  "the neck; everything outside it stays as it is")
        self.head_size.valueChanged.connect(self._resize_heads)
        guard_wheel(self.head_size)
        hrow.addWidget(label("Head size", level="tertiary"))
        hrow.addWidget(self.head_size)
        hrow.addStretch(1)
        left.addLayout(hrow)
        self.found_text = label("", level="tertiary")
        self.found_text.setWordWrap(True)
        left.addWidget(self.found_text)
        left.addStretch(1)
        top.addLayout(left, 3)

        right = QVBoxLayout()
        right.addWidget(label("The new person", type_="heading"))
        self.ref_tile = FrameTile(300, 300)
        self.ref_tile.clear_frame("Choose a picture of the person")
        right.addWidget(self.ref_tile)
        self.ref_combo = QComboBox()
        self.ref_combo.addItem("Choose a picture …", None)
        for text, path in pictures:
            self.ref_combo.addItem(text, path)
        self.ref_combo.addItem("Other picture (project or file)…", "")
        self.ref_combo.activated.connect(self._ref_picked)
        right.addWidget(self.ref_combo)
        crow = QHBoxLayout()
        self.head_crop = QCheckBox("Crop to the head")
        self.head_crop.setChecked(True)
        self.head_crop.setToolTip("H3 head swap: the picture is reduced to 256 x 256 px; cut to a square around the "
                                  "head (with hair and chin), the face keeps its detail. Off: the whole picture "
                                  "(black bars where it is not square)")
        self.head_crop.toggled.connect(lambda _on: self._show_reference())
        self.b_crop = QPushButton("Adjust crop…")
        self.b_crop.setToolTip("Move or resize the square around the head")
        self.b_crop.clicked.connect(self.adjust_crop)
        crow.addWidget(self.head_crop)
        crow.addWidget(self.b_crop)
        crow.addStretch(1)
        right.addLayout(crow)
        self.crop_info = label("", level="tertiary")
        self.crop_info.setWordWrap(True)
        right.addWidget(self.crop_info)
        right.addWidget(label("One picture: the head and shoulders, the face well lit and seen from the front work "
                              "best.", level="tertiary"))
        form = QFormLayout()
        self.method = QComboBox()
        for key, text in METHODS:
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(0, self.method.findData(qs_value(settings, "swapMethod", "animate+lock", str)
                                                                if settings else "animate+lock")))
        self.method.setToolTip(
            "<p><b>Head + face lock</b>: Wan 2.2 Animate draws the new head (face, hair, head shape), then a face "
            "swap (HyperSwap) on every frame makes the face the new person's.<br><b>Head only</b>: Wan 2.2 Animate "
            "alone.<br><b>Face only</b>: no Animate: only the face is swapped on every frame, so all motion is the "
            "clip's own (fast, robust in any pose); hair and head shape stay.<br><b>Head swap with H3</b>: H3's "
            "reference model with the head swap LoRA re-renders the clip with the new head; the whole clip guides "
            "it (motion, camera, light come from the clip itself, no pose estimation); up to 15 s.</p>")
        self.method.currentIndexChanged.connect(lambda *_: (self._forget_auto_details(), self._update()))
        form.addRow("Method", self.method)
        self.body = QCheckBox("Follow the body pose")
        self.body.setChecked(qs_value(settings, "swapBody", True, bool) if settings else True)
        self.body.setToolTip("On: the body skeleton (DWPose) drives Animate too. Off: only the face motion; the "
                             "body stays as in the clip anyway (only the head is drawn anew): better when the body "
                             "is in unusual poses DWPose gets wrong")
        form.addRow("", self.body)
        self.sharpen = QCheckBox("Sharpen the face (GPEN)")
        self.sharpen.setChecked(qs_value(settings, "swapSharpen", True, bool) if settings else True)
        self.sharpen.setToolTip("Face restoration after the face swap (GPEN-BFR-512, 80 %): a sharper face")
        form.addRow("", self.sharpen)
        self.size = QComboBox()
        for key in SIZES:
            self.size.addItem({"480p": "480p (faster)", "720p": "720p (sharper, slower)"}.get(key, key), key)
        self.size.setCurrentIndex(max(0, self.size.findData(qs_value(settings, "swapSize", "480p", str)
                                                            if settings else "480p")))
        self.size.setToolTip("The size the new head is made at (only the head area: the rest of the clip keeps "
                             "its own size and sharpness)")
        form.addRow("Made at", self.size)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText(DEFAULT_PROMPT)
        self.prompt.setToolTip("What happens in the clip (optional); the motion comes from the clip itself")
        form.addRow("Prompt", self.prompt)
        right.addLayout(form)
        self.bar = QProgressBar()
        self.bar.hide()
        right.addWidget(self.bar)
        self.state = label("", level="secondary")
        self.state.setWordWrap(True)
        right.addWidget(self.state)
        brow = QHBoxLayout()
        self.b_gen = QPushButton("Generate")
        self.b_gen.clicked.connect(self.generate)
        self.b_stop = QPushButton("Stop")
        self.b_stop.clicked.connect(self.stop)
        self.b_play = QPushButton("Play result")
        self.b_play.setToolTip("Play the result here, the original left of the divider (drag it over the "
                               "picture); step frame by frame with the buttons under it")
        self.b_play.clicked.connect(self.play_result)
        self.b_use = QPushButton("Use this")
        self.b_use.setToolTip("The video plays this clip from now on (its menu can go back to the original)")
        self.b_use.clicked.connect(self.accept)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.reject)
        for b in (self.b_gen, self.b_stop, self.b_play, self.b_use):
            brow.addWidget(b)
        brow.addStretch(1)
        brow.addWidget(b_close)
        right.addLayout(brow)
        right.addStretch(1)
        top.addLayout(right, 2)
        self.ref: str | None = None
        self.ref_png: Path | None = None              # the picture upright as PNG (what is shown and cropped)
        self.ref_crop: tuple[int, int, int] | None = None   # the head square (x, y, side) in it
        self._update()
        self._find()

    # ---- the head
    def _no_heads(self, why: str):
        """No head boxes (no detector, no face, stopped): the clip itself still goes to Face only and H3 (which find
        the face themselves or need none); the Animate methods need the head."""
        try:
            w, h, n, fps = swap.probe(self.clip)
        except Exception as e:  # noqa: BLE001
            self.found_text.setText(f"Cannot read the clip: {e}")
            return
        self.heads = {"boxes": [], "found": 0, "frames": n, "size": (w, h), "fps": fps, "track": []}
        self.boxes = []
        self.player.load(self.clip, n, (w, h))
        self.found_text.setText(why + " Face only and the H3 head swap work without it; the Animate methods need "
                                "the head.")
        self._update()

    def _find(self):
        if self.detector is None:
            self._no_heads("No face detector found (insightface's det_10g.onnx in ComfyUI's "
                           "models/insightface/models/buffalo_l): the head is not marked.")
            return
        sig, clip, det, scale = self._sig, self.clip, self.detector, self.head_size.value()

        def work():
            try:
                h = swap.find_heads(clip, det, scale, progress=lambda k, n: sig.progress.emit("find", k, n),
                                    cancelled=lambda: self._cancel)
                sig.found.emit(h, "")
            except Exception as e:  # noqa: BLE001
                safe_emit(sig.found, None, str(e))
        self._busy = True
        self.bar.show()
        threading.Thread(target=work, daemon=True).start()

    def _found(self, heads, error: str):
        self._busy = False
        self.bar.hide()
        if heads is None:
            self._cancel = False
            self._no_heads("The head was not marked (stopped)." if error == "cancelled" else
                           f"Could not find the head: {error}.")
            return
        self.heads = heads
        self.boxes = heads["boxes"]
        n = heads["frames"]
        self.player.load(self.clip, n, heads["size"])
        missing = n - heads["found"]
        self.found_text.setText(
            f"The face was found in {heads['found']} of {n} frames"
            + (f"; in the other {missing} the head is placed between the frames around them" if missing else "")
            + f". The clip is made in {window_count(n)} part{'s' if window_count(n) != 1 else ''} of up to 77 "
              "frames.")
        self._update()

    def _resize_heads(self, *_):
        if self.qs is not None:
            self.qs.setValue("swapHeadSize", self.head_size.value())
        if self.heads is not None:
            w, h = self.heads["size"]
            self.boxes = swap.head_boxes(self.heads["track"], w, h, self.head_size.value())
            self.player.refresh()                         # the red head area is drawn at the new size
    
    def _draw_head(self, k: int, img: QImage):
        """The head area of frame k onto the original's picture (stored size)."""
        if not self.show_head.isChecked() or not 0 <= k < len(self.boxes):
            return
        x1, y1, x2, y2 = (v * self.player.scale() for v in self.boxes[k])
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(theme().negative, max(2, img.width() // 300)))
        p.setBrush(QColor(255, 40, 40, 70))
        p.drawEllipse(QRectF(x1, y1, x2 - x1, y2 - y1))
        p.end()

    def play_result(self):
        if self.result is not None:
            self.player.play(from_start=True)

    # ---- the new person
    def _ref_picked(self, i: int):
        path = self.ref_combo.itemData(i)
        if path == "":
            if self.pick is not None:
                path = self.pick("Picture of the new person")
            else:
                from .image_picker import pick_image
                path = pick_image(self, "Picture of the new person", self.qs)
            if not path:                                  # cancelled: the picture before stays shown
                k = self.ref_combo.findData(self.ref) if self.ref else 0
                self.ref_combo.setCurrentIndex(max(0, k))
                return
        self.set_reference(path)

    def _auto_details_made(self, text: str):
        self._auto_details = text
        self.prompt.setText(text)

    def _forget_auto_details(self):
        """Another picture (or method): the written face details belonged to the one before."""
        if self._auto_details and self.prompt.text().strip() == self._auto_details:
            self.prompt.clear()
        self._auto_details = ""

    def set_reference(self, path: str | None):
        """The new person's picture (turned upright as a PNG; the head found for the crop). The picture is read and
        searched in a worker: the window stays usable and the tile says so; Generate waits for it."""
        if str(path or "") != str(self.ref or ""):
            self._forget_auto_details()
        self.ref, self.ref_png, self.ref_crop = None, None, None
        self._ref_gen += 1
        self.ref_busy = bool(path)
        if not path:
            self._show_reference()
            self._update()
            return
        self.ref_tile.clear_frame("Looking at the picture …")
        self.crop_info.setText("Looking for the face …")
        png = Path(self._tmp.name) / f"ref_{uuid.uuid4().hex[:6]}.png"
        gen, sig = self._ref_gen, self._sig

        def work():
            crop = None
            try:
                try:
                    chain.to_png(Path(path), png)        # upright (EXIF), as shown and as cropped
                    upright_ok = True
                except Exception:  # noqa: BLE001
                    use, upright_ok = Path(path), False
                else:
                    use = png
                ok = not QImage(str(use)).isNull()
                crop = self._find_head(use) if ok and upright_ok else None
                out = use if ok else None
            except Exception:  # noqa: BLE001  (the temporary folder removed by a close, ...)
                out = None
            safe_emit(sig.reference, gen, str(path), out, crop)
        threading.Thread(target=work, daemon=True).start()
        self._update()

    def _reference_ready(self, gen: int, path: str, png, crop):
        if gen != self._ref_gen or self._closed:         # another picture was chosen, or the window closed
            return
        self.ref_busy = False
        if png is not None:
            self.ref, self.ref_png, self.ref_crop = path, Path(png), crop
        self._show_reference()
        self._update()

    def _find_head(self, png: Path):
        """The square (x, y, side) around the head in the picture, or None (no face found / no detector)."""
        from .. import faces as F
        if self.detector is None or F.available(self.detector) is not None:
            return None
        try:
            found = F.detect(png, self.detector)
        except Exception:  # noqa: BLE001
            return None
        if not found:
            return None
        w, h = F.image_size(png)
        return F.face_square(found[0], w, h, swap.HEAD_CROP_MARGIN)

    def head_picture(self) -> str | None:
        """The picture the H3 head swap gets: the upright picture, cut to the head square when cropping is on."""
        return str(self.ref_png) if self.ref_png else None

    def adjust_crop(self):
        if not self.ref_png:
            return
        from .face_crop import FaceCropDialog
        try:
            dlg = FaceCropDialog(self, self.ref_png, self.detector, "Crop to the head (with hair and chin)",
                                 self.ref_crop)
        except ValueError:
            return
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self.ref_crop = dlg.crop()
                self.head_crop.setChecked(self.ref_crop is not None)
                self._show_reference()
        finally:
            dlg.deleteLater()

    def _show_reference(self):
        if not self.ref_png:
            self.ref_tile.clear_frame("Choose a picture of the person")
            self.crop_info.setText("")
            return
        img = QImage(str(self.ref_png)).convertToFormat(QImage.Format.Format_ARGB32)
        crop = self.ref_crop if self.head_crop.isChecked() else None
        if crop is not None:
            x, y, side = crop
            p = QPainter(img)
            p.setPen(QPen(theme().negative, max(2, img.width() // 200)))
            p.drawRect(QRectF(x, y, side, side))
            p.end()
        self.ref_tile.set_image(img)
        if self.head_crop.isChecked() and crop is None:
            self.crop_info.setText("No face found in the picture: the whole picture is used (Adjust crop… to set "
                                   "the square by hand).")
        elif crop is not None:
            self.crop_info.setText(f"The head square ({crop[2]} px) goes to the H3 head swap at 256 px"
                                   + (" (small: a larger picture of the face gives more detail)."
                                      if crop[2] < 256 else "."))
        else:
            self.crop_info.setText("The whole picture (black bars where it is not square).")

    # ---- generating
    def _update(self):
        ready = self.heads is not None and bool(self.ref) and not self._busy and not self.ref_busy
        self.b_gen.setEnabled(ready)
        self.b_gen.setToolTip("" if ready else "Wait for the head to be found and choose the new person's picture")
        self.b_stop.setEnabled(self._busy)
        self.b_play.setEnabled(self.result is not None and not self._busy)
        self.b_use.setEnabled(self.result is not None and not self._busy)
        m = self.method.currentData()
        animate = m in ("animate", "animate+lock")
        idle = not self._busy
        # one rule per control: busy and the method decide (nothing is enabled and then switched off again)
        for w, on in ((self.ref_combo, True), (self.method, True), (self.head_size, animate), (self.size, animate),
                      (self.body, animate), (self.prompt, m != "face"),
                      (self.sharpen, m in ("face", "animate+lock")),
                      (self.head_crop, bool(self.ref_png)), (self.b_crop, bool(self.ref_png))):
            w.setEnabled(idle and on)
        for w in (self.head_crop, self.b_crop):           # (the H3 head swap's picture)
            w.setVisible(m == "h3")
        self.crop_info.setVisible(m == "h3")
        self.prompt.setPlaceholderText("The new person's face (empty: described automatically), e.g. “Male, oval "
                                       "face, short black hair, dark beard, brown eyes”" if m == "h3" else DEFAULT_PROMPT)
        why = ""
        if animate and self.heads is not None and not self.boxes:
            why = "The Animate methods need the head marked in the clip (no face was found)"
        elif m == "h3" and not self.h3.get("lora"):
            why = "The head swap LoRA (minimax_h3_head_swap_v1.0_r32) is not among the H3 LoRAs"
        elif m == "h3" and not self.h3.get("turbo"):
            why = (f"The DMD 8-step turbo LoRA ({swap.H3_SWAP_TURBO}.safetensors, drbaph/MiniMax-H3-Turbo-Lora-"
                   "ComfyUI on Hugging Face) is not among the H3 LoRAs")
        elif m == "h3" and self.heads is not None and swap.h3_length(self.heads["frames"]) > swap.H3_MAX_FRAMES:
            why = "H3 makes at most 15 s: use a shorter range of the video"
        if why:
            self.b_gen.setEnabled(False)
            self.b_gen.setToolTip(why)

    def generate(self):
        if not self.b_gen.isEnabled():
            return
        if self.on_start is not None:
            try:
                self.on_start()
            except Exception as e:  # noqa: BLE001  (a hook that fails must not leave the window half started)
                self.state.setText(f"Cannot start: {e}")
                return
        if self.qs is not None:
            for k, v in (("swapSize", self.size.currentData()), ("swapMethod", self.method.currentData()),
                         ("swapBody", self.body.isChecked()), ("swapSharpen", self.sharpen.isChecked())):
                self.qs.setValue(k, v)
        self._busy, self._cancel = True, False
        out_dir = self.project / swap.SWAPS_DIR
        stem = f"swap_{self.nid}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        boxes, (w, h), fps = list(self.boxes), self.heads["size"], self.heads["fps"]
        n, clip, ref = self.heads["frames"], self.clip, self.ref
        size, prompt, seed = self.size.currentData(), self.prompt.text().strip(), random.randint(1, 2**31 - 1)
        scale, sig, connect = self.head_size.value(), self._sig, self.connect
        project, source = self.project, self.source
        ref_png = self.ref_png
        crop = self.ref_crop if self.head_crop.isChecked() else None
        method, body = self.method.currentData(), self.body.isChecked()
        enhance = 0.8 if self.sharpen.isChecked() else 0.0
        persons = next(([Path(ref)] + [Path(x) for x in g if x != ref] for g in self.groups if ref in g),
                       [Path(ref)])[:4]
        self.bar.setRange(0, 0)
        self.bar.show()
        self.state.setText("Connecting to ComfyUI … (loading Wan 2.2 Animate takes a while the first time)")
        self._update()

        def work():
            mask = out_dir / f"{stem}_mask.mp4"
            out = out_dir / f"{stem}.mp4"
            temp = [out_dir / f"{stem}_{x}" for x in ("h3.mp4", "h3.png", "head_ref.png", "pose.mkv", "face.mkv")]
            ok = False

            def check():
                if self._cancel:
                    raise InterruptedError("cancelled")
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                if method == "h3":
                    gen = self.h3["connect"](lambda i, s: sig.progress.emit("make", i, s))
                    self.animator = gen
                    check()
                    length = swap.h3_length(n)
                    head_ref = out_dir / f"{stem}_head_ref.png"   # as the LoRA was trained: 256 px, letterboxed
                    if not letterbox(str(ref_png or ref), head_ref, swap.H3_REF_SIZE, crop):
                        raise RuntimeError(f"cannot read the picture {Path(ref).name}")
                    details = prompt
                    if not details and self.h3.get("describe"):  # the face details, as the creator's workflow
                        sig.progress.emit("describe", 0, 0)
                        details = self.h3["describe"](head_ref)  # (the head the model gets)
                        sig.details.emit(details)
                        check()
                    st = dict(self.h3["settings"] or {})
                    for speed in ("turbo", "hyperflow", "pika"):   # the recipe's own turbo LoRA instead
                        st.pop(speed, None)
                    st.update({"steps": swap.H3_SWAP_STEPS, "frames": length, "width": w, "height": h,
                               "lora": self.h3["turbo"], "strength": 1.0, "recipe": dict(swap.H3_SWAP_RECIPE),
                               "extra_loras": [{"lora": self.h3["lora"], "strength": 1.0}],
                               "ref_paths": [str(head_ref)], "ref_size": "match", "ref_hint": False,
                               "audio": True, "guides": [{"video": str(clip), "frame_idx": 0, "audio": True,
                                                          "at": "start"}]})
                    made_raw = out_dir / f"{stem}_h3.mp4"
                    gen.generate(swap.head_swap_prompt(details), None, made_raw, out_dir / f"{stem}_h3.png", seed,
                                 settings=st)
                    check()
                    swap.trim_with_sound(made_raw, clip, out, n, fps=fps)
                    meta = {"method": "h3", "lora": self.h3["lora"], "frames_made": length, "details": details,
                            "steps": st.get("steps"), "seed": seed}
                elif method == "face":
                    self.animator = connect(lambda i, s: sig.progress.emit("make", i, s))
                    check()
                    meta = self.animator.face_only(clip, out, persons, fps, enhance)
                else:
                    swap.write_mask(boxes, w, h, mask, fps)
                    self.animator = connect(lambda i, s: sig.progress.emit("make", i, s))
                    check()
                    if self.animator.missing:
                        raise RuntimeError(self.animator.missing)
                    gen = gen_size(w, h, SIZES.get(size, 480))
                    key = swap.pose_key(clip, source, gen)
                    cached = swap.cached_pose(project, key)
                    need = ["face"] + (["pose"] if body else [])
                    if all(k in cached for k in need):
                        sig.progress.emit("cached", 0, 0)
                    keep = {k: out_dir / f"{stem}_{k}.mkv" for k in need if k not in cached}
                    try:
                        meta = self.animator.replace(clip, mask, Path(ref), out, seed, n, (w, h), size, prompt,
                                                     fps, pose=cached, keep_pose=keep, body=body,
                                                     lock=persons if method == "animate+lock" else None,
                                                     enhance=enhance)
                    finally:                           # DWPose's work is kept even when the rest failed
                        made = {k: f for k, f in keep.items() if f.is_file()}
                        if made:
                            swap.store_pose(project, key, made, {"source": source, "size": list(gen),
                                                                 "frames": n, "clip": str(clip),
                                                                 "made": time.strftime("%Y-%m-%d %H:%M:%S")})
                    meta.update({"head_size": scale, "mask": str(mask)})
                meta.update({"reference": str(ref), "clip": str(clip), "made": time.strftime("%Y-%m-%d %H:%M:%S")})
                got = swap.probe(out)[2]
                if got != n:
                    raise RuntimeError(f"the made clip has {got} frames instead of {n}")
                ok = True
                sig.finished.emit((out, meta, n), "")
            except Exception as e:  # noqa: BLE001
                safe_emit(sig.finished, None, "cancelled" if self._cancel else str(e))
            finally:                                   # nothing of a failed or stopped try stays in swaps/
                for f in temp + ([] if ok else [out, mask]):
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass
        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        self._cancel = True
        a = self.animator
        if a is not None:
            try:
                a.interrupt()
            except Exception:  # noqa: BLE001
                pass

    def _progress(self, what: str, k: int, n: int):
        if what == "describe":
            self.state.setText("Describing the new person's face (vision model in ComfyUI) …")
            return
        if what == "cached":
            self.state.setText("Using the stored pose and face motion of this clip (DWPose is skipped) …")
            return
        if what == "find":
            self.state.setText(f"Looking for the head: frame {k} of {n}")
        else:
            self.state.setText(f"Making the new head: step {k} of {n}" if n else "Making the new head …")
        if n:
            self.bar.setRange(0, n)
            self.bar.setValue(k)

    def _finished(self, res, error: str):
        self._busy = False
        self.bar.hide()
        self.animator = None
        if res is None:
            self.state.setText(("Stopped." if error == "cancelled" else f"Could not replace the person: {error}")
                               + (" The result before is still there." if self.result is not None else ""))
        else:
            self.result = res
            out, meta, n = res
            if meta.get("method") == "face":
                text = f"Done: the face swapped in {n} frames (all motion is the clip's own)."
            elif meta.get("method") == "h3":
                text = f"Done: {n} frames re-rendered by H3 with the head swap LoRA (the clip's own sound)."
            else:
                gen, windows = meta.get("gen_size") or ("?", "?"), meta.get("windows")
                text = (f"Done: {n} frames, made at {gen[0]}×{gen[1]}"
                        + (f" in {windows} part{'s' if windows != 1 else ''}" if windows else "")
                        + (f" (stored motion used: {', '.join(meta['pose_cached'])})" if meta.get("pose_cached")
                           else "")
                        + ("; its motion is stored for the next try" if meta.get("pose_made") else "")
                        + (", the face locked with HyperSwap" if meta.get("lock") else "") + ".")
            self.state.setText(text + " Compare it frame by frame on the left (drag the divider), then “Use "
                               "this” to put it into the video.")
            self.show_head.setChecked(False)
            self.player.load_result(out)
        self._update()

    def reject(self):
        """Close: a made result is kept only with "Use this"; ask before it is dropped."""
        if self.result is not None and not self._busy and not confirm(
                self, "Replace a person", "Close without using the result? It stays in the project's swaps folder "
                "but the video keeps playing its own clip."):
            return
        super().reject()

    def done(self, r):
        self._closed = True
        if self._busy:
            self.stop()
        self.player.stop_loading()
        super().done(r)
        self._tmp.cleanup()
