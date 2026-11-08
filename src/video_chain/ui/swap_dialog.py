"""Replace a person's head in a real video with another person (Wan 2.2 Animate in ComfyUI, see swap.py).

The clip with the head that is drawn anew (found by the face detector, followed through the clip): check it with
the frame slider, make it larger or smaller; choose the new person's picture; Generate. The result is the clip with
the new head, at its own size, frames and sound; "Use this" makes the video node play it (it can go back)."""
from __future__ import annotations

import random
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
                               QProgressBar, QPushButton, QSlider, QVBoxLayout)

from .. import chain, swap
from ..animate import DEFAULT_PROMPT, SIZES, gen_size, window_count
from .widgets import FrameTile, guard_wheel, label


class _Sig(QObject):
    found = Signal(object, str)             # find_heads result (or None), error
    progress = Signal(str, int, int)        # what, done, of
    finished = Signal(object, str)          # (out path, meta) or None, error


class SwapDialog(QDialog):
    """clip: the converted clip of the video node; pictures: [(label, path)] offered as the new person;
    connect(progress) -> ComfyAnimator (worker thread); on_start(): before Generate (GUI thread);
    play(path, title): show a clip. .result: (path, meta, frames) of the clip to use."""

    def __init__(self, parent, project: Path, nid: str, clip: Path, detector: Path | None,
                 pictures: list[tuple[str, str]], connect, settings=None, on_start=None, play=None,
                 source: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Replace a person in video #{nid[1:]}")
        self.resize(1180, 720)
        self.project, self.nid, self.clip, self.detector = Path(project), nid, Path(clip), detector
        self.connect, self.qs, self.on_start, self.play = connect, settings, on_start, play
        self.source = dict(source or {})          # the video's file and range (the DWPose cache is bound to it)
        self.heads: dict | None = None
        self.boxes: list[tuple] = []
        self.result: tuple[Path, dict, int] | None = None
        self.animator = None
        self._busy = False
        self._cancel = False
        self._frames: dict[int, QImage] = {}
        self._tmp = tempfile.TemporaryDirectory(prefix="video-chain-swap-")
        self._sig = _Sig()
        self._sig.found.connect(self._found)
        self._sig.progress.connect(self._progress)
        self._sig.finished.connect(self._finished)

        top = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(label("The head that is drawn anew", type_="heading"))
        self.view = FrameTile(640, 360)
        self.view.clear_frame("Looking for the head …")
        left.addWidget(self.view)
        srow = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._show_frame)
        self.frame_text = label("", level="tertiary")
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.frame_text)
        left.addLayout(srow)
        hrow = QHBoxLayout()
        self.head_size = QDoubleSpinBox()
        self.head_size.setRange(0.7, 2.0)
        self.head_size.setSingleStep(0.05)
        self.head_size.setValue(float(settings.value("swapHeadSize", 1.0, type=float)) if settings else 1.0)
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
        self.ref_combo.addItem("Choose file…", "")
        self.ref_combo.activated.connect(self._ref_picked)
        right.addWidget(self.ref_combo)
        right.addWidget(label("One picture: the head and shoulders, the face well lit and seen from the front work "
                              "best.", level="tertiary"))
        form = QFormLayout()
        self.size = QComboBox()
        for key in SIZES:
            self.size.addItem({"480p": "480p (faster)", "720p": "720p (sharper, slower)"}[key], key)
        self.size.setCurrentIndex(max(0, self.size.findData(settings.value("swapSize", "480p", type=str)
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
        self.b_play.clicked.connect(lambda: self.result and self.play and
                                    self.play(self.result[0], f"Video #{nid[1:]}: person replaced"))
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
        self._update()
        self._find()

    # ---- the head
    def _find(self):
        if self.detector is None:
            self.found_text.setText("No face detector found (insightface's det_10g.onnx in ComfyUI's "
                                    "models/insightface/models/buffalo_l): the head cannot be found.")
            return
        sig, clip, det, scale = self._sig, self.clip, self.detector, self.head_size.value()

        def work():
            try:
                h = swap.find_heads(clip, det, scale, progress=lambda k, n: sig.progress.emit("find", k, n),
                                    cancelled=lambda: self._cancel)
                sig.found.emit(h, "")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.found.emit(None, str(e))
                except RuntimeError:
                    pass
        self._busy = True
        self.bar.show()
        threading.Thread(target=work, daemon=True).start()

    def _found(self, heads, error: str):
        self._busy = False
        self.bar.hide()
        if heads is None:
            self.found_text.setText(f"Could not find the head: {error}")
            self.view.clear_frame("No head found")
            self._update()
            return
        self.heads = heads
        self.boxes = heads["boxes"]
        n = heads["frames"]
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setEnabled(n > 1)
        missing = n - heads["found"]
        self.found_text.setText(
            f"The face was found in {heads['found']} of {n} frames"
            + (f"; in the other {missing} the head is placed between the frames around them" if missing else "")
            + f". The clip is made in {window_count(n)} part{'s' if window_count(n) != 1 else ''} of up to 77 "
              "frames.")
        self._show_frame(self.slider.value())
        self._update()

    def _resize_heads(self, *_):
        if self.qs is not None:
            self.qs.setValue("swapHeadSize", self.head_size.value())
        if self.heads is not None:
            w, h = self.heads["size"]
            self.boxes = swap.head_boxes(self.heads["track"], w, h, self.head_size.value())
            self._show_frame(self.slider.value())

    def _frame(self, k: int) -> QImage:
        if k not in self._frames:
            png = Path(self._tmp.name) / f"f{k}.png"
            try:
                chain.frame_at(self.clip, k, png)
                self._frames[k] = QImage(str(png))
            except Exception:  # noqa: BLE001
                self._frames[k] = QImage()
        return self._frames[k]

    def _show_frame(self, k: int):
        if self.heads is None:
            return
        img = self._frame(k).convertToFormat(QImage.Format_ARGB32)
        if img.isNull():
            return
        if 0 <= k < len(self.boxes):
            x1, y1, x2, y2 = self.boxes[k]
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(255, 60, 60), max(2, img.width() // 300)))
            p.setBrush(QColor(255, 40, 40, 70))
            p.drawEllipse(QRectF(x1, y1, x2 - x1, y2 - y1))
            p.end()
        self.view.set_image(img)
        self.frame_text.setText(f"frame {k + 1} / {self.heads['frames']}")

    # ---- the new person
    def _ref_picked(self, i: int):
        path = self.ref_combo.itemData(i)
        if path == "":
            start = self.qs.value("swapRefDir", "", type=str) if self.qs is not None else ""
            path, _ = QFileDialog.getOpenFileName(self, "Picture of the new person", start or str(Path.home()),
                                                  "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if not path:
                return
            if self.qs is not None:
                self.qs.setValue("swapRefDir", str(Path(path).parent))
        self.set_reference(path)

    def set_reference(self, path: str | None):
        img = QImage(path) if path else QImage()
        self.ref = str(path) if path and not img.isNull() else None
        if self.ref:
            self.ref_tile.set_image(img)
        else:
            self.ref_tile.clear_frame("Choose a picture of the person")
        self._update()

    # ---- generating
    def _update(self):
        ready = self.heads is not None and bool(self.ref) and not self._busy
        self.b_gen.setEnabled(ready)
        self.b_gen.setToolTip("" if ready else "Wait for the head to be found and choose the new person's picture")
        self.b_stop.setEnabled(self._busy)
        self.b_play.setEnabled(self.result is not None and self.play is not None and not self._busy)
        self.b_use.setEnabled(self.result is not None and not self._busy)
        for w in (self.head_size, self.ref_combo, self.size, self.prompt):
            w.setEnabled(not self._busy)

    def generate(self):
        if not self.b_gen.isEnabled():
            return
        if self.qs is not None:
            self.qs.setValue("swapSize", self.size.currentData())
        if self.on_start is not None:
            self.on_start()
        self._busy, self._cancel = True, False
        self.result = None
        out_dir = self.project / swap.SWAPS_DIR
        stem = f"swap_{self.nid}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        boxes, (w, h), fps = list(self.boxes), self.heads["size"], self.heads["fps"]
        n, clip, ref = self.heads["frames"], self.clip, self.ref
        size, prompt, seed = self.size.currentData(), self.prompt.text().strip(), random.randint(1, 2**31 - 1)
        scale, sig, connect = self.head_size.value(), self._sig, self.connect
        project, source = self.project, self.source
        self.bar.setRange(0, 0)
        self.bar.show()
        self.state.setText("Connecting to ComfyUI … (loading Wan 2.2 Animate takes a while the first time)")
        self._update()

        def work():
            mask = out_dir / f"{stem}_mask.mp4"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                swap.write_mask(boxes, w, h, mask, fps)
                self.animator = connect(lambda i, s: sig.progress.emit("make", i, s))
                if self._cancel:
                    raise InterruptedError("cancelled")
                out = out_dir / f"{stem}.mp4"
                gen = gen_size(w, h, SIZES.get(size, 480))
                key = swap.pose_key(clip, source, gen)
                cached = swap.cached_pose(project, key)
                if cached:
                    sig.progress.emit("cached", 0, 0)
                keep = None if cached else (out_dir / f"{stem}_pose.mkv", out_dir / f"{stem}_face.mkv")
                meta = self.animator.replace(clip, mask, Path(ref), out, seed, n, (w, h), size, prompt, fps,
                                             pose=cached, keep_pose=keep)
                if keep is not None and all(p.is_file() for p in keep):
                    swap.store_pose(project, key, *keep, {"source": source, "size": list(gen), "frames": n,
                                                          "clip": str(clip), "made": time.strftime("%Y-%m-%d %H:%M:%S")})
                meta.update({"reference": str(ref), "head_size": scale, "mask": str(mask), "clip": str(clip),
                             "made": time.strftime("%Y-%m-%d %H:%M:%S")})
                got = swap.probe(out)[2]
                if got != n:
                    raise RuntimeError(f"the made clip has {got} frames instead of {n}")
                sig.finished.emit((out, meta, n), "")
            except Exception as e:  # noqa: BLE001
                try:
                    sig.finished.emit(None, "cancelled" if self._cancel else str(e))
                except RuntimeError:
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
            self.state.setText("Stopped." if error == "cancelled" else f"Could not replace the person: {error}")
        else:
            self.result = res
            out, meta, n = res
            self.state.setText(f"Done: {n} frames, made at {meta['gen_size'][0]}×{meta['gen_size'][1]} in "
                               f"{meta['windows']} part{'s' if meta['windows'] != 1 else ''}"
                               + (" (the stored pose and face motion)" if meta.get("pose_cached") else
                                  "; its pose and face motion are stored for the next try") +
                               ". Play it, then “Use this” to put it into the video.")
            first = Path(self._tmp.name) / "result.png"
            try:
                chain.frame_at(out, min(self.slider.value(), n - 1), first)
                self.view.set_image(QImage(str(first)))
            except Exception:  # noqa: BLE001
                pass
        self._update()

    def done(self, r):
        if self._busy:
            self.stop()
        super().done(r)
