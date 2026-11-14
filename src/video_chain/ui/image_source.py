"""The image dialog's image to edit: the source list, a folder of images, the painted mask.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtGui import QImage, QImageReader
from PySide6.QtWidgets import QDialog, QFileDialog

from .image_common import FOLDER_ITEM, FOLDER_PREFIX
from .image_picker import IMAGE_EXTS, pick_image
from .imageio import upright
from .mask_painter import MaskPainterDialog, mask_area, overlay
from .image_host import ImageEditHost
from .qsvalue import qs_value



class SourceMixin(ImageEditHost):
    # ---- source
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
        elif path == FOLDER_ITEM:
            self._choose_folder()
        elif path.startswith(FOLDER_PREFIX):
            self.set_folder(path[len(FOLDER_PREFIX):])
        else:
            self.set_source(path)

    def _choose_folder(self):
        start = qs_value(self.qs, "editFolder", "", str) if self.qs is not None else ""
        d = QFileDialog.getExistingDirectory(self, "Folder of images to edit (each one in turn)", start)
        if d:
            self.set_folder(d)
        else:
            self._select_source_item()

    def _select_source_item(self):
        """The source list shows what is chosen: the folder, the image or (any other file) Choose file…"""
        key = (FOLDER_PREFIX + self._batch_dir) if self.batch else self.source
        i = self.src_combo.findData(key) if key else -1
        self.src_combo.setCurrentIndex(i if i >= 0 else self.src_combo.findData(None))

    @staticmethod
    def folder_images(folder: str | Path) -> list[str]:
        """The images directly in a folder (not its subfolders), by name (img2 before img10)."""
        try:
            files = [f for f in Path(folder).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                     and not f.name.startswith(".")]
        except OSError:
            return []
        return [str(f) for f in sorted(files, key=lambda f: [int(t) if t.isdigit() else t for t in
                                                             re.split(r"(\d+)", f.name.lower())])]

    def set_folder(self, folder: str) -> bool:
        """Edit every image of a folder in turn (shown: the first). False if it has none."""
        imgs = self.folder_images(folder)
        if not imgs:
            self.state.setText(f"No images in {Path(folder).name or folder} (PNG, JPG, WEBP …; subfolders are "
                               "not looked in).")
            self._select_source_item()
            return False
        first = next((f for f in imgs if QImageReader(f).canRead()), None)
        if first is None:
            self.state.setText(f"None of the files in {Path(folder).name} could be read as an image.")
            self._select_source_item()
            return False
        if self.qs is not None:
            self.qs.setValue("editFolder", str(folder))
        self.set_source(first)
        self.batch, self._batch_dir = imgs, str(folder)
        self.set_mask(None)                           # (a mask belongs to one picture)
        key, text = FOLDER_PREFIX + str(folder), f"Folder: {Path(folder).name} ({len(imgs)} images)"
        i = self.src_combo.findData(key)
        if i < 0:
            i = self.src_combo.findData(None)          # (before Choose file…)
            self.src_combo.insertItem(i, text, key)
        else:
            self.src_combo.setItemText(i, text)
        self.src_combo.setCurrentIndex(i)
        self.src_tile.setToolTip(f"{folder}\nShown: the first image, {Path(first).name}")
        self.src_info.setText(f"Folder {Path(folder).name}: {len(imgs)} image{'s' if len(imgs) != 1 else ''}, "
                              "each edited in turn with the same prompt and settings (shown: the first, "
                              f"{Path(first).name}).")
        self._show_mask()
        self._update_buttons()
        self._variations_changed()
        return True

    def set_source(self, path: str):
        img = upright(path)
        if img.isNull():
            self.state.setText(f"Not an image: {Path(path).name}")
            return
        was_batch, self.batch = bool(self.batch), []    # (one image again: not the folder)
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
        if was_batch:
            self._select_source_item()
            self._variations_changed()
        self._update_buttons()

    def paint_mask(self):
        if not self.source:
            return
        img = upright(self.source)
        dlg = MaskPainterDialog(self, img, self.mask, settings=self.qs)
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted:
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
        self.b_mask.setEnabled(bool(self.source) and not self._busy and not self.batch)
        self.b_mask.setToolTip("A mask belongs to one picture: choose a single image to paint one" if self.batch
                               else "")
        self.b_mask_clear.setVisible(on)
        why = self._mask_problem()
        a = mask_area(self.mask) if on else 0.0
        self.mask_info.setText(why or (f"Only the red part ({f'{a * 100:.0f} %' if a >= 0.01 else 'under 1 %'}) is "
                                       "drawn anew; the rest stays exactly as it is. Say what it should show, e.g. “She tilts her head back "
                                       "and looks up at the ceiling” (a short “She is looking up” only moves the "
                                       "eyes)." if on else ""))
