"""Image picker with thumbnails: folders first, then the images of the folder as a grid, a large preview of the
selected one. Thumbnails are made in background threads (folders on network shares stay responsive)."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QSlider, QSplitter, QStyle, QVBoxLayout, QWidget)

from .theme import theme
from .widgets import icon_button, label

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_RECENT_DIRS = 8
THUMB_MAX = 320                    # thumbnails are made at this size once and scaled for the grid


class _Thumbs(QObject):
    ready = Signal(int, str, QImage)            # generation, path, thumbnail


def _thumbnail(path: str) -> QImage:
    """A small QImage of an image file, read at reduced size (QImage is fine outside the GUI thread)."""
    r = QImageReader(path)
    r.setAutoTransform(True)
    size = r.size()
    if size.isValid() and (size.width() > THUMB_MAX or size.height() > THUMB_MAX):
        r.setScaledSize(size.scaled(THUMB_MAX, THUMB_MAX, Qt.KeepAspectRatio))
    img = r.read()
    return img if not img.isNull() else QImage()


class ImagePickerDialog(QDialog):
    def __init__(self, parent=None, title: str = "Choose an image", start_dir: str | None = None,
                 settings: QSettings | None = None, places: list[tuple[str, str]] | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 700)
        self.qs = settings or QSettings()
        self._gen = 0
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._thumbs = _Thumbs()
        self._thumbs.ready.connect(self._thumb_ready)
        self._items: dict[str, QListWidgetItem] = {}
        self._cache: dict[tuple, QImage] = {}
        self.folder: Path = Path.home()
        self._chosen: str | None = None
        self._extra_places = places or []

        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.b_up = icon_button("chu", "Up one folder (Backspace)")
        self.b_up.clicked.connect(self.go_up)
        self.path = QLineEdit()
        self.path.setToolTip("Type a folder and press Enter")
        self.path.returnPressed.connect(lambda: self.open_folder(Path(os.path.expanduser(self.path.text().strip()))))
        self.sort = QComboBox()
        self.sort.addItem("Name", "name")
        self.sort.addItem("Newest first", "date")
        self.sort.currentIndexChanged.connect(lambda *_: self.open_folder(self.folder))
        self.zoom = QSlider(Qt.Horizontal)
        self.zoom.setRange(80, 260)
        self.zoom.setFixedWidth(140)
        self.zoom.setToolTip("Thumbnail size")
        top.addWidget(self.b_up)
        top.addWidget(self.path, 1)
        top.addWidget(label("Sort", level="tertiary"))
        top.addWidget(self.sort)
        top.addWidget(label("Size", level="tertiary"))
        top.addWidget(self.zoom)
        v.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.places = QListWidget()
        self.places.setMaximumWidth(240)
        self.places.itemClicked.connect(lambda it: self.open_folder(Path(it.data(Qt.UserRole))))
        split.addWidget(self.places)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(True)
        self.grid.setSpacing(6)
        self.grid.currentItemChanged.connect(self._selected)
        self.grid.itemDoubleClicked.connect(self._activated)
        split.addWidget(self.grid)
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(6, 0, 0, 0)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(280, 280)
        self.preview.setStyleSheet(f"background: {theme().bgSunken.name()}; border-radius: 6px;")
        self.info = label("", level="secondary")
        self.info.setWordWrap(True)
        sv.addWidget(self.preview, 1)
        sv.addWidget(self.info)
        split.addWidget(side)
        split.setStretchFactor(1, 1)
        split.setSizes([200, 620, 300])
        v.addWidget(split, 1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Open).setEnabled(False)
        v.addWidget(self.buttons)

        self.zoom.setValue(self.qs.value("pickerZoom", 150, type=int))
        self.zoom.valueChanged.connect(self._zoomed)
        self._zoomed(self.zoom.value())
        self._fill_places()
        start = Path(os.path.expanduser(start_dir)) if start_dir else None
        if start is None or not start.is_dir():
            last = self.qs.value("pickerDir", "", type=str)
            start = Path(last) if last and Path(last).is_dir() else Path.home()
        self.open_folder(start)

    # ---- places
    def _recent_dirs(self) -> list[str]:
        try:
            return [d for d in json.loads(self.qs.value("pickerRecent", "[]", type=str)) if isinstance(d, str)]
        except ValueError:
            return []

    def _fill_places(self):
        self.places.clear()
        dir_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        entries = [("Home", str(Path.home()))]
        pics = Path.home() / "Pictures"
        if pics.is_dir():
            entries.append(("Pictures", str(pics)))
        entries += list(self._extra_places)
        recent = [d for d in self._recent_dirs() if Path(d).is_dir()]
        for name, path in entries:
            it = QListWidgetItem(dir_icon, name)
            it.setData(Qt.UserRole, path)
            it.setToolTip(path)
            self.places.addItem(it)
        if recent:
            head = QListWidgetItem("Recent folders")
            head.setFlags(Qt.NoItemFlags)
            self.places.addItem(head)
            for d in recent:
                it = QListWidgetItem(dir_icon, Path(d).name or d)
                it.setData(Qt.UserRole, d)
                it.setToolTip(d)
                self.places.addItem(it)

    # ---- folder content
    def open_folder(self, folder: Path):
        try:
            folder = Path(folder).expanduser().resolve()
            entries = list(os.scandir(folder))
        except OSError as e:
            self.info.setText(f"Cannot open {folder}: {e}")
            self.path.setText(str(self.folder))
            return
        self.folder = folder
        self.path.setText(str(folder))
        self._gen += 1
        gen = self._gen
        self.grid.clear()
        self._items = {}
        self.preview.clear()
        self.info.setText("")
        self.buttons.button(QDialogButtonBox.Open).setEnabled(False)
        dirs = sorted((e for e in entries if e.is_dir() and not e.name.startswith(".")), key=lambda e: e.name.lower())
        imgs = [e for e in entries if e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS]
        if self.sort.currentData() == "date":
            imgs.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        else:
            imgs.sort(key=lambda e: e.name.lower())
        dir_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        for e in dirs:
            it = QListWidgetItem(dir_icon, e.name)
            it.setData(Qt.UserRole, ("dir", e.path))
            self.grid.addItem(it)
        blank = self._blank_icon()
        for e in imgs:
            it = QListWidgetItem(blank, e.name)
            it.setData(Qt.UserRole, ("img", e.path))
            it.setToolTip(e.name)
            self.grid.addItem(it)
            self._items[e.path] = it
            st = e.stat()
            key = (e.path, st.st_mtime_ns, st.st_size)
            if key in self._cache:
                self._set_icon(it, self._cache[key])
            else:
                self._pool.submit(self._make_thumb, gen, e.path, key)
        n = len(imgs)
        self.setWindowTitle(f"{self.windowTitle().split(' — ')[0]} — {folder.name or folder} ({n} images)")

    def _make_thumb(self, gen: int, path: str, key: tuple):
        if gen != self._gen:                      # the folder changed meanwhile
            return
        img = _thumbnail(path)
        self._cache[key] = img
        self._thumbs.ready.emit(gen, path, img)

    def _thumb_ready(self, gen: int, path: str, img: QImage):
        if gen != self._gen or path not in self._items:
            return
        self._set_icon(self._items[path], img)

    def _set_icon(self, item: QListWidgetItem, img: QImage):
        if not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))
            item.setData(Qt.UserRole + 1, f"{img.width()}×{img.height()}")

    def _blank_icon(self) -> QIcon:
        pm = QPixmap(8, 8)
        pm.fill(theme().bgSunken)
        return QIcon(pm)

    def _zoomed(self, px: int):
        self.grid.setIconSize(QSize(px, px))
        self.grid.setGridSize(QSize(px + 24, px + 40))
        self.qs.setValue("pickerZoom", px)

    # ---- selection
    def _selected(self, item, _prev=None):
        if item is None:
            return
        kind, path = item.data(Qt.UserRole)
        if kind == "dir":
            self.preview.clear()
            self.info.setText(f"Folder {Path(path).name}")
            self.buttons.button(QDialogButtonBox.Open).setEnabled(False)
            return
        r = QImageReader(path)
        r.setAutoTransform(True)
        size = r.size()
        box = self.preview.size()
        if size.isValid() and (size.width() > box.width() or size.height() > box.height()):
            r.setScaledSize(size.scaled(box, Qt.KeepAspectRatio))
        img = r.read()
        if not img.isNull():
            self.preview.setPixmap(QPixmap.fromImage(img))
        try:
            st = Path(path).stat()
            mb = st.st_size / 2**20
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
        except OSError:
            mb, when = 0.0, ""
        dims = f"{size.width()}×{size.height()}" if size.isValid() else "?"
        self.info.setText(f"{Path(path).name}\n{dims} px · {mb:.1f} MB · {when}")
        self.buttons.button(QDialogButtonBox.Open).setEnabled(True)

    def _activated(self, item):
        kind, path = item.data(Qt.UserRole)
        if kind == "dir":
            self.open_folder(Path(path))
        else:
            self._chosen = path
            self._accept()

    def go_up(self):
        if self.folder.parent != self.folder:
            self.open_folder(self.folder.parent)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Backspace and not self.path.hasFocus():
            self.go_up()
            return
        super().keyPressEvent(e)

    def _accept(self):
        item = self.grid.currentItem()
        if self._chosen is None and item is not None and item.data(Qt.UserRole)[0] == "img":
            self._chosen = item.data(Qt.UserRole)[1]
        if self._chosen is None:
            if item is not None and item.data(Qt.UserRole)[0] == "dir":
                self.open_folder(Path(item.data(Qt.UserRole)[1]))
            return
        self.qs.setValue("pickerDir", str(self.folder))
        recent = [str(self.folder)] + [d for d in self._recent_dirs() if d != str(self.folder)]
        self.qs.setValue("pickerRecent", json.dumps(recent[:MAX_RECENT_DIRS]))
        self.accept()

    def chosen(self) -> str | None:
        return self._chosen

    def done(self, r):
        self._gen += 1                           # pending thumbnails are dropped
        self._pool.shutdown(wait=False, cancel_futures=True)
        super().done(r)


def pick_image(parent, title: str, settings: QSettings | None = None, start_dir: str | None = None,
               places: list[tuple[str, str]] | None = None) -> str | None:
    """Show the picker; the chosen image's path or None."""
    dlg = ImagePickerDialog(parent, title, start_dir, settings, places)
    try:
        return dlg.chosen() if dlg.exec() == QDialog.Accepted else None
    finally:
        dlg.deleteLater()
