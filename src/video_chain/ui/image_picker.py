"""Image picker with thumbnails: folders first, then the images of the folder as a grid, a large preview of the
selected one. Thumbnails are made in background threads (folders on network shares stay responsive).

Thumbnails are kept on the local disk (~/.cache/video-chain/thumbs, keyed by path, size and date), so an image
on a network share is read in full only once; thumbnails the desktop already made (the freedesktop cache in
~/.cache/thumbnails, e.g. from the file manager) are used without reading the image at all."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QSlider, QSplitter, QStyle, QVBoxLayout, QWidget)

from .theme import icon, theme
from .widgets import icon_button, label

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mts", ".m2ts", ".wmv", ".flv", ".mpg", ".mpeg"}
SOUND_EXTS = {".mp3", ".wav", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
# what each kind of picker lists: image (pictures), video (videos), media (sounds and videos: voices, tracks)
TEXT_EXTS = {".txt", ".json"}
KINDS = {"image": (IMAGE_EXTS, "images", "pickerDir"), "video": (VIDEO_EXTS, "videos", "videoDir"),
         "media": (SOUND_EXTS | VIDEO_EXTS, "files", "soundDir"), "text": (TEXT_EXTS, "lists", "textDir")}
MAX_RECENT_DIRS = 8
THUMB_MAX = 320                    # thumbnails are made at this size once and scaled for the grid
THREADS = 8                        # parallel reads: on a network share every file costs round trips
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "video-chain" / "thumbs"
if os.name == "nt":                                # (%LOCALAPPDATA%\\video-chain\\cache\\thumbs)
    from .. import paths as _paths
    CACHE_DIR = _paths.cache_dir() / "thumbs"
DESKTOP_THUMBS = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "thumbnails"
MAX_CACHED = 20000                 # beyond this many cached thumbnails the oldest are removed


class _Thumbs(QObject):
    ready = Signal(int, str, QImage)            # generation, path, thumbnail


def _cache_file(cache_dir: Path, path: str, st: os.stat_result) -> Path:
    key = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}|{THUMB_MAX}".encode()).hexdigest()
    return cache_dir / key[:2] / f"{key}.jpg"


def _desktop_thumbnail(desktop: Path, path: str, st: os.stat_result) -> QImage:
    """A thumbnail the desktop already made (freedesktop thumbnail spec), if it is for this version of the file."""
    uri = "file://" + urllib.parse.quote(path)
    name = hashlib.md5(uri.encode()).hexdigest() + ".png"
    for size in ("x-large", "large", "normal"):
        f = desktop / size / name
        if f.is_file():
            img = QImage(str(f))
            if not img.isNull() and img.text("Thumb::MTime") == str(int(st.st_mtime)):
                return img
    return QImage()


def prune_cache(cache_dir: Path, keep: int = MAX_CACHED) -> None:
    """Remove the oldest cached thumbnails when there are more than keep."""
    try:
        files = [f for d in Path(cache_dir).iterdir() if d.is_dir() for f in d.iterdir() if f.suffix == ".jpg"]
    except OSError:
        return
    if len(files) <= keep:
        return
    files.sort(key=lambda f: f.stat().st_mtime)
    for f in files[:len(files) - keep]:
        f.unlink(missing_ok=True)


def _video_frame(path: str, box: int = THUMB_MAX) -> QImage:
    """A frame of a video (at 1 s, else the first), at most box pixels on its long side (ffmpeg; any thread)."""
    import subprocess
    for at in ("1", "0"):
        try:
            r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", at, "-i", path, "-frames:v", "1", "-vf",
                                f"scale='min({box},iw)':'min({box},ih)':force_original_aspect_ratio=decrease",
                                "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return QImage()
        img = QImage.fromData(r.stdout, "PNG") if r.stdout else QImage()
        if not img.isNull():
            return img
    return QImage()


def _media_info(path: str) -> str:
    """'0:12.5 · 1920×1080 · 24 fps' (videos) or '0:08.0 · sound' (ffprobe; '' if unreadable)."""
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,"
                            "r_frame_rate", "-of", "json", path], capture_output=True, text=True, timeout=20)
        d = json.loads(r.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ""
    secs = float((d.get("format") or {}).get("duration") or 0)
    m, sec = divmod(secs, 60)
    parts = [f"{int(m)}:{sec:04.1f}"]
    v = next((x for x in d.get("streams", []) if x.get("codec_type") == "video"), None)
    if v and v.get("width"):
        parts.append(f"{v['width']}×{v['height']}")
        num, _sl, den = str(v.get("r_frame_rate") or "0/1").partition("/")
        try:
            fps = float(num) / float(den or 1)
            if fps:
                parts.append(f"{fps:.3g} fps")
        except ValueError:
            pass
    if any(x.get("codec_type") == "audio" for x in d.get("streams", [])):
        parts.append("sound" if not v else "with sound")
    return " · ".join(parts)


def _thumbnail(path: str) -> QImage:
    """A small QImage of an image file (or a frame of a video), read at reduced size (any thread)."""
    if Path(path).suffix.lower() in VIDEO_EXTS:
        return _video_frame(path)
    if Path(path).suffix.lower() in SOUND_EXTS | TEXT_EXTS:
        return QImage()
    r = QImageReader(path)
    r.setAutoTransform(True)
    size = r.size()
    if size.isValid() and (size.width() > THUMB_MAX or size.height() > THUMB_MAX):
        r.setScaledSize(size.scaled(THUMB_MAX, THUMB_MAX, Qt.KeepAspectRatio))
    img = r.read()
    return img if not img.isNull() else QImage()


class ImagePickerDialog(QDialog):
    """collections: the open project's pictures by kind, [(name, [(path, caption)])], shown first under "This
    project" (captions such as "Clip #5 · last frame"); folders can be browsed as well.
    kind: what is listed (KINDS): 'image', 'video' (thumbnails are frames) or 'media' (sounds and videos)."""

    def __init__(self, parent=None, title: str = "Choose an image", start_dir: str | None = None,
                 settings: QSettings | None = None, places: list[tuple[str, str]] | None = None,
                 cache_dir: Path | None = CACHE_DIR, desktop_thumbs: Path | None = DESKTOP_THUMBS,
                 collections: list[tuple[str, list[tuple[str, str]]]] | None = None, kind: str = "image"):
        super().__init__(parent)
        self.kind = kind if kind in KINDS else "image"
        self.exts, self.noun, self.dir_key = KINDS[self.kind]
        self.setWindowTitle(title)
        self.resize(1100, 700)
        self.qs = settings or QSettings()
        self._gen = 0
        self._pool = ThreadPoolExecutor(max_workers=THREADS)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.desktop_thumbs = Path(desktop_thumbs) if desktop_thumbs else None
        self.stats = {"memory": 0, "disk": 0, "desktop": 0, "read": 0}
        if self.cache_dir is not None:
            self._pool.submit(prune_cache, self.cache_dir)
        self._thumbs = _Thumbs()
        self._thumbs.ready.connect(self._thumb_ready)
        self._items: dict[str, QListWidgetItem] = {}
        from collections import OrderedDict
        self._cache: OrderedDict[tuple, QImage] = OrderedDict()     # the latest thumbnails (bounded)
        self.folder: Path = Path.home()
        self._chosen: str | None = None
        self._extra_places = places or []
        self.collections = [(n, [(str(p), c) for p, c in items if Path(p).is_file()]) for n, items in collections or []]
        self.collections = [(n, items) for n, items in self.collections if items]   # (all files gone: not listed)
        self.collection: str | None = None        # the project collection shown (None: a folder)

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
        self.sort.currentIndexChanged.connect(lambda *_: self.open_collection(self.collection) if self.collection
                                              else self.open_folder(self.folder))
        self.zoom = QSlider(Qt.Horizontal)
        self.zoom.setRange(80, 260)
        self.zoom.setFixedWidth(140)
        self.zoom.setToolTip("Thumbnail size")
        self.b_fav = icon_button("star", "Add this folder to the favorites")
        self.b_fav.clicked.connect(self.toggle_favorite)
        top.addWidget(self.b_up)
        top.addWidget(self.path, 1)
        top.addWidget(self.b_fav)
        top.addWidget(label("Sort", level="tertiary"))
        top.addWidget(self.sort)
        top.addWidget(label("Size", level="tertiary"))
        top.addWidget(self.zoom)
        v.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.places = QListWidget()
        self.places.setMaximumWidth(240)
        self.places.itemClicked.connect(self._place_clicked)
        self.places.setContextMenuPolicy(Qt.CustomContextMenu)
        self.places.customContextMenuRequested.connect(self._places_menu)
        split.addWidget(self.places)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setUniformItemSizes(True)     # every cell the size of the grid (see _cell)
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
            last = self.qs.value(self.dir_key, "", type=str)
            start = Path(last) if last and Path(last).is_dir() else Path.home()
        names = [n for n, _items in self.collections]
        self._place_key = "pickerPlace" if self.kind == "image" else f"pickerPlace_{self.kind}"
        place = self.qs.value(self._place_key, "", type=str)
        if self.collections and (start_dir is None or place.startswith("col:")):
            # the project's pictures first (the collection used last), unless a folder was used last and given
            self.folder = start
            want = place[4:] if place.startswith("col:") and place[4:] in names else names[0]
            if place == "dir" and start_dir is None:
                self.open_folder(start)
            else:
                self.open_collection(want)
        else:
            self.open_folder(start)

    # ---- places
    def favorites(self) -> list[str]:
        """The favorite folders (shared by every kind of browser), in the order they were added."""
        try:
            return [d for d in json.loads(self.qs.value("pickerFavorites", "[]", type=str)) if isinstance(d, str)]
        except ValueError:
            return []

    def _set_favorites(self, dirs: list[str]):
        self.qs.setValue("pickerFavorites", json.dumps(dirs))
        self._fill_places()
        self._update_fav()

    def toggle_favorite(self, folder: str | None = None):
        """The open folder (or folder) into the favorites, or out of them when it is one already."""
        folder = str(folder or self.folder)
        favs = self.favorites()
        self._set_favorites([d for d in favs if d != folder] if folder in favs else favs + [folder])

    def _update_fav(self):
        on = self.collection is None and str(self.folder) in self.favorites()
        self.b_fav.setEnabled(self.collection is None)
        self.b_fav.setIcon(icon("starf" if on else "star"))
        self.b_fav.setToolTip("Remove this folder from the favorites" if on else "Add this folder to the favorites")

    def _places_menu(self, pos):
        it = self.places.itemAt(pos)
        data = it.data(Qt.UserRole) if it is not None else None
        if not isinstance(data, str) or data not in self.favorites():
            return
        m = QMenu(self)
        act = m.addAction("Remove from favorites")
        if self._exec_menu(m, self.places.viewport().mapToGlobal(pos)) is act:
            self.toggle_favorite(data)

    def _exec_menu(self, menu: QMenu, pos):
        return menu.exec(pos)

    def _recent_dirs(self) -> list[str]:
        try:
            return [d for d in json.loads(self.qs.value("pickerRecent", "[]", type=str)) if isinstance(d, str)]
        except ValueError:
            return []

    def _fill_places(self):
        self.places.clear()
        dir_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        if self.collections:
            head = QListWidgetItem("This project")
            head.setFlags(Qt.NoItemFlags)
            self.places.addItem(head)
            pic_icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
            for name, items in self.collections:
                it = QListWidgetItem(pic_icon, f"{name} ({len(items)})")
                it.setData(Qt.UserRole, ("col", name))
                it.setToolTip(f"{len(items)} {self.noun if len(items) != 1 else self.noun[:-1]} of the project")
                self.places.addItem(it)
        favs = self.favorites()
        if favs:
            head = QListWidgetItem("Favorites")
            head.setFlags(Qt.NoItemFlags)
            self.places.addItem(head)
            for d in favs:
                it = QListWidgetItem(icon("starf"), Path(d).name or d)
                it.setData(Qt.UserRole, d)
                ok = Path(d).is_dir()
                it.setToolTip(d + ("" if ok else " (not reachable right now)") + " · right-click to remove")
                if not ok:
                    it.setForeground(theme().colour("textDisabled") or theme().accent)
                self.places.addItem(it)
        if self.collections or favs:
            head = QListWidgetItem("Folders")
            head.setFlags(Qt.NoItemFlags)
            self.places.addItem(head)
        entries = [("Home", str(Path.home()))]
        for name in {"image": ("Pictures",), "video": ("Videos",), "media": ("Music", "Videos")}[self.kind]:
            if (Path.home() / name).is_dir():
                entries.append((name, str(Path.home() / name)))
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

    def _place_clicked(self, it):
        data = it.data(Qt.UserRole)
        if isinstance(data, (tuple, list)) and data and data[0] == "col":
            self.open_collection(data[1])
        elif data:
            self.open_folder(Path(data))

    def _clear_grid(self):
        self._gen += 1
        self.grid.clear()
        self._items = {}
        self.preview.clear()
        self.info.setText("")
        self.buttons.button(QDialogButtonBox.Open).setEnabled(False)

    def open_collection(self, name: str):
        """Show one kind of the project's pictures (with their captions)."""
        items = next((it for n, it in self.collections if n == name), None)
        if items is None:
            return
        self.collection = name
        self.path.setText(f"This project › {name}")
        self._update_fav()
        self._clear_grid()
        gen = self._gen
        if self.sort.currentData() == "date":
            def mtime(pc):
                try:
                    return os.stat(pc[0]).st_mtime
                except OSError:
                    return 0.0
            items = sorted(items, key=mtime, reverse=True)
        blank = self._blank_icon()
        for path, caption in items:
            if path in self._items:               # (the same file in two roles: shown once)
                continue
            it = QListWidgetItem(blank, caption)
            it.setData(Qt.UserRole, ("img", path))
            it.setData(Qt.UserRole + 2, caption)
            it.setToolTip(f"{caption}\n{path}")
            it.setSizeHint(self._cell())
            self.grid.addItem(it)
            self._items[path] = it
            self._pool.submit(self._make_thumb, gen, path)
        self.setWindowTitle(f"{self.windowTitle().split(' — ')[0]} — {name} ({len(self._items)} {self.noun})")

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
        self.collection = None
        self.path.setText(str(folder))
        self._update_fav()
        self._clear_grid()
        gen = self._gen
        dirs = sorted((e for e in entries if e.is_dir() and not e.name.startswith(".")), key=lambda e: e.name.lower())
        imgs = [e for e in entries if e.is_file() and Path(e.name).suffix.lower() in self.exts]
        if self.sort.currentData() == "date":
            imgs.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        else:
            imgs.sort(key=lambda e: e.name.lower())
        dir_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        for e in dirs:
            it = QListWidgetItem(dir_icon, e.name)
            it.setData(Qt.UserRole, ("dir", e.path))
            it.setSizeHint(self._cell())
            self.grid.addItem(it)
        blank = self._blank_icon()
        for e in imgs:
            it = QListWidgetItem(blank, e.name)
            it.setData(Qt.UserRole, ("img", e.path))
            it.setToolTip(e.name)
            it.setSizeHint(self._cell())
            self.grid.addItem(it)
            self._items[e.path] = it
            self._pool.submit(self._make_thumb, gen, e.path)   # file details are read there (network shares)
        n = len(imgs)
        self.setWindowTitle(f"{self.windowTitle().split(' — ')[0]} — {folder.name or folder} ({n} {self.noun})")

    def _make_thumb(self, gen: int, path: str):
        if gen != self._gen:                      # the folder changed meanwhile
            return
        try:
            st = os.stat(path)
            key = (path, st.st_mtime_ns, st.st_size)
        except OSError:
            return
        img = self._cache.get(key)
        if img is not None:
            self.stats["memory"] += 1
        if img is None and self.cache_dir is not None:          # local disk: no network read
            f = _cache_file(self.cache_dir, path, st)
            if f.is_file():
                img = QImage(str(f))
                if img.isNull():
                    img = None
                else:
                    self.stats["disk"] += 1
                    os.utime(f)                                  # recently used: kept when pruning
        if img is None and self.desktop_thumbs is not None:     # the file manager's thumbnail
            img = _desktop_thumbnail(self.desktop_thumbs, path, st)
            if img.isNull():
                img = None
            else:
                self.stats["desktop"] += 1
        if img is None:                                          # read the image (over the network)
            if gen != self._gen:
                return
            img = _thumbnail(path)
            self.stats["read"] += 1
            if not img.isNull() and self.cache_dir is not None:
                f = _cache_file(self.cache_dir, path, st)
                try:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    tmp = f.with_name(f.stem + ".part.jpg")
                    if img.save(str(tmp), "JPG", 85):
                        tmp.replace(f)
                except OSError:
                    pass
        self._cache[key] = img
        while len(self._cache) > 600:                  # a huge folder must not keep every thumbnail in memory
            self._cache.popitem(last=False)
        self._thumbs.ready.emit(gen, path, img)

    def _thumb_ready(self, gen: int, path: str, img: QImage):
        if gen != self._gen or path not in self._items:
            return
        self._set_icon(self._items[path], img)

    def _set_icon(self, item: QListWidgetItem, img: QImage):
        if not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))
            item.setData(Qt.UserRole + 1, f"{img.width()}×{img.height()}")
        elif Path(item.data(Qt.UserRole)[1]).suffix.lower() in SOUND_EXTS:     # a sound: its icon
            item.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        elif Path(item.data(Qt.UserRole)[1]).suffix.lower() in TEXT_EXTS:      # a list: a file icon
            item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))

    def _blank_icon(self) -> QIcon:
        pm = QPixmap(THUMB_MAX, THUMB_MAX)
        pm.fill(theme().bgSunken)
        return QIcon(pm)

    def _cell(self) -> QSize:
        """The size of every grid cell: the whole thumbnail (icon size) plus two lines of the name. Set on every
        item, because with uniform item sizes the view would otherwise take the first item's (a folder icon or
        the placeholder) for all of them and cut the thumbnails off."""
        px = self.grid.iconSize().width()
        return QSize(px + 18, px + 2 * self.grid.fontMetrics().height() + 10)

    def _zoomed(self, px: int):
        self.grid.setIconSize(QSize(px, px))
        cell = self._cell()
        self.grid.setGridSize(QSize(cell.width() + 6, cell.height() + 6))
        for i in range(self.grid.count()):
            self.grid.item(i).setSizeHint(cell)
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
        ext = Path(path).suffix.lower()
        if ext in TEXT_EXTS:                                     # a list: its first lines
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:   # (only the start of a big file)
                    head = fh.read(4000).splitlines()[:14]
            except OSError:
                head = []
            self.preview.setPixmap(self.style().standardIcon(QStyle.SP_FileIcon).pixmap(64, 64))
            self.info.setText(f"{Path(path).name}\n" + "\n".join(line[:80] for line in head))
            self.buttons.button(QDialogButtonBox.Open).setEnabled(True)
            return
        if ext in VIDEO_EXTS or ext in SOUND_EXTS:              # a frame of the video / the sound's icon, its details
            box = self.preview.size()
            img = _video_frame(path, max(box.width(), box.height())) if ext in VIDEO_EXTS else QImage()
            if not img.isNull():
                self.preview.setPixmap(QPixmap.fromImage(img).scaled(box, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.preview.setPixmap(self.style().standardIcon(QStyle.SP_MediaVolume).pixmap(96, 96))
            try:
                st = Path(path).stat()
                mb, when = st.st_size / 2**20, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
            except OSError:
                mb, when = 0.0, ""
            caption = item.data(Qt.UserRole + 2)
            self.info.setText((f"{caption}\n" if caption else "") + f"{Path(path).name}\n{_media_info(path)}\n"
                              f"{mb:.1f} MB · {when}")
            self.buttons.button(QDialogButtonBox.Open).setEnabled(True)
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
        caption = item.data(Qt.UserRole + 2)
        self.info.setText((f"{caption}\n" if caption else "") + f"{Path(path).name}\n{dims} px · {mb:.1f} MB · {when}")
        self.buttons.button(QDialogButtonBox.Open).setEnabled(True)

    def _activated(self, item):
        kind, path = item.data(Qt.UserRole)
        if kind == "dir":
            self.open_folder(Path(path))
        else:
            self._chosen = path
            self._accept()

    def go_up(self):
        if self.collection is not None:           # from the project's pictures to the folder shown before
            self.open_folder(self.folder)
            return
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
        if self.collection is not None:
            self.qs.setValue(self._place_key, f"col:{self.collection}")
        else:
            self.qs.setValue(self._place_key, "dir")
            self.qs.setValue(self.dir_key, str(self.folder))
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
               places: list[tuple[str, str]] | None = None,
               collections: list[tuple[str, list[tuple[str, str]]]] | None = None, kind: str = "image") -> str | None:
    """Show the picker; the chosen file's path or None. kind: 'image', 'video' or 'media' (sounds and videos).
    The thumbnail cache can be moved with the setting 'thumbCacheDir' (empty = no disk cache)."""
    qs = settings or QSettings()
    cache = qs.value("thumbCacheDir", None)
    cache_dir = CACHE_DIR if cache is None else (Path(cache) if cache else None)
    dlg = ImagePickerDialog(parent, title, start_dir, qs, places, cache_dir=cache_dir, collections=collections,
                            kind=kind)
    try:
        return dlg.chosen() if dlg.exec() == QDialog.Accepted else None
    finally:
        dlg.deleteLater()
