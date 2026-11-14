"""Reading pictures for the image windows, in one place: turned upright by their EXIF orientation (phone photos),
decoded at a reduced size when only a thumbnail or a preview is shown, and the file facts shown next to them."""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QImageReader


def decode_upright(path) -> QImage:
    r = QImageReader(str(path))
    r.setAutoTransform(True)
    return r.read()


_UPRIGHT: OrderedDict = OrderedDict()           # (path, mtime_ns, size) -> image: the last few pictures, decoded once
_UPRIGHT_LOCK = threading.Lock()
_UPRIGHT_KEEP = 3                               # (a big photo is tens of MB decoded)


def upright(path) -> QImage:
    """An image as ComfyUI's LoadImage sees it: turned by its EXIF orientation (phone photos). The latest few are
    kept, so choosing, masking and browsing do not decode the same file again (a changed file is read anew)."""
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return QImage()
    with _UPRIGHT_LOCK:
        hit = _UPRIGHT.get(key)
        if hit is not None:
            _UPRIGHT.move_to_end(key)
            return hit
    img = decode_upright(path)                  # (looked up here, so a test can count the decodes)
    if not img.isNull():
        with _UPRIGHT_LOCK:
            _UPRIGHT[key] = img
            while len(_UPRIGHT) > _UPRIGHT_KEEP:
                _UPRIGHT.popitem(last=False)
    return img


def read_scaled(path, box: QSize | int) -> tuple[QImage, QSize]:
    """An image turned upright and decoded at most as large as `box` (a size, or a number for a square), with the
    size the file has (invalid if it cannot be read). The picture is not read in full when only a thumbnail is
    shown. Any thread."""
    if isinstance(box, int):
        box = QSize(box, box)
    r = QImageReader(str(path))
    r.setAutoTransform(True)
    size = r.size()
    if size.isValid() and (size.width() > box.width() or size.height() > box.height()):
        r.setScaledSize(size.scaled(box, Qt.AspectRatioMode.KeepAspectRatio))
    img = r.read()
    return (img if not img.isNull() else QImage()), size


def image_size(path) -> QSize:
    """The size of an image file from its header (not turned by its orientation; the picture is not decoded)."""
    return QImageReader(str(path)).size()


def can_read(path) -> bool:
    """Whether Qt can read the file as an image (the header only)."""
    return QImageReader(str(path)).canRead()


def file_facts(path) -> tuple[float, str]:
    """(size in MB, modified 'YYYY-MM-DD HH:MM') of a file; (0.0, '') if it cannot be read."""
    try:
        st = Path(path).stat()
        return st.st_size / 2**20, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
    except OSError:
        return 0.0, ""
