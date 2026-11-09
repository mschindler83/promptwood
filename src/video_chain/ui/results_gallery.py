"""Browsing the image window's results: a gallery grouped by Generate run, thumbnails with badges (model, in use,
favourite), filters, and the info card of the selected result. Thumbnails are decoded in a background thread
(each once per file version), so a window with hundreds of results opens at once."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from .theme import theme
from .widgets import label

THUMB_DECODE = 400                    # thumbnails are decoded once at this size (the gallery's largest)
RUN_GAP = 15 * 60                     # older results without a run id: the same prompt within 15 min is one run

MODEL_TAGS = {"krea2": "K2", "qwen": "Q", "qwen21": "Q2.1", "moss": "MOSS", "flux3": "F3", "flux1": "F1",
              "chroma": "CH", "sdxl": "IL"}
EDIT_TAGS = {"2511": "EDIT", "qwen21": "Q2.1 EDIT", "moss": "MOSS EDIT"}


def model_key(meta: dict) -> str:
    """A result's model, for the filter: create:<family> / edit:<model>."""
    if meta.get("mode") == "create":
        return f"create:{meta.get('family') or '?'}"
    return f"edit:{meta.get('edit_model') or '2511'}"


def model_tag(meta: dict) -> str:
    if meta.get("mode") == "create":
        return MODEL_TAGS.get(meta.get("family"), "NEW")
    return EDIT_TAGS.get(meta.get("edit_model") or "2511", "EDIT")


def model_name(key: str) -> str:
    from ..comfy import EDIT_MODELS
    from ..imagegen import FAMILIES
    kind, _, name = key.partition(":")
    if kind == "create":
        return FAMILIES.get(name, {}).get("label", name)
    return "Edit: " + EDIT_MODELS.get(name, name)


def when_text(t: float, now: float | None = None) -> str:
    now = time.time() if now is None else now
    day, today = time.localtime(t), time.localtime(now)
    clock = time.strftime("%H:%M", day)
    if day[:3] == today[:3]:
        return f"Today {clock}"
    if day[:3] == time.localtime(now - 86400)[:3]:
        return f"Yesterday {clock}"
    return time.strftime("%Y-%m-%d %H:%M", day)


def run_groups(items: list[tuple[str, dict, float]]) -> list[dict]:
    """Results (png, meta, mtime; newest first) grouped by Generate run: {"key", "title", "items"} newest run first.
    A run is the results of one Generate (their "run" id); older results without one: the same prompt and model
    made within RUN_GAP of each other."""
    groups: list[dict] = []
    for png, meta, t in items:
        run = meta.get("run")
        g = groups[-1] if groups else None
        same = g is not None and (
            (run and g["run"] == run) or
            (not run and not g["run"] and g["prompt"] == meta.get("prompt", "") and g["model"] == model_key(meta)
             and g["oldest"] - t <= RUN_GAP))
        if not same:
            g = {"key": run or f"{png}", "run": run, "prompt": meta.get("prompt", ""), "model": model_key(meta),
                 "newest": t, "oldest": t, "items": []}
            groups.append(g)
        g["items"].append((png, meta, t))
        g["oldest"] = t
    for g in groups:
        n = len(g["items"])
        prompt = " ".join((g["prompt"] or "").split())
        short = prompt if len(prompt) <= 70 else prompt[:68].rstrip() + "…"
        g["title"] = " · ".join(x for x in (when_text(g["oldest"]), model_name(g["model"]),
                                            f"“{short}”" if short else "", f"{n} result{'s' if n != 1 else ''}") if x)
    return groups


def badged(pix: QPixmap, size: QSize, tag: str, fav: bool, in_use: bool, selected: bool = False) -> QPixmap:
    """A thumbnail fitted into size, with its badges painted on: the model tag (bottom left), IN USE (top left),
    the star (top right) and the selection ring."""
    t = theme()
    dpr = QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0
    out = QPixmap(size * dpr)
    out.setDevicePixelRatio(dpr)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    r = QRect(QPoint(0, 0), size)
    clip = QPainterPath()
    clip.addRoundedRect(r.adjusted(1, 1, -1, -1), 6, 6)
    p.fillPath(clip, t.bgSunken)
    if not pix.isNull():                   # filled (cropped to the tile's shape), as photo galleries do
        img = pix.scaled(size * dpr, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        img.setDevicePixelRatio(dpr)
        w, h = img.width() / dpr, img.height() / dpr
        p.setClipPath(clip)
        p.drawPixmap(QRect(int((size.width() - w) / 2), int((size.height() - h) / 2), int(w), int(h)), img)
        p.setClipping(False)
    f = QFont(p.font())
    f.setPixelSize(10)
    f.setBold(True)
    p.setFont(f)
    fm = p.fontMetrics()

    def pill(text: str, x: int, y: int, bg: QColor, fg: QColor, right: bool = False) -> None:
        w = fm.horizontalAdvance(text) + 10
        box = QRect(x - w if right else x, y, w, 16)
        path = QPainterPath()
        path.addRoundedRect(box, 8, 8)
        p.fillPath(path, bg)
        p.setPen(fg)
        p.drawText(box, Qt.AlignCenter, text)
    dark = QColor(0, 0, 0, 170)
    if tag:
        pill(tag, 5, size.height() - 21, dark, QColor(235, 235, 235))
    if in_use:
        pill("IN USE", 5, 5, t.accent, t.onAccent)
    if fav:
        pill("★", size.width() - 5, 5, QColor(0, 0, 0, 150), QColor(255, 196, 0), right=True)
    if selected:
        p.setPen(QPen(t.accent, 3))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(2, 2, -2, -2), 6, 6)
    p.end()
    return out


class FlowBox(QWidget):
    """Places its child widgets left to right and wraps them into rows; its height follows its width. (A widget
    that places its children itself: a Python QLayout subclass crashed in PySide when items were taken out.)"""

    def __init__(self, parent=None, spacing: int = 8):
        super().__init__(parent)
        self.items: list[QWidget] = []
        self.spacing = spacing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def add(self, w: QWidget):
        w.setParent(self)
        w.show()
        self.items.append(w)
        self._place()

    def clear(self):
        for w in self.items:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        self.items = []
        self._place()

    def _place(self):
        width = max(1, self.width())
        x = y = row = 0
        for w in self.items:
            hint = w.sizeHint().expandedTo(w.minimumSize())
            if w.maximumWidth() < hint.width():
                hint.setWidth(w.maximumWidth())
            if x + hint.width() > width and row > 0:
                x, y, row = 0, y + row + self.spacing, 0
            w.setGeometry(x, y, hint.width(), hint.height())
            x += hint.width() + self.spacing
            row = max(row, hint.height())
        h = y + row
        if self.minimumHeight() != h:
            self.setFixedHeight(h)

    def resizeEvent(self, e):                    # noqa: N802  (Qt naming)
        super().resizeEvent(e)
        self._place()

    def sizeHint(self):                          # noqa: N802
        return QSize(200, self.minimumHeight())


class ResultView(QWidget):
    """The selected result large: fitted into all the space there is (kept in proportion), a text when empty."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = QImage()
        self._text = ""
        self._scaled: tuple | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 200)

    def sizeHint(self):                          # noqa: N802
        return QSize(780, 440)

    def clear_frame(self, text: str = "") -> None:
        self._img, self._text, self._scaled = QImage(), text, None
        self.update()

    def set_image(self, img: QImage) -> None:
        if img.isNull():
            self.clear_frame("no picture")
            return
        self._img, self._text, self._scaled = img, "", None
        self.update()

    def has_image(self) -> bool:
        return not self._img.isNull()

    def paintEvent(self, _e):                    # noqa: N802
        t = theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self._img.isNull():
            p.setPen(t.textTertiary)
            p.drawText(self.rect(), Qt.AlignCenter, self._text)
            p.end()
            return
        dpr = self.devicePixelRatioF()
        target = self._img.size().scaled(self.size() * dpr, Qt.KeepAspectRatio)
        key = (self._img.cacheKey(), target.width(), target.height())
        if self._scaled is None or self._scaled[0] != key:     # smooth (area) scaling once per picture and size
            pm = QPixmap.fromImage(self._img.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            pm.setDevicePixelRatio(dpr)
            self._scaled = (key, pm)
        pm = self._scaled[1]
        w, h = pm.width() / dpr, pm.height() / dpr
        p.drawPixmap(QRect(int((self.width() - w) / 2), int((self.height() - h) / 2), int(w), int(h)), pm)
        p.end()


class Thumb(QWidget):
    """One result in the gallery: click selects, double-click opens it large, the star corner toggles ★."""
    clicked = Signal(str)
    opened = Signal(str)
    starred = Signal(str)

    def __init__(self, png: str, meta: dict, parent=None):
        super().__init__(parent)
        self.png, self.meta = png, meta
        self.pix = QPixmap()
        self.in_use = False
        self.selected = False
        self._size = QSize(200, 150)
        self._cache: tuple | None = None
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(meta.get("prompt", Path(png).name))

    def set_size(self, w: int):
        self._size = QSize(w, int(w * 0.75))
        self._cache = None
        self.setFixedSize(self._size)

    def sizeHint(self):                          # noqa: N802
        return self._size

    def set_state(self, selected: bool | None = None, in_use: bool | None = None, meta: dict | None = None):
        if selected is not None:
            self.selected = selected
        if in_use is not None:
            self.in_use = in_use
        if meta is not None:
            self.meta = meta
        self._cache = None
        self.update()

    def set_pixmap(self, pix: QPixmap):
        self.pix = pix
        self._cache = None
        self.update()

    def rendered(self) -> QPixmap:
        key = (self.pix.cacheKey(), self._size.width(), self.selected, self.in_use, bool(self.meta.get("favorite")))
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, badged(self.pix, self._size, model_tag(self.meta), bool(self.meta.get("favorite")),
                                       self.in_use, self.selected))
        return self._cache[1]

    def paintEvent(self, _e):                    # noqa: N802
        p = QPainter(self)
        p.drawPixmap(0, 0, self.rendered())
        p.end()

    def _star_hit(self, pos) -> bool:
        return pos.x() > self.width() - 30 and pos.y() < 26

    def mousePressEvent(self, e):                # noqa: N802
        if e.button() == Qt.LeftButton:
            if self._star_hit(e.position().toPoint()):
                self.starred.emit(self.png)
            else:
                self.clicked.emit(self.png)

    def mouseDoubleClickEvent(self, e):          # noqa: N802
        if e.button() == Qt.LeftButton and not self._star_hit(e.position().toPoint()):
            self.opened.emit(self.png)


class ResultGallery(QScrollArea):
    """The results as a grid, grouped by Generate run (newest first)."""
    clicked = Signal(str)
    opened = Signal(str)
    starred = Signal(str)
    back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.StrongFocus)
        self.thumbs: dict[str, Thumb] = {}
        self.headers: list[QLabel] = []
        self.size_px = 200
        self._selected: str | None = None
        self._build([])

    def _build(self, groups: list[dict]):
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)
        old = self.thumbs
        self.thumbs, self.headers, self.boxes = {}, [], []
        for g in groups:
            head = label(g["title"], level="secondary")
            head.setProperty("role", "runHeader")
            head.setWordWrap(True)
            head.setToolTip(g["prompt"])
            self.headers.append(head)
            v.addWidget(head)
            box = FlowBox(spacing=8)
            for png, meta, _t in g["items"]:
                th = Thumb(png, meta)
                th.set_size(self.size_px)
                if png in old and not old[png].pix.isNull():
                    th.pix = old[png].pix
                th.selected = png == self._selected
                th.clicked.connect(self.clicked)
                th.opened.connect(self.opened)
                th.starred.connect(self.starred)
                self.thumbs[png] = th
                box.add(th)
            self.boxes.append(box)
            v.addWidget(box)
            v.addSpacing(6)
        if not groups:
            empty = label("No results here (yet, or not with these filters).", level="tertiary")
            empty.setAlignment(Qt.AlignCenter)
            v.addWidget(empty)
        v.addStretch(1)
        self.setWidget(inner)

    def set_groups(self, groups: list[dict], in_use: set[str]):
        from PySide6.QtCore import QTimer
        if getattr(self, "_scroll_pending", None) is None:  # (two rebuilds in a row: the first position counts)
            self._scroll_pending = self.verticalScrollBar().value()
        self._build(groups)
        for png, th in self.thumbs.items():
            th.in_use = png in in_use

        def restore():
            if self._scroll_pending is not None:
                self.verticalScrollBar().setValue(self._scroll_pending)
                self._scroll_pending = None
        QTimer.singleShot(0, restore)                          # (where it was, once laid out)

    def update_thumb(self, png: str, meta: dict, in_use: bool | None = None):
        """One result's badges anew (e.g. its star), without rebuilding the gallery."""
        th = self.thumbs.get(png)
        if th is not None:
            th.set_state(meta=meta, in_use=in_use)

    def set_thumb(self, png: str, pix: QPixmap):
        th = self.thumbs.get(png)
        if th is not None:
            th.set_pixmap(pix)

    def set_selected(self, png: str | None, scroll: bool = True):
        if self._selected in self.thumbs:
            self.thumbs[self._selected].set_state(selected=False)
        self._selected = png
        th = self.thumbs.get(png) if png else None
        if th is not None:
            th.set_state(selected=True)
            if scroll:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: png in self.thumbs and self.ensureWidgetVisible(self.thumbs[png], 20, 20))

    def set_thumb_size(self, px: int):
        self.size_px = int(px)
        for th in self.thumbs.values():
            th.set_size(self.size_px)
        for box in getattr(self, "boxes", []):
            box._place()

    def order(self) -> list[str]:
        return list(self.thumbs)

    def keyPressEvent(self, e):                  # noqa: N802  (arrows step, Enter opens, Esc back to Single)
        order = self.order()
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and self._selected:
            self.opened.emit(self._selected)
        elif e.key() == Qt.Key_Escape:
            self.back.emit()
        elif e.key() in (Qt.Key_Right, Qt.Key_Left) and order:
            i = order.index(self._selected) if self._selected in order else -1
            self.clicked.emit(order[max(0, min(len(order) - 1, i + (1 if e.key() == Qt.Key_Right else -1)))])
        elif e.key() in (Qt.Key_Down, Qt.Key_Up) and order:
            cur = self.thumbs.get(self._selected)
            if cur is None:
                self.clicked.emit(order[0])
                return
            inner = self.widget()
            here = cur.mapTo(inner, cur.rect().center())
            down = e.key() == Qt.Key_Down
            rows = []                                   # the thumbnails in the nearest row above / below
            for png, th in self.thumbs.items():
                c = th.mapTo(inner, th.rect().center())
                if (c.y() > here.y() + 4) if down else (c.y() < here.y() - 4):
                    rows.append((abs(c.y() - here.y()), abs(c.x() - here.x()), png))
            if rows:
                nearest = min(r[0] for r in rows)
                self.clicked.emit(min((r for r in rows if r[0] == nearest), key=lambda r: r[1])[2])
        else:
            super().keyPressEvent(e)


class ThumbLoader(QObject):
    """Decodes result thumbnails in a background thread, newest request first; ready(png, mtime_ns, image)."""
    ready = Signal(str, object, QImage)            # (object: a nanosecond time does not fit a Qt int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: queue.LifoQueue = queue.LifoQueue()
        self._stop = threading.Event()
        threading.Thread(target=self._work, daemon=True).start()

    def request(self, png: str, mtime_ns: int):
        self._q.put((png, mtime_ns))

    def stop(self):
        self._stop.set()
        self._q.put(None)

    def _work(self):
        while not self._stop.is_set():
            job = self._q.get()
            if job is None or self._stop.is_set():
                return
            png, mt = job
            img = QImage(png)
            if not img.isNull():
                img = img.scaled(THUMB_DECODE, THUMB_DECODE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            try:
                self.ready.emit(png, mt, img)
            except RuntimeError:                  # the window is gone
                return


class Chip(QLabel):
    def __init__(self, text: str, tip: str = "", accent: bool = False, parent=None):
        super().__init__(text, parent)
        self.setProperty("role", "badge")
        if accent:
            self.setProperty("severity", "info")
        self.setToolTip(tip)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)


class InfoCard(QFrame):
    """The selected result's facts as chips (model, quality, sampling, size, LoRAs, seed, date), where it is used,
    and its prompt (collapsible, with Copy). text() / toolTip() give the facts as one line too."""
    use_seed = Signal(object)                     # (object: a seed can be bigger than a Qt int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("infoCard")
        t = theme()
        self.setStyleSheet(f"QFrame#infoCard {{ background: {t.bgRaised.name()}; border: 1px solid "
                           f"{t.lineSubtle.name()}; border-radius: 6px; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        self.chips = FlowBox(spacing=4)
        v.addWidget(self.chips)
        prow = QHBoxLayout()
        prow.setSpacing(4)
        self.b_prompt = QToolButton()
        self.b_prompt.setCheckable(True)
        self.b_prompt.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.b_prompt.setArrowType(Qt.RightArrow)
        self.b_prompt.setText("Prompt")
        self.b_prompt.setAutoRaise(True)
        self.b_prompt.toggled.connect(self._toggle)
        self.prompt_line = label("", level="tertiary")
        self.prompt_line.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.b_copy = QPushButton("Copy")
        self.b_copy.setAutoDefault(False)
        self.b_copy.setToolTip("Copy the prompt (as you wrote it) to the clipboard")
        self.b_copy.clicked.connect(self._copy)
        prow.addWidget(self.b_prompt)
        prow.addWidget(self.prompt_line, 1)
        self.prompt_gap = QWidget()
        self.prompt_gap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.prompt_gap.hide()
        prow.addWidget(self.prompt_gap, 1)
        prow.addWidget(self.b_copy)
        v.addLayout(prow)
        self.prompt_box = QPlainTextEdit()
        self.prompt_box.setReadOnly(True)
        self.prompt_box.setMaximumHeight(110)
        self.prompt_box.hide()
        v.addWidget(self.prompt_box)
        self._text, self._prompt = "", ""
        self.set(None)

    def _toggle(self, on: bool):
        self.b_prompt.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.prompt_box.setVisible(on and bool(self._prompt))
        self.prompt_line.setVisible(not on)
        self.prompt_gap.setVisible(on)

    def _copy(self):
        if self._prompt:
            QGuiApplication.clipboard().setText(self._prompt)
            self.b_copy.setText("Copied")

    def text(self) -> str:
        return self._text

    def _clear(self):
        self.chips.clear()

    def set(self, info: dict | None):
        """info: {"original", "where", "model", "sampling", "size", "loras", "seed", "when", "prompt", "trigger_words",
        "negative"} or None (nothing selected)."""
        self._clear()
        self.b_copy.setText("Copy")
        if not info:
            self._text, self._prompt = "", ""
            self.setToolTip("")
            self.prompt_line.setText("")
            self.prompt_box.setPlainText("")
            self.setVisible(False)
            return
        self.setVisible(True)
        parts: list[str] = []

        def chip(text, tip="", accent=False):
            self.chips.add(Chip(text, tip, accent))
            parts.append(text)
        if info.get("original"):
            chip("ORIGINAL", "The image this result was made from (B switches back)", accent=True)
        if info.get("where"):
            chip("IN USE: " + ", ".join(info["where"]), "It stays: the project uses it", accent=True)
        for name, val in (info.get("values") or {}).items():           # made from variations
            chip(f"{name}: {val if str(val).strip() else '∅'}", "The value of this variation", accent=True)
        if info.get("model"):
            chip(info["model"])
        if info.get("sampling"):
            chip(info["sampling"])
        if info.get("size"):
            chip(info["size"])
        for lo in info.get("loras") or []:
            chip(lo, "LoRA × strength")
        if info.get("seed") is not None:
            b = QPushButton(f"seed {info['seed']}  ↻")
            b.setAutoDefault(False)
            b.setProperty("role", "chipButton")
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(max(16, self.chips.items[-1].sizeHint().height()) if self.chips.items else 18)
            b.setStyleSheet("padding: 0 6px;")
            b.setToolTip("Use this seed for the next Generate (and keep it: New seed after each Generate is "
                         "switched off)")
            seed = info["seed"]
            b.clicked.connect(lambda: self.use_seed.emit(int(seed)))
            self.chips.add(b)
            parts.append(f"seed {seed}")
        if info.get("when"):
            chip(info["when"])
        self._text = " · ".join(parts)
        self._prompt = info.get("prompt") or ""
        trig = info.get("trigger_words") or []
        flat = " ".join(self._prompt.split())
        self.prompt_line.setText(flat if len(flat) <= 160 else flat[:158] + "…")
        self.prompt_box.setPlainText(self._prompt + (f"\n\nTrigger words added: {', '.join(trig)}" if trig else "")
                                     + (f"\nNegative: {info['negative']}" if info.get("negative") else ""))
        self.b_copy.setEnabled(bool(self._prompt))
        self._toggle(self.b_prompt.isChecked())
        tip = (f"Prompt: {self._prompt}" + (f"\nTrigger words added: {', '.join(trig)}" if trig else "")
               + (f"\nNegative: {info['negative']}" if info.get("negative") else "")
               + "\n\n“Use its prompt and LoRAs” puts it into the prompt field")
        self.setToolTip(tip)
