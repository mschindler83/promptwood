"""A list of LoRAs with a strength each (rows of LoRA, strength, remove; '+ LoRA'), the LoRA picker (search, a grid
of covers from CivitAI, other files), and the helpers to fill a LoRA combo box. Used by the clip cards (H3 LoRAs) and the image edit window (Qwen LoRAs)."""
from __future__ import annotations

import os
from pathlib import Path

import threading

from PySide6.QtCore import QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLineEdit,
                               QListView, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

from .. import tree as T
from .widgets import guard_wheel, icon_button, label


LORA_MIN, LORA_MAX = -10.0, 10.0       # strengths (ComfyUI allows -100..100 for LoRAs, -10..10 for the turbo LoRA)


def lora_path(p: str) -> str:
    """How a chosen LoRA file is stored: absolute, links kept as they are (a link in ComfyUI's LoRA folder keeps
    its own name)."""
    return os.path.abspath(os.path.expanduser(p))


def lora_label(path: str | None) -> str:
    return Path(path).stem if path else "none"


def fill_lora_combo(combo: QComboBox, known: list[tuple[str, str]], current: str | None) -> None:
    """Append the offered LoRAs to a combo box and select current (also when it is stored as the target of a
    listed link, or is not in the list at all)."""
    for path, text in known:
        combo.addItem(text, path)
        combo.setItemData(combo.count() - 1, path, Qt.ToolTipRole)
    if not current or current == T.NO_LORA:
        combo.setCurrentIndex(1 if current == T.NO_LORA and combo.findData(T.NO_LORA) >= 0 else 0)
        return
    i = combo.findData(current)
    if i < 0:
        real = os.path.realpath(current)
        i = next((k for k in range(combo.count()) if isinstance(combo.itemData(k), str)
                  and combo.itemData(k) != T.NO_LORA and os.path.realpath(combo.itemData(k)) == real), -1)
    if i < 0:
        combo.addItem(lora_label(current) + ("" if Path(current).is_file() else " (missing)"), current)
        i = combo.count() - 1
    combo.setCurrentIndex(i)


def matches(query: str, *texts: str) -> bool:
    """Every word of the query is in one of the texts (case does not matter; '-', '_' and spaces count the same)."""
    norm = lambda t: t.lower().replace("_", " ").replace("-", " ")   # noqa: E731
    hay = " ".join(norm(t) for t in texts)
    return all(w in hay for w in norm(query).split())


COVER = 156                      # pixels: the side of a card's cover in the LoRA picker
CARD_W, CARD_H = COVER + 16, COVER + 62
COVER_LOADERS = 2


class _CoverSig(QObject):
    cover = Signal(str, str, object)             # LoRA path, Pictures setting, QImage or None


_COVERS: dict[tuple[str, str], QImage] = {}      # (path, setting) -> the cover, kept while the app runs


class _CardDelegate(QStyledItemDelegate):
    """A LoRA as a card: its cover (or its initials), its name in up to two lines, base model and trigger words."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dlg = dialog

    def sizeHint(self, option, index):            # noqa: N802  (Qt naming)
        return QSize(CARD_W, CARD_H)

    def paint(self, p: QPainter, option, index):
        from PySide6.QtWidgets import QStyle
        from .theme import ThemeManager
        t = ThemeManager.instance().theme
        r = option.rect.adjusted(3, 3, -3, -3)
        selected = bool(option.state & QStyle.State_Selected)
        hover = bool(option.state & QStyle.State_MouseOver)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        card = QPainterPath()
        card.addRoundedRect(QRectF(r), 8, 8)
        p.fillPath(card, t.accent_muted() if selected else t.bgOverlay if hover else t.bgRaised)
        if selected:
            p.setPen(QPen(t.accent, 2))
            p.drawPath(card)
        pic = QRectF(r.x() + 5, r.y() + 5, r.width() - 10, r.width() - 10)
        clip = QPainterPath()
        clip.addRoundedRect(pic, 6, 6)
        img = index.data(LoraPickerDialog.COVER)
        name = index.data(Qt.DisplayRole) or ""
        if isinstance(img, QImage) and not img.isNull():
            p.setClipPath(clip)
            p.drawImage(pic, img)
            p.setClipping(False)
        else:
            p.fillPath(clip, t.bgSunken)
            f = QFont(p.font())
            f.setPixelSize(int(pic.height() * 0.28))
            f.setBold(True)
            p.setFont(f)
            p.setPen(t.textDisabled)
            p.drawText(pic, Qt.AlignCenter, initials(name))
        if index.data(LoraPickerDialog.IN_USE):
            f = QFont(option.font)
            f.setPixelSize(t.font.micro)
            f.setBold(True)
            p.setFont(f)
            text = "IN USE"
            w = QFontMetrics(f).horizontalAdvance(text) + 12
            badge = QRectF(pic.right() - w - 5, pic.top() + 5, w, 18)
            path = QPainterPath()
            path.addRoundedRect(badge, 9, 9)
            p.fillPath(path, t.accent)
            p.setPen(t.onAccent)
            p.drawText(badge, Qt.AlignCenter, text)
        f = QFont(option.font)
        f.setPixelSize(t.font.body)
        f.setBold(selected)
        p.setFont(f)
        p.setPen(t.textPrimary)
        fm = QFontMetrics(f)
        box = QRectF(r.x() + 7, pic.bottom() + 5, r.width() - 14, fm.height() * 2 + 1)
        p.drawText(box, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWrapAnywhere, elide_lines(name, fm, int(box.width()), 2))
        sub = index.data(LoraPickerDialog.SUBTITLE) or ""
        if sub:
            f2 = QFont(option.font)
            f2.setPixelSize(t.font.small)
            p.setFont(f2)
            p.setPen(t.textTertiary)
            fm2 = QFontMetrics(f2)
            line = QRectF(r.x() + 7, r.bottom() - fm2.height() - 4, r.width() - 14, fm2.height())
            p.drawText(line, Qt.AlignHCenter | Qt.AlignVCenter, fm2.elidedText(sub, Qt.ElideRight, int(line.width())))
        p.restore()


def initials(name: str) -> str:
    """Up to two letters standing for a LoRA without a cover ("smoke_exhale_v2" -> "SE")."""
    words = [w for w in name.replace("-", " ").replace("_", " ").replace(".", " ").split() if w[:1].isalnum()]
    return "".join(w[0] for w in words[:2]).upper() or "?"


def elide_lines(text: str, fm: QFontMetrics, width: int, lines: int) -> str:
    """The text broken into at most `lines` lines of `width` pixels (anywhere: LoRA names have few spaces), the
    last one shortened with an ellipsis."""
    out, rest = [], text
    while rest and len(out) < lines - 1:
        n = len(rest)
        while n > 1 and fm.horizontalAdvance(rest[:n]) > width:
            n -= 1
        cut = rest.rfind(" ", 0, n) if n < len(rest) else -1
        n = cut + 1 if cut > n // 2 else n
        out.append(rest[:n].rstrip())
        rest = rest[n:].lstrip()
    if rest:
        out.append(fm.elidedText(rest, Qt.ElideRight, width))
    return "\n".join(out)


def _square(img: QImage, side: int) -> QImage:
    """The middle of a picture, as a square of `side` pixels (the cover fills its card)."""
    img = img.scaled(side, side, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    return img.copy((img.width() - side) // 2, (img.height() - side) // 2, side, side)


class LoraPickerDialog(QDialog):
    """Choose a LoRA: a search field over the full names and folders, a grid of cards (each LoRA's cover from its
    CivitAI page, its name, base model and trigger words), 'Other file…' for a file from anywhere. The covers are
    loaded in the background (lora_pictures.cover_file) as the "Pictures" setting allows (off / safe only / all,
    the LoRA window's setting). chosen() -> the path, or None. fetch(item) -> the local still of a picture (tests
    give their own)."""
    COVER, SUBTITLE, IN_USE, SEARCH = Qt.UserRole + 2, Qt.UserRole + 3, Qt.UserRole + 4, Qt.UserRole + 1

    def __init__(self, parent, known: list[tuple[str, str]], current: str | None = None,
                 used: set[str] | None = None, title: str = "Choose a LoRA", file_dir: str = "", settings=None,
                 db=None, fetch=None):
        super().__init__(parent)
        from .. import triggers
        from .lora_window import SHOW
        self.setWindowTitle(title)
        self.resize(1000, 740)
        self.known, self.file_dir, self._chosen = list(known), file_dir, None
        self.qs = settings if settings is not None else getattr(parent, "qs", None)
        try:
            self.db = db or triggers.db()
        except Exception:  # noqa: BLE001
            self.db = None
        self.fetch = fetch
        used = {os.path.realpath(u) for u in (used or set())}
        v = QVBoxLayout(self)
        v.setSpacing(8)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search LoRAs by name or folder … (several words: all must match)")
        self.search.setClearButtonEnabled(True)
        bar.addWidget(self.search, 1)
        bar.addWidget(label("Pictures", level="tertiary"))
        self.pictures = QComboBox()
        for key, text in SHOW:
            self.pictures.addItem(text, key)
        saved = self.qs.value("loraPictures", "safe", type=str) if self.qs is not None else "safe"
        self.pictures.setCurrentIndex(max(0, self.pictures.findData(saved)))
        self.pictures.setToolTip("Each LoRA's cover is the first picture of its CivitAI page, loaded once and kept "
                                 "on disk (this tells CivitAI which pictures you look at).\nSafe only: pictures "
                                 "CivitAI rates as safe.\nOff: nothing is loaded. The LoRA window uses the same "
                                 "setting.")
        bar.addWidget(self.pictures)
        v.addLayout(bar)
        self.count = label("", level="tertiary")
        v.addWidget(self.count)
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setWrapping(True)
        self.list.setUniformItemSizes(True)
        self.list.setGridSize(QSize(CARD_W + 6, CARD_H + 6))
        self.list.setSelectionMode(QListView.SingleSelection)
        self.list.setMouseTracking(True)
        self.list.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list.verticalScrollBar().setSingleStep(24)
        self.list.setItemDelegate(_CardDelegate(self))
        v.addWidget(self.list, 1)
        info = QFrame()
        info.setProperty("role", "notice")
        iv = QVBoxLayout(info)
        iv.setContentsMargins(10, 8, 10, 8)
        iv.setSpacing(2)
        self.sel_name = label("", type_="title")
        self.sel_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sel_facts = label("", level="secondary")
        self.sel_facts.setWordWrap(True)
        iv.addWidget(self.sel_name)
        iv.addWidget(self.sel_facts)
        v.addWidget(info)
        for path, text in sorted(self.known, key=lambda x: Path(x[0]).stem.lower()):
            name = Path(path).stem
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, path)
            it.setData(self.SEARCH, (name, str(Path(path).parent), text))
            it.setData(self.IN_USE, os.path.realpath(path) in used)
            it.setData(self.SUBTITLE, self._subtitle(path))
            it.setData(self.COVER, None)
            it.setToolTip(name)
            self.list.addItem(it)
            if current and os.path.realpath(path) == os.path.realpath(current):
                self.list.setCurrentItem(it)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Choose")
        self.buttons.button(QDialogButtonBox.Ok).setProperty("variant", "primary")
        self.b_other = self.buttons.addButton("Other file…", QDialogButtonBox.ResetRole)
        self.b_other.setToolTip("A .safetensors file from any folder (ComfyUI must find it in its LoRA folders to "
                                "use it)")
        self.b_other.clicked.connect(self._other)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        v.addWidget(self.buttons)

        self._items = {self.list.item(i).data(Qt.UserRole): self.list.item(i) for i in range(self.list.count())}
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._wanted: list[tuple[str, str]] = []
        self._busy: set[tuple[str, str]] = set()
        self._wake = threading.Event()
        self._sig = _CoverSig()
        self._sig.cover.connect(self._cover)
        self._side = int(COVER * max(1.0, self.devicePixelRatioF()))
        self._prioritise = QTimer(self, singleShot=True, interval=60)
        self._prioritise.timeout.connect(self._want_covers)
        self.list.verticalScrollBar().valueChanged.connect(lambda *_: self._prioritise.start())
        for _ in range(COVER_LOADERS):
            threading.Thread(target=self._load_covers, daemon=True).start()

        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._accept)
        self.search.installEventFilter(self)
        self.list.itemDoubleClicked.connect(lambda _it: self._accept())
        self.list.activated.connect(lambda _i: self._accept())
        self.list.currentItemChanged.connect(lambda *_: self._update())
        self.pictures.currentIndexChanged.connect(self._pictures_changed)
        self._filter("")
        self.search.setFocus()

    def _subtitle(self, path: str) -> str:
        """Base model and trigger words of a LoRA, as far as they are known."""
        try:
            info = self.db.info(path) if self.db is not None and Path(path).is_file() else None
        except Exception:  # noqa: BLE001
            info = None
        if not info:
            return ""
        parts = [info.get("base_model") or ""]
        if info.get("words"):
            parts.append(", ".join(info["words"]))
        return " · ".join(x for x in parts if x)

    # ---- the covers
    def show_mode(self) -> str:
        return self.pictures.currentData() or "off"

    def _want_covers(self):
        """The covers still missing, those in sight first; the loaders take them in this order."""
        mode = self.show_mode()
        if self._closed.is_set():
            return
        area = self.list.viewport().rect()
        seen, later = [], []
        for path, it in self._items.items():
            if it.isHidden() or it.data(self.COVER) is not None:
                continue
            (seen if self.list.visualItemRect(it).intersects(area) else later).append((path, mode))
        with self._lock:
            self._wanted = [] if mode == "off" else seen + later
        if self._wanted:
            self._wake.set()

    def _load_covers(self):
        """A loader thread: the covers wanted, one after the other (kept in memory while the app runs)."""
        from .. import lora_pictures
        while not self._closed.is_set():
            with self._lock:
                job = next((j for j in self._wanted if j not in self._busy), None)
                if job is not None:
                    self._wanted.remove(job)
                    self._busy.add(job)
                else:
                    self._wake.clear()
            if job is None:
                self._wake.wait(0.3)
                continue
            path, mode = job
            img = _COVERS.get(job)
            if img is None and self.db is not None:
                try:
                    f = lora_pictures.cover_file(path, self.db, mode, self.fetch)
                    if f is not None:
                        q = QImage(str(f))
                        if not q.isNull():
                            img = _square(q, self._side)
                            img.setDevicePixelRatio(self._side / COVER)
                            _COVERS[job] = img
                except Exception:  # noqa: BLE001  (a LoRA without a cover is shown with its initials)
                    img = None
            with self._lock:
                self._busy.discard(job)
            try:
                self._sig.cover.emit(path, mode, img)
            except RuntimeError:
                return

    def _cover(self, path: str, mode: str, img):
        it = self._items.get(path)
        if it is not None and img is not None and mode == self.show_mode():
            it.setData(self.COVER, img)

    def _pictures_changed(self, *_):
        if self.qs is not None:
            self.qs.setValue("loraPictures", self.show_mode())
        for it in self._items.values():
            it.setData(self.COVER, None)
        self._want_covers()

    def done(self, result: int):                   # noqa: N802  (Qt naming)
        self._closed.set()
        self._wake.set()
        super().done(result)

    def showEvent(self, e):                        # noqa: N802  (Qt naming): the cards in sight come first
        super().showEvent(e)
        self._prioritise.start()

    def resizeEvent(self, e):                      # noqa: N802  (Qt naming)
        super().resizeEvent(e)
        self._prioritise.start()

    def eventFilter(self, obj, e):                 # noqa: N802  (Qt naming): Down from the search goes to the list
        from PySide6.QtCore import QEvent
        if obj is self.search and e.type() == QEvent.KeyPress and e.key() == Qt.Key_Down:
            self.list.setFocus()
            if self.list.currentItem() is None or self.list.currentItem().isHidden():
                self._select_first()
            return True
        return False

    def visible_paths(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if not self.list.item(i).isHidden()]

    def _select_first(self):
        for i in range(self.list.count()):
            if not self.list.item(i).isHidden():
                self.list.setCurrentRow(i)
                return

    def _filter(self, text: str):
        shown = 0
        for i in range(self.list.count()):
            it = self.list.item(i)
            ok = matches(text, *it.data(self.SEARCH))
            it.setHidden(not ok)
            shown += ok
        if self.list.currentItem() is None or self.list.currentItem().isHidden():
            self._select_first()
        total = self.list.count()
        self.count.setText(f"{shown} of {total} LoRA{'s' if total != 1 else ''}" if text.strip() else
                           f"{total} LoRA{'s' if total != 1 else ''}" if total else
                           "No LoRAs found in ComfyUI's LoRA folders (Other file… for one from elsewhere)")
        self._update()
        self._want_covers()

    def _update(self):
        it = self.list.currentItem()
        ok = it is not None and not it.isHidden()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if not ok:
            self.sel_name.setText("No LoRA chosen")
            self.sel_facts.setText("")
            return
        path = it.data(Qt.UserRole)
        self.sel_name.setText(it.text())
        self.sel_name.setToolTip(path)
        try:
            info = self.db.info(path) if self.db is not None and Path(path).is_file() else None
        except Exception:  # noqa: BLE001
            info = None
        facts = []
        if it.data(self.IN_USE):
            facts.append("in use")
        if info and info.get("base_model"):
            facts.append(f"base model {info['base_model']}")
        if info is None:
            facts.append("trigger words: not looked up yet")
        elif info.get("words"):
            facts.append("trigger words: " + ", ".join(info["words"]))
        elif info.get("suggested"):
            facts.append("no trigger words; its training tags suggest " + ", ".join(info["suggested"]))
        else:
            facts.append("no trigger words")
        if info and info.get("model") and info["model"] != it.text():
            facts.insert(0, f"“{info['model']}” on CivitAI")
        if not Path(path).is_file():
            facts.append("the file is missing")
        self.sel_facts.setText("  ·  ".join(facts))

    def _accept(self):
        it = self.list.currentItem()
        if it is not None and not it.isHidden():
            self._chosen = it.data(Qt.UserRole)
            self.accept()

    def _other(self):
        from PySide6.QtWidgets import QFileDialog
        f, _ = QFileDialog.getOpenFileName(self, "LoRA file", self.file_dir, "LoRA (*.safetensors)")
        if f:
            self._chosen = lora_path(f)
            self.accept()

    def chosen(self) -> str | None:
        return self._chosen


def pick_lora(parent, known, current=None, used=None, file_dir: str = "") -> str | None:
    """Show the LoRA picker; the chosen path or None. (Tests replace LoraStack.pick.)"""
    dlg = LoraPickerDialog(parent, known, current, used, file_dir=file_dir)
    try:
        return dlg.chosen() if dlg.exec() == QDialog.Accepted else None
    finally:
        dlg.deleteLater()


class LoraStack(QWidget):
    """A list of LoRAs: rows of (LoRA, strength, remove) and '+ LoRA'; a row's LoRA button and '+ LoRA' open the
    LoRA picker (search, full names, other files). items() -> [{"path", "strength"}] in order. None by default."""
    changed = Signal()
    fileWanted = Signal()                          # kept for callers that add a file themselves
    picked = Signal(str)                           # a LoRA was chosen in the picker (e.g. a file from elsewhere)
    filled = Signal()                              # set_items put a new list in (e.g. the remembered LoRAs)
    pick = staticmethod(pick_lora)

    def __init__(self, parent=None, add_text: str = "+ LoRA"):
        super().__init__(parent)
        self.known: list[tuple[str, str]] = []
        self.file_dir = ""
        self.rows: list[tuple[QWidget, QPushButton, QDoubleSpinBox]] = []
        self.use_prompt = None                     # set by the owner: put a text into the prompt these LoRAs are for
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(4)
        brow = QHBoxLayout()
        brow.setSpacing(4)
        self.none = label("none", level="tertiary")
        self.b_add = QPushButton(add_text)
        self.b_add.setToolTip("Add a LoRA (an add-on that teaches the model a look, a person or a motion): search "
                              "by name, or take a file from another folder; several are applied in this order")
        self.b_add.clicked.connect(self.add_from_picker)
        self.b_add.setAutoDefault(False)               # Enter in a text field of the window must not open it
        brow.addWidget(self.none)
        brow.addWidget(self.b_add)
        brow.addStretch(1)
        self.v.addLayout(brow)
        self._enabled = True

    def set_items(self, known: list[tuple[str, str]], items: list[dict]):
        self.known = list(known)
        for w, _b, _s in self.rows:
            self.v.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self.rows = []
        for it in items or []:
            if it.get("path"):
                self._add_row(it["path"], float(it.get("strength", 1.0)))
        self.none.setVisible(not self.rows)
        self.filled.emit()

    def _used(self) -> set[str]:
        return {b.property("path") for _w, b, _s in self.rows}

    def add_from_picker(self):
        path = self.pick(self.window(), self.known, None, self._used(), self.file_dir)
        if path:
            self._add_row(lora_path(path), 1.0)
            self.changed.emit()
            self.picked.emit(lora_path(path))

    def _choose_again(self, button: QPushButton):
        path = self.pick(self.window(), self.known, button.property("path"), self._used(), self.file_dir)
        if path and path != button.property("path"):
            self._set_button(button, lora_path(path))
            self.changed.emit()
            self.picked.emit(lora_path(path))

    def _set_button(self, button: QPushButton, path: str):
        button.setProperty("path", path)
        missing = "" if Path(path).is_file() else " (missing)"
        button.setText(Path(path).stem + missing)
        button.setToolTip(f"{path}{missing}\nClick to choose another LoRA; right-click for its trigger words and CivitAI details")
        trig = getattr(button.parentWidget(), "trigger_label", None)
        if trig is not None:
            self._show_trigger(trig, path)

    @staticmethod
    def _show_trigger(lab, path: str):
        """The LoRA's trigger words under its row (triggers.py), with where they come from."""
        try:
            from .. import triggers
            info = triggers.db().info(path) if Path(path).is_file() else None
        except Exception:  # noqa: BLE001
            info = None
        if info is None:
            text, tip = "trigger words: not looked up yet", "Looked up in the background (file, then CivitAI)"
        elif info.get("suggested"):
            text = (f"trigger words: none used; its training tags suggest {', '.join(info['suggested'])} "
                    "(right-click to use them)")
            tip = "A guess from the tags on its training images: not put into prompts unless you take it"
        elif not info["words"]:
            text = "trigger words: none" + (" (CivitAI lists none)" if info["source"] == "civitai" else
                                            " (set by you)" if info["source"] == "user" else "")
            tip = info.get("url") or "Neither the file nor CivitAI names any"
        else:
            src = {"user": "set by you", "file": "from the file", "civitai": "from CivitAI"}.get(info["source"], "")
            text = f"trigger words: {', '.join(info['words'])}  ({src})"
            tip = (f"{info.get('model') or ''}  {info.get('url') or ''}".strip() or src)
        if info and info.get("base_model"):
            text += f"  · base model {info['base_model']}"
        lab.setText(text)
        lab.setToolTip(tip + "\nRight-click the LoRA to change them or to see all that is known about it "
                       "(example prompts with their pictures, description)")

    def strengths(self) -> dict[str, float]:
        return {b.property("path"): round(s.value(), 2) for _w, b, s in self.rows}

    def set_strength(self, path: str, value: float) -> bool:
        """Set a LoRA's strength as if typed (the owner hears `changed`); False when it is not in the stack or
        the stack cannot be changed now."""
        for _w, b, spin in self.rows:
            if b.property("path") == path and self._enabled:
                spin.setValue(max(LORA_MIN, min(LORA_MAX, float(value))))
                return True
        return False

    def window_args(self) -> dict:
        """What the LoRA window may do for this stack: set a strength, put an example into its prompt."""
        on = self._enabled
        return {"strengths": self.strengths(), "set_strength": self.set_strength if on else None,
                "use_prompt": self.use_prompt if on else None}

    def refresh_triggers(self):
        for w, b, _s in self.rows:
            if getattr(w, "trigger_label", None) is not None:
                self._show_trigger(w.trigger_label, b.property("path"))

    def _trigger_menu(self, button: QPushButton, pos):
        from PySide6.QtWidgets import QInputDialog, QMenu
        from .. import triggers
        path = button.property("path")
        if not path or not Path(path).is_file():
            return
        info = triggers.db().info(path) or {}
        menu = QMenu(self)
        a_take = menu.addAction(f"Use the suggested words ({', '.join(info['suggested'])})") \
            if info.get("suggested") else None
        a_edit = menu.addAction("Change the trigger words…")
        a_reset = menu.addAction("Use the found trigger words again")
        menu.addSeparator()
        a_details = menu.addAction("About this LoRA (example prompts with pictures, what the app knows)…")
        chosen = menu.exec(button.mapToGlobal(pos))
        if chosen is not None and chosen is a_details:
            from .lora_window import LoraWindow
            LoraWindow(self.window(), self.known, current=path, **self.window_args()).exec()
            self.refresh_triggers()
            return
        try:
            if chosen is not None and chosen is a_take:
                triggers.db().set_user(path, list(info["suggested"]))
            elif chosen is a_edit:
                words, ok = QInputDialog.getText(
                    self, "Trigger words", f"Trigger words of {Path(path).stem} (comma-separated; empty: none):",
                    text=", ".join(triggers.db().words(path) or info.get("suggested") or []))
                if not ok:
                    return
                triggers.db().set_user(path, [w for w in words.split(",") if w.strip()])
            elif chosen is a_reset:
                triggers.db().set_user(path, None)
            else:
                return
        except OSError as e:
            QMessageBox.warning(self, "Trigger words", f"Could not save the trigger words: {e}")
            return
        self.refresh_triggers()
        self.changed.emit()

    def _add_row(self, path: str, strength: float):
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        outer.addLayout(h)
        w.trigger_label = label("", level="tertiary")
        w.trigger_label.setWordWrap(True)
        outer.addWidget(w.trigger_label)
        button = QPushButton(w)
        button.setAutoDefault(False)                   # Enter in a text field must not open the picker
        button.setStyleSheet("text-align: left; padding-left: 6px;")
        button.setMinimumWidth(60)
        from PySide6.QtWidgets import QSizePolicy
        button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)        # long names shorten, the row stays
        self._set_button(button, path)
        button.clicked.connect(lambda: self._choose_again(button))
        button.setContextMenuPolicy(Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(lambda pos: self._trigger_menu(button, pos))
        spin = QDoubleSpinBox(w)
        spin.setRange(LORA_MIN, LORA_MAX)              # some LoRAs need negative strengths (e.g. sliders)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(strength)
        spin.setToolTip(f"Strength of this LoRA ({LORA_MIN:g} … {LORA_MAX:g}; negative values reverse its effect, "
                        "e.g. for slider LoRAs)")
        rm = icon_button("x", "Remove this LoRA", w)
        h.addWidget(button, 1)
        h.addWidget(spin)
        h.addWidget(rm)
        self.v.insertWidget(len(self.rows), w)
        row = (w, button, spin)
        self.rows.append(row)
        spin.valueChanged.connect(lambda *_: self.changed.emit())
        rm.clicked.connect(lambda: self._remove(row))
        for x in (button, spin, rm):
            x.setEnabled(self._enabled)
        guard_wheel(spin)
        self.none.hide()

    def _remove(self, row):
        if row in self.rows:
            self.rows.remove(row)
            self.v.removeWidget(row[0])
            row[0].setParent(None)
            row[0].deleteLater()
            self.none.setVisible(not self.rows)
            self.changed.emit()

    def items(self) -> list[dict]:
        return [{"path": b.property("path"), "strength": round(spin.value(), 2)} for _w, b, spin in self.rows]

    def setEnabled(self, on: bool):          # noqa: N802  (Qt naming)
        self._enabled = on
        self.b_add.setEnabled(on)
        for _w, b, spin in self.rows:
            b.setEnabled(on)
            spin.setEnabled(on)
