"""The LoRA window: everything the app knows about each LoRA, in one place.

Left, the LoRAs with what is known of each (how many example prompts, whether the prompt refiner has a note). Right,
for the chosen one:
 * Examples: the pictures and videos users posted with it on CivitAI, each with the prompt that made it, its
   settings and likes; searchable; a prompt is copied with one click. Pictures are small stills loaded from CivitAI
   when shown (lora_pictures; switched with "Pictures": off / safe only / all) and kept on disk.
 * How to prompt it: the trigger words and where they come from, the settings its examples were made with (the
   usual strength can be set in the prompt's LoRAs with one click), exactly what the refiner is told about this
   LoRA, the prompt refiner's note (it can be edited) and the user's own remarks for the refiner.
An example can be put into the prompt the window was opened for, or the prompt rewritten in its manner.
 * About: what its creator writes on CivitAI.
Opened from the refine window it also shows what the language model was told (the whole text) and marks the example
prompts that were part of it."""
from __future__ import annotations

import html
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSplitter,
                               QTabWidget, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .. import lora_notes, lora_pictures, triggers
from .lora_details import details_html, themed
from .dialogs import add_action, button_row, confirm
from .lora_stack import lora_label
from .widgets import fit_size, guard_wheel, label, safe_emit
from .qsvalue import qs_value

PER_PAGE = 8
PICTURE = 200                    # pixels: the side of a card's picture
SHOW = (("off", "Off"), ("safe", "Safe only"), ("all", "All"))
LOADERS = 4
TYPING_MS = 200                  # the wait after the last key in "Find in the prompts" before the cards are rebuilt
ARROW_MS = 120                   # the wait on a LoRA in the list before its files are read (arrow keys pass by)


class _Sig(QObject):
    known = Signal(int, object)              # generation, what_is_known of one LoRA
    picture = Signal(int, int, str, str)     # generation, card index, file, error


class ExampleCard(QFrame):
    """One gallery picture: its still, the prompt that made it, its settings."""
    copied = Signal(str)
    used = Signal(str)                       # put this example into the prompt
    adapted = Signal(str)                    # rewrite the prompt in this example's manner

    def __init__(self, ex: dict, told: bool, parent=None, can_use: bool = False, can_adapt: bool = False):
        super().__init__(parent)
        self.ex = ex
        self.setProperty("role", "notice")
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(10)
        self.picture = QLabel()
        self.picture.setFixedSize(PICTURE, PICTURE)
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setWordWrap(True)
        self.picture.setProperty("level", "tertiary")
        h.addWidget(self.picture, 0, Qt.AlignmentFlag.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(4)
        if told:
            v.addWidget(label("Told to the language model in this refine", severity="positive"))
        self.prompt = QLabel(ex["prompt"])
        self.prompt.setWordWrap(True)
        self.prompt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        v.addWidget(self.prompt)
        if ex["negative"]:
            neg = label("Negative: " + ex["negative"], level="tertiary")
            neg.setWordWrap(True)
            neg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.addWidget(neg)
        item = ex["item"]
        facts = [x for x in ("video" if lora_pictures.is_video(item) else "picture",
                             f"{ex['likes']} like{'s' if ex['likes'] != 1 else ''}" if ex["likes"] else "",
                             f"by {ex['user']}" if ex["user"] else "", ex["settings"],
                             f"rated {lora_pictures.level(item)}" if lora_pictures.level(item) != "None" else "") if x]
        line = label(" · ".join(facts), level="tertiary")
        line.setWordWrap(True)
        v.addWidget(line)
        row = QHBoxLayout()
        if can_use and ex["clean"]:
            b_use = QPushButton("Use as my prompt")
            b_use.setToolTip("Put this example into your prompt in place of what is there (Ctrl+Z in the prompt "
                             "takes it back), as plain wording: without <lora:…> tags and (word:1.2) weights\n\n"
                             + ex["clean"])
            b_use.clicked.connect(lambda: self.used.emit(ex["clean"]))
            row.addWidget(b_use)
        if can_adapt and ex["clean"]:
            b_adapt = QPushButton("Rewrite mine like this")
            b_adapt.setProperty("variant", "primary")
            b_adapt.setToolTip("The language model rewrites your prompt in the manner of this example (its wording, "
                               "structure and level of detail) and keeps what yours is about: your subjects, your "
                               "cast and what happens. Nothing of the example's content is taken.")
            b_adapt.clicked.connect(lambda: self.adapted.emit(ex["clean"]))
            row.addWidget(b_adapt)
        b_copy = QPushButton("Copy prompt")
        b_copy.setToolTip("Copy this prompt as it was posted (with its weights and LoRA tags)")
        b_copy.clicked.connect(lambda: self.copied.emit(ex["prompt"]))
        row.addWidget(b_copy)
        if ex["clean"] and ex["clean"] != ex["prompt"]:
            b_clean = QPushButton("Copy as plain wording")
            b_clean.setToolTip("Copy it the way the prompt refiner reads it: without <lora:…> tags and (word:1.2) "
                               "weights\n\n" + ex["clean"])
            b_clean.clicked.connect(lambda: self.copied.emit(ex["clean"]))
            row.addWidget(b_clean)
        page = lora_pictures.page_url(item)
        if page:
            b_open = QPushButton("Open on CivitAI")
            b_open.setToolTip(("Play the video" if lora_pictures.is_video(item) else "See it full size")
                              + f" in your browser\n{page}")
            b_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(page)))
            row.addWidget(b_open)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        h.addLayout(v, 1)

    def show_text(self, text: str):
        self.picture.setPixmap(QPixmap())
        self.picture.setText(text)

    def show_file(self, path: str) -> bool:
        img = QImage(path)
        if img.isNull():
            return False
        self.picture.setText("")
        self.picture.setPixmap(QPixmap.fromImage(img.scaled(PICTURE, PICTURE, Qt.AspectRatioMode.KeepAspectRatio,
                                                            Qt.TransformationMode.SmoothTransformation)))
        return True


class LoraWindow(QDialog):
    """loras: [(path, label)] or paths, the LoRAs to list. current: the one shown first. prompt: the user's
    prompt (from the refine window: examples can be sorted by nearness to it). told: the LoRA part of the
    instruction the language model got (or will get), shown word for word. fetch(item) -> the local file of a
    gallery picture's still (lora_pictures.fetch; tests give their own).
    What the window may do where it was opened from (each None: not offered): use_prompt(text) puts an example
    into the prompt; adapt(example) has the prompt rewritten in the example's manner (the window closes);
    strengths {path: strength} of the LoRAs in the prompt's stack and set_strength(path, value) -> bool."""
    # (these names shadow QWidget methods, which this class never uses)
    find: QLineEdit  # type: ignore[assignment]
    scroll: QScrollArea  # type: ignore[assignment]

    def __init__(self, parent=None, loras=(), current: str | None = None, settings: QSettings | None = None,
                 prompt: str = "", told: str = "", told_title: str = "What the language model was told",
                 fetch: Callable[[dict], Path | str | None] | None = None, db: triggers.TriggerDB | None = None,
                 use_prompt: Callable[[str], None] | None = None, adapt: Callable[[str], None] | None = None,
                 strengths: dict[str, float] | None = None,
                 set_strength: Callable[[str, float], bool] | None = None):
        super().__init__(parent)
        self.setWindowTitle("LoRAs: what is known of them")
        fit_size(self, 1180, 780)
        self.qs = settings if settings is not None else getattr(parent, "qs", None)
        self.db = db or triggers.db()
        self.fetch = fetch or (lambda item: lora_pictures.fetch(item, self.db))
        self.prompt_text, self.told = prompt or "", told or ""
        self.use_prompt, self.adapt, self.set_strength = use_prompt, adapt, set_strength
        self.strengths = dict(strengths or {})
        self.loras = [(str(x), lora_label(str(x))) if isinstance(x, (str, Path)) else (str(x[0]), str(x[1]))
                      for x in loras]
        if current and str(current) not in [p for p, _l in self.loras]:
            self.loras.insert(0, (str(current), lora_label(str(current))))
        self.path: str | None = None
        self.material: dict | None = None
        self.examples: list[dict] = []
        self.shown: list[dict] = []
        self.cards: list[ExampleCard] = []
        self.page = 0
        self._gen = 0                              # (answers of an earlier page or LoRA are dropped)
        self._closed = threading.Event()
        self._kept_of: dict = {}                   # (is a kept still on disk? per example: looked up once)
        self._ctx = None                           # (details, info, mat, k) of the LoRA shown
        self._pending: str | None = None           # the LoRA chosen in the list, shown when the arrow keys rest
        self._reverting = False
        self._find_timer, self._pick_timer = QTimer(self), QTimer(self)
        for timer, ms, fn in ((self._find_timer, TYPING_MS, self._refill), (self._pick_timer, ARROW_MS, self._show_pending)):
            timer.setSingleShot(True)
            timer.setInterval(ms)
            timer.timeout.connect(fn)
        self._jobs: queue.Queue = queue.Queue()
        self._sig = _Sig()
        self._sig.known.connect(self._known)
        self._sig.picture.connect(self._picture)

        v = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.find_lora = QLineEdit()
        self.find_lora.setPlaceholderText("Find a LoRA")
        self.find_lora.setClearButtonEnabled(True)
        self.find_lora.textChanged.connect(self._filter_loras)
        lv.addWidget(self.find_lora)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._selected)
        lv.addWidget(self.list, 1)
        self.total = label("", level="tertiary")
        self.total.setWordWrap(True)
        lv.addWidget(self.total)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.title = label("", type_="title")
        rv.addWidget(self.title)
        self.facts = label("", level="secondary")
        self.facts.setWordWrap(True)
        rv.addWidget(self.facts)
        self.tabs = QTabWidget()
        rv.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        v.addWidget(split, 1)

        # -- examples
        ex = QWidget()
        ev = QVBoxLayout(ex)
        bar = QHBoxLayout()
        self.find = QLineEdit()
        self.find.setPlaceholderText("Find in the prompts, e.g. “rain night” (all words)")
        self.find.setClearButtonEnabled(True)
        self.find.textChanged.connect(lambda *_: self._find_timer.start())
        bar.addWidget(self.find, 1)
        self.sort = QComboBox()
        self.sort.addItem("Most liked first", "liked")
        if self.prompt_text.strip():
            self.sort.addItem("Nearest to my prompt first", "near")
        self.sort.setToolTip("Most liked: as CivitAI's users rated the pictures. Nearest: the prompts that share "
                             "the most words with yours (how the prompt refiner chooses its examples)")
        self.sort.currentIndexChanged.connect(lambda *_: self._refill())
        bar.addWidget(self.sort)
        bar.addWidget(label("Pictures", level="tertiary"))
        self.pictures = QComboBox()
        for key, text in SHOW:
            self.pictures.addItem(text, key)
        saved = qs_value(self.qs, "loraPictures", "safe", str) if self.qs is not None else "safe"
        self.pictures.setCurrentIndex(max(0, self.pictures.findData(saved)))
        self.pictures.setToolTip("The pictures are small stills loaded from CivitAI when a page is shown (a video: "
                                 "one frame of it) and kept on disk, so each is loaded once. This tells CivitAI "
                                 "which pictures you look at.\nSafe only: pictures CivitAI rates as safe; the "
                                 "others show their prompt without the picture.\nOff: nothing is loaded.")
        self.pictures.currentIndexChanged.connect(self._pictures_changed)
        bar.addWidget(self.pictures)
        ev.addLayout(bar)
        self.count = label("", level="tertiary")
        self.count.setWordWrap(True)
        ev.addWidget(self.count)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.box = QWidget()
        self.box_v = QVBoxLayout(self.box)
        self.box_v.setContentsMargins(0, 0, 6, 0)
        self.box_v.setSpacing(8)
        self.scroll.setWidget(self.box)
        ev.addWidget(self.scroll, 1)
        nav = QHBoxLayout()
        self.b_prev, self.b_next = QToolButton(), QToolButton()
        self.b_prev.setText("‹ Previous")
        self.b_next.setText("Next ›")
        self.b_prev.clicked.connect(lambda: self.show_page(self.page - 1))
        self.b_next.clicked.connect(lambda: self.show_page(self.page + 1))
        self.page_label = label("", level="tertiary")
        nav.addStretch(1)
        nav.addWidget(self.b_prev)
        nav.addWidget(self.page_label)
        nav.addWidget(self.b_next)
        nav.addStretch(1)
        ev.addLayout(nav)
        self.tabs.addTab(ex, "Examples")
        # -- how to prompt it / about / told
        how = QWidget()
        hv = QVBoxLayout(how)
        self.how = QTextBrowser()
        self.how.setOpenLinks(False)
        self.how.anchorClicked.connect(self._link)
        hv.addWidget(self.how, 3)
        self.note_title = label("", level="tertiary")
        self.note_title.setWordWrap(True)
        hv.addWidget(self.note_title)
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("How to prompt this LoRA. Empty: the language model writes a note at the "
                                          "next refine with this LoRA.")
        self.note_edit.setToolTip("The note the prompt refiner is given on this LoRA. Change it or write your own: "
                                  "yours stays until you change it, also when CivitAI tells something new.")
        hv.addWidget(self.note_edit, 2)
        hv.addWidget(label("Your own remarks for the prompt refiner (it follows them; they stay when the note is "
                           "written again)", level="tertiary"))
        self.remarks_edit = QPlainTextEdit()
        self.remarks_edit.setPlaceholderText("e.g. “works best in close-ups”, “never write ‘portrait’ with it”, "
                                             "“put the trigger word at the very start”")
        self.remarks_edit.setMaximumHeight(72)
        hv.addWidget(self.remarks_edit, 1)
        nrow = QHBoxLayout()
        self.b_save = QPushButton("Save note and remarks")
        self.b_save.clicked.connect(lambda: self.save_note())
        self.b_rewrite = QPushButton("Let the model write the note again")
        self.b_rewrite.setToolTip("Forget this note (yours or the model's); the language model writes a new one at "
                                  "the next refine with this LoRA. Your remarks stay.")
        self.b_rewrite.clicked.connect(self._forget_note)
        nrow.addWidget(self.b_save)
        nrow.addWidget(self.b_rewrite)
        nrow.addStretch(1)
        hv.addLayout(nrow)
        self.note_edit.textChanged.connect(self._note_buttons)
        self.remarks_edit.textChanged.connect(self._note_buttons)
        self._saved = {"note": "", "remarks": ""}
        self.tabs.addTab(how, "How to prompt it")
        self.about = QTextBrowser()
        self.about.setOpenLinks(False)
        self.about.anchorClicked.connect(self._link)
        self.tabs.addTab(self.about, "About (CivitAI)")
        self.told_view = None
        if self.told.strip():
            self.told_view = QPlainTextEdit(self.told)
            self.told_view.setReadOnly(True)
            self.told_view.setToolTip("Word for word: the part of the language model's instruction about the "
                                      "LoRAs of this prompt")
            self.tabs.addTab(self.told_view, told_title)
            self.tabs.setCurrentWidget(self.told_view)

        self.state = label("", level="secondary")
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        buttons, _none = button_row(self)
        self.b_forget = add_action(buttons, "Delete the kept pictures", destructive=True)
        self.b_forget.setToolTip("Delete the stills kept on disk (they are loaded again when shown)")
        self.b_forget.clicked.connect(self._forget_pictures)
        v.addWidget(buttons)
        guard_wheel(self)

        for path, text in self.loras:
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, path)
            it.setData(Qt.ItemDataRole.UserRole + 1, text)
            it.setToolTip(path)
            self.list.addItem(it)
        self._known_of: dict[str, dict] = {}
        self._loaders: list[threading.Thread] = []      # (started when the first picture is wanted)
        self._read_all()
        first = str(current) if current else (self.loras[0][0] if self.loras else None)
        if first:
            self._select(first)
            self.flush()
        else:
            self.title.setText("No LoRA")
            self.facts.setText("Add a LoRA to a clip or an image first.")

    # ---- the list
    def _select(self, path: str):
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                self.list.setCurrentRow(i)
                return

    def flush(self):
        """Do now what the typing / arrow-key waits still hold back."""
        if self._pick_timer.isActive():
            self._pick_timer.stop()
            self._show_pending()
        if self._find_timer.isActive():
            self._find_timer.stop()
            self._refill()

    def _selected(self, cur, old):
        """The list's choice changed: ask about unsaved edits first, then show the LoRA when the keys rest."""
        if self._reverting or cur is None:
            return
        path = cur.data(Qt.ItemDataRole.UserRole)
        if path != self.path and not self._leave_ok():
            self._pending = None
            self._pick_timer.stop()
            self._reverting = True
            try:
                if old is not None:
                    self.list.setCurrentItem(old)
            finally:
                self._reverting = False
            return
        self._pending = path
        self._pick_timer.start()

    def _show_pending(self):
        path, self._pending = self._pending, None
        if path and path != self.path:
            self.show_lora(path)

    def _read_all(self):
        """What is known of every listed LoRA, read in the background (each has a file of its own)."""
        paths, sig, gen, db, closed = [p for p, _l in self.loras], self._sig, self._gen, self.db, self._closed

        def work():
            for p in paths:
                if closed.is_set():
                    return
                k = lora_notes.what_is_known(p, db)
                if not safe_emit(sig.known, gen, k):
                    return
        threading.Thread(target=work, daemon=True).start()

    def _known(self, _gen: int, k: dict):
        self._known_of[k["path"]] = k
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == k["path"]:
                it.setText(f"{it.data(Qt.ItemDataRole.UserRole + 1)}\n   {lora_notes.known_text(k)}")
                it.setToolTip(f"{k['path']}\n{k['name']}" + (f" (for {k['base']})" if k["base"] else ""))
        ks = list(self._known_of.values())
        if len(ks) == len({p for p, _l in self.loras}):
            self.total.setText(f"{len(ks)} LoRA{'s' if len(ks) != 1 else ''}: {sum(1 for k in ks if k['prompts'])} "
                               f"with example prompts ({sum(k['prompts'] for k in ks)} prompts), "
                               f"{sum(1 for k in ks if k['note'])} with a note, "
                               f"{sum(1 for k in ks if not k['known'])} nothing known of.")

    def _filter_loras(self, text: str):
        words = text.lower().split()
        for i in range(self.list.count()):
            it = self.list.item(i)
            hay = (it.text() + " " + str(it.data(Qt.ItemDataRole.UserRole))).lower()
            it.setHidden(not all(w in hay for w in words))

    # ---- one LoRA
    def show_lora(self, path: str):
        self.path = path
        self._gen += 1
        self._kept_of = {}
        details = self.db.details(path)
        info = self.db.info(path) or {}
        k = lora_notes.what_is_known(path, self.db)
        mat = lora_notes.material(details, self.db.words(path), path) if (details or info.get("words")) else None
        self.material = mat
        self.title.setText(k["name"])
        if details:
            self.facts.setText(" · ".join(x for x in (
                f"for {k['base']}" if k["base"] else "",
                f"{k['prompts']} different example prompt{'s' if k['prompts'] != 1 else ''}"
                + (f" ({k['pictures']} gallery pictures tell theirs" + (f", of {k['seen']} looked at" if k["seen"]
                                                                        else "") + ")" if k["pictures"] else ""),
                "the refiner has a note" if k["note"] else
                "the refiner writes its note at the next refine" if k["note_missing"] else "no note",
                Path(path).name) if x))
        else:
            self.facts.setText(("CivitAI does not know this file. " if info.get("source") == "none" else
                                "CivitAI was not asked about this file yet (it is looked up in the background when "
                                "the lookup is on in the settings). ") + Path(path).name)
        self.examples = lora_notes.examples(details)
        told = set()
        if self.told and mat:                      # (the examples a refine shows: those in the text it was told)
            told = {e["clean"] for e in self.examples if e["clean"] and
                    lora_notes._cut(e["clean"], lora_notes.GUIDE_EXAMPLE_CHARS) in self.told}
        for e in self.examples:
            e["told"] = e["clean"] in told
        self._ctx = (details, info, mat, k)
        self.how.setHtml(self._how_html(path, details, info, mat, k))
        n = lora_notes.note_of(self.db, path, mat)
        self._saved = {"note": n["note"], "remarks": n["remarks"]}
        for edit, text in ((self.note_edit, n["note"]), (self.remarks_edit, n["remarks"])):
            edit.blockSignals(True)
            edit.setPlainText(text)
            edit.blockSignals(False)
        self.note_title.setText(
            "The prompt refiner's note: yours (it stays as you wrote it)" if n["edited"] else
            "The prompt refiner's note, written by the language model from the creator's text and the most liked "
            "example prompts (you can change it)" if n["note"] else
            "The prompt refiner's note: none yet; the language model writes it at the next refine with this LoRA "
            "(you can write your own)" if k["note_missing"] else
            "The prompt refiner's note: none (too little is known for the model to write one; you can write your "
            "own)")
        self._note_buttons()
        self.about.setHtml(details_html(details, None) if details else
                           "<p>CivitAI has told nothing about this LoRA.</p>")
        self._refill()

    def _how_html(self, path: str, details, info: dict, mat, k: dict) -> str:
        e = html.escape
        out = []
        source = {"user": "set by you", "file": "named in the LoRA file", "civitai": "from CivitAI",
                  "tags": "a guess from its training tags (not used until you take them)",
                  "none": "none known"}.get(info.get("source", ""), "not looked up yet")
        words = info.get("words") or []
        out.append("<h3>Trigger words</h3><p>" + (e(", ".join(words)) if words else "<i>none</i>")
                   + f" <span style='color:gray'>({e(source)})</span></p>")
        if info.get("suggested"):
            out.append(f"<p style='color:gray'>Suggested: {e(', '.join(info['suggested']))} (right-click the LoRA "
                       "in a clip to use them)</p>")
        out.append("<h3>Settings its example pictures were made with</h3><p>"
                   + (e(k["settings"]) + f" <span style='color:gray'>(counted over "
                      f"{(mat or {}).get('settings', {}).get('count', 0)} pictures: the usual range, not a rule)</span>"
                      if k["settings"] else "<i>the examples do not tell</i>") + "</p>")
        usual, now = k.get("strength"), self.strengths.get(path)
        if usual is not None and now is not None:
            if abs(now - usual) < 0.005:
                out.append(f"<p>This prompt uses it at {now:.2f}, the usual strength of its examples.</p>")
            elif self.set_strength is not None:
                out.append(f"<p>This prompt uses it at {now:.2f}. <a href='strength:{usual:.2f}'>Set its strength "
                           f"to {usual:.2f}</a>, the usual one of its examples.</p>")
            else:
                out.append(f"<p>This prompt uses it at {now:.2f}; its examples usually use {usual:.2f}.</p>")
        elif usual is not None:
            out.append(f"<p style='color:gray'>Usual strength: {usual:.2f}"
                       + (" (this LoRA is not in the prompt's stack)" if self.strengths else "") + ".</p>")
        guide = lora_notes.guidance([path], self.db, prompt=self.prompt_text)
        out.append("<h3>What the prompt refiner is told about this LoRA</h3>")
        if guide:
            out.append("<p style='color:gray'>Word for word, when this is the only LoRA of the prompt (with more "
                       "LoRAs each gets less room)" + (", with the examples nearest to your prompt" if
                                                       self.prompt_text.strip() else "") + ":</p>"
                       "<pre style='white-space:pre-wrap'>" + e(guide) + "</pre>")
        else:
            out.append("<p><i>Nothing: the app knows nothing that helps writing a prompt for it.</i></p>")
        return themed("\n".join(out))

    def _link(self, url: QUrl):
        s = url.toString()
        if s.startswith("strength:") and self.path and self.set_strength is not None:
            try:
                value = float(s[9:])
            except ValueError:
                return
            if self.set_strength(self.path, value):
                self.strengths[self.path] = value
                self.state.setText(f"{self.title.text()}: strength set to {value:.2f} in this prompt's LoRAs.")
                if self._ctx is not None:          # (only the How tab changes: what is typed in the note stays)
                    self.how.setHtml(self._how_html(self.path, *self._ctx[:3], self._ctx[3]))
            else:
                self.state.setText("The strength could not be set (the LoRA is not in the stack, or the clip "
                                   "cannot be changed now).")
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    # ---- the note and the user's remarks
    def _note_buttons(self):
        changed = (self.note_edit.toPlainText().strip() != self._saved["note"]
                   or self.remarks_edit.toPlainText().strip() != self._saved["remarks"])
        self.b_save.setEnabled(bool(self.path) and changed)
        self.b_rewrite.setEnabled(bool(self.path) and bool(self._saved["note"]))

    def _forget_note(self):
        if confirm(self, "Forget the note", "Forget this note (yours or the model's)? The language model writes a "
                         "new one at the next refine with this LoRA. Your remarks stay."):
            self.save_note(forget=True)

    def _unsaved(self) -> bool:
        return bool(self.path) and (self.note_edit.toPlainText().strip() != self._saved["note"]
                                    or self.remarks_edit.toPlainText().strip() != self._saved["remarks"])

    def _ask_unsaved(self) -> str:
        """"save", "discard" or "cancel": what to do with the note and remarks typed but not kept."""
        box = QMessageBox(QMessageBox.Icon.Question, "LoRAs", "The note or remarks of this LoRA are not kept yet.",
                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, self)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        got = box.exec()
        return "save" if got == QMessageBox.StandardButton.Save else "discard" if got == QMessageBox.StandardButton.Discard else "cancel"

    def _leave_ok(self) -> bool:
        """True when the shown LoRA may be left: nothing unsaved, or kept / thrown away as the user chose."""
        if not self._unsaved():
            return True
        answer = self._ask_unsaved()
        return self.save_note() if answer == "save" else answer == "discard"

    def save_note(self, forget: bool = False) -> bool:
        """Keep the note as edited and the remarks (forget: no note, so the model writes one again)."""
        if not self.path:
            return False
        note = "" if forget else self.note_edit.toPlainText()
        same = not forget and note.strip() == self._saved["note"]
        ok = lora_notes.save_user(self.db, self.path, None if same else note, self.remarks_edit.toPlainText())
        if not ok:
            self.state.setText("Could not be kept: the app has not looked at this LoRA file yet (it does so in the "
                               "background after a LoRA scan). Try again in a moment.")
            return False
        self.state.setText("The note is forgotten; the language model writes a new one at the next refine with "
                           "this LoRA. Your remarks stay." if forget else
                           "Kept. The prompt refiner is given this from the next refine on.")
        self.show_lora(self.path)
        self._read_all()
        return True

    def _use(self, text: str):
        if self.use_prompt is not None:
            self.use_prompt(text)
            self.accept()

    def _adapt(self, text: str):
        if self.adapt is not None:
            self.accept()
            self.adapt(text)

    # ---- the examples
    def show_mode(self) -> str:
        return self.pictures.currentData() or "off"

    def _refill(self):
        words = self.find.text().lower().split()
        shown = [e for e in self.examples if all(w in e["prompt"].lower() for w in words)]
        if self.sort.currentData() == "near" and self.material:
            order = {p: i for i, p in enumerate(lora_notes.nearest(self.material, self.prompt_text,
                                                                   len(self.material["pool"])))}
            shown.sort(key=lambda e: order.get(e["clean"], len(order)))
        elif self.told:
            shown.sort(key=lambda e: not e.get("told"))          # (stable: the told ones first, then by likes)
        self.shown = shown
        self.show_page(0)

    def show_page(self, page: int):
        pages = max(1, -(-len(self.shown) // PER_PAGE))
        self.page = min(max(0, page), pages - 1)
        self._gen += 1
        while True:                                # (pictures still wanted for the page before: not loaded)
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        for c in self.cards:
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        while self.box_v.count():
            self.box_v.takeAt(0)
        mode = self.show_mode()
        for i, ex in enumerate(self.shown[self.page * PER_PAGE:(self.page + 1) * PER_PAGE]):
            card = ExampleCard(ex, bool(ex.get("told")), self.box, can_use=self.use_prompt is not None,
                               can_adapt=self.adapt is not None)
            card.copied.connect(self._copy)
            card.used.connect(self._use)
            card.adapted.connect(self._adapt)
            self.cards.append(card)
            self.box_v.addWidget(card)
            item = ex["item"]
            if mode == "off":
                card.show_text("Pictures are off")
            elif not lora_pictures.allowed(item, mode):
                card.show_text(f"Not shown (rated {lora_pictures.level(item)}; Pictures: All shows it)")
            elif not lora_pictures.still_url(item) and not self._kept(item):
                card.show_text("CivitAI gave no picture for it")
            else:
                card.show_text("Loading …")
                self._start_loaders()
                self._jobs.put((self._gen, i, item))
        self.box_v.addStretch(1)
        self.scroll.verticalScrollBar().setValue(0)
        n, all_ = len(self.shown), len(self.examples)
        self.page_label.setText(f"Page {self.page + 1} of {pages}")
        self.b_prev.setEnabled(self.page > 0)
        self.b_next.setEnabled(self.page < pages - 1)
        if not all_:
            self.count.setText("No example prompts. CivitAI's gallery has none for this LoRA that tell their "
                               "prompt (videos often do not), or it was not looked up yet. “About” has what its "
                               "creator writes.")
        else:
            self.count.setText((f"{n} of {all_} match." if n != all_ else
                                f"{all_} picture{'s' if all_ != 1 else ''} and videos from the LoRA's gallery, "
                                "each with the prompt that made it.")
                               + " The prompt refiner reads the prompts without weights and LoRA tags.")

    def _start_loaders(self):
        if not self._loaders:
            for _ in range(LOADERS):
                t = threading.Thread(target=self._load_pictures, daemon=True)
                t.start()
                self._loaders.append(t)

    def _kept(self, item: dict) -> bool:
        key = item.get("id", id(item))
        if key not in self._kept_of:
            f = lora_pictures.cached(item, self.db)
            self._kept_of[key] = bool(f and f.is_file())
        return self._kept_of[key]

    def _load_pictures(self):
        """A loader thread: the stills of the page shown, one after the other."""
        while not self._closed.is_set():
            try:
                gen, i, item = self._jobs.get(timeout=0.3)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                f = self.fetch(item)
                out = (str(f), "") if f else ("", "CivitAI gave no picture for it")
            except Exception as e:  # noqa: BLE001
                out = ("", f"Could not be loaded ({e})")
            if not safe_emit(self._sig.picture, gen, i, *out):
                return

    def _picture(self, gen: int, i: int, path: str, error: str):
        if gen != self._gen or not 0 <= i < len(self.cards):
            return
        if error or not self.cards[i].show_file(path):
            self.cards[i].show_text(error or "Not a picture")

    def _pictures_changed(self, *_):
        if self.qs is not None:
            self.qs.setValue("loraPictures", self.show_mode())
        self.show_page(self.page)

    def _forget_pictures(self):
        if not confirm(self, "Delete the kept pictures", "Delete the pictures kept on disk? They are loaded from "
                             "CivitAI again when shown."):
            return
        n = lora_pictures.forget(self.db)
        self._kept_of = {}
        self.state.setText(f"{n} kept picture{'s' if n != 1 else ''} deleted.")
        self.show_page(self.page)

    def _copy(self, text: str):
        QGuiApplication.clipboard().setText(text)
        self.state.setText("Copied: " + (text if len(text) <= 90 else text[:88] + "…"))

    def done(self, result: int):                   # noqa: N802  (Qt naming)
        if not self._closed.is_set() and not self._leave_ok():
            return                                 # (the user chose to stay with the unsaved edits)
        self._closed.set()
        self._gen += 1
        super().done(result)
