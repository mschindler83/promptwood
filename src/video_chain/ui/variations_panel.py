"""Prompt variations, the visual way (see variations.py for the rules): write {name} in the prompt, give each name
its values as pills (type + Enter, or paste a list), and the panel shows how many images that makes and exactly
which prompts the next Generate uses. No JSON to write (the Dynamic Prompter node's JSON can be imported)."""
from __future__ import annotations

import hashlib
import html
import json
import re

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
                               QSizePolicy, QSpinBox, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .. import variations as V
from .results_gallery import FlowBox
from .theme import theme
from .widgets import guard_wheel, icon_button, label

COLORS = ("#e8a33d", "#5aa9e6", "#7bc96f", "#e46f8e", "#a98be0", "#4fc3b9", "#e07b53", "#c9c44f")
MAX_ALL = 200                    # "every combination" above this asks first
POSITIONS_KEPT = 60
HELP = """<b>Prompt variations</b> make many prompts from one.<br><br>
1. Write <b>{name}</b> in the prompt where something should vary, e.g. <i>a {animal} in the {place}</i>.<br>
2. Give each name its values below: type one and press Enter, or paste a list (one per line, or comma-separated).<br>
3. Generate makes one image per combination: 3 animals × 2 places = 6 images.<br><br>
<b>Order</b>: <i>In order</i> goes through every combination and continues there next time; <i>Shuffled</i> takes
every combination once in a random order; <i>Random</i> picks freely (weights count: a value ×3 comes three times as
often).<br>
<b>Repeat each</b> makes every combination several times with other seeds.<br>
<b>(nothing)</b> is an empty value: the placeholder disappears. For text that should go with it, write an
optional part: <i>{?style|, in {style} style}</i> is left out when <i>style</i> is empty.<br>
Placeholders work in the negative prompt too. Click a value to switch it off, double-click to edit it,
right-click for its weight."""


def color_of(name: str, names: list[str]) -> QColor:
    i = names.index(name) if name in names else sum(map(ord, name))
    return QColor(COLORS[i % len(COLORS)])


class PlaceholderHighlighter(QSyntaxHighlighter):
    """Colours {name} / {?name|…} in a prompt by variable; red wavy underline for names without values."""

    def __init__(self, doc, panel: "VariationsPanel"):
        super().__init__(doc)
        self.panel = panel

    def highlightBlock(self, text: str):         # noqa: N802  (Qt naming)
        if not self.panel.enabled():
            return
        names = self.panel.names()
        for m in re.finditer(r"\{\??(" + V.NAME + r")(\|[^{}]*(?:\{" + V.NAME + r"\}[^{}]*)*)?\}", text):
            name = m.group(1)
            f = QTextCharFormat()
            var = self.panel.variable(name)
            if var is None or not var.active():
                f.setUnderlineStyle(QTextCharFormat.WaveUnderline)
                f.setUnderlineColor(QColor("#e05555"))
                f.setToolTip(f"{{{name}}} has no values yet: add them in Variations")
            else:
                c = color_of(name, names)
                bg = QColor(c)
                bg.setAlpha(55)
                f.setBackground(bg)
                f.setForeground(c.lighter(125))
            f.setFontWeight(600)
            self.setFormat(m.start(), m.end() - m.start(), f)


class Pill(QFrame):
    """One value: click switches it off / on, double-click edits, right-click: weight, edit, delete."""
    changed = Signal()
    removed = Signal(object)

    def __init__(self, value: V.Value, color: QColor, show_weight: bool, parent=None):
        super().__init__(parent)
        self.value, self.color = value, color
        self.setObjectName("pill")
        self.setFrameShape(QFrame.NoFrame)
        self.setCursor(Qt.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 1, 2, 1)
        h.setSpacing(2)
        self.text = QLabel()
        h.addWidget(self.text)
        self.weight = QLabel()
        h.addWidget(self.weight)
        rm = QToolButton()
        rm.setText("×")
        rm.setAutoRaise(True)
        rm.setFixedSize(16, 16)
        rm.setToolTip("Remove this value")
        rm.clicked.connect(lambda: self.removed.emit(self))
        h.addWidget(rm)
        self.show_weight = show_weight
        self.refresh()

    def refresh(self):
        v, c = self.value, self.color
        shown = html.escape(v.text) if v.text.strip() else "<i>(nothing)</i>"
        self.text.setText(f"<s>{shown}</s>" if not v.on else shown)
        self.weight.setText(f"×{v.weight}")
        self.weight.setVisible(self.show_weight or v.weight != 1)
        fg = theme().textPrimary if v.on else theme().textTertiary
        self.setStyleSheet(f"QLabel {{ color: {fg.name()}; background: transparent; border: none; }} "
                           f"QToolButton {{ color: {fg.name()}; border: none; background: transparent; }}")
        self.update()
        self.setToolTip(("Click: switch off (kept, but not used)" if v.on else "Switched off: click to use it again")
                        + "\nDouble-click: edit · Right-click: weight (for Random), edit, remove")

    def paintEvent(self, _e):                    # noqa: N802  (the rounded pill, painted: the theme's QFrame style won)
        from PySide6.QtGui import QPainter, QPen
        c, on = self.color, self.value.on
        bg = QColor(c)
        bg.setAlpha(60 if on else 16)
        border = QColor(c)
        border.setAlpha(255 if on else 90)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(border, 1, Qt.SolidLine if on else Qt.DashLine))
        p.setBrush(bg)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.end()

    def mousePressEvent(self, e):                # noqa: N802
        if e.button() == Qt.LeftButton:
            self.value.on = not self.value.on
            self.refresh()
            self.changed.emit()
        elif e.button() == Qt.RightButton:
            self._menu(e.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, e):          # noqa: N802
        if e.button() == Qt.LeftButton:
            self.value.on = not self.value.on    # (the first click of the double-click switched it)
            self.refresh()
            self.changed.emit()                  # (also when the edit is cancelled)
            self.edit()

    def edit(self):
        text, ok = QInputDialog.getText(self, "Edit value", "Value (empty: nothing):", text=self.value.text)
        if ok:
            self.value.text = " ".join(text.split())
            self.refresh()
            self.changed.emit()

    def _menu(self, pos):
        m = QMenu(self)
        ws = {}
        for w in (1, 2, 3, 5, 10):
            a = m.addAction(f"Weight ×{w}" + ("  (as often as the others)" if w == 1 else ""))
            a.setCheckable(True)
            a.setChecked(self.value.weight == w)
            ws[a] = w
        m.addSeparator()
        a_edit = m.addAction("Edit…")
        a_toggle = m.addAction("Switch off" if self.value.on else "Switch on")
        a_rm = m.addAction("Remove")
        chosen = m.exec(pos)
        if chosen in ws:
            self.value.weight = ws[chosen]
            self.refresh()
            self.changed.emit()
        elif chosen is a_edit:
            self.edit()
        elif chosen is a_toggle:
            self.value.on = not self.value.on
            self.refresh()
            self.changed.emit()
        elif chosen is a_rm:
            self.removed.emit(self)


class VariableCard(QFrame):
    """A placeholder and its values."""
    changed = Signal()
    typing = Signal(str)
    insert = Signal(str)
    remove = Signal(str)

    def __init__(self, var: V.Variable, color: QColor, used: bool, show_weight: bool, pick_file=None, parent=None):
        super().__init__(parent)
        self.var, self.color, self.pick_file = var, color, pick_file
        self.setObjectName("varCard")
        self.setStyleSheet(f"QFrame#varCard {{ border: 1px solid {theme().lineSubtle.name()}; border-left: 3px solid "
                           f"{color.name()}; border-radius: 6px; background: {theme().bgRaised.name()}; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 5, 6, 6)
        v.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(4)
        name = QLabel(f"<b style='color:{color.name()}'>{{{html.escape(var.name)}}}</b>")
        head.addWidget(name)
        n_on, n = len(var.active()), len(var.values)
        self.count = label(f"{n_on} value{'s' if n_on != 1 else ''}" + (f" ({n - n_on} off)" if n != n_on else "")
                           + ("" if used else " · not in the prompt"), level="tertiary")
        head.addWidget(self.count, 1)
        b_ins = icon_button("plus", f"Write {{{var.name}}} into the prompt at the cursor")
        b_ins.clicked.connect(lambda: self.insert.emit(var.name))
        b_list = icon_button("list", "Add a list of values (paste: one per line, or comma-separated)")
        b_list.clicked.connect(self.paste_list)
        b_more = icon_button("more", "More: wildcard file, (nothing), sort, rename, clear, remove")
        b_more.clicked.connect(lambda: self._more(b_more))
        for b in (b_ins, b_list, b_more):
            head.addWidget(b)
        v.addLayout(head)
        self.pills = FlowBox(spacing=4)
        for val in var.values:
            p = Pill(val, color, show_weight)
            p.changed.connect(self._changed)
            p.removed.connect(self._remove_pill)
            self.pills.add(p)
        v.addWidget(self.pills)
        self.add = QLineEdit()
        self.add.setPlaceholderText("add a value … (Enter; paste several lines at once)")
        self.add.returnPressed.connect(self._add_typed)
        v.addWidget(self.add)
        if not used:
            self.setToolTip(f"{{{var.name}}} is not in the prompt: its values are kept, but not used")
            for w in (name, self.pills):
                w.setEnabled(False)

    def _changed(self):
        self.changed.emit()

    def _remove_pill(self, pill: Pill):
        if pill.value in self.var.values:
            self.var.values.remove(pill.value)
        self.changed.emit()

    def add_values(self, texts: list[str]):
        have = {v.text for v in self.var.values}
        for t in texts:
            t = " ".join(t.split())
            if t not in have:
                self.var.values.append(V.Value(t))
                have.add(t)
        self.changed.emit()

    def _add_typed(self):
        text = self.add.text()
        if not text.strip():
            return
        self.add.clear()
        self.typing.emit(self.var.name)          # (the panel puts the cursor back here after it rebuilt)
        self.add_values(V.split_values(text) if ("\n" in text or "," in text) else [text])

    def paste_list(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Values of {{{self.var.name}}}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(label("One value per line (or one line, comma-separated). Lines starting with # are skipped.",
                            level="tertiary"))
        box = QPlainTextEdit()
        box.setMinimumSize(420, 260)
        lay.addWidget(box)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() == QDialog.Accepted:
            text = "\n".join(x for x in box.toPlainText().splitlines() if not x.strip().startswith("#"))
            self.add_values(V.split_values(text))
        dlg.deleteLater()

    def _more(self, button):
        m = QMenu(self)
        a_file = m.addAction("Add values from a wildcard file (.txt, one per line)…")
        a_none = m.addAction("Add (nothing): sometimes leave it out")
        m.addSeparator()
        a_sort = m.addAction("Sort A–Z")
        a_on = m.addAction("Switch all values on")
        a_rename = m.addAction("Rename…")
        a_clear = m.addAction("Remove all values")
        a_rm = m.addAction(f"Remove {{{self.var.name}}}")
        chosen = m.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if chosen is a_file:
            path = self.pick_file() if self.pick_file else None
            if path:
                try:
                    from pathlib import Path
                    self.add_values(V.wildcard_values(Path(path).read_text(encoding="utf-8", errors="replace")))
                except OSError as e:
                    QMessageBox.warning(self, "Wildcard file", f"Could not read it: {e}")
        elif chosen is a_none:
            if not any(not v.text.strip() for v in self.var.values):
                self.var.values.append(V.Value(""))
                self.changed.emit()
        elif chosen is a_sort:
            self.var.values.sort(key=lambda v: (not v.text.strip(), v.text.lower()))
            self.changed.emit()
        elif chosen is a_on:
            for v in self.var.values:
                v.on = True
            self.changed.emit()
        elif chosen is a_rename:
            new, ok = QInputDialog.getText(self, "Rename", "New name (letters, digits, _ and -):", text=self.var.name)
            if ok and re.fullmatch(V.NAME, new.strip()) and new.strip() != self.var.name:
                self.remove.emit(f"rename:{self.var.name}:{new.strip()}")
        elif chosen is a_clear:
            self.var.values.clear()
            self.changed.emit()
        elif chosen is a_rm:
            self.remove.emit(self.var.name)


class VariationsPanel(QFrame):
    """The variables of the prompt, the order and amount, and the preview of the next Generate."""
    changed = Signal()
    insert = Signal(str)                      # write {name} into the prompt
    rename = Signal(str, str)                 # old, new: the window renames it in the prompt

    def __init__(self, parent=None, pick_file=None):
        super().__init__(parent)
        self.pick_file = pick_file
        self.vars: list[V.Variable] = []
        self.prompt, self.negative = "", ""
        self.positions: dict[str, int] = {}
        self.setObjectName("variations")
        t = theme()
        self.setStyleSheet(f"QFrame#variations {{ border: 1px solid {t.lineSubtle.name()}; border-radius: 8px; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        self.on = QCheckBox("Variations")
        self.on.setStyleSheet("font-weight: 600;")
        self.on.setToolTip("Vary parts of the prompt: {name} placeholders with lists of values")
        self.on.toggled.connect(lambda _on: self._rebuild())
        head.addWidget(self.on)
        self.summary = label("", level="tertiary")
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        head.addWidget(self.summary, 1)
        b_new = icon_button("plus", "New variable: a name for {name}, written into the prompt at the cursor")
        b_new.clicked.connect(lambda: self.new_variable())
        b_help = icon_button("info", "How variations work")
        b_help.clicked.connect(lambda: QMessageBox.information(self, "Prompt variations", HELP))
        b_menu = icon_button("more", "Import / export (the Dynamic Prompter node's JSON), remove all")
        b_menu.clicked.connect(lambda: self._menu(b_menu))
        for b in (b_new, b_help, b_menu):
            head.addWidget(b)
        self.b_new = b_new
        v.addLayout(head)
        self.body = QWidget()
        bv = QVBoxLayout(self.body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(6)
        self.hint = label("", level="tertiary")
        self.hint.setWordWrap(True)
        bv.addWidget(self.hint)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(6)
        bv.addLayout(self.cards_box)
        orow = QHBoxLayout()
        orow.setSpacing(4)
        orow.addWidget(label("Order", level="tertiary"))
        self.modes = QButtonGroup(self)
        self.mode_buttons = {}
        for key, text, tip in (("order", "In order", "Every combination in turn; the next Generate continues there"),
                               ("shuffle", "Shuffled", "Every combination once, in a random order"),
                               ("random", "Random", "Free random picks (values with a higher weight come more often)")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setAutoDefault(False)
            self.modes.addButton(b)
            self.mode_buttons[key] = b
            orow.addWidget(b)
        self.mode_buttons["order"].setChecked(True)
        self.modes.buttonClicked.connect(lambda _b: self._rebuild())
        orow.addStretch(1)
        orow.addWidget(label("repeat each", level="tertiary"))
        self.repeat = QSpinBox()
        self.repeat.setRange(1, 50)
        self.repeat.setPrefix("×")
        self.repeat.setToolTip("Make every combination this many times (each with its own seed)")
        self.repeat.valueChanged.connect(lambda *_: self._refresh())
        orow.addWidget(self.repeat)
        bv.addLayout(orow)
        arow = QHBoxLayout()
        arow.setSpacing(4)
        arow.addWidget(label("Make", level="tertiary"))
        self.all = QRadioButton("every combination")
        self.next = QRadioButton("the next")
        self.all.setChecked(True)
        self.count = QSpinBox()
        self.count.setRange(1, 1000)
        self.count.setValue(4)
        self.count.setSuffix(" images")
        self.count.setToolTip("How many images the next Generate makes; In order / Shuffled continue where the last "
                              "Generate stopped")
        for w in (self.all, self.next):
            w.toggled.connect(lambda *_: self._refresh())
        self.count.valueChanged.connect(lambda *_: (self.next.setChecked(True), self._refresh()))
        arow.addWidget(self.all)
        arow.addWidget(self.next)
        arow.addWidget(self.count)
        arow.addStretch(1)
        bv.addLayout(arow)
        prow = QHBoxLayout()
        self.position = label("", level="tertiary")
        prow.addWidget(self.position, 1)
        self.b_restart = QPushButton("Start over")
        self.b_restart.setAutoDefault(False)
        self.b_restart.setToolTip("Begin again at the first combination")
        self.b_restart.clicked.connect(self.restart)
        prow.addWidget(self.b_restart)
        bv.addLayout(prow)
        self.preview = QTextBrowser()
        self.preview.setOpenLinks(False)
        self.preview.setMinimumHeight(90)
        self.preview.setMaximumHeight(170)
        self.preview.setToolTip("The prompts the next Generate makes, in this order (values in their colour)")
        bv.addWidget(self.preview)
        v.addWidget(self.body)
        guard_wheel(self)
        self._rebuild()

    # ---- state
    def enabled(self) -> bool:
        return self.on.isChecked()

    def mode(self) -> str:
        return next(k for k, b in self.mode_buttons.items() if b.isChecked())

    def names(self) -> list[str]:
        return [v.name for v in self.vars]

    def variable(self, name: str) -> V.Variable | None:
        return next((v for v in self.vars if v.name == name), None)

    def state(self) -> dict:
        return {"on": self.enabled(), "vars": V.to_state(self.vars), "mode": self.mode(), "repeat": self.repeat.value(),
                "amount": "all" if self.all.isChecked() else "next", "count": self.count.value(),
                "positions": self.positions}

    def set_state(self, st: dict | None, keep_positions: bool = True):
        st = st or {}
        self.vars = V.from_state(st.get("vars"))
        for w in (self.on, self.repeat, self.count, self.all, self.next):
            w.blockSignals(True)
        self.on.setChecked(bool(st.get("on")))
        self.mode_buttons.get(st.get("mode"), self.mode_buttons["order"]).setChecked(True)
        self.repeat.setValue(int(st.get("repeat") or 1))
        self.count.setValue(int(st.get("count") or 4))
        (self.next if st.get("amount") == "next" else self.all).setChecked(True)
        for w in (self.on, self.repeat, self.count, self.all, self.next):
            w.blockSignals(False)
        if not keep_positions or isinstance(st.get("positions"), dict):
            self.positions = dict(st.get("positions") or {}) if not keep_positions or not self.positions \
                else {**st.get("positions", {}), **self.positions}
        self._rebuild()

    def set_templates(self, prompt: str, negative: str = ""):
        """The prompt (and negative prompt) changed: cards for new placeholders, the counts, the preview."""
        if (prompt, negative) == (self.prompt, self.negative):
            return
        old = set(V.names_in(self.prompt, self.negative))
        self.prompt, self.negative = prompt or "", negative or ""
        new = V.names_in(self.prompt, self.negative)
        added = [n for n in new if self.variable(n) is None]
        for n in added:
            self.vars.append(V.Variable(n))
        if added or set(new) != old:
            self._rebuild()
        else:
            self._refresh()

    def active(self) -> bool:
        """Variations change the next Generate (on, and the prompt has placeholders with values)."""
        return self.enabled() and bool(V.Plan(self.vars, self.prompt, self.negative).vars)

    def _key(self) -> str:
        plan = V.Plan(self.vars, self.prompt, self.negative)
        raw = json.dumps([self.prompt, self.negative, [(v.name, [x.text for x in v.active()]) for v in plan.vars],
                          self.mode(), self.repeat.value()], ensure_ascii=False)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def position_key(self) -> str:
        return self._key() if self.mode() != "random" else "random"

    def start(self) -> int:
        return int(self.positions.get(self.position_key(), 0))

    def amount(self) -> int:
        """How many images the next Generate makes (without building them)."""
        if not self.active():
            return 0
        n = V.Plan(self.vars, self.prompt, self.negative).total * self.repeat.value() if self.all.isChecked() \
            else self.count.value()
        return min(n, V.MAX_IMAGES)

    def jobs(self, seed: int = 0) -> list[V.Job]:
        """The prompts of the next Generate (empty when variations are not active)."""
        if not self.active():
            return []
        return V.plan_jobs(self.vars, self.prompt, self.negative, self.mode(), self.repeat.value(),
                           None if self.all.isChecked() else self.count.value(), self.start(), seed)

    def done(self, step: int, key: str | None = None):
        """An image of position `step` was made (planned under `key`): the next Generate continues after it."""
        key = key or self.position_key()
        pos = max(int(self.positions.pop(key, 0)), int(step) + 1)
        self.positions[key] = pos                    # (re-inserted: the most recently used are kept)
        if len(self.positions) > POSITIONS_KEPT:
            for k in list(self.positions)[:len(self.positions) - POSITIONS_KEPT]:
                self.positions.pop(k, None)
        self._refresh()
        self.changed.emit()

    def restart(self):
        self.positions.pop(self.position_key(), None)
        self._refresh()
        self.changed.emit()

    def too_many(self) -> int:
        """The number of images 'every combination' would make when that is more than MAX_ALL (else 0)."""
        if not self.active() or not self.all.isChecked():
            return 0
        n = V.Plan(self.vars, self.prompt, self.negative).total * self.repeat.value()
        return n if n > MAX_ALL else 0

    # ---- editing
    def new_variable(self, name: str | None = None) -> str | None:
        if name is None:
            name, ok = QInputDialog.getText(self, "New variable", "Name (letters, digits, _ and -), e.g. animal:")
            if not ok:
                return None
        name = (name or "").strip().strip("{}")
        if not re.fullmatch(V.NAME, name):
            if name:
                QMessageBox.information(self, "New variable", "Use letters, digits, _ and - (no spaces), e.g. hair_color.")
            return None
        if self.variable(name) is None:
            self.vars.append(V.Variable(name))
        self.on.setChecked(True)
        self.insert.emit(name)
        self._rebuild()
        return name

    def _remove(self, what: str):
        if what.startswith("rename:"):
            _r, old, new = what.split(":", 2)
            var = self.variable(old)
            if var is not None and self.variable(new) is None:
                var.name = new
                self.rename.emit(old, new)
        else:
            self.vars = [v for v in self.vars if v.name != what]
        self._rebuild()

    def _menu(self, button):
        m = QMenu(self)
        a_imp = m.addAction("Import values (Dynamic Prompter JSON)…")
        a_exp = m.addAction("Copy values as Dynamic Prompter JSON")
        m.addSeparator()
        a_unused = m.addAction("Remove variables that are not in the prompt")
        a_all = m.addAction("Remove all variables")
        chosen = m.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if chosen is a_imp:
            self.import_json()
        elif chosen is a_exp:
            from PySide6.QtGui import QGuiApplication
            QGuiApplication.clipboard().setText(V.to_json(self.vars))
        elif chosen is a_unused:
            used = set(V.names_in(self.prompt, self.negative))
            self.vars = [v for v in self.vars if v.name in used]
            self._rebuild()
        elif chosen is a_all and QMessageBox.question(self, "Variations", "Remove all variables and their values?") \
                == QMessageBox.Yes:
            self.vars = []
            self._rebuild()

    def import_json(self, text: str | None = None) -> bool:
        if text is None:
            text, ok = QInputDialog.getMultiLineText(self, "Import values", "The Dynamic Prompter node's values JSON, "
                                                     "e.g. {\"p1\": [\"cat\", \"dog\"]}:")
            if not ok:
                return False
        try:
            new = V.from_json(text)
        except ValueError as e:
            QMessageBox.warning(self, "Import values", f"That is not valid: {e}")
            return False
        for nv in new:
            cur = self.variable(nv.name)
            if cur is None:
                self.vars.append(nv)
            else:
                have = {v.text for v in cur.values}
                cur.values += [v for v in nv.values if v.text not in have]
        self.on.setChecked(True)
        self._rebuild()
        return True

    # ---- showing
    def _rebuild(self):
        while self.cards_box.count():
            it = self.cards_box.takeAt(0)
            if it.widget() is not None:
                it.widget().hide()
                it.widget().setParent(None)
                it.widget().deleteLater()
        on = self.enabled()
        self.body.setVisible(on)
        self.b_new.setVisible(True)
        used = V.names_in(self.prompt, self.negative)
        order = [n for n in used if self.variable(n) is not None] + [v.name for v in self.vars if v.name not in used]
        self.vars.sort(key=lambda v: order.index(v.name) if v.name in order else 999)
        if on:
            names = self.names()
            for var in self.vars:
                card = VariableCard(var, color_of(var.name, names), var.name in used, self.mode() == "random",
                                    self.pick_file)
                card.changed.connect(self._rebuild_later)
                card.typing.connect(self._remember_typing)
                card.insert.connect(self.insert)
                card.remove.connect(self._remove)
                self.cards_box.addWidget(card)
        focus = getattr(self, "_typing_in", None)
        if focus:
            self._typing_in = None
            for i in range(self.cards_box.count()):
                card = self.cards_box.itemAt(i).widget()
                if card is not None and card.var.name == focus:
                    card.add.setFocus()
        self._refresh()

    def _remember_typing(self, name: str):
        self._typing_in = name

    def _rebuild_later(self):
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._rebuild)      # (the pill that asked is still on the stack)

    def _refresh(self):
        plan = V.Plan(self.vars, self.prompt, self.negative)
        on = self.enabled()
        if not on:
            self.summary.setText("off: one prompt" if not plan.vars else
                                 f"off (the prompt has {len(V.names_in(self.prompt, self.negative))} placeholders)")
        elif not V.names_in(self.prompt, self.negative):
            self.summary.setText("write {name} in the prompt")
        else:
            factors = " × ".join(str(len(v.active())) for v in plan.vars) or "1"
            rep = self.repeat.value()
            n = plan.total * rep
            self.summary.setText(f"{factors} = {plan.total} combination{'s' if plan.total != 1 else ''}"
                                 + (f" × {rep} = {n} images" if rep > 1 else ""))
        self.all.setText(f"every combination ({plan.total * self.repeat.value()})")
        if not V.names_in(self.prompt, self.negative):
            self.hint.setText("Write {name} in the prompt where something should vary, e.g. “a {animal} in the "
                              "{place}”, or press + to add one. Each name gets its values here.")
        elif plan.missing:
            self.hint.setText("Without values yet: " + ", ".join("{" + n + "}" for n in plan.missing)
                              + " (they stay as written until they get values).")
        else:
            self.hint.setText("")
        self.hint.setVisible(bool(self.hint.text()))
        rand = self.mode() == "random"
        start = self.start()
        if rand:
            self.position.setText("Random: every Generate picks anew")
        else:
            total = plan.total * self.repeat.value()
            self.position.setText(f"Next: #{start % max(1, total) + 1} of {total}"
                                  + (f" (round {start // max(1, total) + 1})" if start >= total else ""))
        self.b_restart.setEnabled(start > 0 and not rand)
        self.count.setEnabled(True)
        self._preview(plan)
        self.changed.emit()

    def _preview(self, plan):
        if not self.enabled() or not plan.vars:
            self.preview.setHtml("")
            self.preview.setVisible(False)
            return
        self.preview.setVisible(True)
        n = self.amount()
        jobs = V.plan_jobs(self.vars, self.prompt, self.negative, self.mode(), self.repeat.value(),
                           min(n, 100), self.start(), 0)                    # (only what is shown is built)
        names = self.names()
        rows = []
        for k, j in enumerate(jobs[:100]):
            vals = " ".join(f"<span style='color:{color_of(n, names).name()}'>{html.escape(t) if t.strip() else '∅'}"
                            f"</span>" for n, t in j.combo.items())
            short = j.prompt if len(j.prompt) <= 150 else j.prompt[:148] + "…"
            rows.append(f"<p style='margin:0 0 4px 0'><span style='color:gray'>{k + 1}.</span> {vals}"
                        f"<br><span style='color:gray'>{html.escape(short)}</span></p>")
        more = f"<p style='color:gray'>… and {n - 100} more</p>" if n > 100 else ""
        big = self.mode() == "shuffle" and plan.total > V.SHUFFLE_LIMIT
        head = (f"<p style='margin:0 0 6px 0'><b>The next Generate makes {n} image{'s' if n != 1 else ''}</b>"
                + (" (random picks)" if self.mode() == "random" else
                   f" (more than {V.SHUFFLE_LIMIT:,} combinations cannot be shuffled: random picks)" if big else "")
                + ":</p>")
        self.preview.setHtml(head + "".join(rows) + more)

    def sizeHint(self):                          # noqa: N802
        return QSize(400, super().sizeHint().height())
