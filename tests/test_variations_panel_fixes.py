"""Variations panel fixes (UIL-10 … UIL-18). Made-up data, offscreen."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from video_chain import variations as V
from video_chain.ui import variations_panel as VP

app = QApplication.instance() or QApplication([])


def panel(values=("a", "b", "c", "d", "e")):
    p = VP.VariationsPanel()
    p.set_templates("{x} {y}")
    p.on.setChecked(True)
    p.variable("x").values = [V.Value(t) for t in values]
    p.variable("y").values = [V.Value("only")]
    p._rebuild()
    return p


def cards(p):
    return [p.cards_box.itemAt(i).widget() for i in range(p.cards_box.count())]


def key(widget, k):
    widget.keyPressEvent(QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier))


def test_a_typed_comma_stays_inside_one_value():
    p = panel(())
    c = cards(p)[0]
    c.add.setText("a red, shiny car")
    c.add.returnPressed.emit()
    assert [v.text for v in p.variable("x").values] == ["a red, shiny car"]


def test_destructive_variation_actions_ask(monkeypatch):
    p = panel()
    asked = []
    answer = [QMessageBox.No]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: (asked.append(a[1]), answer[0])[1])
    c = cards(p)[0]
    c.clear_values()
    assert len(p.variable("x").values) == 5 and asked
    c.remove_variable()
    p.remove_unused()
    assert p.variable("x") is not None
    answer[0] = QMessageBox.Yes
    c.clear_values()
    assert p.variable("x").values == []
    p.set_templates("{y}")
    p.remove_unused()
    assert p.variable("x") is None


def test_pills_work_from_the_keyboard():
    p = panel()
    pill = cards(p)[0].pills.items[0]
    assert pill.focusPolicy() == Qt.StrongFocus
    key(pill, Qt.Key_Space)
    assert not pill.value.on
    key(pill, Qt.Key_Space)
    assert pill.value.on
    gone = []
    pill.removed.connect(gone.append)
    key(pill, Qt.Key_Delete)
    assert gone == [pill]


def test_rename_says_why_it_is_rejected(monkeypatch):
    p = panel()
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[2]))
    c = cards(p)[0]
    c.rename_to("bad name")
    assert said and "letters" in said[-1] and p.variable("x") is not None
    c.rename_to("y")
    assert "already" in said[-1] and p.variable("x") is not None
    got = []
    p.rename.connect(lambda a, b: got.append((a, b)))
    c.rename_to("z")
    assert got == [("x", "z")] and p.variable("z") is not None


def test_a_damaged_saved_state_does_not_crash():
    p = VP.VariationsPanel()
    p.set_state({"on": True, "repeat": "x", "count": None, "positions": "no", "vars": []})
    assert p.repeat.value() == 1 and p.count.value() == 4


def test_done_and_restart_tell_once():
    p = panel()
    seen = []
    p.changed.connect(lambda: seen.append(1))
    p.done(0)
    assert len(seen) == 1
    seen.clear()
    p.restart()
    assert len(seen) == 1


def test_a_refresh_builds_the_plan_once(monkeypatch):
    p = panel()
    built = []
    real = V.Plan
    monkeypatch.setattr(V, "Plan", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    p._refresh()
    assert len(built) <= 1


def test_a_placeholder_without_values_is_underlined_in_the_themes_negative_colour(monkeypatch):
    from video_chain.ui.theme import theme
    p = panel(())
    seen = []
    monkeypatch.setattr(VP.PlaceholderHighlighter, "setFormat", lambda self, s, n, f: seen.append(f))
    VP.PlaceholderHighlighter(None, p).highlightBlock("a {x} here")
    assert seen and seen[0].underlineColor() == theme().negative


def test_the_help_does_not_promise_that_a_typed_comma_splits():
    step = VP.HELP.split("2.", 1)[1].split("<br>", 1)[0]
    assert "a comma stays in the value" in step and "list button" in step
