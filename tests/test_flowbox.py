"""FlowBox: children placed left to right in wrapping rows; a hidden child takes no place, and a change of the
children is placed once (not once per child), never touching a child that went to another box."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QPushButton  # noqa: E402

from ui_helpers import app  # noqa: E402,F401
from video_chain.ui.widgets import FlowBox  # noqa: E402


def _box(n):
    box = FlowBox(spacing=4)
    box.resize(300, 10)
    for k in range(n):
        b = QPushButton(f"b{k}")
        b.setFixedSize(60, 20)
        box.add(b)
    return box


def test_showing_or_clearing_many_children_places_them_once(app, monkeypatch):
    box = _box(300)
    calls = []
    real = FlowBox._place
    monkeypatch.setattr(FlowBox, "_place", lambda self: (calls.append(1), real(self)))
    box.show()
    app.processEvents()
    assert len(calls) <= 3                                   # (not one per child: 300 children made it n²)
    calls.clear()
    box.clear()
    app.processEvents()
    assert len(calls) <= 3 and box.items == []
    box.deleteLater()


def test_a_hidden_child_takes_no_place(app):
    box = _box(3)
    box.show()
    app.processEvents()
    second = box.items[1].geometry().topLeft()
    box.items[1].hide()
    app.processEvents()
    assert box.items[2].geometry().topLeft() == second
    box.deleteLater()


def test_a_child_moved_to_another_box_is_left_alone(app):
    old, new = _box(2), FlowBox(spacing=4)
    new.resize(300, 10)
    old.show()
    new.show()
    b = old.items[0]
    new.add(b)
    app.processEvents()
    b.move(123, 45)
    old._place()                                             # (the old box places again: not b, it is new's now)
    assert b.pos().x() == 123
    old.deleteLater()
    new.deleteLater()
