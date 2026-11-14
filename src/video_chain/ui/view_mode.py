"""Simple and Advanced view: which controls the window shows.

Simple hides the controls registered as advanced-only; it never changes a value, so whatever they are set to keeps
applying (an AdvancedNote says so where it matters). The registry is declarative: code registers what is
advanced-only once (a card when it is built) and calls ViewMode.show() instead of setVisible() for such a control's
own visibility; the view then decides what is really shown.

Targets: a QWidget; (QFormLayout, field widget): that form row, label and field together; (QTabWidget, page): that
tab. A form row with a layout as its field is wrapped in a widget first (row_widget): QFormLayout.setRowVisible on a
layout shows every widget in it again, also the ones the app had hidden.
"""
from __future__ import annotations

import shiboken6
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLayout, QTabWidget, QToolButton, QWidget

from .widgets import label


def row_widget(layout: QLayout) -> QWidget:
    """A layout as one widget (a form row that can be hidden and shown without touching its children)."""
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


def _alive(t) -> bool:
    return all(shiboken6.isValid(x) for x in (t if isinstance(t, tuple) else (t,)))


def _shown(t) -> bool:
    """The target's own visibility now (not hidden on purpose; a child not shown yet counts as shown)."""
    if isinstance(t, tuple):
        a, b = t
        if isinstance(a, QTabWidget):
            return a.isTabVisible(a.indexOf(b))
        return not b.isHidden() if isinstance(a, QFormLayout) else True
    return not t.isHidden()


def _set(t, on: bool) -> None:
    if isinstance(t, tuple):
        a, b = t
        if isinstance(a, QTabWidget):
            i = a.indexOf(b)
            if i >= 0:
                if not on and a.currentIndex() == i:       # (the first shown tab takes over)
                    shown = [k for k in range(a.count()) if k != i and a.isTabVisible(k)]
                    if shown:
                        a.setCurrentIndex(shown[0])
                a.setTabVisible(i, on)
        elif isinstance(a, QFormLayout):
            a.setRowVisible(b, on)
        return
    t.setVisible(on)


class ViewMode(QObject):
    """The window's view: simple (True) or advanced, and the advanced-only controls. changed(simple) after a switch;
    the registered controls are already shown / hidden then."""
    changed = Signal(bool)

    def __init__(self, simple: bool = False, parent=None):
        super().__init__(parent)
        self._simple = bool(simple)
        self._own: dict = {}            # advanced-only target -> its own visibility (what Advanced shows)
        self._revealed: set = set()     # shown in Simple on request (View → Log) until the next switch

    @property
    def simple(self) -> bool:
        return self._simple

    def set_simple(self, on: bool) -> None:
        on = bool(on)
        if on == self._simple:
            return
        self._simple = on
        self._revealed.clear()
        self.apply()
        self.changed.emit(on)

    def advanced(self, *targets) -> None:
        """Register advanced-only controls (hidden in Simple); they follow the current view at once. Their own
        visibility is taken as it is now."""
        self._prune()
        for t in targets:
            if t is None:
                continue
            if t not in self._own:
                self._own[t] = _shown(t)
            _set(t, self.visible(t))

    def is_advanced(self, t) -> bool:
        return t in self._own

    def show(self, t, on: bool = True) -> None:
        """Set a control's own visibility (use it instead of setVisible / setRowVisible for an advanced-only
        control: Simple keeps it hidden, Advanced shows it as set here). Other controls are shown / hidden plainly."""
        if t in self._own:
            self._own[t] = bool(on)
            _set(t, self.visible(t))
        else:
            _set(t, on)

    def visible(self, t) -> bool:
        """Whether the control is shown in the current view (its own visibility and the view's)."""
        if t not in self._own:
            return _shown(t)
        return self._own[t] and (not self._simple or t in self._revealed)

    def reveal(self, t) -> None:
        """Show an advanced-only control in Simple too, until the view is switched (an explicit request for it)."""
        if t in self._own:
            self._revealed.add(t)
            _set(t, self.visible(t))

    def apply(self) -> None:
        self._prune()
        for t in self._own:
            _set(t, self.visible(t))

    def _prune(self) -> None:
        """Forget the controls of deleted widgets (cards are rebuilt often)."""
        dead = [t for t in self._own if not _alive(t)]
        for t in dead:
            del self._own[t]
            self._revealed.discard(t)


class AdvancedNote(QWidget):
    """One muted line, shown in Simple only when hidden settings still apply here: "3 advanced settings are active:
    Show" (Show switches to Advanced). set_active(names): the settings, listed in its tooltip."""

    def __init__(self, view: ViewMode, parent=None):
        super().__init__(parent)
        self.view = view
        self.names: list[str] = []
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self.text = label("", level="tertiary")
        self.b_show = QToolButton()
        self.b_show.setText("Show")
        self.b_show.setAutoRaise(True)
        self.b_show.setCursor(Qt.PointingHandCursor)
        self.b_show.setToolTip("Switch to the Advanced view, which shows every setting")
        self.b_show.clicked.connect(lambda: self.view.set_simple(False))
        h.addWidget(self.text)
        h.addWidget(self.b_show)
        h.addStretch(1)
        view.changed.connect(self._update)
        self._update()

    def set_active(self, names: list[str]) -> None:
        self.names = list(names)
        n = len(self.names)
        self.text.setText(f"{n} advanced setting{'s are' if n != 1 else ' is'} active:")
        tip = ("Hidden in the Simple view, but they apply: " + ", ".join(self.names)) if n else ""
        self.text.setToolTip(tip)
        self.setToolTip(tip)
        self._update()

    def _update(self, *_):
        self.setVisible(bool(self.names) and self.view.simple)
