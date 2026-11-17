"""Small Qt helpers of the main window: drop target, the signals worker threads send, a stack as big as its page."""
from __future__ import annotations

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QComboBox, QStackedWidget, QWidget

from .ui_logic import frames_text


def select_frames(combo: QComboBox, frames: int | None) -> None:
    """Select a frame count; a saved one that is no longer offered is shown as it is, not replaced."""
    if frames is None:
        combo.setCurrentIndex(0)
        return
    i = combo.findData(int(frames))
    if i < 0:
        combo.addItem(frames_text(int(frames)) + " – no longer offered", int(frames))
        i = combo.count() - 1
    combo.setCurrentIndex(i)


class FileDrop(QObject):
    """Makes a widget take files dropped on it: on_files(list of local paths). accept(): whether it takes them now.
    (For a list or other scroll area pass its viewport(): that is where the drop arrives.)"""

    def __init__(self, widget: QWidget, on_files, accept=lambda: True):
        super().__init__(widget)
        self.on_files, self.accept = on_files, accept
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    @staticmethod
    def _files(e) -> list[str]:
        md = e.mimeData()
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()] if md.hasUrls() else []

    def eventFilter(self, obj, e):
        kind = e.type()
        if kind in (e.Type.DragEnter, e.Type.DragMove):
            if self._files(e) and self.accept():
                e.acceptProposedAction()
                return True
        elif kind == e.Type.Drop:
            files = self._files(e)
            if files and self.accept():
                e.acceptProposedAction()
                self.on_files(files)
                return True
        return False


class _CurrentStack(QStackedWidget):
    """A stack whose size is that of the page shown (not of its largest hidden page)."""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()
