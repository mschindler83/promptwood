"""Small Qt helpers of the main window: drop target, the signals worker threads send, a stack that is as big as its shown page."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
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


class _Bridge(QObject):
    joined = Signal(object, object, str, str, str)  # path or None, error or None, purpose, status prefix, project
    film = Signal(object, object, str)               # the joined film or None, error or None, 'play' / 'export'
    duplicated = Signal(object, object, str)         # (scene id, title) or None, error or None, film folder
    comfy = Signal(bool)                       # ComfyUI reachable
    checked = Signal(int, list, list)          # Generate's pre-flight: its number, what is missing, empty prompts
    loras = Signal(list, bool)                 # H3 LoRAs found in ComfyUI's LoRA folders, report them
    triggers = Signal(dict)                    # the trigger word scan finished (its counts)
    trigger_progress = Signal(int, int, str)   # the scan asks CivitAI about the k-th of n LoRAs (its file name)
    copied = Signal(str, str)                  # Save as: target folder, error ('' = done)
    converted = Signal(str, object, str, str, str)  # video node id, record or None, error, token, project
    remixed = Signal(object, str, str)         # [(clip id, its new record)], error ('' = done), project
    exported = Signal(str, str, str)           # a single clip was copied out: its id, the file, error ('' = done)
    freed = Signal(bool)                       # ComfyUI's answer to "unload your models" (True: done)
    room_made = Signal(str)                     # ComfyUI's models were unloaded for a model of this app (the llama.cpp server, text to speech)
    spoken = Signal(str, object, str, str)     # clip id, rendered lines (tts.render_lines) or None, error, project
    tts_log = Signal(str)                      # a step of text to speech
    eleven_models = Signal(object, str)        # ElevenLabs' TTS models or None, error
    eleven_cloned = Signal(str, object, str, str)   # character, cloned voice (tts.clone_sample) or None, error, project


class _CurrentStack(QStackedWidget):
    """A stack whose size is that of the page shown (not of its largest hidden page)."""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()
