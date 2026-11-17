"""The voice library window (Cast → Library…): the voices kept for every project (video_chain.voice_library).
The selected character's voice is saved into it, a saved voice is given to the selected character (copied into the
project), and saved voices are listened to, renamed and deleted."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout)

from .. import tree as T, voice_library as VL
from . import dialogs as D
from .widgets import fit_size, label, remember_geometry

if TYPE_CHECKING:
    from .main_window import MainWindow


def voice_text(v: dict) -> str:
    """One line of the list: the name, its length and what comes with it."""
    parts = [f"{float(v.get('secs') or 0):.1f} s"]
    if v.get("ref_secs"):
        parts.append(f"cloned from {float(v['ref_secs']):.1f} s")
    if v.get("text"):
        parts.append("transcript")
    if v.get("eleven"):
        parts.append(f"ElevenLabs: {v['eleven'].get('name') or v['eleven'].get('id')}")
    return f"{v['name']}   ({', '.join(parts)})"


class VoiceLibraryDialog(QDialog):
    def __init__(self, win: MainWindow, lib: VL.VoiceLibrary | None = None):
        super().__init__(win)
        self.win, self.lib = win, lib or VL.library()
        self.setWindowTitle("Voice library")
        fit_size(self, 620, 440)
        remember_geometry(self, win.qs, "voiceLibrary")
        v = QVBoxLayout(self)
        info = label("Voices kept for all your projects. A voice taken from here is copied into the project, with "
                     "its longer clone part, its transcript and its ElevenLabs voice; deleting or renaming it here "
                     f"changes no project. They are kept in {self.lib.dir}. Use a voice only with the speaker's "
                     "consent.", level="secondary")
        info.setWordWrap(True)
        v.addWidget(info)
        self.list = QListWidget()
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.itemDoubleClicked.connect(lambda _i: self.use())
        v.addWidget(self.list, 1)
        self.empty = label("No voices saved yet: select a character with a voice and click Save its voice.",
                           level="tertiary")
        v.addWidget(self.empty)
        row = QHBoxLayout()
        self.b_save = QPushButton("Save its voice…")
        self.b_save.clicked.connect(lambda: self.save())
        self.b_play = QPushButton("Listen")
        self.b_play.clicked.connect(self.play)
        self.b_rename = QPushButton("Rename…")
        self.b_rename.clicked.connect(lambda: self.rename())
        self.b_delete = QPushButton("Delete")
        self.b_delete.setToolTip("Take the voice out of the library (projects that use it keep their copy)")
        self.b_delete.clicked.connect(lambda: self.delete())
        for b in (self.b_save, self.b_play, self.b_rename, self.b_delete):
            row.addWidget(b)
        row.addStretch(1)
        self.b_use = QPushButton("Use")
        self.b_use.setDefault(True)
        self.b_use.clicked.connect(self.use)
        self.b_close = QPushButton("Close")
        self.b_close.clicked.connect(self.reject)
        row.addWidget(self.b_use)
        row.addWidget(self.b_close)
        v.addLayout(row)
        self.fill()

    # ---- the character the window works for (the one selected in the cast)
    def _character(self):
        return self.win._selected_character()

    def fill(self, select: str | None = None) -> None:
        keep = select or self.selected_id()
        self.list.clear()
        for item in self.lib.list():
            it = QListWidgetItem(voice_text(item))
            it.setData(Qt.ItemDataRole.UserRole, item["id"])
            self.list.addItem(it)
            if item["id"] == keep:
                self.list.setCurrentItem(it)
        self.list.setVisible(self.list.count() > 0)
        self.empty.setVisible(self.list.count() == 0)
        self._update_buttons()

    def selected_id(self) -> str | None:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def _update_buttons(self) -> None:
        c = self._character()
        person = c is not None and T.kind_of(c) == "person"
        idle = self.win.project is not None and not self.win.running
        picked = self.selected_id() is not None
        self.b_save.setText(f"Save the voice of @{c['name']}…" if c is not None else "Save its voice…")
        self.b_save.setEnabled(bool(c and c.get("voice")))
        self.b_save.setToolTip("Keep the selected character's voice in the library" if c and c.get("voice") else
                               "Select a character with a voice in the cast first")
        self.b_use.setText(f"Use for @{c['name']}" if person and c is not None else "Use")
        self.b_use.setEnabled(picked and person and idle)
        self.b_use.setToolTip("The selected voice becomes the voice of the selected character (it replaces its "
                              "voice)" if person else "Select a person in the cast first")
        for b in (self.b_play, self.b_rename, self.b_delete):
            b.setEnabled(picked)

    # ---- the buttons (the arguments: what a test or a caller gives instead of asking)
    def save(self, name: str | None = None) -> dict | None:
        c = self._character()
        if c is None or not c.get("voice") or self.win.project is None:
            return None
        if name is None:
            start = c["name"] + " " + time.strftime("%Y-%m-%d")
            name, ok = QInputDialog.getText(self, "Save the voice", f"Name of the voice of @{c['name']} in the "
                                            "library:", text=start)
            if not ok:
                return None
        try:
            entry = self.lib.save(self.win.project, c, name)
        except (ValueError, OSError) as e:
            D.failed(self, "Voice library", "save the voice", e)
            return None
        self.fill(select=entry["id"])
        self.win.status.setText(f"The voice of @{c['name']} is in the voice library as “{entry['name']}”.")
        return entry

    def use(self) -> bool:
        vid = self.selected_id()
        if vid is None or not self.b_use.isEnabled():
            return False
        if self.win.use_library_voice(vid, self.lib):
            self.accept()
            return True
        return False

    def play(self) -> None:
        vid = self.selected_id()
        v = self.lib.get(vid) if vid else None
        if v is not None:
            self.win.preview.open(self.lib.voice_file(v["id"]), f"Voice “{v['name']}” (voice library)")

    def rename(self, new: str | None = None) -> bool:
        vid = self.selected_id()
        v = self.lib.get(vid) if vid else None
        if v is None:
            return False
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename the voice", "New name:", text=v["name"])
            if not ok:
                return False
        try:
            self.lib.rename(v["id"], new)
        except (ValueError, OSError) as e:
            D.failed(self, "Voice library", "rename the voice", e)
            return False
        self.fill(select=v["id"])
        return True

    def delete(self, confirm: bool = True) -> bool:
        vid = self.selected_id()
        v = self.lib.get(vid) if vid else None
        if v is None:
            return False
        if confirm and not D.confirm(self, "Delete the voice", f"Delete “{v['name']}” from the voice library? "
                                     "Projects that use it keep their own copy."):
            return False
        try:
            self.lib.delete(v["id"])
        except OSError as e:
            D.failed(self, "Voice library", "delete the voice", e)
            return False
        self.fill()
        return True
