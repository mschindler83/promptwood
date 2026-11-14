"""What the image dialog's parts share: its limits and hints, and the small Picture 2 / 3 slot."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from .imageio import upright
from .widgets import FrameTile, icon_button, label

MAX_RESULTS = 20                           # results per Generate (made one after another, Stop ends it)
SEED_MAX = 2**31 - 1                       # (the seeds ComfyUI's image samplers take here)
SIDE_PANEL_WIDTH = 430                     # the left column (source, prompt, options)
RESULT_ICON = QSize(150, 100)              # a result in the strip: its picture,
RESULT_GRID = QSize(158, 108)              # its cell,
RESULT_STRIP_HEIGHT = 130                  # and the strip's height
FAMILY_RETRIES, FAMILY_RETRY_MS = 36, 5000   # asking for the models while ComfyUI starts: up to 3 minutes
# the model file a quality needs (key in the family's files); a quality whose file is not installed is not offered
QUALITY_FILE = {"qwen21": {"fast": "turbo", "best": "base"}, "chroma": {"fast": "flash", "best": "hd"}}
FOLDER_ITEM, FOLDER_PREFIX = "\x00choose-folder", "\x00folder:"   # the source list's folder entries
ASK_ABOVE = 10                             # a folder Generate of more images than this asks first
PROMPT_HINT = ("Describe the change, e.g. “she smiles and looks into the camera”, “night, lit by a street lamp”, "
               "“replace the red jacket with a black leather jacket”. Say what must stay: “keep her face, pose and "
               "the background”.")
CREATE_HINT = ("Describe the image: who or what, doing what, where; light, mood, camera and style, e.g. “a woman "
               "in a red coat on a rainy Paris street at night, neon reflections, 35 mm photo”. With reference "
               "pictures, say what to take from them.")


class PictureSlot(QWidget):
    """A small optional picture (Picture 2 / 3) with choose / remove."""
    changed = Signal()

    def __init__(self, title: str, chooser: Callable[[str], str | None], parent=None):
        super().__init__(parent)
        self.path: str | None = None
        self.chooser = chooser
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.tile = FrameTile(120, 90)
        self.tile.clear_frame(title)
        self.tile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tile.setToolTip(f"{title}: click to choose a picture (e.g. a face or a piece of clothing)")
        self.tile.clicked.connect(lambda _t: self.choose())
        self.title = title
        row = QHBoxLayout()
        row.addWidget(label(title, level="tertiary"))
        row.addStretch(1)
        self.b_clear = icon_button("x", f"Remove {title}")
        self.b_clear.clicked.connect(lambda: self.set(None))
        self.b_clear.setEnabled(False)
        row.addWidget(self.b_clear)
        v.addWidget(self.tile)
        v.addLayout(row)

    def choose(self):
        f = self.chooser(f"{self.title} (optional)")
        if f:
            self.set(f)

    def set(self, path: str | None):
        self.path = path
        if path:
            self.tile.set_image(upright(path))
            self.tile.setToolTip(path)
        else:
            self.tile.clear_frame(self.title)
        self.b_clear.setEnabled(bool(path))
        self.changed.emit()
