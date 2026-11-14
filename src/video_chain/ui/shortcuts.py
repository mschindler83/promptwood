"""All keyboard shortcuts in one table: windows bind from it and write their tooltips with it (`tip`), and the
Help → Keyboard shortcuts dialog shows it (`help_html`). A shortcut is changed here and nowhere else."""
from __future__ import annotations

import html
from typing import NamedTuple

from PySide6.QtGui import QKeySequence


class Shortcut(NamedTuple):
    id: str
    keys: str
    text: str
    where: str


MAIN = "main window"
SHORTCUTS: list[Shortcut] = [
    Shortcut("new_project", "Ctrl+Shift+N", "New project", MAIN),
    Shortcut("open", "Ctrl+O", "Open a project", MAIN),
    Shortcut("save", "Ctrl+S", "Save", MAIN),
    Shortcut("save_as", "Ctrl+Shift+S", "Save as", MAIN),
    Shortcut("quit", "Ctrl+Q", "Quit", MAIN),
    Shortcut("preferences", "Ctrl+,", "Preferences (video engine, API keys)", MAIN),
    Shortcut("continue", "Ctrl+N", "Continue: a new clip after the selected one", MAIN),
    Shortcut("add_video", "Ctrl+Shift+V", "Add a real video after the selected clip", MAIN),
    Shortcut("branch", "Ctrl+B", "Branch: another continuation from the same point", MAIN),
    Shortcut("variant", "Ctrl+D", "Variant: same prompt, other seed", MAIN),
    Shortcut("generate", "Ctrl+Return", "Generate the selected path", MAIN),
    Shortcut("generate_all", "Ctrl+Shift+Return", "Generate every branch", MAIN),
    Shortcut("prev_clip", "Alt+Left", "The clip before it on the path", MAIN),
    Shortcut("next_clip", "Alt+Right", "The clip after it on the path", MAIN),
    Shortcut("tab_clips", "Alt+1", "The Clips tab", MAIN),
    Shortcut("tab_cast", "Alt+2", "The Cast & references tab", MAIN),
    Shortcut("tab_sounds", "Alt+3", "The Sounds tab", MAIN),
    Shortcut("log", "Alt+L", "Log", MAIN),
    Shortcut("left_panel", "Ctrl+Alt+B", "Show or hide the left panel", MAIN),
    Shortcut("layout_edit", "Ctrl+Alt+1", "Layout: Edit (big viewer)", MAIN),
    Shortcut("layout_write", "Ctrl+Alt+2", "Layout: Write (wide prompt)", MAIN),
    Shortcut("layout_watch", "Ctrl+Alt+3", "Layout: Watch (viewer and timeline only)", MAIN),
    Shortcut("shortcuts", "F1", "Keyboard shortcuts", MAIN),
    Shortcut("graph_delete", "Delete", "Delete the selected clip (asks first)", "clip graph"),
    Shortcut("graph_fit", "F", "Fit the whole graph in the view", "clip graph"),
    Shortcut("generate_image", "Ctrl+Return", "Make or edit the image", "image dialog"),
    Shortcut("refine", "Ctrl+Return", "Refine the prompt (Send, in a chat)", "prompt refiner"),
    Shortcut("frame_back", "Left", "One frame back", "frame pickers"),
    Shortcut("frame_on", "Right", "One frame on", "frame pickers"),
    Shortcut("second_back", "Shift+Left", "One second back", "frame pickers"),
    Shortcut("second_on", "Shift+Right", "One second on", "frame pickers"),
    Shortcut("play", "Space", "Play / pause", "players"),
    Shortcut("range_start", "Ctrl+Left / Ctrl+Right", "Move the range start by one frame", "range editors"),
    Shortcut("range_end", "Ctrl+Shift+Left / Right", "Move the range end by one frame", "range editors"),
    Shortcut("rotate", "R / Shift+R", "Turn the video right / left", "video editor"),
    Shortcut("mask_paint", "B / E", "Paint / erase", "mask painter"),
    Shortcut("mask_brush", "[ / ]", "Smaller / bigger brush", "mask painter"),
    Shortcut("undo", "Ctrl+Z", "Undo", "text fields, mask painter"),
    Shortcut("redo", "Ctrl+Shift+Z", "Redo", "mask painter"),
]
_BY_ID = {s.id: s for s in SHORTCUTS}


def keys(id: str) -> str:
    """The key sequence of a shortcut as written in the table."""
    return _BY_ID[id].keys


def sequence(id: str) -> QKeySequence:
    return QKeySequence(keys(id))


def shown(id: str) -> str:
    """The key as the user's platform writes it (a tooltip's text)."""
    s = keys(id)
    q = QKeySequence(s)
    return q.toString(QKeySequence.NativeText) if " / " not in s and not q.isEmpty() else s


def tip(text: str, id: str) -> str:
    """A tooltip: text and its shortcut."""
    return f"{text} ({keys(id)})"


def help_html() -> str:
    """The table as HTML for the Keyboard shortcuts dialog, grouped by where the keys work."""
    out, last = [], None
    for s in SHORTCUTS:
        if s.where != last:
            out.append(f"<h4>{html.escape(s.where.capitalize())}</h4>")
            last = s.where
        out.append(f"<p><b>{html.escape(s.keys)}</b> &nbsp; {html.escape(s.text)}</p>")
    return "".join(out)


def shortcuts_dialog(parent=None):
    """The Help → Keyboard shortcuts window (a Close-only dialog showing the table). The caller runs exec() / show()."""
    from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

    from .dialogs import button_row
    from .widgets import fit_size
    dlg = QDialog(parent)
    dlg.setWindowTitle("Keyboard shortcuts")
    fit_size(dlg, 520, 640)
    v = QVBoxLayout(dlg)
    view = QTextBrowser()
    view.setHtml(help_html())
    v.addWidget(view)
    box, _none = button_row(dlg)
    v.addWidget(box)
    return dlg
