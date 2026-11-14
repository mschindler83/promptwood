"""Bugs found while typing ui/ (XC-23, MWC-8, MWC-11)."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import logging  # noqa: E402

from ui_helpers import app, image, new_project, run_and_wait, win  # noqa: E402,F401


def _menu_of(win, nid, monkeypatch):
    """The entries of the clip's context menu: [(text, enabled)]."""
    seen = []

    def grab(menu, _pos):
        seen.extend((a.text(), a.isEnabled()) for a in menu.actions() if a.text())
        return None
    monkeypatch.setattr(win, "_exec_menu", grab)
    win._context(nid, win.mapToGlobal(win.rect().center()))
    return dict(seen)


def test_frozen_clip_menu_disables_what_it_refuses(app, win, tmp_path, image, monkeypatch):
    """MWC-11: set_row_start refuses a frozen clip after the click; the menu entries are disabled instead."""
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert win.freeze_clip(ids[0], confirm=False)
    menu = _menu_of(win, ids[0], monkeypatch)
    for text in ("Its own start image…", "Hand-over frame…", "End image…", "End image from a video frame…",
                 "Add a real video after this…"):
        assert text in menu and not menu[text], text
    # Delete and Keep this branch are not refused for a frozen clip: they stay as they were
    assert menu["Delete (with everything after it)"]
    # ... and every entry that is enabled is one the frozen check lets through
    assert menu["Branch (another continuation)"] and menu["Continue from here"]


def test_malformed_node_event_is_ignored_with_one_warning(app, win, caplog):
    with caplog.at_level(logging.WARNING):
        win._event({"e": "node", "state": "generating"})
        win._event({"e": "node", "id": "abc"})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2 and "node event" in warnings[0].getMessage()
    assert not win.live


def test_frozen_row_start_menu_disables_the_picture_changes(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)
    assert win.freeze_clip(ids[0], confirm=False)
    from video_chain.ui import mw_nodes
    monkeypatch.setattr(mw_nodes, "row_start_of", lambda nid: ids[0])
    menu = _menu_of(win, "whatever", monkeypatch)
    assert not menu["Another picture for this row…"]
    assert not menu["Start from the project's start image instead"]
    assert menu["More start images… (the same clip for each)"]
