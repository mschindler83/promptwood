"""Main window (project, film, cast, sound, tracks): state of a closed / opened project, save errors, busy guards."""
import json
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from test_ui import (app, crop_accepted, image, no_runpod_key, silent_player, win)  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402


def opened(win, tmp_path, image, name="p"):
    assert win.create_project(tmp_path / name, str(image))
    return win


def test_failures_cleared_on_close_and_open(win, tmp_path, image):
    opened(win, tmp_path, image)
    win._failures["n1"] = {"text": "x", "tip": "x", "error": True, "action": "log", "button": "Details"}
    win.close_project()
    assert win._failures == {}
    win._failures["n1"] = {"text": "x"}
    assert win.open_project(tmp_path / "p")
    assert win._failures == {}


def test_close_project_clears_lists_and_canvas(win, tmp_path, image):
    opened(win, tmp_path, image)
    T.add_character(win.project, win.tree, "Anna") if hasattr(T, "add_character") else None
    win.cast_list.addItem("Anna")
    win.sounds.addItem("s")
    win.track_list.addItem("t")
    win._canvas = (3840, 2160)
    win.close_project()
    assert win.cast_list.count() == 0 and win.sounds.count() == 0 and win.track_list.count() == 0
    assert win._canvas is None


def test_failed_open_keeps_waits(win, tmp_path, image):
    opened(win, tmp_path, image)
    win._waiting_comfy = "waiting"
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / T.TREE_FILE).write_text("{ not json")
    assert win.open_project(tmp_path / "bad", quiet=True) is False
    assert win._waiting_comfy == "waiting"


def test_open_missing_folder_does_not_create_it(win, tmp_path, image):
    opened(win, tmp_path, image)
    assert win.open_project(tmp_path / "nowhere", quiet=True) is False
    assert not (tmp_path / "nowhere").exists()


def test_open_project_with_broken_option_values(win, tmp_path, image):
    opened(win, tmp_path, image)
    p = win.project
    win.close_project()
    data = json.loads((p / T.TREE_FILE).read_text())
    data["options"].update({"frames": None, "steps": "28 x", "turbo": {"on": True, "strength": "high"}})
    (p / T.TREE_FILE).write_text(json.dumps(data))
    assert win.open_project(p, quiet=True)
    assert win.project is not None and win.steps.value() == 28 and win.tree.options["frames"] == 73


def test_save_error_is_not_overwritten(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(T, "save", boom)
    assert win.clear_image()
    assert "Could not save" in win.status.text()


def test_remove_character_oserror_is_reported(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image)
    win.new_character("Anna") if hasattr(win, "new_character") else None
    c = {"name": "Anna", "images": [], "desc": "", "lead": False}
    win.tree.options["cast"] = [c]
    win._show_references(select="Anna")

    def boom(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(T, "remove_character", boom)
    win._remove_character()                        # no exception in the slot
    assert "read-only" in win.status.text()


def test_recent_menu_lists_projects_once(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image, "a")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "chain.json").write_text("{}")
    (tmp_path / "none").mkdir()
    calls = []
    monkeypatch.setattr(win, "recent_projects", lambda: calls.append(1) or [str(win.project), str(tmp_path / "old"),
                                                                           str(tmp_path / "none")])
    win._fill_recent()
    assert len(calls) <= 2
    assert [a.text().split("  ")[0] for a in win.recent_menu.actions()] == ["a", "old"]


def test_eleven_models_without_current(win):
    win.eleven_model.clear()
    win._fill_eleven_models([{"name": "M", "id": "m1"}], "")
    assert win.eleven_model.count() == 1


def test_checker_error_keeps_status(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image)
    nid = win.tree.selected
    win.status_of = {nid: "done"}

    def boom(self, *a, **k):
        raise ValueError("unreadable")
    monkeypatch.setattr(T.Checker, "status", boom)
    win.refresh()
    assert win.status_of == {nid: "done"}
    assert "unreadable" in win.status.text()


def test_failure_classification_not_loose(win, tmp_path, image):
    opened(win, tmp_path, image)
    nid = win.tree.selected
    win._targets = [nid]
    win._error = "CUDA out of memory while loading model"
    win._note_failure(1)
    assert "missing something" not in win._failures[nid]["text"]
    win._error = "Missing model: foo.safetensors"
    win._note_failure(1)
    assert "missing something" in win._failures[nid]["text"]


def test_busy_guards_say_why(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image)
    monkeypatch.setattr(type(win), "running", property(lambda self: True))
    for name, fn in (("add_scene", win.add_scene), ("delete_scene", lambda: win.delete_scene("s1")),
                     ("export_film", win.export_film), ("generate_film", win.generate_film),
                     ("join_film", win.join_film), ("save_project_as", win.save_project_as),
                     ("close_project", win.close_project), ("new_project", win.new_project),
                     ("open_project", lambda: win.open_project(win.project))):
        win.status.setText("")
        fn()
        assert "Wait until" in win.status.text(), name


def test_busy_copy_blocks_scene_switch(win, tmp_path, image):
    opened(win, tmp_path, image)
    win._copying = True
    win.status.setText("")
    assert win.close_project() is None and win.project is not None
    assert "Wait until" in win.status.text()


def test_film_next_without_film(win):
    win.film = None
    win._film_queue = ["s1"]
    win._film_next()                               # no TypeError


def test_state_tip_names_details():
    from video_chain.ui import main_window as M
    assert "status bar" not in M.STATE_TIP["failed"]


def test_stop_llama_can_wait(win):
    calls = []

    class Fake:
        def stop(self, wait=True):
            calls.append(wait)
            return True
    win._llama = Fake()
    win.stop_llama(wait=True)
    assert calls == [True]


def test_start_image_size_is_read_from_header(win, tmp_path, image, monkeypatch):
    from video_chain.ui import main_window as M
    opened(win, tmp_path, image)

    class NoDecode:
        def __init__(self, *a, **k):
            raise AssertionError("the picture was decoded")
    monkeypatch.setattr(M, "QImage", NoDecode)
    win._update_size_info()
    assert "192×108" in win.size_info.text()
    assert win.add_reference(str(tmp_path / "nope.png")) is False
    assert "Not an image" in win.status.text()


def test_pixmap_cache_is_bounded(win, tmp_path, image, monkeypatch):
    from PIL import Image
    from video_chain.ui import main_window as M
    opened(win, tmp_path, image)
    monkeypatch.setattr(M, "PIXMAP_CACHE_MAX", 5)
    win._pix.clear()
    for i in range(12):
        p = tmp_path / f"f{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        assert win._pixmap(p) is not None
    assert len(win._pix) <= 5


def test_add_reference_unknown_owner_is_reported(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image)
    monkeypatch.setattr(T, "add_reference", lambda *a, **k: "refs/ghost.png")
    assert win.add_reference(str(image), crop=None) is False
    assert "no character" in win.status.text()


def test_help_menu_and_preferences_name(win):
    titles = [a.text() for a in win.menuBar().actions()]
    assert "&Help" in titles
    settings_menu = [a for a in win.menuBar().actions() if a.text() == "&Settings"][0].menu()
    assert settings_menu.actions()[0].text() == "Preferences…"


def test_remove_voice_asks_first(win, tmp_path, image, monkeypatch):
    opened(win, tmp_path, image)
    win.tree.options["cast"] = [{"name": "Anna", "images": [], "desc": "", "lead": False,
                                 "voice": {"file": "refs/v.wav", "secs": 3.0}}]
    win._show_references(select="Anna")
    called = []
    monkeypatch.setattr(T, "remove_voice", lambda *a, **k: called.append(1))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    win._remove_voice()
    assert called == []


def test_cast_tooltip_plural(win, tmp_path, image):
    opened(win, tmp_path, image)
    win.tree.options["cast"] = [{"name": "Anna", "images": [], "desc": "", "lead": False}]
    win._show_references(select="Anna")
    assert "clips mention it" not in win.cast_list.item(0).toolTip() or "0 clips" in win.cast_list.item(0).toolTip()
