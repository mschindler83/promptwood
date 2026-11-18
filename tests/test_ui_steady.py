"""Clip settings: "Keep the look steady along a chain" sets the project's steady_look (on for new projects, off for
a project saved before it)."""
import json
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pytest  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, flush, image, new_project, silent_player, win  # noqa: E402,F401

pytestmark = pytest.mark.steady_default


def test_the_steady_look_box_sets_the_option(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a", "b"))
    assert win.steady_look.isChecked() and win.tree.options["steady_look"] is True
    assert "carried" in win.steady_look.toolTip() or "hands" in win.steady_look.toolTip()
    win.steady_look.setChecked(False)
    assert win.tree.options["steady_look"] is False
    flush(app, win)
    assert T.load(win.project).options["steady_look"] is False


def test_a_project_from_before_opens_with_it_off(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a", "b"))
    root = win.project
    f = root / T.TREE_FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    del d["options"]["steady_look"]
    f.write_text(json.dumps(d), encoding="utf-8")
    assert win.open_project(root)
    assert not win.steady_look.isChecked() and win.tree.options["steady_look"] is False


def test_a_new_project_gets_it_on_after_an_old_project_was_open(app, win, tmp_path, image):
    """Review: the last used settings (from an old project: off) were copied into every new project."""
    from video_chain.ui import settings_keys as SK
    win.qs.setValue(SK.LAST_OPTIONS, json.dumps({"steady_look": False, "frames": 73}))
    new_project(win, tmp_path, image, ("a",))
    assert win.tree.options["steady_look"] is True and win.steady_look.isChecked()
