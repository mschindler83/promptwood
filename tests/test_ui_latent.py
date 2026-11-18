"""Clip settings: "Carry over from the latent" sets the project's latent_handover (off unless switched on)."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import app, flush, image, new_project, silent_player, win  # noqa: E402,F401


def test_the_latent_box_sets_the_option(app, win, tmp_path, image):
    new_project(win, tmp_path, image, ("a", "b"))
    assert not win.latent_handover.isChecked() and win.tree.options["latent_handover"] is False
    assert "Motion-Context" in win.latent_handover.toolTip()
    win.latent_handover.setChecked(True)
    assert win.tree.options["latent_handover"] is True
    flush(app, win)
    assert T.load(win.project).options["latent_handover"] is True
    assert ("latent_handover", "Carry-over from the latent") in __import__(
        "video_chain.ui.ui_logic", fromlist=["x"]).project_setting_changes(win.tree.options)
