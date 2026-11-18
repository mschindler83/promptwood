"""Play film / Export film… for a range of scenes: with two scenes or more, a window asks from which scene to which
(the whole film by default, the last choice offered again); one scene: nothing is asked."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path  # noqa: E402

from PySide6.QtWidgets import QFileDialog, QInputDialog  # noqa: E402

from conftest import settings  # noqa: E402,F401
from video_chain import film as F  # noqa: E402
from ui_helpers import app, image, new_project, silent_player, wait, win  # noqa: E402,F401


def _joins(monkeypatch):
    """F.join_film as called by the window (nothing is joined)."""
    calls = []

    def join(film, out, keep=False, scenes=None):
        calls.append({"out": Path(out), "keep": keep, "scenes": scenes})
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"")
        return Path(out)
    monkeypatch.setattr(F, "join_film", join)
    return calls


def _asks(monkeypatch, answers):
    from video_chain.ui import scene_range as SR
    asked = []

    def pick(parent, titles, first, last, action):
        asked.append({"titles": titles, "first": first, "last": last, "action": action})
        return answers.pop(0)
    monkeypatch.setattr(SR, "pick_range", pick)
    return asked


def _film(win, tmp_path, image, monkeypatch, scenes=3):
    new_project(win, tmp_path, image, ("a",))
    for k in range(2, scenes + 1):
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, k=k, **kw: (f"Scene {k}", True))
        win.add_scene()
    win.storyboard.sceneChosen.emit("s1")
    return win.project


def test_one_scene_is_played_without_asking(app, win, tmp_path, image, monkeypatch):
    new_project(win, tmp_path, image, ("a",))
    calls, asked = _joins(monkeypatch), _asks(monkeypatch, [])
    win.b_film_play.click()
    wait(app, lambda: win.status.text().startswith("Film joined"))
    assert asked == [] and calls[0]["scenes"] is None and calls[0]["out"] == win.project / "paths" / "film.mp4"


def test_play_asks_for_the_scenes_and_offers_the_last_choice_again(app, win, tmp_path, image, monkeypatch):
    root = _film(win, tmp_path, image, monkeypatch)
    calls = _joins(monkeypatch)
    asked = _asks(monkeypatch, [(2, 3), (1, 3), None])
    win.b_film_play.click()
    wait(app, lambda: win.status.text().startswith("Film joined"))
    assert asked[0]["titles"] == ["Scene 1", "Scene 2", "Scene 3"] and (asked[0]["first"], asked[0]["last"]) == (1, 3)
    assert calls[-1]["scenes"] == (2, 3) and calls[-1]["out"] == root / "paths" / "film-scenes-2-3.mp4"
    assert win.played[-1] == root / "paths" / "film-scenes-2-3.mp4"
    win.b_play_film.click()                                          # (▶ Film asks the same)
    wait(app, lambda: len(calls) == 2 and not win._joins)
    assert (asked[1]["first"], asked[1]["last"]) == (2, 3)            # the last choice
    assert calls[-1]["scenes"] is None and calls[-1]["out"] == root / "paths" / "film.mp4"   # all: as before
    win.b_film_play.click()                                          # cancelled: nothing joined
    assert len(calls) == 2 and len(asked) == 3


def test_export_asks_for_the_scenes_first(app, win, tmp_path, image, monkeypatch):
    root = _film(win, tmp_path, image, monkeypatch)
    calls = _joins(monkeypatch)
    _asks(monkeypatch, [(2, 2), None])
    offered = []

    def save(parent, title, start, filt):
        offered.append(start)
        return str(tmp_path / "mine.mp4"), ""
    monkeypatch.setattr(QFileDialog, "getSaveFileName", save)
    win.export_film()
    wait(app, lambda: win.status.text().startswith("Exported the film"))
    assert offered[0] == str(root / f"{root.name}-film-scene-2.mp4")
    assert calls[-1] == {"out": tmp_path / "mine.mp4", "keep": False, "scenes": (2, 2)}
    win.export_film()                                                # cancelled: no file asked for
    assert len(offered) == 1 and len(calls) == 1


def test_the_last_choice_is_forgotten_when_its_scenes_are_gone(app, win, tmp_path, image, monkeypatch):
    _film(win, tmp_path, image, monkeypatch)
    _joins(monkeypatch)
    asked = _asks(monkeypatch, [(3, 3), None])
    win.b_film_play.click()
    wait(app, lambda: win.status.text().startswith("Film joined"))
    monkeypatch.setattr("video_chain.ui.dialogs.confirm", lambda *a, **k: True)
    win.delete_scene(win.film.scenes[2].id)
    win.b_film_play.click()
    assert (asked[1]["first"], asked[1]["last"]) == (1, 2)


def test_the_range_window_keeps_the_last_scene_after_the_first(app):
    from video_chain.ui.scene_range import SceneRangeDialog
    dlg = SceneRangeDialog(None, ["Kitchen", "Street", "Park"], 1, 3, "Export")
    assert dlg.windowTitle() == "Export the film" and dlg.first.itemText(0) == "1 · Kitchen"
    assert dlg.value() == (1, 3)
    dlg.first.setCurrentIndex(2)
    assert dlg.value() == (3, 3)                                     # (the last moved along)
    dlg.first.setCurrentIndex(1)
    dlg.last.setCurrentIndex(0)
    assert dlg.value() == (1, 1)                                     # (the first moved back)
    dlg.deleteLater()


def test_play_says_first_that_a_run_or_join_is_busy(app, win, tmp_path, image, monkeypatch):
    """Review: the scenes were asked for, and only then was it said to wait."""
    _film(win, tmp_path, image, monkeypatch)
    calls, asked = _joins(monkeypatch), _asks(monkeypatch, [(1, 2)])
    monkeypatch.setattr(type(win), "running", property(lambda self: True))
    win.play_film()
    assert asked == [] and calls == [] and "Wait" in win.status.text()
    monkeypatch.setattr(type(win), "running", property(lambda self: False))
    win._joins = 1
    win.play_film()
    assert asked == [] and calls == [] and "join is still running" in win.status.text()
    win._joins = 0


def test_the_whole_film_is_remembered_as_the_whole_film(app, win, tmp_path, image, monkeypatch):
    """Review: choosing the whole film kept its first and last scene; a scene added after it was then left out."""
    _film(win, tmp_path, image, monkeypatch, scenes=2)
    _joins(monkeypatch)
    asked = _asks(monkeypatch, [(1, 2), (2, 3)])
    win.b_film_play.click()
    wait(app, lambda: win.status.text().startswith("Film joined") and not win._joins)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Scene 3", True))
    win.add_scene()
    win.b_film_play.click()
    assert (asked[1]["first"], asked[1]["last"]) == (1, 3)
    wait(app, lambda: "scenes 2 to 3" in win.status.text() and not win._joins)   # (which scenes were joined)
