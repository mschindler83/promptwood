"""Leftovers: cast problems carry a kind, the speed mode has a tree accessor, events live outside ui/."""
from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PIL import Image

from video_chain import tree as T


@pytest.fixture
def project(tmp_path):
    img = tmp_path / "in.jpg"
    Image.new("RGB", (96, 54), (250, 250, 250)).save(img)
    p = tmp_path / "proj"
    t = T.Tree()
    T.set_start_image(p.mkdir() or p, img, t)
    T.save(p, t)
    return p


def test_cast_issues_carry_a_kind_not_only_text(project, tmp_path):
    t = T.load(project)
    face = tmp_path / "b.jpg"
    Image.new("RGB", (64, 80)).save(face)
    T.add_character(project, t, "Ben", face)
    T.add_character(project, t, "Nobody")
    a, b, c = t.add(None, "@Bem waves."), t.add(None, "@Nobody waves."), t.add(None, "@Ben <Picture 3> x @Ben")
    T.save(project, t)
    issues = T.cast_issues(project, t, [a.id, b.id, c.id])
    assert [(i.kind, i.name, i.suggestion, i.person) for i in issues[:2]] == [
        (T.CastKind.UNKNOWN, "Bem", "Ben", False), (T.CastKind.NO_PICTURE, "Nobody", "", True)]
    assert all(i.kind is T.CastKind.OTHER for i in issues[2:])
    assert [i.text for i in issues] == T.cast_problems(project, t, [a.id, b.id, c.id])


def test_speed_mode_is_read_and_set_in_the_tree():
    o = dict(T.DEFAULT_OPTIONS)
    assert T.speed_mode(o) == "normal"
    for mode in ("turbo", "hyperflow", "pika", "normal"):
        T.set_speed_mode(o, mode)
        assert T.speed_mode(o) == mode
        assert [m for m in ("turbo", "hyperflow", "pika") if o[m]["on"]] == ([] if mode == "normal" else [mode])
    with pytest.raises(ValueError):
        T.set_speed_mode(o, "warp")


def test_tts_has_a_public_set_voice_fields():
    from video_chain import tts
    assert callable(getattr(tts, "set_voice_fields", None))
    import inspect
    from video_chain.ui import mw_tts
    assert "_set_voice_fields" not in inspect.getsource(mw_tts)


def test_the_worker_builds_its_events_from_the_typed_events_module():
    import ast
    from pathlib import Path
    from video_chain import events
    from video_chain.ui import generator_state as gs
    src = (Path(events.__file__).parent / "__main__.py").read_text()
    tree = ast.parse(src)
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m == "ui" or m.startswith("ui.") for m in imported) and ".ui" not in src.replace("video_chain.ui", "")
    # no emit({...}) with a hand-written dict
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "emit"
           and n.args and isinstance(n.args[0], ast.Dict)]
    assert bad == []
    assert gs.StartEvent is events.StartEvent and gs.event_problem is events.event_problem


def test_event_builders_make_events_the_window_accepts():
    from video_chain import events as E
    for ev in (E.loading("comfyui"), E.loaded("runpod"), E.note("n1", "hi"), E.step("n1", 0, 1, 8),
               E.error("x", comfy_down=True), E.idle(60.0, 1.5), E.bye("idle")):
        assert E.event_problem(ev) is None
    assert E.error("x") == {"e": "error", "msg": "x"} and E.idle(None) == {"e": "idle", "timeout": None}
