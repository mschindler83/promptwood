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
