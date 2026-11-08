"""Scenes: a film is its folder (the first scene) plus scenes/<id>/, sharing the cast, sounds and size settings."""
import json

import pytest
from fakes import FakeGenerator
from PIL import Image

from video_chain import film as F
from video_chain import tree as T


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "film"
    p.mkdir()
    img = tmp_path / "start.jpg"
    Image.new("RGB", (192, 108), (90, 120, 150)).save(img)
    t = T.Tree()
    T.set_start_image(p, img, t)
    t.add(None, "a clip")
    T.save(p, t)
    return p


def test_an_old_project_is_a_film_with_one_scene(project):
    f = F.Film.load(project)
    assert [(s.id, s.title, s.dir) for s in f.scenes] == [("s1", "Scene 1", ".")]
    f.save()
    assert not (project / F.FILM_FILE).exists()                       # nothing written for one scene
    assert F.root_of(project) == project.resolve()


def test_scenes_add_rename_move_remove(project, tmp_path):
    f = F.Film.load(project)
    img = tmp_path / "street.png"
    Image.new("RGB", (192, 108), (10, 10, 10)).save(img)
    s2 = f.add_scene("Street", start_image=img)
    s3 = f.add_scene("", after="s1")                                  # in between, a default title
    assert [s.id for s in f.scenes] == ["s1", "s3", "s2"] and s3.title == "Scene 3"
    d2 = f.path(s2.id)
    assert d2 == (project / "scenes" / "s2").resolve() and (d2 / T.START_FILE).is_file()
    assert (d2 / T.REFS_DIR).is_symlink() and (d2 / T.REFS_DIR).resolve() == (project / T.REFS_DIR).resolve()
    assert F.root_of(d2) == project.resolve()                          # a scene folder belongs to its film
    data = json.loads((project / F.FILM_FILE).read_text())
    assert [s["id"] for s in data["scenes"]] == ["s1", "s3", "s2"]
    f.rename("s2", "Night street")
    f.move("s2", 0)
    g = F.Film.load(d2)                                               # read again (from a scene folder)
    assert [s.title for s in g.scenes] == ["Night street", "Scene 1", "Scene 3"]
    with pytest.raises(ValueError, match="cannot be deleted"):
        g.remove("s1")
    (project / T.REFS_DIR / "keep.png").write_bytes(b"x")
    g.remove("s2")
    assert not d2.exists() and (project / T.REFS_DIR / "keep.png").is_file()   # shared refs stay
    assert [s.id for s in F.Film.load(project).scenes] == ["s1", "s3"]


def test_shared_settings_reach_every_scene(project, tmp_path):
    f = F.Film.load(project)
    s2 = f.add_scene("Two")
    t = T.load(project)
    T.add_character(project, t, "Anna", tmp_path / "start.jpg")
    t.options["resolution"] = 768
    t.options["style"] = {"prefix": "only scene 1", "suffix": ""}
    T.save(project, t)
    assert f.sync_shared(project, t.options) == [s2.id]
    t2 = T.load(f.path(s2.id))
    assert T.cast_of(t2)[0]["name"] == "Anna" and t2.options["resolution"] == 768
    assert t2.options["style"]["prefix"] == ""                        # not shared: a scene's own look
    rel = T.cast_of(t2)[0]["images"][0]
    assert (f.path(s2.id) / rel).is_file()                            # the picture through the shared refs
    assert f.sync_shared(project, t.options) == []                    # nothing to do the second time


def test_scene_summary(project):
    f = F.Film.load(project)
    t = T.load(project)
    t.selected = next(iter(t.nodes))
    T.save(project, t)
    s = F.summary(f, "s1")
    assert s["new"] == 1 and s["clips"] == 1 and s["thumb"].name == T.START_FILE and s["secs"] > 0
    T.run_tree(FakeGenerator(frames=17), project, [t.selected], log=lambda _m: None)
    assert F.summary(f, "s1")["done"] == 1



def test_scene_endings_and_the_joined_film(project, tmp_path):
    t1 = T.load(project)
    assert any("has no clip yet" in m for m in F.film_problems(F.Film.load(project)))   # nothing chosen yet
    t1.selected = next(iter(t1.nodes))
    T.save(project, t1)
    f = F.Film.load(project)
    img = tmp_path / "tall.png"
    Image.new("RGB", (108, 192), (200, 50, 50)).save(img)                # another shape: fitted into the film
    s2 = f.add_scene("Two", start_image=img)
    d2 = f.path(s2.id)
    t2 = T.load(d2)
    a = t2.add(None, "two a")
    b = t2.add(a.id, "two b")
    t2.selected = a.id
    T.save(d2, t2)
    assert f.take_of(s2.id, t2) == a.id                               # no ending set: the selected clip
    f.set_take(s2.id, b.id)
    assert F.Film.load(project).take_of(s2.id, t2) == b.id and F.summary(f, s2.id)["fixed"]
    assert any("not made yet" in m for m in F.film_problems(f))
    with pytest.raises(ValueError, match="not made yet"):
        F.join_film(f, tmp_path / "film.mp4")
    T.run_tree(FakeGenerator(frames=17, size=None), project, [t1.selected], log=lambda _m: None)
    T.run_tree(FakeGenerator(frames=17, size=None), d2, [b.id], log=lambda _m: None)
    assert F.film_problems(f) == []
    out = F.join_film(f, tmp_path / "film.mp4")
    from video_chain import chain
    info, one = chain.probe(out), chain.probe(T.join_path(project, T.load(project), t1.selected))
    two = chain.probe(T.join_path(d2, T.load(d2), b.id))
    assert info["frames"] == one["frames"] + two["frames"]            # hard cuts: nothing left out
    assert (info["width"], info["height"]) == (one["width"], one["height"])   # the first scene's frame


def test_damaged_or_doubtful_film_json(project, tmp_path, monkeypatch):
    f = F.Film.load(project)
    s2 = f.add_scene("Two")
    s3 = f.add_scene("Three")
    f.reorder([s3.id, "s1", s2.id])
    assert [s.id for s in F.Film.load(project).scenes] == [s3.id, "s1", s2.id]
    data = json.loads((project / F.FILM_FILE).read_text())
    data["scenes"].append({"id": "s9", "dir": f"scenes/{s2.id}", "title": "points at s2"})   # id != folder
    data["scenes"] = [e for e in data["scenes"] if e["id"] != s3.id]                       # s3 not listed
    (project / F.FILM_FILE).write_text(json.dumps(data))
    g = F.Film.load(project)
    assert "doubtful" in g.problem and {s.id for s in g.scenes} == {"s1", s2.id, s3.id}       # s3 adopted
    with pytest.raises(OSError, match="not written"):
        g.save()                                                       # never rewritten while doubtful
    (project / F.FILM_FILE).write_text("{broken")
    h = F.Film.load(project)                                           # still opens (the scenes on disk)
    assert "cannot be read" in h.problem and [s.dir for s in h.scenes][0] == "."
    monkeypatch.setattr(F.os, "symlink", lambda *a, **k: (_ for _ in ()).throw(OSError("no links")))
    (project / F.FILM_FILE).unlink()
    k = F.Film.load(project)
    with pytest.raises(OSError, match="links"):
        k.add_scene("No links here")
    assert not any(p.name not in (s2.id, s3.id) for p in (project / "scenes").iterdir())   # no half scene left


def test_shared_settings_go_on_after_a_bad_scene(project, tmp_path):
    f = F.Film.load(project)
    s2, s3 = f.add_scene("Two"), f.add_scene("Three")
    (f.path(s2.id) / T.TREE_FILE).write_text("{broken")
    t = T.load(project)
    t.options["resolution"] = 768
    with pytest.raises(OSError, match="Two"):
        f.sync_shared(project, t.options)
    assert T.load(f.path(s3.id)).options["resolution"] == 768           # the others still got it
    assert F.Film.load(project).shared_state["resolution"] == 768
