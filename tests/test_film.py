"""Scenes: a film is its folder (the first scene) plus scenes/<id>/, sharing the cast, sounds and size settings."""
import json

import pytest
from fakes import FakeGenerator
from PIL import Image

from video_chain import chain
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
    kept = project / chain.OLD_DIR / "scenes" / "s2"                  # the scene's folder is never deleted: moved
    assert (kept / T.START_FILE).is_file() and (kept / T.TREE_FILE).is_file() and not (kept / T.REFS_DIR).exists()
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
    # the app's own file of the film: joined again only when something changed, the one before goes to old/
    own = project / "paths" / "film.mp4"
    assert F.join_film(f, own, keep=True) == own
    first, stamp = own.read_bytes(), own.stat().st_mtime_ns
    assert F.join_film(f, own, keep=True) == own and own.stat().st_mtime_ns == stamp
    assert not (project / chain.OLD_DIR / "paths").exists()
    f.scenes[1].join = "fade" if f.scenes[1].join != "fade" else "cut"
    F.join_film(f, own, keep=True)
    assert (project / chain.OLD_DIR / "paths" / "film.mp4").read_bytes() == first and own.read_bytes() != first
    f.scenes[1].join = "fade" if f.scenes[1].join != "fade" else "cut"
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



def test_crossfade_and_continuing_scenes(project, tmp_path):
    from video_chain import chain
    t1 = T.load(project)
    a1 = next(iter(t1.nodes))
    t1.selected = a1
    T.save(project, t1)
    f = F.Film.load(project)
    s2 = f.add_scene("Two")
    d2 = f.path(s2.id)
    t2 = T.load(d2)
    b = t2.add(None, "two")
    t2.selected = b.id
    T.save(d2, t2)
    f.set_join(s2.id, "continue")
    assert any("continues scene 1, which is not made yet" in m for m in F.film_problems(f))
    T.run_tree(FakeGenerator(frames=17, size=None), project, [a1], log=lambda _m: None)
    assert f.update_continues() == [s2.id]                              # its start: scene 1's last frame
    end = f.end_frame("s1")
    assert end is not None and (d2 / T.START_FILE).is_file()
    assert f.update_continues() == []                                   # nothing new: not set again
    T.run_tree(FakeGenerator(frames=17, size=None), d2, [b.id], log=lambda _m: None)
    one = chain.probe(T.join_path(project, T.load(project), a1))["frames"]
    two = chain.probe(T.join_path(d2, T.load(d2), b.id))["frames"]
    out = F.join_film(f, tmp_path / "cont.mp4")
    assert chain.probe(out)["frames"] == one + two - 1                  # the repeated first frame left out
    f.set_join(s2.id, "fade")
    out = F.join_film(f, tmp_path / "fade.mp4")
    info = chain.probe(out)
    fade = min(F.FADE_SECS, one / 24 / 2, two / 24 / 2)
    assert abs(info["frames"] - (one + two - fade * 24)) <= 1 and info["audio"]   # they overlap by the fade


def test_duplicate_notes_and_cast_by_scene(project, tmp_path):
    f = F.Film.load(project)
    t = T.load(project)
    T.add_character(project, t, "Anna", tmp_path / "start.jpg")
    first = next(iter(t.nodes))
    t.nodes[first].prompt = "@Anna smiles"
    t.selected = first
    T.save(project, t)
    s2 = f.add_scene("Two")
    copy = f.duplicate("s1")                                      # the film folder's own scene, copied
    d = f.path(copy.id)
    assert [s.id for s in f.scenes] == ["s1", copy.id, s2.id] and copy.title == "Scene 1 (copy)"
    assert (d / T.REFS_DIR).is_symlink() and not (d / "scenes").exists() and not (d / F.FILM_FILE).exists()
    assert [n.prompt for n in T.load(d).nodes.values()] == ["@Anna smiles"]
    f.set_notes(s2.id, "  Anna walks home.  ")
    assert F.Film.load(project).scene(s2.id).notes == "Anna walks home."
    names, rows = F.cast_by_scene(F.Film.load(project))
    assert names == ["Anna"] and rows[0] == ("Scene 1", {"Anna": "face"}) and rows[2] == ("Two", {})


def test_continue_stays_honest_after_changes(project, tmp_path):
    from video_chain import chain
    t1 = T.load(project)
    a1 = next(iter(t1.nodes))
    b1 = t1.add(a1, "scene one goes on")
    t1.selected = a1
    T.save(project, t1)
    f = F.Film.load(project)
    s2, s3 = f.add_scene("Two"), f.add_scene("Three", start_image=tmp_path / "start.jpg")
    for sc in (s2, s3):
        d = f.path(sc.id)
        t = T.load(d)
        n = t.add(None, sc.title)
        t.selected = n.id
        T.save(d, t)
    f.set_join(s2.id, "continue")
    f.set_join(s3.id, "fade")
    T.run_tree(FakeGenerator(frames=17, size=None), project, [b1.id], log=lambda _m: None)
    f.update_continues()
    for sc in (s2, s3):
        d = f.path(sc.id)
        T.run_tree(FakeGenerator(frames=17, size=None), d, [T.load(d).selected], log=lambda _m: None)
    out = F.join_film(f, tmp_path / "mixed.mp4")                       # continue (a cut) then a crossfade
    assert chain.probe(out)["frames"] > 30
    f.set_take("s1", b1.id)                                             # scene 1 ends later now
    assert any("whose end changed" in m for m in F.film_problems(f))    # never joined with the old start
    assert f.reorder([s2.id, "s1", s3.id]) == [s2.id]                   # first now: its continue is a cut
    assert F.Film.load(project).scene(s2.id).join == "cut"


def test_duplicating_continuing_scenes(project, tmp_path):
    f = F.Film.load(project)
    s2, s3 = f.add_scene("Two"), f.add_scene("Three")
    f.set_join(s2.id, "continue")
    f.set_join(s3.id, "continue")
    c = f.duplicate(s2.id)                           # s3 continues s2: the copy goes to the end, not in between
    assert [s.id for s in f.scenes] == ["s1", s2.id, s3.id, c.id] and c.join == "cut" and c.title == "Two (copy)"
    assert f.duplicate(c.id).title == "Two (copy 2)"


def _three_scenes(project, tmp_path, make=(1, 2, 3)):
    """A film of three scenes with one clip each; the scenes in `make` are made."""
    t1 = T.load(project)
    t1.selected = next(iter(t1.nodes))
    T.save(project, t1)
    f = F.Film.load(project)
    dirs = {1: (project, t1.selected)}
    for k in (2, 3):
        s = f.add_scene(f"Scene {k}", start_image=tmp_path / "start.jpg")
        d = f.path(s.id)
        t = T.load(d)
        n = t.add(None, f"clip {k}")
        t.selected = n.id
        T.save(d, t)
        dirs[k] = (d, n.id)
    for k in make:
        T.run_tree(FakeGenerator(frames=17, size=None), dirs[k][0], [dirs[k][1]], log=lambda _m: None)
    return f, dirs


def test_a_range_of_scenes_is_joined_and_only_its_scenes_are_checked(project, tmp_path):
    f, dirs = _three_scenes(project, tmp_path, make=(1, 2))           # scene 3 not made yet
    assert any("scene 3" in m for m in F.film_problems(f))
    assert F.film_problems(f, scenes=(1, 2)) == []                     # (scene 3 is not in the range)
    frames = {k: chain.probe(T.join_path(d, T.load(d), n))["frames"] for k, (d, n) in dirs.items() if k < 3}
    out = F.join_film(f, tmp_path / "one-two.mp4", scenes=(1, 2))
    assert chain.probe(out)["frames"] == frames[1] + frames[2]
    out = F.join_film(f, tmp_path / "two.mp4", scenes=(2, 2))
    assert chain.probe(out)["frames"] == frames[2]
    with pytest.raises(ValueError, match="scene 3"):
        F.join_film(f, tmp_path / "all.mp4")                           # (no range: the whole film, as before)
    for bad in ((0, 1), (2, 1), (1, 4)):
        with pytest.raises(ValueError, match="range"):
            F.join_film(f, tmp_path / "bad.mp4", scenes=bad)


def test_a_range_may_begin_with_a_continuing_scene(project, tmp_path):
    """Its join to the scene before is not part of the range: it plays whole, and the scene before is not checked."""
    f, dirs = _three_scenes(project, tmp_path, make=(2,))
    f.set_join("s2", "continue")
    assert any("continues scene 1" in m for m in F.film_problems(f))
    assert F.film_problems(f, scenes=(2, 2)) == []
    d, n = dirs[2]
    out = F.join_film(f, tmp_path / "two.mp4", scenes=(2, 2))
    assert chain.probe(out)["frames"] == chain.probe(T.join_path(d, T.load(d), n))["frames"]


def test_a_range_is_kept_in_its_own_file_and_joins_inside_it_are_used(project, tmp_path):
    """Review: the app's own file of a range (keep), and a crossfade / continuation inside the range."""
    f, dirs = _three_scenes(project, tmp_path, make=(1, 2))
    whole = project / "paths" / "film.mp4"
    own = project / "paths" / "film-scenes-1-2.mp4"
    assert F.join_film(f, own, keep=True, scenes=(1, 2)) == own
    stamp = own.stat().st_mtime_ns
    assert F.join_film(f, own, keep=True, scenes=(1, 2)) == own and own.stat().st_mtime_ns == stamp
    assert not whole.exists()
    frames = {k: chain.probe(T.join_path(d, T.load(d), n))["frames"] for k, (d, n) in dirs.items() if k < 3}
    f.set_join("s2", "fade")
    out = F.join_film(f, tmp_path / "fade.mp4", scenes=(1, 2))
    fade = min(F.FADE_SECS, frames[1] / 24 / 2, frames[2] / 24 / 2)
    assert abs(chain.probe(out)["frames"] - (frames[1] + frames[2] - fade * 24)) <= 1
    f.set_join("s3", "continue")                                       # (scene 3 continues scene 2 in the range)
    assert any("scene 3" in m and "continues scene 2" in m for m in F.film_problems(f, scenes=(2, 3)))
