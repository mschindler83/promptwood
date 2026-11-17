"""film.py: unknown scenes, a failed remove, the joined film's stamp, a damaged scene while continues update."""
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


def made_film(project, tmp_path):
    t1 = T.load(project)
    t1.selected = next(iter(t1.nodes))
    T.save(project, t1)
    f = F.Film.load(project)
    s2 = f.add_scene("Two", start_image=tmp_path / "start.jpg")
    d2 = f.path(s2.id)
    t2 = T.load(d2)
    n2 = t2.add(None, "two")
    t2.selected = n2.id
    T.save(d2, t2)
    T.run_tree(FakeGenerator(frames=17, size=None), project, [t1.selected], log=lambda _m: None)
    T.run_tree(FakeGenerator(frames=17, size=None), d2, [n2.id], log=lambda _m: None)
    return f


def test_unknown_scene_is_a_clear_error_not_stopiteration(project):
    f = F.Film.load(project)
    for call in (lambda: f.scene("s9"), lambda: f.before("s9")):
        with pytest.raises(KeyError, match="s9"):
            call()
        with pytest.raises(ValueError):                  # (the window catches OSError / ValueError)
            call()


def test_remove_that_cannot_move_the_folder_keeps_the_scene(project, tmp_path, monkeypatch):
    f = F.Film.load(project)
    s2 = f.add_scene("Two")
    monkeypatch.setattr(chain, "keep_aside", lambda root, d: False)
    with pytest.raises(OSError, match="could not be moved"):
        f.remove(s2.id)
    assert [s.id for s in F.Film.load(project).scenes] == ["s1", s2.id]     # not dropped, not resurrected later
    assert [s.id for s in f.scenes] == ["s1", s2.id]


def test_joined_film_is_joined_again_when_the_fade_length_changes(project, tmp_path, monkeypatch):
    f = made_film(project, tmp_path)
    own = project / "paths" / "film.mp4"
    F.join_film(f, own, keep=True)
    stamp = own.stat().st_mtime_ns
    F.join_film(f, own, keep=True)
    assert own.stat().st_mtime_ns == stamp                                # nothing changed: reused
    f.scenes[1].join = "fade"
    F.join_film(f, own, keep=True)
    stamp = own.stat().st_mtime_ns
    monkeypatch.setattr(F, "FADE_SECS", 1.0)
    F.join_film(f, own, keep=True)
    assert own.stat().st_mtime_ns != stamp
    assert not list(own.parent.glob("*.part*"))


def test_join_film_reads_each_scene_once(project, tmp_path, monkeypatch):
    f = made_film(project, tmp_path)
    real, calls = T.load, []
    monkeypatch.setattr(T, "load", lambda d, *a, **k: calls.append(d) or real(d, *a, **k))
    F.join_film(f, tmp_path / "out.mp4")
    assert len(calls) == len(f.scenes)


def test_one_damaged_scene_does_not_stop_the_other_continues(project, tmp_path, monkeypatch):
    f = F.Film.load(project)
    s2, s3 = f.add_scene("Two"), f.add_scene("Three")
    f.set_join(s2.id, "continue")
    f.set_join(s3.id, "continue")
    png = tmp_path / "end.png"
    Image.new("RGB", (192, 108), (1, 2, 3)).save(png)
    monkeypatch.setattr(f, "end_frame", lambda sid: png)
    bad, real = f.path(s3.id), T.load

    def load(d, *a, **k):
        if d == bad:
            raise ValueError("damaged")
        return real(d, *a, **k)
    monkeypatch.setattr(T, "load", load)
    assert f.update_continues() == [s2.id]
    assert F.Film.load(project).scene(s2.id).from_sha == chain.file_signature(png)


def test_a_duplicate_that_fails_to_save_leaves_no_scene_and_no_folder(project, monkeypatch):
    f = F.Film.load(project)
    before = [s.id for s in f.scenes]
    folders = sorted(p.name for p in (project / F.SCENES_DIR).glob("*")) if (project / F.SCENES_DIR).exists() else []

    def broken(self):
        raise OSError("disk full")
    monkeypatch.setattr(F.Film, "save", broken)
    with pytest.raises(OSError):
        f.duplicate("s1")
    assert [s.id for s in f.scenes] == before
    assert (sorted(p.name for p in (project / F.SCENES_DIR).glob("*")) if (project / F.SCENES_DIR).exists() else []) == folders


def test_a_scene_that_cannot_be_checked_still_gets_its_card(project, monkeypatch):
    f = F.Film.load(project)

    def boom(*_a, **_k):
        raise RuntimeError("ffprobe failed")
    monkeypatch.setattr(T, "path_cuts", boom)
    card = F.summary(f, "s1")
    assert card["clips"] == 1 and card["secs"] == 0.0


def test_a_scene_with_a_damaged_tree_gets_an_empty_card(project):
    f = F.Film.load(project)
    (project / "tree.json").write_text("{not json")
    (project / "tree.json.bak").unlink(missing_ok=True)
    assert F.summary(f, "s1")["clips"] == 0
