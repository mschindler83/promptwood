"""The sound remix of a made clip: what it keeps, what it must not break (synthetic clips, fake generator)."""
import shutil

import pytest
from fakes import FakeGenerator
from test_video import _Recording, _sound_project, project  # noqa: F401  (fixture)

from video_chain import chain
from video_chain import tree as T

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def _made(project, tmp_path, **extra):
    t, tid = _sound_project(project, tmp_path)
    a = t.add(None, "she dances")
    a.overrides = {"source_sound": f"track:{tid}", "sound_over": True}
    T.save(project, t)
    T.run_tree(_Recording(frames=17, size=None), project, [a.id], log=lambda _m: None)
    t = T.load(project)
    return t, a.id, tid


def test_a_remix_keeps_the_clip_before_in_old(project, tmp_path):
    t, nid, tid = _made(project, tmp_path)
    video = project / t.nodes[nid].made["video"]
    before = video.read_bytes()
    t.nodes[nid].overrides = {"source_sound": f"track:{tid}", "sound_over": "mix"}
    assert T.remix_sound(project, t, nid)
    kept = list((project / chain.OLD_DIR / "nodes").glob(video.stem + "*.mp4"))
    assert len(kept) == 1 and kept[0].read_bytes() == before and video.read_bytes() != before


def test_a_remix_that_fails_does_not_stop_remix_sounds(project, tmp_path):
    t, nid, tid = _made(project, tmp_path)
    t.nodes[nid].overrides = {"source_sound": f"track:{tid}", "sound_over": "mix"}
    for f in (project / "refs").glob("track_*"):
        f.unlink()                                      # its sound file is gone
    logged = []
    assert T.remix_sounds(project, t, [nid], log=logged.append) == []
    assert logged and nid in logged[0]


def test_a_remix_only_rebaselines_the_sound_overrides(project, tmp_path):
    t, nid, tid = _made(project, tmp_path)
    t.nodes[nid].made["overrides"]["lead"] = False      # an edit Undo goes back to; not the remix's business
    t.nodes[nid].overrides = {"source_sound": f"track:{tid}", "sound_over": "mix"}
    assert T.remix_sound(project, t, nid, persist=False)
    m = t.nodes[nid].made["overrides"]
    assert m["lead"] is False and m["sound_over"] == "mix"


def test_a_failed_run_leaves_no_own_sound_behind(project, tmp_path, monkeypatch):
    t, tid = _sound_project(project, tmp_path)
    a = t.add(None, "she dances")
    a.overrides = {"source_sound": f"track:{tid}", "sound_over": True}
    T.save(project, t)

    def boom(*_a, **_k):
        raise RuntimeError("ffmpeg died")
    monkeypatch.setattr(chain, "replace_audio", boom)
    with pytest.raises(RuntimeError):
        T.run_tree(FakeGenerator(frames=17, size=None), project, [a.id], log=lambda _m: None)
    assert not list((project / "nodes").glob("*.own.*"))


def test_remix_sounds_saves_once(project, tmp_path, monkeypatch):
    t, tid = _sound_project(project, tmp_path)
    ids = []
    for k in range(2):
        n = t.add(None, f"clip {k}")
        n.overrides = {"source_sound": f"track:{tid}", "sound_over": True}
        ids.append(n.id)
    T.save(project, t)
    T.run_tree(_Recording(frames=17, size=None), project, ids, log=lambda _m: None)
    t = T.load(project)
    for i in ids:
        t.nodes[i].overrides = {"source_sound": f"track:{tid}", "sound_over": "mix"}
    saves = []
    real = T.save
    monkeypatch.setattr(T.engine, "save", lambda *a, **k: saves.append(1) or real(*a, **k))
    assert sorted(T.remix_sounds(project, t, ids)) == sorted(ids)
    assert len(saves) == 1
