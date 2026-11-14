"""Deleting a clip: its files move to old/ only after tree.json was saved without it (TRE-24, UI part)."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from video_chain import chain, tree as T
from test_ui import app, flush, image, new_project, win  # noqa: F401  (fixtures)


def _made_clip(win, nid):
    clip = win.project / "nodes" / f"{nid}.mp4"
    clip.parent.mkdir(exist_ok=True)
    clip.write_bytes(b"clip")
    win.tree.nodes[nid].made = {"prompt": "p", "seed": 1, "settings": {}, "video": f"nodes/{nid}.mp4",
                                "frame": f"nodes/{nid}.png"}
    win._write_tree()
    return clip


def test_a_failed_save_leaves_the_deleted_clips_files_where_they_are(app, win, tmp_path, image, monkeypatch):
    a, b = new_project(win, tmp_path, image)
    clip = _made_clip(win, b)
    monkeypatch.setattr(T, "save", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(win, "_report_error", lambda *_a, **_k: None)
    win.node_action("delete", b)
    assert clip.exists() and not (win.project / chain.OLD_DIR / "nodes" / clip.name).exists()


def test_a_deleted_clips_files_move_to_old_after_the_save(app, win, tmp_path, image):
    a, b = new_project(win, tmp_path, image)
    clip = _made_clip(win, b)
    win.node_action("delete", b)
    flush(app, win)
    assert not clip.exists() and (win.project / chain.OLD_DIR / "nodes" / clip.name).read_bytes() == b"clip"


def test_keeping_only_one_clip_retires_the_others_after_the_save(app, win, tmp_path, image, monkeypatch):
    a, b = new_project(win, tmp_path, image)
    win.node_action("branch", a)                               # a second clip after the first, beside b
    other = win.selected_id()
    clip = _made_clip(win, other)
    monkeypatch.setattr(T, "save", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(win, "_report_error", lambda *_a, **_k: None)
    win.keep_only(b, [other], confirm=False)
    assert clip.exists()


def test_opening_a_project_sweeps_the_job_folders_of_a_killed_run(app, win, tmp_path):
    proj = tmp_path / "proj"
    (proj / "nodes" / "video-chain-job-1").mkdir(parents=True)
    (proj / "nodes" / "n1.mp4").write_bytes(b"clip")
    win.open_project(proj)
    assert not (proj / "nodes" / "video-chain-job-1").exists() and (proj / "nodes" / "n1.mp4").exists()
