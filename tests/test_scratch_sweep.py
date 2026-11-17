"""Scratch folders (media.scratch: video-chain-<name>-...) that a killed run left behind are swept at startup."""
import os
import time

from video_chain import media


def test_old_scratch_folders_are_swept_and_other_things_stay(tmp_path):
    old = tmp_path / "video-chain-job-abc"
    new = tmp_path / "video-chain-tts-def"
    other = tmp_path / "somebody-else"
    for d in (old, new, other):
        d.mkdir()
        (d / "x").write_bytes(b"1")
    a_file = tmp_path / "video-chain-notes.txt"
    a_file.write_text("mine")
    long_ago = time.time() - 3 * 86400
    for p in (old, other, a_file):
        os.utime(p, (long_ago, long_ago))
    gone = media.sweep_scratch(tmp_path, older_than=86400)
    assert gone == [old] and not old.exists()
    assert new.exists() and other.exists() and a_file.exists()          # (young, not ours, not a folder)


def test_a_project_sweep_takes_the_job_folders_of_a_killed_run(tmp_path):
    from video_chain import tree as T
    (tmp_path / "nodes" / "video-chain-job-1").mkdir(parents=True)
    (tmp_path / "nodes" / "n1.mp4").write_bytes(b"clip")
    T.sweep_scratch(tmp_path)
    assert not (tmp_path / "nodes" / "video-chain-job-1").exists() and (tmp_path / "nodes" / "n1.mp4").exists()


def test_squash_makes_one_line_of_any_white_space():
    assert media.squash("  a \n b\t\tc  ") == "a b c" and media.squash(None) == "" and media.squash(5) == "5"
