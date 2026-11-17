from video_chain import tree as T


def test_inside_only_accepts_files_below_the_folder(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "a.png").write_bytes(b"x")
    assert T.inside(tmp_path, "refs", "refs/a.png") == (tmp_path / "refs" / "a.png").resolve()
    assert T.inside(tmp_path, "refs", "refs/../tree.json") is None
    assert T.inside(tmp_path, "refs", "../elsewhere.png") is None
    assert T.inside(tmp_path, "refs", "refs") is None


def test_retire_moves_to_old_and_ignores_files_outside(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "a.png").write_bytes(b"x")
    (tmp_path / "other.png").write_bytes(b"y")
    assert T.retire(tmp_path, "refs", "refs/a.png")
    assert not (tmp_path / "refs" / "a.png").exists()
    assert (tmp_path / "old" / "refs" / "a.png").is_file()
    assert not T.retire(tmp_path, "refs", "other.png")
    assert (tmp_path / "other.png").is_file()
