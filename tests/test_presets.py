"""PresetStore: a damaged or unreadable file is never overwritten with an empty one; unknown names are said."""
import os

import pytest

from video_chain.presets import PresetStore


def store(tmp_path):
    return PresetStore(tmp_path / "p.json")


def test_corrupt_file_is_kept_aside_not_overwritten(tmp_path):
    s = store(tmp_path)
    s.save("/l/a.safetensors", "one", {"prompt": "x"})
    (tmp_path / "p.json").write_text('{"version": 1, "loras": {"truncated', encoding="utf-8")
    s.save("/l/b.safetensors", "two", {"prompt": "y"})
    kept = [f for f in os.listdir(tmp_path) if f.startswith("p.json.") and f.endswith(".bak")]
    assert len(kept) == 1 and "truncated" in (tmp_path / kept[0]).read_text(encoding="utf-8")
    assert s.get("/l/b.safetensors", "two")["prompt"] == "y"


def test_unreadable_file_stops_a_save_instead_of_wiping(tmp_path, monkeypatch):
    s = store(tmp_path)
    s.save("/l/a.safetensors", "one", {"prompt": "x"})
    real = type(s.path).read_text

    def boom(self, *a, **k):
        if self.name == "p.json":
            raise PermissionError("share gone")
        return real(self, *a, **k)
    monkeypatch.setattr(type(s.path), "read_text", boom)
    with pytest.raises(OSError):
        s.save("/l/b.safetensors", "two", {})
    assert s.list("/l/a.safetensors") == []              # (listing just shows nothing)
    monkeypatch.undo()
    assert s.get("/l/a.safetensors", "one")["prompt"] == "x"      # nothing was lost


def test_rename_and_delete_of_an_unknown_preset(tmp_path):
    s = store(tmp_path)
    s.save("/l/a.safetensors", "one", {})
    before = (tmp_path / "p.json").stat().st_mtime_ns
    with pytest.raises(ValueError, match="no preset"):
        s.rename("/l/a.safetensors", "nope", "new")
    s.delete("/l/a.safetensors", "nope")
    assert (tmp_path / "p.json").stat().st_mtime_ns == before      # nothing rewritten
    s.rename("/l/a.safetensors", "one", "uno")
    assert [p["name"] for p in s.list("/l/a.safetensors")] == ["uno"]


def test_two_stores_do_not_lose_each_others_saves(tmp_path):
    import threading
    a, b = store(tmp_path), store(tmp_path)
    ts = [threading.Thread(target=(a if i % 2 else b).save, args=("/l/x.safetensors", f"n{i}", {}))
          for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(a.list("/l/x.safetensors")) == 20
