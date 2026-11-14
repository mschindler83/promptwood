"""tree.json on disk: safe writing by several writers, a backup to fall back to (synthetic trees only)."""
import json
import multiprocessing
import threading

import pytest

from video_chain import tree as T


def _tree(n: int) -> T.Tree:
    t = T.Tree()
    a = t.add(None, f"prompt {n}")
    for k in range(5):
        t.add(a.id, f"{n} child {k}")
    return t


def _save_many(project, base):
    for k in range(25):
        T.save(project, _tree(base + k))


def test_concurrent_saves_in_threads_never_corrupt(tmp_path):
    errors = []

    def work(base):
        try:
            _save_many(tmp_path, base)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(i * 100,)) for i in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert len(T.load(tmp_path).nodes) == 6
    assert not list(tmp_path.glob("*.part"))


def test_concurrent_saves_from_processes_never_corrupt(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_save_many, args=(tmp_path, i * 100)) for i in range(3)]
    for p in procs:
        p.start()
    _save_many(tmp_path, 900)
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert len(T.load(tmp_path).nodes) == 6
    assert not list(tmp_path.glob("*.part"))


def test_save_keeps_the_previous_file_as_backup(tmp_path):
    T.save(tmp_path, _tree(1))
    first = (tmp_path / T.TREE_FILE).read_text()
    T.save(tmp_path, _tree(2))
    assert (tmp_path / (T.TREE_FILE + ".bak")).read_text() == first


def test_damaged_tree_falls_back_to_the_backup(tmp_path):
    T.save(tmp_path, _tree(1))
    T.save(tmp_path, _tree(2))
    (tmp_path / T.TREE_FILE).write_text('{"nodes": [ truncated')
    t = T.load(tmp_path)
    assert t.nodes["n1"].prompt == "prompt 1"                 # the file before the last save
    assert t.recovered_from == tmp_path / (T.TREE_FILE + ".bak")


def test_a_good_tree_says_nothing_about_recovery(tmp_path):
    T.save(tmp_path, _tree(1))
    assert T.load(tmp_path).recovered_from is None


def test_damaged_tree_and_damaged_backup_are_refused(tmp_path):
    T.save(tmp_path, _tree(1))
    T.save(tmp_path, _tree(2))
    (tmp_path / T.TREE_FILE).write_text("{ nope")
    (tmp_path / (T.TREE_FILE + ".bak")).write_text("{ nope too")
    with pytest.raises(ValueError, match="damaged"):
        T.load(tmp_path)


def test_saving_over_a_damaged_file_keeps_the_good_backup_and_the_damaged_copy(tmp_path):
    T.save(tmp_path, _tree(1))
    T.save(tmp_path, _tree(2))
    good_bak = (tmp_path / (T.TREE_FILE + ".bak")).read_text()
    (tmp_path / T.TREE_FILE).write_text("{ nope")
    T.save(tmp_path, _tree(3))
    assert (tmp_path / (T.TREE_FILE + ".bak")).read_text() == good_bak      # not replaced by the damaged file
    assert (tmp_path / (T.TREE_FILE + ".damaged")).read_text() == "{ nope"
    assert T.load(tmp_path).nodes["n1"].prompt == "prompt 3"


def test_a_node_without_a_seed_is_repaired_not_fatal(tmp_path):
    T.save(tmp_path, _tree(1))
    d = json.loads((tmp_path / T.TREE_FILE).read_text())
    d["nodes"][0]["seed"] = None
    (tmp_path / T.TREE_FILE).write_text(json.dumps(d))
    t = T.load(tmp_path)
    assert t.recovered_from is None and isinstance(t.nodes["n1"].seed, int)


def test_status_of_a_very_deep_chain_does_not_hit_the_recursion_limit(tmp_path, monkeypatch):
    t = T.Tree()
    nid = None
    for k in range(1500):
        nid = t.add(nid, f"p{k}").id
        t.nodes[nid].made = {"prompt": f"p{k}", "seed": t.nodes[nid].seed, "settings": None, "start_sha1": None}
    ck = T.Checker(tmp_path, t)
    ck._files_there = lambda node: True
    ck.settings = lambda nid: None
    ck.start_frame = lambda node: None
    ck.end_key = lambda node: None
    ck.keyframe_key = lambda node: None
    ck.guide_from = lambda node: None
    assert ck.status(nid) == "done"


def test_node_settings_are_worked_out_once_per_round(tmp_path, monkeypatch):
    t = T.Tree()
    a = t.add(None, "a")
    calls = []
    monkeypatch.setattr(T, "node_settings", lambda *a_, **k: calls.append(1) or {})
    ck = T.Checker(tmp_path, t)
    ck.base = lambda: {}
    for _ in range(4):
        ck.settings(a.id)
    assert len(calls) == 1


def test_a_newer_file_version_is_not_opened_or_replaced_by_the_backup(tmp_path):
    T.save(tmp_path, _tree(1))
    T.save(tmp_path, _tree(2))
    d = json.loads((tmp_path / T.TREE_FILE).read_text())
    d["version"] = T.TREE_VERSION + 1
    (tmp_path / T.TREE_FILE).write_text(json.dumps(d))
    with pytest.raises(T.NewerTreeError, match="newer version"):
        T.load(tmp_path)


def test_stored_sub_options_keep_the_defaults_of_missing_keys():
    t = T.Tree.from_json({"options": {"turbo": {"on": True}}, "nodes": []})
    assert t.options["turbo"]["on"] is True and t.options["turbo"]["steps"] == 6


def test_from_json_does_not_change_its_argument():
    t = _tree(1)
    t.nodes["n1"].made = {"prompt": "x", "overrides": {"lora": "x"}, "settings": {}}
    d = json.loads(json.dumps(t.to_json()))
    before = json.dumps(d, sort_keys=True)
    T.Tree.from_json(d)
    assert json.dumps(d, sort_keys=True) == before


def test_keyframe_indexes_are_checked(tmp_path):
    t = _tree(1)
    with pytest.raises(ValueError, match="no keyframe"):
        T.remove_keyframe(tmp_path, t, "n1", 3)
    with pytest.raises(ValueError, match="no keyframe"):
        T.set_keyframe(tmp_path, t, "n1", None, 1.0, "x", index=2)


def test_sidecar_text_is_written_whole(tmp_path):
    from video_chain import chain
    d = tmp_path / "project"                     # (tmp_path also holds the test's own home folder)
    d.mkdir()
    f = d / "pins.json"
    chain.atomic_write_text(f, '{"a": 1}')
    chain.atomic_write_text(f, '{"a": 2}')
    assert json.loads(f.read_text()) == {"a": 2} and [p.name for p in d.iterdir()] == ["pins.json"]
