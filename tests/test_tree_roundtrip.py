"""tree.json: unknown keys survive a load and save; older schemas open; to_json(from_json(x)) is stable."""
import copy

import pytest

from video_chain import tree as T


def _tree():
    t = T.Tree()
    a = t.add(None, "she walks")
    t.add(a.id, "she runs")
    return t


def test_unknown_keys_survive_a_load_and_save():
    d = _tree().to_json()
    d["something_new"] = {"x": 1}
    d["nodes"][0]["extra_field"] = [1, 2]
    again = T.Tree.from_json(copy.deepcopy(d)).to_json()
    assert again["something_new"] == {"x": 1} and again["nodes"][0]["extra_field"] == [1, 2]


def test_to_json_from_json_round_trip_is_stable():
    t = _tree()
    t.nodes["n1"].seed = 7
    t.nodes["n1"].overrides = {"frames": 49}
    d = t.to_json()
    assert T.Tree.from_json(copy.deepcopy(d)).to_json() == d
    assert T.Tree.from_json(copy.deepcopy(d)) == t


def test_a_tree_of_version_1_without_cast_or_loras_opens():
    """The oldest shape: no cast, no version-3 fields, project LoRAs in the options moved to the clips."""
    old = {"tool": "video-chain", "version": 1, "image": None, "selected": "n1", "next_num": 2,
           "options": {"backend": "comfyui", "lora": "/x/a.safetensors", "strength": 0.5},
           "nodes": [{"id": "n1", "parent": None, "prompt": "p", "seed": 3, "made": None, "overrides": {}}]}
    t = T.Tree.from_json(old)
    assert "backend" not in t.options and t.options["prompt_format"] == "classic"
    assert "cast" in t.options and t.next_num == 2
    assert t.nodes["n1"].overrides.get("loras") == [{"path": "/x/a.safetensors", "strength": 0.5}]
    assert T.Tree.from_json(t.to_json()) == t


def test_a_newer_file_is_still_refused():
    with pytest.raises(T.NewerTreeError):
        T.Tree.from_json({"version": 999, "nodes": []})


def test_a_linear_chain_json_becomes_a_one_branch_tree(tmp_path):
    """The pre-branching project: clips in order, settings and the project LoRA become each clip's own stack."""
    import json
    (tmp_path / "chain.json").write_text(json.dumps({
        "image": "x.png", "settings": {"model": "m", "lora": "/l/a.safetensors", "strength": 0.7, "frames": 49},
        "clips": [{"prompt": "one", "seed": 1}, {"prompt": "two", "seed": 2}]}), encoding="utf-8")
    t = T.load(tmp_path)
    assert [n.prompt for n in t.path_to(t.selected)] == ["one", "two"]
    assert t.nodes["n2"].parent == "n1" and t.selected == "n2" and t.image == "x.png"
    assert t.options["prompt_format"] == "classic" and t.options["frames"] == 49
    assert t.nodes["n1"].overrides["loras"] == [{"path": "/l/a.safetensors", "strength": 0.7}]
    assert all(n.made is None for n in t.nodes.values())        # no video files there: nothing counts as made


def test_the_face_pictures_of_an_older_project_become_person1():
    old = _tree().to_json()
    del old["options"]["cast"]
    old["options"]["identity"] = {"images": ["refs/a.png", 5], "mode": "reference"}
    t = T.Tree.from_json(old)
    assert t.options["cast"] == [{"name": "Person1", "images": ["refs/a.png"], "desc": "", "lead": True}]
    assert "images" not in t.options["identity"] and "mode" not in t.options["identity"]
    old["options"]["identity"] = {"images": ["refs/a.png"], "mode": "off"}
    assert T.Tree.from_json(old).options["cast"][0]["lead"] is False


def test_remove_later_moves_the_files_only_after_the_tree_is_saved(tmp_path):
    """A crash between the move and the save must not leave tree.json listing clips whose files are in old/."""
    from video_chain import chain
    t = T.Tree()
    a = t.add(None, "a")
    T.save(tmp_path, t)
    (tmp_path / "nodes").mkdir()
    clip = tmp_path / "nodes" / f"{a.id}.mp4"
    clip.write_bytes(b"clip")
    a.made = {"prompt": "a", "seed": 1, "settings": {}, "video": f"nodes/{a.id}.mp4", "frame": f"nodes/{a.id}.png"}
    T.save(tmp_path, t)
    t.remove(a.id, tmp_path, later=True)
    assert clip.exists()                                   # not yet: tree.json still lists the clip
    T.save(tmp_path, t)
    assert not clip.exists() and (tmp_path / chain.OLD_DIR / "nodes" / clip.name).read_bytes() == b"clip"
    T.save(tmp_path, t)                                    # (nothing is moved twice)
