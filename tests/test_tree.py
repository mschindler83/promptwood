"""Branching tree with the fake generator (synthetic clips); no model."""
import json
import shutil
from pathlib import Path

import pytest
from fakes import FakeGenerator
from PIL import Image

from video_chain import chain
from video_chain import tree as T

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


@pytest.fixture
def project(tmp_path):
    img = tmp_path / "in.jpg"
    Image.new("RGB", (96, 54), (250, 250, 250)).save(img)
    p = tmp_path / "proj"
    t = T.Tree()
    T.set_start_image(p.mkdir() or p, img, t)
    T.save(p, t)
    return p


def build(project):
    """start -> a -> (b1 -> c, b2)"""
    t = T.load(project)
    a = t.add(None, "a")
    b1 = t.add(a.id, "b1")
    b2 = t.add(a.id, "b2")
    c = t.add(b1.id, "c")
    T.save(project, t)
    return t, a, b1, b2, c


def statuses(project):
    t = T.load(project)
    ch = T.Checker(project, t)
    return {n.prompt: ch.status(n.id) for n in t.nodes.values()}


def run(project, targets, gen=None):
    gen = gen or FakeGenerator(frames=17)
    T.run_tree(gen, project, targets, log=lambda _m: None)
    return gen


def test_structure(project):
    t, a, b1, b2, c = build(project)
    assert [n.id for n in t.path_to(c.id)] == [a.id, b1.id, c.id]
    assert {n.id for n in t.children(a.id)} == {b1.id, b2.id}
    assert {n.id for n in t.leaves()} == {c.id, b2.id}
    assert len({n.seed for n in t.nodes.values()}) == 4               # every node its own seed
    t2 = T.Tree.from_json(json.loads(json.dumps(t.to_json())))
    assert t2.to_json() == t.to_json()


def test_generate_a_path_makes_only_that_path(project):
    t, a, b1, b2, c = build(project)
    gen = run(project, [c.id])
    assert [x["prompt"] for x in gen.calls] == ["a", "b1", "c"]
    starts = [x["start"] for x in gen.calls]
    assert starts[0] == "start.png" and starts[1].startswith(f"{a.id}_") and starts[2].startswith(f"{b1.id}_")
    assert statuses(project) == {"a": "done", "b1": "done", "c": "done", "b2": "new"}
    gen2 = run(project, [b2.id])                                        # the branch reuses a
    assert [x["prompt"] for x in gen2.calls] == ["b2"] and gen2.calls[0]["start"].startswith(f"{a.id}_")
    assert set(statuses(project).values()) == {"done"}


def test_a_change_touches_only_its_branch(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id, b2.id])
    t = T.load(project)
    t.nodes[b1.id].prompt = "b1 new"
    T.save(project, t)
    assert statuses(project) == {"a": "done", "b1 new": "changed", "c": "changed", "b2": "done"}
    gen = run(project, [c.id])
    assert [x["prompt"] for x in gen.calls] == ["b1 new", "c"]
    assert set(statuses(project).values()) == {"done"}


def test_changing_the_root_changes_everything(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id, b2.id])
    t = T.load(project)
    t.nodes[a.id].seed += 1
    T.save(project, t)
    assert set(statuses(project).values()) == {"changed"}
    gen = FakeGenerator(frames=17)
    gen.calls = [{}, {}]                     # other colours than the first run: a's new last frame differs
    run(project, [b2.id], gen)
    assert [x["prompt"] for x in gen.calls[2:]] == ["a", "b2"]
    assert statuses(project)["c"] == "changed" and statuses(project)["b1"] == "changed"


def test_identical_new_frame_keeps_children(project):
    """If a regenerated parent ends on exactly the same frame, its children's input is unchanged: still done."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    t.nodes[a.id].prompt = "a again"
    T.save(project, t)
    run(project, [a.id])                     # the fake makes the same frames again for the first call
    assert statuses(project) == {"a again": "done", "b1": "done", "c": "done", "b2": "new"}


def test_settings_and_start_image_count(project, tmp_path):
    t, a, b1, b2, c = build(project)
    run(project, [b2.id])
    t = T.load(project)
    t.options["steps"] = 20
    T.save(project, t)
    assert statuses(project)["a"] == "changed"
    t.options["steps"] = 28
    T.save(project, t)
    assert statuses(project)["a"] == "done"
    img2 = tmp_path / "other.png"
    Image.new("RGB", (96, 54), (10, 10, 10)).save(img2)
    T.set_start_image(project, img2, t)
    T.save(project, t)
    assert statuses(project)["a"] == "changed"


def test_join_path(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    out = T.join_path(project, t, c.id)
    assert out == project / "paths" / f"{c.id}.mp4" and chain.probe(out)["frames"] == 17 + 16 + 16
    mtime = out.stat().st_mtime_ns
    assert T.join_path(project, t, c.id).stat().st_mtime_ns == mtime           # unchanged: not joined again
    with pytest.raises(ValueError, match="not made yet"):
        T.join_path(project, t, b2.id)
    t.options["keep_repeated"] = True
    assert chain.probe(T.join_path(project, t, c.id))["frames"] == 51
    export = project.parent / "export.mp4"
    T.join_path(project, t, c.id, out=export)
    assert export.exists()


def test_remove_subtree_and_files(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id, b2.id])
    t = T.load(project)
    c_files = T.clip_files(project, t.nodes[c.id])
    t.selected = c.id
    gone = t.remove(b1.id, project)
    assert set(gone) == {b1.id, c.id} and t.selected == a.id
    assert not c_files[0].exists() and T.clip_files(project, t.nodes[b2.id])[0].exists()
    assert set(t.nodes) == {a.id, b2.id}


def test_stop_keeps_finished_nodes(project):
    t, a, b1, b2, c = build(project)

    class Dies(FakeGenerator):
        def generate(self, *args, **kw):
            if len(self.calls) == 2:
                raise KeyboardInterrupt
            super().generate(*args, **kw)
    with pytest.raises(KeyboardInterrupt):
        run(project, [c.id], Dies(frames=17))
    assert statuses(project) == {"a": "done", "b1": "done", "c": "new", "b2": "new"}


def test_empty_prompt_is_refused(project):
    t = T.load(project)
    n = t.add(None, "  ")
    T.save(project, t)
    with pytest.raises(ValueError, match="empty prompt"):
        run(project, [n.id])


def test_old_linear_chain_becomes_a_tree(tmp_path):
    img = tmp_path / "in.png"
    Image.new("RGB", (64, 64), (200, 0, 0)).save(img)
    out = tmp_path / "old"
    settings = chain.settings_dict(model="Comfy-Org/MiniMax-H3", lora=None, strength=1.0, frames=73, steps=28,
                                   width=512, height=512, audio=True)
    chain.run_chain(FakeGenerator(frames=17), img, ["one", "two"], out, settings=settings, log=lambda _m: None)
    t = T.load(out)
    assert [n.prompt for n in t.path_to(t.selected)] == ["one", "two"]
    ch = T.Checker(out, t)
    assert [ch.status(n.id) for n in t.path_to(t.selected)] == ["done", "done"]


def test_cli_tree_mode_without_loading_the_model(project, capsys):
    from video_chain.__main__ import main
    t, a, b1, b2, c = build(project)
    assert main(["--tree", str(project), "--targets", "n99", "--events"]) == 2
    assert "unknown node n99" in capsys.readouterr().out
    assert main(["--tree", str(project), "--events"]) == 2
    run(project, [c.id])
    capsys.readouterr()
    assert main(["--tree", str(project), "--targets", c.id, "--events"]) == 0     # all made: no model needed
    events = [json.loads(line[3:]) for line in capsys.readouterr().out.splitlines() if line.startswith("@@ ")]
    kinds = [e["e"] for e in events]
    assert "loading" not in kinds and kinds[-1] == "done" and events[-1]["made"] == []
    assert [e["state"] for e in events if e["e"] == "node"] == ["reused"] * 3


def test_node_overrides(project, tmp_path):
    lora_a, lora_b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    lora_a.write_bytes(b"x")
    lora_b.write_bytes(b"x")
    t, a, b1, b2, c = build(project)
    t = T.load(project)
    t.options["lora"] = str(lora_a)
    t.nodes[b1.id].overrides = {"lora": str(lora_b), "strength": 0.7, "frames": 124, "steps": 20}
    t.nodes[b2.id].overrides = {"lora": T.NO_LORA}
    T.save(project, t)
    t = T.load(project)
    ch = T.Checker(project, t)
    sa, sb1, sb2, sc = (ch.settings(n.id) for n in (a, b1, b2, c))
    assert (sa["lora"], sa["strength"], sa["frames"], sa["steps"]) == (str(lora_a), 1.0, 73, 28)
    assert (sb1["lora"], sb1["strength"], sb1["frames"], sb1["steps"]) == (str(lora_b), 0.7, 124, 20)
    assert sb2["lora"] is None and sb2["strength"] == 1.0
    assert sc["lora"] == str(lora_a)                                   # overrides are not inherited
    assert T.loras_needed(t, ch, [a.id, b1.id, b2.id, c.id]) == [str(lora_a), str(lora_b)]
    gen = run(project, [c.id, b2.id])
    used = {x["prompt"]: x["settings"] for x in gen.calls}
    assert used["b1"]["lora"] == str(lora_b) and used["b1"]["frames"] == 124 and used["b2"]["lora"] is None
    # changing one node's override changes it and what follows, nothing else
    t = T.load(project)
    t.nodes[b1.id].overrides["strength"] = 0.9
    T.save(project, t)
    assert statuses(project) == {"a": "done", "b1": "changed", "c": "changed", "b2": "done"}


def test_missing_lora_file_is_refused(project, tmp_path):
    t = T.load(project)
    n = t.add(None, "x")
    n.overrides = {"lora": str(tmp_path / "gone.safetensors")}
    T.save(project, t)
    with pytest.raises(FileNotFoundError, match="LoRA not found"):
        run(project, [n.id])


def test_lora_file_info(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from video_chain.h3 import lora_file_info
    peft = tmp_path / "peft.safetensors"                      # ai-toolkit format: no alpha -> scale 1
    save_file({"diffusion_model.blocks.0.attn.q.lora_A.weight": np.zeros((16, 64), np.float32),
               "diffusion_model.blocks.0.attn.q.lora_B.weight": np.zeros((64, 16), np.float32)}, str(peft))
    assert lora_file_info(peft) == (16, 1.0)
    kohya = tmp_path / "kohya.safetensors"                    # alpha 16 at rank 32 -> scale 0.5
    save_file({"lora_unet_b0_q.lora_down.weight": np.zeros((32, 64), np.float32),
               "lora_unet_b0_q.lora_up.weight": np.zeros((64, 32), np.float32),
               "lora_unet_b0_q.alpha": np.array(16.0, np.float32)}, str(kohya))
    assert lora_file_info(kohya) == (32, 0.5)
    mixed = tmp_path / "mixed.safetensors"
    save_file({"l1.lora_down.weight": np.zeros((8, 4), np.float32), "l1.alpha": np.array(8.0, np.float32),
               "l2.lora_down.weight": np.zeros((8, 4), np.float32), "l2.alpha": np.array(4.0, np.float32)},
              str(mixed))
    with pytest.raises(ValueError, match="different alpha"):
        lora_file_info(mixed)
    empty = tmp_path / "empty.safetensors"
    save_file({"x": np.zeros(2, np.float32)}, str(empty))
    with pytest.raises(ValueError, match="no LoRA weights"):
        lora_file_info(empty)


def test_retrained_lora_in_the_same_file_changes_its_clips(project, tmp_path):
    lora = tmp_path / "nose.safetensors"
    lora.write_bytes(b"weights v1")
    t, a, b1, b2, c = build(project)
    t = T.load(project)
    t.nodes[b1.id].overrides = {"lora": str(lora)}
    T.save(project, t)
    run(project, [c.id, b2.id])
    assert set(statuses(project).values()) == {"done"}
    lora.write_bytes(b"weights v2, retrained")                       # same path, new content
    assert statuses(project) == {"a": "done", "b1": "changed", "c": "changed", "b2": "done"}


def test_failed_regeneration_keeps_the_old_clip(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    old_file = T.clip_files(project, T.load(project).nodes[b1.id])[0]
    old = old_file.read_bytes()
    t = T.load(project)
    t.nodes[b1.id].prompt = "b1 new"
    T.save(project, t)

    class Dies(FakeGenerator):
        def generate(self, *args, **kw):
            raise RuntimeError("out of memory")
    with pytest.raises(RuntimeError):
        run(project, [c.id], Dies(frames=17))
    assert T.clip_files(project, T.load(project).nodes[b1.id])[0] == old_file and old_file.read_bytes() == old
    assert statuses(project) == {"a": "done", "b1 new": "changed", "c": "changed", "b2": "new"}
    run(project, [c.id])
    assert not list((project / "nodes").glob("*.new.*"))


def test_damaged_tree_files_are_refused(project):
    t, a, b1, b2, c = build(project)
    good = (project / T.TREE_FILE).read_text()
    (project / T.TREE_FILE).write_text("{ not json")
    with pytest.raises(ValueError, match="damaged"):
        T.load(project)
    d = json.loads(good)
    d["nodes"][0]["parent"] = d["nodes"][-1]["id"]                     # a loop
    (project / T.TREE_FILE).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="loop"):
        T.load(project)
    d = json.loads(good)
    d["nodes"][1]["parent"] = "n99"
    (project / T.TREE_FILE).write_text(json.dumps(d))
    with pytest.raises(ValueError, match="missing parent"):
        T.load(project)


def test_cli_checks_prompts_before_loading_the_model(project, capsys):
    from video_chain.__main__ import main
    t = T.load(project)
    n = t.add(None, "")
    T.save(project, t)
    assert main(["--tree", str(project), "--targets", n.id, "--events"]) == 2
    out = capsys.readouterr().out
    assert "empty prompt" in out and '"loading"' not in out


def test_start_image_conversion_failure_leaves_the_old_one(project, tmp_path):
    before = (project / T.START_FILE).read_bytes()
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(RuntimeError):
        T.set_start_image(project, bad, T.load(project))
    assert (project / T.START_FILE).read_bytes() == before
    assert not list(project.glob("*.part.png"))


def test_identity_references(project, tmp_path):
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    face = tmp_path / "face.jpg"
    Image.new("RGB", (64, 80), (180, 140, 120)).save(face)
    rel = T.add_reference(project, face, t)
    assert rel == "refs/ref1.png" and (project / rel).is_file()
    t.options["identity"]["mode"] = "reference"
    T.save(project, t)
    assert set(statuses(project).values()) == {"changed", "new"}               # references change every clip
    gen = run(project, [c.id])
    s = gen.calls[0]["settings"]
    assert s["refs"][0]["file"] == rel and s["ref_paths"] == [str(project / rel)]
    t = T.load(project)
    assert "ref_paths" not in t.nodes[a.id].made["settings"]                    # the record keeps names + hashes
    assert statuses(project)["c"] == "done"
    Image.new("RGB", (64, 80), (10, 10, 10)).save(project / rel)                 # another reference image
    assert statuses(project)["a"] == "changed"
    T.remove_reference(project, rel, t)
    assert not (project / rel).exists() and t.options["identity"]["images"] == []
    assert T.Tree().options["identity"] is not T.Tree().options["identity"]     # defaults are not shared


def test_stacked_loras(project, tmp_path):
    files = {}
    for n in ("a", "b", "c", "d"):
        files[n] = tmp_path / f"{n}.safetensors"
        files[n].write_bytes(n.encode())
    t, a, b1, b2, c = build(project)
    t = T.load(project)
    t.options["lora"], t.options["strength"] = str(files["a"]), 0.9
    t.options["extra_loras"] = [{"path": str(files["b"]), "strength": 0.5}]
    t.nodes[b1.id].overrides = {"extra_loras": [{"path": str(files["c"]), "strength": 0.3}]}     # on top
    t.nodes[b2.id].overrides = {"lora": str(files["d"]), "strength": 0.7}                         # own, no extras
    T.save(project, t)
    t = T.load(project)
    ch = T.Checker(project, t)
    sa, sb1, sb2 = ch.settings(a.id), ch.settings(b1.id), ch.settings(b2.id)
    assert (sa["lora"], [e["lora"] for e in sa["extra_loras"]]) == (str(files["a"]), [str(files["b"])])
    assert [e["lora"] for e in sb1["extra_loras"]] == [str(files["b"]), str(files["c"])]
    assert sb1["extra_loras"][1]["strength"] == 0.3
    assert sb2["lora"] == str(files["d"]) and "extra_loras" not in sb2
    assert T.loras_needed(t, ch, [a.id, b1.id, b2.id]) == [str(files[k]) for k in "abcd"]
    gen = run(project, [c.id, b2.id])
    assert {x["prompt"]: len(x["settings"].get("extra_loras", [])) for x in gen.calls} == {"a": 1, "b1": 2,
                                                                                            "c": 1, "b2": 0}
    files["c"].write_bytes(b"retrained")                                       # an extra LoRA counts by content
    assert statuses(project) == {"a": "done", "b1": "changed", "c": "changed", "b2": "done"}


def test_single_lora_record_unchanged(project, tmp_path):
    """Projects with one LoRA keep exactly their old settings record (their clips stay done)."""
    lo = tmp_path / "x.safetensors"
    lo.write_bytes(b"x")
    s = chain.settings_dict(model="m", lora=str(lo), strength=0.8, frames=73, steps=28, width=512, height=512,
                            audio=True, backend="comfyui", extra_loras=[])
    assert "extra_loras" not in s and set(s) == {"model", "lora", "lora_sha1", "strength", "frames", "steps",
                                                 "width", "height", "audio", "backend"}


def test_regenerating_never_overwrites_and_tolerates_open_files(project, monkeypatch):
    """SMB shares refuse to replace or delete a file that is open somewhere: a new clip gets new names, the old
    file is removed if possible, else on a later run."""
    t, a, b1, b2, c = build(project)
    run(project, [a.id])
    first = T.clip_files(project, T.load(project).nodes[a.id])
    t = T.load(project)
    t.nodes[a.id].prompt = "a again"
    T.save(project, t)
    real_unlink = Path.unlink
    busy = {first[0]}

    def unlink(self, missing_ok=False):
        if self in busy:
            raise PermissionError(13, "Permission denied", str(self))     # open in a player on the share
        return real_unlink(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", unlink)
    gen = FakeGenerator(frames=17)
    gen.calls = [{}]                                                      # other frames than the first version
    run(project, [a.id], gen)
    now = T.clip_files(project, T.load(project).nodes[a.id])
    assert now != first and all(f.exists() for f in now)
    assert first[0].exists() and not first[1].exists()                   # the open one stays, the other is gone
    assert statuses(project)["a again"] == "done"
    busy.clear()                                                          # closed again: the next run cleans up
    assert T.remove_stale_files(project, T.load(project)) == [] and not first[0].exists()


def test_stale_cleanup_also_takes_old_fixed_names(project):
    t, a, b1, b2, c = build(project)
    run(project, [a.id])
    d = project / "nodes"
    for name in ("n1.mp4", "n1.png", "n1.new.mp4", "n7_0123abcd.mp4"):
        (d / name).write_bytes(b"old")
    keep = d / "notes.txt"
    keep.write_text("not ours")
    T.remove_stale_files(project, T.load(project))
    assert sorted(f.name for f in d.iterdir()) == sorted([f.name for f in T.clip_files(project,
                                                          T.load(project).nodes[a.id])] + ["notes.txt"])


def test_turbo_mode(project, tmp_path):
    turbo = tmp_path / "turbo.safetensors"
    turbo.write_bytes(b"t")
    t, a, b1, b2, c = build(project)
    t = T.load(project)
    t.nodes[b1.id].overrides = {"steps": 20}
    t.options["turbo"] = {"on": True, "lora": str(turbo), "steps": 6, "strength": 1.0}
    T.save(project, t)
    t = T.load(project)
    ch = T.Checker(project, t)
    for n in (a, b1, c):
        st = ch.settings(n.id)
        assert st["steps"] == 6 and st["turbo"]["lora"] == str(turbo)            # also over a clip's own steps
    assert str(turbo) in T.loras_needed(t, ch, [a.id])
    run(project, [c.id])
    t = T.load(project)
    t.options["turbo"]["on"] = False
    T.save(project, t)
    assert statuses(project)["a"] == "changed"
    t.options["turbo"]["on"] = True
    T.save(project, t)
    assert statuses(project)["a"] == "done"
