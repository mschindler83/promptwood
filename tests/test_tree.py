"""Branching tree with the fake generator (synthetic clips); no model."""
import json
import shutil

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
    assert [x["start"] for x in gen.calls] == ["start.png", f"{a.id}.png", f"{b1.id}.png"]
    assert statuses(project) == {"a": "done", "b1": "done", "c": "done", "b2": "new"}
    gen2 = run(project, [b2.id])                                        # the branch reuses a
    assert [x["prompt"] for x in gen2.calls] == ["b2"] and gen2.calls[0]["start"] == f"{a.id}.png"
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
    t.selected = c.id
    gone = t.remove(b1.id, project)
    assert set(gone) == {b1.id, c.id} and t.selected == a.id
    assert not T.node_files(project, c.id)[0].exists() and T.node_files(project, b2.id)[0].exists()
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
