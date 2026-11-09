"""Branching tree with the fake generator (synthetic clips); no model."""
import json
import shutil
import subprocess
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
    t.nodes[a.id].overrides = {"loras": [{"path": str(lora_a), "strength": 1.0}]}
    t.nodes[b1.id].overrides = {"loras": [{"path": str(lora_b), "strength": 0.7}, {"path": str(lora_a),
                                                                                    "strength": 0.4}],
                                "frames": 124, "steps": 20}
    T.save(project, t)
    t = T.load(project)
    ch = T.Checker(project, t)
    sa, sb1, sb2, sc = (ch.settings(n.id) for n in (a, b1, b2, c))
    assert (sa["lora"], sa["strength"], sa["frames"], sa["steps"]) == (str(lora_a), 1.0, 73, 28)
    assert (sb1["lora"], sb1["strength"], sb1["frames"], sb1["steps"]) == (str(lora_b), 0.7, 124, 20)
    assert sb1["extra_loras"][0]["lora"] == str(lora_a) and sb1["extra_loras"][0]["strength"] == 0.4
    assert sb2["lora"] is None and sb2["strength"] == 1.0                # no LoRA by default
    assert sc["lora"] is None                                           # LoRAs are not inherited
    assert T.loras_needed(t, ch, [a.id, b1.id, b2.id, c.id]) == [str(lora_a), str(lora_b)]
    gen = run(project, [c.id, b2.id])
    used = {x["prompt"]: x["settings"] for x in gen.calls}
    assert used["b1"]["lora"] == str(lora_b) and used["b1"]["frames"] == 124 and used["b2"]["lora"] is None
    # changing one clip's LoRA strength changes it and what follows, nothing else
    t = T.load(project)
    t.nodes[b1.id].overrides["loras"][1]["strength"] = 0.9
    T.save(project, t)
    assert statuses(project) == {"a": "done", "b1": "changed", "c": "changed", "b2": "done"}


def test_missing_lora_file_is_refused(project, tmp_path):
    t = T.load(project)
    n = t.add(None, "x")
    n.overrides = {"loras": [{"path": str(tmp_path / "gone.safetensors"), "strength": 1.0}]}
    T.save(project, t)
    with pytest.raises(FileNotFoundError, match="LoRA not found"):
        run(project, [n.id])


def test_retrained_lora_in_the_same_file_changes_its_clips(project, tmp_path):
    lora = tmp_path / "nose.safetensors"
    lora.write_bytes(b"weights v1")
    t, a, b1, b2, c = build(project)
    t = T.load(project)
    t.nodes[b1.id].overrides = {"loras": [{"path": str(lora), "strength": 1.0}]}
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


def face(path, color):
    Image.new("RGB", (64, 80), color).save(path)
    return path


def test_identity_references(project, tmp_path):
    """One face picture without any @mention: the first character is the lead, every clip uses its pictures."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    rel = T.add_reference(project, face(tmp_path / "face.jpg", (180, 140, 120)), t)
    assert rel == "refs/ref1.png" and (project / rel).is_file()
    assert T.cast_of(t) == [{"name": "Person1", "images": [rel], "desc": "", "lead": True}]
    T.save(project, t)
    assert set(statuses(project).values()) == {"changed", "new"}               # references change every clip
    gen = run(project, [c.id])
    s = gen.calls[0]["settings"]
    assert s["refs"][0]["file"] == rel and s["ref_paths"] == [str(project / rel)] and "cast" not in s
    assert s["start_pin"] == chain.START_PIN == 5
    assert gen.calls[0]["prompt"] == "a"                                       # the generator adds its sentence
    t = T.load(project)
    assert "ref_paths" not in t.nodes[a.id].made["settings"]                    # the record keeps names + hashes
    assert statuses(project)["c"] == "done"
    face(project / rel, (10, 10, 10))                                          # another reference image
    assert statuses(project)["a"] == "changed"
    T.remove_reference(project, rel, t)
    assert not (project / rel).exists() and T.cast_of(t)[0]["images"] == []
    assert T.Tree().options["cast"] is not T.Tree().options["cast"]             # defaults are not shared


def test_old_face_list_becomes_the_lead_and_clips_stay_done(project, tmp_path):
    t, a, b1, b2, c = build(project)
    T.add_reference(project, face(tmp_path / "f.jpg", (180, 140, 120)), t)
    T.add_picture(project, t, "Person1", face(tmp_path / "g.jpg", (90, 60, 50)))
    T.save(project, t)
    run(project, [c.id])
    assert set(statuses(project).values()) == {"done", "new"}
    raw = json.loads((project / T.TREE_FILE).read_text())
    images = raw["options"].pop("cast")[0]["images"]
    for mode, want in (("reference", {"done", "new"}), ("off", {"changed", "new"})):
        raw["options"]["identity"] = {"mode": mode, "images": images, "size": "match", "hint": True}
        (project / T.TREE_FILE).write_text(json.dumps(raw))
        t = T.load(project)
        assert T.cast_of(t) == [{"name": "Person1", "images": images, "desc": "", "lead": mode == "reference"}]
        assert t.options["identity"] == {"size": "match", "hint": True, "language": "English"}
        assert set(statuses(project).values()) == want                        # off: the clips had no faces


def test_cast_mentions(project, tmp_path):
    t = T.load(project)
    T.add_character(project, t, "Anna", face(tmp_path / "a1.jpg", (200, 150, 130)), desc="woman, 30s, grey sweater")
    T.add_picture(project, t, "anna", face(tmp_path / "a2.jpg", (190, 140, 120)))
    T.add_character(project, t, "Ben", face(tmp_path / "b.jpg", (90, 70, 60)))
    with pytest.raises(ValueError, match="already"):
        T.add_character(project, t, "ben")
    with pytest.raises(ValueError, match="no spaces"):
        T.add_character(project, t, "Anna B")
    both = t.add(None, "@Anna sits at the table. @Ben walks in and sits opposite @anna.")
    ben = t.add(None, "@Ben looks out of the window.")
    nobody = t.add(None, "An empty kitchen.")
    T.save(project, t)
    use = T.clip_cast(t.options, both.prompt)
    assert use["mode"] == "mention" and use["images"] == ["refs/ref1.png", "refs/ref2.png", "refs/ref3.png"]
    assert use["prompt"].startswith("Anna (the person in <Picture 1> and <Picture 2>) sits at the table. Ben (the "
                                    "person in <Picture 3>) walks in and sits opposite Anna. ")
    assert "Ben has exactly the face of the person in <Picture 3>." in use["prompt"]
    assert "Anna: woman, 30s, grey sweater." in use["prompt"]
    assert T.clip_cast(t.options, ben.prompt)["images"] == ["refs/ref3.png"]   # renumbered: Ben is <Picture 1>
    assert "Ben (the person in <Picture 1>)" in T.clip_cast(t.options, ben.prompt)["prompt"]
    assert T.clip_cast(t.options, nobody.prompt) == {"mode": "lead", "characters": [T.character(t, "Anna")],
                                                     "images": ["refs/ref1.png", "refs/ref2.png"],
                                                     "voices": [], "prompt": "An empty kitchen.", "record": None}
    gen = run(project, [both.id, ben.id, nobody.id])
    by = {c["prompt"][:4]: c for c in gen.calls}
    assert by["Ben "]["settings"]["ref_paths"] == [str(project / "refs/ref3.png")]
    assert by["Ben "]["settings"]["cast"] == [{"name": "Ben", "desc": ""}]
    assert set(statuses(project).values()) == {"done"}
    t = T.load(project)
    T.add_character(project, t, "Cara", face(tmp_path / "c.jpg", (30, 30, 30)))        # a new one changes nothing
    T.update_character(t, "Ben", desc="man, 40s")                                        # Ben's clips only
    T.save(project, t)
    st = statuses(project)
    assert st[both.prompt] == st[ben.prompt] == "changed" and st[nobody.prompt] == "done"
    T.update_character(t, "Ben", lead=True)                                              # one lead only
    assert [c["name"] for c in T.cast_of(t) if c["lead"]] == ["Ben"]
    assert T.rename_character(t, "Ben", "Tom") == [both.id, ben.id]
    assert t.nodes[ben.id].prompt == "@Tom looks out of the window." and T.character(t, "Ben") is None
    T.remove_character(project, t, "Cara")
    assert not (project / "refs/ref4.png").exists()


def test_cast_problems(project, tmp_path):
    t = T.load(project)
    T.add_character(project, t, "Ben", face(tmp_path / "b.jpg", (90, 70, 60)))
    T.add_character(project, t, "Nobody")
    a = t.add(None, "@Bem waves.")
    b = t.add(None, "The man from <Image 2> waves.")
    c = t.add(None, "The man from <Picture 3> waves.")
    d = t.add(None, "@Nobody waves.")
    e = t.add(None, "@Ben waves.")
    T.save(project, t)
    probs = T.cast_problems(project, t, [a.id, b.id, c.id, d.id, e.id])
    assert probs == [f"clip #{a.id[1:]} mentions @Bem, but there is no such character — did you mean @Ben?",
                     f"clip #{b.id[1:]} writes <Image 2>; the model only knows <Picture 2> (or mention the "
                     "character as @Name)",
                     f"clip #{c.id[1:]} names <Picture 3>, but the cast has only 1 face picture",
                     f"@Nobody (clip #{d.id[1:]}) has no face picture or voice yet"]
    (project / "refs/ref1.png").unlink()
    assert T.cast_problems(project, t, [e.id]) == [f"a face picture of clip #{e.id[1:]} is missing in refs/"]
    with pytest.raises(ValueError, match="@Bem"):
        run(project, [a.id])


def sound(path, secs, video=False):
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=330:duration={secs}"]
    if video:
        cmd += ["-f", "lavfi", "-i", f"color=gray:s=64x48:d={secs}", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True)
    return path


def test_sound_references(project, tmp_path):
    """<Audio N> in a clip's prompt: only those clips record (and get) the sounds, up to the highest one named."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id, b2.id])
    t = T.load(project)
    one = T.add_audio_reference(project, sound(tmp_path / "voice.mp3", 6), t)
    assert one["file"] == "refs/audio1.wav" and one["name"] == "voice.mp3" and not one["cut"]
    assert abs(one["secs"] - 6) < 0.1 and (project / one["file"]).is_file()
    two = T.add_audio_reference(project, sound(tmp_path / "clip.mp4", 12, video=True), t, name="clip sound")
    assert two["cut"] and abs(two["secs"] - 9) < 0.1                          # 15 s together at most
    with pytest.raises(ValueError, match="15 s"):
        T.add_audio_reference(project, sound(tmp_path / "beat.wav", 3), t)
    T.save(project, t)
    assert set(statuses(project).values()) == {"done"}                          # nobody names a sound yet
    t = T.load(project)
    t.nodes[b1.id].prompt = "b1, she speaks in the voice of <Audio 2>"
    T.save(project, t)
    st = statuses(project)
    assert st["b1, she speaks in the voice of <Audio 2>"] == "changed" and st["b2"] == "done" and st["a"] == "done"
    s = T.Checker(project, T.load(project)).settings(b1.id)
    assert [x["file"] for x in s["audio_refs"]] == ["refs/audio1.wav", "refs/audio2.wav"]     # 1 too: numbered
    gen = run(project, [c.id])
    call = next(x["settings"] for x in gen.calls if x["prompt"].startswith("b1"))
    assert call["audio_paths"] == [str(project / "refs/audio1.wav"), str(project / "refs/audio2.wav")]
    assert "audio_paths" not in T.load(project).nodes[b1.id].made["settings"]
    assert all("audio_refs" not in x["settings"] for x in gen.calls if not x["prompt"].startswith("b1"))
    sound(project / "refs/audio2.wav", 4)                                       # another sound in its place
    assert statuses(project)["b1, she speaks in the voice of <Audio 2>"] == "changed"
    t = T.load(project)
    t.nodes[b2.id].prompt = "b2 with <Audio 3>"
    assert T.audio_problems(project, t, [b2.id]) == ["clip #" + b2.id[1:] + " names <Audio 3>, but there are only 2 "
                                                      "sound references"]
    T.remove_audio_reference(project, "refs/audio1.wav", t)
    assert not (project / "refs/audio1.wav").exists() and [x["file"] for x in T.audio_refs(t)] == ["refs/audio2.wav"]
    with pytest.raises(ValueError, match="no sound"):
        T.add_audio_reference(project, tmp_path / "in.jpg", t)
    with pytest.raises(ValueError, match="at least 2 s"):
        T.add_audio_reference(project, sound(tmp_path / "short.wav", 1), t)
    assert T.audio_mentioned("x <audio 1> and <Audio  3>") == 3 and T.audio_mentioned("none") == 0


def test_old_project_loras_become_each_clips_own(project, tmp_path):
    """Older trees had a project LoRA (+ extra LoRAs) and per-clip overrides; on load every clip gets the stack it
    was made with, so its clips stay done."""
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
    t = T.load(project)
    assert "lora" not in t.options and "extra_loras" not in t.options
    assert t.nodes[b1.id].overrides["loras"] == [{"path": str(files["a"]), "strength": 0.9},
                                                 {"path": str(files["b"]), "strength": 0.5},
                                                 {"path": str(files["c"]), "strength": 0.3}]
    T.save(project, t)                                                         # the new form: still done
    assert set(statuses(project).values()) == {"done"}
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


def test_old_records_count_as_ai_toolkit(project):
    """Clips made by the ai-toolkit generator of early versions (records without a backend, or with it) count as
    changed: ComfyUI samples differently and makes them again. A stored backend option is dropped."""
    t, a, b1, b2, c = build(project)
    run(project, [a.id])
    d = json.loads((project / T.TREE_FILE).read_text())
    d["options"]["backend"] = "ai-toolkit"
    for nd in d["nodes"]:
        if nd.get("made"):
            nd["made"]["settings"].pop("backend")
    (project / T.TREE_FILE).write_text(json.dumps(d))
    t = T.load(project)
    assert "backend" not in t.options
    assert statuses(project)["a"] == "changed"


def test_records_never_point_outside_the_project(project, tmp_path):
    t, a, b1, b2, c = build(project)
    victim = tmp_path / "precious.txt"
    victim.write_text("keep me")
    t = T.load(project)
    t.nodes[a.id].made = {"video": "../precious.txt", "frame": str(victim)}
    assert T.clip_files(project, t.nodes[a.id]) == []
    t.remove(a.id, project)
    assert victim.read_text() == "keep me"


def test_undo_changes_of_a_made_clip(project, tmp_path):
    t, a, b1, b2, c = build(project)
    end = tmp_path / "end.png"
    Image.new("RGB", (40, 30), (255, 255, 0)).save(end)
    made_end = T.set_end_image(project, t, a.id, end)
    T.save(project, t)
    run(project, [c.id])
    t = T.load(project)
    a = t.nodes[a.id]
    assert T.changed_inputs(t, a.id) == []
    a.prompt, a.seed, a.overrides = "a typo", a.seed + 1, {"frames": 25}
    T.set_end_image(project, t, a.id, end)                    # another copy: the made one is kept for undo
    assert (project / made_end).is_file()
    ch = T.Checker(project, t)
    assert ch.status(a.id) == "changed" and ch.why(a.id) == "its prompt and seed and settings and end image changed"
    assert ch.why(c.id) == f"clip #{b1.id[1:]} before it changes"
    newer = a.end_image
    assert T.revert(project, t, a.id) == ["prompt", "seed", "settings", "end image"]
    assert (a.prompt, a.overrides, a.end_image) == ("a", {}, made_end) and not (project / newer).exists()
    ch = T.Checker(project, t)
    assert [ch.status(n) for n in (a.id, b1.id, c.id)] == ["done"] * 3 and ch.why(a.id) == ""
    # a project setting is not undone here: the clip stays changed and says why
    t.options["steps"] = 9
    a.prompt = "x"
    T.revert(project, t, a.id)
    assert T.Checker(project, t).why(a.id) == "its settings changed (the project's or its own)"
    # after making it again with another end image, the old copy goes
    t.options["steps"] = 28
    T.clear_end_image(project, t, a.id)
    assert (project / made_end).is_file()                    # still the made one
    T.save(project, t)
    run(project, [a.id])
    assert not (project / made_end).exists()


def test_clip_states_do_not_read_every_frame_again(project, monkeypatch):
    """The states are checked after every edit: the frames are hashed and the start image probed once per version
    of the file, not on every check (slow on a network share)."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id, b2.id])
    assert set(statuses(project).values()) == {"done"}
    reads, probes = [], []
    real_sha, real_size = chain.sha1_file, chain.image_size
    monkeypatch.setattr(chain, "sha1_file", lambda p: reads.append(p) or real_sha(p))
    monkeypatch.setattr(chain, "image_size", lambda p: probes.append(p) or real_size(p))
    for _ in range(3):
        assert set(statuses(project).values()) == {"done"}
    assert reads == [] and probes == []
    T.set_start_image(project, project / "nodes" / Path(T.load(project).nodes[c.id].made["frame"]).name,
                      T.load(project))                              # another start image: probed and hashed once
    statuses(project)
    statuses(project)
    assert len(probes) == 1 and len([r for r in reads if Path(r).name == T.START_FILE]) <= 1


def test_hand_over_at_an_earlier_frame(project):
    """The clips after a clip start from the frame it hands over at; it stays done, the frames after that frame
    are left out of the joined path, and the frame file goes with its clip version."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    assert T.made_frames(t.nodes[a.id]) == 17                      # the real count of the made clip
    t.nodes[a.id].end_at = 5
    h = T.ensure_handover(project, t, t.nodes[a.id])
    assert h.is_file() and h.name.endswith("_f5.png")
    T.save(project, t)
    st = statuses(project)
    assert st["a"] == "done" and st["b1"] == "changed" and st["c"] == "changed"
    gen = run(project, [c.id])
    assert [x["prompt"] for x in gen.calls] == ["b1", "c"] and gen.calls[0]["start"] == h.name
    t = T.load(project)
    path = t.path_to(c.id)
    assert T.path_cuts(t, path) == [(0, 11), (1, 0), (1, 0)]       # a: frames 6..16 left out
    joined = T.join_path(project, t, c.id)
    assert chain.probe(joined)["frames"] == 6 + 16 + 16
    assert not T.remove_stale_files(project, t) and h.is_file()   # kept: it is in use
    # a missing hand-over frame is made again before the worker plans (its children stay done)
    h.unlink()
    from video_chain.__main__ import prepare_job
    assert prepare_job(project, [c.id])["todo"] == [] and h.is_file()
    # another frame of the same take: the earlier frame file goes
    t.nodes[a.id].end_at = 7
    h7 = T.ensure_handover(project, t, t.nodes[a.id])
    assert h7.is_file() and not h.exists()
    # the clip made again: a new take hands over at its last frame again, the old frame file goes
    t.nodes[a.id].prompt = "a2"
    T.save(project, t)
    run(project, [a.id])
    t = T.load(project)
    assert not h7.exists() and t.nodes[a.id].end_at is None and T.handover_file(project, t, t.nodes[a.id]) is None
    ch = T.Checker(project, t)
    assert ch.start_frame(t.nodes[b1.id]) == project / t.nodes[a.id].made["frame"]


def test_hand_over_with_a_motion_guide(project):
    t, a, b1, b2, c = build(project)
    t.nodes[b1.id].overrides = {"guide": 5}
    T.save(project, t)
    run(project, [b1.id])
    t = T.load(project)
    t.nodes[a.id].end_at = 8
    T.save(project, t)
    ch = T.Checker(project, t)
    assert ch.guide_from(t.nodes[b1.id]).endswith("@8")
    assert T.guide_problem(t, ch, [b1.id]) is None
    t.nodes[a.id].end_at = 3                                        # 4 frames < the 5-frame guide
    assert "shorter than the 5-frame motion guide" in T.guide_problem(t, T.Checker(project, t), [b1.id])


def test_style_text_around_every_prompt(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    t.options["style"] = {"prefix": "cinematic,", "suffix": "soft light"}
    T.save(project, t)
    assert set(statuses(project).values()) == {"changed", "new"}
    gen = run(project, [b1.id])
    assert [x["prompt"] for x in gen.calls] == ["cinematic, a soft light", "cinematic, b1 soft light"]
    made = T.load(project).nodes[a.id].made
    assert made["prompt"] == "a" and made["settings"]["prompt_prefix"] == "cinematic,"
    t = T.load(project)
    t.options["style"] = {"prefix": "", "suffix": ""}               # back: a and b1 (made with it) change
    T.save(project, t)
    assert statuses(project)["a"] == "changed" and "prompt_prefix" not in T.Checker(project, t).settings(c.id)


def test_hyperflow_speed(project):
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    t.options["hyperflow"] = {"on": True, "strength": 1.0}
    t.nodes[b1.id].overrides = {"steps": 40}                      # a clip's own steps do not apply
    T.save(project, t)
    ch = T.Checker(project, t)
    s = ch.settings(b1.id)
    assert s["steps"] == 8 and s["hyperflow"] == {"strength": 1.0} and ch.settings(a.id)["steps"] == 8
    assert statuses(project) == {"a": "changed", "b1": "changed", "c": "changed", "b2": "new"}   # another sampler
    t.options["hyperflow"]["on"] = False
    t.nodes[b1.id].overrides = {}
    T.save(project, t)
    assert statuses(project) == {"a": "done", "b1": "done", "c": "done", "b2": "new"}


def test_export_a_path_refined(project, tmp_path):
    t, a, b1, b2, c = build(project)
    t.options["style"] = {"prefix": "cinematic,", "suffix": ""}
    T.save(project, t)
    gen = run(project, [b1.id])
    t = T.load(project)
    t.nodes[a.id].end_at = 10                                      # a hands over at frame 10
    T.ensure_handover(project, t, t.nodes[a.id])
    T.save(project, t)
    run(project, [b1.id], gen)
    out = tmp_path / "out2k.mp4"
    T.refine_path(gen, project, b1.id, "2k", out, log=lambda _m: None)
    assert [r["prompt"] for r in gen.refines] == ["cinematic, a", "cinematic, b1"]
    # pinned: the first clip at the start image, the next at the refined hand-over frame of the clip before
    assert gen.refines[0]["first"] == T.START_FILE and gen.refines[1]["first"] == f"{b1.id}_first.png"
    t = T.load(project)
    w, h = gen.refines[0]["size"]
    info = chain.probe(out)
    assert (info["width"], info["height"]) == (w, h) and info["frames"] == 11 + 16   # the same cuts as made size
    assert all(T.refined_file(project, t.nodes[i], "2k").is_file() for i in (a.id, b1.id))
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "again.mp4", log=lambda _m: None)
    assert len(gen.refines) == 2                                   # refined copies reused
    assert not T.remove_stale_files(project, t) and T.refined_file(project, t.nodes[a.id], "2k").is_file()
    old = T.refined_file(project, t.nodes[b1.id], "2k")
    t.nodes[b1.id].prompt = "b1 again"
    T.save(project, t)
    run(project, [b1.id], gen)                                     # a new take: its old refined copy goes
    assert not old.exists()
    with pytest.raises(ValueError, match="not made yet"):
        T.refine_path(gen, project, c.id, "2k", tmp_path / "x.mp4", log=lambda _m: None)


def test_refined_export_with_fast_attention_makes_its_own_copies(project, tmp_path):
    t, a, b1, b2, c = build(project)
    gen = run(project, [b1.id])
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "exact.mp4", log=lambda _m: None)
    assert [r["sparse"] for r in gen.refines] == [None, None]
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "fast.mp4", log=lambda _m: None, fast_attention=True)
    assert [r["sparse"] for r in gen.refines[2:]] == [1.3, 1.3]                 # made again with Sol-Attn
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "fast2.mp4", log=lambda _m: None, fast_attention=True)
    assert len(gen.refines) == 4                                                # and then reused


def test_pika_speed(project):
    t, a, b1, b2, c = build(project)
    t.options["pika"] = {"on": True, "strength": 1.0}
    t.nodes[b1.id].overrides = {"steps": 40}
    ch = T.Checker(project, t)
    assert ch.settings(b1.id)["steps"] == 6 and ch.settings(b1.id)["pika"] == {"strength": 1.0}
    t.options["hyperflow"] = {"on": True, "strength": 1.0}                 # one speed mode at a time
    assert "pika" not in T.Checker(project, t).settings(a.id) and T.Checker(project, t).settings(a.id)["steps"] == 8


def test_dialogue_speakers():
    o = {"cast": [{"name": "Anna", "images": [], "lead": True}, {"name": "Ben", "images": []}]}
    text = '@Anna looks at @Ben and says: "Morning." @Ben smiles. "Hi," says @Ben. „Kaffee?“ @Anna nods. «Oui»'
    got = [(d["line"], d["speaker"]["name"] if d["speaker"] else None) for d in T.dialogue(o, text)]
    # first @Name of the sentence; else the one after the line in its sentence; else the last one before it
    assert got == [("Morning.", "Anna"), ("Hi,", "Ben"), ("Kaffee?", "Anna"), ("Oui", "Anna")]
    assert [d["speaker"] for d in T.dialogue(o, 'She says "hi".')] == [None]
    assert T.dialogue(o, 'She says "hi".', default=T.character(o, "Anna"))[0]["speaker"]["name"] == "Anna"
    assert T.dialogue(o, "<d>[German] Hallo</d> says @Ben")[0]["line"] == "Hallo"
    # "she" / "he" before a line: the last-mentioned character of that gender (from the description)
    o = {"cast": [{"name": "Anna", "images": [], "desc": "woman, 30s"}, {"name": "Ben", "images": [], "desc": "a man"}]}
    text = '@Anna and @Ben sit down; she says "good morning" and he answers "coffee?" @Ben looks at her and says "Hi."'
    assert [d["speaker"]["name"] for d in T.dialogue(o, text)] == ["Anna", "Ben", "Ben"]
    assert T.gender_of({"desc": "girl"}) == "f" and T.gender_of({"desc": "robot"}) is None


def test_cast_voices(project, tmp_path):
    """A character's voice: used (MiniMax's structured format, after the sounds the prompt names) only in clips
    where it says a line; such clips record the voice and the speech language, and hold their start image."""
    t = T.load(project)
    T.add_character(project, t, "Anna", face(tmp_path / "a.jpg", (200, 150, 130)), desc="woman, 30s")
    T.add_character(project, t, "Ben", face(tmp_path / "b.jpg", (90, 70, 60)))
    v = T.set_voice(project, t, "Anna", sound(tmp_path / "anna.mp4", 10, video=True))
    assert v["file"] == "refs/voice1.wav" and v["cut"] and abs(v["secs"] - T.VOICE_MAX_SECS) < 0.1
    with pytest.raises(ValueError, match="at least 2 s"):
        T.set_voice(project, t, "Ben", sound(tmp_path / "short.wav", 1))
    talk = t.add(None, '@Ben walks in. @Anna looks up and says "Good morning."')
    quiet = t.add(None, "@Anna drinks her coffee.")
    lead = t.add(None, 'She smiles. "Hello."')                     # nobody mentioned: the lead (Anna) speaks
    T.save(project, t)
    use = T.clip_cast(t.options, talk.prompt)
    assert use["mode"] == "voices" and use["voices"] == ["refs/voice1.wav"] and use["language"] == "English"
    assert use["images"] == ["refs/ref2.png", "refs/ref1.png"]                 # Ben is mentioned first
    p = use["prompt"]
    assert p.startswith("subject_definitions:\n<Subject 1> is Ben, the person in <Picture 1>.\n<Subject 2> is "
                        "Anna (woman, 30s), the person in <Picture 2>.\n<Audio 1> is the voice-timbre reference "
                        "for <Subject 2> (S1)")
    assert ("[Shot 1] <Subject 1> walks in. <Subject 2> (S1) looks up and says, using the voice timbre referenced "
            "from <Audio 1>, <d>[English] Good morning.</d>") in p
    assert "@" not in p and "non_diegetic_music:\nN/A" in p
    assert T.clip_cast(t.options, quiet.prompt)["mode"] == "mention"           # no line: no voice
    lu = T.clip_cast(t.options, lead.prompt)
    assert lu["mode"] == "voices" and lu["images"] == ["refs/ref1.png"] and "<Subject 1> (S1) says, using" in lu["prompt"]
    gen = run(project, [talk.id, quiet.id, lead.id])
    call = next(c["settings"] for c in gen.calls if c["prompt"].startswith("subject_definitions") and "Ben" in c["prompt"])
    assert call["audio_paths"] == [str(project / "refs/voice1.wav")] and call["voice_language"] == "English"
    assert call["start_pin"] == 5 and call["cast"] == [{"name": "Ben", "desc": ""}, {"name": "Anna", "desc": "woman, 30s"}]
    q = next(c["settings"] for c in gen.calls if c["prompt"].startswith("Anna"))
    assert "audio_refs" not in q and "voice_language" not in q
    assert set(statuses(project).values()) == {"done"}
    t = T.load(project)
    t.options["identity"]["language"] = "German"                               # the voiced clips only
    T.save(project, t)
    st = statuses(project)
    assert st[talk.prompt] == st[lead.prompt] == "changed" and st[quiet.prompt] == "done"
    t.options["identity"]["language"] = "English"
    T.set_voice(project, t, "Anna", sound(tmp_path / "other.wav", 4))         # a new voice: those clips again
    assert not (project / "refs/voice1.wav").exists() and T.character(t, "Anna")["voice"]["file"] == "refs/voice2.wav"
    T.save(project, t)
    st = statuses(project)
    assert st[talk.prompt] == "changed" and st[quiet.prompt] == "done"
    # the sounds a clip needs: the <Audio N> it names first, then the voices; at most 3 and 15 s
    T.add_audio_reference(project, sound(tmp_path / "s1.wav", 6), t)
    T.add_audio_reference(project, sound(tmp_path / "s2.wav", 6), t)
    T.set_voice(project, t, "Ben", sound(tmp_path / "ben.wav", 6))
    both = t.add(None, '<Audio 2> is the soundtrack. @Anna says "Hi." @Ben says "Hey."')
    use = T.clip_cast(t.options, both.prompt)
    assert use["voices"] == ["refs/voice2.wav", "refs/voice1.wav"] and "<Audio 3> is the voice-timbre" in use["prompt"]
    assert "<Audio 4> is the voice-timbre reference for <Subject 2> (S2)" in use["prompt"]
    assert T.cast_problems(project, t, [both.id]) == [f"clip #{both.id[1:]} needs 2 voices and 2 sounds; the model "
                                                      "takes at most 3 sounds"]
    t.nodes[both.id].prompt = '<Audio 1> is the soundtrack. @Anna says "Hi." @Ben says "Hey."'
    assert T.cast_problems(project, t, [both.id]) == [f"clip #{both.id[1:]} needs 2 voices and 1 sound of 16.0 s "
                                                      "together; the model takes at most 15 s (shorten a voice: set "
                                                      "it again from a shorter part)"]
    T.remove_voice(project, t, "Ben")
    assert not (project / "refs/voice1.wav").exists() and T.cast_problems(project, t, [both.id]) == []
    T.remove_character(project, t, "Anna")
    assert not (project / "refs/voice2.wav").exists()


def test_refine_problems():
    a = '@Ben and @Anna stand at the window. He points: "Look, it is snowing!" She laughs: "Finally."'
    assert T.refine_problems(a, '@Ben and @Anna stand there. @Ben says “Look, it is snowing!” @Anna: "Finally!"') == []
    assert T.refine_problems(a, '@Ben and Anna stand there. Anna laughs: "Finally." "Finally."') == [
        "the line \u201cLook, it is snowing!\u201d is gone or changed", "the line \u201cFinally.\u201d is there 2 times",
        "@Anna is no longer mentioned"]


def test_face_crop(project, tmp_path):
    from video_chain import faces as F
    t = T.load(project)
    src = tmp_path / "wide.png"
    Image.new("RGB", (300, 200), (90, 80, 70)).save(src)
    rel = T.add_character(project, t, "Anna", src, crop=(50, 20, 120))["images"][0]
    assert Image.open(project / rel).size == (120, 120) and Image.open(project / T.source_of(rel)).size == (300, 200)
    T.recrop_picture(project, t, rel, (0, 0, 200))
    assert Image.open(project / rel).size == (200, 200)
    T.recrop_picture(project, t, rel, None)                                    # back to the whole picture
    assert Image.open(project / rel).size == (300, 200)
    plain = T.add_picture(project, t, "Anna", src)                             # uncropped: no full copy ...
    assert not (project / T.source_of(plain)).exists()
    T.recrop_picture(project, t, plain, (10, 10, 50))                          # ... it is its own full picture
    assert Image.open(project / plain).size == (50, 50) and (project / T.source_of(plain)).is_file()
    T.remove_character(project, t, "Anna")
    assert not list((project / "refs").glob("ref*"))
    # the square around a face: margin, the face a little above the middle, inside the picture
    assert F.face_square((100, 50, 140, 110), 400, 300) == (75, 38, 90)
    assert F.face_square((0, 0, 40, 60), 400, 300) == (0, 0, 90)
    assert F.face_square((10, 10, 390, 290), 400, 300) == (50, 0, 300)
    assert F.centre_square(300, 200) == (50, 0, 200)


@pytest.mark.models
def test_face_detector_finds_no_face_in_a_plain_picture(tmp_path):
    from video_chain import faces as F
    model = F.find_detector(Path.home() / "ComfyUI")
    if F.available(model):
        pytest.skip(F.available(model))
    p = tmp_path / "grey.png"
    Image.new("RGB", (320, 180), (128, 128, 128)).save(p)
    assert F.detect(p, model) == []


def test_cast_review_fixes(project, tmp_path):
    o = {"cast": [{"name": "Anna", "images": ["refs/a.png"], "desc": "woman", "voice": {"file": "refs/v.wav"}},
                  {"name": "Ben", "images": ["refs/b.png"], "desc": "man"}], "identity": {"hint": True}}
    # only a speech verb after a line names its speaker ("Then @Anna" is ordinary text)
    p = T.clip_cast(o, '@Anna turns to @Ben. "Hello." Then @Anna smiles.')["prompt"]
    assert "<d>[English] Hello.</d> Then <Subject 1> (S1) smiles." in p
    # an @Name inside a quoted line is a spoken word: no picture, no tag, no @ read aloud
    u = T.clip_cast(o, '@Anna says "Hi @Ben"')
    assert u["images"] == ["refs/a.png"] and "<d>[English] Hi Ben</d>" in u["prompt"]
    u = T.clip_cast(o, '@Ben waves "Hi @Anna"')
    assert u["images"] == ["refs/b.png"] and '"Hi Anna"' in u["prompt"]
    assert [d["speaker"]["name"] for d in T.dialogue(o, '@Ben waves and says "Hi @Anna"')] == ["Ben"]
    # with the hint off the description is not in the prompt, so it is not recorded either
    o["identity"]["hint"] = False
    assert T.clip_cast(o, "@Ben waves")["record"] == [{"name": "Ben", "desc": ""}]
    t = T.load(project)
    T.add_character(project, t, "Ben", face(tmp_path / "b.jpg", (90, 70, 60)))
    with pytest.raises(ValueError, match="no character called Bob"):
        T.add_reference(project, face(tmp_path / "x.jpg", (1, 2, 3)), t, "Bob")
    n = t.add(None, "@Ben stands next to <Picture 1>.")
    assert any("writes <Picture N> and mentions @Names" in m for m in T.cast_problems(project, t, [n.id]))
    real = T.chain.to_png

    def failing_crop(image, png, crop=None):                                   # the crop step fails
        if crop:
            raise RuntimeError("ffmpeg failed")
        real(image, png)
    T.chain.to_png = failing_crop
    try:
        with pytest.raises(RuntimeError):
            T.add_picture(project, t, "Ben", face(tmp_path / "y.jpg", (5, 5, 5)), crop=(0, 0, 32))
    finally:
        T.chain.to_png = real
    assert sorted(f.name for f in (project / "refs").iterdir()) == ["ref1.png"]    # no stray full picture


def test_lora_trigger_words_in_clips(project, tmp_path):
    """A clip's LoRAs' trigger words go before its prompt and are recorded (new words: the clip is made again)."""
    import numpy as np
    from safetensors.numpy import save_file

    from video_chain import triggers as TR
    lo = tmp_path / "loras" / "braces.safetensors"
    lo.parent.mkdir()
    save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(lo), metadata={"modelspec.trigger_word": "braces"})
    plain = tmp_path / "loras" / "plain.safetensors"
    save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(plain))
    t = T.load(project)
    a = t.add(None, "a woman smiles")
    a.overrides = {"loras": [{"path": str(lo), "strength": 0.8}, {"path": str(plain), "strength": 1.0}]}
    b = t.add(None, "no lora")
    T.save(project, t)
    TR.db().scan([str(lo), str(plain)], online=False)
    gen = run(project, [a.id, b.id])
    call = next(c for c in gen.calls if "smiles" in c["prompt"])
    assert call["prompt"] == "braces, a woman smiles" and call["settings"]["lora_triggers"] == ["braces"]
    assert "lora_triggers" not in next(c for c in gen.calls if "no lora" in c["prompt"])["settings"]
    assert set(statuses(project).values()) == {"done"}
    TR.db().set_user(str(plain), ["plainword"])                                # new words: made again
    assert statuses(project)["a woman smiles"] == "changed"
    t = T.load(project)
    t.options["lora_triggers"] = False                                         # off: the prompt as written
    T.save(project, t)
    assert "lora_triggers" not in T.Checker(project, T.load(project)).settings(a.id)


def test_refined_export_can_refine_a_video_with_a_replaced_person(project, tmp_path):
    from video_chain import swap
    clip = tmp_path / "swapped.mp4"                    # 1 s at 24 fps (off H3's frame grid), with sound
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=96x54:r=24:d=1", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(clip)], check=True)
    t = T.load(project)
    v = t.add_video(None, str(clip), 0.0, 1.0)
    a = t.add(v.id, "a")
    swap.apply(v, clip, 24, {"method": "h3", "details": "Male, short black hair", "seed": 7})
    T.save(project, t)
    gen = run(project, [a.id])
    out = tmp_path / "out.mp4"
    T.refine_path(gen, project, a.id, "2k", out, log=lambda _m: None, refine_swapped=True)
    r = gen.refines[0]
    assert r["prompt"] == "Male, short black hair" and r["frames"] == 39 and r["draft"] == f"{v.id}_draft.mp4"
    assert r["loras"] == [] and gen.refines[1]["first"] == f"{a.id}_first.png"   # the next clip pinned to it
    t = T.load(project)
    refined = T.refined_file(project, t.nodes[v.id], "2k")
    info = chain.probe(refined)
    assert info["frames"] == 24 and info["audio"] and (info["width"], info["height"]) == r["size"]
    T.refine_path(gen, project, a.id, "2k", tmp_path / "again.mp4", log=lambda _m: None, refine_swapped=True)
    assert len(gen.refines) == 2                                     # reused
    T.refine_path(gen, project, a.id, "2k", tmp_path / "plain.mp4", log=lambda _m: None)
    assert json.loads(refined.with_suffix(".json").read_text())["video"] == "converted"   # only scaled up now
    assert len(gen.refines) == 3 and gen.refines[2]["prompt"] == "a"   # (pinned to the other first frame)


def test_new_clips_get_random_unused_seeds():
    t = T.Tree()
    seeds = [t.add(None, str(k)).seed for k in range(20)]
    assert len(set(seeds)) == 20 and all(1 <= s <= T.SEED_MAX for s in seeds)
    assert len({T.Tree().add(None, "x").seed for _ in range(5)}) > 1     # random, not a fixed start seed


def test_refine_strength_is_used_and_recorded(project, tmp_path):
    from video_chain import comfy as C
    assert C.refine_sigmas("natural") == "0.7, 0.66, 0.5136, 0" and C.REFINE_DEFAULT == "natural"
    assert C.refine_sigmas("crisp").startswith("0.8, ") and C.refine_sigmas("soft").startswith("0.6, ")
    t, a, b1, b2, c = build(project)
    gen = run(project, [b1.id])
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "n.mp4", log=lambda _m: None)
    assert [r["strength"] for r in gen.refines] == ["natural", "natural"]
    t = T.load(project)
    assert T.refine_record(T.refined_file(project, t.nodes[a.id], "2k"))["strength"] == "natural"
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "c.mp4", log=lambda _m: None, strength="crisp")
    assert [r["strength"] for r in gen.refines[2:]] == ["crisp", "crisp"]        # another strength: made again
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "c2.mp4", log=lambda _m: None, strength="crisp")
    assert len(gen.refines) == 4                                                   # then reused
