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
    assert starts[0] == "start_672x384.png" and starts[1].startswith(f"{a.id}_") and starts[2].startswith(f"{b1.id}_")
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
    (project / (T.TREE_FILE + ".bak")).unlink(missing_ok=True)         # (with a backup it opens: test_tree_store)
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
    T.remove_picture(project, t, rel)
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


def test_regenerating_never_deletes_and_tolerates_open_files(project, monkeypatch):
    """A clip made again gets new names; the version before is never deleted: it is moved to old/ in the project. A
    file that is open somewhere (SMB shares refuse to move it) stays where it is and is moved on a later run."""
    import os
    t, a, b1, b2, c = build(project)
    run(project, [a.id])
    first = T.clip_files(project, T.load(project).nodes[a.id])
    content = [f.read_bytes() for f in first]
    t = T.load(project)
    t.nodes[a.id].prompt = "a again"
    T.save(project, t)
    real = os.replace
    busy = {first[0]}

    def replace(src, dst):
        if Path(src) in busy:
            raise PermissionError(13, "Permission denied", str(src))      # open in a player on the share
        return real(src, dst)
    monkeypatch.setattr(chain.os, "replace", replace)
    gen = FakeGenerator(frames=17)
    gen.calls = [{}]                                                      # other frames than the first version
    run(project, [a.id], gen)
    now = T.clip_files(project, T.load(project).nodes[a.id])
    old = [project / chain.OLD_DIR / "nodes" / f.name for f in first]
    assert now != first and all(f.exists() for f in now)
    assert first[0].exists() and not first[1].exists()                   # the open one stays, the other was moved
    assert not old[0].exists() and old[1].read_bytes() == content[1]
    assert statuses(project)["a again"] == "done"
    busy.clear()                                                          # closed again: the next run moves it too
    assert T.remove_stale_files(project, T.load(project)) == [] and not first[0].exists()
    assert [f.read_bytes() for f in old] == content                       # nothing was lost
    # a name that is taken in old/ gets a number; what is in old/ stays there
    (project / "nodes" / first[0].name).write_bytes(b"another")
    assert chain.keep_aside(project, project / "nodes" / first[0].name)
    assert (old[0].with_name(old[0].stem + "_2.mp4")).read_bytes() == b"another" and old[0].read_bytes() == content[0]
    assert chain.keep_aside(project, old[0]) and old[0].exists()
    # deleting the clip in the app: its files go to old/ as well
    t = T.load(project)
    t.remove(a.id, project)
    assert all(not f.exists() and (project / chain.OLD_DIR / "nodes" / f.name).is_file() for f in now)


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
    old = project / chain.OLD_DIR / "nodes"                # finished ones are kept in old/, a half-written one is not
    assert sorted(f.name for f in old.iterdir()) == ["n1.mp4", "n1.png", "n7_0123abcd.mp4"]


def test_a_joined_path_made_again_keeps_the_one_before(project):
    """paths/<id>.mp4 is joined again when a clip of it changed: the join before goes to old/paths/, never deleted;
    an unchanged path is not joined again (and nothing is moved)."""
    t, a, b1, b2, c = build(project)
    run(project, [b1.id])
    joined = T.join_path(project, T.load(project), b1.id)
    first = joined.read_bytes()
    old = project / chain.OLD_DIR / "paths" / joined.name
    assert T.join_path(project, T.load(project), b1.id) == joined and not old.exists()
    t = T.load(project)
    t.nodes[b1.id].prompt = "b1 again"
    T.save(project, t)
    gen = FakeGenerator(frames=17)
    gen.calls = [{}, {}]
    run(project, [b1.id], gen)
    assert T.join_path(project, T.load(project), b1.id) == joined
    assert old.read_bytes() == first and joined.read_bytes() != first
    assert not list(joined.parent.glob("*.new.*"))
    t = T.load(project)
    t.remove(b1.id, project)                                              # the clip deleted: its join is kept too
    assert not joined.exists() and (old.with_name(old.stem + "_2.mp4")).is_file()


def test_nothing_the_app_removes_is_deleted(project, tmp_path):
    """Pictures and sounds the app stops using (a start image, face picture, voice, sound reference, end image that
    is replaced or removed) are moved to old/ in the project, never deleted."""
    old = project / chain.OLD_DIR
    t = T.load(project)
    before = (project / T.START_FILE).read_bytes()
    T.set_start_image(project, face(tmp_path / "s2.jpg", (10, 200, 10)), t)
    assert (old / T.START_FILE).read_bytes() == before and (project / T.START_FILE).read_bytes() != before
    second = (project / T.START_FILE).read_bytes()
    T.clear_start_image(project, t)
    assert not (project / T.START_FILE).exists() and (old / "start_2.png").read_bytes() == second
    T.add_character(project, t, "Anna", face(tmp_path / "a.jpg", (200, 150, 130)))
    pic = T.character(t, "Anna")["images"][0]
    T.set_voice(project, t, "Anna", sound(tmp_path / "long.wav", 12))
    v = T.character(t, "Anna")["voice"]
    T.set_voice(project, t, "Anna", sound(tmp_path / "other.wav", 4))       # replaced: the voice and its reference
    assert (old / v["file"]).is_file() and (old / v["ref"]["file"]).is_file() and not (project / v["file"]).exists()
    v2 = T.character(t, "Anna")["voice"]["file"]
    ref = T.add_audio_reference(project, sound(tmp_path / "s1.wav", 3), t)["file"]
    T.remove_audio_reference(project, ref, t)
    T.remove_character(project, t, "Anna")
    assert all((old / r).is_file() and not (project / r).exists() for r in (pic, v2, ref))


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
    assert "cfg" not in T.Checker(project, T.load(project)).settings(a.id)       # CFG 1: settings as before
    t.options["turbo"]["cfg"] = 1.5                                  # turbo's own CFG: in the settings, clips change
    T.save(project, t)
    assert T.Checker(project, T.load(project)).settings(a.id)["cfg"] == 1.5
    t.options["cfg"] = 2.0                                           # (Normal's CFG does not count in turbo mode)
    T.save(project, t)
    assert T.Checker(project, T.load(project)).settings(a.id)["cfg"] == 1.5
    assert statuses(project)["a"] == "changed"
    t.options["turbo"]["cfg"] = 1.0
    t.options.pop("cfg")
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
    T._IMG_SIZE.clear()                       # (caches of earlier tests must not decide the counts)
    chain._SHA_CACHE.clear()
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
    assert "shorter than the 5-frame carry-over" in T.guide_problem(t, T.Checker(project, t), [b1.id])


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
    assert gen.refines[0]["first"] == "start_2528x1440.png" and gen.refines[1]["first"] == f"{b1.id}_first.png"
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
    # the longer source is kept as the clone reference (for text to speech); the clip model hears the short voice
    assert v["ref"]["file"] == "refs/voice1_ref.wav" and abs(v["ref"]["secs"] - 10) < 0.1
    assert abs(chain.audio_seconds(project / "refs/voice1_ref.wav") - 10) < 0.1
    assert T.clone_file(v) == "refs/voice1_ref.wav" and T.clone_secs(v) == v["ref"]["secs"]
    T.save(project, t)
    assert T.character(T.load(project), "Anna")["voice"]["ref"] == v["ref"]              # kept in tree.json
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
    assert not (project / "refs/voice1_ref.wav").exists()                     # its clone reference went with it
    assert "ref" not in T.character(t, "Anna")["voice"]                       # 4 s: the voice is all there is
    assert T.clone_file(T.character(t, "Anna")["voice"]) == "refs/voice2.wav"
    long = T.set_voice(project, t, "Ben", sound(tmp_path / "long.wav", 40), start=5.0)
    assert abs(long["ref"]["secs"] - T.VOICE_REF_MAX_SECS) < 0.1 and abs(long["secs"] - T.VOICE_MAX_SECS) < 0.1
    T.remove_voice(project, t, "Ben")
    assert not list((project / "refs").glob("voice1*"))
    # a chosen part of the source: nothing past its end, neither in the voice nor in the clone reference
    part = T.set_voice(project, t, "Ben", tmp_path / "long.wav", start=5.0, end=17.0)
    assert abs(part["secs"] - T.VOICE_MAX_SECS) < 0.1 and abs(part["ref"]["secs"] - 12) < 0.1 and part["cut"]
    part = T.set_voice(project, t, "Ben", tmp_path / "long.wav", start=5.0, end=9.0)
    assert abs(part["secs"] - 4) < 0.1 and "ref" not in part and not part["cut"]
    assert abs(chain.audio_seconds(project / part["file"]) - 4) < 0.1
    with pytest.raises(ValueError, match="at least 2 s.*in the chosen part"):
        T.set_voice(project, t, "Ben", tmp_path / "long.wav", start=5.0, end=6.0)
    T.remove_voice(project, t, "Ben")
    assert not list((project / "refs").glob("voice1*"))
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


def test_export_with_seedvr2(project, tmp_path):
    """SeedVR2: every generated clip upscaled as made (no pins, no prompt); a real video converted from its file;
    copies recorded per method, so the refine or the other model makes new ones."""
    t, a, b1, b2, c = build(project)
    gen = run(project, [b1.id])
    out = tmp_path / "s.mp4"
    T.refine_path(gen, project, b1.id, "2k", out, log=lambda _m: None, method="seedvr2_7b")
    assert [x["method"] for x in gen.seedvrs] == ["seedvr2_7b", "seedvr2_7b"] and not getattr(gen, "refines", [])
    t = T.load(project)
    assert chain.probe(out)["width"] == gen.seedvrs[0]["size"][0]
    assert T.refine_record(T.refined_file(project, t.nodes[a.id], "2k")) == {"first": None, "last": None,
                                                                               "method": "seedvr2_7b"}
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "s2.mp4", log=lambda _m: None, method="seedvr2_7b")
    assert len(gen.seedvrs) == 2                                                   # reused
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "s3.mp4", log=lambda _m: None, method="seedvr2_3b")
    assert [x["method"] for x in gen.seedvrs[2:]] == ["seedvr2_3b", "seedvr2_3b"]   # another model: made again
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "r.mp4", log=lambda _m: None)
    assert len(gen.refines) == 2                                                   # the refine: its own copies
    with pytest.raises(ValueError, match="unknown method"):
        T.refine_path(gen, project, b1.id, "2k", tmp_path / "x.mp4", log=lambda _m: None, method="topaz")


def test_veda_recorded_for_large_clips_only(project):
    t, a, b1, b2, c = build(project)
    base = {"model": "Comfy-Org/MiniMax-H3", "steps": 28, "width": 1344, "height": 768, "audio": True}
    assert T.node_settings(base, t.options, t.nodes[a.id]).get("veda") is None          # 73 frames: 89k tokens
    t.nodes[a.id].overrides = {"frames": 241}
    assert T.node_settings(base, t.options, t.nodes[a.id])["veda"] is True               # 10 s: 290k tokens
    assert T.node_settings({**base, "width": 672, "height": 384}, t.options, t.nodes[a.id]).get("veda") is None
    t.options["veda"] = {"on": False}
    assert T.node_settings(base, t.options, t.nodes[a.id]).get("veda") is None           # switched off


def test_export_one_clip(project, tmp_path):
    """single: only that clip, whole; SeedVR2 copies are shared with path exports; a refine reuses the path's
    (pinned) copy, else makes an unpinned one that is not kept (the path's copy stays as it is)."""
    t, a, b1, b2, c = build(project)
    gen = run(project, [b1.id])
    t = T.load(project)
    out = tmp_path / "b1.mp4"
    T.refine_path(gen, project, b1.id, "2k", out, log=lambda _m: None, method="seedvr2_7b", single=True)
    assert [x["draft"] for x in gen.seedvrs] == [Path(t.nodes[b1.id].made["video"]).name]   # b1 only
    assert chain.probe(out)["frames"] == chain.probe(project / t.nodes[b1.id].made["video"])["frames"]   # whole
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "p.mp4", log=lambda _m: None, method="seedvr2_7b")
    assert len(gen.seedvrs) == 2                                    # the path: b1's copy reused, a made
    kept = T.refined_file(project, t.nodes[a.id], "2k")
    T.refine_path(gen, project, a.id, "2k", tmp_path / "a1.mp4", log=lambda _m: None, single=True)
    assert gen.refines[-1]["first"] is None and gen.refines[-1]["strength"] == "natural"   # unpinned
    assert T.refine_record(kept).get("method") == "seedvr2_7b"     # the kept (SeedVR2) copy untouched
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "r.mp4", log=lambda _m: None)      # path refine: pinned
    n = len(gen.refines)
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "b1r.mp4", log=lambda _m: None, single=True)
    assert len(gen.refines) == n                                    # the pinned copy is the same clip: reused
    with pytest.raises(ValueError, match="not made yet"):
        T.refine_path(gen, project, c.id, "2k", tmp_path / "x.mp4", log=lambda _m: None, single=True)


def test_seedvr2_light_denoise_is_recorded(project, tmp_path):
    t, a, b1, b2, c = build(project)
    gen = run(project, [b1.id])
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "s.mp4", log=lambda _m: None, method="seedvr2_7b")
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "d.mp4", log=lambda _m: None, method="seedvr2_7b",
                  denoise=True)
    assert [x["denoise"] for x in gen.seedvrs] == [False, False, True, True]        # the other choice: made again
    t = T.load(project)
    assert T.refine_record(T.refined_file(project, t.nodes[a.id], "2k"))["denoise"] is True
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "d2.mp4", log=lambda _m: None, method="seedvr2_7b",
                  denoise=True)
    assert len(gen.seedvrs) == 4                                                     # then reused


def test_camera_move_is_recorded_and_sent_to_the_model(project):
    t, a, b1, b2, c = build(project)
    t.nodes[a.id].prompt = "She lifts the cup. #pan-right-slow toward the window"
    t.nodes[a.id].overrides = {"camera": {"move": "push-in", "amplitude": "small", "speed": "slow"}}
    T.save(project, t)
    gen = run(project, [a.id])
    sent = gen.calls[0]["prompt"]
    assert "cup. The camera pans right at slow speed toward the window." in sent and "#pan" not in sent
    assert sent.endswith("The camera pushes in with small amplitude at slow speed.")
    t = T.load(project)
    assert t.nodes[a.id].made["settings"]["camera"] == {"move": "push-in", "amplitude": "small", "speed": "slow"}
    assert statuses(project)["She lifts the cup. #pan-right-slow toward the window"] == "done"
    t.nodes[a.id].overrides = {"camera": {"move": "push-in", "speed": "fast"}}       # another move: made again
    T.save(project, t)
    assert statuses(project)["She lifts the cup. #pan-right-slow toward the window"] == "changed"
    t.nodes[a.id].overrides = {}
    T.save(project, t)
    assert "camera" not in T.Checker(project, T.load(project)).settings(a.id)


def test_h3_native_canvas():
    """ComfyUI's H3 rule (adapt_canvas): short edge 768, area at most 768 x 1344, multiples of 32."""
    assert chain.h3_canvas(1920, 1080) == (1344, 768) and chain.h3_canvas(1080, 1920) == (768, 1344)
    assert chain.h3_canvas(1000, 1000) == (768, 768) and chain.h3_canvas(1600, 2000) == (768, 960)
    assert chain.h3_canvas(2560, 1080) == (1568, 672)                       # 21:9: capped, then rounded (as ComfyUI)
    assert chain.canvas_size(1600, 2000, chain.H3_NATIVE) == (768, 960)    # the size setting "768p native"
    assert chain.canvas_size(1920, 1080, 512) == (672, 384)                # (the area rule as before)


def test_start_image_prepared_for_h3(project, tmp_path):
    """Cut to the canvas's exact ratio (not stretched), Lanczos to the exact size, transparency on grey, EXIF
    rotation applied; kept as start_<w>x<h>.png until start.png changes; the clip records start.png."""
    from PIL import Image
    src = tmp_path / "wide.png"                                           # 2:1, red | green | blue thirds,
    im = Image.new("RGBA", (900, 450), (0, 0, 0, 0))                        # a transparent band at the top
    for i, c in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        im.paste(c + (255,), (300 * i, 60, 300 * (i + 1), 450))
    im.save(src)
    out = tmp_path / "p.png"
    chain.prepare_image(src, out, 672, 384)                               # 1.75: the sides are cut, not squeezed
    p = Image.open(out)
    assert p.size == (672, 384) and p.mode == "RGB"
    assert p.getpixel((5, 300))[0] > 200 and p.getpixel((336, 300))[1] > 200 and p.getpixel((666, 300))[2] > 200
    assert p.getpixel((336, 5)) == (128, 128, 128)                          # transparent -> neutral grey
    third = sum(1 for x in range(672) if p.getpixel((x, 300))[1] > 200)
    assert 672 / 3 * 1.05 < third                                            # the green middle got wider: cut sides
    rot = tmp_path / "rot.jpg"                                             # EXIF "rotate 90": a portrait photo
    exif = Image.Exif(); exif[0x0112] = 6
    Image.new("RGB", (400, 200), (200, 50, 50)).save(rot, exif=exif)
    chain.prepare_image(rot, out, 384, 672)
    assert Image.open(out).size == (384, 672)
    # in a project: made once per canvas, again when start.png changes
    a = T.prepared_start(project, 672, 384)
    assert a.name == "start_672x384.png" and T.prepared_start(project, 672, 384) == a
    stamp = a.stat().st_mtime_ns
    Image.new("RGB", (96, 54), (10, 200, 10)).save(project / T.START_FILE)
    b = T.prepared_start(project, 672, 384)
    assert b.stat().st_mtime_ns != stamp and Image.open(b).getpixel((300, 200))[1] > 150
    T.prepared_start(project, 1344, 768)
    assert sorted(f.name for f in project.glob("start_*x*.png")) == ["start_1344x768.png"]   # one at a time


def test_first_clip_without_a_start_image(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    t = T.Tree()
    a = t.add(None, "a lighthouse at dusk")
    T.save(p, t)
    assert T.size_origin(p, t) == "aspect" and T.size_source(p, t) == (16 * 120, 9 * 120)
    gen = run(p, [a.id])
    assert gen.calls[0]["start"] is None                                       # from text: no start frame
    made = T.load(p).nodes[a.id].made
    assert (made["settings"]["width"], made["settings"]["height"]) == (672, 384)
    t = T.load(p)
    t.options["aspect"] = "4:5"
    T.save(p, t)
    assert statuses(p)["a lighthouse at dusk"] == "changed"


def test_split_a_video_and_insert_a_clip(project, tmp_path):
    t = T.load(project)
    src = tmp_path / "real.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=96x54:r=24:d=10", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    v = t.add_video(None, str(src), 0.0, 9.0)
    after = t.add(v.id, "goes on")
    clip, rest = t.split_video(v.id, 5.0)
    assert t.nodes[v.id].source == {"file": str(src), "start": 0.0, "end": 5.0}
    assert clip.parent == v.id and not clip.is_video and clip.prompt == ""
    assert rest.parent == clip.id and rest.source == {"file": str(src), "start": 5.0, "end": 9.0}
    assert t.nodes[after.id].parent == rest.id                                 # what followed goes on after the rest
    assert t.video_child(clip.id).id == rest.id                                 # the clip ends at the rest's first frame
    with pytest.raises(ValueError):
        t.split_video(v.id, 0.0)                                                # not at the very start / end
    with pytest.raises(ValueError):
        t.split_video(clip.id, 1.0)                                             # (only real videos)
    # a video whose person was replaced: both parts keep it, the original's range split at the same moment
    from video_chain import swap
    w = t.add_video(None, str(src), 2.0, 6.0)
    swap.apply(w, src, 96, {"method": "h3"}, fps=24.0)                          # now plays the swap (0..4 s)
    c2, r2 = t.split_video(w.id, 1.5)
    assert swap.replaced(w)["original"] == {"file": str(src), "start": 2.0, "end": 3.5}
    assert swap.replaced(r2)["original"] == {"file": str(src), "start": 3.5, "end": 6.0}
    T.save(project, t)
    t2 = T.load(project)
    assert t2.nodes[r2.id].source["start"] == 1.5 and swap.replaced(t2.nodes[r2.id]) is not None


def test_middle_keyframes(project, tmp_path):
    """Keyframes: copied in, sent with their frame, recorded; editing one makes the clip changed (Undo goes back);
    a variant gets its own copies; a keyframe past the end, on a motion guide or a missing picture is refused."""
    t, a, b1, b2, c = build(project)
    pic, pic2 = tmp_path / "k.png", tmp_path / "k2.png"
    Image.new("RGB", (40, 30), (0, 255, 0)).save(pic)
    Image.new("RGB", (40, 30), (0, 0, 255)).save(pic2)
    t.options["frames"] = 22
    assert T.set_keyframe(project, t, a.id, pic, 0.5, "she turns") == 0
    assert T.set_keyframe(project, t, a.id, pic2, 0.25) == 0                     # sorted by time
    assert [k["at"] for k in a.keyframes] == [0.25, 0.5] and all((project / k["image"]).is_file() for k in a.keyframes)
    T.save(project, t)
    assert T.load(project).nodes[a.id].keyframes == a.keyframes                   # kept in tree.json
    gen = run(project, [a.id])
    sent = gen.calls[0]["settings"]["keyframes"]
    assert [k["frame_idx"] for k in sent] == [6, 12] and sent[1]["image"] == str(project / a.keyframes[1]["image"])
    assert "At 0.5 s: she turns." in gen.calls[0]["prompt"]
    t = T.load(project)
    a = t.nodes[a.id]
    assert a.made["keyframes"] == a.keyframes and T.Checker(project, t).status(a.id) == "done"
    made_pics = [k["image"] for k in a.keyframes]
    # move one, change a picture: changed, with the reason; Undo restores them and deletes the newer copy
    T.set_keyframe(project, t, a.id, None, 0.75, index=0)
    T.set_keyframe(project, t, a.id, pic, 0.5, index=0)
    newer = next(k["image"] for k in a.keyframes if k["image"] not in made_pics)
    ch = T.Checker(project, t)
    assert ch.status(a.id) == "changed" and T.changed_inputs(t, a.id) == ["keyframes"]
    assert T.revert(project, t, a.id) == ["keyframes"] and [k["image"] for k in a.keyframes] == made_pics
    assert not (project / newer).exists() and T.Checker(project, t).status(a.id) == "done"
    # the note alone changes the clip (it is in the prompt)
    T.set_keyframe(project, t, a.id, None, 0.5, "she waves", index=1)
    assert T.Checker(project, t).why(a.id) == "its keyframes changed"
    T.revert(project, t, a.id)
    # removing: the made copy stays for Undo; a variant gets its own copies
    T.remove_keyframe(project, t, a.id, 0)
    assert (project / made_pics[0]).is_file()
    T.copy_keyframes(project, t, a.id, b2.id)
    assert len(t.nodes[b2.id].keyframes) == 1 and t.nodes[b2.id].keyframes[0]["image"] != a.keyframes[0]["image"]
    # problems: past the end, on a motion guide, a missing picture
    ch = T.Checker(project, t)
    T.set_keyframe(project, t, b1.id, pic, 0.9)                                   # frame 22 of 22
    assert "past its end" in T.keyframe_problem(t, ch, [b1.id])
    t.nodes[b1.id].keyframes[0]["at"] = 0.1                                       # frame 2
    t.nodes[b1.id].overrides["guide"] = 5                                         # frames 0-4 are the guide
    assert "inside its carry-over" in T.keyframe_problem(t, T.Checker(project, t), [b1.id])
    t.nodes[b1.id].keyframes[0]["at"] = 5 / 24                                    # the first frame after it
    assert T.keyframe_problem(t, T.Checker(project, t), [b1.id]) is None
    t.nodes[b1.id].overrides.pop("guide")
    assert T.keyframe_problem(t, T.Checker(project, t), [b1.id]) is None
    (project / t.nodes[b1.id].keyframes[0]["image"]).unlink()
    assert T.missing_keyframes(project, t, [b1.id]) == [b1.id]
    # removing a clip removes its keyframe pictures
    files = [project / k["image"] for k in t.nodes[b2.id].keyframes]
    t.remove(b2.id, project)
    assert not any(f.exists() for f in files)
    # damaged entries are left out when loading
    assert T.clean_keyframes([{"image": "../x.png", "at": 1}, {"image": "ends/a.png", "at": 0}, "x",
                              {"image": "ends/b.png", "at": 1.5, "note": None}]) == [
        {"image": "ends/b.png", "at": 1.5, "note": ""}]


def test_hard_cuts(tmp_path):
    """A jump to another picture is a cut; motion is not (lavfi test pictures)."""
    smooth, cut = tmp_path / "smooth.mp4", tmp_path / "cut.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=2",
                    "-pix_fmt", "yuv420p", str(smooth)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1",
                    "-f", "lavfi", "-i", "mandelbrot=size=160x90:rate=24", "-filter_complex",
                    "[1]trim=duration=1,setpts=PTS-STARTPTS[b];[0][b]concat=n=2:v=1", "-pix_fmt", "yuv420p", str(cut)],
                   check=True)
    assert chain.hard_cuts(smooth) == []
    assert chain.hard_cuts(cut) == [24]
    assert chain.hard_cuts(tmp_path / "missing.mp4") == []


def test_freeze_keeps_clips_as_made(project, tmp_path):
    """Freezing a clip keeps it and every clip before it done whatever the settings say; the clips after it follow
    the settings; a frozen path of older-size clips joins at the size clips have now; a seam that no longer fits
    and an unmade clip are refused; unfreezing takes the frozen clips after it along; edits are undone."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id], FakeGenerator(frames=17, size=None))
    t = T.load(project)
    old = T.Checker(project, t).base()
    assert T.freeze_problem(project, t, b2.id) == f"clip #{b2.id[1:]} is not made yet (or its files are missing)"
    assert T.freeze(project, t, b1.id) == [a.id, b1.id]
    t.options["resolution"] = 256                          # another size: frozen ones stay done, c is made again
    T.save(project, t)
    t = T.load(project)
    assert t.nodes[b1.id].frozen and t.nodes[a.id].frozen and not t.nodes[c.id].frozen       # kept in tree.json
    st = statuses(project)
    assert st["a"] == st["b1"] == "done" and st["c"] == "changed"
    base = T.Checker(project, t).base()
    assert T.frozen_note(t, t.nodes[a.id], base) == f"made at {old['width']}×{old['height']}"
    gen = run(project, [c.id], FakeGenerator(frames=17, size=None))
    assert len(gen.calls) == 1 and gen.calls[0]["prompt"].startswith("c")
    assert gen.calls[0]["settings"]["width"] == base["width"]
    t = T.load(project)
    out = T.join_path(project, t, c.id)                    # two sizes: every clip at the size of now
    info = chain.probe(out)
    assert (info["width"], info["height"]) == (base["width"], base["height"]) and info["frames"] == 17 + 16 + 16
    mixed = project.parent / "mixed.mp4"                  # (the export's join: no size given, the largest one)
    chain.concat([project / t.nodes[c.id].made["video"], project / t.nodes[a.id].made["video"]], mixed)
    assert (chain.probe(mixed)["width"], chain.probe(mixed)["height"]) == (old["width"], old["height"])
    # unfreezing a takes b1 along (a frozen clip needs the ones before it frozen); both follow the settings again
    assert T.unfreeze(t, a.id) == [a.id, b1.id]
    T.save(project, t)
    assert statuses(project)["a"] == "changed"
    # an edit is undone by freezing
    t.nodes[b1.id].prompt = "b1 edited"
    assert T.freeze_edits(t, b1.id) == {b1.id: ["prompt"]}
    T.freeze(project, t, b1.id)
    assert t.nodes[b1.id].prompt == "b1" and T.Checker(project, t).status(b1.id) == "done"
    # a made again: b1 no longer starts where a ends, so it cannot be frozen with it
    T.unfreeze(t, a.id)
    t.nodes[a.id].prompt = "a again"
    T.save(project, t)
    run(project, [a.id], FakeGenerator(frames=17, size=None))
    t = T.load(project)
    assert "no longer starts at the frame it was made from" in T.freeze_problem(project, t, b1.id)
    assert T.freeze_problem(project, t, a.id) is None


def test_frozen_clips_stay_consistent(project):
    """A frozen clip needs every clip before it frozen and the video it ends at: a tree file that says otherwise is
    healed; unfreezing a video unfreezes the clip that ends at it; freezing a clip takes its video along."""
    t = T.load(project)
    a = t.add(None, "a")
    v = t.add_video(a.id, "/nowhere/x.mp4", 0.0, 1.0)
    b = t.add(v.id, "b")
    for n in (a, v, b):
        n.made = {"video": "x"}
    assert [n.id for n in T.freeze_set(t, a.id)] == [a.id, v.id]          # a ends at v's first frame
    assert [n.id for n in T.freeze_set(t, b.id)] == [a.id, v.id, b.id]
    a.frozen = v.frozen = b.frozen = True
    assert T.unfreeze(t, v.id) == [v.id, a.id, b.id] and not (a.frozen or b.frozen)
    b.frozen = True                                         # b frozen, the ones before it not: healed on load
    T.save(project, t)
    assert not T.load(project).nodes[b.id].frozen
    a.made, a.frozen = None, True                           # never made: not frozen
    T.save(project, t)
    assert not T.load(project).nodes[a.id].frozen


def test_seedvr2_cloud_variant_is_recorded(project, tmp_path):
    """Copies upscaled on a big RunPod GPU (fp16, whole frames) record that: a copy made otherwise (int8, tiles)
    is made again there instead of reused, and the other way round; the same variant is reused."""
    t, a, b1, b2, c = build(project)
    gen = run(project, [b1.id])
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "s.mp4", log=lambda _m: None, method="seedvr2_7b")
    gen.seedvr_variant = "fp16, whole frames"
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "c.mp4", log=lambda _m: None, method="seedvr2_7b")
    assert len(gen.seedvrs) == 4                                                     # made again
    t = T.load(project)
    assert T.refine_record(T.refined_file(project, t.nodes[a.id], "2k"))["variant"] == "fp16, whole frames"
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "c2.mp4", log=lambda _m: None, method="seedvr2_7b")
    assert len(gen.seedvrs) == 4                                                     # then reused
    gen.seedvr_variant = None
    T.refine_path(gen, project, b1.id, "2k", tmp_path / "s2.mp4", log=lambda _m: None, method="seedvr2_7b")
    assert len(gen.seedvrs) == 6                                                     # back home: made again


# ---------------------------------------------------------------------------- the official H3 prompt format

from video_chain import h3_prompt as H  # noqa: E402

ANNA = "woman, 30s, grey sweater"


def h3_project(tmp_path, start=True, name="h3"):
    """A project in the h3 format (the tests' default is classic), with a start image unless start=False."""
    p = tmp_path / name
    p.mkdir()
    t = T.Tree()
    t.options["prompt_format"] = "h3"
    if start:
        img = tmp_path / f"{name}_in.jpg"
        Image.new("RGB", (96, 54), (250, 250, 250)).save(img)
        T.set_start_image(p, img, t)
    T.save(p, t)
    return p, t


def anna_and_kitchen(p, t, tmp_path):
    T.add_character(p, t, "Anna", face(tmp_path / "a1.jpg", (200, 150, 130)), desc=ANNA)
    T.add_picture(p, t, "Anna", face(tmp_path / "a2.jpg", (190, 140, 120)))
    T.add_character(p, t, "Kitchen", face(tmp_path / "k.jpg", (90, 90, 200)), desc="a small sunny kitchen",
                    kind="place")


@pytest.mark.h3_default
def test_prompt_format_of_new_and_loaded_projects(tmp_path):
    """New projects write H3's official formats; a file from before the setting loads (and stays) classic."""
    assert T.Tree().options["prompt_format"] == "h3" and T.is_h3(T.Tree())
    empty = tmp_path / "new"
    empty.mkdir()
    assert T.load(empty).options["prompt_format"] == "h3"                       # a new project folder
    d = T.Tree().to_json()
    del d["options"]["prompt_format"]
    assert T.Tree.from_json(d).options["prompt_format"] == "classic"
    d["options"]["prompt_format"] = "fancy"
    assert T.Tree.from_json(d).options["prompt_format"] == "classic"
    for fmt in T.PROMPT_FORMATS:                                                 # saved once: stays as it is
        t = T.Tree()
        T.set_prompt_format(t, fmt)
        T.save(empty, t)
        assert T.load(empty).options["prompt_format"] == fmt
    with pytest.raises(ValueError):
        T.set_prompt_format(t, "fancy")
    old = tmp_path / "linear"                                                    # a linear chain.json
    old.mkdir()
    (old / "chain.json").write_text(json.dumps({"tool": "video-chain", "settings": {}, "clips": []}))
    assert T.load(old).options["prompt_format"] == "classic"


def test_a_project_from_before_the_format_stays_done(project, tmp_path):
    """Made clips of a project file without the setting stay done (classic, recorded as before); switching to h3
    marks them changed."""
    t = T.load(project)
    T.add_character(project, t, "Anna", face(tmp_path / "a.jpg", (200, 150, 130)), desc=ANNA)
    a = t.add(None, "@Anna waves.")
    b = t.add(a.id, "She smiles.")
    T.save(project, t)
    run(project, [b.id])
    d = json.loads((project / T.TREE_FILE).read_text())
    del d["options"]["prompt_format"]
    (project / T.TREE_FILE).write_text(json.dumps(d))
    t = T.load(project)
    assert t.options["prompt_format"] == "classic" and set(statuses(project).values()) == {"done"}
    s = t.nodes[a.id].made["settings"]
    assert "prompt_format" not in s and "frame_pictures" not in s and s["cast"] == [{"name": "Anna", "desc": ANNA}]
    T.set_prompt_format(t, "h3")
    T.save(project, t)
    assert set(statuses(project).values()) == {"changed"}


def test_h3_base_formats(tmp_path):
    """Without references: three fields, with the header of what the clip has (nothing, start, start + end, end)."""
    fields = ("integrated_multimodal_description: [Shot 1] A lighthouse at dusk.\n\noverall_soundscape: Natural "
              "ambient sound that fits the scene.\n\nnon_diegetic_music: N/A")
    p, t = h3_project(tmp_path, start=False, name="text")
    a = t.add(None, "A lighthouse at dusk.")
    T.save(p, t)
    gen = run(p, [a.id])
    assert gen.calls[0]["start"] is None and gen.calls[0]["prompt"] == fields
    s = T.load(p).nodes[a.id].made["settings"]
    assert s["prompt_format"] == "h3" and "frame_pictures" not in s and "refs" not in s
    end = face(tmp_path / "end.png", (0, 0, 250))
    T.set_end_image(p, t, a.id, end)                                            # end only: L2VA
    T.save(p, t)
    gen = run(p, [a.id])
    assert gen.calls[0]["prompt"] == H.L2VA_HEADER.format(n=1, s="3.04") + "\n\n" + fields
    assert gen.calls[0]["end_rgb"] is not None

    p, t = h3_project(tmp_path, name="start")
    a = t.add(None, "A lighthouse at dusk.")
    T.save(p, t)
    gen = run(p, [a.id])
    assert gen.calls[0]["prompt"] == H.I2VA_HEADER + "\n\n" + fields                 # start frame: I2VA
    t = T.load(p)
    t.options["frames"] = 49
    T.set_end_image(p, t, a.id, end)
    b = t.add(a.id, "A lighthouse at dusk.")
    b.overrides = {"guide": 5}                                                  # its start is a motion guide
    T.save(p, t)
    gen = run(p, [b.id])
    assert gen.calls[0]["prompt"] == H.FL2VA_HEADER.format(n=1, s="2.04") + "\n\n" + fields   # 49 / 24 s
    assert gen.calls[1]["guides"] and gen.calls[1]["prompt"] == fields          # no plain start frame: no header


def test_h3_reference_format_numbers_the_start_frame(tmp_path):
    """A person with 2 pictures and a place, from the start image: the start frame is <Picture 4> (after the
    subject pictures), recorded as frame_pictures, sent after them; the end frame is <Picture 5>."""
    p, t = h3_project(tmp_path)
    anna_and_kitchen(p, t, tmp_path)
    a = t.add(None, "@Anna walks into @Kitchen and looks around, smiling.")
    T.save(p, t)
    gen = run(p, [a.id])
    call = gen.calls[0]
    assert call["prompt"] == (
        "subject_definitions:\n"
        f"<Subject 1> is Anna ({ANNA}), the person in <Picture 1> and <Picture 2>.\n"
        "<Subject 2> is Kitchen (a small sunny kitchen), the environment in <Picture 3>.\n"
        "<Picture 4> is the first frame of [Shot 1].\n\n"
        "summary:\n[keyframe completion + reference generation] The target video shows <Subject 1> and "
        "<Subject 2>. It begins from <Picture 4>.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - Anna's face and identity are retained.\n"
        "<Subject 2> (appears in [Shot 1]): fully_preserved - the layout and look of Kitchen are retained.\n"
        "<Picture 4> ([Shot 1] first frame): fully_preserved - the target video begins exactly from this picture."
        "\n\ndetailed_description:\n[Shot 1] The shot begins from <Picture 4>. <Subject 1>, woman, 30s, grey "
        "sweater, walks into <Subject 2>, a small sunny kitchen, and looks around, smiling.\n\n"
        "overall_soundscape:\nNatural ambient sound that fits the scene.\n\nnon_diegetic_music:\nN/A")
    s = call["settings"]
    assert s["frame_pictures"] == ["start"] and s["prompt_format"] == "h3" and s["ref_hint"] is True
    assert s["ref_paths"] == [str(p / r) for r in ("refs/ref1.png", "refs/ref2.png", "refs/ref3.png")]
    assert s["cast"] == [{"name": "Anna", "desc": ANNA},
                         {"name": "Kitchen", "desc": "a small sunny kitchen", "kind": "place"}]
    made = T.load(p).nodes[a.id].made["settings"]
    assert made["frame_pictures"] == ["start"] and "ref_paths" not in made
    assert statuses(p)[a.prompt] == "done"
    t = T.load(p)
    T.set_end_image(p, t, a.id, face(tmp_path / "end.png", (0, 0, 250)))
    T.save(p, t)
    info = T.clip_inputs(p, t, a.id)
    assert info["settings"]["frame_pictures"] == ["start", "end"] and info["model"] == "ref2va"
    assert [(x["n"], x["what"]) for x in info["pictures"]] == [(1, "@Anna"), (2, "@Anna"), (3, "@Kitchen"),
                                                               (4, "start frame"), (5, "end frame")]
    assert info["end"] == {"given": "frame", "file": p / t.nodes[a.id].end_image, "picture": 5}
    assert "<Picture 5> is the last frame of [Shot 1]." in info["prompt"]
    assert info["prompt"].split("detailed_description:\n")[1].split("\n\n")[0].endswith(
        "smiling. The shot ends on <Picture 5>.")
    gen = run(p, [a.id])
    assert gen.calls[0]["prompt"] == info["prompt"] and gen.calls[0]["settings"]["frame_pictures"] == ["start", "end"]


def test_h3_frame_pictures_fit_in_nine(tmp_path):
    """At most 9 pictures: the end frame loses its number first, then the start frame (both then in words)."""
    p, t = h3_project(tmp_path)
    for k, name in enumerate(("Ann", "Bea", "Cid")):
        T.add_character(p, t, name, face(tmp_path / f"{name}0.jpg", (10 * k, 0, 0)))
        for i in (1, 2):
            if k < 2 or i < 2:                                                  # 3 + 3 + 2 = 8 pictures
                T.add_picture(p, t, name, face(tmp_path / f"{name}{i}.jpg", (10 * k, 10 * i, 0)))
    a = t.add(None, "@Ann, @Bea and @Cid dance.")
    T.save(p, t)
    T.set_end_image(p, t, a.id, face(tmp_path / "end.png", (0, 0, 250)))
    info = T.clip_inputs(p, t, a.id)
    assert info["settings"]["frame_pictures"] == ["start"] and info["start"]["picture"] == 9
    assert info["end"]["picture"] is None and "The video ends exactly on the given end frame." in info["prompt"]
    assert "[keyframe completion + reference generation]" in info["prompt"]
    T.add_picture(p, t, "Cid", face(tmp_path / "Cid2.jpg", (20, 20, 0)))       # 9: no frame gets a number
    info = T.clip_inputs(p, t, a.id)
    assert "frame_pictures" not in info["settings"] and info["start"]["picture"] is None
    assert "The video begins exactly from the given start frame." in info["prompt"]
    assert "[reference generation]" in info["prompt"] and "<Picture 10>" not in info["prompt"]


def test_h3_lead_and_the_lead_switch(project, tmp_path):
    """A clip that mentions nobody gets the lead (h3: Subject 1, said so before [Shot 1]); switched off, it gets no
    pictures in either format; switched on again, it is recorded exactly as before."""
    t = T.load(project)
    T.add_character(project, t, "Anna", face(tmp_path / "a.jpg", (200, 150, 130)), desc=ANNA)
    a = t.add(None, 'She waves. "Hi."')
    T.save(project, t)
    ch = T.Checker(project, t)
    classic = ch.settings(a.id)
    assert classic["refs"] and T.node_cast(t.options, a)["mode"] == "lead"
    T.set_lead_on(a, False)
    assert a.overrides == {"lead": False}
    ch = T.Checker(project, t)
    assert "refs" not in ch.settings(a.id) and T.node_cast(t.options, a)["mode"] is None
    assert T.model_prompt(t.options, a, ch.settings(a.id)) == 'She waves. "Hi."'
    T.set_lead_on(a, True)
    assert a.overrides == {} and T.Checker(project, t).settings(a.id) == classic
    T.set_prompt_format(t, "h3")
    info = T.clip_inputs(project, t, a.id)
    assert info["subjects"] == [{"name": "Anna", "kind": "person", "pictures": [1], "files": ["refs/ref1.png"],
                                 "lead": True, "voice": None}]
    assert ("detailed_description:\nThe person in the video is <Subject 1>.\n[Shot 1] The shot begins from "
            "<Picture 2>. She waves. <Subject 1> (S1), woman, 30s, grey sweater, says, <d>[English] Hi.</d>") \
        in info["prompt"]
    T.set_lead_on(a, False)
    info = T.clip_inputs(project, t, a.id)
    assert info["subjects"] == [] and info["model"] == "fl2va" and "refs" not in info["settings"]
    assert info["prompt"] == (H.I2VA_HEADER + "\n\nintegrated_multimodal_description: [Shot 1] She waves. "
                              "<d>[English] Hi.</d>\n\noverall_soundscape: Natural ambient sound that fits the scene."
                              "\n\nnon_diegetic_music: N/A")
    T.save(project, t)
    assert T.load(project).nodes[a.id].overrides == {"lead": False}              # kept in tree.json


def test_references_of_any_kind(project, tmp_path):
    """Places, objects and styles: whole pictures, never the lead, no voice; named as what they are."""
    t = T.load(project)
    big = tmp_path / "room.png"
    Image.new("RGB", (120, 90), (90, 90, 200)).save(big)
    k = T.add_character(project, t, "Kitchen", big, desc="a small sunny kitchen", kind="place", crop=(0, 0, 40))
    assert k["kind"] == "place" and k["lead"] is False
    assert Image.open(project / k["images"][0]).size == (120, 90)               # not cropped
    T.add_picture(project, t, "Kitchen", big, crop=(0, 0, 40))
    assert Image.open(project / T.character(t, "Kitchen")["images"][1]).size == (120, 90)
    anna = T.add_character(project, t, "Anna", face(tmp_path / "a.jpg", (200, 150, 130)), crop=(0, 0, 40))
    assert anna["lead"] is True and "kind" not in anna and T.kind_of(anna) == "person"   # the first person
    assert Image.open(project / anna["images"][0]).size == (40, 40)
    with pytest.raises(ValueError, match="only a person can have a voice"):
        T.set_voice(project, t, "Kitchen", sound(tmp_path / "v.wav", 3))
    with pytest.raises(ValueError, match="only a person can be the lead"):
        T.update_character(t, "Kitchen", lead=True)
    with pytest.raises(ValueError, match="unknown kind"):
        T.set_kind(t, "Kitchen", "animal")
    T.add_character(project, t, "Cup", kind="object")
    T.save(project, t)
    t = T.load(project)
    assert [T.kind_of(c) for c in T.cast_of(t)] == ["place", "person", "object"]
    assert "kind" not in T.character(t, "Anna")                                  # persons are stored as before
    # classic: the tying sentence names the kind; a person's stays as it was
    use = T.clip_cast(t.options, "@Anna cooks in @Kitchen.")
    assert use["prompt"] == ("Anna (the person in <Picture 1>) cooks in Kitchen (the place in <Picture 2> and "
                             "<Picture 3>). Anna has exactly the face of the person in <Picture 1>. Kitchen is the "
                             "place shown in <Picture 2> and <Picture 3>. Kitchen: a small sunny kitchen. The face "
                             "pictures only show who the people are; their pose and framing come from the video.")
    assert use["record"] == [{"name": "Anna", "desc": ""},
                             {"name": "Kitchen", "desc": "a small sunny kitchen", "kind": "place"}]
    assert T.clip_cast(t.options, "A view of @Kitchen.")["prompt"] == (
        "A view of Kitchen (the place in <Picture 1> and <Picture 2>). Kitchen is the place shown in <Picture 1> and "
        "<Picture 2>. Kitchen: a small sunny kitchen.")
    assert T.dialogue(t.options, 'In @Kitchen, @Anna says "Hi."')[0]["speaker"]["name"] == "Anna"   # a place never speaks
    a = t.add(None, "@Cup falls.")
    assert T.cast_problems(project, t, [a.id]) == [f"@Cup (clip #{a.id[1:]}) has no picture yet"]
    T.set_voice(project, t, "Anna", sound(tmp_path / "v.wav", 3))
    with pytest.raises(ValueError, match="has a voice"):
        T.set_kind(t, "Anna", "object")
    T.remove_voice(project, t, "Anna")
    T.set_kind(t, "Anna", "object")                                              # no longer the lead
    assert T.lead_of(t) is None and T.character(t, "Anna")["lead"] is False
    T.set_kind(t, "Anna", "person")
    assert "kind" not in T.character(t, "Anna")
    assert T.add_reference(project, face(tmp_path / "b.jpg", (1, 2, 3)), t) in T.character(t, "Anna")["images"]


def test_h3_voices_sounds_and_a_given_soundtrack(tmp_path):
    """Voices come after the sounds the prompt names (and are sent in that order); with the clip's whole sound
    given, no voice is sent and the soundtrack is the last <Audio N>, reused 1:1."""
    p, t = h3_project(tmp_path)
    t.options["frames"] = 48
    T.add_character(p, t, "Anna", face(tmp_path / "a.jpg", (200, 150, 130)), desc=ANNA)
    T.set_voice(p, t, "Anna", sound(tmp_path / "anna.wav", 4))
    T.add_audio_reference(p, sound(tmp_path / "rain.wav", 3), t)
    a = t.add(None, '<Audio 1> rains outside. @Anna looks up and says "Good morning."\n'
                    "overall_soundscape: Rain on the window.\nnon_diegetic_music: N/A")
    T.save(p, t)
    gen = run(p, [a.id])
    call = gen.calls[0]
    pr, s = call["prompt"], call["settings"]
    assert s["audio_paths"] == [str(p / "refs/audio1.wav"), str(p / "refs/voice1.wav")]
    assert s["voice_language"] == "English" and s["frame_pictures"] == ["start"]
    assert "<Audio 2> is the voice-timbre reference for <Subject 1> (S1), the voice of one speaker." in pr
    assert ("[Shot 1] The shot begins from <Picture 2>. <Audio 1> rains outside. <Subject 1> (S1), woman, 30s, grey "
            "sweater, looks up and says, using the voice timbre referenced from <Audio 2>, <d>[English] Good "
            "morning.</d>\n\noverall_soundscape:\nRain on the window.\n\nnon_diegetic_music:\nN/A") in pr
    assert "[keyframe completion + reference generation + audio reference]" in pr
    info = T.clip_inputs(p, T.load(p), a.id)
    assert [(x["n"], x["what"]) for x in info["sounds"]] == [(1, "sound"), (2, "voice of @Anna")]
    assert info["subjects"][0]["voice"] == 2

    src = sound(tmp_path / "song.wav", 6)
    t = T.load(p)
    tid = T.add_track(p, t, src, 0.0, 5.0, "Song")["id"]
    t.nodes[a.id].overrides = {"source_sound": f"track:{tid}"}
    T.save(p, t)
    gen = run(p, [a.id])
    call = gen.calls[0]
    pr, s = call["prompt"], call["settings"]
    assert len(s["audio_paths"]) == 2 and s["audio_paths"][0] == str(p / "refs/audio1.wav")   # sound + soundtrack
    assert "voice_language" not in s and s["frame_pictures"] == ["start"]
    assert "<Audio 2> is the complete audio track of the target video, reused 1:1." in pr
    assert "[keyframe completion + reference generation + audio reuse + audio reference]" in pr
    assert T.TRACK_SOUND_TEXT.format(k=2) in pr and "voice timbre" not in pr
    info = T.clip_inputs(p, T.load(p), a.id)
    assert [x["what"] for x in info["sounds"]] == ["sound", "soundtrack"] and info["sounds"][1]["n"] == 2


def test_h3_style_triggers_camera_and_notes_inside_the_description(tmp_path):
    """The style text, trigger words, camera move and keyframe notes end up in the description; the user's own
    sound lines are cut off before the style suffix; a structured prompt of the user's goes as written."""
    p, t = h3_project(tmp_path, start=False)
    t.options["style"] = {"prefix": "Cinematic, warm light.", "suffix": "Film grain."}
    a = t.add(None, "A cat jumps. #push-in-slow.\noverall_soundscape: Soft thud.")
    a.overrides = {"camera": {"move": "pan-left"}}
    T.set_keyframe(p, t, a.id, face(tmp_path / "k.png", (1, 2, 3)), 1.0, "it lands")
    T.save(p, t)
    s = T.Checker(p, t).settings(a.id)
    s["lora_triggers"] = ["mycat"]
    pr = T.model_prompt(t.options, a, s)
    assert pr == ("integrated_multimodal_description: [Shot 1] mycat. Cinematic, warm light. A cat jumps. The camera "
                  "pushes in at slow speed. Film grain. The camera pans left. At 1.0 s: it lands.\n\n"
                  "overall_soundscape: Soft thud.\n\nnon_diegetic_music: N/A")
    own = "integrated_multimodal_description: [Shot 1] A dog.\n\noverall_soundscape: Barks.\n\nnon_diegetic_music: N/A"
    a.prompt = own
    assert T.model_prompt(t.options, a, s) == own


def test_multi_line_prompts_survive(tmp_path):
    """A refined h3 prompt has line breaks (shots, then the sound lines): kept by save / load and the checks; the
    2K refine gets the description without the sound lines (its camera sentence after the shots)."""
    p, t = h3_project(tmp_path)
    text = ('A man says "Hi."\n[Shot 2] At 00:00.500, he leaves.\noverall_soundscape: Steps.\n'
            "non_diegetic_music: N/A")
    a = t.add(None, text)
    a.overrides = {"camera": {"move": "static"}}
    T.save(p, t)
    assert T.load(p).nodes[a.id].prompt == text
    assert T.refine_problems('A man says "Hi."', text) == [] and T.spoken_lines(text) == ["Hi."]
    gen = run(p, [a.id])
    assert gen.calls[0]["prompt"].endswith('[Shot 1] A man says <d>[English] Hi.</d>\n[Shot 2] At 00:00.500, he '
                                           "leaves. The camera holds a static shot.\n\noverall_soundscape: Steps."
                                           "\n\nnon_diegetic_music: N/A")
    T.refine_path(gen, p, a.id, "2k", tmp_path / "out.mp4", log=lambda _m: None)
    assert gen.refines[0]["prompt"] == ('A man says "Hi."\n[Shot 2] At 00:00.500, he leaves. The camera holds a '
                                        "static shot.")


def test_the_same_first_clip_for_more_start_pictures(project, tmp_path):
    """A first clip can have a start picture of its own; add_start_rows makes the same clip once per picture
    (a set): each starts from its picture, fitted to the project's size, and can then go its own way."""
    t = T.load(project)
    a = t.add(None, "she waves", seed=7)
    a.overrides = {"frames": 49, "steps": 30}
    pics = []
    for k, (size, colour) in enumerate((((54, 96), (200, 10, 10)), ((96, 54), (10, 10, 200)))):   # a tall one too
        f = tmp_path / f"p{k}.jpg"
        Image.new("RGB", size, colour).save(f)
        pics.append(f)
    with pytest.raises(ValueError):
        T.add_start_rows(project, t, t.add(a.id, "later").id, pics)          # only a first clip
    new = T.add_start_rows(project, t, a.id, pics + [tmp_path / "missing.jpg"])
    assert len(new) == 2 and all(n.parent is None and n.prompt == "she waves" and n.seed == 7 for n in new)
    assert all(n.overrides == a.overrides and n.overrides is not a.overrides for n in new)
    assert a.row == a.id and {n.row for n in new} == {a.id} and a.start_image is None
    assert [n.id for n in T.row_members(t, new[0].id)] == [a.id, new[0].id, new[1].id]
    assert all(n.start_image.startswith("starts/") and (project / n.start_image).is_file() for n in new)
    ck = T.Checker(project, t)
    assert ck.start_frame(a) == project / T.START_FILE and ck.start_frame(new[0]) == project / new[0].start_image
    T.save(project, t)
    t2 = T.load(project)                                                      # kept in tree.json
    assert t2.nodes[new[0].id].start_image == new[0].start_image and t2.nodes[new[1].id].row == a.id
    assert t2.to_json() == t.to_json()
    # each row is made from its own picture, cut to the project's shape (the tall one is not stretched)
    gen = run(project, [a.id, new[0].id, new[1].id])
    starts = {c["start"] for c in gen.calls}
    stem0, stem1 = Path(new[0].start_image).stem, Path(new[1].start_image).stem
    assert starts == {"start_672x384.png", f"{stem0}_672x384.png", f"{stem1}_672x384.png"}
    with Image.open(project / "starts" / f"{stem0}_672x384.png") as im:
        assert im.size == (672, 384) and im.getpixel((5, 5))[0] > 150 and im.getpixel((660, 380))[0] > 150
    assert set(statuses(project).values()) == {"done"} | ({"new"} if "later" in statuses(project) else set())
    # a changed prompt goes to the other rows on request; seeds, pictures and frozen clips stay
    t = T.load(project)
    a, r0, r1 = (t.nodes[i] for i in (a.id, new[0].id, new[1].id))
    r0.seed = 99
    r1.frozen = True
    a.prompt, a.overrides = "she waves and smiles", {"frames": 73}
    assert T.apply_to_row(t, a.id) == [r0.id]
    assert r0.prompt == "she waves and smiles" and r0.overrides == {"frames": 73} and r0.seed == 99
    assert r1.prompt == "she waves" and r0.start_image == new[0].start_image
    assert T.apply_to_row(t, a.id) == [] and T.row_members(t, t.add(None, "alone").id) == []
    # another picture for one row: its clip counts as changed; back to the project's start image; removing
    T.save(project, t)
    old = r0.start_image
    T.set_node_start(project, t, r0.id, pics[1])
    assert r0.start_image != old and not (project / old).exists() and not list((project / "starts").glob(f"{stem0}_*"))
    T.save(project, t)
    assert T.Checker(project, T.load(project)).status(r0.id) == "changed"
    T.clear_node_start(project, t, r0.id)
    assert r0.start_image is None and T.Checker(project, t).start_frame(r0) == project / T.START_FILE
    kept = project / r1.start_image
    t.remove(r1.id, project)
    assert not kept.exists() and (project / "old" / "starts" / kept.name).is_file()
    # a damaged record never points outside starts/
    d = t.to_json()
    d["nodes"][0]["start_image"] = "../elsewhere.png"
    assert T.Tree.from_json(d).nodes[d["nodes"][0]["id"]].start_image is None


# ----------------------------------------------------------------------------- continue the clip before (<Video 1>)

class TailGenerator(FakeGenerator):
    """FakeGenerator that also looks at the reference video it gets (cut in a temp folder that is gone after the
    clip): its frame count and the colour of its last frame."""

    def generate(self, prompt, start_image, out_video, out_last_frame, seed, settings=None):
        vids = []
        for v in (settings or {}).get("ref_videos") or []:
            last = Path(v).with_suffix(".last.png")
            chain.last_frame(Path(v), last)
            vids.append({"frames": chain.probe(Path(v))["frames"],
                         "last_rgb": Image.open(last).convert("RGB").getpixel((5, 5))})
        super().generate(prompt, start_image, out_video, out_last_frame, seed, settings)
        self.calls[-1]["ref_videos"] = vids


CONT_DEF = ("<Video 1> is the source video for the target video continuation; the target video continues directly "
            "from its end.")
CONT_KEEP = ("<Video 1> (continuation source, its end): fully_preserved - the subjects, scene, motion and camera "
             "movement at its end carry on directly into [Shot 1].")


def test_h3_continue_the_clip_before(tmp_path):
    """Override "continue": the clip before's last n frames (up to its hand-over frame) go as the reference video,
    recorded as settings["continue"] in h3 only; the prompt says so; the lengths offered fit both clips; the clip
    becomes changed when the clip before is made again or hands over elsewhere."""
    p, t = h3_project(tmp_path)
    t.options["frames"] = 49
    a = t.add(None, "A red car drives along a coast road.")
    b = t.add(a.id, "The car drives on.")
    assert T.continue_choices(t, a) == [] and T.continue_choices(t, b) == [22, 39]     # 49 frames each
    b.overrides = {"frames": 25}
    assert T.continue_choices(t, b) == [22]                     # (no longer than the clip itself)
    b.overrides = {"continue": 39}
    a.overrides = {"continue": 22}                              # a first clip: nothing to continue, not recorded
    T.save(p, t)
    ch = T.Checker(p, t)
    assert ch.settings(b.id)["continue"] == {"frames": 39} and "continue" not in ch.settings(a.id)
    assert ch.settings(b.id)["frame_pictures"] == ["start"]     # the reference model: the start frame is numbered
    assert T.uses_ref_model(ch.settings(b.id)) and not T.uses_ref_model(ch.settings(a.id))
    classic = T.Tree.from_json(t.to_json())
    T.set_prompt_format(classic, "classic")
    s = T.Checker(p, classic).settings(b.id)
    assert "continue" not in s and not T.uses_ref_model(s)      # classic: no effect
    gen = run(p, [b.id], TailGenerator(frames=49))
    call = gen.calls[1]
    assert call["ref_videos"][0]["frames"] == 39
    a_last = Image.open(p / T.load(p).nodes[a.id].made["frame"]).convert("RGB").getpixel((5, 5))
    assert call["ref_videos"][0]["last_rgb"] == a_last == call["start_rgb"]   # it ends where this clip starts
    assert call["prompt"] == (
        "subject_definitions:\n<Picture 1> is the first frame of [Shot 1] and the last frame of <Video 1>.\n"
        f"{CONT_DEF}\n\n"
        "summary:\n[keyframe completion + video continuation] The target video continues directly from the end of "
        "<Video 1>. It follows the detailed description. It begins from <Picture 1>, the last frame of <Video 1>."
        "\n\nretention_analysis:\n<Picture 1> ([Shot 1] first frame): fully_preserved - the target video begins "
        f"exactly from this picture.\n{CONT_KEEP}\n\n"
        "detailed_description:\n[Shot 1] The shot continues directly from the end of <Video 1>. It begins from "
        "<Picture 1>, the last frame of <Video 1>. The car drives on.\n\n"
        "overall_soundscape:\nNatural ambient sound that fits the scene.\n\nnon_diegetic_music:\nN/A")
    assert "continue" not in gen.calls[0]["settings"] and not gen.calls[0]["ref_videos"]
    t = T.load(p)
    assert t.nodes[b.id].made["settings"]["continue"] == {"frames": 39} and "ref_videos" not in t.nodes[b.id].made["settings"]
    assert t.nodes[b.id].made["continue_from"] == t.nodes[a.id].made["video"]
    info = T.clip_inputs(p, t, b.id)
    assert info["videos"] == [{"n": 1, "what": f"end of clip #{a.id[1:]}", "frames": 39, "node": a.id}]
    assert info["model"] == "ref2va" and info["prompt"] == call["prompt"]
    assert statuses(p) == {a.prompt: "done", b.prompt: "done"}
    # the clip before as another file (same frame, so only the continuation sees it): changed
    old = p / t.nodes[a.id].made["video"]
    new = old.with_name(old.stem + "x.mp4")
    shutil.copy(old, new)
    t.nodes[a.id].made["video"] = f"nodes/{new.name}"
    ch = T.Checker(p, t)
    assert ch.status(a.id) == "done" and ch.status(b.id) == "changed" and ch.why(b.id) == "the clip it continues changed"
    t.nodes[a.id].made["video"] = f"nodes/{old.name}"
    t.nodes[a.id].end_at = 30                                  # it hands over earlier: the tail ends there
    ch = T.Checker(p, t)
    assert ch.continue_from(t.nodes[b.id]).endswith("@30") and ch.status(b.id) == "changed"
    assert T.continue_choices(t, t.nodes[b.id]) == [22]        # 31 frames up to the hand-over
    assert T.guide_problem(t, ch, [b.id]) == f"clip #{a.id[1:]} (31 frames) is shorter than the 39-frame " \
                                              f"continuation of clip #{b.id[1:]}"
    t.nodes[b.id].overrides["frames"] = 25
    t.nodes[a.id].end_at = None
    assert T.guide_problem(t, T.Checker(p, t), [b.id]) == (f"clip #{b.id[1:]} (25 frames) is shorter than its "
                                                           "39-frame continuation; make it longer or the "
                                                           "continuation shorter")


def test_h3_continue_with_carry_over_and_a_remade_clip_before(tmp_path):
    """With Carry-over at the start there is no plain start frame: the continuation sentence alone, no picture; the
    clip still becomes changed when the clip before is made again (its frames and its video change)."""
    p, t = h3_project(tmp_path)
    t.options["frames"] = 49
    a = t.add(None, "A red car drives along a coast road.")
    b = t.add(a.id, "The car drives on.")
    b.overrides = {"guide": 5, "continue": 22}
    T.save(p, t)
    gen = run(p, [b.id], TailGenerator(frames=49))
    call = gen.calls[1]
    assert call["guides"] and call["ref_videos"][0]["frames"] == 22
    assert call["prompt"] == (
        f"subject_definitions:\n{CONT_DEF}\n\n"
        "summary:\n[video continuation] The target video continues directly from the end of <Video 1>. It follows "
        f"the detailed description.\n\nretention_analysis:\n{CONT_KEEP}\n\n"
        "detailed_description:\n[Shot 1] The shot continues directly from the end of <Video 1>. The car drives "
        "on.\n\noverall_soundscape:\nNatural ambient sound that fits the scene.\n\nnon_diegetic_music:\nN/A")
    assert "frame_pictures" not in call["settings"] and "Picture" not in call["prompt"]
    assert statuses(p) == {a.prompt: "done", b.prompt: "done"}
    t = T.load(p)
    t.nodes[a.id].seed += 1                                   # the clip before is made again
    T.save(p, t)
    run(p, [a.id], gen)                                       # (the same fake: another last-frame colour)
    t = T.load(p)
    ch = T.Checker(p, t)
    assert ch.status(a.id) == "done" and ch.status(b.id) == "changed"
    m = t.nodes[b.id].made
    assert m["continue_from"] != ch.continue_from(t.nodes[b.id]) and m["guide_from"] != ch.guide_from(t.nodes[b.id])
    assert m["start_sha1"] != ch.sha(ch.start_frame(t.nodes[b.id]))   # (the start frame is checked with carry-over too)


def test_joined_sound_start_is_where_the_joined_path_hears_the_clip(tmp_path):
    """The sound the joined path has in a clip starts after the clip's repeated first frame(s), whatever its Sound
    setting; the clip's own sound span starts that much earlier."""
    t = T.Tree()
    f = str(tmp_path / "talk.mp4")
    v = t.add_video(None, f, 2.0, 3.0)
    a = t.add(v.id, "she keeps talking")
    b = t.add(a.id, "and talking")
    b.overrides = {"source_sound": True, "guide": 5}
    lone = t.add(None, "no video before")
    before = dict(a.overrides or {})
    assert T.joined_sound_start(t, a) == {"video": v, "file": f, "at": 3.0}
    assert (a.overrides or {}) == before                              # (the node is not changed)
    a.overrides = {"source_sound": True}
    assert T.source_sound_span(t, a)["start"] == round(3.0 - 1 / chain.FPS, 3)
    assert T.joined_sound_start(t, a)["at"] == 3.0
    assert T.joined_sound_start(t, b)["at"] == round(T.source_sound_span(t, b)["start"] + 5 / chain.FPS, 3)
    assert T.joined_sound_start(t, v) is None and T.joined_sound_start(t, lone) is None


def test_seam_problem_names_a_continuation_cut_from_another_clip(tmp_path, monkeypatch):
    """status and why count a changed continue_from; the seam check says it too (as it does for guide_from)."""
    p, t = h3_project(tmp_path)
    a = t.add(None, "A red car.")
    b = t.add(a.id, "It drives on.")
    ch = T.Checker(p, t)
    b.made = {"video": "b.mp4", "start_sha1": "s", "guide_from": None, "end_key": None, "continue_from": "a1.mp4@3"}
    monkeypatch.setattr(ch, "sha", lambda _f: "s")
    monkeypatch.setattr(ch, "start_frame", lambda _n: "frame")
    monkeypatch.setattr(ch, "guide_from", lambda _n: None)
    monkeypatch.setattr(ch, "end_key", lambda _n: None)
    monkeypatch.setattr(ch, "continue_from", lambda _n: "a2.mp4@3")       # the clip before was made again
    assert ch.seam_problem(b) and "continu" in ch.seam_problem(b)
    monkeypatch.setattr(ch, "continue_from", lambda _n: "a1.mp4@3")
    assert ch.seam_problem(b) is None


def test_another_h3_checkpoint(project):
    """A chosen H3 checkpoint is recorded with every clip (made clips change); a distilled one takes its few steps
    for every clip and switches the speed modes off; without one the clips are as before."""
    t, a, b1, b2, c = build(project)
    run(project, [c.id])
    t = T.load(project)
    t.options["checkpoint"] = {"name": "h3/fine.safetensors", "distilled": True}
    t.options["hyperflow"] = {"on": True, "strength": 1.0}
    t.nodes[b1.id].overrides = {"steps": 40}
    T.save(project, t)
    ch = T.Checker(project, t)
    s = ch.settings(b1.id)
    assert s["checkpoint"] == {"name": "h3/fine.safetensors", "distilled": True}
    assert s["steps"] == T.DISTILLED_STEPS == 8 and "hyperflow" not in s and "turbo" not in s
    assert ch.base()["checkpoint"] == s["checkpoint"]
    assert statuses(project) == {"a": "changed", "b1": "changed", "c": "changed", "b2": "new"}
    t.options["checkpoint"]["distilled"] = False                  # a full fine-tune: its speed mode and steps
    t.options["hyperflow"]["on"] = False
    T.save(project, t)
    s = T.Checker(project, t).settings(b1.id)
    assert s["checkpoint"]["distilled"] is False and s["steps"] == 40
    t.options["checkpoint"] = {"name": "", "distilled": False}    # the standard model again
    t.nodes[b1.id].overrides = {}
    T.save(project, t)
    assert "checkpoint" not in T.Checker(project, t).settings(b1.id)
    assert statuses(project) == {"a": "done", "b1": "done", "c": "done", "b2": "new"}
