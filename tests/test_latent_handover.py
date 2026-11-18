"""Latent handover (option "latent_handover", NikoDemon80's ComfyUI-H3-Motion-Context nodes): a clip's carry-over is
sliced straight out of the clip before's latent (picture and sound) instead of its decoded frames, so the join has no
decode / re-encode round trip. Every clip then saves its latent next to its video. Without the latent (a clip made
before, a real video, another size, an earlier hand-over frame) the carry-over is cut from the frames as before."""
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from video_chain import chain, comfy as C, tree as T
from video_chain.setup import manifest as M


def _wf(**kw):
    return C.build_workflow(prompt="x", image=kw.pop("image", None), seed=1, width=512, height=288, frames=73,
                            steps=6, loras=[], audio=True, prefix="video-chain/p", **kw)


# ---------------------------------------------------------------- the workflow
def test_without_it_the_workflow_is_as_before():
    assert _wf() == _wf(context_latent=None, save_latent=False)


def test_the_context_is_loaded_pinned_and_the_new_latent_saved():
    wf = _wf(context_latent="video-chain/abc.safetensors", context_frames=22, save_latent=True,
             refs=["r.png"], keyframes=[("k.png", 60)], pin_noise=0.95)
    load = wf["mc_load"]
    assert load["class_type"] == "MiniMaxH3MotionContextLoadLatent"
    assert load["inputs"] == {"latent_path": "video-chain/abc.safetensors", "clip_index": 1, "folder": "input"}
    mc = wf["mc"]
    assert mc["class_type"] == "MiniMaxH3MotionContext"
    assert mc["inputs"]["context_latent"] == ["mc_load", 0] and mc["inputs"]["context_length"] == "22"
    assert mc["inputs"]["audio_context_length"] == C.CONTEXT_AUDIO_FRAMES == 18
    assert mc["inputs"]["conditioning"] == ["key_guide0", 0] and mc["inputs"]["latent"] == ["refcond", 1]
    assert wf["pin_noise"]["inputs"]["conditioning"] == ["mc", 0]               # (after the pinned head)
    assert "start" not in wf and "cond" not in wf                               # (no start frame: the head is it)
    save = wf["mc_save"]
    assert save["class_type"] == "MiniMaxH3MotionContextSaveLatent"
    assert save["inputs"]["latent"] == ["sample", 0] and save["inputs"]["clip_index"] == 0
    assert save["inputs"]["filename_prefix"] == "video-chain/p_latent"


def test_a_first_clip_only_saves_its_latent():
    wf = _wf(image="s.png", save_latent=True)
    assert "mc" not in wf and "mc_load" not in wf and wf["mc_save"]["inputs"]["latent"] == ["sample", 0]
    with pytest.raises(ValueError, match="carry-over"):
        _wf(context_latent="a.safetensors", context_frames=10)


def test_two_passes_save_the_full_size_latent_but_cannot_pin_one():
    """Pass 1 is made at half the size; a latent cannot be resized (Motion Context refuses), so a two-pass clip keeps
    the carry-over from the frames; its full-size latent is saved for the clip after it."""
    from video_chain.comfy.video import build_two_pass_workflow
    kw = dict(two_pass={"scale": 0.5, "denoise": 0.3}, upscaler="u.safetensors", prompt="x", image=None, seed=1,
              width=512, height=288, frames=73, steps=8, loras=[], audio=True, prefix="p")
    wf = build_two_pass_workflow(**kw, save_latent=True)
    assert wf["mc_save"]["inputs"]["latent"] == ["sample", 0] and "p1_mc_save" not in wf
    with pytest.raises(ValueError, match="two passes"):
        build_two_pass_workflow(**kw, context_latent="a.safetensors", context_frames=22)


# ---------------------------------------------------------------- the client
def test_generate_uploads_the_context_and_downloads_the_new_latent(tmp_path):
    from fake_comfy import FakeComfy
    server = FakeComfy(tmp_path / "comfy")
    try:
        (server.root / "main.py").write_text("")
        ctx = tmp_path / "before.latent.safetensors"
        ctx.write_bytes(b"latent of the clip before")
        s = chain.settings_dict(model="Comfy-Org/MiniMax-H3", lora=None, strength=1.0, frames=73, steps=6,
                                width=512, height=288, audio=True)
        call = {**s, "context_latent": {"file": str(ctx), "frames": 22}, "latent_out": str(tmp_path / "new.lat")}
        C.ComfyGenerator(server.url, comfy_dir=server.root).generate("x", None, tmp_path / "v.mp4", tmp_path / "v.png",
                                                                    1, call)
        wf = server.prompts[-1]
        assert wf["mc_load"]["inputs"]["latent_path"].endswith(".safetensors")
        assert server.context_latents == [ctx.read_bytes()]
        assert (tmp_path / "new.lat").read_bytes() == b"fake H3 latent"
        server.motion_context = False
        with pytest.raises(C.ComfyError, match="Motion-Context"):
            C.ComfyGenerator(server.url, comfy_dir=server.root).generate("x", None, tmp_path / "w.mp4",
                                                                        tmp_path / "w.png", 1, call)
    finally:
        server.close()


# ---------------------------------------------------------------- the tree
class LatentGenerator:
    """Clips as ffmpeg test pictures; with latent_out also a latent file (its bytes name the clip)."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, start_image, out_video, out_last_frame, seed, settings=None):
        s = settings or {}
        self.calls.append({"start": start_image, "settings": s})
        w, h, n = s.get("width", 96), s.get("height", 64), s.get("frames", 73)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=24",
                        "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_video)], check=True)
        chain.frame_at(Path(out_video), n - 1, Path(out_last_frame))
        if s.get("latent_out"):
            Path(s["latent_out"]).write_bytes(f"latent {len(self.calls)}".encode())


@pytest.fixture
def chained(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    img = tmp_path / "in.png"
    Image.new("RGB", (96, 64), (90, 120, 150)).save(img)
    t = T.Tree()
    T.set_start_image(p, img, t)
    t.options["frames"] = 73
    t.options["latent_handover"] = True
    a = t.add(None, "a")
    b = t.add(a.id, "b")
    t.nodes[b.id].overrides = {"guide": 22}
    T.save(p, t)
    return p, a.id, b.id


def _run(p, ids):
    gen = LatentGenerator()
    T.run_tree(gen, p, ids, log=lambda _m: None)
    return gen


def test_the_carry_over_comes_from_the_latent(chained):
    p, a, b = chained
    gen = _run(p, [b])
    t = T.load(p)
    lat_a = p / t.nodes[a].made["latent"]
    assert lat_a.read_bytes() == b"latent 1" and t.nodes[b].made["latent"]
    first, second = gen.calls
    assert "context_latent" not in first["settings"] and first["settings"]["latent_out"]
    ctx = second["settings"]["context_latent"]
    assert Path(ctx["file"]) == lat_a and ctx["frames"] == 22
    assert second["start"] is None and not any(g.get("at") == "start" for g in second["settings"].get("guides") or [])
    ch = T.Checker(p, t)
    assert ch.status(a) == "done" and ch.status(b) == "done"
    assert lat_a in T.clip_files(p, t.nodes[a])                                  # (moved and removed with it)


def test_switching_it_off_makes_the_latent_clips_again(chained):
    p, a, b = chained
    _run(p, [b])
    t = T.load(p)
    t.options["latent_handover"] = False
    T.save(p, t)
    ch = T.Checker(p, T.load(p))
    assert ch.status(a) == "done" and ch.status(b) == "changed"


def test_without_a_latent_the_carry_over_is_cut_from_the_frames(chained):
    p, a, b = chained
    t = T.load(p)
    t.options["latent_handover"] = False
    T.save(p, t)
    _run(p, [a])                                                                 # (made before: no latent)
    t = T.load(p)
    assert "latent" not in t.nodes[a].made
    t.options["latent_handover"] = True
    T.save(p, t)
    gen = _run(p, [b])
    s = gen.calls[0]["settings"]
    assert "context_latent" not in s and any(g.get("at") == "start" for g in s.get("guides") or [])
    assert T.Checker(p, T.load(p)).status(b) == "done"


def test_an_earlier_hand_over_frame_uses_the_frames(chained):
    p, a, b = chained
    _run(p, [a])
    t = T.load(p)
    t.nodes[a].end_at = 30
    T.save(p, t)
    gen = _run(p, [b])
    assert "context_latent" not in gen.calls[0]["settings"]


def test_projects_from_before_have_it_off():
    assert T.DEFAULT_OPTIONS["latent_handover"] is False
    assert T.Tree.from_json({"options": {}, "nodes": []}).options["latent_handover"] is False
    assert "latent" in T.CLIP_KEYS


def test_setup_installs_the_motion_context_nodes():
    man = M.load()
    n = man["nodes"]["h3-motion-context"]
    assert n["git"] == "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context"
    assert n["rev"] == "518c083d11d80fa5920222ad89889924ee4401c9" and n["licence"] == "GPL-3.0"
    assert n["dir"] == "ComfyUI-H3-Motion-Context"
    assert "h3-motion-context" in json.dumps(man["components"])


def test_a_clip_before_with_a_replaced_sound_hands_over_from_the_frames(chained):
    """Review: the latent holds the model's own sound; a clip before whose sound was laid over or mixed afterwards
    would have handed over a sound that is not the one heard at the join."""
    p, a, b = chained
    _run(p, [a])
    t = T.load(p)
    t.nodes[a].made["own_sound"] = "nodes/x.own.m4a"
    T.save(p, t)
    assert T.Checker(p, T.load(p)).latent_source(T.load(p).nodes[b]) is None
    t = T.load(p)
    del t.nodes[a].made["own_sound"]
    t.nodes[a].made["settings"]["source_sound"] = {"file": "v.mp4", "start": 0}
    T.save(p, t)
    assert T.Checker(p, T.load(p)).latent_source(T.load(p).nodes[b]) is None


def test_a_clip_with_its_own_given_sound_takes_the_frames(chained, monkeypatch):
    """Review: a clip made to a given sound (lip sync) would have got the clip before's sound pinned on top."""
    p, a, b = chained
    _run(p, [a])
    ch = T.Checker(p, T.load(p))
    s = dict(ch.settings(b))
    s["source_sound"] = {"file": "v.mp4", "start": 0}
    monkeypatch.setattr(ch, "settings", lambda nid: s if nid == b else T.Checker.settings(ch, nid))
    assert ch.latent_source(T.load(p).nodes[b]) is None


def test_a_latent_no_record_names_is_cleaned_up(chained):
    """Review: an orphaned latent (a killed run) was never moved aside like the other clip files."""
    p, a, b = chained
    _run(p, [a])
    stray = p / "nodes" / "n9_deadbeef.latent.safetensors"
    stray.write_bytes(b"x")
    T.remove_stale_files(p, T.load(p))
    assert not stray.exists()


def test_switching_it_on_leaves_finished_clips_done(chained):
    """Clips made before have no latent: they stay done; a clip made again afterwards keeps its latent."""
    p, a, b = chained
    t = T.load(p)
    t.options["latent_handover"] = False
    T.save(p, t)
    _run(p, [b])
    t = T.load(p)
    t.options["latent_handover"] = True
    T.save(p, t)
    ch = T.Checker(p, T.load(p))
    assert ch.status(a) == "done" and ch.status(b) == "done"
