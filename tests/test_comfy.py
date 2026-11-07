"""ComfyGenerator against a fake ComfyUI server (synthetic clips; no ComfyUI, no GPU)."""
import shutil

import pytest
from fake_comfy import FakeComfy
from PIL import Image

from video_chain import chain
from video_chain import comfy as C

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


@pytest.fixture
def server(tmp_path):
    s = FakeComfy(tmp_path / "comfy")
    yield s
    s.close()


@pytest.fixture
def start(tmp_path):
    p = tmp_path / "start.png"
    Image.new("RGB", (192, 108), (10, 200, 10)).save(p)
    return p


def settings(**kw):
    s = chain.settings_dict(model="Comfy-Org/MiniMax-H3", lora=None, strength=1.0, frames=39, steps=20, width=672,
                            height=384, audio=True, backend="comfyui")
    s.update(kw)
    return s


def test_workflow_follows_the_blueprint():
    wf = C.build_workflow(prompt="a -- b", image="video-chain/x.png", seed=7, width=672, height=384, frames=73,
                          steps=20, loras=[("A/one.safetensors", 0.8), ("two.safetensors", 0.5)], audio=True,
                          prefix="video-chain/abc")
    t = {k: v["class_type"] for k, v in wf.items()}
    assert t["cond"] == "MiniMaxH3ImageToVideo" and t["sample"] == "SamplerCustomAdvanced"
    assert wf["cond"]["inputs"]["prompt"] == "a -- b" and wf["cond"]["inputs"]["first_frame"] == ["start", 0]
    assert wf["clip"]["inputs"]["type"] == "minimax" and wf["sampler"]["inputs"]["sampler_name"] == "res_multistep"
    # LoRAs chained: unet -> lora0 -> lora1 -> scheduler and guider
    assert wf["lora0"]["inputs"]["model"] == ["unet", 0] and wf["lora1"]["inputs"]["model"] == ["lora0", 0]
    assert wf["guider"]["inputs"]["model"] == ["lora1", 0] and wf["sigmas"]["inputs"]["model"] == ["lora1", 0]
    assert wf["lora0"]["inputs"]["strength_model"] == 0.8
    assert wf["last"]["inputs"]["batch_index"] == 72 and wf["save_last"]["class_type"] == "SaveImage"
    assert wf["video"]["inputs"]["audio"] == ["decode_audio", 0] and wf["video"]["inputs"]["fps"] == 24.0
    silent = C.build_workflow(prompt="x", image="i.png", seed=1, width=512, height=512, frames=39, steps=4,
                              loras=[], audio=False, prefix="p")
    assert "audio" not in silent["video"]["inputs"] and "decode_audio" not in silent
    assert silent["guider"]["inputs"]["model"] == ["unet", 0]


def test_lora_names():
    names = ["MiniMax H3/glow.safetensors", "nose.safetensors", "other/x.safetensors"]
    assert C.lora_name_for("/m/my_models/loras/MiniMax H3/glow.safetensors", names) == "MiniMax H3/glow.safetensors"
    assert C.lora_name_for("/elsewhere/glow.safetensors", names) == "nose.safetensors"
    assert C.lora_name_for("/elsewhere/unknown.safetensors", names) is None


def test_generate_one_clip(server, start, tmp_path):
    steps = []
    gen = C.ComfyGenerator(server.url, progress=lambda i, n: steps.append((i, n)))
    video, frame = tmp_path / "c.mp4", tmp_path / "c.png"
    lora = "/data/my_models/loras/MiniMax H3/glow.safetensors"
    gen.generate("a soft glow appears", start, video, frame, 5, settings(lora=lora, strength=0.7))
    info = chain.probe(video)
    assert (info["width"], info["height"], info["frames"], info["audio"]) == (672, 384, 39, True)
    assert Image.open(frame).size == (672, 384)
    wf = server.prompts[-1]
    assert wf["lora0"]["inputs"]["lora_name"] == "MiniMax H3/glow.safetensors"
    assert wf["noise"]["inputs"]["noise_seed"] == 5 and wf["sigmas"]["inputs"]["steps"] == 20
    assert list((server.root / "input" / "video-chain").glob("*.png"))               # start image uploaded


def test_missing_lora_and_models(server, tmp_path):
    gen = C.ComfyGenerator(server.url)
    assert gen.missing_loras(["/x/MiniMax H3/glow.safetensors", "/x/mine.safetensors"]) == ["/x/mine.safetensors"]
    bad = FakeComfy(tmp_path / "c2", models_missing=True)
    try:
        with pytest.raises(C.ComfyError, match="does not list minimax_h3_fl2va"):
            C.ComfyGenerator(bad.url)
    finally:
        bad.close()


def test_error_in_comfyui(tmp_path, start):
    s = FakeComfy(tmp_path / "c3", fail_with="CUDA out of memory")
    try:
        gen = C.ComfyGenerator(s.url)
        with pytest.raises(C.ComfyError, match="CUDA out of memory"):
            gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, settings())
    finally:
        s.close()


def test_not_running():
    with pytest.raises(C.ComfyUnavailable):
        C.ComfyGenerator("http://127.0.0.1:9")


def test_free_and_interrupt(server):
    gen = C.ComfyGenerator(server.url)
    gen.free()
    gen.current = "p9"
    gen.interrupt()
    bodies = [c for c in server.calls if c[0] == "BODY"]
    assert ("BODY", "/free", {"unload_models": True, "free_memory": True}) in bodies
    assert ("BODY", "/queue", {"delete": ["p9"]}) in bodies and ("BODY", "/interrupt", {"prompt_id": "p9"}) in bodies


def test_websocket_messages(server):
    seen = []
    gen = C.ComfyGenerator(server.url, progress=lambda i, n: seen.append((i, n)))
    assert gen.handle_message({"type": "progress", "data": {"value": 3, "max": 20, "prompt_id": "p1"}}, "p1") is None
    assert gen.handle_message({"type": "progress", "data": {"value": 9, "max": 20, "prompt_id": "other"}},
                              "p1") is None
    assert seen == [(3, 20)]
    assert gen.handle_message({"type": "executing", "data": {"node": None, "prompt_id": "p1"}}, "p1") == "done"
    with pytest.raises(C.ComfyError, match="boom"):
        gen.handle_message({"type": "execution_error", "data": {"prompt_id": "p1", "exception_message": "boom"}},
                           "p1")


def test_real_worker_with_comfyui(server, tmp_path):
    """The real CLI worker (--serve) end to end against the fake ComfyUI."""
    import json
    import subprocess
    import sys

    from video_chain import tree as T
    img = tmp_path / "in.png"
    Image.new("RGB", (192, 108), (200, 10, 10)).save(img)
    proj = tmp_path / "proj"
    t = T.Tree()
    T.set_start_image(proj.mkdir() or proj, img, t)
    t.options["frames"] = 39
    a = t.add(None, "first")
    b = t.add(a.id, "second")
    T.save(proj, t)
    p = subprocess.Popen([sys.executable, "-m", "video_chain", "--serve", "--events",
                          "--comfy-url", server.url, "--idle-minutes", "0"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    events = []

    def until(kind):
        for line in p.stdout:
            if line.startswith("@@ "):
                events.append(json.loads(line[3:]))
                if events[-1]["e"] == kind:
                    return
        raise AssertionError(f"no {kind}: {events}")
    until("idle")
    p.stdin.write(json.dumps({"cmd": "run", "tree": str(proj), "targets": [b.id]}) + "\n")
    p.stdin.flush()
    until("done")
    until("idle")
    p.stdin.write('{"cmd": "quit", "free": true}\n')
    p.stdin.flush()
    until("bye")
    p.wait(timeout=10)
    kinds = [e["e"] for e in events]
    assert kinds[:3] == ["loading", "loaded", "idle"] and "error" not in kinds
    assert [e["state"] for e in events if e["e"] == "node"] == ["generating", "done", "generating", "done"]
    t = T.load(proj)
    ch = T.Checker(proj, t)
    assert ch.status(a.id) == ch.status(b.id) == "done"
    assert t.nodes[b.id].made["settings"]["backend"] == "comfyui"
    assert chain.probe(proj / t.nodes[b.id].made["video"])["frames"] == 39
    # clip 2 started from clip 1's last frame (uploaded to ComfyUI)
    assert len(list((server.root / "input" / "video-chain").glob("*.png"))) == 2
    assert ("BODY", "/free", {"unload_models": True, "free_memory": True}) in server.calls


def test_cleanup_removes_only_comfyuis_copies(server, start, tmp_path):
    (server.root / "main.py").write_text("# fake ComfyUI folder\n")
    keep = server.root / "output" / "other.png"
    keep.write_bytes(b"not ours")
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, settings())
    assert (tmp_path / "v.mp4").stat().st_size > 0 and (tmp_path / "v.png").stat().st_size > 0
    assert not list((server.root / "output" / "video-chain").glob("*"))         # clip and last frame removed
    assert not list((server.root / "input" / "video-chain").glob("*"))          # uploaded start image removed
    assert keep.read_bytes() == b"not ours"
    gen2 = C.ComfyGenerator(server.url, comfy_dir=server.root, cleanup=False)
    gen2.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 1, settings())
    assert len(list((server.root / "output" / "video-chain").glob("*"))) == 2


def test_uploads_removed_when_generation_fails(server, start, tmp_path):
    (server.root / "main.py").write_text("# fake ComfyUI folder\n")
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)

    def fail(*_a, **_k):
        raise C.ComfyError("stopped")
    gen._wait = fail
    with pytest.raises(C.ComfyError):
        gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, settings())
    assert not list((server.root / "input" / "video-chain").glob("*"))


def test_lora_found_through_a_link(tmp_path):
    """A LoRA chosen at its original path is found when ComfyUI lists a link to it (extra_model_paths.yaml)."""
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    models = tmp_path / "my_models"
    (models / "loras" / "MiniMax H3").mkdir(parents=True)
    (comfy / "extra_model_paths.yaml").write_text(
        f"my_models:\n    base_path: {models}/\n    loras: loras/\n    vae: vae/\n")
    trained = tmp_path / "training" / "glow_h3_lora.safetensors"
    trained.parent.mkdir()
    trained.write_bytes(b"weights")
    (models / "loras" / "MiniMax H3" / "glow_step500.safetensors").symlink_to(trained)
    roots = C.comfy_lora_roots(comfy)
    assert models / "loras" in roots
    names = ["MiniMax H3/glow_step500.safetensors", "MiniMax H3/other.safetensors"]
    assert C.lora_name_for(str(trained), names, roots) == "MiniMax H3/glow_step500.safetensors"
    assert C.lora_name_for(str(trained), names) is None                         # without the folders: unknown



def test_h3_lora_detection(tmp_path):
    from test_ui import H3_KEYS, write_lora
    write_lora(tmp_path / "peft.safetensors", H3_KEYS)
    write_lora(tmp_path / "kohya.safetensors", ["lora_unet_blocks_3_attn_qkv_proj.lora_down.weight",
                                                "lora_unet_blocks_3_mlp_fc1.lora_down.weight"])
    write_lora(tmp_path / "lokr.safetensors", ["diffusion_model.blocks.0.attn.qkv_proj.lokr_w1",
                                               "diffusion_model.blocks.0.mlp.fc1.lokr_w1"])
    write_lora(tmp_path / "wan.safetensors", ["diffusion_model.blocks.0.self_attn.q.lora_A.weight",
                                              "diffusion_model.blocks.0.ffn.0.lora_A.weight"])
    write_lora(tmp_path / "flux.safetensors", ["transformer.double_blocks.0.img_attn.qkv.lora_A.weight"])
    (tmp_path / "broken.safetensors").write_bytes(b"\xff" * 20)
    got = {p.name: C.is_h3_lora(p) for p in tmp_path.glob("*.safetensors")}
    assert got == {"peft.safetensors": True, "kohya.safetensors": True, "lokr.safetensors": True,
                   "wan.safetensors": False, "flux.safetensors": False, "broken.safetensors": False}


def test_find_h3_loras_in_folders_and_links(tmp_path):
    from test_ui import H3_KEYS, write_lora
    root = tmp_path / "loras"
    write_lora(root / "MiniMax H3" / "b.safetensors", H3_KEYS)
    write_lora(root / "a.safetensors", H3_KEYS)
    write_lora(tmp_path / "training" / "mine.safetensors", H3_KEYS)
    (root / "MiniMax H3" / "mine_step500.safetensors").symlink_to(tmp_path / "training" / "mine.safetensors")
    names = [n for _p, n in C.find_h3_loras([root])]
    assert names == ["a.safetensors", "MiniMax H3/b.safetensors", "MiniMax H3/mine_step500.safetensors"]


def test_reference_workflow():
    wf = C.build_workflow(prompt="She exhales.", image="video-chain/s.png", seed=1, width=512, height=512,
                          frames=39, steps=20, loras=[], audio=True, prefix="p",
                          refs=["video-chain/r1.png", "video-chain/r2.png"], ref_size="max")
    assert wf["unet"]["inputs"]["unet_name"] == C.UNET_REF
    ref = wf["refcond"]
    assert ref["class_type"] == "MiniMaxH3ReferenceToVideo" and ref["inputs"]["ref_image_size"] == "max"
    assert ref["inputs"]["ref_images.ref_image_0"] == ["ref0", 0] and ref["inputs"]["ref_images.ref_image_1"] == ["ref1", 0]
    guide = wf["cond"]
    assert guide["class_type"] == "MiniMaxH3AddGuide" and guide["inputs"]["frame_idx"] == 0
    assert guide["inputs"]["image"] == ["start", 0] and guide["inputs"]["latent"] == ["refcond", 1]
    assert wf["guider"]["inputs"]["conditioning"] == ["cond", 0] and wf["sample"]["inputs"]["latent_image"] == ["refcond", 1]
    assert C.with_ref_hint("She exhales.") == "She exhales. " + C.REF_HINT
    assert C.with_ref_hint("The woman from <Picture 1> exhales.") == "The woman from <Picture 1> exhales."


def test_generate_with_references(server, start, tmp_path):
    (server.root / "main.py").write_text("")
    ref = tmp_path / "face.png"
    Image.new("RGB", (300, 400), (200, 150, 120)).save(ref)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    s = settings(ref_paths=[str(ref)], ref_size="match", ref_hint=True)
    gen.generate("a soft glow appears", start, tmp_path / "v.mp4", tmp_path / "v.png", 3, s)
    wf = server.prompts[-1]
    assert wf["refcond"]["inputs"]["prompt"].endswith(C.REF_HINT)
    assert chain.probe(tmp_path / "v.mp4")["frames"] == 39
    assert not list((server.root / "input" / "video-chain").glob("*"))           # start + reference cleaned up


def test_stacked_loras_are_chained(server, start, tmp_path):
    server.loras = ["MiniMax H3/glow.safetensors", "MiniMax H3/motion.safetensors"]
    gen = C.ComfyGenerator(server.url)
    s = settings(lora="/m/MiniMax H3/glow.safetensors", strength=0.8,
                 extra_loras=[{"lora": "/m/MiniMax H3/motion.safetensors", "sha1": "x", "strength": 0.4}])
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, s)
    wf = server.prompts[-1]
    assert wf["lora0"]["inputs"]["lora_name"] == "MiniMax H3/glow.safetensors" and wf["lora0"]["inputs"]["strength_model"] == 0.8
    assert wf["lora1"]["inputs"]["lora_name"] == "MiniMax H3/motion.safetensors" and wf["lora1"]["inputs"]["model"] == ["lora0", 0]
    assert wf["lora1"]["inputs"]["strength_model"] == 0.4 and wf["guider"]["inputs"]["model"] == ["lora1", 0]


def test_turbo_workflow_and_generation(server, start, tmp_path):
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=6,
                          loras=[("MiniMax H3/glow.safetensors", 0.8)], audio=True, prefix="p",
                          turbo=("MiniMax H3/minimax_h3_turbo_v4_step600_ema.safetensors", 1.0))
    assert wf["turbo"]["class_type"] == "MiniMaxH3TurboLoRA" and wf["turbo"]["inputs"]["model"] == ["unet", 0]
    assert wf["turbo"]["inputs"]["low_vram"] is False and wf["lora0"]["inputs"]["model"] == ["turbo", 0]
    assert wf["sampler"]["class_type"] == "MiniMaxH3TurboSampler" and wf["sigmas"]["inputs"]["steps"] == 6
    assert wf["sigmas"]["inputs"]["scheduler"] == "simple"
    server.loras = ["MiniMax H3/minimax_h3_turbo_v4_step600_ema.safetensors"]
    gen = C.ComfyGenerator(server.url)
    s = settings(steps=6, turbo={"lora": "/m/MiniMax H3/minimax_h3_turbo_v4_step600_ema.safetensors", "sha1": "x",
                                 "strength": 1.0})
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, s)
    assert server.prompts[-1]["turbo"]["inputs"]["lora_name"] == "MiniMax H3/minimax_h3_turbo_v4_step600_ema.safetensors"
    server.turbo_nodes = False
    with pytest.raises(C.ComfyError, match="ComfyUI-MiniMax-H3-Turbo"):
        gen.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 1, s)


def test_turbo_lora_detection(tmp_path):
    from test_ui import H3_KEYS, write_lora
    write_lora(tmp_path / "minimax_h3_turbo_v9.safetensors", H3_KEYS)
    write_lora(tmp_path / "fast.safetensors", H3_KEYS + ["final_layer.linear.lora_A.weight"])
    write_lora(tmp_path / "normal.safetensors", H3_KEYS)
    assert C.is_turbo_lora(tmp_path / "minimax_h3_turbo_v9.safetensors") and C.is_turbo_lora(tmp_path / "fast.safetensors")
    assert not C.is_turbo_lora(tmp_path / "normal.safetensors")


def test_same_name_elsewhere_is_not_comfyuis_lora(tmp_path):
    root = tmp_path / "loras"
    (root).mkdir()
    (root / "style.safetensors").write_bytes(b"comfy's")
    other = tmp_path / "mine" / "style.safetensors"
    other.parent.mkdir()
    other.write_bytes(b"another one")
    assert C.lora_name_for(str(other), ["style.safetensors"], [root]) is None
    assert C.lora_name_for(str(root / "style.safetensors"), ["style.safetensors"], [root]) == "style.safetensors"
    assert C.lora_name_for(str(other), ["style.safetensors"]) == "style.safetensors"   # no folders known: by name


def test_end_frame_and_guide_workflows():
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=20, loras=[],
                          audio=True, prefix="p", end_image="e.png")
    cond = wf["cond"]["inputs"]
    assert cond["first_frame"] == ["start", 0] and cond["last_frame"] == ["end", 0]
    assert wf["end"]["inputs"]["image"] == "e.png"
    wf = C.build_workflow(prompt="x", image=None, seed=1, width=512, height=512, frames=73, steps=20, loras=[],
                          audio=True, prefix="p", end_image="e.png")                 # no start frame at all
    assert "first_frame" not in wf["cond"]["inputs"] and "start" not in wf
    guides = [{"name": "g0.mp4", "frame_idx": 0, "audio": True, "at": "start"},
              {"name": "g1.mp4", "frame_idx": 68, "audio": False, "at": "end"}]
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=20, loras=[],
                          audio=True, prefix="p", end_image="e.png", guides=guides)
    assert "start" not in wf and "end" not in wf                # the guides replace the single frames
    assert set(wf["cond"]["inputs"]) >= {"width", "length"} and "first_frame" not in wf["cond"]["inputs"]
    g0, g1 = wf["guide0"]["inputs"], wf["guide1"]["inputs"]
    assert wf["gvideo0"]["inputs"]["file"] == "g0.mp4" and g0["image"] == ["gparts0", 0] and g0["frame_idx"] == 0
    assert g0["audio"] == ["gparts0", 1] and g0["audio_vae"] == ["audio_vae", 0] and g0["positive"] == ["cond", 0]
    assert "audio" not in g1 and g1["positive"] == ["guide0", 0] and g1["frame_idx"] == 68
    assert g1["latent"] == ["cond", 1] and wf["guider"]["inputs"]["conditioning"] == ["guide1", 0]
    # identity references: the end frame is a guide at the last frame
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=20, loras=[],
                          audio=False, prefix="p", refs=["r.png"], end_image="e.png")
    assert wf["end_guide"]["inputs"]["frame_idx"] == -1 and wf["end_guide"]["inputs"]["positive"] == ["cond", 0]
    assert wf["guider"]["inputs"]["conditioning"] == ["end_guide", 0]


def test_generate_with_end_frame_and_guides(server, start, tmp_path):
    (server.root / "main.py").write_text("")
    end = tmp_path / "end.png"
    Image.new("RGB", (192, 108), (0, 0, 250)).save(end)
    clip = tmp_path / "g.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=672x384:r=24", "-f", "lavfi",
                    "-i", "sine=d=1", "-frames:v", "5", "-t", "0.2083", "-c:a", "aac", str(clip)], check=True)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, settings(end_image=str(end)))
    wf = server.prompts[-1]
    assert wf["cond"]["inputs"]["last_frame"] == ["end", 0]
    s = settings(end_image=str(end), guides=[{"video": str(clip), "frame_idx": 0, "frames": 5, "audio": True,
                                              "at": "start"}])
    gen.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 1, s)
    wf = server.prompts[-1]
    assert wf["gvideo0"]["inputs"]["file"].endswith(".mp4") and "start" not in wf and "end" in wf
    assert not list((server.root / "input" / "video-chain").glob("*"))      # uploads cleaned up
    gen.generate("x", None, tmp_path / "u.mp4", tmp_path / "u.png", 1, settings(end_image=str(end)))
    assert "start" not in server.prompts[-1]
