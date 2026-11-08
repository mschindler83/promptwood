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
    wf = C.build_workflow(prompt="She exhales.", image="video-chain/s.png", seed=1, width=512, height=512,
                          frames=39, steps=20, loras=[], audio=True, prefix="p", refs=["video-chain/r1.png"],
                          start_pin=5)
    assert wf["start_still"]["inputs"] == {"image": ["start", 0], "amount": 5}
    assert wf["cond"]["inputs"]["image"] == ["start_still", 0] and wf["cond"]["inputs"]["frame_idx"] == 0
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


def test_generate_with_sound_references(server, start, tmp_path):
    """Sounds alone also take the ref2va model (with the audio VAE), the start image as guide; Pika picks ref2va."""
    import subprocess
    (server.root / "main.py").write_text("")
    wav = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=3", str(wav)],
                   check=True)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    s = settings(audio_paths=[str(wav)], audio=False, steps=6, pika={"strength": 1.0})
    gen.generate("she says hello in the voice of <Audio 1>", start, tmp_path / "v.mp4", tmp_path / "v.png", 3, s)
    wf = server.prompts[-1]
    cond = wf["refcond"]["inputs"]
    assert wf["unet"]["inputs"]["unet_name"] == C.UNET_REF and cond["ref_audios.ref_audio_0"] == ["aref0", 0]
    assert wf["aref0"]["class_type"] == "LoadAudio" and wf["audio_vae"]["inputs"]["vae_name"] == C.AUDIO_VAE
    assert not [k for k in cond if k.startswith("ref_images.")] and cond["prompt"].endswith("<Audio 1>")  # no hint
    assert wf["pika"]["inputs"]["lora_name"].endswith("pika_h3_flex_ref2va_v1_bf16.safetensors")
    assert not list((server.root / "input" / "video-chain").glob("*"))           # the sound is cleaned up too
    server.audio_refs = False                                                     # an older ComfyUI
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    with pytest.raises(C.ComfyError, match="no sound references"):
        gen.generate("x <Audio 1>", start, tmp_path / "w.mp4", tmp_path / "w.png", 3, s)


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


def test_a_clip_that_is_given_up_is_cancelled_in_comfyui(server, start, tmp_path, monkeypatch):
    """A timeout (or a lost connection) leaves the job running in ComfyUI: it is cancelled, so the next clip does
    not wait behind it. A job ComfyUI itself ended with an error is not 'cancelled' again."""
    gen = C.ComfyGenerator(server.url)

    def give_up(pid, ws):
        raise C.ComfyError("ComfyUI did not finish within 60 min")
    monkeypatch.setattr(gen, "_wait", give_up)
    with pytest.raises(C.ComfyError, match="did not finish"):
        gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, settings())
    bodies = [c for c in server.calls if c[0] == "BODY"]
    assert any(c[1] == "/interrupt" for c in bodies) and gen.current is None
    failed = FakeComfy(tmp_path / "c4", fail_with="CUDA out of memory")
    try:
        g2 = C.ComfyGenerator(failed.url)
        with pytest.raises(C.ComfyJobFailed):
            g2.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 1, settings())
        assert not any(c[0] == "BODY" and c[1] == "/interrupt" for c in failed.calls)
    finally:
        failed.close()


def test_qwen_image_edit(server, start, tmp_path):
    """Qwen Image Edit as ComfyUI's blueprint: Lightning LoRA for Fast, refs as Picture 2/3, uploads removed."""
    ed = C.ComfyImageEditor(server.url, comfy_dir=None)
    assert ed.files["unet"].endswith("qwen_image_edit_2511_fp8mixed.safetensors")
    assert ed.qualities() == ["fast", "standard", "best"]
    face = tmp_path / "face.png"
    Image.new("RGB", (64, 64), (200, 150, 120)).save(face)
    out = tmp_path / "e.png"
    meta = ed.edit(start, "she smiles", out, 7, "fast", [face])
    assert Image.open(out).size == (1024, 576) and meta["steps"] == 4 and meta["cfg"] == 1.0
    wf = server.prompts[-1]
    assert wf["lightning"]["inputs"]["lora_name"].endswith("Lightning-4steps-V1.0-bf16.safetensors")
    assert wf["pos"]["inputs"]["prompt"] == "she smiles" and wf["neg"]["inputs"]["prompt"] == ""
    assert wf["pos"]["inputs"]["image2"] == ["ref0", 0] and wf["sample"]["inputs"]["seed"] == 7
    ed.edit(start, "night", out, 8, "best")
    wf = server.prompts[-1]
    assert "lightning" not in wf and wf["sample"]["inputs"]["steps"] == 40 and wf["sample"]["inputs"]["cfg"] == 4.0
    assert C.pick_qwen_files(["x.safetensors"], [], [], [])["unet"] is None
    assert meta["edit_model"] == "2511" and ed.models() == ["2511", "qwen21", "moss"]
    # Qwen-Image 2.1 (ComfyUI's image_qwen_image_2_1_image_edit): the source as image_1, its own latent size
    meta = ed.edit(start, "she smiles", out, 9, "fast", [face], model="qwen21")
    wf = server.edited
    assert meta["edit_model"] == "qwen21" and meta["steps"] == 8 and ed.qualities("qwen21") == ["fast", "best"]
    assert wf["unet"]["inputs"]["unet_name"].endswith("qwen_image_2.1_turbo_int8_convrot.safetensors")
    te = wf["text"]["inputs"]
    assert wf["text"]["class_type"] == "TextEncodeQwenImage21" and te["images.image_1"] == ["pic0", 0]
    assert te["images.image_2"] == ["pic1", 0] and te["vae"] == ["vae", 0] and te["resolution"] == 1024
    assert wf["sample"]["inputs"]["latent_image"] == ["text", 2] and "latent" not in wf
    assert wf["sigmas"]["inputs"]["sigmas"] == C.QWEN21_SIGMAS and wf["guider"]["inputs"]["cfg"] == 1.0
    assert wf["cache"]["inputs"]["model"] == ["unet", 0] and wf["clip"]["inputs"]["type"] == "qwen_image"
    meta = ed.edit(start, "night", out, 9, "best", model="qwen21")
    wf = server.edited
    assert meta["steps"] == 25 and wf["sample"]["class_type"] == "KSampler" and wf["sample"]["inputs"]["cfg"] == 1.0
    assert wf["unet"]["inputs"]["unet_name"].endswith("qwen_image_2.1_int8_convrot.safetensors")
    # MOSS-Image (ComfyUI's image_moss_image_edit): SFT + edit LoRA, skip-layer guidance, the source at ~1 MP
    meta = ed.edit(start, "she smiles", out, 4, "standard", [face], model="moss")
    wf = server.edited
    assert meta["edit_model"] == "moss" and meta["steps"] == 30 and ed.qualities("moss") == ["standard"]
    assert wf["unet"]["inputs"]["unet_name"].endswith("moss_image_sft_int8_convrot.safetensors")
    assert wf["edit_lora"]["inputs"]["lora_name"].endswith("moss_image_edit_lora_rank_64_bf16.safetensors")
    assert wf["slg"]["inputs"]["single_layers"] == "22" and wf["slg"]["inputs"]["end_percent"] == 1.0
    p = wf["pos"]["inputs"]
    assert wf["pos"]["class_type"] == "TextEncodeMOSSImageEdit" and p["edit_format"] == "Moss-Image-Edit"
    assert p["images.image_0"] == ["scale0", 0] and p["images.image_1"] == ["scale1", 0]
    assert wf["neg"]["inputs"]["prompt"] == C.MOSS_NEGATIVE and wf["clip"]["inputs"]["type"] == "moss_image"
    assert wf["latent"]["inputs"]["width"] == ["size", 0] and wf["scale0"]["inputs"]["megapixels"] == 1.0
    s = wf["sample"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"], s["scheduler"]) == (30, 4.0, "euler", "normal")
    server.qwen = False                                    # only 2.1 and MOSS left: editing still works with them
    ed = C.ComfyImageEditor(server.url)
    assert ed.models() == ["qwen21", "moss"]
    with pytest.raises(C.ComfyError, match="Qwen Image Edit model"):
        ed.edit(start, "x", out, 1, "fast")
    server.qwen21 = server.moss = False
    with pytest.raises(C.ComfyError, match="Qwen Image Edit model"):
        C.ComfyImageEditor(server.url)


def test_hyperflow_workflow(server, start, tmp_path):
    """HyperFlow: ApplyHyperFlowH3 (pruned weights, curve refit) right after the model, the clip's LoRAs after it,
    Euler on the sigma grid the node returns instead of a scheduler."""
    gen = C.ComfyGenerator(server.url)
    lora = "/data/my_models/loras/MiniMax H3/glow.safetensors"
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 3,
                 settings(lora=lora, strength=0.8, steps=8, hyperflow={"strength": 1.0}))
    wf = server.prompts[-1]
    hf = wf["hyperflow"]["inputs"]
    assert wf["hyperflow"]["class_type"] == "ApplyHyperFlowH3" and hf["model"] == ["unet", 0]
    assert hf["hyperflow_file"].endswith("_pruned.safetensors") and hf["experimental_curve_refit"] is True
    assert hf["download_if_missing"] is False and wf["lora0"]["inputs"]["model"] == ["hyperflow", 0]
    assert wf["sample"]["inputs"]["sigmas"] == ["hyperflow", 1] and "sigmas" not in wf
    assert wf["sampler"]["inputs"]["sampler_name"] == "euler"
    server.hyperflow = False
    with pytest.raises(C.ComfyError, match="no ApplyHyperFlowH3 node"):
        gen.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 3, settings(hyperflow={"strength": 1.0}))


def test_refine_a_clip_at_2k(server, start, tmp_path):
    """Open-H3-Regenerate-2K: the finished clip (video + sound) encoded to H3's AV latent, upscaled with the latent
    upscaler, 3 refine steps with the clip's LoRAs + the refiner LoRA, the clip's own sound kept."""
    gen = C.ComfyGenerator(server.url)
    draft, frame = tmp_path / "d.mp4", tmp_path / "d.png"
    gen.generate("x", start, draft, frame, 1, settings())
    lora = "/data/my_models/loras/MiniMax H3/glow.safetensors"
    size = C.refine_size(672, 384, "2k")
    assert size == (2528, 1440) and C.refine_size(672, 384, "1080p") == (1888, 1088)
    out = tmp_path / "r.mp4"
    gen.refine(draft, "smoke", out, 5, size, 39, [(lora, 0.7)])
    info = chain.probe(out)
    assert (info["width"], info["height"], info["frames"], info["audio"]) == (2528, 1440, 39, True)
    wf = server.prompts[-1]
    assert wf["lora0"]["inputs"]["lora_name"] == "MiniMax H3/glow.safetensors"
    assert wf["lora1"]["inputs"]["lora_name"].endswith("regenerate_2k_refiner_rank_384_bf16.safetensors")
    assert wf["guider"]["inputs"]["model"] == ["lora1", 0] and wf["sample"]["inputs"]["latent_image"] == ["up", 0]
    assert wf["video"]["inputs"]["audio"] == ["parts", 1] and wf["cond"]["inputs"]["length"] == 39
    assert C.pick_refine_files(["a/x_regenerate_2k_refiner_rank_16_bf16.safetensors",
                                "a/x_regenerate_2k_refiner_rank_384_bf16.safetensors"], ["u_fp16", "minimax_h3_u_bf16"]) \
        == {"refiner": "a/x_regenerate_2k_refiner_rank_384_bf16.safetensors", "upscaler": "minimax_h3_u_bf16"}
    server.refine = False
    with pytest.raises(C.ComfyError, match="update ComfyUI"):
        gen.refine(draft, "smoke", out, 5, size, 39, [])


def test_models_unloaded_only_from_hyperflow_to_turbo(server, start, tmp_path):
    """HyperFlow's patch on ComfyUI's cached model breaks the turbo LoRA after it (measured on the real GPU; every
    other order gives identical clips): unload only for that switch; a patched model left from before this
    process is unloaded once when turbo fails with that error."""
    gen = C.ComfyGenerator(server.url)
    turbo = {"lora": "/x/MiniMax H3/glow.safetensors", "strength": 1.0}

    def frees():
        return sum(1 for c in server.calls if c[0] == "BODY" and c[1] == "/free")

    def make(**kw):
        gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, settings(**kw))
    make(turbo=turbo)
    make(hyperflow={"strength": 1.0})
    make()
    assert frees() == 0                                            # turbo -> HyperFlow -> normal: nothing unloaded
    make(hyperflow={"strength": 1.0})
    make(turbo=turbo)
    assert frees() == 1                                            # HyperFlow -> turbo: unloaded first
    make(turbo=turbo)
    assert frees() == 1
    fresh = C.ComfyGenerator(server.url)                           # another process: does not know what ran before
    server.fail_next = "mat1 and mat2 shapes cannot be multiplied (2x8 and 2688x16)"
    fresh.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 1, settings(turbo=turbo))
    assert frees() == 2 and (tmp_path / "w.mp4").is_file()


def test_refine_with_sol_attn(server, start, tmp_path):
    gen = C.ComfyGenerator(server.url)
    draft = tmp_path / "d.mp4"
    gen.generate("x", start, draft, tmp_path / "d.png", 1, settings())
    gen.refine(draft, "x", tmp_path / "r.mp4", 5, (1888, 1088), 39, [], sparse=C.SOL_TAU)
    wf = server.prompts[-1]
    sp = wf["sparse"]["inputs"]
    assert wf["sparse"]["class_type"] == "BlockSparseAttention" and sp["selection"] == "sol-attn"
    assert sp["selection.tau"] == 1.3 and wf["guider"]["inputs"]["model"] == ["sparse", 0]
    assert sp["model"] == ["lora0", 0]                                          # after the refiner LoRA


def _write_lora(path, keys):
    import json as _json
    import struct
    path.parent.mkdir(parents=True, exist_ok=True)
    head = _json.dumps({k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for k in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0\0\0")


def test_qwen_loras_found_and_chained_before_lightning(server, start, tmp_path):
    qwen = ["diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight",
            "diffusion_model.transformer_blocks.0.txt_mlp.net.0.proj.lora_A.weight"]
    flux = qwen + ["single_transformer_blocks.0.proj_out.lora_A.weight"]
    root = tmp_path / "loras"
    _write_lora(root / "qwen-image-edit" / "style.safetensors", qwen)
    _write_lora(root / "qwen-image-edit" / "Qwen-Image-Edit-2511-Lightning-4steps.safetensors", qwen)
    _write_lora(root / "flux" / "flux.safetensors", flux)
    _write_lora(root / "h3" / "nose.safetensors", ["blocks.0.attn.qkv_proj.lora_A.weight",
                                                   "blocks.0.mlp.fc1.lora_A.weight"])
    assert [n for _p, n in C.find_qwen_loras([root])] == ["qwen-image-edit/style.safetensors"]
    assert [n for _p, n in C.find_h3_loras([root])] == ["h3/glow.safetensors"]
    ed = C.ComfyImageEditor(server.url)
    meta = ed.edit(start, "x", tmp_path / "e.png", 1, "fast", [], [("/any/MiniMax H3/glow.safetensors", 0.6)])
    wf = server.prompts[-1]
    assert wf["lora0"]["inputs"]["lora_name"] == "MiniMax H3/glow.safetensors" and wf["lora0"]["inputs"]["model"] == ["unet", 0]
    assert wf["lightning"]["inputs"]["model"] == ["lora0", 0] and wf["lora0"]["inputs"]["strength_model"] == 0.6
    assert meta["loras"] == [{"path": "/any/MiniMax H3/glow.safetensors", "strength": 0.6}]
    with pytest.raises(C.ComfyError, match="not in ComfyUI's LoRA folders"):
        ed.edit(start, "x", tmp_path / "e.png", 1, "fast", [], [("/any/unknown.safetensors", 1.0)])


def test_prompt_refiner(server):
    r = C.ComfyPromptRefiner(server.url)
    assert [m["label"] for m in r.models][0].startswith("Gemma 3 12B abliterated")     # uncensored first
    assert r.models[0]["type"] == "ltxv" and not any("qwen_2.5" in m["file"] for m in r.models)   # cannot generate
    out = r.refine_prompt("a man walks", "video", None, 3, hint="more cinematic")
    assert out.startswith("REFINED a man walks")                                          # thinking, preface, quotes off
    g = server.refined
    assert "image-to-video" in g["system_prompt"] and "minors" in g["system_prompt"]
    assert "more cinematic" in g["prompt"] and g["sampling_mode.seed"] == 3
    r.refine_prompt("the man from <Picture 1> waves", "image", r.models[0]["file"])
    assert server.prompts[-1]["llm"]["inputs"]["clip_name"] == r.models[0]["file"]
    assert "<Picture 1>" in server.refined["system_prompt"] and "<Picture" not in C.REFINE_INSTRUCTIONS["image"]
    r.refine_prompt("@Anna waves at @Ben-2, mail me@home", "video", None)
    assert "exactly as they are, where they are: @Anna, @Ben-2." in server.refined["system_prompt"]
    assert "The cast" not in server.refined["system_prompt"]
    r.refine_prompt("@Anna waves", "video", None, cast=[{"name": "Anna", "desc": "woman, 30s.", "voice": True},
                                                        {"name": "Ben", "desc": "", "voice": False}])
    system = server.refined["system_prompt"]
    assert "The cast, written as @Name: @Anna (woman, 30s), has a voice; @Ben." in system
    assert "in its own sentence that starts with the speaker's @Name, never with he, she or they" in system
    r.refine_prompt("a red car", "image", None, cast=[{"name": "Anna", "desc": "", "voice": False}])
    assert "The cast" not in server.refined["system_prompt"]                  # images: no cast rules
    assert C.clean_refined("**“A calm sea.”**") == "A calm sea."


def test_long_clips_are_refined_in_context_windows():
    """Measured on an RTX 4090: a 10 s clip at 2K (1M tokens) runs out of memory in one pass; refined in H3
    context windows of 124 frames it works (no visible seams). Short clips stay in one pass."""
    assert C.latent_frames(73) == 22 and C.latent_frames(124) == 37 and C.latent_frames(243) == 72
    assert C.context_window(73, 2528, 1440) is None and C.context_window(124, 2528, 1440) is None
    assert C.context_window(243, 2528, 1440) == 124 and C.context_window(243, 1888, 1088) == 209
    common = dict(prompt="x", draft="d.mp4", seed=1, loras=[], refiner="r.safetensors", upscaler="u.safetensors",
                  prefix="p")
    wf = C.build_refine_workflow(width=2528, height=1440, frames=243, **common)
    win = wf["windows"]["inputs"]
    assert wf["windows"]["class_type"] == "MiniMaxH3ContextWindows" and win["context_length"] == 124
    assert win["context_overlap"] == 39 and wf["guider"]["inputs"]["model"] == ["windows", 0]
    assert "windows" not in C.build_refine_workflow(width=2528, height=1440, frames=73, **common)


def test_image_families_from_comfyui_lists():
    from video_chain import imagegen as G
    files = G.pick_files(
        ["krea2Turbo_v10.safetensors", "krea2/krea2_turbo_fp8_scaled.safetensors", "sickOllie_krea2.safetensors",
         "qwen-image/qwen_image_fp8_e4m3fn.safetensors", "qwen-image/qwen_image_2512_fp8_e4m3fn.safetensors",
         "qwen-image/qwen_image_edit_2511_fp8mixed.safetensors", "FLUX1/flux1-dev-fp8.safetensors",
         "FLUX1/flux1-schnell.safetensors"],
        ["illustrij_v20.safetensors", "Qwen-Rapid-AIO-NSFW-v23.safetensors"],
        ["qwen3vl_4b_fp8_scaled.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "clip_l.safetensors",
         "t5xxl_fp8_e4m3fn_scaled.safetensors"],
        ["qwen_image_vae.safetensors", "FLUX1/ae.safetensors"],
        ["qwen-image/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", "krea2/krea2_style_reference.safetensors"])
    assert files["krea2"]["models"][0] == "krea2/krea2_turbo_fp8_scaled.safetensors"        # official build first
    assert set(files["krea2"]["models"]) >= {"krea2Turbo_v10.safetensors", "sickOllie_krea2.safetensors"}
    assert files["qwen"]["models"][0].endswith("qwen_image_2512_fp8_e4m3fn.safetensors")
    assert all("edit" not in m for m in files["qwen"]["models"]) and files["qwen"]["edit"].endswith("2511_fp8mixed.safetensors")
    assert files["qwen"]["lightning4"] and files["qwen"]["lightning8"] is None
    assert "licence" in files["flux3"]["missing"] and files["flux1"]["models"] == ["FLUX1/flux1-dev-fp8.safetensors"]
    assert files["sdxl"]["models"] == ["illustrij_v20.safetensors"]
    assert G.size_for("qwen", 16 / 9) == (1776, 992) and G.size_for("sdxl", 1.0) == (1024, 1024)
    assert G.lora_family(["lora_unet_double_blocks_0_img_attn_proj.alpha"]) == "flux1"
    assert G.lora_family(["lora_te2_text_model_encoder_layers_0_mlp_fc1.alpha"]) == "sdxl"
    assert G.lora_family(["diffusion_model.blocks.0.attn.wq.lora_A.weight",
                          "diffusion_model.blocks.0.mlp.gate.lora_A.weight"]) == "krea2"
    assert G.lora_family(["diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight"]) is None   # Z-Image
    assert G.lora_family(["diffusion_model.blocks.0.self_attn.q.lora_A.weight"]) is None          # e.g. Wan
    assert "Qwen-Image 2.1" in files["qwen21"]["missing"]
    q21 = C.pick_qwen21_files(["qwen-image/qwen_image_2.1_turbo_int8_convrot.safetensors",
                               "qwen-image/qwen_image_2.1_bf16.safetensors",
                               "qwen-image/qwen_image_2.1_int8_convrot.safetensors"],
                              ["qwen3vl_8b_bf16.safetensors", "qwen3vl_8b_int8_convrot.safetensors"],
                              ["qwen_image_2.1_vae_bf16.safetensors"])
    assert q21["base"].endswith("2.1_int8_convrot.safetensors") and q21["turbo"].endswith("turbo_int8_convrot.safetensors")
    assert q21["clip"] == "qwen3vl_8b_int8_convrot.safetensors" and q21["vae"] == "qwen_image_2.1_vae_bf16.safetensors"
    assert C.pick_qwen21_files(["qwen-image/qwen_image_2.1_bf16.safetensors"], [], []) is None   # no encoder
    assert G.size_for("qwen21", 16 / 9) == (1376, 768)
    mo = C.pick_moss_files(["moss-image/moss_image_base_bf16.safetensors", "moss-image/moss_image_sft_int8_convrot.safetensors"],
                           ["qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors"],
                           ["flux2-vae.safetensors", "flux2-vae_small.safetensors"], [])
    assert mo["sft"].endswith("sft_int8_convrot.safetensors") and mo["base"].endswith("base_bf16.safetensors")
    assert mo["clip"] == "qwen3.5_9b_w4a8_gptq.safetensors" and mo["vae"] == "flux2-vae_small.safetensors"
    assert mo["edit_lora"] is None and "MOSS-Image" in files["moss"]["missing"]


def test_create_images_in_every_family(server, start, tmp_path):
    from video_chain import imagegen as G
    cr = G.ComfyImageCreator(server.url)
    assert "missing" in cr.files["flux3"] and "models" in cr.files["krea2"]
    out = tmp_path / "n.png"
    meta = cr.create("krea2", None, "a red fox", out, 5, "standard", 1376, 768)
    wf = server.created
    assert wf["sample"]["inputs"]["steps"] == 8 and wf["sample"]["inputs"]["cfg"] == 1.0
    assert wf["clip"]["inputs"]["type"] == "krea2" and wf["neg"]["class_type"] == "ConditioningZeroOut"
    assert Image.open(out).size == (1376, 768) and meta["family"] == "krea2"
    cr.create("krea2", None, "in this style", out, 5, "standard", 1024, 1024, refs=[start])     # style reference
    wf = server.created
    assert wf["style0"]["inputs"]["lora_name"].endswith("krea2_style_reference.safetensors")
    assert wf["pos0"]["class_type"] == "TextEncodeQwenImageEditPlus" and wf["guider"]["inputs"]["cfg"] == 1.0
    cr.create("qwen", None, "a lighthouse", out, 5, "fast", 1328, 1328, loras=[("/x/MiniMax H3/glow.safetensors", 0.5)])
    wf = server.created
    assert wf["lora0"]["inputs"]["model"] == ["unet", 0] and wf["lightning0"]["inputs"]["model"] == ["lora0", 0]
    assert wf["sample"]["inputs"]["steps"] == 4 and wf["shift"]["inputs"]["shift"] == 3.1
    assert wf["neg"]["inputs"]["text"] == G.QWEN_NEGATIVE
    cr.create("qwen", None, "a lighthouse", out, 5, "best", 1328, 1328)
    assert server.created["sample"]["inputs"]["steps"] == 50 and server.created["sample"]["inputs"]["cfg"] == 4.0
    cr.create("qwen", None, "the woman from Picture 1 on a beach", out, 5, "fast", 1328, 1328, refs=[start])
    assert server.created["unet"]["inputs"]["unet_name"].endswith("qwen_image_edit_2511_fp8mixed.safetensors")
    cr.create("sdxl", None, "1girl, smiling", out, 5, "best", 1024, 1024,
              loras=[("/x/MiniMax H3/glow.safetensors", 0.7)])
    wf = server.created
    assert wf["lora0"]["class_type"] == "LoraLoader" and wf["pos"]["inputs"]["clip"] == ["lora0", 1]
    assert wf["pos"]["inputs"]["text"].startswith("masterpiece") and wf["sample"]["inputs"]["sampler_name"] == "dpmpp_2m"
    cr.create("sdxl", None, "1girl", out, 5, "fast", 832, 1216, refs=[start], ref_strength=0.6)
    assert server.created["sample"]["inputs"]["denoise"] == 0.6 and Image.open(out).size == (832, 1216)
    cr.create("flux1", None, "a cat", out, 5, "standard", 1024, 1024)
    assert server.created["pos"]["class_type"] == "FluxGuidance" and server.created["clip"]["inputs"]["type"] == "flux"
    with pytest.raises(C.ComfyError, match="not available"):
        cr.create("flux3", None, "x", out, 5, "fast", 1024, 1024)
    # Qwen-Image 2.1 (ComfyUI's image_qwen_image_2_1_t2i / turbo_t2i): Turbo 8 fixed steps, base 25, cfg 1
    assert cr.files["qwen21"]["models"] == ["qwen-image/qwen_image_2.1_int8_convrot.safetensors"]   # Turbo: Fast
    meta = cr.create("qwen21", None, "a lighthouse at dusk", out, 5, "fast", 1376, 768)
    wf = server.created
    assert wf["unet"]["inputs"]["unet_name"].endswith("turbo_int8_convrot.safetensors") and meta["steps"] == 8
    assert wf["latent"]["inputs"]["width"] == 1376 and Image.open(out).size == (1376, 768)
    assert "vae" not in wf["text"]["inputs"] and wf["text"]["inputs"]["prompt"] == "a lighthouse at dusk"
    meta = cr.create("qwen21", None, "the woman from the first picture on a beach", out, 5, "best", 1024, 1024,
                     refs=[start])
    wf = server.created
    assert meta["steps"] == 25 and wf["text"]["inputs"]["images.image_1"] == ["pic0", 0]
    assert wf["sample"]["inputs"]["latent_image"] == ["latent", 0]          # the chosen size, not the picture's
    # MOSS-Image (image_moss_image_t2i): SFT, 25 steps, cfg 2.5, er_sde; references with its edit LoRA
    meta = cr.create("moss", None, "a lighthouse", out, 5, "standard", 1376, 768)
    wf = server.created
    s = wf["sample"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"]) == (25, 2.5, "er_sde") and "slg" not in wf
    assert wf["pos"]["class_type"] == "CLIPTextEncode" and wf["neg"]["inputs"]["text"] == C.MOSS_NEGATIVE
    assert wf["latent"]["class_type"] == "EmptyFlux2LatentImage" and Image.open(out).size == (1376, 768)
    cr.create("moss", None, "her on a beach", out, 5, "standard", 1024, 1024, refs=[start])
    wf = server.created
    assert "edit_lora" in wf and wf["latent"]["inputs"]["width"] == 1024
    # Chroma1-HD (image_chroma_text_to_image): 26 steps, beta, cfg 3.5, shift 1; Fast: the Flash build, heun, cfg 1
    assert cr.files["chroma"]["models"] == ["Chroma1-HD.safetensors"]
    meta = cr.create("chroma", None, "a lighthouse", out, 5, "best", 1024, 1024)
    wf = server.created
    s = wf["sample"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"], s["scheduler"]) == (26, 3.5, "euler", "beta")
    assert wf["unet"]["inputs"]["unet_name"] == "Chroma1-HD.safetensors" and wf["shift"]["inputs"]["shift"] == 1.0
    assert wf["clip0"]["inputs"]["type"] == "chroma" and wf["clip"]["class_type"] == "T5TokenizerOptions"
    assert wf["neg"]["inputs"]["text"] == G.CHROMA_NEGATIVE and wf["vae"]["inputs"]["vae_name"].endswith("ae.safetensors")
    meta = cr.create("chroma", None, "a lighthouse", out, 5, "fast", 1024, 1024, refs=[start], ref_strength=0.7)
    wf = server.created
    s = wf["sample"]["inputs"]
    assert wf["unet"]["inputs"]["unet_name"] == "Chroma1-HD-Flash.safetensors" and meta["steps"] == 12
    assert (s["cfg"], s["sampler_name"], s["denoise"]) == (1.0, "heun", 0.7) and wf["latent"]["class_type"] == "VAEEncode"
    server.qwen21 = server.moss = False
    cr = G.ComfyImageCreator(server.url)
    assert "missing" in cr.files["qwen21"] and "missing" in cr.files["moss"]


def test_pika_h3_flex_workflow(server, start, tmp_path):
    """Pika H3 Flex as ComfyUI's templates: its LoRA right after the model (fl2va / ref2va to match), the clip's own
    LoRAs after it, lcm on the template's 6 sigmas, no scheduler."""
    gen = C.ComfyGenerator(server.url)
    lora = "/data/my_models/loras/MiniMax H3/glow.safetensors"
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 3,
                 settings(lora=lora, strength=-0.5, steps=6, pika={"strength": 1.0}))
    wf = server.prompts[-1]
    assert wf["pika"]["inputs"]["lora_name"].endswith("pika_h3_flex_fl2va_v1_bf16.safetensors")
    assert wf["pika"]["inputs"]["model"] == ["unet", 0] and wf["lora0"]["inputs"]["model"] == ["pika", 0]
    assert wf["lora0"]["inputs"]["strength_model"] == -0.5
    assert wf["sampler"]["inputs"]["sampler_name"] == "lcm" and wf["sigmas"]["class_type"] == "ManualSigmas"
    assert wf["sigmas"]["inputs"]["sigmas"] == C.PIKA_SIGMAS and len(C.PIKA_SIGMAS.split(",")) == 7
    assert gen.pika_lora(True).endswith("pika_h3_flex_ref2va_v1_bf16.safetensors")
    assert not [n for _p, n in C.find_h3_loras([tmp_path])]                     # never offered as a normal LoRA
    with pytest.raises(ValueError):                                               # one speed mode at a time
        C.build_workflow(prompt="x", image="s.png", seed=1, width=672, height=384, frames=73, steps=6, loras=[],
                         audio=True, prefix="x", turbo=("t.safetensors", 1.0), pika=("p.safetensors", 1.0))


def test_speed_files_checked_before_the_run(server, monkeypatch):
    """A missing HyperFlow / Pika file stops the run up front, not clip by clip."""
    gen = C.ComfyGenerator(server.url)
    gen.check_speed({"pika": {"strength": 1.0}, "refs": [{"file": "r.png"}]})       # both listed by the fake
    import fake_comfy
    monkeypatch.setattr(fake_comfy, "IMAGE_LORAS", [n for n in fake_comfy.IMAGE_LORAS if "ref2va" not in n])
    gen.check_speed({"pika": {"strength": 1.0}})
    with pytest.raises(C.ComfyError, match="ref2va"):
        gen.check_speed({"pika": {"strength": 1.0}, "refs": [{"file": "r.png"}]})


def test_qwen21_and_moss_loras_found_by_contents(tmp_path):
    """Qwen-Image 2.1 LoRAs (one stream, width 4096; attention-only ones too) and MOSS-Image LoRAs (layers.N.proj,
    width 3072) are found by their contents; Qwen 2512 LoRAs, MOSS's own edit LoRA and look-alikes of another width
    are not."""
    import numpy as np
    from safetensors.numpy import save_file

    from video_chain import imagegen as G

    def lora(name, shapes):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: np.zeros(s, np.float16) for k, s in shapes.items()}, str(p))
        return p
    q21 = lora("Qwen_2.1/head.safetensors", {"diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": (4, 4096),
                                             "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": (4096, 4)})
    q2512 = lora("qwen/style.safetensors", {"diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": (4, 3072),
                                            "diffusion_model.transformer_blocks.0.attn.add_q_proj.lora_A.weight": (4, 3072),
                                            "diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight": (4, 3072),
                                            "diffusion_model.transformer_blocks.0.txt_mlp.net.0.proj.lora_A.weight": (4, 3072)})
    narrow = lora("other/attn.safetensors", {"transformer_blocks.0.attn.to_q.lora_A.weight": (4, 3072)})
    moss = lora("moss/detail.safetensors", {"lora_unet_layers_0_proj_ffn_down.lora_down.weight": (4, 8192),
                                            "lora_unet_layers_0_proj_q.lora_down.weight": (4, 3072)})
    edit = lora("moss-image/moss_image_edit_lora_rank_64_bf16.safetensors",
                {"layers.0.proj.q.lora_A": (4, 3072)})
    assert G.lora_file_family(str(q21)) == "qwen21" and G.lora_file_family(str(moss)) == "moss"
    assert G.lora_file_family(str(q2512)) == "qwen" and G.lora_file_family(str(narrow)) is None
    assert G.lora_file_family(str(edit)) == "moss"
    found = G.find_image_loras([tmp_path])
    assert [n for _p, n in found["qwen21"]] == ["Qwen_2.1/head.safetensors"]
    assert [n for _p, n in found["moss"]] == ["moss/detail.safetensors"]      # the edit LoRA: applied by edits
