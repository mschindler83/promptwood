"""ComfyGenerator against a fake ComfyUI server (synthetic clips; no ComfyUI, no GPU)."""
import shutil
import time

import pytest
from fake_comfy import FakeComfy
from PIL import Image

from video_chain import chain
from video_chain import comfy as C
from video_chain.comfy import seedvr

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
    from ui_helpers import H3_KEYS, write_lora
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
    from ui_helpers import H3_KEYS, write_lora
    root = tmp_path / "loras"
    write_lora(root / "MiniMax H3" / "b.safetensors", H3_KEYS)
    write_lora(root / "a.safetensors", H3_KEYS)
    write_lora(tmp_path / "training" / "mine.safetensors", H3_KEYS)
    (root / "MiniMax H3" / "mine_step500.safetensors").symlink_to(tmp_path / "training" / "mine.safetensors")
    for ic in ("open_h3_regenerate_2k_ic_lora", "minimax_h3_open_regen_2k_IC_comfy_fp32"):   # (upscaler parts)
        write_lora(root / "MiniMax H3" / f"{ic}.safetensors", H3_KEYS)
    names = [n for _p, n in C.find_h3_loras([root])]
    assert names == ["a.safetensors", "MiniMax H3/b.safetensors", "MiniMax H3/mine_step500.safetensors"]
    assert C.is_ic_lora("open_h3_regenerate_2k_ic_lora.safetensors") and C.is_ic_lora("x_regen_2k_IC_comfy.safetensors")
    assert not C.is_ic_lora("minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors")
    assert not C.is_ic_lora("my_icecream_style.safetensors")


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
    assert server.prompts[-1]["guider"]["class_type"] == "BasicGuider" and "neg" not in server.prompts[-1]
    s2 = dict(s, cfg=1.5)                                          # own CFG: CFGGuider against an empty prompt
    gen.generate("x", start, tmp_path / "c.mp4", tmp_path / "c.png", 1, s2)
    wf = server.prompts[-1]
    assert wf["guider"]["class_type"] == "CFGGuider" and wf["guider"]["inputs"]["cfg"] == 1.5
    assert wf["neg"] == {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": wf["guider"]["inputs"]["positive"]}}
    gen.generate("x", start, tmp_path / "d.mp4", tmp_path / "d.png", 1, dict(s, cfg=0.9))
    assert server.prompts[-1]["guider"]["inputs"]["cfg"] == 0.9
    server.turbo_nodes = False
    with pytest.raises(C.ComfyError, match="ComfyUI-MiniMax-H3-Turbo"):
        gen.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 1, s)


def test_turbo_lora_detection(tmp_path):
    from ui_helpers import H3_KEYS, write_lora
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
                 settings(lora=lora, strength=0.8, steps=8, hyperflow={"strength": 1.0}, cfg=1.2))
    wf = server.prompts[-1]
    assert wf["guider"]["class_type"] == "CFGGuider" and wf["guider"]["inputs"]["cfg"] == 1.2   # HyperFlow's CFG
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
    assert server.refine_strengths[-1] == "natural"                          # the default strength
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
    assert C.pick_refine_files(["a/open_h3_regenerate_2k_ic_lora.safetensors"], [])["refiner"] is None   # (not a refiner)
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


def test_classic_video_refine_is_unchanged():
    """The classic format keeps its instruction, its length limit and its one-line answer; clip facts given for
    it are not used."""
    assert C.REFINE_INSTRUCTIONS["video"].startswith("You improve prompts for an image-to-video model (MiniMax H3)")
    assert C.REFINE_INSTRUCTIONS["video"].endswith("At most 120 words. Answer with the prompt only, no preface, no "
                                                   "quotes around it, no markdown.")
    model = {"file": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2"}
    plain = C.build_refine_prompt_workflow(prompt="a man walks", kind="video", model=model, seed=1)
    given = C.build_refine_prompt_workflow(prompt="a man walks", kind="video", model=model, seed=1, seconds=3.0,
                                           start="seen")
    assert plain == given and plain["gen"]["inputs"]["max_length"] == 512 and "pic" not in plain
    assert "seconds" not in plain["gen"]["inputs"]["system_prompt"]
    assert C.clean_refined("a\n\nb\noverall_soundscape: x") == "a b overall_soundscape: x"     # one line


def test_h3_refine_instruction_and_clip():
    """The official format: its own instruction (style and composition first, onset -> development -> result,
    camera as type + amplitude + speed, two sound lines, no tags), the clip's seconds and its start frame."""
    text = C.REFINE_INSTRUCTIONS["video_h3"]
    for part in ("present tense", "overall style", "initial composition", "how the action begins, how it develops",
                 "amplitude and speed", "Keep every subject, action, detail and spoken line", "350-500 words",
                 "[Shot 2] At 00:03.500", "\"overall_soundscape: \"", "\"non_diegetic_music: \"", "N/A",
                 "Do not write a header line", "speaker ids like (S1)"):
        assert part in text, part
    system, user = C.refine_messages(prompt="@Anna waves", kind="video_h3", seconds=73 / 24, start="seen",
                                     cast=[{"name": "Anna", "desc": "", "voice": True}])
    assert user == "@Anna waves" and system.startswith(text) and "minors" in system
    assert "The cast, written as @Name: @Anna, has a voice." in system                 # cast rules apply too
    assert "The clip is 3.04 seconds long" in system and "every cut time is before 00:03.042" in system
    assert "is the video's first frame" in system and "Anchor the first sentences" in system
    unseen, _u = C.refine_messages(prompt="x", kind="video_h3", seconds=10, start="unseen")
    assert "The clip is 10.00 seconds long" in unseen and "Do not invent how it looks" in unseen
    assert "first frame (at 0.00" not in unseen
    nothing, _u = C.refine_messages(prompt="x", kind="video_h3")
    assert nothing == text + " " + C._SAFETY                                           # no clip facts: none told
    wf = C.build_refine_prompt_workflow(prompt="x", kind="video_h3", model={"file": "q", "type": "krea2"}, seed=2,
                                        seconds=5, start="seen", image="video-chain/start.png")
    assert wf["gen"]["inputs"]["max_length"] == 1024 and wf["gen"]["inputs"]["image"] == ["pic", 0]
    assert wf["pic"] == {"class_type": "LoadImage", "inputs": {"image": "video-chain/start.png"}}


def test_clean_refined_keeps_the_h3_lines():
    raw = ("<think>plan</think>Here is the refined prompt:\n\n```\n**integrated_multimodal_description:** [Shot 1] "
           "Live-action, cinematic, a wide shot frames @Anna on a pier.\nThe camera pushes in as @Anna (S1) says: "
           "<d>[English] Come on.</d>\n\n[Shot 2] At 00:03.000, the camera cuts to the sea.\n\n"
           "**overall_soundscape:** Waves break on the pilings.\nnon_diegetic_music: N/A\n```")
    assert C.clean_refined(raw, lines=True) == (
        "[Shot 1] Live-action, cinematic, a wide shot frames @Anna on a pier. The camera pushes in as @Anna says: "
        "\"Come on.\"\n[Shot 2] At 00:03.000, the camera cuts to the sea.\n"
        "overall_soundscape: Waves break on the pilings.\nnon_diegetic_music: N/A")
    header = ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
              "referenced.\n\nintegrated_multimodal_description: She smiles.\n\noverall_soundscape: Wind.")
    assert C.clean_refined(header, lines=True) == "She smiles.\noverall_soundscape: Wind."   # music left out
    assert C.clean_refined("“She smiles. Birds sing.”", lines=True) == "She smiles. Birds sing."   # no sound lines
    inline = "She smiles. overall_soundscape: Wind.  non_diegetic_music:   "                # on one line, music empty
    assert C.clean_refined(inline, lines=True) == "She smiles.\noverall_soundscape: Wind."
    assert C.clean_refined("<think>cut off", lines=True) == ""


def test_h3_refine_sees_the_start_frame_only_with_a_vision_model(server, start):
    r = C.ComfyPromptRefiner(server.url)
    gemma = r.models[0]["file"]
    qwen = next(m["file"] for m in r.vision_models())
    assert "gemma" in gemma and "qwen3vl" in qwen
    server.refine_answer = "She waves.\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
    out = r.refine_prompt("she waves", "video_h3", qwen, 1, seconds=73 / 24, start_image=start)
    assert out == "She waves.\noverall_soundscape: Wind.\nnon_diegetic_music: N/A"           # lines kept
    wf = server.prompts[-1]
    assert wf["gen"]["inputs"]["image"] == ["pic", 0] and wf["pic"]["class_type"] == "LoadImage"
    assert "Anchor the first sentences" in wf["gen"]["inputs"]["system_prompt"]
    assert "3.04 seconds" in wf["gen"]["inputs"]["system_prompt"] and wf["gen"]["inputs"]["max_length"] == 1024
    r.refine_prompt("she waves", "video_h3", gemma, 1, seconds=2, start_image=start)           # cannot see
    wf = server.prompts[-1]
    assert "pic" not in wf and "image" not in wf["gen"]["inputs"]
    assert "Do not invent how it looks" in wf["gen"]["inputs"]["system_prompt"]
    r.refine_prompt("she waves", "video_h3", qwen, 1, seconds=2, start_image=start.with_name("later.png"))
    assert "pic" not in server.prompts[-1] and "cannot see" in server.prompts[-1]["gen"]["inputs"]["system_prompt"]
    r.refine_prompt("she waves", "video_h3", qwen, 1, seconds=2)                               # no start frame
    sp = server.prompts[-1]["gen"]["inputs"]["system_prompt"]
    assert "pic" not in server.prompts[-1] and "first frame" not in sp and "2.00 seconds" in sp
    server.refine_answer = "A man walks.\n\noverall_soundscape: Steps."
    assert r.refine_prompt("a man walks", "video", qwen, 1, seconds=2, start_image=start) == \
        "A man walks. overall_soundscape: Steps."                                              # classic: one line
    assert "pic" not in server.prompts[-1] and server.prompts[-1]["gen"]["inputs"]["max_length"] == 512


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
    # accuracy first: the full-precision file before int8 when both are there
    assert q21["base"].endswith("2.1_bf16.safetensors") and q21["turbo"].endswith("turbo_int8_convrot.safetensors")
    assert q21["clip"] == "qwen3vl_8b_bf16.safetensors" and q21["vae"] == "qwen_image_2.1_vae_bf16.safetensors"
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
    gen.generate("x", start, tmp_path / "c.mp4", tmp_path / "c.png", 3, settings(steps=6, pika={"strength": 1.0}, cfg=1.3))
    assert server.prompts[-1]["guider"]["class_type"] == "CFGGuider" and server.prompts[-1]["guider"]["inputs"]["cfg"] == 1.3
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


def test_qwen21_inpaint_with_a_painted_mask(server, start, tmp_path):
    """A painted mask: the Qwen-Image 2.1 Fun ControlNet (Promptwood's node) draws the masked part anew, the
    result is laid into the source along a soft edge; other models and a missing ControlNet / node say why not."""
    mask = tmp_path / "mask.png"
    Image.new("L", (192, 108), 0).save(mask)
    ed = C.ComfyImageEditor(server.url)
    assert ed.can_inpaint("qwen21") == "" and "Qwen-Image 2.1" in ed.can_inpaint("2511")
    assert ed.inpaint_patch == "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"   # not the older Qwen-Image's
    out = tmp_path / "e.png"
    meta = ed.edit(start, "a red scarf", out, 9, "fast", model="qwen21", mask=mask)
    wf = server.edited
    assert meta["inpaint"]["patch"] == ed.inpaint_patch
    ip = wf["inpaint"]["inputs"]
    assert wf["inpaint"]["class_type"] == C.INPAINT21_NODE and ip["inpaint_image"] == ["pic0", 0]
    assert ip["mask"] == ["imask", 0] and ip["model"] == ["unet", 0] and "cache" not in wf   # (no prefix cache)
    assert wf["guider"]["inputs"]["model"] == ["inpaint", 0]
    assert wf["ipatch"]["inputs"]["name"] == ed.inpaint_patch and wf["imask"]["inputs"]["channel"] == "red"
    # the edit's picture: the source as it is (a grey hole there is copied as a grey disc: measured, Turbo)
    assert wf["text"]["inputs"]["images.image_1"] == ["pic0", 0] and "ihole" not in wf and "igrey" not in wf
    comp = wf["icompose"]["inputs"]
    assert comp["destination"] == ["pic0", 0] and comp["source"] == ["decode", 0] and comp["resize_source"] is True
    assert comp["mask"] == ["isoft", 0] and meta["inpaint"]["feather"] == C.inpaint_feather(192, 108) == 4
    assert wf["igrow"]["inputs"]["expand"] == 4 and wf["isoft_blur"]["inputs"]["sigma"] <= 10   # the whole hole soft
    assert C.inpaint_feather(4000, 3000) == 18 and C.inpaint_feather(10000, 9000) == 20
    assert wf["edit_save"]["inputs"]["images"] == ["icompose", 0]
    ed.edit(start, "night", out, 9, "fast", model="qwen21")                 # no mask: as before
    assert "inpaint" not in server.edited and server.edited["edit_save"]["inputs"]["images"] == ["decode", 0]
    assert server.edited["guider"]["inputs"]["model"] == ["cache", 0]
    with pytest.raises(C.ComfyError, match="Qwen-Image 2.1"):
        ed.edit(start, "x", out, 1, "fast", mask=mask)                       # 2511: no mask
    other = tmp_path / "other_mask.png"
    Image.new("L", (100, 100), 0).save(other)
    with pytest.raises(C.ComfyError, match="paint the mask again"):         # a mask of another size
        ed.edit(start, "x", out, 1, "fast", model="qwen21", mask=other)
    server.inpaint21 = False
    ed = C.ComfyImageEditor(server.url)
    assert "Fun ControlNet" in ed.can_inpaint("qwen21")
    with pytest.raises(C.ComfyError, match="Fun ControlNet"):
        ed.edit(start, "x", out, 1, "fast", model="qwen21", mask=mask)
    assert C.pick_qwen21_inpaint(["qwen_image_inpaint_diffsynth_controlnet.safetensors"]) is None


def test_lora_recipe_sampling():
    """A LoRA's own sampling (the head swap recipe): sigma shift after the LoRAs, its sampler and scheduler."""
    wf = C.build_workflow(prompt="head_swap: x", image=None, seed=1, width=672, height=384, frames=107, steps=8,
                          loras=[("MiniMax H3/dmd.safetensors", 1.0), ("MiniMax H3/head_swap.safetensors", 1.0)],
                          audio=True, prefix="p", refs=["ref.png"],
                          guides=[{"name": "clip.mp4", "frame_idx": 0, "audio": True, "at": "start"}],
                          recipe={"sampler": "euler", "scheduler": "beta", "shift": (12.0, 3.0)})
    assert wf["lora1"]["inputs"]["model"] == ["lora0", 0] and wf["shift"]["inputs"]["model"] == ["lora1", 0]
    assert (wf["shift"]["inputs"]["shift_video"], wf["shift"]["inputs"]["shift_audio"]) == (12.0, 3.0)
    assert wf["sampler"]["inputs"]["sampler_name"] == "euler" and wf["sigmas"]["inputs"]["scheduler"] == "beta"
    assert wf["sigmas"]["inputs"]["steps"] == 8 and wf["guider"]["inputs"]["model"] == ["shift", 0]
    assert wf["guide0"]["inputs"]["frame_idx"] == 0 and "audio" in wf["guide0"]["inputs"]
    with pytest.raises(ValueError, match="Turbo"):
        C.build_workflow(prompt="x", image=None, seed=1, width=64, height=64, frames=22, steps=8, loras=[],
                         audio=False, prefix="p", turbo=("t.safetensors", 1.0), recipe={"sampler": "euler"})


def test_describe_face_workflow():
    """The head swap's face details: a vision model (Qwen3-VL) looks at the picture with the creator's instruction."""
    wf = C.build_describe_face_workflow(image="video-chain/ref.png", model={"file": "qwen3vl_4b_fp8_scaled.safetensors",
                                                                         "type": "krea2"})
    g = wf["gen"]["inputs"]
    assert g["image"] == ["pic", 0] and g["prompt"] == C.FACE_INSTRUCTION and "Never name" in g["prompt"]
    assert wf["llm"]["inputs"]["type"] == "krea2" and wf["show"]["inputs"]["source"] == ["gen", 0]


def test_seedvr2_upscale(server, start, tmp_path):
    """SeedVR2 (ComfyUI's core nodes, as its blueprint): the clip scaled, restored in one step, chunked in time,
    colour-matched; its frames and sound kept; the 7B / 3B int8 model (never the "sharp" one) and the VAE."""
    assert C.pick_seedvr_files(["x/seedvr2_7b_fp16.safetensors", "seedvr2_7b_sharp_int8_convrot.safetensors",
                                "seedvr2_7b_int8_convrot.safetensors"], ["a.safetensors",
                                "seedvr2_ema_vae_fp16.safetensors"], "7b") == {
        "unet": "seedvr2_7b_int8_convrot.safetensors", "vae": "seedvr2_ema_vae_fp16.safetensors"}
    assert C.pick_seedvr_files(["seedvr2_7b_fp16.safetensors"], [], "3b") == {"unet": None, "vae": None}
    gen = C.ComfyGenerator(server.url)
    draft, frame = tmp_path / "d.mp4", tmp_path / "d.png"
    gen.generate("x", start, draft, frame, 1, settings())
    out = tmp_path / "s.mp4"
    gen.seedvr(draft, out, (1888, 1088))
    info = chain.probe(out)
    n = chain.probe(draft)["frames"]
    assert (info["width"], info["height"], info["frames"], info["audio"]) == (1888, 1088, n, True)
    wf = server.prompts[-1]
    assert server.seedvr_jobs == ["seedvr2_7b_int8_convrot.safetensors"]
    assert wf["post"]["inputs"]["color_correction_method"] == "lab"
    assert wf["resize"]["inputs"]["upscale_method"] == "lanczos" and wf["vae"]["inputs"]["vae_name"].startswith("seedvr2")
    gen.seedvr(draft, out, (1888, 1088), "seedvr2_3b")
    assert server.seedvr_jobs[-1] == "seedvr2_3b_int8_convrot.safetensors"
    server.seedvr = False                                                       # not installed: a clear message
    with pytest.raises(C.ComfyError, match="SeedVR2"):
        gen.seedvr(draft, out, (1888, 1088))


def test_veda_only_where_it_gains_and_in_the_workflow(server, start, tmp_path):
    """Veda: used from VEDA_MIN_TOKENS (a 5 s clip at 672x384 measured slower, 10 s at 1344x768 2.3x faster); the
    node sits after every LoRA, before the guider; the R2VA predictor for clips with references."""
    assert chain.h3_tokens(672, 384, 121) == 37 * 42 * 24 and not chain.veda_useful(672, 384, 121)
    assert chain.veda_useful(1344, 768, 241) and chain.veda_useful(1344, 768, 124)        # 290k / 149k tokens
    assert chain.veda_from_seconds(672, 384) is None                                      # not even at 15 s
    assert abs(chain.veda_from_seconds(1344, 768) - 107 / 24) < 1e-9
    assert "veda" not in settings() and settings(veda=True)["veda"] is True
    assert chain.settings_dict(model="m", lora=None, strength=1.0, frames=39, steps=8, width=672, height=384,
                               audio=True, veda=True)["veda"] is True
    names = ["minimax_h3_r2va_veda_preview_fp8.safetensors",
             "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"]
    assert C.pick_veda_predictor(names, False).startswith("minimax_h3_t2va")
    assert C.pick_veda_predictor(names, True).startswith("minimax_h3_r2va")
    assert C.pick_veda_predictor(names[1:], True) == names[1] and C.pick_veda_predictor([], False) is None
    gen = C.ComfyGenerator(server.url)
    out, frame = tmp_path / "v.mp4", tmp_path / "v.png"
    lora = "/data/my_models/loras/MiniMax H3/glow.safetensors"
    gen.generate("x", start, out, frame, 1, settings(veda=True, lora=lora))
    wf = server.prompts[-1]
    assert wf["veda"]["inputs"]["model"] == ["lora0", 0] and wf["guider"]["inputs"]["model"] == ["veda", 0]
    assert wf["veda"]["inputs"]["predictor"].startswith("minimax_h3_t2va")
    gen.generate("x", start, out, frame, 1, settings())
    assert "veda" not in server.prompts[-1]                                               # small clip: none
    server.veda = False                                                                   # not installed: said so
    with pytest.raises(C.ComfyError, match="Veda"):
        gen.generate("x", start, out, frame, 1, settings(veda=True))


def test_seedvr2_long_clip_in_pieces_within_ram(server, tmp_path, monkeypatch):
    """A long clip goes to SeedVR2 in pieces that fit the RAM budget (a 10 s 2K clip as one job made systemd-oomd
    kill ComfyUI): overlapping by SEEDVR_PIECE_OVERLAP frames, crossfaded, the clip's own frame count and sound;
    ComfyUI's other models are unloaded before the first SeedVR2 job, its cached frames after every piece."""
    import subprocess

    assert C.seedvr_piece_frames(2560, 1440) == 41 and C.seedvr_piece_frames(1920, 1088) == 73   # (12 GiB fallback)
    budget = C.seedvr_budget(55 * 2**30)                                              # 55 GiB free: 45 GiB for frames
    assert budget == 45 * 2**30 and C.seedvr_budget(None) == C.SEEDVR_RAM_BUDGET and C.seedvr_budget(2**30) == 4 * 2**30
    assert C.seedvr_piece_frames(2560, 1440, budget) == 153                           # about 6.4 s at 2K per piece
    assert C.seedvr_piece_frames(1920, 1088, budget) == 273                           # about 11.4 s at 1080p
    assert C.seedvr_pieces(30, 45) == [(0, 30)]
    assert C.seedvr_pieces(100, 45) == [(0, 45), (37, 45), (55, 45)]                  # the last one ends at the end
    assert C.seedvr_pieces(82, 45) == [(0, 45), (37, 45)]
    draft = tmp_path / "long.mp4"                                                     # 4 s: gray, then white
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=64x36:r=24:d=2", "-f", "lavfi",
                    "-i", "color=white:s=64x36:r=24:d=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(draft)], check=True)
    monkeypatch.setattr(seedvr, "SEEDVR_RAM_BUDGET", C.SEEDVR_FRAME_COPIES * 128 * 72 * 3 * 4 * 33)   # 33 at 128x72
    gen = C.ComfyGenerator(server.url)
    gen.generate("x", None, tmp_path / "c.mp4", tmp_path / "c.png", 1, settings())    # an H3 clip before
    server.calls.clear()
    out = tmp_path / "up.mp4"
    gen.seedvr(draft, out, (128, 72))
    info = chain.probe(out)
    assert (info["width"], info["height"], info["frames"], info["audio"]) == (128, 72, 96, True)
    assert len(server.seedvr_jobs) == 4                                               # 96 frames in 33-frame pieces
    frees = [c[2] for c in server.calls if c[:2] == ("BODY", "/free")]
    assert frees[0] == {"unload_models": True, "free_memory": True}                   # the H3 models' RAM first
    assert frees[1:] == [{"unload_models": False, "free_memory": True}] * 4           # each piece's cache after
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt",
                        "gray", "-"], capture_output=True, check=True).stdout
    assert r[0] < 160 and r[-1] > 230 and all(b >= a - 3 for a, b in zip(r, r[1:]))  # gray -> white, no jumps back
    monkeypatch.setattr(seedvr, "SEEDVR_RAM_MIN", 0)
    server.ram_free = C.SEEDVR_RAM_RESERVE + C.SEEDVR_FRAME_COPIES * 128 * 72 * 3 * 4 * 65   # measured: 65 frames
    server.seedvr_jobs.clear()
    gen.seedvr(draft, out, (128, 72))
    assert len(server.seedvr_jobs) == 2 and chain.probe(out)["frames"] == 96          # pieces from the free RAM
    server.ram_free = 0
    server.calls.clear()
    gen.seedvr(draft, out, (128, 72))                                                 # SeedVR2 again: no unload
    assert [c[2] for c in server.calls if c[:2] == ("BODY", "/free")][0] == {"unload_models": False,
                                                                             "free_memory": True}


def test_seedvr2_progress_is_one_steady_bar():
    """ComfyUI's counters restart for every node of every piece; the clip's progress only goes forward."""
    seen, notes = [], []
    tr = C.SeedvrProgress(2, lambda i, n: seen.append((i, n)))
    tr.piece(0, notes.append)
    for i in range(1, 5):                  # tiled encode: 4 tiles
        tr.step(i, 4)
    tr.step(1, 1)                          # the one sampling step (a new total: next phase)
    for i in range(1, 9):                  # tiled decode: 8 tiles
        tr.step(i, 8)
    tr.piece(1, notes.append)
    tr.step(1, 4)
    tr.step(0, 4)                          # a counter going back: next phase, never back on the bar
    tr.finish()
    values = [i for i, _n in seen]
    assert all(n == 1000 for _i, n in seen) and values == sorted(values) and values[-1] == 1000
    assert max(v for v in values[:14]) <= 500 and notes == ["SeedVR2 piece 1 of 2", "SeedVR2 piece 2 of 2"]
    one = []
    C.SeedvrProgress(1, lambda i, n: None).piece(0, one.append)
    assert one == ["SeedVR2"]


def test_light_denoise_before_seedvr2(server, tmp_path):
    """SEEDVR_DENOISE calms noise that changes from frame to frame, keeps frames and sound; with denoise the
    uploaded clip is the denoised one."""
    import subprocess
    noisy = tmp_path / "noisy.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=gray:s=96x64:r=24:d=1", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1", "-vf", "noise=alls=3:allf=t", "-c:v", "libx264",
                    "-crf", "4", "-pix_fmt", "yuv420p", "-c:a", "aac", str(noisy)], check=True)
    clean = tmp_path / "clean.mp4"
    C._denoise(noisy, clean)
    def change(v):
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(v), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                           capture_output=True, check=True).stdout
        return sum(abs(a - b) for a, b in zip(r, r[96 * 64:])) / (len(r) - 96 * 64)
    assert change(clean) < 0.6 * change(noisy)                 # fine frame-to-frame noise (as in drafts) calmed
    a, b = chain.probe(noisy), chain.probe(clean)
    assert (b["frames"], b["audio"]) == (a["frames"], True)
    gen = C.ComfyGenerator(server.url)
    out = tmp_path / "up.mp4"
    before = set((server.root / "input").rglob("*.mp4"))
    gen.cleanup = False
    gen.seedvr(noisy, out, (192, 128), denoise=True)
    uploaded = [f for f in (server.root / "input").rglob("*.mp4") if f not in before]
    assert uploaded and change(uploaded[0]) < 0.6 * change(noisy)               # SeedVR2 got the denoised clip
    assert chain.probe(out)["frames"] == a["frames"]


def test_every_video_is_saved_at_a_set_quality():
    """SaveVideo's automatic quality wrote drafts at 2 Mbit/s (blocks, flicker SeedVR2 enlarged): every workflow
    saves H.264 at VIDEO_CRF (the fake ComfyUI checks every SaveVideo too)."""
    node = C.save_video("p")
    assert node["inputs"]["format.codec.encoding"] == "re-encode" and node["inputs"]["format.codec.encoding.crf"] == 12.0
    clip = C.build_workflow(prompt="x", image=None, seed=1, width=672, height=384, frames=39, steps=8, loras=[],
                            audio=True, prefix="p")
    up = C.build_seedvr_workflow(draft="d.mp4", width=1888, height=1088, unet="u", vae="v", prefix="p")
    ref = C.build_refine_workflow(prompt="x", draft="d.mp4", seed=1, width=1888, height=1088, frames=39, loras=[],
                                  refiner="r", upscaler="u", prefix="p")
    for wf in (clip, up, ref):
        saves = [n for n in wf.values() if n["class_type"] == "SaveVideo"]
        assert saves and all(n["inputs"]["format.codec.encoding.crf"] == C.VIDEO_CRF for n in saves)


def test_own_sampling_settings_for_new_images(server, start, tmp_path):
    """Steps, cfg, sampler and scheduler can be set for every family: KSampler workflows directly, the custom
    sampler ones (Krea-2 with pictures, Qwen-Image 2.1 Turbo's fixed sigmas) through their nodes."""
    from video_chain import imagegen as G
    cr = G.ComfyImageCreator(server.url)
    assert "euler" in cr.files["sampling"]["samplers"] and "karras" in cr.files["sampling"]["schedulers"]
    out = tmp_path / "n.png"
    own = {"steps": 33, "cfg": 2.5, "sampler": "dpmpp_2m", "scheduler": "karras"}
    meta = cr.create("sdxl", None, "1girl", out, 5, "fast", 1024, 1024, sampling=own)
    s = server.created["sample"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"], s["scheduler"]) == (33, 2.5, "dpmpp_2m", "karras")
    assert {k: meta[k] for k in own} == own and meta["sampling_set"] == own
    meta = cr.create("sdxl", None, "1girl", out, 5, "best", 1024, 1024, sampling={"steps": 12, "cfg": None})
    s = server.created["sample"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"]) == (12, 6.0, "dpmpp_2m") and meta["sampling_set"] == {"steps": 12}
    cr.create("krea2", None, "in this style", out, 5, "standard", 1024, 1024, refs=[start], sampling=own)
    wf = server.created
    assert wf["sigmas"]["inputs"]["steps"] == 33 and wf["sigmas"]["inputs"]["scheduler"] == "karras"
    assert wf["guider"]["inputs"]["cfg"] == 2.5 and wf["sampler"]["inputs"]["sampler_name"] == "dpmpp_2m"
    wf = G.build_create_workflow(family="qwen21", files=cr.files["qwen21"], model=None, prompt="x", negative="",
                                 width=1024, height=1024, seed=1, quality="fast", loras=[], refs=[], ref_strength=0.6,
                                 prefix="p")
    assert wf["sample"]["class_type"] == "SamplerCustomAdvanced"       # Turbo: its fixed sigma list
    assert G.sampling_of(wf)["scheduler"] == "manual" and G.sampling_of(wf)["steps"] == 8
    assert G.apply_sampling(wf, sampler="heun")["scheduler"] == "manual"              # kept unless asked
    used = G.apply_sampling(wf, steps=12)
    assert wf["sigmas"]["class_type"] == "BasicScheduler" and used["steps"] == 12 and used["scheduler"] == "simple"
    auto = G.default_sampling("chroma", cr.files["chroma"], None, "best")
    assert auto == {"steps": 26, "cfg": 3.5, "sampler": "euler", "scheduler": "beta"}
    assert G.default_sampling("flux3", cr.files["flux3"], None, "fast") is None


def test_negative_prompt_for_cfg1_families(server, start, tmp_path):
    """Krea-2 and FLUX.1 encode an empty negative unless one is given (it counts with CFG above 1)."""
    from video_chain import imagegen as G
    cr = G.ComfyImageCreator(server.url)
    out = tmp_path / "n.png"
    cr.create("krea2", None, "a fox", out, 5, "standard", 1024, 1024)
    assert server.created["neg"]["class_type"] == "ConditioningZeroOut"
    cr.create("krea2", None, "a fox", out, 5, "standard", 1024, 1024, negative="blurry", sampling={"cfg": 3.0})
    wf = server.created
    assert wf["neg"] == {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": "blurry"}}
    assert wf["sample"]["inputs"]["cfg"] == 3.0
    cr.create("krea2", None, "in this style", out, 5, "standard", 1024, 1024, refs=[start], negative="blurry")
    wf = server.created
    assert wf["neg0"]["class_type"] == "TextEncodeQwenImageEditPlus" and wf["neg0"]["inputs"]["prompt"] == "blurry"
    assert wf["neg0"]["inputs"]["image1"] == wf["pos0"]["inputs"]["image1"] and wf["guider"]["inputs"]["negative"] == ["neg", 0]
    cr.create("flux1", None, "a cat", out, 5, "standard", 1024, 1024, negative="dog")
    assert server.created["neg"]["inputs"]["text"] == "dog"


def test_middle_keyframes_in_the_workflow(server, start, tmp_path):
    """Each keyframe: a LoadImage and a MiniMaxH3AddGuide at its frame, after the end frame and the motion guides,
    in every model (also the reference one and with CFG: the negative is made from the same conditioning)."""
    keys = [("k0.png", 20), ("k1.png", 40)]
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=20, loras=[],
                          audio=True, prefix="p", end_image="e.png", keyframes=keys)
    k0, k1 = wf["key_guide0"]["inputs"], wf["key_guide1"]["inputs"]
    assert wf["key0"]["inputs"]["image"] == "k0.png" and k0["image"] == ["key0", 0] and k0["frame_idx"] == 20
    assert k0["positive"] == ["cond", 0] and k1["positive"] == ["key_guide0", 0] and k1["latent"] == ["cond", 1]
    assert wf["guider"]["inputs"]["conditioning"] == ["key_guide1", 0]
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=20, loras=[],
                          audio=False, prefix="p", refs=["r.png"], end_image="e.png", keyframes=keys, cfg=2.0)
    assert wf["key_guide0"]["inputs"]["positive"] == ["end_guide", 0]
    assert wf["guider"]["inputs"]["positive"] == ["key_guide1", 0] and \
        wf["neg"]["inputs"]["conditioning"] == ["key_guide1", 0]
    (server.root / "main.py").write_text("")
    pic = tmp_path / "key.png"
    Image.new("RGB", (192, 108), (0, 250, 0)).save(pic)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1,
                 settings(keyframes=[{"image": str(pic), "frame_idx": 12}]))
    wf = server.prompts[-1]
    assert wf["key_guide0"]["inputs"]["frame_idx"] == 12 and wf["key0"]["inputs"]["image"].endswith(".png")
    assert not list((server.root / "input" / "video-chain").glob("*"))      # the upload is cleaned up


def test_seedvr2_timing_per_node_and_transfer(server, tmp_path, monkeypatch):
    """Each SeedVR2 job logs where its time went: upload and download (MB, s, MB/s) and ComfyUI's time by node type,
    from its "executing" messages (a node runs until the next starts; cached nodes are not counted)."""
    import subprocess
    gen = C.ComfyGenerator(server.url)
    clock = iter([0.0, 2.0, 2.5])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    gen.node_times, gen._running_node = {}, None
    for node in ("enc", "sample", None):
        gen.handle_message({"type": "executing", "data": {"node": node, "prompt_id": "p"}}, "p")
    gen.handle_message({"type": "executing", "data": {"node": "x", "prompt_id": "other"}}, "p")   # not ours
    monkeypatch.undo()
    assert gen.node_times == {"enc": 2.0, "sample": 0.5}
    wf = {"enc": {"class_type": "VAEEncodeTiled"}, "dec": {"class_type": "VAEDecodeTiled"},
          "sample": {"class_type": "KSampler"}}
    line = C.seedvr_timing((1920, 1088), "seedvr2_7b_int8_convrot.safetensors", 25e6, 5.0, 40.0, 80e6, 4.0,
                           {"enc": 12.0, "dec": 20.0, "sample": 6.0, "x": 0.01}, wf)
    assert line == ("SeedVR2 1920x1088 (seedvr2_7b_int8_convrot.safetensors): upload 25.0 MB in 5.0 s (5.0 MB/s); "
                    "ComfyUI 40.0 s [VAEDecodeTiled 20.0 s, VAEEncodeTiled 12.0 s, KSampler 6.0 s]; "
                    "download 80.0 MB in 4.0 s (20.0 MB/s)")
    logs = []
    gen = C.ComfyGenerator(server.url, log=logs.append)
    draft = tmp_path / "d.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x36:r=24:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(draft)], check=True)
    gen.seedvr(draft, tmp_path / "up.mp4", (128, 72))
    assert any(x.startswith("SeedVR2 128x72 (seedvr2_7b_int8_convrot.safetensors): upload ") for x in logs)


def test_seedvr2_full_precision_and_whole_frames_when_asked():
    """The model order for big GPUs (fp16 first) and the untiled workflow; never the mxfp8 / nvfp4 files."""
    unets = ["seedvr2_7b_int8_convrot.safetensors", "seedvr2_7b_fp16.safetensors", "seedvr2_7b_mxfp8.safetensors",
             "seedvr2_7b_nvfp4.safetensors", "seedvr2_7b_sharp_fp16.safetensors"]
    assert C.pick_seedvr_files(unets, [], "7b")["unet"] == "seedvr2_7b_int8_convrot.safetensors"     # as at home
    assert C.pick_seedvr_files(unets, [], "7b", C.SEEDVR_PRECISION_FULL)["unet"] == "seedvr2_7b_fp16.safetensors"
    assert C.pick_seedvr_files(unets[2:4], [], "7b")["unet"] is None
    wf = C.build_seedvr_workflow(draft="d.mp4", width=1920, height=1088, unet="u", vae="v", prefix="p", tiled=False)
    assert wf["enc"] == {"class_type": "VAEEncode", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0]}}
    assert wf["dec"] == {"class_type": "VAEDecode", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0]}}
    assert C.build_seedvr_workflow(draft="d.mp4", width=1920, height=1088, unet="u", vae="v",
                                   prefix="p")["enc"]["class_type"] == "VAEEncodeTiled"


def test_seedvr2_lanczos_on_all_cores_when_comfyui_has_the_node(server, tmp_path):
    """With video-chain's VideoChainLanczosScale in ComfyUI (the same frames as ImageScale's lanczos, see
    tests/test_scale_node.py) the resize uses it; without, ImageScale as before."""
    import subprocess
    draft = tmp_path / "d.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x36:r=24:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(draft)], check=True)
    gen = C.ComfyGenerator(server.url)
    gen.seedvr(draft, tmp_path / "a.mp4", (128, 72))
    assert server.prompts[-1]["resize"]["class_type"] == "ImageScale"
    server.scale_node = True
    gen.seedvr(draft, tmp_path / "b.mp4", (128, 72))
    r = server.prompts[-1]["resize"]
    assert r == {"class_type": "VideoChainLanczosScale", "inputs": {"image": ["parts", 0], "width": 128, "height": 72}}


def test_frames_as_numbered_reference_pictures():
    """h3: the plain start / end frame also goes in as a reference picture after the subject pictures (start
    first), the pinning stays; a frame that is not there, or the plain model, refuses it."""
    kw = dict(seed=1, width=512, height=512, frames=39, steps=20, loras=[], audio=False, prefix="p")
    wf = C.build_workflow(prompt="x", image="s.png", end_image="e.png", refs=["r1.png", "r2.png"],
                          frame_refs=["end", "start"], **kw)
    ref = wf["refcond"]["inputs"]
    assert [ref[f"ref_images.ref_image_{i}"] for i in range(4)] == [["ref0", 0], ["ref1", 0], ["start", 0],
                                                                    ["end", 0]]
    assert wf["cond"]["inputs"]["image"] == ["start", 0] and wf["end_guide"]["inputs"]["image"] == ["end", 0]
    assert ref["ref_image_size"] == "match"                    # (the node scales a picture down, never crops it)
    wf = C.build_workflow(prompt="x", image="s.png", audio_refs=["a.wav"], frame_refs=["start"], **kw)
    assert wf["refcond"]["inputs"]["ref_images.ref_image_0"] == ["start", 0]          # sounds only: Picture 1
    with pytest.raises(ValueError, match="no plain end frame"):
        C.build_workflow(prompt="x", image="s.png", refs=["r.png"], frame_refs=["end"], **kw)
    with pytest.raises(ValueError, match="no plain start frame"):
        C.build_workflow(prompt="x", image="s.png", refs=["r.png"], frame_refs=["start"],
                         guides=[{"name": "g.mp4", "frame_idx": 0, "at": "start"}], **kw)
    with pytest.raises(ValueError, match="only with the reference model"):
        C.build_workflow(prompt="x", image="s.png", frame_refs=["start"], **kw)


def test_h3_tree_and_workflow_agree_on_the_numbers(server, tmp_path):
    """A clip of an h3 project made through ComfyUI: the prompt's <Picture N> are the workflow's ref_images in
    that order (a person's 2 pictures, a place, the start frame), no reference sentence is added."""
    from video_chain import tree as T
    (server.root / "main.py").write_text("")
    img = tmp_path / "in.png"
    Image.new("RGB", (192, 108), (200, 10, 10)).save(img)
    proj = tmp_path / "proj"
    t = T.Tree()
    t.options.update({"prompt_format": "h3", "frames": 39})
    T.set_start_image(proj.mkdir() or proj, img, t)
    for i, rgb in enumerate([(200, 150, 130), (190, 140, 120), (90, 90, 200)]):
        Image.new("RGB", (64, 80), rgb).save(tmp_path / f"p{i}.png")
    T.add_character(proj, t, "Anna", tmp_path / "p0.png", desc="woman, 30s")
    T.add_picture(proj, t, "Anna", tmp_path / "p1.png")
    T.add_character(proj, t, "Kitchen", tmp_path / "p2.png", kind="place")
    a = t.add(None, "@Anna walks into @Kitchen.")
    T.save(proj, t)
    T.run_tree(C.ComfyGenerator(server.url, comfy_dir=server.root), proj, [a.id], log=lambda _m: None)
    wf = server.prompts[-1]
    ref = wf["refcond"]["inputs"]
    keys = sorted(k for k in ref if k.startswith("ref_images."))
    assert keys == [f"ref_images.ref_image_{i}" for i in range(4)] and ref["ref_images.ref_image_3"] == ["start", 0]
    assert [wf[ref[k][0]]["inputs"]["image"].startswith("video-chain/") for k in keys[:3]] == [True] * 3
    assert wf["cond"]["inputs"]["image"] == ["start_still", 0]                    # the start is still pinned
    t = T.load(proj)
    assert ref["prompt"] == T.clip_inputs(proj, t, a.id)["prompt"] and C.REF_HINT not in ref["prompt"]
    assert "<Subject 1> is Anna (woman, 30s), the person in <Picture 1> and <Picture 2>." in ref["prompt"]
    assert "<Subject 2> is Kitchen, the environment in <Picture 3>." in ref["prompt"]
    assert "<Picture 4> is the first frame of [Shot 1]." in ref["prompt"]
    assert t.nodes[a.id].made["settings"]["frame_pictures"] == ["start"]
    assert T.Checker(proj, t).status(a.id) == "done"


def test_refiner_is_told_what_a_reference_is():
    persons = [{"name": "Anna", "desc": "woman, 30s.", "voice": True}, {"name": "Ben", "desc": ""}]
    before = C.cast_instruction(persons)
    assert "@Anna (woman, 30s), has a voice; @Ben." in before and "not a person" not in before
    text = C.cast_instruction(persons + [{"name": "Kitchen", "desc": "sunny", "kind": "place"},
                                         {"name": "Noir", "desc": "", "kind": "style"}])
    assert "@Kitchen (a place: sunny); @Noir (a visual style). @Kitchen, @Noir are not a person: never give them " \
           "a spoken line or a person's actions." in text
    assert text.replace("; @Kitchen (a place: sunny); @Noir (a visual style). @Kitchen, @Noir are not a person: "
                        "never give them a spoken line or a person's actions", "") == before
    from video_chain import tree as T
    t = T.Tree()
    t.options["cast"] = [{"name": "Anna", "images": [], "desc": "woman, 30s.", "lead": True, "voice": {"file": "v"}},
                         {"name": "Ben", "images": [], "desc": "", "lead": False},
                         {"name": "Kitchen", "images": [], "desc": "sunny", "lead": False, "kind": "place"},
                         {"name": "Noir", "images": [], "desc": "", "lead": False, "kind": "style"}]
    assert C.cast_instruction(T.refiner_cast(t)) == text


# ----------------------------------------------------------------------------- continue the clip before (<Video 1>)

def node_labels(cond: dict) -> list[str]:
    """The labels MiniMaxH3ReferenceToVideo gives its references, in its order (nodes_minimax_h3.py): the images,
    then the videos (a video's soundtrack gets an <Audio j> right before its <Video k>), then the standalone audio."""
    out, n = [], {"Picture": 0, "Video": 0, "Audio": 0}

    def add(kind):
        n[kind] += 1
        out.append(f"<{kind} {n[kind]}>")
    for k in cond:
        if k.startswith("ref_images."):
            add("Picture")
    for k in cond:
        if k.startswith("ref_videos."):
            if f"ref_video_audios.ref_video_audio_{k.rsplit('_', 1)[-1]}" in cond:
                add("Audio")
            add("Video")
    for k in cond:
        if k.startswith("ref_audios."):
            add("Audio")
    return out


def test_reference_video_workflow():
    """The tail of the clip before goes in as the first reference video (LoadVideo + GetVideoComponents, pictures
    only): the ref2va model also without pictures or sounds, no soundtrack wired, so a sound stays <Audio 1>; the
    pinning nodes are the same as without it."""
    kw = dict(seed=1, width=512, height=512, frames=39, steps=20, loras=[], audio=True, prefix="p")
    wf = C.build_workflow(prompt="x", image="s.png", ref_videos=["video-chain/tail.mp4"], **kw)
    ref = wf["refcond"]["inputs"]
    assert wf["unet"]["inputs"]["unet_name"] == C.UNET_REF and ref["ref_videos.ref_video_0"] == ["rvparts0", 0]
    assert wf["rvparts0"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["rvideo0", 0]}}
    assert wf["rvideo0"] == {"class_type": "LoadVideo", "inputs": {"file": "video-chain/tail.mp4"}}
    assert not [k for k in ref if k.startswith(("ref_video_audios.", "ref_images.", "ref_audios."))]
    assert "audio_vae" not in ref and wf["cond"]["inputs"]["image"] == ["start", 0]     # the start still pinned
    assert wf["cond"]["inputs"]["frame_idx"] == 0 and wf["cond"]["inputs"]["latent"] == ["refcond", 1]
    full = dict(kw, prompt="x", image="s.png", end_image="e.png", refs=["r.png"], frame_refs=["start"],
                audio_refs=["a.wav"], keyframes=[("k.png", 20)],
                guides=[{"name": "g.mp4", "frame_idx": 30, "frames": 5, "audio": True, "at": "end"}])
    without, wf = C.build_workflow(**full), C.build_workflow(**full, ref_videos=["tail.mp4"],
                                                             video_key=lambda i: f"ref_videos.clip_{i}")
    ref = wf["refcond"]["inputs"]
    assert node_labels(ref) == ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"]   # the sound stays Audio 1
    assert ref["ref_videos.clip_0"] == ["rvparts0", 0] and ref["ref_audios.ref_audio_0"] == ["aref0", 0]
    assert {k: v for k, v in wf.items() if k not in ("rvideo0", "rvparts0", "refcond")} == \
        {k: v for k, v in without.items() if k != "refcond"}                              # pinning unchanged
    assert {k: v for k, v in ref.items() if k != "ref_videos.clip_0"} == without["refcond"]["inputs"]


def test_generate_continuing_a_clip(server, start, tmp_path, monkeypatch):
    """A reference video alone forces the reference model (check_ref_model), is uploaded under the input name the
    node reports and cleaned up; an older ComfyUI without reference videos refuses it."""
    import subprocess
    (server.root / "main.py").write_text("")
    tail = tmp_path / "tail.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=192x108:r=24", "-f", "lavfi",
                    "-i", "sine=f=330", "-frames:v", "22", "-t", f"{22 / 24:.4f}", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-c:a", "aac", str(tail)], check=True)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    checked = []
    real = gen.check_ref_model
    monkeypatch.setattr(gen, "check_ref_model", lambda *a: checked.append(1) or real(*a))
    gen.generate("The car drives on.", start, tmp_path / "v.mp4", tmp_path / "v.png", 3,
                 settings(ref_videos=[str(tail)]))
    wf = server.prompts[-1]
    cond = wf["refcond"]["inputs"]
    assert checked == [1] and wf["unet"]["inputs"]["unet_name"] == C.UNET_REF
    assert cond["ref_videos.ref_video_0"] == ["rvparts0", 0] and wf["rvideo0"]["inputs"]["file"].startswith("video-chain/")
    assert not [k for k in cond if k.startswith(("ref_video_audios.", "ref_images."))] and "audio_vae" not in cond
    assert wf["cond"]["class_type"] == "MiniMaxH3AddGuide" and wf["cond"]["inputs"]["image"] == ["start", 0]
    assert cond["prompt"] == "The car drives on."                                  # (no face hint)
    assert not list((server.root / "input" / "video-chain").glob("*"))             # uploads cleaned up
    info = {"input": {"optional": {"ref_videos": ["COMFY_AUTOGROW_V3", {"template": {"prefix": "vid_"}}]}}}
    monkeypatch.setattr(gen, "_object_info", lambda node: info)
    assert gen.video_key()(0) == "ref_videos.vid_0"
    monkeypatch.undo()
    server.video_refs = False                                                       # an older ComfyUI
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    with pytest.raises(C.ComfyError, match="no reference videos"):
        gen.generate("x", start, tmp_path / "w.mp4", tmp_path / "w.png", 3, settings(ref_videos=[str(tail)]))


def test_h3_tree_continuation_through_comfyui(server, tmp_path):
    """A clip of an h3 project that continues the clip before, made through ComfyUI: <Video 1> is the workflow's
    first reference video (22 frames), the start frame is <Picture 1> and still pinned, the prompt is the one the
    Model prompt window shows."""
    from pathlib import Path

    from video_chain import tree as T
    (server.root / "main.py").write_text("")
    img = tmp_path / "in.png"
    Image.new("RGB", (192, 108), (200, 10, 10)).save(img)
    proj = tmp_path / "proj"
    t = T.Tree()
    t.options.update({"prompt_format": "h3", "frames": 39})
    T.set_start_image(proj.mkdir() or proj, img, t)
    a = t.add(None, "A red car on a coast road.")
    b = t.add(a.id, "The car drives on.")
    b.overrides = {"continue": 22}
    T.save(proj, t)
    sent = []
    real = C.ComfyGenerator.upload

    def upload(self, f):
        if Path(f).name == "continue_from.mp4":
            sent.append(chain.probe(Path(f))["frames"])
        return real(self, f)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    gen.upload = upload.__get__(gen)
    T.run_tree(gen, proj, [b.id], log=lambda _m: None)
    wf = server.prompts[-1]
    ref = wf["refcond"]["inputs"]
    assert sent == [22] and node_labels(ref) == ["<Picture 1>", "<Video 1>"]
    assert ref["ref_images.ref_image_0"] == ["start", 0] and ref["ref_videos.ref_video_0"] == ["rvparts0", 0]
    assert wf["cond"]["inputs"]["image"] == ["start", 0] and wf["cond"]["inputs"]["frame_idx"] == 0
    t = T.load(proj)
    assert ref["prompt"] == T.clip_inputs(proj, t, b.id)["prompt"]
    assert "[keyframe completion + video continuation]" in ref["prompt"]
    assert T.Checker(proj, t).status(b.id) == "done"


def test_another_h3_checkpoint_in_the_workflow():
    """A chosen H3 checkpoint (a fine-tune) replaces the standard model for plain and reference clips; a distilled
    one samples with its few-step recipe (shift, sampler, scheduler), with exact attention."""
    plain = C.build_workflow(prompt="x", image="s.png", seed=1, width=672, height=384, frames=73, steps=28,
                             loras=[], audio=True, prefix="p", unet="h3/fine.safetensors")
    assert plain["unet"]["inputs"]["unet_name"] == "h3/fine.safetensors"
    assert plain["sampler"]["inputs"]["sampler_name"] == C.SAMPLER and plain["sigmas"]["inputs"]["steps"] == 28
    d = C.DISTILLED_CHECKPOINT
    from video_chain import tree as T
    assert d["steps"] == T.DISTILLED_STEPS                                        # (the tree records these steps)
    wf = C.build_workflow(prompt="x", image="s.png", seed=1, width=672, height=384, frames=73, steps=d["steps"],
                          loras=[("A/one.safetensors", 0.8)], audio=True, prefix="p", refs=["r.png"],
                          unet="h3/fine.safetensors", recipe=d["recipe"])
    assert wf["unet"]["inputs"]["unet_name"] == "h3/fine.safetensors"            # (not the ref2va file)
    assert (wf["shift"]["inputs"]["shift_video"], wf["shift"]["inputs"]["shift_audio"]) == (12.0, 3.0)
    assert wf["shift"]["inputs"]["model"] == ["lora0", 0]
    assert wf["sampler"]["inputs"]["sampler_name"] == "euler" and wf["sigmas"]["inputs"]["scheduler"] == "simple"
    assert wf["sigmas"]["inputs"]["steps"] == 8
    assert wf["guider"]["inputs"]["model"] == ["shift", 0] and "sparse" not in wf       # (exact attention)


def test_h3_checkpoints_found_in_comfyuis_model_folders(tmp_path):
    """H3 checkpoints are told by their tensors (the video and audio patch layers), not their names; LoRAs, other
    models and the standard H3 files are left out; the name is the one ComfyUI lists."""
    from ui_helpers import H3_KEYS, write_lora
    root, unet = tmp_path / "diffusion_models", tmp_path / "unet"
    full = ["video_patch_proj.weight", "audio_patch_proj.weight", "blocks.0.attn.qkv_proj.weight"]
    write_lora(root / "sub" / "my_tune.safetensors", full)
    write_lora(unet / "style_tune.safetensors", ["model.diffusion_model." + k for k in full])
    write_lora(root / C.UNET, full)
    write_lora(root / "wan.safetensors", ["blocks.0.self_attn.q.weight"])
    write_lora(root / "h3_lora.safetensors", H3_KEYS + ["video_patch_proj.lora_A.weight"])
    (root / "broken_h3.safetensors").write_bytes(b"no header")
    assert C.find_h3_checkpoints([root, unet, tmp_path / "missing"]) == [
        "style_tune.safetensors", "sub/my_tune.safetensors"]


def test_generating_with_another_h3_checkpoint(server, start, tmp_path):
    """The clip's checkpoint goes into the workflow (also with references: no ref2va file needed); a distilled one
    with its recipe; one ComfyUI does not list is refused with its name."""
    server.extra_unets = ["h3/fine_8step.safetensors"]
    gen = C.ComfyGenerator(server.url)
    out, frame = tmp_path / "v.mp4", tmp_path / "v.png"
    face = tmp_path / "face.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(face)
    ck = {"name": "h3/fine_8step.safetensors", "distilled": True}
    gen.generate("x", start, out, frame, 1, settings(checkpoint=ck, steps=8, ref_paths=[str(face)]))
    wf = server.prompts[-1]
    assert wf["unet"]["inputs"]["unet_name"] == "h3/fine_8step.safetensors" and "refcond" in wf
    assert wf["shift"]["inputs"]["shift_video"] == 12.0 and wf["sampler"]["inputs"]["sampler_name"] == "euler"
    gen.generate("x", start, out, frame, 1, settings(checkpoint={**ck, "distilled": False}))
    wf = server.prompts[-1]
    assert wf["unet"]["inputs"]["unet_name"] == "h3/fine_8step.safetensors" and "shift" not in wf
    with pytest.raises(C.ComfyError, match="gone.safetensors"):
        gen.generate("x", start, out, frame, 1, settings(checkpoint={"name": "gone.safetensors", "distilled": True}))


def test_a_clips_own_sampling_over_a_distilled_checkpoints():
    """A clip's own sampling replaces what it sets and keeps the rest of the distilled checkpoint's recipe."""
    from video_chain.comfy.video import DISTILLED_CHECKPOINT, clip_recipe
    d = DISTILLED_CHECKPOINT["recipe"]
    assert clip_recipe(None, None) is None and clip_recipe(None, d) == d
    assert clip_recipe({"scheduler": "beta"}, None) == {"scheduler": "beta"}
    got = clip_recipe({"scheduler": "beta", "shift": [8.0, 3.0]}, d)
    assert got == {"sampler": d["sampler"], "scheduler": "beta", "shift": [8.0, 3.0]} and d["scheduler"] == "simple"
    wf = C.build_workflow(prompt="p", image="s.png", seed=1, width=384, height=384, frames=73, steps=8, loras=[],
                          audio=False, prefix="p", recipe={"shift": [8.0, 3.0]})
    kinds = {n["class_type"]: n["inputs"] for n in wf.values()}
    assert kinds["MiniMaxH3SigmaShift"]["shift_video"] == 8.0
    assert kinds["KSamplerSelect"]["sampler_name"] == C.SAMPLER and kinds["BasicScheduler"]["scheduler"] == C.SCHEDULER
