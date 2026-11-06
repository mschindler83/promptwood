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


def test_real_worker_with_comfyui_backend(server, tmp_path):
    """The real CLI worker (--serve --backend comfyui) end to end against the fake ComfyUI."""
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
    p = subprocess.Popen([sys.executable, "-m", "video_chain", "--serve", "--events", "--backend", "comfyui",
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
