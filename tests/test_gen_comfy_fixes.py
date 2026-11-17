"""imagegen / animate: cancel on any interruption, no truncated pictures, malformed LoRA files, workflow guards."""
import json
import struct

import pytest
from fake_comfy import FakeComfy
from PIL import Image

from video_chain import animate as A
from video_chain import imagegen as G
from video_chain.comfy import ComfyError


@pytest.fixture
def server(tmp_path):
    s = FakeComfy(tmp_path / "comfy")
    yield s
    s.close()


def test_image_is_never_left_truncated(server, tmp_path, monkeypatch):
    out_dir = tmp_path / "pics"
    out_dir.mkdir()
    cr = G.ComfyImageCreator(server.url)
    logged = []
    cr.log = logged.append
    out = out_dir / "n.png"
    cr.create("krea2", None, "a fox", out, 5, "standard", 1024, 1024)
    good = out.read_bytes()
    out_dir.chmod(0o555)                         # the picture's folder cannot be written (as with a full disk)
    try:
        with pytest.raises(OSError) as e:
            cr.create("krea2", None, "a fox", out, 6, "standard", 1024, 1024)
    finally:
        out_dir.chmod(0o755)
    assert not isinstance(e.value, ComfyError)   # (a local disk problem, not "ComfyUI is unavailable")
    assert not [m for m in logged if "trying again" in m]   # ... and not tried again as if the network had failed
    assert out.read_bytes() == good and not list(out_dir.glob("*.part*"))


def test_stop_by_baseexception_cancels_the_job(server, tmp_path, monkeypatch):
    cr = G.ComfyImageCreator(server.url)
    stopped = []
    monkeypatch.setattr(cr, "interrupt", lambda: stopped.append(1))

    def wait(pid, ws):
        raise KeyboardInterrupt
    monkeypatch.setattr(cr, "_wait", wait)
    with pytest.raises(KeyboardInterrupt):
        cr.create("krea2", None, "a fox", tmp_path / "n.png", 5, "standard", 1024, 1024)
    assert stopped == [1]
    an = A.ComfyAnimator(server.url)
    monkeypatch.setattr(an, "interrupt", lambda: stopped.append(2))
    monkeypatch.setattr(an, "_wait", wait)
    monkeypatch.setattr(an, "_queue", lambda wf: None)
    with pytest.raises(KeyboardInterrupt):
        an._run_outputs({"save": {"class_type": "SaveVideo", "inputs": {}}}, {"save": tmp_path / "o.mp4"}, [])
    assert stopped == [1, 2]


def lora_file(path, header):
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return str(path)


def test_malformed_lora_header_values_do_not_break_the_family_scan(tmp_path):
    # a header whose tensor entries are not objects (and one that is a list)
    keys = {f"lora_unet_blocks_{i}_attn_wq.lora_A.weight": "oops" for i in range(3)}
    bad = lora_file(tmp_path / "bad.safetensors", keys)
    listy = lora_file(tmp_path / "list.safetensors", [1, 2])
    G._FAMILY_CACHE.clear()
    assert G.lora_file_family(bad) is None or isinstance(G.lora_file_family(bad), str)
    assert G.lora_file_family(listy) is None
    G.find_image_loras([tmp_path])                                  # (does not raise)


def test_huge_lora_header_is_not_read(tmp_path, monkeypatch):
    p = tmp_path / "huge.safetensors"
    body = b"{" + b" " * (20 * 2**20) + b"}"
    p.write_bytes(struct.pack("<Q", len(body)) + body)
    G._FAMILY_CACHE.clear()
    seen = []
    monkeypatch.setattr(G.json, "loads", lambda *a, **k: seen.append(1) or {})
    assert G.lora_file_family(str(p)) is None and seen == []


def test_symlink_loop_does_not_list_a_lora_twice(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    lora_file(d / "x.safetensors", {"__metadata__": {}})
    (d / "loop").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "again").symlink_to(d, target_is_directory=True)
    G._FAMILY_CACHE.clear()
    found = G.find_image_loras([tmp_path])
    assert sum(len(v) for v in found.values()) <= 1


def test_default_sampling_only_hides_expected_errors(monkeypatch):
    def boom(**k):
        raise TypeError("a bug")
    monkeypatch.setattr(G, "build_create_workflow", boom)
    with pytest.raises(TypeError):
        G.default_sampling("krea2", {"models": ["m"]}, "m", "standard")


def test_sampling_of_a_workflow_without_a_sample_node_is_a_comfy_error():
    with pytest.raises(ComfyError, match="sampl"):
        G.apply_sampling({}, steps=5)


def test_animate_wiring_is_named():
    wf = A.build_animate_workflow(files={"unet": "u", "lightx2v": "l", "relight": "r", "clip": "c", "vae": "v",
                                         "vision": "x"}, video="v.mp4", mask="m.mp4", reference="r.png",
                                  frames=160, src_size=(320, 180), gen=(832, 480), prompt="", seed=1, prefix="p")
    assert wf["anim1"]["inputs"]["video_frame_offset"] == ["anim0", A.OUT_NEXT_OFFSET]
    assert wf["trim0"]["inputs"]["trim_amount"] == ["anim0", A.OUT_TRIM]
    assert wf["new0"]["inputs"]["batch_index"] == ["anim0", A.OUT_NEW_START]


def test_uploads_are_dropped_when_anything_after_the_build_fails(server, tmp_path, monkeypatch):
    an = A.ComfyAnimator(server.url)
    dropped = []
    monkeypatch.setattr(an, "_drop_uploads", lambda ups: dropped.append(len(ups)))
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    Image.new("RGB", (8, 8)).save(tmp_path / "r.png")
    with pytest.raises(AttributeError):                             # (keep_pose of the wrong kind: after the build)
        an.replace(v, v, tmp_path / "r.png", tmp_path / "o.mp4", 1, 30, (320, 180), keep_pose=["x"])
    assert dropped == [3]


@pytest.mark.parametrize("family", ["krea2"])
def test_a_family_without_a_model_file_is_a_clear_error_not_a_null_in_the_workflow(family):
    with pytest.raises(ComfyError, match="no model file"):
        G.build_create_workflow(family=family, files={"clip": "c.safetensors", "vae": "v.safetensors"}, model=None,
                                prompt="x", negative="", width=512, height=512, seed=1, quality="standard",
                                loras=[], refs=[], ref_strength=0.5, prefix="p")
