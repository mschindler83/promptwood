"""comfy.py's helpers and the custom node package: ffmpeg errors, LoRA header sniffing, model folders, node imports."""
import importlib.util
import json
import logging
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fake_comfy import FakeComfy

from video_chain import comfy as C
from video_chain.comfy import loras

NODES = Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes"
sys.path.insert(0, str(NODES))


def safetensors(path: Path, names):
    head = json.dumps({n: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for n in names}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0")
    return path


# ---- ffmpeg (CMF-16)
def test_ffmpeg_failure_is_a_comfy_error_with_the_reason(tmp_path):
    with pytest.raises(C.ComfyError, match="ffmpeg") as e:
        C._ffmpeg(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "o.mp4")])
    assert "missing.mp4" in str(e.value)


def test_ffmpeg_that_hangs_is_stopped(tmp_path):
    with pytest.raises(C.ComfyError, match="did not finish"):
        C._ffmpeg(["-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=600", "-f", "null", "-"], timeout=0.5)


def test_a_missing_ffmpeg_is_a_comfy_error(monkeypatch):
    def gone(*a, **k):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(subprocess, "run", gone)
    with pytest.raises(C.ComfyError, match="ffmpeg"):
        C._ffmpeg(["-version"])


# ---- LoRA headers (CMF-17)
def test_the_header_is_read_once_for_all_three_checks(tmp_path, monkeypatch):
    f = safetensors(tmp_path / "x.safetensors", ["blocks.0.attn.qkv_proj.w", "blocks.0.mlp.fc1.w"])
    reads = []
    real = C._read_safetensors_keys
    monkeypatch.setattr(loras, "_read_safetensors_keys", lambda p: reads.append(p) or real(p))   # (where it is used)
    C._safetensors_keys.cache_clear()
    assert C.is_h3_lora(f) and not C.is_qwen_lora(f) and not C.is_turbo_lora(f)
    assert len(reads) == 1


def test_a_corrupt_header_is_no_lora(tmp_path):
    f = tmp_path / "bad.safetensors"
    f.write_bytes(struct.pack("<Q", 10**12) + b"junk")
    assert not C.is_h3_lora(f)


# ---- model folders (CMF-23)
def test_extra_model_paths_with_a_yaml_list(tmp_path):
    comfy = tmp_path / "comfy"
    (comfy / "models" / "loras").mkdir(parents=True)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (comfy / "extra_model_paths.yaml").write_text(f"x:\n  base_path: {tmp_path}\n  loras:\n    - a\n    - b\n")
    roots = C.comfy_model_roots(comfy, "loras")
    assert a in roots and b in roots


def test_a_symlink_loop_is_walked_once(tmp_path, monkeypatch):
    root = tmp_path / "loras"
    (root / "sub").mkdir(parents=True)
    os.symlink(root, root / "sub" / "loop")
    seen = []
    real = os.walk

    def walk(*a, **k):
        for item in real(*a, **k):
            seen.append(item[0])
            yield item
    monkeypatch.setattr(os, "walk", walk)
    C._find_loras([root], lambda p: True)
    assert len(seen) <= 3


# ---- pieces (CMF-27)
def test_seedvr_pieces_refuses_an_overlap_that_never_advances():
    with pytest.raises(ValueError):
        C.seedvr_pieces(100, 8, 8)
    assert C.seedvr_pieces(18, 17, 8)[-1][0] + C.seedvr_pieces(18, 17, 8)[-1][1] == 18


# ---- settings combination (CMF-20)
def test_frame_pictures_without_references_is_a_comfy_error(tmp_path):
    from PIL import Image
    srv = FakeComfy(tmp_path / "c")
    try:
        g = C.ComfyGenerator(srv.url)
        st = tmp_path / "s.png"
        Image.new("RGB", (64, 64)).save(st)
        with pytest.raises(C.ComfyError, match="reference"):
            g.generate("p", st, tmp_path / "o.mp4", tmp_path / "l.png", 1,
                       {"frame_pictures": ["start"], "width": 64, "height": 64})
    finally:
        srv.close()


# ---- the generator's attributes (CMF-8)
def test_options_are_declared_in_init(tmp_path):
    srv = FakeComfy(tmp_path / "c")
    try:
        g = C.ComfyGenerator(srv.url)
        for name in ("seedvr_tiled", "seedvr_precision", "seedvr_keep_cache", "node_times", "_running_node",
                     "_last_seedvr"):
            assert name in vars(g), name
    finally:
        srv.close()


# ---- the node package (CMF-7, CMF-22)
def load_nodes(monkeypatch, with_qwen: bool):
    for name in list(sys.modules):
        if name.startswith(("comfy", "video_chain_nodes_t")):
            monkeypatch.delitem(sys.modules, name)
    if with_qwen:
        for name in ("comfy", "comfy.ldm", "comfy.ldm.qwen_image21", "comfy.ldm.qwen_image21.model",
                     "comfy_extras", "comfy_extras.nodes_model_patch"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        sys.modules["comfy_extras.nodes_model_patch"].QwenImage21FunControlPatch = object
    else:
        monkeypatch.setitem(sys.modules, "comfy", None)           # (import raises ImportError)
        monkeypatch.setitem(sys.modules, "comfy_extras", None)
    spec = importlib.util.spec_from_file_location("video_chain_nodes_t", NODES / "__init__.py",
                                                  submodule_search_locations=[str(NODES)])
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "video_chain_nodes_t", mod)
    spec.loader.exec_module(mod)
    return mod


def test_the_other_nodes_load_without_qwen_21_internals(monkeypatch):
    mod = load_nodes(monkeypatch, with_qwen=False)
    assert set(mod.NODE_CLASS_MAPPINGS) == {"VideoChainFaceSwap", "VideoChainLanczosScale"}


def test_all_nodes_load_with_qwen_21_internals(monkeypatch):
    mod = load_nodes(monkeypatch, with_qwen=True)
    assert "VideoChainQwen21FunInpaint" in mod.NODE_CLASS_MAPPINGS


def test_the_face_models_run_on_the_cpu_with_a_warning(monkeypatch, caplog):
    import faceswap as F
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    ort.SessionOptions = type("O", (), {})
    ort.InferenceSession = lambda path, opts, providers: providers
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    with caplog.at_level(logging.WARNING):
        assert F._session("x/det.onnx") == ["CPUExecutionProvider"]
    assert "CPU" in caplog.text
