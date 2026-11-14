"""The shared helpers: config (paths with env overrides), net (one HTTP helper), logs (one logging set-up)."""
import http.server
import json
import logging
import threading
import time

import pytest


# ---- config
def test_config_defaults_and_env_override(monkeypatch):
    from video_chain import config
    monkeypatch.delenv("VIDEO_CHAIN_HF_CACHE", raising=False)
    assert str(config.hf_cache()) == "/path/to/ai/hf-models"
    monkeypatch.setenv("VIDEO_CHAIN_HF_CACHE", "/elsewhere/models")
    assert str(config.hf_cache()) == "/elsewhere/models"
    monkeypatch.delenv("VIDEO_CHAIN_LLAMA_SERVER", raising=False)
    assert config.llama_server().endswith("llama.cpp/build/bin/llama-server")
    monkeypatch.setenv("VIDEO_CHAIN_LLAMA_SERVER", "/x/llama")
    assert config.llama_server() == "/x/llama"


def test_modules_take_their_paths_from_config(monkeypatch):
    import video_chain.llm_server as L
    import video_chain.tts as T
    from video_chain import config
    assert L.DEFAULT_SERVER == config.llama_server()
    assert T.HF_CACHE == config.hf_cache()


# ---- net
class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/ok":
            body = b'{"a": 1}'
        elif self.path == "/html":
            body = b"<html>proxy</html>"
        elif self.path == "/slow":
            time.sleep(1.5)
            body = b"{}"
        else:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()
    s.server_close()


def test_net_get_json(server):
    from video_chain import net
    assert net.get_json(server + "/ok") == {"a": 1}


def test_net_not_json_is_a_net_error(server):
    from video_chain import net
    with pytest.raises(net.BadAnswer, match="not JSON"):
        net.get_json(server + "/html")


def test_net_http_error_carries_code_and_body(server):
    from video_chain import net
    with pytest.raises(net.HttpStatus) as e:
        net.get_json(server + "/boom")
    assert e.value.code == 500 and "boom" in e.value.body


def test_net_timeout_is_a_net_error(server):
    from video_chain import net
    with pytest.raises(net.Unreachable):
        net.get_json(server + "/slow", timeout=0.3)


def test_net_unreachable():
    from video_chain import net
    with pytest.raises(net.Unreachable):
        net.get_json("http://127.0.0.1:1/x", timeout=1)


def test_net_local_hosts_skip_the_proxy():
    from video_chain import net
    for u in ("http://127.0.0.1:8188", "http://localhost:1", "http://192.0.2.5/x", "http://studio-pc:80/",
              "http://[::1]:5/"):
        assert net.is_local(u), u
    for u in ("https://civitai.com/x", "http://8.8.8.8/"):
        assert not net.is_local(u), u


def test_net_urlopen_public_uses_urllib_urlopen(monkeypatch):
    import urllib.request
    from video_chain import net
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: seen.append((req, timeout)) or "R")
    assert net.urlopen(urllib.request.Request("https://civitai.com/x"), 7) == "R"
    assert seen[0][1] == 7


# ---- logs
def test_setup_logging_writes_to_file_once(tmp_path):
    from video_chain import logs
    f = tmp_path / "app.log"
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        logs.setup(f)
        logs.setup(f)                                   # (again: no second handler)
        logging.getLogger("video_chain.test").warning("hello log")
        for h in root.handlers:
            h.flush()
        assert "hello log" in f.read_text()
        assert len([h for h in root.handlers if getattr(h, "_video_chain", False)]) == 2   # stderr + file
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
                h.close()


def test_the_two_nms_copies_agree():
    """faces.py and the ComfyUI node package keep their own _nms (the node package cannot import video_chain);
    this keeps them equal."""
    import importlib.util
    from pathlib import Path

    import numpy as np
    from video_chain import faces
    path = Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes" / "faceswap.py"
    spec = importlib.util.spec_from_file_location("faceswap_node", path)
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, (40, 2))
    boxes = np.hstack([xy, xy + rng.uniform(10, 60, (40, 2))])
    scores = rng.uniform(0, 1, 40)
    assert list(faces._nms(boxes, scores)) == list(node._nms(boxes, scores))


# ---- keys and contains_word (LLM-24)
def test_contains_word_matches_whole_words_only():
    from video_chain.h3_prompt import contains_word
    assert contains_word("an ink4sketch fox", "Ink4Sketch")
    assert contains_word("ink, sketch", "ink")
    assert not contains_word("pinkish", "ink")
    assert not contains_word("ink_sketch", "ink")
    assert not contains_word("", "ink") and not contains_word(None, "ink")


def test_keys_roundtrip_and_remove(monkeypatch):
    from video_chain import keys
    keys.set_key("svc", "user", "  abc ")
    assert keys.get_key("svc", "user") == "abc"
    keys.set_key("svc", "user", "")
    assert keys.get_key("svc", "user") == ""
    keys.set_key("svc", "user", None)              # (removing what is not there is fine)


def test_keys_without_a_keyring(monkeypatch):
    import keyring

    from video_chain import keys

    def boom(*a):
        raise RuntimeError("locked")
    monkeypatch.setattr(keyring, "get_password", boom)
    monkeypatch.setattr(keyring, "set_password", boom)
    assert keys.read_key("svc", "user") is None and keys.get_key("svc", "user") == ""
    with pytest.raises(RuntimeError, match="keyring is not available"):
        keys.set_key("svc", "user", "x")


def test_the_three_key_modules_use_keys(monkeypatch):
    from video_chain import keys, runpod, triggers, tts
    seen = []
    monkeypatch.setattr(keys, "read_key", lambda s, u: seen.append((s, u)) or "k")
    monkeypatch.setattr(keys, "set_key", lambda s, u, k: seen.append((s, u, k)))
    for var in ("RUNPOD_API_KEY", "ELEVENLABS_API_KEY", "CIVITAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert tts.eleven_key() == "k"          # (runpod.api_key is faked by conftest in every test)
    monkeypatch.setattr(triggers, "_API_KEY", None)
    assert triggers.api_key() == "k"
    runpod.set_api_key("r")
    tts.set_eleven_key("e")
    triggers.set_api_key("c")
    assert len(seen) == 5
