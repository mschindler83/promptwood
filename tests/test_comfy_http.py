"""ComfyGenerator's HTTP side against the fake ComfyUI: downloads, uploads, answers that are not JSON, waiting."""
import json
import threading
import time

import pytest
from fake_comfy import FakeComfy
from PIL import Image

from video_chain import comfy as C


@pytest.fixture
def server(tmp_path):
    s = FakeComfy(tmp_path / "comfy")
    (s.root / "main.py").write_text("")
    yield s
    s.close()


@pytest.fixture
def pic(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(p)
    return p


def test_a_download_is_retried_after_a_truncated_body(server, tmp_path):
    (server.root / "output").mkdir(exist_ok=True)
    (server.root / "output" / "x.bin").write_bytes(b"0123456789" * 1000)
    server.view_truncate = 1
    g = C.ComfyGenerator(server.url)
    dest = tmp_path / "got.bin"
    g._sleep = lambda s: None
    g._save({"filename": "x.bin", "subfolder": "", "type": "output"}, dest)
    assert dest.read_bytes() == b"0123456789" * 1000
    assert not list(tmp_path.glob("*.part*"))


def test_a_download_that_stays_truncated_is_a_comfy_error(server, tmp_path):
    (server.root / "output" / "x.bin").write_bytes(b"0123456789" * 1000)
    server.view_truncate = 10
    g = C.ComfyGenerator(server.url)
    g._sleep = lambda s: None
    dest = tmp_path / "got.bin"
    with pytest.raises(C.ComfyError):
        g._save({"filename": "x.bin", "subfolder": "", "type": "output"}, dest)
    assert not dest.exists() and not list(tmp_path.glob("*.part*"))


def test_a_page_that_is_not_json_is_a_comfy_error(server, pic):
    g = C.ComfyGenerator(server.url)
    server.junk = {"/upload/image", "/prompt"}
    with pytest.raises(C.ComfyError):
        g.upload(pic)
    with pytest.raises(C.ComfyError):
        g._queue({})


def test_upload_name_and_type_for_odd_suffixes(server, tmp_path):
    g = C.ComfyGenerator(server.url)
    odd = tmp_path / 'x.p"ng'
    odd.write_bytes(b"abc")
    assert g.upload(odd).endswith(".bin")
    jpg = tmp_path / "y.JPG"
    jpg.write_bytes(b"abcd")
    assert g.upload(jpg).endswith(".jpg")


def test_upload_is_streamed_not_one_bytes_object(server, pic, monkeypatch):
    g = C.ComfyGenerator(server.url)
    seen = {}
    real = g._request

    def spy(method, path, data=None, headers=None, timeout=30):
        if path == "/upload/image":
            seen["type"], seen["len"] = type(data), (headers or {}).get("Content-Length")
        return real(method, path, data, headers, timeout)
    monkeypatch.setattr(g, "_request", spy)
    g.upload(pic)
    assert seen["type"] is not bytes and seen["len"]


def test_a_shared_upload_is_sent_once_and_removed_by_the_last_user(server, pic):
    a = C.ComfyGenerator(server.url, comfy_dir=server.root)
    b = C.ComfyGenerator(server.url, comfy_dir=server.root)
    na, nb = a.upload(pic), b.upload(pic)
    assert na == nb
    assert sum(1 for c in server.calls if c == ("POST", "/upload/image")) == 1
    f = server.root / "input" / na
    a._drop_uploads([(na, pic)])
    assert f.exists()                              # b still uses it
    b._drop_uploads([(nb, pic)])
    assert not f.exists()


def test_a_url_that_is_not_http_is_refused():
    with pytest.raises(C.ComfyError, match="http"):
        C.ComfyGenerator("file:///etc/passwd")


def test_object_info_is_asked_once_per_node(server):
    g = C.ComfyGenerator(server.url)
    n = sum(1 for c in server.calls if c[1].startswith("/object_info/"))
    g._object_info("UNETLoader")
    g._object_info("UNETLoader")
    assert sum(1 for c in server.calls if c[1].startswith("/object_info/")) == n


class ScriptedWs:
    """recv() returns the scripted items: a dict (message), 'timeout', 'drop' (raises) or a callable."""
    def __init__(self, script, on_end=None):
        self.script, self.closed, self.on_end = list(script), False, on_end

    def settimeout(self, t):
        pass

    def recv(self):
        item = self.script.pop(0) if self.script else "timeout"
        if item == "timeout":
            time.sleep(0.01)
            raise TimeoutError("timed out")
        if item == "drop":
            raise ConnectionResetError("gone")
        if not self.script and self.on_end:
            self.on_end()                          # (ComfyUI writes the history when the job ends)
        return json.dumps(item)

    def close(self):
        self.closed = True


def test_a_failing_progress_callback_does_not_drop_the_websocket(server):
    def bad(v, m):
        raise RuntimeError("ui bug")
    g = C.ComfyGenerator(server.url, progress=bad, timeout=5)
    g.log = lambda *a: None
    ws = ScriptedWs([{"type": "progress", "data": {"value": 1, "max": 2, "prompt_id": "p1"}},
                     {"type": "progress", "data": {"value": 2, "max": 2, "prompt_id": "p1"}},
                     {"type": "execution_success", "data": {"prompt_id": "p1"}}], on_end=lambda: _done(server))
    g._connect_ws = lambda: pytest.fail("must not reconnect")
    assert g._wait("p1", ws)["status"]["completed"]
    assert not ws.script                           # every message was read from the same socket


def test_a_dropped_websocket_is_connected_again(server):
    g = C.ComfyGenerator(server.url, timeout=10)
    g._sleep = lambda s: time.sleep(0.01)
    ws2 = ScriptedWs([{"type": "execution_success", "data": {"prompt_id": "p1"}}], on_end=lambda: _done(server))
    calls = []
    g._connect_ws = lambda: calls.append(1) or ws2
    g._wait("p1", ScriptedWs(["drop"]))
    assert calls and not ws2.script


def test_polling_without_a_websocket_is_not_every_half_second(server):
    server.hold = True
    g = C.ComfyGenerator(server.url, timeout=30)
    g._connect_ws = lambda: None
    pid = g._queue({"1": {"class_type": "X", "inputs": {}}})
    t = threading.Thread(target=lambda: _quiet(g, pid), daemon=True)
    n0 = sum(1 for c in server.calls if c[1].startswith("/history/"))
    t.start()
    time.sleep(3.2)
    g.interrupt()
    t.join(5)
    polls = sum(1 for c in server.calls if c[1].startswith("/history/")) - n0
    assert polls <= 3


def _done(server):
    server.history["p1"] = {"outputs": {}, "status": {"status_str": "success", "completed": True}}


def _quiet(g, pid):
    try:
        g._wait(pid, None)
    except C.ComfyError:
        pass


def test_a_proxy_from_the_environment_is_not_used_for_the_lan(server, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    g = C.ComfyGenerator(server.url)
    assert g.stats is not None


def test_a_failed_unload_is_logged_and_does_not_count(server, monkeypatch):
    logs = []
    g = C.ComfyGenerator(server.url, log=logs.append)

    def boom(*a, **k):
        raise C.ComfyUnavailable("down")
    monkeypatch.setattr(g, "_json", boom)
    assert g.free() is False
    assert any("unload" in str(m) for m in logs)


def test_a_job_connects_the_socket_before_queueing_and_cancels_a_lost_queue(server):
    order = []
    g = C.ComfyGenerator(server.url)
    g._connect_ws = lambda: order.append("ws")
    real = g._queue

    def queue(wf):
        order.append("queue")
        return real(wf)
    g._queue = queue
    server.render = lambda wf: {}
    g._run_workflow({"1": {"class_type": "X", "inputs": {}}})
    assert order == ["ws", "queue"]
    # a queue request that fails after ComfyUI may have accepted it: the job is cancelled, current is cleared
    server.hold = True

    def lost(wf):
        g.current = "p9"
        raise C.ComfyUnavailable("answer lost")
    g._queue = lost
    with pytest.raises(C.ComfyUnavailable):
        g._run_workflow({})
    assert ("BODY", "/queue", {"delete": ["p9"]}) in server.calls and g.current is None


def test_refine_and_seedvr_save_through_save():
    import inspect
    for fn in (C._refine, C._seedvr_job):
        assert ".part.mp4" not in inspect.getsource(fn) and "_save(" in inspect.getsource(fn)
