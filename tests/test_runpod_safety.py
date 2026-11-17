"""RunPod safety: lost create answers, retryable close, fast failures (stub API, no network, no account)."""
import os
import stat
import subprocess

import pytest

from video_chain import runpod as RP


class StubApi:
    """RunpodApi stand-in. pods: records still on 'RunPod'; create_error: raised by create_pod (the pod is made first
    when create_lost is set: the answer got lost, the pod exists)."""

    def __init__(self, pods=(), create_error=None, create_lost=False, delete_fails=0, list_error=None):
        self.pods = {p["id"]: dict(p) for p in pods}
        self.create_error, self.create_lost, self.delete_fails, self.list_error = (
            create_error, create_lost, delete_fails, list_error)
        self.deleted, self.created, self.n = [], [], 0

    def gpus(self):
        raise RP.RunpodError("no catalog")      # (then every GPU type is tried)

    def create_pod(self, body):
        self.created.append(body)
        if self.create_lost:
            self.n += 1
            self.pods[f"lost{self.n}"] = {"id": f"lost{self.n}", "name": body["name"], "status": "RUNNING"}
        if self.create_error:
            raise self.create_error
        self.n += 1
        pod = {"id": f"p{self.n}", "name": body["name"], "status": "RUNNING"}
        self.pods[pod["id"]] = pod
        return pod

    def list_pods(self):
        if self.list_error:
            raise self.list_error
        return [dict(p) for p in self.pods.values()]

    def delete_pod(self, pid):
        if self.delete_fails:
            self.delete_fails -= 1
            raise RP.RunpodError("boom", 500)
        self.deleted.append(pid)
        self.pods.pop(pid, None)

    def get_pod(self, pid):
        if pid not in self.pods:
            raise RP.RunpodError("gone", 404)
        return self.pods[pid]


def session(api, tmp_path, **kw):
    return RP.RunpodSession(api, config=tmp_path / "cfg", log=lambda _t: None, **kw)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(RP.time, "sleep", lambda _s: None)


# ---- GEN-1
def test_lost_create_answer_removes_the_pod_that_was_made(tmp_path):
    api = StubApi(create_lost=True, create_error=RP.RunpodError("RunPod is not reachable (timed out)"))
    s = session(api, tmp_path)
    with pytest.raises(RP.RunpodError, match="not reachable"):
        s._create("k")
    assert api.deleted == ["lost1"] and not api.pods


def test_lost_create_only_removes_the_pod_of_this_request(tmp_path):
    other = {"id": "other", "name": f"{RP.POD_PREFIX}-ffffffff", "status": "RUNNING"}
    api = StubApi(pods=[other], create_lost=True, create_error=RP.RunpodError("RunPod is not reachable (x)"))
    s = session(api, tmp_path)
    with pytest.raises(RP.RunpodError):
        s._create("k")
    assert api.deleted == ["lost1"] and "other" in api.pods


def test_lost_create_cleanup_failing_keeps_the_original_error(tmp_path):
    api = StubApi(create_lost=True, create_error=RP.RunpodError("RunPod is not reachable (x)"),
                  list_error=RP.RunpodError("RunPod is not reachable (y)"))
    s = session(api, tmp_path)
    with pytest.raises(RP.RunpodError, match=r"not reachable \(x\)"):
        s._create("k")


def test_start_removes_leftover_pods_first(tmp_path):
    old = {"id": "old1", "name": f"{RP.POD_PREFIX}-12345678", "status": "RUNNING"}
    foreign = {"id": "x", "name": "someone-else", "status": "RUNNING"}
    api = StubApi(pods=[old, foreign])
    s = session(api, tmp_path)
    s.setup_script = __file__                      # (exists)

    def stop(_pub):
        raise RP.RunpodError("stop here")
    s._create = stop
    s.ensure = None
    import video_chain.runpod as mod
    mod_ensure = mod.ensure_key
    mod.ensure_key = lambda cfg: (tmp_path / "k", "pub")
    try:
        with pytest.raises(RP.RunpodError, match="stop here"):
            s.start()
    finally:
        mod.ensure_key = mod_ensure
    assert api.deleted == ["old1"] and "x" in api.pods


def test_start_goes_on_when_the_leftover_check_fails(tmp_path):
    api = StubApi(list_error=RP.RunpodError("RunPod is not reachable (y)"))
    s = session(api, tmp_path)
    s._remove_leftovers()                          # must not raise
    assert api.deleted == []


# ---- GEN-2
def test_close_can_be_retried_after_a_failed_termination(tmp_path):
    api = StubApi()
    s = session(api, tmp_path)
    s._create("k")
    api.delete_fails = 100
    with pytest.raises(RP.RunpodError, match="could not terminate"):
        s.close()
    assert os.path.isdir(s._tmp.name)              # the session's files stay while the pod may be up
    api.delete_fails = 0
    s.close()                                      # the retry really ends it
    assert api.deleted == [s.pod_id] and not api.pods
    s.close()                                      # now a no-op
    assert api.deleted == [s.pod_id]


# ---- GEN-3
def test_interrupt_during_start_terminates_the_pod(tmp_path):
    api = StubApi()
    s = session(api, tmp_path)
    import video_chain.runpod as mod
    orig = mod.ensure_key
    mod.ensure_key = lambda cfg: (tmp_path / "k", "pub")
    s.setup_script = __file__

    def wait():
        raise KeyboardInterrupt
    s._wait_running = wait
    try:
        with pytest.raises(KeyboardInterrupt):
            s.start()
    finally:
        mod.ensure_key = orig
    assert api.deleted == [s.pod_id] or not api.pods


# ---- GEN-4
@pytest.mark.parametrize("status,text", [(403, "forbidden"), (401, "bad key"), (400, "invalid volume path")])
def test_real_refusals_are_not_taken_for_no_capacity(tmp_path, status, text):
    api = StubApi(create_error=RP.RunpodError(f"HTTP {status} {text}", status))
    s = session(api, tmp_path, gpu=RP.GPUS[0][0])
    with pytest.raises(RP.RunpodError, match=text):
        s._create("k")
    assert len(api.created) == 1


def test_400_about_capacity_tries_the_next_gpu(tmp_path):
    api = StubApi(create_error=RP.RunpodError("HTTP 400 no instances available", 400))
    s = session(api, tmp_path, gpu=RP.GPUS[0][0])
    with pytest.raises(RP.RunpodError, match="none of these GPUs|has none"):
        s._create("k")
    assert len(api.created) == len(RP.GPUS)


# ---- GEN-5
def test_non_json_answer_is_a_runpod_error(monkeypatch):
    import io

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(RP.urllib.request, "urlopen", lambda *a, **k: R(b"<html>proxy</html>"))
    with pytest.raises(RP.RunpodError, match="not JSON"):
        RP.RunpodApi("k", sleep=lambda _s: None).call("GET", "/pods")


def test_missing_ssh_keygen_is_said_plainly(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("ssh-keygen")
    monkeypatch.setattr(RP.subprocess, "run", boom)
    with pytest.raises(RP.RunpodError, match="OpenSSH"):
        RP.ensure_key(tmp_path / "cfg")


def test_deleted_public_key_is_remade(tmp_path):
    key, pub = RP.ensure_key(tmp_path / "cfg")
    key.with_suffix(".pub").unlink()
    assert RP.ensure_key(tmp_path / "cfg") == (key, pub)


def test_missing_ssh_binary_is_a_runpod_error(tmp_path):
    s = session(StubApi(), tmp_path, ssh="/nonexistent/ssh")
    s.host = ("h", 22)
    with pytest.raises(RP.RunpodError, match="OpenSSH"):
        s.run("true")


# ---- GEN-6
def fake_ssh(tmp_path, body):
    p = tmp_path / "ssh"
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def setup_session(tmp_path, body, **kw):
    s = session(StubApi(), tmp_path, ssh=fake_ssh(tmp_path, body), **kw)
    s.host = ("h", 22)
    s._upload_nodes = lambda: None
    s.setup_script = tmp_path / "setup.sh"
    s.setup_script.write_text("echo hi\n" * 20000)      # more than a pipe buffer: a closed ssh breaks the pipe
    return s


def test_setup_ssh_dying_at_once_reports_its_message(tmp_path):
    s = setup_session(tmp_path, "echo 'Permission denied (publickey)'; exit 255")
    with pytest.raises(RP.RunpodError, match="Permission denied"):
        s._setup()


def test_setup_timeout_is_named(tmp_path):
    s = setup_session(tmp_path, "cat >/dev/null; exec sleep 30", setup_timeout=0.5)
    with pytest.raises(RP.RunpodError, match="timed out"):
        s._setup()


# ---- GEN-9
def test_cli_down_goes_on_after_a_failing_pod(monkeypatch, capsys):
    pods = [{"id": f"p{i}", "name": f"{RP.POD_PREFIX}-{i}", "status": "RUNNING"} for i in range(3)]
    api = StubApi(pods=pods, delete_fails=0)
    orig = api.delete_pod

    def delete(pid):
        if pid == "p0":
            raise RP.RunpodError("boom", 500)
        orig(pid)
    api.delete_pod = delete
    monkeypatch.setattr(RP, "RunpodApi", lambda key: api)
    monkeypatch.setattr(RP, "api_key", lambda: "k")
    assert RP.main(["down"]) == 1
    assert api.deleted == ["p1", "p2"]


def test_cli_gpus_without_id_and_missing_key(monkeypatch, capsys):
    api = StubApi()
    api.gpus = lambda: [{"memory": 24}, {"id": "A", "memory": 24}]
    monkeypatch.setattr(RP, "RunpodApi", lambda key: api)
    monkeypatch.setattr(RP, "api_key", lambda: "k")
    assert RP.main(["gpus"]) == 0

    def nokey():
        raise RP.RunpodError("no RunPod API key")
    monkeypatch.setattr(RP, "api_key", nokey)
    assert RP.main(["status"]) == 1


def test_a_chatty_tunnel_does_not_block_on_its_stderr(tmp_path):
    """The tunnel's stderr goes to a file: ssh writing far more than a pipe holds (64 KB) while the tunnel lives
    must still reach "connected". The fake ssh answers /system_stats on the forwarded port itself."""
    import stat
    import sys
    fake = tmp_path / "fakessh"
    fake.write_text(f"""#!{sys.executable}
import sys, http.server
port = int(sys.argv[sys.argv.index("-L") + 1].split(":")[1])
sys.stderr.write("x" * 400000)
sys.stderr.flush()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{{}}")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
""")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    s = session(StubApi(), tmp_path, ssh=str(fake))
    s.host = ("127.0.0.1", 22)
    s.key = tmp_path / "k"
    try:
        s._open_tunnel_once()
        assert s.url and s.url.startswith("http://127.0.0.1:")
    finally:
        if s.tunnel:
            s.tunnel.kill()
            s.tunnel.wait()
