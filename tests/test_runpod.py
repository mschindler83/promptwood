"""Cloud upscaling on RunPod against a fake RunPod API and a fake ComfyUI (no account, no pod, no network)."""
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from video_chain import chain
from video_chain import comfy as C
from video_chain import runpod as RP


class FakeRunpod:
    """The few v2 endpoints the app uses. no_capacity: GPU ids answered with 400; pods: id -> pod record."""

    def __init__(self, no_capacity=(), keep_running=False, busy_first=0):
        self.no_capacity, self.keep_running, self.busy = set(no_capacity), keep_running, busy_first
        self.pods, self.calls, self.n = {}, [], 0
        self.catalog = None                       # GPU id -> data centres with stock (None: no catalog answer)
        self.host_keys = "ssh-ed25519 AAAAC3Nz host\nssh-rsa AAAAB3 host\n"
        self.ssh = ("203.0.113.5", 40022)
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body=None, headers=()):
                data = b"" if body is None else json.dumps(body).encode()
                self.send_response(code)
                for k, v in headers:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _route(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                fake.calls.append((method, self.path, body, self.headers.get("Authorization")))
                if fake.busy > 0:
                    fake.busy -= 1
                    return self._send(429, {"title": "Too Many Requests", "status": 429, "detail": "slow down"},
                                      [("Retry-After", "0")])
                if self.headers.get("Authorization") != "Bearer k-123":
                    return self._send(401, {"title": "Unauthorized", "status": 401, "detail": "bad key"})
                p = self.path.split("?")[0]
                if method == "POST" and p == "/v2/pods":
                    if body["gpu"]["id"] in fake.no_capacity:
                        return self._send(400, {"title": "Bad Request", "status": 400,
                                                "detail": "no instances available"})
                    fake.n += 1
                    pid = f"pod{fake.n}"
                    fake.pods[pid] = {"id": pid, "name": body["name"], "status": "RUNNING", "cost": 2.5,
                                      "dataCenterId": "EU-RO-1", "gpu": {"id": body["gpu"]["id"], "memory": 120},
                                      "ssh": {"direct": {"host": fake.ssh[0], "port": fake.ssh[1], "username": "root"}}}
                    return self._send(201, fake.pods[pid])
                if method == "GET" and p == "/v2/pods":
                    if "cursor" not in self.path:
                        return self._send(200, {"pods": [fake.pods[k] for k in list(fake.pods)[:1]],
                                                "pagination": {"nextCursor": "c2", "hasNextPage": True}})
                    return self._send(200, {"pods": [fake.pods[k] for k in list(fake.pods)[1:]],
                                            "pagination": {"nextCursor": None}})
                if p.startswith("/v2/pods/"):
                    pid = p.split("/")[3]
                    if pid not in fake.pods:
                        return self._send(404, {"title": "Not Found", "status": 404, "detail": "pod not found"})
                    if method == "GET":
                        return self._send(200, fake.pods[pid])
                    if method == "DELETE":
                        if not fake.keep_running:
                            del fake.pods[pid]
                        return self._send(204)
                    if method == "POST" and p.endswith("/exec"):
                        return self._send(200, {"stdout": fake.host_keys, "stderr": "", "exitCode": 0,
                                                "timedOut": False})
                if method == "GET" and p == "/v2/catalog/gpus" and fake.catalog is not None:
                    assert "include=AVAILABILITY" in self.path and "cloud=SECURE" in self.path
                    assert f"minCudaVersion={RP.MIN_CUDA}" in self.path               # stock that can run it
                    return self._send(200, {"gpus": [
                        {"id": g, "memory": 24, "price": {"secure": 1.0}, "availability": "LOW" if dcs else "NONE",
                         "dataCenters": [{"id": dc, "availability": "LOW"} for dc in dcs]}
                        for g, dcs in fake.catalog.items()]})
                if method == "GET" and p == "/v2/global-volumes/gv1":
                    return self._send(200, {"globalVolume": {"id": "gv1", "name": "video-chain"}})
                if method == "GET" and p == "/v2/network-volumes/vol9":
                    return self._send(200, {"id": "vol9", "dataCenter": "EU-CZ-1", "size": 40})
                return self._send(404, {"title": "Not Found", "status": 404, "detail": p})

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

            def do_DELETE(self):
                self._route("DELETE")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v2"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake():
    f = FakeRunpod()
    yield f
    f.close()


def api(fake):
    return RP.RunpodApi("k-123", base=fake.url, sleep=lambda _s: None)


def test_pod_request_is_secure_cloud_ssh_only_with_our_key():
    """Secure Cloud only, one GPU, nothing exposed but SSH (no HTTP proxy port, no Jupyter), this app's key; a
    network volume at /workspace in its data centre."""
    b = RP.pod_body(name="video-chain-upscale-x", gpu="NVIDIA L40S", public_key="ssh-ed25519 AAA app")
    assert b["cloud"] == "SECURE" and b["ports"] == ["22/tcp"] and b["startJupyter"] is False
    assert b["env"] == {"PUBLIC_KEY": "ssh-ed25519 AAA app"} and b["gpu"]["count"] == 1
    assert b["gpu"]["minCudaVersion"] == RP.MIN_CUDA == "13.2" and "mounts" not in b and "dataCenterIds" not in b
    v = RP.pod_body(name="n", gpu="g", public_key="k", volume="vol9", data_center="EU-CZ-1")
    assert v["mounts"] == {"network": [{"volumeId": "vol9", "path": "/workspace"}]}
    assert v["dataCenterIds"] == ["EU-CZ-1"] and v["disk"] == b["disk"] == 80   # (new weights go on the disk)


def test_api_calls_retry_on_429_and_page_through_pods(fake):
    a = api(fake)
    fake.busy = 2                                              # rate-limited twice, then answered
    pod = a.create_pod(RP.pod_body(name="video-chain-upscale-a", gpu="NVIDIA L40S", public_key="k"))
    assert pod["id"] == "pod1" and fake.calls[-1][3] == "Bearer k-123"
    a.create_pod(RP.pod_body(name="someone-else", gpu="NVIDIA L40S", public_key="k"))
    assert [p["id"] for p in a.list_pods()] == ["pod1", "pod2"]          # both pages
    assert [p["id"] for p in RP.leftover_pods(a)] == ["pod1"]             # only this app's pods
    a.delete_pod("pod1")
    a.delete_pod("pod1")                                                  # gone already: fine
    with pytest.raises(RP.RunpodError, match="bad key") as e:
        RP.RunpodApi("wrong", base=fake.url).get_pod("pod2")
    assert e.value.status == 401
    with pytest.raises(RP.RunpodError, match="no RunPod API key"):
        RP.RunpodApi("")


def test_create_not_retried_after_a_server_error_so_no_second_pod(fake, monkeypatch):
    """A create is retried after a 429 only (refused before anything was done), never after a 5xx or a lost
    answer: it may have made a pod, and a second try would leave two billed pods."""
    calls = []

    def boom(req, timeout):
        calls.append(req.full_url)
        raise RP.urllib.error.URLError("connection reset")
    monkeypatch.setattr(RP.net, "urlopen", boom)   # (the fake RunPod is a local address: no proxy, via net)
    with pytest.raises(RP.RunpodError, match="not reachable"):
        api(fake).create_pod(RP.pod_body(name="video-chain-upscale-a", gpu="g", public_key="k"))
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(RP.RunpodError):
        api(fake).get_pod("x")                                            # a read is retried
    assert len(calls) == 4


def test_ssh_details_and_host_keys():
    assert RP.direct_ssh({"status": "STARTING", "ssh": {"direct": {"host": "h", "port": 1}}}) is None
    assert RP.direct_ssh({"status": "RUNNING", "ssh": {"direct": None}}) is None
    assert RP.direct_ssh({"status": "RUNNING", "ssh": {"direct": {"host": "1.2.3.4", "port": 4022}}}) == ("1.2.3.4",
                                                                                                         4022)
    assert RP.direct_ssh({"status": "RUNNING", "runtime": {"ports": [
        {"private": 8188, "public": 1, "ip": "x"}, {"private": 22, "public": 3022, "ip": "5.6.7.8"}]}}) == ("5.6.7.8",
                                                                                                          3022)
    kh = RP.known_hosts_line("1.2.3.4", 4022, "ssh-ed25519 AAAA root@x\ngarbage\necdsa-sha2-nistp256 BBBB\n")
    assert kh == "[1.2.3.4]:4022 ssh-ed25519 AAAA\n[1.2.3.4]:4022 ecdsa-sha2-nistp256 BBBB\n"
    assert RP.known_hosts_line("h", 1, "") == ""
    assert RP.models_for("seedvr2_3b") == "3b"
    with pytest.raises(RP.RunpodError):
        RP.models_for("refine")


@pytest.mark.skipif(not shutil.which("ssh-keygen"), reason="no ssh-keygen")
def test_own_ssh_key_made_once_private(tmp_path):
    key, pub = RP.ensure_key(tmp_path / "rp")
    assert pub.startswith("ssh-ed25519 ") and pub.endswith("video-chain-runpod")
    assert oct(key.stat().st_mode & 0o777) == "0o600" and oct((tmp_path / "rp").stat().st_mode & 0o777) == "0o700"
    assert RP.ensure_key(tmp_path / "rp") == (key, pub)                 # the same key next time


def session(fake, tmp_path, **kw):
    notes = []
    s = RP.RunpodSession(api(fake), note=notes.append, log=lambda _t: None, config=tmp_path / "cfg", **kw)
    s.notes = notes
    return s


def test_takes_the_next_gpu_without_capacity_and_the_volume_data_centre(fake, tmp_path):
    fake.no_capacity = {RP.GPUS[0][0], RP.GPUS[1][0]}
    s = session(fake, tmp_path, gpu=RP.GPUS[0][0], volume="vol9")
    s._create("ssh-ed25519 AAA app")
    creates = [c[2] for c in fake.calls if c[:2] == ("POST", "/v2/pods")]
    assert [c["gpu"]["id"] for c in creates] == [g for g, _ in RP.GPUS[:3]]
    assert all(c["dataCenterIds"] == ["EU-CZ-1"] and c["cloud"] == "SECURE" for c in creates)
    assert s.pod_id == "pod1" and s.pod["name"].startswith(RP.POD_PREFIX + "-")
    s.close()
    assert not fake.pods                                                  # terminated


def test_no_fallback_and_nothing_free(fake, tmp_path):
    fake.no_capacity = {g for g, _ in RP.GPUS}
    s = session(fake, tmp_path, gpu=RP.GPUS[2][0], fallback=False)
    with pytest.raises(RP.RunpodError, match="none of these GPUs"):
        s._create("k")
    assert [c[2]["gpu"]["id"] for c in fake.calls if c[:2] == ("POST", "/v2/pods")] == [RP.GPUS[2][0]]
    s.close()                                                             # (nothing made: nothing to end)


def test_host_key_read_through_the_api(fake, tmp_path):
    s = session(fake, tmp_path)
    s._create("k")
    s.host = s._wait_running()
    assert s.host == ("203.0.113.5", 40022)
    s._trust_host()
    assert s.known_hosts.read_text() == ("[203.0.113.5]:40022 ssh-ed25519 AAAAC3Nz\n"
                                         "[203.0.113.5]:40022 ssh-rsa AAAAB3\n")
    args = s.ssh_args("-N")
    assert "StrictHostKeyChecking=yes" in args and f"UserKnownHostsFile={s.known_hosts}" in args
    assert args[-1] == "root@203.0.113.5" and "-N" in args
    s.close()
    assert not s.known_hosts.exists()                                     # the session's files are gone


def test_close_terminates_checks_and_reports_a_pod_left(tmp_path):
    f = FakeRunpod(keep_running=True)                                     # DELETE answered, the pod stays
    try:
        s = session(f, tmp_path)
        s._create("k")
        import video_chain.runpod as mod
        sleeps = []
        orig = mod.time.sleep
        mod.time.sleep = sleeps.append
        try:
            with pytest.raises(RP.RunpodError, match="could not terminate pod pod1"):
                s.close()
        finally:
            mod.time.sleep = orig
        assert sum(1 for c in f.calls if c[0] == "DELETE") == 5
        f.keep_running = False
        s.close()                                                         # a failed close is tried again
        assert not f.pods
        s.close()                                                         # and then idempotent
    finally:
        f.close()


class FakeSession:
    """What CloudUpscaler needs of a session: the tunnel URL, remote removal, the pod's RAM."""

    def __init__(self, url, ram):
        self.url, self.ram, self.removed = url, ram, []

    def remove_remote(self, kind, subfolder, filename):
        self.removed.append((kind, subfolder, filename))

    def ram_free(self):
        return self.ram


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_cloud_upscaler_removes_everything_on_the_pod_and_uses_its_ram(tmp_path):
    """SeedVR2 through the pod's ComfyUI: the same workflow as at home; every upload and result removed on the pod
    after the job; the frame budget from the pod's own RAM (ComfyUI in a container sees the host's); clips and
    the refine are refused."""
    from fake_comfy import FakeComfy
    server = FakeComfy(tmp_path / "comfy")
    try:
        draft = tmp_path / "d.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x36:r=24:d=4", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(draft)], check=True)
        server.ram_free = 500 * 2**30                                    # the host's RAM, as ComfyUI reports it
        pod_ram = C.SEEDVR_RAM_RESERVE + C.SEEDVR_FRAME_COPIES * 128 * 72 * 3 * 4 * 33   # the pod: 33 frames
        fs = FakeSession(server.url, pod_ram)
        gen = RP.CloudUpscaler(fs, log=lambda _m: None)
        out = tmp_path / "up.mp4"
        from video_chain.comfy import seedvr as SV      # (the module that reads it)
        old_min = SV.SEEDVR_RAM_MIN
        SV.SEEDVR_RAM_MIN = 0
        try:
            gen.seedvr(draft, out, (128, 72))
        finally:
            SV.SEEDVR_RAM_MIN = old_min
        info = chain.probe(out)
        assert (info["width"], info["height"], info["frames"], info["audio"]) == (128, 72, 96, True)
        assert len(server.seedvr_jobs) == 4                               # pieces from the pod's RAM, not the host's
        kinds = [k for k, _, _ in fs.removed]
        assert kinds.count("input") == 4 and kinds.count("output") == 4
        assert all(sub == C.SUBFOLDER for _, sub, _ in fs.removed)
        with pytest.raises(C.ComfyError, match="only upscales"):
            gen.generate("x", None, out, tmp_path / "f.png", 1, {})
        with pytest.raises(C.ComfyError, match="SeedVR2"):
            gen.refine()
        assert gen.missing_loras(["a.safetensors"]) == []
    finally:
        server.close()


def test_remote_removal_only_inside_the_ram_folder(tmp_path, fake):
    s = session(fake, tmp_path)
    ran = []
    s.run = lambda cmd, timeout=60, stdin=None: ran.append(cmd)
    s.remove_remote("output", C.SUBFOLDER, "seedvr_abc_00001_.mp4")
    s.remove_remote("output", "elsewhere", "x.mp4")                       # not its subfolder
    s.remove_remote("output", C.SUBFOLDER, "../../etc/passwd")            # no paths
    s.remove_remote("temp", C.SUBFOLDER, "x.mp4")                         # not input / output
    assert ran == [f"rm -f -- /dev/shm/vc-io/output/{C.SUBFOLDER}/seedvr_abc_00001_.mp4"]
    s.close()


def test_pod_setup_script_keeps_private_files_in_ram():
    """The script parses, puts every private folder (uploads, results, temp, database, log, TMPDIR) in /dev/shm,
    refuses when that is no RAM file system, and ComfyUI listens on 127.0.0.1 only."""
    script = RP.SETUP_SCRIPT
    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert 'IO="${IO_DIR:-/dev/shm/vc-io}"' in text and "/root/vc-io" not in text
    for flag in ("--input-directory \"$IO/input\"", "--output-directory \"$IO/output\"",
                 "--temp-directory \"$IO/temp\"", "--user-directory \"$IO/user\"",
                 "--database-url \"sqlite:///$IO/user/comfyui.db\"", "--listen 127.0.0.1"):
        assert flag in text
    assert 'export TMPDIR="$IO/tmp"' in text and '!= "tmpfs"' in text
    assert "0.0.0.0" not in text and RP.COMFY_REF.startswith("v")


@pytest.mark.skipif(sys.platform != "linux", reason="needs /proc")
def test_watchdog_sees_ssh_connections(tmp_path):
    """The watchdog's /proc/net/tcp test: an established connection to local port 22 counts, others do not."""
    text = RP.SETUP_SCRIPT.read_text()
    line = next(x for x in text.splitlines() if x.startswith("ssh_up()"))
    awk = line + "\n" + text.splitlines()[text.splitlines().index(line) + 1]
    head = "  sl  local_address rem_address   st\n"
    for rows, up in ((["0: 0100007F:0016 0200A8C0:D431 01"], True),       # :22 established
                     (["0: 0100007F:0016 00000000:0000 0A"], False),      # :22 listening only
                     (["0: 0100007F:1F90 0200A8C0:D431 01"], False)):     # another port
        (tmp_path / "tcp").write_text(head + "\n".join(rows) + "\n")
        (tmp_path / "tcp6").write_text(head)
        body = awk.replace("/proc/net/tcp /proc/net/tcp6", f"{tmp_path}/tcp {tmp_path}/tcp6")
        r = subprocess.run(["bash", "-c", body + "\nssh_up"], check=False)
        assert (r.returncode == 0) is up


def test_cloud_worker_terminates_its_pod_when_told_to_quit(monkeypatch, tmp_path):
    """--serve --cloud runpod: the pod is made when the worker starts and terminated when it quits."""
    from video_chain import __main__ as M
    events, closed = [], []

    class S:
        def __init__(self, api, **kw):
            self.kw, self.url = kw, "http://127.0.0.1:1"

        def start(self):
            return self.url

        def close(self):
            closed.append(True)

    class U:
        upscale_only = True

        def __init__(self, session, progress=None):
            self.session = session

        def free(self):
            pass

    monkeypatch.setattr(RP, "RunpodSession", S)
    monkeypatch.setattr(RP, "CloudUpscaler", U)
    monkeypatch.setattr(RP, "api_key", lambda: "k-123")
    r, w = os.pipe()
    os.write(w, b'{"cmd": "run", "tree": "x", "targets": ["n1"]}\n{"cmd": "quit"}\n')
    os.close(w)
    a = M.build_parser().parse_args(["--serve", "--idle-minutes", "0", "--cloud", "runpod", "--runpod-models", "3b",
                                     "--runpod-gpu", "NVIDIA L40S", "--runpod-no-fallback"])
    monkeypatch.setattr(sys, "stdin", os.fdopen(r))
    assert M._main_serve(a, events.append) == 0
    assert closed == [True]
    assert [e["e"] for e in events][:2] == ["loading", "loaded"] and events[0]["what"] == "runpod"
    assert any(e["e"] == "error" and "only upscales" in e["msg"] for e in events)    # no clips on the pod
    a2 = M.build_parser().parse_args(["--cloud", "runpod"])
    assert M._main(a2, events.append) == 2                                # worker mode only


def test_cloud_worker_failure_still_terminates(monkeypatch):
    from video_chain import __main__ as M
    events, closed = [], []

    class S:
        def __init__(self, api, **kw):
            pass

        def start(self):
            raise RP.RunpodError("preparing the pod failed (exit 5): /dev/shm has only 1 GB")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(RP, "RunpodSession", S)
    monkeypatch.setattr(RP, "api_key", lambda: "k-123")
    a = M.build_parser().parse_args(["--serve", "--cloud", "runpod"])
    with pytest.raises(SystemExit) as e:
        M._cloud_generator(a, events.append, {}, {})
    assert e.value.code == M.CLOUD_FAILED and closed == [True]
    assert any("/dev/shm has only 1 GB" in ev.get("msg", "") for ev in events)


def test_api_key_from_env_first(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", " env-key ")
    assert RP.api_key() == "env-key"


@pytest.mark.skipif(not (Path("/usr/sbin/sshd").is_file() and shutil.which("ssh") and shutil.which("ssh-keygen")),
                    reason="needs OpenSSH (ssh, sshd, ssh-keygen)")
def test_whole_session_over_real_ssh(tmp_path):
    """start() and close() end to end with a real OpenSSH on 127.0.0.1 (a throwaway sshd run as oneself, its own
    host key) standing in for the pod: the host key comes from the API and is enforced, the setup script's notes
    reach the app, the tunnel reaches the "pod's" ComfyUI, remote files are removed, the pod is terminated."""
    import getpass
    import socket
    import time
    from http.server import BaseHTTPRequestHandler as BH

    sshd_bin = shutil.which("sshd") or ("/usr/sbin/sshd" if os.path.exists("/usr/sbin/sshd") else None)
    if not (sshd_bin and shutil.which("ssh") and shutil.which("ssh-keygen")):
        pytest.skip("needs OpenSSH (sshd, ssh, ssh-keygen) on this machine")
    d = tmp_path / "sshd"
    d.mkdir()
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / "host")], check=True)
    _key, pub = RP.ensure_key(tmp_path / "cfg")
    (d / "authorized_keys").write_text(pub + "\n")
    with socket.socket() as s_:
        s_.bind(("127.0.0.1", 0))
        port = s_.getsockname()[1]
    (d / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {d}/host\nAuthorizedKeysFile {d}/authorized_keys\n"
        f"PasswordAuthentication no\nKbdInteractiveAuthentication no\nUsePAM no\nStrictModes no\n"
        f"PidFile {d}/sshd.pid\nAllowTcpForwarding yes\n")
    sshd = subprocess.Popen([sshd_bin, "-D", "-e", "-f", str(d / "sshd_config")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    class Comfy(BH):                                          # the "pod's" ComfyUI, on its own port
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = b'{"system": {"ram_free": 1}}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    comfy = ThreadingHTTPServer(("127.0.0.1", 0), Comfy)
    threading.Thread(target=comfy.serve_forever, daemon=True).start()
    f = FakeRunpod()
    f.ssh = ("127.0.0.1", port)
    f.host_keys = (d / "host.pub").read_text()
    io = tmp_path / "io"
    (io / "output" / C.SUBFOLDER).mkdir(parents=True)
    victim = io / "output" / C.SUBFOLDER / "seedvr_x_00001_.mp4"
    victim.write_bytes(b"result")
    setup = tmp_path / "setup.sh"
    setup.write_text('#!/usr/bin/env bash\nset -e\necho "@@ installing ComfyUI $COMFY_REF"\n'
                     'echo "models: $MODELS idle: $IDLE_MIN volume: [$VOLUME_PATH]"\necho "@@ ready"\n')
    try:
        for _ in range(200):                                  # sshd up
            with socket.socket() as c:
                if c.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        else:
            pytest.fail("the throwaway sshd did not open its port")
        notes, logs = [], []
        s = RP.RunpodSession(api(f), note=notes.append, log=logs.append, config=tmp_path / "cfg", models="3b")
        s.user, s.remote_port, s.setup_script = getpass.getuser(), comfy.server_address[1], setup
        s.remote_code = str(tmp_path / "remote-code")
        url = s.start()
        assert url.startswith("http://127.0.0.1:") and int(url.rsplit(":", 1)[1]) != comfy.server_address[1]
        import urllib.request
        assert json.loads(urllib.request.urlopen(url + "/system_stats", timeout=5).read()) == {
            "system": {"ram_free": 1}}                         # through the tunnel
        assert f"RunPod: installing ComfyUI {RP.COMFY_REF}" in notes and "RunPod: ready" in notes
        assert "models: 3b idle: 20 volume: []" in logs                      # (no volume: no path)
        assert (tmp_path / "remote-code" / "nodes" / "video_chain_scale" / "__init__.py").read_bytes() == \
            RP.SCALE_NODE_FILE.read_bytes()                                   # the node went over SSH first
        assert s.run("echo hi").stdout == b"hi\n"
        old = RP.REMOTE_IO                                     # remote removal, here inside tmp_path
        RP.REMOTE_IO = RP.PurePosixPath(str(io))
        try:
            s.remove_remote("output", C.SUBFOLDER, victim.name)
        finally:
            RP.REMOTE_IO = old
        assert not victim.exists()
        assert s.ram_free() >= 0
        s.close()
        assert s.tunnel.poll() is not None and not f.pods     # tunnel ended, pod terminated
        # a host whose key is not the one RunPod reported is refused
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / "other2")], check=True)
        f.host_keys = (d / "other2.pub").read_text()
        s2 = RP.RunpodSession(api(f), config=tmp_path / "cfg")
        s2.user, s2.setup_script = getpass.getuser(), setup
        s2._create(pub)
        s2.host = s2._wait_running()
        s2._trust_host()
        r = s2.run("true", timeout=30)
        assert r.returncode != 0 and b"host key" in r.stderr.lower()
        s2.close()
    finally:
        sshd.terminate()
        sshd.wait(5)
        comfy.shutdown()
        f.close()


def creates(fake):
    return [c[2] for c in fake.calls if c[:2] == ("POST", "/v2/pods")]


def test_volume_data_centre_without_free_gpus_is_said_plainly(fake, tmp_path):
    """The volume's data centre has none of the GPUs free (as EUR-IS-3 on 2026-11-13): nothing is asked for in
    vain, and the message names the data centre and where GPUs are free."""
    fake.catalog = {g: [] for g, _ in RP.GPUS}
    fake.catalog["NVIDIA GeForce RTX 4090"] = ["EUR-IS-2"]
    fake.catalog["NVIDIA GeForce RTX 5090"] = ["EU-RO-1"]
    s = session(fake, tmp_path, volume="vol9")
    with pytest.raises(RP.RunpodError) as e:
        s._create("k")
    msg = str(e.value)
    assert "EU-CZ-1" in msg and "RTX 4090 in EUR-IS-2" in msg and "RTX 5090 in EU-RO-1" in msg
    assert "without the volume" in msg and creates(fake) == []
    s.close()


def test_without_the_volume_when_its_data_centre_has_none(fake, tmp_path):
    fake.catalog = {g: [] for g, _ in RP.GPUS}
    fake.catalog["NVIDIA GeForce RTX 4090"] = ["EUR-IS-2"]
    s = session(fake, tmp_path, volume="vol9", without_volume=True)
    s._create("k")
    assert [c["gpu"]["id"] for c in creates(fake)] == ["NVIDIA GeForce RTX 4090"]     # only what is free
    body = creates(fake)[0]
    assert "mounts" not in body and "dataCenterIds" not in body and body["disk"] == 80
    assert s.volume is None and any("without the volume" in n for n in s.notes)
    s.close()


def test_only_gpus_free_in_the_volume_data_centre_are_asked_for(fake, tmp_path):
    fake.catalog = {g: ["US-GA-2"] for g, _ in RP.GPUS}
    fake.catalog["NVIDIA L40S"] = ["EU-CZ-1", "US-GA-2"]
    s = session(fake, tmp_path, volume="vol9", without_volume=True)
    s._create("k")
    assert [(c["gpu"]["id"], c["dataCenterIds"]) for c in creates(fake)] == [("NVIDIA L40S", ["EU-CZ-1"])]
    assert s.volume == "vol9" and creates(fake)[0]["mounts"]["network"][0]["volumeId"] == "vol9"
    assert creates(fake)[0]["gpu"]["minRamPerGpu"] == RP.MIN_RAM_PER_GPU == 32
    s.close()


def test_chosen_gpu_not_free_without_fallback(fake, tmp_path):
    fake.catalog = {g: ["EU-RO-1"] for g, _ in RP.GPUS}
    fake.catalog[RP.GPUS[0][0]] = []
    s = session(fake, tmp_path, gpu=RP.GPUS[0][0], fallback=False)
    with pytest.raises(RP.RunpodError, match="allow other GPUs") as e:
        s._create("k")
    assert "free elsewhere now" in str(e.value) and creates(fake) == []
    s.close()


def test_global_volume_mounts_anywhere(fake, tmp_path):
    """A global volume has no data centre: it is mounted as `global`, and the GPU may be wherever one is free."""
    b = RP.pod_body(name="n", gpu="g", public_key="k", volume="gv1", volume_kind="global")
    assert b["mounts"] == {"global": [{"volumeId": "gv1", "path": "/workspace"}]} and "dataCenterIds" not in b
    with pytest.raises(ValueError):
        RP.pod_body(name="n", gpu="g", public_key="k", volume="x", volume_kind="persistent")
    assert RP.RunpodApi("k-123", base=fake.url).volume("gv1")[0] == "global"
    assert RP.RunpodApi("k-123", base=fake.url).volume("vol9")[0] == "network"
    with pytest.raises(RP.RunpodError, match="no network or global volume"):
        RP.RunpodApi("k-123", base=fake.url).volume("nope")
    fake.catalog = {g: [] for g, _ in RP.GPUS}
    fake.catalog["NVIDIA H100 80GB HBM3"] = ["EU-FR-1"]
    s = session(fake, tmp_path, volume="gv1", without_volume=True)
    s._create("k")
    body = creates(fake)[0]
    assert [c["gpu"]["id"] for c in creates(fake)] == ["NVIDIA H100 80GB HBM3"]
    assert body["mounts"] == {"global": [{"volumeId": "gv1", "path": "/workspace"}]} and "dataCenterIds" not in body
    assert s.volume == "gv1" and s.volume_kind == "global" and body["disk"] == 80
    s.close()


SHIMS = {
    # git: a "clone" makes the folder with a requirements.txt; describe answers the tag cloned
    "git": """#!/usr/bin/env bash
if [ "$1" = clone ]; then dest="${@: -1}"; mkdir -p "$dest/models"; printf 'torch\\nnumpy>=1.25\\ntorchvision\\nav\\n' \\
    > "$dest/requirements.txt"; echo "# fake" > "$dest/main.py"; echo clone >> "$SHIM_LOG"; exit 0; fi
if [ "$3" = describe ]; then echo "$COMFY_REF"; exit 0; fi
exit 0
""",
    "python3": """#!/usr/bin/env bash
[ "$1" = -c ] && { echo 2.8.0; exit 0; }
exit 0
""",
    # uv: a venv whose python "serves" ComfyUI (sleeps); pip install records what it was asked for
    "uv": """#!/usr/bin/env bash
if [ "$1" = venv ]; then d="${@: -1}"; mkdir -p "$d/bin"; printf '#!/usr/bin/env bash\\necho "comfy $*" >> "$SHIM_LOG"\\necho "threads OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS" >> "$SHIM_LOG"\\nexec sleep 300\\n' > "$d/bin/python"
    chmod +x "$d/bin/python"; exit 0; fi
if [ "$1" = pip ]; then echo "uv $*" >> "$SHIM_LOG"; for a in "$@"; do [ -f "$a" ] && cat "$a" >> "$SHIM_LOG"; done
    echo pip >> "$SHIM_LOG"; exit 0; fi
""",
    # curl: ComfyUI answers; a HEAD gives the payload's sha256 as x-linked-etag; a download writes the payload
    "curl": """#!/usr/bin/env bash
for a in "$@"; do url="$a"; done
case "$url" in http://127.0.0.1:8188/*) exit 0 ;; esac
if [[ " $* " == *" -fsI "* ]]; then printf 'HTTP/2 302\\r\\nx-linked-etag: "%s"\\r\\n\\r\\n' "$SHIM_SHA"; exit 0; fi
out=""; prev=""; for a in "$@"; do [ "$prev" = -o ] && out="$a"; prev="$a"; done
printf '%s' "$SHIM_PAYLOAD" > "$out"; echo "download ${url%%\\?*}" >> "$SHIM_LOG"
""",
}


def run_setup(tmp_path, volume, sha=None, extra=None):
    import hashlib
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    for name, body in SHIMS.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    log = tmp_path / "shim.log"
    payload = "weights"
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "SHIM_LOG": str(log), "SHIM_PAYLOAD": payload,
           "SHIM_SHA": sha or hashlib.sha256(payload.encode()).hexdigest(), "COMFY_REF": RP.COMFY_REF,
           "MODELS": "7b", "IDLE_MIN": "0", "MIN_RAM_GB": "0", "VOLUME_PATH": str(volume) if volume else "",
           "TORCH_PINS": " ".join(RP.TORCH_PINS),
           "CODE_DIR": str(tmp_path / "code"), "IO_DIR": str(tmp_path / "io"), **(extra or {})}
    r = subprocess.run(["bash", str(RP.SETUP_SCRIPT)], env=env, capture_output=True, text=True, timeout=60,
                       check=False)
    pid = tmp_path / "io" / "comfy.pid"
    if pid.is_file():                                         # the stand-in ComfyUI
        try:
            os.kill(int(pid.read_text()), 15)
        except (ProcessLookupError, ValueError):
            pass
        pid.unlink()
    return r, log.read_text() if log.is_file() else ""


TMPFS = subprocess.run(["stat", "-f", "-c", "%T", "/tmp"], capture_output=True, text=True,
                       check=False).stdout.strip() == "tmpfs"


@pytest.mark.skipif(not TMPFS, reason="needs /tmp on tmpfs")
def test_setup_script_full_precision_on_big_gpus_everything_on_the_disk(tmp_path):
    """The whole setup script with stand-ins for git, uv, python and curl. 80 GB of VRAM: the 7B fp16 weights and
    the VAE fetched to the container disk, even when the volume has weights (a global volume took writes at
    7-9 MB/s and loaded the VAE in 16 s; Hugging Face gave 111-182 MB/s), ComfyUI started with --highvram; ComfyUI
    and its packages on the container disk, never on the volume (git failed there: no chmod); what an old layout
    left on the volume removed, its weights left alone; nothing fetched twice."""
    vol = tmp_path / "workspace"
    w = vol / "video-chain" / "models"
    for sub, name in (("diffusion_models", "seedvr2_7b_int8_convrot.safetensors"),
                      ("vae", "seedvr2_ema_vae_fp16.safetensors")):
        (w / sub).mkdir(parents=True)
        (w / sub / name).write_text("weights")
        (w / sub / f".{name}.ok").write_text("")
    (vol / "video-chain" / "ComfyUI" / ".git").mkdir(parents=True)          # what the failed first run left
    node = tmp_path / "code" / "nodes" / "video_chain_scale"                 # what the app uploads first
    node.mkdir(parents=True)
    (node / "__init__.py").write_bytes(RP.SCALE_NODE_FILE.read_bytes())
    r, log = run_setup(tmp_path, vol, extra={"VRAM_MB": "81559", "CPUS": "23"})
    assert r.returncode == 0, r.stderr
    notes = [x[3:] for x in r.stdout.splitlines() if x.startswith("@@ ")]
    assert notes[-1] == "ready" and "GPU with 79 GB VRAM: SeedVR2 fp16" in notes
    assert any(n.startswith("CPU threads: 23 (the container's share of ") for n in notes)
    assert "threads OMP=23 MKL=23" in log                     # ComfyUI starts with the container's share
    assert "downloading seedvr2_7b_fp16.safetensors" in notes and "downloading seedvr2_ema_vae_fp16.safetensors" in notes
    assert not any("on the volume" in n for n in notes)
    code = tmp_path / "code"
    assert (code / "ComfyUI" / "models" / "diffusion_models" / "seedvr2_7b_fp16.safetensors").read_text() == "weights"
    assert (code / "ComfyUI" / "models" / "vae" / "seedvr2_ema_vae_fp16.safetensors").read_text() == "weights"
    assert log.count("download ") == 2 and "seedvr2_7b_fp16" in log
    assert sorted(p.name for p in (vol / "video-chain").iterdir()) == ["models"]   # old layout gone, weights kept
    assert sorted(p.name for p in (w / "diffusion_models").iterdir()) == [".seedvr2_7b_int8_convrot.safetensors.ok",
                                                                         "seedvr2_7b_int8_convrot.safetensors"]
    assert not (code / "ComfyUI" / "extra_model_paths.yaml").exists()            # ComfyUI does not look there
    assert "comfy main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --highvram" in log
    assert (code / "ComfyUI" / "custom_nodes" / "video_chain_scale" / "__init__.py").read_bytes() == \
        RP.SCALE_NODE_FILE.read_bytes()                                       # the lanczos node installed
    assert "numpy>=1.25" in log and "torch\n" not in log and "torchvision\n" not in log    # (unpinned ones out)
    assert "torch==2.15.0\ntorchvision==0.29.2\n" in log and " -c " in log              # pinned, as constraints
    assert oct((tmp_path / "io").stat().st_mode & 0o777) == "0o700"
    r, log2 = run_setup(tmp_path, vol, extra={"VRAM_MB": "81559"})       # the same pod again: nothing fetched
    assert r.returncode == 0 and log2.count("download ") == 2


@pytest.mark.skipif(not TMPFS, reason="needs /tmp on tmpfs")
def test_setup_script_on_a_24gb_gpu_and_without_a_volume(tmp_path):
    """24 GB: the int8 copy (as at home), no --highvram; fetched to the container disk, also with a volume that
    has it. No volume: the same. A bad checksum stops it, nothing is kept."""
    vol = tmp_path / "workspace"
    w = vol / "video-chain" / "models"
    for sub, name in (("diffusion_models", "seedvr2_7b_int8_convrot.safetensors"),
                      ("vae", "seedvr2_ema_vae_fp16.safetensors")):
        (w / sub).mkdir(parents=True)
        (w / sub / name).write_text("weights")
        (w / sub / f".{name}.ok").write_text("")
    r, log = run_setup(tmp_path, vol, extra={"VRAM_MB": "24564"})
    assert r.returncode == 0, r.stderr
    assert "GPU with 23 GB VRAM: SeedVR2 int8_convrot" in r.stdout and log.count("download ") == 2
    assert (tmp_path / "code" / "ComfyUI" / "models" / "diffusion_models" / "seedvr2_7b_int8_convrot.safetensors").is_file()
    assert "--highvram" not in log and "--disable-auto-launch " in log
    nv = tmp_path / "novol"
    nv.mkdir()
    r, log = run_setup(nv, None, extra={"VRAM_MB": "81559"})
    assert r.returncode == 0, r.stderr
    assert log.count("download ") == 2 and not (nv / "code" / "ComfyUI" / "extra_model_paths.yaml").exists()
    assert (nv / "code" / "ComfyUI" / "models" / "vae" / "seedvr2_ema_vae_fp16.safetensors").is_file()
    bad = tmp_path / "bad"
    bad.mkdir()
    r, _ = run_setup(bad, None, sha="0" * 64, extra={"VRAM_MB": "81559"})
    assert r.returncode == 3 and "checksum mismatch" in r.stderr
    assert not list((bad / "code" / "ComfyUI" / "models").rglob("*.safetensors"))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_cloud_upscaler_uses_the_big_gpu_fully(tmp_path):
    """On 40 GB of VRAM or more the pod's SeedVR2 runs at full precision (the fp16 weights) and on whole frames
    (plain VAEEncode / VAEDecode, no tile inputs); below that as at home (int8, 512 px tiles)."""
    from fake_comfy import FakeComfy
    server = FakeComfy(tmp_path / "comfy")
    try:
        draft = tmp_path / "d.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x36:r=24:d=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(draft)], check=True)
        for vram, unet, enc, dec in ((80 * 2**30, "seedvr2_7b_fp16.safetensors", "VAEEncode", "VAEDecode"),
                                     (24 * 2**30, "seedvr2_7b_int8_convrot.safetensors", "VAEEncodeTiled",
                                      "VAEDecodeTiled")):
            server.vram_total, server.scale_node = vram, vram > 40 * 2**30
            logs = []
            gen = RP.CloudUpscaler(FakeSession(server.url, 0), log=logs.append)
            gen.seedvr(draft, tmp_path / "up.mp4", (128, 72))
            wf = server.prompts[-1]
            assert server.seedvr_jobs[-1] == unet
            assert (wf["enc"]["class_type"], wf["dec"]["class_type"]) == (enc, dec)
            assert gen.seedvr_variant == ("fp16, whole frames" if enc == "VAEEncode" else None)
            assert gen.seedvr_keep_cache                                     # loaders not run again per clip
            frees = [c[2] for c in server.calls if c[:2] == ("BODY", "/free")]
            assert {"unload_models": False, "free_memory": True} not in frees     # the cache is never dropped
            assert wf["resize"]["class_type"] == ("VideoChainLanczosScale" if server.scale_node else "ImageScale")
            if enc == "VAEEncode":
                assert set(wf["enc"]["inputs"]) == {"pixels", "vae"} and set(wf["dec"]["inputs"]) == {"samples", "vae"}
                assert any("full precision" in x for x in logs)
            else:
                assert wf["enc"]["inputs"]["tile_size"] == 512 and any("512 px tiles" in x for x in logs)
    finally:
        server.close()
