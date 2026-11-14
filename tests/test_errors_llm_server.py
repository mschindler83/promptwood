"""llama-server dies with the app (PR_SET_PDEATHSIG), and stop() takes a grace."""
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from video_chain import llm_server as L

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux only")


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


@linux_only
def test_a_child_dies_when_the_app_dies(tmp_path):
    """A python 'app' starts a child through the module's spawner and is killed hard (no atexit): the child ends too."""
    pidfile = tmp_path / "pid"
    code = textwrap.dedent(f"""
        import os, subprocess, sys, time
        from video_chain import llm_server as L
        p = L.spawn(["sleep", "60"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        open({str(pidfile)!r}, "w").write(str(p.pid))
        time.sleep(0.3)
        os._exit(0)
    """)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30, cwd=str(tmp_path))
    pid = int(pidfile.read_text())
    end = time.time() + 5
    while time.time() < end and _alive(pid):
        time.sleep(0.1)
    alive = _alive(pid)
    if alive:
        os.kill(pid, 9)
    assert not alive


@linux_only
def test_a_child_started_from_a_short_lived_thread_lives_on():
    """PDEATHSIG follows the thread that forked; the spawner is one long-lived thread, so the thread that asked
    (a worker that ends after its refine) does not take the server with it."""
    box = []
    t = threading.Thread(target=lambda: box.append(L.spawn(["sleep", "60"], stdin=subprocess.DEVNULL,
                                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                           start_new_session=True)))
    t.start()
    t.join()
    time.sleep(0.5)
    p = box[0]
    try:
        assert p.poll() is None
    finally:
        p.kill()
        p.wait()


def test_stop_takes_a_grace(monkeypatch):
    srv = L.LlamaServer.__new__(L.LlamaServer)
    srv._lock = threading.Lock()
    srv._gen = 0
    seen = []
    srv.proc = object()
    monkeypatch.setattr(srv, "_take_over_leftover", lambda keep, wait=True: False, raising=False)
    monkeypatch.setattr(srv, "_forget", lambda: None, raising=False)
    monkeypatch.setattr(L.LlamaServer, "_kill", staticmethod(lambda p, wait=True, grace=20: seen.append((wait, grace)) or True))
    srv.stop(grace=2)
    srv.proc = object()
    srv.stop()
    assert seen == [(True, 2), (True, 20)]
