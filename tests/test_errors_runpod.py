"""RunPod: no printing from library code, a deadline for close(), a second try for the SSH tunnel."""
import pytest
from test_runpod_safety import StubApi, session  # noqa: F401  (the stub and the session helper)

from video_chain import runpod as RP


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(RP.time, "sleep", lambda _s: None)


def test_the_listing_goes_to_the_caller_not_to_stdout(monkeypatch, capsys):
    api = StubApi(pods=[{"id": "p0", "name": f"{RP.POD_PREFIX}-0", "status": "RUNNING"}])
    api.gpus = lambda: [{"id": "A", "memory": 24}]
    monkeypatch.setattr(RP, "RunpodApi", lambda key: api)
    monkeypatch.setattr(RP, "api_key", lambda: "k")
    lines = []
    assert RP._cli("gpus", out=lines.append) == 0 and any("A" in x for x in lines)
    assert RP._cli("status", out=lines.append) == 0 and any("p0" in x for x in lines)
    assert capsys.readouterr().out == ""
    assert RP.main(["status"]) == 0
    assert "p0" in capsys.readouterr().out                    # (the command line prints)


def test_close_stops_trying_at_its_deadline(tmp_path):
    api = StubApi()
    s = session(api, tmp_path)
    s._create("k")
    api.delete_fails = 100
    s.close_timeout = 0
    with pytest.raises(RP.RunpodError, match="could not terminate"):
        s.close()
    assert api.delete_fails == 99                             # (one try, not five)


def test_the_tunnel_is_tried_once_more_on_another_port(tmp_path, monkeypatch):
    api = StubApi()
    s = session(api, tmp_path)
    ports = []
    monkeypatch.setattr(RP, "free_port", lambda: ports.append(1) or 40000 + len(ports))
    monkeypatch.setattr(s, "ssh_args", lambda *a: ["sh", "-c", "echo 'bind: Address already in use' >&2; exit 255"])
    with pytest.raises(RP.RunpodError, match="tunnel"):
        s._open_tunnel()
    assert len(ports) == 2
