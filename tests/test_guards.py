"""The test run's own guards: no test reaches the user's real ComfyUI (its default port on this machine). 162 tests'
windows polled it and two sent it requests (one an unload, /free); a busy ComfyUI made the whole run hang.

The guard is tested with a stub in place of the real connect, so this test never reaches the port itself, also
if the guard broke."""
import socket

import pytest

import conftest


@pytest.fixture
def passed(monkeypatch):
    """The addresses the guard let through (to the stub, not to a real socket)."""
    got = []
    monkeypatch.setattr(conftest, "_REAL_CONNECT", lambda self, address: got.append(address))
    return got


@pytest.mark.parametrize("address", [("127.0.0.1", 8188), ("localhost", 8188), ("127.0.0.2", 8188),
                                     ("::1", 8188, 0, 0), ("::ffff:127.0.0.1", 8188, 0, 0), ("0.0.0.0", 8188),
                                     ("LOCALHOST", 8188)])
def test_the_real_comfyui_port_on_this_machine_is_refused(passed, address):
    s = socket.socket(socket.AF_INET6 if len(address) == 4 else socket.AF_INET)
    try:
        with pytest.raises(ConnectionRefusedError, match="real ComfyUI"):
            s.connect(address)
    finally:
        s.close()
    assert passed == []


@pytest.mark.parametrize("address", [("127.0.0.1", 8199), ("127.0.0.1", 40123), ("::1", 8189, 0, 0)])
def test_other_local_ports_go_through(passed, address):
    s = socket.socket(socket.AF_INET6 if len(address) == 4 else socket.AF_INET)
    try:
        s.connect(address)
    finally:
        s.close()
    assert passed == [address]


def test_the_guard_is_in_place_for_the_whole_run():
    assert socket.socket.connect is conftest._no_real_comfyui
