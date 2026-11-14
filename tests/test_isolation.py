"""The suite's own guards: every test runs with isolated settings / data / cache folders and without internet."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import urllib.error
import urllib.request

import pytest


def test_default_qsettings_never_reach_the_real_config(tmp_path):
    from PySide6.QtCore import QSettings
    q = QSettings("video-chain-test", "x")
    q.setValue("a", 1)
    q.sync()
    assert str(tmp_path) in q.fileName()


def test_home_and_xdg_folders_are_the_tests_own(tmp_path):
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "HOME"):
        assert str(tmp_path) in os.environ[name], name


def test_the_internet_is_blocked():
    with pytest.raises(AssertionError, match="internet"):
        urllib.request.urlopen("https://example.com/", timeout=1)


def test_localhost_is_allowed():
    from fakes import free_port
    with pytest.raises(urllib.error.URLError):               # (nobody listens: refused, not blocked)
        urllib.request.urlopen(f"http://127.0.0.1:{free_port()}/", timeout=1)
