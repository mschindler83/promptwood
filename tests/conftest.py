"""Every test gets its own home / config / data / cache folders (default QSettings land there, never in the user's real ones),
no internet (only this machine's own servers), and the timers of the windows it leaves open are stopped after it.

Every test gets its own LoRA trigger database and LoRA presets (never the app's real one in ~/.local/share) and an in-memory
keyring (never the system keyring with the user's CivitAI key).

New trees are made in the classic prompt format here (the tests written before the h3 format assert its text); a test
of the h3 format sets tree.options["prompt_format"] = "h3", one marked h3_default gets the app's real default."""
import os
import urllib.request
from urllib.parse import urlparse

import pytest


class _MemoryKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, pw):
        self.store[(service, user)] = pw

    def delete_password(self, service, user):
        import keyring.errors
        if (service, user) not in self.store:
            raise keyring.errors.PasswordDeleteError("none")
        del self.store[(service, user)]


@pytest.fixture(autouse=True)
def classic_prompts(request, monkeypatch):
    if request.node.get_closest_marker("h3_default") is None:
        from video_chain import tree as T
        monkeypatch.setitem(T.DEFAULT_OPTIONS, "prompt_format", "classic")


def settings(tmp_path, name="s.ini", view="advanced"):
    """The test's own settings file. view: the window's view ("advanced": every control, as the tests were written
    for; "simple"; None: none stored, the window decides as on a first run / an update)."""
    from PySide6.QtCore import QSettings
    q = QSettings(str(tmp_path / name), QSettings.IniFormat)
    if view and q.value("viewMode") is None:
        q.setValue("viewMode", view)
    if q.value("comfyDir") is None:                  # never the real ComfyUI folder in tests
        q.setValue("comfyDir", str(tmp_path / "no-comfyui"))
    if q.value("thumbCacheDir") is None:             # nor the real thumbnail cache
        q.setValue("thumbCacheDir", str(tmp_path / "thumb-cache"))
    if q.value("comfyAutoStart") is None:            # never start the real ComfyUI from a test
        q.setValue("comfyAutoStart", False)
    if q.value("loraTriggersOnline") is None:        # never ask CivitAI from a test
        q.setValue("loraTriggersOnline", False)
    return q


@pytest.fixture(autouse=True)
def no_runpod_key(request, monkeypatch):
    """Never the real RunPod key from the system keyring in a test (a test that needs one sets its own; test_runpod
    tests api_key() itself, with its own keyring)."""
    if request.module.__name__ == "test_runpod":
        return
    from video_chain import runpod as RP
    monkeypatch.setattr(RP, "api_key", lambda: "")


@pytest.fixture(autouse=True)
def own_folders(tmp_path, monkeypatch):
    """HOME and the XDG folders inside the test's tmp_path; QSettings' default stores (native and ini, user scope) too,
    so a window made without settings= (QSettings()) cannot read or write the developer's real configuration."""
    home = tmp_path / "home"
    for name, sub in (("XDG_CONFIG_HOME", ".config"), ("XDG_DATA_HOME", ".local/share"),
                      ("XDG_CACHE_HOME", ".cache"), ("XDG_STATE_HOME", ".local/state")):
        (home / sub).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(home / sub))
    monkeypatch.setenv("HOME", str(home))
    from PySide6.QtCore import QSettings
    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(home / ".config"))


_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


@pytest.fixture(autouse=True)
def no_internet(monkeypatch):
    """urlopen only reaches this machine (the fake ComfyUI / llama servers); anything else fails the test at once
    instead of waiting for a timeout on a real network. A test that fakes urlopen itself overrides this."""
    real = urllib.request.urlopen

    def guarded(url, *a, **k):
        target = url.full_url if hasattr(url, "full_url") else str(url)
        host = urlparse(target).hostname or ""
        if host not in _LOCAL_HOSTS and f"[{host}]" not in _LOCAL_HOSTS:
            raise AssertionError(f"a test reached for the internet: {target}")
        return real(url, *a, **k)
    monkeypatch.setattr(urllib.request, "urlopen", guarded)


@pytest.fixture(autouse=True)
def stop_leftover_timers():
    """A window a test leaves open must not keep ticking (GPU / ComfyUI checks) into the next tests."""
    yield
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    if QApplication.instance() is None:
        return
    for w in QApplication.topLevelWidgets():
        for t in w.findChildren(QTimer):
            t.stop()


@pytest.fixture(autouse=True)
def own_trigger_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_CHAIN_TRIGGERS", str(tmp_path / "lora_triggers.json"))
    monkeypatch.setenv("VIDEO_CHAIN_PRESETS", str(tmp_path / "lora_presets.json"))
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    import keyring
    from video_chain import triggers
    mem = _MemoryKeyring()
    for name in ("get_password", "set_password", "delete_password"):
        monkeypatch.setattr(keyring, name, getattr(mem, name))
    monkeypatch.setattr(triggers, "_API_KEY", None)
    monkeypatch.delenv("VIDEO_CHAIN_CIVITAI_SITE", raising=False)
    monkeypatch.setattr(triggers, "_SITE", None)             # (the default site, whatever a test before chose)


def pytest_sessionfinish(session, exitstatus):
    """Close and delete the windows tests left open while Qt still runs: torn down after the application object at
    exit, Qt aborted ("pure virtual method called")."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    import gc
    for w in QApplication.topLevelWidgets():   # (deleted, not closed: a closed window's close handler ran already)
        w.deleteLater()
    for _ in range(3):
        app.processEvents()
    gc.collect()
    app.processEvents()
