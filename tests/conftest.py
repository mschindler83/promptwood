"""Every test gets its own home / config / data / cache folders (default QSettings land there, never in the user's real ones),
no internet (only this machine's own servers), and the timers of the windows it leaves open are stopped after it.

Every test gets its own LoRA trigger database and LoRA presets (never the app's real one in ~/.local/share) and an in-memory
keyring (never the system keyring with the user's CivitAI key).

New trees are made in the classic prompt format here (the tests written before the h3 format assert its text); a test
of the h3 format sets tree.options["prompt_format"] = "h3", one marked h3_default gets the app's real default. Likewise
they hand over the frame as it is (steady_look off); one marked steady_default gets the app's default."""
import atexit
import ipaddress
import os
import shutil
import socket
import tempfile
import urllib.request
from urllib.parse import urlparse

import pytest

# The app's home folder (ComfyUI, model cache, config.json of an installed app) is never the real one in tests: an
# empty temporary folder, set before the app's modules read their defaults from it.
_HOME = tempfile.mkdtemp(prefix="video-chain-test-home-")
os.environ["VIDEO_CHAIN_HOME"] = _HOME
for _name in ("COMFY_DIR", "COMFY_LAUNCHER", "COMFY_URL", "HF_CACHE", "LLAMA_SERVER", "TTS_PYTHON", "FISH_DIR"):
    os.environ.pop("VIDEO_CHAIN_" + _name, None)
atexit.register(shutil.rmtree, _HOME, True)


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


@pytest.fixture(autouse=True)
def plain_handover(request, monkeypatch):
    """New trees hand over the frame as it is (the tests written before the steady look assert it); a test marked
    steady_default gets the app's real default."""
    if request.node.get_closest_marker("steady_default") is None:
        from video_chain import tree as T
        monkeypatch.setitem(T.DEFAULT_OPTIONS, "steady_look", False)


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


REAL_COMFYUI_PORTS = (8188,)       # the user's own ComfyUI on this machine (the app's default address)
_REAL_CONNECT = socket.socket.connect


def _no_real_comfyui(self, address):
    """No test reaches the user's real ComfyUI: a connection to its port on this machine is refused at once (the
    windows' status poll asked it, two tests sent it requests, one an unload; a busy ComfyUI hung the run). For the
    whole run, not per test: a window's poll thread can still ask after its test ended. The fake ComfyUI servers
    run on other ports."""
    if isinstance(address, tuple) and len(address) >= 2 and address[1] in REAL_COMFYUI_PORTS \
            and _this_machine(str(address[0])):
        raise ConnectionRefusedError(f"a test reached for the real ComfyUI at {address[0]}:{address[1]}")
    return _REAL_CONNECT(self, address)


def _this_machine(host: str) -> bool:
    """Any loopback address (127/8, ::1, ::ffff:127.x), 0.0.0.0 / :: or the name localhost."""
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or ip.is_unspecified or bool(mapped and (mapped.is_loopback or mapped.is_unspecified))


socket.socket.connect = _no_real_comfyui    # type: ignore[method-assign]


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
def own_log_handlers():
    """A test that runs an entry point (logs.setup) leaves no log handler of ours behind."""
    import logging
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_video_chain", False):
            root.removeHandler(h)
            h.close()


@pytest.fixture(autouse=True)
def first_photo_as_face(monkeypatch):
    """The cast's question for a person's first photo (character sheets or a face picture) is answered "face":
    the tests from before sheets keep their flow; a test of sheets patches win.ask_first_photo itself."""
    from video_chain.ui.mw_cast import CastMixin
    monkeypatch.setattr(CastMixin, "ask_first_photo", lambda self, who: "face")


@pytest.fixture(autouse=True)
def no_unanswered_dialogs(monkeypatch):
    """A message box a test does not answer fails it at once: offscreen nobody can click it, so the run would
    otherwise wait for ever. A test that expects one patches QMessageBox.question (etc.) itself."""
    from PySide6.QtWidgets import QMessageBox

    def guard(kind):
        def box(parent, title, text, *a, **k):
            raise AssertionError(f"QMessageBox.{kind}({title!r}) opened: answer it in the test "
                                 f"(monkeypatch QMessageBox.{kind}). Text: {str(text)[:200]!r}")
        return staticmethod(box)
    for kind in ("question", "warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, kind, guard(kind))


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
def no_plugins(monkeypatch):
    """The app's own tests run without plugins, whatever is installed next to it (a plugin test switches them on
    for its own made-up plugin)."""
    from video_chain import plugins
    monkeypatch.setenv(plugins.OFF, "1")
    monkeypatch.delenv(plugins.ENV, raising=False)
    plugins.reset()
    yield
    plugins.reset()


@pytest.fixture(autouse=True)
def own_trigger_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_CHAIN_TRIGGERS", str(tmp_path / "lora_triggers.json"))
    monkeypatch.setenv("VIDEO_CHAIN_PRESETS", str(tmp_path / "lora_presets.json"))
    for name in ("VIDEO_CHAIN_PROMPTS", "VIDEO_CHAIN_VOICES", "VIDEO_CHAIN_CASTS"):   # (the libraries: in tmp_path)
        monkeypatch.delenv(name, raising=False)
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
