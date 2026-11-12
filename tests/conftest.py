"""Every test gets its own LoRA trigger database and LoRA presets (never the app's real one in ~/.local/share) and an in-memory
keyring (never the system keyring with the user's CivitAI key)."""
import atexit
import os
import shutil
import tempfile

import pytest

# The app's home folder (ComfyUI, model cache, config.json of an installed app) is never the real one in tests: an
# empty temporary folder, set before the app's modules read their defaults from it.
_HOME = tempfile.mkdtemp(prefix="video-chain-test-home-")
os.environ["VIDEO_CHAIN_HOME"] = _HOME
for _name in ("COMFY_DIR", "COMFY_LAUNCHER", "COMFY_URL", "HF_CACHE", "LLAMA_SERVER", "TTS_PYTHON"):
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
