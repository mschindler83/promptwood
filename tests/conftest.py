"""Every test gets its own LoRA trigger database and LoRA presets (never the app's real one in ~/.local/share) and an in-memory
keyring (never the system keyring with the user's CivitAI key).

New trees are made in the classic prompt format here (the tests written before the h3 format assert its text); a test
of the h3 format sets tree.options["prompt_format"] = "h3", one marked h3_default gets the app's real default."""
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
