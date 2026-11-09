"""Every test gets its own LoRA trigger database (never the app's real one in ~/.local/share)."""
import pytest


@pytest.fixture(autouse=True)
def own_trigger_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_CHAIN_TRIGGERS", str(tmp_path / "lora_triggers.json"))


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
