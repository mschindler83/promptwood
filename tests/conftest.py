"""Every test gets its own LoRA trigger database (never the app's real one in ~/.local/share)."""
import pytest


@pytest.fixture(autouse=True)
def own_trigger_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_CHAIN_TRIGGERS", str(tmp_path / "lora_triggers.json"))
