"""config.data_dir / state_dir / config_dir: the XDG folders, one place; presets, triggers, logs and runpod use them."""
from pathlib import Path

from video_chain import config, logs, presets, triggers


def test_the_folders_follow_the_xdg_variables(monkeypatch, tmp_path):
    for var in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CONFIG_HOME", "VIDEO_CHAIN_PRESETS", "VIDEO_CHAIN_TRIGGERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert config.data_dir() == tmp_path / ".local" / "share" / "video-chain"
    assert config.state_dir() == tmp_path / ".local" / "state" / "video-chain"
    assert config.config_dir() == tmp_path / ".config" / "video-chain"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "d"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "s"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "c"))
    assert presets.default_path() == tmp_path / "d" / "video-chain" / "lora_presets.json"
    assert triggers.default_path() == tmp_path / "d" / "video-chain" / "lora_triggers.json"
    assert logs.default_log_file() == tmp_path / "s" / "video-chain" / "video-chain.log"
    assert config.config_dir() == tmp_path / "c" / "video-chain"
