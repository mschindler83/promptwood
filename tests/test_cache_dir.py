"""config.cache_dir: one cache folder ($XDG_CACHE_HOME or ~/.cache, + video-chain) for the logs and the thumbnails."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from video_chain import config
from video_chain.ui import image_picker, ui_logic


def test_every_cache_folder_follows_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    assert config.cache_dir() == tmp_path / "c" / "video-chain"
    assert ui_logic.app_cache_dir() == config.cache_dir()
    assert image_picker.thumb_cache_dir() == config.cache_dir() / "thumbs"
    assert image_picker.desktop_thumbs_dir() == tmp_path / "c" / "thumbnails"
