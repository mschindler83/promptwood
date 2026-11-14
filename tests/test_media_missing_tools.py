"""Every feature that runs ffmpeg says so plainly when ffmpeg is not there (TST-15): no FileNotFoundError, no hang,
no crash in a slot. PATH is emptied, so no binary is found."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from video_chain import chain, faces, media, swap, tts  # noqa: E402


@pytest.fixture(autouse=True)
def no_tools(monkeypatch):
    monkeypatch.setenv("PATH", "")


@pytest.fixture
def work(tmp_path):                                  # (tmp_path also holds the isolated home)
    d = tmp_path / "work"
    d.mkdir()
    return d


def test_chain_functions_raise_a_media_error(work):
    f = work / "a.mp4"
    f.write_bytes(b"x")
    for call in (lambda: chain.probe(f), lambda: chain.probe_source(f), lambda: chain.to_png(f, work / "o.png"),
                 lambda: chain.to_wav(f, work / "o.wav"), lambda: chain.frame_at(f, 0, work / "o.png"),
                 lambda: chain.audio_seconds(f), lambda: chain.image_size(f)):
        with pytest.raises(media.MediaError, match="not installed"):
            call()
    assert chain.hard_cuts(f) == []
    assert os.listdir(work) == ["a.mp4"]            # (no part file left behind)


def test_swap_and_faces_say_ffmpeg_is_missing(work):
    f = work / "a.mp4"
    f.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="not installed"):
        swap.probe(f)
    with pytest.raises(RuntimeError, match="not installed"):
        faces.image_size(f)
    with pytest.raises(RuntimeError, match="not installed"):
        swap.detect_faces(f, work / "m.onnx", info=(64, 48, 3, 24.0))
    with pytest.raises(RuntimeError, match="not installed"):
        swap.write_mask([(1, 1, 5, 5)], 16, 16, work / "m.mkv")
    with pytest.raises(RuntimeError, match="not installed"):
        swap.trim_with_sound(f, f, work / "o.mp4", 3)
    assert sorted(os.listdir(work)) == ["a.mp4"]


def test_tts_speed_change_is_a_tts_error(work):
    wav = work / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(tts.TtsError, match="not installed"):
        tts._tempo(wav, 1.2)
    assert wav.read_bytes() == b"x" and os.listdir(work) == ["a.wav"]


def test_ui_helpers_do_not_crash_without_ffmpeg(tmp_path):
    pytest.importorskip("PySide6")
    from video_chain.ui import audio_editor, clip_dialogs, compare_player, image_edit, image_picker
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    assert clip_dialogs.grab_frame(f, 0.5, tmp_path / "o.png") is False
    assert audio_editor.waveform(f, 1.0) == []
    with pytest.raises(media.MediaError, match="not installed"):
        compare_player.decode(f, (16, 12))
    with pytest.raises(media.MediaError, match="not installed"):
        clip_dialogs.decode_frames(f)
    with pytest.raises(media.MediaError, match="not installed"):
        image_edit.scale_to(f, tmp_path / "o.png", (16, 12))
    assert image_picker._video_frame(str(f)).isNull()
    assert image_picker._media_info(str(f)) == ""
