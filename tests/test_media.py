"""The shared ffmpeg / ffprobe runner, the probe and the safe file writes (video_chain.media)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from video_chain import media

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg")


def make_video(path, frames=6, size="64x48", audio=False):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={frames / 24:.6f}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 24:.6f}", "-c:a", "aac"]
    subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(path)], check=True)
    return path


# ---- running tools
def test_a_missing_binary_is_a_media_error_that_names_it(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(media.MediaError, match="ffmpeg is not installed"):
        media.ffmpeg(["-version"])
    with pytest.raises(media.MediaError, match="ffprobe is not installed"):
        media.ffprobe_json("x.mp4")


def test_a_missing_binary_in_popen_is_a_media_error(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(media.MediaError, match="ffmpeg is not installed"):
        media.popen(["ffmpeg", "-version"])


def test_require_ffmpeg_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None if name == "ffprobe" else "/bin/" + name)
    with pytest.raises(media.MediaError, match="ffprobe"):
        media.require_ffmpeg()
    monkeypatch.setattr(media.shutil, "which", lambda name: "/bin/" + name)
    media.require_ffmpeg()


@needs_ffmpeg
def test_a_failing_ffmpeg_carries_the_last_lines_of_its_message(tmp_path):
    with pytest.raises(media.MediaError) as e:
        media.ffmpeg(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "o.mp4")])
    assert "ffmpeg failed" in str(e.value) and "missing.mp4" in str(e.value)


@needs_ffmpeg
def test_a_hanging_ffmpeg_is_stopped(tmp_path):
    with pytest.raises(media.MediaError, match="did not finish"):
        media.ffmpeg(["-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=600", "-f", "null", "-"], timeout=0.5)


def test_run_returns_text_or_bytes():
    assert media.run([sys.executable, "-c", "print('hi')"]).strip() == "hi"
    assert media.run([sys.executable, "-c", "print('hi')"], text=False).strip() == b"hi"
    with pytest.raises(media.MediaError, match="boom"):
        media.run([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])


@needs_ffmpeg
def test_ffmpeg_does_not_read_the_terminal(tmp_path):
    seen = []
    real = subprocess.run
    media.subprocess.run = lambda cmd, **kw: (seen.append(cmd), real(cmd, **kw))[1]
    try:
        media.ffmpeg(["-version"])
    finally:
        media.subprocess.run = real
    assert seen[0][:4] == ["ffmpeg", "-nostdin", "-v", "error"]


# ---- probing
@needs_ffmpeg
def test_probe_reads_size_frames_fps_duration_and_sound(tmp_path):
    info = media.probe(make_video(tmp_path / "a.mp4", frames=12, audio=True), count_frames=True)
    assert (info.width, info.height, info.frames, info.has_audio) == (64, 48, 12, True)
    assert abs(info.fps - 24) < 0.01 and abs(info.duration - 0.5) < 0.2
    assert not media.probe(make_video(tmp_path / "b.mp4")).has_audio


@needs_ffmpeg
def test_probe_of_a_file_without_video_says_so(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=0.2", str(wav)], check=True)
    with pytest.raises(media.MediaError, match="no video stream"):
        media.probe(wav)


@needs_ffmpeg
def test_image_size_of_a_picture_and_of_a_video(tmp_path):
    png = tmp_path / "a.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=40x30", "-frames:v", "1", str(png)],
                   check=True)
    assert media.image_size(png) == (40, 30)
    assert media.image_size(make_video(tmp_path / "a.mp4")) == (64, 48)


@needs_ffmpeg
def test_audio_seconds_is_none_without_sound(tmp_path):
    assert media.audio_seconds(make_video(tmp_path / "a.mp4")) is None
    assert media.audio_seconds(make_video(tmp_path / "b.mp4", frames=24, audio=True)) == pytest.approx(1.0, abs=0.2)


@needs_ffmpeg
def test_iter_frames_streams_raw_frames_and_stops_the_process(tmp_path):
    v = make_video(tmp_path / "a.mp4", frames=6)
    frames = list(media.iter_frames(v, (16, 12), "rgb24"))
    assert len(frames) == 6 and all(len(f) == 16 * 12 * 3 for f in frames)
    it = media.iter_frames(v, (16, 12), "gray")
    assert len(next(it)) == 16 * 12
    it.close()                                  # (the reader is killed; nothing hangs)
    assert list(media.iter_frames(v, (16, 12), "gray", cancelled=lambda: True)) == []


@needs_ffmpeg
def test_iter_frames_of_a_bad_file_raises_with_ffmpegs_message(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(media.MediaError, match="bad.mp4"):
        list(media.iter_frames(bad, (16, 12)))


# ---- safe writes
def test_atomic_writes_leave_the_whole_file_and_no_partial(tmp_path):
    w = tmp_path / 'w'
    w.mkdir()
    f = w / "a.json"
    media.atomic_write_json(f, {"a": 1})
    assert json.loads(f.read_text()) == {"a": 1}
    media.atomic_write_text(f, "two")
    media.atomic_write_bytes(f, b"three")
    assert f.read_bytes() == b"three"
    assert os.listdir(w) == ["a.json"]


def test_a_failing_atomic_write_keeps_the_old_file_and_leaves_nothing(tmp_path):
    w = tmp_path / 'w'
    w.mkdir()
    f = w / "a.json"
    f.write_text("old")
    with pytest.raises(TypeError):
        media.atomic_write_json(f, {"a": object()})
    assert f.read_text() == "old" and os.listdir(w) == ["a.json"]


def test_part_file_is_renamed_on_success_and_removed_on_error(tmp_path):
    w = tmp_path / 'w'
    w.mkdir()
    out = w / "clip.mp4"
    with media.part_file(out) as tmp:
        assert tmp != out and tmp.suffix == ".mp4" and ".part" in tmp.name and tmp.parent == w
        tmp.write_bytes(b"x")
    assert out.read_bytes() == b"x" and os.listdir(w) == ["clip.mp4"]
    with pytest.raises(ValueError):
        with media.part_file(out) as tmp:
            tmp.write_bytes(b"half")
            raise ValueError("cut off")
    assert out.read_bytes() == b"x" and os.listdir(w) == ["clip.mp4"]


def test_part_file_without_output_removes_the_part_and_keeps_the_old(tmp_path):
    w = tmp_path / 'w'
    w.mkdir()
    out = w / "clip.mp4"
    with pytest.raises(FileNotFoundError):
        with media.part_file(out) as tmp:
            pass                                # (the writer made nothing)
    assert os.listdir(w) == []


def test_file_digest_and_scratch_names(tmp_path):
    w = tmp_path / 'w'
    w.mkdir()
    f = w / "a.bin"
    f.write_bytes(b"abc")
    import hashlib
    assert media.file_digest(f, "sha1") == hashlib.sha1(b"abc").hexdigest()
    assert media.file_digest(f) == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(OSError):
        media.file_digest(w / "none")
    assert media.file_digest(f, stop=lambda: True) is None
    with media.scratch("crop", dir=w) as d:
        assert os.path.basename(d).startswith("video-chain-crop-") and os.path.isdir(d)
    assert not os.path.exists(d)
