"""swap.py / faces.py robustness: odd ffprobe answers, probing once, trimming at the clip's rate, error messages."""
import errno
import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from video_chain import faces, swap


def clip(path: Path, frames: int = 20, rate: int = 24, audio: str = "aac", size: str = "320x180") -> Path:
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}",
           "-f", "lavfi", "-i", "sine=f=440", "-frames:v", str(frames), "-shortest", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-c:a", audio, str(path)]
    subprocess.run(cmd, check=True)
    return path


def fake_probe_output(monkeypatch, streams):
    real = subprocess.run

    def run(cmd, *a, **k):
        if cmd and cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"streams": streams}), "")
        return real(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", run)


# ---- GEN-10
def test_probe_zero_over_zero_rate_does_not_crash(monkeypatch):
    fake_probe_output(monkeypatch, [{"codec_type": "video", "width": 64, "height": 32, "r_frame_rate": "0/0",
                                     "nb_read_frames": "5"}])
    w, h, n, fps = swap.probe(Path("x.mp4"))
    assert (w, h, n) == (64, 32, 5) and fps > 0


def test_probe_without_a_video_stream_says_so(monkeypatch):
    fake_probe_output(monkeypatch, [{"codec_type": "audio"}])
    with pytest.raises(RuntimeError, match="no video stream"):
        swap.probe(Path("x.mp4"))


def test_image_size_without_a_stream_or_ffprobe_says_so(monkeypatch):
    fake_probe_output(monkeypatch, [])
    with pytest.raises(RuntimeError, match="no picture|no video"):
        faces.image_size(Path("x.png"))

    def missing(*a, **k):
        raise FileNotFoundError("ffprobe")
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="ffprobe"):
        faces.image_size(Path("x.png"))


# ---- GEN-11
def test_find_heads_decodes_the_clip_once_for_probing(tmp_path, monkeypatch):
    v = clip(tmp_path / "v.mp4", 6)
    real, calls = swap.probe, []
    monkeypatch.setattr(swap, "probe", lambda p: calls.append(p) or real(p))
    monkeypatch.setattr(faces, "detect_rgb", lambda img, w, h, m: [(10, 10, 50, 60, .9)])
    swap.find_heads(v, tmp_path / "m.onnx")
    assert len(calls) == 1


# ---- GEN-12 / GEN-13 / GEN-14
def test_trim_keeps_the_clips_frame_rate(tmp_path):
    made, src = clip(tmp_path / "made.mp4", 40, 25), clip(tmp_path / "src.mp4", 25, 25)
    out = tmp_path / "out.mp4"
    swap.trim_with_sound(made, src, out, 25, fps=25)
    w, h, n, fps = swap.probe(out)
    assert n == 25 and abs(fps - 25) < 0.01


def test_trim_failure_is_not_retried_and_shows_ffmpegs_message(tmp_path, monkeypatch):
    src = clip(tmp_path / "src.mp4", 5)
    calls = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: calls.append(cmd) or real(cmd, *a, **k))
    with pytest.raises(RuntimeError, match="made.mp4|No such file"):
        swap.trim_with_sound(tmp_path / "made.mp4", src, tmp_path / "o.mp4", 5)
    assert len(calls) == 1


def test_trim_retries_with_aac_when_the_sound_does_not_fit_mp4(tmp_path):
    made = clip(tmp_path / "made.mp4", 12)
    src = tmp_path / "src.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24", "-f", "lavfi",
                    "-i", "sine=f=440", "-frames:v", "12", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "pcm_s16le", str(src)], check=True)
    # (pcm in mp4 may or may not be accepted by this ffmpeg: either way the result has the clip's frames and sound)
    out = tmp_path / "o.mp4"
    swap.trim_with_sound(made, src, out, 12)
    assert swap.probe(out)[2] == 12


# ---- GEN-15
def test_write_mask_failure_shows_ffmpegs_message(tmp_path):
    boxes = [(10.0, 10.0, 50.0, 50.0)] * 200
    with pytest.raises(RuntimeError, match="head mask"):
        swap.write_mask(boxes, 320, 180, tmp_path / "no-such-dir" / "m.mp4", 24)


def test_write_mask_keeps_a_fractional_frame_rate(tmp_path):
    out = tmp_path / "m.mp4"
    swap.write_mask([(10.0, 10.0, 50.0, 50.0)] * 10, 64, 64, out, 30000 / 1001)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout.strip()
    assert r == "30000/1001"


# ---- GEN-16
def test_store_pose_across_file_systems(tmp_path, monkeypatch):
    src = tmp_path / "scratch" / "pose.mkv"
    src.parent.mkdir()
    src.write_bytes(b"pose")
    real = Path.replace

    def replace(self, target):
        if self.name == "pose.mkv":
            raise OSError(errno.EXDEV, "cross-device")
        return real(self, target)
    monkeypatch.setattr(Path, "replace", replace)
    got = swap.store_pose(tmp_path / "proj", "k1", {"pose": src}, {"a": 1})
    assert got["pose"].read_bytes() == b"pose" and not src.exists()


# ---- GEN-18
def test_head_boxes_is_not_quadratic():
    tr = [(10.0, 10.0, 50.0, 60.0) if i % 2 == 0 else None for i in range(20000)]
    t = time.time()
    boxes = swap.head_boxes(tr, 320, 180)
    assert len(boxes) == 20000 and time.time() - t < 3


# ---- GEN-19
def test_head_swap_prompt_is_unchanged():
    p = swap.head_swap_prompt("a woman")
    assert p.startswith("head_swap:\n\nsubject_definitions:") and "a woman" in p
    import hashlib
    assert hashlib.sha1(p.encode()).hexdigest() == "ace251855bb919801a8e77331b648a6fa297b4c8"


# ---- GEN-36 (session lock, output check), GEN-13 (-nostdin)
def test_one_session_when_threads_ask_together(monkeypatch, tmp_path):
    made = []

    class Sess:
        def __init__(self, *a, **k):
            time.sleep(0.2)
            made.append(1)

    ort = types.SimpleNamespace(SessionOptions=lambda: types.SimpleNamespace(), InferenceSession=Sess)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(faces, "_SESSIONS", {})
    ts = [threading.Thread(target=faces._session, args=(tmp_path / "m.onnx",)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(made) == 1


def test_another_detector_model_is_said_plainly(monkeypatch, tmp_path):
    class Sess:
        def get_inputs(self):
            return [types.SimpleNamespace(name="in")]

        def run(self, *a):
            return [np.zeros((1, 1))] * 3
    monkeypatch.setattr(faces, "_session", lambda m: Sess())
    with pytest.raises(RuntimeError, match="det_10g|outputs"):
        faces.detect_rgb(np.zeros((10, 10, 3), np.uint8), 10, 10, tmp_path / "m.onnx")


def test_detect_runs_ffmpeg_without_stdin(monkeypatch, tmp_path):
    seen = []
    real = subprocess.run

    def run(cmd, *a, **k):
        if cmd[0] == "ffmpeg":
            seen.append(cmd)
        return real(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(faces, "detect_rgb", lambda *a: [])
    png = tmp_path / "p.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1",
                    str(png)], check=True)
    seen.clear()
    faces.detect(png, tmp_path / "m.onnx")
    assert seen and all("-nostdin" in c for c in seen)
