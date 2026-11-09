"""Replace a person's head in a real video: finding and following the head, the mask, the Animate workflow."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_comfy import FakeComfy  # noqa: E402

from video_chain import animate as A  # noqa: E402
from video_chain import faces, swap  # noqa: E402
from video_chain import tree as T  # noqa: E402


def clip(path: Path, frames: int = 30, size: str = "320x180") -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=24",
                    "-f", "lavfi", "-i", "sine=f=440", "-frames:v", str(frames), "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
    return path


def test_track_and_head_boxes():
    """One person followed (the largest face first, then the nearest), gaps filled, the box grown to the head."""
    a, b = (100, 50, 140, 100, .9), (10, 10, 30, 40, .8)          # the person, someone small far away
    found = [[a, b], [b], [(104, 52, 144, 102, .9)], [], [(112, 54, 152, 104, .9), b]]
    tr = swap.track(found)
    assert tr[0] == a[:4] and tr[1] is None and tr[3] is None and tr[4] == (112, 54, 152, 104)
    boxes = swap.head_boxes(tr, 320, 180)
    assert len(boxes) == 5
    x1, y1, x2, y2 = boxes[1]                                      # filled from the frames around it
    assert x1 < 100 and x2 > 140 and y1 < 50 and y2 > 100         # larger than the face: hair, chin
    big = swap.head_boxes(tr, 320, 180, scale=1.5)[1]
    assert big[2] - big[0] > x2 - x1
    with pytest.raises(ValueError, match="no face"):
        swap.head_boxes([None, None], 320, 180)


def test_mask_video_and_detection(tmp_path, monkeypatch):
    v = clip(tmp_path / "v.mp4", 30)
    calls = []

    def fake_detect(img, w, h, model):                             # a face drifting right, frame by frame
        k = len(calls)
        calls.append(img.shape)
        return [] if k == 10 else [(100 + k, 60, 140 + k, 110, .9)]
    monkeypatch.setattr(faces, "detect_rgb", fake_detect)
    heads = swap.find_heads(v, tmp_path / "det.onnx")
    assert heads["frames"] == 30 and heads["found"] == 29 and heads["size"] == (320, 180)
    assert calls[0][:2] == faces.fit_size(320, 180)[::-1]
    m = tmp_path / "mask.mp4"
    swap.write_mask(heads["boxes"], 320, 180, m, heads["fps"])
    w, h, n, fps = swap.probe(m)
    assert (w, h, n) == (320, 180, 30) and abs(fps - 24) < 0.01
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(m), "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo",
                          "-"], capture_output=True, check=True).stdout
    f0 = np.frombuffer(raw, np.uint8).reshape(180, 320)
    x1, y1, x2, y2 = heads["boxes"][0]
    assert f0[int((y1 + y2) / 2), int((x1 + x2) / 2)] > 250 and f0[5, 5] < 5     # white head, black elsewhere


def test_apply_and_restore(tmp_path):
    t = T.Tree()
    v = t.add_video(None, str(tmp_path / "orig.mp4"), 2.0, 5.0)
    assert swap.replaced(v) is None and not swap.restore(v)
    swap.apply(v, tmp_path / "swap.mp4", 72, {"reference": "ref.png"})
    assert v.source["start"] == 0.0 and v.source["end"] == 3.0 and v.source["file"].endswith("swap.mp4")
    assert swap.replaced(v)["original"] == {"file": str(tmp_path / "orig.mp4"), "start": 2.0, "end": 5.0}
    swap.apply(v, tmp_path / "swap2.mp4", 48, {})                  # again: the original stays the first one
    assert swap.replaced(v)["original"]["file"].endswith("orig.mp4")
    assert swap.restore(v) and v.source == {"file": str(tmp_path / "orig.mp4"), "start": 2.0, "end": 5.0}


def test_windows_and_sizes():
    assert A.window_count(30) == 1 and A.window_count(77) == 1 and A.window_count(78) == 2
    assert A.window_count(149) == 2 and A.window_count(150) == 3
    assert A.gen_size(1280, 720, 480) == (848, 480) and A.gen_size(720, 1280, 480) == (480, 848)
    assert A.gen_size(320, 180, 480) == (320, 176)                 # never larger than the clip
    assert "missing" in A.pick_animate_files([], [], [], [], [])


def test_animate_workflow_in_comfy(tmp_path):
    """The workflow as ComfyUI's template (windows chained, the head laid into the original frames)."""
    v = clip(tmp_path / "v.mp4", 160)
    m = tmp_path / "m.mp4"
    swap.write_mask([(100, 40, 160, 120)] * 160, 320, 180, m)
    ref = tmp_path / "ref.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64", "-frames:v", "1",
                    str(ref)], check=True)
    server = FakeComfy(tmp_path / "comfy")
    try:
        an = A.ComfyAnimator(server.url)
        assert an.missing == "" and an.files["relight"] == "WanAnimate_relight_lora_fp16.safetensors"
        out = tmp_path / "out.mp4"
        meta = an.replace(v, m, ref, out, 7, 160, (320, 180), "480p", "")
        wf = server.animated
        assert meta["windows"] == 3 and meta["gen_size"] == [320, 176] and swap.probe(out)[2] == 160
        assert [k for k in wf if k.startswith("anim")] == ["anim0", "anim1", "anim2"]
        assert wf["anim1"]["inputs"]["continue_motion"] == ["new0", 0]
        assert wf["anim2"]["inputs"]["continue_motion"] == ["join1", 0]
        assert wf["anim1"]["inputs"]["video_frame_offset"] == ["anim0", 5] and wf["anim0"]["inputs"]["length"] == 77
        assert wf["made"]["inputs"]["length"] == 160 and wf["compose"]["inputs"]["destination"] == ["src_parts", 0]
        assert wf["video"]["inputs"]["audio"] == ["src_parts", 1]
        assert wf["shift"]["inputs"]["shift"] == 8.0 and wf["sample0"]["inputs"]["steps"] == 6
        assert wf["lora_speed"]["inputs"]["lora_name"].endswith("rank64_bf16.safetensors")
        assert wf["dw_face"]["inputs"]["detect_face"] == "enable" and wf["dw_pose"]["inputs"]["detect_body"] == "enable"
        server.animate = False
        an = A.ComfyAnimator(server.url)
        assert "Wan 2.2 Animate model" in an.missing
        with pytest.raises(Exception, match="Animate"):
            an.replace(v, m, ref, out, 7, 160, (320, 180))
    finally:
        server.close()


def test_pose_cache_key_and_store(tmp_path):
    """The DWPose cache is bound to the exact video file, range, converted clip and size."""
    c = clip(tmp_path / "c.mp4", 10)
    src = {"file": "/v/real.mp4", "start": 1.0, "end": 2.5}
    k = swap.pose_key(c, src, (832, 480))
    assert k == swap.pose_key(c, dict(src), (832, 480))
    assert k != swap.pose_key(c, {**src, "end": 2.6}, (832, 480))           # another range
    assert k != swap.pose_key(c, {**src, "file": "/v/other.mp4"}, (832, 480))
    assert k != swap.pose_key(c, src, (1280, 720))                          # another size
    c2 = clip(tmp_path / "c2.mp4", 11)
    assert k != swap.pose_key(c2, src, (832, 480))                          # another converted clip
    project = tmp_path / "p"
    assert swap.cached_pose(project, k) == {}
    face = tmp_path / "face.mkv"
    face.write_bytes(b"f")
    stored = swap.store_pose(project, k, {"face": face}, {"source": src})       # the face part only (no body)
    assert set(stored) == {"face"} and swap.cached_pose(project, k) == stored and not face.exists()
    pose = tmp_path / "pose.mkv"
    pose.write_bytes(b"p")
    stored = swap.store_pose(project, k, {"pose": pose}, {"source": src})       # the body part later
    assert set(stored) == {"face", "pose"} and stored["pose"].read_bytes() == b"p"


def test_animate_workflow_keeps_and_reuses_the_pose(tmp_path):
    """First run: DWPose runs and its videos are kept (lossless); with them: DWPose is not run again."""
    v = clip(tmp_path / "v.mp4", 30)
    m = tmp_path / "m.mp4"
    swap.write_mask([(100, 40, 160, 120)] * 30, 320, 180, m)
    ref = tmp_path / "ref.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64", "-frames:v", "1",
                    str(ref)], check=True)
    server = FakeComfy(tmp_path / "comfy")
    try:
        an = A.ComfyAnimator(server.url)
        keep = {"pose": tmp_path / "pose.mkv", "face": tmp_path / "face.mkv"}
        meta = an.replace(v, m, ref, tmp_path / "o1.mp4", 1, 30, (320, 180), keep_pose=keep)
        wf = server.animated
        assert "dw_pose" in wf and "dw_face" in wf and not meta["pose_cached"] and meta["pose_made"] == ["pose", "face"]
        assert wf["pose_save"]["inputs"]["format.codec.encoding.crf"] == 0.0
        assert wf["pose_keep"]["inputs"]["chroma_subsampling"] == "444"
        assert all(p.is_file() for p in keep.values())
        meta = an.replace(v, m, ref, tmp_path / "o2.mp4", 1, 30, (320, 180), pose=keep)
        wf = server.animated
        assert meta["pose_cached"] == ["face", "pose"] and "dw_pose" not in wf and "pose_save" not in wf
        assert wf["anim0"]["inputs"]["pose_video"] == ["pose_parts", 0]
        assert wf["anim0"]["inputs"]["face_video"] == ["face_parts", 0]
        assert swap.probe(tmp_path / "o2.mp4")[2] == 30
        # without the body skeleton: no body DWPose, no pose video; only the face
        meta = an.replace(v, m, ref, tmp_path / "o3.mp4", 1, 30, (320, 180), keep_pose={"face": tmp_path / "f2.mkv"},
                          body=False)
        wf = server.animated
        assert "dw_pose" not in wf and "pose_video" not in wf["anim0"]["inputs"] and not meta["body"]
        assert wf["anim0"]["inputs"]["face_video"] == ["dw_face", 0] and meta["pose_made"] == ["face"]
        # the face lock: Promptwood's face swap after the head is laid in, every picture of the person
        meta = an.replace(v, m, ref, tmp_path / "o4.mp4", 1, 30, (320, 180), pose=keep, lock=[ref, ref], enhance=0.8)
        wf = server.animated
        lk = wf["lock"]["inputs"]
        assert wf["lock"]["class_type"] == "VideoChainFaceSwap" and lk["images"] == ["compose", 0]
        assert lk["swapper"] == "hyperswap_1b_256.onnx" and lk["enhancer"] == "GPEN-BFR-512.onnx"
        assert lk["person"] == ["lock_person0", 0] and lk["person_2"] == ["lock_person1", 0]
        assert wf["video"]["inputs"]["images"] == ["lock", 0] and meta["method"] == "animate+lock"
        # face only: no Animate at all
        meta = an.face_only(v, tmp_path / "o5.mp4", [ref], 24.0, 0.0)
        wf = server.faced
        assert "anim0" not in wf and wf["lock"]["inputs"]["images"] == ["src_parts", 0] and meta["method"] == "face"
        assert swap.probe(tmp_path / "o5.mp4")[2] == 30
        server.faceswap = False
        an = A.ComfyAnimator(server.url)
        assert "VideoChainFaceSwap" in an.face_missing and an.missing == ""   # Animate still works without it
        with pytest.raises(Exception, match="VideoChainFaceSwap"):
            an.face_only(v, tmp_path / "o6.mp4", [ref])
    finally:
        server.close()
