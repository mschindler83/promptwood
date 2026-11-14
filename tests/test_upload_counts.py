"""imagegen / animate remove their uploads through the shared reference count (comfy/client.py _drop_uploads)."""
import pytest
from fake_comfy import FakeComfy
from PIL import Image

from video_chain import animate as A
from video_chain import imagegen as G


@pytest.fixture
def server(tmp_path):
    s = FakeComfy(tmp_path / "comfy")
    (s.root / "main.py").write_text("# fake ComfyUI folder\n")
    yield s
    s.close()


@pytest.fixture
def pic(tmp_path):
    p = tmp_path / "pic.png"
    Image.new("RGB", (64, 64), (10, 200, 10)).save(p)
    return p


def _there(server):
    return list((server.root / "input" / "video-chain").glob("*"))


def test_image_create_drops_its_upload_through_the_count(server, pic, tmp_path):
    cr = G.ComfyImageCreator(server.url, comfy_dir=server.root)
    cr.create("krea2", None, "style", tmp_path / "o.png", 5, "standard", 1024, 1024, refs=[pic])
    assert not _there(server)
    name = cr.upload(pic)                        # the same picture again: really sent again, not a stale count
    assert (server.root / "input" / name).is_file()


def test_animator_run_outputs_drops_its_upload_through_the_count(server, pic, tmp_path):
    an = A.ComfyAnimator(server.url, comfy_dir=server.root)
    name = an.upload(pic)
    assert (server.root / "input" / name).is_file()
    wf = {"save": {"class_type": "SaveImage", "inputs": {"images": ["x", 0], "filename_prefix": "video-chain/o"}}}
    try:
        an._run_outputs(wf, {}, [(name, pic)])
    except Exception:
        pass                                      # (the fake may not know this tiny workflow; the cleanup runs anyway)
    assert not _there(server)
    name2 = an.upload(pic)
    assert (server.root / "input" / name2).is_file()


def test_a_fifth_picture_of_the_person_is_said_not_dropped_silently(server, pic, tmp_path):
    import subprocess

    from video_chain.comfy import ComfyError
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24", "-frames:v", "10",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)], check=True)
    an = A.ComfyAnimator(server.url)
    logged = []
    an.log = logged.append
    an.face_only(v, tmp_path / "o.mp4", [pic] * 5, 24.0, 0.0)
    assert any("4 pictures" in line for line in logged)
    assert "person_5" not in server.faced["lock"]["inputs"]
    with pytest.raises(ComfyError, match="at most 4"):
        A.face_lock_nodes({}, ["x", 0], {"swapper": "s", "persons": ["a"] * 5})


def test_a_setting_of_another_shape_is_a_clear_error_not_a_key_error(server, pic, tmp_path):
    from video_chain import comfy as C
    from video_chain import tree
    from video_chain.comfy import settings as CS
    assert CS.DEFAULT_FRAMES == tree.DEFAULT_FRAMES                   # (two copies of one number)
    gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
    base = dict(frames=39, steps=20, width=672, height=384)
    for bad in ({"turbo": {"strength": 1.0}}, {"hyperflow": "x"}, {"pika": 0.5}):
        with pytest.raises(C.ComfyError):
            gen.generate("x", pic, tmp_path / "v.mp4", tmp_path / "v.png", 1, {**base, **bad})
