"""Setup: paths, the manifest, downloads (against a local server), the doctor's file checks, the installer's model
step. Synthetic data only: nothing is fetched from the internet and nothing outside tmp_path is touched."""
import hashlib
import io
import json
import os
import re
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from video_chain import paths
from video_chain.setup import doctor as D
from video_chain.setup import download as DL
from video_chain.setup import installer as I
from video_chain.setup import manifest as M
from video_chain.setup import system as S

SRC = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_CHAIN_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


# ----------------------------------------------------------------------------- no machine assumptions
def test_no_personal_paths_in_the_code():
    bad = re.compile(r"/home/[a-z]|192\.168\.")
    hits = []
    for f in [*SRC.joinpath("src").rglob("*.py"), *SRC.joinpath("comfy_nodes").rglob("*.py"),
              *SRC.joinpath("src").rglob("*.json"), SRC / "install.sh", SRC / "video-chain", SRC / "pyproject.toml"]:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if bad.search(line):
                hits.append(f"{f.relative_to(SRC)}:{n}: {line.strip()[:80]}")
    assert not hits, "\n".join(hits)


def test_paths_defaults_config_and_env(home, monkeypatch):
    assert paths.comfy_dir() == home / "ComfyUI"
    assert paths.hf_cache() == home / "hf-models"
    assert paths.tts_python() == home / "tts-venv" / "bin" / "python"
    assert paths.comfy_url() == "http://127.0.0.1:8188"
    paths.save_config(comfy_dir="/opt/ComfyUI", comfy_url="http://127.0.0.1:8190/")
    assert paths.comfy_dir() == Path("/opt/ComfyUI") and paths.comfy_url() == "http://127.0.0.1:8190"
    paths.save_config(comfy_dir=None)
    assert paths.comfy_dir() == home / "ComfyUI"
    monkeypatch.setenv("VIDEO_CHAIN_COMFY_DIR", "~/elsewhere")
    assert paths.comfy_dir() == Path.home() / "elsewhere"
    assert paths.model_cache_dirs()[0] == home / "hf-models"


def test_llama_server_prefers_the_installed_one(home, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert paths.llama_server() == str(home / "tools" / "llama.cpp" / "llama-server")     # (named, though missing)
    (tmp_path / "bin").mkdir()
    on_path = tmp_path / "bin" / "llama-server"
    on_path.write_text("#!/bin/sh\n")
    on_path.chmod(0o755)
    assert paths.llama_server() == str(on_path)
    ours = home / "tools" / "llama.cpp" / "llama-server"
    ours.parent.mkdir(parents=True)
    ours.write_text("")
    assert paths.llama_server() == str(ours)


def test_private_tools_come_first_on_path(home, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    paths.use_private_tools()
    assert os.environ["PATH"] == "/usr/bin"                 # (no folder: nothing added)
    paths.tools_bin().mkdir(parents=True)
    paths.use_private_tools()
    paths.use_private_tools()
    assert os.environ["PATH"].split(os.pathsep) == [str(paths.tools_bin()), "/usr/bin"]


# ----------------------------------------------------------------------------- manifest
def test_manifest_is_complete_and_checked():
    man = M.load()
    assert M.profile_components(man, "core") == ["video"]
    names = set()
    for cid, c in man["components"].items():
        assert c["profile"] in M.PROFILES and c["files"] and c["bytes"] == sum(f["size"] for f in c["files"])
        for f in c["files"]:
            assert f["size"] > 0
            if "url" not in f:
                assert re.fullmatch(r"[0-9a-f]{40}", f["rev"]), (cid, f)
                if f["size"] > 50 * 2**20:           # (every model; small text files are pinned by rev)
                    assert re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")), (cid, f["path"])
                assert M.file_url(f).startswith(f"https://huggingface.co/{f['repo']}/resolve/{f['rev']}/")
            if not f.get("hf"):
                for t in M.file_targets(f, Path("/c"), Path("/h")):
                    assert str(t) not in names, f"two files for {t}"
                    names.add(str(t))
        for n in c.get("nodes", []):
            assert n in man["nodes"]
        for n in c.get("software", []):
            assert n in man["software"]


def test_manifest_has_the_files_the_app_asks_for():
    from video_chain import comfy, tts
    man = M.load()
    video = {f["name"] for f in man["components"]["video"]["files"]}
    assert video == {comfy.UNET, comfy.TEXT_ENCODER, comfy.VIDEO_VAE, comfy.AUDIO_VAE}
    assert man["components"]["video-refs"]["files"][0]["name"] == comfy.UNET_REF
    assert {f["repo"] for f in man["components"]["tts"]["files"]} == {tts.QWEN_BASE, tts.QWEN_DESIGN, tts.WHISPER}
    # the app's own pickers find what the manifest installs
    names = lambda cid, d: [f["name"] for f in man["components"][cid]["files"] if f.get("dir") == d]   # noqa: E731
    q = comfy.pick_qwen21_files(names("image", "diffusion_models"), names("image", "text_encoders"),
                                names("image", "vae"))
    assert q and q["base"] and q["turbo"]
    assert comfy.pick_qwen21_inpaint(names("image", "model_patches"))
    r = comfy.pick_refine_files(names("video-2k", "loras"), names("video-2k", "latent_upscale_models"))
    assert r["refiner"] and r["upscaler"]
    s = comfy.pick_seedvr_files(names("upscale", "diffusion_models"), names("upscale", "vae"), "7b")
    assert s["unet"] and s["vae"]
    assert all(comfy.is_turbo_lora(Path(n)) for n in names("video-turbo", "loras"))
    from video_chain import animate, imagegen
    a = animate.pick_animate_files(names("headswap-wan", "diffusion_models"), names("headswap-wan", "text_encoders"),
                                   names("headswap-wan", "vae"), names("headswap-wan", "loras"),
                                   names("headswap-wan", "clip_vision"))
    assert "missing" not in a
    k = imagegen.pick_files(names("image-krea2", "diffusion_models"), [], names("image-krea2", "text_encoders"),
                            names("image-krea2", "vae"), names("image-krea2", "loras"))["krea2"]
    assert "missing" not in k and k["style"]
    llm = [f["path"] for f in man["components"]["llm"]["files"]]
    assert any("mmproj" in p for p in llm) and any("instruct" in p.lower() and "mmproj" not in p for p in llm)


def test_select_adds_what_parts_need():
    man = M.load()
    assert M.select(man, "core") == ["video"]
    assert M.select(man, only=["headswap-h3"]) == ["video", "video-refs", "faces", "headswap-h3"]
    std = M.select(man, "standard")
    assert "video-speed" not in std and {"video", "image", "tts", "llm", "faces"} <= set(std)
    assert "tts" not in M.select(man, "standard", without=["tts"])
    assert "image-krea2" in M.select(man, "standard", add=["image-krea2"])
    assert set(M.select(man, "full")) == set(man["components"])
    with pytest.raises(ValueError):
        M.select(man, only=["nope"])


# ----------------------------------------------------------------------------- downloads
class _Server:
    """A local file server with Range support, that can cut a response short once."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.cut, self.ranges = files, {}, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                data = outer.files.get(self.path)
                if data is None:
                    self.send_error(404)
                    return
                start = 0
                rng = self.headers.get("Range")
                outer.ranges.append((self.path, rng))
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_error(416)
                        return
                    self.send_response(206)
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                cut = outer.cut.pop(self.path, None)
                self.wfile.write(body[:cut] if cut else body)
                if cut:
                    self.wfile.flush()
                    self.connection.close()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(DL.time, "sleep", lambda s: None)
    s = _Server({})
    yield s
    s.close()


def test_fetch_checks_and_resumes(server, tmp_path):
    data = os.urandom(300_000)
    sha = hashlib.sha256(data).hexdigest()
    server.files["/a.bin"] = data
    dest = tmp_path / "x" / "a.bin"
    DL.fetch(server.url + "/a.bin", dest, len(data), sha)
    assert dest.read_bytes() == data and not dest.with_name("a.bin.part").exists()
    n = len(server.ranges)
    DL.fetch(server.url + "/a.bin", dest, len(data), sha)                 # there and whole: no request
    assert len(server.ranges) == n
    # a connection that dies in the middle is continued from what arrived
    dest.unlink()
    server.cut["/a.bin"] = 100_000
    DL.fetch(server.url + "/a.bin", dest, len(data), sha)
    assert dest.read_bytes() == data
    assert server.ranges[-1] == ("/a.bin", "bytes=100000-")
    # a part left by an earlier run
    dest.unlink()
    dest.with_name("a.bin.part").write_bytes(data[:250_000])
    DL.fetch(server.url + "/a.bin", dest, len(data), sha)
    assert dest.read_bytes() == data and server.ranges[-1] == ("/a.bin", "bytes=250000-")


def test_fetch_refuses_damaged_and_missing_files(server, tmp_path):
    data = b"x" * 5000
    server.files["/a.bin"] = data
    with pytest.raises(DL.DownloadError, match="damaged"):
        DL.fetch(server.url + "/a.bin", tmp_path / "a.bin", len(data), "0" * 64)
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()
    with pytest.raises(DL.DownloadError, match="404"):
        DL.fetch(server.url + "/none", tmp_path / "b.bin", 10)
    with pytest.raises(DL.DownloadError):                                 # shorter than promised, every time
        DL.fetch(server.url + "/a.bin", tmp_path / "c.bin", len(data) + 10, tries=2)


def test_fetch_all_reports_the_first_error(server, tmp_path):
    server.files["/a"], server.files["/b"] = b"a" * 100, b"b" * 200
    jobs = [{"url": server.url + "/a", "dest": tmp_path / "a", "size": 100},
            {"url": server.url + "/b", "dest": tmp_path / "b", "size": 200}]
    DL.fetch_all(jobs, workers=2)
    assert (tmp_path / "a").read_bytes() == b"a" * 100 and (tmp_path / "b").stat().st_size == 200
    with pytest.raises(DL.DownloadError):
        DL.fetch_all([{"url": server.url + "/zzz", "dest": tmp_path / "z", "size": 5}])


def test_extract_strips_picks_and_stays_inside(tmp_path):
    tar = tmp_path / "t.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for name, data in (("top/main.py", b"m"), ("top/sub/x.txt", b"x"), ("top/../../evil", b"e")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        exe = tarfile.TarInfo("top/bin/tool")
        exe.size, exe.mode = 1, 0o755
        t.addfile(exe, io.BytesIO(b"t"))
    out = tmp_path / "out"
    DL.extract(tar, out, strip=1)
    assert (out / "main.py").read_bytes() == b"m" and (out / "sub" / "x.txt").is_file()
    assert os.access(out / "bin" / "tool", os.X_OK)
    assert not (tmp_path / "evil").exists() and not (out.parent.parent / "evil").exists()
    z = tmp_path / "z.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("pack/det.onnx", b"d")
        f.writestr("pack/other.onnx", b"o")
        f.writestr("../up.onnx", b"u")
    DL.extract(z, tmp_path / "faces", members=["det.onnx", "up.onnx"])
    assert sorted(p.name for p in (tmp_path / "faces").iterdir()) == ["det.onnx", "up.onnx"]   # flat, inside
    assert not (tmp_path / "up.onnx").exists()


# ----------------------------------------------------------------------------- doctor and installer on a fake setup
def _mini_manifest(server) -> dict:
    files = {"/m/unet.safetensors": os.urandom(4000), "/m/lora.safetensors": os.urandom(3000),
             "/hf/config.json": b'{"a": 1}', "/hf/model.gguf": os.urandom(2000)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("det_10g.onnx", b"detector")
        z.writestr("w600k_r50.onnx", b"recognizer")
        z.writestr("genderage.onnx", b"unused")
    files["/pack.zip"] = buf.getvalue()
    server.files.update(files)
    sha = lambda p: hashlib.sha256(files[p]).hexdigest()   # noqa: E731
    rev = "a" * 40
    comps = {
        "video": {"title": "Video", "profile": "core", "files": [
            {"url": server.url + "/m/unet.safetensors", "dir": "diffusion_models", "name": "unet.safetensors",
             "size": 4000, "sha256": sha("/m/unet.safetensors"),
             "licence": {"id": "other", "name": "Test Licence", "link": "https://example.org/terms", "open": False}}]},
        "extra": {"title": "Extra", "profile": "standard", "needs": ["video"], "files": [
            {"url": server.url + "/m/lora.safetensors", "dir": "loras", "name": "lora.safetensors", "size": 3000,
             "sha256": sha("/m/lora.safetensors")},
            {"url": server.url + "/pack.zip", "dir": "insightface/models/buffalo_l",
             "extract": ["det_10g.onnx", "w600k_r50.onnx"], "size": len(files["/pack.zip"])},
            {"url": server.url + "/hf/config.json", "repo": "org/llm", "rev": rev, "path": "config.json", "hf": True,
             "size": 8},
            {"url": server.url + "/hf/model.gguf", "repo": "org/llm", "rev": rev, "path": "model.gguf", "hf": True,
             "size": 2000, "sha256": sha("/hf/model.gguf")}]},
    }
    for c in comps.values():
        c["bytes"] = sum(f["size"] for f in c["files"])
        for f in c["files"]:
            f.setdefault("licence", {"id": "mit", "name": "mit", "link": "https://example.org/mit", "open": True})
    return {"comfy": {"git": "x", "rev": "r", "python": "3.12"}, "software": {}, "nodes": {}, "components": comps}


def test_installer_downloads_models_and_doctor_sees_them(server, home, monkeypatch, capsys):
    man = _mini_manifest(server)
    monkeypatch.setattr(M, "load", lambda: man)
    comfy = home / "ComfyUI"
    (comfy / "models").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    inst = I.Installer(I.Options(yes=True, accept_licenses=True))
    assert inst.components == ["video", "extra"]
    assert len(inst.downloads()) == 5
    assert D.component_state(man, "extra", comfy, paths.hf_cache()) == (0, 4, man["components"]["extra"]["bytes"])
    need = inst.show_plan()
    assert need >= sum(c["bytes"] for c in man["components"].values())
    inst.licences()
    out = capsys.readouterr().out
    assert "Test Licence" in out and "https://example.org/terms" in out and "extra: Apache-2.0 / MIT" in out
    inst.model_files()
    assert (comfy / "models" / "diffusion_models" / "unet.safetensors").stat().st_size == 4000
    assert (comfy / "models" / "insightface" / "models" / "buffalo_l" / "det_10g.onnx").read_bytes() == b"detector"
    assert not (comfy / "models" / "insightface" / "models" / "buffalo_l" / "genderage.onnx").exists()
    snap = paths.hf_cache() / "models--org--llm" / "snapshots" / ("a" * 40)
    assert (snap / "model.gguf").stat().st_size == 2000
    assert (paths.hf_cache() / "models--org--llm" / "refs" / "main").read_text() == "a" * 40
    from video_chain import tts
    assert tts.snapshot("org/llm") == snap                       # the app's own lookup finds the snapshot
    assert inst.downloads() == []                                # a second run has nothing to do
    requests = len(server.ranges)
    inst.model_files()
    assert len(server.ranges) == requests
    for cid in ("video", "extra"):
        have, total, left = D.component_state(man, cid, comfy, paths.hf_cache())
        assert have == total and left == 0
    checks = D.check_components(man, ["video", "extra"])
    assert [c.status for c in checks] == [D.OK, D.OK]
    # a file of the wrong size counts as missing, with the command that repairs it
    (comfy / "models" / "loras" / "lora.safetensors").write_bytes(b"short")
    bad = D.check_components(man, ["video", "extra"])[1]
    assert bad.status == D.FAIL and "--only extra" in bad.fix
    other = D.check_components(man, ["video"])[1]
    assert other.status == D.INFO and "--with extra" in other.fix


def test_doctor_finds_models_in_extra_model_paths(server, home, tmp_path, monkeypatch):
    man = _mini_manifest(server)
    comfy, shared = home / "ComfyUI", tmp_path / "shared"
    (comfy / "models").mkdir(parents=True)
    (shared / "unet").mkdir(parents=True)
    (shared / "unet" / "unet.safetensors").write_bytes(b"0" * 4000)
    (comfy / "extra_model_paths.yaml").write_text(f"mine:\n    base_path: {shared}\n    diffusion_models: unet/\n")
    f = man["components"]["video"]["files"][0]
    assert D.find_file(f, comfy, paths.hf_cache()) == [shared / "unet" / "unet.safetensors"]


def test_only_steps_keep_the_installed_parts(server, home, monkeypatch):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    paths.save_config(components="video")
    inst = I.Installer(I.Options(steps=["nodes"]))
    assert inst.components == ["video"] and not inst.models and inst.wants("nodes") and not inst.wants("system")
    with pytest.raises(ValueError, match="unknown step"):
        I.Installer(I.Options(steps=["video"]))
    with pytest.raises(ValueError, match="unknown component"):
        I.Installer(I.Options(only=["nodes"]))
    inst = I.Installer(I.Options(only=["extra"]))
    assert inst.components == ["video", "extra"] and inst.models and inst.wants("nodes")


def test_preflight_names_what_blocks(server, home, monkeypatch):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    monkeypatch.setattr(S, "gpu", lambda: None)
    monkeypatch.setattr(S, "qt_missing_libs", lambda: None)
    inst = I.Installer(I.Options(yes=True))
    inst.uv = None
    with pytest.raises(I.SetupError) as e:
        inst.preflight(10**18)
    text = str(e.value)
    assert "uv was not found" in text and "NVIDIA" in text and "not enough disk space" in text
    monkeypatch.setattr(S, "gpu", lambda: {"name": "Card", "vram_mb": 24000, "free_mb": 23000, "driver": "600.1"})
    inst.uv = "/usr/bin/uv"
    inst.preflight(1000)                                         # nothing blocks


def test_llama_build_for_this_computer(server, home, monkeypatch):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    monkeypatch.setattr(I.platform, "system", lambda: "Linux")
    monkeypatch.setattr(I.platform, "machine", lambda: "x86_64")
    names = ["llama-b1-bin-ubuntu-x64.tar.gz", "llama-b1-bin-ubuntu-vulkan-x64.tar.gz",
             "llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz", "llama-b1-bin-ubuntu-cuda-13.5-x64.tar.gz",
             "cudart-llama-b1-bin-ubuntu-cuda-13.5-x64.tar.gz", "cudart-llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz",
             "llama-b1-bin-ubuntu-cuda-13.5-arm64.tar.gz", "llama-b1-bin-macos-arm64.tar.gz"]
    assets = [{"name": n} for n in names]
    inst = I.Installer(I.Options())
    card = {"name": "Card", "vram_mb": 24000, "free_mb": 23000, "driver": "615.78"}
    monkeypatch.setattr(S, "gpu", lambda: card)
    assert [a["name"] for a in inst.llama_asset(assets)] == ["llama-b1-bin-ubuntu-cuda-13.5-x64.tar.gz",
                                                             "cudart-llama-b1-bin-ubuntu-cuda-13.5-x64.tar.gz"]
    card["driver"] = "570.10"
    assert inst.llama_asset(assets)[0]["name"] == "llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz"
    assert inst.torch_index().endswith("/cu126")
    monkeypatch.setattr(S, "gpu", lambda: None)
    assert [a["name"] for a in inst.llama_asset(assets)] == ["llama-b1-bin-ubuntu-vulkan-x64.tar.gz"]
    assert inst.torch_index().endswith("/cpu")
    with pytest.raises(I.SetupError):
        inst.llama_asset([{"name": "llama-b1-bin-win-cpu-x64.zip"}])


def test_report_shows_fixes_only_for_what_is_wrong():
    checks = [D.Check("A", "one", D.OK, "fine", "never shown"), D.Check("A", "two", D.FAIL, "broken", "do this"),
              D.Check("B", "three", D.INFO, "optional", "add it")]
    text = D.report(checks)
    assert "[ ok ] one: fine" in text and "never shown" not in text
    assert "[FAIL] two: broken\n         -> do this" in text and "-> add it" in text
    assert D.failed(checks) and "1 problem to fix" in D.summary(checks)
    assert D.summary(checks[:1]) == "Everything is in place."


def test_install_hints_follow_the_package_manager():
    assert S.install_hint("ffmpeg", {"manager": "apt"}) == "sudo apt install -y ffmpeg"
    assert "RPM Fusion" in S.install_hint("ffmpeg", {"manager": "dnf"})
    assert "xcb-util-cursor" in S.install_hint("qt", {"manager": "dnf"})
    assert "ffmpeg.org" in S.install_hint("ffmpeg", {"manager": None})
    assert "ubuntu-drivers" in S.install_hint("nvidia", {"manager": "apt"})


def test_cli_dispatches_setup_and_doctor(home, monkeypatch, capsys):
    from video_chain import __main__ as cli
    from video_chain.setup import __main__ as setup_cli
    seen = {}
    monkeypatch.setattr(setup_cli, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert cli.main(["setup", "--dry-run"]) == 0 and seen["argv"] == ["setup", "--dry-run"]
    a = setup_cli.build_parser().parse_args(["setup", "--only", "tts, llm", "--with", "image-krea2", "--no-test"])
    assert a.only == ["tts", "llm"] and a.add == ["image-krea2"] and a.test is False and a.profile is None
    assert setup_cli.build_parser().parse_args(["setup", "--steps", "comfyui,nodes"]).steps == ["comfyui", "nodes"]


def test_setup_check_dialog_shows_the_report(home, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(D, "check_all", lambda components=None: [D.Check("G", "thing", D.FAIL, "gone", "fix it")])
    from video_chain.ui.setup_check import SetupCheckDialog
    dlg = SetupCheckDialog()
    dlg._thread.wait(10000)
    app.processEvents()
    assert "[FAIL] thing: gone" in dlg.text.toPlainText() and "fix it" in dlg.text.toPlainText()
    assert "1 problem" in dlg.head.text()
    dlg.close()
    dlg.deleteLater()


def test_another_build_of_a_model_counts(home, tmp_path):
    """What the app's pickers accept is not downloaded again: other precisions, names, subfolders."""
    man = M.load()
    comfy = home / "ComfyUI"
    have = {"model_patches/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors": "image",
            "loras/h3/minimax_h3_turbo_v4_step600.safetensors": "video-turbo",
            "loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors": "video-2k",
            "diffusion_models/qwen_image_2.1_bf16.safetensors": "image",
            "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors": "image",
            "facerestore_models/GPEN-BFR-1024.onnx": "faces"}
    for rel in have:
        (comfy / "models" / rel).parent.mkdir(parents=True, exist_ok=True)
        (comfy / "models" / rel).write_bytes(b"x")
    D.forget_listings()
    found = {}
    for cid in set(have.values()):
        for f in man["components"][cid]["files"]:
            hit = D.find_file(f, comfy, paths.hf_cache())
            if hit:
                found[f.get("name")] = hit[0].name
    assert found["qwen_image_2.1_int8_convrot.safetensors"] == "qwen_image_2.1_bf16.safetensors"      # not the turbo
    assert found["qwen_image_2.1_turbo_int8_convrot.safetensors"] == "qwen_image_2.1_turbo_bf16.safetensors"
    assert found["qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"].startswith("Qwen-Image-2.1-Fun")
    assert found["GPEN-BFR-512.onnx"] == "GPEN-BFR-1024.onnx"
    assert len([n for n in found if "turbo" in n and "minimax" in n]) == 2
    assert "open_h3_regenerate_2k_refiner_lora.safetensors" in found
    assert "qwen3vl_8b_int8_convrot.safetensors" not in found            # (nothing like it is there)
    # a language model with its vision part in any Hugging Face cache does for the prompt refiner
    snap = tmp_path / "cache" / "models--me--llm" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "my-model-Q6.gguf").write_bytes(b"m")
    llm = man["components"]["llm"]["files"]
    assert all(D.find_file(f, comfy, tmp_path / "cache") is None for f in llm)        # no projector: cannot see
    (snap / "mmproj-F16.gguf").write_bytes(b"p")
    assert sorted(D.find_file(f, comfy, tmp_path / "cache")[0].name for f in llm) == ["mmproj-F16.gguf",
                                                                                      "my-model-Q6.gguf"]


def _answers(monkeypatch, *replies):
    left = list(replies)
    asked = []

    def fake(question=""):
        asked.append(question)
        return left.pop(0)
    monkeypatch.setattr("builtins.input", fake)
    return asked


def test_setup_asks_for_a_comfyui_that_is_already_there(server, home, tmp_path, monkeypatch, capsys):
    man = _mini_manifest(server)
    monkeypatch.setattr(M, "load", lambda: man)
    monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
    mine = tmp_path / "MyComfy"
    (mine / "models" / "diffusion_models").mkdir(parents=True)
    (mine / "main.py").write_text("")
    (mine / "models" / "diffusion_models" / "unet.safetensors").write_bytes(b"0" * 4000)
    # 2: a new ComfyUI that reads the models of the old one (a wrong folder first)
    inst = I.Installer(I.Options())
    inst.interactive = True
    asked = _answers(monkeypatch, "2", str(tmp_path / "nothing"), "y", str(mine))
    inst.ask_existing()
    assert inst.o.models_from == [str(mine)] and inst.o.comfy_dir is None and len(asked) == 4
    assert "not a ComfyUI folder" in capsys.readouterr().out
    inst.share_models()
    assert [j["file"]["name"] for j in inst.downloads() if "name" in j["file"]] == ["lora.safetensors"]
    assert not list((mine / "models").rglob("*.yaml")) and sorted(p.name for p in mine.iterdir()) == ["main.py",
                                                                                                      "models"]
    # 3: the old one itself; it has no .venv, so its Python is asked for
    inst = I.Installer(I.Options())
    inst.interactive = True
    _answers(monkeypatch, "3", str(mine), "/usr/bin/python3")
    inst.ask_existing()
    assert inst.comfy_dir == mine and inst.o.comfy_python == "/usr/bin/python3" and inst.o.models_from == []
    # 1 (the default, by Enter): a new one; and nothing is asked without a terminal or when the options said it
    inst = I.Installer(I.Options())
    inst.interactive = True
    _answers(monkeypatch, "")
    inst.ask_existing()
    assert inst.o.comfy_dir is None and inst.o.models_from == []
    for opts in (I.Options(yes=True), I.Options(models_from=[str(mine)])):
        inst = I.Installer(opts)
        inst.interactive = not opts.yes
        _answers(monkeypatch)                                # (any question would fail: no answers left)
        inst.ask_existing()


def test_setup_offers_the_users_hugging_face_cache(server, home, tmp_path, monkeypatch):
    man = _mini_manifest(server)
    man["components"]["extra"]["software"] = ["llama"]
    man["software"] = {"llama": {}}
    monkeypatch.setattr(M, "load", lambda: man)
    monkeypatch.setenv("HOME", str(tmp_path / "userhome"))
    cache = tmp_path / "hub"
    snap = cache / "models--org--llm" / "snapshots" / "other-rev"
    snap.mkdir(parents=True)
    (snap / "model.gguf").write_bytes(b"0" * 2000)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    inst = I.Installer(I.Options())
    inst.interactive = True
    asked = _answers(monkeypatch, "1", "y")
    inst.ask_existing()
    assert "Hugging Face cache" not in asked[0] and inst.o.hf_cache == str(cache) and paths.hf_cache() == cache
    assert "model.gguf" not in [j["file"].get("path") for j in inst.downloads()]


def test_hugging_face_token_is_found_and_stays_with_its_host(server, tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert DL.hf_token() is None
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "token").write_text("hf_from_login\n")            # what `hf auth login` writes
    assert DL.hf_token() == "hf_from_login"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert DL.hf_token() == "hf_env"
    assert DL._token_host("https://huggingface.co/a/b") and not DL._token_host("https://huggingface.co.evil.org/x")
    assert not DL._token_host("https://cdn.example.com/huggingface.co/x")
    # sent to its own host, dropped when the download is redirected to another one
    seen = []
    other = _Server({"/blob": b"data"})
    handler = server.httpd.RequestHandlerClass
    plain_get = handler.do_GET

    def do_get(self):
        seen.append(("first", self.headers.get("Authorization")))
        if self.path == "/file":
            self.send_response(302)
            self.send_header("Location", other.url.replace("127.0.0.1", "localhost") + "/blob")
            self.end_headers()
        else:
            plain_get(self)
    other_get = other.httpd.RequestHandlerClass.do_GET

    def other_do_get(self):
        seen.append(("storage", self.headers.get("Authorization")))
        other_get(self)
    monkeypatch.setattr(handler, "do_GET", do_get)
    monkeypatch.setattr(other.httpd.RequestHandlerClass, "do_GET", other_do_get)
    monkeypatch.setattr(DL, "TOKEN_HOSTS", ("127.0.0.1",))
    try:
        DL.fetch(server.url + "/file", tmp_path / "f.bin", 4, token="hf_secret")
    finally:
        other.close()
    assert (tmp_path / "f.bin").read_bytes() == b"data"
    assert seen == [("first", "Bearer hf_secret"), ("storage", None)]
    seen.clear()
    monkeypatch.setattr(DL, "TOKEN_HOSTS", ("huggingface.co",))
    server.files["/plain"] = b"abcd"
    DL.fetch(server.url + "/plain", tmp_path / "g.bin", 4, token="hf_secret")     # another site: no token at all
    assert seen == [("first", None)]


def test_plan_suggests_a_token_only_when_it_helps(home, monkeypatch, capsys, tmp_path):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "none"))
    inst = I.Installer(I.Options(profile="core"))
    inst.show_plan()
    out = capsys.readouterr().out
    assert "no token found" in out and "huggingface.co/settings/tokens" in out
    monkeypatch.setenv("HF_TOKEN", "hf_abc123")
    inst = I.Installer(I.Options(profile="core"))
    inst.show_plan()
    out = capsys.readouterr().out
    assert "your token is used" in out and "hf_abc123" not in out


def test_every_download_carries_its_publishers_licence(home, capsys):
    """The licence shown is the publisher's own statement, for every part; setup claims nothing itself."""
    man = M.load()
    assert re.fullmatch(r"\d{4}-\d\d-\d\d", man["checked"])
    for cid, c in man["components"].items():
        for f in c["files"]:
            lic = f["licence"]
            assert lic["name"] and lic["link"].startswith("https://") and isinstance(lic["open"], bool), (cid, f)
    lics = lambda cid: {f["licence"]["name"]: f["licence"]["open"] for f in man["components"][cid]["files"]}   # noqa: E731
    assert not any(lics("video").values()) and not any(lics("image").values())        # research / community terms
    assert any("non-commercial" in n for n in lics("faces")) and any("no licence stated" in n for n in lics("faces"))
    assert all(lics("upscale").values()) and all(lics("tts").values())
    inst = I.Installer(I.Options(accept_licenses=True))
    inst.licences()
    out = capsys.readouterr().out
    assert "Most are" not in out and "setup cannot judge" in out and man["checked"] in out
    assert "non-commercial research purposes only" in out and "no licence stated by the publisher" in out
    assert "qwen-research" in out and "upscale, tts, llm: Apache-2.0 / MIT" in out
    inst = I.Installer(I.Options())                               # nobody to ask, not accepted: nothing happens
    inst.interactive = False
    with pytest.raises(I.SetupError, match="not accepted"):
        inst.licences()


def test_setup_asks_how_much_to_install(server, home, monkeypatch, capsys):
    man = _mini_manifest(server)
    monkeypatch.setattr(M, "load", lambda: man)
    comfy = home / "ComfyUI" / "models" / "diffusion_models"
    comfy.mkdir(parents=True)
    (comfy / "unet.safetensors").write_bytes(b"0" * 4000)
    inst = I.Installer(I.Options())
    inst.interactive = True
    _answers(monkeypatch, "1")
    inst.ask_profile()
    out = capsys.readouterr().out
    assert inst.o.profile == "core" and inst.components == ["video"]
    assert "all already on this computer" in out and "to download" in out and "--with NAME" in out
    inst = I.Installer(I.Options())
    inst.interactive = True
    _answers(monkeypatch, "")                                     # Enter: the recommended one
    inst.ask_profile()
    assert inst.o.profile == "standard" and inst.components == ["video", "extra"]
    for opts in (I.Options(profile="standard"), I.Options(only=["video"]), I.Options(yes=True)):
        inst = I.Installer(opts)
        inst.interactive = not opts.yes
        _answers(monkeypatch)                                     # (nothing may be asked)
        inst.ask_profile()


def test_answers_travel_from_the_asking_phase_to_the_install(server, home, monkeypatch, tmp_path):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    first = I.Installer(I.Options(phase="ask"))
    first.o.profile, first.o.models_from = "core", [str(tmp_path)]
    first.save_answers()
    second = I.Installer(I.Options(answers=True))
    second.interactive = True
    assert second.load_answers() and second.o.profile == "core" and second.components == ["video"]
    assert second.o.models_from == [str(tmp_path)] and second.o.accept_licenses and not second.interactive
    assert not second.answers_file().exists() and not second.load_answers()        # read once


def test_steps_are_counted_before_they_start(server, home, monkeypatch):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    monkeypatch.setattr(S, "ffmpeg_ok", lambda st=None: False)
    inst = I.Installer(I.Options(yes=True))
    names = [n for n, _ in inst.pending()]
    assert names == ["ffmpeg", "comfyui", "models", "desktop", "verify"] and inst._headed == 4
    monkeypatch.setattr(S, "ffmpeg_ok", lambda st=None: True)
    (home / "ComfyUI").mkdir(parents=True)
    (home / "ComfyUI" / "main.py").write_text("")
    paths.save_config(components="video")
    inst = I.Installer(I.Options(steps=["nodes"]))
    assert [n for n, _ in inst.pending()] == ["comfyui", "desktop", "verify"] and inst._headed == 1


def test_tools_run_quietly_into_the_log_and_show_why_they_failed(home, capsys):
    from video_chain.setup import console as C
    import sys as _sys
    log = C.open_log()
    C.plan_steps(2)
    C.step("First")
    C.run([_sys.executable, "-c", "print('chatter ' * 3)"], "Doing a thing", expect="quick")
    out = capsys.readouterr().out
    assert "[1/2] First" in out and "Doing a thing … (quick)" in out and "Doing a thing: done in" in out
    assert "chatter" not in out and "chatter chatter" in log.read_text()
    with pytest.raises(C.CommandFailed) as e:
        C.run([_sys.executable, "-c", "import sys; print('the real reason'); sys.exit(3)"], "Breaking")
    assert "exit code 3" in str(e.value) and "the real reason" in str(e.value)
    assert "Breaking: failed after" in capsys.readouterr().out
    assert C.short(str(Path.home() / "x")) == "~/x" and C.clock(75) == "1m15s" and C.clock(3700) == "1h01m"


def test_plan_wording(server, home, monkeypatch, capsys):
    man = _mini_manifest(server)
    monkeypatch.setattr(M, "load", lambda: man)
    (home / "ComfyUI" / "models" / "diffusion_models").mkdir(parents=True)
    (home / "ComfyUI" / "models" / "diffusion_models" / "unet.safetensors").write_bytes(b"0" * 4000)
    inst = I.Installer(I.Options(yes=True))
    inst.show_plan()
    out = capsys.readouterr().out
    assert "already on this computer" in out and "(there)" not in out and "to download" in out
    assert "a new one will be installed" in out and "programs: about" in out


def test_package_installs_survive_a_network_hiccup(server, home, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    monkeypatch.setattr(I.time, "sleep", lambda s: None)
    calls = []

    def fake_run(cmd, what, env=None, cwd=None, expect=""):
        calls.append(what)
        if len(calls) < 3:
            raise I.CommandFailed("x failed. Its last lines:\n error: Failed to download `pkg`\n operation timed out")
    monkeypatch.setattr(I, "run", fake_run)
    inst = I.Installer(I.Options(yes=True))
    inst.uv = "uv"
    inst.pip(tmp_path / "bin" / "python", ["torch"], "Installing PyTorch")
    assert len(calls) == 3 and "Trying again (3 of 4)" in capsys.readouterr().out
    calls.clear()

    def broken(cmd, what, env=None, cwd=None, expect=""):
        calls.append(what)
        raise I.CommandFailed("x failed. Its last lines:\n error: No solution found when resolving dependencies")
    monkeypatch.setattr(I, "run", broken)
    with pytest.raises(I.CommandFailed):                       # not the network: no point in trying again
        inst.pip(tmp_path / "bin" / "python", ["torch"], "Installing PyTorch")
    assert len(calls) == 1


def test_a_foreign_comfyui_on_the_port_is_left_alone(home, tmp_path):
    """A ComfyUI that answers at the address but was not started from setup's folder is never used for tests."""
    ours = home / "ComfyUI"
    ours.mkdir(parents=True)
    (ours / ".video-chain-setup").write_text("rev\n")
    assert D.own_comfy_dir() == ours
    hits = []

    class H(BaseHTTPRequestHandler):
        argv: list = []

        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(self.path)
            body = json.dumps({"system": {"argv": self.argv}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}"
    try:
        H.argv = ["main.py", "--port", "8188"]                       # somebody else's
        with pytest.raises(RuntimeError, match="left alone"):
            D.ComfySession(url, expect_dir=ours, log=lambda t: None).__enter__()
        assert set(hits) == {"/system_stats"}                        # asked who it is, nothing else
        H.argv = ["main.py", "--port", "8189", "--base-directory", str(ours)]
        with D.ComfySession(url, expect_dir=ours, log=lambda t: None) as s:
            assert s.proc is None
        (ours / ".video-chain-setup").unlink()
        assert D.own_comfy_dir() is None                             # a ComfyUI the user brought: theirs to use
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_port_choice_survives_a_run_that_stops(server, home, monkeypatch):
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    monkeypatch.setattr(I, "_port_busy", lambda port: port == 8188)
    inst = I.Installer(I.Options(yes=True))
    inst.uv = "uv"
    monkeypatch.setattr(I, "fetch", lambda *a, **k: (_ for _ in ()).throw(I.DownloadError("network down")))
    with pytest.raises(I.DownloadError):
        inst.comfyui()                                               # stops while fetching ComfyUI
    assert paths.comfy_url() == "http://127.0.0.1:8189"
    monkeypatch.setattr(I, "_port_busy", lambda port: False)        # (later the other program is gone)
    again = I.Installer(I.Options(yes=True))
    again.uv = "uv"
    with pytest.raises(I.DownloadError):
        again.comfyui()
    assert paths.comfy_url() == "http://127.0.0.1:8189"             # the first choice stands


def test_nvidia_smi_is_optional(home, tmp_path, monkeypatch, server):
    """Without nvidia-smi the driver tells card and version; the memory is checked once PyTorch can be asked."""
    monkeypatch.setattr(S.shutil, "which", lambda name: None)
    proc = tmp_path / "nvidia"
    assert S.gpu_from_driver(proc) is None                                    # no driver loaded: no card
    (proc / "gpus" / "0000:01:00.0").mkdir(parents=True)
    (proc / "version").write_text("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.78.08  Release "
                                  "Build  (builder)  Tue Oct  6 2026\nGCC version:  gcc version 16.2.1\n")
    (proc / "gpus" / "0000:01:00.0" / "information").write_text("Model: \t\t NVIDIA GeForce RTX 4090\nIRQ:   181\n")
    monkeypatch.setattr(S, "NVIDIA_PROC", proc)
    card = S.gpu()
    assert card == {"name": "NVIDIA GeForce RTX 4090", "vram_mb": None, "free_mb": None, "driver": "615.78.08"}
    assert S.driver_major(card) == 615
    # the installer goes on (right PyTorch and llama.cpp builds), instead of stopping or falling back to the CPU
    monkeypatch.setattr(M, "load", lambda: _mini_manifest(server))
    monkeypatch.setattr(S, "qt_missing_libs", lambda: None)
    monkeypatch.setattr(I.platform, "system", lambda: "Linux")
    monkeypatch.setattr(I.platform, "machine", lambda: "x86_64")
    inst = I.Installer(I.Options(yes=True))
    inst.uv = "uv"
    inst.preflight(1000)
    assert inst.torch_index().endswith("/cu130")
    assert inst.llama_asset([{"name": "llama-b1-bin-ubuntu-cuda-13.5-x64.tar.gz"},
                             {"name": "llama-b1-bin-ubuntu-x64.tar.gz"}])[0]["name"].endswith("cuda-13.5-x64.tar.gz")
    # ... and checks the memory as soon as PyTorch is there
    monkeypatch.setattr(S, "torch_memory", lambda py: (7000, 8000))
    with pytest.raises(I.SetupError, match="8 GB"):
        inst.card_memory(tmp_path / "python")
    monkeypatch.setattr(S, "torch_memory", lambda py: None)
    with pytest.raises(I.SetupError, match="does not see the graphics card"):
        inst.card_memory(tmp_path / "python")
    monkeypatch.setattr(S, "torch_memory", lambda py: (23000, 24000))
    inst.card_memory(tmp_path / "python")
    # the doctor says what it knows
    check = next(c for c in D.check_system() if c.name == "Graphics card")
    assert check.status == D.OK and "615.78.08" in check.detail and "nvidia-smi is not on this computer" in check.detail


def test_app_reads_free_memory_from_comfyui_without_nvidia_smi(monkeypatch):
    from video_chain import llm_server as L
    monkeypatch.setattr(L.shutil, "which", lambda name: None)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"devices": [{"vram_free": 6 * 2**30, "vram_total": 24 * 2**30}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}"
        assert L.comfy_vram_mb(url) == (6144, 24576) and L.free_vram_mb(url) == 6144
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert L.free_vram_mb() is None and L.free_vram_mb("http://127.0.0.1:9") is None      # nobody to ask
