"""A small stand-in for ComfyUI's HTTP API (no websocket): enough for ComfyGenerator's requests.
/prompt checks the workflow's node types and links, then 'renders' it with ffmpeg (a colour clip with the
requested frame count, size and sound, plus the last frame as PNG)."""
import json
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

KNOWN = {"UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "MiniMaxH3ImageToVideo", "RandomNoise",
         "KSamplerSelect", "BasicScheduler", "BasicGuider", "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio",
         "ImageFromBatch", "SaveImage", "CreateVideo", "SaveVideo", "LoraLoaderModelOnly",
         "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide", "MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler",
         "LoadVideo", "GetVideoComponents"}


class FakeComfy:
    def __init__(self, root: Path, loras=("MiniMax H3/glow.safetensors",), fail_with: str | None = None,
                 models_missing: bool = False):
        self.root = Path(root)
        (self.root / "input").mkdir(parents=True, exist_ok=True)
        (self.root / "output").mkdir(parents=True, exist_ok=True)
        self.loras, self.fail_with, self.models_missing = list(loras), fail_with, models_missing
        self.turbo_nodes = True
        self.prompts, self.calls, self.history = [], [], {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def object_info(self, node):
        from video_chain import comfy as C
        files = {"UNETLoader": ("unet_name", [C.UNET, C.UNET_REF]), "CLIPLoader": ("clip_name", [C.TEXT_ENCODER]),
                 "VAELoader": ("vae_name", [C.VIDEO_VAE, C.AUDIO_VAE]),
                 "LoraLoaderModelOnly": ("lora_name", self.loras)}
        if node in ("MiniMaxH3ImageToVideo", "MiniMaxH3TurboSampler", "MiniMaxH3TurboLoRA", "MiniMaxH3AddGuide",
                    "LoadVideo", "GetVideoComponents"):
            return {} if node.startswith("MiniMaxH3Turbo") and not self.turbo_nodes else {node: {"input": {"required": {}}}}
        if node == "MiniMaxH3ReferenceToVideo":
            return {node: {"input": {"required": {}, "optional": {"ref_images": [
                "COMFY_AUTOGROW_V3", {"template": {"prefix": "ref_image_", "min": 0, "max": 9}}]}}}}
        if node in files:
            field, names = files[node]
            names = [] if self.models_missing and node == "UNETLoader" else names
            return {node: {"input": {"required": {field: ["COMBO", {"options": names}]}}}}
        return {}

    def render(self, wf: dict) -> dict:
        for nid, n in wf.items():
            assert n["class_type"] in KNOWN, n["class_type"]
            for v in n["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                    assert v[0] in wf, f"{nid} links to missing node {v[0]}"
        if "refcond" in wf:                         # identity references: ref2va + the start image as guide
            cond = wf["refcond"]["inputs"]
            assert wf["unet"]["inputs"]["unet_name"].startswith("minimax_h3_ref2va")
            if "start" in wf:
                assert wf["cond"]["class_type"] == "MiniMaxH3AddGuide" and wf["cond"]["inputs"]["image"] == ["start", 0]
            for k, v in cond.items():
                if k.startswith("ref_images."):
                    assert (self.root / "input" / wf[v[0]]["inputs"]["image"]).is_file(), "reference not uploaded"
        else:
            cond = wf["cond"]["inputs"]
            assert ("first_frame" in cond) == ("start" in wf) and ("last_frame" in cond) == ("end" in wf)
        for name in ("start", "end"):
            if name in wf:
                assert (self.root / "input" / wf[name]["inputs"]["image"]).is_file(), f"{name} image not uploaded"
        for nid, node in wf.items():
            if node["class_type"] == "LoadVideo":
                assert (self.root / "input" / node["inputs"]["file"]).is_file(), "guide clip not uploaded"
        w, h, n = cond["width"], cond["height"], cond["length"]
        prefix = wf["save_video"]["inputs"]["filename_prefix"]
        out = self.root / "output" / (prefix + "_00001_.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        args = ["-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=24"]
        if "audio" in wf["video"]["inputs"]:
            args += ["-f", "lavfi", "-i", "sine=f=440", "-c:a", "aac", "-t", f"{n / 24:.4f}"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-frames:v", str(n), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(out)], check=True)
        last = self.root / "output" / (wf["save_last"]["inputs"]["filename_prefix"] + "_00001_.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=red:s={w}x{h}",
                        "-frames:v", "1", str(last)], check=True)
        sub = str(Path(prefix).parent)
        return {"save_video": {"images": [{"filename": out.name, "subfolder": sub, "type": "output"}],
                               "animated": [True]},
                "save_last": {"images": [{"filename": last.name, "subfolder": sub, "type": "output"}]}}

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                fake.calls.append(("GET", u.path))
                if u.path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "fake"}})
                if u.path.startswith("/object_info/"):
                    return self._send(200, fake.object_info(u.path.rsplit("/", 1)[1]))
                if u.path.startswith("/history/"):
                    pid = u.path.rsplit("/", 1)[1]
                    return self._send(200, {pid: fake.history[pid]} if pid in fake.history else {})
                if u.path == "/view":
                    q = urllib.parse.parse_qs(u.query)
                    f = fake.root / q["type"][0] / q.get("subfolder", [""])[0] / q["filename"][0]
                    return self._send(200, f.read_bytes(), "application/octet-stream")
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                u = urllib.parse.urlparse(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.calls.append(("POST", u.path))
                if u.path == "/upload/image":
                    ctype = self.headers["Content-Type"]
                    boundary = ctype.split("boundary=")[1].encode()
                    fields, fname, data = {}, None, None
                    for part in body.split(b"--" + boundary):
                        if b"\r\n\r\n" not in part:
                            continue
                        head, val = part.split(b"\r\n\r\n", 1)
                        val = val[:-2] if val.endswith(b"\r\n") else val
                        name = head.split(b'name="')[1].split(b'"')[0].decode()
                        if b"filename=" in head:
                            fname = head.split(b'filename="')[1].split(b'"')[0].decode()
                            data = val
                        else:
                            fields[name] = val.decode()
                    d = fake.root / "input" / fields.get("subfolder", "")
                    d.mkdir(parents=True, exist_ok=True)
                    (d / fname).write_bytes(data)
                    return self._send(200, {"name": fname, "subfolder": fields.get("subfolder", ""),
                                            "type": "input"})
                payload = json.loads(body or b"{}")
                if u.path == "/prompt":
                    fake.prompts.append(payload["prompt"])
                    pid = f"p{len(fake.prompts)}"
                    if fake.fail_with:
                        fake.history[pid] = {"outputs": {}, "status": {"status_str": "error", "completed": False,
                                             "messages": [["execution_error", {"exception_type": "RuntimeError",
                                                                               "exception_message": fake.fail_with}]]}}
                    else:
                        fake.history[pid] = {"outputs": fake.render(payload["prompt"]),
                                             "status": {"status_str": "success", "completed": True}}
                    return self._send(200, {"prompt_id": pid, "number": 1, "node_errors": {}})
                if u.path in ("/free", "/interrupt", "/queue"):
                    fake.calls.append(("BODY", u.path, payload))
                    return self._send(200, {})
                return self._send(404, {"error": "not found"})
        return H
