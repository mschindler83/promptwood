"""The llama.cpp refiner backend with a fake llama-server (a small Python HTTP server; no model, no GPU)."""
import json
import os
import socket
import sys

import pytest

from video_chain import llm_server as L

FAKE = r'''
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
with open(sys.argv[0] + ".args", "w") as f:
    json.dump(args, f)
time.sleep(0.6)                                   # "loading the model"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send({"status": "ok"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(sys.argv[0] + ".req", "w") as f:
            json.dump(body, f)
        self._send({"choices": [{"message": {"content": "<think>x</think>Here is the refined prompt: \"@Anna runs along the shore.\""}}]})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server(tmp_path):
    script = tmp_path / "llama-server"
    script.write_text(f"#!{sys.executable}\n{FAKE}")
    script.chmod(0o755)
    model = tmp_path / "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-Q6_K.gguf"
    model.write_bytes(b"GGUF")
    return script, model


def test_command_and_label():
    cmd = L.server_command("/x/llama-server", "/m/a.gguf", 8091)
    assert cmd[:3] == ["/x/llama-server", "-m", "/m/a.gguf"] and "--cpu-moe" in cmd and "127.0.0.1" in cmd
    assert "--cpu-moe" not in L.server_command("s", "m", 1, cpu_moe=False)
    assert L.model_label("/m/Huihui-x-abliterated-Q6_K.gguf").startswith("Huihui-x-abliterated-Q6_K (uncensored)")


def test_start_refine_and_stop(fake_server, tmp_path):
    script, model = fake_server
    srv = L.LlamaServer(str(script), str(model), free_port(), log_dir=tmp_path / "log")
    assert srv.problems() == [] and not srv.ready()
    assert srv.models[0]["file"] == L.PREFIX + str(model) and srv.models[0]["uncensored"]
    text = srv.refine_prompt("@Anna is running on the beach.", "video", seed=7, hint="more cinematic",
                             cast=[{"name": "Anna", "desc": "a woman", "voice": False}])
    assert text == "@Anna runs along the shore."                       # think block, preface and quotes removed
    req = json.loads((tmp_path / "llama-server.req").read_text())
    assert req["seed"] == 7 and req["messages"][0]["role"] == "system" and "@Anna" in req["messages"][0]["content"]
    assert "more cinematic" in req["messages"][1]["content"]
    args = json.loads((tmp_path / "llama-server.args").read_text())
    assert "--cpu-moe" in args and args[args.index("-m") + 1] == str(model)
    pid = srv.proc.pid
    assert srv.stop() and not srv.ready() and not srv.stop()
    with pytest.raises(OSError):
        os.kill(pid, 0)                                                 # really gone


def test_missing_files_and_combined_list(tmp_path, fake_server):
    srv = L.LlamaServer(str(tmp_path / "nope"), str(tmp_path / "no.gguf"), free_port())
    with pytest.raises(L.LlmServerError, match="program is missing.*model file is missing"):
        srv.start()

    class Comfy:
        models = [{"file": "gemma.safetensors", "type": "ltxv", "label": "Gemma", "uncensored": True}]

        def refine_prompt(self, *a, **k):
            return "from comfy"
    script, model = fake_server
    llama = L.LlamaServer(str(script), str(model), free_port())
    both = L.CombinedRefiner(Comfy(), llama)
    assert [m["file"] for m in both.models] == [L.PREFIX + str(model), "gemma.safetensors"]
    assert both.refine_prompt("x", "image", "gemma.safetensors") == "from comfy"
    only = L.CombinedRefiner(None, llama, "ComfyUI is not reachable")        # ComfyUI down: llama still works
    assert only.models[0]["file"].startswith(L.PREFIX)
    with pytest.raises(L.LlmServerError, match="not reachable"):
        L.CombinedRefiner(None, None, "ComfyUI is not reachable")
    llama.stop()


def test_main_window_settings_and_stop(fake_server, tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from video_chain.ui import main_window as MW
    root = tmp_path / "hub" / "models--x--y" / "snapshots" / "abc"
    root.mkdir(parents=True)
    for n in ("A-Q6_K.gguf", "mmproj-F16.gguf", "A.imatrix.gguf"):
        (root / n).write_bytes(b"GGUF")
    assert [f.name for f in MW.find_gguf_models([tmp_path / "hub"])] == ["A-Q6_K.gguf"]
    script, model = fake_server
    win = MW.MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    try:
        win.llm_on.setChecked(False)
        assert win.llama_server() is None                                # off: no server
        win.llm_on.setChecked(True)
        win.llm_server.setText(str(script))
        win.llm_model.setEditText(str(model))
        win.llm_port.setValue(free_port())
        srv = win.llama_server()
        assert srv is not None and srv.model == str(model) and win.llama_server() is srv
        srv.start()
        assert srv.ready() and win.stop_llama("test") and "Stopped the llama.cpp server" in win.log.toPlainText()
        win.save_settings()
        assert win.qs.value("llmModel") == str(model) and win.qs.value("llmServerOn", type=bool)
    finally:
        win.stop_llama()
        win.deleteLater()
