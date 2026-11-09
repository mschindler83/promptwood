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
        if self.path == "/props":
            self._send({"model_path": args[args.index("-m") + 1],
                        "modalities": {"vision": "--mmproj" in args}})
        else:
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


def test_chat_messages_and_pictures(tmp_path):
    from PIL import Image
    pic = tmp_path / "start.png"
    Image.new("RGB", (2000, 1000), (10, 20, 30)).save(pic)
    msgs = L.chat_messages(prompt="a cat", kind="image", turns=[("", "a fluffy cat"), ("darker", "a cat at night")],
                           instruction="add rain", pictures=[("The start frame", str(pic))])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "conversation about one prompt" in msgs[0]["content"]
    parts = msgs[1]["content"]
    assert parts[0]["text"].startswith("My prompt:\na cat") and "Improve this prompt." in parts[0]["text"]
    assert parts[1]["text"].strip() == "[The start frame]" and parts[2]["image_url"]["url"].startswith("data:image/png")
    import base64, io
    small = Image.open(io.BytesIO(base64.b64decode(parts[2]["image_url"]["url"].split(",", 1)[1])))
    assert max(small.size) == L.IMAGE_SIDE                                  # sent smaller
    assert [m["content"] for m in msgs[2:]] == ["a fluffy cat", "darker", "a cat at night", "add rain"]
    model = tmp_path / "m" / "x-Q6_K.gguf"
    model.parent.mkdir()
    model.write_bytes(b"x")
    assert L.find_mmproj(str(model)) is None
    for n in ("mmproj-F32.gguf", "mmproj-F16.gguf"):
        (model.parent / n).write_bytes(b"x")
    assert L.find_mmproj(str(model)).endswith("mmproj-F16.gguf")
    srv = L.LlamaServer("s", str(model), 1)
    with pytest.raises(L.LlmServerError, match="cannot see"):
        srv.chat("a", "image", [], "x", pictures=[("p", str(pic))])
    assert L.LlamaServer("s", str(model), 1, mmproj="m").models[0]["vision"]


def test_refine_window_conversation(tmp_path, monkeypatch):
    """The refine window with a conversation model: versions build on each other, an earlier version can be
    continued from (the later ones go), the user's edit counts, Another try replaces the answer, pictures are
    sent only when the model sees and they are switched on."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PIL import Image
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from video_chain.ui.prompt_refiner import RefineDialog
    pic = tmp_path / "s.png"
    Image.new("RGB", (64, 48)).save(pic)
    calls = []

    class Fake:
        models = [{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": True},
                  {"file": "gemma.safetensors", "label": "G"}]

        def chat(self, prompt, kind, turns, ask, pictures=(), cast=None, seed=0):
            calls.append((list(turns), ask, list(pictures)))
            return f"v{len(turns) + 1}:{ask or 'improved'}"

        def refine_prompt(self, prompt, kind, model, seed, hint, **k):
            return "one-shot " + hint
    dlg = RefineDialog(None, "a cat", "image", lambda: Fake(), pictures=[("The start frame", str(pic), True),
                                                                          ("A result", str(pic), False)])

    def wait_idle():
        end = __import__("time").time() + 5
        while (dlg._busy or dlg.refiner is None) and __import__("time").time() < end:
            app.processEvents()
        app.processEvents()
    wait_idle()
    assert dlg.chatting() and dlg.seeing() and dlg.talk.isVisibleTo(dlg) and "conversation" in dlg.caps.text()
    dlg.refine()                                                         # empty: improve it
    wait_idle()
    assert calls[-1] == ([], "", [("The start frame", str(pic))]) and dlg.out.toPlainText() == "v1:improved"
    dlg.hint.setText("darker")
    dlg.b_refine.click()
    wait_idle()
    assert calls[-1][0] == [("", "v1:improved")] and dlg.out.toPlainText() == "v2:darker" and dlg.base == 1
    dlg.out.setPlainText("v2 edited by me")                              # the user's edit is the current version
    dlg.hint.setText("rain")
    dlg.refine()
    wait_idle()
    assert calls[-1][0][-1] == ("darker", "v2 edited by me") and len(dlg.turns) == 3
    dlg._version_clicked(QUrl("v:0"))                                    # back to version 1, continue from there
    assert dlg.out.toPlainText() == "v1:improved"
    dlg.hint.setText("brighter")
    dlg.refine()
    wait_idle()
    assert calls[-1][0] == [("", "v1:improved")] and [a for a, _v in dlg.turns] == ["", "brighter"]
    n = len(calls)
    dlg.b_again.click()                                                  # another answer to "brighter"
    wait_idle()
    assert len(calls) == n + 1 and calls[-1][1] == "brighter" and len(dlg.turns) == 2
    assert calls[-1][0] == [("", "v1:improved")] and dlg.turns[0] == ("", "v1:improved")   # version 1 untouched
    assert not dlg._edited.get(0)
    fail = {"on": True}
    orig_chat = Fake.chat

    def broken(self, *a, **k):
        if fail["on"]:
            raise RuntimeError("server down")
        return orig_chat(self, *a, **k)
    Fake.chat = broken
    dlg.b_again.click()                                                  # fails: still at version 2
    wait_idle()
    assert dlg.base == 1 and "server down" in dlg.state.text()
    Fake.chat = orig_chat
    dlg.out.setPlainText("my own words")                                 # an edit, then a version click: kept
    dlg._version_clicked(QUrl("v:0"))
    assert dlg.turns[1][1] == "my own words" and dlg._edited.get(1)
    dlg.model.setCurrentIndex(1)                                         # a ComfyUI model: one refine per press
    assert not dlg.chatting() and not dlg.talk.isVisibleTo(dlg) and dlg.b_refine.text() == "Refine"
    dlg.hint.setText("shorter")
    dlg.refine()
    wait_idle()
    assert dlg.out.toPlainText() == "one-shot shorter"
    dlg.reject()


def test_a_server_on_the_port_must_be_the_right_one(fake_server, tmp_path):
    script, model = fake_server
    port = free_port()
    other = tmp_path / "other.gguf"
    other.write_bytes(b"GGUF")
    foreign = L.LlamaServer(str(script), str(other), port)
    foreign.start()
    try:
        mine = L.LlamaServer(str(script), str(model), port)
        with pytest.raises(L.LlmServerError, match="port .* is used by another llama.cpp server .*other.gguf"):
            mine.start()
    finally:
        foreign.stop()
    plain = L.LlamaServer(str(script), str(model), port)                 # ours, started without vision ...
    plain.start()
    plain.mmproj = str(tmp_path / "mmproj-F16.gguf")                     # ... now wanted with it: restarted
    first = plain.proc.pid
    plain.start()
    assert plain.proc.pid != first and "--mmproj" in json.loads((tmp_path / "llama-server.args").read_text())
    plain.stop()


def test_adding_pictures_keeps_the_thumbnails(tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PIL import Image
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from video_chain.ui import image_picker as IP
    from video_chain.ui.prompt_refiner import RefineDialog
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for f in (a, b):
        Image.new("RGB", (64, 48)).save(f)

    class Fake:
        models = [{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": True}]
    dlg = RefineDialog(None, "x", "image", lambda: Fake(), pictures=[("Start frame", str(a), True)])
    dlg.show()
    end = __import__("time").time() + 3
    while dlg.refiner is None and __import__("time").time() < end:
        app.processEvents()
    app.processEvents()
    monkeypatch.setattr(IP, "pick_image", lambda *a_, **k: str(b))

    def shown():
        ws = [dlg.pics_row.itemAt(i).widget() for i in range(dlg.pics_row.count())]
        return [w.text() for w in ws if w is not None and w.isVisible()]
    add = [w for w in (dlg.pics_row.itemAt(i).widget() for i in range(dlg.pics_row.count()))
           if w is not None and w.text() == "+ Picture"][0]
    add.click()
    app.processEvents()
    assert shown() == ["Start frame", "b", "+ Picture"]
    dlg._remove_picture(0)
    app.processEvents()
    assert shown() == ["b", "+ Picture"]
    dlg.reject()


def test_ideas_parsing_and_bullet_check():
    got = L.parse_ideas('Here:\n```json\n[{"title": "Rain", "angle": "night", "prompt": "a  cat"}, "a dog", {"x": 1}]\n```')
    assert got == [{"title": "Rain", "angle": "night", "prompt": "a cat"},
                   {"title": "Idea 2", "angle": "", "prompt": "a dog"}]
    with pytest.raises(L.LlmServerError):
        L.parse_ideas("no list here")
    bullets = "- @Anna in a red coat\n- an old lighthouse\n* heavy rain\n\n1. seagulls"
    assert L.missing_bullets(bullets, "@Anna wears a red coat near an old lighthouse; seagulls circle") == ["heavy rain"]
    assert L.missing_bullets(bullets, "Anna in a red coat, old lighthouse, heavy rain, seagulls") == ["@Anna in a red coat"]
    msgs = L.idea_messages(bullets="- a cat", n=4, wish="surreal")
    assert "Make 4 creative" in msgs[0]["content"] and "Also: surreal" in msgs[1]["content"]


def test_ideas_mode_in_the_refine_window(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from video_chain.ui.prompt_refiner import RefineDialog
    asked = []

    class Fake:
        models = [{"file": "llama:/m/q.gguf", "label": "Q", "chat": True, "vision": False}]

        def ideas(self, bullets, n, wish, pictures=(), cast=None, seed=0):
            asked.append((bullets, n, wish))
            return [{"title": f"T{k}", "angle": f"a{k}", "prompt": f"cat {k}" + (" in rain" if k else "")}
                    for k in range(n)]

    def make(kind):
        d = RefineDialog(None, "- cat\n- rain", kind, lambda: Fake())
        d.show()
        end = __import__("time").time() + 3
        while d.refiner is None and __import__("time").time() < end:
            app.processEvents()
        app.processEvents()
        return d
    clip = make("video")                                               # clips: no ideas mode
    assert not clip.b_mode_ideas.isVisible()
    clip.reject()
    dlg = make("image")
    assert dlg.b_mode_ideas.isVisible() and dlg.original.isReadOnly()
    dlg.set_ideas_mode(True)
    assert not dlg.original.isReadOnly() and dlg.b_refine.text() == "Make 5 ideas" and not dlg.out.isVisible()
    dlg.count.setValue(3)
    dlg.hint.setText("surreal")
    dlg.b_refine.click()
    end = __import__("time").time() + 3
    while dlg._busy and __import__("time").time() < end:
        app.processEvents()
    assert asked[-1] == ("- cat\n- rain", 3, "surreal") and len(dlg.ideas) == 3
    assert dlg.ideas[0]["missing"] == ["rain"] and "may leave out: rain" in dlg.ideas_view.toPlainText()
    assert dlg.b_all.isEnabled()
    dlg._idea_link(QUrl("more:1"))                                     # continue from idea 2 as version 1
    assert not dlg.ideas_mode() and dlg.turns[0][1] == "cat 1 in rain" and dlg.out.toPlainText() == "cat 1 in rain"
    dlg.set_ideas_mode(True)
    dlg._use_all()
    assert dlg.chosen_ideas() == ["cat 0", "cat 1 in rain", "cat 2 in rain"]
