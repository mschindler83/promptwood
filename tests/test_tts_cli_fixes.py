"""Fixes from the review of the TTS worker, ElevenLabs client, command line and H3 prompt text (no model, no network:
stub libraries for the worker, a fake gen for the worker loop)."""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from video_chain import __main__ as M
from video_chain import h3_prompt as H
from video_chain import tts

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "video_chain" / "tts_worker.py"

STUBS = {
    "torch.py": '''
import os
float16 = bfloat16 = 0
def manual_seed(s): pass
class _Cuda:
    def is_available(self): return True
    def manual_seed_all(self, s): pass
    def empty_cache(self):
        with open(os.environ["MARK"], "a") as f: f.write("x")
cuda = _Cuda()
''',
    "soundfile.py": '''
import os
def write(path, wav, sr):
    with open(path, "wb") as f: f.write(b"RIFF" + bytes(len(wav)))
''',
    "transformers.py": '''
def pipeline(*a, **k):
    return lambda audio, **kw: {"text": " hello "}
''',
    "qwen_tts.py": '''
class Qwen3TTSModel:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def generate_voice_design(self, text, language, instruct): return [[0.0] * 480], 24000
''',
    "audio_separator/__init__.py": "",
    "audio_separator/separator.py": '''
class Separator:
    def __init__(self, **k): pass
    def load_model(self, m): raise RuntimeError("model did not load")
''',
}


class Worker:
    def __init__(self, tmp_path):
        stubs = tmp_path / "stubs"
        for name, text in STUBS.items():
            f = stubs / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        self.mark = tmp_path / "mark"
        env = {**os.environ, "PYTHONPATH": str(stubs), "MARK": str(self.mark)}
        self.p = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, env=env, cwd=str(tmp_path))
        self.ready = self.read()

    def read(self):
        line = self.p.stdout.readline()
        assert line.startswith("@@TTS "), line
        return json.loads(line[6:])

    def ask(self, r):
        self.p.stdin.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")
        self.p.stdin.flush()
        return self.read()

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)


@pytest.fixture
def worker(tmp_path):
    w = Worker(tmp_path)
    yield w
    w.p.kill()
    w.p.wait()


# ---- TST-16: the worker's own protocol

def test_worker_protocol_answers_every_kind_of_line(worker):
    assert worker.ready == {"ok": True, "ready": True}
    assert worker.ask("not json") == {"ok": False, "error": "not JSON"}
    assert worker.ask({"cmd": "nope"})["error"].startswith("unknown request")
    a = worker.ask({"cmd": "design"})                                   # a request without its fields
    assert a["ok"] is False and "KeyError" in a["error"]
    worker.close()
    assert worker.p.returncode == 0


def test_worker_designs_a_voice_and_echoes_the_request_id(worker, tmp_path):
    out = tmp_path / "v.wav"
    a = worker.ask({"cmd": "design", "id": 7, "model": "m", "text": "Hi", "language": "English",
                    "instruct": "warm", "out": str(out), "seed": 3})
    assert a["ok"] and a["sr"] == 24000 and a["secs"] == 0.02 and a.get("id") == 7
    assert out.is_file() and not Path(str(out) + ".part.wav").exists()


def test_worker_says_why_ffmpeg_failed(worker, tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"this is not audio")
    a = worker.ask({"cmd": "transcribe", "model": "m", "audio": str(bad), "language": None})
    assert a["ok"] is False and "Invalid data" in a["error"], a            # ffmpeg's own words


def test_worker_frees_its_memory_when_removing_music_fails(worker, tmp_path):
    d = tmp_path / "sep"
    d.mkdir()
    (d / "m.ckpt").write_bytes(b"x")
    (d / "m.yaml").write_bytes(b"x")
    a = worker.ask({"cmd": "isolate", "dir": str(d), "model": "m.ckpt", "audio": "a.wav", "out": "o.wav"})
    assert a["ok"] is False and "did not load" in a["error"]
    assert worker.mark.read_text().count("x") >= 2          # emptied before it and again after the failure


def test_worker_isolate_names_a_missing_model_file(worker, tmp_path):
    a = worker.ask({"cmd": "isolate", "dir": str(tmp_path), "model": "m.ckpt", "audio": "a", "out": "o"})
    assert a["ok"] is False and "m.ckpt" in a["error"]


# ---- the client: request ids, a late answer is not taken for the next one

def test_a_late_answer_of_an_earlier_request_is_dropped(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(r'''
import json, sys
def say(a): sys.stdout.write("@@TTS " + json.dumps(a) + "\n"); sys.stdout.flush()
say({"ok": True, "ready": True})
for line in sys.stdin:
    r = json.loads(line)
    if r["cmd"] == "quit": break
    say({"ok": True, "id": r["id"] - 1, "text": "late answer of the one before"})
    say({"ok": True, "id": r["id"], "text": "mine"})
''')
    from test_tts import fake_cache
    loc = tts.LocalTTS(python=Path(sys.executable), cache=fake_cache(tmp_path), worker=script, start_timeout=20)
    try:
        assert loc.request({"cmd": "transcribe"})["text"] == "mine"
        assert loc.request({"cmd": "transcribe"})["text"] == "mine"
    finally:
        loc.stop()


# ---- ElevenLabs

def test_eleven_answers_of_a_changed_shape_are_tts_errors(monkeypatch):
    el = tts.ElevenLabs(key="k")
    monkeypatch.setattr(el, "_call", lambda *a, **k: json.dumps({"voices": [{"name": "no id"}]}).encode())
    with pytest.raises(tts.TtsError):
        el.voices()
    monkeypatch.setattr(el, "_call", lambda *a, **k: json.dumps([{"name": "x", "can_do_text_to_speech": True}]).encode())
    with pytest.raises(tts.TtsError):
        el.models()
    monkeypatch.setattr(el, "_call", lambda *a, **k: b"{}")
    with pytest.raises(tts.TtsError):
        el.clone("n", Path(__file__))


def test_clone_escapes_names_in_the_multipart_body(monkeypatch, tmp_path):
    el = tts.ElevenLabs(key="k")
    seen = {}
    monkeypatch.setattr(el, "_call", lambda m, p, body, ctype: seen.update(body=body) or b'{"voice_id": "v"}')
    f = tmp_path / 'we"ird\r\nname.wav'
    f.write_bytes(b"RIFF")
    el.clone('A "quoted"\r\nname', f, "d\r\nx")
    head = seen["body"].split(b"RIFF")[0].decode()
    assert head.count("Content-Disposition") == 3
    assert "\r\nname" not in head.replace('\r\n\r\n', "").replace("\r\nContent-", "").replace("\r\n--", "")
    assert 'filename="we\\"ird' not in head and "we'ird" in head or "weird" in head


def test_a_sample_needs_a_chosen_elevenlabs_voice(tmp_path):
    with pytest.raises(tts.TtsError, match="voice"):
        tts.make_sample(tts.ElevenLabs(key="k"), "Hi", "English", tmp_path / "o.wav", voice_id=None)


def test_eleven_waits_once_when_rate_limited(monkeypatch):
    import io
    import urllib.error
    el = tts.ElevenLabs(key="k")
    n, waits = {"n": 0}, []

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"ok"

    def urlopen(req, timeout=None):
        n["n"] += 1
        if n["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "x", {"Retry-After": "3"}, io.BytesIO(b"{}"))
        return R()
    monkeypatch.setattr(tts.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(tts, "_pause", lambda s: waits.append(s), raising=False)
    assert el._call("GET", "/x") == b"ok" and waits == [3.0]


def test_speed_that_is_not_a_number_is_one():
    for bad in (float("nan"), float("inf"), "nan"):
        assert tts.speak_options(given={"speed": bad})["speed"] == 1.0 or math.isfinite(
            tts.speak_options(given={"speed": bad})["speed"])
    assert tts.speak_options(given={"speed": float("nan")})["speed"] == 1.0
    assert tts.speak_options(given={"speed": 5})["speed"] == tts.SPEED_RANGE[1]


def test_paths_can_be_set_by_environment():
    code = ("from video_chain import tts, llm_server; print(tts.HF_CACHE, llm_server.DEFAULT_SERVER)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "VIDEO_CHAIN_HF_CACHE": "/x/hf",
                            "VIDEO_CHAIN_LLAMA_SERVER": "/x/llama-server"})
    assert r.stdout.split() == ["/x/hf", "/x/llama-server"], r.stderr


def test_a_big_log_is_rotated(tmp_path):
    from video_chain.llm_server import rotate_log
    log = tmp_path / "a.log"
    log.write_bytes(b"x" * 100)
    rotate_log(log, limit=50)
    assert not log.exists() and (tmp_path / "a.log.1").stat().st_size == 100
    log.write_bytes(b"y")
    rotate_log(log, limit=50)
    assert log.exists()


# ---- the command line

@pytest.mark.parametrize("args", [
    ["--resolution", "0"], ["--steps", "-3"], ["--strength", "0"], ["--seed", "-1"], ["--idle-minutes", "-1"],
    ["--targets", "n1"], ["--runpod-gpu", "x"], ["--tree", "/p", "a.png", "p.txt"], ["--frames", "0"],
])
def test_command_line_mistakes_are_exit_code_two_with_a_message(args, capsys):
    with pytest.raises(SystemExit) as e:
        M.main(args)
    assert e.value.code == 2


def test_the_worker_options_still_parse():
    a = M.build_parser().parse_args(["--serve", "--events", "--idle-minutes", "0", "--comfy-url", "http://x"])
    assert a.idle_minutes == 0 and a.serve
    M.build_parser().parse_args(["--serve", "--cloud", "runpod", "--runpod-gpu", "g", "--runpod-models", "7b"])
    M.build_parser().parse_args(["--tree", "/p", "--targets", "a,b"])


# ---- the worker loop

class _Gen:
    def __init__(self, fail_free=False):
        self.freed, self.fail_free = 0, fail_free

    def free(self):
        self.freed += 1
        if self.fail_free:
            raise RuntimeError("ComfyUI is down")


def _args(idle=0.0):
    return M.build_parser().parse_args(["--serve", "--idle-minutes", str(idle)])


def _stdin(*lines):
    r, w = os.pipe()
    os.write(w, "".join(l + "\n" for l in lines).encode())
    return os.fdopen(r), w


def test_a_free_that_fails_still_ends_the_worker_cleanly_on_quit_and_idle():
    for lines, idle, reason in ((['{"cmd": "quit", "free": true}'], 0.0, "quit"), ([], 0.0005, "idle")):
        events, gen = [], _Gen(fail_free=True)
        stdin, w = _stdin(*lines)
        try:
            assert M.serve(_args(idle), events.append, lambda cur: gen, stdin=stdin) == 0
        finally:
            os.close(w)
            stdin.close()
        assert gen.freed == 1 and {"e": "bye", "reason": reason} in events


def test_unknown_and_bad_commands_are_answered_not_fatal():
    events, gen = [], _Gen()
    stdin, w = _stdin("garbage", '{"cmd": "dance"}', '{"cmd": "quit"}')
    try:
        assert M.serve(_args(), events.append, lambda cur: gen, stdin=stdin) == 0
    finally:
        os.close(w)
        stdin.close()
    assert [e["e"] for e in events] == ["idle", "error", "error", "bye"]


def test_a_refine_command_reports_errors_and_goes_on(monkeypatch, tmp_path):
    def refine(*a, **k):
        raise ValueError("no such target")
    monkeypatch.setattr(M.treemod, "refine_path", refine)
    events = []
    cmd = {"cmd": "refine", "tree": str(tmp_path), "target": "n1", "size": 2048, "out": str(tmp_path / "o")}
    stdin, w = _stdin(json.dumps(cmd), '{"cmd": "quit"}')
    try:
        assert M.serve(_args(), events.append, lambda cur: _Gen(), stdin=stdin) == 0
    finally:
        os.close(w)
        stdin.close()
    assert {"e": "error", "msg": "no such target"} in events and events[-1]["reason"] == "quit"


def test_a_gone_gui_ends_the_worker_quietly(monkeypatch, capsys):
    class Gone:
        def write(self, s):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError
    monkeypatch.setattr(sys, "stdout", Gone())
    emit = M._printer(True)
    emit({"e": "idle"})                                   # no BrokenPipeError
    emit({"e": "idle"})
    assert emit.gone is True
    gen = _Gen()
    stdin, w = _stdin('{"cmd": "dance"}')
    try:
        assert M.serve(_args(), emit, lambda cur: gen, stdin=stdin) == 0
    finally:
        os.close(w)
        stdin.close()
    assert gen.freed == 1


def test_a_second_signal_cannot_cut_the_pod_termination_short():
    states = []

    class Session:
        def close(self):
            states.append([signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)])
    old = [signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)]
    try:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, lambda *a: None)
        mine = [signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)]
        M._close_cloud({"session": Session()}, lambda e: None)
        assert states == [[signal.SIG_IGN] * 3]
        assert [signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)] == mine   # put back
    finally:
        for s, h in zip((signal.SIGTERM, signal.SIGINT, signal.SIGHUP), old):
            signal.signal(s, h)


def test_the_launcher_says_what_to_do_without_a_venv(tmp_path):
    shutil.copy(ROOT / "video-chain", tmp_path / "video-chain")
    r = subprocess.run(["bash", str(tmp_path / "video-chain"), "--help"], capture_output=True, text=True)
    assert r.returncode == 1 and "uv sync" in r.stderr


# ---- H3 prompt text

def test_a_subject_of_an_unknown_kind_is_a_person():
    assert H.Subject(name="Rex", kind="robot").kind == "person"
    out = H.compose("[Shot 1] Rex waves.", subjects=[H.Subject(name="Rex", kind="robot", pictures=(1,))])
    assert "Rex" in out


def test_ellipsis_and_cjk_sentence_ends_get_no_extra_full_stop():
    for end in ("He waits…", "彼は待つ。", "待って！", "待って？"):
        assert H._ends_sentence(end)
        assert ".." not in H._join(end, "Then he leaves") and "。." not in H._join(end, "Then he leaves")


def test_a_sound_label_alone_early_does_not_swallow_the_shots():
    text = "[Shot 1] A man walks.\noverall_soundscape:\n[Shot 2] He sits."
    body, scape, music = H.split_sound_sections(text)
    assert "[Shot 2] He sits." in body and scape is None
    body, scape, music = H.split_sound_sections("[Shot 1] A man walks.\n\noverall_soundscape:\nrain\nwind")
    assert body == "[Shot 1] A man walks." and scape == "rain wind"
