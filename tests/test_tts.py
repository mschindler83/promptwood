"""Text to speech: spoken lines as a clip's sound, voices from TTS, the Qwen3-TTS worker protocol (a fake worker),
ElevenLabs against a local fake server; the real model only with -m models."""
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

from video_chain import chain
from video_chain import tree as T
from video_chain import tts

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def tone(path: Path, secs: float, freq: int = 440, rate: int = 24000):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={secs}",
                    "-ar", str(rate), "-ac", "1", str(path)], check=True)


class FakeLocal:
    """Qwen3-TTS without the model: a line is a tone 0.08 s per character long."""
    name = "local"

    def __init__(self):
        self.calls = []

    def problems(self, **_k):
        return []

    def clone(self, text, lang, ref_audio, ref_text, out, seed=None, **how):
        self.calls.append(("clone", text, lang, Path(ref_audio).name, ref_text, seed, *([how] if how else [])))
        tone(Path(out), round(0.08 * len(text), 2))
        return 0.08 * len(text)

    def design(self, text, lang, instruct, out, seed=None):
        self.calls.append(("design", text, lang, instruct, seed))
        tone(Path(out), 4.0, 300)
        return 4.0

    def transcribe(self, audio, lang=None):
        self.calls.append(("transcribe", Path(audio).name, lang))
        return "what the sample says"

    def stop(self):
        return False


class FakeEleven(FakeLocal):
    name = "elevenlabs"

    def speak(self, text, voice_id, out, model="eleven_multilingual_v2", lang=None, seed=None, **how):
        self.calls.append(("speak", text, voice_id, model, lang, seed, *([how] if how else [])))
        raw = Path(out).with_suffix(".src.wav")
        tone(raw, round(0.08 * len(text), 2), 500)
        chain.to_wav(raw, Path(out))
        return 0.08 * len(text)


@pytest.fixture
def project(tmp_path):
    img = tmp_path / "in.png"
    Image.new("RGB", (96, 54), (200, 200, 200)).save(img)
    p = tmp_path / "proj"
    p.mkdir()
    t = T.Tree()
    T.set_start_image(p, img, t)
    t.options["identity"]["language"] = "German"
    T.add_character(p, t, "Anna", desc="young woman")
    T.add_character(p, t, "Ben", desc="man, 40")
    T.save(p, t)
    return p


def give_voice(project, t, name, tmp_path, secs=3.0):
    src = tmp_path / f"{name}.wav"
    tone(src, secs, 220)
    T.set_voice(project, t, name, src)


def test_lines_and_their_problems(project, tmp_path):
    t = T.load(project)
    n = t.add(None, "@Anna looks around.")
    assert "no spoken line" in tts.line_problems(t, n, "local")[0]
    n.prompt = '@Anna says "Hallo." @Ben answers "Na?"'
    assert [(d["speaker"]["name"], d["line"]) for d in tts.clip_lines(t, n)] == [("Anna", "Hallo."), ("Ben", "Na?")]
    assert tts.line_problems(t, n, "local") == ["@Anna has no voice sample yet (Cast → Voice… or Voice with TTS…)",
                                               "@Ben has no voice sample yet (Cast → Voice… or Voice with TTS…)"]
    assert "no ElevenLabs voice" in tts.line_problems(t, n, "elevenlabs")[0]
    T.update_character(t, "Anna", lead=False)               # (the first character is the lead at first)
    n.prompt = 'Someone says "Hallo."'                    # nobody mentioned, no lead: who?
    assert tts.line_problems(t, n, "local") == ["who says “Hallo.”? (write @Name in its sentence)"]
    T.update_character(t, "Anna", lead=True)               # nobody mentioned: the lead says it
    give_voice(project, t, "Anna", tmp_path)
    assert tts.line_problems(t, n, "local") == []


def test_speak_lines_makes_the_clips_sound(project, tmp_path):
    """Each line in its speaker's voice (cloned from the sample, transcribed once), joined with pauses into a new
    track the clip plays with lip sync; the clip gets longer when the lines need it; speaking again replaces its
    own track; everything survives a reload; the clip counts as changed."""
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    give_voice(project, t, "Ben", tmp_path)
    n = t.add(None, '@Anna says "Hallo, wie geht es dir heute so?" @Ben answers "Gut, danke dir, und selbst?"')
    n.overrides["sound_over"] = True
    eng = FakeLocal()
    r = tts.speak_lines(project, t, n.id, eng)
    clones = [c for c in eng.calls if c[0] == "clone"]
    assert [c[1] for c in clones] == ["Hallo, wie geht es dir heute so?", "Gut, danke dir, und selbst?"]
    assert all(c[2] == "German" and c[4] == "what the sample says" for c in clones)
    assert [c[3] for c in clones] == [T.character(t, "Anna")["voice"]["file"].split("/")[-1],
                                      T.character(t, "Ben")["voice"]["file"].split("/")[-1]]
    assert sum(1 for c in eng.calls if c[0] == "transcribe") == 2            # once per voice
    assert T.character(t, "Anna")["voice"]["text"] == "what the sample says"
    tr = T.track_of(t, r["track"])
    lines = 0.08 * 32 + 0.08 * 27
    expect = tts.LEAD_IN + lines + tts.GAP + tts.TAIL
    assert abs(r["secs"] - expect) < 0.1 and abs(chain.audio_seconds(project / tr["file"]) - r["secs"]) < 0.05
    assert tr["tts"]["lines"] == [{"name": "Anna", "line": "Hallo, wie geht es dir heute so?"},
                                  {"name": "Ben", "line": "Gut, danke dir, und selbst?"}]
    assert n.overrides["source_sound"] == T.TRACK_PREFIX + r["track"] and "sound_over" not in n.overrides
    assert r["frames"] == 141 and n.overrides["frames"] == 141                # 5.6 s = 135 frames: next 17n+5
    span = T.track_sound_span(t, n)
    assert span["file"] == tr["file"] and span["start"] == 0.0
    T.save(project, t)
    t = T.load(project)
    assert T.character(t, "Anna")["voice"]["text"] == "what the sample says" and T.track_of(t, r["track"])["tts"]
    eng2 = FakeLocal()
    r2 = tts.speak_lines(project, t, n.id, eng2)                             # again: its own old track goes
    assert not any(c[0] == "transcribe" for c in eng2.calls)                 # (the transcripts are kept)
    assert T.track_of(t, r["track"]) is None and T.track_of(t, r2["track"]) is not None
    assert T.Checker(project, t).status(n.id) == "new"


def test_a_long_clip_speaks_its_lines(project, tmp_path):
    """A 10 s clip whose line was spoken: its soundtrack (10 s) is all the sound the model gets. The speaker's voice
    (7.5 s) is not sent with it (the two would be more than the 15 s the model takes, and the track has the voice
    already); the line stays in the prompt and the soundtrack sentence ties the lips to the sound. Laid over
    instead, the model speaks itself again, with the voice."""
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path, secs=9.0)
    give_voice(project, t, "Ben", tmp_path, secs=9.0)
    assert T.character(t, "Anna")["voice"]["secs"] == pytest.approx(T.VOICE_MAX_SECS, abs=0.1)
    n = t.add(None, '@Anna looks up and says "Guten Morgen." @Ben answers "Hallo."')
    n.overrides["frames"] = 243                                              # 10.1 s
    ch = T.Checker(project, t)
    before = ch.settings(n.id)
    assert len(before["audio_refs"]) == 2 and "voice timbre" in T.model_prompt(t.options, n, before)
    tts.speak_lines(project, t, n.id, FakeLocal())
    assert n.overrides["frames"] == 243 and T.sound_given(n)
    assert T.source_sound_problems(t, [n.id], project) == [] and T.cast_problems(project, t, [n.id]) == []
    s = T.Checker(project, t).settings(n.id)
    assert not s.get("audio_refs") and s["source_sound"]["mode"] == "track"
    prompt = T.model_prompt(t.options, n, s)
    assert "voice timbre" not in prompt and '"Guten Morgen."' in prompt and '"Hallo."' in prompt
    assert prompt.count("<Audio") == 2 and T.TRACK_SOUND_TEXT.format(k=1) in prompt
    assert T.node_cast(t.options, n)["voices"] == [] and len(T.clip_cast(t.options, n.prompt)["voices"]) == 2
    n.overrides["sound_over"] = True                                         # laid over: the model's own speech
    assert not T.sound_given(n)
    s = T.Checker(project, t).settings(n.id)
    assert len(s["audio_refs"]) == 2 and "voice timbre" in T.model_prompt(t.options, n, s)
    assert T.source_sound_problems(t, [n.id], project) == []


def test_takes_and_how_lines_are_spoken(project, tmp_path):
    """Another take: other seeds, another file. Delivery and speed: the temperature and a tempo change (local), the
    voice's stability and speed (ElevenLabs; eleven_v3 takes 0 / 0.5 / 1 only). The defaults change nothing."""
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    n = t.add(None, '@Anna says "Hallo, wie geht es dir heute so?" Then she says "Gut."')
    assert tts.speak_options(t) == tts.SPEAK_DEFAULTS == {"takes": 3, "delivery": "natural", "speed": 1.0}
    assert tts.speak_options(given={"takes": 99, "delivery": "odd", "speed": 5}) == {
        "takes": 8, "delivery": "natural", "speed": 1.2}
    loc = FakeLocal()
    a = tts.render_lines(project, t, n.id, loc)
    b = tts.render_lines(project, t, n.id, loc, take=1)
    clones = [c for c in loc.calls if c[0] == "clone"]
    assert all(len(c) == 6 for c in clones)                                   # nothing extra by default
    assert [c[5] for c in clones] == [n.seed, n.seed + 1, n.seed + 7919, n.seed + 7920]
    assert a["file"] != b["file"] and (a["take"], b["take"]) == (0, 1) and a["secs"] == pytest.approx(b["secs"])
    slow = tts.render_lines(project, t, n.id, loc, take=2, speak={"delivery": "lively", "speed": 0.8})
    assert loc.calls[-1][6] == {"temperature": 1.1} and slow["speak"] == {"takes": 3, "delivery": "lively",
                                                                          "speed": 0.8}
    spoken = 0.08 * (32 + 4)
    assert slow["secs"] - a["secs"] == pytest.approx(spoken / 0.8 - spoken, abs=0.1)     # the lines, not the pauses
    tts.render_lines(project, t, n.id, loc, speak={"delivery": "steady"})
    assert loc.calls[-1][6] == {"temperature": 0.7}
    t.options["tts"] = {"provider": "elevenlabs", "speak": {"delivery": "steady", "speed": 1.1}}     # the project's
    tts.set_voice_fields(t, "Anna", eleven={"id": "v1", "name": "Rachel"})
    el = FakeEleven()
    r = tts.render_lines(project, t, n.id, el)
    assert el.calls[-1][6] == {"settings": {"stability": 0.8, "speed": 1.1}} and r["speak"]["speed"] == 1.1
    t.options["tts"]["eleven_model"] = "eleven_v3"
    tts.render_lines(project, t, n.id, el, speak={"delivery": "lively"})
    assert el.calls[-1][6] == {"settings": {"stability": 0.0}}
    tts.render_lines(project, t, n.id, el, speak={})
    assert len(el.calls[-1]) == 6
    # other words for the lines (the Speak the lines window): spoken, and in the take's prompt; the tree is not changed
    r = tts.render_lines(project, t, n.id, loc, lines=['  Na, "du"  da! ', "Schlecht."])
    assert [c[1] for c in loc.calls[-2:]] == ["Na, 'du' da!", "Schlecht."]
    assert r["prompt"] == """@Anna says "Na, 'du' da!" Then she says "Schlecht.\"""" and r["base_prompt"] == n.prompt
    assert [d["line"] for d in r["lines"]] == ["Na, 'du' da!", "Schlecht."] and "Hallo, wie" in n.prompt
    with pytest.raises(tts.TtsError, match="a line is empty"):
        tts.render_lines(project, t, n.id, loc, lines=["Hallo.", " "])
    with pytest.raises(tts.TtsError, match="2 spoken lines, not 1"):
        tts.render_lines(project, t, n.id, loc, lines=["Hallo."])
    assert T.replace_lines('„Hi“, says @Anna. <d>[German] Ja.</d> «Oui»', ["A", "B", "C"]) == \
        '„A“, says @Anna. <d>[German] B</d> «C»'


def test_a_shared_track_stays_and_short_lines_are_filled(project, tmp_path):
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    a = t.add(None, '@Anna says "Hi."')
    r = tts.speak_lines(project, t, a.id, FakeLocal())
    assert r["secs"] == pytest.approx(T.AUDIO_REF_MIN, abs=0.01) and r["frames"] is None   # filled up to 2 s
    b = t.add(a.id, '@Anna says "Hi."')
    b.overrides["source_sound"] = a.overrides["source_sound"]               # another clip plays it too
    tts.speak_lines(project, t, a.id, FakeLocal())
    assert T.track_of(t, r["track"]) is not None                             # kept for b


def test_clip_length_for_the_lines():
    assert tts.frames_for(2.0, 73) is None
    assert tts.frames_for(3.1, 73) == 90
    assert tts.frames_for(14.9, 73) == 362
    with pytest.raises(tts.TtsError, match="at most 15 s"):
        tts.frames_for(15.5, 73)


def test_elevenlabs_lines_and_voices(project, tmp_path):
    t = T.load(project)
    t.options["tts"] = {"provider": "elevenlabs", "eleven_model": "eleven_v3"}
    n = t.add(None, '@Anna says "Hallo du."')
    el = FakeEleven()
    wav = tmp_path / "sample.wav"
    assert tts.make_sample(el, "Ein ganzer Satz, lang genug zum Hören.", "German", wav, voice_id="v1", model="eleven_v3") > 0
    entry = tts.use_sample(project, t, "Anna", wav, "ElevenLabs: Rachel", "Ein ganzer Satz, lang genug zum Hören.", {"id": "v1", "name": "Rachel"})
    c = T.character(t, "Anna")
    assert entry["file"] == c["voice"]["file"] and c["voice"]["text"] == "Ein ganzer Satz, lang genug zum Hören."
    assert c["tts"]["eleven"] == {"id": "v1", "name": "Rachel"}
    T.save(project, t)
    assert T.character(T.load(project), "Anna")["tts"]["eleven"]["id"] == "v1"      # kept in tree.json
    tts.speak_lines(project, t, n.id, el)
    assert el.calls[-1][:5] == ("speak", "Hallo du.", "v1", "eleven_v3", "German")
    # a designed voice (local) replaces it: no ElevenLabs voice any more
    loc = FakeLocal()
    tts.make_sample(loc, "Ein ganzer Satz, lang genug zum Hören.", "German", wav, description="warm voice")
    tts.use_sample(project, t, "Anna", wav, "designed: warm voice", "Ein ganzer Satz, lang genug zum Hören.")
    assert "eleven" not in (T.character(t, "Anna").get("tts") or {})
    assert loc.calls[0] == ("design", "Ein ganzer Satz, lang genug zum Hören.", "German", "warm voice", None)


# ---- the worker protocol (a fake worker script, no model)

FAKE_WORKER = r'''
import json, sys
print("noise from a library")
sys.stdout.write("@@TTS " + json.dumps({"ok": True, "ready": True}) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    r = json.loads(line)
    if r["cmd"] == "quit":
        break
    if r["cmd"] == "fail":
        a = {"ok": False, "error": "it broke"}
    elif r["cmd"] == "hang":
        import time; time.sleep(30); a = {"ok": True}
    else:
        a = {"ok": True, "echo": r, "secs": 1.5, "text": "said"}
    print("more noise")
    sys.stdout.write("@@TTS " + json.dumps(a) + "\n"); sys.stdout.flush()
'''


def fake_cache(tmp_path):
    for repo in (tts.QWEN_BASE, tts.QWEN_DESIGN, tts.WHISPER):
        (tmp_path / "hf" / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc").mkdir(parents=True)
    return tmp_path / "hf"


def test_local_worker_protocol(tmp_path):
    w = tmp_path / "worker.py"
    w.write_text(FAKE_WORKER)
    loc = tts.LocalTTS(python=Path(sys.executable), cache=fake_cache(tmp_path), worker=w, start_timeout=20)
    launched = []
    loc.before_launch = lambda _x: launched.append(1)
    assert loc.problems(design=True, transcribe=True) == []
    a = loc.request({"cmd": "clone", "x": 1})
    assert {k: v for k, v in a["echo"].items() if k != "id"} == {"cmd": "clone", "x": 1} and loc.running() and launched == [1]
    assert loc.clone("Hallo", "German", tmp_path / "r.wav", "t", tmp_path / "o.wav", 5) == 1.5
    assert launched == [1]                                  # one process for every request
    with pytest.raises(tts.TtsError, match="it broke"):
        loc.request({"cmd": "fail"})
    assert loc.running()
    with pytest.raises(tts.TtsError, match="no answer"):
        loc.request({"cmd": "hang"}, timeout=1)
    assert not loc.running()                                # a worker that hangs is stopped
    loc.request({"cmd": "x"})
    assert launched == [1, 1] and loc.stop() and not loc.running()

    def refuse(_x):
        raise tts.TtsError("no room")
    loc.before_launch = refuse
    with pytest.raises(tts.TtsError, match="no room"):
        loc.request({"cmd": "x"})


def test_removing_music_needs_its_package_and_model(tmp_path):
    w = tmp_path / "worker.py"
    w.write_text(FAKE_WORKER)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    cache = fake_cache(tmp_path)
    loc = tts.LocalTTS(python=env / "bin" / "python", cache=cache, worker=w, start_timeout=20)
    assert loc.problems() == [] and "audio-separator" in loc.problems(isolate=True)[0]
    with pytest.raises(tts.TtsError, match="video-chain setup --only tts"):
        loc.isolate(tmp_path / "in.wav", tmp_path / "out.wav")
    assert not loc.running()                                # (not even started)
    (env / "lib" / "python3.12" / "site-packages" / "audio_separator").mkdir(parents=True)
    assert tts.SEPARATOR_MODEL in loc.problems(isolate=True)[0]          # the package is there, the model is not
    (cache / tts.SEPARATOR_DIR).mkdir()
    for f in (tts.SEPARATOR_MODEL, Path(tts.SEPARATOR_MODEL).stem + ".yaml"):
        (cache / tts.SEPARATOR_DIR / f).write_text("x")
    assert loc.problems(isolate=True) == []
    try:
        assert loc.isolate(tmp_path / "in.wav", tmp_path / "out.wav") == 1.5
        echo = loc.request({"cmd": "x"})["echo"]
    finally:
        loc.stop()
    assert {k: v for k, v in echo.items() if k != "id"} == {"cmd": "x"}


def test_local_problems(tmp_path):
    loc = tts.LocalTTS(python=tmp_path / "nope" / "python", cache=tmp_path)
    bad = loc.problems(design=True)
    assert "TTS environment is missing" in bad[0] and any(tts.QWEN_DESIGN in b for b in bad)


# ---- ElevenLabs against a local fake server

class _Fake(http.server.BaseHTTPRequestHandler):
    mp3: bytes = b""
    seen: list = []

    def log_message(self, *_a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _Fake.seen.append(("GET", self.path, self.headers.get("xi-api-key")))
        if self.headers.get("xi-api-key") != "good":
            return self._send(401, json.dumps({"detail": {"message": "Invalid API key"}}).encode())
        if self.path.startswith("/v2/voices"):
            page2 = "next_page_token=p2" in self.path
            body = {"voices": [{"voice_id": "v2" if page2 else "v1", "name": "Zed" if page2 else "Anna",
                                "labels": {"gender": "female"}}], "has_more": not page2,
                    "next_page_token": None if page2 else "p2"}
            return self._send(200, json.dumps(body).encode())
        if self.path == "/v1/models":
            return self._send(200, json.dumps([
                {"model_id": "eleven_v3", "name": "Eleven v3", "can_do_text_to_speech": True,
                 "languages": [{"language_id": "de"}]},
                {"model_id": "scribe", "name": "Speech to text", "can_do_text_to_speech": False}]).encode())
        self._send(404, b"{}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Fake.seen.append(("POST", self.path, body))
        if self.path.startswith("/v1/text-to-speech/"):
            return self._send(200, _Fake.mp3, "audio/mpeg")
        if self.path == "/v1/voices/add":
            return self._send(200, json.dumps({"voice_id": "new1"}).encode())
        self._send(404, b"{}")


@pytest.fixture
def eleven_server(tmp_path):
    mp3 = tmp_path / "x.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=1.2",
                    "-c:a", "libmp3lame", str(mp3)], check=True)
    _Fake.mp3, _Fake.seen = mp3.read_bytes(), []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_elevenlabs_api(eleven_server, tmp_path):
    el = tts.ElevenLabs(key="good", url=eleven_server)
    assert [v["id"] for v in el.voices()] == ["v1", "v2"]                   # both pages, by name
    assert el.models() == [{"id": "eleven_v3", "name": "Eleven v3", "languages": ["de"]}]
    out = tmp_path / "line.wav"
    secs = el.speak("Hallo du.", "v1", out, "eleven_v3", "German", seed=7)
    assert 1.1 < secs < 1.4 and chain.audio_seconds(out) == pytest.approx(secs)
    post = next(s for s in _Fake.seen if s[0] == "POST")
    assert post[1] == "/v1/text-to-speech/v1?output_format=mp3_44100_128"
    assert json.loads(post[2]) == {"text": "Hallo du.", "model_id": "eleven_v3", "seed": 7, "language_code": "de"}
    el.speak("Hi.", "v1", out, "eleven_multilingual_v2", "German")
    assert "language_code" not in json.loads([s for s in _Fake.seen if s[0] == "POST"][-1][2])   # (not taken)
    assert "voice_settings" not in json.loads([s for s in _Fake.seen if s[0] == "POST"][-1][2])
    el.speak("Hi.", "v1", out, settings={"stability": 0.8, "speed": 1.1})
    assert json.loads([s for s in _Fake.seen if s[0] == "POST"][-1][2])["voice_settings"] == {"stability": 0.8,
                                                                                               "speed": 1.1}
    sample = tmp_path / "s.wav"
    tone(sample, 2.0)
    assert el.clone("Anna", sample) == "new1"
    body = [s for s in _Fake.seen if s[1] == "/v1/voices/add"][0][2]
    assert b'name="name"\r\n\r\nAnna' in body and b'filename="s.wav"' in body
    with pytest.raises(tts.TtsError, match="rejected the API key.*Invalid API key"):
        tts.ElevenLabs(key="bad", url=eleven_server).voices()


def test_a_voice_sample_cloned_at_elevenlabs(eleven_server, project, tmp_path):
    t = T.load(project)
    el = tts.ElevenLabs(key="good", url=eleven_server)
    with pytest.raises(tts.TtsError, match="no voice sample"):
        tts.clone_sample(project, t, "Anna", el)
    give_voice(project, t, "Anna", tmp_path)
    v = tts.clone_sample(project, t, "Anna", el)
    assert v == {"id": "new1", "name": "Anna (video-chain)", "file": T.character(t, "Anna")["voice"]["file"]}
    body = [s for s in _Fake.seen if s[1] == "/v1/voices/add"][-1][2]
    assert b"Anna (video-chain)" in body and b'filename="voice1.wav"' in body and b"RIFF" in body
    assert "tts" not in T.character(t, "Anna")                                # (clone_sample only reads)
    assert tts.set_eleven(t, "Anna", v)
    assert T.character(t, "Anna")["tts"]["eleven"] == {"id": "new1", "name": "Anna (video-chain)"}
    # a voice from a longer source: its clone reference is what is uploaded, and what local TTS clones
    give_voice(project, t, "Ben", tmp_path, secs=12.0)
    assert tts.clone_sample(project, t, "Ben", el)["file"] == T.character(t, "Ben")["voice"]["file"]
    body = [s for s in _Fake.seen if s[1] == "/v1/voices/add"][-1][2]
    assert b'filename="voice2_ref.wav"' in body and len(body) > 12 * 48000 * 4
    n = t.add(None, '@Ben says "Hallo du."')
    loc = FakeLocal()
    r = tts.render_lines(project, t, n.id, loc)
    assert loc.calls[0] == ("transcribe", "voice2_ref.wav", "de")
    assert loc.calls[1][:5] == ("clone", "Hallo du.", "German", "voice2_ref.wav", "what the sample says")
    tts.apply_lines(project, t, n.id, r)
    assert T.character(t, "Ben")["voice"]["text"] == "what the sample says"
    # the sample was replaced while the clone was made: the clone is not used
    give_voice(project, t, "Anna", tmp_path)
    assert not tts.set_eleven(t, "Anna", {**v, "id": "new2"})
    assert T.character(t, "Anna")["tts"]["eleven"]["id"] == "new1"
    assert not tts.set_eleven(t, "Nobody", v)
    assert tts.ElevenLabs(key="").problems() and tts.ElevenLabs(key="x").problems() == []


def test_join_lines(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    for f, secs in ((a, 1.0), (b, 0.5)):
        tone(f, secs)
        chain.to_wav(f, f.with_name(f.stem + "48.wav"))
    secs = tts.join_lines([tmp_path / "a48.wav", tmp_path / "b48.wav"], tmp_path / "out.wav")
    assert secs == pytest.approx(tts.LEAD_IN + 1.0 + tts.GAP + 0.5 + tts.TAIL, abs=0.01)
    with pytest.raises(tts.TtsError, match="48 kHz"):
        tts.join_lines([a], tmp_path / "x.wav")


@pytest.mark.models
def test_qwen3_tts_speaks_german_and_english(tmp_path):
    """The real models (about 5 GB of VRAM): a designed voice, lines cloned from it, heard back by Whisper."""
    loc = tts.LocalTTS(log=tmp_path / "worker.log")
    if loc.problems(design=True, transcribe=True):
        pytest.skip("; ".join(loc.problems(design=True, transcribe=True)))
    try:
        ref = tmp_path / "ref.wav"
        loc.design(tts.SAMPLE_TEXT["German"], "German", "A man, about forty, deep calm voice.", ref, seed=3)
        heard = loc.transcribe(ref, "de").lower()
        assert "schön dich zu sehen" in heard
        for lang, line, code in (("German", "Wir sehen uns morgen im Park.", "de"),
                                 ("English", "See you tomorrow in the park.", "en")):
            out = tmp_path / f"{code}.wav"
            assert loc.clone(line, lang, ref, tts.SAMPLE_TEXT["German"], out, seed=1) > 1.0
            got = loc.transcribe(out, code).lower().strip(" .!")
            assert got == line.lower().strip(" .!"), got
        # steady / lively (the sampling temperature): still the same words
        for temp in (0.7, 1.1):
            out = tmp_path / f"t{temp}.wav"
            assert loc.clone("Wir sehen uns morgen im Park.", "German", ref, tts.SAMPLE_TEXT["German"], out, seed=1,
                             temperature=temp) > 1.0
            assert loc.transcribe(out, "de").lower().strip(" .!") == "wir sehen uns morgen im park"
        # music under the voice is taken out: the tones are gone, the words are still there
        mixed, clean = tmp_path / "mixed.wav", tmp_path / "clean.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ref), "-f", "lavfi", "-i",
                        "sine=frequency=262:duration=30,volume=2", "-f", "lavfi", "-i",
                        "sine=frequency=392:duration=30,volume=2", "-filter_complex",
                        "[0:a][1:a][2:a]amix=inputs=3:duration=first:normalize=0", "-ar", "48000", "-ac", "2",
                        str(mixed)], check=True)

        def tone_db(path):                              # how loud the 262 Hz tone of the "music" is
            out = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-af",
                                  "bandpass=f=262:width_type=h:w=10,volumedetect", "-f", "null", "-"],
                                 capture_output=True, text=True).stderr
            return float(out.split("mean_volume:")[1].split("dB")[0])
        secs = loc.isolate(mixed, clean)
        assert abs(secs - chain.audio_seconds(mixed)) < 0.1
        assert tone_db(clean) < tone_db(mixed) - 20, (tone_db(mixed), tone_db(clean))
        assert "schön dich zu sehen" in loc.transcribe(clean, "de").lower()
        # a long clone reference (a voice set from a longer source: up to 30 s, transcribed by Whisper)
        text = ("Guten Morgen, schön dich zu sehen. Ich wollte dir schon lange erzählen, was letzte Woche passiert "
                "ist. Wir sind früh am Morgen losgefahren, die Straßen waren noch ganz leer, und über den Feldern "
                "lag dichter Nebel. Nach einer Stunde kam endlich die Sonne heraus, und wir haben am See "
                "angehalten, um in aller Ruhe zu frühstücken.")
        long = tmp_path / "long.wav"
        secs = loc.design(text, "German", "A man, about forty, deep calm voice.", long, seed=3)
        cut = tmp_path / "long30.wav"
        chain.to_wav(long, cut, 0.0, T.VOICE_REF_MAX_SECS if secs > T.VOICE_REF_MAX_SECS else None)
        assert chain.audio_seconds(cut) > 15, secs
        said = loc.transcribe(cut, "de")
        assert "nebel" in said.lower(), said
        for lang, line, code in (("German", "Wir sehen uns morgen im Park.", "de"),
                                 ("English", "See you tomorrow in the park.", "en")):
            out = tmp_path / f"long_{code}.wav"
            assert 1.0 < loc.clone(line, lang, cut, said, out, seed=1) < 8.0
            got = loc.transcribe(out, code).lower().strip(" .!")
            assert got == line.lower().strip(" .!"), got
    finally:
        loc.stop()


def test_lines_too_long_for_a_clip_change_nothing(project, tmp_path):
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    n = t.add(None, '@Anna says "Hallo."')
    r = tts.render_lines(project, t, n.id, FakeLocal())
    r["secs"] = 16.0                                          # (more than the 15 s a clip can be)
    before = (json.dumps(t.options, sort_keys=True), dict(n.overrides))
    with pytest.raises(tts.TtsError, match="at most 15 s"):
        tts.apply_lines(project, t, n.id, r)
    assert (json.dumps(t.options, sort_keys=True), n.overrides) == before     # no track, no transcript, no frames


def test_a_transcript_does_not_go_to_a_voice_replaced_meanwhile(project, tmp_path):
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    n = t.add(None, '@Anna says "Hallo."')
    r = tts.render_lines(project, t, n.id, FakeLocal())
    assert r["texts"] and r["voice_files"]["Anna"] == T.character(t, "Anna")["voice"]["file"]
    other = tmp_path / "other.wav"
    tone(other, 3.0, 330)
    T.set_voice(project, t, "Anna", other)                    # replaced while the lines were made
    tts.apply_lines(project, t, n.id, r)
    assert "text" not in T.character(t, "Anna")["voice"]


# ---- Fish S2 Pro and [tags] in lines


class FakeFish(FakeLocal):
    """Fish S2 Pro without the model (the same calls as Qwen3-TTS: clone, transcribe)."""
    name = "fish"


def test_tags_go_only_to_the_engines_that_hear_them(project, tmp_path):
    """Fish S2 Pro and ElevenLabs v3 get a line with its [tags]; Qwen3-TTS and the older ElevenLabs models get it
    without them; the track keeps the line as written."""
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    n = t.add(None, '@Anna says "[angry] Lass mich [sighs] endlich in Ruhe!"')
    fish = FakeFish()
    r = tts.speak_lines(project, t, n.id, fish)
    assert [c[1] for c in fish.calls if c[0] == "clone"] == ["[angry] Lass mich [sighs] endlich in Ruhe!"]
    assert T.track_of(t, r["track"])["tts"] == {"clip": n.id, "provider": "fish", "language": "German",
                                                "lines": [{"name": "Anna",
                                                           "line": "[angry] Lass mich [sighs] endlich in Ruhe!"}]}
    loc = FakeLocal()
    tts.speak_lines(project, t, n.id, loc)
    assert [c[1] for c in loc.calls if c[0] == "clone"] == ["Lass mich endlich in Ruhe!"]
    T.character(t, "Anna")["tts"] = {"eleven": {"id": "v1", "name": "Rachel"}}
    for model, said in (("eleven_multilingual_v2", "Lass mich endlich in Ruhe!"),
                        ("eleven_v3", "[angry] Lass mich [sighs] endlich in Ruhe!")):
        t.options["tts"] = {"provider": "elevenlabs", "eleven_model": model}
        el = FakeEleven()
        tts.speak_lines(project, t, n.id, el)
        assert el.calls[-1][1] == said
    assert tts.takes_tags("fish") and tts.takes_tags("elevenlabs", "eleven_v3")
    assert not tts.takes_tags("local") and not tts.takes_tags("elevenlabs", "eleven_multilingual_v2")
    n.prompt = '@Anna says "[sighs]"'                       # a sound alone: Fish makes it, Qwen has nothing to say
    t.options["tts"] = {"provider": "local"}
    tts.render_lines(project, t, n.id, FakeFish())
    with pytest.raises(tts.TtsError, match="nothing to say besides its"):
        tts.render_lines(project, t, n.id, FakeLocal())


def test_fish_transcribes_first_and_has_its_own_delivery(project, tmp_path):
    """Every voice sample is transcribed before the first line is spoken (Whisper and Fish never share the GPU);
    steady / lively are Fish's own temperatures (natural: the model's own)."""
    t = T.load(project)
    give_voice(project, t, "Anna", tmp_path)
    give_voice(project, t, "Ben", tmp_path)
    n = t.add(None, '@Anna says "[happy] Hallo." @Ben answers "[tired] Na?" @Anna says "[laughs] Ja!"')
    fish = FakeFish()
    tts.render_lines(project, t, n.id, fish, speak={"delivery": "steady"})
    kinds = [c[0] for c in fish.calls]
    assert kinds == ["transcribe", "transcribe", "clone", "clone", "clone"]
    assert all(c[-1] == {"temperature": 0.5} for c in fish.calls if c[0] == "clone")
    fish = FakeFish()
    tts.render_lines(project, t, n.id, fish, speak={"delivery": "natural"})
    assert all(len(c) == 6 for c in fish.calls if c[0] == "clone")      # (no temperature given)
    fish = FakeFish()
    tts.render_lines(project, t, n.id, fish, speak={"delivery": "lively"})
    assert all(c[-1] == {"temperature": 0.9} for c in fish.calls if c[0] == "clone")


def test_the_clip_model_never_sees_the_tags(project, tmp_path):
    assert T.without_tags("[angry] Lass mich [sighs] endlich in Ruhe [quietly]!") == "Lass mich endlich in Ruhe!"
    assert T.without_tags("Nur Text.") == "Nur Text."
    text = '[Shot 2] @Anna says "[angry] Lass mich!" and <d>[German] [laughs] Ja.</d> [a note]'
    assert T.without_speech_tags(text) == '[Shot 2] @Anna says "Lass mich!" and <d>[German] Ja.</d> [a note]'
    t = T.load(project)
    prompt = '@Anna looks at @Ben and says "[whispering, scared] Hörst du das [gasps] auch?"'
    for fmt in ("classic", "h3"):
        t.options["prompt_format"] = fmt
        for voiced in (False, True):
            if voiced:
                give_voice(project, t, "Anna", tmp_path)
            out = T.clip_cast(t.options, prompt)["prompt"]
            assert "Hörst du das auch?" in out, (fmt, voiced, out)
            assert not any(w in out for w in ("whispering", "scared", "gasps")), (fmt, voiced, out)
    assert T.dialogue(t.options, prompt)[0]["line"] == "[whispering, scared] Hörst du das [gasps] auch?"


def test_fish_worker_protocol_and_its_checks(tmp_path, monkeypatch):
    """The Fish worker speaks the same protocol (clone with its model's snapshot); before it starts the Qwen3-TTS
    helper is stopped and free RAM and VRAM are checked; transcribing goes to the helper (Fish stops first, the
    helper right after); it runs in a capped systemd scope when systemd is there."""
    w = tmp_path / "worker.py"
    w.write_text(FAKE_WORKER)
    cache = fake_cache(tmp_path)
    (cache / ("models--" + tts.FISH_MODEL.replace("/", "--")) / "snapshots" / "f1").mkdir(parents=True)
    helper = tts.LocalTTS(python=Path(sys.executable), cache=cache, worker=w, start_timeout=20)
    fish = tts.FishTTS(python=Path(sys.executable), cache=cache, worker=w, start_timeout=20, helper=helper,
                       mem_cap_gb=None)
    from video_chain import llm_server
    monkeypatch.setattr(tts, "mem_available_mb", lambda: 40000)
    monkeypatch.setattr(llm_server, "free_vram_mb", lambda: 23000)
    assert fish.problems(transcribe=True) == []
    try:
        assert fish.transcribe(tmp_path / "v.wav", "de") == "said" and not helper.running() and not fish.running()
        helper.request({"cmd": "x"})                        # (the helper runs) ...
        assert fish.clone("[angry] Hallo", "German", tmp_path / "r.wav", "t", tmp_path / "o.wav", 5) == 1.5
        assert not helper.running() and fish.running()      # ... and goes before Fish starts
        echo = fish.request({"cmd": "x"})["echo"]
        assert fish.transcribe(tmp_path / "v.wav") == "said" and not fish.running()
    finally:
        fish.stop()
        helper.stop()
    assert {k: v for k, v in echo.items() if k != "id"} == {"cmd": "x"}
    monkeypatch.setattr(tts, "mem_available_mb", lambda: 3000)
    with pytest.raises(tts.TtsError, match="GB of RAM is free"):
        fish.clone("Hallo", "German", tmp_path / "r.wav", "t", tmp_path / "o.wav")
    monkeypatch.setattr(tts, "mem_available_mb", lambda: 40000)
    monkeypatch.setattr(llm_server, "free_vram_mb", lambda: 12000)
    with pytest.raises(tts.TtsError, match="GB of VRAM is free"):
        fish.clone("Hallo", "German", tmp_path / "r.wav", "t", tmp_path / "o.wav")
    assert not fish.running()
    capped = tts.FishTTS(python=Path(sys.executable), worker=w, mem_cap_gb=12)
    monkeypatch.setattr(tts, "_user_systemd", lambda: True)
    cmd = capped._command()
    assert cmd[:3] == ["systemd-run", "--user", "--scope"] and "MemoryMax=12G" in cmd and "MemorySwapMax=0" in cmd
    assert cmd[-3:] == [sys.executable, "-I", str(w)]
    monkeypatch.setattr(tts, "_user_systemd", lambda: False)
    assert capped._command() == [sys.executable, "-I", str(w)]


def test_fish_problems_and_engine_choice(tmp_path):
    helper = tts.LocalTTS(python=tmp_path / "nope" / "python", cache=tmp_path)
    fish = tts.FishTTS(python=tmp_path / "nofish" / "python", cache=tmp_path, helper=helper)
    bad = fish.problems(transcribe=True)
    assert "Fish S2 Pro's environment is missing" in bad[0] and "video-chain setup --with tts-fish" in bad[0]
    assert any(tts.FISH_MODEL in b for b in bad) and any("TTS environment" in b for b in bad)
    t = T.Tree()
    t.options["tts"] = {"provider": "fish"}
    loc = tts.LocalTTS()
    eng = tts.engine(t, loc)
    assert eng.name == "fish" and eng.helper is loc
    assert tts.engine(t, loc, fish) is fish
    t.options["tts"] = {"provider": "local"}
    assert tts.engine(t, loc, fish) is loc
    assert "fish" in tts.PROVIDERS and "non-commercial" in tts.PROVIDERS["fish"]


@pytest.mark.models
def test_fish_s2_pro_speaks_tones_in_a_cloned_voice(tmp_path):
    """The real model (about 19 GB of VRAM, run alone): a designed voice (Qwen3-TTS) cloned by Fish S2 Pro, its lines
    heard back by Whisper: the words are right, no [tag] is read out, and [shouting] is louder than [whispering].
    The environments: tts.FISH_PYTHON and tts.TTS_PYTHON, or $FISH_PYTHON and $TTS_PYTHON (another checkout's)."""
    import os

    from video_chain.llm_server import free_vram_mb
    py = Path(os.environ.get("FISH_PYTHON") or tts.FISH_PYTHON)
    helper = tts.LocalTTS(python=Path(os.environ.get("TTS_PYTHON") or tts.TTS_PYTHON), log=tmp_path / "worker.log")
    fish = tts.FishTTS(python=py, log=tmp_path / "fish.log", helper=helper)
    bad = fish.problems(design=True, transcribe=True)
    if bad:
        pytest.skip("; ".join(bad))
    free = free_vram_mb()
    if free is not None and free < tts.FISH_VRAM_MB:
        pytest.skip(f"only {free} MB of VRAM free; Fish S2 Pro needs {tts.FISH_VRAM_MB}")

    def mean_db(path):
        out = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        return float(out.split("mean_volume:")[1].split("dB")[0])
    try:
        ref = tmp_path / "ref.wav"
        fish.design(tts.SAMPLE_TEXT["German"], "German", "Eine Frau um die dreißig, lebhafte, ausdrucksstarke Stimme.",
                    ref, seed=3)
        said = fish.transcribe(ref, "de")
        assert "dich zu sehen" in said.lower(), said          # (the reference only: its umlaut is not the point)
        got, loud = {}, {}
        for key, line in (("plain", "Wir sehen uns morgen im Park."),
                          ("shouting", "[shouting] Wir sehen uns morgen im Park!"),
                          ("whispering", "[whispering] Wir sehen uns morgen im Park."),
                          ("inline", "Wir sehen uns [laughs] morgen im Park.")):
            out = tmp_path / f"{key}.wav"
            assert 1.0 < fish.clone(line, "German", ref, said, out, seed=1) < 10.0
            loud[key] = mean_db(out)
            got[key] = out
        assert fish.running()
        for key, out in got.items():                        # (Fish stops for Whisper)
            heard = fish.transcribe(out, "de").lower()
            assert "wir sehen uns morgen im park" in heard.replace(",", ""), (key, heard)
            assert not any(w in heard for w in ("shouting", "whispering", "laughs")), (key, heard)
        assert loud["shouting"] > loud["whispering"] + 3, loud
    finally:
        fish.stop()
        helper.stop()
