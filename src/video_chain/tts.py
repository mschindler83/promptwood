"""Text to speech: the spoken lines of a clip as its sound (the picture is made to it, lips in sync), and voice samples
for characters. Two providers:

- local: Qwen3-TTS (Apache 2.0; German, English and 8 more languages) in its own environment (.venv-tts next to the
  app), run by tts_worker.py, started when needed and stopped right after. A character speaks in the voice of its
  voice sample (Base model: cloned from the sample and what it says, transcribed once with Whisper); a new voice is
  made from a description (VoiceDesign model).
- elevenlabs: the ElevenLabs API with the user's key (system keyring or $ELEVENLABS_API_KEY); each character is
  given one of the account's voices, or an instant voice clone of its voice sample (made in the account).

Spoken lines are the quoted lines of a prompt ("..." after @Name, see tree.dialogue); they are spoken one after the
other, with a short pause before the first and between them, and become an audio track of the clip."""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

from . import chain
from . import tree as T

APP_DIR = Path(__file__).resolve().parents[2]
TTS_PYTHON = APP_DIR / ".venv-tts" / "bin" / "python"
WORKER = Path(__file__).resolve().with_name("tts_worker.py")
HF_CACHE = Path("/path/to/ai/hf-models")
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
WHISPER = "openai/whisper-large-v3-turbo"
LOCAL_VRAM_MB = 5500                     # measured: 4.5 GB peak for the 1.7B models (bf16), Whisper turbo less
SETUP_HINT = ("in the app folder: uv venv -p 3.12 .venv-tts && VIRTUAL_ENV=.venv-tts uv pip install qwen-tts "
              "soundfile; the models: Qwen/Qwen3-TTS-12Hz-1.7B-Base and -VoiceDesign in " + str(HF_CACHE))

# the dialogue language (project setting) -> Qwen3-TTS's name / ISO 639-1 for ElevenLabs
LANGUAGES = {"English": "en", "German": "de", "French": "fr", "Spanish": "es", "Italian": "it", "Portuguese": "pt",
             "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "Korean": "ko"}
SAMPLE_TEXT = {"German": "Hallo, schön dich zu sehen. Heute ist ein wirklich guter Tag, findest du nicht auch?",
               "English": "Hi, it's good to see you. Today is a really good day, don't you think?"}
LEAD_IN, GAP, TAIL = 0.25, 0.35, 0.3     # seconds of silence before the first line, between lines, after the last
PROVIDERS = {"local": "Qwen3-TTS (local)", "elevenlabs": "ElevenLabs (API key)"}
DEFAULT_OPTIONS = {"provider": "local", "eleven_model": "eleven_multilingual_v2"}

KEYRING_SERVICE, KEYRING_USER = "video-chain", "elevenlabs-api-key"
ELEVEN_URL = "https://api.elevenlabs.io"


class TtsError(RuntimeError):
    pass


def options(tree_or_options) -> dict:
    o = tree_or_options.options if hasattr(tree_or_options, "options") else tree_or_options
    return {**DEFAULT_OPTIONS, **(o.get("tts") or {})}


def language(tree_or_options) -> str:
    """The language the lines are spoken in: the project's dialogue language."""
    o = tree_or_options.options if hasattr(tree_or_options, "options") else tree_or_options
    return ((o.get("identity") or {}).get("language") or "English").strip() or "English"


def snapshot(repo: str, cache: Path = HF_CACHE) -> Path | None:
    """The downloaded snapshot of a Hugging Face repo in the model cache (None: not there)."""
    d = Path(cache) / ("models--" + repo.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.is_file():
        snap = d / "snapshots" / ref.read_text().strip()
        if snap.is_dir():
            return snap
    snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


# ----------------------------------------------------------------------------- local: Qwen3-TTS


class LocalTTS:
    """The Qwen3-TTS worker process: started on the first request, stopped with stop(). before_launch (optional,
    called just before the process starts) can make room in VRAM or raise TtsError to refuse."""

    name = "local"

    def __init__(self, python: Path = TTS_PYTHON, cache: Path = HF_CACHE, log: Path | None = None,
                 worker: Path = WORKER, start_timeout: float = 120.0):
        self.python, self.cache, self.log, self.worker = Path(python), Path(cache), log, Path(worker)
        self.start_timeout = start_timeout
        self.proc: subprocess.Popen | None = None
        self._answers: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.before_launch = None

    def problems(self, design: bool = False, transcribe: bool = False) -> list[str]:
        out = []
        if not self.python.is_file():
            out.append(f"the TTS environment is missing ({self.python}); set it up {SETUP_HINT}")
        for repo, need in ((QWEN_BASE, True), (QWEN_DESIGN, design), (WHISPER, transcribe)):
            if need and snapshot(repo, self.cache) is None:
                out.append(f"the model {repo} is not in {self.cache}")
        return out

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _start(self):
        bad = self.problems()
        if bad:
            raise TtsError("; ".join(bad))
        if self.before_launch is not None:
            self.before_launch(self)
        env = {**os.environ, "HF_HUB_CACHE": str(self.cache), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        err = open(self.log, "ab") if self.log else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen([str(self.python), "-I", str(self.worker)], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=err, env=env, cwd=tempfile.gettempdir(),
                                         text=True, bufsize=1)
        finally:
            if err is not subprocess.DEVNULL:
                err.close()
        self._answers = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self._answers), daemon=True).start()
        first = self._wait(self.start_timeout)
        if not first.get("ready"):
            raise TtsError(f"the TTS worker did not start: {first.get('error', 'no answer')}")

    @staticmethod
    def _read(proc, answers):
        for line in proc.stdout:
            if line.startswith("@@TTS "):
                try:
                    answers.put(json.loads(line[6:]))
                except ValueError:
                    pass
        answers.put({"ok": False, "error": "the TTS worker stopped", "dead": True})

    def _wait(self, timeout: float) -> dict:
        try:
            return self._answers.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise TtsError(f"the TTS worker gave no answer in {timeout:.0f} s") from None

    def request(self, r: dict, timeout: float = 600.0) -> dict:
        with self._lock:
            if not self.running():
                self._start()
            p = self.proc                       # (stop() from the GUI thread may clear self.proc meanwhile)
            if p is None:
                raise TtsError("the TTS worker was stopped")
            try:
                p.stdin.write(json.dumps(r) + "\n")
                p.stdin.flush()
            except (OSError, ValueError):       # (it died, or was stopped, before it read the request)
                self.stop()
                raise TtsError("the TTS worker stopped before it got the request") from None
            a = self._wait(timeout)
        if not a.get("ok"):
            raise TtsError(a.get("error") or "the TTS worker failed")
        return a

    def stop(self) -> bool:
        p, self.proc = self.proc, None
        if p is None or p.poll() is not None:
            return False
        try:
            p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            p.stdin.flush()
            p.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            p.kill()
            p.wait(timeout=10)
        return True

    def _path(self, repo: str) -> str:
        s = snapshot(repo, self.cache)
        if s is None:
            raise TtsError(f"the model {repo} is not in {self.cache}")
        return str(s)

    # ---- what it does
    def design(self, text: str, lang: str, instruct: str, out: Path, seed: int | None = None) -> float:
        return self.request({"cmd": "design", "model": self._path(QWEN_DESIGN), "text": text, "language": lang,
                             "instruct": instruct, "out": str(out), "seed": seed})["secs"]

    def clone(self, text: str, lang: str, ref_audio: Path, ref_text: str | None, out: Path,
              seed: int | None = None) -> float:
        return self.request({"cmd": "clone", "model": self._path(QWEN_BASE), "text": text, "language": lang,
                             "ref_audio": str(ref_audio), "ref_text": ref_text, "out": str(out), "seed": seed})["secs"]

    def transcribe(self, audio: Path, lang: str | None = None) -> str:
        return self.request({"cmd": "transcribe", "model": self._path(WHISPER), "audio": str(audio),
                             "language": lang})["text"]


# ----------------------------------------------------------------------------- ElevenLabs


def eleven_key() -> str:
    """$ELEVENLABS_API_KEY, else the key kept in the system keyring; "" without one."""
    env = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if env:
        return env
    try:
        import keyring
        return (keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or "").strip()
    except Exception:  # noqa: BLE001  (no keyring now)
        return ""


def set_eleven_key(key: str | None) -> None:
    """Keep the ElevenLabs API key in the system keyring (empty: remove it). RuntimeError: no keyring."""
    key = (key or "").strip()
    try:
        import keyring
        import keyring.errors
        if key:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the system keyring is not available ({e})") from e


class ElevenLabs:
    """The ElevenLabs API with the user's key: the account's voices and TTS models, a line in a voice, an instant
    voice clone of a voice sample."""

    name = "elevenlabs"

    def __init__(self, key: str | None = None, url: str = ELEVEN_URL, timeout: float = 120.0):
        self.key = (key if key is not None else eleven_key()).strip()
        self.url, self.timeout = url.rstrip("/"), timeout

    def problems(self, **_kw) -> list[str]:
        return [] if self.key else ["no ElevenLabs API key (Settings → ElevenLabs API key)"]

    def running(self) -> bool:
        return False

    def stop(self) -> bool:
        return False

    def _call(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None) -> bytes:
        if not self.key:
            raise TtsError("no ElevenLabs API key (Settings → ElevenLabs API key)")
        headers = {"xi-api-key": self.key, "Accept": "*/*"}
        if ctype:
            headers["Content-Type"] = ctype
        req = urllib.request.Request(self.url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                d = json.loads(raw).get("detail")
                msg = d.get("message") if isinstance(d, dict) else d
            except (ValueError, AttributeError):
                msg = raw[:200]
            if e.code == 401:
                raise TtsError(f"ElevenLabs rejected the API key ({msg or 'unauthorized'})") from None
            raise TtsError(f"ElevenLabs: {msg or e.reason} (HTTP {e.code})") from None
        except OSError as e:
            raise TtsError(f"ElevenLabs could not be reached ({e})") from None

    def _json(self, method: str, path: str, data: dict | None = None):
        body = json.dumps(data).encode() if data is not None else None
        return json.loads(self._call(method, path, body, "application/json" if body else None) or b"{}")

    def voices(self) -> list[dict]:
        """The account's voices: [{"id", "name", "category", "labels"}], by name."""
        out, token = [], None
        for _ in range(20):
            q = {"page_size": 100, **({"next_page_token": token} if token else {})}
            d = self._json("GET", "/v2/voices?" + urllib.parse.urlencode(q))
            out += [{"id": v["voice_id"], "name": v.get("name") or v["voice_id"], "category": v.get("category"),
                     "labels": v.get("labels") or {}} for v in d.get("voices") or []]
            token = d.get("next_page_token")
            if not d.get("has_more") or not token:
                break
        return sorted(out, key=lambda v: v["name"].lower())

    def models(self) -> list[dict]:
        """The TTS models: [{"id", "name", "languages": [ISO codes]}]."""
        d = self._json("GET", "/v1/models")
        return [{"id": m["model_id"], "name": m.get("name") or m["model_id"],
                 "languages": [x.get("language_id") for x in m.get("languages") or []]}
                for m in (d if isinstance(d, list) else []) if m.get("can_do_text_to_speech")]

    def speak(self, text: str, voice_id: str, out: Path, model: str = "eleven_multilingual_v2",
              lang: str | None = None, seed: int | None = None) -> float:
        """A line in the voice as 48 kHz WAV at out; returns its length."""
        body = {"text": text, "model_id": model}
        if seed is not None:
            body["seed"] = int(seed) % 4294967296
        code = LANGUAGES.get(lang or "")
        if code and "multilingual_v2" not in model:     # (multilingual_v2 does not take a language)
            body["language_code"] = code
        mp3 = self._call("POST", f"/v1/text-to-speech/{urllib.parse.quote(voice_id)}?output_format=mp3_44100_128",
                         json.dumps(body).encode(), "application/json")
        with tempfile.TemporaryDirectory(prefix="video-chain-tts-") as tmp:
            src = Path(tmp) / "line.mp3"
            src.write_bytes(mp3)
            chain.to_wav(src, Path(out))
        return chain.audio_seconds(Path(out)) or 0.0

    def clone(self, name: str, sample: Path, description: str = "") -> str:
        """An instant voice clone of a voice sample in the account (a paid plan); returns its voice id."""
        b = uuid.uuid4().hex
        parts = [("name", name), ("description", description or "made by video-chain from a voice sample")]
        body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                        for k, v in parts)
        body += (f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{Path(sample).name}"\r\n'
                 "Content-Type: audio/wav\r\n\r\n").encode() + Path(sample).read_bytes() + f"\r\n--{b}--\r\n".encode()
        d = json.loads(self._call("POST", "/v1/voices/add", body, f"multipart/form-data; boundary={b}"))
        return d["voice_id"]


def engine(tree_or_options, local: LocalTTS | None = None):
    """The provider the project uses (local: the given worker, one per app)."""
    return ElevenLabs() if options(tree_or_options)["provider"] == "elevenlabs" else (local or LocalTTS())


# ----------------------------------------------------------------------------- the project side


def clip_lines(tree: T.Tree, node: T.Node) -> list[dict]:
    """The spoken lines of a clip, [{"line", "speaker": character or None}]: the quoted lines of its prompt, who says
    each as tree.dialogue reads it (nobody mentioned: the lead)."""
    found, _unknown = T.mentions(tree.options, node.prompt)
    return [{"line": d["line"], "speaker": d["speaker"]}
            for d in T.dialogue(tree.options, node.prompt, default=None if found else T.lead_of(tree.options))]


def needs_transcript(tree: T.Tree, node: T.Node) -> bool:
    """A local speaker's voice sample has not been transcribed yet (Whisper is needed once)."""
    return any(d["speaker"] is not None and (d["speaker"].get("voice") or {}).get("file")
               and not d["speaker"]["voice"].get("text") for d in clip_lines(tree, node))


def line_problems(tree: T.Tree, node: T.Node, provider: str) -> list[str]:
    """Why the clip's lines cannot be spoken (empty: they can)."""
    lines = clip_lines(tree, node)
    if not lines:
        return ["its prompt has no spoken line (write it in quotes after @Name: @Anna says \"Hello.\")"]
    out = []
    for d in lines:
        c = d["speaker"]
        short = d["line"][:30] + ("…" if len(d["line"]) > 30 else "")
        if c is None:
            out.append(f"who says “{short}”? (write @Name in its sentence)")
        elif provider == "elevenlabs" and not (c.get("tts") or {}).get("eleven"):
            out.append(f"@{c['name']} has no ElevenLabs voice yet (Cast → Voice with TTS…)")
        elif provider != "elevenlabs" and not c.get("voice"):
            out.append(f"@{c['name']} has no voice sample yet (Cast → Voice… or Voice with TTS…)")
    return list(dict.fromkeys(out))


def _set_voice_fields(tree: T.Tree, name: str, **fields) -> None:
    """Fields kept with a character's voice (its transcript) or in its "tts" entry (its ElevenLabs voice)."""
    cast = []
    for c in T.cast_of(tree):
        if c["name"] == name:
            c = dict(c)
            if "text" in fields and c.get("voice"):
                c["voice"] = {**c["voice"], "text": fields["text"]}
            if "eleven" in fields:
                tts = dict(c.get("tts") or {})
                if fields["eleven"]:
                    tts["eleven"] = fields["eleven"]
                else:
                    tts.pop("eleven", None)
                c["tts"] = tts
        cast.append(c)
    T._set_cast(tree, cast)


def _wav_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (48000, 2, 2):
            raise TtsError(f"{path.name} is not 48 kHz stereo")
        return w.readframes(w.getnframes())


def _silence(secs: float) -> bytes:
    return b"\0\0\0\0" * round(secs * 48000)


def join_lines(parts: list[Path], out: Path, min_secs: float = T.AUDIO_REF_MIN) -> float:
    """The spoken lines one after the other (48 kHz stereo WAVs): a short pause first and between them, a short one
    after the last, filled up to min_secs. Returns the length."""
    data = _silence(LEAD_IN)
    for k, p in enumerate(parts):
        data += (_silence(GAP) if k else b"") + _wav_frames(p)
    data += _silence(TAIL)
    short = min_secs - len(data) / (48000 * 4)
    if short > 0:
        data += _silence(short)
    tmp = Path(out).with_name(Path(out).stem + ".part.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(data)
    os.replace(tmp, out)
    return len(data) / (48000 * 4)


def frames_for(secs: float, have: int) -> int | None:
    """The clip length (H3's 17n+5 frames, up to 362 = 15 s) that holds secs of sound; None: the clip's own `have`
    frames hold it already. TtsError: longer than a clip can be."""
    need = int(secs * chain.FPS + 0.999)
    if need <= have:
        return None
    for n in range(4, 22):
        if 17 * n + 5 >= need:
            return 17 * n + 5
    raise TtsError(f"the lines take {secs:.1f} s; a clip is at most 15 s: split them over two clips")


def render_lines(project: Path, tree: T.Tree, nid: str, eng, log=lambda _m: None) -> dict:
    """Speak the clip's lines, each in its speaker's voice, into one new WAV in refs/ (see join_lines). Reads the
    tree only (call it with a copy from a worker thread; apply_lines puts the result in). Returns {"file", "secs",
    "lines": [{"name", "line"}], "texts": {name: what its voice sample says, transcribed now}, "provider",
    "language", "prompt": the prompt the lines are from}."""
    project = Path(project)
    node = tree.nodes[nid]
    if node.is_video:
        raise TtsError("a real video keeps its own sound")
    bad = line_problems(tree, node, eng.name)
    if bad:
        raise TtsError("; ".join(bad))
    lang = language(tree)
    opts = options(tree)
    lines = clip_lines(tree, node)
    refs = project / T.REFS_DIR
    refs.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    voice_files = {d["speaker"]["name"]: d["speaker"]["voice"]["file"] for d in lines
                   if eng.name != "elevenlabs" and d["speaker"].get("voice")}
    with tempfile.TemporaryDirectory(prefix="video-chain-tts-") as tmp:
        parts = []
        for k, d in enumerate(lines):
            c = d["speaker"]
            raw = Path(tmp) / f"line{k}_raw.wav"
            seed = (node.seed + k) % 2_000_000_000
            log(f"speaking line {k + 1} of {len(lines)} (@{c['name']}): {d['line'][:60]}")
            if eng.name == "elevenlabs":
                eng.speak(d["line"], c["tts"]["eleven"]["id"], raw, opts["eleven_model"], lang, seed)
            else:
                text = c["voice"].get("text") or texts.get(c["name"])
                if not text:
                    log(f"transcribing the voice sample of @{c['name']} (once)")
                    text = texts[c["name"]] = eng.transcribe(project / c["voice"]["file"], LANGUAGES.get(lang))
                eng.clone(d["line"], lang, project / c["voice"]["file"], text, raw, seed)
            part = Path(tmp) / f"line{k}.wav"
            chain.to_wav(raw, part)
            parts.append(part)
        rel = f"{T.REFS_DIR}/tts_{uuid.uuid4().hex[:8]}.wav"
        secs = join_lines(parts, project / rel)
    return {"file": rel, "secs": round(secs, 3), "lines": [{"name": d["speaker"]["name"], "line": d["line"]}
                                                           for d in lines],
            "texts": texts, "voice_files": voice_files, "provider": eng.name, "language": lang, "prompt": node.prompt}


def apply_lines(project: Path, tree: T.Tree, nid: str, r: dict) -> dict:
    """Make rendered lines (render_lines) the clip's sound: a new audio track it plays, the picture made to it (lips
    in sync); the clip is made longer when the lines need it; its earlier spoken-lines track goes when no other clip
    plays it; the voice transcripts are kept. Returns {"track", "secs", "frames": new length or None, "lines": n}."""
    node = tree.nodes[nid]
    ov = dict(node.overrides or {})
    ov.pop("sound_over", None)                      # the picture is made to it (lips in sync)
    # (before anything is changed: lines longer than a clip can be refuse the whole thing)
    frames = frames_for(r["secs"], int(ov.get("frames", tree.options.get("frames", 73))))
    for name, text in (r.get("texts") or {}).items():
        c = T.character(tree, name)
        files = r.get("voice_files") or {}
        if c is not None and c.get("voice") and (name not in files or files[name] == c["voice"]["file"]):
            _set_voice_fields(tree, name, text=text)   # (not for a voice replaced while the lines were made)
    old = T.track_id(node)
    old_tr = T.track_of(tree, old) if old else None
    have = {t["id"] for t in T.tracks(tree)}
    tid = "t" + uuid.uuid4().hex[:6]
    while tid in have:
        tid = "t" + uuid.uuid4().hex[:6]
    entry = {"id": tid, "name": f"Lines of clip #{nid[1:]}", "file": r["file"], "secs": r["secs"],
             "source": str(Path(project) / r["file"]), "start": 0.0, "end": r["secs"],
             "tts": {"clip": nid, "provider": r["provider"], "language": r["language"], "lines": r["lines"]}}
    tree.options["tracks"] = T.tracks(tree) + [entry]
    ov["source_sound"] = T.TRACK_PREFIX + tid
    if frames:
        ov["frames"] = frames
    node.overrides = ov
    if old_tr and (old_tr.get("tts") or {}).get("clip") == nid and not [u for u in T.track_users(tree, old)
                                                                         if u != nid]:
        T.remove_track(project, tree, old)
    return {"track": tid, "secs": r["secs"], "frames": frames, "lines": len(r["lines"])}


def speak_lines(project: Path, tree: T.Tree, nid: str, eng, log=lambda _m: None) -> dict:
    """render_lines + apply_lines in one go."""
    return apply_lines(project, tree, nid, render_lines(project, tree, nid, eng, log))


def make_sample(eng, text: str, lang: str, out: Path, *, description: str | None = None,
                voice_id: str | None = None, model: str | None = None, seed: int | None = None) -> float:
    """A voice sample to listen to (48 kHz WAV at out): a sentence in an ElevenLabs voice (voice_id), or in a new
    voice made from a description (local, VoiceDesign). Returns its length."""
    if eng.name == "elevenlabs":
        return eng.speak(text, voice_id, out, model or DEFAULT_OPTIONS["eleven_model"], lang, seed)
    raw = Path(out).with_name(Path(out).stem + "_raw.wav")
    eng.design(text, lang, description or "", raw, seed)
    chain.to_wav(raw, Path(out))
    raw.unlink(missing_ok=True)
    return chain.audio_seconds(Path(out)) or 0.0


def use_sample(project: Path, tree: T.Tree, name: str, wav: Path, label: str, text: str,
               eleven: dict | None = None) -> dict:
    """A sample (make_sample) becomes the character's voice: the clip model hears it, local TTS clones it (with
    what it says, unless the sample had to be cut), and with eleven ({"id", "name"}) its lines are spoken in that
    ElevenLabs voice (without: no ElevenLabs voice any more). Returns the voice entry (tree.set_voice)."""
    entry = T.set_voice(project, tree, name, Path(wav), label=label)
    if not entry.get("cut"):
        _set_voice_fields(tree, name, text=text)
    _set_voice_fields(tree, name, eleven=eleven)
    return entry


def clone_sample(project: Path, tree: T.Tree, name: str, eng) -> dict:
    """An instant ElevenLabs voice clone of a character's voice sample, made in the user's account (eng: ElevenLabs).
    Reads the tree only (call it from a worker thread; set_eleven puts the result in). Returns {"id", "name",
    "file": the sample it was made from}."""
    c = T.character(tree, name)
    if c is None or not c.get("voice"):
        raise TtsError(f"@{name} has no voice sample to clone (Cast → Voice…)")
    label = f"{name} (video-chain)"
    vid = eng.clone(label, Path(project) / c["voice"]["file"],
                    f"made by video-chain from the voice sample of {name}")
    return {"id": vid, "name": label, "file": c["voice"]["file"]}


def set_eleven(tree: T.Tree, name: str, voice: dict) -> bool:
    """A cloned voice (clone_sample) becomes the character's ElevenLabs voice. False (nothing changed): the
    character is gone or its voice sample was replaced while the clone was made."""
    c = T.character(tree, name)
    if c is None or (c.get("voice") or {}).get("file") != voice.get("file"):
        return False
    _set_voice_fields(tree, name, eleven={"id": voice["id"], "name": voice["name"]})
    return True
