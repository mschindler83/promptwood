"""A llama.cpp server (llama-server) as a second prompt refiner next to ComfyUI's TextGenerate: for GGUF language
models ComfyUI cannot run (e.g. Huihui Qwen3-VL-30B-A3B abliterated). The app starts the server itself when a
refine needs it, with the mixture-of-experts weights in RAM (--cpu-moe: a few GB of VRAM, the GPU stays free for
ComfyUI), and stops it before clips are made and when the app closes (it holds the model in RAM).

Requests go to its OpenAI-compatible /v1/chat/completions, with the same instructions as ComfyUI's refiner."""
from __future__ import annotations

import atexit
import base64
import functools
import http.client
import io
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import weakref
from pathlib import Path
from typing import Sequence

from . import config, net

logger = logging.getLogger(__name__)

from .comfy import clean_refined, refine_messages

PREFIX = "llama:"                                # model ids of this backend in the refiner's list
DEFAULT_PORT = 8091
DEFAULT_SERVER = config.llama_server()
START_TIMEOUT = 600.0                            # a 25 GB model from a cold disk takes minutes
CONTEXT = 16384                                  # a conversation with pictures (one ~1 MP picture: ~1-2k tokens)
MAX_TOKENS = 1024                                # a prompt is a few hundred tokens; more means it rambles
LONG_TOKENS = {"video_h3": 1536}                 # the official H3 format: up to ~500 words + the sound lines
IMAGE_SIDE = 1024                                # pictures are sent at most this large (fewer tokens, same gist)


class LlmServerError(RuntimeError):
    pass


def is_uncensored(path: str) -> bool:
    """The model's file name says it is abliterated / uncensored (the folder's name does not count)."""
    name = Path(path).name.lower()
    return "abliterated" in name or "uncensored" in name


def model_label(path: str) -> str:
    name = Path(path).name
    stem = name[:-5] if name.endswith(".gguf") else name
    free = is_uncensored(path)
    return f"{stem}{' (uncensored)' if free else ''}  ·  llama.cpp"


def find_mmproj(model: str) -> str | None:
    """The vision projector next to a GGUF model (mmproj-*.gguf; F16 / BF16 before Q8 / F32), or None."""
    files = sorted(Path(model).parent.glob("*mmproj*.gguf"))
    rank = {"f16": 0, "bf16": 1, "q8_0": 2, "f32": 3}
    quant = re.compile(r"(?<![a-z0-9])(bf16|f16|f32|q8_0)(?![a-z0-9])")
    own = set(re.findall(r"[a-z0-9]+", Path(model).stem.lower()))

    def key(f: Path):
        tokens = [t for t in re.findall(r"[a-z0-9]+", f.stem.lower()) if t != "mmproj" and not quant.fullmatch(t)]
        fits = 0 if tokens and set(tokens) <= own else (1 if not tokens else 2)    # its own / any / another model's
        q = quant.search(f.name.lower())
        return fits, rank.get(q.group(1), 9) if q else 9, f.name
    files.sort(key=key)
    return str(files[0]) if files else None


def image_part(path: str | Path) -> dict:
    """A picture for the chat (OpenAI format, a PNG data URL), at most IMAGE_SIDE large (made once per file state:
    every turn of a conversation sends its pictures again)."""
    try:
        st = os.stat(path)
        url = _image_url(str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        url = _image_url(str(path), 0, 0)               # (the error of opening it comes from there)
    return {"type": "image_url", "image_url": {"url": url}}


@functools.lru_cache(maxsize=24)
def _image_url(path: str, _mtime_ns: int, _size: int) -> str:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):                # transparent parts on grey, not black
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
            bg.alpha_composite(im)
            im = bg
        im = im.convert("RGB")
        im.thumbnail((IMAGE_SIDE, IMAGE_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CHAT_RULES = ("This is a conversation about one prompt. The user's first message is their prompt; each later "
              "message asks for a change to the current version. Answer every time with the complete new prompt only "
              "(no explanations, no quotes around it), changing only what is asked and keeping everything else. "
              "If pictures are attached, they show what the prompt is about (e.g. the frame a clip starts from); "
              "use what you see when the user asks for it.")


def _lora_part(loras: str) -> str:
    """What is known of the LoRAs (lora_notes.guidance), after the rules: the long material comes last."""
    return "\n\n" + loras.strip() if loras.strip() else ""


def chat_messages(*, prompt: str, kind: str, turns: list[tuple[str, str]], instruction: str,
                  pictures: Sequence[tuple[str, str]] = (), cast: list[dict] | None = None,
                  seconds: float | None = None, start_image: str | None = None, loras: str = "") -> list[dict]:
    """The conversation for the model. turns: the earlier [(instruction, version)]; the last version is the current
    prompt (also when the user edited it). The first user message holds the user's prompt, the pictures (labelled)
    and the first instruction; then the turns alternate; the new instruction comes last.
    seconds, start_image ("video_h3"): the clip's length and first frame; the model is told it sees that frame
    when it is among the pictures (else not to invent its look). loras: lora_notes.guidance."""
    start = None
    if kind == "video_h3" and start_image:
        start = "seen" if str(start_image) in [str(p) for _t, p in pictures] else "unseen"
    system, first = refine_messages(prompt=prompt, kind=kind, cast=cast,
                                    seconds=seconds if kind == "video_h3" else None, start=start)
    system += " " + CHAT_RULES + _lora_part(loras)
    asks = [a for a, _v in turns] + [instruction]

    def ask_text(a: str) -> str:
        return a.strip() or "Improve this prompt."
    opening = f"My prompt:\n{first}\n\nWhat I want: {ask_text(asks[0])}"
    content: list[dict] = [{"type": "text", "text": opening}]
    for title, path in pictures:
        content += [{"type": "text", "text": f"\n[{title}]"}, image_part(path)]
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": content if pictures else opening}]
    for i, (_ask, version) in enumerate(turns):
        msgs.append({"role": "assistant", "content": version})
        msgs.append({"role": "user", "content": ask_text(asks[i + 1])})
    return msgs


IDEA_RULES = ("Now the user gives bullet points instead of a prompt. Make {n} creative, interesting and clearly "
              "different image ideas from them. Every idea uses every bullet point (each subject, attribute, place and "
              "action stays as given; @Names stay exactly as written) and must not add other people. Make the ideas "
              "differ in: camera and framing, light and mood, time of day or weather, composition, style or medium, "
              "and one surprising but fitting detail; never give two ideas the same look. Answer with JSON only, no "
              "markdown: a list of {n} objects, each {{\"title\": a 2-5 word name, \"angle\": a few words on what makes "
              "it different (e.g. \"low angle, dusk, film noir\"), \"prompt\": the complete prompt as one paragraph}}.")


def idea_messages(*, bullets: str, n: int, wish: str = "", pictures: Sequence[tuple[str, str]] = (),
                  cast: list[dict] | None = None, loras: str = "") -> list[dict]:
    """The request for n image ideas from bullet points (kind 'image': the refiner's image rules apply)."""
    system, _first = refine_messages(prompt=bullets, kind="image", cast=cast)
    text = (f"My bullet points:\n{bullets.strip()}" + (f"\n\nAlso: {wish.strip()}" if wish.strip() else ""))
    content: list[dict] = [{"type": "text", "text": text}]
    for title, path in pictures:
        content += [{"type": "text", "text": f"\n[{title}]"}, image_part(path)]
    system += " " + IDEA_RULES.format(n=int(n)) + _lora_part(loras)
    return [{"role": "system", "content": system},
            {"role": "user", "content": content if pictures else text}]


def parse_ideas(text: str) -> list[dict]:
    """The ideas in a model's answer: [{"title", "angle", "prompt"}] (JSON, also inside code fences or with
    text around it; a missing title / angle is filled in)."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.S | re.I)
    data, err = None, None
    dec = json.JSONDecoder(strict=False)                # (a line break inside a prompt is no error)
    for m in re.finditer(r"\[", text):                 # the first JSON list of ideas (text around it, "[Note]" …)
        try:
            cand, _end = dec.raw_decode(text, m.start())
        except ValueError as e:
            err = err or e
            continue
        if isinstance(cand, list) and cand and all(isinstance(x, (dict, str)) for x in cand):
            data = cand
            break
    if data is None:                                   # objects without the list around them: {…}, {…}
        objs, pos = [], 0
        while (i := text.find("{", pos)) >= 0:
            try:
                cand, end = dec.raw_decode(text, i)
            except ValueError as e:
                err = err or e
                pos = i + 1
                continue
            if isinstance(cand, dict) and cand.get("prompt"):
                objs.append(cand)
            pos = end
        data = objs or None
    if data is None:
        raise LlmServerError("the model gave no list of ideas" + (f" ({err})" if err else ""))
    out = []
    for k, d in enumerate(data if isinstance(data, list) else []):
        if isinstance(d, str):
            d = {"prompt": d}
        if not isinstance(d, dict) or not str(d.get("prompt") or "").strip():
            continue
        out.append({"title": " ".join(str(d.get("title") or f"Idea {k + 1}").split()),
                    "angle": " ".join(str(d.get("angle") or "").split()),
                    "prompt": " ".join(str(d["prompt"]).split())})
    if not out:
        raise LlmServerError("the model gave no usable ideas")
    return out


def missing_bullets(bullets: str, prompt: str) -> list[str]:
    """The bullet points a prompt seems to leave out: fewer than half of a bullet's words (3+ letters, compared by
    their first 4 letters, so "rain" finds "raining") appear in it, or an @Name is not there as a whole word.
    A bullet without such words is not judged."""
    low = prompt.lower()
    stems = {w[:4] for w in re.findall(r"[^\W\d_]{3,}", low)}
    out = []
    for line in bullets.splitlines():
        b = re.sub(r"^\s*([-*•·]|\d+[.)])\s*", "", line).strip()
        if not b:
            continue
        names = re.findall(r"@[\w-]+", b)
        words = [w.lower()[:4] for w in re.findall(r"[^\W\d_]{3,}", re.sub(r"@[\w-]+", "", b))]
        lost_name = any(not re.search(re.escape(n.lower()) + r"(?![\w-])", low) for n in names)
        if lost_name or (words and sum(w in stems for w in words) * 2 < len(words)):
            out.append(b)
    return out


def server_command(server: str, model: str, port: int, cpu_moe: bool = True, mmproj: str | None = None) -> list[str]:
    """llama-server's command: all layers on the GPU except (cpu_moe) the experts, which stay in RAM."""
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(int(port)), "-c", str(CONTEXT),
           "-ngl", "999", "-np", "1", "--jinja", "--no-webui"]       # (-np 1: one slot gets the whole context)
    if cpu_moe:
        cmd.append("--cpu-moe")
    if mmproj:
        cmd += ["--mmproj", mmproj]
    return cmd


def vram_needed_mb(model: str, cpu_moe: bool = True, mmproj: str | None = None) -> int:
    """About how much VRAM the server takes: with the experts in RAM (cpu_moe) the attention layers, KV cache and
    buffers (~2.5-4 GB measured on a 30B-A3B), else the whole model file; plus the vision part."""
    def mb(path):
        try:
            return os.path.getsize(path) // (1024 * 1024)
        except OSError:
            return 0
    return (4000 if cpu_moe else mb(model) + 1500) + (mb(mmproj) if mmproj else 0)


def free_vram_mb() -> int | None:
    """Free VRAM of the first GPU (nvidia-smi), None when it cannot be read."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


_LOOPBACK = net._DIRECT     # (a proxy setting must not catch 127.0.0.1)
LOG_LIMIT = 5 * 1024 * 1024


def rotate_log(path: Path, limit: int = LOG_LIMIT) -> None:
    """A log over `limit` bytes becomes <name>.1 (the older one is dropped), so logs do not grow for ever."""
    try:
        if path.stat().st_size > limit:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError as e:
        logger.debug("log not rotated: %s", e)


def _proc_stat(pid: int) -> tuple[str, str] | None:
    """(state, start time) of a process from /proc, None when there is none."""
    try:
        rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return rest[0], rest[19]
    except (OSError, IndexError):
        return None


class _Adopted:
    """A server of an earlier run of the app (found by its pid file) in the place of a Popen: only what stopping
    it needs. It is no child of this process, so its end is seen in /proc."""

    def __init__(self, pid: int, started: str):
        self.pid, self.started, self.returncode = pid, started, None

    def poll(self):
        st = _proc_stat(self.pid)
        if st is None or st[0] == "Z" or st[1] != self.started:
            self.returncode = 0
            return 0
        return None

    def wait(self, timeout: float | None = None):
        end = None if timeout is None else time.time() + timeout
        while self.poll() is None:
            if end is not None and time.time() > end:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.1)
        return self.returncode


_LIVE: "weakref.WeakSet[LlamaServer]" = weakref.WeakSet()


def _die_with_parent() -> None:
    """In the child, before exec: the kernel sends SIGTERM when the parent thread ends (Linux prctl)."""
    import ctypes
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)       # 1 = PR_SET_PDEATHSIG


class _Spawner:
    """Starts child processes from ONE long-lived daemon thread. PR_SET_PDEATHSIG fires when the *thread* that forked
    ends, not the process: a worker thread that asked for the server and then finished (a refine) must not take it
    down. This thread ends only with the app (also a crash or SIGKILL of it), so llama-server cannot outlive it."""

    def __init__(self):
        import queue
        self._q: "queue.Queue" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _loop(self) -> None:
        while True:
            argv, kw, box, done = self._q.get()
            try:
                box.append(subprocess.Popen(argv, preexec_fn=_die_with_parent, **kw))    # noqa: PLW1509
            except BaseException as e:  # noqa: BLE001  (handed to the caller)
                box.append(e)
            done.set()

    def spawn(self, argv, **kw) -> subprocess.Popen:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="spawner", daemon=True)
                self._thread.start()
        box: list = []
        done = threading.Event()
        self._q.put((argv, kw, box, done))
        done.wait()
        if isinstance(box[0], BaseException):
            raise box[0]
        return box[0]


_SPAWNER = _Spawner()


def spawn(argv, **kw) -> subprocess.Popen:
    """subprocess.Popen that dies with the app where Linux can do that (PR_SET_PDEATHSIG; elsewhere a plain Popen)."""
    if sys.platform.startswith("linux"):
        return _SPAWNER.spawn(argv, **kw)
    return subprocess.Popen(argv, **kw)


_DYING: list = []          # servers sent SIGTERM without waiting for them


def _reap(p, grace: float) -> None:
    """Wait for a server that was sent SIGTERM; SIGKILL for its process group after `grace` s."""
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)                         # (the port is free before a restart)
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning("SIGKILL of the llama-server failed: %s", e)


@atexit.register
def _stop_all() -> None:
    """The app ends: no server it started is left holding the RAM (a short grace, then SIGKILL). A server stopped
    with wait=False is waited for here: the background thread would die with the interpreter before its SIGKILL."""
    for srv in list(_LIVE):
        try:
            srv._kill(srv._take(), grace=5)
        except Exception as e:  # noqa: BLE001  (exiting: nothing more to do about it)
            logger.warning("stopping a llama-server at exit failed: %s", e)
    for p in list(_DYING):
        try:
            if p.poll() is None:
                _reap(p, 5)
        except Exception as e:  # noqa: BLE001
            logger.warning("reaping a llama-server at exit failed: %s", e)


class LlamaServer:
    """Starts / stops one llama-server for one model file and asks it to refine prompts. before_launch (optional,
    called in start()'s thread just before a new server process starts) can make room for it, or raise
    LlmServerError to refuse."""

    def __init__(self, server: str, model: str, port: int = DEFAULT_PORT, cpu_moe: bool = True,
                 mmproj: str | None = None, log_dir: Path | None = None):
        self.server, self.model, self.port, self.cpu_moe, self.mmproj = server, model, int(port), cpu_moe, mmproj
        self.log_dir = log_dir
        self.proc: subprocess.Popen | _Adopted | None = None
        self._lock = threading.Lock()                # around self.proc
        self._start_lock = threading.Lock()          # one start() at a time
        self._gen = 0                                # stop() / interrupt() count it up: a running start() ends
        self.before_launch = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def models(self) -> list[dict]:
        return [{"file": PREFIX + self.model, "type": "llama", "label": model_label(self.model),
                 "uncensored": is_uncensored(self.model), "chat": True, "vision": bool(self.mmproj)}]

    def problems(self) -> list[str]:
        out = []
        if not (Path(self.server).is_file() and os.access(self.server, os.X_OK)) and not shutil.which(self.server):
            out.append(f"the llama.cpp server program is missing ({self.server})")
        if not Path(self.model).is_file():
            out.append(f"the model file is missing ({self.model})")
        return out

    def _get(self, path: str, timeout: float = 5.0) -> dict:
        with _LOOPBACK.open(self.url + path, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace") or "{}")
        return data if isinstance(data, dict) else {}          # (a foreign service may answer a list)

    def ready(self) -> bool:
        try:
            return self._get("/health", 3.0).get("status") == "ok"
        except (OSError, ValueError, http.client.HTTPException):
            return False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def mismatch(self) -> str:
        """Why the server answering on the port is not the one wanted ("" when it is: this model, and the vision
        part when there is one)."""
        try:
            props = self._get("/props", 3.0)
        except (OSError, ValueError, http.client.HTTPException):
            return ""                                  # (an older server without /props: trusted)
        path = props.get("model_path") or ""
        if path and os.path.realpath(path) != os.path.realpath(self.model):
            return f"it runs {Path(path).name}"
        if self.mmproj and (props.get("modalities") or {}).get("vision") is False:
            return "it runs this model without its vision part"
        return ""

    def _command(self) -> list[str]:
        return server_command(self.server, self.model, self.port, self.cpu_moe, self.mmproj)

    def _pidfile(self) -> Path | None:
        return self.log_dir / "llama-server.pid" if self.log_dir is not None else None

    def _remember(self, proc) -> None:
        """Write the pid file: what a later run needs to find this server again (and to know it is ours)."""
        path = self._pidfile()
        st = _proc_stat(proc.pid)
        if path is None or st is None:
            return
        try:
            path.write_text(json.dumps({"pid": proc.pid, "started": st[1], "cmd": self._command(), "port": self.port}))
        except OSError as e:
            logger.warning("could not write the server record: %s", e)

    def _forget(self) -> None:
        path = self._pidfile()
        if path is not None:
            path.unlink(missing_ok=True)

    def _leftover(self) -> tuple[_Adopted, list[str]] | None:
        """The server of an earlier run (pid file), when it still runs and really is that server: the same
        process (start time), its command line is the recorded one, and it leads its own process group. Else the
        stale file is dropped and nothing is touched."""
        path = self._pidfile()
        if path is None or not path.is_file():
            return None
        try:
            d = json.loads(path.read_text())
            pid, started, cmd = int(d["pid"]), str(d["started"]), [str(x) for x in d["cmd"]]
            st = _proc_stat(pid)
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").split("\0")[:-1]
            if st is not None and st[0] != "Z" and st[1] == started and cmd and argv[-len(cmd):] == cmd \
                    and os.getpgid(pid) == pid:
                return _Adopted(pid, started), cmd
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.debug("no adoptable server: %s", e)
        self._forget()
        return None

    def _take_over_leftover(self, keep: bool, wait: bool = True) -> bool:
        """A server of an earlier run (the app crashed or was killed): taken over when it is the one wanted (keep),
        else ended (True: one was). Never while this object has a server of its own."""
        with self._lock:
            if self.proc is not None:
                return False
            found = self._leftover()
            if found is None:
                return False
            old, cmd = found
            if keep and cmd == self._command():
                self.proc = old
                return False
        ended = self._kill(old, wait)
        self._forget()
        return ended

    def start(self, stop=lambda: False, timeout: float = START_TIMEOUT) -> None:
        """Start the server (if it is not answering yet) and wait until the model is loaded. A server already on
        the port must run this model (with its vision part when there is one): one this object started is
        restarted otherwise; another one is an error (its port is taken). One start at a time; stop() and
        interrupt() never wait for a load: they cancel it (a generation counter, so a cancel just before the
        start is not lost), and a cancelled start ends the server it started."""
        gen = self._gen
        while not self._start_lock.acquire(timeout=0.5):        # (a cancel ends the wait behind another start)
            if stop() or gen != self._gen:
                raise LlmServerError("cancelled")
        try:
            self._start_locked(gen, stop, timeout)
        finally:
            self._start_lock.release()

    def _start_locked(self, gen: int, stop, timeout: float) -> None:
        if gen != self._gen:
            raise LlmServerError("cancelled")
        self._take_over_leftover(keep=True)
        if self.ready():
            why = self.mismatch()
            if not why:
                return
            if not self.running():
                raise LlmServerError(f"port {self.port} is used by another llama.cpp server ({why}). Close "
                                     "the program that started it, or choose another port in the settings")
            self._kill(self._take())
        bad = self.problems()
        if bad:
            raise LlmServerError("; ".join(bad))
        if not self.running() and self._port_taken():
            raise LlmServerError(f"port {self.port} is used by another program. Close it, or choose another "
                                 "port in the settings")
        if self.before_launch is not None and not self.running():
            self.before_launch()
        with self._lock:
            if gen != self._gen:
                raise LlmServerError("cancelled")
            if not self.running():
                log = subprocess.DEVNULL
                if self.log_dir is not None:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    rotate_log(self.log_dir / "llama-server.log")
                    log = open(self.log_dir / "llama-server.log", "ab")    # noqa: SIM115
                try:
                    self.proc = spawn(self._command(), stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
                except OSError as e:
                    raise LlmServerError(f"the llama.cpp server could not be started: {e}") from e
                finally:
                    if log is not subprocess.DEVNULL:
                        log.close()                    # (the process has its own copy)
                _LIVE.add(self)
                self._remember(self.proc)
            proc = self.proc
        log_hint = f"; see {self.log_dir / 'llama-server.log'}" if self.log_dir else ""
        end = time.time() + timeout
        while time.time() < end:
            if stop() or gen != self._gen:
                if self.proc is proc:              # (cancelled: not left loading in the background)
                    self._kill(self._take())
                raise LlmServerError("cancelled")
            if proc is not self.proc:
                raise LlmServerError("the llama.cpp server was stopped while loading")
            if proc.poll() is not None:
                raise LlmServerError(f"the llama.cpp server stopped while loading (exit {proc.returncode}, "
                                     f"port {self.port}{log_hint})")
            if self.ready():
                return
            time.sleep(0.5)
        self._kill(self._take())                   # (not left holding the RAM)
        raise LlmServerError(f"the llama.cpp server did not load the model within {int(timeout)} s{log_hint}")

    def _port_taken(self) -> bool:
        try:
            socket.create_connection(("127.0.0.1", self.port), 0.3).close()
            return True
        except OSError:
            return False

    def _take(self):
        with self._lock:
            p, self.proc = self.proc, None
        if p is not None:
            self._forget()
        return p

    @staticmethod
    def _kill(p, wait: bool = True, grace: float = 20) -> bool:
        """SIGTERM to the server's process group; wait for it (SIGKILL after `grace` s), in the background when
        wait is False (the GUI does not wait)."""
        if p is None or p.poll() is not None:
            return False
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            return False

        if wait:
            _reap(p, grace)
        else:
            _DYING[:] = [q for q in _DYING if q.poll() is None]
            _DYING.append(p)                           # (at exit: _stop_all sees it through, see LLM-3)
            threading.Thread(target=_reap, args=(p, grace), daemon=True).start()
        return True

    def stop(self, wait: bool = True, grace: float = 20) -> bool:
        """Stop the server this object started (frees its RAM and VRAM); a load in progress is cancelled. True if
        one was running. wait=False: the signal is sent now, the waiting happens in the background. grace: seconds
        the server gets to end after SIGTERM before SIGKILL (a closing window passes a short one)."""
        self._bump()
        left = self._take_over_leftover(keep=False, wait=wait)    # (a server a crashed run left behind, too)
        return self._kill(self._take(), wait, grace) or left

    def _bump(self) -> None:
        with self._lock:
            self._gen += 1

    def refine_prompt(self, prompt: str, kind: str, model_file: str | None = None, seed: int = 0,
                      hint: str = "", cast: list[dict] | None = None, temperature: float = 0.5,
                      timeout: float = 300.0, seconds: float | None = None,
                      start_image: str | None = None, loras: str = "") -> str:
        """One refine (text only; the refine window holds a conversation with llama.cpp models, see chat, where
        pictures are shown). start_image ("video_h3"): the model is told the clip starts from a frame it cannot see."""
        system, user = refine_messages(prompt=prompt, kind=kind, hint=hint, cast=cast,
                                       seconds=seconds if kind == "video_h3" else None,
                                       start="unseen" if kind == "video_h3" and start_image else None,
                                       loras=loras)
        return self._complete([{"role": "system", "content": system}, {"role": "user", "content": user}], seed,
                              temperature, timeout, max_tokens=LONG_TOKENS.get(kind, MAX_TOKENS),
                              lines=kind == "video_h3")

    def chat(self, prompt: str, kind: str, turns: list[tuple[str, str]], instruction: str,
             pictures: Sequence[tuple[str, str]] = (), cast: list[dict] | None = None, seed: int = 0,
             temperature: float = 0.5, timeout: float = 300.0, seconds: float | None = None,
             start_image: str | None = None, loras: str = "") -> str:
        """The next version in a refine conversation (see chat_messages); pictures need the vision projector."""
        if pictures and not self.mmproj:
            raise LlmServerError("this model has no vision part (mmproj file) next to it: it cannot see pictures")
        return self._complete(chat_messages(prompt=prompt, kind=kind, turns=turns, instruction=instruction,
                                            pictures=pictures, cast=cast, seconds=seconds, start_image=start_image,
                                            loras=loras),
                              seed, temperature, timeout, max_tokens=LONG_TOKENS.get(kind, MAX_TOKENS),
                              lines=kind == "video_h3")

    def ideas(self, bullets: str, n: int = 5, wish: str = "", pictures: Sequence[tuple[str, str]] = (),
              cast: list[dict] | None = None, seed: int = 0, temperature: float = 0.9,
              timeout: float = 600.0, loras: str = "") -> list[dict]:
        """n distinct image ideas from bullet points (see IDEA_RULES): [{"title", "angle", "prompt"}]."""
        if pictures and not self.mmproj:
            raise LlmServerError("this model has no vision part (mmproj file) next to it: it cannot see pictures")
        item = {"type": "object", "properties": {k: {"type": "string"} for k in ("title", "angle", "prompt")},
                "required": ["title", "angle", "prompt"]}
        shape = {"type": "array", "items": item, "minItems": int(n), "maxItems": int(n)}   # the answer's form
        raw = self._complete(idea_messages(bullets=bullets, n=n, wish=wish, pictures=pictures, cast=cast,
                                           loras=loras), seed,
                             temperature, timeout, max_tokens=300 * int(n) + 300, clean=False, schema=shape)
        return parse_ideas(raw)

    def complete(self, system: str, user: str, max_tokens: int = 512, model_file: str | None = None,
                 seed: int = 0, temperature: float = 0.2, timeout: float = 300.0) -> str:
        """The model's raw answer to one question (text only; e.g. a LoRA's note, lora_notes)."""
        return self._complete([{"role": "system", "content": system}, {"role": "user", "content": user}], seed,
                              temperature, timeout, max_tokens=max_tokens, clean=False)

    def interrupt(self):
        """A load in progress is cancelled (the request of a closed window is not awaited)."""
        self._bump()

    def _complete(self, messages: list[dict], seed: int, temperature: float, timeout: float,
                  max_tokens: int = MAX_TOKENS, clean: bool = True, schema: dict | None = None,
                  lines: bool = False) -> str:
        self.start()
        body = {"messages": messages,
                "max_tokens": int(max_tokens), "temperature": float(temperature), "top_k": 64, "top_p": 0.95,
                "min_p": 0.05, "repeat_penalty": 1.05, "seed": int(seed), "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}     # (a thinking model: the prompt, no essay)
        if schema is not None:                     # llama.cpp constrains the answer to this JSON shape
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with _LOOPBACK.open(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            body = e.read()[:2000].decode("utf-8", "replace")
            try:
                body = (json.loads(body).get("error") or {}).get("message") or body
            except (ValueError, AttributeError):
                pass
            raise LlmServerError(f"the llama.cpp server answered {e.code}: {body[:300]}") from e
        except ValueError as e:
            raise LlmServerError(f"unreadable answer from the llama.cpp server: {e}") from e
        except (OSError, http.client.HTTPException) as e:
            raise LlmServerError(f"the llama.cpp server could not be asked: {e}") from e
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LlmServerError(f"unexpected answer from the llama.cpp server: {str(data)[:300]}") from e
        if choice.get("finish_reason") == "length":
            raise LlmServerError(f"the answer was cut off after {max_tokens} tokens (the model went on too long); "
                                 "try again, or ask for a shorter prompt or fewer ideas")
        if not clean:
            return text
        text = clean_refined(text, lines=lines)
        if not text:
            raise LlmServerError("the language model gave no answer")
        return text


class CombinedRefiner:
    """ComfyUI's language models and the llama.cpp server's in one list (for the refine window): each refine goes
    to the backend of its model. Either may be missing; an error only when both are."""

    def __init__(self, comfy=None, llama: LlamaServer | None = None, comfy_error: str = ""):
        self.comfy, self.llama, self.comfy_error = comfy, llama, comfy_error
        self.models = (list(llama.models) if llama is not None else []) + (list(comfy.models) if comfy else [])
        if not self.models:
            raise LlmServerError(comfy_error or "no language model")

    def refine_prompt(self, prompt, kind, model_file=None, seed=0, hint="", cast=None, **clip):
        """clip: seconds / start_image of a "video_h3" refine, to either backend."""
        if (model_file or "").startswith(PREFIX) or self.comfy is None:
            if self.llama is None:
                raise LlmServerError("the llama.cpp server is not set up")
            return self.llama.refine_prompt(prompt, kind, model_file, seed, hint, cast, **clip)
        return self.comfy.refine_prompt(prompt, kind, model_file, seed, hint, cast, **clip)

    def complete(self, system, user, max_tokens=512, model_file=None, seed=0, **k):
        """One plain question, to the backend of the model (a LoRA's note is written by the model that refines)."""
        if (model_file or "").startswith(PREFIX) or self.comfy is None:
            if self.llama is None:
                raise LlmServerError("the llama.cpp server is not set up")
            return self.llama.complete(system, user, max_tokens, model_file, seed, **k)
        return self.comfy.complete(system, user, max_tokens, model_file, seed, **k)

    def ideas(self, *a, **k):
        if self.llama is None:
            raise LlmServerError("ideas need a llama.cpp model")
        return self.llama.ideas(*a, **k)

    def chat(self, *a, **k):
        if self.llama is None:
            raise LlmServerError("conversations need a llama.cpp model")
        return self.llama.chat(*a, **k)

    def interrupt(self):
        if self.comfy is not None:
            self.comfy.interrupt()
        if self.llama is not None:                  # (a model load in progress is cancelled too)
            self.llama.interrupt()
