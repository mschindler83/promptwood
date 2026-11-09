"""A llama.cpp server (llama-server) as a second prompt refiner next to ComfyUI's TextGenerate: for GGUF language
models ComfyUI cannot run (e.g. Huihui Qwen3-VL-30B-A3B abliterated). The app starts the server itself when a
refine needs it, with the mixture-of-experts weights in RAM (--cpu-moe: a few GB of VRAM, the GPU stays free for
ComfyUI), and stops it before clips are made and when the app closes (it holds the model in RAM).

Requests go to its OpenAI-compatible /v1/chat/completions, with the same instructions as ComfyUI's refiner."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .comfy import clean_refined, refine_messages

PREFIX = "llama:"                                # model ids of this backend in the refiner's list
DEFAULT_PORT = 8091
DEFAULT_SERVER = "/path/to/ai/llama.cpp/build/bin/llama-server"
START_TIMEOUT = 600.0                            # a 25 GB model from a cold disk takes minutes
CONTEXT = 16384                                  # a conversation with pictures (one ~1 MP picture: ~1-2k tokens)
IMAGE_SIDE = 1024                                # pictures are sent at most this large (fewer tokens, same gist)


class LlmServerError(RuntimeError):
    pass


def model_label(path: str) -> str:
    name = Path(path).name
    stem = name[:-5] if name.endswith(".gguf") else name
    free = "abliterated" in stem.lower() or "uncensored" in stem.lower()
    return f"{stem}{' (uncensored)' if free else ''}  ·  llama.cpp"


def find_mmproj(model: str) -> str | None:
    """The vision projector next to a GGUF model (mmproj-*.gguf; F16 / BF16 before Q8 / F32), or None."""
    files = sorted(Path(model).parent.glob("*mmproj*.gguf"))
    rank = {"f16": 0, "bf16": 1, "q8_0": 2, "f32": 3}
    files.sort(key=lambda f: next((r for k, r in rank.items() if k in f.name.lower()), 9))
    return str(files[0]) if files else None


def image_part(path: str | Path) -> dict:
    """A picture for the chat (OpenAI format, a PNG data URL), at most IMAGE_SIDE large."""
    import base64

    from PIL import Image, ImageOps
    import io
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((IMAGE_SIDE, IMAGE_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


CHAT_RULES = ("This is a conversation about one prompt. The user's first message is their prompt; each later "
              "message asks for a change to the current version. Answer every time with the complete new prompt only "
              "(no explanations, no quotes around it), changing only what is asked and keeping everything else. "
              "If pictures are attached, they show what the prompt is about (e.g. the frame a clip starts from); "
              "use what you see when the user asks for it.")


def chat_messages(*, prompt: str, kind: str, turns: list[tuple[str, str]], instruction: str,
                  pictures: list[tuple[str, str]] = (), cast: list[dict] | None = None) -> list[dict]:
    """The conversation for the model. turns: the earlier [(instruction, version)]; the last version is the current
    prompt (also when the user edited it). The first user message holds the user's prompt, the pictures (labelled)
    and the first instruction; then the turns alternate; the new instruction comes last."""
    system, first = refine_messages(prompt=prompt, kind=kind, cast=cast)
    asks = [a for a, _v in turns] + [instruction]

    def ask_text(a: str) -> str:
        return a.strip() or "Improve this prompt."
    opening = f"My prompt:\n{first}\n\nWhat I want: {ask_text(asks[0])}"
    content: list[dict] = [{"type": "text", "text": opening}]
    for title, path in pictures:
        content += [{"type": "text", "text": f"\n[{title}]"}, image_part(path)]
    msgs = [{"role": "system", "content": system + " " + CHAT_RULES},
            {"role": "user", "content": content if pictures else opening}]
    for i, (_ask, version) in enumerate(turns):
        msgs.append({"role": "assistant", "content": version})
        msgs.append({"role": "user", "content": ask_text(asks[i + 1])})
    return msgs


def server_command(server: str, model: str, port: int, cpu_moe: bool = True, mmproj: str | None = None,
                   threads: int | None = None) -> list[str]:
    """llama-server's command: all layers on the GPU except (cpu_moe) the experts, which stay in RAM."""
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(int(port)), "-c", str(CONTEXT),
           "-ngl", "999", "--jinja", "--no-webui"]
    if cpu_moe:
        cmd.append("--cpu-moe")
    if mmproj:
        cmd += ["--mmproj", mmproj]
    if threads:
        cmd += ["-t", str(int(threads))]
    return cmd


class LlamaServer:
    """Starts / stops one llama-server for one model file and asks it to refine prompts."""

    def __init__(self, server: str, model: str, port: int = DEFAULT_PORT, cpu_moe: bool = True,
                 mmproj: str | None = None, log_dir: Path | None = None):
        self.server, self.model, self.port, self.cpu_moe, self.mmproj = server, model, int(port), cpu_moe, mmproj
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def models(self) -> list[dict]:
        return [{"file": PREFIX + self.model, "type": "llama", "label": model_label(self.model),
                 "uncensored": "abliterated" in self.model.lower(), "chat": True, "vision": bool(self.mmproj)}]

    def problems(self) -> list[str]:
        out = []
        if not (Path(self.server).is_file() and os.access(self.server, os.X_OK)) and not shutil.which(self.server):
            out.append(f"the llama.cpp server program is missing ({self.server})")
        if not Path(self.model).is_file():
            out.append(f"the model file is missing ({self.model})")
        return out

    def _get(self, path: str, timeout: float = 5.0) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace") or "{}")

    def ready(self) -> bool:
        try:
            return self._get("/health", 3.0).get("status") == "ok"
        except (OSError, ValueError):
            return False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def mismatch(self) -> str:
        """Why the server answering on the port is not the one wanted ("" when it is: this model, and the vision
        part when there is one)."""
        try:
            props = self._get("/props", 3.0)
        except (OSError, ValueError):
            return ""                                  # (an older server without /props: trusted)
        path = props.get("model_path") or ""
        if path and os.path.realpath(path) != os.path.realpath(self.model):
            return f"it runs {Path(path).name}"
        if self.mmproj and (props.get("modalities") or {}).get("vision") is False:
            return "it runs this model without its vision part"
        return ""

    def start(self, stop=lambda: False, timeout: float = START_TIMEOUT) -> None:
        """Start the server (if it is not answering yet) and wait until the model is loaded. A server already on
        the port must run this model (with its vision part when there is one): one this object started is
        restarted otherwise; another one is an error (its port is taken)."""
        with self._lock:
            if self.ready():
                why = self.mismatch()
                if not why:
                    return
                if not self.running():
                    raise LlmServerError(f"port {self.port} is used by another llama.cpp server ({why}). Close the "
                                         "program that started it, or choose another port in the settings")
                self._lock.release()
                try:
                    self.stop()
                finally:
                    self._lock.acquire()
            bad = self.problems()
            if bad:
                raise LlmServerError("; ".join(bad))
            if not self.running():
                log = subprocess.DEVNULL
                if self.log_dir is not None:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    log = open(self.log_dir / "llama-server.log", "ab")    # noqa: SIM115  (kept by the process)
                self.proc = subprocess.Popen(server_command(self.server, self.model, self.port, self.cpu_moe,
                                                            self.mmproj),
                                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             start_new_session=True)
            end = time.time() + timeout
            while time.time() < end:
                if stop():
                    raise LlmServerError("cancelled")
                if self.proc is not None and self.proc.poll() is not None:
                    raise LlmServerError(f"the llama.cpp server stopped while loading (exit {self.proc.returncode}"
                                         + (f"; see {self.log_dir / 'llama-server.log'}" if self.log_dir else "")
                                         + ")")
                if self.ready():
                    return
                time.sleep(0.5)
            raise LlmServerError(f"the llama.cpp server did not load the model within {int(timeout)} s")

    def stop(self) -> bool:
        """Stop the server this object started (frees its RAM and VRAM). True if one was running."""
        with self._lock:
            p, self.proc = self.proc, None
            if p is None or p.poll() is not None:
                return False
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
            return True

    def refine_prompt(self, prompt: str, kind: str, model_file: str | None = None, seed: int = 0,
                      hint: str = "", cast: list[dict] | None = None, temperature: float = 0.5,
                      timeout: float = 300.0) -> str:
        system, user = refine_messages(prompt=prompt, kind=kind, hint=hint, cast=cast)
        return self._complete([{"role": "system", "content": system}, {"role": "user", "content": user}], seed,
                              temperature, timeout)

    def chat(self, prompt: str, kind: str, turns: list[tuple[str, str]], instruction: str,
             pictures: list[tuple[str, str]] = (), cast: list[dict] | None = None, seed: int = 0,
             temperature: float = 0.5, timeout: float = 300.0) -> str:
        """The next version in a refine conversation (see chat_messages); pictures need the vision projector."""
        if pictures and not self.mmproj:
            raise LlmServerError("this model has no vision part (mmproj file) next to it: it cannot see pictures")
        return self._complete(chat_messages(prompt=prompt, kind=kind, turns=turns, instruction=instruction,
                                            pictures=pictures, cast=cast), seed, temperature, timeout)

    def interrupt(self):
        pass                                          # (a request ends by itself; closing the window drops it)

    def _complete(self, messages: list[dict], seed: int, temperature: float, timeout: float) -> str:
        self.start()
        body = {"messages": messages,
                "max_tokens": 768, "temperature": float(temperature), "top_k": 64, "top_p": 0.95, "min_p": 0.05,
                "repeat_penalty": 1.05, "seed": int(seed), "stream": False}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            raise LlmServerError(f"the llama.cpp server answered {e.code}: {e.read()[:300]!r}") from e
        except OSError as e:
            raise LlmServerError(f"the llama.cpp server could not be asked: {e}") from e
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LlmServerError(f"unexpected answer from the llama.cpp server: {str(data)[:300]}") from e
        text = clean_refined(text)
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

    def refine_prompt(self, prompt, kind, model_file=None, seed=0, hint="", cast=None):
        if (model_file or "").startswith(PREFIX) or self.comfy is None:
            if self.llama is None:
                raise LlmServerError("the llama.cpp server is not set up")
            return self.llama.refine_prompt(prompt, kind, model_file, seed, hint, cast)
        return self.comfy.refine_prompt(prompt, kind, model_file, seed, hint, cast)

    def chat(self, *a, **k):
        if self.llama is None:
            raise LlmServerError("conversations need a llama.cpp model")
        return self.llama.chat(*a, **k)

    def interrupt(self):
        if self.comfy is not None:
            self.comfy.interrupt()
