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
CONTEXT = 8192


class LlmServerError(RuntimeError):
    pass


def model_label(path: str) -> str:
    name = Path(path).name
    stem = name[:-5] if name.endswith(".gguf") else name
    free = "abliterated" in stem.lower() or "uncensored" in stem.lower()
    return f"{stem}{' (uncensored)' if free else ''}  ·  llama.cpp"


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
                 "uncensored": "abliterated" in self.model.lower()}]

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

    def start(self, stop=lambda: False, timeout: float = START_TIMEOUT) -> None:
        """Start the server (if it is not answering yet) and wait until the model is loaded."""
        with self._lock:
            if self.ready():
                return
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
        self.start()
        system, user = refine_messages(prompt=prompt, kind=kind, hint=hint, cast=cast)
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": 512, "temperature": float(temperature), "top_k": 64, "top_p": 0.95, "min_p": 0.05,
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
