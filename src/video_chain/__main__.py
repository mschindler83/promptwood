"""video-chain: one start image + N prompts -> N chained H3 clips + one joined video.

Run it with ai-toolkit's Python (see the video-chain launcher next to this project), e.g.
  ./video-chain start.jpg prompts.txt --out ~/chains/test1 --lora glow_h3_lora.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from . import chain
from . import tree as treemod


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="video-chain", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, nargs="?", help="start image of the first clip")
    p.add_argument("prompts", type=Path, nargs="?",
                   help="text file, one prompt per line (# comments and empty lines skipped)")
    p.add_argument("--out", type=Path, help="output folder (clips, last frames, chain.mp4)")
    p.add_argument("--tree", type=Path, help="branching mode: a project folder with tree.json (made by the GUI); "
                                             "image, prompts and clip settings come from there")
    p.add_argument("--targets", default="", help="branching mode: comma-separated node ids whose paths to make")
    p.add_argument("--lora", type=Path, help="H3 LoRA (.safetensors) to apply; none = base model")
    p.add_argument("--strength", type=float, default=1.0, help="LoRA strength (default 1.0)")
    p.add_argument("--frames", type=int, default=73, help="frames per clip, 17n+5: 39, 56, 73 (default), 90, 107, 124")
    p.add_argument("--steps", type=int, default=28, help="sampling steps (default 28; the trainer's samples use 20)")
    p.add_argument("--seed", type=int, default=42, help="seed of clip 1; clip k uses seed+k-1 (default 42)")
    p.add_argument("--same-seed", action="store_true", help="use the same seed for every clip")
    p.add_argument("--resolution", type=int, default=512,
                   help="about resolution² pixels in the start image's aspect ratio (default 512, like training)")
    p.add_argument("--size", help="exact size WxH instead (multiples of 32); the start image is stretched to it")
    p.add_argument("--no-audio", action="store_true", help="generate video only")
    p.add_argument("--keep-repeated-frames", action="store_true",
                   help="keep the first frame of clips 2..n in chain.mp4 (it repeats the last frame before)")
    p.add_argument("--redo", action="store_true", help="generate every clip again (default: reuse unchanged clips)")
    p.add_argument("--offload", type=float, default=0.9, help="share of transformer layers kept in RAM (default 0.9)")
    p.add_argument("--min-free-gb", type=float, default=18.0,
                   help="refuse to start with less free GPU memory (protects a running training; default 18)")
    p.add_argument("--model", default="Comfy-Org/MiniMax-H3")
    p.add_argument("--events", action="store_true", help="also print progress as '@@ {json}' lines (for the GUI)")
    p.add_argument("--serve", action="store_true",
                   help="worker mode (for the GUI): load the model once, then read jobs as JSON lines from stdin "
                        '({"cmd": "run", "tree": DIR, "targets": [ids]} or {"cmd": "quit"})')
    p.add_argument("--idle-minutes", type=float, default=15,
                   help="worker mode: quit (and free the memory) after this long without a job; 0 = never")
    p.add_argument("--loras", default="", help="worker mode: comma-separated LoRA files to prepare when loading")
    p.add_argument("--backend", choices=["ai-toolkit", "comfyui"], default="ai-toolkit",
                   help="worker mode: render with ai-toolkit (loads the model itself) or a running ComfyUI")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI address (backend comfyui)")
    p.add_argument("--comfy-dir", type=Path, help="ComfyUI's folder (its LoRA folders; cleanup of its copies)")
    p.add_argument("--keep-comfy-copies", action="store_true",
                   help="leave ComfyUI's copies of clips and start images in its output/input folders")
    p.add_argument("--aitk", type=Path, default=Path("/path/to/ai/ai-toolkit"),
                   help="ai-toolkit folder")
    return p


def _printer(enabled: bool):
    def emit(event: dict) -> None:
        if enabled:
            print("@@ " + json.dumps(event, ensure_ascii=False), flush=True)
    return emit


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    emit = _printer(a.events)
    try:
        return _main(a, emit)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit({"e": "error", "msg": f"{type(e).__name__}: {e}"})
        return 1


def _fail(emit, msg: str) -> int:
    print(msg, file=sys.stderr)
    emit({"e": "error", "msg": msg})
    return 2


def _generator(a, emit, current, width, height, frames, steps, lora, strength, model, audio, loras=()):
    """Load MiniMax-H3 (minutes). Step progress goes out as events for the node in current."""
    emit({"e": "loading"})
    from .h3 import H3Generator          # imports torch / ai-toolkit: only now, after the cheap checks
    gen = H3Generator(width=width, height=height, frames=frames, steps=steps, lora=lora, strength=strength,
                      model=model, offload=a.offload, audio=audio, aitk=a.aitk, min_free_gb=a.min_free_gb,
                      loras=loras,
                      progress=lambda i, n: emit({"e": "step", "id": current["id"], "k": current["k"],
                                                  "i": i, "n": n}))
    emit({"e": "loaded"})
    return gen


class JobError(Exception):
    pass


def prepare_job(project: Path, targets: list[str]) -> dict:
    """Check a tree job without loading anything; raises JobError with a message for the user."""
    project = Path(project).expanduser().resolve()
    tree = treemod.load(project)
    if not targets:
        raise JobError("no node ids given")
    unknown = [t for t in targets if t not in tree.nodes]
    if unknown:
        raise JobError(f"unknown node {unknown[0]}")
    settings = treemod.current_settings(project, tree.options, tree)
    if settings is None:
        raise JobError("the project has no start image (and no video to take the size from)")
    empty = treemod.empty_prompts(tree, targets)
    if empty:
        raise JobError(f"empty prompt in node {empty[0]}")
    checker = treemod.Checker(project, tree)
    todo = treemod.plan(tree, checker, targets)
    loras = treemod.loras_needed(tree, checker, todo)
    missing = [lo for lo in loras if not Path(lo).is_file()]
    if missing:
        raise JobError(f"LoRA not found: {missing[0]}")
    ai_todo = [nid for nid in todo if not tree.nodes[nid].is_video]     # what needs the model
    if settings.get("backend") == "ai-toolkit":
        why = treemod.needs_comfy(tree, checker, ai_todo)
        if why:
            raise JobError(f"{why}: that needs the ComfyUI generator")
    return {"project": project, "targets": targets, "settings": settings, "todo": todo, "loras": loras,
            "ai_todo": ai_todo}


def run_job(job: dict, gen, emit, current: dict) -> None:
    """Make what a prepared job needs with an already loaded generator (None if nothing is to be made)."""
    s = job["settings"]
    emit({"e": "start", "nodes": len(job["todo"]), "width": s["width"], "height": s["height"],
          "frames": s["frames"], "steps": s["steps"], "out": str(job["project"])})

    def on_event(ev: dict) -> None:
        if ev.get("e") == "node" and ev.get("state") == "generating":
            current["id"], current["k"] = ev["id"], ev.get("k", 0)
        emit(ev)

    treemod.run_tree(gen, job["project"], job["targets"], events=on_event)


def _main_tree(a, emit) -> int:
    try:
        job = prepare_job(a.tree, [t for t in a.targets.split(",") if t])
    except (JobError, ValueError) as e:
        return _fail(emit, str(e))
    current = {"id": None, "k": 0}
    gen = None
    if job["ai_todo"]:                             # nothing to generate: no need to load the model
        s = job["settings"]
        gen = _generator(a, emit, current, s["width"], s["height"], s["frames"], s["steps"], None, 1.0,
                         s["model"], s["audio"], loras=job["loras"])
    run_job(job, gen, emit, current)
    return 0


class _LineReader:
    """Lines from stdin without Python's buffering, so select() sees every waiting line (two jobs in one read
    would otherwise leave the second one in the buffer until the idle timeout)."""

    def __init__(self, stream):
        self.fd = stream.fileno()
        self.buf = b""

    def ready(self, timeout: float) -> bool:
        if b"\n" in self.buf:
            return True
        r, _, _ = select.select([self.fd], [], [], timeout)
        return bool(r)

    def readline(self) -> str:
        while b"\n" not in self.buf:
            chunk = os.read(self.fd, 65536)
            if not chunk:                           # end of input
                line, self.buf = self.buf, b""
                return line.decode(errors="replace")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="replace") + "\n"


RESTART = 3          # worker exit code: the job needs a fresh worker (other model, sound or a bigger LoRA)


def serve(a, emit, make_generator: Callable[[list[str], dict], object], stdin=None) -> int:
    """Worker: load once, then run jobs from stdin until 'quit', end of input, or the idle timeout.

    make_generator(loras, current) loads the model with these LoRAs prepared. A job the loaded model cannot do
    (another model or sound setting, or a LoRA of a higher rank than the prepared ones) is answered with a
    'restart' event and exit code RESTART; the GUI then starts a new worker for it."""
    stdin = stdin or sys.stdin
    lines = _LineReader(stdin)
    current = {"id": None, "k": 0}
    model, audio = a.model, not a.no_audio
    gen = make_generator([lo for lo in a.loras.split(",") if lo], current)
    idle = a.idle_minutes * 60 if a.idle_minutes and a.idle_minutes > 0 else None
    emit({"e": "idle", "timeout": idle})
    while True:
        if idle is not None and not lines.ready(idle):
            if hasattr(gen, "free"):
                gen.free()                          # ComfyUI: unload its models after the idle time
            emit({"e": "bye", "reason": "idle"})
            return 0
        line = lines.readline()
        if not line:                                # the GUI is gone
            return 0
        try:
            cmd = json.loads(line)
        except ValueError:
            emit({"e": "error", "msg": f"bad command: {line.strip()[:80]}"})
            continue
        if cmd.get("cmd") == "quit":
            if cmd.get("free") and hasattr(gen, "free"):
                gen.free()
            emit({"e": "bye", "reason": "quit"})
            return 0
        if cmd.get("cmd") != "run":
            emit({"e": "error", "msg": f"unknown command {cmd.get('cmd')!r}"})
            continue
        started = time.monotonic()
        try:
            job = prepare_job(Path(cmd["tree"]), list(cmd.get("targets") or []))
            s = job["settings"]
            if job["ai_todo"] and s.get("backend", a.backend) != a.backend:
                emit({"e": "restart", "backend": s.get("backend"), "loras": job["loras"]})
                return RESTART
            if job["ai_todo"] and getattr(gen, "restart_helps", True):
                if s["model"] != model or s["audio"] != audio or not gen.can_use(job["loras"]):
                    emit({"e": "restart", "model": s["model"], "audio": s["audio"], "loras": job["loras"]})
                    return RESTART
            elif job["ai_todo"]:
                missing = gen.missing_loras(job["loras"])
                if missing:
                    raise JobError(f"{Path(missing[0]).name} is not in ComfyUI's LoRA folders; copy or link it "
                                   "there (e.g. into my_models/loras)")
            run_job(job, gen if job["ai_todo"] else None, emit, current)
        except (JobError, ValueError, FileNotFoundError, KeyError) as e:
            emit({"e": "error", "msg": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            emit({"e": "error", "msg": f"{type(e).__name__}: {e}"})
        emit({"e": "idle", "timeout": idle, "took": round(time.monotonic() - started, 1)})


COMFY_DOWN = 4      # worker exit code: ComfyUI is not running


def _main_serve(a, emit) -> int:
    holder: dict = {}

    def on_term(_sig, _frame):                       # Stop in the GUI: cancel the clip in ComfyUI too
        gen = holder.get("gen")
        if gen is not None and hasattr(gen, "interrupt"):
            gen.interrupt()
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, on_term)

    if a.backend == "comfyui":
        from .comfy import ComfyGenerator, ComfyUnavailable

        def make(_loras: list[str], current: dict):
            emit({"e": "loading", "what": "comfyui"})
            try:
                gen = ComfyGenerator(a.comfy_url, progress=lambda i, n: emit(
                    {"e": "step", "id": current["id"], "k": current["k"], "i": i, "n": n}),
                    comfy_dir=a.comfy_dir, cleanup=not a.keep_comfy_copies)
            except ComfyUnavailable as e:
                emit({"e": "error", "msg": str(e), "comfy_down": True})
                raise SystemExit(COMFY_DOWN)
            holder["gen"] = gen
            emit({"e": "loaded", "what": "comfyui"})
            return gen
    else:
        def make(loras: list[str], current: dict):
            gen = _generator(a, emit, current, 512, 512, 73, 28, None, 1.0, a.model, not a.no_audio, loras=loras)
            holder["gen"] = gen
            return gen
    return serve(a, emit, make)


def _main(a, emit) -> int:
    if a.serve:
        return _main_serve(a, emit)
    if a.tree:
        return _main_tree(a, emit)
    if a.image is None or a.prompts is None or a.out is None:
        return _fail(emit, "give IMAGE PROMPTS --out DIR, or --tree PROJECT --targets IDS")
    if a.frames < 22 or a.frames % 17 != 5:
        return _fail(emit, f"--frames must be on H3's 17n+5 grid (39, 56, 73, 90, 107, 124), got {a.frames}")
    if not a.prompts.is_file():
        return _fail(emit, f"prompt file not found: {a.prompts}")
    prompts = chain.read_prompts(a.prompts)
    if not prompts:
        return _fail(emit, f"no prompts in {a.prompts}")
    if not a.image.is_file():
        return _fail(emit, f"start image not found: {a.image}")
    try:
        width, height = chain.parse_size(a.size) if a.size else chain.canvas_size(*chain.image_size(a.image),
                                                                                   a.resolution)
    except ValueError as e:
        return _fail(emit, str(e))
    image, prompts_file, out = a.image.resolve(), a.prompts.resolve(), a.out.expanduser().resolve()
    lora = a.lora.resolve() if a.lora else None
    settings = chain.settings_dict(model=a.model, lora=str(lora) if lora else None, strength=a.strength,
                                   frames=a.frames, steps=a.steps, width=width, height=height, audio=not a.no_audio)
    print(f"{len(prompts)} prompts, {width}x{height}, {a.frames} frames ({a.frames / 24:.1f} s) per clip, "
          f"{a.steps} steps, seed {a.seed}{' (same for all)' if a.same_seed else '+'}")

    emit({"e": "start", "clips": len(prompts), "width": width, "height": height, "frames": a.frames,
          "steps": a.steps, "out": str(out)})
    current = {"k": 0}

    def on_event(ev: dict) -> None:
        if ev.get("e") == "clip":
            current["k"] = ev["k"]
        emit(ev)

    current.setdefault("id", None)
    gen = _generator(a, emit, current, width, height, a.frames, a.steps, lora, a.strength, a.model, not a.no_audio)
    result = chain.run_chain(gen, image, prompts, out, seed=a.seed, same_seed=a.same_seed, settings=settings,
                             resume=not a.redo, drop_first=not a.keep_repeated_frames, events=on_event)
    chain.copy_prompts(prompts_file, out)
    print(f"done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
