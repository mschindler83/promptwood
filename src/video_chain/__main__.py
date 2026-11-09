"""video-chain: one start image + N prompts -> N chained MiniMax-H3 clips + one joined video, rendered by a running
ComfyUI (see comfy.py). E.g.
  ./video-chain start.jpg prompts.txt --out ~/chains/test1 --lora glow_h3_lora.safetensors
The LoRA must be in one of ComfyUI's LoRA folders.
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
    p.add_argument("--events", action="store_true", help="also print progress as '@@ {json}' lines (for the GUI)")
    p.add_argument("--serve", action="store_true",
                   help="worker mode (for the GUI): load the model once, then read jobs as JSON lines from stdin "
                        '({"cmd": "run", "tree": DIR, "targets": [ids]} or {"cmd": "quit"})')
    p.add_argument("--idle-minutes", type=float, default=15,
                   help="worker mode: quit (and free the memory) after this long without a job; 0 = never")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI address")
    p.add_argument("--comfy-dir", type=Path, help="ComfyUI's folder (its LoRA folders; cleanup of its copies)")
    p.add_argument("--keep-comfy-copies", action="store_true",
                   help="leave ComfyUI's copies of clips and start images in its output/input folders")
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


COMFY_DOWN = 4      # exit code: ComfyUI is not running


def _generator(a, emit, current):
    """Connect to ComfyUI (it loads the model on the first clip and keeps it). Step progress goes out as events
    for the node in current."""
    from .comfy import ComfyGenerator, ComfyUnavailable
    emit({"e": "loading", "what": "comfyui"})
    try:
        gen = ComfyGenerator(a.comfy_url, progress=lambda i, n: emit(
            {"e": "step", "id": current.get("id"), "k": current.get("k", 0), "i": i, "n": n}),
            comfy_dir=a.comfy_dir, cleanup=not a.keep_comfy_copies)
    except ComfyUnavailable as e:
        emit({"e": "error", "msg": str(e), "comfy_down": True})
        raise SystemExit(COMFY_DOWN)
    gen.note = lambda text: emit({"e": "note", "id": current.get("id"), "text": str(text)})
    emit({"e": "loaded", "what": "comfyui"})
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
    treemod.ensure_handovers(project, tree)       # a missing hand-over frame would make its children look changed
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
    problem = treemod.guide_problem(tree, checker, todo)
    if problem:
        raise JobError(problem)
    sounds = treemod.audio_problems(project, tree, todo) + treemod.cast_problems(project, tree, todo) + \
        treemod.source_sound_problems(tree, todo, project)
    if sounds:
        raise JobError(sounds[0])
    no_end = treemod.missing_end_images(project, tree, todo)
    if no_end:
        raise JobError(f"the end image of clip #{no_end[0][1:]} is missing")
    ai_todo = [nid for nid in todo if not tree.nodes[nid].is_video]     # what needs the model
    # which H3 models the clips use: the reference one (face references or a sound) and/or the normal one
    ref_kinds = sorted({treemod.uses_ref_model(checker.settings(nid) or {}) for nid in ai_todo})
    return {"project": project, "targets": targets, "settings": settings, "todo": todo, "loras": loras,
            "ai_todo": ai_todo, "ref_kinds": ref_kinds}


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
    if job["ai_todo"]:                             # nothing to generate: no need for ComfyUI
        gen = _generator(a, emit, current)
        missing = gen.missing_loras(job["loras"])
        if missing:
            return _fail(emit, _missing_msg(missing[0]))
        problem = _speed_problem(gen, job["settings"], job["ref_kinds"])
        if problem:
            return _fail(emit, problem)
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


def _speed_problem(gen, settings: dict, ref_kinds=(None,)) -> str | None:
    """Why the HyperFlow weights / Pika LoRA these settings need cannot be used (None: fine, or nothing needed).
    ref_kinds: whether the clips use the reference model (True), the normal one (False), or both."""
    if not hasattr(gen, "check_speed"):
        return None
    from .comfy import ComfyError
    try:
        for kind in ref_kinds:
            gen.check_speed(settings if kind is None else {**settings, "refs": [True] if kind else None})
    except ComfyError as e:
        return str(e)
    return None


def _missing_msg(lora: str) -> str:
    return (f"{Path(lora).name} is not in ComfyUI's LoRA folders; copy or link it there "
            "(e.g. into my_models/loras)")


def serve(a, emit, make_generator: Callable[[dict], object], stdin=None) -> int:
    """Worker: connect once, then run jobs from stdin until 'quit', end of input, or the idle timeout.
    make_generator(current) connects to ComfyUI; ComfyUI keeps the model loaded between jobs, and the worker asks
    it to unload after the idle time (or on quit with free)."""
    stdin = stdin or sys.stdin
    lines = _LineReader(stdin)
    current = {"id": None, "k": 0}
    gen = make_generator(current)
    idle = a.idle_minutes * 60 if a.idle_minutes and a.idle_minutes > 0 else None
    emit({"e": "idle", "timeout": idle})
    while True:
        if idle is not None and not lines.ready(idle):
            if hasattr(gen, "free"):
                gen.free()                          # unload ComfyUI's models after the idle time
            emit({"e": "bye", "reason": "idle"})
            return 0
        line = lines.readline()
        if not line:                                # the GUI is gone: give ComfyUI's memory back too
            if hasattr(gen, "free"):
                try:
                    gen.free()
                except Exception:  # noqa: BLE001
                    pass
            return 0
        try:
            cmd = json.loads(line)
        except ValueError:
            cmd = None
        if not isinstance(cmd, dict):
            emit({"e": "error", "msg": f"bad command: {line.strip()[:80]}"})
            continue
        if cmd.get("cmd") == "quit":
            if cmd.get("free") and hasattr(gen, "free"):
                gen.free()
            emit({"e": "bye", "reason": "quit"})
            return 0
        if cmd.get("cmd") == "refine":               # export a path at a higher size (2K refine)
            started = time.monotonic()
            if cmd.get("free_first") and hasattr(gen, "free"):
                gen.free()
            if "last_hyperflow" in cmd and getattr(gen, "last_hyperflow", False) is None:
                gen.last_hyperflow = cmd["last_hyperflow"]
            try:
                treemod.refine_path(gen, Path(cmd["tree"]), cmd["target"], cmd["size"], Path(cmd["out"]),
                                    fast_attention=bool(cmd.get("fast_attention")),
                                    refine_swapped=bool(cmd.get("refine_swapped")),
                                    strength=str(cmd.get("strength") or "natural"),
                                    method=str(cmd.get("method") or "refine"),
                                    single=bool(cmd.get("single")),
                                    denoise=bool(cmd.get("denoise")),
                                    events=lambda ev: (ev.get("e") == "node" and ev.get("state") == "generating"
                                                       and current.update(id=ev["id"], k=ev.get("k", 0)),
                                                       emit(ev)))
            except (JobError, ValueError, FileNotFoundError, KeyError) as e:
                emit({"e": "error", "msg": str(e)})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                emit({"e": "error", "msg": f"{type(e).__name__}: {e}"})
            emit({"e": "idle", "timeout": idle, "took": round(time.monotonic() - started, 1)})
            continue
        if cmd.get("cmd") != "run":
            emit({"e": "error", "msg": f"unknown command {cmd.get('cmd')!r}"})
            continue
        started = time.monotonic()
        if cmd.get("free_first") and hasattr(gen, "free"):
            gen.free()                              # ComfyUI holds another model (an image edit): unload it first
        if "last_hyperflow" in cmd and getattr(gen, "last_hyperflow", False) is None:
            gen.last_hyperflow = cmd["last_hyperflow"]          # what the window knows of earlier workers
        try:
            job = prepare_job(Path(cmd["tree"]), list(cmd.get("targets") or []))
            if job["ai_todo"]:
                missing = gen.missing_loras(job["loras"])
                if missing:
                    raise JobError(_missing_msg(missing[0]))
                problem = _speed_problem(gen, job["settings"], job["ref_kinds"])
                if problem:
                    raise JobError(problem)
            run_job(job, gen if job["ai_todo"] else None, emit, current)
        except (JobError, ValueError, FileNotFoundError, KeyError) as e:
            emit({"e": "error", "msg": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            emit({"e": "error", "msg": f"{type(e).__name__}: {e}"})
        emit({"e": "idle", "timeout": idle, "took": round(time.monotonic() - started, 1)})


def _main_serve(a, emit) -> int:
    holder: dict = {}

    def on_term(_sig, _frame):                       # Stop in the GUI: cancel the clip in ComfyUI too
        gen = holder.get("gen")
        try:
            if gen is not None and hasattr(gen, "interrupt"):
                gen.interrupt()
        except Exception:  # noqa: BLE001  (Stop must end the worker whatever ComfyUI answers)
            pass
        finally:
            raise SystemExit(143)
    signal.signal(signal.SIGTERM, on_term)

    def make(current: dict):
        holder["gen"] = _generator(a, emit, current)
        return holder["gen"]
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
    settings = chain.settings_dict(model=treemod.DEFAULT_OPTIONS["model"], lora=str(lora) if lora else None,
                                   strength=a.strength, frames=a.frames, steps=a.steps, width=width, height=height,
                                   audio=not a.no_audio)
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
    gen = _generator(a, emit, current)
    if lora is not None and gen.missing_loras([str(lora)]):
        return _fail(emit, _missing_msg(str(lora)))
    result = chain.run_chain(gen, image, prompts, out, seed=a.seed, same_seed=a.same_seed, settings=settings,
                             resume=not a.redo, drop_first=not a.keep_repeated_frames, events=on_event)
    chain.copy_prompts(prompts_file, out)
    print(f"done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
