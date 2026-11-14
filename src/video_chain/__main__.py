"""video-chain: one start image + N prompts -> N chained MiniMax-H3 clips + one joined video, rendered by a running
ComfyUI (see comfy/). E.g.
  ./video-chain start.jpg prompts.txt --out ~/chains/test1 --lora glow_h3_lora.safetensors
The LoRA must be in one of ComfyUI's LoRA folders.
"""
from __future__ import annotations

import argparse
import io
import logging
import json
import os
import select
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from . import chain, logs, media
from . import events as EV
from . import tree as treemod

logger = logging.getLogger(__name__)


def _positive(kind):
    def parse(text: str):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not v > 0:                                  # (also rejects nan)
            raise argparse.ArgumentTypeError(f"must be more than 0, got {text}")
        return v
    return parse


def _not_negative(kind):
    def parse(text: str):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if not v >= 0:
            raise argparse.ArgumentTypeError(f"must not be negative, got {text}")
        return v
    return parse


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
    p.add_argument("--strength", type=_positive(float), default=1.0, help="LoRA strength (default 1.0)")
    p.add_argument("--frames", type=_positive(int), default=73, help="frames per clip, 17n+5: 39, 56, 73 (default), 90, 107, 124")
    p.add_argument("--steps", type=_positive(int), default=28, help="sampling steps (default 28; the trainer's samples use 20)")
    p.add_argument("--seed", type=_not_negative(int), default=42, help="seed of clip 1; clip k uses seed+k-1 (default 42)")
    p.add_argument("--same-seed", action="store_true", help="use the same seed for every clip")
    p.add_argument("--resolution", type=_positive(int), default=512,
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
    p.add_argument("--idle-minutes", type=_not_negative(float), default=15,
                   help="worker mode: quit (and free the memory) after this long without a job; 0 = never")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI address")
    p.add_argument("--comfy-dir", type=Path, help="ComfyUI's folder (its LoRA folders; cleanup of its copies)")
    p.add_argument("--keep-comfy-copies", action="store_true",
                   help="leave ComfyUI's copies of clips and start images in its output/input folders")
    p.add_argument("--cloud", choices=("runpod",),
                   help="worker mode: upscale (SeedVR2) on a RunPod pod made for this worker and terminated when it "
                        "ends, instead of the local ComfyUI (API key: $RUNPOD_API_KEY or the system keyring)")
    p.add_argument("--runpod-gpu", default=None, help="RunPod GPU type id (default: the fastest offered)")
    p.add_argument("--runpod-no-fallback", action="store_true",
                   help="only that GPU type (default: the others in turn when it has no capacity)")
    p.add_argument("--runpod-volume", default=None,
                   help="RunPod network volume id for ComfyUI and the SeedVR2 weights (public files only): "
                        "faster starts")
    p.add_argument("--runpod-without-volume", action="store_true",
                   help="when the network volume's data centre has no GPU free, start without the volume")
    p.add_argument("--runpod-models", default="7b", help="SeedVR2 sizes to fetch on the pod: 7b, 3b or '7b 3b'")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line; options that do nothing in the chosen mode are a mistake (exit code 2, a message)."""
    p = build_parser()
    a = p.parse_args(argv)
    if a.tree and (a.image is not None or a.prompts is not None):
        p.error("--tree reads the start image and prompts from the project: give no IMAGE PROMPTS with it")
    if a.targets and not a.tree:
        p.error("--targets belongs to --tree")
    if not a.cloud:
        wanted = [name for name, given in (("--runpod-gpu", a.runpod_gpu), ("--runpod-no-fallback", a.runpod_no_fallback),
                                           ("--runpod-volume", a.runpod_volume),
                                           ("--runpod-without-volume", a.runpod_without_volume),
                                           ("--runpod-models", a.runpod_models != "7b")) if given]
        if wanted:
            p.error(f"{', '.join(wanted)} only work with --cloud runpod")
    return a


class _Printer:
    """emit(event): one "@@ {json}" line on stdout for the GUI; gone: the GUI is gone, nothing more is written."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.gone = False

    def __call__(self, event: Mapping[str, Any]) -> None:
        if not self.enabled or self.gone:
            return
        try:
            print("@@ " + json.dumps(event, ensure_ascii=False), flush=True)
        except (OSError, ValueError):                  # the GUI is gone (broken pipe) or stdout is closed
            self.gone = True


def _printer(enabled: bool) -> _Printer:
    return _Printer(enabled)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    logs.setup(to_file=False)                  # (stderr only: the GUI that starts this keeps it)
    try:
        cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8", errors="replace")      # (the events do not depend on the locale)
    except (AttributeError, ValueError, OSError):
        pass
    emit = _printer(a.events)
    try:
        media.require_ffmpeg()                # (a plain message at the start, not a FileNotFoundError deep inside)
    except media.MediaError as e:
        return _fail(emit, str(e))
    try:
        return _main(a, emit)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(EV.error(f"{type(e).__name__}: {e}"))
        return 1


def _fail(emit, msg: str) -> int:
    print(msg, file=sys.stderr)
    emit(EV.error(msg))
    return 2


COMFY_DOWN = 4      # exit code: ComfyUI is not running


CLOUD_FAILED = 5    # exit code: the RunPod pod could not be made ready


def _cloud_generator(a, emit, current, holder: dict):
    """A RunPod pod made ready for this worker (holder["session"]: terminated when the worker ends), and the
    upscaler talking to its ComfyUI through the SSH tunnel."""
    from . import runpod as rp
    emit(EV.loading("runpod"))
    def note(text):
        emit(EV.note(current.get("id"), text))

    try:
        session = rp.RunpodSession(rp.RunpodApi(rp.api_key()), gpu=a.runpod_gpu or rp.DEFAULT_GPU,
                                   fallback=not a.runpod_no_fallback, volume=a.runpod_volume,
                                   models=a.runpod_models, note=note, log=lambda t: print(t, flush=True),
                                   without_volume=a.runpod_without_volume)
        holder["session"] = session
        session.start()
        gen = rp.CloudUpscaler(session, progress=lambda i, n: emit(
            EV.step(current.get("id"), current.get("k", 0), i, n)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(EV.error(f"RunPod: {e}"))
        _close_cloud(holder, emit)
        raise SystemExit(CLOUD_FAILED)
    gen.note = note  # type: ignore[method-assign]  # (the worker's own hook replaces the no-op on this instance)
    emit(EV.loaded("runpod"))
    return gen


def _close_cloud(holder: dict, emit) -> None:
    """Terminate the worker's pod (if any); a pod that may be left is reported."""
    session = holder.pop("session", None)
    if session is None:
        return
    sigs = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    old = {}
    try:
        for sig in sigs:                                # (a second Stop must not cut the termination short)
            old[sig] = signal.signal(sig, signal.SIG_IGN)
    except ValueError:                                  # (not the main thread)
        pass
    try:
        session.close()
    except Exception as e:  # noqa: BLE001
        print(f"RunPod: {e}", file=sys.stderr, flush=True)
        emit(EV.error(f"RunPod: {e}"))
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _generator(a, emit, current):
    """Connect to ComfyUI (it loads the model on the first clip and keeps it). Step progress goes out as events
    for the node in current."""
    from .comfy import ComfyGenerator, ComfyUnavailable
    emit(EV.loading("comfyui"))
    try:
        gen = ComfyGenerator(a.comfy_url, progress=lambda i, n: emit(
            EV.step(current.get("id"), current.get("k", 0), i, n)),
            comfy_dir=a.comfy_dir, cleanup=not a.keep_comfy_copies)
    except ComfyUnavailable as e:
        emit(EV.error(str(e), comfy_down=True))
        raise SystemExit(COMFY_DOWN)
    gen.note = lambda text: emit(EV.note(current.get("id"), text))  # type: ignore[method-assign]  # (the worker's hook replaces the no-op on this instance)
    emit(EV.loaded("comfyui"))
    from . import plugins                          # (a plugin may wrap the generator or give another backend)
    return plugins.wrap_generator(gen, a, log=lambda m: print(m, file=sys.stderr, flush=True))


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
    # a sound-only change is mixed again now (seconds, no model), so the plan knows what repeats or carries over
    # that sound and has to be made again
    try:
        treemod.remix_sounds(project, tree, sorted({n.id for t in targets for n in tree.path_to(t)}, key=tree.depth))
    except (OSError, RuntimeError) as e:
        raise JobError(f"could not mix a clip's sound again: {e}") from e
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
    no_key = treemod.missing_keyframes(project, tree, todo)
    if no_key:
        raise JobError(f"a keyframe picture of clip #{no_key[0][1:]} is missing")
    problem = treemod.keyframe_problem(tree, checker, todo)
    if problem:
        raise JobError(problem)
    ai_todo = [nid for nid in todo if not tree.nodes[nid].is_video]     # what needs the model
    # which H3 models the clips use: the reference one (face references or a sound) and/or the normal one
    ref_kinds = sorted({treemod.uses_ref_model(checker.settings(nid) or {}) for nid in ai_todo})
    return {"project": project, "targets": targets, "settings": settings, "todo": todo, "loras": loras,
            "ai_todo": ai_todo, "ref_kinds": ref_kinds}


def run_job(job: dict, gen, emit, current: dict) -> None:
    """Make what a prepared job needs with an already loaded generator (None if nothing is to be made)."""
    s = job["settings"]
    emit(EV.StartEvent(e="start", nodes=len(job["todo"]), width=s["width"], height=s["height"],
                       frames=s["frames"], steps=s["steps"], out=str(job["project"])))

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
    current: dict[str, Any] = {"id": None, "k": 0}
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


def _free_quietly(gen) -> None:
    """Ask ComfyUI to unload its models; a ComfyUI that is down is not the worker's end."""
    if hasattr(gen, "free"):
        try:
            gen.free()
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def _prologue(gen, cmd: dict) -> None:
    """What a run / refine command asks before it starts."""
    if cmd.get("free_first") and hasattr(gen, "free"):
        gen.free()                                  # ComfyUI holds another model (an image edit): unload it first
    if "last_hyperflow" in cmd and getattr(gen, "last_hyperflow", False) is None:
        gen.last_hyperflow = cmd["last_hyperflow"]  # what the window knows of earlier workers


def _guarded(emit, job: Callable[[], None]) -> None:
    """Run one job of the worker loop; its errors are events, never the worker's end."""
    try:
        job()
    except (JobError, ValueError, FileNotFoundError, KeyError) as e:
        emit(EV.error(str(e)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(EV.error(f"{type(e).__name__}: {e}"))


def serve(a, emit, make_generator: Callable[[dict], Any], stdin=None) -> int:
    """Worker: connect once, then run jobs from stdin until 'quit', end of input, or the idle timeout.
    make_generator(current) connects to ComfyUI; ComfyUI keeps the model loaded between jobs, and the worker asks
    it to unload after the idle time (or on quit with free)."""
    stdin = stdin or sys.stdin
    lines = _LineReader(stdin)
    current: dict[str, Any] = {"id": None, "k": 0}
    gen = make_generator(current)
    idle = a.idle_minutes * 60 if a.idle_minutes and a.idle_minutes > 0 else None
    emit(EV.idle(idle))
    while True:
        if getattr(emit, "gone", False):            # the GUI's pipe is closed: nobody to answer
            _free_quietly(gen)
            return 0
        if idle is not None and not lines.ready(idle):
            _free_quietly(gen)                      # unload ComfyUI's models after the idle time
            emit(EV.bye("idle"))
            return 0
        line = lines.readline()
        if not line:                                # the GUI is gone: give ComfyUI's memory back too
            _free_quietly(gen)
            return 0
        try:
            cmd = json.loads(line)
        except ValueError:
            cmd = None
        if not isinstance(cmd, dict):
            emit(EV.error(f"bad command: {line.strip()[:80]}"))
            continue
        if cmd.get("cmd") == "quit":
            if cmd.get("free"):
                _free_quietly(gen)
            emit(EV.bye("quit"))
            return 0
        if cmd.get("cmd") == "refine":               # export a path at a higher size (2K refine)
            started = time.monotonic()

            def on_refine_event(ev: dict) -> None:
                if ev.get("e") == "node" and ev.get("state") == "generating":
                    current.update(id=ev["id"], k=ev.get("k", 0))
                emit(ev)

            def refine():
                _prologue(gen, cmd)
                treemod.refine_path(gen, Path(cmd["tree"]), cmd["target"], cmd["size"], Path(cmd["out"]),
                                    fast_attention=bool(cmd.get("fast_attention")),
                                    refine_swapped=bool(cmd.get("refine_swapped")),
                                    strength=str(cmd.get("strength") or "natural"),
                                    method=str(cmd.get("method") or "refine"),
                                    single=bool(cmd.get("single")),
                                    denoise=bool(cmd.get("denoise")),
                                    events=on_refine_event)
            _guarded(emit, refine)
            emit(EV.idle(idle, round(time.monotonic() - started, 1)))
            continue
        if cmd.get("cmd") != "run":
            emit(EV.error(f"unknown command {cmd.get('cmd')!r}"))
            continue
        if getattr(gen, "upscale_only", False):     # (the cloud worker: an export only)
            emit(EV.error("this worker only upscales on RunPod; clips are made by the local ComfyUI"))
            emit(EV.idle(idle))
            continue
        started = time.monotonic()

        def run():
            _prologue(gen, cmd)
            job = prepare_job(Path(cmd["tree"]), list(cmd.get("targets") or []))
            if job["ai_todo"]:
                missing = gen.missing_loras(job["loras"])
                if missing:
                    raise JobError(_missing_msg(missing[0]))
                problem = _speed_problem(gen, job["settings"], job["ref_kinds"])
                if problem:
                    raise JobError(problem)
            run_job(job, gen if job["ai_todo"] else None, emit, current)
        _guarded(emit, run)
        emit(EV.idle(idle, round(time.monotonic() - started, 1)))


def _main_serve(a, emit) -> int:
    holder: dict = {}

    def on_term(_sig, _frame):                       # Stop in the GUI: cancel the clip in ComfyUI too
        gen = holder.get("gen")
        try:
            if gen is not None and hasattr(gen, "interrupt"):
                gen.interrupt()
        except Exception as e:  # noqa: BLE001  (Stop must end the worker whatever ComfyUI answers)
            logger.warning("interrupt on stop failed: %s", e)
        finally:
            raise SystemExit(143)
    signal.signal(signal.SIGTERM, on_term)

    def make(current: dict):
        holder["gen"] = (_cloud_generator(a, emit, current, holder) if a.cloud
                         else _generator(a, emit, current))
        return holder["gen"]
    if not a.cloud:
        return serve(a, emit, make)
    signal.signal(signal.SIGHUP, on_term)            # (the pod is terminated whichever way the worker ends)
    signal.signal(signal.SIGINT, on_term)
    try:
        return serve(a, emit, make)
    finally:
        _close_cloud(holder, emit)


def _main(a, emit) -> int:
    if a.cloud and not a.serve:
        return _fail(emit, "--cloud works in worker mode (--serve) only")
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
    print(f"{len(prompts)} prompts, {width}x{height}, {a.frames} frames ({a.frames / chain.FPS:.1f} s) per clip, "
          f"{a.steps} steps, seed {a.seed}{' (same for all)' if a.same_seed else '+'}")

    emit(EV.StartEvent(e="start", clips=len(prompts), width=width, height=height, frames=a.frames,
                       steps=a.steps, out=str(out)))
    current: dict[str, Any] = {"k": 0}

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
