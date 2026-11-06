"""Stands in for `video-chain` in GUI tests: same arguments and events, synthetic clips instead of H3.

FAKE_FAIL=1 fails after clip 1, FAKE_DELAY=<s> waits that long per sampling step."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeGenerator  # noqa: E402

from video_chain import __main__ as cli  # noqa: E402
from video_chain import chain  # noqa: E402


class Failing(FakeGenerator):
    def generate(self, *a, **kw):
        if self.calls:
            raise RuntimeError("fake failure in clip 2")
        super().generate(*a, **kw)


def tree_mode(a, emit) -> int:
    from video_chain import tree as T
    project = a.tree.resolve()
    tree = T.load(project)
    targets = [t for t in a.targets.split(",") if t]
    settings = T.current_settings(project, tree.options, tree)
    todo = T.plan(tree, T.Checker(project, tree), targets)
    emit({"e": "start", "nodes": len(todo), "width": settings["width"], "height": settings["height"],
          "frames": settings["frames"], "steps": settings["steps"], "out": str(project)})
    cur = {"id": None}

    def on_event(ev):
        if ev.get("e") == "node" and ev.get("state") == "generating":
            cur["id"] = ev["id"]
        emit(ev)

    if todo:
        emit({"e": "loading"})
        emit({"e": "loaded"})
    cls = Failing if os.environ.get("FAKE_FAIL") else FakeGenerator
    gen = cls(frames=17, audio=bool(tree.options.get("audio", True)), size=None,
              step_delay=float(os.environ.get("FAKE_DELAY", "0")),
              progress=lambda i, n: emit({"e": "step", "id": cur["id"], "i": i, "n": n}))
    T.run_tree(gen, project, targets, events=on_event, log=lambda m: print(m, flush=True))
    return 0


def serve_mode(a, emit) -> int:
    """The real worker loop with fake clips. Every connect is logged to FAKE_LOADS (a file) if set;
    FAKE_COMFY_DOWN: ComfyUI is not running; FAKE_COMFY_MISSING=<name>: LoRAs ComfyUI does not list."""
    def make(current):
        if os.environ.get("FAKE_COMFY_DOWN"):
            emit({"e": "error", "msg": f"ComfyUI is not reachable at {a.comfy_url}", "comfy_down": True})
            raise SystemExit(4)
        emit({"e": "loading"})
        if os.environ.get("FAKE_LOADS"):
            with open(os.environ["FAKE_LOADS"], "a") as f:
                f.write(json.dumps({"url": a.comfy_url}) + "\n")
        time.sleep(float(os.environ.get("FAKE_LOAD_DELAY", "0")))
        cls = Failing if os.environ.get("FAKE_FAIL") else FakeGenerator
        gen = cls(frames=17, audio=True, size=None, step_delay=float(os.environ.get("FAKE_DELAY", "0")),
                  progress=lambda i, n: emit({"e": "step", "id": current["id"], "i": i, "n": n}))
        missing = os.environ.get("FAKE_COMFY_MISSING", "")
        gen.missing_loras = lambda need: [lo for lo in need if missing and missing in lo]

        def free():
            if os.environ.get("FAKE_FREES"):
                with open(os.environ["FAKE_FREES"], "a") as f:
                    f.write("free\n")
        gen.free = free
        emit({"e": "loaded"})
        return gen
    return cli.serve(a, emit, make)


def main() -> int:
    a = cli.build_parser().parse_args(sys.argv[1:])
    emit = cli._printer(a.events)
    try:
        if a.serve:
            return serve_mode(a, emit)
        if a.tree:
            return tree_mode(a, emit)
        prompts = chain.read_prompts(a.prompts)
        w, h = chain.canvas_size(*chain.image_size(a.image), a.resolution)
        settings = chain.settings_dict(model="Comfy-Org/MiniMax-H3", lora=str(a.lora.resolve()) if a.lora else None,
                                       strength=a.strength, frames=a.frames, steps=a.steps, width=w, height=h,
                                       audio=not a.no_audio)
        emit({"e": "start", "clips": len(prompts), "width": w, "height": h, "frames": a.frames, "steps": a.steps,
              "out": str(a.out)})
        emit({"e": "loading"})
        emit({"e": "loaded"})
        cur = {"k": 0}

        def on_event(ev):
            if ev.get("e") == "clip":
                cur["k"] = ev["k"]
            emit(ev)

        cls = Failing if os.environ.get("FAKE_FAIL") else FakeGenerator
        gen = cls(frames=17, audio=not a.no_audio, size=None, step_delay=float(os.environ.get("FAKE_DELAY", "0")),
                  progress=lambda i, n: emit({"e": "step", "k": cur["k"], "i": i, "n": n}))
        chain.run_chain(gen, a.image.resolve(), prompts, a.out.resolve(), seed=a.seed, same_seed=a.same_seed,
                        settings=settings, resume=not a.redo, drop_first=not a.keep_repeated_frames,
                        events=on_event, log=lambda m: print(m, flush=True))
        return 0
    except Exception as e:  # noqa: BLE001
        emit({"e": "error", "msg": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
