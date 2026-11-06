"""Stands in for `video-chain` in GUI tests: same arguments and events, synthetic clips instead of H3.

FAKE_FAIL=1 fails after clip 1, FAKE_DELAY=<s> waits that long per sampling step."""
import os
import sys
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


def main() -> int:
    a = cli.build_parser().parse_args(sys.argv[1:])
    emit = cli._printer(a.events)
    try:
        prompts = chain.read_prompts(a.prompts)
        w, h = chain.canvas_size(*chain.image_size(a.image), a.resolution)
        settings = chain.settings_dict(model=a.model, lora=str(a.lora.resolve()) if a.lora else None,
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
        gen = cls(frames=17, audio=not a.no_audio, step_delay=float(os.environ.get("FAKE_DELAY", "0")),
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
