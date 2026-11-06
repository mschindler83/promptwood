"""video-chain: one start image + N prompts -> N chained H3 clips + one joined video.

Run it with ai-toolkit's Python (see the video-chain launcher next to this project), e.g.
  ./video-chain start.jpg prompts.txt --out ~/chains/test1 --lora glow_h3_lora.safetensors
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import chain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="video-chain", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="start image of the first clip")
    p.add_argument("prompts", type=Path, help="text file, one prompt per line (# comments and empty lines skipped)")
    p.add_argument("--out", type=Path, required=True, help="output folder (clips, last frames, chain.mp4)")
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


def _main(a, emit) -> int:
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

    emit({"e": "loading"})
    from .h3 import H3Generator          # imports torch / ai-toolkit: only now, after the cheap checks
    gen = H3Generator(width=width, height=height, frames=a.frames, steps=a.steps, lora=lora, strength=a.strength,
                      model=a.model, offload=a.offload, audio=not a.no_audio, aitk=a.aitk,
                      min_free_gb=a.min_free_gb,
                      progress=lambda i, n: emit({"e": "step", "k": current["k"], "i": i, "n": n}))
    emit({"e": "loaded"})
    result = chain.run_chain(gen, image, prompts, out, seed=a.seed, same_seed=a.same_seed, settings=settings,
                             resume=not a.redo, drop_first=not a.keep_repeated_frames, events=on_event)
    chain.copy_prompts(prompts_file, out)
    print(f"done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
