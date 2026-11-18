"""Random LoRA variations (the image window's Random LoRAs): from the LoRAs of the image model, every image gets the
pinned ones and at least one random one of the others (those not left out), at most `most` in all, each at a random
strength within its own range. No Qt."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

DEFAULT_MOST = 3            # LoRAs per image at most
MAX_MOST = 20               # (the field's limit; an image never gets more LoRAs than there are to choose from)
DEFAULT_N = 8               # images per run (not in continuous mode)
MAX_N = 500
DECIMALS = 2                # strengths as the LoRA list shows them
NONE_TO_CHOOSE = "There are no LoRAs to choose from (the model has none, or every one is left out)"


@dataclass(frozen=True)
class Choice:
    """One LoRA to choose from: the strengths it may get (low … high), whether every image has it (pinned) and
    whether it may be chosen at all (on; a pinned one always is)."""
    path: str
    low: float
    high: float
    pinned: bool = False
    on: bool = True

    @property
    def usable(self) -> bool:
        return self.on or self.pinned


def strength(low: float, high: float, rng: random.Random) -> float:
    """A random strength from low … high (either order), rounded as the list shows it."""
    a, b = sorted((float(low), float(high)))
    return round(rng.uniform(a, b), DECIMALS)


def problems(pool: list[Choice], most: int) -> list[str]:
    """Why no image can be made with these LoRAs and most (empty: it can)."""
    if not any(c.usable for c in pool):
        return [NONE_TO_CHOOSE]
    if most < 1:
        return ["An image needs at least 1 LoRA"]
    pinned = sum(1 for c in pool if c.pinned)
    if pinned > most:
        return [f"{pinned} LoRAs are pinned, but an image may have at most {most}"]
    if pinned == most and any(c.on and not c.pinned for c in pool):
        return [f"{pinned} LoRA{'s are' if pinned != 1 else ' is'} pinned and an image may have at most {most}: "
                "that leaves no room for a random one (raise the most)"]
    return []


def pick(pool: list[Choice], most: int, rng: random.Random) -> list[tuple[str, float]]:
    """The LoRAs of one image, in the pool's order: every pinned one and at least one random one of the others that
    are on, up to `most` in all (a random number, so often fewer; only the pinned ones when there are no others);
    each at a random strength from its range. problems() must be empty."""
    pinned = [c for c in pool if c.pinned]
    others = [c for c in pool if c.on and not c.pinned]
    room = min(int(most) - len(pinned), len(others))       # random ones at most
    extra = rng.randint(1, room) if room >= 1 else 0
    chosen = {id(c) for c in pinned} | {id(c) for c in rng.sample(others, extra)}
    return [(c.path, strength(c.low, c.high, rng)) for c in pool if id(c) in chosen]


def default_range(s: float) -> tuple[float, float]:
    """A new LoRA's range: from half its strength in the list up to that strength (a negative one: towards 0)."""
    s = round(float(s), DECIMALS)
    half = round(s / 2, DECIMALS)
    return (half, s) if s >= 0 else (s, half)


def to_saved(pool: list[Choice], *, most: int, n: int, same_seed: bool) -> dict[str, Any]:
    """What is remembered (per LoRA by its file, so a LoRA keeps its range when LoRAs come and go)."""
    return {"most": int(most), "n": int(n), "same_seed": bool(same_seed),
            "loras": {c.path: {"low": c.low, "high": c.high, "pinned": c.pinned, **({} if c.on else {"on": False})}
                      for c in pool}}


def _number(value: Any, default: float) -> float:
    try:
        got = float(value)
    except (TypeError, ValueError):
        return default
    return got if math.isfinite(got) else default


def from_saved(saved: dict | None, items: list[dict]) -> dict[str, Any]:
    """The pool for these LoRAs (items: {"path"[, "strength"]}), with the remembered ranges, pins and left-out ones
    (a LoRA not seen before: default_range of its strength, 1.0 without one), and the remembered most / n /
    same_seed."""
    saved = saved if isinstance(saved, dict) else {}
    loras = saved.get("loras")
    known: dict = loras if isinstance(loras, dict) else {}
    pool: list[Choice] = []
    seen: set[str] = set()
    for it in items:
        if it["path"] in seen:                          # (the same file twice: one choice, never both in an image)
            continue
        seen.add(it["path"])
        lo, hi = default_range(_number(it.get("strength", 1.0), 1.0))
        entry = known.get(it["path"])
        got: dict = entry if isinstance(entry, dict) else {}
        pool.append(Choice(it["path"], round(_number(got.get("low"), lo), DECIMALS),
                           round(_number(got.get("high"), hi), DECIMALS), bool(got.get("pinned", False)),
                           bool(got.get("on", True))))
    most = int(_number(saved.get("most"), DEFAULT_MOST))
    n = int(_number(saved.get("n"), DEFAULT_N))
    return {"choices": pool, "most": min(MAX_MOST, max(1, most)), "n": min(MAX_N, max(1, n)),
            "same_seed": bool(saved.get("same_seed", False))}
