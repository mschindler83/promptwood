"""Random LoRA variations (the image window's Random LoRAs): every image gets the pinned LoRAs and a random choice of
the others, at least one LoRA and at most `most` in all, each at a random strength within its own range. No Qt."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

DEFAULT_MOST = 3            # LoRAs per image at most
DEFAULT_N = 8               # images per run (not in continuous mode)
MAX_N = 500
DECIMALS = 2                # strengths as the LoRA list shows them


@dataclass(frozen=True)
class Choice:
    """One LoRA of the list: the strengths it may get (low … high) and whether every image has it (pinned)."""
    path: str
    low: float
    high: float
    pinned: bool = False


def strength(low: float, high: float, rng: random.Random) -> float:
    """A random strength from low … high (either order), rounded as the list shows it."""
    a, b = sorted((float(low), float(high)))
    return round(rng.uniform(a, b), DECIMALS)


def problems(pool: list[Choice], most: int) -> list[str]:
    """Why no image can be made with this list and most (empty: it can)."""
    if not pool:
        return ["Add at least one LoRA to the list"]
    if most < 1:
        return ["An image needs at least 1 LoRA"]
    pinned = sum(1 for c in pool if c.pinned)
    if pinned > most:
        return [f"{pinned} LoRAs are pinned, but an image may have at most {most}"]
    return []


def pick(pool: list[Choice], most: int, rng: random.Random) -> list[tuple[str, float]]:
    """The LoRAs of one image, in the list's order: every pinned one, then a random number of the others, so the
    image has between 1 (or the number pinned) and `most` LoRAs (fewer when the list has fewer); each at a random
    strength from its range. problems() must be empty."""
    pinned = [c for c in pool if c.pinned]
    others = [c for c in pool if not c.pinned]
    least = max(1, len(pinned))
    upto = min(int(most), len(pinned) + len(others))
    total = rng.randint(least, upto)
    chosen = {id(c) for c in pinned} | {id(c) for c in rng.sample(others, total - len(pinned))}
    return [(c.path, strength(c.low, c.high, rng)) for c in pool if id(c) in chosen]


def default_range(s: float) -> tuple[float, float]:
    """A new LoRA's range: from half its strength in the list up to that strength (a negative one: towards 0)."""
    s = round(float(s), DECIMALS)
    half = round(s / 2, DECIMALS)
    return (half, s) if s >= 0 else (s, half)


def to_saved(pool: list[Choice], *, most: int, n: int, same_seed: bool) -> dict[str, Any]:
    """What is remembered (per LoRA by its file, so a LoRA keeps its range when the list changes)."""
    return {"most": int(most), "n": int(n), "same_seed": bool(same_seed),
            "loras": {c.path: {"low": c.low, "high": c.high, "pinned": c.pinned} for c in pool}}


def _number(value: Any, default: float) -> float:
    try:
        got = float(value)
    except (TypeError, ValueError):
        return default
    return got if math.isfinite(got) else default


def from_saved(saved: dict | None, items: list[dict]) -> dict[str, Any]:
    """The pool for the list's LoRAs (items: {"path", "strength"}), with the remembered ranges and pins (a LoRA
    not seen before: default_range), and the remembered most / n / same_seed."""
    saved = saved if isinstance(saved, dict) else {}
    loras = saved.get("loras")
    known: dict = loras if isinstance(loras, dict) else {}
    pool: list[Choice] = []
    for it in items:
        if any(c.path == it["path"] for c in pool):     # (the same file twice: one choice, never both in an image)
            continue
        lo, hi = default_range(_number(it.get("strength", 1.0), 1.0))
        entry = known.get(it["path"])
        got: dict = entry if isinstance(entry, dict) else {}
        pool.append(Choice(it["path"], round(_number(got.get("low"), lo), DECIMALS),
                           round(_number(got.get("high"), hi), DECIMALS), bool(got.get("pinned", False))))
    most = int(_number(saved.get("most"), DEFAULT_MOST))
    n = int(_number(saved.get("n"), DEFAULT_N))
    return {"choices": pool, "most": max(1, most), "n": min(MAX_N, max(1, n)),
            "same_seed": bool(saved.get("same_seed", False))}
