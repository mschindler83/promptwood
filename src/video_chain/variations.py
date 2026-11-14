"""Prompt variations (a port of the Dynamic Prompter ComfyUI node, with its planned ideas): a prompt with
placeholders, values for each, and a plan of which prompts the next Generate makes.

Template syntax (prompt and negative prompt):
  {name}                  replaced by one of the variable's values
  {?name|text with {name}} an optional section: left out when the value is empty ("(nothing)")
Values: text, a weight (Random picks a value with weight 3 three times as often as one with 1) and on/off.

Combinations are counted like the node: the first variable changes fastest (mixed radix). Orders:
  order     every combination in turn (the node's "sequential"), continuing where the last Generate stopped
  shuffle   every combination once, in a seeded random order, then again (no repeats within a round)
  random    each image picks every variable's value at random (weighted); repeats can happen
Each combination can be used `repeat` times in a row (the node's repeat_count; each image gets its own seed)."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

NAME = r"[A-Za-z_][\w-]*"
_COND = re.compile(r"\{\?(" + NAME + r")\|((?:[^{}]|\{" + NAME + r"\})*)\}")
_PLAIN = re.compile(r"\{(" + NAME + r")\}")
_ANY = re.compile(_COND.pattern + "|" + _PLAIN.pattern)       # both in one pass (values are not read again)
MAX_IMAGES = 100_000                       # one Generate makes at most this many
SHUFFLE_LIMIT = 100_000                     # bigger products: shuffle falls back to random picks
MODES = ("order", "shuffle", "random")


@dataclass
class Value:
    text: str
    weight: int = 1
    on: bool = True


@dataclass
class Variable:
    name: str
    values: list[Value] = field(default_factory=list)

    def active(self) -> list[Value]:
        return [v for v in self.values if v.on]


def _weight(x) -> int:
    """A weight from stored or imported data: a whole number of at least 1 (anything else counts as 1)."""
    try:
        return max(1, int(x))
    except (ValueError, TypeError, OverflowError):
        return 1


def names_in(*templates: str) -> list[str]:
    """The placeholder names used in the templates, in order of first appearance."""
    out: list[str] = []
    for t in templates:
        for m in re.finditer(r"\{(?:\?(" + NAME + r")\||(" + NAME + r")\})", t or ""):   # ({cat|dog} is not one)
            name = m.group(1) or m.group(2)
            if name not in out:
                out.append(name)
    return out


def _tidy(line: str) -> str:
    """One line after an empty value: no doubled spaces or commas, nothing dangling at its ends (its indentation
    stays)."""
    indent = line[:len(line) - len(line.lstrip(" \t"))]
    body = re.sub(r"[ \t]{2,}", " ", line.strip(" \t"))
    body = re.sub(r"[ \t]+([,;])", r"\1", body)
    body = re.sub(r"([,;])([ \t]*[,;])+", r"\1", body)
    return indent + body.strip(" \t,;")


def substitute(template: str, combo: dict[str, str]) -> str:
    """The template with the chosen values (unknown placeholders stay as they are). Only a line where a value was
    empty is tidied; values are inserted once (a value with {x} in it stays as it is)."""
    def one_line(line: str) -> str:
        empty = False

        def one(m):
            nonlocal empty
            if m.group(1) is not None:                      # {?name|text with {name}}
                val = combo.get(m.group(1))
                if val is None:
                    return m.group(0)
                if not val.strip():
                    empty = True
                    return ""
                return _PLAIN.sub(lambda p: combo.get(p.group(1), p.group(0)), m.group(2))
            val = combo.get(m.group(3))
            if val is None:
                return m.group(0)
            empty = empty or not val.strip()
            return val
        out = _ANY.sub(one, line)
        return _tidy(out) if empty else out
    return "\n".join(one_line(line) for line in (template or "").split("\n"))


class Plan:
    """The variables used by the templates and how many combinations they make."""

    def __init__(self, variables: list[Variable], *templates: str):
        used = names_in(*templates)
        by_name = {v.name: v for v in variables}
        self.vars = [by_name[n] for n in used if n in by_name and by_name[n].active()]
        self.missing = [n for n in used if n not in by_name or not by_name[n].active()]   # no values (yet)
        self.unused = [v.name for v in variables if v.name not in used]
        self.total = 1
        for v in self.vars:
            self.total *= len(v.active())

    def combination(self, index: int) -> dict[str, str]:
        out, index = {}, index % max(1, self.total)
        for v in self.vars:
            vals = v.active()
            out[v.name] = vals[index % len(vals)].text
            index //= len(vals)
        return out

    def index_of(self, combo: dict[str, str]) -> int:
        idx, mul = 0, 1
        for v in self.vars:
            texts = [x.text for x in v.active()]
            idx += texts.index(combo[v.name]) * mul if combo.get(v.name) in texts else 0
            mul *= len(texts)
        return idx

    def random_combination(self, rng: random.Random) -> dict[str, str]:
        return self.random_pick(rng)[0]

    def random_pick(self, rng: random.Random) -> tuple[dict[str, str], int]:
        """A weighted random combination and its number (as combination() counts: the picked positions)."""
        combo, idx, mul = {}, 0, 1
        for v in self.vars:
            vals = v.active()
            pos = rng.choices(range(len(vals)), weights=[max(1, x.weight) for x in vals])[0]
            combo[v.name] = vals[pos].text
            idx += pos * mul
            mul *= len(vals)
        return combo, idx

    def shuffled(self, seed: int) -> list[int] | None:
        if self.total > SHUFFLE_LIMIT:
            return None
        order = list(range(self.total))
        random.Random(seed).shuffle(order)
        return order


@dataclass
class Job:
    step: int                     # the position in the order (counts across Generates)
    index: int                    # the combination's number (0 … total-1)
    combo: dict[str, str]
    prompt: str
    negative: str
    key: str = ""                 # (the panel's position key it was planned under)


def iter_jobs(variables: list[Variable], prompt: str, negative: str = "", mode: str = "order", repeat: int = 1,
              count: int | None = None, start: int = 0, seed: int = 0):
    """The prompts of the next Generate, one Job at a time (see plan_jobs)."""
    plan = Plan(variables, prompt, negative)
    repeat = max(1, int(repeat))
    n = min(MAX_IMAGES, plan.total * repeat if count is None else max(0, int(count)))
    order = plan.shuffled(seed) if mode == "shuffle" else None
    rng = random.Random(f"{seed}:{start}")
    combo: dict[str, str] = {}
    index = 0
    for k in range(n):
        step = start + k
        if mode == "random" or (mode == "shuffle" and order is None):
            if k % repeat == 0:
                combo, index = plan.random_pick(rng)
        else:
            pos = (step // repeat) % plan.total
            index = order[pos] if order is not None else pos
            combo = plan.combination(index)
        yield Job(step, index, dict(combo), substitute(prompt, combo), substitute(negative, combo))


def plan_jobs(variables: list[Variable], prompt: str, negative: str = "", mode: str = "order", repeat: int = 1,
              count: int | None = None, start: int = 0, seed: int = 0) -> list[Job]:
    """The prompts of the next Generate: count images (None: every combination once, each `repeat` times),
    from position `start` (order / shuffle continue there; random uses it only to vary its picks)."""
    return list(iter_jobs(variables, prompt, negative, mode, repeat, count, start, seed))


# ---- the Dynamic Prompter node's JSON ({"p1": ["a", "b"], "p2": [["c", 3], "d"]}) and wildcard files
def from_json(text: str) -> list[Variable]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected an object like {\"p1\": [\"a\", \"b\"]}")
    out = []
    for name, vals in data.items():
        if not re.fullmatch(NAME, str(name)) or not isinstance(vals, list):
            continue
        var = Variable(str(name))
        for v in vals:
            if isinstance(v, list) and not v:
                continue                                    # (an empty list is no value)
            if isinstance(v, list):
                var.values.append(Value(str(v[0]), _weight(v[1]) if len(v) > 1 else 1))
            elif isinstance(v, dict):
                var.values.append(Value(str(v.get("text", "")), _weight(v.get("weight", 1)), bool(v.get("on", True))))
            else:
                var.values.append(Value(str(v)))
        out.append(var)
    return out


def to_json(variables: list[Variable]) -> str:
    """The node's format (weights as [text, weight]; switched-off values left out)."""
    return json.dumps({v.name: [x.text if x.weight == 1 else [x.text, x.weight] for x in v.values if x.on]
                       for v in variables}, indent=2, ensure_ascii=False)


def to_state(variables: list[Variable]) -> list[dict]:
    return [{"name": v.name, "values": [{"text": x.text, "weight": x.weight, "on": x.on} for x in v.values]}
            for v in variables]


def from_state(state) -> list[Variable]:
    out = []
    for d in state or []:
        if isinstance(d, dict) and re.fullmatch(NAME, str(d.get("name", ""))):
            out.append(Variable(d["name"], [Value(str(x.get("text", "")), _weight(x.get("weight", 1)),
                                                  bool(x.get("on", True)))
                                            for x in d.get("values") or [] if isinstance(x, dict)]))
    return out


def wildcard_values(text: str) -> list[str]:
    """A wildcard file (A1111 / ComfyUI style): one value per line, empty lines and # comments skipped."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def split_values(text: str) -> list[str]:
    """Pasted values: one per line, or comma-separated when it is a single line."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [x.strip() for x in lines[0].split(",") if x.strip()]
    return lines
