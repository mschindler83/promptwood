"""The events the generator worker (video_chain.__main__) sends the window as '@@ {json}' lines (MWC-8): one TypedDict
per kind, a builder per kind the worker emits (so a field is spelled in one place) and the check the window makes once
at the boundary. Not in ui/: the worker must not import the window's modules."""
from __future__ import annotations

from typing import Any, TypedDict

# One TypedDict per kind of event the worker (video_chain.__main__) sends. `total=False` fields are optional on the wire.
# The window checks an event once, at the boundary (event_problem), so a handler can index the required fields.
class _Event(TypedDict):
    e: str


class LoadingEvent(_Event, total=False):
    what: str                           # "runpod" | "comfyui"


class IdleEvent(_Event, total=False):
    timeout: float | None
    took: float


class ByeEvent(_Event, total=False):
    reason: str


class StartEvent(_Event, total=False):
    nodes: int
    clips: int                          # (the chain command)
    steps: int
    out: str
    width: int
    height: int
    frames: int


class PlanEvent(_Event):
    make: list[str]


class NodeEvent(_Event, total=False):
    id: str
    state: str
    k: int


class StepEvent(_Event, total=False):
    id: str | None
    k: int
    i: float
    n: float


class NoteEvent(_Event, total=False):
    id: str | None
    text: str


class ErrorEvent(_Event, total=False):
    msg: str
    comfy_down: bool


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# kind -> {field: (check, what it must be)} for the fields that must be there
_REQUIRED: dict[str, dict[str, tuple[Any, str]]] = {
    "start": {"width": (_number, "a number"), "height": (_number, "a number"), "frames": (_number, "a number")},
    "plan": {"make": (lambda v: isinstance(v, list), "a list")},
    "node": {"id": (lambda v: isinstance(v, str), "text"), "state": (lambda v: isinstance(v, str), "text")},
    "step": {"i": (_number, "a number"), "n": (_number, "a number")},
}


def event_problem(ev: Any) -> str | None:
    """Why a worker event cannot be handled (None: it can). Unknown kinds are fine: the plugins hear them."""
    if not isinstance(ev, dict):
        return f"it is not an object: {ev!r}"
    for field, (ok, what) in _REQUIRED.get(str(ev.get("e")), {}).items():
        if field not in ev:
            return f"a {ev.get('e')} event without {field!r}: {ev!r}"
        if not ok(ev[field]):
            return f"a {ev.get('e')} event whose {field!r} is not {what}: {ev!r}"
    return None


# ---- builders: what the worker sends (typed, so a misspelt field is a type error)
def loading(what: str) -> LoadingEvent:
    return {"e": "loading", "what": what}


def loaded(what: str) -> LoadingEvent:
    return {"e": "loaded", "what": what}


def note(node_id: str | None, text: object) -> NoteEvent:
    return {"e": "note", "id": node_id, "text": str(text)}


def step(node_id: str | None, k: int, i: float, n: float) -> StepEvent:
    return {"e": "step", "id": node_id, "k": k, "i": i, "n": n}


def error(msg: str, comfy_down: bool = False) -> ErrorEvent:
    ev: ErrorEvent = {"e": "error", "msg": msg}
    if comfy_down:
        ev["comfy_down"] = True
    return ev


def idle(timeout: float | None, took: float | None = None) -> IdleEvent:
    ev: IdleEvent = {"e": "idle", "timeout": timeout}
    if took is not None:
        ev["took"] = took
    return ev


def bye(reason: str) -> ByeEvent:
    return {"e": "bye", "reason": reason}
