"""What the window knows about the generator worker, as types instead of strings and tuples: the model ComfyUI
holds, the phase of a run, the job in progress, and what a running worker is bound to (MWC-6, MWC-7)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict


class ComfyModel(str, Enum):
    """The model this app last used in ComfyUI (None where it is kept: nothing of this app is loaded). A str enum:
    the values are what was written before, so a plain string still compares equal."""
    H3 = "h3"
    H3_HYPERFLOW = "h3-hyperflow"
    QWEN = "qwen"                       # the image edit model
    LLM = "llm"                         # the language model of the prompt refiner


CLIP_MODELS = (ComfyModel.H3, ComfyModel.H3_HYPERFLOW)      # the clip model: the generator worker holds it
IMAGE_MODELS = (ComfyModel.QWEN, ComfyModel.LLM)            # a model of an editing window: ComfyUI holds it


class Phase(str, Enum):
    """Where a generator run is."""
    IDLE = ""
    STARTING = "starting"
    DONE = "done"                       # the worker said the run is over (its exit decides success)


class ModelState(str, Enum):
    """The worker's model: loading it takes a while (preparing a RunPod pod, loading into ComfyUI)."""
    NONE = ""
    LOADING = "loading"
    LOADED = "loaded"


@dataclass
class Job:
    """The generation in progress: the clips to make, the export's refine settings (None: a normal run) and whether
    the worker has been sent it."""
    targets: list[str]
    refine: dict | None = None
    sent: bool = False


@dataclass(frozen=True)
class ComfyBinding:
    """A worker bound to a ComfyUI: its address, its folder and whether it cleans up the copies it makes."""
    url: str
    comfy_dir: str
    cleanup: bool


@dataclass(frozen=True)
class RunpodBinding:
    """A worker bound to a RunPod pod (a cloud export): GPU, volume, fallback, SeedVR2 size, no-volume option."""
    gpu: str
    volume: str
    fallback: bool
    models: str
    without_volume: bool


WorkerBinding = ComfyBinding | RunpodBinding


@dataclass
class ComfyService:
    """What this app has loaded in ComfyUI (the model it last used, None: nothing) and whether the worker's next quit
    unloads it. One place for what mw_comfy, mw_generator, mw_ends and mw_llama each used to set by hand (MWC-13)."""
    model: ComfyModel | None = None
    quit_frees: bool = True         # the worker's next quit unloads the clip model

    def holds_clip(self) -> bool:
        """The generator worker holds the clip model."""
        return self.model in CLIP_MODELS

    def holds_image(self) -> bool:
        """An editing window's model (image edit, language model) is loaded in ComfyUI: it has to be unloaded."""
        return self.model in IMAGE_MODELS

    def forget(self) -> None:
        """Nothing of this app is loaded in ComfyUI any more."""
        self.model = None

    def language_model_loaded(self) -> None:
        """The prompt refiner's language model is in ComfyUI now (the image model stays the one to unload)."""
        if self.model != ComfyModel.QWEN:
            self.model = ComfyModel.LLM


@dataclass
class ModelSwitch:
    """ComfyUI changing from the clip model to the model of an editing window (an image edit, a replaced person):
    the clip model is unloaded once, before the first job of the window, and the window says when its own model was
    loaded (MWC-13: three windows had their own copy of this)."""
    free: bool = False         # the clip model is still loaded: the first connection unloads it
    loaded: bool = False       # a model of the window is loaded in ComfyUI now

    def prepare(self, current: ComfyModel | None):
        """(GUI thread, before each job) Does the clip model have to go? Once: after the first job it is gone."""
        self.free = not self.loaded and current in CLIP_MODELS

    def free_clip_model(self, editor):
        """(worker thread, once connected) Unload the clip model if it is still there."""
        if self.free:
            editor.free()
            self.free = False


# ---------------------------------------------------------------- the worker's events ('@@ {json}' lines, MWC-8)
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
