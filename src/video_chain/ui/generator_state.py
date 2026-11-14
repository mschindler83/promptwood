"""What the window knows about the generator worker, as types instead of strings and tuples: the model ComfyUI
holds, the phase of a run, the job in progress, and what a running worker is bound to (MWC-6, MWC-7)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
