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
