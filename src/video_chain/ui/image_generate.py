"""The background half of the image dialog's Generate: one worker per Generate that makes the results one after
another, writes each one's record and tells the window (progress, result, finished). Edit and Create differ only in
what they ask the editor for; the loop, the file names, the record and the error handling are shared."""
from __future__ import annotations

import itertools
import random as _random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from .workers import run_in_background
from .. import edits
from .. import lora_random
from ..comfy import EDIT_MODELS
from ..triggers import stack_words, with_triggers
from .imageio import upright
from .image_common import SEED_MAX
from .widgets import safe_emit


def _even(n: int) -> int:
    """The size the video codecs and the edit model take: an even number of pixels."""
    return n // 2 * 2


def variation_meta(job, prompt: str, key: str) -> dict:
    """The record keys of a result made from a variation (job: variations.Job; prompt: the template)."""
    return {"template": prompt, "variation": {"values": job.combo, "step": job.step, "index": job.index, "key": key}}


@dataclass
class Plan:
    """What one Generate makes, as the window decided it (read on the GUI thread, then only read by the worker)."""
    edits: Path                                   # the project's edits folder
    connect: Callable                             # connect(progress) -> the editor (called in the worker's thread)
    prompt: str
    quality: str
    n: int                                        # results in all (the progress shows "k of n")
    seeds: list[int]
    runs: list[list[tuple[str, float]]]           # the LoRAs of each image in turn: [(path, strength)]
    run_words: list[list[str]]                    # the trigger words of each of `runs`
    refs: list[str]
    jobs: list | None = None                      # the variations' jobs (None: the plain prompt)
    run: str = field(default_factory=lambda: uuid.uuid4().hex[:12])      # this Generate's results belong together


@dataclass
class EditPlan(Plan):
    model: str = ""
    sources: list[str] = field(default_factory=list)   # the images to edit (a folder: each in turn, with all of it)
    per: int = 1                                  # results per source
    mask: QImage | None = None
    keep_size: bool = False
    out_size: tuple[int, int] | None = None       # another shape: the model draws at this size (keep_size is off)


@dataclass
class RandomRun:
    """Random LoRAs (lora_random): each image the pinned LoRAs and a random choice of the others (at most `most` in
    all), at random strengths; same_seed: every image the window's seed, else a new random one each."""
    pool: list[lora_random.Choice]
    most: int
    same_seed: bool
    words: dict[str, list[str]]                   # the trigger words of each LoRA (read on the GUI thread)
    seed: int | None = None                       # the random generator's seed (None: unpredictable; for the tests)


@dataclass
class CreatePlan(Plan):
    family: str = ""
    model_file: str | None = None
    negative: str = ""
    width: int = 0
    height: int = 0
    strength: float = 0.65
    tags: bool = False
    sampling: dict = field(default_factory=dict)
    random: RandomRun | None = None               # random LoRAs; then n = 0 means "until Stop" (continuous)


class GenerateWorker(QObject):
    """Runs a plan in a thread. progress(k, n, step, steps), result(png, record), finished(error): '' = all done,
    'cancelled' = stopped. It has no parent: the window may close while the thread still emits (the thread's own
    reference keeps the worker alive, and a gone window is simply not told)."""
    progress = Signal(int, int, int, int)       # try k, of n, step i, of steps
    result = Signal(str, object)                # png, meta
    finished = Signal(str)                      # error ('' = all done or cancelled)

    def __init__(self, plan: Plan, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.editor: Any = None                 # (the connected ComfyImageEditor / ComfyImageCreator)
        self.cancelled = False
        self.skipped: list[str] = []            # (a folder's unreadable files)
        self._k = 1                             # the result being made now (the editor's progress belongs to it)

    def start(self) -> None:
        run_in_background(self._run, on_done=lambda _r: None, owner=self)

    def cancel(self) -> None:
        self.cancelled = True
        ed = self.editor
        if ed is not None:
            run_in_background(ed.interrupt, on_done=lambda _r: None)

    # ---- the thread
    def _run(self) -> None:
        p = self.plan
        try:
            p.edits.mkdir(parents=True, exist_ok=True)
            self.editor = p.connect(lambda i, s: self.progress.emit(self._k, p.n, i, s))
            self._make()
            self.finished.emit("")
        except Exception as e:  # noqa: BLE001
            safe_emit(self.finished, "cancelled" if self.cancelled else str(e))
        finally:
            self._cleanup()

    def _make(self) -> None:
        raise NotImplementedError

    def _cleanup(self) -> None:
        pass

    # ---- shared by both kinds
    def _pick(self, i: int):
        """What the i-th image (0-based, within its source) is made with: (variation job, seed, LoRAs, words)."""
        p = self.plan
        b, r = divmod(i, len(p.runs))
        return (p.jobs[b] if p.jobs else None), p.seeds[b], p.runs[r], p.run_words[r]

    def _prompt_of(self, job) -> str:
        return job.prompt if job else self.plan.prompt

    def _finish(self, out: Path, meta: dict, job, seed: int, words: list[str], extra: dict) -> None:
        """The record of a made result: written first (atomically), then the window is told."""
        meta.update({"run": self.plan.run, "prompt": self._prompt_of(job), "trigger_words": words, "seed": seed})
        meta.update(extra)
        edits.write_meta(out.with_suffix(".json"), meta)
        self.result.emit(str(out), meta)

    def _variation(self, job) -> dict:
        return variation_meta(job, self.plan.prompt, job.key) if job else {}


class EditWorker(GenerateWorker):
    """Edit mode: every source image, each `per` times (seeds x LoRA runs)."""
    plan: EditPlan

    def __init__(self, plan: EditPlan, parent=None):
        super().__init__(plan, parent)
        self._copies: list[Any] = []           # [kept source, mask file, results made from it]

    def _make(self) -> None:
        p = self.plan
        if p.model not in self.editor.models():
            raise RuntimeError(f"{EDIT_MODELS[p.model]} is not available in ComfyUI")
        if p.quality == "fast" and "fast" not in self.editor.qualities(p.model):
            self.progress.emit(0, p.n, 0, 0)
        many = len(p.sources) > 1
        for si, src in enumerate(p.sources):
            if self.cancelled:
                break
            size = None
            if many or p.keep_size:
                img = upright(src)                # the size the edit comes back at (a phone photo turned)
                if img.isNull():
                    self.skipped.append(Path(src).name)   # (a folder's unreadable file: the next one)
                    continue
                if p.keep_size and p.out_size is None:
                    size = (_even(img.width()), _even(img.height()))
            kept = edits.keep_source(p.edits, src)           # the original stays with its results
            entry = [kept, None, 0]
            self._copies.append(entry)
            if p.mask is not None:                           # the mask too (white = drawn anew)
                entry[1] = edits.mask_path(kept)
                # (the stubs want bytes as the format, PySide6 6.12 only takes str)
                if not p.mask.save(str(entry[1]), "PNG"):  # type: ignore[call-overload]
                    raise RuntimeError("could not save the mask")
            for j in range(p.per):
                if self.cancelled:
                    break
                k = si * p.per + j + 1
                self._k = k
                self._edit_one(src, kept, entry, j, k, size, many)

    def _edit_one(self, src: str, kept: Path, entry: list, j: int, k: int, size, many: bool) -> None:
        p, mask_file = self.plan, entry[1]
        job, seed, loras, words = self._pick(j)
        out = edits.unique_out(p.edits, "edit", seed, k)
        shape = {"size": p.out_size} if p.out_size else {}
        meta = self.editor.edit(Path(src), with_triggers(self._prompt_of(job), words), out, seed, p.quality,
                                [Path(x) for x in p.refs], loras, model=p.model, mask=mask_file, **shape)
        entry[2] += 1
        if size is not None:
            try:
                edits.scale_to(out, out, size)
            except Exception:
                out.unlink(missing_ok=True)       # (no result without its record)
                raise
        self._finish(out, meta, job, seed, words, {
            "source": src, "source_copy": str(kept), **({"mask": str(mask_file)} if mask_file else {}),
            "refs": p.refs, "size": list(shown) if (shown := size or p.out_size) else None,
            **({"folder": str(Path(src).parent)} if many else {}), **self._variation(job)})

    def _cleanup(self) -> None:
        edits.drop_unused_copies(self._copies)      # no result refers to them: the copies of source and mask go


class CreateWorker(GenerateWorker):
    """Create mode: n new images from the prompt (and reference pictures)."""
    plan: CreatePlan

    def _make(self) -> None:
        p = self.plan
        kept = edits.keep_source(p.edits, p.refs[0]) if p.refs else None   # the first picture is the "original"
        rng = _random.Random(p.random.seed) if p.random else None
        for k in (itertools.count(1) if p.random and p.n == 0 else range(1, p.n + 1)):
            if self.cancelled:
                break
            self._k = k
            job, seed, loras, words = self._random_pick(rng) if rng else self._pick(k - 1)
            out = edits.unique_out(p.edits, "new", seed, k)
            neg = (" ".join(job.negative.split()) if p.negative else "") if job else p.negative
            meta = self.editor.create(p.family, p.model_file, with_triggers(self._prompt_of(job), words), out, seed,
                                      p.quality, p.width, p.height, neg, loras, [Path(r) for r in p.refs],
                                      p.strength, p.tags, p.sampling)
            self._finish(out, meta, job, seed, words, {
                "refs": p.refs, **self._variation(job), "tags": p.tags,
                "source_copy": str(kept) if kept else None, **self._random_meta()})

    def _random_pick(self, rng: _random.Random):
        """(job, seed, LoRAs, words) of the next random-LoRA image."""
        r = self.plan.random
        assert r is not None
        loras = lora_random.pick(r.pool, r.most, rng)
        seed = self.plan.seeds[0] if r.same_seed else rng.randint(1, SEED_MAX)
        return None, seed, loras, stack_words([path for path, _s in loras], lookup=lambda q: r.words.get(q, []))

    def _random_meta(self) -> dict:
        r = self.plan.random
        return {"random_loras": {"pinned": [c.path for c in r.pool if c.pinned], "most": r.most}} if r else {}
