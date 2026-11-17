"""Joining and refining a path into one video."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Protocol

from ..triggers import with_triggers
from .. import camera as _camera
from .. import chain
from .. import h3_prompt as _h3
from .. import joinmatch
from .. import media
from .consts import DEFAULT_OPTIONS
from .consts import REFINED, checkpoint_of
from .model import Node, Tree, short_id
from .timeline import SWAPPED_REFINE_PROMPT, _guide_of, handover_index, made_frames, path_cuts
from .files import load, refined_file
from .starts import prepared_start, start_source
from .sources import source_rotation
from .audio_tracks import sound_file
from .checker import Checker
from .engine import _clip_frames, given_sound, remix_sounds


class RefineGenerator(Protocol):
    """What refine_path needs of a generator (comfy.ComfyGenerator): re-render a clip at a size, or upscale it."""

    def refine(self, draft: Path, prompt: str, out: Path, seed: int, size: tuple[int, int], frames: int,
               loras: list[tuple[str, float]], first_frame: Path | None = None, last_frame: Path | None = None,
               sparse: float | None = None, strength: str = ..., checkpoint: str | None = None) -> None: ...

    def seedvr(self, draft: Path, out: Path, size: tuple[int, int], method: str = ..., seed: int = ...,
               denoise: bool = ...) -> None: ...


def swapped_video(node: Node) -> dict | None:
    """A video node whose person was replaced: what it was replaced with (see swap.replaced), else None."""
    if not node.is_video:
        return None
    from ..swap import replaced
    return replaced(node)


def refine_record(target: Path) -> dict:
    """What a refined copy was made with (its .json; {} if none)."""
    try:
        return json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _refined_sound(project: Path, node: Node, target: Path, tmp: Path, log: Callable[[str], None], who: str) -> None:
    """The sound of a generated clip's refined copy: its mix of both sounds exactly as the clip has it, or its given
    sound cut again from its file (as recorded); without a given sound the refine's own sound stays."""
    s = node.made_dict().get("settings") or {}
    draft = project / node.made_dict()["video"]
    if (s.get("source_sound") or {}).get("mix") and chain.audio_seconds(draft):
        wav = tmp / f"{node.id}_sound.wav"     # the made clip's mix of both sounds, exactly
        chain.to_wav(draft, wav)
        chain.replace_audio(target, wav)
    elif s.get("source_sound") and not sound_file(project, s["source_sound"]).is_file():
        log(f"{who}: the video file of its sound is missing ({Path(s['source_sound']['file']).name}); the refined "
            "copy keeps the model's sound")
    elif s.get("source_sound"):
        ss = s["source_sound"]
        wav = tmp / f"{node.id}_sound.wav"     # the recorded sound, as in the clip
        given_sound(project, node, ss, wav, check=False)
        chain.replace_audio(target, wav)


def _pin_frames(project: Path, tree: Tree, path: list[Node], node: Node, size: str,
                tmp: Path) -> tuple[Path | None, Path | None]:
    """The frames a refined clip is pinned to: its first (the start image, or the frame the refined clip before it
    hands over at) and its last (the first frame of the refined video after it)."""
    from ..comfy import refine_size
    first = last = None
    i = path.index(node)
    if i == 0:
        s0 = node.made_dict().get("settings") or {}
        first = prepared_start(project, *refine_size(int(s0["width"]), int(s0["height"]), size),
                               source=start_source(project, node), crop=s0.get("start_crop")) \
            if s0.get("width") and s0.get("height") else None
    else:
        before = path[i - 1]
        n_before = made_frames(before) or _clip_frames(project, before)
        at = handover_index(tree, before, n_before)
        first = tmp / f"{node.id}_first.png"
        chain.frame_at(refined_file(project, before, size), n_before - 1 if at is None else at, first)
    vc = tree.video_child(node.id)
    if vc is not None and vc in path:
        last = tmp / f"{node.id}_last.png"
        chain.frame_at(refined_file(project, vc, size), 0, last)
    return first, last


def _refine_checkpoint(tree: Tree, node: Node) -> str | None:
    """The H3 checkpoint a node's refine loads: the one a generated clip was made with, for a video with a replaced
    person the project's choice; None: the standard file."""
    if node.is_video:
        ck = checkpoint_of(tree.options)
        return ck["name"] if ck else None
    ck = (node.made_dict().get("settings") or {}).get("checkpoint") or {}
    if not isinstance(ck, dict):
        raise ValueError(f"clip #{short_id(node.id)}: its recorded checkpoint setting is damaged (expected a dict, "
                         f"got {type(ck).__name__}); make the clip again")
    return ck.get("name") or None


def _refine_choices(generator: RefineGenerator, method: str, strength: str, fast_attention: bool,
                    denoise: bool, checkpoint: str | None = None) -> dict:
    """What a re-rendered copy records of how it was made (a copy is reused only while these are the same);
    checkpoint: the H3 checkpoint a refine loads (recorded only when there is one: standard copies stay as they
    were)."""
    pins: dict = {}
    if method != "refine":
        pins["method"] = method
        if denoise:
            pins["denoise"] = True
        variant = getattr(generator, "seedvr_variant", None)   # (RunPod's big GPUs: fp16, whole frames)
        if variant:
            pins["variant"] = variant
    else:
        pins["strength"] = strength
        if fast_attention:
            pins["attention"] = "sol-attn"
        if checkpoint:
            pins["checkpoint"] = checkpoint
    return pins


def _check_recorded_size(node: Node) -> None:
    s = node.made_dict().get("settings") or {}
    if node.is_video and not all(k_ in (node.made_dict().get("key") or {}) for k_ in ("width", "height")) or \
            not node.is_video and not all(k_ in s for k_ in ("width", "height")):
        raise ValueError(f"{'video' if node.is_video else 'clip'} #{short_id(node.id)} has no recorded size; "
                         "make it again, then export")


def _seedvr_clip(generator: RefineGenerator, project: Path, node: Node, target: Path, size: str, method: str,
                 denoise: bool, log: Callable[[str], None], who: str) -> None:
    """The clip as made, restored at the size by SeedVR2."""
    from ..comfy import SEEDVR_METHODS, refine_size
    dims = (node.made_dict().get("key") if node.is_video else node.made_dict().get("settings")) or {}
    w, h = refine_size(int(dims["width"]), int(dims["height"]), size)
    name = "SeedVR2 " + SEEDVR_METHODS[method].upper()
    log(f"{who}: upscaling with {name} to {w}x{h}")
    generator.seedvr(project / node.made_dict()["video"], target, (w, h), method, seed=node.seed, denoise=denoise)


def _refine_swapped(generator: RefineGenerator, project: Path, node: Node, target: Path, size: str, tmp: Path,
                    fast_attention: bool, strength: str, log: Callable[[str], None], who: str,
                    checkpoint: str | None = None) -> None:
    """A video whose person was replaced, re-rendered like a generated clip (with the recorded sound)."""
    from ..comfy import SOL_TAU, refine_size
    key = node.made_dict().get("key") or {}
    w, h = refine_size(int(key["width"]), int(key["height"]), size)
    info = swapped_video(node) or {}
    prompt = info.get("details") or info.get("prompt") or SWAPPED_REFINE_PROMPT
    log(f"{who}: refining the video with the replaced person at {w}x{h}")
    draft, n = project / node.made_dict()["video"], _clip_frames(project, node)
    grid = n if n <= 5 else chain.h3_frames(n)       # H3's 17n+5 frame grid
    padded = tmp / f"{node.id}_draft.mp4"
    if grid != n:                            # the last frame held to the grid, cut off again after
        media.ffmpeg(["-y", "-i", draft, "-vf", f"tpad=stop_mode=clone:stop={grid - n}", "-c:v", "libx264",
                      "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "copy", padded])
    made = tmp / f"{node.id}_refined.mp4" if grid != n else target
    generator.refine(padded if grid != n else draft, prompt, made, int(info.get("seed") or 0), (w, h),
                     grid, [], None, None, sparse=SOL_TAU if fast_attention else None, strength=strength,
                     **({"checkpoint": checkpoint} if checkpoint else {}))
    if grid != n:
        media.ffmpeg(["-y", "-i", made, "-frames:v", str(n), "-c:v", "libx264", "-crf", "12", "-pix_fmt",
                      "yuv420p", "-an", target])
    if chain.audio_seconds(draft):                # the recorded sound exactly, not the model's
        wav = tmp / f"{node.id}_sound.wav"
        chain.to_wav(draft, wav)
        chain.replace_audio(target, wav)


def _convert_video(project: Path, node: Node, target: Path, size: str, log: Callable[[str], None],
                   who: str) -> None:
    """A real video converted again from its file at the size."""
    from ..comfy import refine_size
    src = node.source_dict()
    if not Path(src["file"]).is_file():
        raise FileNotFoundError(f"video file not found: {src['file']}")
    key = node.made_dict().get("key") or {}
    w, h = refine_size(int(key["width"]), int(key["height"]), size)
    log(f"{who}: converting {Path(src['file']).name} at {w}x{h}")
    chain.convert_range(Path(src["file"]), target, src["start"], src["end"], w, h, rotate=source_rotation(node.source))


def _refine_clip(generator: RefineGenerator, project: Path, node: Node, target: Path, size: str, tmp: Path,
                 first: Path | None, last: Path | None, fast_attention: bool, strength: str,
                 log: Callable[[str], None], who: str, checkpoint: str | None = None) -> None:
    """A generated clip re-rendered as it was made (its prompt, style text, LoRAs, seed), with its sound."""
    from ..comfy import SOL_TAU, refine_size
    s = node.made_dict().get("settings") or {}
    w, h = refine_size(int(s["width"]), int(s["height"]), size)
    made_prompt = node.made_dict().get("prompt", node.prompt)
    if s.get("prompt_format") == "h3" and not _h3.is_structured(made_prompt):
        made_prompt = _h3.split_sound_sections(made_prompt)[0]   # (the camera sentence before them)
    made_prompt = _camera.apply(made_prompt, s.get("camera"))
    prompt = " ".join(x for x in (s.get("prompt_prefix", ""), made_prompt, s.get("prompt_suffix", "")) if x)
    prompt = with_triggers(prompt, list(s.get("lora_triggers") or []))   # as the clip was made
    loras = ([(s["lora"], s.get("strength", 1.0))] if s.get("lora") else []) + \
        [(e["lora"], e.get("strength", 1.0)) for e in s.get("extra_loras") or []]
    log(f"{who}: refining at {w}x{h}")
    generator.refine(project / node.made_dict()["video"], prompt, target, node.seed, (w, h),
                     made_frames(node) or _clip_frames(project, node), loras, first, last,
                     sparse=SOL_TAU if fast_attention else None, strength=strength,
                     **({"checkpoint": checkpoint} if checkpoint else {}))
    _refined_sound(project, node, target, tmp, log, who)


def refine_path(generator: RefineGenerator, project: Path, nid: str, size: str, out: Path,
                events: Callable[[dict], None] | None = None, log: Callable[[str], None] = print,
                pin: bool = True, fast_attention: bool = False, refine_swapped: bool = False,
                strength: str = "natural", method: str = "refine", single: bool = False,
                denoise: bool = False) -> Path:
    """Export the path start -> nid at a higher size ('1080p' / '2k'): every generated clip is re-rendered by
    generator.refine (Open-H3-Regenerate-2K) as it was made (its prompt, style text, LoRAs, seed), every real video
    is converted again from its file at that size. Refined copies are kept (nodes/<clip>_<size>.mp4) and reused.
    pin: a refined clip starts exactly at the frame the refined clip before it hands over at (the first one at the
    start image) and ends at the refined first frame of a real video after it, so the joins stay seamless at the
    higher size; a kept copy is reused only while those frames are the same (its .json records them).
    fast_attention: Sol-Attn block-sparse attention for the refine (about 2x faster at 2K, not exact); the
    attention a copy was made with is recorded too, so the other choice makes new copies.
    strength: how much the refine draws anew (comfy.REFINE_STRENGTHS: soft / natural / crisp); recorded with each
    copy, so another strength makes new copies.
    refine_swapped: a real video whose person was replaced (made at the generation size) is re-rendered by the
    refine like a generated clip (prompt: the new person's face details), not only converted at the higher size.
    method: "refine" (above), or "seedvr2_7b" / "seedvr2_3b": every generated clip (and, with refine_swapped, every
    video with a replaced person) is upscaled by generator.seedvr instead, which restores what is there rather than
    drawing it anew (no pins needed: neighbouring clips stay as alike as they were made); recorded per copy too.
    denoise: with SeedVR2, a light denoise of each clip first (comfy.SEEDVR_DENOISE); recorded per copy too.
    single: only the clip nid itself (whole, nothing left out); its kept copy is shared with path exports, except a
    refine without one: that is made unpinned (no neighbours) into a temporary file, the path's copy stays.
    Then the path is joined (the same frames left out as at the normal size) into out."""
    from ..comfy import SEEDVR_METHODS
    if method != "refine" and method not in SEEDVR_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if size not in REFINED:
        raise ValueError(f"unknown size {size!r}")
    project = Path(project)
    emit = events or (lambda _e: None)
    tree = load(project)
    checker = Checker(project, tree)
    if single and nid not in tree.nodes:
        raise ValueError(f"no clip {nid}")
    path = [tree.nodes[nid]] if single else tree.path_to(nid)
    # (the refined copies take the clip's sound)
    if remix_sounds(project, tree, [n.id for n in path], checker, log=log):
        checker = Checker(project, tree)
    missing = [n.id for n in path if checker.status(n.id) != "done"]
    if missing:
        raise ValueError(f"not made yet: {', '.join(missing)}")
    order = [n for n in path if n.is_video] + [n for n in path if not n.is_video]   # videos first: pins
    files: dict[str, Path] = {}
    emit({"e": "plan", "make": [n.id for n in order]})
    made_now = []
    with media.scratch("job", dir=str(project / "nodes")) as scratch:
        tmp = Path(scratch)
        for k, node in enumerate(order, start=1):
            target = refined_file(project, node, size)
            first = last = None
            if pin and not single and not node.is_video and method == "refine":
                first, last = _pin_frames(project, tree, path, node, size, tmp)
            pins = {"first": chain.file_signature(first), "last": chain.file_signature(last)}
            regen = not node.is_video or bool(refine_swapped and swapped_video(node))
            if node.is_video:
                pins["video"] = "refined" if regen else "converted"
            ck_name = _refine_checkpoint(tree, node) if regen and method == "refine" else None
            if regen:
                pins.update(_refine_choices(generator, method, strength, fast_attention, denoise, ck_name))
            record = refine_record(target)
            sound_rev = None if node.is_video else node.made_dict().get("sound_rev")   # (its sound mixed again since)
            copy_rev = record.pop("sound", None)
            if node.is_video and not regen and record.get("video", "converted") == "converted":
                reuse = target.is_file()                 # (copies converted before the record named it)
            else:
                reuse = target.is_file() and record == pins
            if single and regen and method == "refine" and not reuse:
                loose = {k_: v for k_, v in pins.items() if k_ not in ("first", "last")}
                if target.is_file() and {k_: v for k_, v in record.items() if k_ not in ("first", "last")} == loose:
                    reuse = True                         # (a copy pinned to its path: the same clip refined)
                else:
                    target = tmp / f"{node.id}_{size}_single.mp4"   # unpinned: not kept, the path's copy stays
            files[node.id] = target
            if reuse and (copy_rev or None) != (sound_rev or None) and target.parent != tmp:
                log(f"[{k}/{len(order)}] {node.id}: its sound was mixed again; the refined copy gets the new sound")
                _refined_sound(project, node, target, tmp, log, f"[{k}/{len(order)}] {node.id}")
                chain.atomic_write_text(target.with_suffix(".json"), json.dumps({**pins, "sound": sound_rev}))
            if reuse:
                emit({"e": "node", "id": node.id, "state": "reused"})
                continue
            emit({"e": "node", "id": node.id, "state": "generating", "k": k, "n": len(order)})
            target.with_suffix(".json").unlink(missing_ok=True)   # a copy made now must not keep older pins
            _check_recorded_size(node)
            who = f"[{k}/{len(order)}] {node.id}"
            if regen and method != "refine":             # SeedVR2: the clip as made, restored at the size
                _seedvr_clip(generator, project, node, target, size, method, denoise, log, who)
            elif node.is_video and regen:                # a replaced person: re-rendered like a generated clip
                _refine_swapped(generator, project, node, target, size, tmp, fast_attention, strength, log, who,
                                ck_name)
            elif node.is_video:
                _convert_video(project, node, target, size, log, who)
            else:
                _refine_clip(generator, project, node, target, size, tmp, first, last, fast_attention, strength, log,
                             who, ck_name)
            if sound_rev:
                pins["sound"] = sound_rev
            chain.atomic_write_text(target.with_suffix(".json"), json.dumps(pins))
            made_now.append(node.id)
            emit({"e": "node", "id": node.id, "state": "done"})
        videos = [files.get(n.id) or refined_file(project, n, size) for n in path]
        cuts = [(0, 0)] if single else path_cuts(tree, path, bool(tree.options.get("keep_repeated")))
        smooth = None
        if not single:
            cuts, smooth = smooth_cuts(project, tree, path, videos, cuts)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        emit({"e": "joining"})
        chain.concat(videos, out, cuts=cuts, smooth=smooth)   # (inside: a temporary single copy is still there)
    emit({"e": "exported", "file": str(out)})
    emit({"e": "done", "made": made_now})
    return out


logger = logging.getLogger(__name__)

JOINS_FILE = "joins.json"          # paths/: the kept measures of smooth joins
JOINS_KEPT = 500                   # measures kept at most (the oldest go)


def _join_key(a: Path, a_end: int, b: Path, guide: int) -> str | None:
    try:
        return json.dumps([str(a), a.stat().st_mtime_ns, int(a_end), str(b), b.stat().st_mtime_ns, int(guide),
                           joinmatch.VERSION])
    except OSError:
        return None


def smooth_cuts(project: Path, tree: Tree, path: list[Node], videos: list[Path], cuts: list[tuple[int, int]],
                measure: bool = True) -> tuple[list[tuple[int, int]], list[dict | None]]:
    """The cuts and chain.concat's smooth of a path's join (option "smooth_joins"): every generated clip after
    another clip or video is measured against it (joinmatch.measure; kept in paths/joins.json for these files),
    its start cut moved by the measured offset, with the colour correction and blend. A join that cannot be
    measured (another picture, an unreadable file) stays as it is. measure=False: only the kept measures (no
    ffmpeg). Off, or with the repeated frames kept: the cuts as given and no smooth."""
    cuts = list(cuts)
    smooth: list[dict | None] = [None] * len(path)
    o = tree.options
    if not o.get("smooth_joins", DEFAULT_OPTIONS["smooth_joins"]) or o.get("keep_repeated") or len(path) < 2:
        return cuts, smooth
    file = Path(project) / "paths" / JOINS_FILE
    try:
        kept = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
        kept = kept if isinstance(kept, dict) else {}
    except (OSError, ValueError):
        kept = {}
    changed = False
    for k in range(1, len(path)):
        b = path[k]
        if b.is_video or not b.made:
            continue
        guide = _guide_of(b)
        key = _join_key(Path(videos[k - 1]), cuts[k - 1][1], Path(videos[k]), guide)
        if key is None:
            continue
        m = kept.get(key, False)
        if m is False and measure:
            try:
                stop = chain.probe(videos[k - 1])["frames"] - cuts[k - 1][1]
                m = joinmatch.measure(videos[k - 1], stop, videos[k], guide)
            except (OSError, RuntimeError, ValueError) as e:      # (media errors too: a join is never refused)
                logger.warning("join before clip %s not measured: %s", b.id, e)
                continue
            kept[key] = m
            changed = True
        if not m:
            continue
        start, end = cuts[k]
        frames = chain.probe(videos[k])["frames"] if measure else (made_frames(b) or 0)
        new = max(1, start + int(m.get("offset") or 0))
        if frames and new + end >= frames:
            continue
        cuts[k] = (new, end)
        smooth[k] = {"blend": int(m.get("blend") or 0), "lut": joinmatch.lut(m.get("planes") or [])}
    if changed:
        try:
            while len(kept) > JOINS_KEPT:
                kept.pop(next(iter(kept)))
            file.parent.mkdir(parents=True, exist_ok=True)
            chain.atomic_write_text(file, json.dumps(kept))
        except OSError as e:
            logger.warning("the join measures could not be kept: %s", e)
    return cuts, smooth


def join_cuts(project: Path, tree: Tree, nid: str) -> list[tuple[int, int]]:
    """The cuts the joined path to nid has (the kept measures of its smooth joins; no ffmpeg), for path_spans."""
    path = tree.path_to(nid)
    cuts = path_cuts(tree, path, bool(tree.options.get("keep_repeated")))
    videos = [Path(project) / n.made_dict()["video"] for n in path]
    return smooth_cuts(project, tree, path, videos, cuts, measure=False)[0]


def join_path(project: Path, tree: Tree, nid: str, out: Path | None = None, persist: bool = False,
              remix: bool = True, remixed: list[str] | None = None) -> Path:
    """Join the clips from the start to nid into one video (paths/<id>.mp4 unless out is given).
    Every clip on the path must be done. A clip whose sound is due to be mixed again is remixed first (in tree;
    persist: also saved; a window saves its own tree, so by default a remix not saved is simply done again later;
    remix=False: not at all, for a caller that did it itself; remixed: a list the ids remixed are appended to, so
    the caller knows to save the tree).
    A kept join is reused while every clip file is unchanged (by date: a remixed clip is joined again)."""
    project = Path(project)
    path = tree.path_to(nid)
    if remix:
        done = remix_sounds(project, tree, [n.id for n in path], persist=persist)
        if remixed is not None:
            remixed.extend(done)
    checker = Checker(project, tree)
    missing = [n.id for n in path if checker.status(n.id) != "done"]
    if missing:
        raise ValueError(f"not made yet: {', '.join(missing)}")
    videos = [project / n.made_dict()["video"] for n in path]
    cuts, smooth = smooth_cuts(project, tree, path, videos, path_cuts(tree, path,
                                                                     bool(tree.options.get("keep_repeated"))))
    target = Path(out) if out else project / "paths" / f"{nid}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    size = None                    # frozen clips of an older size: every clip scaled to the size clips have now
    if any(n.frozen for n in path):
        base = checker.base()
        sizes = {(i["width"], i["height"]) for i in (chain.probe_source(v) for v in videos)}
        if base is not None and len(sizes) > 1:
            size = (int(base["width"]), int(base["height"]))
    key = json.dumps([[str(v), v.stat().st_mtime_ns] for v in videos] + [[list(c) for c in cuts]] +
                     ([list(size)] if size else []) + ([smooth] if any(smooth) else []))
    stamp = target.with_suffix(".json")
    if out is None and target.exists() and stamp.exists() and stamp.read_text() == key:
        return target
    def write(f: Path) -> None:
        chain.concat(videos, f, cuts=cuts, size=size, fill=True, smooth=smooth)
    if out is None:                                # the join before is never deleted: it goes to old/
        chain.replace_kept(project, target, write)
        chain.atomic_write_text(stamp, key)
    else:
        write(target)
    return target
