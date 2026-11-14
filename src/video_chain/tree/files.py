"""A project's files: which belong to which clip, load/save of tree.json (backup, lock), LoRA migration."""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:                                  # (Windows)
    fcntl = None

from .. import chain
from .consts import (
    CLIP_KEYS,
    DEFAULT_OPTIONS,
    NO_LORA,
    NewerTreeError,
    REFINED,
    TREE_FILE,
    _CLIP_NAME,
    _OLD_LORA_KEYS)
from .guard import inside
from .model import Node, Tree
from .timeline import handover_file

logger = logging.getLogger(__name__)


def handover_frames(video: Path) -> list[Path]:
    """The hand-over frames written for a clip version (<stem>_f<index>.png; not a video's <stem>_first.png)."""
    video = Path(video)
    if not video.parent.is_dir():
        return []
    pat = re.compile(re.escape(video.stem) + r"_f\d+\.png$")
    return [f for f in video.parent.glob(f"{video.stem}_f*.png") if pat.match(f.name)]


def _unlink(f: Path) -> bool:
    """Delete a file; False if that is not possible right now (e.g. open in a player on an SMB share)."""
    try:
        f.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def clip_files(project: Path, node: "Node") -> list[Path]:
    """The files a node's current record points to."""
    if not node.made:
        return []
    out = []
    for k in CLIP_KEYS:
        if node.made.get(k):
            f = Path(project) / node.made[k]
            if inside(project, "nodes", f):           # a damaged/hand-edited record never reaches elsewhere
                out.append(f)
    if node.made.get("video"):                    # hand-over frames and refined copies of this version
        v = Path(project) / node.made["video"]
        if inside(project, "nodes", v) and v.parent.is_dir():
            out += sorted(handover_frames(v))
            out += [r for size in REFINED for r in (refined_file(project, node, size),
                                                     refined_file(project, node, size).with_suffix(".json"))
                    if r.is_file()]
    return out


def new_clip_files(project: Path, nid: str) -> tuple[Path, Path]:
    """Fresh, unique names for a node's next clip and last frame. A clip is never written over an existing file:
    on SMB/CIFS shares a file that is open anywhere cannot be replaced."""
    tag = uuid.uuid4().hex[:8]
    d = Path(project) / "nodes"
    return d / f"{nid}_{tag}.mp4", d / f"{nid}_{tag}.png"


def remove_stale_files(project: Path, tree: "Tree") -> list[Path]:
    """Clip files of this app that no record points to any more: finished ones (old versions that could not be
    moved while they were open) go to old/ (chain.keep_aside), half-written ones of a stopped run are deleted.
    Returns what could not be moved."""
    d = Path(project) / "nodes"
    if not d.is_dir():
        return []
    used = {Path(project) / n.made[k] for n in tree.nodes.values() if n.made for k in CLIP_KEYS if n.made.get(k)}
    used |= {h for n in tree.nodes.values() if (h := handover_file(project, tree, n)) is not None}
    used |= {f for n in tree.nodes.values() if n.made and n.made.get("video") for size in REFINED
             for f in (refined_file(project, n, size), refined_file(project, n, size).with_suffix(".json"))}
    stuck = []
    for f in d.iterdir():
        if not f.is_file() or f in used:
            continue
        if ".new." in f.name or ".part." in f.name:
            _unlink(f)                              # (never finished)
        elif _CLIP_NAME.match(f.name) and not chain.keep_aside(project, f):
            stuck.append(f)
    return stuck


def node_files(project: Path, nid: str) -> tuple[Path, Path]:
    d = Path(project) / "nodes"
    return d / f"{nid}.mp4", d / f"{nid}.png"


def load(project: Path) -> Tree:
    """tree.json of a project; a linear chain.json from before branching is turned into a one-branch tree."""
    project = Path(project)
    p = project / TREE_FILE
    if p.exists():
        try:
            return Tree.from_json(json.loads(p.read_text(encoding="utf-8")))
        except NewerTreeError:
            raise
        except Exception as e:  # noqa: BLE001  (any damage: the backup, else one clear message)
            bak = project / (TREE_FILE + ".bak")
            try:
                t = Tree.from_json(json.loads(bak.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                raise ValueError(f"{p} is damaged ({type(e).__name__}: {e})") from e
            t.recovered_from = bak                  # the caller tells the user; the next save keeps the damaged file
            return t
    old = chain.load_record(project)
    t = Tree()
    if old:
        t.options["prompt_format"] = "classic"      # made before the setting existed
        t.image = old.get("image")
        s = dict(old.get("settings") or {})
        s.setdefault("backend", "ai-toolkit")        # linear chains were always made with ai-toolkit
        for key in ("model", "lora", "strength", "frames", "steps", "audio"):
            if key in s:
                t.options[key] = s[key]
        parent = None
        project_loras = {k: t.options.pop(k) for k in _OLD_LORA_KEYS if k in t.options}
        for c in old.get("clips", []):
            n = t.add(parent, c.get("prompt", ""), c.get("seed"))
            n.overrides = migrate_loras(project_loras, {})
            video, frame = project / c.get("video", ""), project / c.get("last_frame", "")
            if video.is_file() and frame.is_file():
                n.made = {"prompt": n.prompt, "seed": n.seed, "settings": s, "start_sha1": c.get("start_sha1"),
                          "video": os.path.relpath(video, project), "frame": os.path.relpath(frame, project)}
            parent = n.id
        t.selected = parent
    return t


def _merged_options(stored: dict) -> dict:
    """The defaults with the stored options on top; a stored sub-dict (turbo, style ...) is merged key by key, so a
    file saved before a sub-key existed still gets its default."""
    out = copy.deepcopy(DEFAULT_OPTIONS)
    for k, v in stored.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "cast":
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _int_or(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _parses(p: Path) -> bool:
    try:
        Tree.from_json(json.loads(p.read_text(encoding="utf-8")))
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("%s does not parse as a tree: %s", p, e)
        return False


def _sync_dir(d: Path) -> None:
    """The rename itself on disk too (power cut); not possible everywhere (network shares), then skipped."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as e:
        logger.debug("fsync of the folder failed: %s", e)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _write_lock(project: Path):
    """One writer at a time, across threads and processes (window, worker, command line): an advisory lock on
    tree.json.lock. Without fcntl (or a share that refuses locks) the unique temp name still keeps the file whole."""
    f = None
    try:
        f = open(project / (TREE_FILE + ".lock"), "a")
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except OSError as e:
        logger.warning("project lock not taken: %s", e)
    try:
        yield
    finally:
        if f is not None:
            f.close()                                # (closing releases the lock)


def save(project: Path, tree: Tree) -> None:
    """Writes tree.json whole or not at all. The file before stays as tree.json.bak (load() falls back to it); a
    damaged tree.json is kept as tree.json.damaged instead of replacing the good backup."""
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    final = project / TREE_FILE
    text = json.dumps(tree.to_json(), indent=2, ensure_ascii=False)
    with _write_lock(project):
        fd, name = tempfile.mkstemp(prefix=TREE_FILE + ".", suffix=".part", dir=project)   # (a name of its own)
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())               # on disk before it replaces the old file (power cut)
            if final.is_file():
                if _parses(final):
                    bak = project / (TREE_FILE + ".bak")
                    shutil.copyfile(final, tmp.with_name(tmp.name + ".bak"))
                    tmp.with_name(tmp.name + ".bak").replace(bak)
                else:
                    shutil.copyfile(final, project / (TREE_FILE + ".damaged"))
            tmp.replace(final)
            _sync_dir(project)
        finally:
            tmp.unlink(missing_ok=True)
            tmp.with_name(tmp.name + ".bak").unlink(missing_ok=True)


def _old_record(s: dict | None) -> dict | None:
    """Records written before the backend setting existed were made with ai-toolkit (the generator of the first
    versions). Clips it made sample differently, so they count as changed: ComfyUI makes them again."""
    if isinstance(s, dict) and "backend" not in s:
        return {**s, "backend": "ai-toolkit"}
    return s


def _extras(items) -> list[tuple[str, float]]:
    return [(e["path"], float(e.get("strength", 1.0))) for e in (items or []) if e.get("path")]


def migrate_loras(project: dict, ov: dict) -> dict:
    """Older trees had a project LoRA (+ strength, extra LoRAs) that clips used unless they had their own; now
    every clip has its own stack. The stack is what the clip was made with, so its clips stay done."""
    if not any(k in ov for k in _OLD_LORA_KEYS) and not project.get("lora") and not project.get("extra_loras"):
        return {k: v for k, v in ov.items() if k not in _OLD_LORA_KEYS}
    if "loras" in ov:
        return {k: v for k, v in ov.items() if k not in _OLD_LORA_KEYS}
    if "lora" in ov:
        first = None if ov["lora"] == NO_LORA else ov["lora"]
        extras = _extras(ov.get("extra_loras"))
    else:
        first = project.get("lora")
        extras = _extras(project.get("extra_loras")) + _extras(ov.get("extra_loras"))
    strength = float(ov.get("strength", project.get("strength", 1.0)))
    stack = ([{"path": first, "strength": strength}] if first else []) + \
        [{"path": p, "strength": st} for p, st in extras]
    out = {k: v for k, v in ov.items() if k not in _OLD_LORA_KEYS}
    if stack:
        out["loras"] = stack
    return out


def node_loras(node: Node) -> list[tuple[str, float]]:
    """A clip's LoRA stack as (path, strength), in order."""
    return _extras((node.overrides or {}).get("loras"))


def refined_file(project: Path, node: Node, size: str) -> Path:
    """The refined copy of a node's current clip at an export size (named after its clip version)."""
    v = Path(node.made["video"])
    return Path(project) / v.parent / f"{v.stem}_{size}.mp4"
