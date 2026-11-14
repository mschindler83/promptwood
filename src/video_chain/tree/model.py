"""The data model: a clip (Node) and the tree of clips (Tree), with their JSON form."""
from __future__ import annotations

import copy
import random
import re
from typing import Any, cast
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .. import chain
from .guard import retire
from .schema import Keyframe, MadeRecord, Options, Overrides, Source
from .consts import (
    AI,
    ENDS_DIR,
    DEFAULT_OPTIONS,
    NewerTreeError,
    OVERRIDE_KEYS,
    PROMPT_FORMATS,
    SEED_MAX,
    STARTS_DIR,
    TREE_VERSION,
    VIDEO,
    _OLD_LORA_KEYS)


@dataclass
class Node:
    id: str
    parent: str | None             # None: starts from the start image
    prompt: str = ""
    seed: int = 42
    made: MadeRecord | None = None       # what the current files were made from (prompt, seed, settings, files)
    overrides: Overrides = field(default_factory=lambda: Overrides())   # loras [{"path", "strength"}], frames, steps, guide
    kind: str = AI                 # AI: generated clip; VIDEO: a range of a real video file
    source: Source | None = None     # VIDEO: {"file": absolute path, "start": s, "end": s}
    end_image: str | None = None   # AI: its own end image (ends/..., project-relative); a video after it wins
    end_at: int | None = None      # AI: hand over at this frame instead of the last one (see handover_index)
    keyframes: list[Keyframe] = field(default_factory=list)   # AI: [{"image": ends/..., "at": s, "note": str}], by time
    frozen: bool = False           # kept as made: done whatever the settings say (see freeze)
    start_image: str | None = None  # AI, a first clip: its own start picture (starts/..., project-relative)
    #                                 instead of the project's start image (see add_start_rows)
    row: str | None = None         # the first clips made as one set for several start pictures share this id
    extra: dict[str, Any] = field(default_factory=dict)   # keys this version does not know: kept, written back

    @property
    def is_video(self) -> bool:
        return self.kind == VIDEO

    def made_dict(self) -> dict[str, Any]:
        """The made record as a plain dict ({} for a clip not made yet), for reading it by a key that is a variable
        or a name from outside (a TypedDict allows only literal keys)."""
        return cast(dict[str, Any], self.made or {})

    def source_dict(self) -> dict[str, Any]:
        """The source of a real video as a plain dict ({} for a clip)."""
        return cast(dict[str, Any], self.source or {})


def short_id(nid: str) -> str:
    """The number shown for a clip id ("n3" -> "3", as in "clip #3"): the id without its one prefix letter."""
    return str(nid)[1:]


@dataclass
class Tree:
    nodes: dict[str, Node] = field(default_factory=dict)
    options: Options = field(default_factory=lambda: copy.deepcopy(DEFAULT_OPTIONS))
    image: str | None = None       # where the start image came from (start.png is the copy that counts)
    selected: str | None = None
    next_num: int = 1
    recovered_from: Path | None = field(default=None, compare=False, repr=False)   # set by load(): the backup used
    extra: dict[str, Any] = field(default_factory=dict)   # top-level keys this version does not know: kept
    _removed: list[tuple[str, Node]] = field(default_factory=list, compare=False, repr=False)   # see remove(later=)

    # ---- structure
    def children(self, nid: str | None) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent == nid]

    def path_to(self, nid: str | None) -> list[Node]:
        out = []
        while nid is not None:
            n = self.nodes[nid]
            out.append(n)
            nid = n.parent
        return out[::-1]

    def depth(self, nid: str) -> int:
        return len(self.path_to(nid))

    def leaves(self) -> list[Node]:
        parents = {n.parent for n in self.nodes.values()}
        return [n for n in self.nodes.values() if n.id not in parents]

    def descendants(self, nid: str) -> list[Node]:
        out, todo = [], [nid]
        while todo:
            cur = todo.pop()
            for c in self.children(cur):
                out.append(c)
                todo.append(c.id)
        return out

    def add(self, parent: str | None, prompt: str = "", seed: int | None = None) -> Node:
        if parent is not None and parent not in self.nodes:
            raise KeyError(parent)
        nid = f"n{self.next_num}"
        self.next_num += 1
        if seed is None:
            seed = self.new_seed()
        node = Node(nid, parent, prompt, int(seed))
        self.nodes[nid] = node
        return node

    def new_seed(self) -> int:
        """A random seed no clip of the tree has."""
        used = {n.seed for n in self.nodes.values()}
        while True:
            s = random.randint(1, SEED_MAX)
            if s not in used:
                return s

    def video_child(self, nid: str | None) -> Node | None:
        """The video a generated clip leads into (its end frame), or None. Only generated clips have one."""
        if nid is None or nid not in self.nodes or self.nodes[nid].is_video:
            return None
        vids = sorted((c for c in self.children(nid) if c.is_video), key=lambda c: self.number(c.id))
        return vids[0] if vids else None

    def can_add_video(self, parent: str | None) -> bool:
        return self.video_child(parent) is None

    def add_video(self, parent: str | None, file: str, start: float, end: float, rotate: int = 0) -> Node:
        """A real video (file, start..end seconds, turned clockwise by rotate degrees) after parent. A generated clip
        can lead into one video only."""
        if parent is not None and not self.can_add_video(parent):      # (a first video after the start image: always)
            raise ValueError(f"clip #{short_id(parent)} already leads into a video")
        if float(end) <= float(start):
            raise ValueError("the end of the range must be after its start")
        from .sources import video_source               # (the modules below this one import this module)
        node = self.add(parent, "")
        node.kind = VIDEO
        node.source = video_source(file, start, end, rotate)
        return node

    def split_video(self, nid: str, at: float) -> tuple[Node, Node]:
        """Insert a generated clip into a real video at `at` seconds (of its file): the video ends there, a new empty
        clip follows it, then the rest of the same video (at .. its old end), whose first frame the clip ends at.
        Clips that followed the video follow its rest now. A video whose person was replaced keeps that in both
        parts (its original range split at the same moment). Returns (the new clip, the rest)."""
        v = self.nodes[nid]
        if not v.is_video:
            raise ValueError(f"#{short_id(nid)} is not a real video")
        assert v.source is not None                   # (a real video has its source)
        start, end = float(v.source["start"]), float(v.source["end"])
        at = round(float(at), 4)
        if not start + 1 / chain.FPS <= at <= end - 1 / chain.FPS:
            raise ValueError("split inside the video's range (not at its first or last frame)")
        kids = self.children(nid)
        rest_source: Source = {**v.source, "start": at, "end": round(end, 4)}
        v.source = {**v.source, "end": at}
        rep = v.source.get("replaced")
        if isinstance(rep, dict) and isinstance(rep.get("original"), dict):   # the original's range, split too
            o = rep["original"]
            cut = round(float(o["start"]) + (at - start), 4)
            v.source["replaced"] = {**rep, "original": {**o, "end": cut}}
            rest_source["replaced"] = {**rep, "original": {**o, "start": cut}}
        clip = self.add(nid, "")
        rest = self.add_video(clip.id, rest_source["file"], at, end)
        rest.source = rest_source
        for k in kids:
            k.parent = rest.id
        return clip, rest

    def remove(self, nid: str, project: Path | None = None, later: bool = False) -> list[str]:
        """Remove a node and everything below it. Their clips and pictures are not deleted: they are moved to
        old/ in the project (chain.keep_aside). Returns the removed ids. later: the files move when the tree is
        saved next (save() calls retire_removed), so a crash in between never leaves tree.json listing clips
        whose files are already in old/."""
        from .freezing import heal_frozen
        gone = [nid] + [n.id for n in self.descendants(nid)]
        parent = self.nodes[nid].parent
        for g in gone:
            node = self.nodes.pop(g)
            if project is not None:
                if later:
                    self._removed.append((g, node))
                else:
                    _retire_node(project, g, node)
        if self.selected in gone:
            self.selected = parent
        heal_frozen(self)
        return gone

    def retire_removed(self, project: Path) -> None:
        """Move the files of the nodes removed with later=True to old/ (after tree.json was saved)."""
        done, self._removed = self._removed, []
        for g, node in done:
            _retire_node(project, g, node)

    def siblings(self, nid: str) -> list[Node]:
        """The other clips that start from the same point as nid."""
        parent = self.nodes[nid].parent
        return [n for n in self.children(parent) if n.id != nid]

    def others_than_path(self, nid: str) -> list[str]:
        """Root ids of every branch that is not on the path start -> nid (the siblings along the path)."""
        return [s.id for n in self.path_to(nid) for s in self.siblings(n.id)]

    def number(self, nid: str) -> int:
        s = short_id(nid)
        return int(s) if s.isdigit() else 0

    # ---- files
    def to_json(self) -> dict:
        return {**self.extra, "tool": "video-chain", "version": TREE_VERSION, "image": self.image,
                "options": self.options, "selected": self.selected, "next_num": self.next_num,
                "nodes": [_node_to_json(n) for n in sorted(self.nodes.values(), key=lambda n: self.number(n.id))]}

    @classmethod
    def from_json(cls, d: dict) -> Tree:
        """Raises ValueError for a tree that cannot be right (duplicate ids, missing parents, cycles)."""
        from .cast import _migrate_cast
        from .files import _merged_options
        from .freezing import heal_frozen
        if not isinstance(d, dict):
            raise ValueError("not a tree file")
        if isinstance(d.get("version"), int) and d["version"] > TREE_VERSION:
            raise NewerTreeError(f"this project was saved by a newer version of the app (file version "
                                 f"{d['version']}, this one reads up to {TREE_VERSION}); saving would drop what it "
                                 "added")
        stored = copy.deepcopy(d.get("options") or {})
        stored.pop("backend", None)                       # ComfyUI is the only generator now
        project_loras = {k: stored.pop(k) for k in _OLD_LORA_KEYS if k in stored}   # older: project LoRAs
        t = cls(options=_merged_options(stored),
                image=d.get("image"),
                selected=d.get("selected"), next_num=int(d.get("next_num") or 1),
                extra={k: copy.deepcopy(v) for k, v in d.items() if k not in _TREE_KEYS})
        if stored.get("prompt_format") not in PROMPT_FORMATS:
            t.options["prompt_format"] = "classic"   # a project from before the setting: as it was made
        _migrate_cast(t.options, "cast" in stored)
        for nd in copy.deepcopy(d.get("nodes", [])):        # (a copy: the caller's dict is not changed)
            n = _node_from_json(nd, project_loras, t.nodes)
            t.nodes[n.id] = n
        _check_graph(t.nodes)
        t.next_num = max([t.next_num] + [t.number(i) + 1 for i in t.nodes])
        heal_frozen(t)
        if t.selected not in t.nodes:
            t.selected = None
        return t


def _retire_node(project: Path, g: str, node: Node) -> None:
    """A removed node's files (never deleted: moved to old/)."""
    from .files import _unlink, clip_files, node_files
    from .keyframes import keyframe_files
    from .starts import _drop_start_file
    for f in list(node_files(project, g)) + clip_files(project, node):
        chain.keep_aside(project, f)
    for rel in {node.end_image, (node.made or {}).get("end_image")} | keyframe_files(node):
        if rel is not None:
            retire(project, ENDS_DIR, rel, resolved=False)
    _drop_start_file(project, node.start_image)
    chain.keep_aside(project, Path(project) / "paths" / f"{g}.mp4")     # its joined path too
    _unlink(Path(project) / "paths" / f"{g}.json")


_NODE_KEYS = frozenset(f.name for f in fields(Node)) - {"extra"}
_TREE_KEYS = frozenset({"tool", "version", "image", "options", "selected", "next_num", "nodes"})


def _node_to_json(n: Node) -> dict:
    d = asdict(n)
    extra = d.pop("extra")
    return {**extra, **d}


def _node_from_json(nd: object, project_loras: dict, taken: dict[str, Node]) -> Node:
    """One node of tree.json, checked and cleaned (a value that does not fit is dropped or refused). taken: the
    nodes read before it (an id appears once)."""
    from .files import _int_or, migrate_loras
    from .keyframes import clean_keyframes
    if not isinstance(nd, dict) or not nd.get("id"):
        raise ValueError("a node without id")
    if not re.fullmatch(r"n\d+", str(nd["id"])) or (
            nd.get("parent") is not None and not re.fullmatch(r"n\d+", str(nd["parent"]))):
        raise ValueError(f"node {str(nd['id'])[:20]!r} has an invalid id")       # ids become file names
    if nd.get("made") is not None and not isinstance(nd.get("made"), dict):
        raise ValueError(f"node {nd['id']} has a damaged record")
    if nd["id"] in taken:
        raise ValueError(f"node {nd['id']} appears twice")
    kind = nd.get("kind") or AI
    raw = dict(nd.get("overrides") or {})
    if kind == AI:
        raw = migrate_loras(project_loras, raw)
        made = nd.get("made")
        if isinstance(made, dict) and isinstance(made.get("overrides"), dict):   # what Undo goes back to
            made["overrides"] = {k: v for k, v in migrate_loras(project_loras, made["overrides"]).items()
                                 if k in OVERRIDE_KEYS and v is not None}
    ov: Any = {k: v for k, v in raw.items() if k in OVERRIDE_KEYS and v is not None}
    src: Any = nd.get("source")
    if kind not in (AI, VIDEO):
        raise ValueError(f"node {nd['id']} has an unknown kind {kind!r}")
    if kind == VIDEO and not (isinstance(src, dict) and src.get("file") and "start" in src and "end" in src):
        raise ValueError(f"video node {nd['id']} has no source")
    end = nd.get("end_image") if kind == AI and isinstance(nd.get("end_image"), str) else None
    at = nd.get("end_at")
    at = int(at) if kind == AI and isinstance(at, (int, float)) and not isinstance(at, bool) and at >= 1 \
        else None
    own = nd.get("start_image")
    own = own if kind == AI and nd.get("parent") is None and isinstance(own, str) \
        and own.startswith(STARTS_DIR + "/") else None
    row = nd.get("row") if isinstance(nd.get("row"), str) and re.fullmatch(r"n\d+", nd["row"]) else None
    n = Node(nd["id"], nd.get("parent"), nd.get("prompt", ""), _int_or(nd.get("seed"), 42), nd.get("made"), ov,
             kind, cast(Source, dict(src)) if kind == VIDEO else None, end, at,
             clean_keyframes(nd.get("keyframes")) if kind == AI else [],
             bool(nd.get("frozen")) and isinstance(nd.get("made"), dict), own,
             row if kind == AI and nd.get("parent") is None else None,
             {k: v for k, v in nd.items() if k not in _NODE_KEYS})
    return n


def _check_graph(nodes: dict[str, Node]) -> None:
    """Every parent is there and every chain of parents ends at the start image."""
    for n in nodes.values():
        if n.parent is not None and n.parent not in nodes:
            raise ValueError(f"node {n.id} has a missing parent {n.parent}")
    for n in nodes.values():                 # every chain of parents must end at the start image
        seen: set[str] = set()
        cur: str | None = n.id
        while cur is not None:
            if cur in seen:
                raise ValueError(f"node {n.id} is part of a loop")
            seen.add(cur)
            cur = nodes[cur].parent
