"""Branching chains: a tree of clips. Every clip (node) starts from the last frame of its parent; clips directly
under the start image start from that image. A path from the start to a node is one video.

Project folder:
  tree.json            nodes (id, parent, prompt, seed, what was made), options, the selected node
  start.png            the start image (copied in)
  nodes/<id>.mp4       a node's clip
  nodes/<id>.png       its last frame, the start image of its children
  paths/<id>.mp4       a joined path (start -> node), made on demand

Every node can override the project's LoRA (another one or none), LoRA strength, clip length and steps
(Node.overrides); the video size and sound stay per project, so the clips of a path fit together.

A node counts as done when its clip exists and was made from exactly its current prompt, seed, settings and start
frame (the parent's current last frame). If any of that changed, the node and everything below it is "changed"
and is made again when a path through it is generated. Other branches are not touched.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import chain

TREE_FILE = "tree.json"
START_FILE = "start.png"
NO_LORA = "none"                   # override value: this node without any LoRA
OVERRIDE_KEYS = ("lora", "strength", "frames", "steps")
DEFAULT_OPTIONS = {"model": "Comfy-Org/MiniMax-H3", "lora": None, "strength": 1.0, "frames": 73, "steps": 28,
                   "resolution": 512, "audio": True, "keep_repeated": False, "seed": 42}


@dataclass
class Node:
    id: str
    parent: str | None             # None: starts from the start image
    prompt: str = ""
    seed: int = 42
    made: dict | None = None       # what the current files were made from (prompt, seed, settings, start_sha1, files)
    overrides: dict = field(default_factory=dict)   # lora (path or NO_LORA), strength, frames, steps


@dataclass
class Tree:
    nodes: dict[str, Node] = field(default_factory=dict)
    options: dict = field(default_factory=lambda: dict(DEFAULT_OPTIONS))
    image: str | None = None       # where the start image came from (start.png is the copy that counts)
    selected: str | None = None
    next_num: int = 1

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
            seed = int(self.options.get("seed", 42)) + len(self.nodes)
        node = Node(nid, parent, prompt, int(seed))
        self.nodes[nid] = node
        return node

    def remove(self, nid: str, project: Path | None = None) -> list[str]:
        """Remove a node and everything below it (and their files). Returns the removed ids."""
        gone = [nid] + [n.id for n in self.descendants(nid)]
        parent = self.nodes[nid].parent
        for g in gone:
            node = self.nodes.pop(g)
            if project is not None:
                for f in node_files(project, g):
                    f.unlink(missing_ok=True)
                (Path(project) / "paths" / f"{g}.mp4").unlink(missing_ok=True)
        if self.selected in gone:
            self.selected = parent
        return gone

    def number(self, nid: str) -> int:
        return int(nid[1:]) if nid[1:].isdigit() else 0

    # ---- files
    def to_json(self) -> dict:
        return {"tool": "video-chain", "version": 2, "image": self.image, "options": self.options,
                "selected": self.selected, "next_num": self.next_num,
                "nodes": [asdict(n) for n in sorted(self.nodes.values(), key=lambda n: self.number(n.id))]}

    @classmethod
    def from_json(cls, d: dict) -> "Tree":
        t = cls(options={**DEFAULT_OPTIONS, **(d.get("options") or {})}, image=d.get("image"),
                selected=d.get("selected"), next_num=int(d.get("next_num") or 1))
        for nd in d.get("nodes", []):
            ov = {k: v for k, v in (nd.get("overrides") or {}).items() if k in OVERRIDE_KEYS and v is not None}
            n = Node(nd["id"], nd.get("parent"), nd.get("prompt", ""), int(nd.get("seed", 42)), nd.get("made"), ov)
            t.nodes[n.id] = n
        t.next_num = max([t.next_num] + [t.number(i) + 1 for i in t.nodes])
        if t.selected not in t.nodes:
            t.selected = None
        return t


def node_files(project: Path, nid: str) -> tuple[Path, Path]:
    d = Path(project) / "nodes"
    return d / f"{nid}.mp4", d / f"{nid}.png"


def load(project: Path) -> Tree:
    """tree.json of a project; a linear chain.json from before branching is turned into a one-branch tree."""
    project = Path(project)
    p = project / TREE_FILE
    if p.exists():
        return Tree.from_json(json.loads(p.read_text(encoding="utf-8")))
    old = chain.load_record(project)
    t = Tree()
    if old:
        t.image = old.get("image")
        s = old.get("settings") or {}
        for key in ("model", "lora", "strength", "frames", "steps", "audio"):
            if key in s:
                t.options[key] = s[key]
        parent = None
        for c in old.get("clips", []):
            n = t.add(parent, c.get("prompt", ""), c.get("seed", 42))
            video, frame = project / c.get("video", ""), project / c.get("last_frame", "")
            if video.is_file() and frame.is_file():
                n.made = {"prompt": n.prompt, "seed": n.seed, "settings": s, "start_sha1": c.get("start_sha1"),
                          "video": os.path.relpath(video, project), "frame": os.path.relpath(frame, project)}
            parent = n.id
        t.selected = parent
    return t


def save(project: Path, tree: Tree) -> None:
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    tmp = project / (TREE_FILE + ".part")
    tmp.write_text(json.dumps(tree.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(project / TREE_FILE)


def set_start_image(project: Path, image: Path, tree: Tree) -> None:
    chain.to_png(Path(image), Path(project) / START_FILE)
    tree.image = str(Path(image).resolve())


# ----------------------------------------------------------------------------- state

class Checker:
    """Works out which nodes are done; caches file hashes and the start image size for one round of questions."""

    def __init__(self, project: Path, tree: Tree):
        self.project, self.tree = Path(project), tree
        self._sha: dict[tuple, str] = {}
        self._status: dict[str, str] = {}
        self._settings: dict | None = None

    def sha(self, path: Path) -> str | None:
        try:
            st = path.stat()
        except OSError:
            return None
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key not in self._sha:
            self._sha[key] = chain.sha1_file(path)
        return self._sha[key]

    def base(self) -> dict | None:
        if self._settings is None:
            self._settings = current_settings(self.project, self.tree.options)
        return self._settings

    def settings(self, nid: str) -> dict | None:
        base = self.base()
        return None if base is None else node_settings(base, self.tree.options, self.tree.nodes[nid])

    def start_frame(self, node: Node) -> Path:
        if node.parent is None:
            return self.project / START_FILE
        return self.project / self.tree.nodes[node.parent].made["frame"] if self.tree.nodes[node.parent].made \
            else node_files(self.project, node.parent)[1]

    def status(self, nid: str) -> str:
        """done | changed (made, but something it depends on changed) | new (never made)."""
        if nid in self._status:
            return self._status[nid]
        node = self.tree.nodes[nid]
        m = node.made
        if not m or not (self.project / m.get("video", "")).is_file() or not (self.project / m.get("frame", "")).is_file():
            st = "new"
        elif (m.get("prompt") != node.prompt or m.get("seed") != node.seed or m.get("settings") != self.settings(nid)
              or (node.parent is not None and self.status(node.parent) != "done")
              or m.get("start_sha1") != self.sha(self.start_frame(node))):
            st = "changed"
        else:
            st = "done"
        self._status[nid] = st
        return st


def current_settings(project: Path, options: dict) -> dict | None:
    """What every clip depends on besides prompt, seed and start frame (None while there is no start image)."""
    start = Path(project) / START_FILE
    if not start.is_file():
        return None
    w, h = chain.canvas_size(*chain.image_size(start), int(options.get("resolution", 512)))
    return chain.settings_dict(model=options.get("model", DEFAULT_OPTIONS["model"]), lora=options.get("lora"),
                               strength=options.get("strength", 1.0), frames=options.get("frames", 73),
                               steps=options.get("steps", 28), width=w, height=h, audio=options.get("audio", True))


def node_settings(base: dict, options: dict, node: Node) -> dict:
    """The settings one node is made with: the project's, with the node's overrides applied."""
    ov = node.overrides or {}
    if "lora" in ov:
        lora = None if ov["lora"] == NO_LORA else ov["lora"]
    else:
        lora = options.get("lora")
    return chain.settings_dict(model=base["model"], lora=lora,
                               strength=ov.get("strength", options.get("strength", 1.0)),
                               frames=ov.get("frames", options.get("frames", 73)),
                               steps=ov.get("steps", options.get("steps", 28)),
                               width=base["width"], height=base["height"], audio=base["audio"])


def loras_needed(tree: Tree, checker: "Checker", nids: list[str]) -> list[str]:
    """The LoRA files these nodes are made with (each once, in order)."""
    out: list[str] = []
    for nid in nids:
        lora = checker.settings(nid)["lora"]
        if lora and lora not in out:
            out.append(lora)
    return out


# ----------------------------------------------------------------------------- generating

def plan(tree: Tree, checker: Checker, targets: list[str]) -> list[str]:
    """Nodes to make for these targets: every not-done node on their paths, parents first, each once."""
    out: list[str] = []
    for t in targets:
        for n in tree.path_to(t):
            if n.id not in out and checker.status(n.id) != "done":
                out.append(n.id)
    return out


def run_tree(generator: chain.Generator, project: Path, targets: list[str],
             events: Callable[[dict], None] | None = None, log: Callable[[str], None] = print) -> list[str]:
    """Make what the target paths need. Saves tree.json after every clip, so a stop keeps finished clips."""
    project = Path(project)
    emit = events or (lambda _e: None)
    tree = load(project)
    for t in targets:
        if t not in tree.nodes:
            raise KeyError(f"no node {t}")
    if not (project / START_FILE).is_file():
        raise FileNotFoundError("the project has no start image")
    empty = [n.id for t in targets for n in tree.path_to(t) if not n.prompt.strip()]
    if empty:
        raise ValueError(f"empty prompt in node {empty[0]}")
    checker = Checker(project, tree)
    todo = plan(tree, checker, targets)
    on_paths = {n.id for t in targets for n in tree.path_to(t)}
    missing = [lo for lo in loras_needed(tree, checker, todo) if not Path(lo).is_file()]
    if missing:
        raise FileNotFoundError(f"LoRA not found: {missing[0]}")
    for nid in sorted(on_paths - set(todo), key=lambda i: tree.depth(i)):
        emit({"e": "node", "id": nid, "state": "reused"})
    emit({"e": "plan", "make": todo})
    (project / "nodes").mkdir(exist_ok=True)
    for k, nid in enumerate(todo, start=1):
        node = tree.nodes[nid]
        start = checker.start_frame(node)
        video, frame = node_files(project, nid)
        log(f"[{k}/{len(todo)}] {nid}: generating (seed {node.seed}) from {start.name}")
        emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": len(todo)})
        for f in (video, frame):
            f.unlink(missing_ok=True)
        node.made = None
        save(project, tree)
        settings = checker.settings(nid)
        generator.generate(node.prompt, start, video, frame, node.seed, settings=settings)
        if not video.exists() or not frame.exists():
            raise RuntimeError(f"the generator did not write {video.name} and {frame.name}")
        node.made = {"prompt": node.prompt, "seed": node.seed, "settings": settings,
                     "start_sha1": checker.sha(start), "video": f"nodes/{video.name}", "frame": f"nodes/{frame.name}"}
        save(project, tree)
        checker = Checker(project, tree)          # the new frame changes what depends on it
        emit({"e": "node", "id": nid, "state": "done"})
    emit({"e": "done", "made": todo})
    return todo


def join_path(project: Path, tree: Tree, nid: str, out: Path | None = None) -> Path:
    """Join the clips from the start to nid into one video (paths/<id>.mp4 unless out is given).
    Every clip on the path must be done."""
    project = Path(project)
    checker = Checker(project, tree)
    path = tree.path_to(nid)
    missing = [n.id for n in path if checker.status(n.id) != "done"]
    if missing:
        raise ValueError(f"not made yet: {', '.join(missing)}")
    videos = [project / n.made["video"] for n in path]
    target = Path(out) if out else project / "paths" / f"{nid}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    key = json.dumps([[str(v), v.stat().st_mtime_ns] for v in videos] +
                     [bool(tree.options.get("keep_repeated"))])
    stamp = target.with_suffix(".json")
    if out is None and target.exists() and stamp.exists() and stamp.read_text() == key:
        return target
    chain.concat(videos, target, drop_first=not tree.options.get("keep_repeated"))
    if out is None:
        stamp.write_text(key)
    return target
