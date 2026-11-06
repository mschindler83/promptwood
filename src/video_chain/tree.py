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

import copy
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import chain

TREE_FILE = "tree.json"
START_FILE = "start.png"
NO_LORA = "none"                   # override value: this node without any LoRA
OVERRIDE_KEYS = ("lora", "strength", "frames", "steps", "extra_loras")
DEFAULT_OPTIONS = {"model": "Comfy-Org/MiniMax-H3", "lora": None, "strength": 1.0, "frames": 73, "steps": 28,
                   "resolution": 512, "audio": True, "keep_repeated": False, "seed": 42,
                   "backend": "comfyui",         # comfyui | ai-toolkit (they sample differently)
                   "extra_loras": [],            # further LoRAs stacked after "lora": [{"path", "strength"}]
                   # identity: reference images (copies in refs/) every clip sees, so the face stays the same
                   "identity": {"mode": "off", "images": [], "size": "match", "hint": True}}
REFS_DIR = "refs"
MAX_REFS = 9


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
    options: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_OPTIONS))
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
                for ext in (".mp4", ".json"):
                    (Path(project) / "paths" / f"{g}{ext}").unlink(missing_ok=True)
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
        """Raises ValueError for a tree that cannot be right (duplicate ids, missing parents, cycles)."""
        if not isinstance(d, dict):
            raise ValueError("not a tree file")
        t = cls(options={**copy.deepcopy(DEFAULT_OPTIONS), **copy.deepcopy(d.get("options") or {})},
                image=d.get("image"),
                selected=d.get("selected"), next_num=int(d.get("next_num") or 1))
        for nd in d.get("nodes", []):
            if not isinstance(nd, dict) or not nd.get("id"):
                raise ValueError("a node without id")
            if nd["id"] in t.nodes:
                raise ValueError(f"node {nd['id']} appears twice")
            ov = {k: v for k, v in (nd.get("overrides") or {}).items() if k in OVERRIDE_KEYS and v is not None}
            n = Node(nd["id"], nd.get("parent"), nd.get("prompt", ""), int(nd.get("seed", 42)), nd.get("made"), ov)
            t.nodes[n.id] = n
        for n in t.nodes.values():
            if n.parent is not None and n.parent not in t.nodes:
                raise ValueError(f"node {n.id} has a missing parent {n.parent}")
        for n in t.nodes.values():                 # every chain of parents must end at the start image
            seen, cur = set(), n.id
            while cur is not None:
                if cur in seen:
                    raise ValueError(f"node {n.id} is part of a loop")
                seen.add(cur)
                cur = t.nodes[cur].parent
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
        try:
            return Tree.from_json(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{p} is damaged ({e})") from e
    old = chain.load_record(project)
    t = Tree()
    if old:
        t.image = old.get("image")
        s = dict(old.get("settings") or {})
        s.setdefault("backend", "ai-toolkit")        # linear chains were always made with ai-toolkit
        for key in ("model", "lora", "strength", "frames", "steps", "audio", "backend"):
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


def add_reference(project: Path, image: Path, tree: Tree) -> str:
    """Copy an image into refs/ as a reference for the face; returns its name in the project."""
    ident = dict(tree.options.get("identity") or DEFAULT_OPTIONS["identity"])
    images = list(ident.get("images") or [])
    if len(images) >= MAX_REFS:
        raise ValueError(f"at most {MAX_REFS} reference images")
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    k = 1
    while (d / f"ref{k}.png").exists() or f"{REFS_DIR}/ref{k}.png" in images:
        k += 1
    rel = f"{REFS_DIR}/ref{k}.png"
    chain.to_png(Path(image), Path(project) / rel)
    images.append(rel)
    ident["images"] = images
    tree.options["identity"] = ident
    return rel


def remove_reference(project: Path, rel: str, tree: Tree) -> None:
    ident = dict(tree.options.get("identity") or {})
    ident["images"] = [r for r in ident.get("images") or [] if r != rel]
    tree.options["identity"] = ident
    target = (Path(project) / rel).resolve()
    if (Path(project) / REFS_DIR).resolve() in target.parents:
        target.unlink(missing_ok=True)


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
    ident = options.get("identity") or {}
    refs = None
    if ident.get("mode") == "reference" and ident.get("images"):
        refs = []
        for rel in ident["images"]:
            sha = chain.file_signature(Path(project) / rel)
            if sha is None:
                raise FileNotFoundError(f"reference image missing: {rel}")
            refs.append({"file": rel, "sha1": sha})
    return chain.settings_dict(model=options.get("model", DEFAULT_OPTIONS["model"]), lora=options.get("lora"),
                               strength=options.get("strength", 1.0), frames=options.get("frames", 73),
                               steps=options.get("steps", 28), width=w, height=h, audio=options.get("audio", True),
                               backend=options.get("backend", "comfyui"), refs=refs,
                               ref_size=ident.get("size", "match"), ref_hint=ident.get("hint", True))


def _extras(items) -> list[tuple[str, float]]:
    return [(e["path"], float(e.get("strength", 1.0))) for e in (items or []) if e.get("path")]


def node_settings(base: dict, options: dict, node: Node) -> dict:
    """The settings one node is made with: the project's, with the node's overrides applied.

    LoRAs: the first one is the node's own (or none), else the project's; a node on the project default also
    gets the project's extra LoRAs; the node's own extra LoRAs come on top, in order."""
    ov = node.overrides or {}
    if "lora" in ov:
        lora = None if ov["lora"] == NO_LORA else ov["lora"]
        extras = _extras(ov.get("extra_loras"))
    else:
        lora = options.get("lora")
        extras = _extras(options.get("extra_loras")) + _extras(ov.get("extra_loras"))
    return chain.settings_dict(model=base["model"], lora=lora, extra_loras=extras,
                               strength=ov.get("strength", options.get("strength", 1.0)),
                               frames=ov.get("frames", options.get("frames", 73)),
                               steps=ov.get("steps", options.get("steps", 28)),
                               width=base["width"], height=base["height"], audio=base["audio"],
                               backend=base.get("backend", "comfyui"), refs=base.get("refs"),
                               ref_size=base.get("ref_size", "match"), ref_hint=base.get("ref_hint", True))


def loras_needed(tree: Tree, checker: "Checker", nids: list[str]) -> list[str]:
    """The LoRA files these nodes are made with (each once, in order)."""
    out: list[str] = []
    for nid in nids:
        s = checker.settings(nid)
        for lora in [s["lora"]] + [e["lora"] for e in s.get("extra_loras") or []]:
            if lora and lora not in out:
                out.append(lora)
    return out


# ----------------------------------------------------------------------------- generating

def empty_prompts(tree: Tree, targets: list[str]) -> list[str]:
    """Nodes on the target paths without a prompt."""
    return [n.id for t in targets for n in tree.path_to(t) if not n.prompt.strip()]


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
    empty = empty_prompts(tree, targets)
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
    nodes_dir = project / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    for left in list(nodes_dir.glob("*.new.*")) + list(nodes_dir.glob("video-chain-*")):   # from a killed run
        shutil.rmtree(left, ignore_errors=True) if left.is_dir() else left.unlink(missing_ok=True)
    for k, nid in enumerate(todo, start=1):
        node = tree.nodes[nid]
        start = checker.start_frame(node)
        video, frame = node_files(project, nid)
        log(f"[{k}/{len(todo)}] {nid}: generating (seed {node.seed}) from {start.name}")
        emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": len(todo)})
        # made under new names: a failed or stopped generation keeps the previous clip and its record
        new_video, new_frame = video.with_name(f"{nid}.new.mp4"), frame.with_name(f"{nid}.new.png")
        for f in (new_video, new_frame):
            f.unlink(missing_ok=True)
        settings = checker.settings(nid)
        start_sha = checker.sha(start)
        call = dict(settings)
        if settings.get("refs"):                   # the generator gets the files; the record keeps names + hashes
            call["ref_paths"] = [str(project / r["file"]) for r in settings["refs"]]
        generator.generate(node.prompt, start, new_video, new_frame, node.seed, settings=call)
        if not new_video.exists() or not new_frame.exists():
            raise RuntimeError(f"the generator did not write {video.name} and {frame.name}")
        new_video.replace(video)
        new_frame.replace(frame)
        node.made = {"prompt": node.prompt, "seed": node.seed, "settings": settings,
                     "start_sha1": start_sha, "video": f"nodes/{video.name}", "frame": f"nodes/{frame.name}"}
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
