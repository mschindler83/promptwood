"""Branching chains: a tree of clips. Every clip (node) starts from the last frame of its parent; clips directly
under the start image start from that image. A path from the start to a node is one video.

A node is a generated clip ("ai") or a piece of a real video ("video": a file and a time range, converted to the
chain's frame rate and size). A generated clip right before a video ends at the video's first frame (H3's
last-frame input), one right after it starts from the video's last frame. So real and generated clips can be
chained in any order. A generated clip can lead into at most one video (its end can only match one).
With a motion guide (override "guide": 5, 22 or 39 frames) a generated clip gets the last frames of the clip
before and the first frames of a following video (with their sound) instead of a single frame at each end.
A generated clip with no video after it can have an end image of its own (Node.end_image, a copy in ends/): the
frame H3 steers the clip's end to. It is not a frame of the chain, so nothing is cut when the path is joined.

Project folder:
  tree.json            nodes (id, parent, prompt, seed, what was made), options, the selected node
  start.png            the start image (copied in)
  nodes/<id>_<tag>.mp4 a node's clip (a new name for every version)
  nodes/<id>_<tag>.png its last frame, the start image of its children
  nodes/<id>_<tag>_first.png   a video node's first frame (the end frame of the generated clip before it)
  paths/<id>.mp4       a joined path (start -> node), made on demand
  ends/<id>_<tag>.png  a generated clip's own end image

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
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import chain

TREE_FILE = "tree.json"
START_FILE = "start.png"
NO_LORA = "none"                   # override value: this node without any LoRA
OVERRIDE_KEYS = ("lora", "strength", "frames", "steps", "extra_loras", "guide")
AI, VIDEO = "ai", "video"
GUIDE_CHOICES = (5, 22, 39)        # motion guide lengths on H3's 17n+5 grid
DEFAULT_OPTIONS = {"model": "Comfy-Org/MiniMax-H3", "lora": None, "strength": 1.0, "frames": 73, "steps": 28,
                   "resolution": 512, "audio": True, "keep_repeated": False, "seed": 42,
                   "extra_loras": [],            # further LoRAs stacked after "lora": [{"path", "strength"}]
                   # turbo: a step-distilled LoRA + its sampler, few steps for every clip (ComfyUI only)
                   "turbo": {"on": False, "lora": None, "steps": 6, "strength": 1.0},
                   # identity: reference images (copies in refs/) every clip sees, so the face stays the same
                   "identity": {"mode": "off", "images": [], "size": "match", "hint": True}}
REFS_DIR = "refs"
ENDS_DIR = "ends"
MAX_REFS = 9


@dataclass
class Node:
    id: str
    parent: str | None             # None: starts from the start image
    prompt: str = ""
    seed: int = 42
    made: dict | None = None       # what the current files were made from (prompt, seed, settings, start_sha1, files)
    overrides: dict = field(default_factory=dict)   # lora (path or NO_LORA), strength, frames, steps, guide
    kind: str = AI                 # AI: generated clip; VIDEO: a range of a real video file
    source: dict | None = None     # VIDEO: {"file": absolute path, "start": s, "end": s}
    end_image: str | None = None   # AI: its own end image (ends/..., project-relative); a video after it wins

    @property
    def is_video(self) -> bool:
        return self.kind == VIDEO


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

    def video_child(self, nid: str | None) -> Node | None:
        """The video a generated clip leads into (its end frame), or None. Only generated clips have one."""
        if nid is None or nid not in self.nodes or self.nodes[nid].is_video:
            return None
        vids = sorted((c for c in self.children(nid) if c.is_video), key=lambda c: self.number(c.id))
        return vids[0] if vids else None

    def can_add_video(self, parent: str | None) -> bool:
        return self.video_child(parent) is None

    def add_video(self, parent: str | None, file: str, start: float, end: float) -> Node:
        """A real video (file, start..end seconds) after parent. A generated clip can lead into one video only."""
        if not self.can_add_video(parent):
            raise ValueError(f"clip #{parent[1:]} already leads into a video")
        if float(end) <= float(start):
            raise ValueError("the end of the range must be after its start")
        node = self.add(parent, "")
        node.kind = VIDEO
        node.source = {"file": str(file), "start": round(float(start), 4), "end": round(float(end), 4)}
        return node

    def remove(self, nid: str, project: Path | None = None) -> list[str]:
        """Remove a node and everything below it (and their files). Returns the removed ids."""
        gone = [nid] + [n.id for n in self.descendants(nid)]
        parent = self.nodes[nid].parent
        for g in gone:
            node = self.nodes.pop(g)
            if project is not None:
                for f in list(node_files(project, g)) + clip_files(project, node):
                    _unlink(f)
                if node.end_image and _in_ends(project, node.end_image):
                    _unlink(Path(project) / node.end_image)
                for ext in (".mp4", ".json"):
                    _unlink(Path(project) / "paths" / f"{g}{ext}")
        if self.selected in gone:
            self.selected = parent
        return gone

    def siblings(self, nid: str) -> list["Node"]:
        """The other clips that start from the same point as nid."""
        parent = self.nodes[nid].parent
        return [n for n in self.children(parent) if n.id != nid]

    def others_than_path(self, nid: str) -> list[str]:
        """Root ids of every branch that is not on the path start -> nid (the siblings along the path)."""
        return [s.id for n in self.path_to(nid) for s in self.siblings(n.id)]

    def number(self, nid: str) -> int:
        return int(nid[1:]) if nid[1:].isdigit() else 0

    # ---- files
    def to_json(self) -> dict:
        return {"tool": "video-chain", "version": 3, "image": self.image, "options": self.options,
                "selected": self.selected, "next_num": self.next_num,
                "nodes": [asdict(n) for n in sorted(self.nodes.values(), key=lambda n: self.number(n.id))]}

    @classmethod
    def from_json(cls, d: dict) -> "Tree":
        """Raises ValueError for a tree that cannot be right (duplicate ids, missing parents, cycles)."""
        if not isinstance(d, dict):
            raise ValueError("not a tree file")
        stored = copy.deepcopy(d.get("options") or {})
        stored.pop("backend", None)                       # ComfyUI is the only generator now
        t = cls(options={**copy.deepcopy(DEFAULT_OPTIONS), **stored},
                image=d.get("image"),
                selected=d.get("selected"), next_num=int(d.get("next_num") or 1))
        for nd in d.get("nodes", []):
            if not isinstance(nd, dict) or not nd.get("id"):
                raise ValueError("a node without id")
            if nd["id"] in t.nodes:
                raise ValueError(f"node {nd['id']} appears twice")
            ov = {k: v for k, v in (nd.get("overrides") or {}).items() if k in OVERRIDE_KEYS and v is not None}
            kind = nd.get("kind") or AI
            src = nd.get("source")
            if kind not in (AI, VIDEO):
                raise ValueError(f"node {nd['id']} has an unknown kind {kind!r}")
            if kind == VIDEO and not (isinstance(src, dict) and src.get("file") and "start" in src and "end" in src):
                raise ValueError(f"video node {nd['id']} has no source")
            end = nd.get("end_image") if kind == AI and isinstance(nd.get("end_image"), str) else None
            n = Node(nd["id"], nd.get("parent"), nd.get("prompt", ""), int(nd.get("seed", 42)), nd.get("made"), ov,
                     kind, dict(src) if kind == VIDEO else None, end)
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


_CLIP_NAME = re.compile(r"^n\d+(_[0-9a-f]{8})?(_first)?\.(mp4|png)$")   # also the fixed names of older versions


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
    area = (Path(project) / "nodes").resolve()
    out = []
    for k in ("video", "frame", "first"):
        if node.made.get(k):
            f = Path(project) / node.made[k]
            if area in f.resolve().parents:           # a damaged/hand-edited record never reaches elsewhere
                out.append(f)
    return out


def new_clip_files(project: Path, nid: str) -> tuple[Path, Path]:
    """Fresh, unique names for a node's next clip and last frame. A clip is never written over an existing file:
    on SMB/CIFS shares a file that is open anywhere cannot be replaced."""
    tag = uuid.uuid4().hex[:8]
    d = Path(project) / "nodes"
    return d / f"{nid}_{tag}.mp4", d / f"{nid}_{tag}.png"


def remove_stale_files(project: Path, tree: "Tree") -> list[Path]:
    """Delete clip files of this app that no record points to any more (old versions that could not be deleted
    while they were open, leftovers of a stopped run). Returns what could not be deleted."""
    d = Path(project) / "nodes"
    if not d.is_dir():
        return []
    used = {Path(project) / n.made[k] for n in tree.nodes.values() if n.made for k in ("video", "frame", "first")
            if n.made.get(k)}
    stuck = []
    for f in d.iterdir():
        if f.is_file() and (_CLIP_NAME.match(f.name) or ".new." in f.name or ".part." in f.name) and f not in used:
            if not _unlink(f):
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
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"{p} is damaged ({e})") from e
    old = chain.load_record(project)
    t = Tree()
    if old:
        t.image = old.get("image")
        s = dict(old.get("settings") or {})
        s.setdefault("backend", "ai-toolkit")        # linear chains were always made with ai-toolkit
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


def _in_ends(project: Path, rel: str) -> bool:
    """rel points into the project's ends/ folder (a damaged/hand-edited record never reaches elsewhere)."""
    area = (Path(project) / ENDS_DIR).resolve()
    return area in (Path(project) / rel).resolve().parents


def set_end_image(project: Path, tree: Tree, nid: str, image: Path) -> str:
    """Copy an image in as the end image of a generated clip (a new name every time; the old copy goes)."""
    node = tree.nodes[nid]
    if node.is_video:
        raise ValueError("a real video has no end image of its own")
    if tree.video_child(nid) is not None:
        raise ValueError(f"clip #{nid[1:]} ends at the video after it")
    d = Path(project) / ENDS_DIR
    d.mkdir(parents=True, exist_ok=True)
    rel = f"{ENDS_DIR}/{nid}_{uuid.uuid4().hex[:8]}.png"
    chain.to_png(Path(image), Path(project) / rel)
    clear_end_image(project, tree, nid)
    node.end_image = rel
    return rel


def clear_end_image(project: Path, tree: Tree, nid: str) -> None:
    node = tree.nodes[nid]
    if node.end_image and _in_ends(project, node.end_image):
        _unlink(Path(project) / node.end_image)
    node.end_image = None


def copy_end_image(project: Path, tree: Tree, src: str, dst: str) -> None:
    """A variant gets its own copy of the end image (deleting one never touches the other's)."""
    rel = tree.nodes[src].end_image
    if rel and (Path(project) / rel).is_file() and tree.video_child(dst) is None:
        set_end_image(project, tree, dst, Path(project) / rel)


def end_image_used(tree: Tree, node: Node) -> str | None:
    """The end image a generated clip is made with: its own, unless a video follows (that one's first frame)."""
    if node.is_video or not node.end_image or tree.video_child(node.id) is not None:
        return None
    return node.end_image


def missing_end_images(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    return [nid for nid in todo if end_image_used(tree, tree.nodes[nid])
            and not (Path(project) / tree.nodes[nid].end_image).is_file()]


def set_start_image(project: Path, image: Path, tree: Tree) -> None:
    chain.to_png(Path(image), Path(project) / START_FILE)
    tree.image = str(Path(image).resolve())


# ----------------------------------------------------------------------------- state

_SRC_INFO: dict[tuple, dict] = {}


def source_info(file: str | Path) -> dict | None:
    """chain.probe_source of a real video, cached by (path, size, date); None if the file is not there."""
    try:
        st = Path(file).stat()
    except OSError:
        return None
    key = (str(file), st.st_size, st.st_mtime_ns)
    if key not in _SRC_INFO:
        _SRC_INFO[key] = chain.probe_source(Path(file))
    return _SRC_INFO[key]


def size_source(project: Path, tree: "Tree | None") -> tuple[int, int] | None:
    """The picture whose aspect ratio the video size follows: the start image, else the first video of the tree
    (by number). None while there is neither."""
    start = Path(project) / START_FILE
    if start.is_file():
        return chain.image_size(start)
    if tree is not None:
        for n in sorted(tree.nodes.values(), key=lambda n: tree.number(n.id)):
            if not n.is_video:
                continue
            info = source_info(n.source["file"])
            if info:
                return info["width"], info["height"]
            key = (n.made or {}).get("key") or {}
            if key.get("width") and key.get("height"):     # the file is gone, its converted clip is still here
                return int(key["width"]), int(key["height"])
    return None


def video_key(node: Node, base: dict) -> dict:
    """What a video node's converted clip depends on: the source file (by size and date), the range and the size."""
    src = node.source or {}
    key = {"file": src.get("file"), "start": src.get("start"), "end": src.get("end"),
           "width": base["width"], "height": base["height"], "size": None, "mtime_ns": None}
    try:
        st = Path(src.get("file") or "").stat()
        key["size"], key["mtime_ns"] = st.st_size, st.st_mtime_ns
    except OSError:
        pass
    return key


def _same_key(made: dict | None, current: dict) -> bool:
    if not isinstance(made, dict):
        return False
    if current.get("size") is None:              # the source is gone: its clip stays valid for the same range
        skip = ("size", "mtime_ns")
        return {k: made.get(k) for k in current if k not in skip} == {k: v for k, v in current.items()
                                                                       if k not in skip}
    return made == current


class Checker:
    """Works out which nodes are done; caches file hashes and the start image size for one round of questions."""

    def __init__(self, project: Path, tree: Tree):
        self.project, self.tree = Path(project), tree
        self._sha: dict[tuple, str] = {}
        self._status: dict[str, str] = {}
        self._settings: dict | None = None

    def sha(self, path: Path | None) -> str | None:
        if path is None:
            return None
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
            self._settings = current_settings(self.project, self.tree.options, self.tree)
        return self._settings

    def settings(self, nid: str) -> dict | None:
        base = self.base()
        return None if base is None else node_settings(base, self.tree.options, self.tree.nodes[nid])

    def start_frame(self, node: Node) -> Path | None:
        """The frame a generated clip starts from: the parent's last frame, the start image, or None (a first
        clip without a start image)."""
        if node.parent is None:
            start = self.project / START_FILE
            return start if start.is_file() else None
        parent = self.tree.nodes[node.parent]
        return self.project / parent.made["frame"] if parent.made else node_files(self.project, node.parent)[1]

    def end_key(self, node: Node):
        """What a generated clip's end depends on: the key of the video it leads into (False: that video is not
        ready), its own end image ({"image": sha1}), or None (a free end)."""
        vc = self.tree.video_child(node.id)
        if vc is None:
            own = end_image_used(self.tree, node)
            return {"image": self.sha(self.project / own)} if own else None
        return vc.made.get("key") if self.status(vc.id) == "done" else False

    def guide_from(self, node: Node) -> str | None:
        """The parent clip a start motion guide is cut from (its file name: every new clip gets a new one), or None
        without a start guide."""
        s = self.settings(node.id)
        if node.parent is None or not s or not int(s.get("guide") or 0):
            return None
        parent = self.tree.nodes[node.parent]
        return parent.made.get("video") if parent.made else None

    def _files_there(self, node: Node) -> bool:
        m = node.made
        if not m or not m.get("video") or not m.get("frame") or (node.is_video and not m.get("first")):
            return False
        return all((self.project / m[k]).is_file() for k in ("video", "frame", "first") if m.get(k))

    def status(self, nid: str) -> str:
        """done | changed (made, but something it depends on changed) | new (never made)."""
        if nid in self._status:
            return self._status[nid]
        node = self.tree.nodes[nid]
        m = node.made
        if not self._files_there(node):
            st = "new"
        elif node.is_video:                        # a real video depends only on its file, range and the size
            base = self.base()
            st = "done" if base is not None and _same_key(m.get("key"), video_key(node, base)) else "changed"
        elif (m.get("prompt") != node.prompt or m.get("seed") != node.seed
              or _old_record(m.get("settings")) != self.settings(nid)
              or (node.parent is not None and self.status(node.parent) != "done")
              or m.get("start_sha1") != self.sha(self.start_frame(node))
              or m.get("end_key") != self.end_key(node)
              or m.get("guide_from") != self.guide_from(node)):
            st = "changed"
        else:
            st = "done"
        self._status[nid] = st
        return st


def _old_record(s: dict | None) -> dict | None:
    """Records written before the backend setting existed were made with ai-toolkit (the generator of the first
    versions). Clips it made sample differently, so they count as changed: ComfyUI makes them again."""
    if isinstance(s, dict) and "backend" not in s:
        return {**s, "backend": "ai-toolkit"}
    return s


def current_settings(project: Path, options: dict, tree: "Tree | None" = None) -> dict | None:
    """What every clip depends on besides prompt, seed and start frame (None while there is neither a start image
    nor a video to take the size from)."""
    dims = size_source(project, tree)
    if dims is None:
        return None
    w, h = chain.canvas_size(*dims, int(options.get("resolution", 512)))
    ident = options.get("identity") or {}
    refs = None
    if ident.get("mode") == "reference" and ident.get("images"):
        refs = []
        for rel in ident["images"]:
            sha = chain.file_signature(Path(project) / rel)
            if sha is None:
                raise FileNotFoundError(f"reference image missing: {rel}")
            refs.append({"file": rel, "sha1": sha})
    turbo = options.get("turbo") or {}
    turbo_on = bool(turbo.get("on") and turbo.get("lora"))
    return chain.settings_dict(model=options.get("model", DEFAULT_OPTIONS["model"]), lora=options.get("lora"),
                               strength=options.get("strength", 1.0), frames=options.get("frames", 73),
                               steps=int(turbo.get("steps", 6)) if turbo_on else options.get("steps", 28),
                               width=w, height=h, audio=options.get("audio", True),
                               turbo=(turbo["lora"], turbo.get("strength", 1.0)) if turbo_on else None,
                               backend="comfyui", refs=refs,
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
                               # turbo: its steps for every clip (a clip's own steps would break it)
                               steps=base["steps"] if base.get("turbo") else ov.get("steps", options.get("steps", 28)),
                               width=base["width"], height=base["height"], audio=base["audio"],
                               turbo=(base["turbo"]["lora"], base["turbo"]["strength"]) if base.get("turbo") else None,
                               backend="comfyui", refs=base.get("refs"),
                               ref_size=base.get("ref_size", "match"), ref_hint=base.get("ref_hint", True),
                               guide=int(ov.get("guide") or 0))


def loras_needed(tree: Tree, checker: "Checker", nids: list[str]) -> list[str]:
    """The LoRA files these nodes are made with (each once, in order)."""
    out: list[str] = []
    for nid in nids:
        if tree.nodes[nid].is_video:
            continue
        s = checker.settings(nid)
        for lora in ([s["lora"]] + [e["lora"] for e in s.get("extra_loras") or []]
                     + ([s["turbo"]["lora"]] if s.get("turbo") else [])):
            if lora and lora not in out:
                out.append(lora)
    return out


# ----------------------------------------------------------------------------- generating

def empty_prompts(tree: Tree, targets: list[str]) -> list[str]:
    """Generated clips on the target paths without a prompt."""
    return [n.id for t in targets for n in tree.path_to(t) if not n.is_video and not n.prompt.strip()]


def plan(tree: Tree, checker: Checker, targets: list[str]) -> list[str]:
    """Nodes to make for these targets: every not-done node on their paths, parents first, each once. A video a
    generated clip leads into comes before that clip (its first frame is the clip's end), also when the video
    itself is not on the path."""
    out: list[str] = []

    def need(nid: str):
        if nid not in out and checker.status(nid) != "done":
            out.append(nid)
    for t in targets:
        for n in tree.path_to(t):
            vc = tree.video_child(n.id)
            if vc is not None:
                need(vc.id)
            need(n.id)
    return out


def path_videos(tree: Tree, targets: list[str]) -> list[str]:
    """Video nodes the targets need (on their paths, or led into by a clip on them)."""
    out = []
    for t in targets:
        for n in tree.path_to(t):
            for v in ([n] if n.is_video else []) + ([tree.video_child(n.id)] if tree.video_child(n.id) else []):
                if v.id not in out:
                    out.append(v.id)
    return out


def make_video(project: Path, node: Node, base: dict) -> dict:
    """Convert a video node's range to a chain clip (new file names); returns its record."""
    project = Path(project)
    src = node.source or {}
    if not Path(src.get("file") or "").is_file():
        raise FileNotFoundError(f"video file not found: {src.get('file')}")
    video, frame = new_clip_files(project, node.id)
    first = frame.with_name(frame.stem + "_first.png")
    video.parent.mkdir(parents=True, exist_ok=True)
    try:
        n = chain.convert_range(Path(src["file"]), video, src["start"], src["end"], base["width"], base["height"])
        chain.frame_at(video, 0, first)
        chain.frame_at(video, n - 1, frame)
    except BaseException:
        for f in (video, frame, first):
            _unlink(f)
        raise
    return {"kind": VIDEO, "key": video_key(node, base), "frames": n, "video": f"nodes/{video.name}",
            "frame": f"nodes/{frame.name}", "first": f"nodes/{first.name}"}


def copy_video(project: Path, tree: Tree, src: Node, parent: str | None) -> Node:
    """A copy of a video node (same file and range) after parent; its converted files are copied too (new names),
    so the copy is ready at once and deleting one never touches the other's files."""
    node = tree.add_video(parent, src.source["file"], src.source["start"], src.source["end"])
    if src.made and all((Path(project) / src.made[k]).is_file() for k in ("video", "frame", "first")):
        video, frame = new_clip_files(project, node.id)
        first = frame.with_name(frame.stem + "_first.png")
        try:
            for k, target in (("video", video), ("frame", frame), ("first", first)):
                shutil.copyfile(Path(project) / src.made[k], target)
        except OSError:
            for f in (video, frame, first):
                _unlink(f)
            return node                              # made later by converting again
        node.made = {**copy.deepcopy(src.made), "video": f"nodes/{video.name}", "frame": f"nodes/{frame.name}",
                     "first": f"nodes/{first.name}"}
    return node


def clip_length(tree: Tree, node: Node) -> int:
    """Frames of a node's clip: a video's range (or what was converted), a generated clip's length setting."""
    if node.is_video:
        if node.made and node.made.get("frames"):
            return int(node.made["frames"])
        return chain.range_frames(node.source["start"], node.source["end"])
    return int((node.overrides or {}).get("frames", tree.options.get("frames", 73)))


def _clip_frames(project: Path, node: Node) -> int:
    if node.is_video and node.made.get("frames"):
        return int(node.made["frames"])
    return chain.probe(Path(project) / node.made["video"])["frames"]


def _guides(project: Path, tree: Tree, node: Node, settings: dict, tmp: Path) -> tuple[list[dict], Path | None]:
    """Motion guide clips (and the end frame) for a generated clip: (guides, end image)."""
    guide = int(settings.get("guide") or 0)
    frames = int(settings["frames"])
    vc = tree.video_child(node.id)
    guides, end_image = [], None
    if guide and node.parent is not None:
        parent = tree.nodes[node.parent]
        clip, n = project / parent.made["video"], _clip_frames(project, parent)
        if n < guide:
            raise ValueError(f"clip #{parent.id[1:]} ({n} frames) is shorter than the {guide}-frame motion guide "
                             f"of clip #{node.id[1:]}")
        g = tmp / "guide_start.mp4"
        audio = chain.cut_frames(clip, g, n - guide, guide)
        guides.append({"video": str(g), "frame_idx": 0, "frames": guide, "audio": audio, "at": "start"})
    if vc is not None:
        if guide:
            clip, n = project / vc.made["video"], _clip_frames(project, vc)
            if n < guide:
                raise ValueError(f"video #{vc.id[1:]} ({n} frames) is shorter than the {guide}-frame motion guide "
                                 f"of clip #{node.id[1:]}")
            g = tmp / "guide_end.mp4"
            audio = chain.cut_frames(clip, g, 0, guide)
            guides.append({"video": str(g), "frame_idx": frames - guide, "frames": guide, "audio": audio,
                           "at": "end"})
        else:
            end_image = project / vc.made["first"]
    elif end_image_used(tree, node):
        end_image = project / node.end_image
    used = sum(g["frames"] for g in guides)
    if used and used >= frames:
        raise ValueError(f"clip #{node.id[1:]} ({frames} frames) is too short for its motion guide ({used} frames); "
                         "make it longer or the guide shorter")
    return guides, end_image


def guide_problem(tree: Tree, checker: "Checker", todo: list[str]) -> str | None:
    """Why a motion guide of a clip to be made cannot work (a neighbour shorter than the guide, a clip too short
    for its guides), or None. Checked before anything is made, with the lengths the clips will have."""
    def length(n: Node) -> int:
        if n.is_video and n.id in todo:             # converted again: the current range
            return chain.range_frames(n.source["start"], n.source["end"])
        return clip_length(tree, n)                 # converted (a source that ends early gives fewer frames)

    for nid in todo:
        node = tree.nodes[nid]
        s = checker.settings(nid) if not node.is_video else None
        guide = int((s or {}).get("guide") or 0)
        if not guide:
            continue
        used = 0
        near = ([("clip", tree.nodes[node.parent])] if node.parent is not None else [])
        vc = tree.video_child(nid)
        if vc is not None:
            near.append(("video", vc))
        for what, other in near:
            n = length(other)
            if n < guide:
                return (f"{what} #{other.id[1:]} ({n} frames) is shorter than the {guide}-frame motion guide "
                        f"of clip #{nid[1:]}")
            used += guide
        frames = int(s["frames"])
        if used and used >= frames:
            return (f"clip #{nid[1:]} ({frames} frames) is too short for its motion guide ({used} frames); "
                    "make it longer or the guide shorter")
    return None


def run_tree(generator: chain.Generator, project: Path, targets: list[str],
             events: Callable[[dict], None] | None = None, log: Callable[[str], None] = print) -> list[str]:
    """Make what the target paths need. Saves tree.json after every clip, so a stop keeps finished clips."""
    project = Path(project)
    emit = events or (lambda _e: None)
    tree = load(project)
    for t in targets:
        if t not in tree.nodes:
            raise KeyError(f"no node {t}")
    empty = empty_prompts(tree, targets)
    if empty:
        raise ValueError(f"empty prompt in node {empty[0]}")
    checker = Checker(project, tree)
    if checker.base() is None:
        raise FileNotFoundError("the project has no start image (and no video to take the size from)")
    todo = plan(tree, checker, targets)
    on_paths = {n.id for t in targets for n in tree.path_to(t)}
    missing = [lo for lo in loras_needed(tree, checker, todo) if not Path(lo).is_file()]
    if missing:
        raise FileNotFoundError(f"LoRA not found: {missing[0]}")
    gone = [nid for nid in todo if tree.nodes[nid].is_video and not Path(tree.nodes[nid].source["file"]).is_file()]
    if gone:
        raise FileNotFoundError(f"video file not found: {tree.nodes[gone[0]].source['file']}")
    problem = guide_problem(tree, checker, todo)
    if problem:
        raise ValueError(problem)
    no_end = missing_end_images(project, tree, todo)
    if no_end:
        raise FileNotFoundError(f"the end image of clip #{no_end[0][1:]} is missing")
    for nid in sorted(on_paths - set(todo), key=lambda i: tree.depth(i)):
        emit({"e": "node", "id": nid, "state": "reused"})
    emit({"e": "plan", "make": todo})
    nodes_dir = project / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    for left in list(nodes_dir.glob("video-chain-*")):                  # temp folders of a killed run
        shutil.rmtree(left, ignore_errors=True)
    remove_stale_files(project, tree)
    for k, nid in enumerate(todo, start=1):
        node = tree.nodes[nid]
        old = clip_files(project, node)
        if node.is_video:
            src = node.source
            log(f"[{k}/{len(todo)}] {nid}: converting {Path(src['file']).name} {src['start']:.2f}-{src['end']:.2f} s")
            emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": len(todo)})
            node.made = make_video(project, node, checker.base())
            new_files = [project / node.made[x] for x in ("video", "frame", "first")]
        else:
            start = checker.start_frame(node)
            log(f"[{k}/{len(todo)}] {nid}: generating (seed {node.seed}) from "
                f"{start.name if start else 'nothing (no start frame)'}")
            emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": len(todo)})
            # a new clip gets new file names: nothing is overwritten, a failed or stopped generation keeps the
            # previous clip and its record, and a still-open old file (SMB) does not stop anything
            new_video, new_frame = new_clip_files(project, nid)
            settings = checker.settings(nid)
            start_sha = checker.sha(start)
            end_key = checker.end_key(node)
            guide_from = checker.guide_from(node)
            call = dict(settings)
            if settings.get("refs"):               # the generator gets the files; the record keeps names + hashes
                call["ref_paths"] = [str(project / r["file"]) for r in settings["refs"]]
            try:
                with tempfile.TemporaryDirectory(prefix="video-chain-", dir=str(nodes_dir)) as tmp:
                    guides, end_image = _guides(project, tree, node, settings, Path(tmp))
                    if guides:
                        call["guides"] = guides
                    if end_image is not None:
                        call["end_image"] = str(end_image)
                    generator.generate(node.prompt, start, new_video, new_frame, node.seed, settings=call)
                if not new_video.exists() or not new_frame.exists():
                    raise RuntimeError(f"the generator did not write the clip of {nid}")
            except BaseException:
                for f in (new_video, new_frame):
                    _unlink(f)
                raise
            node.made = {"prompt": node.prompt, "seed": node.seed, "settings": settings,
                         "start_sha1": start_sha, "video": f"nodes/{new_video.name}",
                         "frame": f"nodes/{new_frame.name}"}
            if end_key is not None:
                node.made["end_key"] = end_key
            if guide_from is not None:
                node.made["guide_from"] = guide_from
            new_files = [new_video, new_frame]
        save(project, tree)
        for f in old:                              # the previous version; left for later if it is open somewhere
            if f not in new_files and not _unlink(f):
                log(f"{f.name} is still in use; it is removed on a later run")
        checker = Checker(project, tree)          # the new frame changes what depends on it
        emit({"e": "node", "id": nid, "state": "done"})
    emit({"e": "done", "made": todo})
    return todo


def _guide_of(node: Node, planned: bool = False) -> int:
    if planned:
        return int((node.overrides or {}).get("guide") or 0)
    return int((((node.made or {}).get("settings")) or {}).get("guide") or 0)


def path_cuts(tree: Tree, path: list[Node], keep_repeated: bool = False,
              planned: bool = False) -> list[tuple[int, int]]:
    """Frames left out of each clip of a path when it is joined, as (at the start, at the end). Where a generated
    clip continues from the clip before, its first frame (or its motion guide frames) repeats that clip's end;
    where it leads into a video, its last frame (or guide frames) repeats the video's start. Two videos in a row
    and a first clip lose nothing. The repeated frames of generated clips are left out, real ones are kept.
    planned: the motion guides as set now (for display), not as the clips were made."""
    cuts = [[0, 0] for _ in path]
    if not keep_repeated:
        for k in range(1, len(path)):
            a, b = path[k - 1], path[k]
            if not b.is_video:
                cuts[k][0] = max(1, _guide_of(b, planned))
            elif not a.is_video and tree.video_child(a.id) is b:
                cuts[k - 1][1] = max(1, _guide_of(a, planned))
    return [(s, e) for s, e in cuts]


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
    cuts = path_cuts(tree, path, bool(tree.options.get("keep_repeated")))
    target = Path(out) if out else project / "paths" / f"{nid}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    key = json.dumps([[str(v), v.stat().st_mtime_ns] for v in videos] + [[list(c) for c in cuts]])
    stamp = target.with_suffix(".json")
    if out is None and target.exists() and stamp.exists() and stamp.read_text() == key:
        return target
    chain.concat(videos, target, cuts=cuts)
    if out is None:
        stamp.write_text(key)
    return target
