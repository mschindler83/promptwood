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
Middle keyframes (Node.keyframes: [{"image": ends/..., "at": seconds, "note": text}]) pin pictures at moments inside
a generated clip (MiniMaxH3AddGuide at that frame); a keyframe the action cannot reach makes H3 cut to it.
A made generated clip without a video after it can hand over at an earlier frame (Node.end_at, a frame index): the
clips after it start from that frame, and the frames after it are left out when a path is joined. The clip itself
stays as made (nothing is generated again for it); only what follows it changes.
The project's style text (options "style": prefix / suffix) is put around every prompt sent to the model.
The cast (options "cast"): characters with face pictures in refs/. A prompt mentions them as @Name; the clip is made
with exactly the pictures of the characters it mentions (renumbered <Picture 1> ... for the model, see clip_cast).
A clip that mentions nobody gets the lead's pictures; one that writes <Picture N> itself gets every picture.

Project folder:
  tree.json            nodes (id, parent, prompt, seed, what was made), options, the selected node
  start.png            the start image (copied in)
  nodes/<id>_<tag>.mp4 a node's clip (a new name for every version)
  nodes/<id>_<tag>.png its last frame, the start image of its children
  nodes/<id>_<tag>_first.png   a video node's first frame (the end frame of the generated clip before it)
  nodes/<id>_<tag>_f<k>.png    a clip's hand-over frame k (Node.end_at), the start image of its children
  paths/<id>.mp4       a joined path (start -> node), made on demand
  ends/<id>_<tag>.png  a generated clip's own end image
  ends/<id>_k<tag>.png a middle keyframe of a generated clip (Node.keyframes)

Every generated clip has its own LoRA stack (Node.overrides["loras"]: [{"path", "strength"}], applied in order;
none by default) and can override the project's clip length and steps
(Node.overrides); the video size and sound stay per project, so the clips of a path fit together.

A node counts as done when its clip exists and was made from exactly its current prompt, seed, settings and start
frame (the parent's current last frame). If any of that changed, the node and everything below it is "changed"
and is made again when a path through it is generated. Other branches are not touched.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import camera as _camera
from . import chain

TREE_FILE = "tree.json"
START_FILE = "start.png"
NO_LORA = "none"                   # old override value: this node without any LoRA
OVERRIDE_KEYS = ("loras", "frames", "steps", "guide", "source_sound", "sound_over", "camera")
_OLD_LORA_KEYS = ("lora", "strength", "extra_loras")
AI, VIDEO = "ai", "video"
GUIDE_CHOICES = (5, 22, 39)        # motion guide lengths on H3's 17n+5 grid
DEFAULT_OPTIONS = {"model": "Comfy-Org/MiniMax-H3", "frames": 73, "steps": 28,
                   "resolution": 512, "audio": True, "keep_repeated": False, "seed": 42,
                   # the aspect ratio of a project without a start image or video (its first clip starts from text)
                   "aspect": "16:9",
                   # turbo: a step-distilled LoRA + its sampler, few steps for every clip (ComfyUI only)
                   "turbo": {"on": False, "lora": None, "steps": 6, "strength": 1.0},
                   # HyperFlow: Video Rebirth's 8-step LoRA with its own sigma grid (ComfyUI-HyperFlow-H3 nodes)
                   "hyperflow": {"on": False, "strength": 1.0},
                   # Pika H3 Flex: Pika's post-trained H3 as a LoRA, 6 steps (ComfyUI template recipe)
                   "pika": {"on": False, "strength": 1.0},
                   # Veda sparse attention: used automatically for clips large enough to gain (chain.veda_useful)
                   "veda": {"on": True},
                   # the cast: characters with face pictures (copies in refs/); a prompt mentions them as @Name and
                   # the clip is made with exactly their pictures. A clip that mentions nobody gets the lead's.
                   # [{"name", "images": [rel], "desc", "lead": bool, "voice": {"file", "name", "secs", "text",
                   #   "ref": {"file", "secs"} (a longer part of the source, for text to speech to clone)}
                   #   (optional), "tts": {"eleven": {"id", "name"}} (optional: its ElevenLabs voice)}]
                   "cast": [],
                   # how the face pictures are used: their size for the model, and the sentence tying a person to
                   # their picture, the language of spoken lines (dialogue tags for voices)
                   "identity": {"size": "match", "hint": True, "language": "English"},
                   # sound references (WAV copies in refs/): <Audio 1> ... in a clip's prompt uses them (a voice,
                   # a soundtrack to copy, a beat); only the clips that name one are made with them
                   "audio_refs": [],
                   # style text put before / after every prompt (e.g. look, camera, a LoRA's trigger word)
                   "style": {"prefix": "", "suffix": ""},
                   # the trigger words of a clip's LoRAs (triggers.py: from the files and CivitAI) before its prompt
                   "lora_triggers": True}
REFS_DIR = "refs"
ENDS_DIR = "ends"
MAX_CAST_PICTURES = 9              # MiniMax H3 ref2va takes up to 9 reference images
MAX_PICTURES_PER_CHARACTER = 3
MAX_AUDIO_REFS = 3                 # MiniMax H3 ref2va: up to 3 sounds, each 2-15 s, together at most 15 s
AUDIO_REF_MIN, AUDIO_REF_TOTAL = 2.0, 15.0
AUDIO_TAG = re.compile(r"<Audio\s*(\d+)>", re.IGNORECASE)
MENTION = re.compile(r"(?<![\w@])@(\w[\w-]*)")        # @Anna in a prompt
NAME_OK = re.compile(r"^[^\W\d_][\w-]*$")              # a character name: a letter first, no spaces
PICTURE_TAG = re.compile(r"<Picture\s*\d+>", re.IGNORECASE)
# a spoken line: "...", “...”, „...“, «...» or MiniMax's own <d>[Language] ...</d>
DIALOGUE = re.compile(r'<d>\s*(?:\[[^\]\n]*\]\s*)?(.*?)</d>|"([^"\n]+)"|“([^”\n]+)”|„([^“”\n]+)[“”]|«([^»\n]+)»',
                      re.DOTALL)
SAID_BY = re.compile(r"[ \t]*,?[ \t]*(says|said|asks|asked|answers|answered|replies|replied|whispers|whispered|"
                     r"shouts|shouted|calls|called|adds|added|cries|cried|murmurs|murmured|mutters|muttered|laughs|"
                     r"laughed|sagt|fragt|antwortet|ruft|flüstert)[ \t]+@(\w[\w-]*)", re.IGNORECASE)   # "...", says @Anna
VOICE_MAX_SECS = 7.5               # a character's voice is cut to this (two voices fit the 15 s a clip may have)
VOICE_REF_MAX_SECS = 30.0          # ... and text to speech clones this much of a longer source (voice["ref"])
WRONG_TAG = re.compile(r"<(Image|Img|Face|Person|Photo|Pic|Ref|Reference)\s*(\d+)>", re.IGNORECASE)


@dataclass
class Node:
    id: str
    parent: str | None             # None: starts from the start image
    prompt: str = ""
    seed: int = 42
    made: dict | None = None       # what the current files were made from (prompt, seed, settings, start_sha1, files)
    overrides: dict = field(default_factory=dict)   # loras [{"path", "strength"}], frames, steps, guide
    kind: str = AI                 # AI: generated clip; VIDEO: a range of a real video file
    source: dict | None = None     # VIDEO: {"file": absolute path, "start": s, "end": s}
    end_image: str | None = None   # AI: its own end image (ends/..., project-relative); a video after it wins
    end_at: int | None = None      # AI: hand over at this frame instead of the last one (see handover_index)
    keyframes: list = field(default_factory=list)   # AI: [{"image": ends/..., "at": s, "note": str}], by time
    frozen: bool = False           # kept as made: done whatever the settings say (see freeze)

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

    def split_video(self, nid: str, at: float) -> tuple[Node, Node]:
        """Insert a generated clip into a real video at `at` seconds (of its file): the video ends there, a new empty
        clip follows it, then the rest of the same video (at .. its old end), whose first frame the clip ends at.
        Clips that followed the video follow its rest now. A video whose person was replaced keeps that in both
        parts (its original range split at the same moment). Returns (the new clip, the rest)."""
        v = self.nodes[nid]
        if not v.is_video:
            raise ValueError(f"#{nid[1:]} is not a real video")
        start, end = float(v.source["start"]), float(v.source["end"])
        at = round(float(at), 4)
        if not start + 1 / 24 <= at <= end - 1 / 24:
            raise ValueError("split inside the video's range (not at its first or last frame)")
        kids = self.children(nid)
        rest_source = {**v.source, "start": at, "end": round(end, 4)}
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

    def remove(self, nid: str, project: Path | None = None) -> list[str]:
        """Remove a node and everything below it (and their files). Returns the removed ids."""
        gone = [nid] + [n.id for n in self.descendants(nid)]
        parent = self.nodes[nid].parent
        for g in gone:
            node = self.nodes.pop(g)
            if project is not None:
                for f in list(node_files(project, g)) + clip_files(project, node):
                    _unlink(f)
                for rel in ({node.end_image, (node.made or {}).get("end_image")} | keyframe_files(node)) - {None}:
                    if _in_ends(project, rel):
                        _unlink(Path(project) / rel)
                for ext in (".mp4", ".json"):
                    _unlink(Path(project) / "paths" / f"{g}{ext}")
        if self.selected in gone:
            self.selected = parent
        heal_frozen(self)
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
        project_loras = {k: stored.pop(k) for k in _OLD_LORA_KEYS if k in stored}   # older: project LoRAs
        t = cls(options={**copy.deepcopy(DEFAULT_OPTIONS), **stored},
                image=d.get("image"),
                selected=d.get("selected"), next_num=int(d.get("next_num") or 1))
        _migrate_cast(t.options, "cast" in stored)
        for nd in d.get("nodes", []):
            if not isinstance(nd, dict) or not nd.get("id"):
                raise ValueError("a node without id")
            if not re.fullmatch(r"n\d+", str(nd["id"])) or nd.get("parent") is not None and \
                    not re.fullmatch(r"n\d+", str(nd["parent"])):
                raise ValueError(f"node {str(nd['id'])[:20]!r} has an invalid id")       # ids become file names
            if nd.get("made") is not None and not isinstance(nd.get("made"), dict):
                raise ValueError(f"node {nd['id']} has a damaged record")
            if nd["id"] in t.nodes:
                raise ValueError(f"node {nd['id']} appears twice")
            kind = nd.get("kind") or AI
            raw = dict(nd.get("overrides") or {})
            if kind == AI:
                raw = migrate_loras(project_loras, raw)
                made = nd.get("made")
                if isinstance(made, dict) and isinstance(made.get("overrides"), dict):   # what Undo goes back to
                    made["overrides"] = {k: v for k, v in migrate_loras(project_loras, made["overrides"]).items()
                                         if k in OVERRIDE_KEYS and v is not None}
            ov = {k: v for k, v in raw.items() if k in OVERRIDE_KEYS and v is not None}
            src = nd.get("source")
            if kind not in (AI, VIDEO):
                raise ValueError(f"node {nd['id']} has an unknown kind {kind!r}")
            if kind == VIDEO and not (isinstance(src, dict) and src.get("file") and "start" in src and "end" in src):
                raise ValueError(f"video node {nd['id']} has no source")
            end = nd.get("end_image") if kind == AI and isinstance(nd.get("end_image"), str) else None
            at = nd.get("end_at")
            at = int(at) if kind == AI and isinstance(at, (int, float)) and not isinstance(at, bool) and at >= 1 \
                else None
            n = Node(nd["id"], nd.get("parent"), nd.get("prompt", ""), int(nd.get("seed", 42)), nd.get("made"), ov,
                     kind, dict(src) if kind == VIDEO else None, end, at,
                     clean_keyframes(nd.get("keyframes")) if kind == AI else [],
                     bool(nd.get("frozen")) and isinstance(nd.get("made"), dict))
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
        heal_frozen(t)
        if t.selected not in t.nodes:
            t.selected = None
        return t


SEED_MAX = 2_000_000_000            # (the seed field's range)
REFINED = ("1080p", "2k")          # refined copies of a clip version for export: nodes/<id>_<tag>_<size>.mp4
_CLIP_NAME = re.compile(r"^n\d+(_[0-9a-f]{8})?((_first|_f\d+)?\.(mp4|png)|_(1080p|2k)\.(mp4|json))$")   # also the fixed names of older versions


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
    area = (Path(project) / "nodes").resolve()
    out = []
    for k in ("video", "frame", "first"):
        if node.made.get(k):
            f = Path(project) / node.made[k]
            if area in f.resolve().parents:           # a damaged/hand-edited record never reaches elsewhere
                out.append(f)
    if node.made.get("video"):                    # hand-over frames and refined copies of this version
        v = Path(project) / node.made["video"]
        if area in v.resolve().parents and v.parent.is_dir():
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
    """Delete clip files of this app that no record points to any more (old versions that could not be deleted
    while they were open, leftovers of a stopped run). Returns what could not be deleted."""
    d = Path(project) / "nodes"
    if not d.is_dir():
        return []
    used = {Path(project) / n.made[k] for n in tree.nodes.values() if n.made for k in ("video", "frame", "first")
            if n.made.get(k)}
    used |= {h for n in tree.nodes.values() if (h := handover_file(project, tree, n)) is not None}
    used |= {f for n in tree.nodes.values() if n.made and n.made.get("video") for size in REFINED
             for f in (refined_file(project, n, size), refined_file(project, n, size).with_suffix(".json"))}
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
        except Exception as e:  # noqa: BLE001  (any damage: one clear message)
            raise ValueError(f"{p} is damaged ({type(e).__name__}: {e})") from e
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


def save(project: Path, tree: Tree) -> None:
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    tmp = project / (TREE_FILE + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(tree.to_json(), indent=2, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())                       # on disk before it replaces the old file (power cut)
    tmp.replace(project / TREE_FILE)


def _migrate_cast(options: dict, has_cast: bool) -> None:
    """Older projects had one list of face pictures (identity "images", used by every clip while "mode" was
    "reference"). They become one character, Person1, the lead while they were on: every clip is made with exactly
    the same pictures as before, so made clips stay done."""
    ident = dict(options.get("identity") or {})
    if not has_cast:
        images = [r for r in ident.get("images") or [] if isinstance(r, str)]
        options["cast"] = ([{"name": "Person1", "images": images, "desc": "",
                             "lead": ident.get("mode") == "reference"}] if images else [])
    for k in ("images", "mode"):
        ident.pop(k, None)
    options["identity"] = {**DEFAULT_OPTIONS["identity"], **ident}
    cast = []
    for c in options.get("cast") or []:
        if isinstance(c, dict) and isinstance(c.get("name"), str) and c["name"]:
            entry = {"name": c["name"], "images": [r for r in c.get("images") or [] if isinstance(r, str)],
                     "desc": str(c.get("desc") or ""), "lead": bool(c.get("lead"))}
            v = c.get("voice")
            if isinstance(v, dict) and isinstance(v.get("file"), str) and v["file"]:
                entry["voice"] = {"file": v["file"], "name": str(v.get("name") or ""),
                                  "secs": float(v.get("secs") or 0)}
                if isinstance(v.get("text"), str) and v["text"].strip():
                    entry["voice"]["text"] = v["text"].strip()      # what clone_file says (text to speech clones it)
                r = v.get("ref")
                if isinstance(r, dict) and isinstance(r.get("file"), str) and r["file"]:
                    entry["voice"]["ref"] = {"file": r["file"], "secs": float(r.get("secs") or 0)}
            el = (c.get("tts") or {}).get("eleven") if isinstance(c.get("tts"), dict) else None
            if isinstance(el, dict) and isinstance(el.get("id"), str) and el["id"]:
                entry["tts"] = {"eleven": {"id": el["id"], "name": str(el.get("name") or el["id"])}}
            cast.append(entry)
    options["cast"] = cast


def cast_of(tree_or_options) -> list[dict]:
    """The characters ({"name", "images", "desc", "lead"}), in the order of the cast list."""
    options = tree_or_options.options if isinstance(tree_or_options, Tree) else tree_or_options
    return list(options.get("cast") or [])


def character(tree_or_options, name: str) -> dict | None:
    """The character with this name (case does not matter), or None."""
    low = (name or "").lower()
    return next((c for c in cast_of(tree_or_options) if c["name"].lower() == low), None)


def lead_of(tree_or_options) -> dict | None:
    return next((c for c in cast_of(tree_or_options) if c.get("lead")), None)


def all_pictures(tree_or_options) -> list[str]:
    """Every face picture of the cast, in cast order (what <Picture 1> ... number in a prompt that writes them)."""
    return [r for c in cast_of(tree_or_options) for r in c["images"]]


def check_name(tree: Tree, name: str, keep: str | None = None) -> str:
    """A valid new name (stripped), or ValueError. keep: the character being renamed (its own name is fine)."""
    name = (name or "").strip()
    if not NAME_OK.match(name):
        raise ValueError("a name starts with a letter and has only letters, digits, - and _ (no spaces), "
                         "e.g. Anna or Anna_B")
    other = character(tree, name)
    if other is not None and (keep is None or other["name"].lower() != keep.lower()):
        raise ValueError(f"there is already a character called {other['name']}")
    return name


def _set_cast(tree: Tree, cast: list[dict]) -> None:
    tree.options["cast"] = cast


def source_of(rel: str) -> str:
    """Where the full picture a face picture was cropped from is kept (refs/ref1.png -> refs/ref1_full.png)."""
    return rel[:-4] + "_full.png" if rel.endswith(".png") else rel + "_full.png"


def _write_picture(project: Path, image: Path, rel: str, crop: tuple[int, int, int] | None) -> None:
    """The face picture rel from image, cropped to the square crop (x, y, side) if given; the full picture is kept
    beside it (source_of) for cropping again."""
    full = Path(project) / source_of(rel)
    if crop:
        made = Path(image).resolve() != full.resolve()
        if made:
            chain.to_png(Path(image), full)
        x, y, side = (int(v) for v in crop)
        try:
            chain.to_png(full, Path(project) / rel, (x, y, side, side))
        except BaseException:
            if made:
                full.unlink(missing_ok=True)
            raise
    else:
        chain.to_png(Path(image), Path(project) / rel)
        full.unlink(missing_ok=True)


def recrop_picture(project: Path, tree: Tree, rel: str, crop: tuple[int, int, int] | None) -> None:
    """Crop a face picture again from its full picture (crop None: the whole full picture). A picture added
    uncropped is its own full picture."""
    if rel not in all_pictures(tree):
        raise ValueError(f"{rel} is no face picture of the cast")
    full = Path(project) / source_of(rel)
    if not full.is_file():                          # added uncropped (or before cropping existed): it is the full one
        shutil.copyfile(Path(project) / rel, full)
    if crop:
        x, y, side = (int(v) for v in crop)
        chain.to_png(full, Path(project) / rel, (x, y, side, side))
    else:
        chain.to_png(full, Path(project) / rel)


def _copy_picture(project: Path, image: Path, tree: Tree, crop: tuple[int, int, int] | None = None) -> str:
    if len(all_pictures(tree)) >= MAX_CAST_PICTURES:
        raise ValueError(f"at most {MAX_CAST_PICTURES} face pictures in a project")
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = set(all_pictures(tree))
    k = 1
    while (d / f"ref{k}.png").exists() or f"{REFS_DIR}/ref{k}.png" in used:
        k += 1
    rel = f"{REFS_DIR}/ref{k}.png"
    _write_picture(project, image, rel, crop)
    return rel


def add_character(project: Path, tree: Tree, name: str, image: Path | None = None, desc: str = "",
                  crop: tuple[int, int, int] | None = None) -> dict:
    """A new character (with a first face picture, cropped to crop = x, y, side). The first character of a cast is
    the lead."""
    name = check_name(tree, name)
    entry = {"name": name, "images": [], "desc": (desc or "").strip(), "lead": not cast_of(tree)}
    if image is not None:
        entry["images"].append(_copy_picture(project, image, tree, crop))
    _set_cast(tree, cast_of(tree) + [entry])
    return entry


def next_character_name(tree: Tree) -> str:
    k = 1
    while character(tree, f"Person{k}") is not None:
        k += 1
    return f"Person{k}"


def add_picture(project: Path, tree: Tree, name: str, image: Path,
                crop: tuple[int, int, int] | None = None) -> str:
    """Copy an image into refs/ as one more face picture of a character (cropped to the square crop = x, y, side;
    the full picture is kept for cropping again); returns its name in the project."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    if len(c["images"]) >= MAX_PICTURES_PER_CHARACTER:
        raise ValueError(f"at most {MAX_PICTURES_PER_CHARACTER} pictures per character")
    rel = _copy_picture(project, image, tree, crop)
    _set_cast(tree, [{**x, "images": x["images"] + [rel]} if x is c else x for x in cast_of(tree)])
    return rel


def add_reference(project: Path, image: Path, tree: Tree, name: str | None = None,
                  crop: tuple[int, int, int] | None = None) -> str:
    """A face picture for a character (name), else for the lead, else for a new character PersonN."""
    target = character(tree, name) if name else (lead_of(tree) or (cast_of(tree) or [None])[0])
    if name and target is None:
        raise ValueError(f"no character called {name}")
    if target is None:
        return add_character(project, tree, next_character_name(tree), image, crop=crop)["images"][0]
    return add_picture(project, tree, target["name"], image, crop)


def _drop_file(project: Path, rel: str, tree: Tree) -> None:
    if rel in all_pictures(tree):
        return                                     # another character still uses it
    for f in (rel, source_of(rel)):
        target = (Path(project) / f).resolve()
        if (Path(project) / REFS_DIR).resolve() in target.parents:
            target.unlink(missing_ok=True)


def remove_picture(project: Path, tree: Tree, rel: str) -> None:
    _set_cast(tree, [{**c, "images": [r for r in c["images"] if r != rel]} for c in cast_of(tree)])
    _drop_file(project, rel, tree)


def remove_reference(project: Path, rel: str, tree: Tree) -> None:
    remove_picture(project, tree, rel)


def remove_character(project: Path, tree: Tree, name: str) -> None:
    c = character(tree, name)
    if c is None:
        return
    _set_cast(tree, [x for x in cast_of(tree) if x is not c])
    for rel in c["images"]:
        _drop_file(project, rel, tree)
    if c.get("voice"):
        _drop_voice(project, c["voice"], tree)


def voices_in_use(tree_or_options) -> list[str]:
    return [c["voice"]["file"] for c in cast_of(tree_or_options) if c.get("voice")]


def clone_file(voice: dict) -> str:
    """The file text to speech clones a voice from: its longer clone reference when it has one, else the voice."""
    return (voice.get("ref") or {}).get("file") or voice["file"]


def clone_secs(voice: dict) -> float:
    return float((voice.get("ref") or voice).get("secs") or 0)


def _drop_voice(project: Path, voice: dict, tree: Tree) -> None:
    """Delete a voice's files (the voice and its clone reference) that nothing uses any more."""
    kept = {a["file"] for a in audio_refs(tree)}
    for c in cast_of(tree):
        if c.get("voice"):
            kept |= {c["voice"]["file"], clone_file(c["voice"])}
    for rel in {voice["file"], clone_file(voice)} - kept:
        target = (Path(project) / rel).resolve()
        if (Path(project) / REFS_DIR).resolve() in target.parents:
            target.unlink(missing_ok=True)


def set_voice(project: Path, tree: Tree, name: str, source: Path, start: float = 0.0,
              label: str | None = None, end: float | None = None) -> dict:
    """Copy the sound of an audio or video file into refs/ as the voice of a character (from start, at most
    VOICE_MAX_SECS, not past end); it replaces the character's old voice. Returns the entry, with "cut": the source was longer.
    A clip uses the voice when the character says a line in it ("..." after @Name). Of a longer source, up to
    VOICE_REF_MAX_SECS (from the same start) are kept next to it as the clone reference ("ref"): text to speech
    clones the voice from that, the clip model still hears the short voice."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    length = chain.audio_seconds(Path(source))
    if length is None:
        raise ValueError(f"{Path(source).name} has no sound")
    cut_end = end is not None and float(end) < length - 0.01
    have = (float(end) if cut_end else length) - float(start)
    if have < AUDIO_REF_MIN:
        raise ValueError(f"a voice needs at least {AUDIO_REF_MIN:.0f} s of speech ({Path(source).name} has "
                         f"{max(have, 0):.1f} s" + (" in the chosen part)" if cut_end or start else ")"))
    secs = min(have, VOICE_MAX_SECS)
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = set(voices_in_use(tree))
    k = 1
    while (d / f"voice{k}.wav").exists() or f"{REFS_DIR}/voice{k}.wav" in used:
        k += 1
    rel = f"{REFS_DIR}/voice{k}.wav"
    chain.to_wav(Path(source), Path(project) / rel, start, secs if have > secs or cut_end else None)
    real = chain.audio_seconds(Path(project) / rel) or secs
    entry = {"file": rel, "name": label or Path(source).name, "secs": round(real, 2)}
    if have >= secs + 1.0:                          # (worth it: at least a second more to clone from)
        long = min(have, VOICE_REF_MAX_SECS)
        ref = f"{REFS_DIR}/voice{k}_ref.wav"
        chain.to_wav(Path(source), Path(project) / ref, start, long if have > long or cut_end else None)
        entry["ref"] = {"file": ref, "secs": round(chain.audio_seconds(Path(project) / ref) or long, 2)}
    old = c.get("voice")
    _set_cast(tree, [{**x, "voice": entry} if x is c else x for x in cast_of(tree)])
    if old:
        _drop_voice(project, old, tree)
    return {**entry, "cut": have > secs + 0.01}


def remove_voice(project: Path, tree: Tree, name: str) -> None:
    c = character(tree, name)
    if c is None or not c.get("voice"):
        return
    _set_cast(tree, [{k: v for k, v in x.items() if k != "voice"} if x is c else x for x in cast_of(tree)])
    _drop_voice(project, c["voice"], tree)


def update_character(tree: Tree, name: str, *, desc: str | None = None, lead: bool | None = None) -> None:
    """Change a character's description, or make it the lead (lead=False: no lead)."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    cast = []
    for x in cast_of(tree):
        x = dict(x)
        if x["name"] == c["name"]:
            if desc is not None:
                x["desc"] = desc.strip()
            if lead is not None:
                x["lead"] = bool(lead)
        elif lead:
            x["lead"] = False                       # one lead only
        cast.append(x)
    _set_cast(tree, cast)


def rename_character(tree: Tree, old: str, new: str) -> list[str]:
    """Rename a character and every @old in the clips' prompts (and the style text). Returns the clips changed."""
    c = character(tree, old)
    if c is None:
        raise ValueError(f"no character called {old}")
    new = check_name(tree, new, keep=c["name"])

    def swap(text: str) -> str:
        return MENTION.sub(lambda m: f"@{new}" if m.group(1).lower() == c["name"].lower() else m.group(0),
                           text or "")
    changed = []
    for n in tree.nodes.values():
        if not n.is_video and swap(n.prompt) != n.prompt:
            n.prompt = swap(n.prompt)
            changed.append(n.id)
    style = dict(tree.options.get("style") or {})
    for k in ("prefix", "suffix"):
        if style.get(k):
            style[k] = swap(style[k])
    tree.options["style"] = style
    _set_cast(tree, [{**x, "name": new} if x["name"] == c["name"] else x for x in cast_of(tree)])
    return changed


def _in_lines(text: str):
    """pos -> True when it lies inside a quoted line (an @Name there is a spoken word, not a mention)."""
    spans = [(m.start(), m.end()) for m in DIALOGUE.finditer(text or "")]
    return lambda pos: any(a < pos < b for a, b in spans)


def _unmention(line: str) -> str:
    """A spoken line with its @Names as plain names (nobody says the @)."""
    return MENTION.sub(lambda m: m.group(1), line)


def mentions(options: dict, text: str) -> tuple[list[dict], list[str]]:
    """The characters a prompt mentions as @Name, in the order they first appear, and the @words that name
    nobody in the cast. An @Name inside a quoted line is a spoken word, not a mention."""
    found, unknown = [], []
    inside = _in_lines(text)
    for m in MENTION.finditer(text or ""):
        if inside(m.start()):
            continue
        c = character(options, m.group(1))
        if c is None:
            if m.group(1) not in unknown:
                unknown.append(m.group(1))
        elif c not in found:
            found.append(c)
    return found, unknown


_FEMALE = re.compile(r"\b(woman|women|girl|female|lady|mother|mom|mum|wife|daughter|sister|grandmother|frau|"
                     r"mädchen|mutter)\b", re.IGNORECASE)
_MALE = re.compile(r"\b(man|men|boy|male|guy|gentleman|father|dad|husband|son|brother|grandfather|mann|junge|"
                   r"vater)\b", re.IGNORECASE)
_PRONOUN = re.compile(r"\b(she|he|sie|er)\b", re.IGNORECASE)


def gender_of(c: dict) -> str | None:
    """'f' / 'm' from a character's description (woman, man, girl ...), else None."""
    desc = c.get("desc") or ""
    f, m = bool(_FEMALE.search(desc)), bool(_MALE.search(desc))
    return "f" if f and not m else "m" if m and not f else None


def dialogue(options: dict, text: str, default: dict | None = None) -> list[dict]:
    """The spoken lines of a prompt, [{"start", "end", "line", "speaker": character or None}]. Who says a line,
    looking at the words since the sentence began (or since the line before, in the same sentence):
    - a "she" / "he" there, the last one: the last-mentioned character of that gender (from its description);
    - else the first @Name there (@Anna looks at @Ben and says "Hi." -> Anna);
    - else the first @Name after the line in its sentence ("Hi," says @Anna.);
    - else the last @Name before the line, else default."""
    text = text or ""
    spans = [(m.start(), m.end(), next(g for g in m.groups() if g is not None).strip())
             for m in DIALOGUE.finditer(text)]
    masked = list(text)
    for a, b, line in spans:
        masked[a:b] = "x" * (b - a)                 # sentence ends inside a quote do not count ...
        if line[-1:] in (".", "!", "?"):
            masked[b - 1] = "."                     # ... but a line that ends one ends its sentence
    masked = "".join(masked)
    inside = _in_lines(text)
    marks = [(m.start(), character(options, m.group(1))) for m in MENTION.finditer(text) if not inside(m.start())]
    marks = [(pos, c) for pos, c in marks if c is not None]
    out, prev_end = [], 0
    for a, b, line in spans:
        if not line:
            prev_end = b
            continue
        cut = max(masked.rfind(e, 0, a) for e in (". ", "! ", "? ", "\n"))
        seg = max(cut, prev_end - 1)                # the words that lead up to this line
        after = [masked.find(e, b) for e in (". ", "! ", "? ", "\n")]
        stop = min([x for x in after if x >= 0] or [len(text)])
        who = None
        pron = [m for m in _PRONOUN.finditer(masked, seg + 1, a)]
        if pron:
            g = "f" if pron[-1].group(1).lower() in ("she", "sie") else "m"
            if pron[-1].group(1).lower() == "sie" and any(gender_of(c) == "m" for _p, c in marks):
                g = None                            # German "sie" is also "they" / "you"
            if g:
                who = next((c for pos, c in reversed(marks) if pos < pron[-1].start() and gender_of(c) == g), None)
        who = who or next((c for pos, c in marks if seg < pos < a), None) \
            or next((c for pos, c in marks if b <= pos <= stop), None) \
            or next((c for pos, c in reversed(marks) if pos < a), None) or default
        out.append({"start": a, "end": b, "line": line, "speaker": who})
        prev_end = b
    return out


def replace_lines(text: str, lines: list[str]) -> str:
    """A prompt with its spoken lines (those dialogue() gives, in order) replaced by `lines`; everything around
    them, the quotes too, stays. ValueError: another number of lines."""
    spans = []
    for m in DIALOGUE.finditer(text or ""):
        g = next(k for k in range(1, (m.lastindex or 0) + 1) if m.group(k) is not None)
        if m.group(g).strip():
            spans.append(m.span(g))
    if len(spans) != len(lines):
        raise ValueError(f"the prompt has {len(spans)} spoken lines, not {len(lines)}")
    for (a, b), line in reversed(list(zip(spans, lines))):
        text = text[:a] + line + text[b:]
    return text


def spoken_lines(text: str) -> list[str]:
    """The quoted lines of a prompt, as written (normalised spaces, no trailing , ; :)."""
    return [" ".join(next(g for g in m.groups() if g is not None).split()).rstrip(",;: ")
            for m in DIALOGUE.finditer(text or "")]


def refine_problems(original: str, refined: str) -> list[str]:
    """What a refined prompt lost of the original: spoken lines (dropped, changed or repeated) and @mentions."""
    def key(x: str) -> str:
        return re.sub(r"[^\w]+", " ", x.lower()).strip()
    before, after = spoken_lines(original), spoken_lines(refined)
    have = [key(x) for x in after]
    out = []
    for line in dict.fromkeys(before):
        n_want, n_got = [key(x) for x in before].count(key(line)), have.count(key(line))
        short = line if len(line) <= 40 else line[:39] + "…"
        if n_got == 0:
            out.append(f"the line \u201c{short}\u201d is gone or changed")
        elif n_got > n_want:
            out.append(f"the line \u201c{short}\u201d is there {n_got} times")
    gone = [f"@{m}" for m in dict.fromkeys(MENTION.findall(original or ""))
            if not re.search(rf"(?<![\w@])@{re.escape(m)}(?![\w-])", refined or "", re.IGNORECASE)]
    if gone:
        out.append(f"{', '.join(gone)} {'is' if len(gone) == 1 else 'are'} no longer mentioned")
    return out


def _picture_tags(found: list[dict]) -> tuple[dict, list[str]]:
    """<Picture> tags per character (numbered in order of their first mention) and the pictures in that order."""
    tags, images = {}, []
    for c in found:
        if c["images"]:
            first = len(images) + 1
            images += c["images"]
            tags[c["name"]] = " and ".join(f"<Picture {k}>" for k in range(first, len(images) + 1))
    return tags, images


def _voiced_prompt(options: dict, text: str, found: list[dict], lines: list[dict], n_sounds: int,
                   implicit: bool) -> tuple[str, list[str], list[str]]:
    """MiniMax's structured reference format (the only one a voice took effect with in the tests): subjects,
    <Audio N> as the voice-timbre reference of a speaker (S1, S2 ... in speaking order), each line as
    <d>[Language] ...</d>. The voices come after the n_sounds sounds the prompt names itself.
    Returns (prompt, pictures, voice files)."""
    lang = ((options.get("identity") or {}).get("language") or "English").strip() or "English"
    hint = (options.get("identity") or {}).get("hint", True)
    tags, images = _picture_tags(found)
    subj = {c["name"]: k for k, c in enumerate(found, start=1)}
    order: list[str] = []
    for d in lines:
        sp = d["speaker"]
        if sp is not None and sp["name"] in subj and sp["name"] not in order:
            order.append(sp["name"])
    sid = {n: k for k, n in enumerate(order, start=1)}
    voiced = [n for n in order if character(options, n).get("voice")]
    audio_no = {n: n_sounds + k for k, n in enumerate(voiced, start=1)}

    def token(name: str) -> str:
        return f"<Subject {subj[name]}>" + (f" (S{sid[name]})" if name in sid else "")

    def names(t: str) -> str:
        def one(m):
            c = character(options, m.group(1))
            if c is None:
                return m.group(0)
            return token(c["name"]) if c["name"] in subj else c["name"]
        return MENTION.sub(one, t)

    body, pos, said = "", 0, set()
    for d in lines:
        body += names(text[pos:d["start"]])
        sp = d["speaker"]
        end = d["end"]
        line = _unmention(d["line"].rstrip(",;: ") or d["line"])
        say = f"<d>[{lang}] {line}</d>"
        # "Hi," says @Anna.  ->  <Subject 1> (S1) says, ... <d>Hi</d>.
        tail = SAID_BY.match(text, end)
        verb = "says"
        if sp is not None and tail and (character(options, tail.group(2)) or {}).get("name") == sp["name"]:
            verb, end = tail.group(1), tail.end()
        elif tail:
            tail = None
        head = body.rstrip().rstrip(":,").rstrip()
        in_sentence = bool(head) and not tail and (head[-1].isalnum() or head[-1] in ")>")
        if sp is not None and sp["name"] in subj:
            voice = ""
            if sp["name"] in audio_no and sp["name"] not in said:
                said.add(sp["name"])
                voice = f"using the voice timbre referenced from <Audio {audio_no[sp['name']]}>, "
            if in_sentence:                         # "... and says" (the speaker leads the sentence)
                body = f"{head}, {voice}{say}"
            else:                                   # a line on its own: say who speaks
                body = f"{head} {token(sp['name'])} {verb}, {voice}{say}".lstrip()
        else:
            body += say
        pos = end
    body += names(text[pos:])

    defs = []
    for c in found:
        desc = f" ({c['desc'].rstrip('.')})" if c.get("desc") else ""
        defs.append(f"<Subject {subj[c['name']]}> is {c['name']}{desc}"
                    + (f", the person in {tags[c['name']]}." if c["name"] in tags else "."))
    defs += [f"<Audio {k}> is a sound reference, used as the description says." for k in range(1, n_sounds + 1)]
    defs += [f"<Audio {audio_no[n]}> is the voice-timbre reference for {token(n)}, the voice of one speaker."
             for n in voiced]
    who = [f"<Subject {subj[c['name']]}>" for c in found]
    shows = who[0] if len(who) == 1 else ", ".join(who[:-1]) + " and " + who[-1]
    summary = (f"[reference generation + audio reference] The target video shows {shows}. "
               + " ".join(f"{token(n)} speaks with the voice timbre of <Audio {audio_no[n]}>." for n in voiced))
    keep = [f"<Subject {subj[c['name']]}> (appears in [Shot 1]): fully_preserved - {c['name']}'s face and "
            "identity are retained." for c in found if c["name"] in tags]
    keep += [f"<Audio {audio_no[n]}>: reference - its vocal timbre guides the dialogue delivery of {token(n)} "
             "without copying the original signal or spoken words." for n in voiced]
    detail = ""
    if hint and images:
        detail += "The pictures only show who the people are; the scene, pose and framing come from the video.\n"
    if implicit and order:
        detail += f"{token(order[0])} is the one who speaks.\n"
    detail += f"[Shot 1] {body.strip()}"
    prompt = "\n\n".join([
        "subject_definitions:\n" + "\n".join(defs),
        "summary:\n" + summary.strip(),
        "retention_analysis:\n" + "\n".join(keep),
        "detailed_description:\n" + detail,
        "overall_soundscape:\nThe natural sound of the scene; the spoken lines are clear.",
        "non_diegetic_music:\n" + ("N/A" if not n_sounds else "As the sound references above say.")])
    return prompt, images, [character(options, n)["voice"]["file"] for n in voiced]


def sound_given(node: Node) -> bool:
    """The clip's whole sound is given (its Sound setting plays a track, a video's or the clip before's sound, and
    the picture is made to it): the model speaks nothing itself, so it gets no voices (see clip_cast)."""
    ov = node.overrides or {}
    return not node.is_video and bool(ov.get("source_sound")) and not ov.get("sound_over")


def node_cast(options: dict, node: Node) -> dict:
    """clip_cast for a clip as it is set up: without the voices when its sound is given."""
    return clip_cast(options, node.prompt, voices=not sound_given(node))


def clip_cast(options: dict, prompt: str, voices: bool = True) -> dict:
    """How a clip uses the cast. The prompt is the clip's own (the style text is put around it here).
    voices=False: the clip's sound is given (sound_given), its spoken lines are in that sound already, so no voice
    is sent and the characters are used as if they had none (the lines stay in the prompt as written; the
    soundtrack sentence of model_prompt ties the lips to the sound). The voices would only use up the 15 s of
    sound the model takes, which the soundtrack needs.
    {"mode": "voices" | "mention" | "tags" | "lead" | None, "characters": [...], "images": [rel], "voices": [rel],
    "language": the dialogue language and "lines": [{"name": speaker or None, "line"}] (voices only), "prompt": what
    the model gets, "record": [{"name", "desc"}]
    (voices / mention: what the model prompt says about them)}.
    voices:  a character with a voice says a line ("..." in the sentence of its @Name; with nobody mentioned, the
             lead's lines): MiniMax's structured format (see _voiced_prompt); the voices come after the sounds the
             prompt names as <Audio N>.
    mention: the characters named as @Name, numbered in order of their first mention; each @Name becomes its name
             (the first time with its <Picture> tags), and a sentence ties each to their face.
    tags:    the prompt writes <Picture N> itself: every face picture of the cast, in cast order.
    lead:    nobody is mentioned: the lead's pictures (the generator adds its <Picture 1> sentence).
    None:    no face pictures (no cast, or no lead)."""
    text = full_prompt(options, prompt)
    found, _ = mentions(options, text)
    lead = lead_of(options)
    implicit = not found and lead is not None and bool(lead.get("voice")) and not PICTURE_TAG.search(text)
    lines = dialogue(options, text, default=lead if implicit else None)
    talking = [c for c in ([lead] if implicit else found)
               if (c["images"] or c.get("voice"))]
    if voices and any(d["speaker"] is not None and d["speaker"].get("voice")
                      and any(d["speaker"]["name"] == c["name"] for c in talking) for d in lines):
        out, images, voices = _voiced_prompt(options, text, talking, lines, audio_mentioned(text), implicit)
        lang = ((options.get("identity") or {}).get("language") or "English").strip() or "English"
        said = [{"name": d["speaker"]["name"] if d["speaker"] else None, "line": d["line"]} for d in lines]
        return {"mode": "voices", "characters": talking, "images": images, "voices": voices, "language": lang,
                "lines": said, "prompt": out,
                "record": [{"name": c["name"], "desc": c.get("desc", "")} for c in talking]}
    found = [c for c in found if c["images"]]
    if found:
        tags, images = _picture_tags(found)
        seen: set[str] = set()
        inside = _in_lines(text)

        def name(m):
            c = character(options, m.group(1))
            if c is None:
                return m.group(1) if inside(m.start()) else m.group(0)
            if not c["images"] or c["name"] in seen or inside(m.start()):
                return c["name"]
            seen.add(c["name"])
            return f"{c['name']} (the person in {tags[c['name']]})"
        out = MENTION.sub(name, text)
        if (options.get("identity") or {}).get("hint", True):
            extra = [f"{c['name']} has exactly the face of the person in {tags[c['name']]}." for c in found]
            extra += [f"{c['name']}: {c['desc'].rstrip('.')}." for c in found if c.get("desc")]
            extra.append("The pictures only show who the people are; the scene, pose and framing come from the "
                         "video.")
            out = f"{out.rstrip()} {' '.join(extra)}"
        hint = (options.get("identity") or {}).get("hint", True)      # the descriptions are in the prompt only then
        return {"mode": "mention", "characters": found, "images": images, "voices": [], "prompt": out,
                "record": [{"name": c["name"], "desc": c.get("desc", "") if hint else ""} for c in found]}
    plain = MENTION.sub(lambda m: (character(options, m.group(1)) or {"name": m.group(0)})["name"], text)
    if PICTURE_TAG.search(text) and all_pictures(options):
        return {"mode": "tags", "characters": cast_of(options), "images": all_pictures(options), "voices": [],
                "prompt": plain, "record": None}
    if lead and lead["images"] and not mentions(options, text)[0]:
        return {"mode": "lead", "characters": [lead], "images": list(lead["images"]), "voices": [],
                "prompt": plain, "record": None}
    return {"mode": None, "characters": [], "images": [], "voices": [], "prompt": plain, "record": None}


def cast_problems(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    """Why clips to be made cannot get the faces their prompts name (one message per problem)."""
    out = []
    names = [c["name"] for c in cast_of(tree)]
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video:
            continue
        text = full_prompt(tree.options, node.prompt)
        found, unknown = mentions(tree.options, text)
        for word in unknown:
            close = _closest(word, names)
            out.append(f"clip #{nid[1:]} mentions @{word}, but there is no such character"
                       + (f" — did you mean @{close}?" if close else " (add it in the Cast)"))
        for c in found:
            if not c["images"] and not c.get("voice"):
                out.append(f"@{c['name']} (clip #{nid[1:]}) has no face picture or voice yet")
        for m in WRONG_TAG.finditer(text):
            out.append(f"clip #{nid[1:]} writes <{m.group(1)} {m.group(2)}>; the model only knows <Picture "
                       f"{m.group(2)}> (or mention the character as @Name)")
        use = node_cast(tree.options, node)
        if use["mode"] in ("mention", "voices") and PICTURE_TAG.search(text):
            out.append(f"clip #{nid[1:]} writes <Picture N> and mentions @Names; with @Names the app numbers the "
                       "pictures itself, so write the people only as @Name")
        if use["mode"] == "tags":
            top = max(int(re.sub(r"\D", "", t)) for t in PICTURE_TAG.findall(text))
            if top > len(use["images"]):
                n = len(use["images"])
                out.append(f"clip #{nid[1:]} names <Picture {top}>, but the cast has only {n} face "
                           f"picture{'' if n == 1 else 's'}")
        if len(use["images"]) > MAX_CAST_PICTURES:
            out.append(f"clip #{nid[1:]} needs {len(use['images'])} face pictures; the model takes at most "
                       f"{MAX_CAST_PICTURES}")
        if any(not (Path(project) / r).is_file() for r in use["images"]):
            out.append(f"a face picture of clip #{nid[1:]} is missing in refs/")
        if use["voices"]:
            if any(not (Path(project) / r).is_file() for r in use["voices"]):
                out.append(f"a voice of clip #{nid[1:]} is missing in refs/")
            n_sounds = audio_mentioned(text)
            secs = {a["file"]: float(a.get("secs") or 0) for a in audio_refs(tree)[:n_sounds]}
            secs.update({c["voice"]["file"]: float(c["voice"].get("secs") or 0) for c in cast_of(tree)
                         if c.get("voice")})
            n = n_sounds + len(use["voices"])
            total = sum(secs.get(r, 0) for r in [a["file"] for a in audio_refs(tree)[:n_sounds]] + use["voices"])
            what = (f"{len(use['voices'])} voice{'s' if len(use['voices']) != 1 else ''}"
                    + (f" and {n_sounds} sound{'s' if n_sounds != 1 else ''}" if n_sounds else ""))
            if n > MAX_AUDIO_REFS:
                out.append(f"clip #{nid[1:]} needs {what}; the model takes at most {MAX_AUDIO_REFS} sounds")
            elif total > AUDIO_REF_TOTAL + 0.05:
                out.append(f"clip #{nid[1:]} needs {what} of {total:.1f} s together; the model takes at most "
                           f"{AUDIO_REF_TOTAL:.0f} s (shorten a voice: set it again from a shorter part)")
    return list(dict.fromkeys(out))


def _closest(word: str, names: list[str]) -> str | None:
    import difflib
    hit = difflib.get_close_matches(word.lower(), [n.lower() for n in names], n=1, cutoff=0.6)
    return next((n for n in names if hit and n.lower() == hit[0]), None)


def audio_refs(tree: Tree) -> list[dict]:
    """The project's sound references ({"file", "name", "secs"}), <Audio 1> first."""
    return list(tree.options.get("audio_refs") or [])


def add_audio_reference(project: Path, source: Path, tree: Tree, start: float = 0.0,
                        name: str | None = None) -> dict:
    """Copy the sound of an audio or video file into refs/ as <Audio N> (from start, cut to what is left of the
    15 s all sounds may have together). Returns the new entry, with "cut": the source was longer."""
    refs = audio_refs(tree)
    if len(refs) >= MAX_AUDIO_REFS:
        raise ValueError(f"at most {MAX_AUDIO_REFS} sound references")
    total = sum(float(a.get("secs") or 0) for a in refs)
    left = round(AUDIO_REF_TOTAL - total, 3)
    if left < AUDIO_REF_MIN:
        raise ValueError(f"the sounds already have {total:.1f} of the {AUDIO_REF_TOTAL:.0f} s MiniMax H3 takes; "
                         "remove one first")
    length = chain.audio_seconds(Path(source))
    if length is None:
        raise ValueError(f"{Path(source).name} has no sound")
    have = length - float(start)
    if have < AUDIO_REF_MIN:
        raise ValueError(f"a sound reference needs at least {AUDIO_REF_MIN:.0f} s ({Path(source).name} has "
                         f"{max(have, 0):.1f} s)")
    secs = min(have, left)
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = {a["file"] for a in refs}
    k = 1
    while (d / f"audio{k}.wav").exists() or f"{REFS_DIR}/audio{k}.wav" in used:
        k += 1
    rel = f"{REFS_DIR}/audio{k}.wav"
    chain.to_wav(Path(source), Path(project) / rel, start, secs if have > left else None)
    real = chain.audio_seconds(Path(project) / rel) or secs
    entry = {"file": rel, "name": name or Path(source).name, "secs": round(real, 2)}
    tree.options["audio_refs"] = refs + [entry]
    return {**entry, "cut": have > left + 0.01}


def remove_audio_reference(project: Path, rel: str, tree: Tree) -> None:
    tree.options["audio_refs"] = [a for a in audio_refs(tree) if a["file"] != rel]
    target = (Path(project) / rel).resolve()
    if (Path(project) / REFS_DIR).resolve() in target.parents:
        target.unlink(missing_ok=True)


def audio_mentioned(prompt: str) -> int:
    """The highest <Audio N> a prompt names (0: none). The model numbers its sounds in order, so a clip naming
    <Audio 2> gets <Audio 1> and <Audio 2>."""
    return max((int(m) for m in AUDIO_TAG.findall(prompt or "")), default=0)


def audio_problems(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    """Why clips to be made cannot get the sounds their prompts name (one message per problem)."""
    refs = audio_refs(tree)
    out = []
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video:
            continue
        n = audio_mentioned(full_prompt(tree.options, node.prompt))
        if n > len(refs):
            out.append(f"clip #{nid[1:]} names <Audio {n}>, but there {'is' if len(refs) == 1 else 'are'} only "
                       f"{len(refs)} sound reference{'' if len(refs) == 1 else 's'}")
        elif any(not (Path(project) / a["file"]).is_file() for a in refs[:n]):
            out.append(f"a sound reference of clip #{nid[1:]} is missing in refs/")
    return out


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
    """Remove a clip's end image. The copy its current clip was made with stays until the clip is made again
    (Undo changes can go back to it)."""
    node = tree.nodes[nid]
    if node.end_image and node.end_image != (node.made or {}).get("end_image") and _in_ends(project, node.end_image):
        _unlink(Path(project) / node.end_image)
    node.end_image = None


def changed_inputs(tree: Tree, nid: str) -> list[str]:
    """What was edited on a made generated clip since it was made: prompt, seed, settings (its own),
    end image. Only what the record can restore (older records keep no own settings / end image)."""
    node = tree.nodes.get(nid)
    m = node.made if node is not None else None
    if node is None or node.is_video or not m:
        return []
    out = []
    if m.get("prompt") != node.prompt:
        out.append("prompt")
    if m.get("seed") != node.seed:
        out.append("seed")
    if "overrides" in m and m["overrides"] != (node.overrides or {}):
        out.append("settings")
    if "end_image" in m and m["end_image"] != node.end_image:
        out.append("end image")
    if "keyframes" in m and clean_keyframes(m["keyframes"]) != node.keyframes:
        out.append("keyframes")
    return out


def revert(project: Path, tree: Tree, nid: str) -> list[str]:
    """Undo the edits of a made generated clip: back to the prompt, seed, own settings and end image it was
    made with. Returns what was restored."""
    done = changed_inputs(tree, nid)
    node = tree.nodes[nid]
    m = node.made
    if "prompt" in done:
        node.prompt = m["prompt"]
    if "seed" in done:
        node.seed = int(m["seed"])
    if "settings" in done:
        node.overrides = copy.deepcopy(m["overrides"])
    if "end image" in done:
        clear_end_image(project, tree, nid)       # the newer image goes (unless it is the made one)
        node.end_image = m["end_image"]
    if "keyframes" in done:
        newer = {k["image"] for k in node.keyframes}
        node.keyframes = clean_keyframes(m["keyframes"])
        for rel in newer:
            _drop_keyframe_file(project, node, rel)
    return done


def _revert_fields(tree: Tree, nid: str) -> None:
    """Undo the edits of a made clip in the tree only (no files touched): for trying out what freezing finds."""
    node, done = tree.nodes[nid], changed_inputs(tree, nid)
    m = node.made
    if "prompt" in done:
        node.prompt = m["prompt"]
    if "seed" in done:
        node.seed = int(m["seed"])
    if "settings" in done:
        node.overrides = copy.deepcopy(m["overrides"])
    if "end image" in done:
        node.end_image = m["end_image"]
    if "keyframes" in done:
        node.keyframes = clean_keyframes(m["keyframes"])


def freeze_set(tree: Tree, nid: str) -> list[Node]:
    """What freezing nid freezes: it and every clip / video before it (so every seam up to it stays as made), and
    the real video each of those generated clips leads into (the clip ends at its first frame)."""
    out = tree.path_to(nid)
    for n in list(out):
        vc = tree.video_child(n.id)
        if vc is not None and vc not in out:
            out.append(vc)
    return out


def freeze_edits(tree: Tree, nid: str) -> dict[str, list[str]]:
    """The not yet made edits freezing nid undoes: {clip id: ["prompt", ...]}."""
    return {n.id: e for n in freeze_set(tree, nid) if (e := changed_inputs(tree, n.id))}


def freeze_problem(project: Path, tree: Tree, nid: str) -> str | None:
    """Why nid cannot be frozen (None: it can): a clip or video of it is not made, or one no longer joins the one
    before it as it was made (then it has to be made again first, else the frozen path would jump there)."""
    trial = copy.deepcopy(tree)
    nodes = freeze_set(trial, nid)
    ck = Checker(project, trial)
    for n in nodes:
        if not ck._files_there(n):
            return f"{'video' if n.is_video else 'clip'} #{n.id[1:]} is not made yet (or its files are missing)"
    for n in nodes:
        if not n.is_video:
            _revert_fields(trial, n.id)
        n.frozen = True
    ck = Checker(project, trial)
    for n in nodes:
        why = ck.seam_problem(n)
        if why:
            return f"clip #{n.id[1:]}: {why}; make it again first"
    return None


def freeze(project: Path, tree: Tree, nid: str) -> list[str]:
    """Keep nid as it is made, and with it every clip before it: they count as done whatever the project's
    settings (size, speed LoRA, steps ...) say, are not made again and cannot be edited until unfrozen. Their not
    yet made edits are undone first. Returns the ids frozen now. Raises ValueError (freeze_problem)."""
    why = freeze_problem(project, tree, nid)
    if why:
        raise ValueError(why)
    out = []
    for n in freeze_set(tree, nid):
        if not n.is_video:
            revert(project, tree, n.id)
        if not n.frozen:
            n.frozen = True
            out.append(n.id)
    return out


def unfreeze(tree: Tree, nid: str) -> list[str]:
    """nid follows the settings again, and so does every frozen clip after it (a frozen clip needs the ones before
    it frozen) and a frozen clip that ends at it (a video). Returns the ids unfrozen."""
    node = tree.nodes[nid]
    if not node.frozen:
        return []
    node.frozen = False
    return [nid] + heal_frozen(tree)


def heal_frozen(tree: Tree) -> list[str]:
    """Unfreeze what may not stay frozen: a clip after one that is not frozen, a generated clip whose video after it
    is not frozen, one that was never made. Returns those ids."""
    out, again = [], True
    while again:
        again = False
        for n in tree.nodes.values():
            if not n.frozen:
                continue
            p = tree.nodes.get(n.parent) if n.parent is not None else None
            vc = tree.video_child(n.id)
            if not isinstance(n.made, dict) or (p is not None and not p.frozen) or (vc is not None and not vc.frozen):
                n.frozen = False
                out.append(n.id)
                again = True
    return out


def frozen_note(tree: Tree, node: Node, base: dict | None) -> str:
    """How a frozen clip differs from what the settings would make now, in a few words ('' when it does not, or
    the record does not say)."""
    if base is None or not node.made:
        return ""
    if node.is_video:
        k = node.made.get("key") or {}
        same = (k.get("width"), k.get("height")) == (base.get("width"), base.get("height"))
        return "" if same or not k.get("width") else f"made at {k['width']}×{k['height']}"
    s = node.made.get("settings") or {}
    parts = []
    if s.get("width") and (s.get("width"), s.get("height")) != (base.get("width"), base.get("height")):
        parts.append(f"{s['width']}×{s['height']}")

    def speed(d: dict) -> str:
        if d.get("hyperflow"):
            return "HyperFlow"
        if d.get("pika"):
            return "Pika"
        return "Turbo" if d.get("turbo") else "no speed LoRA"
    now = {"turbo": base.get("turbo"), "hyperflow": base.get("hyperflow"), "pika": base.get("pika")}
    if speed(s) != speed(now):
        parts.append(speed(s))
    return ("made at " + ", ".join(parts)) if parts else ""


def copy_end_image(project: Path, tree: Tree, src: str, dst: str) -> None:
    """A variant gets its own copy of the end image (deleting one never touches the other's)."""
    rel = tree.nodes[src].end_image
    if rel and (Path(project) / rel).is_file() and tree.video_child(dst) is None:
        set_end_image(project, tree, dst, Path(project) / rel)


def clean_keyframes(raw) -> list[dict]:
    """Keyframes as tree.json holds them: {"image": "ends/...", "at": s > 0, "note"}, sorted by time (damaged
    entries are left out)."""
    out = []
    for k in raw if isinstance(raw, list) else []:
        if not isinstance(k, dict) or not isinstance(k.get("image"), str) or not k["image"].startswith(ENDS_DIR + "/"):
            continue
        at = k.get("at")
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not at > 0:
            continue
        out.append({"image": k["image"], "at": round(float(at), 3), "note": str(k.get("note") or "")})
    return sorted(out, key=lambda k: k["at"])


def keyframe_files(node: Node) -> set[str]:
    """The keyframe pictures a node and its made clip use (project-relative)."""
    return {k["image"] for k in node.keyframes} | {k["image"] for k in (node.made or {}).get("keyframes") or []
                                                    if isinstance(k, dict) and k.get("image")}


def _drop_keyframe_file(project: Path, node: Node, rel: str) -> None:
    """Delete a keyframe picture no longer used (the made clip's stay until it is made again: Undo)."""
    made = {k.get("image") for k in (node.made or {}).get("keyframes") or [] if isinstance(k, dict)}
    if rel not in made and rel not in {k["image"] for k in node.keyframes} and _in_ends(project, rel):
        _unlink(Path(project) / rel)


def set_keyframe(project: Path, tree: Tree, nid: str, image: Path | None, at: float, note: str | None = None,
                 index: int | None = None) -> int:
    """Add a middle keyframe (index None) or change one (a new picture, time and / or note; image None keeps the
    picture). The picture is copied in (a new name every time). Returns its index after sorting by time."""
    node = tree.nodes[nid]
    if node.is_video:
        raise ValueError("a real video has no keyframes")
    if not at > 0:
        raise ValueError("a keyframe needs a time after the start")
    old = dict(node.keyframes[index]) if index is not None else None
    rel = old["image"] if old else None
    if image is not None:
        d = Path(project) / ENDS_DIR
        d.mkdir(parents=True, exist_ok=True)
        rel = f"{ENDS_DIR}/{nid}_k{uuid.uuid4().hex[:8]}.png"
        chain.to_png(Path(image), Path(project) / rel)
    if rel is None:
        raise ValueError("a new keyframe needs a picture")
    entry = {"image": rel, "at": round(float(at), 3), "note": (old or {}).get("note", "") if note is None else note}
    kfs = [k for i, k in enumerate(node.keyframes) if i != index] + [entry]
    node.keyframes = sorted(kfs, key=lambda k: k["at"])
    if old and old["image"] != rel:
        _drop_keyframe_file(project, node, old["image"])
    return node.keyframes.index(entry)


def remove_keyframe(project: Path, tree: Tree, nid: str, index: int) -> None:
    node = tree.nodes[nid]
    gone = node.keyframes.pop(index)
    _drop_keyframe_file(project, node, gone["image"])


def copy_keyframes(project: Path, tree: Tree, src: str, dst: str) -> None:
    """A variant gets its own copies of the keyframe pictures."""
    for k in list(tree.nodes[src].keyframes):
        if (Path(project) / k["image"]).is_file():
            set_keyframe(project, tree, dst, Path(project) / k["image"], k["at"], k["note"])


def keyframe_frame(at: float) -> int:
    return int(round(float(at) * chain.FPS))


def missing_keyframes(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    return [nid for nid in todo if any(not (Path(project) / k["image"]).is_file() for k in tree.nodes[nid].keyframes)]


def keyframe_problem(tree: Tree, checker: "Checker", todo: list[str]) -> str | None:
    """Why a clip's keyframes cannot be used (one past its end, on its first or last frame, inside a motion guide,
    two on the same frame), or None. Checked with the length the clip will have."""
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video or not node.keyframes:
            continue
        s = checker.settings(nid) or {}
        frames = int(s.get("frames") or clip_length(tree, node))
        guide = int(s.get("guide") or 0)
        lo = guide if guide and node.parent is not None else 1
        hi = frames - 2 - (guide if guide and tree.video_child(nid) is not None else 0)
        seen = set()
        for k in node.keyframes:
            f = keyframe_frame(k["at"])
            where = f"clip #{nid[1:]}'s keyframe at {k['at']:.2f} s"
            if f > hi:
                return (f"{where} is at or past its end ({frames / chain.FPS:.2f} s{', minus its motion guide' if hi < frames - 2 else ''}): "
                        "move it earlier or make the clip longer")
            if f < lo:
                return f"{where} is {'inside its motion guide' if lo > 1 else 'on its first frame'}: move it later"
            if f in seen:
                return f"{where} falls on the same frame as another one: move one of them"
            seen.add(f)
    return None


def end_image_used(tree: Tree, node: Node) -> str | None:
    """The end image a generated clip is made with: its own, unless a video follows (that one's first frame)."""
    if node.is_video or not node.end_image or tree.video_child(node.id) is not None:
        return None
    return node.end_image


def missing_end_images(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    return [nid for nid in todo if end_image_used(tree, tree.nodes[nid])
            and not (Path(project) / tree.nodes[nid].end_image).is_file()]


def prepared_start(project: Path, width: int, height: int) -> Path | None:
    """The start image prepared for a canvas (chain.prepare_image): start_<w>x<h>.png in the project, made again
    when start.png changes. None without a start image."""
    src = Path(project) / START_FILE
    if not src.is_file():
        return None
    dst = Path(project) / f"start_{int(width)}x{int(height)}.png"
    rec = dst.with_suffix(".json")
    sig = chain.file_signature(src)
    try:
        if dst.is_file() and json.loads(rec.read_text(encoding="utf-8")).get("source") == sig:
            return dst
    except (OSError, ValueError):
        pass
    for old in Path(project).glob("start_*x*.png"):        # (one prepared copy at a time)
        if old != dst:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    chain.prepare_image(src, dst, width, height)
    rec.write_text(json.dumps({"source": sig}), encoding="utf-8")
    return dst


def clear_start_image(project: Path, tree: Tree) -> None:
    """No start image any more: start.png and its prepared copies go (the first clip then starts from its text, the
    size follows the first video or the Aspect setting)."""
    _unlink(Path(project) / START_FILE)
    for f in Path(project).glob("start_*x*.png"):
        _unlink(f)
        _unlink(f.with_suffix(".json"))
    tree.image = None


def set_start_image(project: Path, image: Path, tree: Tree) -> None:
    chain.to_png(Path(image), Path(project) / START_FILE)
    tree.image = str(Path(image).resolve())


# ----------------------------------------------------------------------------- state

_SRC_INFO: dict[tuple, dict] = {}
_IMG_SIZE: dict[tuple, tuple[int, int]] = {}


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


ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "5:4": (5, 4), "3:2": (3, 2),
           "2:3": (2, 3), "21:9": (21, 9)}


def size_origin(project: Path, tree: "Tree | None") -> str:
    """What the video size follows: 'image' (the start image), 'video' (the first video) or 'aspect' (the project's
    aspect setting: a project without either)."""
    if (Path(project) / START_FILE).is_file():
        return "image"
    if tree is not None and any(n.is_video for n in tree.nodes.values()):
        return "video"
    return "aspect"


def size_source(project: Path, tree: "Tree | None") -> tuple[int, int] | None:
    """The picture whose aspect ratio the video size follows: the start image, else the first video of the tree
    (by number), else the project's aspect setting (ASPECTS; a first clip then starts from its text). None only
    without a tree and a start image."""
    start = Path(project) / START_FILE
    try:
        st = start.stat()
    except OSError:
        st = None
    if st is not None:
        key = (str(start), st.st_size, st.st_mtime_ns)
        if key not in _IMG_SIZE:                     # ffprobe once per version of the image, not per check
            _IMG_SIZE[key] = chain.image_size(start)
        return _IMG_SIZE[key]
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
        a = ASPECTS.get(str(tree.options.get("aspect") or "16:9"), ASPECTS["16:9"])
        return a[0] * 120, a[1] * 120
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
        self._status: dict[str, str] = {}
        self._settings: dict | None = None
        self._audio: list[dict] | None = None
        self._pictures: dict[str, str | None] = {}

    def sha(self, path: Path | None) -> str | None:
        """SHA-1 of a frame, cached across checks by (path, size, date): a check after every edit must not read
        every clip's frame again (slow on a network share)."""
        return chain.file_signature(path) if path is not None else None

    def base(self) -> dict | None:
        if self._settings is None:
            self._settings = current_settings(self.project, self.tree.options, self.tree)
        return self._settings

    def audio_pool(self) -> list[dict]:
        """The sound references with their content hash (None: the file is missing)."""
        if self._audio is None:
            self._audio = [{"file": a["file"], "sha1": self.sha(self.project / a["file"])}
                           for a in audio_refs(self.tree)]
        return self._audio

    def picture_sha(self, rel: str) -> str | None:
        """The content hash of a face picture (None: missing), cached for this round."""
        if rel not in self._pictures:
            self._pictures[rel] = self.sha(self.project / rel)
        return self._pictures[rel]

    def settings(self, nid: str) -> dict | None:
        base = self.base()
        node = self.tree.nodes[nid]
        span = source_sound_span(self.tree, node, self.project) \
            if not node.is_video and (node.overrides or {}).get("source_sound") else None
        if span is not None and (node.overrides or {}).get("sound_over"):
            span = {**span, "overlay": True}       # laid over the clip afterwards: the plain model, no lip sync
        return None if base is None else node_settings(base, self.tree.options, node, self.audio_pool(),
                                                       self.picture_sha, source_sound=span)

    def start_frame(self, node: Node) -> Path | None:
        """The frame a generated clip starts from: the parent's last frame, the start image, or None (a first
        clip without a start image)."""
        if node.parent is None:
            start = self.project / START_FILE
            return start if start.is_file() else None
        parent = self.tree.nodes[node.parent]
        if not parent.made:
            return node_files(self.project, node.parent)[1]
        return handover_file(self.project, self.tree, parent) or self.project / parent.made["frame"]

    def end_key(self, node: Node):
        """What a generated clip's end depends on: the key of the video it leads into (False: that video is not
        ready), its own end image ({"image": sha1}), or None (a free end)."""
        vc = self.tree.video_child(node.id)
        if vc is None:
            own = end_image_used(self.tree, node)
            return {"image": self.sha(self.project / own)} if own else None
        return vc.made.get("key") if self.status(vc.id) == "done" else False

    def keyframe_key(self, node: Node) -> list | None:
        """What a generated clip's middle keyframes are: [[picture sha1, frame, note], ...], None without any."""
        if node.is_video or not node.keyframes:
            return None
        return [[self.sha(self.project / k["image"]), keyframe_frame(k["at"]), k["note"]] for k in node.keyframes]

    def guide_from(self, node: Node) -> str | None:
        """The parent clip a start motion guide is cut from (its file name: every new clip gets a new one), or None
        without a start guide."""
        s = self.settings(node.id)
        if node.parent is None or not s or not int(s.get("guide") or 0):
            return None
        parent = self.tree.nodes[node.parent]
        if not parent.made:
            return None
        k = handover_index(self.tree, parent)            # an earlier hand-over: other guide frames
        v = parent.made.get("video")
        return None if v is None else v + (f"@{k}" if k is not None else "")

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
        elif node.frozen:                          # kept as made, whatever the settings say now
            st = "done"
        elif node.is_video:                        # a real video depends only on its file, range and the size
            base = self.base()
            st = "done" if base is not None and _same_key(m.get("key"), video_key(node, base)) else "changed"
        elif (m.get("prompt") != node.prompt or m.get("seed") != node.seed
              or _old_record(m.get("settings")) != self.settings(nid)
              or (node.parent is not None and self.status(node.parent) != "done")
              or m.get("start_sha1") != self.sha(self.start_frame(node))
              or m.get("end_key") != self.end_key(node)
              or m.get("keyframe_key") != self.keyframe_key(node)
              or m.get("guide_from") != self.guide_from(node)):
            st = "changed"
        else:
            st = "done"
        self._status[nid] = st
        return st


    def seam_problem(self, node: Node) -> str | None:
        """Why a made generated clip no longer joins the clips around it as made (None: it does): what it starts
        from, its motion guide's source or its end changed. Its own edits are not seams (Undo changes fixes them)."""
        if node.is_video or not node.made:
            return None
        m = node.made
        if m.get("start_sha1") != self.sha(self.start_frame(node)):
            return "it no longer starts at the frame it was made from"
        if m.get("guide_from") != self.guide_from(node):
            return "the clip its motion guide comes from was made again"
        if m.get("end_key") != self.end_key(node):
            return "the video it ends at changed"
        return None

    def why(self, nid: str) -> str:
        """Why a clip counts as changed, in words ('' if it does not)."""
        if self.status(nid) != "changed":
            return ""
        node = self.tree.nodes[nid]
        m = node.made
        if node.is_video:
            return "its range, file or the video size changed"
        edited = changed_inputs(self.tree, nid)
        if edited:
            return f"its {' and '.join(edited)} changed"
        s = m.get("settings")
        if isinstance(s, dict) and "backend" not in s:
            return "it was made with the old generator"
        if _old_record(s) != self.settings(nid):
            return "its settings changed (the project's or its own)"
        if node.parent is not None and self.status(node.parent) != "done":
            p = self.tree.nodes[node.parent]
            return f"{'video' if p.is_video else 'clip'} #{p.id[1:]} before it changes"
        if m.get("start_sha1") != self.sha(self.start_frame(node)):
            return "the frame it starts from changed"
        if m.get("end_key") != self.end_key(node):
            return "its end (the video after it or its end image) changed"
        if m.get("keyframe_key") != self.keyframe_key(node):
            return "its keyframes changed"
        if m.get("guide_from") != self.guide_from(node):
            return "the clip its motion guide comes from changed"
        return "something it depends on changed"


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
    turbo = options.get("turbo") or {}
    turbo_on = bool(turbo.get("on") and turbo.get("lora"))
    hyper = options.get("hyperflow") or {}
    hyper_on = bool(hyper.get("on")) and not turbo_on
    pika = options.get("pika") or {}
    pika_on = bool(pika.get("on")) and not turbo_on and not hyper_on
    style = options.get("style") or {}
    return chain.settings_dict(model=options.get("model", DEFAULT_OPTIONS["model"]), lora=None,
                               strength=1.0, frames=options.get("frames", 73),
                               steps=(int(turbo.get("steps", 6)) if turbo_on else chain.HYPERFLOW_STEPS if hyper_on
                                      else chain.PIKA_STEPS if pika_on else options.get("steps", 28)),
                               hyperflow=float(hyper.get("strength", 1.0)) if hyper_on else None,
                               pika=float(pika.get("strength", 1.0)) if pika_on else None,
                               width=w, height=h, audio=options.get("audio", True),
                               turbo=(turbo["lora"], turbo.get("strength", 1.0)) if turbo_on else None,
                               cfg=float((turbo if turbo_on else hyper if hyper_on else pika if pika_on
                                          else options).get("cfg", 1.0)),
                               backend="comfyui",
                               style=(style.get("prefix") or "", style.get("suffix") or ""))


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


REPEAT_MIN = 0.5                    # s: a shorter repeated sound would only buzz
TRACK_PREFIX = "track:"             # a clip's source_sound override "track:<id>": that audio track
TRACK_SOUND_TEXT = ("<Audio {k}> is the complete audio track of the target video, reused 1:1: every word, voice "
                    "and sound comes from <Audio {k}>, and anyone speaking moves their lips in sync with it.")
REPEAT_SOUND_TEXT = ("<Audio {k}> is the complete audio track of the target video, reused 1:1: every sound of the "
                     "video comes from <Audio {k}>, and what is seen matches it.")
SOURCE_SOUND_TEXT = ("<Audio {k}> is the complete original audio track of the target video, reused 1:1: every word, "
                     "voice and sound comes from <Audio {k}>, and the speaking person's lips move in sync with it.")


def uses_ref_model(settings: dict) -> bool:
    """Whether a clip is made with H3's reference model: face pictures, sounds or voices, or the real video's sound
    (all go to MiniMaxH3ReferenceToVideo)."""
    ss = settings.get("source_sound")
    return bool(settings.get("refs") or settings.get("audio_refs")
                or (ss and not (isinstance(ss, dict) and ss.get("overlay"))))


def source_video(tree: Tree, node: Node) -> Node | None:
    """The real video a generated clip continues (the nearest one before it on its path), or None."""
    path = tree.path_to(node.id)
    return next((n for n in reversed(path[:-1]) if n.is_video), None)


def source_sound_span(tree: Tree, node: Node, project: Path | None = None) -> dict | None:
    """Where in its real video's file a clip's sound continues ("Continue the video's sound"): the part right after
    what the path has shown so far. {"file", "start", "secs", "video"} (seconds in the file), or None without a real
    video before the clip. A generated clip's first frame (or its carry-over frames) repeats the end of the clip
    before, so its sound starts that much earlier; a clip that hands over early counts up to that frame.
    With "Repeat the sound of the clip before" (source_sound == "previous") see previous_sound_span."""
    mode = (node.overrides or {}).get("source_sound")
    if mode == "previous":
        return previous_sound_span(tree, node, project)
    if isinstance(mode, str) and mode.startswith(TRACK_PREFIX):
        return track_sound_span(tree, node)
    path = tree.path_to(node.id)
    vi = max((i for i, n in enumerate(path[:-1]) if n.is_video), default=None)
    if vi is None:
        return None
    v = path[vi]
    src = v.source or {}
    t = float(src["start"]) + clip_length(tree, v) / chain.FPS   # the frames the video really has (not the end)
    for p in path[vi + 1:-1]:
        n = made_frames(p) or clip_length(tree, p)
        h = handover_index(tree, p, n)
        t += ((h + 1 if h is not None else n) - _join_lead(tree, p)) / chain.FPS
    start = t - _join_lead(tree, node) / chain.FPS
    return {"file": str(src["file"]), "start": round(max(0.0, start), 3),
            "secs": round(clip_length(tree, node) / chain.FPS, 3), "video": v.id}


def _join_lead(tree: Tree, node: Node) -> int:
    """Frames of a generated clip (after another clip) left out at its start when the path is joined: its first
    frame or its carry-over frames repeat the clip before; none with "keep the repeated frames"."""
    return 0 if tree.options.get("keep_repeated") else max(1, _guide_of(node, planned=True))


# ---- audio tracks: imported sound (cut to a range), played by clips one after another
def tracks(tree: Tree) -> list[dict]:
    """The project's audio tracks ({"id", "name", "file", "secs", "source", "start", "end"}); shared by the film's
    scenes. file: the cut sound as WAV in refs/ (a new name for every cut)."""
    return [t for t in (tree.options.get("tracks") or []) if isinstance(t, dict) and t.get("id") and t.get("file")]


def track_of(tree: Tree, tid: str) -> dict | None:
    return next((t for t in tracks(tree) if t["id"] == tid), None)


def track_id(node: Node) -> str | None:
    """The audio track a clip plays (its Sound setting), or None."""
    mode = (node.overrides or {}).get("source_sound")
    return mode[len(TRACK_PREFIX):] if isinstance(mode, str) and mode.startswith(TRACK_PREFIX) else None


def _cut_track(project: Path, source: Path, start: float, end: float | None) -> tuple[str, float]:
    length = chain.audio_seconds(Path(source))
    if length is None:
        raise ValueError(f"{Path(source).name} has no sound")
    start = max(0.0, float(start))
    end = length if end is None else min(float(end), length)
    if end - start < AUDIO_REF_MIN:
        raise ValueError(f"an audio track needs at least {AUDIO_REF_MIN:.0f} s (the range has {max(0, end - start):.1f} s)")
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    rel = f"{REFS_DIR}/track_{uuid.uuid4().hex[:8]}.wav"
    chain.to_wav(Path(source), Path(project) / rel, start, end - start)
    return rel, round(chain.audio_seconds(Path(project) / rel) or (end - start), 3)


def add_track(project: Path, tree: Tree, source: Path, start: float = 0.0, end: float | None = None,
              name: str | None = None) -> dict:
    """Cut the sound of an audio or video file (start-end, seconds) into refs/ as a new audio track."""
    rel, secs = _cut_track(project, source, start, end)
    have = {t["id"] for t in tracks(tree)}
    tid = "t" + uuid.uuid4().hex[:6]               # never reused: a removed track's clips keep pointing nowhere
    while tid in have:
        tid = "t" + uuid.uuid4().hex[:6]
    entry = {"id": tid, "name": name or Path(source).stem, "file": rel, "secs": secs, "source": str(source),
             "start": round(float(start), 3), "end": round(float(start) + secs, 3)}
    tree.options["tracks"] = tracks(tree) + [entry]
    return entry


def recut_track(project: Path, tree: Tree, tid: str, source: Path, start: float, end: float | None,
                name: str | None = None) -> dict:
    """Cut an audio track again (another range, or another file); the clips playing it count as changed."""
    old = track_of(tree, tid)
    if old is None:
        raise ValueError("this audio track is gone")
    rel, secs = _cut_track(project, source, start, end)
    entry = {**old, "name": name or old["name"], "file": rel, "secs": secs, "source": str(source),
             "start": round(float(start), 3), "end": round(float(start) + secs, 3)}
    tree.options["tracks"] = [entry if t["id"] == tid else t for t in tracks(tree)]
    return entry


def remove_track(project: Path, tree: Tree, tid: str) -> None:
    """Take a track out of the list; its file stays until no scene lists it (unused_track_files)."""
    tree.options["tracks"] = [t for t in tracks(tree) if t["id"] != tid]


def unused_track_files(project: Path, scene_dirs: list[Path], open_tree: Tree | None = None) -> list[Path]:
    """Cut track files in refs/ that the saved tree of no scene lists any more (a track cut again or removed), nor
    the open tree. A scene whose tree cannot be read keeps everything."""
    listed = {tr["file"] for tr in tracks(open_tree)} if open_tree is not None else set()
    for d in scene_dirs:
        try:
            t = load(Path(d))
        except Exception:  # noqa: BLE001
            return []
        listed |= {tr["file"] for tr in tracks(t)}
    refs = Path(project) / REFS_DIR
    return sorted(f for f in refs.glob("track_*.wav") if f"{REFS_DIR}/{f.name}" not in listed) \
        if refs.is_dir() else []


def track_users(tree: Tree, tid: str) -> list[str]:
    return [n.id for n in tree.nodes.values() if not n.is_video and track_id(n) == tid]


def track_sound_span(tree: Tree, node: Node) -> dict | None:
    """Which part of its audio track a clip plays: the track goes on where the clip before stopped when that clip
    plays the same track, else it starts at the track's beginning. Laid so the joined path hears it without a gap
    or a repeat: a clip's first frame (or carry-over frames) is left out when joined, so its sound starts that much
    earlier; the first clip of a run gets that much silence first (pad). Past the track's end: silence.
    {"file", "start", "secs", "pad", "mode": "track", "track", "from"} (from: the clip it goes on from, or None);
    None when the track is gone."""
    tid = track_id(node)
    track = track_of(tree, tid) if tid else None
    if track is None:
        return None
    path = tree.path_to(node.id)

    def lead(k: int) -> int:                       # frames of path[k] left out at its start when joined
        return 0 if k == 0 else _join_lead(tree, path[k])
    i = len(path) - 1
    while i > 0 and not path[i - 1].is_video and track_id(path[i - 1]) == tid:
        i -= 1                                     # the run of clips playing this track, up to this one
    t = 0
    for k in range(i, len(path) - 1):
        p = path[k]
        n = made_frames(p) or clip_length(tree, p)
        h = handover_index(tree, p, n)
        t += (h + 1 if h is not None else n) - lead(k)
    start = (t - lead(len(path) - 1)) / chain.FPS
    return {"file": track["file"], "start": round(max(0.0, start), 3), "pad": round(max(0.0, -start), 3),
            "secs": round(clip_length(tree, node) / chain.FPS, 3), "mode": "track", "track": tid,
            "from": path[-2].id if i < len(path) - 1 else None}


def previous_sound_span(tree: Tree, node: Node, project: Path | None = None) -> dict | None:
    """'Repeat the sound of the clip before': the sound of the clip before this one as the joined path plays it
    (without its carry-over frames and the frames after an early hand-over), repeated as often as this clip needs.
    It is laid so that the joined path hears it from its beginning: this clip's own first frame (or carry-over
    frames) is left out when joined, so its sound starts that much before the loop's end (offset).
    {"file", "start", "secs", "loop", "offset", "video", "mode": "previous"}; a generated clip's file is named
    relative to the project folder (its names are unique per version; see sound_file), "" while it is not made yet.
    None for a first clip."""
    path = tree.path_to(node.id)
    if len(path) < 2:
        return None
    k = len(path) - 2
    p = path[k]
    secs = round(clip_length(tree, node) / chain.FPS, 3)
    lead = _join_lead(tree, node) / chain.FPS
    if p.is_video:
        file, base = str((p.source or {}).get("file") or ""), float((p.source or {}).get("start") or 0.0)
    elif p.made:
        file, base = str(p.made["video"]), 0.0
    else:
        file, base = "", 0.0
    n = (made_frames(p) if not p.is_video else 0) or clip_length(tree, p)
    a, b = path_cuts(tree, path, bool(tree.options.get("keep_repeated")), planned=not (p.made or p.is_video))[k]   # (as planned until it is made)
    loop = max(1, n - a - b) / chain.FPS
    return {"file": file, "start": round(base + a / chain.FPS, 3), "secs": secs, "loop": round(loop, 3),
            "offset": round((loop - lead) % loop, 3), "video": p.id, "mode": "previous"}


def sound_file(project: Path | None, sound: dict) -> Path:
    """The file of a clip's soundtrack (source_sound span or record): a real video's own path, or a clip of the
    project (relative)."""
    f = Path(sound["file"])
    return f if f.is_absolute() or project is None else Path(project) / f


def source_sound_problems(tree: Tree, todo: list[str], project: Path | None = None) -> list[str]:
    """Why clips that continue their real video's sound (or repeat the clip before's) cannot be made (one message
    per problem)."""
    out = []
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video or not (node.overrides or {}).get("source_sound"):
            continue
        span = source_sound_span(tree, node, project)
        over = bool((node.overrides or {}).get("sound_over"))   # laid over: not a sound for the model
        tid = track_id(node)
        if tid is not None:
            if span is None:
                out.append(f"clip #{nid[1:]} plays an audio track that was removed; choose another sound")
                continue
            track = track_of(tree, tid)
            if not sound_file(project, span).is_file():
                out.append(f"the file of the audio track “{track['name']}” is missing ({Path(span['file']).name})")
                continue
            if span["start"] >= float(track["secs"]) - 0.05:
                out.append(f"the audio track “{track['name']}” ({float(track['secs']):.1f} s) has ended before clip "
                           f"#{nid[1:]} (it would start at {span['start']:.1f} s); choose another sound")
                continue
            if not over and span["secs"] < AUDIO_REF_MIN:
                out.append(f"clip #{nid[1:]} is shorter than {AUDIO_REF_MIN:.0f} s, the least sound the model "
                           "takes; make it longer to play the audio track")
            if not over:
                out += _sound_room(tree, node, span)
            continue
        if (node.overrides or {}).get("source_sound") == "previous":
            if span is None:
                out.append(f"clip #{nid[1:]} repeats the sound of the clip before, but it is the first clip")
                continue
            before = tree.nodes[span["video"]]
            if not over and span["secs"] < AUDIO_REF_MIN:
                out.append(f"clip #{nid[1:]} is shorter than {AUDIO_REF_MIN:.0f} s, the least sound the model "
                           "takes; make it longer to repeat the sound of the clip before")
            if span["loop"] < REPEAT_MIN:
                out.append(f"the clip before clip #{nid[1:]} plays only {span['loop']:.2f} s in the film, too short "
                           f"to repeat (at least {REPEAT_MIN} s)")
            if not over:
                out += _sound_room(tree, node, span)
            if not before.is_video and before.id in todo:   # made (again) first in this run: checked then
                continue
            if not span["file"]:
                out.append(f"clip #{nid[1:]} repeats the sound of clip #{before.id[1:]}, which is not made yet")
                continue
            if not sound_file(project, span).is_file():
                out.append(f"the video file of clip #{nid[1:]}'s sound is missing ({Path(span['file']).name})")
                continue
            out += _repeat_sound_problems(project, node, span)
            continue
        if span is None:
            out.append(f"clip #{nid[1:]} continues a real video's sound, but there is no real video before it")
            continue
        if not Path(span["file"]).is_file():
            out.append(f"the video file of clip #{nid[1:]}'s sound is missing ({Path(span['file']).name})")
            continue
        have = chain.audio_seconds(Path(span["file"]))
        need = span["start"] + span["secs"]
        if not over and span["secs"] < AUDIO_REF_MIN:
            out.append(f"clip #{nid[1:]} is shorter than {AUDIO_REF_MIN:.0f} s, the least sound the model takes; "
                       "make it longer to continue the video's sound")
        if have is None:
            out.append(f"{Path(span['file']).name} has no sound, so clip #{nid[1:]} cannot continue it")
        elif have < need - 0.05:
            out.append(f"clip #{nid[1:]} needs the sound of {Path(span['file']).name} up to "
                       f"{int(need // 60)}:{need % 60:04.1f}, but it ends at {int(have // 60)}:{have % 60:04.1f}; "
                       "make the clip shorter")
        if not over:
            out += _sound_room(tree, node, span)
    return out


def _repeat_sound_problems(project: Path | None, node: Node, span: dict) -> list[str]:
    """The clip before's file must have sound, about as long as the part that is repeated."""
    have = chain.audio_seconds(sound_file(project, span))
    who = f"clip #{(span.get('video') or node.parent or '?')[1:]}"
    if have is None:
        return [f"{who} has no sound, so clip #{node.id[1:]} cannot repeat it"]
    if have < span["start"] + span["loop"] - 0.25:
        return [f"{who}'s sound ends at {have:.1f} s, before the part clip #{node.id[1:]} repeats "
                f"(up to {span['start'] + span['loop']:.1f} s)"]
    return []


def _sound_room(tree: Tree, node: Node, span: dict) -> list[str]:
    """The model's limits on sound references, with the clip's soundtrack as one more."""
    nid = node.id
    n_sounds = audio_mentioned(full_prompt(tree.options, node.prompt))
    voices = node_cast(tree.options, node).get("voices") or []
    secs = sum(float(a.get("secs") or 0) for a in audio_refs(tree)[:n_sounds]) + span["secs"] + \
        sum(float(c["voice"].get("secs") or 0) for c in cast_of(tree) if c.get("voice")
            and c["voice"]["file"] in voices)
    if n_sounds + len(voices) + 1 > MAX_AUDIO_REFS:
        return [f"clip #{nid[1:]} has too many sounds with its soundtrack (the model takes "
                f"{MAX_AUDIO_REFS}); remove a sound or voice from its prompt"]
    if secs > AUDIO_REF_TOTAL + 0.05:
        return [f"clip #{nid[1:]} needs {secs:.1f} s of sound with its soundtrack (the model takes "
                f"{AUDIO_REF_TOTAL:.0f} s); make it shorter or leave out its other sounds"]
    return []


def lora_trigger_words(paths: list[str]) -> list[str] | None:
    """The trigger words of a LoRA stack (the app's trigger database; None if it cannot be read)."""
    try:
        from . import triggers
        return triggers.stack_words(paths)
    except Exception:  # noqa: BLE001
        return None                                # unknown (e.g. the database cannot be read right now)


def _stack_triggers(options: dict, node: Node, stack: list) -> list[str] | None:
    """A clip's trigger words; while the database cannot be read, those it was made with (no false CHANGED)."""
    if not stack or not options.get("lora_triggers", True):
        return None
    words = lora_trigger_words([p for p, _s in stack])
    if words is None:
        words = (((node.made or {}).get("settings")) or {}).get("lora_triggers")
    return words


def model_prompt(options: dict, node: Node, settings: dict | None) -> str:
    """What the model gets for a generated clip: the cast's prompt (clip_cast) with its LoRAs' trigger words, and
    with "Continue the video's sound" the sentence that makes the last sound reference its soundtrack."""
    from .triggers import with_triggers
    prompt = _camera.apply(node.prompt, (settings or {}).get("camera"))    # #tokens and the clip's camera move
    ss = (settings or {}).get("source_sound")
    given = bool(ss) and not (isinstance(ss, dict) and ss.get("overlay"))      # (then without the voices)
    text = with_triggers(clip_cast(options, prompt, voices=not given)["prompt"],
                         list((settings or {}).get("lora_triggers") or []))
    if ss and not (isinstance(ss, dict) and ss.get("overlay")):   # (a sound laid over is not the model's)
        k = len((settings or {}).get("audio_refs") or []) + 1
        mode = ss.get("mode") if isinstance(ss, dict) else ss
        sentence = (REPEAT_SOUND_TEXT if mode == "previous" else
                    TRACK_SOUND_TEXT if mode == "track" or (isinstance(mode, str) and mode.startswith(TRACK_PREFIX))
                    else SOURCE_SOUND_TEXT)
        text = f"{text.rstrip()} {sentence.format(k=k)}"
    notes = [f"At {k['at']:.1f} s: {k['note'].strip().rstrip('.')}." for k in node.keyframes if k["note"].strip()]
    if notes:                                      # what happens at each keyframe, in time order
        text = f"{text.rstrip()} {' '.join(notes)}"
    return text


def node_settings(base: dict, options: dict, node: Node, audio_pool: list[dict] | None = None,
                  picture_sha: Callable[[str], str | None] | None = None,
                  source_sound: dict | None = None) -> dict:
    """The settings one node is made with: the project's, with the node's own LoRA stack and overrides.
    The first LoRA of the stack is recorded as "lora", the others as "extra_loras" (as before).
    audio_pool: the project's sound references ({"file", "sha1"}); the clip records those up to the highest
    <Audio N> its prompt names (none: nothing recorded, as before), then the voices of the characters who speak.
    picture_sha: the content hash of a face picture (project-relative name); the clip records the pictures it is
    made with (see clip_cast), and with @mentions what the model prompt says about the characters."""
    ov = node.overrides or {}
    n_audio = 0 if node.is_video else audio_mentioned(full_prompt(options, node.prompt))
    given = source_sound is not None and not source_sound.get("overlay")       # (then without the voices)
    use = clip_cast(options, node.prompt, voices=not given) if not node.is_video \
        else {"images": [], "record": None, "voices": []}
    sha = picture_sha or (lambda rel: None)
    refs = [{"file": r, "sha1": sha(r)} for r in use["images"]]
    sounds = (audio_pool or [])[:n_audio] if n_audio else []
    sounds = sounds + [{"file": r, "sha1": sha(r)} for r in use.get("voices", [])]
    ident = options.get("identity") or {}
    stack = node_loras(node)
    lora, strength = stack[0] if stack else (None, 1.0)
    return chain.settings_dict(model=base["model"], lora=lora, extra_loras=stack[1:],
                               strength=strength,
                               frames=ov.get("frames", options.get("frames", 73)),
                               # turbo / HyperFlow: their steps for every clip (a clip's own steps would break it)
                               steps=(base["steps"] if base.get("turbo") or base.get("hyperflow") or base.get("pika")
                                      else ov.get("steps", options.get("steps", 28))),
                               hyperflow=(base["hyperflow"]["strength"] if base.get("hyperflow") else None),
                               pika=(base["pika"]["strength"] if base.get("pika") else None),
                               width=base["width"], height=base["height"], audio=base["audio"],
                               turbo=(base["turbo"]["lora"], base["turbo"]["strength"]) if base.get("turbo") else None,
                               cfg=float(base.get("cfg", 1.0)),
                               backend="comfyui", refs=refs or None, cast=use["record"],
                               ref_size=ident.get("size", "match"), ref_hint=ident.get("hint", True),
                               guide=int(ov.get("guide") or 0),
                               style=(base.get("prompt_prefix", ""), base.get("prompt_suffix", "")),
                               audio_refs=sounds or None, voice_language=use.get("language"),
                               lora_triggers=_stack_triggers(options, node, stack),
                               source_sound=source_sound,
                               veda=veda_for(options, base["width"], base["height"],
                                             int(ov.get("frames", options.get("frames", 73)))),
                               camera=_camera.clean(ov.get("camera")))


def veda_for(options: dict, width: int, height: int, frames: int) -> bool:
    """A clip of this size is made with Veda sparse attention: the project allows it and it is large enough."""
    return bool((options.get("veda") or {"on": True}).get("on", True)) and chain.veda_useful(width, height, frames)


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


def made_frames(node: Node) -> int | None:
    """Frames of a node's made clip (from its record), or None."""
    if not node.made:
        return None
    if node.made.get("frames"):
        return int(node.made["frames"])
    if node.is_video:
        return None
    return int(((node.made.get("settings") or {}).get("frames")) or 0) or None   # older records


def handover_index(tree: Tree, node: Node, frames: int | None = None) -> int | None:
    """The frame (0-based) the clips after a generated clip start from, when it is not its last frame (end_at);
    None: the last frame. Not for a clip that leads into a video (its end is that video's start). frames: the clip's
    length (default: the made clip's, else its length setting)."""
    if node.is_video or node.end_at is None or tree.video_child(node.id) is not None:
        return None
    n = frames or made_frames(node) or clip_length(tree, node)
    if node.end_at >= n - 1:
        return None
    return max(1, int(node.end_at))


def handover_file(project: Path, tree: Tree, node: Node) -> Path | None:
    """The PNG of a made clip's hand-over frame (named after its clip version), or None (the last frame counts)."""
    k = handover_index(tree, node)
    if k is None or not node.made or not node.made.get("video"):
        return None
    v = Path(node.made["video"])
    return Path(project) / v.parent / f"{v.stem}_f{k}.png"


def ensure_handover(project: Path, tree: Tree, node: Node) -> Path | None:
    """Write the hand-over frame of a made clip if it is not there yet; returns it (None: the last frame counts)."""
    if node.end_at is not None and node.made and not node.is_video and not node.made.get("frames") \
            and node.made.get("video") and (Path(project) / node.made["video"]).is_file():
        node.made["frames"] = chain.probe(Path(project) / node.made["video"])["frames"]   # older record: count once
    f = handover_file(project, tree, node)
    if f is not None and not f.is_file():
        tmp = f.with_name(f.stem + ".part.png")
        try:
            chain.frame_at(Path(project) / node.made["video"], handover_index(tree, node), tmp)
            tmp.replace(f)
        finally:
            tmp.unlink(missing_ok=True)
    if node.made and node.made.get("video") and not node.is_video:   # frames of an earlier choice in this take
        v = Path(project) / node.made["video"]
        area = (Path(project) / "nodes").resolve()
        if area in v.resolve().parents:
            for old in handover_frames(v):
                if old != f:
                    _unlink(old)
    return f


def ensure_handovers(project: Path, tree: Tree) -> list[str]:
    """All hand-over frames of the tree (e.g. after a copy); returns the nodes whose frame could not be made."""
    bad = []
    for n in tree.nodes.values():
        try:
            ensure_handover(project, tree, n)
        except (OSError, RuntimeError):
            bad.append(n.id)
    return bad


def full_prompt(options: dict, prompt: str) -> str:
    """A clip's prompt as the model gets it: the project's style text before and after it."""
    style = options.get("style") or {}
    return " ".join(x for x in ((style.get("prefix") or "").strip(), prompt.strip(),
                                (style.get("suffix") or "").strip()) if x)


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
        k = handover_index(tree, parent, n)
        if k is not None:                           # it hands over earlier: the frames up to that one
            n = k + 1
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
            k = handover_index(tree, other, n) if what == "clip" else None
            if k is not None:                       # it hands over earlier
                n = k + 1
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
    ensure_handovers(project, tree)               # before the states: children start from them
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
    no_key = missing_keyframes(project, tree, todo)
    if no_key:
        raise FileNotFoundError(f"a keyframe picture of clip #{no_key[0][1:]} is missing")
    problem = keyframe_problem(tree, checker, todo)
    if problem:
        raise ValueError(problem)
    faces = cast_problems(project, tree, todo) + source_sound_problems(tree, todo, project)
    if faces:
        raise ValueError(faces[0])
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
            send = start                                 # what H3 gets: start.png prepared for the canvas
            if start is not None and start == project / START_FILE:   # (the record keeps start.png's hash)
                base = checker.base() or {}
                if base.get("width") and base.get("height"):
                    send = prepared_start(project, base["width"], base["height"])
            emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": len(todo)})
            # a new clip gets new file names: nothing is overwritten, a failed or stopped generation keeps the
            # previous clip and its record, and a still-open old file (SMB) does not stop anything
            new_video, new_frame = new_clip_files(project, nid)
            settings = checker.settings(nid)
            start_sha = checker.sha(start)
            end_key = checker.end_key(node)
            key_key = checker.keyframe_key(node)
            guide_from = checker.guide_from(node)
            call = dict(settings)
            if node.keyframes:
                call["keyframes"] = [{"image": str(project / k["image"]), "frame_idx": keyframe_frame(k["at"])}
                                     for k in node.keyframes]
            if settings.get("refs"):               # the generator gets the files; the record keeps names + hashes
                call["ref_paths"] = [str(project / r["file"]) for r in settings["refs"]]
            if settings.get("audio_refs"):
                call["audio_paths"] = [str(project / a["file"]) for a in settings["audio_refs"]]
            try:
                with tempfile.TemporaryDirectory(prefix="video-chain-", dir=str(nodes_dir)) as tmp:
                    guides, end_image = _guides(project, tree, node, settings, Path(tmp))
                    if guides:
                        call["guides"] = guides
                    if end_image is not None:
                        call["end_image"] = str(end_image)
                    sound = None
                    if settings.get("source_sound"):   # the real video's sound after its range: the soundtrack
                        sound = Path(tmp) / "source_sound.wav"
                        ss = settings["source_sound"]
                        if ss.get("mode") == "previous":
                            bad = _repeat_sound_problems(project, node, ss)
                            if bad:
                                raise RuntimeError(bad[0])
                        chain.to_wav(sound_file(project, ss), sound, ss["start"], ss["secs"], ss.get("loop"),
                                     ss.get("offset") or 0.0, ss.get("pad") or 0.0,
                                     fill=ss.get("mode") == "track")
                        if not ss.get("overlay"):  # the model makes the picture to it (lip sync)
                            call["audio_paths"] = list(call.get("audio_paths") or []) + [str(sound)]
                    generator.generate(model_prompt(tree.options, node, settings), send, new_video, new_frame,
                                       node.seed, settings=call)
                    if sound is not None and new_video.exists():
                        chain.replace_audio(new_video, sound)    # exactly the recorded speech, not a remake
                if not new_video.exists() or not new_frame.exists():
                    raise RuntimeError(f"the generator did not write the clip of {nid}")
                made_n = chain.probe(new_video)["frames"]      # the real count (hand-over frames, joins)
            except BaseException:
                for f in (new_video, new_frame):
                    _unlink(f)
                raise
            old_end = (node.made or {}).get("end_image")
            old_keys = {k.get("image") for k in (node.made or {}).get("keyframes") or [] if isinstance(k, dict)}
            cut = chain.hard_cuts(new_video) if node.keyframes or end_image_used(tree, node) else []
            node.made = {"prompt": node.prompt, "seed": node.seed, "settings": settings,
                         "overrides": copy.deepcopy(node.overrides or {}), "end_image": node.end_image,
                         "keyframes": copy.deepcopy(node.keyframes),
                         "start_sha1": start_sha, "video": f"nodes/{new_video.name}",
                         "frame": f"nodes/{new_frame.name}", "frames": made_n}
            if old_end and old_end != node.end_image and _in_ends(project, old_end):
                old.append(project / old_end)          # the end image of the previous version
            for rel in old_keys - {k["image"] for k in node.keyframes} - {None}:
                if _in_ends(project, rel):
                    old.append(project / rel)          # keyframe pictures of the previous version
            if cut:
                node.made["cuts"] = cut                # hard cuts H3 made (a keyframe / end it could not reach)
                log(f"{nid}: the clip cuts at {', '.join(f'{c / chain.FPS:.2f} s' for c in cut)} "
                    "(a keyframe or the end image is probably too far from what comes before it)")
            if end_key is not None:
                node.made["end_key"] = end_key
            if key_key is not None:
                node.made["keyframe_key"] = key_key
            if guide_from is not None:
                node.made["guide_from"] = guide_from
            new_files = [new_video, new_frame]
            if node.end_at is not None:
                # a new take: the frame chosen in the old one means nothing here; it hands over at its last
                # frame until a frame of this take is chosen
                log(f"{nid}: made again, it hands over at its last frame again (choose a frame of the new take)")
                node.end_at = None
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
    A clip that hands over at an earlier frame (end_at) loses the frames after it.
    planned: the motion guides as set now (for display), not as the clips were made."""
    cuts = [[0, 0] for _ in path]
    for k, a in enumerate(path):                   # an earlier hand-over: the frames after it are left out
        n = (clip_length(tree, a) if planned else made_frames(a)) or clip_length(tree, a)
        h = handover_index(tree, a, n)
        if h is not None:
            cuts[k][1] = n - 1 - h
    if not keep_repeated:
        for k in range(1, len(path)):
            a, b = path[k - 1], path[k]
            if not b.is_video:
                cuts[k][0] = max(1, _guide_of(b, planned))
            elif not a.is_video and tree.video_child(a.id) is b:
                cuts[k - 1][1] = max(1, _guide_of(a, planned))
    for k, a in enumerate(path):                   # a short clip keeps at least one frame (carry-over + hand-over)
        n = (clip_length(tree, a) if planned else made_frames(a)) or clip_length(tree, a)
        if cuts[k][0] + cuts[k][1] >= n:
            cuts[k][1] = max(0, n - 1 - cuts[k][0])
            cuts[k][0] = min(cuts[k][0], max(0, n - 1))
    return [(s, e) for s, e in cuts]


def refined_file(project: Path, node: Node, size: str) -> Path:
    """The refined copy of a node's current clip at an export size (named after its clip version)."""
    v = Path(node.made["video"])
    return Path(project) / v.parent / f"{v.stem}_{size}.mp4"


SWAPPED_REFINE_PROMPT = "A realistic video of a person, natural skin texture, sharp detailed face."


def swapped_video(node: Node) -> dict | None:
    """A video node whose person was replaced: what it was replaced with (see swap.replaced), else None."""
    if not node.is_video:
        return None
    from .swap import replaced
    return replaced(node)


def refine_record(target: Path) -> dict:
    """What a refined copy was made with (its .json; {} if none)."""
    try:
        return json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def refine_path(generator, project: Path, nid: str, size: str, out: Path,
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
    from .comfy import SEEDVR_METHODS, refine_size
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
    missing = [n.id for n in path if checker.status(n.id) != "done"]
    if missing:
        raise ValueError(f"not made yet: {', '.join(missing)}")
    order = [n for n in path if n.is_video] + [n for n in path if not n.is_video]   # videos first: pins
    files: dict[str, Path] = {}
    emit({"e": "plan", "make": [n.id for n in order]})
    made_now = []
    with tempfile.TemporaryDirectory(prefix="video-chain-", dir=str(project / "nodes")) as tmp:
        tmp = Path(tmp)
        for k, node in enumerate(order, start=1):
            target = refined_file(project, node, size)
            first = last = None
            if pin and not single and not node.is_video and method == "refine":
                i = path.index(node)
                if i == 0:
                    s0 = node.made.get("settings") or {}
                    first = prepared_start(project, *refine_size(int(s0["width"]), int(s0["height"]), size)) \
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
            pins = {"first": chain.file_signature(first), "last": chain.file_signature(last)}
            regen = not node.is_video or bool(refine_swapped and swapped_video(node))
            if node.is_video:
                pins["video"] = "refined" if regen else "converted"
            if regen and method != "refine":
                pins["method"] = method
                if denoise:
                    pins["denoise"] = True
            elif regen:
                pins["strength"] = strength
                if fast_attention:
                    pins["attention"] = "sol-attn"
            record = refine_record(target)
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
            if reuse:
                emit({"e": "node", "id": node.id, "state": "reused"})
                continue
            emit({"e": "node", "id": node.id, "state": "generating", "k": k, "n": len(order)})
            target.with_suffix(".json").unlink(missing_ok=True)   # a copy made now must not keep older pins
            s = node.made.get("settings") or {}
            if node.is_video and not all(k_ in (node.made.get("key") or {}) for k_ in ("width", "height")) or \
                    not node.is_video and not all(k_ in s for k_ in ("width", "height")):
                raise ValueError(f"{'video' if node.is_video else 'clip'} #{node.id[1:]} has no recorded size; "
                                 "make it again, then export")
            if regen and method != "refine":             # SeedVR2: the clip as made, restored at the size
                dims = node.made.get("key") if node.is_video else s
                w, h = refine_size(int(dims["width"]), int(dims["height"]), size)
                name = "SeedVR2 " + SEEDVR_METHODS[method].upper()
                log(f"[{k}/{len(order)}] {node.id}: upscaling with {name} to {w}x{h}")
                generator.seedvr(project / node.made["video"], target, (w, h), method, seed=node.seed,
                                 denoise=denoise)
            elif node.is_video and regen:                # a replaced person: re-rendered like a generated clip
                key = node.made.get("key") or {}
                w, h = refine_size(int(key["width"]), int(key["height"]), size)
                info = swapped_video(node)
                prompt = info.get("details") or info.get("prompt") or SWAPPED_REFINE_PROMPT
                log(f"[{k}/{len(order)}] {node.id}: refining the video with the replaced person at {w}x{h}")
                from .comfy import SOL_TAU
                import subprocess
                draft, n = project / node.made["video"], _clip_frames(project, node)
                grid = n if n <= 5 or n % 17 == 5 else n + (5 - n) % 17       # H3's 17n+5 frame grid
                padded = tmp / f"{node.id}_draft.mp4"
                if grid != n:                            # the last frame held to the grid, cut off again after
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(draft), "-vf",
                                    f"tpad=stop_mode=clone:stop={grid - n}", "-c:v", "libx264", "-crf", "10",
                                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(padded)], check=True,
                                   capture_output=True)
                made = tmp / f"{node.id}_refined.mp4" if grid != n else target
                generator.refine(padded if grid != n else draft, prompt, made, int(info.get("seed") or 0), (w, h),
                                 grid, [], None, None, sparse=SOL_TAU if fast_attention else None, strength=strength)
                if grid != n:
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(made), "-frames:v", str(n),
                                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-an", str(target)],
                                   check=True, capture_output=True)
                if chain.audio_seconds(draft):                # the recorded sound exactly, not the model's
                    wav = tmp / f"{node.id}_sound.wav"
                    chain.to_wav(draft, wav)
                    chain.replace_audio(target, wav)
            elif node.is_video:
                src = node.source
                if not Path(src["file"]).is_file():
                    raise FileNotFoundError(f"video file not found: {src['file']}")
                key = node.made.get("key") or {}
                w, h = refine_size(int(key["width"]), int(key["height"]), size)
                log(f"[{k}/{len(order)}] {node.id}: converting {Path(src['file']).name} at {w}x{h}")
                chain.convert_range(Path(src["file"]), target, src["start"], src["end"], w, h)
            else:
                s = node.made.get("settings") or {}
                w, h = refine_size(int(s["width"]), int(s["height"]), size)
                made_prompt = _camera.apply(node.made.get("prompt", node.prompt), s.get("camera"))
                prompt = " ".join(x for x in (s.get("prompt_prefix", ""), made_prompt, s.get("prompt_suffix", "")) if x)
                from .triggers import with_triggers
                prompt = with_triggers(prompt, list(s.get("lora_triggers") or []))   # as the clip was made
                loras = ([(s["lora"], s.get("strength", 1.0))] if s.get("lora") else []) + \
                    [(e["lora"], e.get("strength", 1.0)) for e in s.get("extra_loras") or []]
                log(f"[{k}/{len(order)}] {node.id}: refining at {w}x{h}")
                from .comfy import SOL_TAU
                generator.refine(project / node.made["video"], prompt, target, node.seed, (w, h),
                                 made_frames(node) or _clip_frames(project, node), loras, first, last,
                                 sparse=SOL_TAU if fast_attention else None, strength=strength)
                if s.get("source_sound") and not sound_file(project, s["source_sound"]).is_file():
                    log(f"[{k}/{len(order)}] {node.id}: the video file of its sound is missing "
                        f"({Path(s['source_sound']['file']).name}); the refined copy keeps the model's sound")
                if s.get("source_sound") and sound_file(project, s["source_sound"]).is_file():
                    wav = tmp / f"{node.id}_sound.wav"     # the recorded sound, as in the clip
                    chain.to_wav(sound_file(project, s["source_sound"]), wav, s["source_sound"]["start"],
                                 s["source_sound"]["secs"], s["source_sound"].get("loop"),
                                 s["source_sound"].get("offset") or 0.0, s["source_sound"].get("pad") or 0.0,
                                 fill=s["source_sound"].get("mode") == "track")
                    chain.replace_audio(target, wav)
            target.with_suffix(".json").write_text(json.dumps(pins), encoding="utf-8")
            made_now.append(node.id)
            emit({"e": "node", "id": node.id, "state": "done"})
        videos = [files.get(n.id) or refined_file(project, n, size) for n in path]
        cuts = [(0, 0)] if single else path_cuts(tree, path, bool(tree.options.get("keep_repeated")))
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        emit({"e": "joining"})
        chain.concat(videos, out, cuts=cuts)            # (inside: a temporary single copy is still there)
    emit({"e": "exported", "file": str(out)})
    emit({"e": "done", "made": made_now})
    return out


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
    size = None                    # frozen clips of an older size: every clip scaled to the size clips have now
    if any(n.frozen for n in path):
        base = checker.base()
        sizes = {(i["width"], i["height"]) for i in (chain.probe_source(v) for v in videos)}
        if base is not None and len(sizes) > 1:
            size = (int(base["width"]), int(base["height"]))
    key = json.dumps([[str(v), v.stat().st_mtime_ns] for v in videos] + [[list(c) for c in cuts]] +
                     ([list(size)] if size else []))
    stamp = target.with_suffix(".json")
    if out is None and target.exists() and stamp.exists() and stamp.read_text() == key:
        return target
    chain.concat(videos, target, cuts=cuts, size=size, fill=True)
    if out is None:
        stamp.write_text(key)
    return target
