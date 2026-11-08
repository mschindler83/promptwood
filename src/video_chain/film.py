"""A film: scenes in an order. Each scene is a project as tree.py knows it (its own tree.json, start image and
clips); the film folder is the first scene, further scenes live in scenes/<id>/. film.json (written once there is
more than one scene) holds their order and titles.

Shared by all scenes: the cast (faces, voices), the sound references and the settings in SHARED_OPTIONS. Their
files are in the film folder's refs/; a scene's refs/ is a link to it. A change of a shared setting in one scene is
written into the others (sync_shared), so every scene is made with the same people at the same size.

A project from before scenes is a film with one scene: nothing on disk changes until a second scene is added."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from . import tree as T

FILM_FILE = "film.json"
SCENES_DIR = "scenes"
SHARED_OPTIONS = ("cast", "identity", "audio_refs", "resolution", "audio")


@dataclass
class Scene:
    id: str
    title: str
    dir: str                    # relative to the film folder ("." for the first scene)
    take: str | None = None     # the clip the scene ends at in the film (None: its selected clip)


def root_of(folder: Path) -> Path:
    """The film folder a project folder belongs to (a scene folder scenes/<id> -> the film; else itself)."""
    folder = Path(folder).expanduser().resolve()
    if folder.parent.name == SCENES_DIR and (folder.parent.parent / FILM_FILE).is_file():
        return folder.parent.parent
    return folder


class Film:
    def __init__(self, root: Path, scenes: list[Scene], current: str | None = None):
        self.root = Path(root)
        self.scenes = scenes
        self.current = current if any(s.id == current for s in scenes) else scenes[0].id

    # ---- loading and saving
    @staticmethod
    def load(root: Path) -> "Film":
        root = root_of(root)
        f = root / FILM_FILE
        scenes, current = [], None
        if f.is_file():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                for s in d.get("scenes") or []:
                    sid, sdir = str(s.get("id") or ""), str(s.get("dir") or "")
                    if re.fullmatch(r"s\d+", sid) and (sdir == "." or re.fullmatch(rf"{SCENES_DIR}/s\d+", sdir)):
                        take = s.get("take") if isinstance(s.get("take"), str) and re.fullmatch(r"n\d+", s["take"]) \
                            else None
                        scenes.append(Scene(sid, str(s.get("title") or sid), sdir, take))
                current = d.get("current")
            except (OSError, ValueError, AttributeError) as e:
                raise ValueError(f"{f} is damaged ({e})") from e
        if not any(s.dir == "." for s in scenes):             # the film folder is always a scene
            scenes.insert(0, Scene("s1" if not any(s.id == "s1" for s in scenes) else _next_id(scenes),
                                   "Scene 1", "."))
        return Film(root, scenes, current)

    def save(self) -> None:
        """Write film.json (only once there is more than one scene, or it exists already)."""
        f = self.root / FILM_FILE
        if len(self.scenes) < 2 and not f.exists():
            return
        data = {"version": 1, "scenes": [asdict(s) for s in self.scenes], "current": self.current}
        tmp = f.with_suffix(".part")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(f)

    # ---- scenes
    def scene(self, sid: str) -> Scene:
        return next(s for s in self.scenes if s.id == sid)

    def path(self, sid: str) -> Path:
        return (self.root / self.scene(sid).dir).resolve()

    def scene_of(self, folder: Path) -> Scene | None:
        folder = Path(folder).resolve()
        return next((s for s in self.scenes if self.path(s.id) == folder), None)

    def add_scene(self, title: str, after: str | None = None, start_image: Path | None = None,
                  like: str | None = None) -> Scene:
        """A new scene (after the scene `after`, else at the end): its own folder, the film's shared settings and
        refs, the other settings like the scene `like` (e.g. its LoRAs and speed), an optional start image."""
        sid = _next_id(self.scenes)
        rel = f"{SCENES_DIR}/{sid}"
        d = self.root / rel
        d.mkdir(parents=True, exist_ok=False)
        refs = self.root / T.REFS_DIR
        refs.mkdir(exist_ok=True)
        os.symlink(os.path.relpath(refs, d), d / T.REFS_DIR, target_is_directory=True)
        base = T.load(self.path(like)) if like else T.load(self.root)
        t = T.Tree()
        t.options.update({k: v for k, v in base.options.items() if k in T.DEFAULT_OPTIONS})
        t.options["style"] = {"prefix": "", "suffix": ""}           # a scene's look is its own
        if start_image is not None:
            T.set_start_image(d, Path(start_image), t)
        T.save(d, t)
        scene = Scene(sid, (title or "").strip() or f"Scene {len(self.scenes) + 1}", rel)
        k = next((i + 1 for i, s in enumerate(self.scenes) if s.id == after), len(self.scenes))
        self.scenes.insert(k, scene)
        self.save()
        return scene

    def rename(self, sid: str, title: str) -> None:
        title = (title or "").strip()
        if not title:
            raise ValueError("a scene needs a title")
        self.scene(sid).title = title
        self.save()

    def set_take(self, sid: str, nid: str | None) -> None:
        """The clip a scene ends at in the film (None: its selected clip)."""
        self.scene(sid).take = nid
        self.save()

    def take_of(self, sid: str, tree: "T.Tree") -> str | None:
        """The clip the film plays a scene up to: its chosen ending, else its selected clip (None: nothing)."""
        t = self.scene(sid).take
        if t in tree.nodes:
            return t
        return tree.selected if tree.selected in tree.nodes else None

    def move(self, sid: str, index: int) -> None:
        s = self.scene(sid)
        self.scenes.remove(s)
        self.scenes.insert(max(0, min(int(index), len(self.scenes))), s)
        self.save()

    def remove(self, sid: str) -> None:
        """Delete a scene and its folder (its clips). The film folder's own scene (the first one made) stays."""
        s = self.scene(sid)
        if s.dir == ".":
            raise ValueError("the film folder's own scene cannot be deleted (move it instead)")
        if len(self.scenes) < 2:
            raise ValueError("a film keeps at least one scene")
        d = self.path(sid)
        if (self.root / SCENES_DIR).resolve() not in d.parents:
            raise ValueError("this scene's folder is not inside the film")
        link = d / T.REFS_DIR
        if link.is_symlink():
            link.unlink()                          # the shared refs are never deleted with a scene
        shutil.rmtree(d)
        self.scenes.remove(s)
        if self.current == sid:
            self.current = self.scenes[0].id
        self.save()

    # ---- shared settings
    def shared(self, options: dict) -> dict:
        return {k: options.get(k) for k in SHARED_OPTIONS if k in options}

    def sync_shared(self, from_dir: Path, options: dict) -> list[str]:
        """Write the shared settings of one scene into the others; returns the ids of the scenes changed."""
        want = self.shared(options)
        changed = []
        src = Path(from_dir).resolve()
        for s in self.scenes:
            d = self.path(s.id)
            if d == src or not (d / T.TREE_FILE).is_file():
                continue
            t = T.load(d)
            if all(t.options.get(k) == v for k, v in want.items()):
                continue
            t.options.update(json.loads(json.dumps(want)))
            T.save(d, t)
            changed.append(s.id)
        return changed


def _next_id(scenes: list[Scene]) -> str:
    n = max((int(s.id[1:]) for s in scenes if re.fullmatch(r"s\d+", s.id)), default=0) + 1
    return f"s{n}"


def summary(film: Film, sid: str, tree: "T.Tree | None" = None, status_of: dict | None = None) -> dict:
    """What a scene's card shows: {"clips", "secs" (its selected path, joined), "done", "changed", "new",
    "thumb" (a picture of its start, or None)}. tree / status_of: the open scene as the window has it."""
    d = film.path(sid)
    out = {"clips": 0, "secs": 0.0, "done": 0, "changed": 0, "new": 0, "thumb": None}
    try:
        t = tree if tree is not None else T.load(d)
    except (OSError, ValueError):
        return out
    start = d / T.START_FILE
    take = film.take_of(sid, t)
    path = t.path_to(take) if take else []
    out["clips"] = len(t.nodes)
    out["take"] = take
    out["fixed"] = film.scene(sid).take == take and take is not None
    try:
        ch = None if status_of is not None else T.Checker(d, t)
        for n in path:
            st = status_of.get(n.id, "new") if status_of is not None else ch.status(n.id)
            out[st if st in ("done", "changed", "new") else "changed"] += 1
        cuts = T.path_cuts(t, path, bool(t.options.get("keep_repeated")), planned=True)
        frames = sum(max(0, T.clip_length(t, n) - a - b) for n, (a, b) in zip(path, cuts))
        out["secs"] = round(frames / 24, 1)
    except Exception:  # noqa: BLE001  (a scene that cannot be checked still gets its card)
        pass
    if start.is_file():
        out["thumb"] = start
    elif path and path[0].made:
        m = path[0].made
        f = d / (m.get("first") or m.get("frame") or "")
        out["thumb"] = f if f.is_file() else None
    return out


def film_problems(film: Film) -> list[str]:
    """Why the film cannot be played / exported yet (a scene without clips, clips not made)."""
    out = []
    for k, sc in enumerate(film.scenes, start=1):
        d = film.path(sc.id)
        try:
            t = T.load(d)
        except (OSError, ValueError) as e:
            out.append(f"scene {k} ({sc.title}) cannot be read: {e}")
            continue
        take = film.take_of(sc.id, t)
        if take is None:
            out.append(f"scene {k} ({sc.title}) has no clip yet")
            continue
        ch = T.Checker(d, t)
        todo = [n.id for n in t.path_to(take) if ch.status(n.id) != "done"]
        if todo:
            out.append(f"scene {k} ({sc.title}): {len(todo)} clip{'s' if len(todo) != 1 else ''} not made yet "
                       f"(#{todo[0][1:]}{' …' if len(todo) > 1 else ''})")
    return out


def join_film(film: Film, out: Path) -> Path:
    """The film as one video: every scene's path up to its ending, joined with hard cuts. Scenes of another size
    than the first one are fitted into its frame (black bars)."""
    problems = film_problems(film)
    if problems:
        raise ValueError(problems[0])
    parts = []
    for sc in film.scenes:
        d = film.path(sc.id)
        t = T.load(d)
        parts.append(T.join_path(d, t, film.take_of(sc.id, t)))
    from . import chain
    first = chain.probe(parts[0])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chain.concat(parts, out, cuts=[(0, 0)] * len(parts), size=(first["width"], first["height"]))
    return out
