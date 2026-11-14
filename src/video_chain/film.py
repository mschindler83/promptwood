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

from . import chain
from . import tree as T

FILM_FILE = "film.json"
SCENES_DIR = "scenes"
SHARED_OPTIONS = ("cast", "identity", "audio_refs", "tracks", "resolution", "audio")
JOINS = {"cut": "Cut", "fade": "Crossfade", "continue": "Continue"}   # how a scene follows the one before
FADE_SECS = 0.5


@dataclass
class Scene:
    id: str
    title: str
    dir: str                    # relative to the film folder ("." for the first scene)
    take: str | None = None     # the clip the scene ends at in the film (None: its selected clip)
    join: str = "cut"           # how it follows the scene before (JOINS)
    from_sha: str | None = None  # "continue": the end frame its start image was taken from
    notes: str = ""             # the user's notes (a line of script, what happens)


def root_of(folder: Path) -> Path:
    """The film folder a project folder belongs to (a scene folder scenes/<id> -> the film; else itself)."""
    folder = Path(folder).expanduser().resolve()
    if folder.parent.name == SCENES_DIR and (folder.parent.parent / FILM_FILE).is_file():
        return folder.parent.parent
    return folder


class Film:
    def __init__(self, root: Path, scenes: list[Scene], current: str | None = None, shared: dict | None = None):
        self.root = Path(root)
        self.scenes = scenes
        self.current = current if any(s.id == current for s in scenes) else scenes[0].id
        self.shared_state = shared             # the shared settings as last passed on (film.json)
        self.problem = ""                      # film.json could not be read fully: not written back

    # ---- loading and saving
    @staticmethod
    def load(root: Path) -> "Film":
        root = root_of(root)
        f = root / FILM_FILE
        scenes, current, shared, problem = [], None, None, ""
        if f.is_file():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                entries = d.get("scenes") or []
                if not isinstance(entries, list):
                    raise ValueError("no list of scenes")
                for e in entries:
                    if not isinstance(e, dict):
                        problem = "an entry that is not a scene"
                        continue
                    sid, sdir = str(e.get("id") or ""), str(e.get("dir") or "")
                    ok = re.fullmatch(r"s\d+", sid) and (sdir == "." or sdir == f"{SCENES_DIR}/{sid}")
                    if not ok or any(x.id == sid or x.dir == sdir for x in scenes):
                        problem = f"a doubtful scene entry ({sid or '?'} in {sdir or '?'})"
                        continue
                    if sdir != "." and not (root / sdir / T.TREE_FILE).is_file():
                        problem = f"the folder of scene {sid} is missing"
                        continue
                    take = e.get("take") if isinstance(e.get("take"), str) and re.fullmatch(r"n\d+", e["take"]) \
                        else None
                    join = e.get("join") if e.get("join") in JOINS else "cut"
                    frm = e.get("from_sha") if isinstance(e.get("from_sha"), str) else None
                    notes = str(e.get("notes") or "")
                    scenes.append(Scene(sid, str(e.get("title") or sid), sdir, take, join, frm, notes))
                current = d.get("current")
                shared = d.get("shared") if isinstance(d.get("shared"), dict) else None
            except (OSError, ValueError, AttributeError, TypeError) as e:
                scenes, problem = [], f"film.json cannot be read ({e})"
        for d in sorted((root / SCENES_DIR).glob("s*")) if (root / SCENES_DIR).is_dir() else []:
            if re.fullmatch(r"s\d+", d.name) and (d / T.TREE_FILE).is_file() and \
                    not any(x.dir == f"{SCENES_DIR}/{d.name}" for x in scenes) and not any(x.id == d.name for x in scenes):
                scenes.append(Scene(d.name, d.name, f"{SCENES_DIR}/{d.name}"))     # a scene folder not listed
        if not any(s.dir == "." for s in scenes):             # the film folder is always a scene
            scenes.insert(0, Scene("s1" if not any(s.id == "s1" for s in scenes) else _next_id(scenes, root),
                                   "Scene 1", "."))
        film = Film(root, scenes, current, shared)
        film.problem = problem
        return film

    def reorder(self, ids: list[str]) -> list[str]:
        """The scenes in this order (ids not given keep their place at the end). Returns the scenes whose
        'continue' became a cut (their scene before changed)."""
        before = self._befores()
        by = {s.id: s for s in self.scenes}
        new = [by[i] for i in ids if i in by]
        self.scenes = new + [s for s in self.scenes if s not in new]
        reset = self._keep_joins(before)
        self.save()
        return reset

    def _befores(self) -> dict[str, str | None]:
        return {s.id: (self.scenes[k - 1].id if k else None) for k, s in enumerate(self.scenes)}

    def _keep_joins(self, before: dict[str, str | None]) -> list[str]:
        """A scene continued the scene before it: if that is another scene now (or none), it starts with a cut
        (its start image is never replaced by the new neighbour's end)."""
        reset = []
        now = self._befores()
        for s in self.scenes:
            if s.join == "continue" and (now[s.id] is None or now[s.id] != before.get(s.id)):
                s.join, s.from_sha = "cut", None
                reset.append(s.id)
        return reset

    def save(self) -> None:
        """Write film.json (only once there is more than one scene, or it exists already)."""
        f = self.root / FILM_FILE
        if self.problem:
            raise OSError(f"film.json is not written ({self.problem}); fix or remove it")
        if len(self.scenes) < 2 and not f.exists():
            return
        data = {"version": 1, "scenes": [asdict(s) for s in self.scenes], "current": self.current,
                "shared": self.shared_state}
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
        sid = _next_id(self.scenes, self.root)
        rel = f"{SCENES_DIR}/{sid}"
        d = self.root / rel
        d.mkdir(parents=True, exist_ok=False)
        try:
            refs = self.root / T.REFS_DIR
            refs.mkdir(exist_ok=True)
            try:
                os.symlink(os.path.relpath(refs, d), d / T.REFS_DIR, target_is_directory=True)
            except OSError as e:
                raise OSError(f"the scene's link to the film's refs/ cannot be made here ({e}); scenes need a "
                              "file system with links") from e
            base = T.load(self.path(like)) if like else T.load(self.root)
            t = T.Tree()
            t.options.update({k: v for k, v in base.options.items() if k in T.DEFAULT_OPTIONS})
            t.options["style"] = {"prefix": "", "suffix": ""}           # a scene's look is its own
            if start_image is not None:
                T.set_start_image(d, Path(start_image), t)
            T.save(d, t)
        except BaseException:
            link = d / T.REFS_DIR
            if link.is_symlink():
                link.unlink()
            shutil.rmtree(d, ignore_errors=True)                       # no half scene blocking the next add
            raise
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

    def set_notes(self, sid: str, notes: str) -> None:
        self.scene(sid).notes = (notes or "").strip()
        self.save()

    def duplicate(self, sid: str) -> Scene:
        """A copy of a scene with its clips (to try a variant), right after it."""
        if self.problem:
            raise OSError(f"film.json is not written ({self.problem}); fix or remove it first")
        src = self.scene(sid)
        nid = _next_id(self.scenes, self.root)
        rel = f"{SCENES_DIR}/{nid}"
        d = self.root / rel
        skip = {T.REFS_DIR, SCENES_DIR, FILM_FILE, "paths"}
        ignore = shutil.ignore_patterns("*.part", "*.part.*", "video-chain-*",         # (refined copies can be
                                        "*_1080p.mp4", "*_1080p.json", "*_2k.mp4", "*_2k.json")   # made again)

        def ign(folder, names):
            out = set(ignore(folder, names))
            if Path(folder).resolve() == self.path(sid):
                out |= {n for n in names if n in skip}
            return out
        try:
            shutil.copytree(self.path(sid), d, ignore=ign)
            refs = self.root / T.REFS_DIR
            refs.mkdir(exist_ok=True)
            os.symlink(os.path.relpath(refs, d), d / T.REFS_DIR, target_is_directory=True)
            T.save(d, T.load(d))
        except BaseException:
            link = d / T.REFS_DIR
            if link.is_symlink():
                link.unlink()
            shutil.rmtree(d, ignore_errors=True)
            raise
        base = re.sub(r" \(copy( \d+)?\)$", "", src.title)
        titles = {s.title for s in self.scenes}
        title, n = f"{base} (copy)", 2
        while title in titles:
            title, n = f"{base} (copy {n})", n + 1
        # a copy keeps its own start (it does not continue the original); it does not come between the original
        # and a scene continuing it
        join = "cut" if src.join == "continue" else src.join
        scene = Scene(nid, title, rel, src.take, join, None if join == "cut" else src.from_sha, src.notes)
        k = next(i for i, s in enumerate(self.scenes) if s.id == sid)
        after = self.scenes[k + 1] if k + 1 < len(self.scenes) else None
        at = len(self.scenes) if after is not None and after.join == "continue" else k + 1
        self.scenes.insert(at, scene)
        try:
            self.save()
        except BaseException:
            self.scenes.remove(scene)
            link = d / T.REFS_DIR
            if link.is_symlink():
                link.unlink()
            shutil.rmtree(d, ignore_errors=True)
            raise
        return scene

    def set_join(self, sid: str, join: str) -> None:
        if join not in JOINS:
            raise ValueError(f"unknown join {join!r}")
        sc = self.scene(sid)
        sc.join = join
        sc.from_sha = None
        self.save()

    def before(self, sid: str) -> Scene | None:
        k = next(i for i, s in enumerate(self.scenes) if s.id == sid)
        return self.scenes[k - 1] if k else None

    def end_frame(self, sid: str) -> Path | None:
        """The last frame of a scene's ending as made (its hand-over frame if it hands over early), or None."""
        d = self.path(sid)
        try:
            t = T.load(d)
        except (OSError, ValueError):
            return None
        take = self.take_of(sid, t)
        node = t.nodes.get(take) if take else None
        if node is None or not node.made:
            return None
        try:
            f = T.ensure_handover(d, t, node) or (d / node.made.get("frame", ""))
        except (OSError, RuntimeError, ValueError):
            f = d / node.made.get("frame", "")
        return f if f.is_file() else None

    def update_continues(self) -> list[str]:
        """Scenes that continue the scene before them start from its end frame: their start image is set again when
        that frame changed (their clips then count as changed). Returns the ids of the scenes updated."""
        from . import chain
        out = []
        for k, sc in enumerate(self.scenes):
            if k == 0 or sc.join != "continue":
                continue
            frame = self.end_frame(self.scenes[k - 1].id)
            if frame is None:
                continue
            sha = chain.file_signature(frame)
            d = self.path(sc.id)
            if sha == sc.from_sha and (d / T.START_FILE).is_file():
                continue
            t = T.load(d)
            T.set_start_image(d, frame, t)
            T.save(d, t)
            sc.from_sha = sha
            out.append(sc.id)
        if out:
            self.save()
        return out

    def take_of(self, sid: str, tree: "T.Tree") -> str | None:
        """The clip the film plays a scene up to: its chosen ending, else its selected clip (None: nothing)."""
        t = self.scene(sid).take
        if t in tree.nodes:
            return t
        return tree.selected if tree.selected in tree.nodes else None

    def move(self, sid: str, index: int) -> list[str]:
        before = self._befores()
        s = self.scene(sid)
        self.scenes.remove(s)
        self.scenes.insert(max(0, min(int(index), len(self.scenes))), s)
        reset = self._keep_joins(before)
        self.save()
        return reset

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
        before = self._befores()
        self.scenes.remove(s)                      # out of the film first: a failed delete leaves no broken entry
        self._keep_joins(before)
        if self.current == sid:
            self.current = self.scenes[0].id
        self.save()
        link = d / T.REFS_DIR
        if link.is_symlink():
            link.unlink()                          # (only the link to the shared refs)
        chain.keep_aside(self.root, d)             # the scene's clips are never deleted: its folder goes to old/

    # ---- shared settings
    def shared(self, options: dict) -> dict:
        return {k: options.get(k) for k in SHARED_OPTIONS if k in options}

    def sync_shared(self, from_dir: Path, options: dict) -> list[str]:
        """Write the shared settings of one scene into the others (and film.json); returns the ids of the scenes
        changed. A scene that cannot be written is skipped (raised at the end, the others are done); a scene whose
        tree.json changes meanwhile (another window, a run) is not overwritten."""
        want = json.loads(json.dumps(self.shared(options)))
        changed, failed = [], []
        src = Path(from_dir).resolve()
        for s in self.scenes:
            d = self.path(s.id)
            if d == src or not (d / T.TREE_FILE).is_file():
                continue
            try:
                before = (d / T.TREE_FILE).stat().st_mtime_ns
                t = T.load(d)
                if all(t.options.get(k) == v for k, v in want.items()):
                    continue
                t.options.update(want)
                if (d / T.TREE_FILE).stat().st_mtime_ns != before:
                    raise OSError("it was changed meanwhile")
                T.save(d, t)
                changed.append(s.id)
            except (OSError, ValueError) as e:
                failed.append(f"{s.title}: {e}")
        if len(self.scenes) > 1 and not self.problem:
            self.shared_state = want
            self.save()
        if failed:
            raise OSError("not passed on to " + "; ".join(failed))
        return changed


def _next_id(scenes: list[Scene], root: Path | None = None) -> str:
    """A scene id used neither by the film nor by a folder in scenes/ (e.g. one left by a failed add)."""
    n = max((int(s.id[1:]) for s in scenes if re.fullmatch(r"s\d+", s.id)), default=0) + 1
    while root is not None and (Path(root) / SCENES_DIR / f"s{n}").exists():
        n += 1
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
    out["join"] = film.scene(sid).join if film.scenes[0].id != sid else None
    out["notes"] = film.scene(sid).notes
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
        if k > 1 and sc.join == "continue":
            end = film.end_frame(film.scenes[k - 2].id)
            if end is None:
                out.append(f"scene {k} ({sc.title}) continues scene {k - 1}, which is not made yet")
                continue
            from . import chain
            if chain.file_signature(end) != sc.from_sha:
                out.append(f"scene {k} ({sc.title}) continues scene {k - 1}, whose end changed: it has to be "
                           "made again from there (Generate film)")
                continue
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


def join_film(film: Film, out: Path, keep: bool = False) -> Path:
    """The film as one video: every scene's path up to its ending, joined as each scene says (cut, crossfade, or
    continuing: its repeated first frame left out). Scenes of another size than the first are fitted into its frame
    (black bars). keep: out is the app's own file of the film (played again and again): it is joined again only
    when something changed, and the join before is then moved to old/ in the film's folder, never deleted."""
    problems = film_problems(film)
    if problems:
        raise ValueError(problems[0])
    parts = []
    for sc in film.scenes:
        d = film.path(sc.id)
        t = T.load(d)
        take = film.take_of(sc.id, t)
        if take is None:
            raise ValueError(f"scene {sc.title} has no clip yet")
        parts.append(T.join_path(d, t, take))
    from . import chain
    first = chain.probe(parts[0])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = (first["width"] // 2 * 2, first["height"] // 2 * 2)     # even (the video codec needs it)
    joins = [s.join for s in film.scenes]

    def write(f: Path) -> None:
        chain.join_parts(parts, f, size, joins, fade=FADE_SECS)
    if not keep:
        write(out)
        return out
    key = json.dumps([[str(p), p.stat().st_mtime_ns] for p in parts] + [joins, list(size)])
    stamp = out.with_suffix(".json")
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        return out
    chain.replace_kept(film.root, out, write)
    stamp.write_text(key)
    return out


def cast_by_scene(film: Film) -> tuple[list[str], list[tuple[str, dict[str, str]]]]:
    """Who appears in which scene (along the path the film plays): (character names, [(scene title,
    {name: mark})]); mark: "face", "voice", "face+voice" or "picture" (a place, object or style)."""
    names: list[str] = []
    rows = []
    for sc in film.scenes:
        d = film.path(sc.id)
        try:
            t = T.load(d)
        except (OSError, ValueError):
            rows.append((sc.title, {}))
            continue
        for c in T.cast_of(t):
            if c["name"] not in names:
                names.append(c["name"])
        take = film.take_of(sc.id, t)
        marks: dict[str, set] = {}
        for n in t.path_to(take) if take else []:
            if n.is_video:
                continue
            use = T.node_cast(t.options, n)
            for c in use["characters"]:
                if c["images"]:
                    marks.setdefault(c["name"], set()).add("face" if T.kind_of(c) == "person" else "picture")
                if c.get("voice") and c["voice"]["file"] in (use.get("voices") or []):
                    marks.setdefault(c["name"], set()).add("voice")
        rows.append((sc.title, {k: "+".join(sorted(v)) for k, v in marks.items()}))
    return names, rows
