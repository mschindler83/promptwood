"""The cast library: the people of a project's cast kept for the app, so they can be brought into any project.
In the app's data folder ($XDG_DATA_HOME/video-chain/casts, VIDEO_CHAIN_CASTS):
  library.json  {"version": 1, "casts": [{"id", "name", "saved", "people": [person]}]}
  person        {"name", "desc", "lead", "faces": n, "voice": what voice_library.keep_voice keeps (or absent)}
  <id>/<k>/     the k-th person's files: face1.png … (face1_full.png: the full picture it was cropped from),
                voice.wav, ref.wav
Only people are kept (places, objects and styles are not). A person brought into a project is copied into it
(tree.put_person, tree.put_voice): a project never points into the library."""
from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Sequence
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from . import tree as T
from .voice_library import FolderStore, keep_voice, put_kept_voice


def default_dir() -> Path:
    env = os.environ.get("VIDEO_CHAIN_CASTS")
    return Path(env) if env else config.data_dir() / "casts"


def people_of(src: T.Tree) -> list[T.CastEntry]:
    """The people of a cast (what the library keeps)."""
    return [c for c in T.cast_of(src) if T.kind_of(c) == "person"]


@dataclass
class Choice:
    """What becomes of one saved person when a cast is brought in: as (the name it gets in the project) and
    replace (a character of that name is replaced; else the name must be free)."""
    person: str
    as_name: str
    replace: bool = False


class CastLibrary(FolderStore):
    KEY = "casts"

    def __init__(self, folder: Path | None = None):
        super().__init__(Path(folder) if folder else default_dir())

    def list(self) -> list[dict]:
        """The saved casts (with their people) whose files are there, by name."""
        return [c for c in self._entries() if (self.dir / c["id"]).is_dir() and isinstance(c.get("people"), list)]

    def get(self, cid: str) -> dict | None:
        return next((c for c in self.list() if c["id"] == cid), None)

    def person_dir(self, cid: str, k: int) -> Path:
        return self.dir / cid / str(k)

    def faces(self, cid: str, k: int, person: dict) -> Sequence[tuple[Path, Path | None]]:
        """A saved person's face pictures with their full pictures (None: none kept)."""
        d = self.person_dir(cid, k)
        out: list[tuple[Path, Path | None]] = []
        for j in range(1, int(person.get("faces") or 0) + 1):
            full = d / f"face{j}_full.png"
            out.append((d / f"face{j}.png", full if full.is_file() else None))
        return out

    def save(self, project: Path, tree: T.Tree, name: str) -> dict:
        """Keep the people of a project's cast (face pictures, descriptions, the lead, voices) under name."""
        people = people_of(tree)
        if not people:
            raise ValueError("the cast has no people to save")
        cid, folder = self._new_folder()
        with self._exclusive():
            data = self._read(update=True)
            name = self._name(data["casts"], name)
            folder.mkdir(parents=True)
            try:
                kept = []
                for k, c in enumerate(people):
                    d = self.person_dir(cid, k)
                    d.mkdir()
                    for j, rel in enumerate(c["images"], start=1):
                        shutil.copyfile(Path(project) / rel, d / f"face{j}.png")
                        full = Path(project) / T.source_of(rel)
                        if full.is_file():
                            shutil.copyfile(full, d / f"face{j}_full.png")
                    p: dict = {"name": c["name"], "desc": c.get("desc", ""), "lead": bool(c.get("lead")),
                               "faces": len(c["images"])}
                    if c.get("voice"):
                        p["voice"] = keep_voice(project, c, d)
                    kept.append(p)
                entry = {"id": cid, "name": name, "saved": time.time(), "people": kept}
                data["casts"].append(entry)
                self._write(data)
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return entry

    def bring_in(self, cid: str, project: Path, tree: T.Tree, choices: Sequence[Choice]) -> Sequence[str]:
        """Bring the chosen people of a saved cast into a project (copied in). Every choice is checked first
        (names, the number of pictures), so nothing is changed when one of them cannot be made. Returns the
        names they got."""
        saved = self.get(cid)
        if saved is None:
            raise ValueError("that cast is not in the library any more")
        index = {p["name"]: k for k, p in enumerate(saved["people"])}
        probe = T.Tree()                             # (the checks on a copy of the cast, without files)
        probe.options["cast"] = [copy.copy(c) for c in T.cast_of(tree)]
        for ch in choices:
            if ch.person not in index:
                raise ValueError(f"@{ch.person} is not in the saved cast “{saved['name']}”")
            old = T.character(probe, ch.as_name) if ch.replace else None
            name = T.check_name(probe, ch.as_name, keep=old["name"] if old else None)
            faces = int(saved["people"][index[ch.person]].get("faces") or 0)
            if faces > T.MAX_PICTURES_PER_CHARACTER:
                raise ValueError(f"at most {T.MAX_PICTURES_PER_CHARACTER} pictures per character")
            rest = [c for c in T.cast_of(probe) if c is not old]
            pictures = sum(len(c["images"]) for c in rest) + faces
            if pictures > T.MAX_CAST_PICTURES:
                raise ValueError(f"at most {T.MAX_CAST_PICTURES} face pictures in a project ({pictures} with "
                                 f"@{name})")
            probe.options["cast"] = rest + [T.CastEntry(name=name, images=["-"] * faces, desc="", lead=False)]
        names, had_lead, lead = [], T.lead_of(tree) is not None, None
        for ch in choices:
            k = index[ch.person]
            p = saved["people"][k]
            entry = T.put_person(project, tree, ch.as_name, self.faces(cid, k, p), p.get("desc", ""),
                                 replace=ch.replace)
            if p.get("voice"):
                put_kept_voice(project, tree, entry["name"], self.person_dir(cid, k), p["voice"],
                               f"{p['name']} ({saved['name']})")
            names.append(entry["name"])
            if p.get("lead"):
                lead = entry["name"]
        if lead and not had_lead:                     # (the saved lead, when the project had none)
            T.update_character(tree, lead, lead=True)
        return names


def library() -> CastLibrary:
    """The app's cast library (at default_dir(), read fresh each time)."""
    return CastLibrary()
