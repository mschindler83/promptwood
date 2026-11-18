"""The cast library: a project's whole cast kept for the app, so its entries can be brought into any project.
In the app's data folder ($XDG_DATA_HOME/video-chain/casts, VIDEO_CHAIN_CASTS):
  library.json  {"version": 1, "casts": [{"id", "name", "saved", "entries": [entry]}]}
  entry         {"name", "kind" (absent: a person), "desc", "lead", "faces": n (its pictures),
                 "voice": what voice_library.keep_voice keeps (a person's, or absent)}
  <id>/<k>/     the k-th entry's files: face1.png … (its pictures; face1_full.png: the full picture one was
                cropped from), voice.wav, ref.wav
People, places, objects and styles are kept; casts saved when only people were kept list them under "people".
An entry brought into a project is copied into it (tree.put_entry, tree.put_voice): a project never points into
the library."""
from __future__ import annotations

import copy
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config
from . import tree as T
from .voice_library import FolderStore, keep_voice, put_kept_voice


def default_dir() -> Path:
    env = os.environ.get("VIDEO_CHAIN_CASTS")
    return Path(env) if env else config.data_dir() / "casts"


@dataclass
class Choice:
    """What becomes of one saved entry (name) when a cast is brought in: as_name (the name it gets in the
    project) and replace (an entry of that name is replaced; else the name must be free)."""
    name: str
    as_name: str
    replace: bool = False


def kind_of(saved: dict) -> str:
    """A saved entry's kind (person when it has none)."""
    k = saved.get("kind")
    return k if k in T.CAST_KINDS else "person"


class CastLibrary(FolderStore):
    KEY = "casts"

    def __init__(self, folder: Path | None = None):
        super().__init__(Path(folder) if folder else default_dir())

    def list(self) -> list[dict]:
        """The saved casts whose files are there, by name, each with its "entries" (also an old one's people)."""
        out = []
        for c in self._entries():
            entries = c.get("entries", c.get("people"))
            if (self.dir / c["id"]).is_dir() and isinstance(entries, list):
                c.pop("people", None)
                out.append({**c, "entries": [e for e in entries if isinstance(e, dict) and e.get("name")]})
        return out

    def get(self, cid: str) -> dict | None:
        return next((c for c in self.list() if c["id"] == cid), None)

    def entry_dir(self, cid: str, k: int) -> Path:
        return self.dir / cid / str(k)

    def faces(self, cid: str, k: int, saved: dict) -> Sequence[tuple[Path, Path | None]]:
        """A saved entry's pictures with their full pictures (None: none kept)."""
        d = self.entry_dir(cid, k)
        out: list[tuple[Path, Path | None]] = []
        for j in range(1, int(saved.get("faces") or 0) + 1):
            full = d / f"face{j}_full.png"
            out.append((d / f"face{j}.png", full if full.is_file() else None))
        return out

    @staticmethod
    def roles(saved: dict) -> Sequence[str] | None:
        """A saved person's picture roles (one per picture; None: plain pictures, or ones not readable)."""
        roles = saved.get("roles")
        n = int(saved.get("faces") or 0)
        if not isinstance(roles, list) or len(roles) != n:
            return None
        return [r if r in T.PICTURE_ROLES else "" for r in roles]

    def source(self, cid: str, k: int, saved: dict) -> Path | None:
        """The photo a saved person's sheets were made from (None: none kept)."""
        f = self.entry_dir(cid, k) / "source.png"
        return f if saved.get("source") and f.is_file() else None

    def save(self, project: Path, tree: T.Tree, name: str) -> dict:
        """Keep a project's whole cast (pictures, descriptions, kinds, the lead, voices) under name."""
        cast = T.cast_of(tree)
        if not cast:
            raise ValueError("the cast is empty: nothing to save")
        cid, folder = self._new_folder()
        with self._exclusive():
            data = self._read(update=True)
            name = self._name(data["casts"], name)
            folder.mkdir(parents=True)
            try:
                kept = []
                for k, c in enumerate(cast):
                    d = self.entry_dir(cid, k)
                    d.mkdir()
                    for j, rel in enumerate(c["images"], start=1):
                        shutil.copyfile(Path(project) / rel, d / f"face{j}.png")
                        full = Path(project) / T.source_of(rel)
                        if full.is_file():
                            shutil.copyfile(full, d / f"face{j}_full.png")
                    p: dict = {"name": c["name"], "desc": c.get("desc", ""), "lead": bool(c.get("lead")),
                               "faces": len(c["images"])}
                    if T.kind_of(c) != "person":
                        p["kind"] = T.kind_of(c)
                    roles = [T.role_of(c, rel) for rel in c["images"]]
                    if any(roles):
                        p["roles"] = roles                      # (absent: plain pictures, as before)
                    src = T.sheet_source(c) if T.kind_of(c) == "person" else None
                    if src and (Path(project) / src).is_file():
                        shutil.copyfile(Path(project) / src, d / "source.png")
                        p["source"] = True
                    if c.get("voice") and T.kind_of(c) == "person":
                        p["voice"] = keep_voice(project, c, d)
                    kept.append(p)
                entry = {"id": cid, "name": name, "saved": time.time(), "entries": kept}
                data["casts"].append(entry)
                self._write(data)
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return entry

    def bring_in(self, cid: str, project: Path, tree: T.Tree, choices: Sequence[Choice]) -> Sequence[str]:
        """Bring the chosen entries of a saved cast into a project (copied in). Every choice is checked first
        (names, the number of pictures, the saved files), so nothing is changed when one of them cannot be made.
        Returns the names they got."""
        saved = self.get(cid)
        if saved is None:
            raise ValueError("that cast is not in the library any more")
        index = {p["name"]: k for k, p in enumerate(saved["entries"])}
        probe = T.Tree()                             # (the checks on a copy of the cast, without files)
        probe.options["cast"] = [copy.copy(c) for c in T.cast_of(tree)]
        for ch in choices:
            if ch.name not in index:
                raise ValueError(f"@{ch.name} is not in the saved cast “{saved['name']}”")
            old = T.character(probe, ch.as_name) if ch.replace else None
            name = T.check_name(probe, ch.as_name, keep=old["name"] if old else None)
            faces = int(saved["entries"][index[ch.name]].get("faces") or 0)
            if faces > T.MAX_PICTURES_PER_CHARACTER:
                raise ValueError(f"at most {T.MAX_PICTURES_PER_CHARACTER} pictures per character")
            rest = [c for c in T.cast_of(probe) if c is not old]
            pictures = sum(len(c["images"]) for c in rest) + faces
            if pictures > T.MAX_CAST_PICTURES:
                raise ValueError(f"at most {T.MAX_CAST_PICTURES} pictures in a project ({pictures} with "
                                 f"@{name})")
            k = index[ch.name]
            need = [f for f, _full in self.faces(cid, k, saved["entries"][k])]
            if saved["entries"][k].get("voice") and kind_of(saved["entries"][k]) == "person":
                need.append(self.entry_dir(cid, k) / "voice.wav")
            gone = [f for f in need if not f.is_file()]
            if gone:
                raise ValueError(f"files of @{ch.name} are missing in the cast library ({gone[0].name}): bring "
                                 "in the others, or save the cast again")
            placed = T.CastEntry(name=name, images=["-"] * faces, desc="", lead=False)
            if kind_of(saved["entries"][k]) != "person":
                placed["kind"] = kind_of(saved["entries"][k])
            probe.options["cast"] = rest + [placed]
        names, had_lead, lead = [], T.lead_of(tree) is not None, None
        for ch in choices:
            k = index[ch.name]
            p = saved["entries"][k]
            entry = T.put_entry(project, tree, ch.as_name, self.faces(cid, k, p), p.get("desc", ""),
                                replace=ch.replace, kind=kind_of(p), roles=self.roles(p),
                                source=self.source(cid, k, p))
            if p.get("voice") and kind_of(p) == "person":
                put_kept_voice(project, tree, entry["name"], self.entry_dir(cid, k), p["voice"],
                               f"{p['name']} ({saved['name']})")
            names.append(entry["name"])
            if p.get("lead") and kind_of(p) == "person":
                lead = entry["name"]
        if lead and not had_lead:                     # (the saved lead, when the project had none)
            T.update_character(tree, lead, lead=True)
        elif had_lead and T.lead_of(tree) is None:    # (the lead was replaced by a place, object or style)
            first = next((c["name"] for c in T.cast_of(tree) if T.kind_of(c) == "person"), None)
            if first:
                T.update_character(tree, lead or first, lead=True)
        return names


def library() -> CastLibrary:
    """The app's cast library (at default_dir(), read fresh each time)."""
    return CastLibrary()
