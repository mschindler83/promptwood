"""The prompt library: prompts kept for the app, not for one project, to use again (a clip's description, an image
edit or a new image's prompt). In the app's data folder ($XDG_DATA_HOME/video-chain/prompts, VIDEO_CHAIN_PROMPTS):
  library.json   {"version": 1, "prompts": [{"id", "name", "kind", "folder", "text", "saved", "updated"}]}
kind: one of KINDS; folder: "" (none) or a folder name the prompt is sorted into. A name is used once per kind and
folder (case does not matter). A prompt is copied into the field it is used in, so changing the library changes no
project."""
from __future__ import annotations

import builtins
import os
import time
from pathlib import Path

from . import config
from .media import squash
from .voice_library import FolderStore

KINDS = {"video": "Video", "edit": "Image edit", "create": "Image create"}


def default_dir() -> Path:
    env = os.environ.get("VIDEO_CHAIN_PROMPTS")
    return Path(env) if env else config.data_dir() / "prompts"


def _folder_of(p: dict) -> str:
    f = p.get("folder")
    return squash(f) if isinstance(f, str) else ""


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"no such kind of prompt: {kind}")
    return kind


class PromptLibrary(FolderStore):
    KEY = "prompts"

    def __init__(self, folder: Path | None = None):
        super().__init__(Path(folder) if folder else default_dir())

    # ---- reading
    def list(self, kind: str | None = None) -> list[dict]:
        """The saved prompts (of one kind), by folder and name."""
        items = [p for p in self._entries() if isinstance(p.get("kind"), str) and p["kind"] in KINDS
                 and isinstance(p.get("text"), str) and (kind is None or p["kind"] == kind)]
        for p in items:
            p["folder"], p["name"] = _folder_of(p), str(p["name"])
        return sorted(items, key=lambda p: (p["folder"].lower(), str(p["name"]).lower()))

    def get(self, pid: str) -> dict | None:
        return next((p for p in self.list() if p["id"] == pid), None)

    def folders(self, kind: str) -> builtins.list[str]:
        """The folders that hold prompts of this kind, by name."""
        return sorted({p["folder"] for p in self.list(kind) if p["folder"]}, key=str.lower)

    def find(self, kind: str, folder: str, name: str) -> dict | None:
        """The prompt of that name in that folder (case does not matter), or None."""
        folder, name = squash(folder or "").lower(), squash(name or "").lower()
        return next((p for p in self.list(kind) if p["folder"].lower() == folder and p["name"].lower() == name),
                    None)

    # ---- changing
    @staticmethod
    def _canon(items, kind: str, folder: str) -> str:
        """The folder's spelling as the library has it (a folder differing only in case is the same folder)."""
        return next((_folder_of(p) for p in items if p.get("kind") == kind
                     and _folder_of(p).lower() == folder.lower()), folder) if folder else ""

    def _clash(self, items, kind: str, folder: str, name: str, keep: str | None = None) -> None:
        for p in items:
            if (p.get("kind") == kind and _folder_of(p).lower() == folder.lower()
                    and str(p.get("name") or "").lower() == name.lower() and p.get("id") != keep):
                where = f" in “{folder}”" if folder else ""
                raise ValueError(f"there is already a prompt “{name}”{where}")

    def save(self, kind: str, name: str, text: str, folder: str = "", replace: bool = False) -> dict:
        """Keep a prompt. A name already in that folder: ValueError, unless replace (its text is replaced)."""
        kind, name, folder = _kind(kind), squash(name), squash(folder or "")
        if not name:
            raise ValueError("a name is needed")
        text = (text or "").strip()
        if not text:
            raise ValueError("the prompt is empty: nothing to save")
        now = time.strftime("%Y-%m-%d %H:%M")
        with self._exclusive():
            data = self._read(update=True)
            folder = self._canon(data[self.KEY], kind, folder)
            old = next((p for p in data[self.KEY] if p.get("kind") == kind
                        and _folder_of(p).lower() == folder.lower()
                        and str(p.get("name") or "").lower() == name.lower()), None)
            if old is not None and not replace:
                self._clash(data[self.KEY], kind, folder, name)
            if old is not None:
                old.update({"text": text, "updated": now})
                entry = old
            else:
                pid, _ = self._new_folder()
                entry = {"id": pid, "name": name, "kind": kind, "folder": folder, "text": text, "saved": now,
                         "updated": now}
                data[self.KEY].append(entry)
            self._write(data)
        return dict(entry)

    def _change(self, pid: str, change) -> dict:
        with self._exclusive():
            data = self._read(update=True)
            p = next((p for p in data[self.KEY] if p.get("id") == pid), None)
            if p is None:
                raise ValueError("that prompt is not in the library any more")
            change(data[self.KEY], p)
            self._write(data)
            return dict(p)

    def update_text(self, pid: str, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("the prompt is empty")

        def change(_items, p):
            p.update({"text": text, "updated": time.strftime("%Y-%m-%d %H:%M")})
        return self._change(pid, change)

    def rename(self, vid: str, new: str) -> None:
        new = squash(new)
        if not new:
            raise ValueError("a name is needed")

        def change(items, p):
            self._clash(items, p.get("kind"), _folder_of(p), new, keep=p.get("id"))
            p["name"] = new
        self._change(vid, change)

    def move(self, pid: str, folder: str) -> dict:
        """Into another folder ("" : none)."""
        folder = squash(folder or "")

        def change(items, p):
            into = self._canon([q for q in items if q is not p], p.get("kind"), folder)
            self._clash(items, p.get("kind"), into, str(p.get("name") or ""), keep=p.get("id"))
            p["folder"] = into
        return self._change(pid, change)

    def rename_folder(self, kind: str, old: str, new: str) -> int:
        """Rename a folder of this kind (its prompts move along; into a folder that exists: both are one). Returns
        how many prompts moved. A name both folders have: ValueError, nothing changes."""
        kind, old, new = _kind(kind), squash(old or ""), squash(new or "")
        if not old or not new:
            raise ValueError("a folder name is needed")
        with self._exclusive():
            data = self._read(update=True)
            moving = [p for p in data[self.KEY] if p.get("kind") == kind and _folder_of(p).lower() == old.lower()]
            ids = {p.get("id") for p in moving}
            if new.lower() != old.lower():             # (only its case changes: the new spelling as typed)
                new = self._canon([q for q in data[self.KEY] if q.get("id") not in ids], kind, new)
            for p in moving:
                self._clash([q for q in data[self.KEY] if q.get("id") not in ids], kind, new, str(p["name"]))
            for p in moving:
                p["folder"] = new
            if moving:
                self._write(data)
        return len(moving)

    def delete_folder(self, kind: str, folder: str) -> int:
        """Take a folder of this kind and its prompts out of the library. Returns how many went."""
        kind, folder = _kind(kind), squash(folder or "")
        if not folder:
            raise ValueError("a folder name is needed")
        with self._exclusive():
            data = self._read(update=True)
            left = [p for p in data[self.KEY] if not (p.get("kind") == kind and _folder_of(p).lower() == folder.lower())]
            gone = len(data[self.KEY]) - len(left)
            if gone:
                data[self.KEY] = left
                self._write(data)
        return gone


def library() -> PromptLibrary:
    return PromptLibrary()
