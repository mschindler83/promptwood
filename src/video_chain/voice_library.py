"""The voice library: voices kept for the app, not for one project, so a cloned or recorded voice can be given to a
character in any project. In the app's data folder ($XDG_DATA_HOME/video-chain/voices, VIDEO_CHAIN_VOICES):
  library.json   {"version": 1, "voices": [{"id", "name", "secs", "text", "ref_secs", "eleven", "saved"}]}
  <id>/voice.wav the short sample the clip model hears
  <id>/ref.wav   its longer clone reference (when the voice had one)
with text what the clone file (ref.wav, else voice.wav) says and eleven ({"id", "name"}) its ElevenLabs voice.
A voice taken from the library is copied into the project (tree.put_voice): a project never points into the
library, so renaming or deleting a library voice changes no project."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from . import config, media
from . import tree as T
from .media import squash

LIBRARY_FILE = "library.json"


def default_dir() -> Path:
    env = os.environ.get("VIDEO_CHAIN_VOICES")
    return Path(env) if env else config.data_dir() / "voices"


class FolderStore:
    """A list of named entries in <folder>/library.json (under KEY), each with its files in <folder>/<id>/; one
    writer at a time (threads and processes), a damaged list is kept as a .bak file."""
    KEY = "items"

    def __init__(self, folder: Path):
        self.dir = Path(folder)
        self.path = self.dir / LIBRARY_FILE
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def _exclusive(self):
        """This process's threads and other processes (a second window) one at a time."""
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / (LIBRARY_FILE + ".lock"), "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def _read(self, update: bool = False) -> dict:
        """The file's content. update=True (before a write): an unreadable file raises OSError, so nothing is
        written over it; a damaged one (not JSON) is moved to a .bak file first and a new one is started."""
        empty: dict = {"version": 1, self.KEY: []}
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return empty
        except OSError:
            if update:
                raise
            return empty
        try:
            d = json.loads(text)
            if isinstance(d, dict) and isinstance(d.get(self.KEY), list):
                return d
        except ValueError:
            pass
        if update:
            os.replace(self.path, self.path.with_name(f"{self.path.name}.{int(time.time() * 1000)}.bak"))
        return empty

    def _write(self, data: dict) -> None:
        media.atomic_write_json(self.path, data, indent=1)

    def _entries(self) -> list[dict]:
        """The entries with an id and a name, by name."""
        with self._lock:
            items = [dict(v) for v in self._read()[self.KEY]
                     if isinstance(v, dict) and isinstance(v.get("id"), str) and v.get("name")]
        return sorted(items, key=lambda v: str(v["name"]).lower())

    def _name(self, items: Sequence[dict], name: str, keep: str | None = None) -> str:
        name = squash(name)
        if not name:
            raise ValueError("a name is needed")
        if any(v.get("name") == name and v.get("id") != keep for v in items):
            raise ValueError(f"there is already “{name}” in the library")
        return name

    def _new_folder(self) -> tuple[str, Path]:
        vid = uuid.uuid4().hex[:12]
        return vid, self.dir / vid

    def rename(self, vid: str, new: str) -> None:
        with self._exclusive():
            data = self._read(update=True)
            new = self._name(data[self.KEY], new, keep=vid)
            for v in data[self.KEY]:
                if v.get("id") == vid:
                    v["name"] = new
                    break
            else:
                raise ValueError("that is not in the library any more")
            self._write(data)

    def delete(self, vid: str) -> None:
        """Take an entry out of the library (projects that use it keep their own copy)."""
        with self._exclusive():
            data = self._read(update=True)
            left = [v for v in data[self.KEY] if v.get("id") != vid]
            if len(left) != len(data[self.KEY]):
                data[self.KEY] = left
                self._write(data)
            if vid and "/" not in vid and vid not in (".", ".."):
                shutil.rmtree(self.dir / vid, ignore_errors=True)


class VoiceLibrary(FolderStore):
    KEY = "voices"

    def __init__(self, folder: Path | None = None):
        super().__init__(Path(folder) if folder else default_dir())

    # ---- reading
    def list(self) -> list[dict]:
        """The saved voices whose sample is there, by name."""
        return [v for v in self._entries() if self.voice_file(v["id"]).is_file()]

    def get(self, vid: str) -> dict | None:
        return next((v for v in self.list() if v["id"] == vid), None)

    def voice_file(self, vid: str) -> Path:
        return self.dir / vid / "voice.wav"

    def ref_file(self, vid: str) -> Path | None:
        f = self.dir / vid / "ref.wav"
        return f if f.is_file() else None

    # ---- writing
    def save(self, project: Path, character: T.CastEntry, name: str) -> dict:
        """Keep a character's voice (its sample, clone reference, transcript and ElevenLabs voice) under name."""
        if not character.get("voice"):
            raise ValueError(f"@{character.get('name')} has no voice to save")
        vid, folder = self._new_folder()
        with self._exclusive():
            data = self._read(update=True)
            name = self._name(data["voices"], name)
            folder.mkdir(parents=True)
            try:
                entry = {"id": vid, "name": name, **keep_voice(project, character, folder), "saved": time.time()}
                data["voices"].append(entry)
                self._write(data)
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return entry

    def use(self, vid: str, project: Path, tree: T.Tree, name: str) -> dict:
        """A saved voice becomes the voice of the character name in a project (copied in, with its transcript and
        ElevenLabs voice; a character with an ElevenLabs voice of its own and none saved keeps none: the voice
        is replaced as a whole). Returns the project's voice entry."""
        v = self.get(vid)
        if v is None:
            raise ValueError("that voice is not in the library any more")
        return put_kept_voice(project, tree, name, self.dir / vid, v, v["name"])


def keep_voice(project: Path, character: T.CastEntry, folder: Path) -> dict:
    """A character's voice copied into folder (voice.wav, ref.wav); returns what is kept of it ("secs",
    "ref_secs", "text", "eleven")."""
    v = character.get("voice")
    assert v
    shutil.copyfile(Path(project) / v["file"], folder / "voice.wav")
    out: dict = {"secs": float(v.get("secs") or 0)}
    ref = v.get("ref")
    if ref:
        shutil.copyfile(Path(project) / ref["file"], folder / "ref.wav")
        out["ref_secs"] = float(ref.get("secs") or 0)
    if v.get("text"):
        out["text"] = v["text"]
    el = (character.get("tts") or {}).get("eleven")
    if el:
        out["eleven"] = {"id": el["id"], "name": el.get("name") or el["id"]}
    return out


def put_kept_voice(project: Path, tree: T.Tree, name: str, folder: Path, kept: dict, label: str) -> dict:
    """A voice kept in folder (keep_voice) becomes the voice of the character name (with its ElevenLabs voice, or
    none). Returns the project's voice entry."""
    from . import tts as TTS                       # (tts imports a lot; only needed here)
    ref = folder / "ref.wav"
    entry = T.put_voice(project, tree, name, folder / "voice.wav", ref if ref.is_file() else None, label=label,
                        text=str(kept.get("text") or ""))
    TTS.set_voice_fields(tree, name, eleven=kept.get("eleven") or None)
    return dict(entry)


def library() -> VoiceLibrary:
    """The app's voice library (at default_dir(), read fresh each time)."""
    return VoiceLibrary()
