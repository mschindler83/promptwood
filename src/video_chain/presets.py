"""Named presets of a LoRA for new images: prompt, negative prompt, quality, sampling (steps, cfg, sampler,
scheduler), prompt variations and the LoRA's strength. Several per LoRA, kept in one JSON file for the app (next to the LoRA trigger
database), keyed by the LoRA's real path (a link and its target share them)."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path

FIELDS = ("prompt", "negative", "quality", "sampling", "strength", "family", "variations")


def default_path() -> Path:
    env = os.environ.get("VIDEO_CHAIN_PRESETS")
    if env:
        return Path(env)
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "video-chain" / "lora_presets.json"


def lora_key(path: str) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


class PresetStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_path()
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def _exclusive(self):
        """This process's threads and other processes (a second window, the worker) one at a time."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path.with_name(self.path.name + ".lock"), "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def _read(self, update: bool = False) -> dict:
        """The file's content. update=True (before a write): an unreadable file raises OSError, so nothing is
        written over it; a damaged one (not JSON) is moved to a .bak file first and a new one is started."""
        empty = {"version": 1, "loras": {}}
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
            if isinstance(d, dict) and isinstance(d.get("loras"), dict):
                return d
        except ValueError:
            pass
        if update:                                           # damaged: kept, not destroyed
            os.replace(self.path, self.path.with_name(f"{self.path.name}.{int(time.time() * 1000)}.bak"))
        return empty

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".part", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def list(self, lora: str) -> list[dict]:
        """The presets of a LoRA, by name."""
        with self._lock:
            items = (self._read()["loras"].get(lora_key(lora)) or {}).get("presets") or []
        return sorted((dict(p) for p in items if isinstance(p, dict) and p.get("name")),
                      key=lambda p: str(p["name"]).lower())

    def get(self, lora: str, name: str) -> dict | None:
        return next((p for p in self.list(lora) if p["name"] == name), None)

    def save(self, lora: str, name: str, values: dict) -> dict:
        """Save (or replace) the preset `name` of a LoRA; returns it."""
        name = " ".join(str(name or "").split())
        if not name:
            raise ValueError("a preset needs a name")
        preset = {"name": name, **{k: values.get(k) for k in FIELDS if k in values}, "updated": time.time()}
        with self._exclusive():
            data = self._read(update=True)
            entry = data["loras"].setdefault(lora_key(lora), {"file": Path(lora).name, "presets": []})
            entry["presets"] = [p for p in entry.get("presets") or [] if p.get("name") != name] + [preset]
            self._write(data)
        return preset

    def rename(self, lora: str, old: str, new: str) -> None:
        new = " ".join(str(new or "").split())
        if not new:
            raise ValueError("a preset needs a name")
        with self._exclusive():
            data = self._read(update=True)
            presets = (data["loras"].get(lora_key(lora)) or {}).get("presets") or []
            if not any(p.get("name") == old for p in presets):
                raise ValueError(f"no preset “{old}” (it may have been changed meanwhile)")
            if new != old and any(p.get("name") == new for p in presets):
                raise ValueError(f"there is already a preset “{new}”")
            for p in presets:
                if p.get("name") == old:
                    p["name"], p["updated"] = new, time.time()
            self._write(data)

    def delete(self, lora: str, name: str) -> None:
        with self._exclusive():
            data = self._read(update=True)
            key = lora_key(lora)
            entry = data["loras"].get(key)
            if not entry or not any(p.get("name") == name for p in entry.get("presets") or []):
                return                                       # (nothing to remove: nothing written)
            entry["presets"] = [p for p in entry["presets"] if p.get("name") != name]
            if not entry["presets"]:
                data["loras"].pop(key, None)
            self._write(data)


def store() -> PresetStore:
    """The app's presets (at default_path(), read fresh each time)."""
    return PresetStore()
