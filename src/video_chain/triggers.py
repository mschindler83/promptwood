"""Trigger words of LoRAs: a small database (one JSON file for the app), filled from the LoRA files themselves and,
in the background, from CivitAI (looked up by the file's SHA-256, as civitai.com shows them as "trained words").

Which words count, best first: words the user set; an explicit trigger field in the file (modelspec.trigger_phrase /
trigger_word); CivitAI's trained words (an empty list there means the creator gives none); a guess from kohya's
training tags (the tags on nearly every training image). A LoRA CivitAI does not know is asked again after
RECHECK_DAYS; one whose lookup failed (network) is asked again next time.

The database is keyed by the file's content hash, so a LoRA keeps its words when it is moved or renamed; the hash of
each path is cached by (size, date)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

RECHECK_DAYS = 30
SITES = ("https://civitai.com", "https://civitai.red")    # the same site; .red answers too when .com does not
REQUEST_GAP = 1.0                                          # seconds between lookups (be gentle to the API)
USER_AGENT = "video-chain (LoRA trigger lookup)"


def default_path() -> Path:
    env = os.environ.get("VIDEO_CHAIN_TRIGGERS")
    if env:
        return Path(env)
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "video-chain" / "lora_triggers.json"


def _clean(words) -> list[str]:
    out = []
    for w in words or []:
        for part in str(w).split(","):
            part = " ".join(part.split()).strip()
            if part and part.lower() not in [x.lower() for x in out]:
                out.append(part)
    return out


def file_words(path: str | Path) -> tuple[list[str] | None, str]:
    """Trigger words in a LoRA file's header: (words, 'file') from an explicit field, (words, 'tags') guessed from
    kohya's training tags, or (None, '') when the file says nothing."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 100 * 2**20:
                return None, ""
            meta = json.loads(f.read(n)).get("__metadata__") or {}
    except (OSError, ValueError):
        return None, ""
    for key in ("modelspec.trigger_phrase", "modelspec.trigger_word", "trigger_phrase", "trigger_word",
                "trigger_words", "ss_trigger_word"):
        if meta.get(key):
            val = meta[key]
            try:
                val = json.loads(val) if isinstance(val, str) and val.strip().startswith("[") else val
            except ValueError:
                pass
            words = _clean(val if isinstance(val, list) else [val])
            if words:
                return words, "file"
    return _tag_guess(meta), "tags"


def _tag_guess(meta: dict) -> list[str] | None:
    """The tags on (nearly) every training image of a kohya LoRA: usually the trigger. None without tag data."""
    try:
        freq = json.loads(meta.get("ss_tag_frequency") or "{}")
        dirs = json.loads(meta.get("ss_dataset_dirs") or "{}")
    except ValueError:
        return None
    if not isinstance(freq, dict) or not freq:
        return None
    counts: dict[str, int] = {}
    for tags in freq.values():
        if isinstance(tags, dict):
            for t, c in tags.items():
                counts[t.strip()] = counts.get(t.strip(), 0) + int(c or 0)
    images = sum(int((d or {}).get("img_count") or 0) for d in dirs.values()) if isinstance(dirs, dict) else 0
    if not counts:
        return None
    top = max(counts.values())
    images = images or top
    words = [t for t, c in sorted(counts.items(), key=lambda x: -x[1]) if t and c >= 0.9 * images][:3]
    return words or None


def sha256_of(path: str | Path, stop: Callable[[], bool] = lambda: False) -> str | None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            if stop():
                return None
            h.update(chunk)
    return h.hexdigest()


def civitai_lookup(sha: str, timeout: float = 20.0) -> dict | None:
    """CivitAI's model version for a file hash: {"words", "model", "url"}; None when CivitAI does not know it.
    Raises OSError when no site could be asked (offline, server error)."""
    last: Exception | None = None
    for site in SITES:
        req = urllib.request.Request(f"{site}/api/v1/model-versions/by-hash/{sha}",
                                     headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
            continue
        except (OSError, ValueError) as e:
            last = e
            continue
        if not isinstance(data, dict) or data.get("error"):
            return None
        model = data.get("model") or {}
        mid = data.get("modelId") or model.get("id")
        return {"words": _clean(data.get("trainedWords") or []), "model": str(model.get("name") or ""),
                "url": f"https://civitai.com/models/{mid}?modelVersionId={data.get('id')}" if mid else ""}
    raise OSError(f"CivitAI could not be asked: {last}")


class TriggerDB:
    """The trigger word database (thread-safe in this process; written atomically, re-read when another process
    changed it)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_path()
        self._lock = threading.RLock()
        self._mtime = None
        self.data = {"version": 1, "by_sha": {}, "files": {}}
        self._load()

    def _load(self):
        try:
            st = self.path.stat()
        except OSError:
            return
        if st.st_mtime_ns == self._mtime:
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                self.data = {"version": 1, "by_sha": dict(d.get("by_sha") or {}), "files": dict(d.get("files") or {})}
                self._mtime = st.st_mtime_ns
        except (OSError, ValueError):
            pass

    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".part")
            tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
            self._mtime = self.path.stat().st_mtime_ns

    def _file(self, path: str) -> dict | None:
        """The cached entry of a path while its size and date are unchanged."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        e = self.data["files"].get(str(path))
        if e and e.get("size") == st.st_size and e.get("mtime") == st.st_mtime_ns:
            return e
        return None

    def info(self, path: str) -> dict | None:
        """What is known of a LoRA: {"words", "source", "model", "url"} (source: user / file / civitai / tags /
        none), or None while it was not looked at yet."""
        with self._lock:
            self._load()
            f = self._file(path)
            if f is None:
                return None
            rec = self.data["by_sha"].get(f.get("sha256") or "") or {}
            if rec.get("user") is not None:
                return {"words": list(rec["user"]), "source": "user", "model": rec.get("model", ""),
                        "url": rec.get("url", "")}
            if f.get("file_source") == "file" and f.get("file_words"):
                return {"words": list(f["file_words"]), "source": "file", "model": rec.get("model", ""),
                        "url": rec.get("url", "")}
            if rec.get("civitai") is not None:
                return {"words": list(rec["civitai"]), "source": "civitai", "model": rec.get("model", ""),
                        "url": rec.get("url", "")}
            if f.get("file_words"):
                return {"words": list(f["file_words"]), "source": "tags", "model": "", "url": ""}
            if rec.get("checked"):                 # CivitAI does not know it, the file says nothing
                return {"words": [], "source": "none", "model": "", "url": ""}
            return None                            # not looked up yet

    def words(self, path: str) -> list[str]:
        i = self.info(path)
        return list(i["words"]) if i else []

    def set_user(self, path: str, words: list[str] | None) -> None:
        """The user's own words for a LoRA (None: back to the found ones)."""
        with self._lock:
            f = self._file(path) or self._remember(path)
            sha = f.get("sha256") or sha256_of(path)
            f["sha256"] = sha
            rec = self.data["by_sha"].setdefault(sha, {})
            if words is None:
                rec.pop("user", None)
            else:
                rec["user"] = _clean(words)
            self.save()

    def _remember(self, path: str) -> dict:
        st = os.stat(path)
        words, source = file_words(path)
        f = {"size": st.st_size, "mtime": st.st_mtime_ns, "file_words": words, "file_source": source}
        self.data["files"][str(path)] = f
        return f

    def needs_lookup(self, path: str) -> bool:
        with self._lock:
            f = self._file(path)
            if f is None:
                return True
            if f.get("file_source") == "file" and f.get("file_words"):
                return False                       # the file names its trigger
            rec = self.data["by_sha"].get(f.get("sha256") or "") if f.get("sha256") else None
            if rec is None:
                return True
            if rec.get("civitai") is not None:
                return False                       # CivitAI knows it (its words, maybe none)
            if rec.get("checked"):
                return time.time() - float(rec["checked"]) > RECHECK_DAYS * 86400
            return True

    def scan(self, paths: list[str], online: bool = True, stop: Callable[[], bool] = lambda: False,
             lookup: Callable[[str], dict | None] = civitai_lookup, gap: float = REQUEST_GAP) -> dict:
        """Look at LoRA files not known yet (header; then CivitAI by hash). Returns counts
        {"looked", "found", "unknown", "failed"}."""
        n = {"looked": 0, "found": 0, "unknown": 0, "failed": 0}
        asked = False
        for path in paths:
            if stop():
                break
            if not os.path.isfile(path):
                continue
            with self._lock:
                f = self._file(path)
                if f is None:
                    f = self._remember(path)
                    self.save()
            if not online or not self.needs_lookup(path):
                continue
            sha = f.get("sha256")
            if not sha:
                sha = sha256_of(path, stop)
                if sha is None:
                    break
                with self._lock:
                    f["sha256"] = sha
                if not self.needs_lookup(path):    # known by its content (moved, renamed, a copy)
                    with self._lock:
                        self.save()
                    continue
            if asked and gap:
                time.sleep(gap)
            asked = True
            n["looked"] += 1
            try:
                hit = lookup(sha)
            except OSError:
                n["failed"] += 1
                with self._lock:
                    self.save()                    # the hash is kept; asked again next time
                continue
            with self._lock:
                rec = self.data["by_sha"].setdefault(sha, {})
                rec["checked"] = time.time()
                if hit is None:
                    n["unknown"] += 1
                else:
                    rec.update({"civitai": hit["words"], "model": hit["model"], "url": hit["url"]})
                    n["found"] += bool(hit["words"])
                self.save()
        return n


_DB: TriggerDB | None = None
_DB_LOCK = threading.Lock()


def db() -> TriggerDB:
    """The app's database (one per process, at default_path())."""
    global _DB
    with _DB_LOCK:
        if _DB is None or _DB.path != default_path():
            _DB = TriggerDB()
        return _DB


def stack_words(paths: list[str], lookup: Callable[[str], list[str]] | None = None) -> list[str]:
    """The trigger words of a LoRA stack, in its order, each once."""
    get = lookup or (lambda p: db().words(p))
    out: list[str] = []
    for p in paths:
        for w in get(p):
            if w.lower() not in [x.lower() for x in out]:
                out.append(w)
    return out


def with_triggers(prompt: str, words: list[str]) -> str:
    """The prompt with the trigger words in front (those it does not contain yet). In MiniMax's structured
    format they go at the start of the shot description."""
    missing = [w for w in words if not re.search(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)", (prompt or "").lower())]
    if not missing:
        return prompt
    lead = ", ".join(missing)
    if (prompt or "").startswith("subject_definitions:") and "[Shot 1] " in prompt:
        return prompt.replace("[Shot 1] ", f"[Shot 1] {lead}. ", 1)
    return f"{lead}, {prompt}" if prompt else lead
