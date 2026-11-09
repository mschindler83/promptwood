"""Trigger words of LoRAs: a small database (one JSON file for the app), filled from the LoRA files themselves and,
in the background, from CivitAI (looked up by the file's SHA-256, as civitai.com shows them as "trained words").

Which words count, best first: words the user set; an explicit trigger field in the file (modelspec.trigger_phrase /
trigger_word); CivitAI's trained words (an empty list there means the creator gives none); a guess from kohya's
training tags (the tags on nearly every training image). A LoRA CivitAI does not know is asked again after
RECHECK_DAYS; one whose lookup failed (network) is asked again next time.

The database is keyed by the file's content hash, so a LoRA keeps its words when it is moved or renamed; the hash of
each path is cached by (size, date).

Everything CivitAI tells about a LoRA is kept too (details_path(): one JSON file per hash next to the database): the
model version (base model, description, files, images), the model page (its description, often with sample prompts,
tags, creator, license) and the generation data of its example images (prompts, seeds, settings; text only, no
pictures are downloaded). LoRAs whose files name their trigger are looked up for these details as well."""
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


KEYRING_SERVICE, KEYRING_USER = "video-chain", "civitai-api-key"
_API_KEY: str | None = None


def api_key() -> str:
    """The CivitAI API key: $CIVITAI_API_KEY, else the one kept in the system keyring (set_api_key); "" if none.
    With it CivitAI answers as your account (e.g. the generation data of example images it holds back otherwise)."""
    global _API_KEY
    env = os.environ.get("CIVITAI_API_KEY", "").strip()
    if env:
        return env
    if _API_KEY is None:
        try:
            import keyring
            _API_KEY = (keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or "").strip()
        except Exception:  # noqa: BLE001  (no keyring backend: no key)
            _API_KEY = ""
    return _API_KEY


def set_api_key(key: str | None) -> None:
    """Keep the CivitAI API key in the system keyring (empty / None: remove it). Raises RuntimeError when there is
    no keyring to keep it in."""
    global _API_KEY
    key = (key or "").strip()
    try:
        import keyring
        import keyring.errors
        if key:
            keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
            except keyring.errors.PasswordDeleteError:
                pass
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the system keyring is not available ({e})") from e
    _API_KEY = key


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
    try:
        return _file_words(path)
    except Exception:  # noqa: BLE001  (an odd header must not stop a scan)
        return None, ""


def _file_words(path: str | Path) -> tuple[list[str] | None, str]:
    """Trigger words in a LoRA file's header: (words, 'file') from an explicit field, (words, 'tags') guessed from
    kohya's training tags, or (None, '') when the file says nothing."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 100 * 2**20:
                return None, ""
            head = json.loads(f.read(n))
    except (OSError, ValueError):
        return None, ""
    meta = head.get("__metadata__") if isinstance(head, dict) else None
    if not isinstance(meta, dict):
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


def _get_json(path: str, timeout: float) -> dict | None:
    """A CivitAI API answer (the first site that answers); None for 404. OSError when no site could be asked."""
    last: Exception | None = None
    for site in SITES:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        key = api_key()
        if key:                                        # (only ever sent to CivitAI's own sites)
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{site}{path}", headers=headers)
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
        return data if isinstance(data, dict) and not data.get("error") else None
    raise OSError(f"CivitAI could not be asked: {last}")


EXAMPLE_IMAGES = 100        # example images whose generation data (prompts) is kept
DETAILS_VERSION = 2         # raised when the details asked for change: every LoRA is asked once more
                            # (2: the example images' generation data, which CivitAI sends only with withMeta=true)


def civitai_lookup(sha: str, timeout: float = 20.0, gap: float = REQUEST_GAP) -> dict | None:
    """CivitAI's model version for a file hash: {"words", "model", "url", "details"}; None when CivitAI does not
    know it. details = {"version": the version as CivitAI gives it, "model": the model page (description, tags,
    creator, license; its other versions only as id / name / base model), "examples": the example images (their
    generation data: prompt, negative prompt, seed, settings)}. The model page and the examples are optional (a
    failed request leaves them out). Raises OSError when the version could not be asked (offline, server error)."""
    data = _get_json(f"/api/v1/model-versions/by-hash/{sha}", timeout)
    if data is None:
        return None
    model = data.get("model") or {}
    mid = data.get("modelId") or model.get("id")
    details: dict = {"version": data, "fetched": time.time()}
    if mid:
        try:
            time.sleep(gap)
            page = _get_json(f"/api/v1/models/{int(mid)}", timeout)
            if page:
                page = dict(page)
                page["modelVersions"] = [{"id": v.get("id"), "name": v.get("name"), "baseModel": v.get("baseModel")}
                                         for v in page.get("modelVersions") or [] if isinstance(v, dict)]
                details["model"] = page
        except (OSError, ValueError, TypeError):
            pass
    if data.get("id"):
        try:
            time.sleep(gap)
            imgs = _get_json(f"/api/v1/images?modelVersionId={int(data['id'])}&limit={EXAMPLE_IMAGES}"
                             "&sort=Most%20Reactions&nsfw=X&withMeta=true", timeout)
            if imgs and isinstance(imgs.get("items"), list):
                details["examples"] = imgs["items"]
        except (OSError, ValueError, TypeError):
            pass
    return {"words": _clean(data.get("trainedWords") or []), "model": str(model.get("name") or ""),
            "url": f"https://civitai.com/models/{mid}?modelVersionId={data.get('id')}" if mid else "",
            "details": details}


def example_prompts(details: dict | None) -> list[dict]:
    """The generation data of a LoRA's example images (details from civitai_lookup), each prompt once:
    [{"prompt", "negative", "seed", "steps", "cfg", "sampler", "image"}], the most liked first."""
    out, seen = [], set()
    for it in (details or {}).get("examples") or []:
        meta = (it or {}).get("meta") or {}
        if not isinstance(meta, dict):
            continue
        prompt = str(meta.get("prompt") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        out.append({"prompt": prompt, "negative": str(meta.get("negativePrompt") or "").strip(),
                    "seed": meta.get("seed"), "steps": meta.get("steps"), "cfg": meta.get("cfgScale"),
                    "sampler": meta.get("sampler"), "image": it.get("url") or ""})
    return out


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
        """Write the database; what another process added in between is kept (merged, ours first)."""
        import tempfile
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                disk = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                disk = {}
            if isinstance(disk, dict):
                for part in ("by_sha", "files"):
                    for k, v in (disk.get(part) or {}).items():
                        self.data[part].setdefault(k, v)
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".part", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=1, ensure_ascii=False)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            self._mtime = self.path.stat().st_mtime_ns

    def details_path(self, sha: str) -> Path:
        return self.path.parent / "civitai" / f"{sha}.json"

    def details(self, path: str) -> dict | None:
        """Everything CivitAI told about a LoRA (civitai_lookup's details), None when there is nothing."""
        with self._lock:
            self._load()
            f = self._file(path)
            sha = (f or {}).get("sha256")
        if not sha:
            return None
        try:
            d = json.loads(self.details_path(sha).read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else None
        except (OSError, ValueError):
            return None

    def _save_details(self, sha: str, details: dict):
        import tempfile
        target = self.details_path(sha)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(details, f, ensure_ascii=False)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

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
            out = self._info(f, rec)
            if out is not None and rec.get("base_model"):
                out["base_model"] = rec["base_model"]
            return out

    @staticmethod
    def _info(f: dict, rec: dict) -> dict | None:
        user = f["user"] if f.get("user") is not None else rec.get("user")
        if user is not None:
            return {"words": list(user), "source": "user", "model": rec.get("model", ""),
                    "url": rec.get("url", "")}
        if f.get("file_source") == "file" and f.get("file_words"):
            return {"words": list(f["file_words"]), "source": "file", "model": rec.get("model", ""),
                    "url": rec.get("url", "")}
        if rec.get("civitai") is not None:
            return {"words": list(rec["civitai"]), "source": "civitai", "model": rec.get("model", ""),
                    "url": rec.get("url", "")}
        if f.get("file_words"):                # a guess: shown as a suggestion, not put into prompts
            return {"words": [], "source": "tags", "model": "", "url": "", "suggested": list(f["file_words"])}
        if rec.get("checked"):                 # CivitAI does not know it, the file says nothing
            return {"words": [], "source": "none", "model": "", "url": ""}
        return None                            # not looked up yet

    def words(self, path: str) -> list[str]:
        i = self.info(path)
        return list(i["words"]) if i else []

    def set_user(self, path: str, words: list[str] | None) -> None:
        """The user's own words for a LoRA (None: back to the found ones). Kept with the file (no hashing here, it
        may be big) and with its content hash when that is known (then also for its copies)."""
        with self._lock:
            f = self._file(path) or self._remember(path)
            rec = self.data["by_sha"].setdefault(f["sha256"], {}) if f.get("sha256") else None
            if words is None:
                f.pop("user", None)
                if rec is not None:
                    rec.pop("user", None)
            else:
                f["user"] = _clean(words)
                if rec is not None:
                    rec["user"] = f["user"]
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
            rec = self.data["by_sha"].get(f.get("sha256") or "") if f.get("sha256") else None
            if rec is None:
                return True                        # (also a LoRA whose file names its trigger: for the details)
            if rec.get("civitai") is not None:     # CivitAI knows it (its words, maybe none); asked again once
                if int(rec.get("details_v") or 0) < DETAILS_VERSION:   # when it was found before these details
                    return True                                         # were kept,
                return bool(api_key()) and not rec.get("asked_with_key")   # and once with an API key
            if rec.get("checked"):
                return time.time() - float(rec["checked"]) > RECHECK_DAYS * 86400
            return True

    def scan(self, paths: list[str], online: bool = True, stop: Callable[[], bool] = lambda: False,
             lookup: Callable[[str], dict | None] = civitai_lookup, gap: float = REQUEST_GAP) -> dict:
        """Look at LoRA files not known yet (header; then CivitAI by hash). Returns counts
        {"looked", "found", "unknown", "failed"}."""
        n = {"looked": 0, "found": 0, "unknown": 0, "failed": 0, "new": 0}
        asked, dirty, failed_in_a_row = False, 0, 0

        def flush(force=False):
            nonlocal dirty
            if dirty and (force or dirty >= 20):
                with self._lock:
                    self.save()
                dirty = 0
        for path in paths:
            if stop():
                break
            try:
                if not os.path.isfile(path):
                    continue
                with self._lock:
                    if self._file(path) is None:
                        self._remember(path)
                        n["new"] += 1
                        dirty += 1
                if not online or not self.needs_lookup(path):
                    flush()
                    continue
                with self._lock:
                    sha = (self._file(path) or {}).get("sha256")
                if not sha:
                    sha = sha256_of(path, stop)
                    if sha is None:
                        break
                    with self._lock:
                        f = self._file(path) or self._remember(path)   # (the data may have been re-read)
                        f["sha256"] = sha
                        if f.get("user") is not None:
                            self.data["by_sha"].setdefault(sha, {}).setdefault("user", f["user"])
                        dirty += 1
                    if not self.needs_lookup(path):    # known by its content (moved, renamed, a copy)
                        flush()
                        continue
                if asked and gap:
                    time.sleep(gap)
                asked = True
                n["looked"] += 1
                try:
                    hit = lookup(sha)
                except OSError:
                    n["failed"] += 1
                    failed_in_a_row += 1
                    if failed_in_a_row >= 2:           # offline: try again with the next scan
                        break
                    continue
                failed_in_a_row = 0
                with self._lock:
                    rec = self.data["by_sha"].setdefault(sha, {})
                    rec["checked"] = time.time()
                    if hit is None:
                        n["unknown"] += 1
                    else:
                        rec.update({"civitai": hit["words"], "model": hit["model"], "url": hit["url"]})
                        rec["asked_details"] = time.time()
                        rec["details_v"] = DETAILS_VERSION
                        if api_key():
                            rec["asked_with_key"] = time.time()
                        if hit.get("details"):
                            try:
                                self._save_details(sha, hit["details"])
                                rec["details"] = True
                                base = (hit["details"].get("version") or {}).get("baseModel")
                                if base:
                                    rec["base_model"] = str(base)
                            except (OSError, TypeError, ValueError):
                                pass
                        n["found"] += bool(hit["words"])
                    dirty += 1
                flush()
            except (OSError, ValueError, TypeError, AttributeError):
                continue                               # a file that vanished or cannot be read: the next one
        flush(force=True)
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
    if (prompt or "").startswith("subject_definitions:") and re.search(r"\[Shot 1\]\s+", prompt):
        return re.sub(r"\[Shot 1\]\s+", lambda _m: f"[Shot 1] {lead}. ", prompt, count=1)
    return f"{lead}, {prompt}" if prompt else lead
