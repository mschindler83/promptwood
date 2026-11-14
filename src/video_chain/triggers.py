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
tags, creator, license) and the generation data of the pictures in its gallery, the creator's and other users'
(prompts, seeds, settings; text only: the lookup loads no picture. The LoRA window loads small stills of the
pictures it shows, lora_pictures.py). Next to it, <hash>.note.json: the prompt refiner's own note on the LoRA
(lora_notes.py). LoRAs whose files name their trigger are looked up for these details as well.

CivitAI has two sites, civitai.com and civitai.red (SITES). The one chosen in the settings (site(); civitai.com
unless chosen otherwise) is asked first and linked to; the other is asked when the first does not know a LoRA or
cannot be reached."""
from __future__ import annotations

import logging
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import net

logger = logging.getLogger(__name__)

RECHECK_DAYS = 30
SITES = ("https://civitai.com", "https://civitai.red")    # CivitAI's two sites (the first is the default)
_SITE: str | None = None


def site() -> str:
    """The CivitAI site to ask first and to link to: $VIDEO_CHAIN_CIVITAI_SITE, else the one chosen in the
    settings (set_site), else civitai.com. Always one of SITES."""
    for wanted in (os.environ.get("VIDEO_CHAIN_CIVITAI_SITE", ""), _SITE or ""):
        wanted = wanted.strip().lower().rstrip("/")
        for s in SITES:
            if wanted and wanted in (s, s.split("//", 1)[1]):
                return s
    return SITES[0]


def set_site(name: str | None) -> str:
    """Choose the site (its address or just "civitai.red"; None / unknown: the default); the one now used."""
    global _SITE
    _SITE = str(name or "") or None
    return site()


def sites() -> list[str]:
    """The sites in the order they are asked: the chosen one, then the other."""
    first = site()
    return [first] + [s for s in SITES if s != first]


def on_site(url: str) -> str:
    """A link to a CivitAI page as a link to the chosen site (the two show the same pages)."""
    for s in SITES:
        if url and (url == s or url.startswith(s + "/")):
            return site() + url[len(s):]
    return url
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
        except Exception:  # noqa: BLE001  (no keyring now, e.g. locked: no key this time, asked again later)
            return ""
    return _API_KEY


def set_api_key(key: str | None) -> None:
    """Keep the CivitAI API key in the system keyring (empty / None: remove it). Raises RuntimeError when there is
    no keyring to keep it in."""
    global _API_KEY
    key = (key or "").strip()
    _REJECTED.discard(key)                              # (a key saved anew gets another chance)
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


def _never() -> bool:
    return False


def sha256_of(path: str | Path, stop: Callable[[], bool] = _never) -> str | None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            if stop():
                return None
            h.update(chunk)
    return h.hexdigest()


MAX_ANSWER = 32 * 2**20      # bytes read from one CivitAI answer at most


class KeyRejected(OSError):
    """CivitAI turned the API key down (401 / 403)."""


_REJECTED: set[str] = set()     # keys CivitAI turned down (this run): not sent again until a new key is saved


def key_rejected() -> bool:
    """CivitAI turned the current API key down (the lookups go on without it)."""
    key = api_key()
    return bool(key) and key in _REJECTED


RETRY_WAIT_MAX = 30.0       # seconds to wait for a "slow down" (429 / 503 Retry-After) at most


def _pause(seconds: float, stop: Callable[[], bool] = _never) -> None:
    """Wait (sleep) in short steps so that a stop ends the wait."""
    end = time.monotonic() + max(0.0, seconds)
    while (left := end - time.monotonic()) > 0 and not stop():
        time.sleep(min(0.25, left))


def _kw(stop) -> dict:
    """{"stop": stop} unless it is the default (a replacement of _get_json need not know it)."""
    return {} if stop is _never else {"stop": stop}


def _retry_after(e: urllib.error.HTTPError) -> float:
    try:
        wait = float((e.headers or {}).get("Retry-After") or 2)
    except (TypeError, ValueError, AttributeError):
        wait = 2.0
    return min(max(wait, 0.0), RETRY_WAIT_MAX)


def _get_json(path: str, timeout: float, use_key: bool = True,
              stop: Callable[[], bool] = _never) -> dict | None:
    """A CivitAI API answer: of the chosen site, else of the other (when the first cannot be asked or does not
    know what is asked for: a LoRA may be listed on one only); None only when every site answered that it does not
    know it (404). OSError when a site could not be asked (offline, rate limited, server error) and none had it:
    "not known" must never come from a failed request.
    The key goes only to CivitAI's own sites: as an unredirected header (a redirect elsewhere does not carry it).
    A rejected key (401 / 403): asked once more without it. A 429 (slow down): waits (Retry-After, at most
    RETRY_WAIT_MAX s) and asks that site once more."""
    last: Exception | None = None
    key = api_key() if use_key else ""
    if key in _REJECTED:
        key = ""
    unknown = False
    for base in sites():
        req = urllib.request.Request(f"{base}{path}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        if key:
            req.add_unredirected_header("Authorization", f"Bearer {key}")
        data = None
        for attempt in (0, 1):
            try:
                with net.urlopen(req, timeout) as r:
                    data = json.loads(r.read(MAX_ANSWER).decode("utf-8", "replace"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    unknown = True                      # (the other site may know it)
                    break
                if e.code == 429 and not attempt and not stop():
                    _pause(_retry_after(e), stop)
                    continue
                if e.code in (401, 403) and key:
                    _REJECTED.add(key)                  # (not sent again: one request per lookup, not two)
                    try:
                        return _get_json(path, timeout, use_key=False, stop=stop)
                    except OSError as e2:
                        raise KeyRejected(f"CivitAI rejected the API key ({e.code}) and could not be asked "
                                          f"without it: {e2}") from e2
                last = e
                break
            except (OSError, ValueError) as e:
                last = e
                break
        if data is None:
            continue
        if isinstance(data, dict) and not data.get("error"):
            return data
        unknown = True
    if last is not None:                                # (a site failed: that is not "unknown")
        raise OSError(f"CivitAI could not be asked: {last}")
    return None if unknown else _no_site()


def _no_site():
    raise OSError("CivitAI could not be asked")


EXAMPLE_PAGE = 200          # gallery pictures per request (CivitAI's maximum)
EXAMPLE_PAGES = 10          # requests per LoRA at most: the 2000 most liked pictures and videos of its gallery
EXAMPLE_PROMPTS = 300       # different prompts kept per LoRA (the gallery is read until it has that many)
META_VALUE = 4000           # a longer value in a picture's generation data is left out (a whole workflow)
DETAILS_VERSION = 3         # raised when the details asked for change: every LoRA is asked once more
                            # (2: the example images' generation data, which CivitAI sends only with withMeta=true;
                            #  3: the whole gallery of all time, not only the last year's first 100 pictures)


def _slim(item: dict) -> dict:
    """A gallery picture as it is kept: its generation data without values that are whole workflows."""
    meta = {k: v for k, v in item["meta"].items()
            if k in ("prompt", "negativePrompt") or len(json.dumps(v, default=str)) <= META_VALUE}
    return {**item, "meta": meta}


def gallery(version_id: int, timeout: float = 20.0, gap: float = REQUEST_GAP, pages: int = EXAMPLE_PAGES,
            want: int = EXAMPLE_PROMPTS, stop: Callable[[], bool] = _never) -> tuple[list[dict], int, bool]:
    """The pictures and videos users posted with a LoRA version (the gallery on its CivitAI page: the creator's
    examples and everyone else's), of all time, the most liked first; only those that tell their prompt, each
    prompt once. -> (the pictures, how many were looked at, complete). The gallery is read page by page until it
    ends, `want` prompts are found or `pages` requests were made. complete is False when a later page could not
    be asked (what was read so far is returned). Raises OSError when the first page could not be asked."""
    base = (f"/api/v1/images?modelVersionId={int(version_id)}&limit={EXAMPLE_PAGE}&sort=Most%20Reactions"
            "&period=AllTime&nsfw=X&withMeta=true")
    out, seen, looked, cursor = [], set(), 0, None
    for page in range(max(1, pages)):
        if page and stop():
            return out, looked, False
        if page and gap:
            _pause(gap, stop)
        try:
            data = _get_json(base + (f"&cursor={urllib.parse.quote(str(cursor), safe='')}" if cursor else ""),
                             timeout, **_kw(stop))
        except (OSError, ValueError, TypeError):
            if not page:
                raise
            return out, looked, False
        items = (data or {}).get("items")
        for it in items if isinstance(items, list) else []:
            looked += 1
            meta = it.get("meta") if isinstance(it, dict) else None
            prompt = str(meta.get("prompt") or "").strip() if isinstance(meta, dict) else ""
            if prompt and prompt not in seen:
                seen.add(prompt)
                out.append(_slim(it))
        meta = (data or {}).get("metadata")
        cursor = meta.get("nextCursor") if isinstance(meta, dict) else None
        if not cursor or len(out) >= want:
            break
    return out[:want], looked, True


def civitai_lookup(sha: str, timeout: float = 20.0, gap: float = REQUEST_GAP,
                   stop: Callable[[], bool] = _never) -> dict | None:
    """CivitAI's model version for a file hash: {"words", "model", "url", "details"}; None when CivitAI does not
    know it. details = {"version": the version as CivitAI gives it, "model": the model page (description, tags,
    creator, license; its other versions only as id / name / base model), "examples": the pictures of its
    gallery that tell their prompt (see gallery: generation data with prompt, negative prompt, seed, settings),
    "examples_seen": how many gallery pictures were looked at}. The model page and the examples are optional (a
    failed request leaves them out). Raises OSError when the version could not be asked (offline, server error)."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(sha)):
        raise ValueError(f"not a SHA-256: {sha!r}")
    data = _get_json(f"/api/v1/model-versions/by-hash/{sha}", timeout, **_kw(stop))
    if data is None:
        return None
    model = data.get("model") or {}
    mid = data.get("modelId") or model.get("id")
    details: dict = {"version": data, "fetched": time.time(), "complete": True}
    if mid:
        try:
            _pause(gap, stop)
            page = _get_json(f"/api/v1/models/{int(mid)}", timeout, **_kw(stop))
            if page:
                page = dict(page)
                page["modelVersions"] = [{"id": v.get("id"), "name": v.get("name"), "baseModel": v.get("baseModel")}
                                         for v in page.get("modelVersions") or [] if isinstance(v, dict)]
                details["model"] = page
        except (OSError, ValueError, TypeError):
            details["complete"] = False                # (asked again with the next scan)
    if data.get("id"):
        try:
            _pause(gap, stop)
            details["examples"], details["examples_seen"], whole = gallery(int(data["id"]), timeout, gap, stop=stop)
            if not whole:
                details["complete"] = False
        except (OSError, ValueError, TypeError):
            details["complete"] = False
    return {"words": _clean(data.get("trainedWords") or []), "model": str(model.get("name") or ""),
            "url": f"{SITES[0]}/models/{mid}?modelVersionId={data.get('id')}" if mid else "",   # (shown: on_site)
            "details": details}


def example_prompts(details: dict | None) -> list[dict]:
    """The generation data of a LoRA's example images (details from civitai_lookup), each prompt once:
    [{"prompt", "negative", "seed", "steps", "cfg", "sampler", "image"}], the most liked first."""
    out, seen = [], set()
    items = (details or {}).get("examples")
    for it in items if isinstance(items, list) else []:
        meta = it.get("meta") if isinstance(it, dict) else None
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


KEEP = object()        # update_note: leave the note as it is


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
        except (OSError, ValueError) as e:
            logger.warning("the trigger database is not readable: %s", e)

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
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(sha)):
            raise ValueError(f"not a SHA-256: {sha!r}")
        return self.path.parent / "civitai" / f"{sha.lower()}.json"

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

    def note_path(self, sha: str) -> Path:
        return self.details_path(sha).with_suffix(".note.json")

    def _sha(self, path: str) -> str | None:
        with self._lock:
            self._load()
            return (self._file(path) or {}).get("sha256")

    def note(self, path: str) -> dict | None:
        """The prompt writer's note on a LoRA (lora_notes: {"note", "stamp", "made"}), None when there is none."""
        sha = self._sha(path)
        if not sha:
            return None
        try:
            d = json.loads(self.note_path(sha).read_text(encoding="utf-8"))
            return d if isinstance(d, dict) and isinstance(d.get("note"), str) else None
        except (OSError, ValueError):
            return None

    def set_note(self, path: str, note: dict | None) -> bool:
        """Keep (None: forget) a LoRA's note, by its content hash; False when the hash is not known yet or the
        file cannot be written."""
        sha = self._sha(path)
        if not sha:
            return False
        try:
            if note is None:
                self.note_path(sha).unlink(missing_ok=True)
            else:
                self._save_details(sha, note, self.note_path(sha))
            return True
        except OSError:
            return False

    def update_note(self, path: str, change: Callable[[dict | None], object]) -> bool:
        """Read, change and write a LoRA's note under one lock, so that two writers (the user's save and a
        background note job) cannot overwrite each other. change(current note or None) returns the new note
        (None: forget it) or KEEP (write nothing). False when nothing was written."""
        with self._lock:
            new = change(self.note(path))
            if new is KEEP:
                return False
            return self.set_note(path, new)

    def _save_details(self, sha: str, details: dict, target: Path | None = None):
        import tempfile
        target = target or self.details_path(sha)
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
            if out is not None and out.get("url"):
                out["url"] = on_site(out["url"])
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

    def needs_lookup(self, path: str, key: str | None = None) -> bool:
        key = api_key() if key is None else key            # (outside the lock: the keyring may ask to unlock)
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
                return bool(key) and not rec.get("asked_with_key")   # and once with an API key
            if rec.get("checked"):
                if not rec.get("both_sites"):      # not known when only one site was asked: the other may know it
                    return True
                return time.time() - float(rec["checked"]) > RECHECK_DAYS * 86400
            return True

    def scan(self, paths: list[str], online: bool = True, stop: Callable[[], bool] = _never,
             lookup: Callable[[str], dict | None] = civitai_lookup, gap: float = REQUEST_GAP,
             progress: Callable[[int, int, str], None] | None = None) -> dict:
        """Look at LoRA files not known yet (header; then CivitAI by hash). Returns counts
        {"looked", "found", "unknown", "failed"}. progress(k, n, file name): the k-th of about n LoRAs is being
        asked on CivitAI now (n: those to ask as far as known before hashing; a copy of a known LoRA is not)."""
        n = {"looked": 0, "found": 0, "unknown": 0, "failed": 0, "new": 0}
        asked, dirty, failed_in_a_row = False, 0, 0
        key = api_key() if online else ""                  # (once, outside the lock)
        total = 0
        if online and progress:
            for path in paths:
                try:
                    total += os.path.isfile(path) and self.needs_lookup(path, key)
                except (OSError, ValueError, TypeError, AttributeError) as e:
                    logger.debug("lookup count: %s", e)

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
                if not online or not self.needs_lookup(path, key):
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
                    if not self.needs_lookup(path, key):    # known by its content (moved, renamed, a copy)
                        flush()
                        continue
                if asked and gap:
                    _pause(gap, stop)
                asked = True
                n["looked"] += 1
                if progress:
                    try:
                        progress(n["looked"], max(total, n["looked"]), Path(path).name)
                    except Exception as e:  # noqa: BLE001  (a window that is gone: the scan goes on)
                        logger.warning("the progress callback failed: %s", e)
                try:
                    hit = lookup(sha) if lookup is not civitai_lookup else civitai_lookup(sha, stop=stop)
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
                    rec["both_sites"] = True
                    if hit is None:
                        n["unknown"] += 1
                        if rec.get("civitai") is not None:    # known before, gone now: not asked on every scan
                            rec["details_v"] = DETAILS_VERSION
                            if key:
                                rec["asked_with_key"] = time.time()
                    else:
                        rec.update({"civitai": hit["words"], "model": hit["model"], "url": hit["url"]})
                        rec["asked_details"] = time.time()
                        complete = (hit.get("details") or {}).get("complete", True)
                        if complete:                   # (a part failed, e.g. rate limited: asked again later)
                            rec["details_v"] = DETAILS_VERSION
                            if key:
                                rec["asked_with_key"] = time.time()
                        if hit.get("details"):
                            try:
                                new = dict(hit["details"])
                                old = self.details_path(sha)
                                if not complete and old.is_file():      # keep what an earlier answer had
                                    try:
                                        prev = json.loads(old.read_text(encoding="utf-8"))
                                        for part in ("model", "examples"):
                                            if part not in new and part in prev:
                                                new[part] = prev[part]
                                        if len(prev.get("examples") or []) > len(new.get("examples") or []):
                                            new["examples"] = prev["examples"]     # (a gallery read only in part)
                                    except (OSError, ValueError) as e:
                                        logger.debug("merge of earlier details failed: %s", e)
                                self._save_details(sha, new)
                                rec["details"] = True
                                base = (hit["details"].get("version") or {}).get("baseModel")
                                if base:
                                    rec["base_model"] = str(base)
                            except (OSError, TypeError, ValueError) as e:
                                logger.debug("details not read: %s", e)
                        n["found"] += bool(hit["words"])
                    dirty += 1
                flush()
            except (OSError, ValueError, TypeError, AttributeError) as e:
                logger.warning("scan of a file failed: %s", e)
                continue                               # a file that vanished or cannot be read: the next one
        flush(force=True)
        n["key_rejected"] = key_rejected() if online else False
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
