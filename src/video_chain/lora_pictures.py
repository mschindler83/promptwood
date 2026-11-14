"""The pictures of a LoRA's CivitAI gallery, small: what an example prompt's result looks like. triggers.py keeps
the gallery's text (prompts, settings); a picture is loaded only when the LoRA window shows it, as a small still
(a video: its thumbnail frame), and kept in pictures/ next to the details so it is loaded once.

A LoRA's cover (the thumbnail in the LoRA picker) is the first picture the Pictures setting allows of the creator's
showcase on the LoRA's page (the version's images, CivitAI's own cover first), else of its gallery: covers().

Only CivitAI's picture server is asked, without the API key, and only for a picture address CivitAI gave."""
from __future__ import annotations

import hashlib
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import media, triggers

logger = logging.getLogger(__name__)

WIDTH = 320                      # pixels wide (CivitAI's server scales)
MAX_BYTES = 4 * 2**20            # a small still is ~50 kB; more is not a still
HOSTS = tuple(site.split("//", 1)[1] for site in triggers.SITES)    # (one list, in triggers.py)
LEVELS = ("None", "Soft", "Mature", "X")       # CivitAI's ratings, mildest first
_LEVEL_NUMBERS = {1: "None", 2: "Soft", 4: "Mature"}   # (a version's images rate by number: 1 PG … 8 X, 16 XXX)
COVER_TRIES = 3                  # pictures tried for a cover before the LoRA is shown without one
USER_AGENT = "video-chain (LoRA example pictures)"
_SIZE = re.compile(r"/(original=true|width=\d+)[^/]*/")


def level(item: dict) -> str:
    """A gallery picture's rating: None (safe) / Soft / Mature / X; one CivitAI did not rate counts as X."""
    raw = (item or {}).get("nsfwLevel")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return _LEVEL_NUMBERS.get(raw, "X")
    lv = str(raw or "")
    if lv in LEVELS:
        return lv
    return "None" if (item or {}).get("nsfw") is False else "X"


def allowed(item: dict, show: str) -> bool:
    """show: "off" (no pictures), "safe" (only those rated None), "all"."""
    return show == "all" or (show == "safe" and level(item) == "None")


def is_video(item: dict) -> bool:
    return str((item or {}).get("type") or "").lower() == "video" or \
        str((item or {}).get("url") or "").lower().rsplit(".", 1)[-1] in ("mp4", "webm", "mov")


def _ours(url: str) -> bool:
    try:
        u = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host == h or host.endswith("." + h) for h in HOSTS)


class _OursOnly(urllib.request.HTTPRedirectHandler):
    """A redirect is followed only to CivitAI's own picture server (checked before the next request is made)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, newurl) if _ours(newurl) else None


def _open(req, timeout: float):
    return urllib.request.build_opener(_OursOnly).open(req, timeout=timeout)


def still_url(item: dict, width: int = WIDTH) -> str | None:
    """The address of a small still of a gallery picture (a video: its thumbnail frame), None when CivitAI gave
    none that can be asked for."""
    if not isinstance(item, dict):
        return None
    if is_video(item):
        thumb = item.get("thumbnail")
        url = str(thumb.get("url") or "") if isinstance(thumb, dict) else ""
    else:
        url = str(item.get("url") or "")
        if _SIZE.search(url):
            url = _SIZE.sub(f"/width={int(width)}/", url, count=1)
    return url if url and _ours(url) else None


def page_url(item: dict) -> str | None:
    """The picture's page on CivitAI, on the chosen site (full size, the video, its comments)."""
    from . import triggers
    try:
        return f"{triggers.site()}/images/{int(item['id'])}"
    except (KeyError, TypeError, ValueError):
        return None


def cache_dir(db=None) -> Path:
    from . import triggers
    return (db or triggers.db()).path.parent / "civitai" / "pictures"


def cached(item: dict, db=None) -> Path | None:
    """Where a picture's still is kept (the file may not be there yet): by its id, else by its address; None for
    a picture without either."""
    try:
        return cache_dir(db) / f"{int(item['id'])}.jpg"
    except (KeyError, TypeError, ValueError):
        pass
    url = str((item or {}).get("url") or "") if isinstance(item, dict) else ""
    return cache_dir(db) / f"u{hashlib.sha256(url.encode()).hexdigest()[:24]}.jpg" if url else None


def covers(details: dict | None, show: str) -> list[dict]:
    """The pictures that may be a LoRA's cover, best first (details: triggers.TriggerDB.details): the creator's
    showcase on its page, then the gallery; only those `show` allows and that have a still to load."""
    if not isinstance(details, dict) or show == "off":
        return []
    version = details.get("version") if isinstance(details.get("version"), dict) else {}
    out = []
    for group in (version.get("images"), details.get("examples")):
        for it in group if isinstance(group, list) else []:
            if isinstance(it, dict) and allowed(it, show) and still_url(it):
                out.append(it)
    return out


def cover_file(path: str, db=None, show: str = "safe", load=None) -> Path | None:
    """The local still of a LoRA's cover, loaded when needed (load(item) -> file, default: fetch); the next
    picture is tried when one cannot be loaded. None when there is none to show."""
    from . import triggers
    db = db or triggers.db()
    load = load or (lambda item: fetch(item, db))
    for item in covers(db.details(path), show)[:COVER_TRIES]:
        try:
            f = load(item)
        except OSError:
            continue
        if f:
            return Path(f)
    return None


def fetch(item: dict, db=None, timeout: float = 20.0) -> Path | None:
    """The still of a gallery picture as a local file: from pictures/, else loaded from CivitAI and kept. None
    when there is none to load; OSError when it could not be loaded."""
    target = cached(item, db)
    if target is None:
        return None
    if target.is_file() and target.stat().st_size:
        return target
    url = still_url(item)
    if not url:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
    try:
        with _open(req, timeout) as r:
            if not _ours(r.geturl()):
                raise OSError("the picture was redirected away from CivitAI")
            kind = str(r.headers.get("Content-Type") or "").lower()
            if not kind.startswith("image/"):
                raise OSError(f"not a picture ({kind or 'unknown type'})")
            data = r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise OSError(f"CivitAI answered {e.code}") from e
    except ValueError as e:
        raise OSError(str(e)) from e
    if not data or len(data) > MAX_BYTES:
        raise OSError("the picture is empty or too large")
    target.parent.mkdir(parents=True, exist_ok=True)
    media.atomic_write_bytes(target, data)
    return target


def kept(db=None) -> tuple[int, int]:
    """(files, bytes) of the stills kept."""
    n = size = 0
    try:
        for f in cache_dir(db).iterdir():
            if f.is_file():
                n += 1
                size += f.stat().st_size
    except OSError as e:
        logger.debug("cache size unknown: %s", e)
    return n, size


def forget(db=None) -> int:
    """Delete the stills kept (they are loaded again when shown); how many."""
    n = 0
    try:
        for f in cache_dir(db).iterdir():
            if f.is_file() and f.suffix in (".jpg", ".part"):
                f.unlink(missing_ok=True)
                n += 1
    except OSError as e:
        logger.warning("cache not fully cleared: %s", e)
    return n
