"""Downloads that survive a bad connection: resumed from where they stopped, checked by size and SHA-256.

A file is written as <name>.part and renamed only when it is complete and correct, so a file that exists is whole.
Standard library only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AGENT = "video-chain-setup"
CHUNK = 4 * 2**20


class DownloadError(RuntimeError):
    pass


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Progress:
    """One line of progress for many downloads (plain lines every 30 s when the output is not a terminal)."""

    def __init__(self, total: int, label: str = "", out=None):
        self.total, self.label, self.out = max(int(total), 1), label, out or sys.stderr
        self.done = 0
        self.start = self.last = time.monotonic()
        self.lock = threading.Lock()
        self.tty = hasattr(self.out, "isatty") and self.out.isatty()
        self.started_at = 0                      # bytes that were already on disk (not counted for the speed)

    def add(self, n: int, resumed: bool = False):
        with self.lock:
            self.done += n
            if resumed:
                self.started_at += n
            now = time.monotonic()
            if now - self.last >= (0.5 if self.tty else 30.0):
                self.last = now
                self._show(now)

    def _show(self, now: float, end: str = ""):
        speed = (self.done - self.started_at) / max(now - self.start, 1e-6)
        left = (self.total - self.done) / speed if speed > 0 else 0
        eta = f"{int(left // 3600)}h{int(left % 3600 // 60):02d}m" if left >= 3600 else f"{int(left // 60)}m{int(left % 60):02d}s"
        text = (f"{self.label}{human(self.done)} of {human(self.total)} ({100 * self.done / self.total:.0f} %), "
                f"{human(speed)}/s, about {eta} left")
        if self.tty:
            self.out.write("\r\033[K" + text + end)
        else:
            self.out.write(text + "\n")
        self.out.flush()

    def finish(self):
        with self.lock:
            if self.tty:
                self._show(time.monotonic(), end="\n")


TOKEN_HOSTS = ("huggingface.co",)       # the only hosts a Hugging Face token is ever sent to
TOKEN_HINT = ("Hugging Face limits downloads without an account. Create a free token (read access) at "
              "https://huggingface.co/settings/tokens and run setup again with it: HF_TOKEN=hf_… ./install.sh "
              "(or log in once with: hf auth login). Setup only sends it to huggingface.co and never stores it.")


def hf_token() -> str | None:
    """The user's Hugging Face token from the usual places: $HF_TOKEN, else the file `hf auth login` writes
    ($HF_TOKEN_PATH, $HF_HOME/token, ~/.cache/huggingface/token). Read only; setup never writes or prints it."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser()
    for f in (os.environ.get("HF_TOKEN_PATH"), home / "token"):
        try:
            if f and Path(f).is_file():
                return Path(f).read_text(encoding="utf-8").strip() or None
        except OSError:
            pass
    return None


def _token_host(url: str) -> bool:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in TOKEN_HOSTS)


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but the token stays with the host it is for (downloads are redirected to storage
    servers, which need no token and must not get one)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _token_host(newurl):
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_Redirects)


def _open(url: str, start: int = 0, timeout: float = 60, token: str | None = None):
    headers = {"User-Agent": AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    if token and _token_host(url):
        headers["Authorization"] = f"Bearer {token}"
    return _OPENER.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def fetch(url: str, dest: Path, size: int | None = None, sha256: str | None = None, progress: Progress | None = None,
          tries: int = 8, token: str | None = None) -> Path:
    """Download url to dest (resuming dest.part), check size and SHA-256, then rename. A dest that is already there
    with the right size is kept as it is."""
    dest = Path(dest)
    if dest.is_file() and (size is None or dest.stat().st_size == size):
        if progress and size:
            progress.add(size, resumed=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if size is not None and have > size:
        part.unlink()
        have = 0
    if progress and have:
        progress.add(have, resumed=True)
    error: Exception | None = None
    limited = False
    for attempt in range(tries):
        if size is not None and have == size:
            break
        try:
            with _open(url, have, token=token) as r:
                if have and r.status != 206:           # the server ignored the range: start over
                    if progress:
                        progress.add(-have, resumed=True)
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if progress:
                            progress.add(len(b))
            if size is None or have == size:
                break
            error = DownloadError(f"the connection ended after {human(have)} of {human(size)}")
        except urllib.error.HTTPError as e:
            if e.code == 416 and size is not None and have == size:
                break
            if e.code in (401, 403):
                raise DownloadError(
                    f"{url}: access denied (HTTP {e.code}). " + ("The token was refused or its account has not "
                    "accepted this model's licence: open the model's page in a browser, accept, then run setup "
                    "again." if token else "This file needs a Hugging Face account that accepted its licence. "
                    + TOKEN_HINT)) from e
            if e.code == 429:
                limited = True
            if e.code == 404:
                raise DownloadError(f"{url}: not found (HTTP 404); the file was moved or removed upstream") from e
            error = e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            error = e
        time.sleep(min(60, 3 * (attempt + 1)))
    else:
        raise DownloadError(f"{url}: gave up after {tries} tries ({error})"
                            + (". " + TOKEN_HINT if limited and not token and _token_host(url) else ""))
    if size is not None and part.stat().st_size != size:
        raise DownloadError(f"{dest.name}: {part.stat().st_size} bytes instead of {size}")
    if sha256:
        got = sha256_of(part)
        if got != sha256:
            part.unlink()
            raise DownloadError(f"{dest.name}: the download is damaged (checksum {got[:12]}… instead of "
                                f"{sha256[:12]}…); it was removed, run setup again")
    os.replace(part, dest)
    return dest


def fetch_all(jobs: list[dict], workers: int = 4, label: str = "", token: str | None = None) -> None:
    """jobs: [{"url", "dest", "size", "sha256"}]; the first error stops the rest and is raised."""
    todo = [j for j in jobs if not (Path(j["dest"]).is_file() and Path(j["dest"]).stat().st_size == j.get("size"))]
    if not todo:
        return
    progress = Progress(sum(j.get("size") or 0 for j in todo), label)
    stop = threading.Event()

    def one(j):
        if stop.is_set():
            return
        try:
            fetch(j["url"], Path(j["dest"]), j.get("size"), j.get("sha256"), progress, token=token)
        except BaseException:
            stop.set()
            raise
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, j) for j in sorted(todo, key=lambda j: -(j.get("size") or 0))]
        errors = [f.exception() for f in futures]
    progress.finish()
    for e in errors:
        if e is not None:
            raise e


def extract(archive: Path, dest: Path, strip: int = 0, members: list[str] | None = None) -> None:
    """Unpack a .zip / .tar.* into dest, dropping `strip` leading folders; members: only these file names (flat).
    Entries that would land outside dest are refused."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    def target(name: str) -> Path | None:
        parts = [p for p in Path(name).parts if p not in ("", ".")]
        if members is not None:
            if not parts or parts[-1] not in members:
                return None
            parts = parts[-1:]
        else:
            parts = parts[strip:]
        if not parts or ".." in parts or Path(*parts).is_absolute():
            return None
        out = (dest / Path(*parts)).resolve()
        return out if out == root or root in out.parents else None

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                out = target(info.filename)
                if out is None or info.is_dir():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                mode = info.external_attr >> 16
                if mode & 0o111:
                    out.chmod(0o755)
        return
    with tarfile.open(archive) as t:
        for member in t:
            out = target(member.name)
            if out is None:
                continue
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                link = member.linkname
                if os.path.isabs(link) or ".." in Path(link).parts:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.is_symlink() or out.exists():
                    out.unlink()
                os.symlink(link, out)
            elif member.isfile():
                out.parent.mkdir(parents=True, exist_ok=True)
                data = t.extractfile(member)
                assert data is not None                 # (a regular file always has its data)
                with data as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                out.chmod(0o755 if member.mode & 0o111 else 0o644)
