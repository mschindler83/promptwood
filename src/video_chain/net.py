"""One small HTTP helper for the modules that talk to ComfyUI, llama.cpp, RunPod, CivitAI and ElevenLabs: a timeout on
every request, answers that are not JSON become an error that says so, and addresses on this machine or the home network
never go through a proxy from the environment."""
from __future__ import annotations

import http.client
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request


class NetError(OSError):
    """Base of the errors below (an OSError, so callers that already catch OSError keep working)."""


class Unreachable(NetError):
    """No answer: refused, timed out, DNS, a cut connection."""


class HttpStatus(NetError):
    def __init__(self, code: int, body: str = "", url: str = ""):
        super().__init__(f"HTTP {code} {body[:300]}")
        self.code, self.body, self.url = code, body, url


class BadAnswer(NetError):
    """The server answered, but not with JSON."""


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_local(url: str) -> bool:
    """True for loopback, private (LAN) addresses, and plain host names without a dot (a home network name)."""
    host = urllib.parse.urlparse(url).hostname or ""
    try:
        return host == "localhost" or "." not in host and ":" not in host or ipaddress.ip_address(host).is_private
    except ValueError:
        return "." not in host


def urlopen(req, timeout: float):
    """urllib's urlopen; local and LAN addresses without the proxy of the environment. The request's address decides."""
    url = req.full_url if hasattr(req, "full_url") else str(req)
    if is_local(url):
        return _DIRECT.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def parse_json(raw: bytes | str, what: str = "the answer"):
    try:
        return json.loads(raw)
    except ValueError as e:
        shown = raw[:80] if isinstance(raw, (bytes, str)) else raw
        raise BadAnswer(f"{what} is not JSON ({shown!r})") from e


def request_json(url: str, *, payload=None, method: str | None = None, headers: dict | None = None,
                 timeout: float = 30, opener=urlopen):
    """GET (or POST with payload) and the decoded JSON answer (None for an empty body). Raises HttpStatus,
    Unreachable or BadAnswer, all NetError."""
    data = None if payload is None else json.dumps(payload).encode()
    h = dict(headers or {})
    if data is not None:
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with opener(req, timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise HttpStatus(e.code, e.read().decode("utf-8", "replace"), url) from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise Unreachable(f"{url} is not reachable ({e})") from e
    return parse_json(body, url) if body.strip() else None


def get_json(url: str, **kw):
    return request_json(url, **kw)
