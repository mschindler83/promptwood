"""Cloud upscaling on RunPod: a Secure Cloud GPU pod runs ComfyUI with SeedVR2 for one export, then is terminated.

How a session goes (RunpodSession):
  1. a pod is created through RunPod's REST API v2 (Secure Cloud only, never Community Cloud; one GPU of the chosen
     type, others in turn if it has no capacity and that is allowed), with only 22/tcp exposed and this app's own
     SSH key (~/.config/video-chain/runpod/id_ed25519) as its PUBLIC_KEY;
  2. the pod's SSH host key is read through the API (authenticated HTTPS, /pods/{id}/exec), so the SSH connection
     is verified, not trusted on first use;
  3. cloud/pod_setup.sh is run on the pod over SSH: ComfyUI at a pinned tag and the SeedVR2 weights (kept on an
     optional network volume, so later pods start in about a minute), ComfyUI started on the pod's 127.0.0.1;
  4. an SSH tunnel forwards a local port to it: the app talks to that ComfyUI as to a local one (CloudUpscaler);
  5. close(): the tunnel ends and the pod is terminated (DELETE), checked; a pod-side watchdog removes the pod by
     itself after IDLE_MIN minutes without an SSH connection (the app was killed, the network went away).

Privacy: nothing private is written to a RunPod disk; uploads, results and every temporary file stay in the pod's
RAM (see pod_setup.sh) and are removed after each job. The API key is kept in the system keyring and sent only to
RunPod. Needs: ssh and ssh-keygen (OpenSSH) on this machine. Standard library only.
"""
from __future__ import annotations

import logging
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

from .comfy import (SEEDVR_FULL_VRAM, SEEDVR_METHODS, SEEDVR_NODES, SEEDVR_PRECISION_FULL, SUBFOLDER, ComfyError,
                    ComfyGenerator)

from . import net

logger = logging.getLogger(__name__)

API_URL = "https://api.runpod.io/v2"
# RunPod's official PyTorch image (starts sshd from PUBLIC_KEY); CUDA 12.8 runs on Ampere, Ada, Hopper and Blackwell
IMAGE = "runpod/pytorch:2.1.0-cu1281-torch280-ubuntu2404"
# PyTorch on the pod: pinned to what ran (2026-11-13, H100, driver 595 / CUDA 13.2). Unpinned, uv put the newest
# torchvision (for spandrel) and with it the newest torch into the venv anyway, over the image's 2.8: pods would
# have differed by the day, and a host with an older driver could not run it.
TORCH_PINS = ("torch==2.15.0", "torchvision==0.29.2")
MIN_CUDA = "13.2"                          # host driver at least this (the pinned PyTorch is built for CUDA 13.2)
COMFY_REF = "v0.45.3"                      # ComfyUI tag on the pod (has the SeedVR2 core nodes this app's workflow uses)
POD_PREFIX = "video-chain-upscale"         # every pod this app makes is named so (leftovers are found by it)
REMOTE_IO = PurePosixPath("/dev/shm/vc-io")   # the pod's RAM-only folder for uploads and results (pod_setup.sh)
REMOTE_PORT = 8188
VOLUME_PATH = "/workspace"                 # where the pod's network or global volume is mounted
IDLE_MIN = 20                              # the pod removes itself after this long without an SSH connection
MIN_RAM_PER_GPU = 32                       # GB of system RAM: the SeedVR2 pieces are cut to what the pod has
                                           # (64 left out most RTX 4090 / 5090 hosts)
KEYRING_SERVICE, KEYRING_USER = "video-chain", "runpod-api-key"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "video-chain" / "runpod"
SETUP_SCRIPT = Path(__file__).with_name("cloud") / "pod_setup.sh"
# video-chain's lanczos node (the same frames as ComfyUI's, on all cores), copied to the pod before the setup
SCALE_NODE_FILE = Path(__file__).resolve().parents[2] / "comfy_nodes" / "video_chain_nodes" / "scale.py"
REMOTE_CODE = "/root/video-chain"          # the pod's container-disk folder for ComfyUI (pod_setup.sh CODE)

# GPU types (RunPod's ids) offered in the settings, fastest first. Prices change: RunPod's console shows them.
GPUS = (
    ("NVIDIA H100 80GB HBM3", "H100 SXM 80 GB (fastest)"),
    ("NVIDIA H200", "H200 141 GB (about as fast, pricier)"),
    ("NVIDIA H100 NVL", "H100 NVL 94 GB"),
    ("NVIDIA H100 PCIe", "H100 PCIe 80 GB"),
    ("NVIDIA GeForce RTX 5090", "RTX 5090 32 GB"),
    ("NVIDIA L40S", "L40S 48 GB"),
    ("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada 48 GB"),
    ("NVIDIA GeForce RTX 4090", "RTX 4090 24 GB (as an RTX 4090 at home)"),
)
DEFAULT_GPU = GPUS[0][0]


class RunpodError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


# ----------------------------------------------------------------------------- the API key


def api_key() -> str:
    """$RUNPOD_API_KEY, else the key kept in the system keyring; "" without one."""
    env = os.environ.get("RUNPOD_API_KEY", "").strip()
    if env:
        return env
    try:
        import keyring
        return (keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or "").strip()
    except Exception:  # noqa: BLE001  (no keyring now)
        return ""


def set_api_key(key: str | None) -> None:
    """Keep the RunPod API key in the system keyring (empty: remove it). RuntimeError: no keyring."""
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


# ----------------------------------------------------------------------------- REST API v2


class RunpodApi:
    """The few calls of RunPod's REST API v2 this app needs (https://api.runpod.io/v2)."""

    def __init__(self, key: str, base: str = API_URL, sleep=time.sleep):
        if not key:
            raise RunpodError("no RunPod API key (Settings → Cloud upscaling)")
        self.key, self.base, self.sleep = key, base.rstrip("/"), sleep

    def call(self, method: str, path: str, payload=None, timeout: float = 30, tries: int = 4,
             idempotent: bool = True):
        """idempotent=False: retried only after a 429 (refused before anything was done), never after a 5xx or a
        lost answer, which may have been carried out."""
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json",
                   "User-Agent": "video-chain"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(tries):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
            try:
                with net.urlopen(req, timeout) as r:
                    body = r.read()
                    try:
                        return json.loads(body) if body.strip() else None
                    except ValueError as e:        # (a proxy's HTML page with status 200)
                        raise RunpodError(f"RunPod {method} {path}: the answer is not JSON ({body[:80]!r})") from e
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                retry = e.code == 429 or (idempotent and e.code >= 500)
                if retry and attempt + 1 < tries:
                    try:
                        wait = float(e.headers.get("Retry-After") or 0)
                    except ValueError:
                        wait = 0
                    self.sleep(min(30.0, wait or 2.0 * (attempt + 1)))
                    continue
                raise RunpodError(f"RunPod {method} {path}: HTTP {e.code} {_detail(body)}", e.code) from e
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                if idempotent and attempt + 1 < tries:
                    self.sleep(2.0 * (attempt + 1))
                    continue
                raise RunpodError(f"RunPod is not reachable ({e})") from e
        raise RunpodError(f"RunPod {method} {path}: no answer")

    def create_pod(self, body: dict) -> dict:
        return self.call("POST", "/pods", body, timeout=60, idempotent=False)   # (a 2nd try could make 2 pods)

    def get_pod(self, pod_id: str) -> dict:
        res = self.call("GET", f"/pods/{urllib.parse.quote(pod_id)}") or {}
        return res.get("pod", res) if isinstance(res, dict) else {}

    def delete_pod(self, pod_id: str) -> None:
        try:
            self.call("DELETE", f"/pods/{urllib.parse.quote(pod_id)}")
        except RunpodError as e:
            if e.status != 404:                    # gone already
                raise

    def list_pods(self) -> list[dict]:
        pods, cursor = [], None
        while True:
            q = "?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else ""
            res = self.call("GET", f"/pods{q}") or {}
            pods += res.get("pods") or []
            cursor = (res.get("pagination") or {}).get("nextCursor")
            if not cursor:
                return pods

    def exec(self, pod_id: str, argv: list[str], timeout: int = 20) -> dict:
        return self.call("POST", f"/pods/{urllib.parse.quote(pod_id)}/exec",
                         {"command": list(argv), "timeoutSeconds": int(timeout)}, timeout=timeout + 15,
                         idempotent=False)

    def network_volume(self, vol_id: str) -> dict:
        res = self.call("GET", f"/network-volumes/{urllib.parse.quote(vol_id)}") or {}
        return res.get("networkVolume", res) if isinstance(res, dict) else {}

    def global_volume(self, vol_id: str) -> dict:
        res = self.call("GET", f"/global-volumes/{urllib.parse.quote(vol_id)}") or {}
        return res.get("globalVolume", res) if isinstance(res, dict) else {}

    def volume(self, vol_id: str) -> tuple[str, dict]:
        """("network" | "global", its record) for a volume id; RunpodError if the account has no such volume."""
        for kind, get in (("network", self.network_volume), ("global", self.global_volume)):
            try:
                return kind, get(vol_id)
            except RunpodError as e:
                if e.status not in (400, 404, 422):
                    raise
        raise RunpodError(f"RunPod has no network or global volume {vol_id!r} in this account", 404)

    def gpus(self) -> list[dict]:
        """GPU types with their Secure Cloud price and current pod stock."""
        res = self.call("GET", f"/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&minCudaVersion={MIN_CUDA}")
        res = res or {}
        return res.get("gpus") or []


_CAPACITY_WORDS = re.compile(r"availab|capacity|instances|stock|insufficient|no longer|not enough", re.I)


def _detail(body: str) -> str:
    try:
        d = json.loads(body)
    except ValueError:
        return body[:400]
    if not isinstance(d, dict):
        return body[:400]
    parts = [str(d.get("detail") or d.get("title") or "")] + [str(x) for x in d.get("errors") or []]
    return "; ".join(p for p in parts if p)[:600]


def pod_body(*, name: str, gpu: str, public_key: str, volume: str | None = None, data_center: str | None = None,
             image: str = IMAGE, volume_kind: str = "network") -> dict:
    """The create request for an upscale pod: Secure Cloud, one GPU, SSH only (22/tcp; no HTTP port, no Jupyter),
    this app's key; the volume (public files only; volume_kind "network": in one data centre, "global": any) at
    /workspace when one is given."""
    body = {
        "name": name,
        "image": image,
        "cloud": "SECURE",
        "gpu": {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA, "minRamPerGpu": MIN_RAM_PER_GPU},
        "disk": 80,                         # ComfyUI, its packages and new weights (7B fp16: 16.5 GB) go on the disk
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": public_key},  # (set here, RunPod does not put the account's own keys in)
        "startSsh": True,
        "startJupyter": False,
    }
    if volume:
        if volume_kind not in ("network", "global"):
            raise ValueError(f"unknown volume kind {volume_kind!r}")
        body["mounts"] = {volume_kind: [{"volumeId": volume, "path": VOLUME_PATH}]}
    if data_center:
        body["dataCenterIds"] = [data_center]
    return body


def direct_ssh(pod: dict) -> tuple[str, int] | None:
    """(host, port) of the pod's own sshd once RUNNING and its 22/tcp is published, else None."""
    if pod.get("status") != "RUNNING":
        return None
    d = (pod.get("ssh") or {}).get("direct") or {}
    if d.get("host") and d.get("port"):
        return str(d["host"]), int(d["port"])
    for p in (pod.get("runtime") or {}).get("ports") or []:
        if p.get("private") == 22 and p.get("public") and p.get("ip"):
            return str(p["ip"]), int(p["public"])
    return None


def models_for(method: str) -> str:
    """The SeedVR2 sizes pod_setup.sh fetches for an export method ('seedvr2_7b' -> '7b')."""
    if method not in SEEDVR_METHODS:
        raise RunpodError(f"cloud upscaling does SeedVR2 only, not {method!r}")
    return SEEDVR_METHODS[method]


def leftover_pods(api: RunpodApi) -> list[dict]:
    """This app's pods that still exist (named POD_PREFIX-…): a crash or lost connection left them."""
    return [p for p in api.list_pods() if str(p.get("name", "")).startswith(POD_PREFIX + "-")
            and p.get("status") != "TERMINATED"]


# ----------------------------------------------------------------------------- SSH


def ensure_key(config: Path = CONFIG_DIR) -> tuple[Path, str]:
    """This app's own SSH key for RunPod pods (made once, ed25519, no passphrase, only readable by you):
    (private key path, public key line)."""
    config = Path(config)
    config.mkdir(parents=True, exist_ok=True)
    os.chmod(config, 0o700)
    key = config / "id_ed25519"
    try:
        if not key.is_file():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "video-chain-runpod", "-f",
                            str(key)], check=True, capture_output=True)
        os.chmod(key, 0o600)
        pub = key.with_suffix(".pub")
        if not pub.is_file():                  # (deleted by hand: made again from the private key)
            r = subprocess.run(["ssh-keygen", "-y", "-f", str(key)], check=True, capture_output=True, text=True)
            pub.write_text(r.stdout.strip() + "\n", encoding="utf-8")
        return key, pub.read_text(encoding="utf-8").strip()
    except FileNotFoundError as e:
        raise RunpodError("OpenSSH is not installed (ssh-keygen is missing): install it to use RunPod") from e
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr or "")
        raise RunpodError("ssh-keygen failed: " + err.strip()[:300]) from e


def known_hosts_line(host: str, port: int, host_keys: str) -> str:
    """known_hosts lines for [host]:port from the pod's /etc/ssh/ssh_host_*_key.pub contents."""
    out = []
    for line in host_keys.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"(ssh|ecdsa|sk)-[A-Za-z0-9@.-]+", parts[0]):
            out.append(f"[{host}]:{port} {parts[0]} {parts[1]}")
    return "\n".join(out) + ("\n" if out else "")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")


# ----------------------------------------------------------------------------- one session


class RunpodSession:
    """One pod for one export (see the module doc). start() -> the local ComfyUI URL; close() terminates the pod.
    note(text): progress for the user; log(text): details for the log."""

    def __init__(self, api: RunpodApi, *, gpu: str = DEFAULT_GPU, fallback: bool = True, volume: str | None = None,
                 models: str = "7b", note=None, log=None, config: Path = CONFIG_DIR, idle_min: int = IDLE_MIN,
                 ssh: str = "ssh", boot_timeout: float = 900, setup_timeout: float = 3600,
                 without_volume: bool = False):
        """without_volume: when the network volume's data centre has none of the GPUs free, start without the
        volume anywhere (ComfyUI and the weights are fetched again: a slower start)."""
        self.api, self.gpu, self.fallback, self.volume, self.models = api, gpu, fallback, volume or None, models
        self.without_volume = bool(without_volume)
        self.volume_kind = "network"
        self.note, self.log = note or (lambda _t: None), log or (lambda _t: None)
        self.config, self.idle_min, self.ssh_bin = Path(config), int(idle_min), ssh
        self.boot_timeout, self.setup_timeout = boot_timeout, setup_timeout
        self.key: Path = self.config / "id_ed25519"
        self.user, self.remote_port = "root", REMOTE_PORT      # (tests: a local sshd as oneself)
        self.remote_code = REMOTE_CODE
        self.pod_id: str | None = None
        self.pod: dict = {}
        self.host: tuple[str, int] | None = None
        self.url: str | None = None
        self.tunnel: subprocess.Popen | None = None
        self._tunnel_err = None
        self.started = None
        self._tmp = tempfile.TemporaryDirectory(prefix="video-chain-runpod-")
        self.known_hosts = Path(self._tmp.name) / "known_hosts"
        self._closed = False
        self._closing = False
        self._lock = threading.Lock()

    # ---- start
    setup_script = SETUP_SCRIPT

    def start(self) -> str:
        if not Path(self.setup_script).is_file():
            raise RunpodError(f"the pod setup script is missing ({self.setup_script})")
        self.key, pub = ensure_key(self.config)
        self._remove_leftovers()
        try:
            self._create(pub)
            self.host = self._wait_running()
            self._trust_host()
            self._wait_ssh()
            self._setup()
            self._open_tunnel()
        except BaseException:                    # (also Ctrl-C / SystemExit: never leave a billed pod behind)
            try:
                self.close()
            except RunpodError as e:
                self.log(str(e))
            raise
        return self.url

    def _candidates(self) -> list[str]:
        rest = [g for g, _ in GPUS if g != self.gpu] if self.fallback else []
        return [self.gpu] + rest

    def _stock(self) -> dict[str, set[str]] | None:
        """GPU type -> the data centres where RunPod's Secure Cloud has it free now (None: the catalog did not
        answer; then every type is tried)."""
        try:
            gpus = self.api.gpus()
        except RunpodError as e:
            self.log(f"RunPod: no GPU catalog ({e}); trying every GPU type")
            return None
        return {g["id"]: {x["id"] for x in g.get("dataCenters") or []
                          if x.get("id") and x.get("availability") not in (None, "NONE")}
                for g in gpus if g.get("id")}

    def _create(self, public_key: str) -> None:
        dc = None
        if self.volume:                          # a network volume: the pod must be in its data centre; global: any
            self.volume_kind, vol = self.api.volume(self.volume)
            if self.volume_kind == "network":
                dc = vol.get("dataCenterId") or vol.get("dataCenter") or None
                if isinstance(dc, dict):
                    dc = dc.get("id")
            self.log(f"RunPod: {self.volume_kind} volume {self.volume}" + (f" in {dc}" if dc else ""))
        stock = self._stock()
        order = self._candidates()

        def free_in(where):                      # the candidates RunPod has free there (anywhere: None)
            if stock is None:
                return order
            return [g for g in order if stock.get(g) and (where is None or where in stock[g])]

        def elsewhere() -> str:
            if not stock:
                return ""
            free = [f"{re.sub(r' [0-9]+ GB.*$', '', dict(GPUS)[g])} in {', '.join(sorted(stock[g])[:4])}"
                    for g, _ in GPUS if stock.get(g)]
            return f"; free elsewhere now: {'; '.join(free)}" if free else "; RunPod lists none of them free anywhere"

        cands, volume = free_in(dc), self.volume
        if volume and dc and not cands and stock is not None:
            if self.without_volume:
                self.note(f"RunPod: no GPU free in {dc} (the network volume's data centre): starting without "
                          "the volume (a slower start)")
                volume, dc, cands = None, None, free_in(None)
            else:
                raise RunpodError(f"none of these GPUs is free in {dc}, your network volume's data centre, right "
                                  f"now{elsewhere()}. Try again later, or switch on “start without the volume” "
                                  "(Settings → Cloud upscaling), or make a volume in a data centre with more GPUs")
        if not cands:
            raise RunpodError("none of these GPUs is free on RunPod's Secure Cloud right now"
                              + (elsewhere() if not self.fallback else "") + "; try again later"
                              + ("" if self.fallback else " or allow other GPUs"))
        self.volume = volume
        last = None
        for gpu in cands:
            name = f"{POD_PREFIX}-{uuid.uuid4().hex[:8]}"
            self.note(f"RunPod: asking for a {gpu}" + (f" in {dc}" if dc else ""))
            try:
                pod = self.api.create_pod(pod_body(name=name, gpu=gpu, public_key=public_key, volume=volume,
                                                   data_center=dc, volume_kind=self.volume_kind))
            except RunpodError as e:
                if e.status == 400 and _CAPACITY_WORDS.search(str(e)):   # no capacity for this one: the next type
                    self.log(f"RunPod: {gpu}: {e}")
                    last = e
                    continue
                if e.status is None or e.status >= 500:      # the answer was lost: the pod may exist all the same
                    self._remove_pods_named(name)
                raise                            # (401/403: key or credit; other 400: our request: not a stock matter)
            if not isinstance(pod, dict) or not pod.get("id") or "podId" in pod:   # (a queued pod: never asked for)
                raise RunpodError(f"RunPod did not create a pod: {json.dumps(pod)[:300]}")
            self.pod_id, self.pod, self.started = pod["id"], pod, time.monotonic()
            cost = pod.get("cost")
            self.log(f"RunPod: pod {self.pod_id} ({gpu}, {pod.get('dataCenterId') or '?'}"
                     + (f", ${cost:.2f}/h" if isinstance(cost, (int, float)) else "") + ")")
            return
        raise RunpodError(("RunPod listed these GPUs as free but gave none of them" if stock is not None else
                           "RunPod has none of these GPUs free right now")
                          + (f" (last answer: {last})" if last else "") + "; try again in a minute")

    def _wait_running(self) -> tuple[str, int]:
        self.note("RunPod: starting the pod")
        end = time.monotonic() + self.boot_timeout
        while time.monotonic() < end:
            pod = self.api.get_pod(self.pod_id)
            self.pod = pod or self.pod
            st = pod.get("status")
            if st in ("ERROR", "TERMINATED", "EXITED"):
                raise RunpodError(f"the pod ended while starting (status {st})")
            hp = direct_ssh(pod)
            if hp:
                return hp
            time.sleep(3)
        raise RunpodError(f"the pod did not start within {self.boot_timeout / 60:.0f} min")

    def _trust_host(self) -> None:
        """The pod's SSH host keys, read through the API (HTTPS with the API key) into a known_hosts file of this
        session only: the SSH connection then verifies the pod, nothing is trusted on first use."""
        end = time.monotonic() + 180
        last = ""
        while time.monotonic() < end:
            try:
                res = self.api.exec(self.pod_id, ["sh", "-c", "cat /etc/ssh/ssh_host_*_key.pub 2>/dev/null"])
                lines = known_hosts_line(*self.host, res.get("stdout") or "")
                if lines:
                    self.known_hosts.write_text(lines, encoding="utf-8")
                    return
                last = "no host keys yet"
            except RunpodError as e:
                last = str(e)
                if e.status not in (409, 503, None):
                    break
            time.sleep(3)
        raise RunpodError(f"could not read the pod's SSH host key through RunPod's API ({last}); "
                          "not connecting to an unverified host")

    def ssh_args(self, *extra: str) -> list[str]:
        host, port = self.host
        return [self.ssh_bin, "-i", str(self.key), "-p", str(port),
                "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "GlobalKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-o", "LogLevel=ERROR", *extra, f"{self.user}@{host}"]

    def run(self, command: str, timeout: float = 60, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self.ssh_args() + [command], input=stdin, capture_output=True, timeout=timeout,
                                  check=False)
        except FileNotFoundError as e:
            raise RunpodError("OpenSSH is not installed (ssh is missing): install it to use RunPod") from e

    def _wait_ssh(self) -> None:
        end = time.monotonic() + 300
        err = ""
        while time.monotonic() < end:
            try:
                r = self.run("true", timeout=30)
                if r.returncode == 0:
                    return
                err = r.stderr.decode(errors="replace").strip()
                if "host key" in err.lower() and "verification failed" in err.lower():
                    raise RunpodError(f"the pod's SSH host key does not match what RunPod reported: {err}")
            except subprocess.TimeoutExpired:
                err = "timeout"
            time.sleep(3)
        raise RunpodError(f"could not log in to the pod over SSH ({err})")

    def _upload_nodes(self) -> None:
        """video-chain's lanczos node to the pod (pod_setup.sh puts it into ComfyUI's custom_nodes)."""
        if not SCALE_NODE_FILE.is_file():
            self.log(f"RunPod: {SCALE_NODE_FILE} is missing: the pod resizes with ComfyUI's ImageScale (slow)")
            return
        dest = f"{self.remote_code}/nodes/video_chain_scale"
        r = self.run(f"mkdir -p {shlex.quote(dest)} && cat > {shlex.quote(dest + '/__init__.py')}", timeout=60,
                     stdin=SCALE_NODE_FILE.read_bytes())
        if r.returncode != 0:
            raise RunpodError("could not copy video-chain's node to the pod: "
                              + r.stderr.decode(errors="replace").strip()[:300])

    def _setup(self) -> None:
        self._upload_nodes()
        env = {"COMFY_REF": COMFY_REF, "MODELS": self.models, "IDLE_MIN": str(self.idle_min),
               "VOLUME_PATH": VOLUME_PATH if self.volume else "", "TORCH_PINS": " ".join(TORCH_PINS),
               "CODE_DIR": self.remote_code}
        cmd = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " bash -s"
        try:
            p = subprocess.Popen(self.ssh_args() + [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        except FileNotFoundError as e:
            raise RunpodError("OpenSSH is not installed (ssh is missing): install it to use RunPod") from e
        tail: list[str] = []
        timed_out = threading.Event()

        def expire():
            timed_out.set()
            p.kill()
        timer = threading.Timer(self.setup_timeout, expire)
        timer.start()

        def feed():                              # (apart: ssh's answer is read meanwhile, a dead ssh must not block)
            try:
                p.stdin.write(Path(self.setup_script).read_bytes())
                p.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        threading.Thread(target=feed, daemon=True).start()
        try:
            for raw in p.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line.startswith("@@ "):
                    self.note("RunPod: " + line[3:])
                elif line:
                    tail = (tail + [line])[-20:]
                    self.log(line)
            code = p.wait()
        finally:
            timer.cancel()
            if p.poll() is None:
                p.kill()
                p.wait()
        if timed_out.is_set():
            raise RunpodError(f"preparing the pod timed out after {self.setup_timeout / 60:.0f} min: "
                              + " | ".join(tail[-3:]))
        if code != 0:
            raise RunpodError(f"preparing the pod failed (exit {code}): " + " | ".join(tail[-5:]))

    def _open_tunnel(self) -> None:
        port = free_port()
        self._tunnel_err = tempfile.TemporaryFile()      # (a file, not a pipe: nobody reads it while the tunnel lives)
        self.tunnel = subprocess.Popen(
            self.ssh_args("-N", "-o", "ExitOnForwardFailure=yes",
                          "-L", f"127.0.0.1:{port}:127.0.0.1:{self.remote_port}"),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._tunnel_err)
        url = f"http://127.0.0.1:{port}"
        end = time.monotonic() + 60
        while time.monotonic() < end:
            if self.tunnel.poll() is not None:
                self._tunnel_err.seek(0)
                err = self._tunnel_err.read().decode(errors="replace").strip()
                raise RunpodError(f"the SSH tunnel to the pod ended: {err}")
            try:
                with net.urlopen(urllib.request.Request(url + "/system_stats"), 5) as r:
                    if r.status == 200:
                        self.url = url
                        self.note("RunPod: connected")
                        return
            except Exception:  # noqa: BLE001  (not forwarded yet)
                pass
            time.sleep(1)
        raise RunpodError("ComfyUI on the pod does not answer through the SSH tunnel")

    # ---- while it runs
    def remove_remote(self, kind: str, subfolder: str, filename: str) -> None:
        """Delete one upload or result on the pod (only inside its RAM folder's video-chain subfolders)."""
        if kind not in ("input", "output") or subfolder != SUBFOLDER or not _SAFE_NAME.fullmatch(filename):
            return
        path = REMOTE_IO / kind / SUBFOLDER / filename
        try:
            self.run(f"rm -f -- {shlex.quote(str(path))}", timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def ram_free(self) -> int:
        """RAM the pod's container may still use, in bytes, from its cgroup (ComfyUI sees the host's RAM in a
        container, which is far more than the pod has); 0 when it cannot be read."""
        script = ("if [ -f /sys/fs/cgroup/memory.max ]; then m=$(cat /sys/fs/cgroup/memory.max); "
                  "c=$(cat /sys/fs/cgroup/memory.current); i=$(awk '$1==\"inactive_file\"{print $2}' "
                  "/sys/fs/cgroup/memory.stat); else m=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes); "
                  "c=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes); i=$(awk '$1==\"total_inactive_file\""
                  "{print $2}' /sys/fs/cgroup/memory/memory.stat); fi; echo \"$m $c ${i:-0}\"")
        try:
            r = self.run(script, timeout=30)
            limit, current, inactive = r.stdout.decode().split()[:3]
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return 0
        if not limit.isdigit():                       # "max": no limit inside, the pod record says its RAM
            gb = (self.pod.get("gpu") or {}).get("memory")
            if not gb:
                return 0
            limit = str(int(gb) * 2**30)
        return max(0, int(limit) - (int(current) - int(inactive or 0)))

    @property
    def cost_so_far(self) -> float | None:
        cost = self.pod.get("cost")
        if not isinstance(cost, (int, float)) or self.started is None:
            return None
        return cost * (time.monotonic() - self.started) / 3600

    # ---- end
    def close(self) -> None:
        """End the tunnel and terminate the pod (checked). Raises RunpodError if the pod may still exist, so the
        caller can tell the user; close() can then be called again to retry. Once it succeeded it does nothing."""
        with self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
        try:
            if self.tunnel is not None and self.tunnel.poll() is None:
                self.tunnel.terminate()
                try:
                    self.tunnel.wait(5)
                except subprocess.TimeoutExpired:
                    self.tunnel.kill()
            if self.pod_id:
                spent = self.cost_so_far
                err = None
                for attempt in range(5):
                    try:
                        self.api.delete_pod(self.pod_id)
                        st = self._status()
                        if st in (None, "TERMINATED"):
                            self.log(f"RunPod: pod {self.pod_id} terminated"
                                     + (f" (about ${spent:.2f})" if spent is not None else ""))
                            break
                        err = f"status {st}"
                    except RunpodError as e:
                        err = str(e)
                    time.sleep(2 * (attempt + 1))
                else:
                    raise RunpodError(f"could not terminate pod {self.pod_id} ({err}): end it in RunPod's console "
                                      f"(it removes itself after {self.idle_min} min without a connection)")
            with self._lock:
                self._closed = True              # only now: a failed close can be tried again
            self._tmp.cleanup()
        finally:
            self._closing = False

    def _remove_pods_named(self, name: str) -> None:
        """A create whose answer was lost may have made the pod: find it by its unique name and terminate it
        (best effort: the original error is what the caller gets)."""
        try:
            found = [p for p in leftover_pods(self.api) if p.get("name") == name]
            for p in found:
                self.api.delete_pod(p["id"])
                self.log(f"RunPod: removed pod {p['id']} whose creation was not answered")
        except RunpodError as e:
            self.log(f"RunPod: could not look for a pod made without an answer ({e}); check RunPod's console")

    def _remove_leftovers(self) -> None:
        """Pods of this app left by a crash or a lost connection (they bill): terminated before a new one is made.
        Best effort; a failure here never stops the start."""
        try:
            for p in leftover_pods(self.api):
                self.note(f"RunPod: ending a pod left over from an earlier run ({p.get('id')})")
                self.api.delete_pod(p["id"])
        except RunpodError as e:
            self.log(f"RunPod: could not check for leftover pods ({e})")

    def _status(self) -> str | None:
        try:
            return self.api.get_pod(self.pod_id).get("status")
        except RunpodError as e:
            if e.status == 404:
                return None
            raise


# ----------------------------------------------------------------------------- ComfyUI on the pod


def vram_total(stats: dict | None) -> int:
    """The first GPU's VRAM in bytes from ComfyUI's /system_stats (0: not said)."""
    for d in (stats or {}).get("devices") or []:
        if d.get("type") == "cuda" and d.get("vram_total"):
            return int(d["vram_total"])
    return 0


class CloudUpscaler(ComfyGenerator):
    """ComfyGenerator for the pod's ComfyUI (through the tunnel): SeedVR2 upscaling only. Uploads and results are
    removed on the pod after each job (its RAM folder), the frame budget comes from the pod's own RAM."""

    upscale_only = True

    def __init__(self, session: RunpodSession, log=print, progress=None, timeout: float = 3600):
        self.session = session
        super().__init__(session.url, log=log, progress=progress, timeout=timeout, comfy_dir=None, cleanup=False)
        self.cleanup = True
        self.input_dir, self.output_dir = REMOTE_IO / "input", REMOTE_IO / "output"
        self.vram = vram_total(self.stats)
        self.seedvr_variant = None                     # (as at home: copies made there or here are the same)
        # ComfyUI keeps its cache between clips: the VAE and model loaders are not run again for every clip (24 s
        # per clip on an H100 pod, 2026-11-13). At home the cache is dropped after each clip for its RAM; a pod has
        # far more (the pieces are cut to what is free, after the last clip's frames).
        self.seedvr_keep_cache = True
        if self.vram >= SEEDVR_FULL_VRAM:            # (an H100 & co.: what costs no accuracy and uses the GPU)
            self.seedvr_tiled, self.seedvr_precision = False, SEEDVR_PRECISION_FULL
            self.seedvr_variant = "fp16, whole frames"   # recorded with each kept copy: others are made again
        log(f"RunPod: {self.vram / 2**30:.0f} GB VRAM: SeedVR2 "
            + ("full precision (fp16 weights), whole frames (no tiles)" if self.vram >= SEEDVR_FULL_VRAM else
               "int8 weights, 512 px tiles (as at home)"))

    def _check_models(self):
        for node in SEEDVR_NODES:
            if not self._object_info(node):
                raise ComfyError(f"the pod's ComfyUI has no {node} node")

    def _remove(self, top, f: dict, local: Path, same_size: bool = True) -> None:
        kind = "output" if top == self.output_dir else "input" if top == self.input_dir else None
        if kind:
            self.session.remove_remote(kind, str(f.get("subfolder") or ""), str(f.get("filename") or ""))

    def _ram_free(self) -> int:
        comfy = super()._ram_free()
        pod = self.session.ram_free()
        return min(x for x in (comfy, pod) if x) if (comfy or pod) else 0

    def missing_loras(self, loras):
        return []

    def generate(self, *a, **k):
        raise ComfyError("the RunPod worker only upscales with SeedVR2; clips are made by the local ComfyUI")

    def refine(self, *a, **k):
        raise ComfyError("the RunPod worker only upscales with SeedVR2 (choose SeedVR2 7B or 3B, or switch off "
                         "“on RunPod”)")


# ----------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    """python -m video_chain.runpod status | down | gpus: this app's pods on RunPod, terminate them, GPU prices."""
    import argparse
    p = argparse.ArgumentParser(prog="python -m video_chain.runpod", description=main.__doc__)
    p.add_argument("what", choices=("status", "down", "gpus"))
    a = p.parse_args(argv)
    try:
        return _cli(a.what)
    except RunpodError as e:
        print(f"error: {e}")
        return 1


def _cli(what: str) -> int:
    api = RunpodApi(api_key())
    if what == "gpus":
        for g in api.gpus():
            price = (g.get("price") or {}).get("secure")
            print(f"{str(g.get('id') or '?'):40} {g.get('memory', '?'):>3} GB  "
                  + (f"${price:.2f}/h" if isinstance(price, (int, float)) else "   -   ")
                  + f"  {g.get('availability') or ''}")
        return 0
    pods, failed = leftover_pods(api), 0
    for pod in pods:
        print(f"{pod.get('id')}  {pod.get('name')}  {pod.get('status')}  {(pod.get('gpu') or {}).get('id', '')}")
        if what == "down":
            try:
                api.delete_pod(pod["id"])
                print("  terminated")
            except RunpodError as e:         # (the others still get their turn)
                failed += 1
                print(f"  NOT terminated: {e}")
    if not pods:
        print("no video-chain pods on RunPod")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
