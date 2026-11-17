"""What this computer has: operating system, package manager, graphics card, the programs the app calls.
Standard library only; nothing here changes the system."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .. import paths

MIN_VRAM_MB = 16000          # below this H3 does not fit even with ComfyUI's offloading (tested with 24 GB)
GOOD_VRAM_MB = 23000
MIN_DRIVER_CU13 = 580        # NVIDIA driver needed by CUDA 13 builds (PyTorch cu130, llama.cpp cuda-13)

# what to install with the system's package manager when something is missing
PACKAGES = {
    "ffmpeg": {"apt": "ffmpeg", "dnf": "ffmpeg (from RPM Fusion; Fedora's ffmpeg-free cannot write H.264)",
               "pacman": "ffmpeg", "zypper": "ffmpeg", "brew": "ffmpeg"},
    "qt": {"apt": "libxcb-cursor0 libegl1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 "
                  "libfontconfig1 libdbus-1-3 libpulse0 libnss3 libasound2t64",
           "dnf": "xcb-util-cursor libxkbcommon-x11 xcb-util-wm xcb-util-keysyms mesa-libEGL fontconfig dbus-libs "
                  "pulseaudio-libs nss alsa-lib",
           "pacman": "xcb-util-cursor libxkbcommon-x11 xcb-util-wm xcb-util-keysyms fontconfig",
           "zypper": "libxcb-cursor0 libxkbcommon-x11-0 Mesa-libEGL1 fontconfig"},
    "nvidia": {"apt": "the NVIDIA driver: sudo ubuntu-drivers install   (Debian: sudo apt install nvidia-driver)",
               "dnf": "the NVIDIA driver from RPM Fusion: sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda",
               "pacman": "the NVIDIA driver: sudo pacman -S nvidia-open nvidia-utils",
               "zypper": "the NVIDIA driver: https://en.opensuse.org/SDB:NVIDIA_drivers"},
}
INSTALL = {"apt": "sudo apt install -y", "dnf": "sudo dnf install -y", "pacman": "sudo pacman -S --needed",
           "zypper": "sudo zypper install -y", "brew": "brew install"}


def run(cmd: list[str], timeout: float = 30, **kw) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except (OSError, subprocess.SubprocessError):
        return None


def os_info() -> dict:
    info = {"system": platform.system(), "machine": platform.machine().lower(), "name": platform.platform(),
            "manager": None}
    if info["system"] == "Linux":
        rel = {}
        try:
            for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    rel[k] = v.strip().strip('"')
        except OSError:
            pass
        info["name"] = rel.get("PRETTY_NAME") or info["name"]
        for tool in ("apt-get", "dnf", "pacman", "zypper"):
            if shutil.which(tool):
                info["manager"] = "apt" if tool == "apt-get" else tool
                break
    elif info["system"] == "Darwin" and shutil.which("brew"):
        info["manager"] = "brew"
    return info


def install_hint(what: str, osi: dict | None = None) -> str:
    """The command that installs `what` (a key of PACKAGES) on this system."""
    osi = osi or os_info()
    pk = PACKAGES.get(what, {}).get(osi.get("manager") or "")
    if not pk:
        if paths.WINDOWS and what == "ffmpeg":
            return "winget install Gyan.FFmpeg   (or: video-chain setup --steps system)"
        return {"ffmpeg": "install ffmpeg (https://ffmpeg.org/download.html)",
                "nvidia": "install the NVIDIA driver (https://www.nvidia.com/drivers)",
                "qt": "install the libraries Qt needs (https://doc.qt.io/qt-6/linux-requirements.html)"}[what]
    if what == "nvidia":
        return pk
    return f"{INSTALL[osi['manager']]} {pk}"


NVIDIA_PROC = Path("/proc/driver/nvidia")


def gpu() -> dict | None:
    """The first NVIDIA card: {"name", "vram_mb", "free_mb", "driver"}; None when there is none (or no driver).
    nvidia-smi is asked when it is installed. Without it the driver itself tells the card and its version
    (Linux: /proc/driver/nvidia); the memory is then not known yet (vram_mb and free_mb are None) until PyTorch is
    installed and can be asked (torch_memory)."""
    smi = shutil.which("nvidia-smi")
    if smi:
        r = run([smi, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                15)
        if r is not None and r.returncode == 0 and r.stdout.strip():
            try:
                name, total, free, driver = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
                return {"name": name, "vram_mb": int(float(total)), "free_mb": int(float(free)), "driver": driver}
            except ValueError:
                pass
    return gpu_from_driver()


def gpu_from_driver(root: Path | None = None) -> dict | None:
    """The card and driver version from the loaded NVIDIA kernel driver, without nvidia-smi."""
    root = Path(root or NVIDIA_PROC)
    try:
        version = (root / "version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"\b(\d{3}\.\d+(?:\.\d+)?)\b", version)
    if not m:
        return None
    name = "NVIDIA graphics card"
    try:
        for info in sorted((root / "gpus").glob("*/information")):
            model = re.search(r"^Model:\s*(.+)$", info.read_text(encoding="utf-8", errors="replace"), re.M)
            if model:
                name = model.group(1).strip()
                break
    except OSError:
        pass
    return {"name": name, "vram_mb": None, "free_mb": None, "driver": m.group(1)}


def torch_memory(python: Path) -> tuple[int, int] | None:
    """(free MB, total MB) of the first card as PyTorch in this environment sees it; None when it sees none."""
    r = run([str(python), "-c", "import torch\nf, t = torch.cuda.mem_get_info(0)\nprint(f // 2**20, t // 2**20)"],
            180)
    if r is None or r.returncode != 0:
        return None
    try:
        free, total = r.stdout.strip().splitlines()[-1].split()
        return int(free), int(total)
    except (ValueError, IndexError):
        return None


def driver_major(g: dict | None) -> int:
    m = re.match(r"(\d+)", (g or {}).get("driver", ""))
    return int(m.group(1)) if m else 0


def ffmpeg_state() -> dict:
    """{"ffmpeg", "ffprobe": paths or None, "version", "x264", "aac": the encoders the app writes with}."""
    paths.use_private_tools()
    out: dict = {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"), "version": "", "x264": False,
           "aac": False}
    ffmpeg = out["ffmpeg"]
    if isinstance(ffmpeg, str):
        v = run([ffmpeg, "-hide_banner", "-version"])
        out["version"] = v.stdout.split("\n")[0].replace("ffmpeg version ", "").split(" ")[0] if v and v.stdout else ""
        e = run([ffmpeg, "-hide_banner", "-encoders"])
        text = e.stdout if e else ""
        out["x264"] = bool(re.search(r"^\s*V\S*\s+libx264\s", text, re.M))
        out["aac"] = bool(re.search(r"^\s*A\S*\s+aac\s", text, re.M))
    return out


def ffmpeg_ok(st: dict | None = None) -> bool:
    st = st or ffmpeg_state()
    return bool(st["ffmpeg"] and st["ffprobe"] and st["x264"] and st["aac"])


def find_uv() -> str | None:
    exe = "uv.exe" if paths.WINDOWS else "uv"
    for cand in (shutil.which("uv"), paths.tools_bin() / exe, Path.home() / ".local" / "bin" / exe,
                 Path.home() / ".cargo" / "bin" / exe):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def qt_missing_libs() -> list[str] | None:
    """Shared libraries Qt's window plugins need that this system lacks (Linux; None: could not be checked)."""
    if platform.system() != "Linux" or not shutil.which("ldd"):
        return None
    try:
        import PySide6
    except ImportError:
        return None
    plug = Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms"
    missing: set[str] = set()
    found = False
    for name in ("libqxcb.so", "libqwayland.so", "libqwayland-generic.so"):
        f = plug / name
        if not f.is_file():
            continue
        r = run(["ldd", str(f)])
        if r is None or r.returncode != 0:
            continue
        found = True
        libs = {m.group(1) for m in re.finditer(r"^\s*(\S+) => not found", r.stdout, re.M)}
        if name == "libqxcb.so" or not (plug / "libqxcb.so").is_file():
            missing |= libs
    return sorted(missing) if found else None


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def python_ok() -> bool:
    return sys.version_info >= (3, 12)
