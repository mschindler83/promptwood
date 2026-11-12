"""`video-chain setup`: brings a computer from "the app is unpacked" to "every part of the app works".

It installs ComfyUI (or uses one that is there), its custom nodes, the model files, the text to speech environment,
llama.cpp and, when the system has none that fits, ffmpeg; then it checks the result and renders one real clip.
Every step looks first at what is there, so running it again only does what is still missing (downloads resume).
Standard library only; Python environments are made with uv.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from . import doctor as D
from . import manifest as M
from . import system as S
from .console import CommandFailed, ask, ask_text, log_path, open_log, plan_steps, run, say, short, step, working
from .download import AGENT, DownloadError, extract, fetch, fetch_all, hf_token, human

STEPS = ("system", "comfyui", "nodes")          # what --steps accepts
# what the programs take on disk (measured on a fresh install; mostly PyTorch with NVIDIA's libraries)
PROGRAM_BYTES = {"app": 0.8 * 2**30, "comfyui": 7.5 * 2**30, "tts": 6 * 2**30, "llama": 0.6 * 2**30}
TEST_FREE_MB = 14000                            # free graphics memory the tests at the end need to be meaningful
FFMPEG_BUILDS = {"x86_64": "linux64", "amd64": "linux64", "aarch64": "linuxarm64", "arm64": "linuxarm64"}
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-{}-gpl.tar.xz"


class SetupError(RuntimeError):
    pass


@dataclass
class Options:
    profile: str | None = None                  # None: standard, and a person at the terminal is asked
    only: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    phase: str | None = None                    # "ask": questions, plan, checks and consent only (install.sh)
    answers: bool = False                       # continue from what the "ask" phase stored
    add: list[str] = field(default_factory=list)
    without: list[str] = field(default_factory=list)
    comfy_dir: str | None = None
    comfy_python: str | None = None
    comfy_launcher: str | None = None
    comfy_port: int | None = None
    models_from: list[str] = field(default_factory=list)
    hf_cache: str | None = None
    torch_index: str | None = None
    yes: bool = False
    accept_licenses: bool = False
    force: bool = False
    test: bool = True
    desktop: bool = True
    workers: int = 4
    dry_run: bool = False


def _port_busy(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Installer:
    def __init__(self, opts: Options):
        self.o = opts
        self.man = M.load()
        self.home = paths.home()
        self.interactive = sys.stdin.isatty() and not opts.yes
        self.profile_given = opts.profile is not None
        opts.profile = opts.profile or "standard"
        for s in opts.steps:
            if s not in STEPS:
                raise ValueError(f"unknown step '{s}' (there are: {', '.join(STEPS)})")
        self.steps = list(opts.steps)
        before = D.installed_components() or []
        self.models = not (self.steps and not opts.only)      # (--steps alone installs no parts)
        self.choose_components(before)
        self.before = before
        self.own_comfy = True
        self.uv = S.find_uv()
        if opts.hf_cache:                             # an existing Hugging Face cache: used (and added to) as it is
            self.home.mkdir(parents=True, exist_ok=True)
            paths.save_config(hf_cache=Path(opts.hf_cache).expanduser().resolve())
        self.token = hf_token()

    def choose_components(self, before: list[str] | None = None) -> None:
        o = self.o
        if not self.models:
            self.components = M.select(self.man, o.profile, (before if before is not None else self.before) or (),
                                       o.add, o.without)
        else:
            self.components = M.select(self.man, o.profile, o.only, o.add, o.without)

    def wants(self, name: str) -> bool:
        """Whether a step runs: all of them, unless --steps / --only narrowed the run."""
        if not self.o.only and not self.steps:
            return True
        if name in STEPS:
            return name in self.steps or (name == "nodes" and self.models) or \
                (name == "comfyui" and self.models and not (self.comfy_dir / "main.py").is_file())
        return self.models

    # ---- where things go
    @property
    def comfy_dir(self) -> Path:
        return Path(self.o.comfy_dir).expanduser().resolve() if self.o.comfy_dir else paths.comfy_dir()

    def comfy_py(self) -> Path:
        if self.o.comfy_python:
            return Path(self.o.comfy_python).expanduser()
        if not self.own_comfy:
            for venv in (".venv", "venv"):
                p = paths.venv_python(self.comfy_dir / venv)
                if p.is_file():
                    return p
            cfg = paths.load_config().get("comfy_python")
            if cfg and Path(cfg).is_file():
                return Path(cfg)
            raise SetupError(f"{self.comfy_dir} is a ComfyUI that setup did not install and it has no .venv: say "
                             "which Python runs it with --comfy-python /path/to/python")
        return paths.venv_python(self.comfy_dir / ".venv")

    # ---- questions (only when someone is there to answer)
    def ask_existing(self) -> None:
        """Ask for a ComfyUI and a Hugging Face cache that are already on the computer, so their models are used
        instead of downloaded. Skipped when the command line said it, or setup has run here before."""
        o = self.o
        if not self.interactive:
            return
        known = paths.load_config()
        if not (o.comfy_dir or o.models_from or known.get("comfy_dir") or (self.comfy_dir / "main.py").is_file()):
            guess = next((str(p) for p in (Path.home() / "ComfyUI", Path.home() / "comfy" / "ComfyUI",
                                           Path.home() / "Documents" / "ComfyUI") if (p / "main.py").is_file()), "")
            say("\nDo you already have ComfyUI on this computer?\n"
                "  1) No: install a new one for Promptwood\n"
                "  2) Yes: install a new one, but use the models of mine in place (nothing there is copied or "
                "changed)\n"
                "  3) Yes: use mine as it is (Promptwood's custom nodes and Python packages are installed into it)")
            choice = ask_text("Your choice", "2" if guess else "1")
            while choice in ("2", "3"):
                where = ask_text("Its folder (the one with main.py; for 2 a models folder works too)", guess)
                d = Path(where).expanduser() if where else None
                if d is not None and ((d / "main.py").is_file() or (choice == "2" and d.is_dir())):
                    if choice == "2":
                        o.models_from = [str(d)]
                    else:
                        o.comfy_dir = str(d)
                        if not any(paths.venv_python(d / v).is_file() for v in (".venv", "venv")):
                            o.comfy_python = ask_text("It has no .venv: which Python runs it (full path)") or None
                    break
                say(f"    {where or 'That'} is not a ComfyUI folder.")
                if not ask("    Try another folder?"):
                    break
        if not (o.hf_cache or known.get("hf_cache")) and M.software_for(self.man, self.components):
            mine = next((c for c in paths.model_cache_dirs()[1:] if c.is_dir()), None)
            files = [f for cid in self.components for f in self.man["components"][cid]["files"] if f.get("hf")]
            if mine is not None and any(D.find_file(f, self.comfy_dir, mine) for f in files):
                say(f"\nYour Hugging Face cache ({mine}) already has text to speech or language models that "
                    "Promptwood uses.")
                if ask("Use that cache (what is missing is downloaded into it) instead of a new one?"):
                    o.hf_cache = str(mine)
                    self.home.mkdir(parents=True, exist_ok=True)
                    paths.save_config(hf_cache=mine)

    def left_to_download(self, components: list[str]) -> int:
        cache = paths.hf_cache()
        return sum(f["size"] for cid in components for f in self.man["components"][cid]["files"]
                   if D.find_file(f, self.comfy_dir, cache) is None)

    def ask_profile(self) -> None:
        """Ask how much to install, with what each choice still has to download on this computer."""
        o = self.o
        if not self.interactive or self.profile_given or o.only or o.add or o.without or not self.models \
                or self.before:
            return
        say("\nHow much do you want to install?")
        for i, prof in enumerate(M.PROFILES, 1):
            ids = M.profile_components(self.man, prof)
            total = sum(self.man["components"][c]["bytes"] for c in ids)
            left = self.left_to_download(ids)
            size = (f"{human(total)} of models, all already on this computer" if not left else
                    f"{human(left)} to download" + (f" (of {human(total)}; the rest is already here)"
                                                   if left < total else ""))
            say(f"  {i}) {prof:9} {M.PROFILE_TEXT[prof]}\n               {size}")
        say("     Single parts can be added later at any time: video-chain setup --with NAME")
        choice = ask_text("Your choice", "2")
        o.profile = dict(zip("123", M.PROFILES)).get(choice, choice if choice in M.PROFILES else "standard")
        self.choose_components()

    # ---- between install.sh's two phases
    def answers_file(self) -> Path:
        return self.home / "setup-answers.json"

    def save_answers(self) -> None:
        o = self.o
        self.home.mkdir(parents=True, exist_ok=True)
        self.answers_file().write_text(json.dumps({
            "profile": o.profile, "only": o.only, "add": o.add, "without": o.without, "models_from": o.models_from,
            "comfy_dir": o.comfy_dir, "comfy_python": o.comfy_python, "hf_cache": o.hf_cache}), encoding="utf-8")

    def load_answers(self) -> bool:
        """What the person chose and agreed to in the "ask" phase (read once, then removed)."""
        try:
            data = json.loads(self.answers_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        self.answers_file().unlink(missing_ok=True)
        for k, v in data.items():
            setattr(self.o, k, v)
        self.o.yes = self.o.accept_licenses = True
        self.interactive = False
        self.choose_components()
        return True

    # ---- the plan
    def downloads(self) -> list[dict]:
        """The model files still to fetch: [{"url", "dest", "size", "sha256", "file"}]."""
        jobs, seen = [], set()
        cache = paths.hf_cache()
        for cid in self.components:
            for f in self.man["components"][cid]["files"]:
                if D.find_file(f, self.comfy_dir, cache) is not None:
                    continue
                if "extract" in f:
                    dest = self.home / "downloads" / f["url"].rsplit("/", 1)[-1]
                else:
                    dest = M.file_targets(f, self.comfy_dir, cache)[0]
                if str(dest) in seen:
                    continue
                seen.add(str(dest))
                jobs.append({"url": M.file_url(f), "dest": dest, "size": f["size"], "sha256": f.get("sha256"),
                             "file": f})
        return jobs

    def share_models(self) -> None:
        """--models-from: the new ComfyUI reads the model folders of another ComfyUI (or a plain models folder) in
        place, through its extra_model_paths.yaml. Nothing is copied and nothing there is changed."""
        if not self.o.models_from:
            return
        d = self.comfy_dir
        target = d / "extra_model_paths.yaml"
        if target.is_file() and "written by video-chain setup" not in target.read_text(encoding="utf-8"):
            raise SetupError(f"{target} exists and was not written by setup: add your model folders there yourself")
        sections: dict = {}
        for i, src in enumerate(self.o.models_from):
            src = Path(src).expanduser().resolve()
            models = src / "models" if (src / "models").is_dir() else src
            if not models.is_dir():
                raise SetupError(f"--models-from: {src} is not a folder")
            kinds = sorted(p.name for p in models.iterdir() if p.is_dir() and not p.name.startswith("."))
            sections[f"shared_{i}"] = {"base_path": str(models) + "/", **{k: k + "/" for k in kinds}}
            extra = src / "extra_model_paths.yaml"
            if extra.is_file():
                try:
                    import yaml
                    data = yaml.safe_load(extra.read_text(encoding="utf-8")) or {}
                except Exception as e:  # noqa: BLE001
                    say(f"    Note: {extra} could not be read ({e}); only {models} is shared")
                    data = {}
                for name, sec in (data.items() if isinstance(data, dict) else ()):
                    if not isinstance(sec, dict):
                        continue
                    sec = {k: v for k, v in sec.items() if k != "is_default"}
                    base = Path(os.path.expanduser(str(sec.get("base_path", ""))))
                    sec["base_path"] = str(base if base.is_absolute() else src / base) + "/"
                    sections[f"shared_{i}_{name}"] = sec
        lines = ["# written by video-chain setup (--models-from): model folders used in place, never changed"]
        for name, sec in sections.items():
            lines.append(f"{name}:")
            for k, v in sec.items():
                v = str(v)
                if "\n" in v.strip():
                    lines.append(f"    {k}: |")
                    lines += [f"        {x.strip()}" for x in v.splitlines() if x.strip()]
                else:
                    lines.append(f"    {k}: {v.strip()}")
        d.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def program_bytes(self) -> int:
        """Disk space the programs still to install take."""
        soft = M.software_for(self.man, self.components) if self.models else []
        need = 0 if (self.comfy_dir / "main.py").is_file() else PROGRAM_BYTES["comfyui"]
        need += PROGRAM_BYTES["tts"] if "tts" in soft and not paths.tts_python().is_file() else 0
        need += PROGRAM_BYTES["llama"] if "llama" in soft and not Path(paths.llama_server()).is_file() else 0
        need += PROGRAM_BYTES["app"] if self.o.phase == "ask" else 0
        return int(need)

    def show_plan(self) -> int:
        say("\nThe plan")
        say(f"    Home folder   {short(self.home)}")
        new = not (self.comfy_dir / "main.py").is_file()
        say(f"    ComfyUI       {short(self.comfy_dir)}  ({'a new one will be installed' if new else 'already there'})")
        for src in self.o.models_from:
            say(f"    Models from   {short(src)}  (used in place, nothing there is changed)")
        if self.steps and not self.o.only:
            say(f"    Steps         {', '.join(self.steps)}")
        elif self.o.only:
            say(f"    Only          {', '.join(self.o.only)}")
        else:
            say(f"    Profile       {self.o.profile}: {M.PROFILE_TEXT[self.o.profile]}")
        jobs = self.downloads() if self.models else []
        todo: dict[str, int] = {}
        for j in jobs:
            for cid in self.components:
                if j["file"] in self.man["components"][cid]["files"]:
                    todo[cid] = todo.get(cid, 0) + j["size"]
                    break
        say("    Parts")
        for cid in self.components:
            c = self.man["components"][cid]
            left = todo.get(cid, 0)
            state = "" if not self.models else "already on this computer" if not left else \
                f"{human(left)} to download"
            say(f"      {cid:13} {c['title']:68} {state}")
        rest = [cid for cid in self.man["components"] if cid not in self.components]
        if rest and self.models and not self.o.only:
            say("    Left out      " + ", ".join(rest) + "\n                  (add one with --with NAME; "
                "video-chain setup --help lists what each is)")
        need, software = sum(j["size"] for j in jobs), self.program_bytes()
        free = S.free_bytes(self.comfy_dir)
        say(f"    To download   models: {human(need) if need else 'nothing'}"
            + (f"; programs: about {human(software)} on disk (PyTorch, ComfyUI, …)" if software else ""))
        say(f"    Free space    {human(free)} on that disk")
        if any("huggingface.co" in j["url"] for j in jobs):
            say("    Hugging Face  " + ("your token is used for the downloads (sent to huggingface.co only)"
                                        if self.token else
                                        "no token found: downloads work without one, but Hugging Face may slow or "
                                        "limit them.\n                  For smoother downloads create a free read "
                                        "token (https://huggingface.co/settings/tokens)\n                  and "
                                        "start setup as  HF_TOKEN=hf_… ./install.sh  (or log in once: hf auth "
                                        "login).\n                  Setup reads it, never stores or prints it."))
        return need + software

    # ---- before anything is changed
    def preflight(self, need: int) -> None:
        say("\nThis computer")
        checks = [c for c in D.check_system() if not (self.o.phase == "ask" and c.name.startswith("Window"))]
        say(D.report(checks, colour=sys.stdout.isatty(), headings=False))
        problems = []
        if not S.python_ok():
            problems.append("Python 3.12 or newer is needed (install.sh brings its own)")
        if self.uv is None and self.o.phase != "ask":
            problems.append("uv was not found (install.sh installs it; or: https://docs.astral.sh/uv/)")
        gpu = S.gpu()
        if (gpu is None or gpu["vram_mb"] < S.MIN_VRAM_MB) and not self.o.force:
            problems.append(("no NVIDIA graphics card with a working driver was found: " + S.install_hint("nvidia")
                             if gpu is None else
                             f"the graphics card has {gpu['vram_mb'] / 1024:.0f} GB; the video model needs "
                             f"{S.MIN_VRAM_MB // 1000} GB or more") + "   (--force installs anyway)")
        free = S.free_bytes(self.comfy_dir)
        if need > free and not self.o.force:
            problems.append(f"not enough disk space: {human(need)} needed, {human(free)} free on the disk of "
                            f"{self.comfy_dir}. Choose a smaller set (--profile core) or another place "
                            "(VIDEO_CHAIN_HOME=/big/disk/video-chain)")
        if problems:
            raise SetupError("Setup cannot start:\n  - " + "\n  - ".join(problems))

    def window_libraries(self) -> None:
        """The system libraries Qt's window needs (known only once the app's own packages are installed)."""
        libs = S.qt_missing_libs()
        if libs:
            say()
            self.system_packages("qt", f"The app's window needs system libraries that are missing ({', '.join(libs)}).")

    def system_packages(self, what: str, why: str) -> bool:
        """Install system packages when that is possible without surprises: as root directly, else with sudo after
        asking. Otherwise the command is shown (and the final check reports what is still missing)."""
        osi = S.os_info()
        hint = S.install_hint(what, osi)
        say(f"    {why}")
        pk = S.PACKAGES.get(what, {}).get(osi.get("manager") or "")
        if not pk or what == "nvidia" or "(" in pk:
            say(f"    Please run: {hint}")
            return False
        cmd = hint.split()
        root = hasattr(os, "geteuid") and os.geteuid() == 0
        if root:
            cmd = cmd[1:]
            if osi["manager"] == "apt":
                subprocess.run(["apt-get", "update", "-qq"])
        elif not (self.interactive and shutil.which("sudo") and ask(f"    Run '{hint}' now?")):
            say(f"    Please run: {hint}")
            return False
        try:
            run(cmd, "Installing system packages", env={"DEBIAN_FRONTEND": "noninteractive"})
            return True
        except CommandFailed as e:
            say(f"    {e}\n    Please run: {hint}")
            return False

    def licences(self) -> None:
        """Show, for every part, the licence its publisher states, and ask for consent. Nothing here is setup's
        own judgement: the names and links are the publishers' (read on the manifest's date)."""
        say(f"\nLicences  (as each publisher states them on its page; read on {self.man.get('checked', '?')})")
        plain: list[str] = []
        for cid in self.components:
            seen: dict[str, dict] = {}
            for f in self.man["components"][cid]["files"]:
                lic = f.get("licence") or {"name": "not recorded", "link": "", "open": False}
                seen.setdefault(lic["name"], lic)
            special = [v for v in seen.values() if not v.get("open")]
            if not special:
                plain.append(cid)
                continue
            for i, lic in enumerate(special):
                say(f"    {cid if i == 0 else '':13} {lic['name']}\n                  {lic['link']}")
        if plain:
            say(f"    {', '.join(plain)}: Apache-2.0 / MIT")
        say("    Several of these allow research or non-commercial use only, and where no licence is stated you have\n"
            "    no stated permission at all. Whether your use is allowed is for you to check on the pages above;\n"
            "    setup cannot judge it. You accept a model's licence by downloading it. Parts you do not want can\n"
            "    be left out (--without NAME). docs/MODELS.md lists every file with its source and licence.")
        if self.o.accept_licenses:
            return
        if not self.interactive or not ask("    Do you accept these licences?"):
            raise SetupError("The licences were not accepted (pass --accept-licenses to accept them without being "
                             "asked).")

    # ---- ffmpeg
    def ffmpeg(self) -> None:
        st = S.ffmpeg_state()
        step("ffmpeg")
        say("    " + ("This system has no ffmpeg." if not st["ffmpeg"] else
                      f"The system's ffmpeg ({st['ffmpeg']}) cannot write H.264 with libx264."))
        build = FFMPEG_BUILDS.get(platform.machine().lower()) if platform.system() == "Linux" else None
        if build is None:
            raise SetupError("Please install ffmpeg with libx264: " + S.install_hint("ffmpeg"))
        say(f"    Downloading a complete one into {short(paths.tools_bin())} (the system's stays untouched) …")
        archive = self.home / "downloads" / "ffmpeg.tar.xz"
        fetch(FFMPEG_URL.format(build), archive)
        extract(archive, paths.tools_bin(), members=["ffmpeg", "ffprobe"])
        archive.unlink(missing_ok=True)
        if not S.ffmpeg_ok():
            raise SetupError("The downloaded ffmpeg does not run here. Please install one: "
                             + S.install_hint("ffmpeg"))

    # ---- Python packages
    def pip(self, py: Path, args: list[str], what: str, expect: str = "") -> None:
        if (Path(py).parent.parent / "pyvenv.cfg").is_file():
            run([self.uv, "pip", "install", "--python", py, *args], what, expect=expect)
        else:                                          # a ComfyUI running on a system Python: its own pip
            run([py, "-m", "pip", "install", *args], what, expect=expect)

    def torch_index(self) -> str | None:
        if self.o.torch_index:
            return None if self.o.torch_index == "pypi" else self.o.torch_index
        if platform.system() == "Darwin":
            return None
        gpu = S.gpu()
        if gpu is None:
            return "https://download.pytorch.org/whl/cpu"
        return ("https://download.pytorch.org/whl/cu130" if S.driver_major(gpu) >= S.MIN_DRIVER_CU13 else
                "https://download.pytorch.org/whl/cu126")

    # ---- ComfyUI
    def comfyui(self) -> None:
        d, spec = self.comfy_dir, self.man["comfy"]
        there = (d / "main.py").is_file()
        marker = d / ".video-chain-setup"
        self.own_comfy = not there or marker.is_file()
        port, moved = self.o.comfy_port, None
        if port is None:
            m = re.search(r":(\d+)$", paths.load_config().get("comfy_url", ""))
            port = int(m.group(1)) if m else 8188
            if not m and self.own_comfy and not there:
                # a fresh ComfyUI next to another program on 8188 (often: another ComfyUI) takes the next free port
                first = port
                while port < first + 20 and _port_busy(port):
                    port += 1
                moved = first if port != first else None
        url = f"http://127.0.0.1:{port}"
        if self.wants("comfyui") or not there:
            if self.own_comfy:
                step(f"ComfyUI in {short(d)}")
                if moved is not None:
                    say(f"    Port {moved} is taken by another program (often another ComfyUI, which stays as it "
                        f"is): this one gets port {port}.")
                if not there:
                    archive = self.home / "downloads" / "comfyui.tar.gz"
                    say(f"    ComfyUI's code (revision {spec['rev']})")
                    fetch(f"{spec['git']}/archive/{spec['rev']}.tar.gz", archive)
                    extract(archive, d, strip=1)
                    archive.unlink(missing_ok=True)
                    marker.write_text(spec["rev"] + "\n")
                py = self.comfy_py()
                if not py.is_file():
                    run([self.uv, "venv", "--python", spec["python"], d / ".venv"], "Creating ComfyUI's environment")
                done = d / ".venv" / ".video-chain-packages"
                if not done.is_file() or done.read_text().strip() != spec["rev"]:
                    index = self.torch_index()
                    kind = index.rsplit("/", 1)[-1] if index else "standard build"
                    self.pip(py, ["torch", "torchvision", "torchaudio", *(["--index-url", index] if index else [])],
                             f"Installing PyTorch ({kind})", "about 3 GB, the longest step here")
                    self.pip(py, ["-r", d / "requirements.txt",
                                  *(["--extra-index-url", index] if index else [])], "Installing ComfyUI's packages")
                    done.write_text(spec["rev"] + "\n")
            else:
                step(f"ComfyUI: using the one in {short(d)}")
                say(f"    Its Python: {short(self.comfy_py())}")
                say("    Promptwood needs a recent ComfyUI (MiniMax H3, SeedVR2 and Qwen-Image 2.1 nodes); the final "
                    "check tells if this one is too old.")
        py = self.comfy_py()
        launcher = Path(self.o.comfy_launcher).expanduser() if self.o.comfy_launcher else None
        if launcher is None:
            cfg = paths.load_config().get("comfy_launcher")
            ours = self.home / "bin" / "comfyui"
            launcher = Path(cfg) if cfg and Path(cfg).is_file() and Path(cfg) != ours else ours
            if launcher == ours:
                ours.parent.mkdir(parents=True, exist_ok=True)
                ours.write_text("#!/usr/bin/env bash\n# Starts ComfyUI for Promptwood (written by video-chain "
                                "setup). Extra arguments go to ComfyUI.\n"
                                f'cd "{d}" || exit 1\nexec "{py}" main.py --port {port} "$@"\n')
                ours.chmod(0o755)
        paths.save_config(comfy_dir=d, comfy_python=py, comfy_launcher=launcher, comfy_url=url)

    def nodes(self) -> None:
        d = self.comfy_dir
        names = M.nodes_for(self.man, self.components)
        if not names:
            return
        step("ComfyUI's custom nodes")
        py = self.comfy_py()
        for name in names:
            n = self.man["nodes"][name]
            target = D.node_dir(self.man, name, d)
            if "bundled" in n:
                src = Path(__file__).resolve().parents[3] / n["bundled"]
                if not src.is_dir():
                    raise SetupError(f"{src} is missing: Promptwood must be installed from its source folder "
                                     "(install.sh does that)")
                target.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*.py"):
                    shutil.copy2(f, target / f.name)
                say(f"    {target.name}: Promptwood's own nodes copied")
            elif D.node_installed(self.man, name, d):
                say(f"    {target.name}: already there")
                continue
            else:
                say(f"    {target.name}: {n['git']} at {n['rev']}")
                archive = self.home / "downloads" / f"node-{name}.tar.gz"
                fetch(f"{n['git']}/archive/{n['rev']}.tar.gz", archive)
                extract(archive, target, strip=1)
                archive.unlink(missing_ok=True)
            done = target / ".video-chain-packages"
            req = target / "requirements.txt"
            want = ",".join(n.get("pip", [])) + (f"|{req.stat().st_size}" if req.is_file() else "")
            if want and not (done.is_file() and done.read_text() == want):
                if n.get("pip"):
                    self.pip(py, list(n["pip"]), f"Installing packages for {target.name}")
                if req.is_file():
                    self.pip(py, ["-r", req], f"Installing packages for {target.name}")
                done.write_text(want)

    # ---- models
    def model_files(self) -> None:
        jobs = self.downloads()
        cache = paths.hf_cache()
        if jobs:
            step(f"Models: {len(jobs)} files, {human(sum(j['size'] for j in jobs))}")
            say("    Safe to interrupt (Ctrl+C): the next run continues where this one stopped.")
            fetch_all(jobs, self.o.workers, "    ", self.token)
            for j in jobs:
                f = j["file"]
                if "extract" in f:
                    extract(j["dest"], self.comfy_dir / "models" / f["dir"], members=f["extract"])
                    Path(j["dest"]).unlink(missing_ok=True)
        for cid in self.components:                      # the Hugging Face cache's pointer to each snapshot
            for f in self.man["components"][cid]["files"]:
                if f.get("hf"):
                    ref = M.hf_snapshot(f, cache).parent.parent / "refs" / "main"
                    if not ref.is_file() and M.hf_snapshot(f, cache).is_dir():
                        ref.parent.mkdir(parents=True, exist_ok=True)
                        ref.write_text(f["rev"])
        D.forget_listings()
        for cid in self.components:
            have, total, _ = D.component_state(self.man, cid, self.comfy_dir, cache)
            if have != total:
                raise SetupError(f"{cid}: {total - have} of {total} files are still missing after the download")

    # ---- programs next to the app
    def tts_done(self) -> bool:
        py = paths.tts_python()
        done = py.parent.parent / ".video-chain-packages"
        return py.is_file() and done.is_file() and done.read_text() == ",".join(self.man["software"]["tts"]["pip"])

    def llama_done(self) -> bool:
        done = self.home / "tools" / "llama.cpp" / ".video-chain-version"
        return done.is_file() and done.read_text().strip() == self.man["software"]["llama"]["tag"] \
            and Path(paths.llama_server()).is_file()

    def tts(self) -> None:
        spec = self.man["software"]["tts"]
        py = paths.tts_python()
        done = py.parent.parent / ".video-chain-packages"
        want = ",".join(spec["pip"])
        step("Text to speech (Qwen3-TTS)")
        if not py.is_file():
            run([self.uv, "venv", "--python", spec["python"], py.parent.parent], "Creating the TTS environment")
        self.pip(py, list(spec["pip"]), "Installing Qwen3-TTS with its own PyTorch", "about 3 GB")
        done.write_text(want)
        paths.save_config(tts_python=py)

    def llama_asset(self, assets: list[dict]) -> list[dict]:
        """The release files for this computer: the server build, and NVIDIA's runtime libraries with a CUDA one."""
        sysname, machine = platform.system(), platform.machine().lower()
        arch = "x64" if machine in ("x86_64", "amd64") else "arm64"
        gpu = S.gpu()
        if sysname == "Darwin":
            wants = [rf"-bin-macos-{arch}\."]
        elif gpu is not None:
            cuda = r"cuda-13[\d.]*" if S.driver_major(gpu) >= S.MIN_DRIVER_CU13 else r"cuda-12[\d.]*"
            wants = [rf"^llama-.*-bin-ubuntu-{cuda}-{arch}\.", rf"^llama-.*-bin-ubuntu-vulkan-{arch}\."]
        else:
            wants = [rf"^llama-.*-bin-ubuntu-vulkan-{arch}\.", rf"^llama-.*-bin-ubuntu-{arch}\."]
        for pat in wants:
            hit = next((a for a in assets if re.search(pat, a["name"])), None)
            if hit:
                extra = next((a for a in assets if a["name"] == "cudart-" + hit["name"]), None)
                return [hit] + ([extra] if extra else [])
        raise SetupError(f"llama.cpp has no ready-made build for {sysname} {machine}; install llama-server yourself "
                         "and put it on PATH (https://github.com/ggml-org/llama.cpp)")

    def llama(self) -> None:
        spec = self.man["software"]["llama"]
        target = self.home / "tools" / "llama.cpp"
        done = target / ".video-chain-version"
        step("llama.cpp (runs the prompt refiner's language model)")
        api = f"https://api.github.com/repos/{spec['repo']}/releases/tags/{spec['tag']}"
        try:
            with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": AGENT}), timeout=60) as r:
                assets = json.loads(r.read())["assets"]
        except Exception as e:  # noqa: BLE001
            raise SetupError(f"could not read llama.cpp's release list ({e}); try again later") from e
        if target.is_dir():
            shutil.rmtree(target)
        for a in self.llama_asset(assets):
            archive = self.home / "downloads" / a["name"]
            say(f"    {a['name']} ({human(a['size'])})")
            fetch(a["browser_download_url"], archive, a["size"])
            extract(archive, target)
            archive.unlink(missing_ok=True)
        server = next((p for p in sorted(target.rglob("llama-server*"), key=lambda p: len(p.parts))
                       if p.is_file() and os.access(p, os.X_OK)), None)
        if server is None:
            raise SetupError("llama.cpp's download has no llama-server in it")
        for lib in target.rglob("*.so*"):                # NVIDIA's libraries next to the server, where it looks
            if lib.parent != server.parent and not (server.parent / lib.name).exists():
                os.symlink(os.path.relpath(lib, server.parent), server.parent / lib.name)
        paths.save_config(llama_server=server)
        r = S.run([str(server), "--version"], 60)
        if r is None or r.returncode != 0:
            raise SetupError(f"{server} does not start here: {(r.stderr or r.stdout).strip()[-600:] if r else ''}")
        done.write_text(spec["tag"] + "\n")

    # ---- starting the app
    def desktop(self) -> None:
        """Commands in <home>/bin (linked into ~/.local/bin when that is on PATH) and a menu entry."""
        bin_dir = self.home / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        py = Path(sys.executable)
        for name, module in (("video-chain", "video_chain"), ("video-chain-gui", "video_chain.app")):
            script = bin_dir / name
            script.write_text(f'#!/usr/bin/env bash\n# Promptwood (written by video-chain setup)\n'
                              f'export VIDEO_CHAIN_HOME="{self.home}"\nexec "{py}" -m {module} "$@"\n')
            script.chmod(0o755)
        if not self.o.desktop or platform.system() != "Linux":
            return
        local = Path.home() / ".local" / "bin"
        if str(local) in os.environ.get("PATH", "").split(os.pathsep) and local.is_dir():
            for name in ("video-chain", "video-chain-gui"):
                link = local / name
                if link.is_symlink() or not link.exists():
                    if link.is_symlink():
                        link.unlink()
                    link.symlink_to(bin_dir / name)
        apps = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "applications"
        icon = Path(__file__).resolve().parents[1] / "theme" / "icons" / "app.svg"
        try:
            apps.mkdir(parents=True, exist_ok=True)
            (apps / "video-chain.desktop").write_text(
                "[Desktop Entry]\nType=Application\nName=Promptwood\n"
                "Comment=Chain image-to-video clips into one film\n"
                f"Exec={bin_dir / 'video-chain-gui'}\n" + (f"Icon={icon}\n" if icon.is_file() else "")
                + "Terminal=false\nCategories=AudioVideo;Video;\nStartupWMClass=video-chain\n")
        except OSError:
            pass

    # ---- the end: check, and prove it
    def verify(self) -> bool:
        step("Checking the result" + (" and testing with the real models" if self.o.test else ""))
        installed = sorted(set(self.before) | set(self.components), key=list(self.man["components"]).index)
        if self.models:                                  # (a run of single steps installs no parts)
            paths.save_config(components=",".join(installed))
        with working("Looking at what is installed", quiet_end=True):
            checks = D.check_all(installed)
        gpu = S.gpu()
        busy = gpu is not None and gpu["free_mb"] < TEST_FREE_MB
        if self.o.test and busy and not D.failed(checks):
            checks.append(D.Check(D.TESTS, "Test clip, test line, language model", D.WARN,
                                  f"skipped: other programs use the graphics card ({gpu['free_mb'] / 1024:.0f} GB "
                                  "free)", "close them, then: video-chain doctor --render --speech --llm"))
        if self.o.test and not busy and not D.failed(checks):
            checks += run_tests(self.man, installed)
        say()
        say(D.report(checks, colour=sys.stdout.isatty()))
        say()
        say(D.summary(checks))
        return not D.failed(checks)

    def pending(self) -> list[tuple[str, object]]:
        """The steps this run will do, in order (so they can be counted before the first one starts)."""
        soft = M.software_for(self.man, self.components) if self.models else []
        there = (self.comfy_dir / "main.py").is_file()
        todo: list[tuple[str, object]] = []
        if self.wants("system") and not S.ffmpeg_ok():
            todo.append(("ffmpeg", self.ffmpeg))
        todo.append(("comfyui", self.comfyui))
        if self.wants("nodes") and M.nodes_for(self.man, self.components):
            todo.append(("nodes", self.nodes))
        if self.models:
            todo.append(("models", self.model_files))
            if "tts" in soft and not self.tts_done():
                todo.append(("tts", self.tts))
            if "llama" in soft and not self.llama_done():
                todo.append(("llama", self.llama))
        todo.append(("desktop", self.desktop))
        todo.append(("verify", None))
        # the ones that print a heading
        self._headed = sum(1 for name, _ in todo if name in ("ffmpeg", "nodes", "tts", "llama", "verify")
                           or (name == "comfyui" and (self.wants("comfyui") or not there))
                           or (name == "models" and self.downloads()))
        return todo

    def run(self) -> int:
        resumed = self.o.answers and self.load_answers()
        log = None if self.o.dry_run else open_log()       # (a dry run writes nothing at all)
        say("Promptwood setup" + ("" if resumed or log is None else
                                   f"\n    Everything it prints, and more, is kept in {short(log)}"))
        if not resumed:
            self.ask_existing()
            if not self.o.dry_run:
                self.share_models()
            self.ask_profile()
            need = self.show_plan()
            if self.o.dry_run:
                say("\n(--dry-run: nothing was installed)")
                return 0
            self.preflight(need)
            if self.models:
                self.licences()
            if self.interactive and not ask("\nStart the installation?"):
                say("Nothing was installed.")
                return 1
            if self.o.phase == "ask":
                self.save_answers()
                return 0
        else:
            self.share_models()
        self.home.mkdir(parents=True, exist_ok=True)
        self.window_libraries()
        todo = self.pending()
        plan_steps(self._headed)
        ok = True
        for name, fn in todo:
            if name == "verify":
                ok = self.verify()
            else:
                fn()
        shutil.rmtree(self.home / "downloads", ignore_errors=True)
        if ok:
            say(f"\nPromptwood is ready. Start it from your applications menu, or with:\n"
                f"    {short(self.home / 'bin' / 'video-chain-gui')}\n"
                "ComfyUI is started by the app when it is needed (the Start button next to its status).")
        else:
            say(f"\nThe full output of this run is in {short(log)}")
        return 0 if ok else 1


def run_tests(man: dict, installed: list[str]) -> list:
    """The tests with the real models, one after the other (the graphics card holds one model at a time): ComfyUI
    lists every node and model, a short clip is rendered, a line is spoken, the language model answers."""
    checks: list = []
    url = paths.comfy_url()
    quiet = lambda _t: None                                  # noqa: E731
    try:
        with working("Starting ComfyUI", "up to a minute") as w:
            session = D.ComfySession(url, log=w.note)
            session.__enter__()
        try:
            with working("Asking ComfyUI for its nodes and models", quiet_end=True):
                checks += D.deep_checks(url, installed)
            if "video" in installed and not D.failed(checks):
                with working("Rendering a test clip", "loads about 35 GB of models: 1 to 5 minutes"):
                    checks.append(D.render_test(url, log=quiet))
            if session.proc is None:                         # a ComfyUI that was running already: memory back
                try:
                    req = urllib.request.Request(url + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                                 headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=30).close()
                except OSError:
                    pass
        finally:
            session.__exit__(None, None, None)
    except Exception as e:  # noqa: BLE001
        checks.append(D.Check(D.LIVE, "Start ComfyUI and ask it", D.FAIL, str(e)))
    soft = M.software_for(man, installed)
    if "tts" in soft and not D.failed(checks):
        with working("Speaking a test line", "loads the voice model: about a minute"):
            checks.append(D.speech_test(log=quiet))
    if "llama" in soft and not D.failed(checks):
        with working("Asking the language model a question", "loads the model: about a minute"):
            checks.append(D.llm_test(log=quiet))
    return checks


def main_setup(opts: Options) -> int:
    try:
        return Installer(opts).run()
    except (SetupError, DownloadError, CommandFailed, ValueError) as e:
        say(f"\nSetup stopped: {e}")
        if log_path() is not None:
            say(f"\nThe full output of this run is in {short(log_path())}\n"
                "After fixing the cause, run the same command again: what is done is kept.")
        return 1
    except KeyboardInterrupt:
        say("\nInterrupted. Run the same command again to continue: what is done is kept, downloads resume.")
        return 130
