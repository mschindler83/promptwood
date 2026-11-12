"""What can be installed: the components (model files, custom nodes, programs) from manifest.json, which
tools/build_manifest.py writes."""
from __future__ import annotations

import json
from pathlib import Path

PROFILES = ("core", "standard", "full")
PROFILE_TEXT = {"core": "video clips only", "standard": "every part of the app with one model each (recommended)",
                "full": "also the extra speed modes, Krea-2 and both head swaps"}


def load() -> dict:
    return json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))


def profile_components(man: dict, profile: str) -> list[str]:
    rank = PROFILES.index(profile)
    return [cid for cid, c in man["components"].items() if PROFILES.index(c["profile"]) <= rank]


def select(man: dict, profile: str = "standard", only: list[str] = (), add: list[str] = (),
           without: list[str] = ()) -> list[str]:
    """The component ids to install, in the manifest's order, with what they need (video always)."""
    comps = man["components"]
    for cid in [*only, *add, *without]:
        if cid not in comps:
            raise ValueError(f"unknown component '{cid}' (there are: {', '.join(comps)})")
    want = set(only) if only else set(profile_components(man, profile)) | set(add)
    want -= set(without)
    grew = True
    while grew:
        grew = False
        for cid in list(want):
            for need in comps[cid].get("needs", []):
                if need not in want:
                    want.add(need)
                    grew = True
    return [cid for cid in comps if cid in want]


def file_url(f: dict) -> str:
    if "url" in f:
        return f["url"]
    return f"https://huggingface.co/{f['repo']}/resolve/{f['rev']}/{f['path']}"


def hf_snapshot(f: dict, hf_cache: Path) -> Path:
    return Path(hf_cache) / ("models--" + f["repo"].replace("/", "--")) / "snapshots" / f["rev"]


def file_targets(f: dict, comfy_dir: Path, hf_cache: Path) -> list[Path]:
    """Where a manifest file ends up (several for an archive whose members are unpacked)."""
    if f.get("hf"):
        return [hf_snapshot(f, hf_cache) / f["path"]]
    base = Path(comfy_dir) / "models" / f["dir"]
    if "extract" in f:
        return [base / name for name in f["extract"]]
    return [base / f["name"]]


def nodes_for(man: dict, components: list[str]) -> list[str]:
    out: list[str] = []
    for cid in components:
        for n in man["components"][cid].get("nodes", []):
            if n not in out:
                out.append(n)
    return out


def software_for(man: dict, components: list[str]) -> list[str]:
    out: list[str] = []
    for cid in components:
        for n in man["components"][cid].get("software", []):
            if n not in out:
                out.append(n)
    return out
