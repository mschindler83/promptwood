"""The image-edit results kept in a project (no Qt): edits/<name>.png with its record edits/<name>.json, the copies of
what a result was made from in edits/sources/, the trash of deleted results, and the words about how a result was made.
The image dialog (ui/image_edit.py) and its generate worker use this; so can anything without a window."""
from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import TypedDict, cast

from . import media
from .comfy import EDIT_MODELS
from .imagegen import FAMILIES

EDITS_DIR = "edits"
TRASH_DIR = ".deleted"                      # (inside edits/: Undo delete brings a result back from here)
SOURCES_DIR = "sources"
SOURCE_GRACE = 5.0                          # seconds: a source copy this young may belong to a result being made
MOSS_EDIT_QUALITIES = {"standard": "Standard (30 steps)"}
LORA_FAMILY = {"chroma": "flux1"}           # Chroma LoRAs have FLUX.1's layer names (and FLUX.1 LoRAs often work)
QUALITY_TEXT = {"fast": "Fast (Lightning, 4 steps)", "standard": "Standard (20 steps)", "best": "Best (40 steps, slow)"}


class ResultRecord(TypedDict, total=False):
    """What edits/<name>.json holds. The generator adds its own keys (model, steps, cfg ...); every key is optional,
    an unreadable record is {}."""
    run: str                                # the Generate this result belongs to
    mode: str                               # "create" (else an edit)
    prompt: str
    negative: str
    template: str                           # the prompt with its {placeholders} (made from variations)
    variation: dict                         # {"values", "step", "index", "key"}
    trigger_words: list[str]
    seed: int
    source: str                             # the image that was edited
    source_copy: str | None                 # its copy in edits/sources/ (create: the first reference picture)
    mask: str
    refs: list[str]
    size: list[int] | None
    folder: str
    width: int
    height: int
    family: str
    edit_model: str
    quality: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    sampling_set: dict
    loras: list[dict]
    tags: bool
    favorite: bool
    uses: list[dict]                        # {"kind": "start", "sha1"} / {"kind": "end", "file"} / {"kind": "ref", "file"}


def scale_to(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Scale an image to exactly this size (the result keeps the size of the image it was made from)."""
    with media.part_file(dst) as tmp:
        media.ffmpeg(["-y", "-i", src, "-vf", f"scale={size[0]}:{size[1]}:flags=lanczos", "-frames:v", "1", tmp],
                     timeout=120)


def mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def project_edits(project: Path) -> list[tuple[Path, ResultRecord]]:
    """The edit results kept in a project, newest first, with what they were made from."""
    d = Path(project) / EDITS_DIR
    out = []
    for png in d.glob("*.png") if d.is_dir() else []:
        if ".part" in png.name:
            continue
        meta = _record(png.with_suffix(".json"))
        out.append((png, meta))
    return sorted(out, key=lambda x: mtime(x[0]), reverse=True)


_RECORDS: dict[str, tuple[tuple[int, int], ResultRecord]] = {}     # record file -> ((mtime_ns, size), parsed)


def _record(f: Path) -> ResultRecord:
    """A result's record, parsed once per version of the file (a reload of hundreds of results reads none again)."""
    try:
        st = f.stat()
        key = (st.st_mtime_ns, st.st_size)
        hit = _RECORDS.get(str(f))
        if hit is None or hit[0] != key:
            hit = (key, json.loads(f.read_text(encoding="utf-8")))
            _RECORDS[str(f)] = hit
    except (OSError, ValueError):
        return {}
    return copy.deepcopy(hit[1])


def read_meta(png: str | Path) -> ResultRecord | None:
    """A result's record as it is on disk now (not the cached one); None if it is missing, unreadable or not a
    record."""
    try:
        meta = json.loads(Path(png).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cast(ResultRecord, meta) if isinstance(meta, dict) else None


def write_meta(meta_file: Path, meta: ResultRecord | dict) -> None:
    """A result's record, written atomically (a reader never sees half a file)."""
    media.atomic_write_json(meta_file, meta)


def record_use(png: str | Path, use: dict) -> None:
    """Note in a result's record which project copy it became ({"kind": "start", "sha1"} / {"kind": "end",
    "file"} / {"kind": "ref", "file"}), so the window knows it is in use."""
    meta_file = Path(png).with_suffix(".json")
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        meta = {}
    except (OSError, ValueError):
        return                                      # (an unreadable record is not replaced by a near-empty one)
    uses = [u for u in meta.get("uses") or [] if u != use]
    meta["uses"] = uses + [use]
    write_meta(meta_file, meta)


def how_made(meta: ResultRecord) -> str:
    """The model and quality a result was made with, in words (for the result's facts)."""
    if meta.get("mode") == "create":
        fam = FAMILIES.get(meta.get("family") or "", {})
        return f"{fam.get('label', '')} {fam.get('qualities', {}).get(meta.get('quality'), '')}"
    if meta.get("edit_model") == "qwen21":
        return f"{EDIT_MODELS['qwen21']} {FAMILIES['qwen21']['qualities'].get(meta.get('quality'), '')}"
    if meta.get("edit_model") == "moss":
        return f"{EDIT_MODELS['moss']} {MOSS_EDIT_QUALITIES['standard']}"
    return QUALITY_TEXT.get(meta.get("quality") or "", "")


def lora_note(mode: str, family: dict | None, fam: str, edit_model: str, family_loras: dict) -> str:
    """The line above the LoRA list: which LoRAs fit the chosen model, and how many were found."""
    if mode == "create":
        if not family:
            return ""
        found = len(family_loras.get(LORA_FAMILY.get(fam, fam), []))
        if fam == "chroma":
            return ("LoRAs (optional): Chroma and FLUX.1 LoRAs (the same layer names, so they are listed together; "
                    f"FLUX.1 LoRAs often work on Chroma, not always; {found} found), applied in this order.")
        return (f"LoRAs (optional): {family['label']} LoRAs from ComfyUI's LoRA folders (found by their contents; "
                f"{found} found), applied in this order.")
    if edit_model == "qwen21":
        return ("LoRAs (optional): Qwen-Image 2.1 LoRAs from ComfyUI's LoRA folders (found by their contents; "
                f"{len(family_loras.get('qwen21', []))} found), applied in this order.")
    if edit_model == "moss":
        return ("LoRAs (optional): MOSS-Image LoRAs from ComfyUI's LoRA folders (found by their contents; "
                f"{len(family_loras.get('moss', []))} found), applied before its edit LoRA.")
    return ("LoRAs (optional): Qwen Image / Image Edit LoRAs from ComfyUI's LoRA folders, applied in this order. "
            "Text-to-image Qwen LoRAs often work for edits too, not always.")


def uses_index(tree) -> dict:
    """What `uses_of` looks up, built once for the whole project (a reload asks once per result): who owns each
    face picture, and which clips have which end image."""
    from . import tree as T
    owner = {r: c["name"] for c in (T.cast_of(tree) if tree is not None else []) for r in c["images"]}
    ends: dict[str, list[str]] = {}
    for n in (tree.nodes.values() if tree is not None else []):
        for f in (n.end_image, (n.made or {}).get("end_image")):
            if f:
                ends.setdefault(f, []).append(n.id)
    return {"owner": owner, "ends": ends}


def uses_of(project: Path, tree, meta: ResultRecord, index: dict | None = None) -> list[str]:
    """Where a result is used now, in words: the start image (still that picture), a clip's end image (set now or
    made with), a face reference. Empty: it can go. index: `uses_index(tree)`, when many results are asked in a row."""
    from . import chain
    from . import tree as T
    out = []
    index = index or uses_index(tree)
    owner = index["owner"]
    for u in meta.get("uses") or []:
        kind = u.get("kind")
        if kind == "start" and u.get("sha1") and chain.file_signature(Path(project) / T.START_FILE) == u["sha1"]:
            out.append("the start image")
        elif kind == "end" and tree is not None:
            for nid in dict.fromkeys(index["ends"].get(u.get("file"), [])) if u.get("file") else []:
                out.append(f"the end image of clip #{T.short_id(nid)}")
        elif kind == "ref" and u.get("file") in owner:
            out.append(f"face of @{owner[u['file']]}")
    return list(dict.fromkeys(out))


# ---- files a Generate writes
def unique_out(edits: Path, prefix: str, seed: int, k: int) -> Path:
    """The png a result is written to: edit_<time>_<seed>.png (new_ for a created image); the same seed again within
    a second gets _<k> behind it."""
    stem = f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{seed}"
    out = edits / f"{stem}.png"
    if out.exists() or out.with_suffix(".json").exists():
        out = edits / f"{stem}_{k}.png"
    return out


def keep_source(edits: Path, src: str | Path) -> Path:
    """A copy of the original in edits/sources/, so it stays with its results."""
    kept = edits / SOURCES_DIR / (f"source_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
                                  f"{Path(src).suffix.lower() or '.png'}")
    kept.parent.mkdir(exist_ok=True)
    kept.write_bytes(Path(src).read_bytes())
    return kept


def mask_path(kept: Path) -> Path:
    """Where the mask of a kept source is saved (next to it)."""
    return kept.with_name(kept.stem.replace("source_", "mask_", 1) + ".png")


def drop_unused_copies(copies: list[tuple[Path, Path | None, int]]) -> None:
    """Remove the copies of source and mask that no result refers to (copies: [kept, mask file, results made])."""
    for kept, mask_file, made in copies:
        if not made:
            for f in (kept, mask_file):
                if f is not None:
                    Path(f).unlink(missing_ok=True)


# ---- deleting bad results
def trash_result(project: Path, png: str | Path) -> Path:
    """Move a result and its record to the trash (earlier ones stay: several deletes can be undone); returns the
    trashed png."""
    png = Path(png)
    trash = Path(project) / EDITS_DIR / TRASH_DIR
    trash.mkdir(exist_ok=True)
    for f in (png, png.with_suffix(".json")):
        if f.exists():
            f.replace(trash / f.name)
    return trash / png.name


def restore_result(project: Path, trashed: Path) -> Path:
    """Bring a trashed result (and its record) back; returns its png."""
    back = Path(project) / EDITS_DIR / trashed.name
    for f in (trashed, trashed.with_suffix(".json")):
        if f.exists():
            f.replace(back.with_suffix(f.suffix))
    return back


def empty_trash(project: Path, keep_sources: bool = False) -> None:
    """Delete for good what was deleted before, and (not keep_sources) the originals no result needs any more."""
    trash = Path(project) / EDITS_DIR / TRASH_DIR
    if trash.is_dir():
        for f in trash.iterdir():
            f.unlink(missing_ok=True)
        trash.rmdir()
    if keep_sources:
        return
    sources = Path(project) / EDITS_DIR / SOURCES_DIR
    needed = {m.get(k) for _p, m in project_edits(project) for k in ("source_copy", "mask")}
    for f in sources.glob("*") if sources.is_dir() else []:
        if str(f) not in needed and f.stat().st_mtime < time.time() - SOURCE_GRACE:
            f.unlink(missing_ok=True)
