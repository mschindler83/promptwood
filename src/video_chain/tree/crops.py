"""Which part of a start picture a clip starts from, when the picture's shape is not the clips' shape.

H3 stretches a start picture onto the canvas, so a picture of another shape is cut to the canvas's shape first
(chain.prepare_image). Without a crop of its own that cut is taken from the centre at the largest size that fits.
The user can move and shrink it: tree.options["crops"] = {picture: {"cx", "cy", "zoom", "seen"}} with
  picture  the start picture, relative to the project (START_FILE, or a first clip's own in starts/),
  cx, cy   the centre of the cut, as a share of the picture's width / height (0.5, 0.5: the centre),
  zoom     its size as a share of the largest cut of the canvas's shape (1: the largest),
  seen     the canvas shape (shape_key) the user last looked at this cut for: a picture whose shape does not match
           is shown before its clips are made until it is seen for the clips' shape.
Centre and zoom do not depend on the canvas's shape, so a cut stays where it was when the shape changes (it is
then shown again). A clip records its cut in its settings ("start_crop") unless it is the centre one, so clips made
before cuts existed stay done, and a moved cut makes the clip changed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import Tree

MIN_ZOOM = 0.1
MISMATCH = 0.05                   # shapes that differ by more than this are cut visibly (the canvas sides are
                                  # multiples of 32, so a picture of its own shape differs by a few per cent)


def shape_key(width: int, height: int) -> str:
    """The canvas shape a cut was looked at for (sizes of one shape at other resolutions round to the same)."""
    return f"{int(width) / int(height):.2f}"


def mismatched(pic_w: int, pic_h: int, width: int, height: int) -> bool:
    """Whether a picture of pic_w x pic_h is cut to fit a canvas of width x height."""
    if min(pic_w, pic_h, width, height) <= 0:
        return False
    canvas = width / height
    return abs(pic_w / pic_h - canvas) > MISMATCH * canvas


def clean(crop: Any) -> dict | None:
    """A stored cut with its values in range (None: none, or damaged)."""
    if not isinstance(crop, dict):
        return None
    try:
        cx, cy, zoom = float(crop.get("cx", 0.5)), float(crop.get("cy", 0.5)), float(crop.get("zoom", 1.0))
    except (TypeError, ValueError):
        return None
    out: dict[str, Any] = {"cx": min(1.0, max(0.0, cx)), "cy": min(1.0, max(0.0, cy)),
                           "zoom": min(1.0, max(MIN_ZOOM, zoom))}
    if isinstance(crop.get("seen"), str):
        out["seen"] = crop["seen"]
    return out


def is_centre(crop: dict | None) -> bool:
    """The cut the app makes without one: the largest, at the centre."""
    c = clean(crop)
    return c is None or (abs(c["cx"] - 0.5) < 1e-4 and abs(c["cy"] - 0.5) < 1e-4 and c["zoom"] > 1 - 1e-4)


def crop_value(crop: dict | None) -> list[float] | None:
    """What a clip records of its cut ([cx, cy, zoom]); None for the centre cut (recorded as before: nothing)."""
    c = clean(crop)
    if c is None or is_centre(c):
        return None
    return [round(c["cx"], 4), round(c["cy"], 4), round(c["zoom"], 4)]


def crop_box(pic_w: int, pic_h: int, width: int, height: int, crop: Any = None) -> tuple[int, int, int, int]:
    """The cut (x, y, w, h in the picture's pixels) for a canvas of width x height. crop: a stored cut or its
    recorded value [cx, cy, zoom]; None: the centre cut."""
    if isinstance(crop, (list, tuple)) and len(crop) == 3:
        crop = {"cx": crop[0], "cy": crop[1], "zoom": crop[2]}
    c = clean(crop) or {"cx": 0.5, "cy": 0.5, "zoom": 1.0}
    r = int(width) / int(height)
    big_w = min(float(pic_w), pic_h * r)
    w = max(1.0, big_w * c["zoom"])
    h = max(1.0, w / r)
    if h > pic_h:                                  # (rounding at the largest size)
        h, w = float(pic_h), pic_h * r
    x = min(max(0.0, c["cx"] * pic_w - w / 2), pic_w - w)
    y = min(max(0.0, c["cy"] * pic_h - h / 2), pic_h - h)
    wi, hi = max(1, round(w)), max(1, round(h))
    return min(round(x), pic_w - wi), min(round(y), pic_h - hi), wi, hi


def picture_key(project: Path, picture: Path) -> str | None:
    """How a start picture is named in options["crops"] (relative to the project); None outside it."""
    try:
        return Path(picture).resolve().relative_to(Path(project).resolve()).as_posix()
    except ValueError:
        return None


def get_crop(tree: Tree, project: Path, picture: Path | None) -> dict | None:
    if picture is None:
        return None
    key = picture_key(project, picture)
    return clean((tree.options.get("crops") or {}).get(key)) if key else None


def set_crop(tree: Tree, project: Path, picture: Path, cx: float, cy: float, zoom: float,
             seen: str | None = None) -> None:
    """Store a picture's cut (seen: the canvas shape it was looked at for)."""
    key = picture_key(project, picture)
    if key is None:
        return
    crops = dict(tree.options.get("crops") or {})
    crops[key] = clean({"cx": cx, "cy": cy, "zoom": zoom, "seen": seen}) or {}
    tree.options["crops"] = crops


def drop_crop(tree: Tree, rel: str | None) -> None:
    """A start picture is replaced or gone: its cut goes with it (a new picture starts at the centre)."""
    crops = tree.options.get("crops")
    if rel and isinstance(crops, dict) and rel in crops:
        crops = dict(crops)
        crops.pop(rel)
        if crops:
            tree.options["crops"] = crops
        else:
            tree.options.pop("crops", None)


def copy_crop(tree: Tree, src_rel: str | None, dst_rel: str) -> None:
    crops = tree.options.get("crops")
    if src_rel and isinstance(crops, dict) and src_rel in crops:
        tree.options["crops"] = {**crops, dst_rel: dict(crops[src_rel])}
