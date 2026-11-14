"""Real videos as clip sources: the source record, probing, and the video size the project follows."""
from __future__ import annotations

from pathlib import Path

from .. import chain
from .consts import START_FILE
from .schema import Source
from .model import Tree


def video_source(file: str, start: float, end: float, rotate: int = 0) -> Source:
    """A video node's source: the file, the range (seconds) and, when it is turned, "rotate" (clockwise degrees;
    left out at 0, so videos without it keep the records and keys they had)."""
    src: Source = {"file": str(file), "start": round(float(start), 4), "end": round(float(end), 4)}
    if chain.rotation(rotate):
        src["rotate"] = chain.rotation(rotate)
    return src


def source_rotation(src: Source | None) -> int:
    return chain.rotation(src.get("rotate", 0) if src else 0)


_SRC_INFO: dict[tuple, dict] = {}


_IMG_SIZE: dict[tuple, tuple[int, int]] = {}


def source_info(file: str | Path) -> dict | None:
    """chain.probe_source of a real video, cached by (path, size, date); None if the file is not there."""
    try:
        st = Path(file).stat()
    except OSError:
        return None
    key = (str(file), st.st_size, st.st_mtime_ns)
    if key not in _SRC_INFO:
        _SRC_INFO[key] = chain.probe_source(Path(file))
    return _SRC_INFO[key]


ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5), "5:4": (5, 4), "3:2": (3, 2),
           "2:3": (2, 3), "21:9": (21, 9)}


def size_origin(project: Path, tree: Tree | None) -> str:
    """What the video size follows: 'image' (the start image), 'video' (the first video) or 'aspect' (the project's
    aspect setting: a project without either)."""
    if (Path(project) / START_FILE).is_file():
        return "image"
    if tree is not None and any(n.is_video for n in tree.nodes.values()):
        return "video"
    return "aspect"


def size_source(project: Path, tree: Tree | None) -> tuple[int, int] | None:
    """The picture whose aspect ratio the video size follows: the start image, else the first video of the tree
    (by number), else the project's aspect setting (ASPECTS; a first clip then starts from its text). None only
    without a tree and a start image."""
    start = Path(project) / START_FILE
    try:
        st = start.stat()
    except OSError:
        st = None
    if st is not None:
        key = (str(start), st.st_size, st.st_mtime_ns)
        if key not in _IMG_SIZE:                     # ffprobe once per version of the image, not per check
            _IMG_SIZE[key] = chain.image_size(start)
        return _IMG_SIZE[key]
    if tree is not None:
        for n in sorted(tree.nodes.values(), key=lambda n: tree.number(n.id)):
            if not n.is_video or n.source is None:
                continue
            info = source_info(n.source["file"])
            if info:                                 # (as the video is turned)
                return chain.rotated_size(info["width"], info["height"], source_rotation(n.source))
            made_key = (n.made or {}).get("key") or {}
            if made_key.get("width") and made_key.get("height"):     # the file is gone, its converted clip is still here
                return int(made_key["width"]), int(made_key["height"])
        a = ASPECTS.get(str(tree.options.get("aspect") or "16:9"), ASPECTS["16:9"])
        return a[0] * 120, a[1] * 120
    return None
