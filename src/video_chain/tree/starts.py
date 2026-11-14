"""Start pictures: the project's and a first clip's own."""
from __future__ import annotations

import copy
import json
import logging
import uuid
from pathlib import Path

from .. import chain
from .consts import STARTS_DIR, START_FILE
from .guard import inside
from .model import Node, Tree
from .files import _unlink

logger = logging.getLogger(__name__)


def _in_starts(project: Path, rel: str) -> bool:
    """rel points into the project's starts/ folder (a damaged/hand-edited record never reaches elsewhere)."""
    return inside(project, STARTS_DIR, rel) is not None


def _drop_start_file(project: Path, rel: str | None) -> None:
    """A first clip's own start picture that is no longer used goes to old/, its prepared copies are removed."""
    if not rel or not _in_starts(project, rel):
        return
    src = Path(project) / rel
    for f in src.parent.glob(f"{src.stem}_*x*.png"):
        _unlink(f)
        _unlink(f.with_suffix(".json"))
    chain.keep_aside(project, src)


def start_source(project: Path, node: Node) -> Path | None:
    """The picture a first clip starts from: its own start picture, else the project's start image; None when
    there is neither (it starts from its text)."""
    if node.start_image:
        return Path(project) / node.start_image
    start = Path(project) / START_FILE
    return start if start.is_file() else None


def set_node_start(project: Path, tree: Tree, nid: str, image: Path) -> str:
    """Copy a picture in as a first clip's own start picture (a new name every time; the copy before goes to
    old/). The clip then starts from it instead of the project's start image."""
    node = tree.nodes[nid]
    if node.is_video or node.parent is not None:
        raise ValueError("only a first generated clip has a start picture of its own")
    d = Path(project) / STARTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    rel = f"{STARTS_DIR}/{nid}_{uuid.uuid4().hex[:8]}.png"
    chain.to_png(Path(image), Path(project) / rel)
    _drop_start_file(project, node.start_image)
    node.start_image = rel
    return rel


def clear_node_start(project: Path, tree: Tree, nid: str) -> None:
    """A first clip starts from the project's start image again (its own picture goes to old/)."""
    node = tree.nodes[nid]
    _drop_start_file(project, node.start_image)
    node.start_image = None


def copy_node_start(project: Path, tree: Tree, src: str, dst: str) -> None:
    """A variant of a first clip with its own start picture gets its own copy of that picture."""
    rel = tree.nodes[src].start_image
    target = tree.nodes[dst]
    if rel and (Path(project) / rel).is_file() and target.parent is None and not target.is_video:
        set_node_start(project, tree, dst, Path(project) / rel)
        target.row = tree.nodes[src].row


def row_members(tree: Tree, nid: str) -> list[Node]:
    """The first clips made as one set with nid (the same clip for several start pictures), nid among them, by
    number; [] when nid is in no such set."""
    row = tree.nodes[nid].row if nid in tree.nodes else None
    if not row:
        return []
    return sorted((n for n in tree.nodes.values() if n.row == row and n.parent is None),
                  key=lambda n: tree.number(n.id))


def add_start_rows(project: Path, tree: Tree, nid: str, images: list[Path]) -> list[Node]:
    """The same first clip for more start pictures: for each picture a new first clip with nid's prompt, seed and
    own settings, starting from that picture (copied into starts/). nid and the new clips form a set (Node.row).
    A picture that cannot be read is left out: -> the new clips, in the pictures' order. Raises ValueError when
    nid is not a first generated clip."""
    src = tree.nodes[nid]
    if src.is_video or src.parent is not None:
        raise ValueError("more start pictures can be given to a first generated clip only")
    out = []
    for image in images:
        new = tree.add(None, src.prompt, src.seed)
        try:
            set_node_start(project, tree, new.id, Path(image))
        except (OSError, RuntimeError, ValueError):
            del tree.nodes[new.id]
            continue
        new.overrides = copy.deepcopy(src.overrides)
        out.append(new)
    if out:
        src.row = src.row or src.id
        for new in out:
            new.row = src.row
    return out


def apply_to_row(tree: Tree, nid: str) -> list[str]:
    """Give nid's prompt and own settings (LoRAs, length, steps …) to the other clips of its set; each keeps its
    start picture, seed and end image. A frozen clip is left as it is. -> the ids that changed."""
    src = tree.nodes[nid]
    changed = []
    for n in row_members(tree, nid):
        if n.id == nid or n.frozen or (n.prompt == src.prompt and n.overrides == src.overrides):
            continue
        n.prompt, n.overrides = src.prompt, copy.deepcopy(src.overrides)
        changed.append(n.id)
    return changed


def prepared_start(project: Path, width: int, height: int, source: Path | None = None) -> Path | None:
    """A start picture prepared for a canvas (chain.prepare_image): the project's start image as
    start_<w>x<h>.png in the project, a first clip's own one (source, in starts/) as <its name>_<w>x<h>.png next
    to it; made again when the picture changes. None without a start picture."""
    src = Path(source) if source is not None else Path(project) / START_FILE
    if not src.is_file():
        return None
    main = src.resolve() == (Path(project) / START_FILE).resolve()
    stem = "start" if main else src.stem
    dst = src.parent / f"{stem}_{int(width)}x{int(height)}.png"
    rec = dst.with_suffix(".json")
    sig = chain.file_signature(src)
    try:
        if dst.is_file() and json.loads(rec.read_text(encoding="utf-8")).get("source") == sig:
            return dst
    except (OSError, ValueError) as e:
        logger.debug("cached record unreadable: %s", e)
    for old in src.parent.glob(f"{stem}_*x*.png"):         # (one prepared copy at a time)
        if old != dst:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    chain.prepare_image(src, dst, width, height)
    chain.atomic_write_text(rec, json.dumps({"source": sig}))
    return dst


def clear_start_image(project: Path, tree: Tree) -> None:
    """No start image any more: start.png goes to old/ (never deleted), its prepared copies (scaled from it) are
    removed (the first clip then starts from its text, the size follows the first video or the Aspect setting)."""
    chain.keep_aside(project, Path(project) / START_FILE)
    for f in Path(project).glob("start_*x*.png"):
        _unlink(f)
        _unlink(f.with_suffix(".json"))
    tree.image = None


def set_start_image(project: Path, image: Path, tree: Tree) -> None:
    start = Path(project) / START_FILE
    if start.is_file() and Path(image).resolve() != start.resolve():
        new = start.with_name("start.new.part.png")     # (converted first: a picture that cannot be read changes nothing)
        try:
            chain.to_png(Path(image), new)
            chain.keep_aside(project, start)        # the start image before: never deleted, moved to old/
            new.replace(start)
        finally:
            new.unlink(missing_ok=True)
    else:
        chain.to_png(Path(image), start)
    tree.image = str(Path(image).resolve())
