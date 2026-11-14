"""End images and middle keyframes of a generated clip."""
from __future__ import annotations

import uuid
from pathlib import Path

from .. import chain
from .consts import ENDS_DIR
from .guard import inside, retire
from .model import Node, Tree, short_id


def _in_ends(project: Path, rel: str) -> bool:
    """rel points into the project's ends/ folder (a damaged/hand-edited record never reaches elsewhere)."""
    return inside(project, ENDS_DIR, rel) is not None


def set_end_image(project: Path, tree: Tree, nid: str, image: Path) -> str:
    """Copy an image in as the end image of a generated clip (a new name every time; the old copy goes)."""
    node = tree.nodes[nid]
    if node.is_video:
        raise ValueError("a real video has no end image of its own")
    if tree.video_child(nid) is not None:
        raise ValueError(f"clip #{short_id(nid)} ends at the video after it")
    d = Path(project) / ENDS_DIR
    d.mkdir(parents=True, exist_ok=True)
    rel = f"{ENDS_DIR}/{nid}_{uuid.uuid4().hex[:8]}.png"
    chain.to_png(Path(image), Path(project) / rel)
    clear_end_image(project, tree, nid)
    node.end_image = rel
    return rel


def clear_end_image(project: Path, tree: Tree, nid: str) -> None:
    """Remove a clip's end image. The copy its current clip was made with stays until the clip is made again
    (Undo changes can go back to it)."""
    node = tree.nodes[nid]
    if node.end_image and node.end_image != (node.made or {}).get("end_image"):
        retire(project, ENDS_DIR, node.end_image, resolved=False)
    node.end_image = None


def copy_end_image(project: Path, tree: Tree, src: str, dst: str) -> None:
    """A variant gets its own copy of the end image (deleting one never touches the other's)."""
    rel = tree.nodes[src].end_image
    if rel and (Path(project) / rel).is_file() and tree.video_child(dst) is None:
        set_end_image(project, tree, dst, Path(project) / rel)


def clean_keyframes(raw) -> list[dict]:
    """Keyframes as tree.json holds them: {"image": "ends/...", "at": s > 0, "note"}, sorted by time (damaged
    entries are left out)."""
    out = []
    for k in raw if isinstance(raw, list) else []:
        if not isinstance(k, dict) or not isinstance(k.get("image"), str) or not k["image"].startswith(ENDS_DIR + "/"):
            continue
        at = k.get("at")
        if isinstance(at, bool) or not isinstance(at, (int, float)) or not at > 0:
            continue
        out.append({"image": k["image"], "at": round(float(at), 3), "note": str(k.get("note") or "")})
    return sorted(out, key=lambda k: k["at"])


def keyframe_files(node: Node) -> set[str]:
    """The keyframe pictures a node and its made clip use (project-relative)."""
    return {k["image"] for k in node.keyframes} | {k["image"] for k in (node.made or {}).get("keyframes") or []
                                                    if isinstance(k, dict) and k.get("image")}


def _drop_keyframe_file(project: Path, node: Node, rel: str) -> None:
    """A keyframe picture no longer used goes to old/ (the made clip's stay until it is made again: Undo)."""
    made = {k.get("image") for k in (node.made or {}).get("keyframes") or [] if isinstance(k, dict)}
    if rel not in made and rel not in {k["image"] for k in node.keyframes}:
        retire(project, ENDS_DIR, rel, resolved=False)


def set_keyframe(project: Path, tree: Tree, nid: str, image: Path | None, at: float, note: str | None = None,
                 index: int | None = None) -> int:
    """Add a middle keyframe (index None) or change one (a new picture, time and / or note; image None keeps the
    picture). The picture is copied in (a new name every time). Returns its index after sorting by time."""
    node = tree.nodes[nid]
    if node.is_video:
        raise ValueError("a real video has no keyframes")
    if not at > 0:
        raise ValueError("a keyframe needs a time after the start")
    if index is not None and not 0 <= index < len(node.keyframes):
        raise ValueError(f"no keyframe {index}")
    old = dict(node.keyframes[index]) if index is not None else None
    rel = old["image"] if old else None
    if image is not None:
        d = Path(project) / ENDS_DIR
        d.mkdir(parents=True, exist_ok=True)
        rel = f"{ENDS_DIR}/{nid}_k{uuid.uuid4().hex[:8]}.png"
        chain.to_png(Path(image), Path(project) / rel)
    if rel is None:
        raise ValueError("a new keyframe needs a picture")
    entry = {"image": rel, "at": round(float(at), 3), "note": (old or {}).get("note", "") if note is None else note}
    kfs = [k for i, k in enumerate(node.keyframes) if i != index] + [entry]
    node.keyframes = sorted(kfs, key=lambda k: k["at"])
    if old and old["image"] != rel:
        _drop_keyframe_file(project, node, old["image"])
    return node.keyframes.index(entry)


def remove_keyframe(project: Path, tree: Tree, nid: str, index: int) -> None:
    node = tree.nodes[nid]
    if not 0 <= index < len(node.keyframes):
        raise ValueError(f"no keyframe {index}")
    gone = node.keyframes.pop(index)
    _drop_keyframe_file(project, node, gone["image"])


def copy_keyframes(project: Path, tree: Tree, src: str, dst: str) -> None:
    """A variant gets its own copies of the keyframe pictures."""
    for k in list(tree.nodes[src].keyframes):
        if (Path(project) / k["image"]).is_file():
            set_keyframe(project, tree, dst, Path(project) / k["image"], k["at"], k["note"])


def keyframe_frame(at: float) -> int:
    return int(round(float(at) * chain.FPS))


def missing_keyframes(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    return [nid for nid in todo if any(not (Path(project) / k["image"]).is_file() for k in tree.nodes[nid].keyframes)]


def end_image_used(tree: Tree, node: Node) -> str | None:
    """The end image a generated clip is made with: its own, unless a video follows (that one's first frame)."""
    if node.is_video or not node.end_image or tree.video_child(node.id) is not None:
        return None
    return node.end_image


def missing_end_images(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    return [nid for nid in todo if end_image_used(tree, tree.nodes[nid])
            and not (Path(project) / tree.nodes[nid].end_image).is_file()]
