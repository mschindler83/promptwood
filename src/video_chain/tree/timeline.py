"""Where a clip is in the film: lengths, hand-over frames, the prompt with the project's style, path cuts."""
from __future__ import annotations

from pathlib import Path

from .. import chain
from .consts import DEFAULT_FRAMES
from .model import Node, Tree
from .schema import Options


def clip_length(tree: Tree, node: Node) -> int:
    """Frames of a node's clip: a video's range (or what was converted), a generated clip's length setting."""
    if node.is_video:
        if node.made and node.made.get("frames"):
            return int(node.made["frames"])
        return chain.range_frames(node.source["start"], node.source["end"])
    return int((node.overrides or {}).get("frames", tree.options.get("frames", DEFAULT_FRAMES)))


def made_frames(node: Node) -> int | None:
    """Frames of a node's made clip (from its record), or None."""
    if not node.made:
        return None
    if node.made.get("frames"):
        return int(node.made["frames"])
    if node.is_video:
        return None
    return int(((node.made.get("settings") or {}).get("frames")) or 0) or None   # older records


def handover_index(tree: Tree, node: Node, frames: int | None = None) -> int | None:
    """The frame (0-based) the clips after a generated clip start from, when it is not its last frame (end_at);
    None: the last frame. Not for a clip that leads into a video (its end is that video's start). frames: the clip's
    length (default: the made clip's, else its length setting)."""
    if node.is_video or node.end_at is None or tree.video_child(node.id) is not None:
        return None
    n = frames or made_frames(node) or clip_length(tree, node)
    if node.end_at >= n - 1:
        return None
    return max(1, int(node.end_at))


def handover_file(project: Path, tree: Tree, node: Node) -> Path | None:
    """The PNG of a made clip's hand-over frame (named after its clip version), or None (the last frame counts)."""
    k = handover_index(tree, node)
    if k is None or not node.made or not node.made.get("video"):
        return None
    v = Path(node.made["video"])
    return Path(project) / v.parent / f"{v.stem}_f{k}.png"


def full_prompt(options: Options, prompt: str) -> str:
    """A clip's prompt as the model gets it: the project's style text before and after it."""
    style = options.get("style") or {}
    return " ".join(x for x in ((style.get("prefix") or "").strip(), prompt.strip(),
                                (style.get("suffix") or "").strip()) if x)


def _guide_of(node: Node, planned: bool = False) -> int:
    if planned:
        return int((node.overrides or {}).get("guide") or 0)
    return int((((node.made or {}).get("settings")) or {}).get("guide") or 0)


def path_cuts(tree: Tree, path: list[Node], keep_repeated: bool = False,
              planned: bool = False) -> list[tuple[int, int]]:
    """Frames left out of each clip of a path when it is joined, as (at the start, at the end). Where a generated
    clip continues from the clip before, its first frame (or its motion guide frames) repeats that clip's end;
    where it leads into a video, its last frame (or guide frames) repeats the video's start. Two videos in a row
    and a first clip lose nothing. The repeated frames of generated clips are left out, real ones are kept.
    A clip that hands over at an earlier frame (end_at) loses the frames after it.
    planned: the motion guides as set now (for display), not as the clips were made."""
    cuts = [[0, 0] for _ in path]
    for k, a in enumerate(path):                   # an earlier hand-over: the frames after it are left out
        n = (clip_length(tree, a) if planned else made_frames(a)) or clip_length(tree, a)
        h = handover_index(tree, a, n)
        if h is not None:
            cuts[k][1] = n - 1 - h
    if not keep_repeated:
        for k in range(1, len(path)):
            a, b = path[k - 1], path[k]
            if not b.is_video:
                cuts[k][0] = max(1, _guide_of(b, planned))
            elif not a.is_video and tree.video_child(a.id) is b:
                cuts[k - 1][1] = max(1, _guide_of(a, planned))
    for k, a in enumerate(path):                   # a short clip keeps at least one frame (carry-over + hand-over)
        n = (clip_length(tree, a) if planned else made_frames(a)) or clip_length(tree, a)
        if cuts[k][0] + cuts[k][1] >= n:
            cuts[k][1] = max(0, n - 1 - cuts[k][0])
            cuts[k][0] = min(cuts[k][0], max(0, n - 1))
    return [(s, e) for s, e in cuts]


def path_spans(tree: Tree, nid: str) -> list[tuple[str, float, float]]:
    """Where each clip of the path to nid is in the joined video (join_path): (clip id, from, to) in seconds."""
    path = tree.path_to(nid)
    cuts = path_cuts(tree, path, bool(tree.options.get("keep_repeated")))
    out, at = [], 0
    for n, (a, b) in zip(path, cuts):
        frames = max(1, (made_frames(n) or clip_length(tree, n)) - a - b)
        out.append((n.id, at / chain.FPS, (at + frames) / chain.FPS))
        at += frames
    return out


SWAPPED_REFINE_PROMPT = "A realistic video of a person, natural skin texture, sharp detailed face."
