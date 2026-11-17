"""Changed inputs, revert and freeze (a clip kept as it is although its inputs changed)."""
from __future__ import annotations

import copy
from pathlib import Path

from .model import Node, Tree
from .keyframes import _drop_keyframe_file, clean_keyframes, clear_end_image


def changed_inputs(tree: Tree, nid: str) -> list[str]:
    """What was edited on a made generated clip since it was made: prompt, seed, settings (its own),
    end image. Only what the record can restore (older records keep no own settings / end image)."""
    node = tree.nodes.get(nid)
    m = node.made if node is not None else None
    if node is None or node.is_video or not m:
        return []
    out = []
    if m.get("prompt") != node.prompt:
        out.append("prompt")
    if m.get("seed") != node.seed:
        out.append("seed")
    if "overrides" in m and m["overrides"] != (node.overrides or {}):
        out.append("settings")
    if "end_image" in m and m["end_image"] != node.end_image:
        out.append("end image")
    if "keyframes" in m and clean_keyframes(m["keyframes"]) != node.keyframes:
        out.append("keyframes")
    return out


def revert(project: Path, tree: Tree, nid: str) -> list[str]:
    """Undo the edits of a made generated clip: back to the prompt, seed, own settings and end image it was
    made with. Returns what was restored."""
    node = tree.nodes[nid]
    newer = {k["image"] for k in node.keyframes}
    if "end image" in changed_inputs(tree, nid):
        clear_end_image(project, tree, nid)       # the newer image goes (unless it is the made one)
    done = _revert_fields(tree, nid)
    if "keyframes" in done:
        for rel in newer:
            _drop_keyframe_file(project, node, rel)
    return done


def _revert_fields(tree: Tree, nid: str) -> list[str]:
    """Undo the edits of a made clip in the tree only (no files touched; revert adds those): also for trying out
    what freezing finds. Returns what was restored."""
    node, done = tree.nodes[nid], changed_inputs(tree, nid)
    m = node.made
    assert m is not None                              # (only a made clip has edits to undo)
    if "prompt" in done:
        node.prompt = m["prompt"]
    if "seed" in done:
        node.seed = int(m["seed"])
    if "settings" in done:
        node.overrides = copy.deepcopy(m["overrides"])
    if "end image" in done:
        node.end_image = m["end_image"]
    if "keyframes" in done:
        node.keyframes = clean_keyframes(m["keyframes"])
    return done


def freeze_set(tree: Tree, nid: str) -> list[Node]:
    """What freezing nid freezes: it and every clip / video before it (so every seam up to it stays as made), and
    the real video each of those generated clips leads into (the clip ends at its first frame)."""
    out = tree.path_to(nid)
    for n in list(out):
        vc = tree.video_child(n.id)
        if vc is not None and vc not in out:
            out.append(vc)
    return out


def freeze_edits(tree: Tree, nid: str) -> dict[str, list[str]]:
    """The not yet made edits freezing nid undoes: {clip id: ["prompt", ...]}."""
    return {n.id: e for n in freeze_set(tree, nid) if (e := changed_inputs(tree, n.id))}


def unfreeze(tree: Tree, nid: str) -> list[str]:
    """nid follows the settings again, and so does every frozen clip after it (a frozen clip needs the ones before
    it frozen) and a frozen clip that ends at it (a video). Returns the ids unfrozen."""
    node = tree.nodes[nid]
    if not node.frozen:
        return []
    node.frozen = False
    return [nid] + heal_frozen(tree)


def heal_frozen(tree: Tree) -> list[str]:
    """Unfreeze what may not stay frozen: a clip after one that is not frozen, a generated clip whose video after it
    is not frozen, one that was never made. Returns those ids."""
    out, again = [], True
    while again:
        again = False
        for n in tree.nodes.values():
            if not n.frozen:
                continue
            p = tree.nodes.get(n.parent) if n.parent is not None else None
            vc = tree.video_child(n.id)
            if not isinstance(n.made, dict) or (p is not None and not p.frozen) or (vc is not None and not vc.frozen):
                n.frozen = False
                out.append(n.id)
                again = True
    return out


def frozen_note(tree: Tree, node: Node, base: dict | None) -> str:
    """How a frozen clip differs from what the settings would make now, in a few words ('' when it does not, or
    the record does not say)."""
    if base is None or not node.made:
        return ""
    if node.is_video:
        k = node.made.get("key") or {}
        same = (k.get("width"), k.get("height")) == (base.get("width"), base.get("height"))
        return "" if same or not k.get("width") else f"made at {k['width']}×{k['height']}"
    s = node.made.get("settings") or {}
    parts = []
    if s.get("width") and (s.get("width"), s.get("height")) != (base.get("width"), base.get("height")):
        parts.append(f"{s['width']}×{s['height']}")

    def speed(d: dict) -> str:
        if d.get("hyperflow"):
            return "HyperFlow"
        if d.get("pika"):
            return "Pika"
        return "Turbo" if d.get("turbo") else "no speed LoRA"
    now = {"turbo": base.get("turbo"), "hyperflow": base.get("hyperflow"), "pika": base.get("pika")}
    if speed(s) != speed(now):
        parts.append(speed(s))
    return ("made at " + ", ".join(parts)) if parts else ""
