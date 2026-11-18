"""The Checker: what a clip was made from, whether it is up to date, and why not."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, cast

from .. import chain
from .consts import CONTEXT_LENGTHS, DEFAULT_FRAMES, DEFAULT_OPTIONS, DISTILLED_STEPS, TWO_PASS, checkpoint_of
from .model import Node, Tree, short_id
from .timeline import clip_length, handover_file, handover_index
from .files import _old_record, node_files
from .cast import audio_refs
from .clipcast import mix_level, sound_mixed
from .keyframes import end_image_used, keyframe_frame
from .freezing import _revert_fields, changed_inputs, freeze_set, revert
from .crops import crop_value, get_crop
from .starts import start_source
from .sources import size_source, source_rotation, video_size
from .audio_tracks import source_sound_span
from .inputs import node_settings, plain_frames
from .schema import MadeRecord, Options

logger = logging.getLogger(__name__)


def freeze_problem(project: Path, tree: Tree, nid: str) -> str | None:
    """Why nid cannot be frozen (None: it can): a clip or video of it is not made, or one no longer joins the one
    before it as it was made (then it has to be made again first, else the frozen path would jump there)."""
    trial = copy.deepcopy(tree)
    nodes = freeze_set(trial, nid)
    ck = Checker(project, trial)
    for n in nodes:
        if not ck.files_there(n):
            return f"{'video' if n.is_video else 'clip'} #{short_id(n.id)} is not made yet (or its files are missing)"
    for n in nodes:
        if not n.is_video:
            _revert_fields(trial, n.id)
        n.frozen = True
    ck = Checker(project, trial)
    for n in nodes:
        why = ck.seam_problem(n)
        if why:
            return f"clip #{short_id(n.id)}: {why}; make it again first"
    return None


def freeze(project: Path, tree: Tree, nid: str) -> list[str]:
    """Keep nid as it is made, and with it every clip before it: they count as done whatever the project's
    settings (size, speed LoRA, steps ...) say, are not made again and cannot be edited until unfrozen. Their not
    yet made edits are undone first. Returns the ids frozen now. Raises ValueError (freeze_problem)."""
    why = freeze_problem(project, tree, nid)
    if why:
        raise ValueError(why)
    out = []
    for n in freeze_set(tree, nid):
        if not n.is_video:
            revert(project, tree, n.id)
        if not n.frozen:
            n.frozen = True
            out.append(n.id)
    return out


def keyframe_problem(tree: Tree, checker: Checker, todo: list[str]) -> str | None:
    """Why a clip's keyframes cannot be used (one past its end, on its first or last frame, inside a motion guide,
    two on the same frame), or None. Checked with the length the clip will have."""
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video or not node.keyframes:
            continue
        s = checker.settings(nid) or {}
        frames = int(s.get("frames") or clip_length(tree, node))
        guide = int(s.get("guide") or 0)
        lo = guide if guide and node.parent is not None else 1
        hi = frames - 2 - (guide if guide and tree.video_child(nid) is not None else 0)
        seen = set()
        for k in node.keyframes:
            f = keyframe_frame(k["at"])
            where = f"clip #{short_id(nid)}'s keyframe at {k['at']:.2f} s"
            if f > hi:
                minus = ", minus its carry-over" if hi < frames - 2 else ""
                return (f"{where} is at or past its end ({frames / chain.FPS:.2f} s{minus}): "
                        "move it earlier or make the clip longer")
            if f < lo:
                return f"{where} is {'inside its carry-over' if lo > 1 else 'on its first frame'}: move it later"
            if f in seen:
                return f"{where} falls on the same frame as another one: move one of them"
            seen.add(f)
    return None


def video_key(node: Node, base: dict) -> dict:
    """What a video node's converted clip depends on: the source file (by size and date), the range, its rotation
    (only when turned) and the size."""
    src = node.source or {}
    key = {"file": src.get("file"), "start": src.get("start"), "end": src.get("end"),
           "width": base["width"], "height": base["height"], "size": None, "mtime_ns": None}
    if source_rotation(src):
        key["rotate"] = source_rotation(src)
    try:
        st = Path(src.get("file") or "").stat()
        key["size"], key["mtime_ns"] = st.st_size, st.st_mtime_ns
    except OSError as e:
        logger.debug("source not stat-able: %s", e)
    return key


def _same_key(made: dict | None, current: dict) -> bool:
    if not isinstance(made, dict):
        return False
    if current.get("size") is None:              # the source is gone: its clip stays valid for the same range
        skip = ("size", "mtime_ns")
        return {k: made.get(k) for k in current if k not in skip} == {k: v for k, v in current.items()
                                                                       if k not in skip}
    return made == current


class Checker:
    """Works out which nodes are done; caches file hashes and the start image size for one round of questions."""

    def __init__(self, project: Path, tree: Tree):
        self.project, self.tree = Path(project), tree
        self._status: dict[str, str] = {}
        self._remix: set[str] = set()              # done, but its sound is to be built again (remix_due)
        self._settings: dict | None = None
        self._audio: list[dict] | None = None
        self._pictures: dict[str, str | None] = {}
        self._node_settings: dict[str, dict | None] = {}

    def sha(self, path: Path | None) -> str | None:
        """SHA-1 of a frame, cached across checks by (path, size, date): a check after every edit must not read
        every clip's frame again (slow on a network share)."""
        return chain.file_signature(path) if path is not None else None

    def base(self) -> dict | None:
        if self._settings is None:
            self._settings = current_settings(self.project, self.tree.options, self.tree)
        return self._settings

    def audio_pool(self) -> list[dict]:
        """The sound references with their content hash (None: the file is missing)."""
        if self._audio is None:
            self._audio = [{"file": a["file"], "sha1": self.sha(self.project / a["file"])}
                           for a in audio_refs(self.tree)]
        return self._audio

    def picture_sha(self, rel: str) -> str | None:
        """The content hash of a face picture (None: missing), cached for this round."""
        if rel not in self._pictures:
            self._pictures[rel] = self.sha(self.project / rel)
        return self._pictures[rel]

    def settings(self, nid: str) -> dict | None:
        """The settings a node would be made with now (worked out once per node and round, like status)."""
        if nid not in self._node_settings:
            self._node_settings[nid] = self._work_out_settings(nid)
        return self._node_settings[nid]

    def _work_out_settings(self, nid: str) -> dict | None:
        base = self.base()
        node = self.tree.nodes[nid]
        span = source_sound_span(self.tree, node, self.project) \
            if not node.is_video and (node.overrides or {}).get("source_sound") else None
        if span is not None and (node.overrides or {}).get("sound_over"):
            span = {**span, "overlay": True}       # laid over the clip afterwards: the plain model, no lip sync
            if base is not None and sound_mixed(node, base.get("audio", True)):   # and mixed with the model's sound
                span.update({"mix": True}, **({"mix_level": mix_level(node)} if mix_level(node) else {}))
        if base is None:
            return None
        s = node_settings(base, self.tree.options, node, self.audio_pool(), self.picture_sha, source_sound=span,
                          frames_given=plain_frames(self.project, self.tree, node))
        if node.parent is None and not node.is_video:     # the cut of its start picture (not the centre one)
            cut = crop_value(get_crop(self.tree, self.project, start_source(self.project, node)))
            if cut:
                s["start_crop"] = cut
        return s

    def start_frame(self, node: Node) -> Path | None:
        """The frame a generated clip starts from: the parent's last frame, a first clip's start picture (its
        own, else the project's start image), or None (a first clip without one)."""
        if node.parent is None:
            return start_source(self.project, node)
        parent = self.tree.nodes[node.parent]
        if not parent.made:
            return node_files(self.project, node.parent)[1]
        return handover_file(self.project, self.tree, parent) or self.project / parent.made["frame"]

    def end_key(self, node: Node):
        """What a generated clip's end depends on: the key of the video it leads into (False: that video is not
        ready), its own end image ({"image": sha1}), or None (a free end)."""
        vc = self.tree.video_child(node.id)
        if vc is None:
            own = end_image_used(self.tree, node)
            return {"image": self.sha(self.project / own)} if own else None
        return (vc.made or {}).get("key") if self.status(vc.id) == "done" else False

    def keyframe_key(self, node: Node) -> list | None:
        """What a generated clip's middle keyframes are: [[picture sha1, frame, note], ...], None without any."""
        if node.is_video or not node.keyframes:
            return None
        return [[self.sha(self.project / k["image"]), keyframe_frame(k["at"]), k["note"]] for k in node.keyframes]

    def guide_from(self, node: Node) -> str | None:
        """The parent clip a start motion guide is cut from (its file name: every new clip gets a new one), or None
        without a start guide."""
        s = self.settings(node.id)
        if node.parent is None or not s or not int(s.get("guide") or 0):
            return None
        parent = self.tree.nodes[node.parent]
        if not parent.made:
            return None
        k = handover_index(self.tree, parent)            # an earlier hand-over: other guide frames
        v = parent.made.get("video")
        rev = parent.made.get("sound_rev")              # its sound built again in place: the guide hears the new one
        lat = "#latent" if self.latent_source(node) is not None else ""   # (from the latent: another carry-over)
        return None if v is None else v + (f"@{k}" if k is not None else "") + (f"~s{rev}" if rev else "") + lat

    def latent_source(self, node: Node) -> Path | None:
        """The latent node's carry-over is sliced from (option "latent_handover"): the clip before's saved latent,
        when it has one of this clip's size, hands over at its last frame, its sound is the model's own (none laid
        over, mixed or given: the latent holds the model's), and the carry-over is one Motion Context takes, in one
        pass, without a given sound of its own; None: the carry-over is cut from its frames."""
        if not self.tree.options.get("latent_handover") or node.parent is None:
            return None
        s = self.settings(node.id)
        if not s or int(s.get("guide") or 0) not in CONTEXT_LENGTHS or s.get("two_pass") or s.get("source_sound"):
            return None                              # (a given sound: the latent's would be pinned on top)
        parent = self.tree.nodes[node.parent]
        made = parent.made or {}
        ps = made.get("settings") or {}
        if parent.is_video or not made.get("latent") or made.get("sound_rev") or made.get("own_sound") \
                or ps.get("source_sound") \
                or handover_index(self.tree, parent) is not None \
                or (ps.get("width"), ps.get("height")) != (s.get("width"), s.get("height")):
            return None
        f = Path(self.project) / made["latent"]
        return f if f.is_file() else None

    def continue_from(self, node: Node) -> str | None:
        """The clip a continuation's reference video is cut from (its file name: every new clip gets a new one,
        and its hand-over frame, where the cut ends), or None for a clip that does not continue the one before."""
        s = self.settings(node.id)
        if node.parent is None or not s or not s.get("continue"):
            return None
        parent = self.tree.nodes[node.parent]
        v = (parent.made or {}).get("video")
        if v is None:
            return None
        k = handover_index(self.tree, parent)
        return v + (f"@{k}" if k is not None else "")

    def files_there(self, node: Node) -> bool:
        m = node.made
        if not m or not m.get("video") or not m.get("frame") or (node.is_video and not m.get("first")):
            return False
        return all((self.project / m[k]).is_file() for k in ("video", "frame", "first") if m.get(k))

    def status(self, nid: str) -> str:
        """done | changed (made, but something it depends on changed) | new (never made). A clip whose settings
        differ only in how its laid-over sound is mixed counts as done (nothing the model made changes; what follows
        it is not touched): its sound is built again from its kept own sound (remix_due, remix_sound)."""
        if nid in self._status:
            return self._status[nid]
        node = self.tree.nodes[nid]
        if not node.is_video and node.parent is not None and node.parent not in self._status:
            for above in self.tree.path_to(nid)[:-1]:   # parents first (path_to is iterative): no deep recursion
                self.status(above.id)
        m: MadeRecord = node.made or {}
        if not self.files_there(node):
            st = "new"
        elif node.frozen:                          # kept as made, whatever the settings say now
            st = "done"
        elif node.is_video:                        # a real video depends only on its file, range and the size
            base = self.base()
            st = "done" if base is not None and _same_key(m.get("key"), video_key(node, base)) else "changed"
        elif (m.get("prompt") != node.prompt or m.get("seed") != node.seed
              or (not (same := _old_record(m.get("settings")) == self.settings(nid)) and not self._sound_only(node))
              or (node.parent is not None and self.status(node.parent) != "done")
              or m.get("start_sha1") != self.sha(self.start_frame(node))
              or m.get("end_key") != self.end_key(node)
              or m.get("keyframe_key") != self.keyframe_key(node)
              or m.get("guide_from") != self.guide_from(node)
              or m.get("continue_from") != self.continue_from(node)):
            st = "changed"
        else:
            st = "done"
            if not same:                           # only its sound's mix changed
                self._remix.add(nid)
        self._status[nid] = st
        return st

    def _sound_only(self, node: Node) -> bool:
        """The made clip's settings differ from its current ones only in its laid-over sound's mix (mix on / off,
        the level) or in a new version of the sound it repeats from the clip before (sound_rev): the model heard
        none of that, so the clip need not be made again, as long as the model's own sound was kept (own_sound).
        Clips made before it was kept are made again."""
        m: MadeRecord = node.made or {}
        own = m.get("own_sound")
        if not own or not (self.project / own).is_file():
            return False
        made, now = _old_record(m.get("settings")), self.settings(node.id)
        if not isinstance(made, dict) or not isinstance(now, dict):
            return False
        a, b = made.get("source_sound"), now.get("source_sound")
        if not (isinstance(a, dict) and isinstance(b, dict) and a.get("overlay") and b.get("overlay")):
            return False
        loose = ("mix", "mix_level", "rev")

        def strip(d: dict) -> dict:
            return {k: v for k, v in d.items() if k not in loose}
        return strip(a) == strip(b) and {**made, "source_sound": None} == {**now, "source_sound": None}

    def remix_due(self, nid: str) -> bool:
        """The clip is done, but its sound has to be built again for its current sound settings (remix_sound)."""
        return self.status(nid) == "done" and nid in self._remix

    def seam_problem(self, node: Node) -> str | None:
        """Why a made generated clip no longer joins the clips around it as made (None: it does): what it starts
        from, its motion guide's source or its end changed. Its own edits are not seams (Undo changes fixes them)."""
        if node.is_video or not node.made:
            return None
        m = node.made
        if m.get("start_sha1") != self.sha(self.start_frame(node)):
            return "it no longer starts at the frame it was made from"
        if m.get("guide_from") != self.guide_from(node):
            return "the clip its carry-over comes from was made again"
        if m.get("continue_from") != self.continue_from(node):
            return "the clip it continues was made again"
        if m.get("end_key") != self.end_key(node):
            return "the video it ends at changed"
        return None

    def why(self, nid: str) -> str:
        """Why a clip counts as changed, in words ('' if it does not)."""
        if self.status(nid) != "changed":
            return ""
        node = self.tree.nodes[nid]
        m: MadeRecord = node.made or {}
        if node.is_video:
            return "its range, file or the video size changed"
        edited = changed_inputs(self.tree, nid)
        if edited:
            return f"its {' and '.join(edited)} changed"
        s = m.get("settings")
        if isinstance(s, dict) and "backend" not in s:
            return "it was made with the old generator"
        if _old_record(s) != self.settings(nid):
            was, now = (s or {}).get("source_sound") or {}, (self.settings(nid) or {}).get("source_sound") or {}
            if now.get("mode") == "previous" and was.get("rev") != now.get("rev"):
                return "the sound of the clip before it, which it repeats, was mixed again"
            return "its settings changed (the project's or its own)"
        if node.parent is not None and self.status(node.parent) != "done":
            p = self.tree.nodes[node.parent]
            return f"{'video' if p.is_video else 'clip'} #{short_id(p.id)} before it changes"
        if m.get("start_sha1") != self.sha(self.start_frame(node)):
            return "the frame it starts from changed"
        if m.get("end_key") != self.end_key(node):
            return "its end (the video after it or its end image) changed"
        if m.get("keyframe_key") != self.keyframe_key(node):
            return "its keyframes changed"
        if m.get("guide_from") != self.guide_from(node):
            return "the clip its carry-over comes from changed"
        if m.get("continue_from") != self.continue_from(node):
            return "the clip it continues changed"
        return "something it depends on changed"


def current_settings(project: Path, options: Options, tree: Tree | None = None) -> dict | None:
    """What every clip depends on besides prompt, seed and start frame (None while there is neither a start image
    nor a video to take the size from)."""
    dims = size_source(project, tree)
    if dims is None:
        return None
    w, h = video_size(options, *dims)
    ck = checkpoint_of(options)
    distilled = bool(ck and ck["distilled"])        # (its turbo built in: no speed mode on top)
    turbo = options.get("turbo") or {}
    turbo_on = bool(turbo.get("on") and turbo.get("lora")) and not distilled
    hyper = options.get("hyperflow") or {}
    hyper_on = bool(hyper.get("on")) and not turbo_on and not distilled
    pika = options.get("pika") or {}
    pika_on = bool(pika.get("on")) and not turbo_on and not hyper_on and not distilled
    style = options.get("style") or {}
    return chain.settings_dict(model=options.get("model", DEFAULT_OPTIONS["model"]), lora=None,
                               strength=1.0, frames=options.get("frames", DEFAULT_FRAMES),
                               steps=(DISTILLED_STEPS if distilled else
                                      int(turbo.get("steps", 6)) if turbo_on else chain.HYPERFLOW_STEPS if hyper_on
                                      else chain.PIKA_STEPS if pika_on else options.get("steps", 28)),
                               hyperflow=float(hyper.get("strength", 1.0)) if hyper_on else None,
                               pika=float(pika.get("strength", 1.0)) if pika_on else None,
                               width=w, height=h, audio=options.get("audio", True),
                               turbo=(turbo["lora"], turbo.get("strength", 1.0)) if turbo_on else None,
                               cfg=float(cast(dict[str, Any], ck if distilled else turbo if turbo_on else hyper
                                              if hyper_on else pika if pika_on else options).get("cfg", 1.0)),
                               checkpoint=ck,
                               two_pass=dict(TWO_PASS) if (options.get("two_pass") or {}).get("on") else None,
                               backend="comfyui",
                               style=(style.get("prefix") or "", style.get("suffix") or ""))
