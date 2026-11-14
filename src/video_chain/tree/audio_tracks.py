"""Sound of a clip: audio tracks, the sound taken from a real video, the soundtrack prompt sentences."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from .. import chain
from .consts import AUDIO_REF_MIN, AUDIO_REF_TOTAL, MAX_AUDIO_REFS, REFS_DIR
from .model import Node, Tree, short_id
from .schema import Track
from .timeline import _guide_of, clip_length, full_prompt, handover_index, made_frames, path_cuts
from .files import load
from .cast import audio_mentioned, audio_refs, cast_of
from .clipcast import node_cast

logger = logging.getLogger(__name__)


REPEAT_MIN = 0.5                    # s: a shorter repeated sound would only buzz


TRACK_PREFIX = "track:"             # a clip's source_sound override "track:<id>": that audio track


TRACK_SOUND_TEXT = ("<Audio {k}> is the complete audio track of the target video, reused 1:1: every word, voice "
                    "and sound comes from <Audio {k}>, and anyone speaking moves their lips in sync with it.")


REPEAT_SOUND_TEXT = ("<Audio {k}> is the complete audio track of the target video, reused 1:1: every sound of the "
                     "video comes from <Audio {k}>, and what is seen matches it.")


SOURCE_SOUND_TEXT = ("<Audio {k}> is the complete original audio track of the target video, reused 1:1: every word, "
                     "voice and sound comes from <Audio {k}>, and the speaking person's lips move in sync with it.")


def uses_ref_model(settings: dict) -> bool:
    """Whether a clip is made with H3's reference model: face pictures, sounds or voices, or the real video's sound
    (all go to MiniMaxH3ReferenceToVideo)."""
    ss = settings.get("source_sound")
    return bool(settings.get("refs") or settings.get("audio_refs")
                or (ss and not (isinstance(ss, dict) and ss.get("overlay"))))


def source_video(tree: Tree, node: Node) -> Node | None:
    """The real video a generated clip continues (the nearest one before it on its path), or None."""
    path = tree.path_to(node.id)
    return next((n for n in reversed(path[:-1]) if n.is_video), None)


def source_sound_span(tree: Tree, node: Node, project: Path | None = None) -> dict | None:
    """Where in its real video's file a clip's sound continues ("Continue the video's sound"): the part right after
    what the path has shown so far. {"file", "start", "secs", "video"} (seconds in the file), or None without a real
    video before the clip. A generated clip's first frame (or its carry-over frames) repeats the end of the clip
    before, so its sound starts that much earlier; a clip that hands over early counts up to that frame.
    With "Repeat the sound of the clip before" (source_sound == "previous") see previous_sound_span."""
    mode = (node.overrides or {}).get("source_sound")
    if mode == "previous":
        return previous_sound_span(tree, node, project)
    if isinstance(mode, str) and mode.startswith(TRACK_PREFIX):
        return track_sound_span(tree, node)
    path = tree.path_to(node.id)
    vi = max((i for i, n in enumerate(path[:-1]) if n.is_video), default=None)
    if vi is None:
        return None
    v = path[vi]
    src = v.source or {}
    t = float(src["start"]) + clip_length(tree, v) / chain.FPS   # the frames the video really has (not the end)
    for p in path[vi + 1:-1]:
        n = made_frames(p) or clip_length(tree, p)
        h = handover_index(tree, p, n)
        t += ((h + 1 if h is not None else n) - _join_lead(tree, p)) / chain.FPS
    start = t - _join_lead(tree, node) / chain.FPS
    return {"file": str(src["file"]), "start": round(max(0.0, start), 3),
            "secs": round(clip_length(tree, node) / chain.FPS, 3), "video": v.id}


def _join_lead(tree: Tree, node: Node) -> int:
    """Frames of a generated clip (after another clip) left out at its start when the path is joined: its first
    frame or its carry-over frames repeat the clip before; none with "keep the repeated frames"."""
    return 0 if tree.options.get("keep_repeated") else max(1, _guide_of(node, planned=True))


# ---- audio tracks: imported sound (cut to a range), played by clips one after another
def tracks(tree: Tree) -> list[Track]:
    """The project's audio tracks ({"id", "name", "file", "secs", "source", "start", "end"}); shared by the film's
    scenes. file: the cut sound as WAV in refs/ (a new name for every cut)."""
    return [t for t in (tree.options.get("tracks") or []) if isinstance(t, dict) and t.get("id") and t.get("file")]


def track_of(tree: Tree, tid: str) -> dict | None:
    return next((t for t in tracks(tree) if t["id"] == tid), None)


def track_id(node: Node) -> str | None:
    """The audio track a clip plays (its Sound setting), or None."""
    mode = (node.overrides or {}).get("source_sound")
    return mode[len(TRACK_PREFIX):] if isinstance(mode, str) and mode.startswith(TRACK_PREFIX) else None


def _cut_track(project: Path, source: Path, start: float, end: float | None) -> tuple[str, float]:
    length = chain.audio_seconds(Path(source))
    if length is None:
        raise ValueError(f"{Path(source).name} has no sound")
    start = max(0.0, float(start))
    end = length if end is None else min(float(end), length)
    if end - start < AUDIO_REF_MIN:
        raise ValueError(f"an audio track needs at least {AUDIO_REF_MIN:.0f} s (the range has {max(0, end - start):.1f} s)")
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    rel = f"{REFS_DIR}/track_{uuid.uuid4().hex[:8]}.wav"
    chain.to_wav(Path(source), Path(project) / rel, start, end - start)
    return rel, round(chain.audio_seconds(Path(project) / rel) or (end - start), 3)


def add_track(project: Path, tree: Tree, source: Path, start: float = 0.0, end: float | None = None,
              name: str | None = None) -> dict:
    """Cut the sound of an audio or video file (start-end, seconds) into refs/ as a new audio track."""
    rel, secs = _cut_track(project, source, start, end)
    have = {t["id"] for t in tracks(tree)}
    tid = "t" + uuid.uuid4().hex[:6]               # never reused: a removed track's clips keep pointing nowhere
    while tid in have:
        tid = "t" + uuid.uuid4().hex[:6]
    entry = {"id": tid, "name": name or Path(source).stem, "file": rel, "secs": secs, "source": str(source),
             "start": round(float(start), 3), "end": round(float(start) + secs, 3)}
    tree.options["tracks"] = tracks(tree) + [entry]
    return entry


def recut_track(project: Path, tree: Tree, tid: str, source: Path, start: float, end: float | None,
                name: str | None = None) -> dict:
    """Cut an audio track again (another range, or another file); the clips playing it count as changed."""
    old = track_of(tree, tid)
    if old is None:
        raise ValueError("this audio track is gone")
    rel, secs = _cut_track(project, source, start, end)
    entry = {**old, "name": name or old["name"], "file": rel, "secs": secs, "source": str(source),
             "start": round(float(start), 3), "end": round(float(start) + secs, 3)}
    tree.options["tracks"] = [entry if t["id"] == tid else t for t in tracks(tree)]
    return entry


def remove_track(project: Path, tree: Tree, tid: str) -> None:
    """Take a track out of the list; its file stays until no scene lists it (unused_track_files)."""
    tree.options["tracks"] = [t for t in tracks(tree) if t["id"] != tid]


def unused_track_files(project: Path, scene_dirs: list[Path], open_tree: Tree | None = None) -> list[Path]:
    """Cut track files in refs/ that the saved tree of no scene lists any more (a track cut again or removed), nor
    the open tree. A scene whose tree cannot be read keeps everything."""
    listed = {tr["file"] for tr in tracks(open_tree)} if open_tree is not None else set()
    for d in scene_dirs:
        try:
            t = load(Path(d))
        except Exception as e:  # noqa: BLE001
            logger.warning("a scene could not be loaded: %s", e)
            return []
        listed |= {tr["file"] for tr in tracks(t)}
    refs = Path(project) / REFS_DIR
    return sorted(f for f in refs.glob("track_*.wav") if f"{REFS_DIR}/{f.name}" not in listed) \
        if refs.is_dir() else []


def track_users(tree: Tree, tid: str) -> list[str]:
    return [n.id for n in tree.nodes.values() if not n.is_video and track_id(n) == tid]


def track_sound_span(tree: Tree, node: Node) -> dict | None:
    """Which part of its audio track a clip plays: the track goes on where the clip before stopped when that clip
    plays the same track, else it starts at the track's beginning. Laid so the joined path hears it without a gap
    or a repeat: a clip's first frame (or carry-over frames) is left out when joined, so its sound starts that much
    earlier; the first clip of a run gets that much silence first (pad). Past the track's end: silence.
    {"file", "start", "secs", "pad", "mode": "track", "track", "from"} (from: the clip it goes on from, or None);
    None when the track is gone."""
    tid = track_id(node)
    track = track_of(tree, tid) if tid else None
    if track is None:
        return None
    path = tree.path_to(node.id)

    def lead(k: int) -> int:                       # frames of path[k] left out at its start when joined
        return 0 if k == 0 else _join_lead(tree, path[k])
    i = len(path) - 1
    while i > 0 and not path[i - 1].is_video and track_id(path[i - 1]) == tid:
        i -= 1                                     # the run of clips playing this track, up to this one
    t = 0
    for k in range(i, len(path) - 1):
        p = path[k]
        n = made_frames(p) or clip_length(tree, p)
        h = handover_index(tree, p, n)
        t += (h + 1 if h is not None else n) - lead(k)
    start = (t - lead(len(path) - 1)) / chain.FPS
    return {"file": track["file"], "start": round(max(0.0, start), 3), "pad": round(max(0.0, -start), 3),
            "secs": round(clip_length(tree, node) / chain.FPS, 3), "mode": "track", "track": tid,
            "from": path[-2].id if i < len(path) - 1 else None}


def previous_sound_span(tree: Tree, node: Node, project: Path | None = None) -> dict | None:
    """'Repeat the sound of the clip before': the sound of the clip before this one as the joined path plays it
    (without its carry-over frames and the frames after an early hand-over), repeated as often as this clip needs.
    It is laid so that the joined path hears it from its beginning: this clip's own first frame (or carry-over
    frames) is left out when joined, so its sound starts that much before the loop's end (offset).
    {"file", "start", "secs", "loop", "offset", "video", "mode": "previous"}; a generated clip's file is named
    relative to the project folder (its names are unique per version; see sound_file), "" while it is not made yet.
    None for a first clip."""
    path = tree.path_to(node.id)
    if len(path) < 2:
        return None
    k = len(path) - 2
    p = path[k]
    secs = round(clip_length(tree, node) / chain.FPS, 3)
    lead = _join_lead(tree, node) / chain.FPS
    if p.is_video:
        file, base = str((p.source or {}).get("file") or ""), float((p.source or {}).get("start") or 0.0)
    elif p.made:
        file, base = str(p.made["video"]), 0.0
    else:
        file, base = "", 0.0
    n = (made_frames(p) if not p.is_video else 0) or clip_length(tree, p)
    a, b = path_cuts(tree, path, bool(tree.options.get("keep_repeated")), planned=not (p.made or p.is_video))[k]   # (as planned until it is made)
    loop = max(1, n - a - b) / chain.FPS
    out = {"file": file, "start": round(base + a / chain.FPS, 3), "secs": secs, "loop": round(loop, 3),
           "offset": round((loop - lead) % loop, 3), "video": p.id, "mode": "previous"}
    if not p.is_video and p.made and p.made.get("sound_rev"):
        out["rev"] = int(p.made["sound_rev"])       # its sound was built again in the same file (remix_sound)
    return out


def sound_file(project: Path | None, sound: dict) -> Path:
    """The file of a clip's soundtrack (source_sound span or record): a real video's own path, or a clip of the
    project (relative)."""
    f = Path(sound["file"])
    return f if f.is_absolute() or project is None else Path(project) / f


def source_sound_problems(tree: Tree, todo: list[str], project: Path | None = None) -> list[str]:
    """Why clips that continue their real video's sound (or repeat the clip before's) cannot be made (one message
    per problem)."""
    out = []
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video or not (node.overrides or {}).get("source_sound"):
            continue
        span = source_sound_span(tree, node, project)
        over = bool((node.overrides or {}).get("sound_over"))   # laid over: not a sound for the model
        tid = track_id(node)
        if tid is not None:
            if span is None:
                out.append(f"clip #{short_id(nid)} plays an audio track that was removed; choose another sound")
                continue
            track = track_of(tree, tid)
            if not sound_file(project, span).is_file():
                out.append(f"the file of the audio track “{track['name']}” is missing ({Path(span['file']).name})")
                continue
            if span["start"] >= float(track["secs"]) - 0.05:
                out.append(f"the audio track “{track['name']}” ({float(track['secs']):.1f} s) has ended before clip "
                           f"#{short_id(nid)} (it would start at {span['start']:.1f} s); choose another sound")
                continue
            if not over and span["secs"] < AUDIO_REF_MIN:
                out.append(f"clip #{short_id(nid)} is shorter than {AUDIO_REF_MIN:.0f} s, the least sound the model "
                           "takes; make it longer to play the audio track")
            if not over:
                out += _sound_room(tree, node, span)
            continue
        if (node.overrides or {}).get("source_sound") == "previous":
            if span is None:
                out.append(f"clip #{short_id(nid)} repeats the sound of the clip before, but it is the first clip")
                continue
            before = tree.nodes[span["video"]]
            if not over and span["secs"] < AUDIO_REF_MIN:
                out.append(f"clip #{short_id(nid)} is shorter than {AUDIO_REF_MIN:.0f} s, the least sound the model "
                           "takes; make it longer to repeat the sound of the clip before")
            if span["loop"] < REPEAT_MIN:
                out.append(f"the clip before clip #{short_id(nid)} plays only {span['loop']:.2f} s in the film, too short "
                           f"to repeat (at least {REPEAT_MIN} s)")
            if not over:
                out += _sound_room(tree, node, span)
            if not before.is_video and before.id in todo:   # made (again) first in this run: checked then
                continue
            if not span["file"]:
                out.append(f"clip #{short_id(nid)} repeats the sound of clip #{short_id(before.id)}, which is not made yet")
                continue
            if not sound_file(project, span).is_file():
                out.append(f"the video file of clip #{short_id(nid)}'s sound is missing ({Path(span['file']).name})")
                continue
            out += _repeat_sound_problems(project, node, span)
            continue
        if span is None:
            out.append(f"clip #{short_id(nid)} continues a real video's sound, but there is no real video before it")
            continue
        if not Path(span["file"]).is_file():
            out.append(f"the video file of clip #{short_id(nid)}'s sound is missing ({Path(span['file']).name})")
            continue
        have = chain.audio_seconds(Path(span["file"]))
        need = span["start"] + span["secs"]
        if not over and span["secs"] < AUDIO_REF_MIN:
            out.append(f"clip #{short_id(nid)} is shorter than {AUDIO_REF_MIN:.0f} s, the least sound the model takes; "
                       "make it longer to continue the video's sound")
        if have is None:
            out.append(f"{Path(span['file']).name} has no sound, so clip #{short_id(nid)} cannot continue it")
        elif have < need - 0.05:
            out.append(f"clip #{short_id(nid)} needs the sound of {Path(span['file']).name} up to "
                       f"{int(need // 60)}:{need % 60:04.1f}, but it ends at {int(have // 60)}:{have % 60:04.1f}; "
                       "make the clip shorter")
        if not over:
            out += _sound_room(tree, node, span)
    return out


def _repeat_sound_problems(project: Path | None, node: Node, span: dict) -> list[str]:
    """The clip before's file must have sound, about as long as the part that is repeated."""
    have = chain.audio_seconds(sound_file(project, span))
    who = f"clip #{short_id(span.get('video') or node.parent or '?')}"
    if have is None:
        return [f"{who} has no sound, so clip #{short_id(node.id)} cannot repeat it"]
    if have < span["start"] + span["loop"] - 0.25:
        return [f"{who}'s sound ends at {have:.1f} s, before the part clip #{short_id(node.id)} repeats "
                f"(up to {span['start'] + span['loop']:.1f} s)"]
    return []


def _sound_room(tree: Tree, node: Node, span: dict) -> list[str]:
    """The model's limits on sound references, with the clip's soundtrack as one more."""
    nid = node.id
    n_sounds = audio_mentioned(full_prompt(tree.options, node.prompt))
    voices = node_cast(tree.options, node).get("voices") or []
    secs = sum(float(a.get("secs") or 0) for a in audio_refs(tree)[:n_sounds]) + span["secs"] + \
        sum(float(c["voice"].get("secs") or 0) for c in cast_of(tree) if c.get("voice")
            and c["voice"]["file"] in voices)
    if n_sounds + len(voices) + 1 > MAX_AUDIO_REFS:
        return [f"clip #{short_id(nid)} has too many sounds with its soundtrack (the model takes "
                f"{MAX_AUDIO_REFS}); remove a sound or voice from its prompt"]
    if secs > AUDIO_REF_TOTAL + 0.05:
        return [f"clip #{short_id(nid)} needs {secs:.1f} s of sound with its soundtrack (the model takes "
                f"{AUDIO_REF_TOTAL:.0f} s); make it shorter or leave out its other sounds"]
    return []


def lora_trigger_words(paths: list[str]) -> list[str] | None:
    """The trigger words of a LoRA stack (the app's trigger database; None if it cannot be read)."""
    try:
        from .. import triggers
        return triggers.stack_words(paths)
    except Exception as e:  # noqa: BLE001
        logger.warning("trigger words unreadable: %s", e)
        return None                                # unknown (e.g. the database cannot be read right now)


def _stack_triggers(options: dict, node: Node, stack: list) -> list[str] | None:
    """A clip's trigger words; while the database cannot be read, those it was made with (no false CHANGED)."""
    if not stack or not options.get("lora_triggers", True):
        return None
    words = lora_trigger_words([p for p, _s in stack])
    if words is None:
        words = (((node.made or {}).get("settings")) or {}).get("lora_triggers")
    return words


def _soundtrack_sentence(ss, k: int) -> str:
    mode = ss.get("mode") if isinstance(ss, dict) else ss
    sentence = (REPEAT_SOUND_TEXT if mode == "previous" else
                TRACK_SOUND_TEXT if mode == "track" or (isinstance(mode, str) and mode.startswith(TRACK_PREFIX))
                else SOURCE_SOUND_TEXT)
    return sentence.format(k=k)
