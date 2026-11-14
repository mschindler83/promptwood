"""Making clips: the inputs of a clip, the generation loop (run_tree), copy_video, sound remix."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Callable, cast

from .. import chain
from .. import media
from .consts import VIDEO
from .guard import inside
from .model import Node, Tree, short_id
from .schema import MadeRecord
from .timeline import clip_length, full_prompt, handover_file, handover_index
from .files import _unlink, clip_files, handover_frames, load, new_clip_files, remove_stale_files, save
from .cast import audio_mentioned, audio_refs, cast_of, is_h3, kind_of, mentions
from .clipcast import cast_problems, node_cast
from .keyframes import _in_ends, end_image_used, keyframe_frame, missing_end_images, missing_keyframes
from .starts import prepared_start
from .sources import source_rotation
from .audio_tracks import _repeat_sound_problems, sound_file, source_sound_problems, uses_ref_model
from .inputs import empty_prompts, frame_numbers, loras_needed, model_prompt, plain_frames, plan
from .checker import Checker, keyframe_problem, video_key


def _clip_sounds(tree: Tree, node: Node, s: dict, use: dict) -> tuple[list[dict], dict]:
    """The <Audio N> list of clip_inputs (the sounds the prompt names, the voices, the soundtrack) and the number of
    each voice by its character's name."""
    sounds_n = audio_mentioned(full_prompt(tree.options, node.prompt))
    pool = audio_refs(tree)
    sounds = [{"n": k, "file": pool[k - 1]["file"] if k <= len(pool) else None, "what": "sound"}
              for k in range(1, sounds_n + 1)]
    by_voice = {c["voice"]["file"]: c["name"] for c in cast_of(tree) if c.get("voice")}
    voice_no = {}
    for f in use["voices"]:
        voice_no[by_voice.get(f)] = len(sounds) + 1
        sounds.append({"n": len(sounds) + 1, "file": f, "what": f"voice of @{by_voice.get(f)}"})
    ss = s.get("source_sound")
    if ss and not ss.get("overlay"):
        sounds.append({"n": len(s.get("audio_refs") or []) + 1, "file": None, "what": "soundtrack"})
    return sounds, voice_no


def _clip_subjects(use: dict, mentioned: set[str], voice_no: dict) -> tuple[list[dict], list[dict]]:
    """The subjects of clip_inputs and the pictures they bring (in ref_images order)."""
    subjects, pictures, k = [], [], 0
    for c in use["characters"]:
        if use["mode"] == "tags" and not c["images"] and c["name"] not in mentioned:
            continue                                 # (classic tags: the whole cast, those without pictures unused)
        n = list(range(k + 1, k + 1 + len(c["images"])))
        k += len(c["images"])
        subjects.append({"name": c["name"], "kind": kind_of(c), "pictures": n, "files": list(c["images"]),
                         "lead": use["mode"] != "tags" and c["name"] not in mentioned,
                         "voice": voice_no.get(c["name"])})
        pictures += [{"n": i, "file": f, "what": f"@{c['name']}"} for i, f in zip(n, c["images"])]
    return subjects, pictures


def clip_inputs(project: Path, tree: Tree, nid: str, checker: Checker | None = None) -> dict:
    """What a generated clip is made with, in the order the model gets it, and the exact prompt (for the clip card's
    chips and the "Model prompt" dialog; nothing is made or written). Either format.
    {"format": "h3" | "classic", "model": "ref2va" | "fl2va", "seconds": clip length,
     "start": {"given": "frame" | "guide" | None, "file": Path | None, "picture": N | None},
     "end":   {"given": "frame" | "guide" | None, "file": Path | None, "picture": N | None},
     "subjects": [{"name", "kind", "pictures": [N ...], "files": [rel ...], "lead": True for the implicit lead,
                   "voice": its <Audio N> or None}]  (in subject order: <Subject 1> ... in the h3 format),
     "pictures": [{"n": N, "file": rel or Path, "what": "@Anna" | "start frame" | "end frame"}]  (ref_images order),
     "videos":   [{"n": 1, "what": "end of clip #3" | "end of video #2", "frames": n, "node": its id}]  (<Video N>:
                 the clip before's end it continues, "continue"; picture only, so it adds no <Audio N>),
     "sounds":   [{"n": N, "file": rel or None, "what": "sound" | "voice of @Anna" | "soundtrack"}]  (<Audio N>),
     "prompt": what the model gets, "settings": the settings it is made with (Checker.settings)}
    picture: the <Picture N> a frame has as a reference picture (h3 + reference model); a frame without one is
    given only as a pinned / first / last frame. The soundtrack's file is cut when the clip is made (file None)."""
    checker = checker or Checker(project, tree)
    node = tree.nodes[nid]
    if node.is_video:
        raise ValueError(f"#{short_id(nid)} is a real video")
    s = checker.settings(nid) or {}
    use = node_cast(tree.options, node)
    plain = plain_frames(project, tree, node)
    nums = frame_numbers(len(use["images"]), s.get("frame_pictures"))
    guide = int((node.overrides or {}).get("guide") or 0)
    vc = tree.video_child(nid)
    start_file = checker.start_frame(node)
    own_end = end_image_used(tree, node)
    end_file = (Path(project) / vc.made["first"] if vc is not None and vc.made and vc.made.get("first") else None) \
        if vc is not None else (Path(project) / own_end if own_end else None)
    start = {"given": "frame" if plain[0] else "guide" if guide and node.parent is not None else None,
             "file": start_file if plain[0] else None, "picture": nums.get("start")}
    end = {"given": "frame" if plain[1] else "guide" if guide and vc is not None else None,
           "file": end_file if plain[1] else None, "picture": nums.get("end")}
    mentioned = {c["name"] for c in mentions(tree.options, full_prompt(tree.options, node.prompt))[0]}
    sounds, voice_no = _clip_sounds(tree, node, s, use)
    subjects, pictures = _clip_subjects(use, mentioned, voice_no)
    for f, d, what in (("start", start, "start frame"), ("end", end, "end frame")):
        if d["picture"] is not None:
            pictures.append({"n": d["picture"], "file": d["file"], "what": what})
    videos = []
    if (s.get("continue") or {}).get("frames") and node.parent is not None:
        p = tree.nodes[node.parent]
        videos.append({"n": 1, "what": f"end of {'video' if p.is_video else 'clip'} #{short_id(p.id)}",
                       "frames": int(s["continue"]["frames"]), "node": p.id})
    return {"format": "h3" if is_h3(tree) else "classic", "model": "ref2va" if uses_ref_model(s) else "fl2va",
            "seconds": int(s.get("frames") or clip_length(tree, node)) / chain.FPS, "start": start, "end": end,
            "subjects": subjects, "pictures": pictures, "videos": videos, "sounds": sounds,
            "prompt": model_prompt(tree.options, node, s, start=plain[0], end=plain[1]), "settings": s}


def make_video(project: Path, node: Node, base: dict) -> dict:
    """Convert a video node's range to a chain clip (new file names); returns its record."""
    project = Path(project)
    src = node.source or {}
    if not Path(src.get("file") or "").is_file():
        raise FileNotFoundError(f"video file not found: {src.get('file')}")
    video, frame = new_clip_files(project, node.id)
    first = frame.with_name(frame.stem + "_first.png")
    video.parent.mkdir(parents=True, exist_ok=True)
    try:
        n = chain.convert_range(Path(src["file"]), video, src["start"], src["end"], base["width"], base["height"],
                                rotate=source_rotation(src))
        chain.frame_at(video, 0, first)
        chain.frame_at(video, n - 1, frame)
    except BaseException:
        for f in (video, frame, first):
            _unlink(f)
        raise
    return {"kind": VIDEO, "key": video_key(node, base), "frames": n, "video": f"nodes/{video.name}",
            "frame": f"nodes/{frame.name}", "first": f"nodes/{first.name}"}


def copy_video(project: Path, tree: Tree, src: Node, parent: str | None) -> Node:
    """A copy of a video node (same file and range) after parent; its converted files are copied too (new names),
    so the copy is ready at once and deleting one never touches the other's files."""
    assert src.source is not None                    # (a video node has its source)
    node = tree.add_video(parent, src.source["file"], src.source["start"], src.source["end"],
                          source_rotation(src.source))
    made = cast(dict[str, Any], src.made)             # (the keys below are names, the checker wants literals)
    if made and all((Path(project) / made[k]).is_file() for k in ("video", "frame", "first")):
        video, frame = new_clip_files(project, node.id)
        first = frame.with_name(frame.stem + "_first.png")
        try:
            for k, target in (("video", video), ("frame", frame), ("first", first)):
                shutil.copyfile(Path(project) / made[k], target)
        except OSError:
            for f in (video, frame, first):
                _unlink(f)
            return node                              # made later by converting again
        node.made = cast(MadeRecord, {**copy.deepcopy(made), "video": f"nodes/{video.name}",
                                      "frame": f"nodes/{frame.name}", "first": f"nodes/{first.name}"})
    return node


def ensure_handover(project: Path, tree: Tree, node: Node) -> Path | None:
    """Write the hand-over frame of a made clip if it is not there yet; returns it (None: the last frame counts)."""
    if node.end_at is not None and node.made and not node.is_video and not node.made.get("frames") \
            and node.made.get("video") and (Path(project) / node.made["video"]).is_file():
        node.made["frames"] = chain.probe(Path(project) / node.made["video"])["frames"]   # older record: count once
    f = handover_file(project, tree, node)
    if f is not None and not f.is_file():
        assert node.made is not None                 # (a hand-over file belongs to a made clip)
        at = handover_index(tree, node)
        assert at is not None
        with media.part_file(f) as tmp:
            chain.frame_at(Path(project) / node.made["video"], at, tmp)
    if node.made and node.made.get("video") and not node.is_video:   # frames of an earlier choice in this take
        v = Path(project) / node.made["video"]
        if inside(project, "nodes", v):
            for old in handover_frames(v):
                if old != f:
                    _unlink(old)
    return f


def ensure_handovers(project: Path, tree: Tree) -> list[str]:
    """All hand-over frames of the tree (e.g. after a copy); returns the nodes whose frame could not be made."""
    bad = []
    for n in tree.nodes.values():
        try:
            ensure_handover(project, tree, n)
        except (OSError, RuntimeError):
            bad.append(n.id)
    return bad


def _clip_frames(project: Path, node: Node) -> int:
    made = node.made or {}
    if node.is_video and made.get("frames"):
        return int(made["frames"])
    return chain.probe(Path(project) / made["video"])["frames"]


def _guides(project: Path, tree: Tree, node: Node, settings: dict, tmp: Path) -> tuple[list[dict], Path | None]:
    """Motion guide clips (and the end frame) for a generated clip: (guides, end image)."""
    guide = int(settings.get("guide") or 0)
    frames = int(settings["frames"])
    vc = tree.video_child(node.id)
    guides: list[dict] = []
    end_image: Path | None = None
    if guide and node.parent is not None:
        parent = tree.nodes[node.parent]
        clip, n = project / (parent.made or {})["video"], _clip_frames(project, parent)
        k = handover_index(tree, parent, n)
        if k is not None:                           # it hands over earlier: the frames up to that one
            n = k + 1
        if n < guide:
            raise ValueError(f"clip #{short_id(parent.id)} ({n} frames) is shorter than the {guide}-frame carry-over "
                             f"of clip #{short_id(node.id)}")
        g = tmp / "guide_start.mp4"
        audio = chain.cut_frames(clip, g, n - guide, guide)
        guides.append({"video": str(g), "frame_idx": 0, "frames": guide, "audio": audio, "at": "start"})
    if vc is not None:
        if guide:
            clip, n = project / (vc.made or {})["video"], _clip_frames(project, vc)
            if n < guide:
                raise ValueError(f"video #{short_id(vc.id)} ({n} frames) is shorter than the {guide}-frame carry-over "
                                 f"of clip #{short_id(node.id)}")
            g = tmp / "guide_end.mp4"
            audio = chain.cut_frames(clip, g, 0, guide)
            guides.append({"video": str(g), "frame_idx": frames - guide, "frames": guide, "audio": audio,
                           "at": "end"})
        else:
            end_image = project / (vc.made or {})["first"]
    elif (own_end := end_image_used(tree, node)):
        end_image = project / own_end
    used = sum(g["frames"] for g in guides)
    if used and used >= frames:
        raise ValueError(f"clip #{short_id(node.id)} ({frames} frames) is too short for its carry-over "
                         f"({used} frames); make it longer or the carry-over shorter")
    return guides, end_image


def _continuation(project: Path, tree: Tree, node: Node, settings: dict, tmp: Path) -> Path | None:
    """The reference video of a clip that continues the clip before (settings "continue"): that clip's last n
    frames up to its hand-over frame, cut to a short clip of their own as a carry-over is (chain.cut_frames; its
    sound stays in the file, but only the pictures are wired). None without a continuation."""
    n = int((settings.get("continue") or {}).get("frames") or 0)
    if not n or node.parent is None:
        return None
    parent = tree.nodes[node.parent]
    clip, have = project / (parent.made or {})["video"], _clip_frames(project, parent)
    k = handover_index(tree, parent, have)
    if k is not None:                               # it hands over earlier: the frames up to that one
        have = k + 1
    what = f"{'video' if parent.is_video else 'clip'} #{short_id(parent.id)}"
    if have < n:
        raise ValueError(f"{what} ({have} frames) is shorter than the {n}-frame continuation of clip "
                         f"#{short_id(node.id)}")
    if n > int(settings["frames"]):
        raise ValueError(f"clip #{short_id(node.id)} ({settings['frames']} frames) is shorter than its {n}-frame "
                         "continuation; make it longer or the continuation shorter")
    out = tmp / "continue_from.mp4"
    chain.cut_frames(clip, out, have - n, n)
    return out


def guide_problem(tree: Tree, checker: Checker, todo: list[str]) -> str | None:
    """Why a motion guide or a continuation of a clip to be made cannot work (a neighbour shorter than the guide or
    the continuation, a clip too short for its guides or its continuation), or None. Checked before anything is
    made, with the lengths the clips will have."""
    def length(n: Node) -> int:
        if n.is_video and n.id in todo:             # converted again: the current range
            src = n.source or {}
            return chain.range_frames(src["start"], src["end"])
        return clip_length(tree, n)                 # converted (a source that ends early gives fewer frames)

    for nid in todo:
        node = tree.nodes[nid]
        s = checker.settings(nid) if not node.is_video else None
        s = s if s is not None else {}
        cont = int(((s or {}).get("continue") or {}).get("frames") or 0)
        if cont and node.parent is not None:
            other = tree.nodes[node.parent]
            n = length(other)
            k = handover_index(tree, other, n)
            n = k + 1 if k is not None else n
            what = f"{'video' if other.is_video else 'clip'} #{short_id(other.id)}"
            if n < cont:
                return f"{what} ({n} frames) is shorter than the {cont}-frame continuation of clip #{short_id(nid)}"
            if cont > int(s["frames"]):
                return (f"clip #{short_id(nid)} ({s['frames']} frames) is shorter than its {cont}-frame "
                        "continuation; make it longer or the continuation shorter")
        guide = int(s.get("guide") or 0)
        if not guide:
            continue
        used = 0
        near: list[tuple[str, Node]] = [("clip", tree.nodes[node.parent])] if node.parent is not None else []
        vc = tree.video_child(nid)
        if vc is not None:
            near.append(("video", vc))
        for what, other in near:
            n = length(other)
            k = handover_index(tree, other, n) if what == "clip" else None
            if k is not None:                       # it hands over earlier
                n = k + 1
            if n < guide:
                return (f"{what} #{short_id(other.id)} ({n} frames) is shorter than the {guide}-frame carry-over "
                        f"of clip #{short_id(nid)}")
            used += guide
        frames = int(s["frames"])
        if used and used >= frames:
            return (f"clip #{short_id(nid)} ({frames} frames) is too short for its carry-over ({used} frames); "
                    "make it longer or the carry-over shorter")
    return None


def given_sound(project: Path, node: Node, ss: dict, wav: Path, check: bool = True) -> None:
    """A clip's given sound (its recorded source_sound span) cut into a WAV, as the clip gets it: by run_tree before
    the model and remix_sound after it, so a remix lays exactly the same sound."""
    if check and ss.get("mode") == "previous":
        bad = _repeat_sound_problems(project, node, ss)
        if bad:
            raise RuntimeError(bad[0])
    chain.to_wav(sound_file(project, ss), wav, ss["start"], ss["secs"], ss.get("loop"), ss.get("offset") or 0.0,
                 ss.get("pad") or 0.0, fill=ss.get("mode") == "track")


_SOUND_OVERRIDES = ("source_sound", "sound_over", "sound_mix_level")


def remix_sound(project: Path, tree: Tree, nid: str, checker: Checker | None = None, persist: bool = True) -> bool:
    """A made clip whose settings changed only in how its laid-over sound is mixed (Checker.remix_due): its sound
    is built again from the model's sound kept next to it and the given sound (cut again as run_tree does), in
    place, the picture copied packet for packet; its record then holds the current settings (sound_rev counts the
    rebuilds: a clip that repeats this one's sound, or carries over from it, depends on it). Seconds instead of a new
    generation; the last frame and everything the clips after it start from stay as they are. False: nothing to do
    (not due: e.g. no own sound kept, then the clip has to be made again). persist: write tree.json (a window saves
    its own tree instead)."""
    project = Path(project)
    ck = checker or Checker(project, tree)
    if nid not in tree.nodes or not ck.remix_due(nid):
        return False
    node = tree.nodes[nid]
    settings = ck.settings(nid)
    assert settings is not None and node.made is not None     # (remix_due: a made clip with settings)
    made = cast(dict[str, Any], node.made)
    ss = settings["source_sound"]
    if not sound_file(project, ss).is_file():
        raise FileNotFoundError(f"the sound of clip #{short_id(nid)} is missing ({Path(ss['file']).name})")
    with media.scratch("job", dir=str(project / "nodes")) as tmp:
        wav = Path(tmp) / "source_sound.wav"
        given_sound(project, node, ss, wav)
        if not chain.keep_copy(project, project / made["video"]):      # (never overwritten: the old mix stays)
            raise OSError(f"the clip of {nid} could not be kept in {chain.OLD_DIR}/ before its sound is mixed again")
        chain.remix_audio(project / made["video"], project / made["own_sound"], wav,
                          float(ss.get("mix_level") or 0.0) if ss.get("mix") else None)
    made["settings"] = settings
    base = made.get("overrides")           # (what Undo goes back to: only the sound keys follow the remix)
    if isinstance(base, dict):
        for k in _SOUND_OVERRIDES:
            if cast(dict[str, Any], node.overrides or {}).get(k) is None:
                base.pop(k, None)
            else:
                base[k] = copy.deepcopy(cast(dict[str, Any], node.overrides)[k])
    made["sound_rev"] = int(made.get("sound_rev") or 0) + 1
    if persist:
        save(project, tree)
    return True


def sweep_scratch(project: Path) -> None:
    """The temporary folders of a killed run in the project's nodes/ (also for the window to call when it opens one)."""
    for left in list((Path(project) / "nodes").glob("video-chain-*")):
        shutil.rmtree(left, ignore_errors=True)


def remix_sounds(project: Path, tree: Tree, nids: list[str], checker: Checker | None = None,
                 persist: bool = True, log: Callable[[str], None] = print) -> list[str]:
    """remix_sound for every due clip of nids (in this order: put a clip before the ones after it), again and again
    while one becomes due (a clip that lays over the sound of the one before, remixed now). Returns the remixed.
    A clip that cannot be remixed (its sound file is gone, ffmpeg fails) is reported to log and left as it is: it
    stays "changed" and does not stop the others."""
    out: list[str] = []
    failed: set[str] = set()
    ck = checker or Checker(project, tree)
    again = True
    try:
        while again and len(out) <= len(nids) ** 2:   # (each remix can make only the clips after it due)
            again = False
            for nid in nids:
                if nid in failed:
                    continue
                try:
                    done = remix_sound(project, tree, nid, ck, persist=False)
                except (OSError, RuntimeError) as e:
                    failed.add(nid)
                    log(f"{nid}: its sound could not be mixed again ({e})")
                    continue
                if done:
                    out.append(nid)
                    ck = Checker(project, tree)         # (what depends on its sound changes)
                    again = True
                    break
    finally:
        if persist and out:
            save(project, tree)                         # once, not after every clip
    return out


def _check_run(project: Path, tree: Tree, checker: Checker, todo: list[str]) -> None:
    """Everything a run needs is there (LoRAs, videos, pictures) and fits (frames, cast, sound): raises if not."""
    missing = [lo for lo in loras_needed(tree, checker, todo) if not Path(lo).is_file()]
    if missing:
        raise FileNotFoundError(f"LoRA not found: {missing[0]}")
    gone = [nid for nid in todo if tree.nodes[nid].is_video and not Path((tree.nodes[nid].source or {})["file"]).is_file()]
    if gone:
        raise FileNotFoundError(f"video file not found: {(tree.nodes[gone[0]].source or {})['file']}")
    problem = guide_problem(tree, checker, todo)
    if problem:
        raise ValueError(problem)
    no_end = missing_end_images(project, tree, todo)
    if no_end:
        raise FileNotFoundError(f"the end image of clip #{short_id(no_end[0])} is missing")
    no_key = missing_keyframes(project, tree, todo)
    if no_key:
        raise FileNotFoundError(f"a keyframe picture of clip #{short_id(no_key[0])} is missing")
    problem = keyframe_problem(tree, checker, todo)
    if problem:
        raise ValueError(problem)
    faces = cast_problems(project, tree, todo) + source_sound_problems(tree, todo, project)
    if faces:
        raise ValueError(faces[0])


def _generator_call(project: Path, node: Node, settings: dict) -> dict:
    """The settings handed to generator.generate: the clip's settings with the files its pictures and sounds are."""
    call: dict[str, Any] = dict(settings)
    if node.keyframes:
        call["keyframes"] = [{"image": str(project / k["image"]), "frame_idx": keyframe_frame(k["at"])}
                             for k in node.keyframes]
    if settings.get("refs"):               # the generator gets the files; the record keeps names + hashes
        call["ref_paths"] = [str(project / r["file"]) for r in settings["refs"]]
    if settings.get("audio_refs"):
        call["audio_paths"] = [str(project / a["file"]) for a in settings["audio_refs"]]
    return call


def _record_made(project: Path, tree: Tree, node: Node, settings: dict, *, new_video: Path, new_frame: Path,
                 own: Path | None, made_n: int, start_sha: str | None, end_key, key_key, guide_from, continue_from=None,
                 log: Callable[[str], None]) -> list[Path]:
    """Write the record of the clip just made (node.made). Returns the pictures of the previous version that go to
    old/ (its end image and keyframes, when they are no longer the clip's)."""
    nid = node.id
    old: list[Path] = []
    old_end = (node.made or {}).get("end_image")
    old_keys = {k.get("image") for k in (node.made or {}).get("keyframes") or [] if isinstance(k, dict)}
    cut = chain.hard_cuts(new_video) if node.keyframes or end_image_used(tree, node) else []
    node.made = {"prompt": node.prompt, "seed": node.seed, "settings": settings,
                 "overrides": copy.deepcopy(node.overrides or {}), "end_image": node.end_image,
                 "keyframes": copy.deepcopy(node.keyframes),
                 "start_sha1": start_sha, "video": f"nodes/{new_video.name}",
                 "frame": f"nodes/{new_frame.name}", "frames": made_n}
    if own is not None:
        node.made["own_sound"] = f"nodes/{own.name}"   # (a sound-only change mixes again: remix_sound)
    if old_end and old_end != node.end_image and _in_ends(project, old_end):
        old.append(project / old_end)          # the end image of the previous version
    for rel in old_keys - {k["image"] for k in node.keyframes} - {None}:
        if _in_ends(project, rel):
            old.append(project / rel)          # keyframe pictures of the previous version
    if cut:
        node.made["cuts"] = cut                # hard cuts H3 made (a keyframe / end it could not reach)
        log(f"{nid}: the clip cuts at {', '.join(f'{c / chain.FPS:.2f} s' for c in cut)} "
            "(a keyframe or the end image is probably too far from what comes before it)")
    if end_key is not None:
        node.made["end_key"] = end_key
    if key_key is not None:
        node.made["keyframe_key"] = key_key
    if guide_from is not None:
        node.made["guide_from"] = guide_from
    if continue_from is not None:
        node.made["continue_from"] = continue_from
    if node.end_at is not None:
        # a new take: the frame chosen in the old one means nothing here; it hands over at its last
        # frame until a frame of this take is chosen
        log(f"{nid}: made again, it hands over at its last frame again (choose a frame of the new take)")
        node.end_at = None
    return old


def _make_ai_clip(generator: chain.Generator, project: Path, tree: Tree, node: Node, checker: Checker,
                  k: int, total: int, emit: Callable[[dict], None],
                  log: Callable[[str], None]) -> tuple[list[Path], list[Path]]:
    """Generate one clip and write its record. Returns (its new files, the files of the previous version that go
    to old/ besides the clip's own)."""
    nid = node.id
    nodes_dir = project / "nodes"
    start = checker.start_frame(node)
    log(f"[{k}/{total}] {nid}: generating (seed {node.seed}) from "
        f"{start.name if start else 'nothing (no start frame)'}")
    send = start                                 # what H3 gets: a start picture prepared for the canvas
    if start is not None and node.parent is None:   # (the record keeps the picture's own hash)
        base = checker.base() or {}
        if base.get("width") and base.get("height"):
            send = prepared_start(project, base["width"], base["height"], start)
    emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": total})
    # a new clip gets new file names: nothing is overwritten, a failed or stopped generation keeps the
    # previous clip and its record, and a still-open old file (SMB) does not stop anything
    new_video, new_frame = new_clip_files(project, nid)
    settings = checker.settings(nid)
    assert settings is not None                    # (checked before the run: the project has its size)
    start_sha = checker.sha(start)
    end_key = checker.end_key(node)
    key_key = checker.keyframe_key(node)
    guide_from = checker.guide_from(node)
    continue_from = checker.continue_from(node)
    call = _generator_call(project, node, settings)
    own: Path | None = None
    try:
        with media.scratch("job", dir=str(nodes_dir)) as tmp:
            guides, end_image = _guides(project, tree, node, settings, Path(tmp))
            if guides:
                call["guides"] = guides
            ref_video = _continuation(project, tree, node, settings, Path(tmp))
            if ref_video is not None:          # <Video 1>: the end of the clip before
                call["ref_videos"] = [str(ref_video)]
            if end_image is not None:
                call["end_image"] = str(end_image)
            sound = None
            if settings.get("source_sound"):   # the real video's sound after its range: the soundtrack
                sound = Path(tmp) / "source_sound.wav"
                ss = settings["source_sound"]
                given_sound(project, node, ss, sound)
                if not ss.get("overlay"):  # the model makes the picture to it (lip sync)
                    call["audio_paths"] = list(call.get("audio_paths") or []) + [str(sound)]
            plain = (send is not None and not any(g.get("at") == "start" for g in guides),
                     end_image is not None)        # what the workflow really gets as plain frames
            numbered = settings.get("frame_pictures") or []
            if ("start" in numbered and not plain[0]) or ("end" in numbered and not plain[1]):
                raise RuntimeError(f"clip #{short_id(nid)}: its frames are not the ones its settings number "
                                   f"({', '.join(numbered)})")
            generator.generate(model_prompt(tree.options, node, settings, start=plain[0], end=plain[1]),
                               send, new_video, new_frame, node.seed, settings=call)
            if sound is not None and new_video.exists() and settings["source_sound"].get("overlay"):
                own = chain.keep_own_sound(new_video)    # the model's sound, before it is replaced / mixed
            if sound is not None and new_video.exists() and settings["source_sound"].get("mix"):
                chain.mix_audio(new_video, sound, settings["source_sound"].get("mix_level") or 0.0)
            elif sound is not None and new_video.exists():
                chain.replace_audio(new_video, sound)    # exactly the recorded speech, not a remake
        if not new_video.exists() or not new_frame.exists():
            raise RuntimeError(f"the generator did not write the clip of {nid}")
        made_n = chain.probe(new_video)["frames"]      # the real count (hand-over frames, joins)
    except BaseException:
        for f in (new_video, new_frame, own):
            if f is not None:
                _unlink(f)
        raise
    old = _record_made(project, tree, node, settings, new_video=new_video, new_frame=new_frame, own=own,
                       made_n=made_n, start_sha=start_sha, end_key=end_key, key_key=key_key,
                       guide_from=guide_from, continue_from=continue_from, log=log)
    return [new_video, new_frame] + ([own] if own is not None else []), old


def run_tree(generator: chain.Generator, project: Path, targets: list[str],
             events: Callable[[dict], None] | None = None, log: Callable[[str], None] = print) -> list[str]:
    """Make what the target paths need. Saves tree.json after every clip, so a stop keeps finished clips."""
    project = Path(project)
    emit = events or (lambda _e: None)
    tree = load(project)
    for t in targets:
        if t not in tree.nodes:
            raise KeyError(f"no node {t}")
    empty = empty_prompts(tree, targets)
    if empty:
        raise ValueError(f"empty prompt in node {empty[0]}")
    ensure_handovers(project, tree)               # before the states: children start from them
    checker = Checker(project, tree)
    if checker.base() is None:
        raise FileNotFoundError("the project has no start image (and no video to take the size from)")
    on_paths = {n.id for t in targets for n in tree.path_to(t)}
    # a sound-only change: the sound is mixed again (seconds), not the clip made again; before the plan, since what
    # repeats or carries over that sound depends on it
    for nid in remix_sounds(project, tree, sorted(on_paths, key=lambda i: tree.depth(i)), checker, log=log):
        log(f"{nid}: sound mixed again from its kept own sound (no new generation)")
        checker = Checker(project, tree)
    todo = plan(tree, checker, targets)
    _check_run(project, tree, checker, todo)
    for nid in sorted(on_paths - set(todo), key=lambda i: tree.depth(i)):
        emit({"e": "node", "id": nid, "state": "reused"})
    emit({"e": "plan", "make": todo})
    nodes_dir = project / "nodes"
    nodes_dir.mkdir(exist_ok=True)
    sweep_scratch(project)
    remove_stale_files(project, tree)
    for k, nid in enumerate(todo, start=1):
        node = tree.nodes[nid]
        old = clip_files(project, node)
        if node.is_video:
            src = node.source or {}
            log(f"[{k}/{len(todo)}] {nid}: converting {Path(src['file']).name} {src['start']:.2f}-{src['end']:.2f} s")
            emit({"e": "node", "id": nid, "state": "generating", "k": k, "n": len(todo)})
            record = make_video(project, node, checker.base() or {})
            node.made = cast(MadeRecord, record)
            new_files = [project / record[x] for x in ("video", "frame", "first")]
        else:
            new_files, older = _make_ai_clip(generator, project, tree, node, checker, k, len(todo), emit, log)
            old += older
        save(project, tree)
        for f in old:                              # the previous version is never deleted: it goes to old/
            if f not in new_files and not chain.keep_aside(project, f):
                log(f"{f.name} is still in use; it is moved to {chain.OLD_DIR}/ on a later run")
        checker = Checker(project, tree)          # the new frame changes what depends on it
        emit({"e": "node", "id": nid, "state": "done"})
    emit({"e": "done", "made": todo})
    return todo
