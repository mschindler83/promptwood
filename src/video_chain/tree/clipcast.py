"""What a clip is made with from the cast: its pictures, voices and sound, and the problems found in them."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .. import chain
from .. import h3_prompt as _h3
from .consts import AUDIO_REF_TOTAL, MAX_AUDIO_REFS, MAX_CAST_PICTURES, MENTION, PICTURE_TAG, WRONG_TAG
from .model import Node, Tree, short_id
from .timeline import full_prompt
from .cast import (
    _in_lines,
    all_pictures,
    audio_mentioned,
    audio_refs,
    cast_of,
    character,
    is_h3,
    kind_of,
    lead_of,
    mentions)
from .prompts import _picture_tags, _pictures_note, _voiced_prompt, dialogue, without_speech_tags
from .schema import CastEntry, Options, Overrides


_KIND_WORD = {"person": "person", "place": "place", "object": "object", "style": "visual style"}


def _tie(c: CastEntry, tags: str) -> str:
    """The classic sentence tying a cast entry to its pictures."""
    if kind_of(c) == "person":
        return f"{c['name']} has exactly the face of the person in {tags}."
    return f"{c['name']} is the {_KIND_WORD[kind_of(c)]} shown in {tags}."


def sound_given(node: Node) -> bool:
    """The clip's whole sound is given (its Sound setting plays a track, a video's or the clip before's sound, and
    the picture is made to it): the model speaks nothing itself, so it gets no voices (see clip_cast)."""
    ov = node.overrides or {}
    return not node.is_video and bool(ov.get("source_sound")) and not ov.get("sound_over")


def sound_mixed(node: Node, model_sound: bool = True) -> bool:
    """The clip's given sound is laid over and mixed with the sound the model makes for it (overrides "sound_over":
    "mix"), not put in instead of it. model_sound: the project's sound is on; without it the model makes none, so
    the given sound simply replaces the silent clip's (as a plain overlay, recorded as one)."""
    ov = node.overrides or {}
    return not node.is_video and bool(ov.get("source_sound")) and ov.get("sound_over") == "mix" and bool(model_sound)


def mix_level(node: Node) -> float:
    """The given sound's level in the mix in dB (overrides "sound_mix_level"; 0: as recorded), within
    chain.MIX_LEVELS."""
    try:
        db = float((node.overrides or {}).get("sound_mix_level") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(max(db, chain.MIX_LEVELS[0]), chain.MIX_LEVELS[1])


def lead_on(node: Node) -> bool:
    """The clip may get the lead's pictures when its prompt mentions nobody (overrides "lead": False switches that
    off for this clip)."""
    return (node.overrides or {}).get("lead") is not False


def set_lead_on(node: Node, on: bool) -> None:
    """Switch a clip's lead on (the default: the key goes, so the clip is recorded as before) or off."""
    ov: Overrides = {**(node.overrides or {})}
    if on:
        ov.pop("lead", None)
    else:
        ov["lead"] = False
    node.overrides = ov


def node_cast(options: Options, node: Node) -> dict[str, Any]:
    """clip_cast for a clip as it is set up: without the voices when its sound is given, without the lead when its
    lead switch is off."""
    return clip_cast(options, node.prompt, voices=not sound_given(node), lead=lead_on(node))


def clip_cast(options: Options, prompt: str, voices: bool = True, lead: bool = True) -> dict[str, Any]:
    """How a clip uses the cast. The prompt is the clip's own (the style text is put around it here).
    voices=False: the clip's sound is given (sound_given), its spoken lines are in that sound already, so no voice
    is sent and the characters are used as if they had none (the lines stay in the prompt as written; the
    soundtrack sentence of model_prompt ties the lips to the sound). The voices would only use up the 15 s of
    sound the model takes, which the soundtrack needs.
    lead=False: the clip's lead switch is off (lead_on): a prompt that mentions nobody gets no lead.
    {"mode": "voices" | "mention" | "tags" | "lead" | None, "characters": [...], "images": [rel], "voices": [rel],
    "language": the dialogue language and "lines": [{"name": speaker or None, "line"}] (voices only), "prompt": what
    the model gets, "record": [{"name", "desc"}]
    (voices / mention: what the model prompt says about them)}.
    voices:  a character with a voice says a line ("..." in the sentence of its @Name; with nobody mentioned, the
             lead's lines): MiniMax's structured format (see _voiced_prompt); the voices come after the sounds the
             prompt names as <Audio N>.
    mention: the characters named as @Name, numbered in order of their first mention; each @Name becomes its name
             (the first time with its <Picture> tags), and a sentence ties each to their face (a place, object or
             style: to what it is).
    tags:    the prompt writes <Picture N> itself: every face picture of the cast, in cast order.
    lead:    nobody is mentioned: the lead's pictures (the generator adds its <Picture 1> sentence).
    None:    no face pictures (no cast, or no lead).
    In the h3 format (is_h3) the same modes pick the same kind of pictures, but the text is written by
    h3_prompt.compose (see _h3_cast; model_prompt adds the clip's frames, camera, notes and soundtrack)."""
    prompt = without_speech_tags(prompt)          # (how a line is spoken is for text to speech only)
    if is_h3(options):
        return _h3_cast(options, prompt, voices, lead)
    text = full_prompt(options, prompt)
    found, _ = mentions(options, text)
    the_lead = lead_of(options) if lead else None
    implicit = not found and the_lead is not None and bool(the_lead.get("voice")) and not PICTURE_TAG.search(text)
    lines = dialogue(options, text, default=the_lead if implicit else None)
    talking = [c for c in ([the_lead] if implicit and the_lead is not None else found)
               if (c["images"] or c.get("voice"))]
    if voices and any(d["speaker"] is not None and d["speaker"].get("voice")
                      and any(d["speaker"]["name"] == c["name"] for c in talking) for d in lines):
        out, images, voice_files = _voiced_prompt(options, text, talking, lines, audio_mentioned(text), implicit)
        lang = ((options.get("identity") or {}).get("language") or "English").strip() or "English"
        said = [{"name": d["speaker"]["name"] if d["speaker"] else None, "line": d["line"]} for d in lines]
        return {"mode": "voices", "characters": talking, "images": images, "voices": voice_files, "language": lang,
                "lines": said, "prompt": out,
                "record": [_cast_record(c) for c in talking]}
    found = [c for c in found if c["images"]]
    if found:
        tags, images = _picture_tags(found)
        seen: set[str] = set()
        inside = _in_lines(text)

        def name(m):
            c = character(options, m.group(1))
            if c is None:
                return m.group(1) if inside(m.start()) else m.group(0)
            if not c["images"] or c["name"] in seen or inside(m.start()):
                return c["name"]
            seen.add(c["name"])
            return f"{c['name']} (the {_KIND_WORD[kind_of(c)]} in {tags[c['name']]})"
        out = MENTION.sub(name, text)
        if (options.get("identity") or {}).get("hint", True):
            extra = [_tie(c, tags[c["name"]]) for c in found]
            extra += [f"{c['name']}: {c['desc'].rstrip('.')}." for c in found if c.get("desc")]
            if _pictures_note(found):
                extra.append(_pictures_note(found))
            out = f"{out.rstrip()} {' '.join(extra)}"
        hint = (options.get("identity") or {}).get("hint", True)      # the descriptions are in the prompt only then
        return {"mode": "mention", "characters": found, "images": images, "voices": [], "prompt": out,
                "record": [_cast_record(c, c.get("desc", "") if hint else "") for c in found]}
    plain = MENTION.sub(lambda m: (character(options, m.group(1)) or {"name": m.group(0)})["name"], text)
    if PICTURE_TAG.search(text) and all_pictures(options):
        return {"mode": "tags", "characters": cast_of(options), "images": all_pictures(options), "voices": [],
                "prompt": plain, "record": None}
    if the_lead and the_lead["images"] and not mentions(options, text)[0]:
        return {"mode": "lead", "characters": [the_lead], "images": list(the_lead["images"]), "voices": [],
                "prompt": plain, "record": None}
    return {"mode": None, "characters": [], "images": [], "voices": [], "prompt": plain, "record": None}


def _cast_record(c: CastEntry, desc: str | None = None) -> dict[str, Any]:
    """What the clip's record keeps of a cast entry the model prompt describes: name, description, and the kind
    of a non-person (a person is recorded as before)."""
    out: dict[str, Any] = {"name": c["name"], "desc": c.get("desc", "") if desc is None else desc}
    if kind_of(c) != "person":
        out["kind"] = kind_of(c)
    return out


def _language(options: Options) -> str:
    return ((options.get("identity") or {}).get("language") or "English").strip() or "English"


def _h3_cast(options: Options, prompt: str, voices: bool, lead_ok: bool) -> dict[str, Any]:
    """clip_cast in the h3 format. The pictures are chosen as in the classic format: the cast entries the prompt
    mentions (in order of first mention, their pictures numbered on), every picture in cast order when it writes
    <Picture N> itself, else the lead (switched on, a person, with pictures or a voice that speaks). Every entry
    the prompt mentions is a subject (also one without pictures: its description is put in at its first
    mention); a person with a voice who speaks gets the voice (after the <Audio N> the prompt names, in speaking
    order: h3_prompt.voice_numbers). Lines nobody is named for are the implicit lead's.
    A prompt that already is a structured H3 prompt (h3_prompt.is_structured) is sent as written: the pictures as
    above, no voices, nothing added.
    Besides clip_cast's keys: "h3" = {"structured", "body" (style prefix + prompt + suffix, without the trailing
    sound lines), "soundscape", "music" (those lines; None: not given), "subjects" (h3_prompt subjects), "lines"
    (spoken lines with offsets into body), "lead_implicit", "sounds" (the highest <Audio N> named)}; "prompt" is the
    text without the clip's frames, camera and notes (model_prompt gives the full one)."""
    structured = _h3.is_structured(prompt)
    if structured:
        body, scape, music = (prompt or "").strip(), None, None
    else:
        own, scape, music = _h3.split_sound_sections(prompt)
        body = full_prompt(options, own)
    n_sounds = audio_mentioned(full_prompt(options, prompt))
    found, _ = mentions(options, body)
    lead = lead_of(options) if lead_ok else None
    tags = bool(PICTURE_TAG.search(body)) and bool(all_pictures(options)) and not any(c["images"] for c in found)
    implicit = not found and lead is not None and not PICTURE_TAG.search(body)
    lines = [] if structured else dialogue(options, body, default=lead if implicit else None)
    if implicit and lead is not None and not lead["images"]:     # a lead without pictures only counts with its voice
        implicit = voices and bool(lead.get("voice")) and any((d["speaker"] or {}).get("name") == lead["name"]
                                                              for d in lines)
        if not implicit:
            lines = dialogue(options, body)
    if tags:
        chars = [c for c in cast_of(options) if c["images"]] + [c for c in found if not c["images"]]
    else:
        chars = [lead] if implicit and lead is not None else list(found)
    subjects: list[dict[str, Any]] = []
    images: list[str] = []
    record: list[dict[str, Any]] = []
    for c in chars:
        first = len(images) + 1
        images += c["images"]
        subjects.append({"name": c["name"], "kind": kind_of(c), "desc": (c.get("desc") or "").strip(),
                         "pictures": list(range(first, len(images) + 1)),
                         "voice": voices and not structured and kind_of(c) == "person" and bool(c.get("voice"))})
        record.append(_cast_record(c))
    said = [{"start": d["start"], "end": d["end"], "line": d["line"],
             "speaker": d["speaker"]["name"] if d["speaker"] else None} for d in lines]
    numbers = _h3.voice_numbers(subjects, said, n_sounds)
    voice_files = [(character(options, n) or {})["voice"]["file"] for n in sorted(numbers, key=lambda n: numbers[n])]
    if voice_files:
        mode = "voices"
    elif tags:
        mode = "tags"
    elif implicit and lead is not None and lead["images"]:
        mode = "lead"
    elif images:
        mode = "mention"
    else:
        mode = None
    h = {"structured": structured, "body": body, "soundscape": scape, "music": music, "subjects": subjects,
         "lines": said, "lead_implicit": implicit, "sounds": n_sounds}
    out: dict[str, Any] = {"mode": mode, "characters": chars, "images": images, "voices": voice_files,
           "prompt": body if structured else _h3.compose(
               body, subjects=subjects, sounds=n_sounds, lines=said, language=_language(options),
               lead_implicit=implicit, soundscape=scape, music=music, resolve=_resolver(options),
               reference=bool(images or voice_files or n_sounds)),
           "record": record or None, "h3": h}
    if mode == "voices":
        out.update({"language": _language(options),
                    "lines": [{"name": d["speaker"], "line": d["line"]} for d in said]})
    return out


def _resolver(options: Options):
    """@word -> the cast name it means (None: nobody in the cast), for h3_prompt.compose."""
    return lambda w: (character(options, w) or {}).get("name")


def cast_problems(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    """Why clips to be made cannot get the faces their prompts name (one message per problem)."""
    out = []
    names = [c["name"] for c in cast_of(tree)]
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video:
            continue
        text = full_prompt(tree.options, node.prompt)
        found, unknown = mentions(tree.options, text)
        for word in unknown:
            close = _closest(word, names)
            out.append(f"clip #{short_id(nid)} mentions @{word}, but there is no such character"
                       + (f" — did you mean @{close}?" if close else " (add it in the Cast)"))
        for c in found:
            if kind_of(c) != "person" and not c["images"]:
                out.append(f"@{c['name']} (clip #{short_id(nid)}) has no picture yet")
            elif not c["images"] and not c.get("voice"):
                out.append(f"@{c['name']} (clip #{short_id(nid)}) has no face picture or voice yet")
        for m in WRONG_TAG.finditer(text):
            out.append(f"clip #{short_id(nid)} writes <{m.group(1)} {m.group(2)}>; the model only knows <Picture "
                       f"{m.group(2)}> (or mention the character as @Name)")
        use = node_cast(tree.options, node)
        if use["mode"] in ("mention", "voices") and PICTURE_TAG.search(text):
            out.append(f"clip #{short_id(nid)} writes <Picture N> and mentions @Names; with @Names the app numbers the "
                       "pictures itself, so write the people only as @Name")
        if use["mode"] == "tags":
            top = max(int(re.sub(r"\D", "", t)) for t in PICTURE_TAG.findall(text))
            if top > len(use["images"]):
                n = len(use["images"])
                out.append(f"clip #{short_id(nid)} names <Picture {top}>, but the cast has only {n} face "
                           f"picture{'' if n == 1 else 's'}")
        faces = "face " if all(kind_of(c) == "person" for c in use["characters"]) else ""
        if len(use["images"]) > MAX_CAST_PICTURES:
            out.append(f"clip #{short_id(nid)} needs {len(use['images'])} {faces}pictures; the model takes at most "
                       f"{MAX_CAST_PICTURES}")
        if any(not (Path(project) / r).is_file() for r in use["images"]):
            out.append(f"a {faces}picture of clip #{short_id(nid)} is missing in refs/")
        if use["voices"]:
            if any(not (Path(project) / r).is_file() for r in use["voices"]):
                out.append(f"a voice of clip #{short_id(nid)} is missing in refs/")
            n_sounds = audio_mentioned(text)
            secs = {a["file"]: float(a.get("secs") or 0) for a in audio_refs(tree)[:n_sounds]}
            secs.update({c["voice"]["file"]: float(c["voice"].get("secs") or 0) for c in cast_of(tree)
                         if c.get("voice")})
            n = n_sounds + len(use["voices"])
            total = sum(secs.get(r, 0) for r in [a["file"] for a in audio_refs(tree)[:n_sounds]] + use["voices"])
            what = (f"{len(use['voices'])} voice{'s' if len(use['voices']) != 1 else ''}"
                    + (f" and {n_sounds} sound{'s' if n_sounds != 1 else ''}" if n_sounds else ""))
            if n > MAX_AUDIO_REFS:
                out.append(f"clip #{short_id(nid)} needs {what}; the model takes at most {MAX_AUDIO_REFS} sounds")
            elif total > AUDIO_REF_TOTAL + 0.05:
                out.append(f"clip #{short_id(nid)} needs {what} of {total:.1f} s together; the model takes at most "
                           f"{AUDIO_REF_TOTAL:.0f} s (shorten a voice: set it again from a shorter part)")
    return list(dict.fromkeys(out))


def _closest(word: str, names: list[str]) -> str | None:
    import difflib
    hit = difflib.get_close_matches(word.lower(), [n.lower() for n in names], n=1, cutoff=0.6)
    return next((n for n in names if hit and n.lower() == hit[0]), None)
