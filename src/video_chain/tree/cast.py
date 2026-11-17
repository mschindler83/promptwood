"""The cast: characters with pictures and voices, sound references (audio_refs) and the @Name mentions."""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import cast as _cast

from .. import chain
from .consts import (
    AUDIO_REF_MIN,
    AUDIO_REF_TOTAL,
    AUDIO_TAG,
    CAST_KINDS,
    DEFAULT_OPTIONS,
    DIALOGUE,
    MAX_AUDIO_REFS,
    MAX_CAST_PICTURES,
    MAX_PICTURES_PER_CHARACTER,
    MENTION,
    NAME_OK,
    PROMPT_FORMATS,
    REFS_DIR,
    VOICE_MAX_SECS,
    VOICE_REF_MAX_SECS)
from .guard import retire
from .model import Tree
from .schema import AudioRef, CastEntry, Options, Voice


def _migrate_cast(options: Options, has_cast: bool) -> None:
    """Older projects had one list of face pictures (identity "images", used by every clip while "mode" was
    "reference"). They become one character, Person1, the lead while they were on: every clip is made with exactly
    the same pictures as before, so made clips stay done."""
    ident = dict(options.get("identity") or {})
    if not has_cast:
        images = [r for r in ident.get("images") or [] if isinstance(r, str)]
        options["cast"] = ([{"name": "Person1", "images": images, "desc": "",
                             "lead": ident.get("mode") == "reference"}] if images else [])
    for k in ("images", "mode"):
        ident.pop(k, None)
    options["identity"] = {**DEFAULT_OPTIONS["identity"], **ident}
    cast: list[CastEntry] = []
    for c in options.get("cast") or []:
        if isinstance(c, dict) and isinstance(c.get("name"), str) and c["name"]:
            entry: Any = {"name": c["name"], "images": [r for r in c.get("images") or [] if isinstance(r, str)],
                     "desc": str(c.get("desc") or ""), "lead": bool(c.get("lead"))}
            if c.get("kind") in CAST_KINDS and c["kind"] != "person":
                entry["kind"] = c["kind"]                # (a person has no "kind": older files stay as they are)
            v = c.get("voice")
            if isinstance(v, dict) and isinstance(v.get("file"), str) and v["file"]:
                entry["voice"] = {"file": v["file"], "name": str(v.get("name") or ""),
                                  "secs": float(v.get("secs") or 0)}
                if isinstance(v.get("text"), str) and v["text"].strip():
                    entry["voice"]["text"] = v["text"].strip()      # what clone_file says (text to speech clones it)
                r = v.get("ref")
                if isinstance(r, dict) and isinstance(r.get("file"), str) and r["file"]:
                    entry["voice"]["ref"] = {"file": r["file"], "secs": float(r.get("secs") or 0)}
            el = (c.get("tts") or {}).get("eleven") if isinstance(c.get("tts"), dict) else None
            if isinstance(el, dict) and isinstance(el.get("id"), str) and el["id"]:
                entry["tts"] = {"eleven": {"id": el["id"], "name": str(el.get("name") or el["id"])}}
            cast.append(entry)
    options["cast"] = cast


def _options_of(src: Tree | Options) -> Options:
    """The options of a tree, or the options handed in."""
    return src.options if isinstance(src, Tree) else src


def cast_of(src: Tree | Options) -> list[CastEntry]:
    """The characters ({"name", "images", "desc", "lead"}), in the order of the cast list."""
    return list(_options_of(src).get("cast") or [])


def character(src: Tree | Options, name: str) -> CastEntry | None:
    """The character with this name (case does not matter), or None."""
    low = (name or "").lower()
    return next((c for c in cast_of(src) if c["name"].lower() == low), None)


def lead_of(src: Tree | Options) -> CastEntry | None:
    """The lead (only a person can be it)."""
    return next((c for c in cast_of(src) if c.get("lead") and kind_of(c) == "person"), None)


def kind_of(c: CastEntry | None) -> str:
    """What a cast entry is: person (also without "kind"), place, object or style. Only a person has a face (the
    crop window), a voice and can be the lead; the others are used with their whole pictures."""
    k = c.get("kind") if c else None
    return k if k in CAST_KINDS else "person"


def refiner_cast(src: Tree | Options) -> list[dict[str, Any]]:
    """The cast as the prompt refiner is told about it (comfy.cast_instruction): [{"name", "desc", "voice": bool}],
    + "kind" for a place, object or style (so the refiner does not make it a person)."""
    return [{"name": c["name"], "desc": c.get("desc", ""), "voice": bool(c.get("voice")) and kind_of(c) == "person",
             **({"kind": kind_of(c)} if kind_of(c) != "person" else {})} for c in cast_of(src)]


def is_h3(src: Tree | Options) -> bool:
    """The project writes H3's official prompt formats (options "prompt_format" == "h3"); else the classic one."""
    options = _options_of(src)
    return (options or {}).get("prompt_format") == "h3"


def set_prompt_format(tree: Tree, fmt: str) -> None:
    """Switch the prompt format ("h3" | "classic"). Made clips are then CHANGED (their prompt and settings are
    written another way)."""
    if fmt not in PROMPT_FORMATS:
        raise ValueError(f"unknown prompt format {fmt!r} (h3 or classic)")
    tree.options["prompt_format"] = fmt


def all_pictures(src: Tree | Options) -> list[str]:
    """Every face picture of the cast, in cast order (what <Picture 1> ... number in a prompt that writes them)."""
    return [r for c in cast_of(src) for r in c["images"]]


def check_name(tree: Tree, name: str, keep: str | None = None) -> str:
    """A valid new name (stripped), or ValueError. keep: the character being renamed (its own name is fine)."""
    name = (name or "").strip()
    if not NAME_OK.match(name):
        raise ValueError("a name starts with a letter and has only letters, digits, - and _ (no spaces), "
                         "e.g. Anna or Anna_B")
    other = character(tree, name)
    if other is not None and (keep is None or other["name"].lower() != keep.lower()):
        raise ValueError(f"there is already a character called {other['name']}")
    return name


def _without(entry: CastEntry, key: str) -> CastEntry:
    """The entry without one key (a TypedDict cannot be rebuilt from a filtered .items() for the checker)."""
    return _cast(CastEntry, {k: v for k, v in entry.items() if k != key})


def _set_cast(tree: Tree, cast: list[CastEntry]) -> None:
    tree.options["cast"] = cast


def source_of(rel: str) -> str:
    """Where the full picture a face picture was cropped from is kept (refs/ref1.png -> refs/ref1_full.png)."""
    return rel[:-4] + "_full.png" if rel.endswith(".png") else rel + "_full.png"


def _write_picture(project: Path, image: Path, rel: str, crop: tuple[int, int, int] | None) -> None:
    """The face picture rel from image, cropped to the square crop (x, y, side) if given; the full picture is kept
    beside it (source_of) for cropping again."""
    full = Path(project) / source_of(rel)
    if crop:
        made = Path(image).resolve() != full.resolve()
        if made:
            chain.to_png(Path(image), full)
        x, y, side = (int(v) for v in crop)
        try:
            chain.to_png(full, Path(project) / rel, (x, y, side, side))
        except BaseException:
            if made:
                full.unlink(missing_ok=True)
            raise
    else:
        chain.to_png(Path(image), Path(project) / rel)
        chain.keep_aside(project, full)


def recrop_picture(project: Path, tree: Tree, rel: str, crop: tuple[int, int, int] | None) -> None:
    """Crop a face picture again from its full picture (crop None: the whole full picture). A picture added
    uncropped is its own full picture."""
    if rel not in all_pictures(tree):
        raise ValueError(f"{rel} is no face picture of the cast")
    full = Path(project) / source_of(rel)
    if not full.is_file():                          # added uncropped (or before cropping existed): it is the full one
        shutil.copyfile(Path(project) / rel, full)
    if crop:
        x, y, side = (int(v) for v in crop)
        chain.to_png(full, Path(project) / rel, (x, y, side, side))
    else:
        chain.to_png(full, Path(project) / rel)


def _copy_picture(project: Path, image: Path, tree: Tree, crop: tuple[int, int, int] | None = None) -> str:
    if len(all_pictures(tree)) >= MAX_CAST_PICTURES:
        raise ValueError(f"at most {MAX_CAST_PICTURES} face pictures in a project")
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = set(all_pictures(tree))
    k = 1
    while (d / f"ref{k}.png").exists() or f"{REFS_DIR}/ref{k}.png" in used:
        k += 1
    rel = f"{REFS_DIR}/ref{k}.png"
    _write_picture(project, image, rel, crop)
    return rel


def _check_kind(kind: str) -> str:
    if kind not in CAST_KINDS:
        raise ValueError(f"unknown kind {kind!r} (one of {', '.join(CAST_KINDS)})")
    return kind


def add_character(project: Path, tree: Tree, name: str, image: Path | None = None, desc: str = "",
                  crop: tuple[int, int, int] | None = None, kind: str = "person") -> CastEntry:
    """A new cast entry (with a first picture; a person's face picture is cropped to crop = x, y, side, any other
    kind takes the whole picture). The first person of a cast is the lead."""
    name = check_name(tree, name)
    kind = _check_kind(kind)
    lead = kind == "person" and not any(kind_of(c) == "person" for c in cast_of(tree))
    images: list[str] = []
    entry: CastEntry = {"name": name, "images": images, "desc": (desc or "").strip(), "lead": lead}
    if kind != "person":
        entry["kind"] = kind
        crop = None
    if image is not None:
        images.append(_copy_picture(project, image, tree, crop))
    _set_cast(tree, cast_of(tree) + [entry])
    return entry


def put_entry(project: Path, tree: Tree, name: str, faces: Sequence[tuple[Path, Path | None]], desc: str = "",
              replace: bool = False, kind: str = "person") -> CastEntry:
    """A ready cast entry (e.g. of the cast library) joins the cast as it is: faces are its pictures (a person's
    face pictures) with the full pictures they were cropped from (None: the picture is its own), copied into
    refs/. replace: an entry of that name is replaced in its place (a person stays the lead if it was; the old
    pictures and voice go to old/); else the name must be free. A person is the lead when the cast has no other
    person. Nothing is changed on an error. Returns the entry (a person without a voice: put_voice gives it one)."""
    kind = _check_kind(kind)
    old = character(tree, name) if replace else None
    name = check_name(tree, name, keep=old["name"] if old else None)
    if len(faces) > MAX_PICTURES_PER_CHARACTER:
        raise ValueError(f"at most {MAX_PICTURES_PER_CHARACTER} pictures per character")
    others = [r for c in cast_of(tree) if c is not old for r in c["images"]]
    if len(others) + len(faces) > MAX_CAST_PICTURES:
        raise ValueError(f"at most {MAX_CAST_PICTURES} pictures in a project (@{name} would make "
                         f"{len(others) + len(faces)})")
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = set(all_pictures(tree))
    images: list[str] = []
    made: list[Path] = []
    k = 1
    try:
        for face, full in faces:
            while ((d / f"ref{k}.png").exists() or (d / f"ref{k}_full.png").exists()
                   or f"{REFS_DIR}/ref{k}.png" in used):
                k += 1
            rel = f"{REFS_DIR}/ref{k}.png"
            made.append(Path(project) / rel)
            shutil.copyfile(face, Path(project) / rel)
            if full is not None:
                made.append(Path(project) / source_of(rel))
                shutil.copyfile(full, Path(project) / source_of(rel))
            images.append(rel)
            k += 1
    except BaseException:
        for f in made:
            f.unlink(missing_ok=True)
        raise
    no_person = not any(kind_of(c) == "person" for c in cast_of(tree) if c is not old)
    person = kind == "person"
    entry: CastEntry = {"name": name, "images": images, "desc": (desc or "").strip(),
                        "lead": person and (bool(old and old.get("lead")) or no_person)}
    if not person:
        entry["kind"] = kind
    if old is None:
        _set_cast(tree, cast_of(tree) + [entry])
        return entry
    _set_cast(tree, [entry if x is old else x for x in cast_of(tree)])
    for rel in old["images"]:
        _drop_file(project, rel, tree)
    if old.get("voice"):
        _drop_voice(project, old["voice"], tree)
    return entry


def next_character_name(tree: Tree) -> str:
    k = 1
    while character(tree, f"Person{k}") is not None:
        k += 1
    return f"Person{k}"


def add_picture(project: Path, tree: Tree, name: str, image: Path,
                crop: tuple[int, int, int] | None = None) -> str:
    """Copy an image into refs/ as one more face picture of a character (cropped to the square crop = x, y, side;
    the full picture is kept for cropping again); returns its name in the project."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    if len(c["images"]) >= MAX_PICTURES_PER_CHARACTER:
        raise ValueError(f"at most {MAX_PICTURES_PER_CHARACTER} pictures per character")
    rel = _copy_picture(project, image, tree, crop if kind_of(c) == "person" else None)   # (others: whole)
    _set_cast(tree, [{**x, "images": x["images"] + [rel]} if x is c else x for x in cast_of(tree)])
    return rel


def add_reference(project: Path, image: Path, tree: Tree, name: str | None = None,
                  crop: tuple[int, int, int] | None = None) -> str:
    """A face picture for a character (name), else for the lead, else the first person, else a new PersonN."""
    target = character(tree, name) if name else (lead_of(tree) or next((c for c in cast_of(tree)
                                                                         if kind_of(c) == "person"), None))
    if name and target is None:
        raise ValueError(f"no character called {name}")
    if target is None:
        return add_character(project, tree, next_character_name(tree), image, crop=crop)["images"][0]
    return add_picture(project, tree, target["name"], image, crop)


def _drop_file(project: Path, rel: str, tree: Tree) -> None:
    if rel in all_pictures(tree):
        return                                     # another character still uses it
    for f in (rel, source_of(rel)):
        retire(project, REFS_DIR, f)                # (never deleted: moved to old/)


def remove_picture(project: Path, tree: Tree, rel: str) -> None:
    _set_cast(tree, [{**c, "images": [r for r in c["images"] if r != rel]} for c in cast_of(tree)])
    _drop_file(project, rel, tree)


def remove_character(project: Path, tree: Tree, name: str) -> None:
    c = character(tree, name)
    if c is None:
        return
    _set_cast(tree, [x for x in cast_of(tree) if x is not c])
    for rel in c["images"]:
        _drop_file(project, rel, tree)
    if c.get("voice"):
        _drop_voice(project, c["voice"], tree)


def voices_in_use(src: Tree | Options) -> list[str]:
    return [c["voice"]["file"] for c in cast_of(src) if c.get("voice")]


def clone_file(voice: Voice) -> str:
    """The file text to speech clones a voice from: its longer clone reference when it has one, else the voice."""
    ref = voice.get("ref")
    return (ref.get("file") if ref else None) or voice["file"]


def clone_secs(voice: Voice) -> float:
    ref = voice.get("ref")
    return float((ref or voice).get("secs") or 0)


def _drop_voice(project: Path, voice: Voice, tree: Tree) -> None:
    """A voice's files (the voice and its clone reference) that nothing uses any more go to old/ (never deleted)."""
    kept = {a["file"] for a in audio_refs(tree)}
    for c in cast_of(tree):
        if c.get("voice"):
            kept |= {c["voice"]["file"], clone_file(c["voice"])}
    for rel in {voice["file"], clone_file(voice)} - kept:
        retire(project, REFS_DIR, rel)


def set_voice(project: Path, tree: Tree, name: str, source: Path, start: float = 0.0,
              label: str | None = None, end: float | None = None) -> dict:
    """Copy the sound of an audio or video file into refs/ as the voice of a character (from start, at most
    VOICE_MAX_SECS, not past end); it replaces the character's old voice. Returns the entry, with "cut": the source
    was longer.
    A clip uses the voice when the character says a line in it ("..." after @Name). Of a longer source, up to
    VOICE_REF_MAX_SECS (from the same start) are kept next to it as the clone reference ("ref"): text to speech
    clones the voice from that, the clip model still hears the short voice."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    if kind_of(c) != "person":
        raise ValueError(f"@{c['name']} is a {kind_of(c)}; only a person can have a voice")
    length = chain.audio_seconds(Path(source))
    if length is None:
        raise ValueError(f"{Path(source).name} has no sound")
    cut_end = end is not None and float(end) < length - 0.01
    have = (float(end) if cut_end and end is not None else length) - float(start)
    if have < AUDIO_REF_MIN:
        raise ValueError(f"a voice needs at least {AUDIO_REF_MIN:.0f} s of speech ({Path(source).name} has "
                         f"{max(have, 0):.1f} s" + (" in the chosen part)" if cut_end or start else ")"))
    secs = min(have, VOICE_MAX_SECS)
    k = _free_voice_slot(project, tree)
    rel = f"{REFS_DIR}/voice{k}.wav"
    made_files = [Path(project) / rel]           # (a failure after the first file takes the new files with it)
    try:
        chain.to_wav(Path(source), Path(project) / rel, start, secs if have > secs or cut_end else None)
        real = chain.audio_seconds(Path(project) / rel) or secs
        entry: Voice = {"file": rel, "name": label or Path(source).name, "secs": round(real, 2)}
        if have >= secs + 1.0:                          # (worth it: at least a second more to clone from)
            long = min(have, VOICE_REF_MAX_SECS)
            ref = f"{REFS_DIR}/voice{k}_ref.wav"
            made_files.append(Path(project) / ref)
            chain.to_wav(Path(source), Path(project) / ref, start, long if have > long or cut_end else None)
            entry["ref"] = {"file": ref, "secs": round(chain.audio_seconds(Path(project) / ref) or long, 2)}
    except BaseException:
        for f in made_files:
            f.unlink(missing_ok=True)
        raise
    old = c.get("voice")
    _set_cast(tree, [{**x, "voice": entry} if x is c else x for x in cast_of(tree)])
    if old:
        _drop_voice(project, old, tree)
    return {**entry, "cut": have > secs + 0.01}


def _free_voice_slot(project: Path, tree: Tree) -> int:
    """The first k with refs/voice{k}.wav neither on disk nor in use (refs/ is made)."""
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = set(voices_in_use(tree))
    k = 1
    while (d / f"voice{k}.wav").exists() or f"{REFS_DIR}/voice{k}.wav" in used:
        k += 1
    return k


def put_voice(project: Path, tree: Tree, name: str, voice: Path, ref: Path | None = None, label: str = "",
              text: str = "") -> Voice:
    """A ready voice (e.g. one of the voice library) becomes a character's voice as it is: voice (the short sample
    the clip model hears) and ref (its longer clone reference, if any) are copied into refs/, text is what the
    clone file says (no new transcript then). It replaces the character's old voice. Returns the entry."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    if kind_of(c) != "person":
        raise ValueError(f"@{c['name']} is a {kind_of(c)}; only a person can have a voice")
    k = _free_voice_slot(project, tree)
    rel = f"{REFS_DIR}/voice{k}.wav"
    made_files = [Path(project) / rel]
    try:
        shutil.copyfile(voice, Path(project) / rel)
        entry: Voice = {"file": rel, "name": label or Path(voice).name,
                        "secs": round(chain.audio_seconds(Path(project) / rel) or 0.0, 2)}
        if ref is not None:
            ref_rel = f"{REFS_DIR}/voice{k}_ref.wav"
            made_files.append(Path(project) / ref_rel)
            shutil.copyfile(ref, Path(project) / ref_rel)
            entry["ref"] = {"file": ref_rel, "secs": round(chain.audio_seconds(Path(project) / ref_rel) or 0.0, 2)}
        if text.strip():
            entry["text"] = text.strip()
    except BaseException:
        for f in made_files:
            f.unlink(missing_ok=True)
        raise
    old = c.get("voice")
    _set_cast(tree, [{**x, "voice": entry} if x is c else x for x in cast_of(tree)])
    if old:
        _drop_voice(project, old, tree)
    return entry


def remove_voice(project: Path, tree: Tree, name: str) -> None:
    c = character(tree, name)
    if c is None or not c.get("voice"):
        return
    _set_cast(tree, [_without(x, "voice") if x is c else x for x in cast_of(tree)])
    _drop_voice(project, c["voice"], tree)


def update_character(tree: Tree, name: str, *, desc: str | None = None, lead: bool | None = None) -> None:
    """Change a character's description, or make it the lead (lead=False: no lead)."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    if lead and kind_of(c) != "person":
        raise ValueError(f"@{c['name']} is a {kind_of(c)}; only a person can be the lead")
    cast: list[CastEntry] = []
    for x in cast_of(tree):
        x = CastEntry(**x)
        if x["name"] == c["name"]:
            if desc is not None:
                x["desc"] = desc.strip()
            if lead is not None:
                x["lead"] = bool(lead)
        elif lead:
            x["lead"] = False                       # one lead only
        cast.append(x)
    _set_cast(tree, cast)


def set_kind(tree: Tree, name: str, kind: str) -> None:
    """Make a cast entry a person, place, object or style. Not a person any more: it stops being the lead (a clip
    that mentions nobody then gets no lead pictures); one with a voice cannot change (remove the voice first). Its
    pictures stay as they are (recrop_picture(rel, None) gives a cropped one its whole picture again)."""
    c = character(tree, name)
    if c is None:
        raise ValueError(f"no character called {name}")
    kind = _check_kind(kind)
    if kind != "person" and c.get("voice"):
        raise ValueError(f"@{c['name']} has a voice; only a person can have one (remove the voice first)")
    cast: list[CastEntry] = []
    for x in cast_of(tree):
        if x["name"] == c["name"]:
            x = _without(x, "kind")
            if kind != "person":
                x["kind"] = kind
                x["lead"] = False
        cast.append(x)
    _set_cast(tree, cast)


def rename_character(tree: Tree, old: str, new: str) -> list[str]:
    """Rename a character and every @old in the clips' prompts (and the style text). Returns the clips changed."""
    c = character(tree, old)
    if c is None:
        raise ValueError(f"no character called {old}")
    new = check_name(tree, new, keep=c["name"])

    def swap(text: str) -> str:
        return MENTION.sub(lambda m: f"@{new}" if m.group(1).lower() == c["name"].lower() else m.group(0),
                           text or "")
    changed = []
    for n in tree.nodes.values():
        if not n.is_video and swap(n.prompt) != n.prompt:
            n.prompt = swap(n.prompt)
            changed.append(n.id)
    style = dict(tree.options.get("style") or {})
    for k in ("prefix", "suffix"):
        if style.get(k):
            style[k] = swap(style[k])
    tree.options["style"] = style
    _set_cast(tree, [{**x, "name": new} if x["name"] == c["name"] else x for x in cast_of(tree)])
    return changed


def _in_lines(text: str):
    """pos -> True when it lies inside a quoted line (an @Name there is a spoken word, not a mention)."""
    spans = [(m.start(), m.end()) for m in DIALOGUE.finditer(text or "")]
    return lambda pos: any(a < pos < b for a, b in spans)


def _unmention(line: str) -> str:
    """A spoken line with its @Names as plain names (nobody says the @)."""
    return MENTION.sub(lambda m: m.group(1), line)


def mentions(options: Options, text: str) -> tuple[list[CastEntry], list[str]]:
    """The characters a prompt mentions as @Name, in the order they first appear, and the @words that name
    nobody in the cast. An @Name inside a quoted line is a spoken word, not a mention."""
    found: list[CastEntry] = []
    unknown: list[str] = []
    inside = _in_lines(text)
    for m in MENTION.finditer(text or ""):
        if inside(m.start()):
            continue
        c = character(options, m.group(1))
        if c is None:
            if m.group(1) not in unknown:
                unknown.append(m.group(1))
        elif c not in found:
            found.append(c)
    return found, unknown


def audio_refs(tree: Tree) -> list[AudioRef]:
    """The project's sound references ({"file", "name", "secs"}), <Audio 1> first."""
    return list(tree.options.get("audio_refs") or [])


def add_audio_reference(project: Path, source: Path, tree: Tree, start: float = 0.0,
                        name: str | None = None) -> dict[str, Any]:
    """Copy the sound of an audio or video file into refs/ as <Audio N> (from start, cut to what is left of the
    15 s all sounds may have together). Returns the new entry, with "cut": the source was longer."""
    refs = audio_refs(tree)
    if len(refs) >= MAX_AUDIO_REFS:
        raise ValueError(f"at most {MAX_AUDIO_REFS} sound references")
    total = sum(float(a.get("secs") or 0) for a in refs)
    left = round(AUDIO_REF_TOTAL - total, 3)
    if left < AUDIO_REF_MIN:
        raise ValueError(f"the sounds already have {total:.1f} of the {AUDIO_REF_TOTAL:.0f} s MiniMax H3 takes; "
                         "remove one first")
    length = chain.audio_seconds(Path(source))
    if length is None:
        raise ValueError(f"{Path(source).name} has no sound")
    have = length - float(start)
    if have < AUDIO_REF_MIN:
        raise ValueError(f"a sound reference needs at least {AUDIO_REF_MIN:.0f} s ({Path(source).name} has "
                         f"{max(have, 0):.1f} s)")
    secs = min(have, left)
    d = Path(project) / REFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    used = {a["file"] for a in refs}
    k = 1
    while (d / f"audio{k}.wav").exists() or f"{REFS_DIR}/audio{k}.wav" in used:
        k += 1
    rel = f"{REFS_DIR}/audio{k}.wav"
    chain.to_wav(Path(source), Path(project) / rel, start, secs if have > left else None)
    real = chain.audio_seconds(Path(project) / rel) or secs
    entry: AudioRef = {"file": rel, "name": name or Path(source).name, "secs": round(real, 2)}
    tree.options["audio_refs"] = refs + [entry]
    return {**entry, "cut": have > left + 0.01}


def remove_audio_reference(project: Path, rel: str, tree: Tree) -> None:
    tree.options["audio_refs"] = [a for a in audio_refs(tree) if a["file"] != rel]
    retire(project, REFS_DIR, rel)                  # (never deleted: moved to old/)


def audio_mentioned(prompt: str) -> int:
    """The highest <Audio N> a prompt names (0: none). The model numbers its sounds in order, so a clip naming
    <Audio 2> gets <Audio 1> and <Audio 2>."""
    return max((int(m) for m in AUDIO_TAG.findall(prompt or "")), default=0)
