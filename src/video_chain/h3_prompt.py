"""MiniMax H3's official prompt format, built from a clip's inputs and the user's text (pure, standard library only).

The wording comes from H3's two prompt guides: "Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)" (the base
formats: an optional header line, then three fields) and "Full-Reference Mode Rewrite Output Format Guide" (six
sections). ComfyUI sends the text as written, so every header and label the model should see is put in here.

Which format (compose):
  base formats (MiniMaxH3ImageToVideo, fl2va model): the tokenizer labels the start frame <Picture 1> and the end
  frame <Picture 2> (an end frame alone <Picture 1>), so the header is one of the guide's lines:
    nothing          three fields, no header (T2VA)
    start frame      I2VA_HEADER, a blank line, three fields
    start + end      FL2VA_HEADER (S.SS = `seconds` with two decimals, Shot N = the last [Shot N] of the body)
    end frame only   L2VA_HEADER
  reference format (MiniMaxH3ReferenceToVideo, ref2va model): when a subject has pictures or a voice, the prompt
  names sounds (`sounds` > 0), a soundtrack is reused (`soundtrack`), the clip continues a reference video
  (`continues`), or `reference=True`. Six sections:
  subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.
  There only ref_images get <Picture N> labels; a frame pinned with MiniMaxH3AddGuide is unseen by the text
  encoder. So a start / end frame that is ALSO sent as a reference picture has a number (`frame_pictures`) and is
  defined as such ("keyframe completion"); one without a number is declared in words only.

Pass-through (is_structured): a text that already starts with a guide header line, "integrated_multimodal_description:"
or "subject_definitions:" is the user's own structured prompt and comes back unchanged (stripped). The caller should
test the user's own prompt (before the style text is put around it) and then send it as it is.

Sound sections contract (split_sound_sections, shared with the refiner): the user's text may END with lines that start
with "overall_soundscape:" and / or "non_diegetic_music:" (either order; a label alone on its line takes the lines
after it). They are cut off the body and fill those fields; a missing one gets DEFAULT_SOUNDSCAPE, and music N/A or,
when the body itself talks about music (talks_about_music), MUSIC_IN_SHOTS, so the default never contradicts it.

The contract with the caller (tree.py), compose(body, ...):
  body        the clip's text as the model should read it, BEFORE any tag is put in: the project's style prefix,
              the user's prompt (camera #tokens expanded), the style suffix (full_prompt). The user's trailing sound
              lines must be split off the user's prompt first (split_sound_sections) and handed over as
              `soundscape` / `music`, because the style suffix would otherwise follow them; compose also cuts
              trailing sound lines off `body` itself, explicit `soundscape` / `music` (not None) win.
              "[Shot 1]" is put in front unless the body has it; text before a "[Shot 1]" the body has (e.g. the
              style prefix) goes right after it (base) or into the style line before it (reference format).
              "[Shot N] At MM:SS.mmm," markers stay as written.
  lines       the spoken lines, as tree.dialogue(body) gives them: Line(start, end, text, speaker) or the dicts
              {"start", "end", "line", "speaker"} (speaker: the cast name, a cast dict with "name", or None).
              start / end are offsets into `body` of the whole quoted span ("..." or <d>...</d>). Lines after the
              cut-off sound lines are ignored. Each becomes <d>[language] text</d> (its @Names as plain names); its
              speaker gets a speaker id (S1, S2 ... in speaking order, every speaker the resolver knows) and is
              written "<Subject k> (Sx)" (reference format, a subject) or "Name (Sx)". Joining, as before:
              '@Anna looks up and says "Hi."' -> '... and says, <d>[English] Hi.</d>' (the line in its sentence);
              '"Hi," says @Anna.' -> '<Subject 1> (S1) says, <d>[English] Hi</d>.' (SAID_BY after the line names its
              speaker); a line on its own -> '<speaker> says, <d>...</d>'; a line without a speaker: just <d>...</d>.
  resolve     @word -> the cast name it means (any cast member, also one that is no subject; None: no cast
              member, the @word stays as written). Default: the subjects' names, case-insensitive. @Names of a
              subject become "<Subject k>" (reference format) / the name (base), "(Sx)" added when it speaks; at its
              first appearance a subject's description follows in commas ("<Subject 1>, a tall woman, walks").
  subjects    Subject(name, kind, desc, pictures, voice) or dicts with those keys, numbered <Subject 1> ... in the
              order given (the order of first mention; with lead_implicit the lead is first). pictures: the
              <Picture N> numbers the caller sends for it (ref_images order). voice: it has a voice reference; the
              voices are numbered after the `sounds` the prompt names, in speaking order (voice_numbers gives the
              same numbers, so the caller can send the voice files in that order).
  frame_pictures  {"start": N, "end": M}: the start / end frame is also reference picture N / M (reference
              format only; ignored in the base formats, where the tokenizer numbers them).
  start / end the clip has a start / end frame (picks the base header; in the reference format an un-numbered one is
              declared in words).
  seconds     the clip's length (FL2VA / L2VA headers need it).
  sounds      the highest <Audio N> the user's text names itself (tree.audio_mentioned).
  soundtrack  the <Audio N> that is the clip's complete audio track, reused 1:1 (its sentence comes in `extra`).
  continues   the <Video k> the clip continues (the END of the clip before, sent as a reference video, picture
              only: it adds no <Audio N>): "video continuation" in the task types, <Video k> defined as the source
              video, a retention line, and [Shot 1] opens with "The shot continues directly from the end of
              <Video k>." A start frame of such a clip IS that video's last frame, and is called so.
  lead_implicit  subjects[0] is the lead used without being mentioned: "The person in the video is <Subject 1>."
              before [Shot 1].
  triggers    LoRA trigger words: right after "[Shot 1] " (after the start-frame sentence), those the body does not
              contain yet, as "a, b. ".
  extra       sentences appended to the body in order (camera sentence, keyframe notes, soundtrack sentence ...),
              before the end-frame sentence, which ends the last shot.
  language    the language tag of the spoken lines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

I2VA_HEADER = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
FL2VA_HEADER = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
                "0.00-second mark of the target video; Picture 2 (from Shot {n}) aligns with the {s}-second mark of "
                "the target video.")
L2VA_HEADER = ("How the reference pictures align with the target video — <Picture 1> (from [Shot {n}]) aligns with "
               "the {s}-second mark of the target video.")
SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape",
            "non_diegetic_music")
FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
TASK_ORDER = ("keyframe completion", "reference generation", "video editing", "video continuation", "audio reuse",
              "audio reference")                                    # the guide's order
# a clip that continues the clip before (shown to the model as <Video k>): worded by the guide's 2.3, 4.1 and 5.3
CONTINUE_DEF = ("<Video {k}> is the source video for the target video continuation; the target video continues "
                "directly from its end.")
CONTINUE_KEEP = ("<Video {k}> (continuation source, its end): fully_preserved - the subjects, scene, motion and camera "
                 "movement at its end carry on directly into [Shot 1].")
CONTINUE_SUMMARY = "The target video continues directly from the end of <Video {k}>."
CONTINUE_SHOT = "The shot continues directly from the end of <Video {k}>."
KINDS = ("person", "place", "object", "style")
DEFAULT_SOUNDSCAPE = "Natural ambient sound that fits the scene."
MUSIC_IN_SHOTS = "As described in the shots."
NO_MUSIC = "N/A"

# the same patterns as tree.py (a test keeps them equal); copied so this module stays free of tree.py
MENTION = re.compile(r"(?<![\w@])@(\w[\w-]*)")
SAID_BY = re.compile(r"[ \t]*,?[ \t]*(says|said|asks|asked|answers|answered|replies|replied|whispers|whispered|"
                     r"shouts|shouted|calls|called|adds|added|cries|cried|murmurs|murmured|mutters|muttered|laughs|"
                     r"laughed|sagt|fragt|antwortet|ruft|flüstert)[ \t]+@(\w[\w-]*)", re.IGNORECASE)
SHOT = re.compile(r"\[Shot\s*(\d+)\]", re.IGNORECASE)
_SHOT_1 = re.compile(r"\[Shot\s*1\]", re.IGNORECASE)
_LATER_SHOT = re.compile(r"\s*(\[Shot\s*(?!1\])\d+\])", re.IGNORECASE)
_STRUCTURED = re.compile(r"(For the target video, at 0\.00 seconds into the target video|How the reference pictures "
                         r"align with the target video|integrated_multimodal_description\s*:|subject_definitions\s*:)",
                         re.IGNORECASE)
_SOUND_LABEL = re.compile(r"^[ \t]*(overall_soundscape|non_diegetic_music)[ \t]*:[ \t]*(.*)$", re.IGNORECASE)
_MUSIC = re.compile(r"\b(music\w*|songs?|score|soundtracks?|melod(y|ies|ic)|beats?)\b", re.IGNORECASE)
_POSSESSIVE = re.compile(r"['’]s\b")

_DEFINES = {"person": "the person in", "place": "the environment in", "object": "the object in",
            "style": "the visual style of"}


@dataclass(frozen=True)
class Subject:
    """A referenced subject: a cast entry (kind person / place / object / style) with the <Picture N> numbers of its
    pictures as sent to the model and whether it has a voice reference."""
    name: str
    kind: str = "person"
    desc: str = ""
    pictures: tuple[int, ...] = ()
    voice: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:                      # (a hand-built Subject: the dict path does the same)
            object.__setattr__(self, "kind", "person")


@dataclass(frozen=True)
class Line:
    """A spoken line: offsets of its whole quoted span in the body, its words, its speaker's cast name (or None)."""
    start: int
    end: int
    text: str
    speaker: str | None = None


# ----------------------------------------------------------------------------- small helpers

def is_structured(text: str) -> bool:
    """The text is already a structured H3 prompt (starts with a guide header line or a field / section label)."""
    return bool(_STRUCTURED.match((text or "").lstrip()))


def _sound_cut(text: str) -> tuple[int, str | None, str | None]:
    """Where the trailing sound lines begin (len(text) if none) and their contents (None: not given)."""
    text = text or ""
    rows, pos = [], 0
    for ln in text.split("\n"):                   # (offset, line) of every line
        rows.append((pos, ln))
        pos += len(ln) + 1
    end = len(rows)
    found: dict[str, str] = {}
    cut = len(text)
    while len(found) < 2:
        while end > 0 and not rows[end - 1][1].strip():
            end -= 1                               # empty lines between / after the sections
        i = end - 1
        while i >= 0 and not _SOUND_LABEL.match(rows[i][1]):
            i -= 1
        if i < 0:
            break
        m = _SOUND_LABEL.match(rows[i][1])
        label, own = m.group(1).lower(), m.group(2).strip()
        if label in found or (own and i != end - 1):
            break                                  # body text (a label with words of its own is one line)
        if not own and any(SHOT.match(rows[k][1].strip()) for k in range(i + 1, end)):
            break                                  # a label alone above more shots: body text, not a sound line
        found[label] = " ".join(x for x in [own] + [rows[k][1].strip() for k in range(i + 1, end)] if x)
        cut, end = rows[i][0], i
    if not found:
        return len(text), None, None
    return cut, found.get("overall_soundscape"), found.get("non_diegetic_music")


def split_sound_sections(text: str) -> tuple[str, str | None, str | None]:
    """(body, soundscape, music): the trailing "overall_soundscape: ..." / "non_diegetic_music: ..." lines cut off
    (either order, each at most once; a label alone on its line takes the lines after it). None: not given."""
    cut, scape, music = _sound_cut(text)
    return (text or "")[:cut].rstrip(), scape, music


def talks_about_music(text: str) -> bool:
    """The text mentions music (music / song / score / soundtrack / melody / beat)."""
    return bool(_MUSIC.search(text or ""))


def last_shot(text: str) -> int:
    """The number of the last [Shot N] in the text (1 without any)."""
    return max((int(n) for n in SHOT.findall(text or "")), default=1)


def header(start: bool, end: bool, seconds: float | None = None, shot: int = 1) -> str:
    """The base format's first line for a clip with a start and / or end frame ('' for neither)."""
    if start and not end:
        return I2VA_HEADER
    if not end:
        return ""
    if seconds is None:
        raise ValueError("an end frame needs the clip's seconds for the header")
    if start:
        return FL2VA_HEADER.format(n=shot, s=f"{float(seconds):.2f}")
    return L2VA_HEADER.format(n=shot, s=f"{float(seconds):.2f}")


def task_types(*, keyframe: bool = False, subjects: bool = False, reuse: bool = False, audio: bool = False,
               continuation: bool = False) -> str:
    """'[keyframe completion + reference generation + audio reference]': in the guide's order, without repeats."""
    have = {"keyframe completion": keyframe, "reference generation": subjects, "video continuation": continuation,
            "audio reuse": reuse, "audio reference": audio}
    types = [t for t in TASK_ORDER if have.get(t)] or ["reference generation"]
    return "[" + " + ".join(types) + "]"


def _subject(s) -> Subject:
    if isinstance(s, Subject):
        return s
    kind = (s.get("kind") or "person").lower()
    return Subject(name=s["name"], kind=kind if kind in KINDS else "person", desc=(s.get("desc") or "").strip(),
                   pictures=tuple(int(p) for p in s.get("pictures") or ()), voice=bool(s.get("voice")))


def _line(d) -> Line:
    if isinstance(d, Line):
        return d
    sp = d.get("speaker")
    if isinstance(sp, Mapping):
        sp = sp.get("name")
    return Line(int(d["start"]), int(d["end"]), d.get("line", d.get("text", "")), sp)


def speaker_ids(lines: Iterable) -> dict[str, int]:
    """name -> speaker id (1 = S1), in the order the speakers first speak."""
    out: dict[str, int] = {}
    for d in map(_line, lines):
        if d.speaker is not None and d.text.strip() and d.speaker not in out:
            out[d.speaker] = len(out) + 1
    return out


def voice_numbers(subjects: Iterable, lines: Iterable, sounds: int = 0) -> dict[str, int]:
    """name -> its <Audio N>: the subjects with a voice who speak, in speaking order, after the `sounds` the prompt
    names itself. The caller sends the voice files in this order."""
    voiced = {s.name for s in map(_subject, subjects) if s.voice}
    order = [n for n in speaker_ids(lines) if n in voiced]
    return {n: sounds + k for k, n in enumerate(order, start=1)}


def _listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _ends_sentence(s: str) -> bool:
    s = s.rstrip()
    while True:
        for tail in ("</d>", '"', "”", "»", ")"):
            if s.endswith(tail):
                s = s[:-len(tail)].rstrip()
                break
        else:
            break
    return s[-1:] in (".", "!", "?", "…", "。", "！", "？")


def _join(a: str, b: str) -> str:
    """Two pieces of text as consecutive sentences."""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return a or b
    return f"{a}{'' if _ends_sentence(a) else '.'} {b}"


def _unmention(line: str) -> str:
    return MENTION.sub(lambda m: m.group(1), line)


# ----------------------------------------------------------------------------- the body

def _render(text: str, cut: int, lines: list[Line], resolve: Callable[[str], str | None], token: Callable[[str], str],
            descs: Mapping[str, str], audio_no: Mapping[str, int], lang: str) -> str:
    """The body up to `cut` with @Names as tokens (a subject's description at its first appearance) and each spoken
    line as <d>[lang] ...</d>, joined into its sentence the way _voiced_prompt did."""
    described: set[str] = set()

    def named(name: str, at: int | None) -> str:
        t = token(name)
        if name in descs and name not in described and descs[name]:
            if at is not None and _POSSESSIVE.match(text, at):
                return t                            # "@Anna's": the description waits for the next mention
            described.add(name)
            nxt = text[at:cut].lstrip()[:1] if at is not None else ""
            t += f", {descs[name]}" + ("" if at is not None and (not nxt or nxt in ".,;:!?)") else ",")
        return t

    def names(a: str, b: int) -> str:
        out, last = "", a
        for m in MENTION.finditer(text, a, b):
            name = resolve(m.group(1))
            out += text[last:m.start()] + (m.group(0) if name is None else named(name, m.end()))
            last = m.end()
        return out + text[last:b]

    body, pos, said = "", 0, set()
    for d in sorted(lines, key=lambda x: x.start):
        if d.start < pos or d.end > cut:
            continue
        body += names(pos, d.start)
        sp, end = d.speaker, d.end
        if not d.text.strip():
            body += text[d.start:d.end]
            pos = end
            continue
        say = f"<d>[{lang}] {_unmention(d.text.strip().rstrip(',;: ') or d.text.strip())}</d>"
        tail = SAID_BY.match(text, end)             # "Hi," says @Anna.  ->  <Subject 1> (S1) says, <d>Hi</d>.
        verb = "says"
        if sp is not None and tail and tail.end() <= cut and resolve(tail.group(2)) == sp:
            verb, end = tail.group(1), tail.end()
        else:
            tail = None
        head = body.rstrip().rstrip(":,").rstrip()
        in_sentence = bool(head) and not tail and (head[-1].isalnum() or head[-1] in ")>")
        if sp is not None:
            voice = ""
            if sp in audio_no and sp not in said:
                said.add(sp)
                voice = f"using the voice timbre referenced from <Audio {audio_no[sp]}>, "
            if in_sentence:                         # "... and says" (the speaker leads the sentence)
                body = f"{head}, {voice}{say}"
            else:                                   # a line on its own: say who speaks
                body = f"{head} {named(sp, None)} {verb}, {voice}{say}".lstrip()
        else:
            body += say
        pos = end
    return body + names(pos, cut)


def _shot_parts(body: str) -> tuple[str, str]:
    """(text before [Shot 1], the rest after the marker)."""
    m = _SHOT_1.search(body)
    if m is None:
        return "", body.strip()
    return body[:m.start()].strip(), body[m.end():].strip()


def contains_word(text: str | None, word: str) -> bool:
    """True when `word` is in `text` as a whole word (case-insensitive): "ink" is not in "pinkish" or "ink_sketch"."""
    return bool(re.search(r"(?<!\w)" + re.escape(word.lower()) + r"(?!\w)", (text or "").lower()))


def _missing_triggers(words: Iterable[str], text: str) -> list[str]:
    return [w for w in words if w and not contains_word(text, w)]


# ----------------------------------------------------------------------------- the prompt

def compose(body: str, *, start: bool = False, end: bool = False, seconds: float | None = None,
            subjects: Iterable = (), frame_pictures: Mapping[str, int] | None = None, sounds: int = 0,
            soundtrack: int | None = None, lines: Iterable = (), language: str = "English",
            lead_implicit: bool = False, triggers: Iterable[str] = (), extra: Iterable[str] = (),
            soundscape: str | None = None, music: str | None = None,
            resolve: Callable[[str], str | None] | None = None, reference: bool | None = None,
            continues: int | None = None) -> str:
    """The prompt the model gets (see the module docstring for the contract)."""
    text = body or ""
    if is_structured(text):
        return text.strip()
    subs = [_subject(s) for s in subjects]
    lns = [_line(d) for d in lines]
    lang = (language or "").strip() or "English"
    cut, scape0, music0 = _sound_cut(text)
    lns = [d for d in lns if d.end <= cut]           # (a quote in the cut-off sound lines is no spoken line)
    scape = soundscape if soundscape is not None else scape0
    mus = music if music is not None else music0
    by_name = {s.name: s for s in subs}
    if resolve is None:
        low = {s.name.lower(): s.name for s in subs}
        resolve = lambda w: low.get(w.lower())       # noqa: E731
    cont = int(continues) if continues else None
    if reference is None or cont:                    # (a reference video needs the reference model)
        reference = bool(cont or reference or sounds or soundtrack or any(s.pictures or s.voice for s in subs))
    sid = speaker_ids(lns)
    audio_no = voice_numbers(subs, lns, sounds) if reference else {}
    subj_no = {s.name: k for k, s in enumerate(subs, start=1)}

    def token(name: str) -> str:
        t = f"<Subject {subj_no[name]}>" if reference and name in subj_no else name
        return t + (f" (S{sid[name]})" if name in sid else "")

    descs = {s.name: s.desc.rstrip(".").strip() for s in subs if s.desc}
    rendered = _render(text, cut, lns, resolve, token, descs, audio_no, lang)
    pre, rest = _shot_parts(rendered)
    user_text = text[:cut]
    missing = _missing_triggers(triggers, user_text)
    trig = ", ".join(missing)
    tail = ""
    for s in extra:
        tail = _join(tail, s)

    if not reference:
        shot1 = _join(pre, rest)
        shot1 = _join(trig, shot1) if trig else shot1
        desc = _join(shot1, tail)
        desc = f"[Shot 1] {desc}".rstrip()
        n = last_shot(desc)
        head = header(start, end, seconds, n)
        scape = scape or DEFAULT_SOUNDSCAPE
        mus = mus or (MUSIC_IN_SHOTS if talks_about_music(user_text) else NO_MUSIC)
        fields = [f"integrated_multimodal_description: {desc}", f"overall_soundscape: {scape}",
                  f"non_diegetic_music: {mus}"]
        return "\n\n".join(([head] if head else []) + fields)

    frames = dict(frame_pictures or {})
    p_start, p_end = frames.get("start"), frames.get("end")
    start, end = start or p_start is not None, end or p_end is not None
    opening = (f"The shot begins from <Picture {p_start}>." if p_start is not None
               else "The video begins exactly from the given start frame." if start else "")
    if cont:                                         # the start frame is the last frame of <Video k>: one story
        vid = f"<Video {cont}>"
        opening = _join(CONTINUE_SHOT.format(k=cont),
                        f"It begins from <Picture {p_start}>, the last frame of {vid}." if p_start is not None else
                        f"It begins exactly from the given start frame, the last frame of {vid}." if start else "")
    closing = (f"The shot ends on <Picture {p_end}>." if p_end is not None
               else "The video ends exactly on the given end frame." if end else "")
    shot1 = rest
    if trig:
        shot1 = _join(trig, shot1)
    shot1 = _join(opening, shot1)
    shot1 = _join(_join(shot1, tail), closing)
    shot1 = _LATER_SHOT.sub(lambda m: "\n" + m.group(1), f"[Shot 1] {shot1}".rstrip())
    lead_line = "The person in the video is <Subject 1>." if lead_implicit and subs else ""
    detail = "\n".join(x for x in (pre, lead_line, shot1) if x)
    n = last_shot(detail)

    def shots_of(k: int) -> list[int]:
        marks = list(SHOT.finditer(detail))
        got = []
        for i, m in enumerate(marks):
            seg = detail[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(detail)]
            if re.search(rf"<Subject {k}>", seg) and int(m.group(1)) not in got:
                got.append(int(m.group(1)))
        return got or sorted({int(m.group(1)) for m in marks}) or [1]

    defs, keep = [], []
    for s in subs:
        k = subj_no[s.name]
        d = f" ({s.desc.rstrip('.').strip()})" if s.desc else ""
        pics = _listing([f"<Picture {p}>" for p in s.pictures]) if s.pictures else ""
        defs.append(f"<Subject {k}> is {s.name}{d}" + (f", {_DEFINES[s.kind]} {pics}." if pics else "."))
        if s.pictures:
            reason = {"person": f"{s.name}'s face and identity are retained.",
                      "place": f"the layout and look of {s.name} are retained.",
                      "object": f"the shape and look of {s.name} are retained.",
                      "style": f"the visual style of {s.name} is retained."}[s.kind]
            where = ", ".join(f"[Shot {x}]" for x in shots_of(k))
            keep.append(f"<Subject {k}> (appears in {where}): fully_preserved - {reason}")
    if p_start is not None:
        defs.append(f"<Picture {p_start}> is the first frame of [Shot 1]"
                    + (f" and the last frame of <Video {cont}>." if cont else "."))
        keep.append(f"<Picture {p_start}> ([Shot 1] first frame): fully_preserved - the target video begins exactly "
                    "from this picture.")
    if p_end is not None:
        defs.append(f"<Picture {p_end}> is the last frame of [Shot {n}].")
        keep.append(f"<Picture {p_end}> ([Shot {n}] last frame): fully_preserved - the target video ends exactly on "
                    "this picture.")
    if cont:                                         # (the node's order: pictures, videos, then the audio)
        defs.append(CONTINUE_DEF.format(k=cont))
        keep.append(CONTINUE_KEEP.format(k=cont))
    defs += [f"<Audio {k}> is a sound reference, used as the description says." for k in range(1, sounds + 1)]
    keep += [f"<Audio {k}>: reference - it is used as the description says, without copying the original signal."
             for k in range(1, sounds + 1)]
    voiced = sorted(audio_no, key=audio_no.get)
    defs += [f"<Audio {audio_no[v]}> is the voice-timbre reference for {token(v)}, the voice of one speaker."
             for v in voiced]
    keep += [f"<Audio {audio_no[v]}>: reference - its vocal timbre guides the dialogue delivery of "
             f"<Subject {subj_no[v]}> without copying the original signal or spoken words." for v in voiced]
    if soundtrack:
        defs.append(f"<Audio {soundtrack}> is the complete audio track of the target video, reused 1:1.")
        keep.append(f"<Audio {soundtrack}>: fully_copy - <Audio {soundtrack}> is reused 1:1 as the target video's "
                    "complete final audio track.")

    shown = [f"<Subject {subj_no[s.name]}>" for s in subs if s.kind != "style"]
    styles = [f"<Subject {subj_no[s.name]}>" for s in subs if s.kind == "style"]
    summary = [task_types(keyframe=p_start is not None or p_end is not None, subjects=bool(subs),
                          reuse=bool(soundtrack), audio=bool(sounds or voiced), continuation=bool(cont))]
    it = "It" if cont else "The target video"
    if cont:                                         # (as the guide begins an edit's summary with its source)
        summary.append(CONTINUE_SUMMARY.format(k=cont))
    summary.append(f"{it} shows {_listing(shown)}." if shown else f"{it} follows the detailed description.")
    if styles:
        summary.append(f"It is in the visual style of {_listing(styles)}.")
    summary += [f"{token(v)} speaks with the voice timbre of <Audio {audio_no[v]}>." for v in voiced]
    if soundtrack:
        summary.append(f"<Audio {soundtrack}> is reused 1:1 as its complete audio track.")
    if p_start is not None:
        summary.append(f"It begins from <Picture {p_start}>" + (f", the last frame of <Video {cont}>." if cont else "."))
    if p_end is not None:
        summary.append(f"It ends on <Picture {p_end}>.")

    scape = scape or DEFAULT_SOUNDSCAPE              # (a sound reference may be the music: never "N/A" then)
    mus = mus or (MUSIC_IN_SHOTS if sounds or soundtrack or talks_about_music(user_text) else NO_MUSIC)
    return "\n\n".join([
        "subject_definitions:\n" + ("\n".join(defs) or "N/A"),
        "summary:\n" + " ".join(summary),
        "retention_analysis:\n" + ("\n".join(keep) or "N/A"),
        "detailed_description:\n" + detail,
        "overall_soundscape:\n" + scape,
        "non_diegetic_music:\n" + mus])
