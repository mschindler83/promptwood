"""Spoken lines and the prompt text the model gets for a cast (voiced prompt, dialogue tags)."""
from __future__ import annotations

import re

from .. import h3_prompt as _h3
from .consts import DIALOGUE, MENTION, SAID_BY
from .cast import _in_lines, _unmention, character, kind_of


_FEMALE = re.compile(r"\b(woman|women|girl|female|lady|mother|mom|mum|wife|daughter|sister|grandmother|frau|"
                     r"mädchen|mutter)\b", re.IGNORECASE)


_MALE = re.compile(r"\b(man|men|boy|male|guy|gentleman|father|dad|husband|son|brother|grandfather|mann|junge|"
                   r"vater)\b", re.IGNORECASE)


_PRONOUN = re.compile(r"\b(she|he|sie|er)\b", re.IGNORECASE)


def gender_of(c: dict) -> str | None:
    """'f' / 'm' from a character's description (woman, man, girl ...), else None."""
    desc = c.get("desc") or ""
    f, m = bool(_FEMALE.search(desc)), bool(_MALE.search(desc))
    return "f" if f and not m else "m" if m and not f else None


def dialogue(options: dict, text: str, default: dict | None = None) -> list[dict]:
    """The spoken lines of a prompt, [{"start", "end", "line", "speaker": character or None}]. Who says a line,
    looking at the words since the sentence began (or since the line before, in the same sentence):
    - a "she" / "he" there, the last one: the last-mentioned character of that gender (from its description);
    - else the first @Name there (@Anna looks at @Ben and says "Hi." -> Anna);
    - else the first @Name after the line in its sentence ("Hi," says @Anna.);
    - else the last @Name before the line, else default.
    Only a person speaks: a place, object or style named there is passed over."""
    text = text or ""
    spans = [(m.start(), m.end(), next(g for g in m.groups() if g is not None).strip())
             for m in DIALOGUE.finditer(text)]
    masked = list(text)
    for a, b, line in spans:
        masked[a:b] = "x" * (b - a)                 # sentence ends inside a quote do not count ...
        if line[-1:] in (".", "!", "?"):
            masked[b - 1] = "."                     # ... but a line that ends one ends its sentence
    masked = "".join(masked)
    inside = _in_lines(text)
    marks = [(m.start(), character(options, m.group(1))) for m in MENTION.finditer(text) if not inside(m.start())]
    marks = [(pos, c) for pos, c in marks if c is not None and kind_of(c) == "person"]
    out, prev_end = [], 0
    for a, b, line in spans:
        if not line:
            prev_end = b
            continue
        cut = max(masked.rfind(e, 0, a) for e in (". ", "! ", "? ", "\n"))
        seg = max(cut, prev_end - 1)                # the words that lead up to this line
        after = [masked.find(e, b) for e in (". ", "! ", "? ", "\n")]
        stop = min([x for x in after if x >= 0] or [len(text)])
        who = None
        pron = [m for m in _PRONOUN.finditer(masked, seg + 1, a)]
        if pron:
            g = "f" if pron[-1].group(1).lower() in ("she", "sie") else "m"
            if pron[-1].group(1).lower() == "sie" and any(gender_of(c) == "m" for _p, c in marks):
                g = None                            # German "sie" is also "they" / "you"
            if g:
                who = next((c for pos, c in reversed(marks) if pos < pron[-1].start() and gender_of(c) == g), None)
        who = who or next((c for pos, c in marks if seg < pos < a), None) \
            or next((c for pos, c in marks if b <= pos <= stop), None) \
            or next((c for pos, c in reversed(marks) if pos < a), None) or default
        out.append({"start": a, "end": b, "line": line, "speaker": who})
        prev_end = b
    return out


def replace_lines(text: str, lines: list[str]) -> str:
    """A prompt with its spoken lines (those dialogue() gives, in order) replaced by `lines`; everything around
    them, the quotes too, stays. ValueError: another number of lines."""
    spans = []
    for m in DIALOGUE.finditer(text or ""):
        g = next(k for k in range(1, (m.lastindex or 0) + 1) if m.group(k) is not None)
        if m.group(g).strip():
            spans.append(m.span(g))
    if len(spans) != len(lines):
        raise ValueError(f"the prompt has {len(spans)} spoken lines, not {len(lines)}")
    for (a, b), line in reversed(list(zip(spans, lines))):
        text = text[:a] + line + text[b:]
    return text


def spoken_lines(text: str) -> list[str]:
    """The quoted lines of a prompt, as written (normalised spaces, no trailing , ; :)."""
    return [" ".join(next(g for g in m.groups() if g is not None).split()).rstrip(",;: ")
            for m in DIALOGUE.finditer(text or "")]


def refine_problems(original: str, refined: str) -> list[str]:
    """What a refined prompt lost of the original: spoken lines (dropped, changed or repeated) and @mentions."""
    def key(x: str) -> str:
        return re.sub(r"[^\w]+", " ", x.lower()).strip()
    before, after = spoken_lines(original), spoken_lines(refined)
    have = [key(x) for x in after]
    out = []
    for line in dict.fromkeys(before):
        n_want, n_got = [key(x) for x in before].count(key(line)), have.count(key(line))
        short = line if len(line) <= 40 else line[:39] + "…"
        if n_got == 0:
            out.append(f"the line \u201c{short}\u201d is gone or changed")
        elif n_got > n_want:
            out.append(f"the line \u201c{short}\u201d is there {n_got} times")
    gone = [f"@{m}" for m in dict.fromkeys(MENTION.findall(original or ""))
            if not re.search(rf"(?<![\w@])@{re.escape(m)}(?![\w-])", refined or "", re.IGNORECASE)]
    if gone:
        out.append(f"{', '.join(gone)} {'is' if len(gone) == 1 else 'are'} no longer mentioned")
    return out


def _picture_tags(found: list[dict]) -> tuple[dict, list[str]]:
    """<Picture> tags per character (numbered in order of their first mention) and the pictures in that order."""
    tags, images = {}, []
    for c in found:
        if c["images"]:
            first = len(images) + 1
            images += c["images"]
            tags[c["name"]] = " and ".join(f"<Picture {k}>" for k in range(first, len(images) + 1))
    return tags, images


def _voiced_prompt(options: dict, text: str, found: list[dict], lines: list[dict], n_sounds: int,
                   implicit: bool) -> tuple[str, list[str], list[str]]:
    """MiniMax's structured reference format (the only one a voice took effect with in the tests): subjects,
    <Audio N> as the voice-timbre reference of a speaker (S1, S2 ... in speaking order), each line as
    <d>[Language] ...</d>. The voices come after the n_sounds sounds the prompt names itself.
    Returns (prompt, pictures, voice files)."""
    lang = ((options.get("identity") or {}).get("language") or "English").strip() or "English"
    hint = (options.get("identity") or {}).get("hint", True)
    tags, images = _picture_tags(found)
    subj = {c["name"]: k for k, c in enumerate(found, start=1)}
    order: list[str] = []
    for d in lines:
        sp = d["speaker"]
        if sp is not None and sp["name"] in subj and sp["name"] not in order:
            order.append(sp["name"])
    sid = {n: k for k, n in enumerate(order, start=1)}
    voiced = [n for n in order if character(options, n).get("voice")]
    audio_no = {n: n_sounds + k for k, n in enumerate(voiced, start=1)}

    def token(name: str) -> str:
        return f"<Subject {subj[name]}>" + (f" (S{sid[name]})" if name in sid else "")

    def names(t: str) -> str:
        def one(m):
            c = character(options, m.group(1))
            if c is None:
                return m.group(0)
            return token(c["name"]) if c["name"] in subj else c["name"]
        return MENTION.sub(one, t)

    body, pos, said = "", 0, set()
    for d in lines:
        body += names(text[pos:d["start"]])
        sp = d["speaker"]
        end = d["end"]
        line = _unmention(d["line"].rstrip(",;: ") or d["line"])
        say = f"<d>[{lang}] {line}</d>"
        # "Hi," says @Anna.  ->  <Subject 1> (S1) says, ... <d>Hi</d>.
        tail = SAID_BY.match(text, end)
        verb = "says"
        if sp is not None and tail and (character(options, tail.group(2)) or {}).get("name") == sp["name"]:
            verb, end = tail.group(1), tail.end()
        elif tail:
            tail = None
        head = body.rstrip().rstrip(":,").rstrip()
        in_sentence = bool(head) and not tail and (head[-1].isalnum() or head[-1] in ")>")
        if sp is not None and sp["name"] in subj:
            voice = ""
            if sp["name"] in audio_no and sp["name"] not in said:
                said.add(sp["name"])
                voice = f"using the voice timbre referenced from <Audio {audio_no[sp['name']]}>, "
            if in_sentence:                         # "... and says" (the speaker leads the sentence)
                body = f"{head}, {voice}{say}"
            else:                                   # a line on its own: say who speaks
                body = f"{head} {token(sp['name'])} {verb}, {voice}{say}".lstrip()
        else:
            body += say
        pos = end
    body += names(text[pos:])

    defs = []
    for c in found:
        desc = f" ({c['desc'].rstrip('.')})" if c.get("desc") else ""
        defs.append(f"<Subject {subj[c['name']]}> is {c['name']}{desc}"
                    + (f", {_h3._DEFINES[kind_of(c)]} {tags[c['name']]}." if c["name"] in tags else "."))
    defs += [f"<Audio {k}> is a sound reference, used as the description says." for k in range(1, n_sounds + 1)]
    defs += [f"<Audio {audio_no[n]}> is the voice-timbre reference for {token(n)}, the voice of one speaker."
             for n in voiced]
    who = [f"<Subject {subj[c['name']]}>" for c in found]
    shows = who[0] if len(who) == 1 else ", ".join(who[:-1]) + " and " + who[-1]
    summary = (f"[reference generation + audio reference] The target video shows {shows}. "
               + " ".join(f"{token(n)} speaks with the voice timbre of <Audio {audio_no[n]}>." for n in voiced))
    keep = [f"<Subject {subj[c['name']]}> (appears in [Shot 1]): fully_preserved - {_kept(c)}" for c in found
            if c["name"] in tags]
    keep += [f"<Audio {audio_no[n]}>: reference - its vocal timbre guides the dialogue delivery of {token(n)} "
             "without copying the original signal or spoken words." for n in voiced]
    detail = ""
    if hint and images and _pictures_note(found):
        detail += _pictures_note(found) + "\n"
    if implicit and order:
        detail += f"{token(order[0])} is the one who speaks.\n"
    detail += f"[Shot 1] {body.strip()}"
    prompt = "\n\n".join([
        "subject_definitions:\n" + "\n".join(defs),
        "summary:\n" + summary.strip(),
        "retention_analysis:\n" + "\n".join(keep),
        "detailed_description:\n" + detail,
        "overall_soundscape:\nThe natural sound of the scene; the spoken lines are clear.",
        "non_diegetic_music:\n" + ("N/A" if not n_sounds else "As the sound references above say.")])
    return prompt, images, [character(options, n)["voice"]["file"] for n in voiced]


PICTURES_NOTE = "The pictures only show who the people are; the scene, pose and framing come from the video."


def _kept(c: dict) -> str:
    """The retention reason of a cast entry with pictures (classic structured format)."""
    k, n = kind_of(c), c["name"]
    return {"person": f"{n}'s face and identity are retained.", "place": f"the layout and look of {n} are retained.",
            "object": f"the shape and look of {n} are retained.",
            "style": f"the visual style of {n} is retained."}[k]


def _pictures_note(found: list[dict]) -> str:
    """The classic sentence that keeps face pictures from setting the scene ('' when no person has pictures; with
    places, objects or styles it speaks of the face pictures only, theirs do set the scene)."""
    shown = [c for c in found if c.get("images")]
    people = [c for c in shown if kind_of(c) == "person"]
    if not people:
        return ""
    if len(people) == len(shown):
        return PICTURES_NOTE
    return "The face pictures only show who the people are; their pose and framing come from the video."
