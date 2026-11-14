"""Camera moves for MiniMax H3 prompts, in the words of H3's own prompt guide (docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md,
"4.3 Camera Motion: Motion Type + Amplitude + Speed"): a motion type, optionally "with small / large amplitude" and
"at slow / fast speed" (medium and normal are left out), written as a natural sentence in the shot ("The camera
pushes in with small amplitude at slow speed."), not as bracket labels.

Two ways in: a clip's own camera setting ({"move", "amplitude", "speed"}: one sentence after its prompt), and
#-tokens in the prompt text (#push-in, #pan-right-fast, #push-in-small-slow ...: the same words in place).
"""
from __future__ import annotations

import re

# slug -> (label, phrase after "the camera", takes amplitude / speed)
MOVES: dict[str, tuple[str, str, bool]] = {
    "push-in": ("Push in", "pushes in", True),
    "pull-out": ("Pull out", "pulls out", True),
    "zoom-in": ("Zoom in", "zooms in", True),
    "zoom-out": ("Zoom out", "zooms out", True),
    "pan-left": ("Pan left", "pans left", True),
    "pan-right": ("Pan right", "pans right", True),
    "truck-left": ("Truck left", "trucks left", True),
    "truck-right": ("Truck right", "trucks right", True),
    "tilt-up": ("Tilt up", "tilts up", True),
    "tilt-down": ("Tilt down", "tilts down", True),
    "pedestal-up": ("Pedestal up", "pedestals up", True),
    "pedestal-down": ("Pedestal down", "pedestals down", True),
    "arc": ("Arc shot", "moves in an arc shot around the subject", True),
    "tracking": ("Tracking shot", "follows the subject in a tracking shot", True),
    "static": ("Static shot", "holds a static shot", False),
    "shake-slightly": ("Shake slightly", "shakes slightly", False),
    "shake-strongly": ("Shake strongly", "shakes strongly", False),
    "pov": ("POV", "shows the subject's point of view (POV)", False),
    "roll-cw": ("Roll clockwise", "rolls clockwise", True),
    "roll-ccw": ("Roll counterclockwise", "rolls counterclockwise", True),
}
AMPLITUDES = {"small": "with small amplitude", "large": "with large amplitude"}     # (medium: left out)
SPEEDS = {"slow": "at slow speed", "fast": "at fast speed"}                          # (normal: left out)

TOKEN = re.compile(r"(?<![\w#&/])#([A-Za-z][A-Za-z0-9-]*)")


def phrase(move: str, amplitude: str | None = None, speed: str | None = None) -> str | None:
    """'the camera pushes in with small amplitude at slow speed' (lower case; None for an unknown move)."""
    if move not in MOVES:
        return None
    _label, words, mods = MOVES[move]
    parts = ["the camera", words]
    if mods and amplitude in AMPLITUDES:
        parts.append(AMPLITUDES[amplitude])
    if mods and speed in SPEEDS:
        parts.append(SPEEDS[speed])
    return " ".join(parts)


def sentence(camera: dict | None) -> str:
    """A clip's camera setting as a sentence ('' without one)."""
    if not camera or camera.get("move") not in MOVES:
        return ""
    p = phrase(camera["move"], camera.get("amplitude"), camera.get("speed"))
    return p[0].upper() + p[1:] + "."


def parse(token: str) -> tuple[str, str | None, str | None] | None:
    """'push-in-small-slow' -> ('push-in', 'small', 'slow'); None if it is not a camera token."""
    t = token.lower()
    amplitude = speed = None
    for _ in range(2):                                  # the modifiers come last, in either order
        head, _sep, last = t.rpartition("-")
        if head and last in SPEEDS and speed is None:
            speed, t = last, head
        elif head and last in AMPLITUDES and amplitude is None:
            amplitude, t = last, head
    if t not in MOVES or ((amplitude or speed) and not MOVES[t][2]):
        return None
    return t, amplitude, speed


def expand(text: str) -> str:
    """The #camera tokens of a prompt in H3's words, in place ('The camera ...' at the start of a sentence, else
    'the camera ...'); other #words stay as they are."""
    def sub(m: re.Match) -> str:
        p = parse(m.group(1))
        if p is None:
            return m.group(0)
        words = phrase(*p)
        before = text[:m.start()].rstrip()
        start = not before or before[-1] in ".!?]:\n"
        return words[0].upper() + words[1:] if start else words
    return TOKEN.sub(sub, text or "")


def apply(prompt: str, camera: dict | None) -> str:
    """A clip's prompt with its #tokens expanded and its camera setting as the sentence after it."""
    text = expand(prompt).strip()
    s = sentence(camera)
    if s:
        text = f"{text}{'' if not text or text[-1] in '.!?' else '.'} {s}".strip()
    return text


def tokens() -> list[str]:
    """The tokens offered while typing #: every move, with the common speed / amplitude endings."""
    out = []
    for slug, (_label, _words, mods) in MOVES.items():
        out.append(slug)
        if mods:
            out += [f"{slug}-slow", f"{slug}-fast", f"{slug}-small", f"{slug}-large", f"{slug}-small-slow",
                    f"{slug}-large-fast"]
    return out


def label(token: str) -> str | None:
    """A token in plain words for menus and the # list: 'push-in-small-slow' -> 'Push in, small, slow' (None: not a
    camera token)."""
    p = parse(token)
    if p is None:
        return None
    move, amplitude, speed = p
    return ", ".join([MOVES[move][0]] + [x for x in (amplitude, speed) if x])


def is_token(token: str) -> bool:
    return parse(token) is not None


def describe(camera: dict | None) -> str:
    """'push in · small · slow' for a clip's summary ('' without one)."""
    if not camera or camera.get("move") not in MOVES:
        return ""
    label, _words, mods = MOVES[camera["move"]]
    extra = [x for x in ((camera.get("amplitude") if mods else None), (camera.get("speed") if mods else None))
             if x in AMPLITUDES or x in SPEEDS]
    return " · ".join([label.lower()] + extra)


def clean(camera) -> dict | None:
    """A stored camera setting, normalised (None when there is none)."""
    if not isinstance(camera, dict) or camera.get("move") not in MOVES:
        return None
    out = {"move": camera["move"]}
    if MOVES[camera["move"]][2]:
        if camera.get("amplitude") in AMPLITUDES:
            out["amplitude"] = camera["amplitude"]
        if camera.get("speed") in SPEEDS:
            out["speed"] = camera["speed"]
    return out
