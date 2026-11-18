"""Character sheets for the cast: a close-up head sheet and a full-body sheet of a person, each the person three times
side by side (front, profile, three-quarter view) on white. Made with Qwen-Image 2.1 from two pictures: the app's
template of that sheet (a mannequin in the three views: the layout) and the person's source photo (who it is).
No Qt; the window is ui/sheets_window.py, the cast keeps the sheets (tree.set_sheets)."""
from __future__ import annotations

from pathlib import Path

from . import media

ROLES = ("head_sheet", "body_sheet")
TEMPLATES = Path(__file__).resolve().parent / "templates"
SIZE = (1376, 768)                  # the templates' size (16:9; Qwen-Image 2.1 takes multiples of 32)
FAMILY = "qwen21"
LABELS = {"head_sheet": "Head sheet", "body_sheet": "Full-body sheet"}

DEFAULT_PROMPTS = {
    "head_sheet": (
        "A character sheet of the person from the second picture, in exactly the layout of the first picture: three "
        "close-up head and shoulders views side by side on a plain white background, on the left facing the camera "
        "(front view), in the middle in profile, on the right a three-quarter view, each at the same position, size "
        "and angle as the mannequin heads in the first picture. Each mannequin head becomes the person: their face, "
        "skin tone, eyes, eyebrows, nose, mouth, ears, hair and hairstyle exactly as in the second picture, with a "
        "neutral expression. The same person in all three views. Soft, even studio light, photorealistic, sharp "
        "focus, no text."),
    "body_sheet": (
        "A full-body character sheet of the person from the second picture, in exactly the layout of the first "
        "picture: three full-body views side by side on a plain white background, on the left facing the camera "
        "(front view), in the middle in profile, on the right a three-quarter view, each standing in the same pose, "
        "position and size as the mannequins in the first picture, from head to feet. Each mannequin becomes the "
        "person: their face, hair, skin tone and their own body build (not the mannequin's), wearing the clothes "
        "and shoes from the second picture; where it does not show them, simple everyday clothes and shoes that "
        "suit the person. The same person in the same clothes in all three views. Soft, even studio light, "
        "photorealistic, sharp focus, no text."),
}


def _role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown sheet role {role!r} (one of {', '.join(ROLES)})")
    return role


def template(role: str) -> Path:
    """The app's template of a sheet (head_sheet / body_sheet)."""
    return TEMPLATES / f"{_role(role)}.png"


def flat(image: Path, out: Path) -> Path:
    """The picture as RGB with any transparency laid on white (what the model is given)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = media.image_size(image)
    with media.part_file(out) as tmp:
        media.ffmpeg(["-y", "-f", "lavfi", "-i", f"color=white:s={w}x{h}", "-i", str(image), "-filter_complex",
                      "[0:v][1:v]overlay=format=auto,format=rgb24", "-frames:v", "1", tmp])
    return out


def make_sheet(creator, role: str, source: Path, prompt: str, out_png: Path, *, seed: int, quality: str,
               work: Path, model: str | None = None) -> dict:
    """One sheet: creator (a connected imagegen.ComfyImageCreator) makes it with Qwen-Image 2.1 at SIZE from the
    template (the first picture) and the source photo (the second). work: a folder for the flattened template.
    Returns the creator's record of it with "sheet" (the role) and "source"."""
    first = flat(template(role), Path(work) / f"template_{role}.png")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    meta = creator.create(FAMILY, model, prompt, Path(out_png), seed, quality, SIZE[0], SIZE[1], "", [],
                          [first, Path(source)])
    return {**meta, "sheet": role, "source": str(source)}
