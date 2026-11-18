"""Making character sheets (sheets.py): the app's templates (a mannequin in front, side and three-quarter view), the
default prompts and one sheet made with Qwen-Image 2.1 from a template and the person's source photo."""
import tomllib
from pathlib import Path

from PIL import Image

from video_chain import sheets as S
from video_chain import tree as T


def test_the_templates_come_with_the_app_at_the_sheet_size():
    assert set(S.ROLES) == set(T.PICTURE_ROLES)
    for role in S.ROLES:
        f = S.template(role)
        assert f.is_file() and Path(f).parent.name == "templates"
        assert Image.open(f).size == S.SIZE
    assert S.SIZE[0] % 32 == 0 and S.SIZE[1] % 32 == 0                        # (Qwen-Image 2.1: multiples of 32)
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["video_chain"]
    assert "templates/*.png" in data                                           # (installed with the package)


def test_a_template_is_flattened_on_white(tmp_path):
    rgba = tmp_path / "t.png"
    im = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    im.putpixel((1, 1), (10, 20, 30, 255))
    im.save(rgba)
    out = S.flat(rgba, tmp_path / "flat.png")
    got = Image.open(out)
    assert got.mode == "RGB" and got.getpixel((5, 5)) == (255, 255, 255) and got.getpixel((1, 1)) == (10, 20, 30)


def test_the_default_prompts_name_both_pictures_and_the_three_views():
    for role in S.ROLES:
        text = S.DEFAULT_PROMPTS[role]
        low = text.lower()
        assert "first picture" in low and "second picture" in low
        assert all(w in low for w in ("front", "profile", "three-quarter"))
    assert "body build" in S.DEFAULT_PROMPTS["body_sheet"].lower()            # (not the male mannequin's)


class _Creator:
    def __init__(self):
        self.calls = []

    def create(self, family, model, prompt, out_png, seed, quality, width, height, negative="", loras=(), refs=(),
               ref_strength=0.65, quality_tags=True, sampling=None):
        self.calls.append(dict(family=family, prompt=prompt, seed=seed, quality=quality, size=(width, height),
                               refs=[Path(r) for r in refs], loras=list(loras)))
        Image.new("RGB", (width, height), (1, 2, 3)).save(out_png)
        return {"mode": "create", "family": family}


def test_one_sheet_is_made_from_its_template_and_the_source_photo(tmp_path):
    photo = tmp_path / "photo.png"
    Image.new("RGB", (300, 400), (200, 100, 50)).save(photo)
    c = _Creator()
    out = tmp_path / "out" / "head.png"
    meta = S.make_sheet(c, "head_sheet", photo, "my prompt", out, seed=7, quality="fast", work=tmp_path / "w")
    call = c.calls[0]
    assert call["family"] == "qwen21" and call["size"] == S.SIZE and call["prompt"] == "my prompt"
    assert call["seed"] == 7 and call["quality"] == "fast" and call["loras"] == []
    first, second = call["refs"]
    assert Image.open(first).mode == "RGB" and Image.open(first).size == S.SIZE   # (the flattened template)
    assert second == photo
    assert out.is_file() and meta["sheet"] == "head_sheet" and meta["source"] == str(photo)


def test_an_unknown_role_is_refused(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="role"):
        S.template("face")
