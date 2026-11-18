"""Character sheets in the cast: a person's pictures can be a head sheet and a full-body sheet (each front, side and
three-quarter view), made from one source photo that is kept (to make them again) but never sent as a picture.
Pictures without a role, and every older project and cast library, stay exactly as they were."""
import shutil

import pytest
from PIL import Image

from video_chain import cast_library as CL
from video_chain import tree as T


def _picture(path, colour, size=(160, 90)):
    Image.new("RGB", size, colour).save(path)
    return path


def _project(tmp_path):
    p = tmp_path / "p"
    p.mkdir()
    t = T.Tree()
    T.add_character(p, t, "Anna", _picture(tmp_path / "anna.png", (200, 150, 120), (120, 160)), desc="woman",
                    crop=(10, 20, 80))
    return p, t


def _sheets(tmp_path, tag=""):
    return (_picture(tmp_path / f"head{tag}.png", (10, 200, 10)), _picture(tmp_path / f"body{tag}.png", (10, 10, 200)),
            _picture(tmp_path / f"photo{tag}.png", (200, 10, 10), (120, 160)))


def test_a_person_gets_a_head_and_a_body_sheet_from_a_kept_source_photo(tmp_path):
    p, t = _project(tmp_path)
    face = T.character(t, "Anna")["images"][0]
    head, body, photo = _sheets(tmp_path)
    T.set_sheets(p, t, "Anna", head=head, body=body, source=photo)
    anna = T.character(t, "Anna")
    assert len(anna["images"]) == 2 and anna["desc"] == "woman" and anna["lead"]
    assert T.sheet_of(anna, "head_sheet") == anna["images"][0] and T.sheet_of(anna, "body_sheet") == anna["images"][1]
    assert [T.role_of(anna, r) for r in anna["images"]] == ["head_sheet", "body_sheet"]
    assert Image.open(p / anna["images"][0]).getpixel((5, 5)) == (10, 200, 10)       # (the whole sheet, not cropped)
    src = T.sheet_source(anna)
    assert src and (p / src).is_file() and src not in T.all_pictures(t)                # (kept, never a picture)
    assert Image.open(p / src).getpixel((5, 5)) == (200, 10, 10)
    assert not (p / face).exists() and any((p / "old").rglob("*.png"))                 # (the old face: to old/)


def test_making_the_sheets_again_retires_the_old_sheets_and_source(tmp_path):
    p, t = _project(tmp_path)
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path))
    first = dict(T.character(t, "Anna"))
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path, "2"))
    anna = T.character(t, "Anna")
    assert set(anna["roles"].values()) == {"head_sheet", "body_sheet"} and len(anna["roles"]) == 2
    for rel in [r for r in first["images"] if r not in anna["images"]] + (
            [first["source"]] if first["source"] != anna["source"] else []):
        assert not (p / rel).exists()
    assert Image.open(p / T.sheet_of(anna, "head_sheet")).getpixel((1, 1)) == (10, 200, 10)


def test_plain_pictures_and_old_projects_are_unchanged(tmp_path):
    p, t = _project(tmp_path)
    T.add_character(p, t, "Ben", _picture(tmp_path / "ben.png", (1, 2, 3)))
    before = [dict(c) for c in T.cast_of(t)]
    assert all("roles" not in c and "source" not in c for c in before)
    assert [T.role_of(c, r) for c in before for r in c["images"]] == ["", ""]
    assert T.sheet_of(before[0], "head_sheet") is None and T.sheet_source(before[0]) is None
    T.save(p, t)
    assert [dict(c) for c in T.cast_of(T.load(p))] == before                     # (no new keys written)


def test_roles_and_source_survive_saving_and_broken_ones_are_dropped(tmp_path):
    p, t = _project(tmp_path)
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path))
    anna = T.character(t, "Anna")
    T.save(p, t)
    back = T.character(T.load(p), "Anna")
    assert back["roles"] == anna["roles"] and back["source"] == anna["source"]
    options = {"cast": [{"name": "X", "images": ["refs/ref1.png"], "desc": "", "lead": True,
                         "roles": {"refs/ref1.png": "head_sheet", "refs/gone.png": "body_sheet", 7: "x"},
                         "source": 5},
                        {"name": "Y", "images": ["refs/ref2.png"], "desc": "", "lead": False,
                         "roles": {"refs/ref2.png": "nonsense"}, "source": "refs/src9.png"}]}
    T._migrate_cast(options, True)
    x, y = options["cast"]
    assert x["roles"] == {"refs/ref1.png": "head_sheet"} and "source" not in x
    assert "roles" not in y and y["source"] == "refs/src9.png"


def test_a_removed_sheet_loses_its_role_and_a_new_picture_never_inherits_it(tmp_path):
    p, t = _project(tmp_path)
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path))
    head = T.sheet_of(T.character(t, "Anna"), "head_sheet")
    T.remove_picture(p, t, head)
    anna = T.character(t, "Anna")
    assert head not in anna["roles"] and T.sheet_of(anna, "head_sheet") is None
    new = T.add_picture(p, t, "Anna", _picture(tmp_path / "x.png", (5, 5, 5)))
    assert T.role_of(T.character(t, "Anna"), new) == ""


def test_removing_the_person_retires_the_source_photo(tmp_path):
    p, t = _project(tmp_path)
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path))
    src = T.sheet_source(T.character(t, "Anna"))
    T.remove_character(p, t, "Anna")
    assert not (p / src).exists()


def test_only_a_person_has_sheets_and_the_picture_limit_holds(tmp_path):
    p, t = _project(tmp_path)
    T.add_character(p, t, "Kitchen", _picture(tmp_path / "k.png", (9, 9, 9)), kind="place")
    with pytest.raises(ValueError, match="person"):
        T.set_sheets(p, t, "Kitchen", *_sheets(tmp_path))
    with pytest.raises(ValueError, match="no character"):
        T.set_sheets(p, t, "Nobody", *_sheets(tmp_path))
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path))
    T.set_kind(t, "Anna", "object")                              # (not a person: its pictures are plain pictures)
    anna = T.character(t, "Anna")
    assert [T.role_of(anna, r) for r in anna["images"]] == ["", ""]
    T.set_kind(t, "Anna", "person")                              # (and back: the sheets are sheets again)
    assert T.sheet_of(T.character(t, "Anna"), "body_sheet")


def test_the_project_picture_limit_is_kept(tmp_path):
    p, t = _project(tmp_path)
    for k in range(T.MAX_CAST_PICTURES - 1):
        T.add_character(p, t, f"P{k}", _picture(tmp_path / f"p{k}.png", (k, k, k)), kind="object")
    head, body, photo = _sheets(tmp_path)
    with pytest.raises(ValueError, match="at most"):
        T.set_sheets(p, t, "Anna", head=head, body=body, source=photo)  # (1 face -> 2 sheets: 10 > 9)
    assert len(T.character(t, "Anna")["images"]) == 1 and "roles" not in T.character(t, "Anna")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_the_cast_library_keeps_sheets_and_their_source(tmp_path):
    p, t = _project(tmp_path)
    T.set_sheets(p, t, "Anna", *_sheets(tmp_path))
    T.add_character(p, t, "Ben", _picture(tmp_path / "ben.png", (1, 2, 3)))
    lib = CL.library()
    saved = lib.save(p, t, "Sheets")
    anna, ben = saved["entries"]
    assert anna["roles"] == ["head_sheet", "body_sheet"] and anna["source"] is True
    assert "roles" not in ben and "source" not in ben                           # (plain: as before)
    q = tmp_path / "q"
    q.mkdir()
    tq = T.Tree()
    lib.bring_in(saved["id"], q, tq, [CL.Choice("Anna", "Anna"), CL.Choice("Ben", "Ben")])
    a2 = T.character(tq, "Anna")
    assert [T.role_of(a2, r) for r in a2["images"]] == ["head_sheet", "body_sheet"]
    assert Image.open(q / T.sheet_source(a2)).getpixel((1, 1)) == (200, 10, 10)
    assert "roles" not in T.character(tq, "Ben") and "source" not in T.character(tq, "Ben")


def test_put_entry_checks_the_roles(tmp_path):
    p, t = _project(tmp_path)
    face = _picture(tmp_path / "f.png", (1, 1, 1))
    with pytest.raises(ValueError, match="role"):
        T.put_entry(p, t, "Cara", [(face, None)], roles=["head_sheet", "body_sheet"])
    with pytest.raises(ValueError, match="role"):
        T.put_entry(p, t, "Cara", [(face, None)], roles=["wing"])
    assert T.character(t, "Cara") is None


def test_the_head_sheet_keeps_its_front_view_for_the_head_swap(tmp_path):
    """The head swap takes one face, never a sheet: the head sheet's front view (its left third, as the template
    has it) is kept beside it, and goes with it."""
    p, t = _project(tmp_path)
    head = tmp_path / "h.png"
    im = Image.new("RGB", (300, 90), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 100, 90))                                    # (the left third: the front view)
    im.save(head)
    _h, body, photo = _sheets(tmp_path)
    T.set_sheets(p, t, "Anna", head, body, photo)
    anna = T.character(t, "Anna")
    rel = T.sheet_of(anna, "head_sheet")
    front = p / T.front_of(rel)
    assert front.is_file() and Image.open(front).size == (100, 90)
    assert Image.open(front).getpixel((50, 45)) == (255, 0, 0)
    assert not (p / T.front_of(T.sheet_of(anna, "body_sheet"))).exists()        # (a body sheet has none)
    assert [f for _label, f in T.swap_pictures(p, t)] == [str(front)]
    T.remove_picture(p, t, rel)
    assert not front.exists()
    assert T.swap_pictures(p, t) == []                                          # (a body sheet is no face)


def test_the_head_swap_offers_each_persons_faces_with_a_label(tmp_path):
    p, t = _project(tmp_path)
    T.add_picture(p, t, "Anna", _picture(tmp_path / "a2.png", (1, 1, 1)))
    T.add_character(p, t, "Ben", _picture(tmp_path / "b.png", (2, 2, 2)))
    T.add_character(p, t, "Kitchen", _picture(tmp_path / "k.png", (3, 3, 3)), kind="place")
    T.set_sheets(p, t, "Ben", *_sheets(tmp_path))
    got = T.swap_pictures(p, t)
    anna = T.character(t, "Anna")
    ben = T.character(t, "Ben")
    assert got == [("Face of @Anna (1)", str(p / anna["images"][0])), ("Face of @Anna (2)", str(p / anna["images"][1])),
                   ("Face of @Ben (head sheet, front view)", str(p / T.front_of(T.sheet_of(ben, "head_sheet"))))]
