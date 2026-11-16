"""The cast library: the people of a cast (faces with their full pictures, descriptions, the lead, voices) kept for
the app and brought into another project, added or replacing a character of the same name."""
import shutil
import subprocess

import pytest
from PIL import Image

from video_chain import cast_library as CL, config
from video_chain import tree as T
from video_chain import tts as TTS

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def _picture(path, colour, size=(120, 160)):
    Image.new("RGB", size, colour).save(path)
    return path


def _saved_project(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    t = T.Tree()
    T.add_character(a, t, "Anna", _picture(tmp_path / "anna.png", (200, 150, 120)), desc="woman, 30s",
                    crop=(10, 20, 80))
    T.add_picture(a, t, "Anna", _picture(tmp_path / "anna2.png", (190, 140, 110)))
    T.add_character(a, t, "Ben", _picture(tmp_path / "ben.png", (90, 90, 200)), desc="man")
    T.add_character(a, t, "Kitchen", _picture(tmp_path / "k.png", (250, 250, 0)), kind="place")
    speech = tmp_path / "speech.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=12",
                    str(speech)], check=True)
    T.set_voice(a, t, "Anna", speech)
    TTS.set_voice_fields(t, "Anna", text="Hello.", eleven={"id": "el1", "name": "Anna EL"})
    return a, t


def test_a_saved_cast_is_brought_into_another_project(tmp_path):
    a, ta = _saved_project(tmp_path)
    lib = CL.library()
    assert lib.dir == config.data_dir() / "casts"
    saved = lib.save(a, ta, "Family")
    assert [p["name"] for p in saved["people"]] == ["Anna", "Ben"]            # people only
    anna = saved["people"][0]
    assert anna["faces"] == 2 and anna["lead"] and anna["voice"]["text"] == "Hello." and "voice" not in saved["people"][1]
    faces = lib.faces(saved["id"], 0, anna)
    assert faces[0][1] is not None and faces[0][1].is_file()                 # cropped: its full picture kept

    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Cleo", _picture(tmp_path / "c.png", (0, 0, 0)))  # Cleo is the lead here
    got = lib.bring_in(saved["id"], b, tb, [CL.Choice("Anna", "Anna"), CL.Choice("Ben", "Ben")])
    assert got == ["Anna", "Ben"] and [c["name"] for c in T.cast_of(tb)] == ["Cleo", "Anna", "Ben"]
    a2 = T.character(tb, "Anna")
    assert a2["desc"] == "woman, 30s" and len(a2["images"]) == 2 and not a2["lead"]   # Cleo stays the lead
    assert (b / a2["images"][0]).read_bytes() == (a / T.character(ta, "Anna")["images"][0]).read_bytes()
    assert (b / T.source_of(a2["images"][0])).is_file()                      # Crop… works again
    assert a2["voice"]["text"] == "Hello." and a2["voice"]["ref"] and a2["tts"]["eleven"]["id"] == "el1"
    assert a2["voice"]["name"] == "Anna (Family)"
    assert len(set(T.all_pictures(tb))) == 4                                  # no picture overwritten

    c = tmp_path / "c"                                                       # an empty project: the saved lead
    c.mkdir()
    tc = T.Tree()
    lib.bring_in(saved["id"], c, tc, [CL.Choice("Ben", "Ben"), CL.Choice("Anna", "Anna")])
    assert T.lead_of(tc)["name"] == "Anna"


def test_same_names_replace_or_get_a_new_name(tmp_path):
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Family")
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Anna", _picture(tmp_path / "other.png", (1, 2, 3)), desc="mine")
    T.add_character(b, tb, "Zoe")
    mine = T.character(tb, "Anna")["images"][0]
    with pytest.raises(ValueError, match="already"):                         # same name, no choice: nothing done
        lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben"), CL.Choice("Anna", "Anna")])
    assert [c["name"] for c in T.cast_of(tb)] == ["Anna", "Zoe"] and not T.character(tb, "Ben")
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Anna", "Anna_2")])
    assert T.character(tb, "Anna")["desc"] == "mine" and T.character(tb, "Anna_2")["desc"] == "woman, 30s"
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Anna", replace=True)])   # Ben in Anna's place, as Anna
    names = [c["name"] for c in T.cast_of(tb)]
    assert names == ["Anna", "Zoe", "Anna_2"] and T.character(tb, "Anna")["desc"] == "man"
    assert T.character(tb, "Anna")["lead"] and not (b / mine).exists()       # (it was the lead; old/ has its picture)


def test_limits_are_checked_before_anything_is_copied(tmp_path):
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Family")
    with pytest.raises(ValueError, match="already"):
        lib.save(a, ta, "Family")
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    for k in range(T.MAX_CAST_PICTURES - 2):
        T.add_character(b, tb, f"P{k}", _picture(tmp_path / f"p{k}.png", (k, k, k)))
    before = sorted(p.name for p in (b / "refs").iterdir())
    with pytest.raises(ValueError, match="face pictures"):                   # Ben fits, Anna (2 more) does not
        lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben"), CL.Choice("Anna", "Anna")])
    assert sorted(p.name for p in (b / "refs").iterdir()) == before and not T.character(tb, "Ben")
    with pytest.raises(ValueError, match="letter"):
        lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben Two")])
    lib.delete(saved["id"])
    assert lib.list() == [] and not (lib.dir / saved["id"]).exists()
    empty = T.Tree()
    T.add_character(b, empty, "Kitchen2", kind="place")
    with pytest.raises(ValueError, match="no people"):
        lib.save(b, empty, "Places")
