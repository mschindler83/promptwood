"""The cast library: a whole cast (people with faces and their full pictures, the lead, voices; places, objects and
styles with their pictures; descriptions) kept for
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
    assert [p["name"] for p in saved["entries"]] == ["Anna", "Ben", "Kitchen"]
    assert saved["entries"][2]["kind"] == "place" and "kind" not in saved["entries"][0]
    anna = saved["entries"][0]
    assert anna["faces"] == 2 and anna["lead"] and anna["voice"]["text"] == "Hello." and "voice" not in saved["entries"][1]
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
    with pytest.raises(ValueError, match="pictures in a project"):           # Ben fits, Anna (2 more) does not
        lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben"), CL.Choice("Anna", "Anna")])
    assert sorted(p.name for p in (b / "refs").iterdir()) == before and not T.character(tb, "Ben")
    with pytest.raises(ValueError, match="letter"):
        lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben Two")])
    lib.delete(saved["id"])
    assert lib.list() == [] and not (lib.dir / saved["id"]).exists()
    with pytest.raises(ValueError, match="empty"):
        lib.save(b, T.Tree(), "Nothing")


def test_a_missing_saved_file_changes_nothing(tmp_path):
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Family")
    (lib.entry_dir(saved["id"], 0) / "voice.wav").unlink()                  # Anna's voice is gone
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Zed")
    with pytest.raises(ValueError, match="missing"):
        lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben"), CL.Choice("Anna", "Anna")])
    assert [c["name"] for c in T.cast_of(tb)] == ["Zed"] and not list((b / "refs").glob("*.png"))


def test_a_person_replacing_a_place_is_the_lead_when_there_is_no_other_person(tmp_path):
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Family")
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Ben", kind="place")
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben", replace=True)])
    assert T.lead_of(tb)["name"] == "Ben"


def test_a_left_over_full_picture_is_not_taken_for_a_new_face(tmp_path):
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Family")                                       # Ben's face has no full picture
    b = tmp_path / "b"
    (b / "refs").mkdir(parents=True)
    _picture(b / "refs" / "ref1_full.png", (7, 7, 7))                        # an orphan
    tb = T.Tree()
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben")])
    rel = T.character(tb, "Ben")["images"][0]
    assert rel != "refs/ref1.png" and not (b / T.source_of(rel)).exists()


def test_library_names_ignore_case_and_a_voice_keeps_its_elevenlabs_voice_by_any_case(tmp_path):
    from video_chain import voice_library as VL
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    lib.save(a, ta, "Family")
    with pytest.raises(ValueError, match="already"):
        lib.save(a, ta, "family")
    voices = VL.VoiceLibrary(tmp_path / "voices")
    v = voices.save(a, T.character(ta, "Anna"), "Anna")
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Cleo")
    voices.use(v["id"], b, tb, "cleo")
    assert T.character(tb, "Cleo")["tts"]["eleven"]["id"] == "el1"


def test_places_objects_and_styles_are_kept_and_brought_in(tmp_path):
    a, ta = _saved_project(tmp_path)
    T.add_character(a, ta, "Bike", _picture(tmp_path / "bike.png", (200, 0, 0)), desc="a red bicycle", kind="object")
    T.add_character(a, ta, "Ink", _picture(tmp_path / "ink.png", (0, 0, 0)), kind="style")
    T.add_picture(a, ta, "Kitchen", _picture(tmp_path / "k2.png", (240, 240, 10)))
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "World")
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Bike", kind="place")                             # a name it has: replaced, as an object
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Kitchen", "Kitchen"), CL.Choice("Bike", "Bike", replace=True),
                                      CL.Choice("Ink", "Ink")])
    k = T.character(tb, "Kitchen")
    assert T.kind_of(k) == "place" and len(k["images"]) == 2 and not k["lead"] and "voice" not in k
    assert (b / k["images"][1]).read_bytes() == (a / T.character(ta, "Kitchen")["images"][1]).read_bytes()
    bike = T.character(tb, "Bike")
    assert T.kind_of(bike) == "object" and bike["desc"] == "a red bicycle" and not bike["lead"]
    assert T.kind_of(T.character(tb, "Ink")) == "style" and T.lead_of(tb) is None      # (no person: no lead)
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben")])
    assert T.lead_of(tb)["name"] == "Ben"                                    # the first person is the lead


def test_a_cast_saved_with_people_only_is_still_read(tmp_path):
    """Casts saved before places were kept list their people under "people"."""
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Old")
    import json
    data = json.loads(lib.path.read_text())
    data["casts"][0]["people"] = [p for p in data["casts"][0].pop("entries") if "kind" not in p]
    lib.path.write_text(json.dumps(data))
    assert [p["name"] for p in lib.get(saved["id"])["entries"]] == ["Anna", "Ben"]
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Ben")])
    assert T.kind_of(T.character(tb, "Ben")) == "person"


def test_replacing_the_lead_with_a_place_keeps_a_lead(tmp_path):
    """Ann is the lead, Bob a person; the saved place Kitchen replaces Ann: Bob becomes the lead (the saved lead
    when one comes in too)."""
    a, ta = _saved_project(tmp_path)
    lib = CL.CastLibrary(tmp_path / "lib")
    saved = lib.save(a, ta, "Family")
    b = tmp_path / "b"
    b.mkdir()
    tb = T.Tree()
    T.add_character(b, tb, "Ann")
    T.add_character(b, tb, "Bob")
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Kitchen", "Ann", replace=True)])
    assert T.kind_of(T.character(tb, "Ann")) == "place" and T.lead_of(tb)["name"] == "Bob"
    T.update_character(tb, "Bob", lead=True)
    lib.bring_in(saved["id"], b, tb, [CL.Choice("Ben", "Bob", replace=True), CL.Choice("Anna", "Anna")])
    assert T.lead_of(tb)["name"] == "Bob"                                    # (Bob's place, a person again: lead)
