"""The prompt library (video_chain.prompt_library): prompts kept for every project, by kind and folder."""
import json

import pytest

from video_chain.prompt_library import PromptLibrary


def test_save_find_and_organize(tmp_path):
    lib = PromptLibrary(tmp_path / "lib")
    assert lib.list() == [] and lib.folders("video") == []
    a = lib.save("video", "Slow push in", "  the camera pushes in slowly  ", folder="Camera")
    b = lib.save("video", "Walk", "@Anna walks to the window")
    c = lib.save("edit", "Sheet", "a character sheet", folder="Sheets")
    assert a["text"] == "the camera pushes in slowly" and a["folder"] == "Camera"
    assert [p["name"] for p in lib.list("video")] == ["Walk", "Slow push in"]       # by folder, then name
    assert lib.folders("video") == ["Camera"] and lib.folders("edit") == ["Sheets"]
    assert lib.find("video", "camera", "SLOW PUSH IN")["id"] == a["id"]
    with pytest.raises(ValueError, match="already"):                                 # same folder and name
        lib.save("video", "slow push in", "other", folder="camera")
    lib.save("video", "Slow push in", "another text", folder="Camera", replace=True)
    assert lib.get(a["id"])["text"] == "another text" and len(lib.list("video")) == 2
    lib.save("video", "Slow push in", "fine elsewhere", folder="Close-ups")          # other folder: allowed
    lib.save("create", "Slow push in", "fine for another kind", folder="Camera")
    with pytest.raises(ValueError, match="empty"):
        lib.save("video", "x", "   ")
    with pytest.raises(ValueError, match="kind"):
        lib.save("audio", "x", "y")
    lib.update_text(b["id"], "@Anna runs")
    assert lib.get(b["id"])["text"] == "@Anna runs"
    lib.rename(b["id"], "Run")
    with pytest.raises(ValueError, match="already"):                                 # Close-ups has one
        lib.move(a["id"], "Close-ups")
    assert lib.move(b["id"], "Camera")["folder"] == "Camera"
    lib.delete(c["id"])
    assert lib.list("edit") == []
    kept = json.loads((tmp_path / "lib" / "library.json").read_text())
    assert kept["version"] == 1 and all(p["kind"] in ("video", "create") for p in kept["prompts"])


def test_folders_are_renamed_and_deleted_with_their_prompts(tmp_path):
    lib = PromptLibrary(tmp_path / "lib")
    lib.save("video", "A", "a", folder="Camera")
    lib.save("video", "B", "b", folder="Camera")
    lib.save("video", "A", "other a", folder="Moves")
    lib.save("edit", "A", "edit a", folder="Camera")                   # another kind's folder of the same name
    with pytest.raises(ValueError, match="already"):                   # both folders have an A: nothing changes
        lib.rename_folder("video", "Camera", "moves")
    assert lib.folders("video") == ["Camera", "Moves"]
    assert lib.rename_folder("video", "camera", "Shots") == 2
    assert lib.folders("video") == ["Moves", "Shots"] and lib.folders("edit") == ["Camera"]
    assert lib.delete_folder("video", "shots") == 2
    assert [p["name"] for p in lib.list("video")] == ["A"] and lib.list("edit")[0]["folder"] == "Camera"


def test_a_damaged_library_is_kept_aside(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    (d / "library.json").write_text("{not json")
    lib = PromptLibrary(d)
    assert lib.list() == []
    lib.save("video", "A", "a")
    assert [p["name"] for p in lib.list()] == ["A"] and list(d.glob("library.json.*.bak"))
