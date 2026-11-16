"""The voice library: a character's voice kept for the app and given to a character of another project (copied in,
with its clone part, transcript and ElevenLabs voice)."""
import json
import shutil
import subprocess

import pytest

from video_chain import config, voice_library as VL
from video_chain import tree as T
from video_chain import tts as TTS

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def _project(path, *names):
    path.mkdir()
    t = T.Tree()
    t.options["cast"] = [{"name": n, "images": [], "desc": "", "lead": False} for n in names]
    return path, t


def _voiced(tmp_path, secs=12):
    src = tmp_path / "speech.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=330:duration={secs}",
                    str(src)], check=True)
    a, ta = _project(tmp_path / "a", "Anna")
    T.set_voice(a, ta, "Anna", src, label="speech.wav")
    TTS.set_voice_fields(ta, "Anna", text="Hello there.", eleven={"id": "el1", "name": "Anna (video-chain)"})
    return a, ta


def test_a_saved_voice_is_used_in_another_project(tmp_path):
    a, ta = _voiced(tmp_path)
    lib = VL.library()
    assert lib.dir == config.data_dir() / "voices" and lib.list() == []
    saved = lib.save(a, T.character(ta, "Anna"), "  Anna   warm ")
    assert saved["name"] == "Anna warm" and saved["text"] == "Hello there." and saved["ref_secs"] > saved["secs"]
    assert saved["eleven"] == {"id": "el1", "name": "Anna (video-chain)"}
    assert lib.voice_file(saved["id"]).is_file() and lib.ref_file(saved["id"]) is not None
    assert [v["name"] for v in json.loads(lib.path.read_text())["voices"]] == ["Anna warm"]

    b, tb = _project(tmp_path / "b", "Ben", "Cleo")
    (b / T.REFS_DIR).mkdir()
    (b / T.REFS_DIR / "voice1.wav").write_bytes(b"taken")                 # (a file of the project stays)
    entry = lib.use(saved["id"], b, tb, "Ben")
    ben = T.character(tb, "Ben")
    assert entry["file"] == "refs/voice2.wav" and ben["voice"]["ref"]["file"] == "refs/voice2_ref.wav"
    assert ben["voice"]["text"] == "Hello there." and ben["voice"]["name"] == "Anna warm"
    assert ben["tts"]["eleven"]["id"] == "el1" and (b / "refs/voice2_ref.wav").is_file()
    assert abs(ben["voice"]["secs"] - saved["secs"]) < 0.1
    assert T.clone_file(ben["voice"]) == "refs/voice2_ref.wav"
    assert (b / T.REFS_DIR / "voice1.wav").read_bytes() == b"taken"

    lib.delete(saved["id"])                                                 # the project keeps its copy
    assert lib.list() == [] and not (lib.dir / saved["id"]).exists()
    assert (b / "refs/voice2.wav").is_file() and (b / "refs/voice2_ref.wav").is_file()
    with pytest.raises(ValueError, match="not in the library"):
        lib.use(saved["id"], b, tb, "Cleo")


def test_a_short_voice_without_extras_replaces_the_old_one_as_a_whole(tmp_path):
    src = tmp_path / "short.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=3", str(src)],
                   check=True)
    a, ta = _project(tmp_path / "a", "Anna")
    T.set_voice(a, ta, "Anna", src)
    lib = VL.VoiceLibrary(tmp_path / "lib")
    saved = lib.save(a, T.character(ta, "Anna"), "plain")
    assert "ref_secs" not in saved and "text" not in saved and "eleven" not in saved
    assert lib.ref_file(saved["id"]) is None
    (tmp_path / "x").mkdir()
    b, tb = _voiced(tmp_path / "x")
    lib.use(saved["id"], b, tb, "Anna")                                     # Anna had a long one, a text, an EL voice
    anna = T.character(tb, "Anna")
    assert "ref" not in anna["voice"] and "text" not in anna["voice"] and "eleven" not in anna.get("tts", {})
    assert not (b / "refs/voice1.wav").exists()                             # the old one is retired (old/)


def test_names_rename_and_a_damaged_file(tmp_path):
    a, ta = _voiced(tmp_path)
    lib = VL.VoiceLibrary(tmp_path / "lib")
    one = lib.save(a, T.character(ta, "Anna"), "One")
    two = lib.save(a, T.character(ta, "Anna"), "two")
    with pytest.raises(ValueError, match="already"):
        lib.save(a, T.character(ta, "Anna"), "One")
    with pytest.raises(ValueError, match="needs a name"):
        lib.save(a, T.character(ta, "Anna"), "   ")
    assert len(list(lib.dir.iterdir())) == 4                                # two voices, the list and its lock
    with pytest.raises(ValueError, match="already"):
        lib.rename(two["id"], "One")
    lib.rename(two["id"], "Alpha")
    lib.rename(one["id"], "One")                                            # (its own name: fine)
    assert [v["name"] for v in lib.list()] == ["Alpha", "One"]
    with pytest.raises(ValueError):
        T.put_voice(a, ta, "Nobody", lib.voice_file(one["id"]))
    lib.path.write_text("{ not json")
    assert lib.list() == []
    lib.save(a, T.character(ta, "Anna"), "Fresh")                          # damaged: kept as .bak, a new list
    assert [v["name"] for v in lib.list()] == ["Fresh"] and list(lib.dir.glob("library.json.*.bak"))
