"""LoRA trigger words: from the file, from CivitAI (faked here), the database, and the prompt."""
import json
import os
import time

import numpy as np
import pytest
from safetensors.numpy import save_file

from video_chain import triggers as TR


def lora(path, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"a.lora_A.weight": np.zeros((2, 2), np.float16)}, str(path), metadata=meta)
    return path


def test_words_in_the_file(tmp_path):
    assert TR.file_words(lora(tmp_path / "a.safetensors", {"modelspec.trigger_phrase": "braces, smile"})) == \
        (["braces", "smile"], "file")
    tags = {"ss_tag_frequency": json.dumps({"10_ohwx": {"ohwx woman": 20, "smile": 7, "outdoors": 19}}),
            "ss_dataset_dirs": json.dumps({"10_ohwx": {"n_repeats": 10, "img_count": 20}})}
    assert TR.file_words(lora(tmp_path / "b.safetensors", tags)) == (["ohwx woman", "outdoors"], "tags")
    assert TR.file_words(lora(tmp_path / "c.safetensors", {"name": "x"})) == (None, "tags")
    assert TR.file_words(tmp_path / "missing.safetensors") == (None, "")


def test_database_and_background_lookup(tmp_path):
    db = TR.TriggerDB(tmp_path / "db.json")
    known = lora(tmp_path / "loras" / "braces.safetensors", {"ss_output_name": "braces"})
    named = lora(tmp_path / "loras" / "named.safetensors", {"modelspec.trigger_word": "zzz"})
    unknown = lora(tmp_path / "loras" / "own.safetensors")
    offline = lora(tmp_path / "loras" / "later.safetensors", {"x": "1"})
    asked = []

    def lookup(sha):
        asked.append(sha)
        if sha == TR.sha256_of(known):
            return {"words": ["Braces"], "model": "Braces", "url": "https://civitai.com/models/1?modelVersionId=2"}
        if sha == TR.sha256_of(offline):
            raise OSError("offline")
        return None
    assert db.info(str(known)) is None                                       # not looked at yet
    n = db.scan([str(p) for p in (known, named, unknown, offline)], lookup=lookup, gap=0)
    assert n == {"looked": 3, "found": 1, "unknown": 1, "failed": 1, "new": 4}   # the named one needs no lookup
    assert db.info(str(known))["words"] == ["Braces"] and db.info(str(known))["source"] == "civitai"
    assert db.info(str(named)) == {"words": ["zzz"], "source": "file", "model": "", "url": ""}
    assert db.info(str(unknown))["source"] == "none" and db.info(str(offline)) is None
    asked.clear()
    db2 = TR.TriggerDB(tmp_path / "db.json")                                 # kept on disk
    db2.scan([str(p) for p in (known, named, unknown, offline)], lookup=lookup, gap=0)
    assert asked == [TR.sha256_of(offline)]                                  # only the failed one again
    rec = db2.data["by_sha"][TR.sha256_of(unknown)]
    rec["checked"] = time.time() - (TR.RECHECK_DAYS + 1) * 86400            # a month later: asked again
    assert db2.needs_lookup(str(unknown))
    moved = tmp_path / "elsewhere" / "braces_copy.safetensors"              # moved / renamed: the same words
    moved.parent.mkdir()
    os.link(known, moved)
    db2.scan([str(moved)], lookup=lambda sha: pytest.fail("known by its hash"), gap=0)
    assert db2.words(str(moved)) == ["Braces"]
    db2.set_user(str(known), ["brace face"])                                 # the user's words win
    assert db2.info(str(moved)) ["words"] == ["brace face"] and db2.info(str(known))["source"] == "user"
    db2.set_user(str(known), None)
    assert db2.words(str(known)) == ["Braces"]


def test_words_before_the_prompt():
    assert TR.with_triggers("a woman smiles", ["braces", "ohwx"]) == "braces, ohwx, a woman smiles"
    assert TR.with_triggers("braces on her teeth", ["braces"]) == "braces on her teeth"      # already there
    s = "subject_definitions:\nx\n\ndetailed_description:\n[Shot 1] She smiles."
    assert "[Shot 1] braces. She smiles." in TR.with_triggers(s, ["braces"])
    assert TR.stack_words(["a", "b"], lambda p: {"a": ["X", "y"], "b": ["x", "z"]}[p]) == ["X", "y", "z"]


def test_tag_guesses_are_suggestions_and_user_words_need_no_hash(tmp_path):
    db = TR.TriggerDB(tmp_path / "db.json")
    tags = {"ss_tag_frequency": json.dumps({"d": {"1girl": 10}}), "ss_dataset_dirs": json.dumps({"d": {"img_count": 10}})}
    p = lora(tmp_path / "guess.safetensors", tags)
    db.scan([str(p)], online=False)
    info = db.info(str(p))
    assert info["words"] == [] and info["suggested"] == ["1girl"]            # not put into prompts by itself
    db.set_user(str(p), ["my word"])                                         # without hashing the file
    assert db.words(str(p)) == ["my word"] and not db.data["files"][str(p)].get("sha256")
    odd = tmp_path / "odd.safetensors"                                        # a header that is no JSON object
    odd.write_bytes((4).to_bytes(8, "little") + b"[1] ")
    assert TR.file_words(odd) == (None, "")
    assert db.scan([str(odd), str(p)], online=False)["new"] == 1             # the scan goes on
