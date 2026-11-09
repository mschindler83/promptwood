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
    assert n == {"looked": 4, "found": 1, "unknown": 2, "failed": 1, "new": 4,      # the named one too (details)
                 "key_rejected": False}
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


def test_civitai_details_are_kept(tmp_path, monkeypatch):
    """Everything CivitAI tells is kept (version, model page, example prompts) in a file per hash; LoRAs found
    before the details were kept are asked once more; the details window shows it as text."""
    db = TR.TriggerDB(tmp_path / "db.json")
    p = lora(tmp_path / "loras" / "walk.safetensors", {"modelspec.trigger_word": "w4lk"})
    sha = TR.sha256_of(p)
    asked = []

    def get(path, timeout):
        asked.append(path)
        if path.startswith("/api/v1/model-versions/by-hash/"):
            return {"id": 7, "modelId": 3, "name": "v2", "baseModel": "Wan Video 14B t2v", "trainedWords": ["w4lk"],
                    "description": "<p>Use 0.8.</p><img src='https://x/y.jpg'>", "model": {"name": "Walk", "type": "LORA"},
                    "files": [{"name": "walk.safetensors", "sizeKB": 150000, "hashes": {"SHA256": sha.upper()}}]}
        if path == "/api/v1/models/3":
            return {"id": 3, "name": "Walk", "description": "<p>Sample: <b>a woman walks, w4lk</b></p>",
                    "tags": ["motion"], "modelVersions": [{"id": 7, "name": "v2", "baseModel": "Wan", "images": [1] * 50}]}
        if path.startswith("/api/v1/images?modelVersionId=7"):
            assert "withMeta=true" in path                       # without it CivitAI leaves the prompts out
            return {"items": [{"url": "u1", "meta": {"prompt": "a man runs, w4lk", "seed": 5, "steps": 20}},
                              {"url": "u2", "meta": {"prompt": "a man runs, w4lk"}}, {"url": "u3", "meta": None}]}
        return None
    monkeypatch.setattr(TR, "_get_json", get)
    hit = TR.civitai_lookup(sha, gap=0)
    assert hit["words"] == ["w4lk"] and hit["url"].endswith("/models/3?modelVersionId=7")
    assert hit["details"]["model"]["modelVersions"] == [{"id": 7, "name": "v2", "baseModel": "Wan"}]
    assert TR.example_prompts(hit["details"]) == [{"prompt": "a man runs, w4lk", "negative": "", "seed": 5,
                                                   "steps": 20, "cfg": None, "sampler": None, "image": "u1"}]

    db.scan([str(p)], lookup=lambda s: TR.civitai_lookup(s, gap=0), gap=0)
    assert db.info(str(p))["source"] == "file" and db.info(str(p))["base_model"] == "Wan Video 14B t2v"
    d = db.details(str(p))
    assert d["version"]["baseModel"] == "Wan Video 14B t2v" and db.details_path(sha).is_file()
    assert "lora_triggers" not in str(db.details_path(sha)) and not db.needs_lookup(str(p))

    rec = db.data["by_sha"][sha]                                     # found before details were kept: asked again
    rec["details_v"] = 1                                             # (asked before withMeta)
    assert db.needs_lookup(str(p))

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from video_chain.ui.lora_details import LoraDetailsDialog, clean_html
    assert "<img" not in clean_html(d["version"]["description"]) and "Use 0.8" in clean_html(d["version"]["description"])
    assert "file:" not in clean_html('<a title="x"href="file:///etc">x</a>')     # (no space before href)
    dirty = clean_html("<p style='background:url(file:///x)'>a</p><table background='/etc/x'><tr><td>b</td></tr></table>"
                       "<a href='javascript:x()'>c</a><a href=\"https://civitai.com/m\">d</a>")
    assert "style" not in dirty and "background" not in dirty and "javascript" not in dirty
    assert 'href="https://civitai.com/m"' in dirty
    dlg = LoraDetailsDialog(str(p), details=d)
    text = dlg.view.toPlainText()
    for want in ("Walk", "Wan Video 14B t2v", "Use 0.8.", "a woman walks, w4lk", "a man runs, w4lk", "motion",
                 "walk.safetensors"):
        assert want in text, want
    from PySide6.QtCore import QUrl
    dlg._link(QUrl("copy:0"))
    assert app.clipboard().text() == "a man runs, w4lk" and "Copied" in dlg.note.text()
    empty = LoraDetailsDialog(str(p), details={})
    assert "nothing" in empty.view.toPlainText()


def test_civitai_api_key(tmp_path, monkeypatch):
    """The key lives in the keyring (here an in-memory one), goes only into CivitAI requests as a Bearer token,
    and LoRAs looked up without it are asked once more with it."""
    sent = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self, n=-1):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        auth = req.get_header("Authorization") or req.unredirected_hdrs.get("Authorization")
        assert "Authorization" not in req.headers                   # (never a header a redirect would carry on)
        sent.append((req.full_url, auth))
        if auth == "Bearer revoked":
            raise TR.urllib.error.HTTPError(req.full_url, 401, "no", {}, None)
        return Resp(b'{"id": 1}')
    monkeypatch.setattr(TR.urllib.request, "urlopen", urlopen)
    assert TR.api_key() == ""
    TR._get_json("/api/v1/x", 5)
    assert sent[-1] == ("https://civitai.com/api/v1/x", None)
    db = TR.TriggerDB(tmp_path / "db.json")
    p = lora(tmp_path / "loras" / "a.safetensors", {"x": "1"})
    db.scan([str(p)], lookup=lambda s: {"words": ["a"], "model": "A", "url": "", "details": {"version": {}}}, gap=0)
    assert not db.needs_lookup(str(p))
    TR.set_api_key("  secret123 ")
    TR._API_KEY = None                                           # read back from the keyring
    assert TR.api_key() == "secret123" and db.needs_lookup(str(p))          # once more, with the key
    TR._get_json("/api/v1/x", 5)
    assert sent[-1] == ("https://civitai.com/api/v1/x", "Bearer secret123")
    db.scan([str(p)], lookup=lambda s: {"words": ["a"], "model": "A", "url": "", "details": {"version": {}}}, gap=0)
    assert not db.needs_lookup(str(p))
    TR.set_api_key("revoked")                                    # a rejected key: asked again without it
    assert TR._get_json("/api/v1/y", 5) == {"id": 1} and sent[-1] == ("https://civitai.com/api/v1/y", None)
    assert TR.key_rejected()
    n = len(sent)
    TR._get_json("/api/v1/z", 5)                                 # not sent again: one request, without it
    assert len(sent) == n + 1 and sent[-1][1] is None
    with pytest.raises(ValueError):
        db.details_path("../../etc/x")                           # only real hashes become file names
    monkeypatch.setenv("CIVITAI_API_KEY", "fromenv")
    assert TR.api_key() == "fromenv"
    monkeypatch.delenv("CIVITAI_API_KEY")
    TR.set_api_key("")
    TR._API_KEY = None
    assert TR.api_key() == "" and not TR.key_rejected()


def test_partial_details_keep_earlier_parts_and_gone_loras_are_not_reasked(tmp_path):
    db = TR.TriggerDB(tmp_path / "db.json")
    p = lora(tmp_path / "loras" / "x.safetensors", {"x": "1"})
    full = {"words": ["w"], "model": "M", "url": "", "details": {"version": {"baseModel": "B"}, "complete": True,
                                                                   "model": {"description": "d"}, "examples": [1]}}
    db.scan([str(p)], lookup=lambda s: full, gap=0)
    sha = TR.sha256_of(p)
    db.data["by_sha"][sha]["details_v"] = 1                      # asked again (e.g. a newer details version) ...
    part = {"words": ["w"], "model": "M", "url": "", "details": {"version": {"baseModel": "B"}, "complete": False}}
    db.scan([str(p)], lookup=lambda s: part, gap=0)              # ... rate limited: the earlier parts stay
    d = db.details(str(p))
    assert d["model"] == {"description": "d"} and d["examples"] == [1] and db.needs_lookup(str(p))
    db.scan([str(p)], lookup=lambda s: None, gap=0)              # deleted on CivitAI: not asked on every scan
    assert not db.needs_lookup(str(p))
