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
    from video_chain.ui.lora_details import clean_html, details_html
    assert "<img" not in clean_html(d["version"]["description"]) and "Use 0.8" in clean_html(d["version"]["description"])
    assert "file:" not in clean_html('<a title="x"href="file:///etc">x</a>')     # (no space before href)
    dirty = clean_html("<p style='background:url(file:///x)'>a</p><table background='/etc/x'><tr><td>b</td></tr></table>"
                       "<a href='javascript:x()'>c</a><a href=\"https://civitai.com/m\">d</a>")
    assert "style" not in dirty and "background" not in dirty and "javascript" not in dirty
    assert 'href="https://civitai.com/m"' in dirty
    page = details_html(d, TR.example_prompts(d))
    for want in ("Walk", "Wan Video 14B t2v", "Use 0.8.", "a woman walks, w4lk", "a man runs, w4lk", "motion",
                 "walk.safetensors", "copy:0"):
        assert want in page, want
    assert "a man runs" not in details_html(d)       # (the LoRA window shows the prompts with their pictures)


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


def test_the_whole_gallery_is_read(monkeypatch):
    """The gallery of a LoRA's page (everyone's pictures, of all time) is read page by page; kept are the pictures
    that tell their prompt, each prompt once, without whole workflows in their generation data."""
    asked = []
    pages = {None: ([{"url": "a", "meta": {"prompt": "one", "comfy": "x" * 9000, "steps": 20}}, {"url": "b", "meta": None},
                     {"url": "c", "meta": {"prompt": "one"}}, "broken"], "feed:2"),
             "feed:2": ([], "c|3"),                                  # (an empty page can still have a next one)
             "c|3": ([{"url": "d", "meta": {"prompt": "two"}}, {"url": "e", "meta": {"seed": 1}}], None)}

    def get(path, timeout):
        asked.append(path)
        cursor = TR.urllib.parse.unquote(path.split("&cursor=")[1]) if "&cursor=" in path else None
        items, nxt = pages[cursor]
        return {"items": items, "metadata": {"nextCursor": nxt} if nxt else {}}
    monkeypatch.setattr(TR, "_get_json", get)
    items, looked, whole = TR.gallery(7, gap=0)
    assert [i["url"] for i in items] == ["a", "d"] and looked == 6 and whole and len(asked) == 3
    assert items[0]["meta"] == {"prompt": "one", "steps": 20}                    # (the workflow is left out)
    for want in ("modelVersionId=7", "limit=200", "period=AllTime", "withMeta=true", "sort=Most%20Reactions"):
        assert want in asked[0], want                # (without period CivitAI gives only the last year)
    assert asked[2].endswith("&cursor=c%7C3")
    assert TR.gallery(7, gap=0, want=1)[0] == items[:1]                          # enough prompts: no further page
    assert len(TR.gallery(7, gap=0, pages=2)[0]) == 1                            # the request limit

    def fails_later(path, timeout):
        if "&cursor=" in path:
            raise OSError("rate limited")
        return get(path, timeout)
    monkeypatch.setattr(TR, "_get_json", fails_later)
    items, looked, whole = TR.gallery(7, gap=0)
    assert [i["url"] for i in items] == ["a"] and not whole                      # what was read is kept

    def down(path, timeout):
        raise OSError("offline")
    monkeypatch.setattr(TR, "_get_json", down)
    with pytest.raises(OSError):
        TR.gallery(7, gap=0)


def test_the_chosen_site_is_asked_first_and_the_other_too(monkeypatch):
    """civitai.com is the default; the settings (or $VIDEO_CHAIN_CIVITAI_SITE) choose civitai.red. The chosen site
    is asked first and linked to; the other is asked when the first does not know what is asked or is down."""
    import io
    import urllib.error
    monkeypatch.delenv("VIDEO_CHAIN_CIVITAI_SITE", raising=False)
    monkeypatch.setattr(TR, "_SITE", None)
    monkeypatch.setattr(TR, "api_key", lambda: "")
    assert TR.site() == "https://civitai.com" and TR.sites() == ["https://civitai.com", "https://civitai.red"]
    asked, answers = [], {}

    class Answer:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            return json.dumps(self.data).encode()

    def urlopen(req, timeout):
        asked.append(req.full_url)
        host = req.full_url.split("/")[2]
        what = answers.get(host, 404)
        if isinstance(what, int):
            raise urllib.error.HTTPError(req.full_url, what, "x", {}, io.BytesIO(b""))
        if isinstance(what, Exception):
            raise what
        return Answer(what)
    monkeypatch.setattr(TR.urllib.request, "urlopen", urlopen)
    answers.update({"civitai.red": {"id": 9}})                       # only .red knows it
    assert TR._get_json("/api/v1/x", 5) == {"id": 9}
    assert asked == ["https://civitai.com/api/v1/x", "https://civitai.red/api/v1/x"]
    asked.clear()
    assert TR.set_site("civitai.red") == "https://civitai.red" and TR.sites()[0] == "https://civitai.red"
    assert TR._get_json("/api/v1/x", 5) == {"id": 9} and asked == ["https://civitai.red/api/v1/x"]
    answers.clear()                                                  # neither knows it: unknown, not an error
    assert TR._get_json("/api/v1/x", 5) is None
    answers.update({"civitai.red": OSError("down"), "civitai.com": {"id": 1}})
    assert TR._get_json("/api/v1/x", 5) == {"id": 1}
    answers.update({"civitai.com": 500})                             # one down, one failing: could not be asked
    with pytest.raises(OSError):
        TR._get_json("/api/v1/x", 5)
    answers.update({"civitai.com": 404})                             # one down, the other does not know: unknown
    assert TR._get_json("/api/v1/x", 5) is None
    # links go to the chosen site
    assert TR.on_site("https://civitai.com/models/3?modelVersionId=7") == "https://civitai.red/models/3?modelVersionId=7"
    assert TR.on_site("https://example.org/civitai.com/x") == "https://example.org/civitai.com/x"
    from video_chain import lora_pictures as LP
    assert LP.page_url({"id": 5}) == "https://civitai.red/images/5"
    assert TR.set_site("https://evil.example") == "https://civitai.com" and TR.set_site(None) == "https://civitai.com"
    monkeypatch.setenv("VIDEO_CHAIN_CIVITAI_SITE", "https://civitai.red/")
    assert TR.site() == "https://civitai.red"


def test_loras_unknown_on_one_site_are_asked_again(tmp_path):
    db = TR.TriggerDB(tmp_path / "db.json")
    p = lora(tmp_path / "loras" / "x.safetensors", {"x": "1"})
    db.scan([str(p)], lookup=lambda s: None, gap=0)
    assert not db.needs_lookup(str(p))
    sha = TR.sha256_of(p)
    del db.data["by_sha"][sha]["both_sites"]                         # checked when only one site was asked
    assert db.needs_lookup(str(p))
