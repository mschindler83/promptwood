"""Describe: a place, object or style's description written from its pictures by the refiner's llama.cpp model
(no model, no GPU: the request is checked and the answer faked)."""
import pytest
from PIL import Image

from video_chain import llm_server as L


def _pic(tmp_path, name="k.png"):
    p = tmp_path / name
    Image.new("RGB", (64, 48), (200, 200, 0)).save(p)
    return str(p)


def test_the_request_names_the_kind_and_shows_every_picture(tmp_path):
    a, b = _pic(tmp_path), _pic(tmp_path, "k2.png")
    msgs = L.describe_messages(kind="place", name="Kitchen", pictures=[a, b], desc="sunny")
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "description of a place" in system and "furniture" in system and "pictures" in system
    assert "Leave out people" in system and "at most 40 words" in system
    assert user[0]["text"] == "Describe @Kitchen, a place. Its description so far: sunny"
    assert [p["type"] for p in user[1:]] == ["text", "image_url", "text", "image_url"]
    assert "Picture 2 of @Kitchen" in user[3]["text"]
    assert "palette" in L.describe_messages(kind="style", name="Ink", pictures=[a])[0]["content"]
    with pytest.raises(ValueError, match="person"):
        L.describe_messages(kind="person", name="Anna", pictures=[a])


def test_the_answer_is_cleaned_and_a_model_without_eyes_is_refused(tmp_path, monkeypatch):
    srv = L.LlamaServer("s", str(tmp_path / "m.gguf"), 1)
    with pytest.raises(L.LlmServerError, match="cannot see"):
        srv.describe("place", "Kitchen", [_pic(tmp_path)])
    srv = L.LlamaServer("s", str(tmp_path / "m.gguf"), 1, mmproj="mm.gguf")
    with pytest.raises(L.LlmServerError, match="no picture"):
        srv.describe("place", "Kitchen", [])
    sent = []
    monkeypatch.setattr(srv, "_complete", lambda msgs, *a, **k: (sent.append((msgs, k)),
                                                                 'The image shows: a small kitchen, white tiles.')[1])
    assert srv.describe("place", "Kitchen", [_pic(tmp_path)]) == "a small kitchen, white tiles."
    assert sent[0][1]["max_tokens"] == 200
    assert L.clean_description("This picture depicts a red bike") == "a red bike"
