"""One implementation each: running a workflow, the combo lists, /free, the model question."""
import pytest
from fake_comfy import FakeComfy

from video_chain import animate as A
from video_chain import comfy as C
from video_chain import imagegen as G


@pytest.fixture
def server(tmp_path):
    s = FakeComfy(tmp_path / "comfy")
    yield s
    s.close()


def test_create_and_run_outputs_go_through_run_workflow(server, tmp_path, monkeypatch):
    seen = []

    def run(self, wf):
        seen.append(type(self).__name__)
        return {"outputs": {}}
    monkeypatch.setattr(C.ComfyGenerator, "_run_workflow", run)
    cr = G.ComfyImageCreator(server.url)
    with pytest.raises(C.ComfyError, match="without the image"):
        cr.create("krea2", None, "a fox", tmp_path / "n.png", 5, "standard", 1024, 1024)
    an = A.ComfyAnimator(server.url)
    with pytest.raises(C.ComfyError, match="without the output"):
        an._run_outputs({"save": {}}, {"save": tmp_path / "o.mp4"}, [])
    assert seen == ["ComfyImageCreator", "ComfyAnimator"]


def test_run_workflow_cancels_on_a_baseexception(server, monkeypatch):
    gen = C.ComfyGenerator(server.url)
    stopped = []
    monkeypatch.setattr(gen, "interrupt", lambda: stopped.append(1))
    monkeypatch.setattr(gen, "_queue", lambda wf: None)

    def wait(pid, ws):
        raise KeyboardInterrupt
    monkeypatch.setattr(gen, "_wait", wait)
    with pytest.raises(KeyboardInterrupt):
        gen._run_workflow({})
    assert stopped == [1]


def test_combo_is_a_generator_method(server):
    gen = C.ComfyGenerator(server.url)
    assert gen._combo("UNETLoader", "unet_name")
    assert gen._combo("NoSuchNode", "x") == []


def test_free_has_one_implementation_that_can_be_used_without_a_generator(server):
    assert C.free_models(server.url) is True
    assert C.free_models(server.url, unload_models=False) is True
    assert C.free_models("http://127.0.0.1:1", timeout=1) is False
    gen = C.ComfyGenerator(server.url)
    assert gen.free() is True and gen.free_cache() is True


def test_complete_and_refine_prompt_share_the_model_question(server, monkeypatch):
    gen = C.ComfyPromptRefiner(server.url)
    gen.models = [{"file": "m.gguf"}]
    asked = []

    def ask(self, wf):
        asked.append(wf)
        return "answer"
    monkeypatch.setattr(C.ComfyPromptRefiner, "_ask_text", ask)
    monkeypatch.setattr(gen, "_begin_job", lambda: None)
    monkeypatch.setattr(C, "build_refine_prompt_workflow", lambda **k: {"gen": {"inputs": {}}})
    assert gen.complete("s", "u") == "answer"
    assert gen.refine_prompt("p", "image") == "answer"
    assert len(asked) == 2
