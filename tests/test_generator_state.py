"""The types of the generator state (ui/generator_state.py): enums that still compare equal to the old strings, the
job and the worker bindings."""
from video_chain.ui.generator_state import (CLIP_MODELS, IMAGE_MODELS, ComfyBinding, ComfyModel, Job, Phase,
                                            RunpodBinding)


def test_models_compare_equal_to_their_old_strings():
    assert "h3" in CLIP_MODELS and "h3-hyperflow" in CLIP_MODELS and "qwen" not in CLIP_MODELS
    assert "qwen" in IMAGE_MODELS and "llm" in IMAGE_MODELS and "h3" not in IMAGE_MODELS
    assert ComfyModel.QWEN == "qwen" and Phase.DONE == "done" and Phase.IDLE == ""


def test_a_job_starts_unsent_and_without_a_refine():
    job = Job(["a", "b"])
    assert job.targets == ["a", "b"] and job.refine is None and job.sent is False


def test_a_comfy_binding_is_never_a_runpod_one():
    comfy = ComfyBinding("http://x", "/c", True)
    pod = RunpodBinding("gpu", "vol", True, "7b", True)
    assert comfy != pod and not isinstance(comfy, RunpodBinding) and isinstance(pod, RunpodBinding)
    assert ComfyBinding("http://x", "/c", True) == comfy          # (a worker is reused when nothing changed)
    assert ComfyBinding("http://x", "/c", False) != comfy


def test_model_switch_frees_the_clip_model_once():
    from video_chain.ui.generator_state import ModelSwitch

    class Editor:
        freed = 0

        def free(self):
            self.freed += 1
    sw = ModelSwitch()
    sw.prepare(ComfyModel.H3)
    ed = Editor()
    sw.free_clip_model(ed)
    sw.free_clip_model(ed)
    assert ed.freed == 1                           # (the clip model goes once)
    sw.loaded = True
    sw.prepare(ComfyModel.H3)
    assert not sw.free                             # (an image model is loaded already: nothing to unload)
    other = ModelSwitch()
    other.prepare(ComfyModel.LLM)
    assert not other.free                          # (only the clip model is the generator's)
