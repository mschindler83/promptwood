"""Edit mode's output shape: an edit can come out in another aspect than the image to edit (a 9:16 portrait in,
a 16:9 character sheet out). Synthetic data, offscreen, no ComfyUI."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_image_edit_split import _edit_plan, _run_worker  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_edit_size_is_about_one_megapixel_in_32_steps():
    from video_chain.comfy.image_edit import edit_size
    w, h = edit_size(16 / 9)
    assert (w, h) == (1376, 768) and w % 32 == 0 and h % 32 == 0
    assert edit_size(9 / 16) == (768, 1376) and edit_size(1.0) == (1024, 1024)


def test_the_workflows_take_the_output_size():
    from video_chain.comfy.image_edit import build_edit_workflow, build_moss_workflow, build_qwen21_workflow
    files = {"clip": "c.safetensors", "vae": "v.safetensors", "unet": "u.safetensors"}
    wf = build_edit_workflow(prompt="p", image="a.png", seed=1, steps=4, cfg=1.0, files=files, lightning=None,
                             prefix="x", size=(1376, 768))
    assert wf["latent"] == {"class_type": "EmptySD3LatentImage",
                            "inputs": {"width": 1376, "height": 768, "batch_size": 1}}
    assert wf["scale"]["class_type"] == "FluxKontextImageScale"          # (the picture is still seen as it is)
    plain = build_edit_workflow(prompt="p", image="a.png", seed=1, steps=4, cfg=1.0, files=files, lightning=None,
                                prefix="x")
    assert plain["latent"]["class_type"] == "VAEEncode"                 # no size: the source's shape, as before
    f21 = {"base": "q21.safetensors", "turbo": None, "clip": "c", "vae": "v"}
    wf = build_qwen21_workflow(files=f21, prompt="p", seed=1, quality="best", prefix="x", save_as="s",
                               images=["a.png"], size=(1376, 768))
    assert wf["latent"]["inputs"]["width"] == 1376 and wf["sample"]["inputs"]["latent_image"] == ["latent", 0]
    fm = {"sft": "m.safetensors", "base": None, "models": ["m.safetensors"], "clip": "c", "vae": "v",
          "edit_lora": "e.safetensors"}
    wf = build_moss_workflow(files=fm, prompt="p", seed=1, prefix="x", save_as="s", images=["a.png"],
                             size=(1376, 768))
    assert wf["latent"]["inputs"] == {"width": 1376, "height": 768, "batch_size": 1}


class _SizedEditor:
    def __init__(self):
        self.sizes = []

    def models(self):
        return ["2511"]

    def qualities(self, model):
        return ["standard"]

    def edit(self, src, prompt, out, seed, quality, refs, loras, model=None, mask=None, size=None):
        self.sizes.append(size)
        Image.new("RGB", size or (16, 16), "red").save(out)
        return {"edit_model": model}


def test_the_worker_asks_for_the_shape_and_does_not_scale_it_back(app, tmp_path):
    src = tmp_path / "portrait.png"
    Image.new("RGB", (90, 160), "green").save(src)
    ed = _SizedEditor()
    got = _run_worker(_edit_worker(_edit_plan(tmp_path, ed, [str(src)], keep_size=True, out_size=(64, 32))))
    assert got["finished"] == [""] and ed.sizes == [(64, 32)]
    png, meta = got["results"][0]
    assert Image.open(png).size == (64, 32) and meta["size"] == [64, 32]


def _edit_worker(plan):
    from video_chain.ui.image_generate import EditWorker
    return EditWorker(plan)


def _dialog(tmp_path, **kw):
    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    q = QSettings(str(tmp_path / "d.ini"), QSettings.IniFormat)
    return ImageEditDialog(None, tmp_path, [], None, None,
                           services=ImageEditServices(connect=lambda p: None, on_use=lambda k, p: False, settings=q,
                                                      **kw))


def test_the_window_offers_the_output_shape(app, tmp_path):
    src = tmp_path / "portrait.png"
    Image.new("RGB", (90, 160), "green").save(src)
    dlg = _dialog(tmp_path)
    dlg.set_source(str(src))
    assert dlg.edit_shape.currentData() is None and dlg.keep_size.isEnabled()      # like the image to edit
    dlg.edit_shape.setCurrentIndex(dlg.edit_shape.findData(16 / 9))
    assert not dlg.keep_size.isEnabled() and dlg.edit_out_size() == (1376, 768)
    assert "1376 × 768" in dlg.edit_shape.toolTip()
    plans = []
    dlg._start_worker = lambda worker, base_n, text: plans.append(worker.plan)
    dlg.prompt.setPlainText("a character sheet")
    dlg.generate()
    assert plans[0].out_size == (1376, 768)
    assert not plans[0].keep_size
    dlg._busy = False                                    # (the stub started nothing)
    dlg.reject()
    again = _dialog(tmp_path)                                                       # remembered
    assert again.edit_shape.currentData() == pytest.approx(16 / 9)
    again.reject()


def test_a_painted_mask_keeps_the_shape(app, tmp_path):
    src = tmp_path / "portrait.png"
    Image.new("RGB", (90, 160), "green").save(src)
    dlg = _dialog(tmp_path)
    dlg.set_source(str(src))
    dlg.edit_model.setCurrentIndex(max(0, dlg.edit_model.findData("qwen21")))
    if dlg.edit_model.currentData() != "qwen21":
        dlg.edit_model.addItem("Qwen-Image 2.1", "qwen21")
        dlg.edit_model.setCurrentIndex(dlg.edit_model.count() - 1)
    mask = QImage(90, 160, QImage.Format.Format_RGB32)
    mask.fill(0)
    dlg.mask = mask
    assert dlg._mask_problem() == ""
    dlg.edit_shape.setCurrentIndex(dlg.edit_shape.findData(16 / 9))
    assert "shape" in dlg._mask_problem()
    dlg.reject()
