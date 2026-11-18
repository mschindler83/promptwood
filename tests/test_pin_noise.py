"""Pinned-frame noise: MiniMax H3 conditions on the frames it pins (start frame, carry-over, keyframes) and on the
reference pictures at a noise level of 0.999 (nearly clean), so it copies their fine detail, its own artifacts too,
and a chain sharpens from clip to clip. ComfyUI's H3 takes another level from the conditioning
("minimax_visual_cond_noise_aug"); Promptwood's VideoChainH3PinNoise node sets it. Off (H3's 0.999) the workflow and
the recorded settings are exactly as before."""
import sys
from pathlib import Path

import pytest

from video_chain import chain, comfy as C, tree as T

NODES = Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes"
sys.path.insert(0, str(NODES))
import pin_noise as P  # noqa: E402


def test_the_node_sets_the_level_on_every_conditioning_entry():
    tensor = object()
    cond = [[tensor, {"pooled_output": 1, "minimax_keyframes": [{"x": 1}]}], [tensor, {}]]
    out, = P.VideoChainH3PinNoise().apply(cond, 0.95)
    assert [e[1]["minimax_visual_cond_noise_aug"] for e in out] == [0.95, 0.95]
    assert out[0][0] is tensor and out[0][1]["minimax_keyframes"] == [{"x": 1}]
    assert "minimax_visual_cond_noise_aug" not in cond[0][1]                   # (the input stays as it is)
    assert "VideoChainH3PinNoise" in P.NODE_CLASS_MAPPINGS


def _wf(**kw):
    return C.build_workflow(prompt="x", image="s.png", seed=1, width=512, height=512, frames=73, steps=6, loras=[],
                            audio=False, prefix="p", **kw)


def test_the_workflow_gets_the_node_only_when_set():
    assert _wf() == _wf(pin_noise=None) == _wf(pin_noise=C.PIN_NOISE_OFF)       # (off: as before)
    wf = _wf(pin_noise=0.95, refs=["r.png"], keyframes=[("k.png", 40)])
    node = next(k for k, v in wf.items() if v["class_type"] == "VideoChainH3PinNoise")
    assert wf[node]["inputs"]["visual"] == 0.95
    assert wf[node]["inputs"]["conditioning"] == ["key_guide0", 0]              # (after every pinned frame)
    assert wf["guider"]["inputs"]["conditioning"] == [node, 0]
    with pytest.raises(ValueError, match="noise"):
        _wf(pin_noise=1.5)


def test_it_is_recorded_only_when_set(tmp_path):
    base = dict(model="m", lora=None, strength=1.0, frames=73, steps=6, width=512, height=512, audio=False)
    assert "pin_noise" not in chain.settings_dict(**base)
    assert chain.settings_dict(**base, pin_noise=0.95)["pin_noise"] == 0.95
    assert T.DEFAULT_OPTIONS.get("pin_noise") in (None, C.PIN_NOISE_OFF)


def test_generate_sends_it_and_says_when_the_node_is_missing(tmp_path):
    from fake_comfy import FakeComfy
    from PIL import Image
    server = FakeComfy(tmp_path / "comfy")
    try:
        (server.root / "main.py").write_text("")
        start = tmp_path / "s.png"
        Image.new("RGB", (192, 108), (10, 200, 10)).save(start)
        s = chain.settings_dict(model="Comfy-Org/MiniMax-H3", lora=None, strength=1.0, frames=39, steps=20,
                                width=672, height=384, audio=True, pin_noise=0.95)
        gen = C.ComfyGenerator(server.url, comfy_dir=server.root)
        gen.generate("x", start, tmp_path / "v.mp4", tmp_path / "v.png", 1, s)
        wf = server.prompts[-1]
        assert any(v["class_type"] == "VideoChainH3PinNoise" and v["inputs"]["visual"] == 0.95 for v in wf.values())
        server.pin_noise_node = False
        with pytest.raises(C.ComfyError, match="VideoChainH3PinNoise.*custom_nodes"):
            C.ComfyGenerator(server.url, comfy_dir=server.root).generate("x", start, tmp_path / "w.mp4",
                                                                       tmp_path / "w.png", 1, s)
    finally:
        server.close()


def test_the_project_option_reaches_the_clips(tmp_path):
    from PIL import Image
    p = tmp_path / "proj"
    p.mkdir()
    img = tmp_path / "in.png"
    Image.new("RGB", (96, 54), (200, 200, 200)).save(img)
    t = T.Tree()
    T.set_start_image(p, img, t)
    a = t.add(None, "a")
    T.save(p, t)
    assert "pin_noise" not in T.Checker(p, T.load(p)).settings(a.id)          # (not set: recorded as before)
    t = T.load(p)
    t.options["pin_noise"] = 0.95
    T.save(p, t)
    assert T.Checker(p, T.load(p)).settings(a.id)["pin_noise"] == 0.95
    t.options["pin_noise"] = C.PIN_NOISE_OFF                                  # (H3's own level: as not set)
    T.save(p, t)
    assert "pin_noise" not in T.Checker(p, T.load(p)).settings(a.id)


def test_the_two_off_values_agree():
    from video_chain.tree import inputs
    assert inputs.PIN_NOISE_OFF == C.PIN_NOISE_OFF


def test_the_clip_settings_field(tmp_path, monkeypatch):
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from ui_helpers import flush, new_project
    from conftest import settings
    from PIL import Image
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    from video_chain.ui.main_window import MainWindow, Preview
    monkeypatch.setattr(Preview, "open", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    win = MainWindow(settings(tmp_path), command=["true"], audio_player=False)
    try:
        img = tmp_path / "i.png"
        Image.new("RGB", (192, 108), (90, 90, 90)).save(img)
        new_project(win, tmp_path, img, ("a",))
        assert win.pin_noise.value() == pytest.approx(C.PIN_NOISE_OFF) and "pin_noise" not in win.tree.options
        win.pin_noise.setValue(0.95)
        assert win.tree.options["pin_noise"] == 0.95
        flush(app, win)
        assert T.load(win.project).options["pin_noise"] == 0.95
        win.pin_noise.setValue(C.PIN_NOISE_OFF)
        assert "pin_noise" not in win.tree.options                             # (H3's own: nothing stored)
    finally:
        win.close()
        win.deleteLater()


@pytest.mark.parametrize("bad", ["z", float("nan"), 0.3, 1.7, None])
def test_a_stored_value_that_cannot_be_used_counts_as_not_set(bad):
    """Review: an unreadable or out-of-range value was recorded (and failed only when the workflow was built), and the
    simple view's note raised on text."""
    from video_chain.ui import ui_logic as UL
    o = {"pin_noise": bad}
    assert T.pin_noise_of(o) is None
    assert "pin_noise" not in [k for k, _ in UL.project_setting_changes(o)]
    assert "pin_noise" in [k for k, _ in UL.project_setting_changes({"pin_noise": 0.95})]


def test_both_passes_and_the_cfg_negative_get_the_level():
    from video_chain.comfy.video import build_two_pass_workflow
    wf = build_two_pass_workflow(two_pass={"scale": 0.5, "denoise": 0.3}, upscaler="u.safetensors", prompt="x",
                                 image="s.png", seed=1, width=512, height=512, frames=73, steps=8, loras=[],
                                 audio=False, prefix="p", pin_noise=0.95)
    assert wf["pin_noise"]["inputs"]["visual"] == wf["p1_pin_noise"]["inputs"]["visual"] == 0.95
    wf = _wf(pin_noise=0.95, cfg=1.5)
    assert wf["neg"]["inputs"]["conditioning"] == ["pin_noise", 0] and wf["guider"]["inputs"]["positive"] == ["pin_noise", 0]
