"""The character sheets window: tries of the head sheet and the full-body sheet from one source photo (Qwen-Image 2.1
in ComfyUI; here a fake creator), editable prompts that are remembered and can be reset, one try chosen of each."""
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

from ui_helpers import app, wait  # noqa: E402,F401
from video_chain import edits, sheets as S  # noqa: E402


class FakeCreator:
    def __init__(self, gate: threading.Event | None = None):
        self.calls = []
        self.gate = gate
        self.interrupted = False
        self.started = threading.Event()

    def create(self, family, model, prompt, out_png, seed, quality, width, height, negative="", loras=(), refs=(),
               ref_strength=0.65, quality_tags=True, sampling=None):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10)
            if self.interrupted:
                raise RuntimeError("interrupted")
        self.calls.append({"prompt": prompt, "seed": seed, "quality": quality, "refs": [str(r) for r in refs]})
        Image.new("RGB", (width, height), (len(self.calls) * 20, 0, 0)).save(out_png)
        return {"mode": "create", "family": family, "quality": quality, "width": width, "height": height}

    def interrupt(self):
        self.interrupted = True
        if self.gate is not None:
            self.gate.set()


def _dialog(tmp_path, creator, qs=None):
    from video_chain.ui.sheets_window import SheetsDialog
    photo = tmp_path / "photo.png"
    if not photo.exists():
        Image.new("RGB", (300, 400), (90, 60, 30)).save(photo)
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    qs = qs or QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    return SheetsDialog(None, project, "Anna", photo, connect=lambda progress: creator, settings=qs), photo, project, qs


def test_the_window_makes_tries_of_both_sheets_and_one_of_each_is_chosen(app, tmp_path):
    c = FakeCreator()
    dlg, photo, project, _qs = _dialog(tmp_path, c)
    assert dlg.windowTitle() == "Character sheets of @Anna"
    assert not dlg.b_save.isEnabled()
    dlg.tries.setValue(2)
    dlg.make("head_sheet")
    wait(app, lambda: not dlg.busy and len(dlg.results["head_sheet"]) == 2)
    assert len(c.calls) == 2 and len({x["seed"] for x in c.calls}) == 2                 # (a new seed per try)
    assert all(x["prompt"] == S.DEFAULT_PROMPTS["head_sheet"] and x["refs"][1] == str(photo) for x in c.calls)
    assert dlg.chosen("head_sheet") == dlg.results["head_sheet"][0] and not dlg.b_save.isEnabled()
    dlg.make_both()
    wait(app, lambda: not dlg.busy and len(dlg.results["body_sheet"]) == 2)
    assert len(dlg.results["head_sheet"]) == 4 and len(c.calls) == 6
    dlg.choose("head_sheet", dlg.results["head_sheet"][2])
    assert dlg.chosen("head_sheet") == dlg.results["head_sheet"][2] and dlg.b_save.isEnabled()
    for png in dlg.results["head_sheet"] + dlg.results["body_sheet"]:                # (kept as image results)
        assert Path(png).parent == project / edits.EDITS_DIR
        meta = edits.read_meta(png)
        assert meta["sheet"] in S.ROLES and meta["sheet_source"] and meta["prompt"]
    dlg.deleteLater()


def test_the_prompts_are_remembered_and_can_be_reset(app, tmp_path):
    c = FakeCreator()
    dlg, _photo, _p, qs = _dialog(tmp_path, c)
    dlg.prompts["body_sheet"].setPlainText("my own body prompt")
    dlg.quality.setCurrentIndex(dlg.quality.findData("best"))
    dlg.make("body_sheet")
    wait(app, lambda: not dlg.busy and dlg.results["body_sheet"])
    assert c.calls[-1]["prompt"] == "my own body prompt" and c.calls[-1]["quality"] == "best"
    dlg.deleteLater()
    again, *_ = _dialog(tmp_path, c, qs)
    assert again.prompts["body_sheet"].toPlainText() == "my own body prompt"
    assert again.quality.currentData() == "best"
    assert again.results["body_sheet"]                                     # (earlier tries of this photo: shown again)
    again.reset_prompt("body_sheet")
    assert again.prompts["body_sheet"].toPlainText() == S.DEFAULT_PROMPTS["body_sheet"]
    again.deleteLater()
    third, *_ = _dialog(tmp_path, c, qs)
    assert third.prompts["body_sheet"].toPlainText() == S.DEFAULT_PROMPTS["body_sheet"]
    third.deleteLater()


def test_an_empty_prompt_is_refused_and_stop_interrupts(app, tmp_path):
    gate = threading.Event()
    c = FakeCreator(gate)
    dlg, *_ = _dialog(tmp_path, c)
    dlg.prompts["head_sheet"].setPlainText("   ")
    dlg.make("head_sheet")
    assert not dlg.busy and "prompt" in dlg.state.text().lower()
    dlg.reset_prompt("head_sheet")
    dlg.tries.setValue(3)
    dlg.make("head_sheet")
    assert dlg.busy and dlg.b_stop.isEnabled() and not dlg.b_both.isEnabled()
    assert c.started.wait(10)                                       # (a try is being made: Stop interrupts it)
    dlg.stop()
    wait(app, lambda: not dlg.busy)
    assert c.interrupted and dlg.results["head_sheet"] == [] and "stopped" in dlg.state.text().lower()
    dlg.deleteLater()


def test_a_failure_is_shown(app, tmp_path):
    class Broken(FakeCreator):
        def create(self, *a, **k):
            raise RuntimeError("Qwen-Image 2.1 is not available: no model files")
    dlg, *_ = _dialog(tmp_path, Broken())
    dlg.make("head_sheet")
    wait(app, lambda: not dlg.busy)
    assert "Qwen-Image 2.1 is not available" in dlg.state.text()
    dlg.deleteLater()
