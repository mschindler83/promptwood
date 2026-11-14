"""GUI tests: exporting clips and paths, the enlargement, the RunPod export.
(Split out of test_ui.py; the shared fixtures and helpers are in ui_helpers.py.)"""
import json
import os


os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop

from video_chain.ui.generator_state import RunpodBinding  # noqa: E402
from video_chain import chain  # noqa: E402
from conftest import settings  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402
from ui_helpers import (  # noqa: E402,F401
    app, crop_accepted, flush, image, loads, new_project, run_and_wait, silent_player, wait, win)


def test_stopping_an_export_leaves_the_clips_as_they_are(app, win, tmp_path, image, monkeypatch):
    """An export works on upscaled copies only: stopped or failed in the middle of a clip, every clip of the path
    stays "done" (it was marked "stopped" and blocked Play path and Export path until the project was reopened),
    and the path can be exported again at once."""
    ids = new_project(win, tmp_path, image, ("a", "b"))
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    assert [win.state(i) for i in ids] == ["done", "done"] and win.b_export.isEnabled()
    monkeypatch.setenv("FAKE_DELAY", "0.5")                           # (each fake upscale takes 5 s)
    win.unload()
    wait(app, lambda: not win.worker_alive, timeout=15)
    win.export(tmp_path / "x.mp4", "2k")
    wait(app, lambda: any(win.live.get(i) == "generating" for i in ids), timeout=30)
    win.stop()
    wait(app, lambda: not win.running, timeout=20)
    assert win.status.text().startswith("Stopped")
    assert [win.state(i) for i in ids] == ["done", "done"]
    assert win.b_export.isEnabled() and win.b_play_path.isEnabled()
    assert not (tmp_path / "x.mp4").exists()


def test_step_progress_and_export(app, win, tmp_path, image, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    ids = new_project(win, tmp_path, image, ("only",))
    seen = []
    card = win.cards[0]
    orig = card.set_progress
    card.set_progress = lambda i, n, text=None: (seen.append((i, n)), orig(i, n, text))
    run_and_wait(app, win, win.generate_path)
    assert seen[-1] == (4, 4)
    target = tmp_path / "export.mp4"
    win.join_selected("export", target)
    wait(app, lambda: target.exists() and "Exported" in win.status.text())
    assert chain.probe(target)["frames"] == 17


def test_export_refined_and_hyperflow_speed(app, win, tmp_path, image):
    from video_chain.ui.main_window import ExportDialog
    ids = new_project(win, tmp_path, image, ("a", "b"))
    win.speed.setCurrentIndex(win.speed.findData("hyperflow"))           # HyperFlow: 8 steps for every clip
    assert win.tree.options["hyperflow"]["on"] and not win.steps.isEnabled()
    assert T.Checker(win.project, win.tree).settings(ids[0])["steps"] == 8
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384), {"2k": False, "1080p": True})
    texts = [dlg.size.itemText(i) for i in range(dlg.size.count())]
    assert texts[0].startswith("As made (672×384)") and "1888×1088 · 2.8× the draft · already upscaled" in texts[1] \
        and "2528×1440 · 3.8× the draft" in texts[2]
    assert dlg.scale_note.isHidden()                                         # (as made: no enlargement)
    assert dlg.swapped.isHidden()                                            # no video with a replaced person
    assert dlg.method.currentData() == "seedvr2_7b" and not dlg.method.isEnabled()      # SeedVR2 7B by default
    assert [dlg.method.itemData(i) for i in range(dlg.method.count())] == ["seedvr2_7b", "seedvr2_3b", "refine"]
    assert dlg.strength.currentData() == "natural" and not dlg.strength.isEnabled()
    assert [dlg.strength.itemData(i) for i in range(dlg.strength.count())] == ["soft", "natural", "crisp"]
    dlg.size.setCurrentIndex(2)
    assert dlg.method.isEnabled() and not dlg.strength.isEnabled() and not dlg.fast.isEnabled()   # (refine only)
    assert dlg.denoise.isEnabled() and not dlg.denoise.isChecked()                # SeedVR2 only, off by default
    dlg.method.setCurrentIndex(dlg.method.findData("refine"))
    assert dlg.strength.isEnabled() and dlg.fast.isEnabled() and not dlg.denoise.isEnabled()
    dlg.reject()
    asked = []
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384),
                       lambda size, how, st, dn=False: asked.append(dn) or dn, denoise=True)
    assert dlg.denoise.isChecked() and "already upscaled" in dlg.size.itemText(2)   # kept copies per choice
    dlg.denoise.setChecked(False)
    assert "already upscaled" not in dlg.size.itemText(2) and asked[-1] is False
    dlg.reject()
    seen = []
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384),
                       lambda size, how, st, dn=False: seen.append((size, how, st)) or how == "seedvr2_3b", swapped=2)
    assert "already upscaled" not in dlg.size.itemText(2) and ("2k", "seedvr2_7b", "natural") in seen
    dlg.method.setCurrentIndex(dlg.method.findData("seedvr2_3b"))         # kept copies are per method
    assert "already upscaled" in dlg.size.itemText(2)
    assert not dlg.swapped.isHidden() and dlg.swapped.isChecked() and not dlg.swapped.isEnabled()
    assert dlg.swapped.text() == "Upscale the videos with a replaced person too"
    dlg.size.setCurrentIndex(2)
    assert dlg.swapped.isEnabled()
    dlg.reject()
    out = tmp_path / "export-2k.mp4"
    win.export(out, "2k")                                                    # SeedVR2 7B
    assert win.running
    wait(app, lambda: not win.running, timeout=120)
    info = chain.probe(out)
    base = T.Checker(win.project, win.tree).base()
    from video_chain.comfy import refine_size
    assert (info["width"], info["height"]) == refine_size(base["width"], base["height"], "2k")
    assert win.status.text().startswith(f"Exported {out.name} (2k, SeedVR2 7B; 2 clips upscaled now"), \
        win.status.text()
    win.export(out, "2k")                                                    # again: the kept copies are reused
    wait(app, lambda: not win.running, timeout=120)
    assert win.status.text() == f"Exported {out.name} (2k, SeedVR2 7B; every clip was upscaled already)."
    win.export(out, "2k", method="refine")                                   # the refine: its own copies
    wait(app, lambda: not win.running, timeout=120)
    assert win.status.text().startswith(f"Exported {out.name} (2k, refined; 2 clips upscaled now")
    assert {win.state(n) for n in ids} == {"done"}


def test_export_on_runpod(app, win, tmp_path, image, monkeypatch):
    """“Upscale on RunPod”: only for SeedVR2 at a higher size and with a key; the export gets its own worker bound
    to the pod settings (GPU, volume, fallback, SeedVR2 size), the local ComfyUI is neither needed nor started,
    the pod's preparation shows in the status bar, and the worker (with its pod) ends after the export."""
    from video_chain import runpod as RP
    from video_chain.ui.main_window import ExportDialog
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384), {}, cloud=None, cloud_on=True)
    assert not dlg.cloud.isEnabled() and not dlg.cloud.isChecked() and "API key" in dlg.cloud.toolTip()
    dlg.reject()
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 384), {}, cloud=True, cloud_on=True)
    assert dlg.cloud.isChecked() and not dlg.cloud.isEnabled() and not dlg.on_cloud()   # as made: nothing to do
    dlg.size.setCurrentIndex(2)
    assert dlg.cloud.isEnabled() and dlg.on_cloud()
    dlg.method.setCurrentIndex(dlg.method.findData("refine"))                # the refine needs the H3 models
    assert not dlg.cloud.isEnabled() and not dlg.on_cloud()
    dlg.reject()

    loads = tmp_path / "loads.jsonl"
    monkeypatch.setenv("FAKE_LOADS", str(loads))
    ids = new_project(win, tmp_path, image, ("a",))
    run_and_wait(app, win, win.generate_path)                                # the clip, by the local worker
    wait(app, lambda: not win._joins, timeout=30)
    out = tmp_path / "cloud-2k.mp4"
    win.export(out, "2k", cloud=True)
    wait(app, lambda: not win.running)
    assert not win.running and "RunPod API key" in win.status.text()        # no key: not started
    monkeypatch.setattr(RP, "api_key", lambda: "k-123")
    win.runpod_gpu.setCurrentIndex(win.runpod_gpu.findData("NVIDIA L40S"))
    win.runpod_volume.setText("vol9")
    win.runpod_volume.editingFinished.emit()
    win.runpod_fallback.setChecked(False)
    win.comfy_up = False                                                     # no local ComfyUI: not needed
    monkeypatch.setenv("FAKE_LOAD_DELAY", "0.5")                             # (the pod "prepares" a moment)
    win.export(out, "2k", method="seedvr2_3b", cloud=True)
    wait(app, lambda: win._job is not None)                                   # (checked in the background first)
    assert win.running and isinstance(win._worker_key(), RunpodBinding)
    seen = []
    wait(app, lambda: seen.append(win.status.text()) or not win.running, timeout=120)
    assert any(t.startswith("RunPod: downloading") for t in seen)
    assert win.status.text().startswith(f"Exported {out.name} (2k, SeedVR2 3B"), win.status.text()
    assert chain.probe(out)["frames"] > 0
    cloud = [json.loads(x) for x in loads.read_text().splitlines() if "cloud" in x]
    assert cloud == [{"cloud": "runpod", "gpu": "NVIDIA L40S", "models": "3b", "volume": "vol9",
                      "fallback": False, "idle": 0, "without_volume": True}]
    wait(app, lambda: not win.worker_alive, timeout=30)                      # its pod is not left waiting
    assert not win._comfy_starting                                            # the local ComfyUI was not started
    assert {win.state(n) for n in ids} == {"done"}


def test_export_one_clip(app, win, tmp_path, image, monkeypatch):
    ids = new_project(win, tmp_path, image, ("a", "b"))
    assert not win.b_export_clip.isEnabled()                               # not made yet
    run_and_wait(app, win, win.generate_path)
    wait(app, lambda: not win._joins, timeout=30)
    win.select(ids[0])
    flush(app, win)
    assert win.focus == ids[0] and win.b_export_clip.isEnabled()
    node = win.tree.nodes[ids[0]]
    out = tmp_path / "clip-a.mp4"
    win.export(out, "made", single=ids[0])                                 # as made: the clip itself
    wait(app, lambda: win.status.text().startswith("Exported clip-a.mp4"), timeout=30)   # (copied in the background)
    assert out.read_bytes() == (win.project / node.made["video"]).read_bytes()
    assert win.status.text() == f"Exported clip-a.mp4 (clip #{ids[0][1:]}, as made)."
    up = tmp_path / "clip-a-2k.mp4"
    win.export(up, "2k", single=ids[0])                                    # upscaled: that clip only
    assert win.running
    wait(app, lambda: not win.running, timeout=120)
    assert chain.probe(up)["frames"] == chain.probe(win.project / node.made["video"])["frames"]
    assert win.status.text().startswith(f"Exported {up.name} (2k, SeedVR2 7B; 1 clip upscaled now"), win.status.text()
    # the dialog: titled for the clip, its default file named after it
    from video_chain.ui import main_window as M
    seen = {}

    def fake_exec(dlg):
        seen.update(title=dlg.windowTitle(), file=dlg.file.text())
        return QDialog.Rejected
    from PySide6.QtWidgets import QDialog
    monkeypatch.setattr(M.ExportDialog, "exec", fake_exec)
    win.export_path(ids[1])
    assert seen["title"] == f"Export clip #{ids[1][1:]}" and seen["file"].endswith(f"-{ids[1]}-clip.mp4")
    # the clip's menu: "Export this clip…" opens the same dialog for that clip
    called = []
    monkeypatch.setattr(win, "export_path", lambda nid=None: called.append(nid))
    monkeypatch.setattr(win, "_exec_menu", lambda m, pos: next(a for a in m.actions() if a.text() == "Export this clip…"))
    from PySide6.QtCore import QPoint
    win._context(ids[1], QPoint(0, 0))
    wait(app, lambda: called, timeout=5)
    assert called == [ids[1]]


def test_export_shows_the_enlargement_and_size_1024(app, win, tmp_path, image):
    from video_chain.ui.main_window import ExportDialog
    dlg = ExportDialog(None, str(tmp_path / "x.mp4"), (672, 832), {})       # a portrait draft at size 768
    assert "1280×1600 · 1.9× the draft" in dlg.size.itemText(1) and "1696×2112 · 2.5× the draft" in dlg.size.itemText(2)
    dlg.size.setCurrentIndex(1)
    assert dlg.scale_note.isHidden()                                         # 1.9×: no note
    dlg.size.setCurrentIndex(2)
    assert not dlg.scale_note.isHidden() and "2.5× the draft" in dlg.scale_note.text()
    assert "1080p stays within 2× here" in dlg.scale_note.text() and "H3's native size" in dlg.scale_note.text()
    dlg.reject()
    win.open_project(tmp_path / "proj")
    from video_chain import chain
    k = win.resolution.findData(chain.H3_NATIVE)
    assert k >= 0 and "native size" in win.resolution.itemText(k)