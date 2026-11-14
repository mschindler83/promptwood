"""Things the split of main_window.py made explicit (structure, caching): the cast list looks at every clip once."""
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from test_ui import app, crop_accepted, image, silent_player, win  # noqa: E402,F401
from video_chain import tree as T  # noqa: E402


def test_cast_list_reads_each_clip_once_not_once_per_character(win, tmp_path, image, monkeypatch):
    assert win.create_project(tmp_path / "p", str(image))
    for name in ("Anna", "Bert", "Cleo"):
        T.add_character(win.project, win.tree, name)
    first = win.tree.add(None, "@Anna walks to @Bert")
    win.tree.add(first.id, "@Cleo waves")
    calls = []
    real = T.mentions
    monkeypatch.setattr(T, "mentions", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(win, "_show_character", lambda *a, **k: None)      # (only the list is looked at)
    win.cards = []
    win._show_references()
    assert len(calls) == len([n for n in win.tree.nodes.values() if not n.is_video])     # (not clips x characters)
    assert win._mentioning("anna") == [first.id]
    assert len(win._mentioning("Cleo")) == 1 and win._mentioning("nobody") == []


def test_film_save_error_at_open_is_reported(win, tmp_path, image, monkeypatch):
    from video_chain import film as F
    assert win.create_project(tmp_path / "p", str(image))
    win.close_project()

    def broken(self):
        raise OSError("disk full")
    monkeypatch.setattr(F.Film, "save", broken)
    assert win.open_project(tmp_path / "p")
    assert "disk full" in win.log.toPlainText()


def test_joined_seconds_of_the_film_is_asked_off_the_ui_thread(app, win, tmp_path, image, monkeypatch):
    import threading
    from test_ui import wait
    from video_chain import chain, film as F
    assert win.create_project(tmp_path / "p", str(image))
    seen = []
    out = tmp_path / "film.mp4"
    out.write_bytes(b"x")
    monkeypatch.setattr(F, "join_film", lambda film, target, keep=False: out)
    monkeypatch.setattr(chain, "joined_seconds", lambda p: seen.append(threading.current_thread()) or 4.0)
    win.join_film("export", out)
    wait(app, lambda: "Exported the film" in win.status.text())
    assert seen and all(t is not threading.main_thread() for t in seen)
    assert "0:04.0" in win.status.text()


def test_a_failed_comfyui_unload_is_said_in_the_log(win, monkeypatch):
    from video_chain import comfy as CMF
    from video_chain import llm_server as LLM
    monkeypatch.setattr(LLM, "free_vram_mb", lambda: 1000)
    monkeypatch.setattr(CMF, "free_models", lambda url, timeout=10: False)
    win._make_room(9000, "http://127.0.0.1:1", "the language model")
    assert "did not unload" in win.log.toPlainText()


def test_export_dialog_hands_its_answers_over_as_one_request(app, tmp_path):
    from video_chain.ui.project_dialogs import ExportDialog, ExportRequest
    dlg = ExportDialog(None, str(tmp_path / "x"), (672, 384), {"1080p": True}, swapped=2, strength="natural",
                       method="seedvr2_3b", denoise=True)
    req = dlg.request()
    assert isinstance(req, ExportRequest)
    assert req.out == tmp_path / "x.mp4" and req.size == dlg.size.currentData()
    assert req.method == "seedvr2_3b" and req.denoise is True and req.cloud is False
    assert req.swapped is True                       # (replaced-person videos exist and the box is checked)
    dlg.swapped.setChecked(False)
    assert dlg.request().swapped is False
    assert ExportDialog(None, str(tmp_path / "y"), (672, 384), {}).request().swapped is False
    dlg.deleteLater()


def test_a_failing_trigger_scan_says_so_in_the_log(app, win, tmp_path, image, monkeypatch):
    from test_ui import wait
    from video_chain import triggers
    assert win.create_project(tmp_path / "p", str(image))

    class Broken:
        def scan(self, *a, **k):
            raise ValueError("the CivitAI answer was nonsense")
    monkeypatch.setattr(triggers, "db", lambda: Broken())
    monkeypatch.setattr(win, "_trigger_paths", lambda: ["/x/some.safetensors"])
    win._scan_triggers()
    wait(app, lambda: win._trigger_scan is None)
    assert "nonsense" in win.log.toPlainText()
