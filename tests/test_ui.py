"""GUI tests: offscreen, own QSettings file, the fake backend instead of ai-toolkit."""
import os
import sys
import time
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # never open windows on the real desktop
from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_backend.py")]


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from video_chain.ui.theme import ThemeManager
    ThemeManager.instance().apply(a)
    return a


def settings(tmp_path, name="s.ini"):
    from PySide6.QtCore import QSettings
    return QSettings(str(tmp_path / name), QSettings.IniFormat)


@pytest.fixture
def win(app, tmp_path):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path), command=FAKE)
    w.show()
    yield w
    if w.running:
        w.stop()
        w.proc and w.proc.waitForFinished(10000)
    w.close()


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "start.png"
    Image.new("RGB", (192, 108), (200, 200, 200)).save(p)
    return p


def wait(app, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out")


def setup_run(win, image, tmp_path, prompts=("one", "two", "three")):
    win.set_image(str(image))
    win.set_prompts(list(prompts))
    win.out.setText(str(tmp_path / "out"))
    win._out_changed()


def test_cards_add_move_remove(app, win, monkeypatch):
    assert len(win.cards) == 1
    win.cards[0].edit.setPlainText("first")
    win.add_card("second")
    win.add_card("third")
    assert win.prompts() == ["first", "second", "third"]
    win._move(win.cards[2], -1)
    assert win.prompts() == ["first", "third", "second"]
    assert [c.title.text() for c in win.cards] == ["Clip 1", "Clip 2", "Clip 3"]
    assert not win.cards[0].b_up.isEnabled() and not win.cards[2].b_down.isEnabled()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    win._remove(win.cards[1])
    assert win.prompts() == ["first", "second"]
    win.cards[0].edit.setPlainText("line one\nline   two")
    assert win.prompts()[0] == "line one line two"               # one line per prompt in the file
    assert "2 clips" in win.total.text()
    # times: 73 frames = 3.04 s, clip 2 starts there and is one frame shorter
    assert win.cards[1].span.text().startswith("0:03.0")


def test_settings_and_prompts_persist(app, tmp_path, image):
    from video_chain.ui.main_window import MainWindow
    w = MainWindow(settings(tmp_path, "p.ini"), command=FAKE)
    w.set_image(str(image))
    w.set_prompts(["a cat", "a dog"])
    w.steps.setValue(20)
    w.frames.setCurrentIndex(5)                                  # 124 frames
    w.close()
    w2 = MainWindow(settings(tmp_path, "p.ini"), command=FAKE)
    assert w2.prompts() == ["a cat", "a dog"] and w2.steps.value() == 20 and w2.frames.currentData() == 124
    assert w2.image_file == str(image.resolve())
    w2.close()


def test_video_size_follows_the_image(app, win, image):
    win.set_image(str(image))                                    # 16:9
    assert win.video_size() == (672, 384)
    assert "192×108 image → 672×384 video" in win.size_info.text()
    win.resolution.setCurrentIndex(win.resolution.findData(384))
    assert win.video_size()[0] % 32 == 0 and win.video_size()[0] < 672


def test_generate_runs_the_chain(app, win, image, tmp_path):
    setup_run(win, image, tmp_path)
    win.generate()
    assert win.running and not win.b_gen.isEnabled() and win.cards[0].edit.isReadOnly()
    wait(app, lambda: not win.running)
    out = tmp_path / "out"
    assert (out / "chain.mp4").exists() and (out / "prompts.txt").read_text().split("\n")[:3] == ["one", "two",
                                                                                                  "three"]
    assert [c.state for c in win.cards] == ["done", "done", "done"]
    assert all(c.thumb.has_image() and c.video for c in win.cards)
    assert win.status.text().startswith("Done: 3 clips") and win.b_chain.isEnabled()
    assert win.b_gen.isEnabled() and not win.cards[0].edit.isReadOnly()
    assert win.progress.value() == 1000


def test_second_run_reuses_and_marks_changes(app, win, image, tmp_path):
    setup_run(win, image, tmp_path)
    win.generate()
    wait(app, lambda: not win.running)
    win.cards[1].edit.setPlainText("two, changed")
    assert [c.state for c in win.cards] == ["done", "changed", "changed"]
    win.generate()
    wait(app, lambda: not win.running)
    assert [c.state for c in win.cards] == ["reused", "done", "done"]
    win.seed.setValue(7)                                        # other seeds: every clip is redone
    assert [c.state for c in win.cards] == ["changed"] * 3
    win.seed.setValue(42)
    assert [c.state for c in win.cards] == ["done"] * 3


def test_load_prompts_from_an_output_folder(app, win, image, tmp_path):
    from video_chain.ui.main_window import MainWindow
    setup_run(win, image, tmp_path, prompts=("x", "y"))
    win.generate()
    wait(app, lambda: not win.running)
    w2 = MainWindow(settings(tmp_path, "other.ini"), command=FAKE)
    w2.out.setText(str(tmp_path / "out"))
    w2._out_changed()
    assert w2.b_load_chain.isVisibleTo(w2)
    w2._load_chain_from_out()
    assert w2.prompts() == ["x", "y"] and w2.image_file == str(image.resolve())
    assert [c.state for c in w2.cards] == ["done", "done"]
    w2.close()


def test_failure_is_shown(app, win, image, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    setup_run(win, image, tmp_path)
    win.generate()
    wait(app, lambda: not win.running)
    assert win.status.text().startswith("Failed:") and "fake failure in clip 2" in win.status.text()
    assert [c.state for c in win.cards] == ["done", "failed", ""]
    assert win.tabs.currentWidget() is win.log


def test_stop(app, win, image, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.5")
    setup_run(win, image, tmp_path)
    win.generate()
    wait(app, lambda: win.cards[0].state == "generating")
    win.stop()
    wait(app, lambda: not win.running, timeout=20)
    assert win.status.text().startswith("Stopped")
    assert win.cards[0].state == "stopped" and win.b_gen.isEnabled()


def test_generate_explains_what_is_missing(app, win):
    win.set_prompts([""])
    win.out.setText("")
    win.generate()
    assert not win.running
    t = win.status.text()
    assert "start image" in t and "prompt" in t and "output folder" in t


def test_step_progress_reaches_the_card(app, win, image, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    setup_run(win, image, tmp_path, prompts=("only",))
    seen = []
    orig = win.cards[0].set_progress
    win.cards[0].set_progress = lambda i, n: (seen.append((i, n)), orig(i, n))
    win.generate()
    wait(app, lambda: not win.running)
    assert seen[-1] == (4, 4) and len(seen) == 4
