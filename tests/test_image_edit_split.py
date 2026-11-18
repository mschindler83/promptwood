"""The image dialog after its split: edits records without Qt, undo of several deletes, one generate worker."""
import json
import os
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _results(tmp_path, names=("r0", "r1", "r2")):
    from PIL import Image
    edits = tmp_path / "edits"
    edits.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        Image.new("RGB", (16, 16), "red").save(edits / f"{n}.png")
        (edits / f"{n}.json").write_text(json.dumps({"prompt": n}))
        os.utime(edits / f"{n}.png", (time.time() - 50 + i, time.time() - 50 + i))
    return edits


def _dialog(tmp_path, **kw):
    from video_chain.ui.image_edit import ImageEditDialog, ImageEditServices
    q = QSettings(str(tmp_path / "t.ini"), QSettings.IniFormat)
    return ImageEditDialog(None, tmp_path, [], None, None, services=ImageEditServices(connect=lambda p: None, on_use=lambda k, p: False, settings=q, **kw))


def test_several_deletes_can_each_be_undone(app, tmp_path):
    edits = _results(tmp_path)
    dlg = _dialog(tmp_path)
    first = Path(dlg.selected()[0]).name
    dlg.delete_selected()
    second = Path(dlg.selected()[0]).name
    dlg.delete_selected()
    assert not (edits / first).exists() and not (edits / second).exists()
    assert dlg.b_undo.isVisibleTo(dlg)
    dlg.undo_delete()
    assert (edits / second).exists() and (edits / second).with_suffix(".json").exists()
    assert dlg.b_undo.isVisibleTo(dlg)                  # one more to bring back
    dlg.undo_delete()
    assert (edits / first).exists() and (edits / first).with_suffix(".json").exists()
    assert not dlg.b_undo.isVisibleTo(dlg)
    dlg.reject()


def test_closing_empties_the_trash_for_good(app, tmp_path):
    edits = _results(tmp_path)
    dlg = _dialog(tmp_path)
    dlg.delete_selected()
    dlg.delete_selected()
    assert any((edits / ".deleted").iterdir())
    dlg.reject()
    assert not (edits / ".deleted").exists()
    assert len(list(edits.glob("*.png"))) == 1


def test_edits_module_needs_no_qt():
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, 'src'); import video_chain.edits as E; "
            "assert 'PySide6' not in sys.modules, 'Qt was imported'; print(E.EDITS_DIR)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "edits"


class _Editor:
    def __init__(self, fail_on=None, on_edit=None):
        self.calls, self.fail_on, self.on_edit = [], fail_on, on_edit

    def models(self):
        return ["2511"]

    def qualities(self, model):
        return ["fast", "standard", "best"]

    def edit(self, src, prompt, out, seed, quality, refs, loras, model=None, mask=None):
        from PIL import Image
        self.calls.append((Path(src).name, prompt, seed))
        if self.fail_on and len(self.calls) == self.fail_on:
            raise RuntimeError("boom")
        if self.on_edit:
            self.on_edit()
        Image.new("RGB", (16, 16), "red").save(out)
        return {"edit_model": model}

    def create(self, fam, model, prompt, out, seed, quality, w, h, negative, loras, refs, strength, tags, sampling):
        from PIL import Image
        self.calls.append((fam, prompt, seed, negative, [lo[0] for lo in loras]))
        Image.new("RGB", (w, h), "blue").save(out)
        return {"family": fam, "mode": "create"}

    def interrupt(self):
        pass


def _run_worker(worker):
    got = {"results": [], "finished": []}
    worker.result.connect(lambda png, meta: got["results"].append((png, meta)))
    worker.finished.connect(got["finished"].append)
    worker._run()                                     # (in this thread: the signals arrive at once)
    return got


def _edit_plan(tmp_path, editor, sources, **kw):
    from video_chain.ui.image_generate import EditPlan
    args = dict(edits=tmp_path / "edits", connect=lambda progress: editor, prompt="make it blue", quality="standard",
                n=len(sources), seeds=[7], runs=[[]], run_words=[[]], refs=[], model="2511", sources=sources, per=1)
    args.update(kw)
    return EditPlan(**args)


def test_edit_worker_keeps_a_source_copy_and_skips_unreadable_files(app, tmp_path):
    from PIL import Image

    from video_chain.ui.image_generate import EditWorker
    good, bad = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (16, 16), "green").save(good)
    bad.write_bytes(b"not an image")
    w = EditWorker(_edit_plan(tmp_path, _Editor(), [str(bad), str(good)], keep_size=False))
    got = _run_worker(w)
    assert got["finished"] == [""] and w.skipped == ["b.png"]
    (png, meta), = got["results"]
    assert meta["source"] == str(good) and Path(meta["source_copy"]).read_bytes() == good.read_bytes()
    assert meta["folder"] == str(tmp_path) and meta["seed"] == 7 and meta["prompt"] == "make it blue"
    assert json.loads(Path(png).with_suffix(".json").read_text())["run"] == meta["run"]


def test_edit_worker_removes_source_copies_no_result_refers_to(app, tmp_path):
    from PIL import Image

    from video_chain.ui.image_generate import EditWorker
    src = tmp_path / "a.png"
    Image.new("RGB", (16, 16), "green").save(src)
    w = EditWorker(_edit_plan(tmp_path, _Editor(fail_on=1), [str(src)]))
    got = _run_worker(w)
    assert got["finished"] == ["boom"] and got["results"] == []
    assert list((tmp_path / "edits" / "sources").glob("*")) == []


def test_edit_worker_cancel_ends_as_cancelled(app, tmp_path):
    from PIL import Image

    from video_chain.ui.image_generate import EditWorker
    src = tmp_path / "a.png"
    Image.new("RGB", (16, 16), "green").save(src)
    holder = []
    editor = _Editor(on_edit=lambda: holder[0].cancel())
    w = EditWorker(_edit_plan(tmp_path, editor, [str(src)], per=3, n=3, seeds=[1, 2, 3], runs=[[]], run_words=[[]]))
    holder.append(w)
    got = _run_worker(w)
    assert len(got["results"]) == 1 and got["finished"] == ["cancelled"]   # the loop stops: Stopped, not Done


def test_create_worker_makes_one_image_per_lora_and_variation(app, tmp_path):
    from types import SimpleNamespace

    from video_chain.ui.image_generate import CreatePlan, CreateWorker
    jobs = [SimpleNamespace(prompt="a cat", negative="", key="k", combo={"x": "cat"}, step=0, index=0),
            SimpleNamespace(prompt="a dog", negative="", key="k", combo={"x": "dog"}, step=1, index=1)]
    editor = _Editor()
    plan = CreatePlan(edits=tmp_path / "edits", connect=lambda progress: editor, prompt="a {x}", quality="fast", n=4,
                      seeds=[5, 6], runs=[[("l1", 1.0)], [("l2", 0.5)]], run_words=[["w1"], ["w2"]], refs=[],
                      jobs=jobs, family="sdxl", model_file="m.safetensors", negative="", width=32, height=16, tags=True)
    got = _run_worker(CreateWorker(plan))
    assert got["finished"] == [""] and len(got["results"]) == 4
    assert [c[1] for c in editor.calls] == ["w1, a cat", "w2, a cat", "w1, a dog", "w2, a dog"]
    assert [c[2] for c in editor.calls] == [5, 5, 6, 6]
    metas = [m for _p, m in got["results"]]
    assert [m["variation"]["step"] for m in metas] == [0, 0, 1, 1] and all(m["tags"] is True for m in metas)
    assert all(m["template"] == "a {x}" and m["source_copy"] is None for m in metas)


def test_unique_out_adds_the_counter_only_when_the_name_is_taken(tmp_path, monkeypatch):
    from video_chain import edits
    monkeypatch.setattr(edits.time, "strftime", lambda fmt: "20260101-000000")
    first = edits.unique_out(tmp_path, "edit", 3, 1)
    assert first.name == "edit_20260101-000000_3.png"
    first.write_bytes(b"x")
    second = edits.unique_out(tmp_path, "edit", 3, 2)
    assert second.name == "edit_20260101-000000_3_2.png"


def _add_result(edits, name, meta=None):
    from PIL import Image
    Image.new("RGB", (16, 16), "green").save(edits / f"{name}.png")
    (edits / f"{name}.json").write_text(json.dumps(meta or {"prompt": name}))
    os.utime(edits / f"{name}.png", (time.time() + 5, time.time() + 5))
    return str(edits / f"{name}.png")


def test_a_new_result_is_added_without_reloading_all_results(app, tmp_path, monkeypatch):
    """UIM-5: each new result used to re-read every record and rebuild every strip item."""
    from video_chain import edits as ED
    edits = _results(tmp_path)
    dlg = _dialog(tmp_path)
    app.processEvents()
    before = [dlg.results.item(i).data(0x100)[0] for i in range(dlg.results.count())]
    reloads = []
    real = ED.project_edits
    monkeypatch.setattr(ED, "project_edits", lambda p: (reloads.append(p), real(p))[1])
    png = _add_result(edits, "r3")
    dlg._result(png, {"prompt": "r3"})
    assert reloads == []
    after = [dlg.results.item(i).data(0x100)[0] for i in range(dlg.results.count())]
    assert after == [png] + before
    assert dlg.results.currentRow() == 0 and dlg.selected()[0] == png
    assert png in dlg._strip_items and png in dlg.gallery.thumbs
    assert dlg.f_count.text() == "4 results"
    monkeypatch.undo()
    full = [str(p) for p, _m in real(tmp_path)]                  # the same order a full reload gives
    assert after == full
    dlg.reject()


def test_a_new_result_the_filter_hides_reloads_everything(app, tmp_path):
    edits = _results(tmp_path)
    dlg = _dialog(tmp_path)
    dlg.f_search.setText("nothing like this")
    dlg._search_timer.stop()
    dlg._load_results(keep=True)
    assert dlg.results.count() == 0
    png = _add_result(edits, "r3")
    dlg._result(png, {"prompt": "r3"})
    assert dlg.results.count() == 0 and dlg.f_count.text() == "0 of 4"
    dlg.reject()


def test_strip_and_seed_chip_are_styled_by_the_theme_not_per_widget(app, tmp_path):
    """UIM-39: no stylesheet on the widgets; the application stylesheet styles them by their role."""
    from video_chain.ui.results_gallery import InfoCard
    from video_chain.ui.theme import ThemeManager
    ThemeManager.instance().apply(app)
    qss = ThemeManager.instance().qss
    assert 'QListWidget[role="resultStrip"]' in qss and 'QPushButton[role="chipButton"]' in qss
    dlg = _dialog(tmp_path)
    assert dlg.results.styleSheet() == "" and dlg.results.property("role") == "resultStrip"
    dlg.reject()
    card = InfoCard()
    card.set({"seed": 5, "prompt": "x"})
    chips = [b for b in card.findChildren(type(dlg.b_gen)) if b.property("role") == "chipButton"]
    assert chips and all(b.styleSheet() == "" for b in chips)


def _random_plan(tmp_path, editor, rnd, n):
    from video_chain.ui.image_generate import CreatePlan
    return CreatePlan(edits=tmp_path / "edits", connect=lambda progress: editor, prompt="a cat", quality="fast", n=n,
                      seeds=[5], runs=[[]], run_words=[[]], refs=[], family="sdxl", model_file="m.safetensors",
                      negative="", width=32, height=16, random=rnd)


def test_create_worker_with_random_loras(app, tmp_path):
    """Random LoRAs: each image its own choice (the pinned one always) and strengths, its trigger words, a new seed
    each (or the window's seed for all); recorded with the image."""
    from video_chain import lora_random as R
    from video_chain.ui.image_generate import CreateWorker, RandomRun
    pool = [R.Choice("pin", 0.5, 0.5, True), R.Choice("b", 0.2, 0.4), R.Choice("c", 0.6, 0.9)]
    words = {"pin": ["pw"], "b": ["bw"], "c": []}
    editor = _Editor()
    got = _run_worker(CreateWorker(_random_plan(tmp_path, editor, RandomRun(pool, 2, False, words, seed=3), 30)))
    assert got["finished"] == [""] and len(got["results"]) == 30
    for (_fam, prompt, _seed, _neg, loras), (_png, meta) in zip(editor.calls, got["results"]):
        assert loras[0] == "pin" and len(loras) == 2
        assert prompt == ", ".join(["pw"] + (["bw"] if "b" in loras else [])) + ", a cat"
        assert meta["random_loras"] == {"pinned": ["pin"], "most": 2}       # (the LoRAs: the editor's record)
    assert {tuple(c[4]) for c in editor.calls} == {("pin", "b"), ("pin", "c")}       # never the pinned one alone
    assert len({c[2] for c in editor.calls}) > 20                          # a new seed each
    editor = _Editor()
    _run_worker(CreateWorker(_random_plan(tmp_path, editor, RandomRun(pool, 2, True, words, seed=3), 5)))
    assert [c[2] for c in editor.calls] == [5] * 5                        # the same seed: the window's


def test_continuous_random_loras_run_until_stopped(app, tmp_path):
    from video_chain import lora_random as R
    from video_chain.ui.image_generate import CreateWorker, RandomRun
    holder = []
    editor = _Editor()
    real = editor.create

    def create(*a):
        out = real(*a)
        if len(editor.calls) == 7:
            holder[0].cancel()
        return out
    editor.create = create
    w = CreateWorker(_random_plan(tmp_path, editor, RandomRun([R.Choice("a", 1, 1)], 1, False, {}, seed=1), 0))
    holder.append(w)
    got = _run_worker(w)
    assert len(got["results"]) == 7 and got["finished"] == ["cancelled"]     # (Stop: "Stopped.", not "Done.")


def test_random_loras_most_is_not_limited_by_a_short_list(app, tmp_path):
    """The most per image can be set beyond the list's length (the list may grow later): a short list neither
    blocks the field nor cuts the remembered number down."""
    from video_chain import lora_random as R
    from video_chain.ui.random_loras import RandomLorasDialog
    one = {"choices": [R.Choice("a", 0.5, 1.0)], "most": 4, "n": 8, "same_seed": False}
    dlg = RandomLorasDialog(None, one, 1)
    assert dlg.most.maximum() >= 10 and dlg.most.value() == 4 and dlg.saved()["most"] == 4
    assert "1 LoRA to choose from" in dlg.most_note.text()
    dlg.most.setValue(6)
    assert dlg.problems() == [] and dlg.b_make.isEnabled()
    dlg.deleteLater()


def test_random_loras_strength_cells_take_negative_strengths(app, tmp_path):
    from PySide6.QtWidgets import QStyleOptionViewItem

    from video_chain import lora_random as R
    from video_chain.ui.random_loras import LOW, RandomLorasDialog
    dlg = RandomLorasDialog(None, {"choices": [R.Choice("a", 0.5, 1.0)], "most": 2, "n": 1, "same_seed": False}, 1)
    index = dlg.table.model().index(0, LOW)
    delegate = dlg.table.itemDelegateForColumn(LOW)
    editor = delegate.createEditor(dlg.table, QStyleOptionViewItem(), index)
    assert editor.minimum() < 0 and editor.value() == 0.5
    editor.setValue(-1.25)
    delegate.setModelData(editor, dlg.table.model(), index)
    assert dlg.pool()[0].low == -1.25
    dlg.deleteLater()
