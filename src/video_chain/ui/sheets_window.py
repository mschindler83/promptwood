"""The character sheets window: a person's head sheet and full-body sheet made from one source photo with Qwen-Image
2.1 (sheets.make_sheet), a few tries of each; one try of each is chosen and the cast keeps them (tree.set_sheets,
done by the main window). The prompts are editable, remembered, and Reset brings the app's default back. Every try
is kept in the project's edits/ like any image result (the image window shows them too) and comes back here when
the window is opened for the same photo again."""
from __future__ import annotations

import random
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QListView, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout)

from .. import chain, edits, sheets as S
from ..imagegen import FAMILIES
from .image_common import SEED_MAX
from .image_generate import CreatePlan, GenerateWorker
from . import settings_keys as SK
from .widgets import fit_size, guard_wheel, label, label_for

THUMB = QSize(240, 134)
MAX_TRIES = 8


@dataclass
class SheetPlan(CreatePlan):
    jobs: list[tuple[str, str, int]] = field(default_factory=list)   # (role, prompt, seed) of each try in turn
    source: str = ""
    signature: str = ""                                              # the source photo's SHA-1 (its tries)
    work: Path | None = None                                         # the flattened templates


class SheetWorker(GenerateWorker):
    """The tries of a plan, one after another (Stop interrupts the one being made)."""
    plan: SheetPlan

    def _make(self) -> None:
        p = self.plan
        assert p.work is not None
        for k, (role, prompt, seed) in enumerate(p.jobs, start=1):
            if self.cancelled:
                break
            self._k = k
            out = edits.unique_out(p.edits, "sheet", seed, k)
            meta = S.make_sheet(self.editor, role, Path(p.source), prompt, out, seed=seed, quality=p.quality,
                                work=p.work)
            self._finish(out, meta, SimpleNamespace(prompt=prompt), seed, [],
                         {"refs": [p.source], "sheet_source": p.signature})

    def _variation(self, job) -> dict:
        return {}

    def _cleanup(self) -> None:
        if self.plan.work is not None:                  # (the run's flattened templates)
            shutil.rmtree(self.plan.work, ignore_errors=True)


class SheetsDialog(QDialog):
    """project: where the tries are kept (edits/); name: the person; source: the photo; connect(progress) -> a
    connected ComfyImageCreator (worker thread); on_start(): before each run (GUI thread); pick() -> another photo
    (None: no Other photo… button). After Save (accepted), chosen(role) is the picked try of each sheet and source
    the photo they were made from."""

    def __init__(self, parent, project: Path, name: str, source: Path, connect: Callable,
                 settings: QSettings, on_start: Callable[[], None] | None = None,
                 pick: Callable[[], str | None] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Character sheets of @{name}")
        self.project, self.name, self.source = Path(project), name, Path(source)
        self.connect_creator, self.qs, self.on_start, self.pick = connect, settings, on_start, pick
        self.worker: SheetWorker | None = None
        self.busy = False
        self.signature = chain.file_signature(self.source) or ""
        self.results: dict[str, list[str]] = {r: [] for r in S.ROLES}
        self.prompts: dict[str, QPlainTextEdit] = {}
        self.galleries: dict[str, QListWidget] = {}
        self.buttons: dict[str, QPushButton] = {}
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        side = QVBoxLayout()
        self.photo = label()
        side.addWidget(self.photo)
        self.b_other = QPushButton("Other photo…")
        self.b_other.setToolTip("Make the sheets from another photo (the tries of this one stay in the image window)")
        self.b_other.clicked.connect(self.other_photo)
        self.b_other.setVisible(pick is not None)
        side.addWidget(self.b_other)
        side.addStretch(1)
        top.addLayout(side)
        intro = label(f"Two character sheets of @{name} are made from this photo with Qwen-Image 2.1: a close-up "
                      "head sheet and a full-body sheet, each the person from the front, in profile and in a "
                      "three-quarter view, laid out like the app's templates. Make a few tries, click the best of "
                      "each and save: the two sheets become @" + name + "'s pictures (the photo is kept to make "
                      "them again). Qwen invents what the photo does not show (body, clothes): describe it in the "
                      "prompt.", level="secondary")
        intro.setWordWrap(True)
        top.addWidget(intro, 1)
        v.addLayout(top)
        grid = QGridLayout()
        for col, role in enumerate(S.ROLES):
            box = QGroupBox(S.LABELS[role])
            b = QVBoxLayout(box)
            edit = QPlainTextEdit()
            edit.setPlainText(str(self.qs.value(self._key(role), S.DEFAULT_PROMPTS[role])))
            edit.setToolTip("The prompt for this sheet: the first picture is the template (the layout), the second "
                            "the photo (the person). Your changes are remembered; Reset brings the default back")
            edit.setMinimumHeight(110)
            self.prompts[role] = edit
            b.addWidget(edit)
            row = QHBoxLayout()
            reset = QPushButton("Reset prompt")
            reset.setToolTip("The app's default prompt for this sheet again")
            reset.clicked.connect(lambda _c=False, r=role: self.reset_prompt(r))
            make = QPushButton(f"Make {S.LABELS[role].lower()}")
            make.clicked.connect(lambda _c=False, r=role: self.make(r))
            self.buttons[role] = make
            row.addWidget(reset)
            row.addStretch(1)
            row.addWidget(make)
            b.addLayout(row)
            gal = QListWidget()
            gal.setViewMode(QListView.ViewMode.IconMode)
            gal.setIconSize(THUMB)
            gal.setResizeMode(QListView.ResizeMode.Adjust)
            gal.setMovement(QListView.Movement.Static)
            gal.setMinimumHeight(THUMB.height() + 40)
            gal.setToolTip("The tries: click the one to keep")
            gal.currentItemChanged.connect(lambda *_a: self._update())
            self.galleries[role] = gal
            b.addWidget(gal, 1)
            grid.addWidget(box, 0, col)
        v.addLayout(grid, 1)
        opts = QHBoxLayout()
        self.tries = QSpinBox()
        self.tries.setRange(1, MAX_TRIES)
        try:
            self.tries.setValue(int(str(self.qs.value(SK.SHEETS_TRIES, 2) or 2)))
        except ValueError:
            self.tries.setValue(2)
        self.tries.setToolTip("How many tries a Make makes (each with a new seed)")
        self.quality = QComboBox()
        for key, text in FAMILIES[S.FAMILY]["qualities"].items():
            self.quality.addItem(text, key)
        k = self.quality.findData(self.qs.value(SK.SHEETS_QUALITY, "fast"))
        self.quality.setCurrentIndex(max(0, k))
        opts.addWidget(label_for("Tries", self.tries, level="tertiary"))
        opts.addWidget(self.tries)
        opts.addWidget(label_for("Quality", self.quality, level="tertiary"))
        opts.addWidget(self.quality)
        opts.addStretch(1)
        self.b_both = QPushButton("Make both")
        self.b_both.setToolTip("The tries of the head sheet, then of the full-body sheet")
        self.b_both.clicked.connect(self.make_both)
        self.b_stop = QPushButton("Stop")
        self.b_stop.setToolTip("Stop interrupts the try being made (those already made stay)")
        self.b_stop.clicked.connect(self.stop)
        opts.addWidget(self.b_both)
        opts.addWidget(self.b_stop)
        v.addLayout(opts)
        self.state = label("", level="secondary")
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        end = QHBoxLayout()
        end.addStretch(1)
        self.b_save = QPushButton(f"Save sheets to @{name}")
        self.b_save.setProperty("variant", "primary")
        self.b_save.setToolTip(f"The chosen head sheet and full-body sheet become @{name}'s pictures (the ones "
                               "before go to old/)")
        self.b_save.clicked.connect(self._save)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        end.addWidget(self.b_save)
        end.addWidget(close)
        v.addLayout(end)
        self._show_photo()
        guard_wheel(self)
        fit_size(self, 1100, 760)
        self._update()

    # ---- the photo
    def _show_photo(self) -> None:
        """The photo and its earlier tries (kept in edits/, oldest first: as they were made)."""
        pm = QPixmap(str(self.source))
        self.photo.setPixmap(pm.scaled(QSize(120, 160), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation) if not pm.isNull() else QPixmap())
        self.photo.setToolTip(str(self.source))
        for role in S.ROLES:
            self.results[role] = []
            self.galleries[role].clear()
        for png, meta in reversed(edits.project_edits(self.project)):
            if meta.get("sheet") in S.ROLES and self.signature and meta.get("sheet_source") == self.signature:
                self._add_result(str(png), meta["sheet"])

    def other_photo(self) -> None:
        if self.busy or self.pick is None:
            return
        f = self.pick()
        if not f:
            return
        self.source = Path(f)
        self.signature = chain.file_signature(self.source) or ""
        self._show_photo()
        self.state.setText("")
        self._update()

    # ---- prompts
    @staticmethod
    def _key(role: str) -> str:
        return SK.SHEETS_PROMPT + role

    def reset_prompt(self, role: str) -> None:
        self.prompts[role].setPlainText(S.DEFAULT_PROMPTS[role])
        self.qs.remove(self._key(role))

    def _remember(self) -> None:
        for role, edit in self.prompts.items():
            text = edit.toPlainText().strip()
            if text and text != S.DEFAULT_PROMPTS[role]:
                self.qs.setValue(self._key(role), text)
            else:
                self.qs.remove(self._key(role))
        self.qs.setValue(SK.SHEETS_TRIES, self.tries.value())
        self.qs.setValue(SK.SHEETS_QUALITY, self.quality.currentData())

    # ---- making tries
    def make(self, role: str) -> None:
        self._start([role])

    def make_both(self) -> None:
        self._start(list(S.ROLES))

    def _start(self, roles: list[str]) -> None:
        if self.busy:
            return
        empty = [S.LABELS[r] for r in roles if not self.prompts[r].toPlainText().strip()]
        if empty:
            self.state.setText(f"The prompt of the {empty[0].lower()} is empty: write one (or Reset prompt).")
            return
        self._remember()
        rng = random.Random()
        jobs = [(r, self.prompts[r].toPlainText().strip(), rng.randint(1, SEED_MAX))
                for r in roles for _ in range(self.tries.value())]
        if self.on_start is not None:
            self.on_start()
        plan = SheetPlan(edits=self.project / edits.EDITS_DIR, connect=self.connect_creator, prompt="",
                         quality=str(self.quality.currentData()), n=len(jobs), seeds=[s for _r, _p, s in jobs],
                         runs=[[]], run_words=[[]], refs=[str(self.source)], family=S.FAMILY, jobs=jobs,
                         source=str(self.source), signature=self.signature,
                         work=Path(tempfile.mkdtemp(prefix="sheets_")))
        w = SheetWorker(plan)
        w.progress.connect(self._progress)
        w.result.connect(self._result)
        w.finished.connect(self._finished)
        self.worker, self.busy = w, True
        self.state.setText("Connecting to ComfyUI … (the first try loads Qwen-Image 2.1, which takes a while)")
        self._update()
        w.start()

    def stop(self) -> None:
        if self.worker is not None and self.busy:
            self.worker.cancel()
            self.state.setText("Stopping …")

    def _progress(self, k: int, n: int, step: int, steps: int) -> None:
        self.state.setText(f"Try {k} of {n}" + (f": step {step} / {steps}" if steps else " …"))

    def _result(self, png: str, meta: dict) -> None:
        self._add_result(png, str(meta.get("sheet")))

    def _finished(self, error: str) -> None:
        self.busy, self.worker = False, None
        self.state.setText("Stopped (the tries already made stay)." if error == "cancelled" else
                           f"Could not make the sheet: {error}" if error else
                           "Done: click the best try of each sheet, then Save.")
        self._update()

    def _add_result(self, png: str, role: str) -> None:
        if role not in self.results:
            return
        self.results[role].append(png)
        item = QListWidgetItem(QIcon(QPixmap(png).scaled(THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                                          Qt.TransformationMode.SmoothTransformation)), "")
        item.setData(Qt.ItemDataRole.UserRole, png)
        item.setToolTip(Path(png).name)
        gal = self.galleries[role]
        gal.addItem(item)
        if gal.currentItem() is None:
            gal.setCurrentItem(item)
        self._update()

    # ---- choosing
    def chosen(self, role: str) -> str | None:
        item = self.galleries[role].currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def choose(self, role: str, png: str) -> None:
        gal = self.galleries[role]
        for i in range(gal.count()):
            if gal.item(i).data(Qt.ItemDataRole.UserRole) == png:
                gal.setCurrentRow(i)

    def _update(self) -> None:
        idle = not self.busy
        for b in (*self.buttons.values(), self.b_both, self.b_other):
            b.setEnabled(idle)
        self.b_stop.setEnabled(self.busy)
        self.b_save.setEnabled(idle and all(self.chosen(r) for r in S.ROLES))

    def _save(self) -> None:
        if self.b_save.isEnabled():
            self._remember()
            self.accept()

    def done(self, r: int) -> None:                   # noqa: D401  (closing: a run still going is stopped)
        self.stop()
        self._remember()
        super().done(r)
