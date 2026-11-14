"""Joining and exporting a path, the GPU poll."""
from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QDialog

from .workers import run_in_background
from . import settings_keys as SK
from .. import chain, media, runpod as RP, tree as T
from .theme import repolish
from .widgets import format_time
from .project_dialogs import ExportDialog
from .generator_state import Refine
from .mw_host import MainWindowHost
from .qsvalue import qs_value


class ExportMixin(MainWindowHost):
    """Joining and exporting a path, the GPU poll."""

    # ------------------------------------------------------------------ paths
    def join_selected(self, purpose: str, out: Path | None = None, prefix: str = ""):
        path = self.selected_path()
        if not path or self.tree is None:
            return
        nid, project = path[-1].id, self.project
        assert project is not None
        tree = T.Tree.from_json(self.tree.to_json())
        pre = f"{prefix} " if prefix else ""
        self.status.setText(f"{pre}Joining the path …")
        lock = self._join_lock
        self._joins += 1
        self._update_buttons()

        def work():
            with lock:
                joined = T.join_path(project, tree, nid, out)
                secs = chain.joined_seconds(joined)             # not in the GUI thread (network shares)
                return joined, secs, T.path_spans(tree, nid)

        def error(e: BaseException):                           # (the answer ends "joining")
            self._joined(None, str(e) or type(e).__name__, purpose, pre, str(project))
        run_in_background(work, on_done=lambda r: self._joined(r, None, purpose, pre, str(project)),
                          on_error=error, owner=self)

    def _joined(self, path, error, purpose, pre, project):
        self._joins = max(0, self._joins - 1)
        self._update_buttons()
        if project != (str(self.project) if self.project is not None else ""):   # another project meanwhile
            return
        if error:
            self._report_error(f"{pre}Could not join the path: {error}")
            return
        path, secs, spans = path
        if purpose == "play":
            self._played_path = (str(path), spans)  # (before it opens: the clip it is at is marked while it plays)
            self.preview.open(path, f"Path to clip #{T.short_id(Path(path).stem)} · {format_time(secs, 1)}")
            self.status.setText(f"{pre}Path joined: {format_time(secs, 1)}.")
        else:
            self.status.setText(f"{pre}Exported {Path(path).name} ({format_time(secs, 1)}).")

    def _preview_opened(self, path: str):
        if self._played_path is not None and self._played_path[0] != path:
            self._played_path = None                # something else is played now
            self._mark_playing(None)

    def _preview_moved(self, secs: float):
        """The played path is at `secs` (-1: it stopped): its clip there is marked in the graph and timeline."""
        if self._played_path is None:
            return
        if secs < 0:
            self._mark_playing(None)
            return
        spans = self._played_path[1]
        nid = next((n for n, a, b in spans if a <= secs < b), spans[-1][0] if spans else None)
        self._mark_playing(nid)

    def _mark_playing(self, nid: str | None):
        self.graph.set_playing(nid)
        self.timeline.set_playing(nid)

    def export_path(self, single: str | None = None):
        """The export dialog for the selected path, or (single: a clip id) for that clip on its own."""
        if single is not None:
            node = self.tree.nodes.get(single) if self.tree else None
            if node is None or not node.made or self.state(single) != "done":
                self.status.setText("Make the clip first, then export it.")
                return
            path = [node]
        else:
            path = self.selected_path()
        if not path:
            return
        assert self.project is not None
        default = str(self.project / f"{self.project.name}-{path[-1].id}{'-clip' if single else ''}.mp4")
        assert self.tree is not None
        base = T.Checker(self.project, self.tree).base()
        strength = qs_value(self.qs, SK.REFINE_STRENGTH, "natural", str)
        method = qs_value(self.qs, SK.EXPORT_UPSCALER, "seedvr2_7b", str)
        denoise = qs_value(self.qs, SK.EXPORT_DENOISE, False, bool)

        def ready(size, how, strength_, denoise_=False):   # every generated clip has a kept copy made this way
            def kept(n):
                assert self.project is not None
                f = T.refined_file(self.project, n, size)
                if not (n.made and f.is_file()):
                    return False
                if n.is_video:
                    return True
                rec = T.refine_record(f)
                return (rec.get("method") == how and bool(rec.get("denoise")) == bool(denoise_)) if how != "refine" \
                    else ("method" not in rec and rec.get("strength") == strength_)
            return all(kept(n) for n in path)
        swapped = sum(1 for n in path if T.swapped_video(n))
        dlg = ExportDialog(self, default, (base["width"], base["height"]) if base else None, ready, swapped,
                           strength, method, denoise, cloud=True if RP.api_key() else None,
                           cloud_on=qs_value(self.qs, SK.EXPORT_CLOUD, False, bool), simple=self.simple, settings=self.qs)
        if single is not None:
            dlg.setWindowTitle(f"Export {'video' if path[0].is_video else 'clip'} #{T.short_id(single)}")
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            req = dlg.request()
        finally:
            dlg.deleteLater()
        if req.size != "made":
            self.qs.setValue(SK.REFINE_STRENGTH, req.strength)
            self.qs.setValue(SK.EXPORT_UPSCALER, req.method)
            self.qs.setValue(SK.EXPORT_DENOISE, req.denoise)
            if req.method != "refine":
                self.qs.setValue(SK.EXPORT_CLOUD, req.cloud)
        self.export(req.out, req.size, req.fast, req.swapped, req.strength, req.method, single,
                    req.denoise and req.method != "refine", cloud=req.cloud and req.method != "refine")

    def export(self, out: Path, size: str = "made", fast_attention: bool = False,
               refine_swapped: bool = False, strength: str = "natural", method: str = "seedvr2_7b",
               single: str | None = None, denoise: bool = False, cloud: bool = False) -> None:
        """Export the selected path (single: that clip alone) to out: as made (joined / copied here), or upscaled at
        '1080p' / '2k' (a generator job; cloud: SeedVR2 on a RunPod pod made for it)."""
        if single is not None:
            node = self.tree.nodes.get(single) if self.tree else None
            if node is None or not node.made:
                return
            if size == "made":
                assert self.project is not None
                self._export_copy(single, self.project / node.made["video"], Path(out))
                return
            target = single
        else:
            path = self.selected_path()
            if not path:
                return
            if size == "made":
                self.join_selected("export", out)
                return
            target = path[-1].id
        self._start([target], refine=Refine(size=size, out=str(out), fast_attention=bool(fast_attention),
                                            refine_swapped=bool(refine_swapped), strength=strength, method=method,
                                            single=single is not None, denoise=bool(denoise),
                                            cloud="runpod" if cloud and method != "refine" else ""))

    def _export_copy(self, nid: str, src: Path, out: Path):
        """Copy a made clip to out in the background (it can be big, and out on a share); like a join it holds back
        generating and deleting meanwhile. The file appears under its name only when it is complete (MWC-33)."""
        if self._joins:
            self.status.setText("A join or copy is still running; wait for it.")
            return
        self._joins += 1
        self._update_buttons()
        self.status.setText(f"Exporting clip #{T.short_id(nid)} to {out.name} …")

        def work():
            out.parent.mkdir(parents=True, exist_ok=True)
            with media.part_file(out) as tmp:
                shutil.copyfile(src, tmp)

        run_in_background(work, on_done=lambda _r: self._clip_copied(nid, str(out), ""),    # (the answer ends "exporting")
                          on_error=lambda e: self._clip_copied(nid, str(out), str(e) or type(e).__name__), owner=self)

    def _clip_copied(self, nid: str, out: str, error: str):
        self._joins = max(0, self._joins - 1)
        self._update_buttons()
        if error:
            self._report_error(f"Could not export clip #{T.short_id(nid)}: {error}")
        else:
            self.status.setText(f"Exported {Path(out).name} (clip #{T.short_id(nid)}, as made).")

    # ------------------------------------------------------------------ GPU
    def _poll_gpu(self):
        if not shutil.which("nvidia-smi"):
            self.gpu.setText("GPU: nvidia-smi not found")
            return
        if self._gpu_proc is not None:
            return
        p = QProcess(self)
        self._gpu_proc = p

        failed: list[int] = []

        def done(*_):
            if self._gpu_proc is not p:          # window closing: the query was abandoned
                return
            try:
                out = bytes(p.readAllStandardOutput().data()).decode(errors="replace").strip().splitlines()
            except RuntimeError:                 # Qt is destroying the window (and with it the query)
                return
            self._gpu_proc = None
            p.deleteLater()
            if failed:
                return
            try:
                used, total = (float(x) for x in out[0].split(","))
            except (IndexError, ValueError):
                self.gpu.setText("GPU: ?")
                return
            free = (total - used) / 1024
            self.gpu.setText(f"GPU {free:.1f} GB free of {total / 1024:.0f}")
            self.gpu.setProperty("severity", "warning" if free < 4 else "")
            repolish(self.gpu)

        def not_started(err):                    # (no "finished" follows a process that never started)
            if err == QProcess.ProcessError.FailedToStart:
                failed.append(1)
                self.gpu.setText("GPU: ?")
                done()
        p.finished.connect(done)
        p.errorOccurred.connect(not_started)
        p.start("nvidia-smi", ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
