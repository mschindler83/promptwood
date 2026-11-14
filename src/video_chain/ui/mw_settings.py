"""Window settings (QSettings), recent projects, the title."""
from __future__ import annotations

import json
from pathlib import Path

from .. import runpod as RP, tree as T, triggers
from . import ui_logic as UL
from .ui_logic import (
    DEFAULT_COMFY_LAUNCHER, DEFAULT_COMFY_URL, LAYOUT_VERSION, LLM_DEFAULT_PORT, LLM_DEFAULT_SERVER, MAX_RECENT)
from .mw_host import MainWindowHost
from .qsvalue import qs_value


class SettingsMixin(MainWindowHost):
    """Window settings (QSettings), recent projects, the title."""

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        q = self.qs
        self.keep_loaded.setChecked(qs_value(q, "keepLoaded", True, bool))
        self.comfy_url.setText(qs_value(q, "comfyUrl", DEFAULT_COMFY_URL, str))
        self.comfy_launcher.setText(qs_value(q, "comfyLauncher", DEFAULT_COMFY_LAUNCHER, str))
        self.comfy_dir.setText(qs_value(q, "comfyDir", UL.DEFAULT_COMFY_DIR, str))
        self.comfy_cleanup.setChecked(qs_value(q, "comfyCleanup", True, bool))
        self.comfy_autostart.setChecked(qs_value(q, "comfyAutoStart", True, bool))
        self.triggers_online.setChecked(qs_value(q, "loraTriggersOnline", True, bool))
        self.civitai_site.blockSignals(True)
        self.civitai_site.setCurrentIndex(max(0, self.civitai_site.findData(
            triggers.set_site(qs_value(q, "civitaiSite", "", str)))))
        self.civitai_site.blockSignals(False)
        self.llm_on.setChecked(qs_value(q, "llmServerOn", self.llm_model.count() > 0, bool))  # on when one is found
        self.llm_server.setText(qs_value(q, "llmServer", LLM_DEFAULT_SERVER, str))
        saved = qs_value(q, "llmModel", "", str)
        if saved:
            i = self.llm_model.findData(saved)
            if i < 0:
                self.llm_model.addItem(Path(saved).name, saved)
                i = self.llm_model.count() - 1
            self.llm_model.setCurrentIndex(i)
        self.llm_port.setValue(qs_value(q, "llmPort", LLM_DEFAULT_PORT, int))
        self.llm_cpu_moe.setChecked(qs_value(q, "llmCpuMoe", True, bool))
        self.video_gpu.blockSignals(True)
        self.video_gpu.setChecked(qs_value(q, "videoGpu", True, bool))
        self.video_gpu.blockSignals(False)
        self.idle_minutes.setValue(qs_value(q, "idleMinutes", 15, int))
        self._runpod_loading = True
        gpu = qs_value(q, "runpodGpu", RP.DEFAULT_GPU, str)
        i = self.runpod_gpu.findData(gpu)
        if i >= 0:
            self.runpod_gpu.setCurrentIndex(i)
        else:
            self.runpod_gpu.setEditText(gpu)
        self.runpod_fallback.setChecked(qs_value(q, "runpodFallback", True, bool))
        self.runpod_volume.setText(qs_value(q, "runpodVolume", "", str))
        self.runpod_without_volume.setChecked(qs_value(q, "runpodWithoutVolume", True, bool))
        self._runpod_loading = False
        if g := q.value("geometry"):
            self.restoreGeometry(g)
        self._restore_layout()
        self._fill_recent()
        proj = qs_value(q, "project", "", str)
        if proj and Path(proj).is_dir():
            self.open_project(Path(proj), quiet=True)

    def save_settings(self):
        q = self.qs
        q.setValue("keepLoaded", self.keep_loaded.isChecked())
        q.setValue("comfyUrl", self.comfy_url.text().strip())
        q.setValue("comfyLauncher", self.comfy_launcher.text().strip())
        q.setValue("comfyDir", self.comfy_dir.text().strip())
        q.setValue("comfyCleanup", self.comfy_cleanup.isChecked())
        q.setValue("comfyAutoStart", self.comfy_autostart.isChecked())
        q.setValue("loraTriggersOnline", self.triggers_online.isChecked())
        q.setValue("civitaiSite", self.civitai_site.currentData())
        q.setValue("videoGpu", self.video_gpu.isChecked())
        q.setValue("llmServerOn", self.llm_on.isChecked())
        q.setValue("llmServer", self.llm_server.text().strip())
        q.setValue("llmModel", self.llm_model_path())
        q.setValue("llmPort", self.llm_port.value())
        q.setValue("llmCpuMoe", self.llm_cpu_moe.isChecked())
        q.setValue("idleMinutes", self.idle_minutes.value())
        q.setValue("project", str(self.project) if self.project else "")
        if self.tree is not None:
            q.setValue("lastOptions", json.dumps(self.tree.options))
        q.setValue("geometry", self.saveGeometry())
        q.setValue("layout", json.dumps({"v": LAYOUT_VERSION, "split": self.split.sizes(),
                                         "center": self.center.sizes(), "left": self.act_left.isChecked(),
                                         "left_w": (self.split.sizes()[0] if self.act_left.isChecked() else 0)
                                         or self._left_w}))
        q.sync()

    # ---- recent projects
    def recent_projects(self) -> list[str]:
        try:
            items = json.loads(qs_value(self.qs, "recentProjects", "[]", str))
        except ValueError:
            items = []
        return [p for p in items if isinstance(p, str)]

    def _remember_project(self, folder: Path):
        items = [str(folder)] + [p for p in self.recent_projects() if p != str(folder)]
        self.qs.setValue("recentProjects", json.dumps(items[:MAX_RECENT]))
        self._fill_recent()

    @staticmethod
    def _is_project(folder: Path) -> bool:
        """A folder with a project (tree.json, or the chain.json of before branching)."""
        return (Path(folder) / T.TREE_FILE).exists() or (Path(folder) / "chain.json").exists()

    def _fill_recent(self):
        recent = self.recent_projects()
        items = [p for p in recent if self._is_project(Path(p))]
        for menu in (self.recent_menu, self.recent_file_menu):
            menu.clear()
            for p in items:
                act = menu.addAction(f"{Path(p).name}   —   {Path(p).parent}")
                act.triggered.connect(lambda _=False, f=p: self.open_project(Path(f)))
            if not items:
                menu.addAction("no recent projects").setEnabled(False)
        self.b_recent.setEnabled(bool(recent))

    def _update_title(self):
        if self.project is None:
            self.setWindowTitle("Promptwood")
            return
        scene = self.film.scene_of(self.project) if self.film else None
        name = self.film.root.name if self.film else self.project.name
        many = self.film is not None and len(self.film.scenes) > 1
        self.setWindowTitle(f"{name} › {scene.title} — Promptwood" if many and scene else f"{name} — Promptwood")

    # ---- the whole film: play, export, generate
    def _busy_reason(self) -> str:
        """Why the project must not be changed, opened, closed or copied right now ("" when it may)."""
        if self.running:
            return "Wait until generating has finished."
        if self._joins or self._converts or self._copying:
            return "Wait until generating, joins, conversions and copies have finished."
        return ""

    def _idle_or_say(self) -> bool:
        """True when nothing runs; else the status says why the button did nothing."""
        why = self._busy_reason()
        if why:
            self.status.setText(why)
        return not why
