"""ComfyUI: poll, start, free its models, close the window."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QProcess, Qt, QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog

from .. import comfy as CMF, net
from . import dialogs as D, shortcuts as S
from .theme import repolish
from .widgets import safe_emit
from .generator_state import ComfyModel, IMAGE_MODELS, ModelState
from .ui_logic import CLOSE_GRACE, COMFY_START_TIMEOUT, DEFAULT_COMFY_LAUNCHER, DEFAULT_COMFY_URL
from .mw_host import MainWindowHost


class ComfyMixin(MainWindowHost):
    """ComfyUI: poll, start, free its models, close the window."""

    def unload(self):
        if self.worker_alive and not self.running:
            self.status.setText("Unloading the model …")
            self._quit_worker()
        elif self._comfy_model in IMAGE_MODELS and not self.running:
            self.status.setText("Unloading the image model …")
            self._free_comfy()

    def _free_comfy(self):
        """Ask ComfyUI to unload its models (in the background)."""
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        self._qwen_timer.stop()

        def work():
            try:
                ok = CMF.free_models(url, timeout=10)
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).warning("unloading ComfyUI's models", exc_info=True)
                ok = False
            safe_emit(self.bridge.freed, ok)
        threading.Thread(target=work, daemon=True).start()

    def _comfy_freed(self, ok: bool):
        """ComfyUI answered the unload request (MWC-30): the model counts as unloaded only when it did."""
        if ok:
            self._comfy_model = None
            self.log.appendPlainText("ComfyUI's models were unloaded.")
        else:
            self._report_error("Could not unload the model: ComfyUI did not answer; it may still hold the memory.")
        self._update_model_label()

    def _qwen_idle(self):
        if self._comfy_model in IMAGE_MODELS and not self.running:
            self._free_comfy()
            self.status.setText("The image model was unloaded after the idle time.")

    # ---- ComfyUI
    def _poll_comfy(self):
        if self._comfy_polling:
            return
        self._comfy_polling = True
        url, bridge = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/"), self.bridge

        def check():
            try:
                with net.urlopen(url + "/system_stats", timeout=1.5) as r:
                    ok = r.status == 200
            except Exception:  # noqa: BLE001
                ok = False
            try:
                safe_emit(bridge.comfy, ok)
            except RuntimeError:                     # (the window was closed meanwhile)
                pass
        threading.Thread(target=check, daemon=True).start()

    def _autostart(self, targets: list[str], refine: dict | None = None) -> bool:
        """ComfyUI is down: start it (if allowed) and generate targets once it answers. False: not started."""
        if not self.comfy_autostart.isChecked():
            return False
        if not (self._comfy_starting and time.monotonic() - self._comfy_starting < COMFY_START_TIMEOUT):
            if not self.start_comfy():
                return False
        self._waiting_comfy = list(targets)
        self._waiting_refine = refine                 # an export waits too, not a plain generate
        self.status.setText("Starting ComfyUI … generating begins as soon as it runs (Stop cancels that).")
        self._update_buttons()
        return True

    def _comfy_state(self, up: bool):
        self._comfy_polling = False
        was, self.comfy_up = self.comfy_up, up
        if up and was is False and self._comfy_starting:
            self._comfy_starting = 0.0
            self.status.setText("ComfyUI is running.")
        if self._waiting_comfy is not None:
            if up and QApplication.activeModalWidget() is not None:
                self.status.setText("ComfyUI is running; generating starts when you close the dialog.")  # (it may
                #                                              change the project: later, at the next poll)
            elif up:
                targets, self._waiting_comfy = self._waiting_comfy, None
                if self.tree is not None and all(t in self.tree.nodes for t in targets):
                    refine = self._waiting_refine
                    QTimer.singleShot(0, lambda: self._start(targets, refine))
                else:
                    self._film_queue = None
            elif not (self._comfy_starting and time.monotonic() - self._comfy_starting < COMFY_START_TIMEOUT):
                self._waiting_comfy = None
                self._film_queue = None
                self.status.setText("ComfyUI did not start within 3 minutes; see its own log.")
            self._update_buttons()
        self._update_comfy_label()

    def _update_comfy_label(self):
        if self.comfy_up:
            text, sev = "Video engine: running", ""
        elif self._comfy_starting and time.monotonic() - self._comfy_starting < COMFY_START_TIMEOUT:
            text, sev = "Video engine: starting …", "warning"
        elif self.comfy_up is False:
            text, sev = "Video engine: not running", "warning"
        else:
            text, sev = "Video engine: …", ""
        self.comfy_label.setText(text)
        self.comfy_label.setToolTip(f"The video engine is ComfyUI, the program that makes the clips "
                                    f"({self.comfy_url.text().strip() or DEFAULT_COMFY_URL}); its settings: "
                                    f"{S.keys('preferences')}")
        self.comfy_label.setProperty("severity", sev)
        repolish(self.comfy_label)
        self.b_start_comfy.setEnabled(not self.comfy_up)
        self._update_engine_banner()

    def start_comfy(self) -> bool:
        """Start ComfyUI in the background with the user's launcher (it runs on its own, not tied to this app)."""
        cmd = os.path.expanduser(self.comfy_launcher.text().strip() or DEFAULT_COMFY_LAUNCHER)
        if not Path(cmd).is_file():
            self.status.setText(f"ComfyUI launcher not found: {cmd}")
            return False
        ok = QProcess.startDetached(cmd, [])
        ok = ok[0] if isinstance(ok, tuple) else ok
        if not ok:
            self._report_error(f"Could not start {cmd}")
            return False
        self._comfy_starting = time.monotonic()
        self.status.setText("Starting ComfyUI … (the clips can be generated once it runs)")
        self._update_comfy_label()
        return True

    def _update_model_label(self):
        if not self.worker_alive and self._comfy_model in IMAGE_MODELS:
            left = self._qwen_timer.remainingTime() if self._qwen_timer.isActive() else -1
            text, sev = (("Model: language model (prompt refiner) loaded in ComfyUI"
                          if self._comfy_model == ComfyModel.LLM
                          else "Model: image model loaded in ComfyUI") +
                         (f" · unloads in {max(1, round(left / 60000))} min" if left >= 0 else "")), ""
        elif not self.worker_alive:
            text, sev = "Model: not loaded", ""
        elif self.running and self._model_state == ModelState.LOADING:
            text, sev = "Model: loading …", "warning"
        elif self.running or not self._worker_idle:
            text, sev = "Model: working", ""
        elif self._idle_timeout:
            left = max(0.0, self._idle_timeout - (time.monotonic() - self._idle_since))
            text, sev = f"Model: loaded in ComfyUI · unloads in {max(1, round(left / 60))} min", ""
        else:
            text, sev = "Model: loaded", ""
        self.model_label.setText(text)
        self.model_label.setProperty("severity", sev)
        repolish(self.model_label)
        self.b_unload.setVisible((self.worker_alive and not self.running and self._worker_idle) or
                                 (not self.worker_alive and self._comfy_model in IMAGE_MODELS))

    def changeEvent(self, e):
        """Back in this window (e.g. after training a LoRA): look for new LoRAs, at most every 30 s."""
        super().changeEvent(e)
        if e.type() == e.Type.ActivationChange and self.isActiveWindow() and \
                time.monotonic() - self._scanned_at > 30:
            self.scan_loras()

    def _post_free(self, url: str, done: threading.Event | None = None):
        """Ask ComfyUI to unload its models, in a thread (the GUI never waits for the network)."""
        def work():
            try:
                CMF.free_models(url, timeout=3)
            except Exception:  # noqa: BLE001  (free_models reports its own failures)
                logging.getLogger(__name__).warning("unloading ComfyUI's models", exc_info=True)
            finally:
                if done is not None:
                    done.set()
        threading.Thread(target=work, daemon=True).start()

    def _wait_for_shutdown(self, p: QProcess | None, cloud: bool, also: threading.Event | None):
        """Wait (the window stays alive and says what it waits for) until the generator process has ended: it gets
        3 s to quit by itself, then is asked to terminate (a cloud worker ends its pod: long), then killed."""
        def over() -> bool:
            return (p is None or p.state() == QProcess.ProcessState.NotRunning) and (also is None or also.is_set())
        if over():
            return
        loop = QEventLoop(self)
        timers: list[QTimer] = []

        def later(ms, fn):
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(fn)
            t.start(ms)
            timers.append(t)

        def check():
            if over():
                loop.quit()
        later_ms = 3000 + (45000 if cloud else 5000)
        if p is not None and p.state() != QProcess.ProcessState.NotRunning:
            later(3000, lambda: p.terminate() if p.state() != QProcess.ProcessState.NotRunning else None)
            later(later_ms, lambda: p.kill() if p.state() != QProcess.ProcessState.NotRunning else None)
        later(later_ms + 3000, loop.quit)
        tick = QTimer(self)
        tick.timeout.connect(check)
        tick.start(50)
        timers.append(tick)
        box = QProgressDialog("Shutting down the generator" + (" (ending the cloud pod, this can take a minute)"
                                                               if cloud else "") + " …", "", 0, 0, self)
        box.setCancelButton(None)
        box.setWindowTitle("Promptwood")
        box.setWindowModality(Qt.WindowModality.WindowModal)
        box.show()
        loop.exec()
        box.close()
        for t in timers:
            t.stop()
            t.deleteLater()

    def closeEvent(self, e):
        if self._checking:                             # Generate is still checking the project: nothing runs yet
            self._cancel_check()
        was_running = self.running
        if was_running:
            if not D.confirm(
                    self, "Quit while generating?",
                    "Clips are being generated. Stop and quit?\n\nClips that are finished are kept; the one being "
                    "made is lost."):
                e.ignore()
                return
        elif self._joins or self._converts or self._copying:
            what = "copied (Save as)" if self._copying else "converted" if self._converts else "joined"
            if not D.confirm(self, "Quit while working?", f"A video is being {what}. Quit anyway? What is not "
                                 "finished is lost."):
                e.ignore()
                return
        if self.project is not None and not was_running and not self._may_leave_project():
            e.ignore()                                # the user keeps the window to save elsewhere / reopen
            return                                    # (idle: the project is saved now, before anything is shut down)
        self._save_timer.stop()
        if self._trigger_scan is not None:
            self._trigger_scan.set()                  # the trigger word lookup stops at the next file
        self.stop_llama(wait=True, grace=CLOSE_GRACE)   # never leave the language model server behind (but be quick)
        self.stop_tts()
        timers = [t for t in (self._qwen_timer, self._model_timer, self._gpu_timer, self._comfy_timer) if t.isActive()]
        for t in timers:
            t.stop()
        if self._gpu_proc is not None:
            gp, self._gpu_proc = self._gpu_proc, None
            gp.finished.disconnect()
            gp.kill()
            gp.waitForFinished(1000)
        freed = None
        if self._comfy_model in IMAGE_MODELS:              # nor the image edit model
            freed = threading.Event()
            self._post_free((self.comfy_url.text().strip() or DEFAULT_COMFY_URL), freed)
            self._comfy_model = None
        p = self.proc
        cloud = self._cloud_worker
        if p is not None:                            # never leave the generator (and the model) behind
            self.proc = None
            p.finished.disconnect()
            if p.state() == QProcess.ProcessState.Running:
                p.write(b'{"cmd": "quit", "free": true}\n')     # ComfyUI: give the memory back too
                p.closeWriteChannel()
        self._wait_for_shutdown(p, cloud, freed)
        if was_running:                              # the generator wrote tree.json meanwhile: take that, then save
            self._job = None
            self._reload_made()
            if not self._may_leave_project():
                e.ignore()
                for t in timers:                     # (the window stays: its GPU / ComfyUI status goes on)
                    t.start()
                self._update_buttons()
                return
        self.preview.stop()
        self.save_settings()
        super().closeEvent(e)
