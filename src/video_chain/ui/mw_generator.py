"""The generator worker: start, jobs, events, finish."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer

from .. import comfy as CMF, runpod as RP, tree as T
from ..__main__ import COMFY_DOWN
from .widgets import safe_emit
from .generator_state import (
    CLIP_MODELS, ComfyBinding, ComfyModel, IMAGE_MODELS, Job, ModelState, Phase, RunpodBinding, WorkerBinding)
from .ui_logic import CLOUD_KILL_GRACE_MS, DEFAULT_COMFY_URL, KILL_GRACE_MS, fmt_left, plural, time_left
from .mw_host import MainWindowHost
from .qsvalue import qs_value


def step_status(nid: str, k: int, total: int, what: str, frac: float, left: float | None,
                left_all: float | None) -> str:
    """The status line of a clip being made: 'Clip #3 (2 of 5): step 4 / 20 · 20 % · about 40 s left (all clips:
    about 3 min)'."""
    text = f"Clip #{T.short_id(nid)} ({k} of {total}): {what} · {round(100 * frac)} %"
    if left is not None:
        text += f" · {fmt_left(left)} left"
    if left_all is not None and total > 1 and k < total:
        text += f" (all clips: {fmt_left(left_all)})"
    return text


class GeneratorMixin(MainWindowHost):
    """The generator worker: start, jobs, events, finish."""

    @property
    def running(self) -> bool:
        """A generation job is in progress (editing is locked)."""
        return self._job is not None or self._checking

    @property
    def worker_alive(self) -> bool:
        return self.proc is not None

    def generate_path(self):
        path = self.selected_path()
        if not path:
            self.status.setText("Select the last clip of the path you want to make.")
            return
        self._start([path[-1].id])

    def generate_all(self):
        if self.tree is not None:
            self._start([n.id for n in self.tree.leaves()])

    @property
    def _refine(self) -> dict | None:
        """The refine settings of the export in progress (None: no job, or a normal run)."""
        return self._job.refine if self._job is not None else None

    def _worker_key(self) -> WorkerBinding:
        """What a running worker is bound to (which ComfyUI, its folder, cleanup; or the RunPod pod of a cloud
        export: GPU, volume, fallback, SeedVR2 size); a change needs a new worker."""
        refine = self._refine or {}
        if refine.get("cloud"):
            return RunpodBinding(qs_value(self.qs, "runpodGpu", RP.DEFAULT_GPU, str),
                                 qs_value(self.qs, "runpodVolume", "", str).strip(),
                                 qs_value(self.qs, "runpodFallback", True, bool),
                                 CMF.SEEDVR_METHODS.get(refine.get("method"), "7b"),
                                 qs_value(self.qs, "runpodWithoutVolume", True, bool))
        return ComfyBinding((self.comfy_url.text().strip() or DEFAULT_COMFY_URL), self.comfy_dir.text().strip(),
                            self.comfy_cleanup.isChecked())

    @property
    def _cloud_worker(self) -> bool:
        """The running worker is a RunPod one (it ends, and its pod with it, after its export)."""
        return bool(self.proc is not None and isinstance(self._worker_bound, RunpodBinding))

    def _start(self, targets: list[str], refine: dict | None = None):
        if self.running or self.tree is None:
            return
        self._style_changed()                      # a style edit not finished with Enter / leaving the field
        if self._joins:
            self.status.setText("Wait until the path is joined.")
            return
        if self._copying:
            self.status.setText("Wait until the project is copied (Save as).")
            return
        if self._converts:
            self.status.setText("Wait until the video is converted.")
            return
        if self._tts_busy:
            self.status.setText("Wait until text to speech is done.")
            return
        self._checking, self._check_gen = True, self._check_gen + 1
        gen, project, tree, bridge = self._check_gen, self.project, self.tree, self.bridge
        self._check_args = (list(targets), refine)
        self.status.setText("Checking the project …")
        for c in self.cards:
            c.set_editable(False)
        self._update_buttons()                     # (running: nothing can be edited while it is checked)

        def work():
            problems, empty = self._preflight(project, tree, targets, refine)
            try:
                safe_emit(bridge.checked, gen, problems, empty)
            except RuntimeError:                   # (the window was closed meanwhile)
                pass
        threading.Thread(target=work, daemon=True).start()

    def _cancel_check(self):
        """Stop waiting for the pre-flight (Stop, closing): its result is not used any more."""
        self._checking, self._check_gen = False, self._check_gen + 1
        for c in self.cards:
            c.set_editable(not self.running)
        self._update_buttons()

    @staticmethod
    def _preflight(project: Path, tree: T.Tree, targets: list[str], refine: dict | None) -> tuple[list[str], list[str]]:
        """What keeps these targets from being made: (the problems as short phrases, ids of clips without a prompt).
        Reads files and hashes LoRAs, so it runs in a thread; nothing is changed (the window is locked meanwhile)."""
        problems = []
        turbo = tree.options.get("turbo") or {}
        if turbo.get("on"):
            if not turbo.get("lora"):
                problems.append("choose a turbo LoRA (Clips → Turbo LoRA) or switch Speed to Normal")
        nodes = {n.id: n for t in targets for n in tree.path_to(t)}
        empty = [n for n in nodes.values() if not n.is_video and not n.prompt.strip()]
        if empty:
            problems.append(f"write the prompt of clip #{T.short_id(empty[0].id)}")
        try:
            checker = T.Checker(project, tree)
            if checker.base() is None:
                problems.append("choose a start image (or begin with a real video)")
            else:
                todo = T.plan(tree, checker, targets)
                if refine and todo:
                    problems.append(f"generate the path first ({len(todo)} clip{'s are' if len(todo) != 1 else ' is'} "
                                    "new or changed)")
                gone = [n for n in todo if tree.nodes[n].is_video
                        and not Path(tree.nodes[n].source["file"]).is_file()]
                if gone:
                    problems.append(f"the file of video #{T.short_id(gone[0])} is missing (Edit range… to choose it)")
                problem = T.guide_problem(tree, checker, todo)
                if problem:
                    problems.append(problem)
                no_end = T.missing_end_images(project, tree, todo)
                if no_end:
                    problems.append(f"the end image of clip #{T.short_id(no_end[0])} is missing (choose it again)")
                sounds = T.audio_problems(project, tree, todo)
                if sounds:
                    problems.append(sounds[0] + " (Sound reference)")
                cont = T.source_sound_problems(tree, todo, project)
                if cont:
                    problems.append(cont[0] + " (the clip's “Sound” setting)")
                faces = T.cast_problems(project, tree, todo)
                if faces:
                    problems.append(faces[0] + " (Cast)")
            if checker.base() is not None:
                for lo in T.loras_needed(tree, checker, list(nodes)):
                    if not Path(lo).is_file():
                        problems.append(f"the LoRA file does not exist ({Path(lo).name})")
                        break
        except Exception as e:  # noqa: BLE001
            problems.append(f"could not check the settings ({e})")
        return problems, [n.id for n in empty]

    def _runpod_key_hint(self) -> str:
        """Where the RunPod key is set; in the Simple view that place is hidden, so the way there is named (UX-3)."""
        where = "Settings → Cloud upscaling"
        if self.simple:
            where = "the Advanced view (View → Advanced view), then Settings → Preferences → Cloud upscaling"
        return f"To upscale on RunPod: save your RunPod API key ({where})."

    def _checked(self, gen: int, problems: list, empty: list):
        """Generate's pre-flight is done: report what is missing, or go on with the job."""
        if gen != self._check_gen or not self._checking or self.tree is None:
            return                                  # (stopped, closed or another project meanwhile)
        self._checking = False
        targets, refine = self._check_args
        if problems:
            self.status.setText("To start: " + ", ".join(problems) + ".")
            self._film_queue = None                 # (a film run does not go on after a scene that cannot start)
            self._cancel_check()
            if empty and len(targets) > 1:
                self.select(empty[0])
            return
        nodes = {n.id: n for t in targets for n in self.tree.path_to(t)}
        cloud = bool((refine or {}).get("cloud"))
        if cloud and not RP.api_key():
            self.status.setText(self._runpod_key_hint())
            self._film_queue = None
            self._cancel_check()
            return
        if not cloud and self.comfy_up is False and not self.worker_alive and self._autostart(targets, refine):
            self._cancel_check()
            return
        self._waiting_comfy = None
        self._save_timer.stop()
        if not self._write_tree():
            self._film_queue = None
            self._cancel_check()
            return
        self.save_settings()
        self._job = Job(targets, refine)
        self._run_started, self._clip_started, self._clip_note = time.monotonic(), time.monotonic(), ""
        self._exported = None
        self._error, self._stopping, self._phase = Phase.IDLE, False, Phase.STARTING
        self._targets, self._plan = targets, []
        for n in nodes:                            # made again now: what went wrong last time goes
            self._failures.pop(n, None)
        self._made_now: set[str] = set()           # the clips really made (an export reuses kept refined copies)
        self.live, self.progress_of = {}, {}
        self.log.clear()
        self.preview.stop()
        for c in self.cards:
            c.set_editable(False)
        self._update_buttons()
        self.progress.setRange(0, 0)
        if (self.worker_alive and self._worker_idle and not self._worker_leaving
                and self._worker_bound == self._worker_key()):
            self._send_job()
        elif self.worker_alive and self._worker_leaving:  # it just said bye (idle timeout): start a new one
            self.status.setText("Starting the generator again …")
            self._replace_worker = True
            self._quit_worker()
        elif self.worker_alive:                    # bound to another ComfyUI address or folder: replace it
            self.status.setText("ComfyUI settings changed: connecting again …")
            self._replace_worker = True
            self._quit_worker()
        else:
            self._start_worker()

    def _program(self) -> tuple[str, list[str]]:
        if self.command:
            return self.command[0], list(self.command[1:])
        return sys.executable, ["-m", "video_chain"]

    def _start_worker(self):
        program, pre = self._program()
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read_out)
        self.proc.readyReadStandardError.connect(self._read_err)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._proc_error)
        self._buf, self._ebuf = b"", ""
        self._worker_idle, self._worker_bound, self._replace_worker = False, self._worker_key(), False
        self._worker_leaving = False
        minutes = self.idle_minutes.value() if self.keep_loaded.isChecked() else 0
        bound = self._worker_bound
        if isinstance(bound, RunpodBinding):          # a pod for this export only: never idles, quits after it
            args = pre + ["--serve", "--events", "--idle-minutes", "0", "--cloud", "runpod", "--runpod-gpu", bound.gpu,
                          "--runpod-models", bound.models]
            if bound.volume:
                args += ["--runpod-volume", bound.volume]
            if not bound.fallback:
                args.append("--runpod-no-fallback")
            if bound.volume and bound.without_volume:
                args.append("--runpod-without-volume")
            self._kill_timer.setInterval(CLOUD_KILL_GRACE_MS)        # (time to terminate the pod on Stop)
        else:
            args = pre + ["--serve", "--events", "--idle-minutes", f"{minutes:g}", "--comfy-url", bound.url]
            if bound.comfy_dir:
                args += ["--comfy-dir", os.path.expanduser(bound.comfy_dir)]
            if not bound.cleanup:
                args.append("--keep-comfy-copies")
            self._kill_timer.setInterval(KILL_GRACE_MS)
        self.log.appendPlainText("$ " + " ".join([program, *args]))
        self.status.setText("Starting …")
        self.proc.start(program, args)
        self._update_model_label()

    def _send_job(self):
        if self._job is None or self.proc is None:
            return
        if self.proc.state() != QProcess.ProcessState.Running:    # (it ended meanwhile: _finished starts a new one)
            return
        self._job.sent = True
        self._worker_idle = False
        self.stop_llama("its RAM and VRAM are needed for the clips")
        cmd = {"cmd": "run", "tree": str(self.project), "targets": self._job.targets,
               "free_first": self._comfy_model in IMAGE_MODELS}          # an image edit model is loaded: unload it
        if self._comfy_model in CLIP_MODELS:                        # earlier workers: was HyperFlow last?
            cmd["last_hyperflow"] = self._comfy_model == ComfyModel.H3_HYPERFLOW
        if self._job.refine:                                       # export refined: re-render the path's clips
            cmd.update({"cmd": "refine", "target": self._job.targets[0], **self._job.refine})
        if self._cloud_worker:                                     # the local ComfyUI is not touched
            cmd.pop("free_first", None)
            cmd.pop("last_hyperflow", None)
            self._write_job(cmd)
            self._update_model_label()
            return
        self._qwen_timer.stop()
        hyper = bool((self.tree.options.get("hyperflow") or {}).get("on")) and not self._job.refine \
            if self.tree is not None else False
        self._comfy_model = ComfyModel.H3_HYPERFLOW if hyper else ComfyModel.H3
        self._write_job(cmd)
        self._update_model_label()

    def _write_job(self, cmd: dict):
        """Send a command line to the generator; if that fails the generator is ended, so the run fails cleanly."""
        if self.proc.write((json.dumps(cmd) + "\n").encode()) < 0:
            self._report_error("Could not send the job to the generator.")
            self._kill()

    def _quit_worker(self, free: bool = True):
        """Ask an idle worker to quit (free: and unload the model); killed if it does not go."""
        if self.proc is None:
            return
        self._quit_frees = free
        self._worker_leaving = True                       # no new job for it (a new worker starts instead)
        if self.proc.state() == QProcess.ProcessState.Running:
            self.proc.write(b'{"cmd": "quit", "free": true}\n' if free else b'{"cmd": "quit"}\n')
        self._kill_timer.start()

    def stop(self):
        if self._checking:                         # still checking the project: nothing was started
            self._cancel_check()
            self._film_queue = None
            self.status.setText("Stopped: nothing was started.")
            return
        if self._waiting_comfy is not None and not self.running:   # generating once ComfyUI runs: not any more
            self._waiting_comfy = None
            self._film_queue = None
            self.status.setText("Cancelled: nothing is generated when ComfyUI is up (ComfyUI itself keeps "
                                "starting).")
            self._update_buttons()
            return
        if not self.running or self.proc is None:
            return
        self._stopping = True
        self.status.setText("Stopping …")
        self.proc.terminate()
        self._kill_timer.start()

    def _kill(self):
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()

    def _read_out(self):
        p = self.proc
        if p is not None:
            self._drain_out(p)

    def _drain_out(self, p: QProcess, final: bool = False):
        """Lines of the worker's output: bytes, so a character split between reads stays whole; the last, unfinished
        line waits for more unless this is the end (final)."""
        self._buf += bytes(p.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        if final:
            lines, self._buf = lines + [self._buf], b""
        for line in lines:
            self._safe_line(line.decode(errors="replace"))

    def _safe_line(self, line: str):
        """One line of the worker's output; a broken event must not lose the lines after it."""
        try:
            self._line(line)
        except Exception as e:  # noqa: BLE001
            self.log.appendPlainText(f"(could not handle the worker's message: {type(e).__name__}: {e})")

    def _read_err(self):
        p = self.proc
        if p is not None:
            self._drain_err(p)

    def _drain_err(self, p: QProcess, final: bool = False):
        """Lines of the worker's error output into the log (a carriage-return progress line shows its last state)."""
        self._ebuf += bytes(p.readAllStandardError()).decode(errors="replace")
        *lines, self._ebuf = self._ebuf.split("\n")
        if final:
            lines, self._ebuf = lines + [self._ebuf], ""
        for line in lines:
            line = line.split("\r")[-1].rstrip()
            if line:
                self.log.appendPlainText(line)

    def _line(self, line: str):
        line = line.split("\r")[-1].rstrip()
        at = line.find("@@ {")                    # an event can follow text printed without a newline
        if at >= 0:
            try:
                ev = json.loads(line[at + 3:])
            except ValueError:
                ev = None
            if ev is not None:
                if line[:at].strip():
                    self.log.appendPlainText(line[:at].rstrip())
                self._event(ev)
                return
        if line:
            self.log.appendPlainText(line)

    def _overall(self, k: int, frac: float):
        if self._plan:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * ((k - 1) + frac) / len(self._plan)))

    def _reload_made(self):
        """Take what the generator recorded (made) from tree.json; nothing else changes during a run."""
        try:
            assert self.project is not None
            fresh = T.load(self.project)
        except (OSError, ValueError):
            return
        for nid, n in fresh.nodes.items():
            assert self.tree is not None
            if nid in self.tree.nodes:
                self.tree.nodes[nid].made = n.made
                self.tree.nodes[nid].end_at = n.end_at   # a clip made again hands over at its last frame again
        self._tree_stamp = self._stamp()            # the generator wrote it; this window is up to date

    def _event(self, ev: dict):
        """One event of the worker ('@@ {json}' line): the handler of its kind, then the plugins hear it."""
        handler = getattr(self, "_ev_" + str(ev.get("e")), None)
        if handler is not None:
            handler(ev)
        self.plugins.emit("run_event", ev)             # (the plugins hear it once the app has handled it)

    def _ev_loading(self, ev: dict):
        self._worker_idle = False
        self.status.setText("Starting a RunPod pod …" if ev.get("what") == "runpod" else "Connecting to ComfyUI …")
        self.progress.setRange(0, 0)
        self._model_state = ModelState.LOADING
        self._update_model_label()

    def _ev_loaded(self, ev: dict):
        self.status.setText("Connected to the RunPod pod." if ev.get("what") == "runpod" else
                            "Connected to ComfyUI.")
        self._model_state = ModelState.LOADED

    def _ev_idle(self, ev: dict):
        self._worker_idle = True
        self._idle_since = time.monotonic()
        self._idle_timeout = ev.get("timeout")
        if self._job is not None and not self._job.sent:
            self._send_job()
        elif self._job is not None:
            self._job_finished()
        self._update_model_label()

    def _ev_bye(self, ev: dict):
        self._worker_idle = False
        self._worker_leaving = True
        if self._job is not None and self._job.sent and self._phase != Phase.DONE:
            self._replace_worker = True               # our job crossed its goodbye: a new worker takes it
        if self._comfy_model in CLIP_MODELS and (ev.get("reason") == "idle" or self._quit_frees):
            self._comfy_model = None                       # it unloaded the clip model
        if ev.get("reason") == "idle" and not self.running:
            self.status.setText("The model was unloaded after the idle time.")

    def _ev_start(self, ev: dict):
        self.status.setText(f"{ev.get('nodes', 0)} clips to make at {ev['width']}×{ev['height']}, "
                            f"{ev['frames']} frames each.")

    def _ev_plan(self, ev: dict):
        self._plan = list(ev.get("make", []))

    def _ev_node(self, ev: dict):
        nid, st = ev.get("id"), ev.get("state")
        if not isinstance(nid, str) or not isinstance(st, str):
            logging.getLogger(__name__).warning("ignoring a node event without id or state: %r", ev)
            return
        if st == "generating":
            self._made_now.add(nid)
            self.live[nid] = "generating"
            self.progress_of[nid] = 0.0
            self._clip_started, self._clip_note = time.monotonic(), ""
            k = self._plan.index(nid) + 1 if nid in self._plan else 1
            self._overall(k, 0)
            node = self.tree.nodes.get(nid) if self.tree else None
            fast = node is not None and self.veda_used(node) and not self._refine
            self.status.setText(f"Clip #{T.short_id(nid)} ({k} of {len(self._plan)}) …"
                                + (" with ⚡ Veda fast attention (a large clip)" if fast else ""))
        else:
            self.live.pop(nid, None)
            self.progress_of.pop(nid, None)
            if nid in self._plan:
                self._overall(self._plan.index(nid) + 1, 1)
            self._reload_made()
        self.refresh()
        if st == "generating":
            self.graph.reveal(nid)

    def _ev_step(self, ev: dict):
        nid = ev.get("id")
        if not (nid and ev.get("n")):
            return
        frac = ev["i"] / ev["n"]
        self.progress_of[nid] = frac
        k = self._plan.index(nid) + 1 if nid in self._plan else 1
        self._overall(k, frac)
        upscaling = bool(self._refine)
        now = time.monotonic()
        left = time_left(now - self._clip_started, frac)
        n_plan = max(1, len(self._plan))
        left_all = time_left(now - self._run_started, ((k - 1) + frac) / n_plan)
        what = (self._clip_note or "upscaling") if upscaling else f"step {ev['i']} / {ev['n']}"
        self.status.setText(step_status(nid, k, len(self._plan), what, frac, left, left_all))
        item = self.graph.items_by_id.get(nid)
        if item is not None:
            item.info.progress = frac
            item.update()
        card = self._card_of(nid)
        if card is not None:
            text = None
            if upscaling:
                text = f"{round(100 * frac)} %" + (f" · {fmt_left(left)} left" if left is not None else "")
            card.set_progress(ev["i"], ev["n"], text)

    def _ev_note(self, ev: dict):
        """What the generator is doing now (e.g. a SeedVR2 piece)."""
        self._clip_note = str(ev.get("text") or "")
        if self._model_state == ModelState.LOADING:         # (preparing the RunPod pod: each step in the status bar)
            self.status.setText(self._clip_note + " …")

    def _ev_done(self, ev: dict):
        self._phase = Phase.DONE

    def _ev_exported(self, ev: dict):
        self._exported = ev.get("file")

    def _ev_joining(self, ev: dict):
        self.status.setText("Joining the refined clips …")

    def _ev_error(self, ev: dict):
        self._error = ev.get("msg", "error")
        if self._job is None:                      # after the job (e.g. a RunPod pod that could not be ended)
            self.status.setText(f"Warning: {self._error}")
            self.log.appendPlainText(f"Warning: {self._error}")

    def _tell_plugins_finished(self, result: int):
        try:
            self.plugins.emit("run_finished", result)
        except RuntimeError:                           # the window was closed before the timer fired
            pass

    def _job_finished(self, code: int = 0):
        """End of a job: the worker reported idle again (code 0), or the worker ended (its exit code)."""
        ok = self._phase == Phase.DONE and code == 0 and not self._error
        result = 0 if ok else (int(code) or 1)           # (told to the plugins once the window is up to date)
        QTimer.singleShot(0, lambda: self._tell_plugins_finished(result))
        refine = self._refine         # an export at a refined size
        for nid in [n for n, s in self.live.items() if s == "generating"]:
            if refine:                                   # an export never touches the clip itself, only its upscaled
                self.live.pop(nid, None)                 # copy: the clip stays as it is (it was marked "stopped" and
            else:                                        # blocked Play and Export path until the project reopened)
                self.live[nid] = "stopped" if self._stopping else "failed"
        if ok:
            self.live.clear()
        self.progress_of.clear()
        self._reload_made()
        self._job = None
        for c in self.cards:
            c.set_editable(True)
        self.progress.setRange(0, 1000)
        self.refresh()
        self._report_job_end(ok, refine, code)
        self._stopping = False
        if self.worker_alive and (self._cloud_worker or not self.keep_loaded.isChecked()):
            self._quit_worker()                    # (a RunPod worker always: its pod is billed while it waits)
        self._save_timer.start()                   # what was chosen during the run (selection …) is saved now
        if ok and self.film is not None and len(self.film.scenes) > 1 and not self.film.problem \
                and self._film_queue is None:
            QTimer.singleShot(0, self._update_continues)   # the next scene starts from this one's (new) end
        if self._film_queue is not None:
            if ok and not refine:
                QTimer.singleShot(0, self._film_next)
            else:
                self._film_queue = None            # stopped or failed: the film stops there
        self._update_buttons()
        self._update_model_label()

    def _report_job_end(self, ok: bool, refine: dict | None, code: int):
        """The status line and the progress bar after a job: done, exported, stopped or failed."""
        if ok and refine:
            self.progress.setValue(1000)
            out = Path(self._exported or refine["out"])
            n = len(self._made_now)                # the export lists every clip, but reuses kept copies
            how = {"seedvr2_7b": "SeedVR2 7B", "seedvr2_3b": "SeedVR2 3B"}.get(refine.get("method"), "refined")
            self.status.setText(f"Exported {out.name} ({refine['size']}, {how}"
                                + (f"; {plural(n, 'clip')} upscaled now" if n else
                                   "; every clip was upscaled already") + ").")
        elif ok:
            self.progress.setValue(1000)
            made = len(self._plan)
            msg = f"Done: {plural(made, 'clip')} made." if made else "Everything was already made."
            self.status.setText(msg)
            path = self.selected_path()
            if len(self._targets) == 1 and path and path[-1].id == self._targets[0] and self._film_queue is None:
                self.join_selected("play", prefix=msg)
        elif self._stopping:
            self.progress.setValue(0)
            self.status.setText("Stopped. Finished clips are kept.")
        else:
            self.progress.setValue(0)
            self.status.setText(f"Failed: {self._error or f'the generator exited with code {code}'}")
            if self.view.visible((self.tabs, self.log)):     # (Simple: the status line and the card say it)
                self.tabs.setCurrentWidget(self.log)
            if not refine:                                   # (an export's failure is not a clip's)
                self._note_failure(code)

    def _proc_error(self, err):
        if err == QProcess.ProcessError.FailedToStart:
            # Qt sends no 'finished' for a process that never started: clean up here
            self._error = f"could not start {self._program()[0]}"
            p = self.proc
            QTimer.singleShot(0, lambda: self.proc is p and self._finished(-1, None))

    def _finished(self, code: int, _status=None):
        self._kill_timer.stop()
        p, self.proc = self.proc, None
        if p is not None:
            self._drain_out(p, final=True)
            self._drain_err(p, final=True)
            p.deleteLater()
        self._buf, self._ebuf = b"", ""
        self._worker_idle = False
        self._model_state = ModelState.NONE
        if self._job is not None and self._replace_worker and not self._stopping:
            self._job.sent = False
            self._start_worker()
        elif self._job is not None:
            refine = self._job.refine                # an export stays an export when it waits for ComfyUI
            film_rest = list(self._film_queue) if self._film_queue is not None else None
            self._job_finished(code if code else 1)
            if code == COMFY_DOWN:
                self.comfy_up = False
                if self._autostart(self._targets, refine):
                    if film_rest is not None:               # Generate film goes on once ComfyUI runs
                        self._film_queue = film_rest
                else:
                    self.status.setText(f"ComfyUI is not running at {self.comfy_url.text().strip()}. "
                                        "Start it (Settings → ComfyUI and preferences → Start ComfyUI) and generate "
                                        "again.")
                self._update_comfy_label()
        self._update_model_label()
        self._update_buttons()
