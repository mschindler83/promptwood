"""The local llama.cpp server and the API keys."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from .workers import post, run_in_background
from .. import comfy as CMF, llm_server as LLM, runpod as RP, triggers, tts as TTS
from . import dialogs as D
from .ui_logic import (
    CLOSE_GRACE, DEFAULT_COMFY_URL, LLM_DEFAULT_SERVER, UNLOAD_TIMEOUT, VRAM_POLL_SECONDS, app_cache_dir)
from .mw_host import MainWindowHost


@dataclass
class KeyField:
    """One API key of the Preferences: its line edit, how to read and set the key (the system keyring), the text of
    the empty field and what happens around saving it."""
    title: str                       # the name in a failure dialog
    service: str
    edit: str                        # the window's attribute of its line edit (built after this is declared)
    get: Callable[[], str | None]
    set: Callable[[str], None]
    none_hint: str                   # the placeholder while no key is kept
    kept: str = ""                   # the status line after a key was kept (default: from the service)
    before: Callable[[str], None] | None = None     # after the key was set, before the field is shown again
    after: Callable[[str], None] | None = None      # after the status line


class LlamaMixin(MainWindowHost):
    """The local llama.cpp server and the API keys."""

    def llm_model_path(self) -> str:
        i = self.llm_model.currentIndex()
        typed = self.llm_model.currentText().strip()
        data = self.llm_model.itemData(i) if i >= 0 else None
        return data if data and self.llm_model.itemText(i) == typed else os.path.expanduser(typed)

    def llama_server(self):
        """The llama.cpp server of the refiner (one per app; renewed when its settings change), None when off."""
        if not self.llm_on.isChecked() or not self.llm_model_path():
            return None
        key = (self.llm_server.text().strip() or LLM_DEFAULT_SERVER, self.llm_model_path(), self.llm_port.value(),
               self.llm_cpu_moe.isChecked(), LLM.find_mmproj(self.llm_model_path()))
        cur = self._llama
        if cur is None or (cur.server, cur.model, cur.port, cur.cpu_moe, cur.mmproj) != key:
            if cur is not None:
                cur.stop(wait=False)
            self._llama = LLM.LlamaServer(*key, log_dir=app_cache_dir())
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL).rstrip("/")
        llama = self._llama
        assert llama is not None
        # (llm_server.py types the hook as None)
        llama.before_launch = lambda llama=llama: self._room_for_llama(llama, url)  # type: ignore[assignment]
        return llama

    def _end_leftover_llama(self):
        """At start-up: a llama-server an earlier run left behind (it crashed; found by its pid file) holds RAM and
        VRAM nobody knows about; it is ended, in the background. Nothing else is touched."""
        log_dir = app_cache_dir()
        server, port = self.llm_server.text().strip() or LLM_DEFAULT_SERVER, self.llm_port.value()

        run_in_background(lambda: LLM.LlamaServer(server, "", port, log_dir=log_dir).stop(wait=True, grace=CLOSE_GRACE),
                          on_done=lambda _r: None, owner=self, name="llama-leftover",
                          on_error=lambda e: logging.getLogger(__name__).warning("ending a leftover llama-server",
                                                                                 exc_info=e))

    def _room_for_llama(self, llama, url: str, settle: float = 15.0) -> None:
        """(start()'s thread, just before the llama.cpp server starts) Too little free VRAM for it: ComfyUI is asked
        to unload its models (the clip / image model loads again when it is needed). While clips are generated it
        refuses instead (their run keeps its VRAM)."""
        self._make_room(LLM.vram_needed_mb(llama.model, llama.cpu_moe, llama.mmproj), url, "the llama.cpp model",
                        LLM.LlmServerError, "refine after the run, or choose a ComfyUI model", settle)

    def _make_room(self, need: int, url: str, what: str, error: type[Exception] = RuntimeError,
                   later: str = "try again after the run", settle: float = 15.0) -> None:
        """(a worker thread, before a model of this app starts) Too little free VRAM for need MB: ComfyUI is asked to
        unload its models; while clips are generated: error instead (their run keeps its VRAM)."""
        free = LLM.free_vram_mb()
        if free is None or free >= need:
            return
        def gb(mb: float) -> str:
            return f"{mb / 1024:.1f} GB"
        if self.running:
            raise error(f"only {gb(free)} of VRAM is free while clips are generated and {what} needs about "
                        f"{gb(need)}; {later}")
        if not CMF.free_models(url, timeout=UNLOAD_TIMEOUT):
            note = (f"ComfyUI did not unload its models (is it running?): only {gb(free)} of VRAM "
                    f"is free and {what} needs about {gb(need)}.")
            post(self, lambda: self.log.appendPlainText(note))
            return                                  # (ComfyUI is not up: the VRAM is used by something else)
        end = time.time() + settle
        now: float | None = free
        while time.time() < end:
            time.sleep(VRAM_POLL_SECONDS)
            now = LLM.free_vram_mb()
            if now is None or now >= need:
                break
        note = (f"ComfyUI's models were unloaded for {what} (it needs about {gb(need)} of VRAM, "
                f"{gb(free)} was free" + (f", now {gb(now)}" if now is not None else "") +
                "); the clip model loads again before the next clips.")
        post(self, lambda: self._room_made(note))

    def _tts_log(self, message: str):
        """A step of text to speech: in the log, and in the status line while it works."""
        self.log.appendPlainText("TTS: " + message)
        self.status.setText(message + "…")

    def _room_made(self, msg: str):
        self.comfy.forget()                    # (nothing of this app is loaded in ComfyUI any more)
        self._qwen_timer.stop()
        if self.worker_alive and not self.running:
            self._quit_worker(free=False)           # its model is gone; a new worker starts for the next clips
        self._update_model_label()
        self.log.appendPlainText(msg)
        self.status.setText(msg)

    def stop_llama(self, why: str = "", wait: bool = False, grace: float | None = None) -> bool:
        """Stop the refiner's llama.cpp server if the app started it (its RAM / VRAM back for ComfyUI). wait: until
        it is gone (when the window closes: a background wait would die with the app, leaving the server). grace:
        seconds it gets to end before it is killed (short when the window closes)."""
        cur = self._llama
        stopped = bool(cur is not None and (cur.stop(wait=wait) if grace is None else cur.stop(wait=wait, grace=grace)))
        if stopped and why:
            self.log.appendPlainText(f"Stopped the llama.cpp server ({why}).")
        return stopped

    def _key_fields(self) -> dict[str, KeyField]:
        """The API keys of the services (looked up when asked: the key functions are the modules' own)."""
        return {
            "civitai": KeyField("CivitAI API key", "CivitAI", "civitai_key", lambda: triggers.api_key(),
                                lambda k: triggers.set_api_key(k),
                                "none: paste your key to get everything CivitAI holds back",
                                kept="CivitAI API key kept in the system keyring; LoRAs are looked up once more "
                                     "with it.",
                                after=lambda key: self._scan_triggers() if key and self.triggers_online.isChecked()
                                else None),
            "eleven": KeyField("ElevenLabs API key", "ElevenLabs", "eleven_key", lambda: TTS.eleven_key(),
                               lambda k: TTS.set_eleven_key(k),
                               "none: paste your key for text to speech with ElevenLabs",
                               before=lambda key: setattr(self, "_eleven_voices", None)),
            "runpod": KeyField("RunPod API key", "RunPod", "runpod_key", lambda: RP.api_key(),
                               lambda k: RP.set_api_key(k), "none: paste your key to upscale on RunPod"),
        }

    def _show_key(self, name: str):
        """The key field of a service: empty, and its placeholder says whether a key is kept."""
        field = self._key_fields()[name]
        has = bool(field.get())
        edit = getattr(self, field.edit)
        edit.clear()
        edit.setPlaceholderText("a key is kept (type a new one, or save empty to remove it)" if has
                                      else field.none_hint)

    def _save_key(self, name: str):
        field = self._key_fields()[name]
        key = getattr(self, field.edit).text().strip()
        try:
            field.set(key)
        except RuntimeError as e:
            D.failed(self, field.title, "keep the key", e)
            return
        if field.before:
            field.before(key)
        self._show_key(name)
        self.status.setText(field.kept or f"{field.service} API key kept in the system keyring." if key
                            else f"{field.service} API key removed.")
        if field.after:
            field.after(key)

    def _show_civitai_key(self):
        self._show_key("civitai")

    def _save_civitai_key(self):
        self._save_key("civitai")

    def _show_eleven_key(self):
        self._show_key("eleven")

    def _save_eleven_key(self):
        self._save_key("eleven")

    def _show_runpod_key(self):
        self._show_key("runpod")

    def _save_runpod_key(self):
        self._save_key("runpod")

    def _runpod_gpu_id(self) -> str:
        i = self.runpod_gpu.currentIndex()
        text = self.runpod_gpu.currentText().strip()
        if i >= 0 and self.runpod_gpu.itemText(i) == text:
            return self.runpod_gpu.itemData(i) or text
        return text or RP.DEFAULT_GPU

    def _runpod_changed(self):
        if self._runpod_loading:
            return
        self.qs.setValue("runpodGpu", self._runpod_gpu_id())
        self.qs.setValue("runpodFallback", self.runpod_fallback.isChecked())
        self.qs.setValue("runpodVolume", self.runpod_volume.text().strip())
        self.qs.setValue("runpodWithoutVolume", self.runpod_without_volume.isChecked())
