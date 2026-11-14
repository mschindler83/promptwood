"""Promptwood window: a start image and a tree of clips. Every clip continues from the last frame of the clip
before it; any clip can be continued in several ways (branches). A path from the start to a clip is one video.
Real videos can be put into the chain too (a range of a file): a generated clip before one ends at its first frame,
one after it starts from its last frame.

The project folder holds tree.json, start.png and the clips (see tree/). Clips are rendered by ComfyUI; a small
worker process (python -m video_chain --serve) talks to it and reports '@@ {json}' events; Stop ends that process
and cancels the clip in ComfyUI.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap  # noqa: F401
from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox  # noqa: F401

from .. import tree as T, tts as TTS
# (re-exported: the tests patch and import these names on main_window)
from .clip_dialogs import SplitDialog, VideoFrameDialog  # noqa: F401
from .video_editor import VideoRangeDialog  # noqa: F401
from .view_mode import ViewMode
from .clip_cards import PromptCard, VideoCard
from .generator_state import ComfyModel, Job, ModelState, ModelSwitch, Phase, WorkerBinding
from .mw_audio import AudioMixin
from .mw_build import BuildMixin
from .mw_cast import CastMixin
from .mw_comfy import ComfyMixin
from .mw_ends import EndsMixin
from .mw_export import ExportMixin
from .mw_film import FilmMixin
from .mw_generator import GeneratorMixin
from .mw_inspector import InspectorMixin
from .mw_llama import LlamaMixin
from .mw_loras import LorasMixin
from .mw_nodes import NodeMixin
from .mw_options import OptionsMixin
from .mw_path import PathMixin
from .mw_project import ProjectMixin
from .mw_settings import SettingsMixin
from .mw_state import StateMixin
from .mw_tts import TtsMixin
from .mw_videos import VideosMixin
from .mw_view import ViewMixin
from .preview import Preview  # noqa: F401
from .project_dialogs import ExportDialog, NewProjectDialog, VariantsDialog  # noqa: F401
from .ui_logic import (  # noqa: F401
    CARD_WIDTH, CONTINUE_CLASSIC_TIP, CONTINUE_FIRST_TIP, CONTINUE_TIP, KILL_GRACE_MS, LAYOUT_VERSION, MIX_TIP,
    REMIX_DELAY_MS, STATE_TIP, find_gguf_models, fmt_left, label_loras, row_start, row_start_of, time_left)
from .qsvalue import qs_value
from .widgets import timer


class MainWindow(BuildMixin, GeneratorMixin, ComfyMixin, ExportMixin, VideosMixin, EndsMixin, NodeMixin, LorasMixin,
                 PathMixin, InspectorMixin, StateMixin, AudioMixin, CastMixin, TtsMixin, LlamaMixin, OptionsMixin,
                 ProjectMixin, FilmMixin, SettingsMixin, ViewMixin, QMainWindow):
    """The window: wiring only. What it does lives in the mixins (mw_*.py), one topic each."""

    viewChanged = Signal(bool)                  # the view was switched: True = Simple (after the controls follow)

    def __init__(self, settings: QSettings, command: list[str] | None = None):
        super().__init__()
        self.setWindowTitle("Promptwood")
        self.qs = settings
        # Simple or Advanced, decided once: Simple on a first run (no settings yet), Advanced for settings from
        # before the views existed (that user knows every control); stored at once, so it never flips later
        mode = qs_value(self.qs, "viewMode", "", str)
        if mode not in ("simple", "advanced"):
            mode = "advanced" if self.qs.allKeys() else "simple"
            self.qs.setValue("viewMode", mode)
        self.view = ViewMode(mode == "simple", self)
        self.view.changed.connect(self._view_switched)
        self._build_menu()
        self.command = command                  # tests replace the worker with a fake one
        self.project: Path | None = None
        self.tree: T.Tree | None = None
        self.status_of: dict[str, str] = {}
        self.why_of: dict[str, str] = {}             # why a changed clip is changed, in words
        self.live: dict[str, str] = {}          # states during a run (generating / failed / stopped)
        self.progress_of: dict[str, float] = {}
        self.cards: list[PromptCard | VideoCard] = []
        self._pix: dict[tuple, QPixmap] = {}
        self.proc: QProcess | None = None
        self._buf, self._ebuf = b"", ""
        self._stopping = False
        self._phase = Phase.IDLE
        self._error = ""
        self._plan: list[str] = []
        self._made_now: set[str] = set()
        self._targets: list[str] = []
        self._loading = False
        self._tree_stamp: tuple | None = None   # tree.json as this window last read or wrote it
        self._copying: str | None = None        # Save as in progress (the source project)
        self._worker_leaving = False            # the worker said bye (idle timeout) and is quitting
        self._job: Job | None = None            # the generation in progress
        self._worker_idle = False
        self._worker_bound: WorkerBinding | None = None
        self._replace_worker = False
        self._idle_since, self._idle_timeout = 0.0, None
        self._model_state = ModelState.NONE
        self._joins = 0                         # path joins still running
        self._converts: dict[str, str] = {}     # video node -> token of its conversion in progress
        self._kill_timer = timer(self, self._kill, KILL_GRACE_MS, single=True)
        self._film_queue: list[str] | None = None    # Generate film: the scenes still to make
        self._trigger_scan = None
        self._tts_local: TTS.LocalTTS | None = None
        self._tts_fish: TTS.FishTTS | None = None
        self._tts_busy = False
        self._uses_cache: tuple | None = None        # (tree, IE.uses_index(tree)) until something changes
        self._first_shown = False
        self._notices: list[QMessageBox] = []        # non-blocking notices, kept until they are closed
        self._eleven_voices: list | None = None
        self.h3_loras: list[tuple[str, str]] = []
        self.qwen_loras: list[tuple[str, str]] = []
        self.image_loras: dict[str, list[tuple[str, str]]] = {}
        self.turbo_loras: list[tuple[str, str]] = []
        self._scanned_at = 0.0                  # last LoRA scan (monotonic)
        self.comfy_up: bool | None = None
        self._comfy_polling = False
        self._comfy_starting = 0.0
        self._remade_note = ""                       # what the last change said about CHANGED clips
        self._failures: dict[str, dict] = {}    # clip -> what went wrong when it was made (on its card until it is
        #                                         made again or edited)
        self._waiting_comfy: list[str] | None = None   # targets to generate once ComfyUI answers
        # the model this app last used in ComfyUI ("h3" clips, "qwen" image edits, None: none / unloaded). ComfyUI
        # is asked to unload only when it changes: switching models frees the other one first instead of letting
        # both fight over the memory; the same model again is used as it is (no reload)
        self._comfy_model: ComfyModel | None = None
        self._quit_frees = True                 # the worker's next quit unloads the model
        self._qwen_timer = timer(self, self._qwen_idle, single=True)
        self._join_lock = threading.Lock()      # joins run one after another (play after generating + export)
        self._save_timer = timer(self, self._save_tree, 400, single=True)
        self._refresh_timer = timer(self, self.refresh, 250, single=True)
        # a made clip's sound mode / level changed: its sound is mixed again (no new generation), once the level
        # settles; remix_due: the clips waiting for that (refresh)
        self._remix_timer = timer(self, self._start_remix, REMIX_DELAY_MS, single=True)
        self._remixing = False
        self.remix_due: set[str] = set()
        self._remix_stuck: set[str] = set()     # the due clips whose remix failed (tried again after an edit)
        self._gpu_proc: QProcess | None = None
        self._gpu_timer = timer(self, self._poll_gpu, 4000)
        self._comfy_timer = timer(self, self._poll_comfy, 4000)
        self._build()
        from .plugin_host import PluginHost      # separately installed features (plugins.py); none: nothing added
        self.plugins = PluginHost(self)          # (before the settings: they open the last project, and the
        self.plugins.start()                     # plugins hear that)
        self._load_settings()
        self._end_leftover_llama()
        self._gpu_timer.start()
        self._comfy_timer.start()
        QTimer.singleShot(0, self._poll_comfy)
        QTimer.singleShot(0, self.scan_loras)
        self.comfy_dir.editingFinished.connect(self.scan_loras)
        QTimer.singleShot(0, self._poll_gpu)


    # ---- class defaults: what is only set once something happens, and is read before that
    _write_error: str | None = None             # why the last _write_tree did not save (None: it did)
    # While the cards are updated every file is looked at once (a path of clips asked about the same pictures and
    # videos several times each; on a network share every look is a round trip).
    _stat_memo: dict | None = None
    _cmp_memo: dict | None = None
    # what a frozen clip / video cannot do: it would change the clip as made (or what it ends at)
    FROZEN_LOCKED = {"end_image", "end_video", "end_ai", "end_clear", "revert", "refine", "speak", "handover",
                     "handover_clear", "edit", "split", "swap", "swap_undo", "video", "keyframe", "start_own",
                     "start_own_clear"}
    _trigger_scan_online = True                 # the running trigger word scan may ask CivitAI
    _left_w = 0                                 # the width of the left panel (kept while it is hidden)
    _shared_sent = None                         # the shared settings last written into the film's other scenes
    _copy_scene, _copy_scene_id = ".", None     # Save as: the scene to open in the copy
    _tracks_shown = None                        # what the audio tracks list shows (a signature)
    _canvas = None                              # the clips' size as _update_size_info last worked it out
    _llama = None                               # the llama.cpp server of this window
    _runpod_loading = False                     # the settings are being loaded (the RunPod fields change unsaved)
    _dialogs_modeless = False                   # tests: dialogs are not run, they are handed back
    # the generator run (mw_generator): checking the project first, then the clips
    _checking = False                           # Generate is checking the project (in the background)
    _check_gen = 0
    _exported = None
    _clip_started = _run_started = 0.0
    _clip_note = ""
    _edit_switch: ModelSwitch | None = None
    _lora_misses: dict = {}
    _lora_hits: dict = {}
    _lora_scan_gen = 0
    _path_end = None
