"""What the window's mixins (mw_*.py) use of the window they are mixed into (MainWindow, main_window.py).

At run time MainWindowHost is an empty class. For the type checker it is the window: it declares the attributes the
mixins share (the widgets built in mw_build.py, the state main_window.py sets up) and falls back to Any for the
methods of the other mixins (those are checked in the mixin that defines them). A new attribute that two mixins
share is declared here; mypy says so ("... has no attribute")."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading
    from typing import Any

    from PySide6.QtCore import QProcess, QSettings, QTimer
    from PySide6.QtGui import QAction, QPixmap
    from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QToolButton, QWidget
    from pathlib import Path
    from threading import Lock
    from ..film import Film
    from ..llm_server import LlamaServer
    from ..tree.model import Tree
    from ..tts import FishTTS, LocalTTS
    from .cast_strip import CastStrip
    from .clip_cards import PromptCard, VideoCard
    from .generator_state import ComfyModel, Job, ModelState, ModelSwitch, Phase, WorkerBinding
    from .graph import GraphView
    from .guide import EngineBanner, GuidePanel, NextHint
    from .mw_support import _Bridge, _CurrentStack
    from .plugin_host import PluginHost
    from .preview import Preview, StartTile
    from .storyboard import Storyboard
    from .timeline import Timeline
    from .view_mode import AdvancedNote, ViewMode

    class MainWindowHost(QMainWindow):
        _check_gen: int
        _comfy_polling: bool
        _eleven_voices: list | None
        _first_shown: bool
        _gpu_proc: QProcess | None
        _lora_hits: dict
        _lora_misses: dict
        _lora_scan_gen: int
        _path_end: str | None
        FROZEN_LOCKED: set[str]
        _worker_bound: WorkerBinding | None
        _edit_switch: ModelSwitch | None
        _film_queue: list[str] | None
        _job: Job | None
        _plan: list[str]
        _played_path: tuple | None
        _shared_sent: str | None
        _tracks_shown: tuple | None
        _tree_stamp: tuple | None
        _tts_fish: FishTTS | None
        _tts_local: LocalTTS | None
        _uses_cache: tuple | None
        _worker_leaving: bool
        proc: QProcess | None
        _cfg_row: QWidget
        _checking: bool
        _civitai_row: QWidget
        _clips_form: QFormLayout
        _cmp_memo: dict | None
        _comfy_dir_row: QWidget
        _comfy_form: QFormLayout
        _comfy_model: ComfyModel | None
        _comfy_starting: float
        _comfy_timer: QTimer
        _converts: dict[str, str]
        _copying: str | None
        _dialogs_modeless: bool
        _eleven_row: QWidget
        _error: str
        _failures: dict[str, dict]
        _gpu_timer: QTimer
        _idle_since: float
        _idle_timeout: float | None
        _join_lock: Lock
        _joins: int
        _keep_row: QWidget
        _kill_timer: QTimer
        _left_w: int
        _llama: LlamaServer | None
        _loading: bool
        _model_state: ModelState
        _model_timer: QTimer
        _notices: list[QMessageBox]
        _pix: dict[tuple[Any, ...], QPixmap]
        _qwen_timer: QTimer
        _refresh_timer: QTimer
        _remade_note: str
        _remix_stuck: set[str]
        _remix_timer: QTimer
        _remixing: bool
        _runpod_loading: bool
        _runpod_section: tuple[QWidget, QWidget]
        _save_timer: QTimer
        _scanned_at: float
        _stat_memo: dict | None
        _targets: list[str]
        _trigger_scan: threading.Event | None
        _tts_busy: bool
        _tts_form: QFormLayout
        _turbo_row: QWidget
        _voice_row: QWidget
        _waiting_comfy: list[str] | None
        _worker_idle: bool
        _write_error: str | None
        act_close: QAction
        act_left: QAction
        act_new: QAction
        act_open: QAction
        act_save: QAction
        act_save_as: QAction
        adv_note: AdvancedNote
        aspect: QComboBox
        audio: QCheckBox
        b_add_video: QToolButton
        b_advanced: QPushButton
        b_ai_create: QPushButton
        b_ai_edit: QPushButton
        b_branch: QToolButton
        b_char_del: QPushButton
        b_char_new: QPushButton
        b_char_rename: QPushButton
        b_continue: QToolButton
        b_delete: QToolButton
        b_export: QPushButton
        b_export_clip: QPushButton
        b_film_cast: QPushButton
        b_film_export: QPushButton
        b_film_gen: QPushButton
        b_film_play: QPushButton
        b_gen: QPushButton
        b_gen_all: QPushButton
        b_new: QPushButton
        b_next_clip: QPushButton
        b_open: QPushButton
        b_play_clip: QPushButton
        b_play_film: QPushButton
        b_play_path: QPushButton
        b_prev_clip: QPushButton
        b_project: QToolButton
        b_recent: QPushButton
        b_ref_add: QPushButton
        b_ref_clip: QPushButton
        b_ref_create: QPushButton
        b_ref_crop: QPushButton
        b_ref_del: QPushButton
        b_ref_start: QPushButton
        b_simple: QPushButton
        b_sound_add: QPushButton
        b_sound_clip: QPushButton
        b_sound_del: QPushButton
        b_sound_play: QToolButton
        b_start_clear: QToolButton
        b_start_comfy: QPushButton
        b_stop: QPushButton
        b_track_add: QPushButton
        b_track_clip: QPushButton
        b_track_cut: QPushButton
        b_track_del: QPushButton
        b_track_play: QToolButton
        b_unload: QPushButton
        b_variant: QToolButton
        b_voice_clip: QPushButton
        b_voice_clone: QPushButton
        b_voice_del: QToolButton
        b_voice_file: QPushButton
        b_voice_play: QToolButton
        b_voice_tts: QPushButton
        bridge: _Bridge
        card_stack: _CurrentStack
        cards: list[PromptCard | VideoCard]
        cast_list: QListWidget
        cast_strip: CastStrip
        center: QSplitter
        cfg_note: QLabel
        char_desc: QLineEdit
        char_kind: QComboBox
        char_lead: QCheckBox
        char_voice: QLabel
        civitai_site: QComboBox
        clip_cfg: QDoubleSpinBox
        comfy_autostart: QCheckBox
        comfy_cleanup: QCheckBox
        comfy_dir: QLineEdit
        comfy_label: QLabel
        comfy_launcher: QLineEdit
        comfy_up: bool | None
        comfy_url: QLineEdit
        command: list[str] | None
        dialogue_lang: QComboBox
        eleven_model: QComboBox
        engine_banner: EngineBanner
        film: Film | None
        film_info: QLabel
        focus: str | None
        focus_info: QLabel
        frames: QComboBox
        gpu: QLabel
        graph: GraphView
        guide_panel: GuidePanel
        h3_loras: list[tuple[str, str]]
        ident_info: QLabel
        idle_minutes: QSpinBox
        image_loras: dict[str, list[tuple[str, str]]]
        inspector: QTabWidget
        keep_loaded: QCheckBox
        keep_rep: QCheckBox
        left_tabs: QTabWidget
        live: dict[str, str]
        llm_cpu_moe: QCheckBox
        llm_model: QComboBox
        llm_on: QCheckBox
        llm_port: QSpinBox
        llm_server: QLineEdit
        log: QPlainTextEdit
        lora_triggers: QCheckBox
        model_label: QLabel
        next_hint: NextHint
        path_hint: QLabel
        path_info: QLabel
        plugins: PluginHost
        prefs: QDialog
        preview: Preview
        progress: QProgressBar
        progress_of: dict[str, float]
        proj_scroll: QScrollArea
        project: Path | None
        project_edit: QLineEdit
        prompt_format: QComboBox
        qs: QSettings
        qwen_loras: list[tuple[str, str]]
        recent_file_menu: QMenu
        recent_menu: QMenu
        ref_add_label: QLabel
        ref_hint: QCheckBox
        ref_size: QComboBox
        refs: QListWidget
        remix_due: set[str]
        resolution: QComboBox
        runpod_fallback: QCheckBox
        runpod_gpu: QComboBox
        runpod_volume: QLineEdit
        runpod_without_volume: QCheckBox
        size_info: QLabel
        sound_info: QLabel
        sounds: QListWidget
        speed: QComboBox
        split: QSplitter
        start_tile: StartTile
        status: QLabel
        status_of: dict[str, str]
        steps: QSpinBox
        storyboard: Storyboard
        style_prefix: QLineEdit
        style_suffix: QLineEdit
        tabs: QTabWidget
        timeline: Timeline
        track_info: QLabel
        track_list: QListWidget
        tree: Tree | None
        tree_info: QLabel
        triggers_online: QCheckBox
        tts_provider: QComboBox
        turbo_lora: QComboBox
        turbo_steps: QSpinBox
        turbo_strength: QDoubleSpinBox
        veda: QCheckBox
        video_gpu: QCheckBox
        view: ViewMode
        why_of: dict[str, str]

        def __getattr__(self, name: str) -> Any: ...
else:
    class MainWindowHost:
        pass
