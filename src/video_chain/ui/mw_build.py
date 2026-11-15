"""The window's widgets."""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMenu, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import chain, runpod as RP, tree as T, triggers, tts as TTS
from ..comfy import REF_HINT
from . import shortcuts as S, ui_logic as UL
from .cast_strip import CastStrip
from .clip_inputs import KIND_NAMES
from .graph import GraphView
from .guide import EngineBanner, GuidePanel, NextHint
from .storyboard import Storyboard
from .timeline import Timeline
from .view_mode import AdvancedNote, row_widget
from .widgets import guard_wheel, icon_button, label, label_for, remember_geometry, section_bar, shortcut, timer
from .mw_support import FileDrop, _CurrentStack
from .preview import Preview, StartTile
from .ui_logic import (
    DEFAULT_COMFY_LAUNCHER, DEFAULT_COMFY_URL, FRAME_CHOICES, LLM_DEFAULT_PORT, LLM_DEFAULT_SERVER, STEPS_TIP,
    frames_text)
from .mw_host import MainWindowHost


def _panel() -> tuple[QWidget, QVBoxLayout]:
    """A panel without margins and its vertical layout."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    return w, v


def _buddy(text: str, field: QWidget, name: str | None = None) -> QLabel:
    """The label of a form row that holds a layout: a buddy of the field (a screen reader reads both)."""
    lab = QLabel(text)
    lab.setBuddy(field)
    field.setAccessibleName(name or text)
    return lab


def _scrolled(w: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(w)
    return sa


class BuildMixin(MainWindowHost):
    """The window's widgets."""

    # ------------------------------------------------------------------ layout
    def _build(self):
        shared = SimpleNamespace()          # what the sections of the build hand to each other
        self.resize(1560, 940)
        shared.split = QSplitter(Qt.Orientation.Horizontal, self)
        shared.split.setChildrenCollapsible(False)
        self.setCentralWidget(shared.split)
        self.split = shared.split

        # panels (an editor's layout): left tabs (clips graph, cast, sounds), the viewer in the middle with the
        # film / path / log below, the project's settings on the right; ComfyUI's settings in Preferences
        shared.left = QWidget()                                    # (parent of the section bars below)
        shared.proj_w, shared.proj_v = _panel()
        shared.cast_w, shared.cast_v = _panel()
        shared.sound_w, shared.sound_v = _panel()
        self.prefs = QDialog(self)
        self.prefs.setWindowTitle("Preferences")
        remember_geometry(self.prefs, self.qs, "prefs")
        shared.prefs_v = QVBoxLayout(self.prefs)
        self._build_project_panel(shared)
        self._build_start_image(shared)
        self._build_cast_panel(shared)
        self._build_tracks_panel(shared)
        self._build_sound_panel(shared)
        self._build_clip_settings(shared)
        self._build_engine_prefs(shared)
        self._build_runpod_prefs(shared)
        self._build_inspector_tabs(shared)
        self._build_film_strip(shared)
        self._build_graph_panel(shared)
        self._build_path_panel(shared)
        self._build_clip_tab(shared)
        self._build_viewer(shared)
        self._build_toolbar(shared)
        self._build_shortcuts(shared)
        self._name_fields()
        self._register_advanced()
        self._show_view()
        self._update_buttons()
        self._update_guide()

    def _name_fields(self) -> None:
        """Accessible names for the fields that have no label of their own beside them (UX-10)."""
        for field, name in ((self.char_desc, "Description of the selected entry"),
                            (self.char_kind, "What the selected entry is"), (self.eleven_model, "ElevenLabs model"),
                            (self.llm_model, "Language model file"), (self.llm_port, "Port of the language model server"),
                            (self.clip_cfg, "CFG (how strictly the model follows the prompt)"),
                            (self.turbo_steps, "Steps of the fast speed"),
                            (self.turbo_strength, "Strength of the fast speed add-on"),
                            (self.idle_minutes, "Minutes the model stays loaded"), (self.log, "Log")):
            field.setAccessibleName(name)

    def _build_project_panel(self, shared):
        """Build the Project tab: the folder."""
        bar, _ = section_bar("Project folder", shared.left)
        shared.proj_v.addWidget(bar)
        pw = QWidget()
        pf = QFormLayout(pw)
        pf.setContentsMargins(10, 6, 10, 8)
        prow = QHBoxLayout()
        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText("a folder for this chain (new or existing)")
        self.project_edit.setReadOnly(True)
        self.b_project = icon_button("fopen", "Open or create a project folder")
        self.b_project.clicked.connect(self._choose_project)
        self.b_open_dir = icon_button("folder", "Show the project folder")
        self.b_open_dir.clicked.connect(self._open_dir)
        prow.addWidget(self.project_edit, 1)
        prow.addWidget(self.b_project)
        prow.addWidget(self.b_open_dir)
        pf.addRow(_buddy("Folder", self.project_edit, "Project folder"), prow)
        brow = QHBoxLayout()
        self.b_new = QPushButton("New…")
        self.b_new.setToolTip(S.tip("Create a new project", "new_project"))
        self.b_new.clicked.connect(self.new_project)
        self.b_open = QPushButton("Open…")
        self.b_open.setToolTip(S.tip("Open a project folder", "open"))
        self.b_open.clicked.connect(self._choose_project)
        self.b_recent = QPushButton("Recent")
        self.b_recent.setToolTip("Open one of the last projects")
        self.recent_menu = QMenu(self)
        self.b_recent.setMenu(self.recent_menu)
        for b in (self.b_new, self.b_open, self.b_recent):
            brow.addWidget(b)
        brow.addStretch(1)
        pf.addRow("", brow)
        shared.proj_v.addWidget(pw)

    def _build_start_image(self, shared):
        """Build the start image tile."""
        bar, _ = section_bar("Start image", shared.left)
        shared.proj_v.addWidget(bar)
        sw = QWidget()
        sv = QVBoxLayout(sw)
        sv.setContentsMargins(10, 8, 10, 8)
        self.start_tile = StartTile(320, 180)
        sv.addWidget(self.start_tile, 0, Qt.AlignmentFlag.AlignHCenter)
        self.size_info = label("", level="tertiary")
        self.size_info.setWordWrap(True)
        sv.addWidget(self.size_info)
        self.b_ai_edit = QPushButton("Edit with AI…")
        self.b_ai_edit.setToolTip("Change an image by a prompt (Qwen Image Edit in ComfyUI) and use the result as "
                                  "the start image or as a clip's end image")
        self.b_ai_edit.clicked.connect(lambda: self.edit_image(start=True, mode="edit"))
        self.b_ai_create = QPushButton("Create a new image…")
        self.b_ai_create.setToolTip("Make a new image from a prompt or from reference pictures (Krea-2, Qwen-Image, "
                                    "FLUX, Illustrious) and use it as the start image, an end image or a face "
                                    "reference")
        self.b_ai_create.clicked.connect(lambda: self.edit_image(start=True, mode="create"))
        airow = QHBoxLayout()
        airow.addWidget(self.b_ai_edit)
        airow.addWidget(self.b_ai_create)
        airow.addStretch(1)
        self.b_start_clear = icon_button("trash", "Remove the start image: the first clip then starts from its "
                                                  "prompt (the size follows the Aspect setting or a real video)")
        self.b_start_clear.clicked.connect(lambda: self.clear_image(confirm=True))
        airow.addWidget(self.b_start_clear)
        sv.addLayout(airow)
        shared.proj_v.addWidget(sw)

    def _build_cast_panel(self, shared):
        """Build the Cast & references tab."""
        bar, _ = section_bar("Cast & references", shared.left)
        shared.cast_v.addWidget(bar)
        iw = QWidget()
        iv = QVBoxLayout(iw)
        iv.setContentsMargins(10, 6, 10, 8)
        iv.setSpacing(6)
        self.cast_list = QListWidget()
        self.cast_list.setIconSize(QSize(40, 40))
        self.cast_list.setFixedHeight(110)
        self.cast_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cast_list.setToolTip("The characters of this video and the references for places, objects and a "
                                  "style. In a prompt write @Name (typing @ offers the names): the clip is made "
                                  "with exactly the faces and pictures it mentions. A clip that mentions nobody "
                                  "uses the lead's face.")
        self.cast_list.currentRowChanged.connect(self._show_character)
        self.cast_list.itemDoubleClicked.connect(self._rename_character)
        iv.addWidget(self.cast_list)
        crow = QHBoxLayout()
        self.b_char_new = QPushButton("New")
        self.b_char_new.setToolTip("A new entry: a person (a character with face pictures, a voice, can be the "
                                   "lead) or a place, object or style (whole pictures); a name to mention as @Name")
        new_menu = QMenu(self.b_char_new)
        for kind in T.CAST_KINDS:
            act = new_menu.addAction(f"{KIND_NAMES[kind]}…")
            act.triggered.connect(lambda _=False, k=kind: self._new_character(k))
        self.b_char_new.setMenu(new_menu)
        self.char_kind = QComboBox()
        for kind in T.CAST_KINDS:
            self.char_kind.addItem(KIND_NAMES[kind], kind)
        self.char_kind.setToolTip("What the selected entry is. A person has face pictures (cropped to the face), can "
                                  "have a voice and be the lead; a place, object or style is sent with its whole "
                                  "pictures and is named as what it is in the prompt")
        self.char_kind.activated.connect(self._character_kind_changed)
        self.b_char_rename = QPushButton("Rename…")
        self.b_char_rename.setToolTip("Rename the character; every @mention in the prompts is renamed too")
        self.b_char_rename.clicked.connect(self._rename_character)
        self.b_char_del = QPushButton("Remove")
        self.b_char_del.setToolTip("Remove the selected entry and its pictures")
        self.b_char_del.clicked.connect(self._remove_character)
        crow.addWidget(self.b_char_new)
        crow.addWidget(self.b_char_rename)
        crow.addWidget(self.char_kind)
        crow.addStretch(1)
        crow.addWidget(self.b_char_del)
        iv.addLayout(crow)
        self.char_desc = QLineEdit()
        self.char_desc.setPlaceholderText("Description (optional), e.g. woman, 30s, grey knit sweater")
        self.char_desc.setToolTip("Goes into the prompt the model gets for clips that mention this character")
        self.char_desc.editingFinished.connect(self._character_desc_changed)
        iv.addWidget(self.char_desc)
        self.char_lead = QCheckBox("Lead: clips that mention nobody get this face")
        self.char_lead.toggled.connect(self._character_lead_changed)
        iv.addWidget(self.char_lead)
        self._voice_row = QWidget()
        vrow = QHBoxLayout(self._voice_row)
        vrow.setContentsMargins(0, 0, 0, 0)
        self.char_voice = label("Voice: none", level="tertiary")
        self.char_voice.setToolTip("A character with a voice speaks its lines with it: write the line in quotes in "
                                   "the sentence of its @Name, e.g. @Anna looks up and says \"Good morning.\" "
                                   "Use a voice only with the speaker's consent.")
        self.b_voice_file = QPushButton("Voice…")
        self.b_voice_file.setToolTip("Set the voice from an audio or video file: choose the part of it where only "
                                     "this speaker talks, with little music or echo (the clip model hears the first "
                                     f"{T.VOICE_MAX_SECS:g} s of the part; text to speech clones the voice from up "
                                     f"to {T.VOICE_REF_MAX_SECS:g} s of it)")
        self.b_voice_file.clicked.connect(self._voice_from_file)
        self.b_voice_clip = QPushButton("From clip")
        self.b_voice_clip.clicked.connect(self._voice_from_selected)
        self.b_voice_tts = QPushButton("TTS…")
        self.b_voice_tts.setToolTip("Make the voice with text to speech: designed from a description (Qwen3-TTS, "
                                    "local) or one of your ElevenLabs voices. Its lines can then be spoken with it "
                                    "(Speak the lines on a clip)")
        self.b_voice_tts.clicked.connect(self._voice_with_tts)
        self.b_voice_clone = QPushButton("Clone")
        self.b_voice_clone.setToolTip("Clone the voice at ElevenLabs: the voice sample is uploaded to your ElevenLabs "
                                      "account and becomes a voice there (an instant voice clone; needs a paid "
                                      "plan). The character's lines are then spoken with it when ElevenLabs is the "
                                      "text to speech provider. Only with the speaker's consent.")
        self.b_voice_clone.clicked.connect(self._clone_voice_at_eleven)
        self.b_voice_play = icon_button("play", "Listen to the voice")
        self.b_voice_play.clicked.connect(self._play_voice)
        self.b_voice_del = icon_button("x", "Remove the voice")
        self.b_voice_del.clicked.connect(self._remove_voice)
        vrow.addWidget(self.char_voice, 1)
        for b in (self.b_voice_file, self.b_voice_clip, self.b_voice_tts, self.b_voice_clone, self.b_voice_play,
                  self.b_voice_del):
            vrow.addWidget(b)
        iv.addWidget(self._voice_row)
        self.refs = QListWidget()
        self.refs.setViewMode(QListWidget.ViewMode.IconMode)
        self.refs.setIconSize(QSize(64, 64))
        self.refs.setFixedHeight(92)
        self.refs.setMovement(QListWidget.Movement.Static)
        self.refs.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.refs.setToolTip("Face pictures of the selected character (up to 3, e.g. front and side): best a "
                             "sharp close-up of the face, without the scene around it")
        iv.addWidget(self.refs)
        rrow = QHBoxLayout()
        self.b_ref_add = QPushButton("Image…")
        self.b_ref_add.setToolTip("Add an image file as a face picture of the selected character")
        self.b_ref_add.clicked.connect(self._add_reference)
        self.b_ref_create = QPushButton("Create…")
        self.b_ref_create.setToolTip("Create a face picture with AI (the Create image window); there “Add as face "
                                     "reference” adds a result to the selected character")
        self.b_ref_create.clicked.connect(lambda: self.create_reference())   # (clicked passes checked=False)
        self.b_ref_start = QPushButton("Start image")
        self.b_ref_start.clicked.connect(self._reference_from_start)
        self.b_ref_clip = QPushButton("Selected clip")
        self.b_ref_clip.clicked.connect(self._reference_from_selected)
        self.b_ref_del = QPushButton("Remove")
        self.b_ref_del.setToolTip("Remove the selected face picture")
        self.b_ref_del.clicked.connect(self._remove_reference)
        self.b_ref_crop = QPushButton("Crop…")
        self.b_ref_crop.setToolTip("Crop the selected face picture again (from the full picture it came from)")
        self.b_ref_crop.clicked.connect(self._recrop_reference)
        self.refs.itemDoubleClicked.connect(self._recrop_reference)
        self.refs.currentRowChanged.connect(self._update_cast_buttons)
        self.ref_add_label = label("Add face:", level="tertiary")
        rrow.addWidget(self.ref_add_label)
        for b in (self.b_ref_add, self.b_ref_create, self.b_ref_start, self.b_ref_clip):
            rrow.addWidget(b)
        rrow.addStretch(1)
        rrow.addWidget(self.b_ref_crop)
        rrow.addWidget(self.b_ref_del)
        iv.addLayout(rrow)
        irow = QFormLayout()
        irow.setContentsMargins(0, 0, 0, 0)
        self.ref_size = QComboBox()
        self.ref_size.setToolTip("Reference size: how big the cast's pictures are sent to the model. Match: the "
                                 "clip's size (fast); max 2048 px: a better likeness, slower")
        self.ref_size.addItem("match (fast)", "match")
        self.ref_size.addItem("max 2048 px (best likeness, slower)", "max")
        irow.addRow("Picture size", self.ref_size)
        self.dialogue_lang = QComboBox()
        self.dialogue_lang.setEditable(True)
        self.dialogue_lang.addItems(["English", "German", "French", "Spanish", "Italian", "Portuguese", "Japanese",
                                     "Chinese", "Korean"])
        self.dialogue_lang.setToolTip("The language of the spoken lines (\"...\" in a prompt); the model gets each "
                                      "line as <d>[Language] ...</d> when a character speaks with its voice")
        irow.addRow("Speech", self.dialogue_lang)
        self.tts_provider = QComboBox()
        for k, text in TTS.PROVIDERS.items():
            self.tts_provider.addItem(text, k)
        self.tts_provider.setToolTip("Who speaks the lines with text to speech (Speak the lines on a clip, TTS… for a "
                                     "voice): Qwen3-TTS on this computer, cloning each character's voice sample; "
                                     "Fish S2 Pro on this computer, cloning it too and taking the tone from [tags] "
                                     "in a line (\"[angry] Lass mich!\"; about 19 GB of VRAM, its licence allows "
                                     "non-commercial use only); or ElevenLabs with your API key (Settings), in each "
                                     "character's ElevenLabs voice (its v3 models hear [tags] too)")
        irow.addRow("Text to speech", self.tts_provider)
        mrow = QHBoxLayout()
        self.eleven_model = QComboBox()
        self.eleven_model.setToolTip("The ElevenLabs model the lines are spoken with")
        self.eleven_model.addItem(TTS.DEFAULT_OPTIONS["eleven_model"], TTS.DEFAULT_OPTIONS["eleven_model"])
        self.b_eleven_models = icon_button("rotate", "Ask ElevenLabs for its models")
        self.b_eleven_models.clicked.connect(self._load_eleven_models)
        mrow.addWidget(self.eleven_model, 1)
        mrow.addWidget(self.b_eleven_models)
        self._eleven_row = QWidget()
        self._eleven_row.setLayout(mrow)
        mrow.setContentsMargins(0, 0, 0, 0)
        irow.addRow("ElevenLabs model", self._eleven_row)
        self._tts_form = irow
        iv.addLayout(irow)
        self.ref_hint = QCheckBox("Tie each person to their face in the prompt")
        self.ref_hint.setChecked(True)
        self.ref_hint.setToolTip("With @mentions: \"Anna has exactly the face of the person in <Picture 1>.\" "
                                 "and that the pictures only show who the people are. Without: "
                                 f"\"{REF_HINT}\"")
        iv.addWidget(self.ref_hint)
        self.ident_info = label("", level="tertiary")
        self.ident_info.setWordWrap(True)
        iv.addWidget(self.ident_info)
        shared.cast_v.addWidget(iw)
        self.ref_size.currentIndexChanged.connect(self._identity_changed)
        self.ref_hint.toggled.connect(self._identity_changed)
        self.dialogue_lang.activated.connect(self._identity_changed)          # a finished choice, not each key
        self.tts_provider.currentIndexChanged.connect(self._tts_changed)
        self.eleven_model.currentIndexChanged.connect(self._tts_changed)
        edit_box = self.dialogue_lang.lineEdit()
        assert edit_box is not None
        edit_box.editingFinished.connect(self._identity_changed)

    def _build_tracks_panel(self, shared):
        """Build the audio tracks."""
        bar, _ = section_bar("Audio tracks", shared.left)
        shared.sound_v.addWidget(bar)
        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.setContentsMargins(10, 6, 10, 8)
        tv.setSpacing(6)
        self.track_list = QListWidget()
        self.track_list.setFixedHeight(96)
        self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.track_list.setToolTip("Audio tracks: music, speech or a background the clips play as their own sound "
                                   "(a clip's Sound: “Audio track: …”), on from clip to clip. "
                                   "Double-click to cut it again")
        self.track_list.itemDoubleClicked.connect(self._recut_track)
        self.track_list.currentRowChanged.connect(self._update_track_buttons)
        tv.addWidget(self.track_list)
        trow = QHBoxLayout()
        self.b_track_add = QPushButton("Import…")
        self.b_track_add.setToolTip("Import the sound of an MP3, WAV, OGG, FLAC … or a video file and cut it to the "
                                    "part you need")
        self.b_track_add.clicked.connect(lambda: self.import_track())
        self.b_track_clip = QPushButton("From selected clip…")
        self.b_track_clip.clicked.connect(self._track_from_selected)
        self.b_track_cut = QPushButton("Cut…")
        self.b_track_cut.setToolTip("Change the part of the file the track plays (or its name); the clips playing it "
                                    "count as changed")
        self.b_track_cut.clicked.connect(self._recut_track)
        self.b_track_play = icon_button("play", "Listen to the selected audio track")
        self.b_track_play.clicked.connect(self._play_track)
        self.b_track_del = QPushButton("Remove")
        self.b_track_del.setToolTip("Remove the selected audio track")
        self.b_track_del.clicked.connect(self._remove_track)
        for w in (self.b_track_add, self.b_track_clip):
            trow.addWidget(w)
        trow.addStretch(1)
        for track_button in (self.b_track_cut, self.b_track_play, self.b_track_del):
            trow.addWidget(track_button)
        tv.addLayout(trow)
        self.track_info = label("", level="tertiary")
        self.track_info.setWordWrap(True)
        tv.addWidget(self.track_info)
        shared.sound_v.addWidget(tw)

    def _build_sound_panel(self, shared):
        """Build the sound reference."""
        bar, _ = section_bar("Sound reference", shared.left)
        shared.sound_v.addWidget(bar)
        aw = QWidget()
        av = QVBoxLayout(aw)
        av.setContentsMargins(10, 6, 10, 8)
        av.setSpacing(6)
        self.sounds = QListWidget()
        self.sounds.setFixedHeight(70)
        self.sounds.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sounds.setToolTip("Sound references: sounds the model listens to (a voice, a sound to copy, a beat). "
                               "In a clip's prompt the first one is <Audio 1>, the second "
                               "<Audio 2> ... Double-click to listen")
        self.sounds.itemDoubleClicked.connect(self._play_sound)
        FileDrop(self.sounds.viewport(), lambda files: [self.add_sound(f) for f in files],
                 lambda: self.project is not None)
        self.sounds.currentRowChanged.connect(self._update_sound_buttons)
        av.addWidget(self.sounds)
        arow = QHBoxLayout()
        self.b_sound_add = QPushButton("File…")
        self.b_sound_add.setToolTip("Add the sound of an audio or video file (WAV, MP3, FLAC, MP4 …)")
        self.b_sound_add.clicked.connect(self._add_sound)
        self.b_sound_clip = QPushButton("Selected clip")
        self.b_sound_clip.clicked.connect(self._sound_from_selected)
        self.b_sound_play = icon_button("play", "Listen to the selected sound reference")
        self.b_sound_play.clicked.connect(self._play_sound)
        self.b_sound_del = QPushButton("Remove")
        self.b_sound_del.setToolTip("Remove the selected sound reference")
        self.b_sound_del.clicked.connect(self._remove_sound)
        arow.addWidget(label("Add:", level="tertiary"))
        arow.addWidget(self.b_sound_add)
        arow.addWidget(self.b_sound_clip)
        arow.addStretch(1)
        arow.addWidget(self.b_sound_play)
        arow.addWidget(self.b_sound_del)
        av.addLayout(arow)
        self.sound_info = label("", level="tertiary")
        self.sound_info.setWordWrap(True)
        av.addWidget(self.sound_info)
        shared.sound_v.addWidget(aw)

    def _build_clip_settings(self, shared):
        """Build the clip settings (the project's defaults)."""
        bar, _ = section_bar("Clip settings (the project's defaults)", shared.left)
        shared.proj_v.addWidget(bar)
        cw = QWidget()
        c = QFormLayout(cw)
        c.setContentsMargins(10, 6, 10, 8)
        self.adv_note = AdvancedNote(self.view)          # Simple: the hidden project settings that still apply
        d = UL.as_dict(T.DEFAULT_OPTIONS)                   # (the widgets start with the project defaults)
        c.addRow(self.adv_note)
        self.frames = QComboBox()
        for n in FRAME_CHOICES:
            self.frames.addItem(frames_text(n), n)
        self.frames.setCurrentIndex(FRAME_CHOICES.index(d["frames"]))
        self.frames.setToolTip("Length of every clip (a clip can have its own). Longer clips mean fewer hand-overs "
                               "between clips, so the face drifts less; they take longer and need more memory")
        c.addRow("Length", self.frames)
        self.resolution = QComboBox()
        for r, txt in ((384, "384 (fast draft)"), (512, "512 (standard)"), (640, "640"), (768, "768"),
                       (chain.H3_NATIVE, "768p — the model's native size (16:9 = 1344×768; slower)")):
            self.resolution.addItem(txt, r)
        self.resolution.setCurrentIndex(max(0, self.resolution.findData(d["resolution"])))
        self.resolution.setToolTip("Size: how big the clips are made, about this many pixels squared in the start "
                                   "image's aspect ratio (bigger is sharper and slower). 512: like the training (the "
                                   "default); 768p: MiniMax H3's native size (fast attention, Veda, helps there)")
        c.addRow("Size", self.resolution)
        self.exact_size = QCheckBox("Exact")
        self.exact_size.setToolTip(
            "Make the clips at exactly this width × height instead of the size worked out from Size and the start "
            "image's ratio (to match a clip made elsewhere). Each side is a multiple of 32; a start image of "
            "another ratio is stretched to fit")
        self.exact_width, self.exact_height = QSpinBox(), QSpinBox()
        for spin, name in ((self.exact_width, "Exact width"), (self.exact_height, "Exact height")):
            spin.setRange(chain.SIZE_MIN, chain.SIZE_MAX)
            spin.setSingleStep(chain.DIVISIBILITY)
            spin.setAccessibleName(name)
            spin.setEnabled(False)
        self.exact_width.setValue(768)
        self.exact_height.setValue(768)
        erow = QHBoxLayout()
        erow.setContentsMargins(0, 0, 0, 0)
        erow.addWidget(self.exact_size)
        erow.addWidget(self.exact_width, 1)
        erow.addWidget(label("×", level="tertiary"))
        erow.addWidget(self.exact_height, 1)
        c.addRow("Exact size", erow)
        self.aspect = QComboBox()
        for key in T.ASPECTS:
            self.aspect.addItem(key, key)
        self.aspect.setToolTip("The picture format of a project without a start image or real video: its first clip "
                               "starts from the prompt alone (with the cast's face pictures as references when it "
                               "mentions someone). With a start image or video, their format is used instead")
        self.aspect.setCurrentIndex(max(0, self.aspect.findData(d["aspect"])))
        c.addRow("Aspect", self.aspect)
        mrow = QHBoxLayout()
        mrow.setContentsMargins(0, 0, 0, 0)
        self.h3_model = QComboBox()
        self.h3_model.addItem("MiniMax H3 (standard)", "")
        self.h3_model.setToolTip(
            "The video model: the standard MiniMax H3 files, or another H3 checkpoint (a fine-tune) found in "
            "ComfyUI's diffusion model folders. A checkpoint makes every clip, with and without references. "
            "Clips made with another model count as changed")
        self.h3_distilled = QCheckBox(f"Distilled ({T.DISTILLED_STEPS} steps)")
        self.h3_distilled.setToolTip(
            f"The checkpoint has its turbo built in (a few-step model): every clip takes {T.DISTILLED_STEPS} steps "
            "with the FastH3 recipe (euler, sigma shift 12 / 3); the speed modes and Veda are off. Off: the "
            "checkpoint is sampled like the standard H3 (steps and speed mode as set)")
        mrow.addWidget(self.h3_model, 1)
        mrow.addWidget(self.h3_distilled)
        self._h3_model_row = row_widget(mrow)
        c.addRow("H3 model", self._h3_model_row)
        self.steps = QSpinBox()
        self.steps.setRange(4, 80)
        self.steps.setValue(d["steps"])
        self.steps.setToolTip(STEPS_TIP)
        c.addRow("Steps", self.steps)
        self.speed = QComboBox()
        self.speed.addItem("Normal", "normal")
        self.speed.addItem("Turbo (turbo LoRA, 4-8 steps)", "turbo")
        self.speed.addItem("HyperFlow (8 steps)", "hyperflow")
        self.speed.addItem("Pika H3 Flex (6 steps)", "pika")
        self.speed.setToolTip("Speed: make the clips faster with an add-on (LoRA) made for few steps; Normal "
                              "uses the full model. "
                              "HyperFlow: Video Rebirth's 8-step LoRA, about 2× faster than Normal (measured on an "
                              "RTX 4090: 19 s instead of 42 s for a 3 s clip at 672×384; needs the "
                              "ComfyUI-HyperFlow-H3 nodes and its pruned weights). "
                              "Pika H3 Flex: Pika's post-trained H3 as a LoRA, 6 steps (lcm), reported close to "
                              "the full model; your own LoRAs can be added. "
                              "Turbo: a step-distilled LoRA with its own sampler renders every clip in 4-8 steps "
                              "instead of ~20 (needs the ComfyUI-MiniMax-H3-Turbo nodes)")
        c.addRow("Speed", self.speed)
        crow = QHBoxLayout()
        crow.setContentsMargins(0, 0, 0, 0)
        self.clip_cfg = QDoubleSpinBox()
        self.clip_cfg.setRange(0.0, 10.0)
        self.clip_cfg.setSingleStep(0.05)
        self.clip_cfg.setDecimals(2)
        self.clip_cfg.setValue(1.0)
        self.clip_cfg.setToolTip(
            "CFG: how strictly the model follows the prompt. It belongs to the chosen speed mode (each mode keeps "
            "its own value). 1: off, one pass per step, as H3 and its speed LoRAs (turbo, HyperFlow, Pika) are made "
            "for. Above 1: real classifier-free guidance against an empty prompt (start / end frames stay): the "
            "prompt is followed more strongly, but every step runs twice "
            "(about twice as slow) and the step-distilled LoRAs can burn colours or over-sharpen from about 1.5-2; "
            "try 1.2-1.5 first. Below 1 (e.g. 0.9): the prompt is followed a little less, also two passes per step")
        self.cfg_note = label("", level="tertiary")
        crow.addWidget(self.clip_cfg)
        crow.addWidget(self.cfg_note, 1)
        self._cfg_row = row_widget(crow)
        c.addRow("CFG", self._cfg_row)
        self._cfg_mode = "normal"
        self.veda = QCheckBox("Fast attention (Veda) for large and long clips")
        self.veda.setChecked(d["veda"]["on"])
        self.veda.setToolTip(
            "Fast attention: makes large and long clips much faster. "
            "Veda sparse attention (the veda-sparse-attention node in ComfyUI) computes only the attention that "
            "matters. It is used automatically for clips large enough to gain, marked “⚡ Veda” on the clip. "
            "Measured on an RTX 4090: a 10 s clip at 1344×768 took 146 s instead of 338 s (2.3× faster) and stayed "
            "coherent; a 5 s clip at 672×384 got 25–30 % slower, so small clips keep full attention. A clip made "
            "with Veda differs slightly from one made without (switching this makes such clips CHANGED)")
        c.addRow("", self.veda)
        self.turbo_lora = QComboBox()
        self.turbo_lora.setToolTip("The turbo LoRA (only turbo LoRAs are listed here, and they are left out of the "
                                   "normal LoRA lists); v4 step600 ema is the recommended one")
        c.addRow("Turbo LoRA", self.turbo_lora)
        trow = QHBoxLayout()
        self.turbo_steps = QSpinBox()
        self.turbo_steps.setRange(4, 8)
        self.turbo_steps.setValue(d["turbo"]["steps"])
        self.turbo_steps.setSuffix(" steps")
        self.turbo_steps.setToolTip("4 is the minimum, 6-8 look clearly better; above 8 it gets over-sharp")
        self.turbo_strength = QDoubleSpinBox()
        self.turbo_strength.setRange(-10.0, 10.0)
        self.turbo_strength.setSingleStep(0.05)
        self.turbo_strength.setValue(d["turbo"]["strength"])
        self.turbo_strength.setToolTip("Keep 1.0; smear/ghosting -> a bit higher (1.05-1.2), over-sharp grain -> "
                                       "a bit lower (0.8-0.95)")
        trow.addWidget(self.turbo_steps)
        trow.addWidget(label_for("strength", self.turbo_strength, level="tertiary"))
        trow.addWidget(self.turbo_strength)
        trow.addStretch(1)
        self._turbo_row = QWidget()
        self._turbo_row.setLayout(trow)
        trow.setContentsMargins(0, 0, 0, 0)
        c.addRow("", self._turbo_row)
        self._clips_form = c
        self.audio = QCheckBox("Generate sound")
        self.audio.setChecked(d["audio"])
        c.addRow("", self.audio)
        self.keep_rep = QCheckBox("Keep repeated frames in paths")
        self.keep_rep.setChecked(d["keep_repeated"])
        self.keep_rep.setToolTip("Each clip starts with the last frame of the clip before; by default that repeated "
                                 "frame is left out when a path is joined")
        c.addRow("", self.keep_rep)
        self.style_prefix = QLineEdit()
        self.style_prefix.setPlaceholderText("e.g. cinematic, soft daylight, handheld camera")
        self.style_prefix.setToolTip("Put before every prompt (look, camera, light, a LoRA's trigger word). "
                                     "Changing it makes every generated clip again")
        self.style_suffix = QLineEdit()
        self.style_suffix.setPlaceholderText("optional")
        self.style_suffix.setToolTip("Put after every prompt. Changing it makes every generated clip again")
        self.prompt_format = QComboBox()
        self.prompt_format.addItem("Official H3", "h3")
        self.prompt_format.addItem("Classic", "classic")
        self.prompt_format.setToolTip(
            "<p><b>Official H3</b>: the prompts are written in MiniMax H3's official formats (a header line for a "
            "start / end frame; with pictures or sounds the six sections of the reference format, each person, place, "
            "object and style named as &lt;Subject N&gt; and tied to its pictures).<br><b>Classic</b>: the app's "
            "earlier format (the prompt as written, with sentences that tie each name to its pictures); projects "
            "made before the official format keep it.<br>Changing it marks the made clips as changed: they are kept, "
            "making them again is your choice.</p>")
        self.prompt_format.currentIndexChanged.connect(self._prompt_format_changed)
        c.addRow("Prompt format", self.prompt_format)
        c.addRow("Style before", self.style_prefix)
        c.addRow("Style after", self.style_suffix)
        for w in (self.style_prefix, self.style_suffix):
            w.editingFinished.connect(self._style_changed)
        self.lora_triggers = QCheckBox("Put the LoRAs' trigger words before the prompt")
        self.lora_triggers.setChecked(d["lora_triggers"])
        self.lora_triggers.setToolTip("Trigger words: the words a LoRA was trained with, which switch on its effect. "
                                      "A clip with LoRAs gets their trigger words in front of its prompt (from the "
                                      "LoRA files and CivitAI; shown under each LoRA, right-click to change them). "
                                      "Clips record them, so new words make those clips again")
        self.lora_triggers.toggled.connect(self._lora_triggers_changed)
        c.addRow("", self.lora_triggers)
        shared.proj_v.addWidget(cw)

    def _build_engine_prefs(self, shared):
        """Build Preferences: the video engine (ComfyUI)."""
        bar, _ = section_bar("Video engine (ComfyUI)", shared.left)
        shared.prefs_v.addWidget(bar)
        aw = QWidget()
        a = QFormLayout(aw)
        a.setContentsMargins(10, 6, 10, 8)
        self._comfy_form = a
        self.comfy_url = QLineEdit(DEFAULT_COMFY_URL)
        self.comfy_url.setToolTip("Address of the ComfyUI that renders the clips")
        a.addRow("Address", self.comfy_url)
        crow = QHBoxLayout()
        self.comfy_launcher = QLineEdit(DEFAULT_COMFY_LAUNCHER)
        self.comfy_launcher.setToolTip("Command that starts ComfyUI (your app-drawer launcher)")
        self.b_start_comfy = QPushButton("Start ComfyUI")
        self.b_start_comfy.setToolTip("Start ComfyUI in the background with this command")
        self.b_start_comfy.clicked.connect(self.start_comfy)
        crow.addWidget(self.comfy_launcher, 1)
        crow.addWidget(self.b_start_comfy)
        a.addRow(_buddy("Start with", self.comfy_launcher, "Command that starts ComfyUI"), crow)
        self.comfy_dir = QLineEdit(UL.DEFAULT_COMFY_DIR)
        self.comfy_dir.setToolTip("ComfyUI's folder: its LoRA folders are searched for the chosen LoRA files, and "
                                  "its copies of the clips are removed after they were taken over")
        drow = QHBoxLayout()
        drow.addWidget(self.comfy_dir, 1)
        self.b_rescan = QPushButton("Rescan LoRAs")
        self.b_rescan.setToolTip("Look for new LoRAs in ComfyUI's LoRA folders (also done when you come back to "
                                 "this window)")
        self.b_rescan.clicked.connect(lambda: self.scan_loras(report=True))
        drow.addWidget(self.b_rescan)
        self._comfy_dir_row = row_widget(drow)
        a.addRow(_buddy("Folder", self.comfy_dir, "ComfyUI folder"), self._comfy_dir_row)
        self.comfy_cleanup = QCheckBox("Clean up ComfyUI's copies after each clip")
        self.comfy_cleanup.setChecked(True)
        self.comfy_cleanup.setToolTip("ComfyUI keeps every clip and start image in output/video-chain and "
                                      "input/video-chain; the project has its own copy, so they are removed after "
                                      "the download (only there, never anything else)")
        a.addRow("", self.comfy_cleanup)
        self.comfy_autostart = QCheckBox("Start ComfyUI when generating")
        self.comfy_autostart.setChecked(True)
        self.comfy_autostart.setToolTip("If ComfyUI is not running when you generate, start it with the command "
                                        "above and begin as soon as it answers")
        a.addRow("", self.comfy_autostart)
        self.triggers_online = QCheckBox("Look up LoRAs on CivitAI (trigger words, example prompts)")
        self.triggers_online.setChecked(True)
        self.triggers_online.setToolTip("In the background, after each LoRA scan: LoRAs are looked up on CivitAI by "
                                        "their file hash (this tells CivitAI which LoRA files you have, nothing else). "
                                        "Found trigger words are kept in ~/.local/share/video-chain/"
                                        "lora_triggers.json, everything else CivitAI tells next to it in civitai/: "
                                        "descriptions, base model, files, license, and the prompts and settings of "
                                        "the pictures in the LoRA's gallery (up to 2000 pictures are looked at per "
                                        "LoRA, so the first lookup of many LoRAs takes a while; the status line "
                                        "says where it is). The prompt refiner uses them to write for a LoRA. "
                                        "Right-click a LoRA to see all that is known about it.")
        self.llm_on = QCheckBox("Prompt refiner: also a llama.cpp server (GGUF language models)")
        self.llm_on.setToolTip("For language models ComfyUI cannot run (GGUF, e.g. Huihui Qwen3-VL-30B-A3B "
                               "abliterated). The app starts llama-server when you refine with such a model and "
                               "stops it before clips are made and when it closes (the model fills a lot of RAM).")
        a.addRow("", self.llm_on)
        self.llm_server = QLineEdit(LLM_DEFAULT_SERVER)
        self.llm_server.setToolTip("The llama-server program (llama.cpp)")
        a.addRow("llama-server", self.llm_server)
        lrow = QHBoxLayout()
        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)
        self.llm_model.setToolTip("The GGUF model file (found in the model folders; a path can be typed too)")
        # (the model folders are searched in the background after the settings are loaded: _scan_gguf_models)
        self.llm_port = QSpinBox()
        self.llm_port.setRange(1024, 65535)
        self.llm_port.setValue(LLM_DEFAULT_PORT)
        self.llm_port.setToolTip("The local port of the llama.cpp server")
        lrow.addWidget(self.llm_model, 1)
        lrow.addWidget(label_for("port", self.llm_port, level="tertiary"))
        lrow.addWidget(self.llm_port)
        self._gguf_row = row_widget(lrow)
        a.addRow("GGUF model", self._gguf_row)
        self.llm_cpu_moe = QCheckBox("Keep the model's experts in RAM (GPU stays free for ComfyUI; slower)")
        self.llm_cpu_moe.setChecked(True)
        self.llm_cpu_moe.setToolTip("Mixture-of-experts models (…-A3B): their expert weights stay in RAM and only "
                                    "attention runs on the GPU, a few GB of VRAM. Off: as much as fits on the GPU "
                                    "(faster, but it competes with ComfyUI for VRAM)")
        a.addRow("", self.llm_cpu_moe)
        self.video_gpu = QCheckBox("Play videos on the GPU (smooth; takes effect after a restart)")
        self.video_gpu.setChecked(True)
        self.video_gpu.setToolTip("On: the preview players decode on the GPU and draw through Qt's GPU video surface: "
                                  "smooth also with big or 2K clips. NVIDIA's driver crashed there once while "
                                  "ComfyUI held nearly all GPU memory; if the app crashes when playing, switch it "
                                  "off. Off: decoded and drawn on the CPU: never crashes, but lags with big clips. "
                                  "(The before / after comparison is always drawn on the CPU.)")
        self.video_gpu.toggled.connect(self._video_gpu_changed)
        a.addRow("", self.video_gpu)
        self.triggers_online.toggled.connect(lambda on: on and self._scan_triggers())
        a.addRow("", self.triggers_online)
        self.civitai_site = QComboBox()
        for s_ in triggers.SITES:
            self.civitai_site.addItem(s_.split("//", 1)[1], s_)
        self.civitai_site.setToolTip("CivitAI has two sites that share their LoRAs: civitai.com and civitai.red. "
                                     "The one chosen here is asked first and is where links open (a LoRA's page, a "
                                     "gallery picture). The other is asked too when the first does not know a LoRA "
                                     "or cannot be reached.")
        self.civitai_site.currentIndexChanged.connect(self._civitai_site_changed)
        a.addRow("CivitAI site", self.civitai_site)
        crow = QHBoxLayout()
        self.civitai_key = QLineEdit()
        self.civitai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.civitai_key.setToolTip("Your CivitAI API key (civitai.com → Account settings → API Keys). With it CivitAI "
                                    "answers as your account, e.g. with the prompts of example images it holds back "
                                    "otherwise. Kept in the system keyring, never in a settings file; sent only to "
                                    "CivitAI. ($CIVITAI_API_KEY wins when it is set.)")
        b_key = QPushButton("Save key")
        b_key.setAutoDefault(False)
        b_key.clicked.connect(self._save_civitai_key)
        self.civitai_key.returnPressed.connect(self._save_civitai_key)    # (Enter saves too)
        crow.addWidget(self.civitai_key, 1)
        crow.addWidget(b_key)
        self._civitai_row = row_widget(crow)
        a.addRow(_buddy("CivitAI API key", self.civitai_key), self._civitai_row)
        self._show_civitai_key()
        erow = QHBoxLayout()
        self.eleven_key = QLineEdit()
        self.eleven_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.eleven_key.setToolTip("Your ElevenLabs API key (elevenlabs.io → Developers → API keys), for text to "
                                   "speech with ElevenLabs. Kept in the system keyring, never in a settings file or "
                                   "a project; sent only to ElevenLabs. ($ELEVENLABS_API_KEY wins when it is set.)")
        b_ekey = QPushButton("Save key")
        b_ekey.setAutoDefault(False)
        b_ekey.clicked.connect(self._save_eleven_key)
        self.eleven_key.returnPressed.connect(self._save_eleven_key)
        erow.addWidget(self.eleven_key, 1)
        erow.addWidget(b_ekey)
        a.addRow(_buddy("ElevenLabs API key", self.eleven_key), erow)
        self._show_eleven_key()
        krow = QHBoxLayout()
        self.keep_loaded = QCheckBox("Keep the model loaded for")
        self.keep_loaded.setChecked(True)
        self.keep_loaded.setToolTip("After generating (or editing images), the model stays in memory, so the next "
                                    "clip or edit starts at once instead of loading for minutes. It is unloaded "
                                    "after this idle time, or with Unload in the status bar. Switching between "
                                    "clips (MiniMax H3) and image edits (Qwen) unloads the other model first.")
        self.idle_minutes = QSpinBox()
        self.idle_minutes.setRange(1, 240)
        self.idle_minutes.setValue(15)
        self.idle_minutes.setSuffix(" min")
        krow.addWidget(self.keep_loaded)
        krow.addWidget(self.idle_minutes)
        krow.addStretch(1)
        self._keep_row = row_widget(krow)
        a.addRow("", self._keep_row)
        self.keep_loaded.toggled.connect(self.idle_minutes.setEnabled)
        shared.prefs_v.addWidget(aw)

    def _build_runpod_prefs(self, shared):
        """Build Preferences: cloud upscaling (RunPod)."""
        bar, _ = section_bar("Cloud upscaling (RunPod)", shared.left)
        shared.prefs_v.addWidget(bar)
        rw = QWidget()
        self._runpod_section = (bar, rw)
        r = QFormLayout(rw)
        r.setContentsMargins(10, 6, 10, 8)
        rrow = QHBoxLayout()
        self.runpod_key = QLineEdit()
        self.runpod_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.runpod_key.setToolTip("Your RunPod API key (runpod.io → Settings → API Keys), for “Upscale on RunPod” "
                                   "in the export. Kept in the system keyring, never in a settings file or a "
                                   "project; sent only to RunPod. ($RUNPOD_API_KEY wins when it is set.)")
        b_rkey = QPushButton("Save key")
        b_rkey.setAutoDefault(False)
        b_rkey.clicked.connect(self._save_runpod_key)
        self.runpod_key.returnPressed.connect(self._save_runpod_key)
        rrow.addWidget(self.runpod_key, 1)
        rrow.addWidget(b_rkey)
        r.addRow(_buddy("RunPod API key", self.runpod_key), rrow)
        self.runpod_gpu = QComboBox()
        self.runpod_gpu.setEditable(True)
        for gid, text in RP.GPUS:
            self.runpod_gpu.addItem(text, gid)
        self.runpod_gpu.setToolTip("The GPU the export's pod gets (Secure Cloud only: RunPod's own data centres, "
                                   "never Community Cloud hosts). A RunPod GPU type id can be typed too. Prices: "
                                   "RunPod's console or: python -m video_chain.runpod gpus")
        r.addRow("GPU", self.runpod_gpu)
        self.runpod_fallback = QCheckBox("Take another of these GPUs when this one is not free")
        self.runpod_fallback.setChecked(True)
        self.runpod_fallback.setToolTip("In the order of the list (fastest first). Off: the export waits for "
                                        "nothing and fails when the chosen GPU has no capacity")
        r.addRow("", self.runpod_fallback)
        self.runpod_volume = QLineEdit()
        self.runpod_volume.setPlaceholderText("optional: a global or network volume id, for fast starts")
        self.runpod_volume.setToolTip("A RunPod volume keeps ComfyUI, its packages and the SeedVR2 weights (about "
                                      "15 GB) between pods: a pod is ready in minutes instead of 5-10 more. It only "
                                      "ever holds these public files, never a clip: those stay in the pod's RAM. "
                                      "A global volume mounts in any data centre (the GPU can be wherever one is "
                                      "free); a network volume ties the pod to its data centre.")
        r.addRow("Volume", self.runpod_volume)
        self.runpod_without_volume = QCheckBox("Start without the volume when its data centre has no GPU free")
        self.runpod_without_volume.setChecked(True)
        self.runpod_without_volume.setToolTip("A pod with the volume must run in the volume's data centre, and that "
                                              "one may have none of the GPUs free. On: the pod then starts anywhere "
                                              "without it, and ComfyUI and the weights are fetched again (about 5-10 "
                                              "min more). Off: the export stops and says where GPUs are free")
        r.addRow("", self.runpod_without_volume)
        note = label("Your clips never touch a RunPod disk: they go through an SSH tunnel (the pod's host key is "
                     "checked through RunPod's API) and sit only in the pod's RAM, removed after each job; the pod is "
                     "terminated when the export ends. Leftovers: python -m video_chain.runpod status / down.",
                     level="tertiary")
        note.setWordWrap(True)
        r.addRow("", note)
        self.runpod_gpu.currentTextChanged.connect(lambda *_: self._runpod_changed())
        self.runpod_fallback.toggled.connect(lambda *_: self._runpod_changed())
        self.runpod_without_volume.toggled.connect(lambda *_: self._runpod_changed())
        self.runpod_volume.editingFinished.connect(self._runpod_changed)
        shared.prefs_v.addWidget(rw)
        self._show_runpod_key()

    def _build_inspector_tabs(self, shared):
        """Build the left and the right tab widgets."""
        pb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        pb.button(QDialogButtonBox.StandardButton.Close).setAutoDefault(False)     # Enter in a field must not close it
        pb.button(QDialogButtonBox.StandardButton.Close).setDefault(False)
        pb.rejected.connect(self.prefs.reject)
        shared.prefs_v.addWidget(pb)
        for v in (shared.proj_v, shared.cast_v, shared.sound_v):
            v.addStretch(1)
        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)
        shared.cast_scroll, shared.sound_scroll = _scrolled(shared.cast_w), _scrolled(shared.sound_w)
        proj_scroll = _scrolled(shared.proj_w)
        self.inspector = QTabWidget()
        self.inspector.setDocumentMode(True)
        self.inspector.addTab(proj_scroll, "Project")
        self.proj_scroll = proj_scroll
        self.inspector.setMinimumWidth(380)
        self._panels = (shared.proj_w, shared.cast_w, shared.sound_w)

    def _build_film_strip(self, shared):
        """Build the film strip under the viewer."""
        # the clips graph (left tab), the film strip (under the viewer)
        shared.gw = QWidget()
        shared.gv = QVBoxLayout(shared.gw)
        shared.gv.setContentsMargins(0, 0, 0, 0)
        shared.gv.setSpacing(0)
        shared.film_w = QWidget()
        film_v = QVBoxLayout(shared.film_w)
        film_v.setContentsMargins(0, 0, 0, 0)
        film_v.setSpacing(0)
        fbar, fbh = section_bar("Film", shared.film_w)
        self.film_info = label("", level="tertiary")
        fbh.addWidget(self.film_info)
        film_v.addWidget(fbar)
        self.storyboard = Storyboard()
        self.storyboard.sceneChosen.connect(self.switch_scene)
        self.storyboard.addWanted.connect(self.add_scene)
        self.storyboard.renameWanted.connect(self.rename_scene)
        self.storyboard.deleteWanted.connect(self.delete_scene)
        self.storyboard.moved.connect(self._scenes_moved)
        self.storyboard.joinWanted.connect(self.set_scene_join)
        self.storyboard.duplicateWanted.connect(self.duplicate_scene)
        self.storyboard.notesWanted.connect(self.scene_notes)
        self.b_film_gen = QPushButton("Generate film")
        self.b_film_gen.setToolTip("Make every scene's missing clips, scene after scene (up to each scene's ending)")
        self.b_film_gen.clicked.connect(self.generate_film)
        self.b_film_play = QPushButton("Play film")
        self.b_film_play.setToolTip("Join all scenes in their order (hard cuts) and play the film")
        self.b_film_play.clicked.connect(lambda: self.join_film("play"))
        self.b_film_export = QPushButton("Export film…")
        self.b_film_export.setToolTip("Save the whole film as one video")
        self.b_film_export.clicked.connect(self.export_film)
        self.b_film_cast = QPushButton("Cast…")
        self.b_film_cast.setToolTip("Who appears in which scene (along the clips the film plays)")
        self.b_film_cast.clicked.connect(self.show_film_cast)
        for b in (self.b_film_cast, self.b_film_gen, self.b_film_play, self.b_film_export):
            fbh.addWidget(b)
        film_v.addWidget(self.storyboard)
        self.film = None

    def _build_graph_panel(self, shared):
        """Build the clips graph with the engine banner and the guide."""
        bar, bh = section_bar("Clips", shared.gw)
        self.tree_info = label("", level="tertiary")
        bh.addWidget(self.tree_info)
        bh.addSpacing(8)
        self.b_continue = icon_button("plus", S.tip("Continue: a new clip after the selected one", "continue"))
        self.b_add_video = icon_button("camera", S.tip("Add a real video after the selected clip", "add_video") + ": a "
                                                 "generated clip before it ends at its first frame, one after it "
                                                 "starts from its last frame")
        self.b_branch = icon_button("split", S.tip("Branch: another continuation from the same point", "branch"))
        self.b_variant = icon_button("clone", S.tip("Variant: same prompt, other seed", "variant"))
        self.b_delete = icon_button("trash", S.tip("Delete the selected clip and everything after it", "graph_delete"))
        self.b_fit = icon_button("frame", S.tip("Fit the view", "graph_fit"))
        for btn, act in ((self.b_continue, "continue"), (self.b_add_video, "video"), (self.b_branch, "branch"),
                         (self.b_variant, "variant"), (self.b_delete, "delete")):
            btn.clicked.connect(lambda _=False, a=act: self.node_action(a, self.target_id()))
            bh.addWidget(btn)
        bh.addSpacing(6)
        self.b_fit.clicked.connect(lambda: self.graph.fit())
        bh.addWidget(self.b_fit)
        shared.gv.addWidget(bar)
        # above the clips: the engine banner (engine not reachable), the guide (no made clip yet), the one-time
        # "how to go on" line (after the first made clip)
        notes = QVBoxLayout()
        notes.setContentsMargins(6, 6, 6, 0)
        notes.setSpacing(6)
        self.engine_banner = EngineBanner()
        self.engine_banner.button.clicked.connect(self._engine_banner_clicked)
        self.engine_banner.hide()
        self.guide_panel = GuidePanel()
        self.guide_panel.hideWanted.connect(lambda: self.show_guide(False))
        self.guide_panel.b_new.clicked.connect(self.new_project)
        self.guide_panel.b_open.clicked.connect(self._choose_project)
        self.guide_panel.b_picture.clicked.connect(self._choose_image)
        self.guide_panel.b_text_only.clicked.connect(self._guide_text_only)
        self.guide_panel.b_cast.clicked.connect(self._guide_cast)
        self.guide_panel.b_skip_cast.clicked.connect(lambda: self._guide_mark("no_cast"))
        self.guide_panel.b_write.clicked.connect(self._guide_write)
        self.guide_panel.b_show_gen.clicked.connect(lambda: self.b_gen.setFocus(Qt.FocusReason.OtherFocusReason))
        self.guide_panel.hide()
        self.next_hint = NextHint()
        self.next_hint.b_next.clicked.connect(lambda: self.node_action("continue", self.target_id()))
        self.next_hint.closeWanted.connect(self._next_hint_done)
        self.next_hint.hide()
        for w in (self.engine_banner, self.guide_panel, self.next_hint):
            notes.addWidget(w)
        shared.gv.addLayout(notes)
        self._guide_ready = False                  # every needed step of the guide is done (Generate is next)
        self.graph = GraphView()
        self.graph.selected.connect(self._graph_clicked)
        self.graph.activated.connect(self._activate)
        self.graph.contextMenu.connect(self._context)
        shared.gv.addWidget(self.graph, 1)
        self.left_tabs.addTab(shared.gw, "Clips")
        self.left_tabs.addTab(shared.cast_scroll, "Cast & references")
        self.left_tabs.addTab(shared.sound_scroll, "Sounds")
        self.left_tabs.setMinimumWidth(340)

    def _build_path_panel(self, shared):
        """Build the selected path: cast strip and timeline."""
        shared.pw2 = QWidget()
        pv = QVBoxLayout(shared.pw2)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        bar, bh = section_bar("Selected path", shared.pw2)
        self.path_info = label("", level="tertiary")
        bh.addWidget(self.path_info)
        pv.addWidget(bar)
        self.cast_strip = CastStrip()
        self.cast_strip.clipClicked.connect(self._show_card_of)
        self.cast_strip.hide()
        pv.addWidget(self.cast_strip)
        self.timeline = Timeline()
        self.timeline.clicked.connect(self.focus_clip)
        self.timeline.activated.connect(self._play_node)
        self.timeline.menu.connect(self._context)
        pv.addWidget(self.timeline)
        pv.addStretch(1)

    def _build_clip_tab(self, shared):
        """Build the inspector's Clip tab."""
        # the inspector's Clip tab: the card of the clip in focus (one at a time; the path's other cards wait in
        # the stack), with ‹ › to step along the path
        clip_tab = QWidget()
        ctv = QVBoxLayout(clip_tab)
        ctv.setContentsMargins(8, 6, 8, 6)
        nav = QHBoxLayout()
        self.b_prev_clip = QPushButton("‹")
        self.b_prev_clip.setToolTip(S.tip("The clip before it on the path", "prev_clip"))
        self.b_prev_clip.clicked.connect(lambda: self._step_focus(-1))
        self.b_next_clip = QPushButton("›")
        self.b_next_clip.setToolTip(S.tip("The clip after it on the path", "next_clip"))
        self.b_next_clip.clicked.connect(lambda: self._step_focus(1))
        self.focus_info = label("", level="secondary")
        for b in (self.b_prev_clip, self.b_next_clip):
            b.setFixedWidth(32)
        nav.addWidget(self.b_prev_clip)
        nav.addWidget(self.focus_info, 1)
        nav.addWidget(self.b_next_clip)
        ctv.addLayout(nav)
        self.card_stack = _CurrentStack()
        self.path_hint = label("", level="tertiary")
        self.path_hint.setWordWrap(True)
        self.path_hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.card_stack.addWidget(self.path_hint)
        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        card_scroll.setFrameShape(QFrame.Shape.NoFrame)
        card_scroll.setWidget(self.card_stack)
        ctv.addWidget(card_scroll, 1)
        self.inspector.insertTab(0, clip_tab, "Clip")
        self.inspector.setCurrentIndex(0)
        self.focus: str | None = None

    def _build_viewer(self, shared):
        """Build the viewer, the bottom tabs and the splitters."""
        # the viewer (the biggest area), under it the film and the selected path / the log
        rp = QWidget()
        rv = QVBoxLayout(rp)
        rv.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview()
        self._played_path: tuple[str, list] | None = None   # the joined path in the preview, its clips' spans
        self.preview.opened.connect(self._preview_opened)
        self.preview.moved.connect(self._preview_moved)
        self.preview.failed.connect(self._preview_failed)
        rv.addWidget(self.preview, 1)
        prow = QHBoxLayout()
        prow.setContentsMargins(8, 0, 8, 8)
        self.b_play_clip = QPushButton("▶ Clip")
        self.b_play_clip.setToolTip("Play the clip shown in the inspector (Space in the timeline: double-click)")
        self.b_play_clip.clicked.connect(lambda: self.focus and self._play_node(self.focus))
        self.b_play_path = QPushButton("▶ Path")
        self.b_play_path.setToolTip("Join the clips of the selected path and play them")
        self.b_play_path.clicked.connect(lambda: self.join_selected("play"))
        self.b_play_film = QPushButton("▶ Film")
        self.b_play_film.setToolTip("Join all scenes of the film and play it")
        self.b_play_film.clicked.connect(lambda: self.join_film("play"))
        self.b_export = QPushButton("Export path…")
        self.b_export.setToolTip("Save the selected path as one video")
        self.b_export.clicked.connect(lambda: self.export_path())          # (clicked passes checked=False)
        self.b_export_clip = QPushButton("Export clip…")
        self.b_export_clip.setToolTip("Save the clip in the inspector on its own (as made, or upscaled)")
        self.b_export_clip.clicked.connect(lambda: self.focus and self.export_path(self.focus))
        prow.addWidget(label("Play", level="tertiary"))
        for b in (self.b_play_clip, self.b_play_path, self.b_play_film):
            prow.addWidget(b)
        prow.addSpacing(12)
        prow.addWidget(self.b_export)
        prow.addWidget(self.b_export_clip)
        prow.addStretch(1)
        rv.addLayout(prow)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("type", "mono")
        self.log.setMaximumBlockCount(5000)
        self.tabs = QTabWidget()                            # under the viewer: the selected path, the log
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(shared.pw2, "Selected path")
        self.tabs.addTab(self.log, "Log")
        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(0)
        bv.addWidget(shared.film_w)
        bv.addWidget(self.tabs, 1)
        center = QSplitter(Qt.Orientation.Vertical)
        center.setChildrenCollapsible(False)
        center.addWidget(rp)
        center.addWidget(bottom)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        self.center = center
        shared.split.addWidget(self.left_tabs)
        shared.split.addWidget(center)
        shared.split.addWidget(self.inspector)
        shared.split.setStretchFactor(1, 1)
        shared.split.setSizes([400, 1300, 420])
        self.viewer = rp

    def _build_toolbar(self, shared):
        """Build the status bar, the run toolbar, the view buttons."""
        sb = self.statusBar()
        self.b_gen = QPushButton("Generate path")
        self.b_gen.setProperty("variant", "primary")
        self.b_gen.setToolTip(S.tip("Make every clip on the selected path that is not made yet", "generate"))
        self.b_gen.clicked.connect(self.generate_path)
        self.b_gen_all = QPushButton("Generate all branches")
        self.b_gen_all.setToolTip(S.tip("Make every clip of every branch that is not made yet", "generate_all"))
        self.b_gen_all.clicked.connect(self.generate_all)
        self.b_stop = QPushButton("Stop")
        self.b_stop.setProperty("variant", "danger")
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("Progress of the run")
        self.progress.setFixedWidth(220)
        self.progress.setRange(0, 1000)
        self.status = label("Open or create a project folder.", role="statusText")
        self.gpu = label("GPU …", role="statusText")
        tb = self.addToolBar("Run")                # making clips: at the top, like an editor's render buttons
        tb.setObjectName("runToolbar")
        tb.setMovable(False)
        for w in (self.b_gen, self.b_gen_all, self.b_stop):
            tb.addWidget(w)
        tb.addSeparator()
        tb.addWidget(self.progress)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.view_buttons = QButtonGroup(self)      # Simple | Advanced, always in sight (also in the View menu)
        self.b_simple = QPushButton("Simple")
        self.b_simple.setToolTip("Simple view: only what you need to make and continue clips. Hidden settings keep "
                                 "their values (a line says when some of them are active)")
        self.b_advanced = QPushButton("Advanced")
        self.b_advanced.setToolTip("Advanced view: every setting (speed, steps, LoRAs, seeds, keyframes, ComfyUI …)")
        for b, simple in ((self.b_simple, True), (self.b_advanced, False)):
            b.setCheckable(True)
            b.setAutoDefault(False)
            self.view_buttons.addButton(b)
            b.clicked.connect(lambda _=False, on=simple: self.set_simple(on))
            tb.addWidget(b)
        self.run_toolbar = tb
        sb.addWidget(self.status, 1)
        self.model_label = label("Model: not loaded", role="statusText")
        self.b_unload = QPushButton("Unload")
        self.b_unload.setToolTip("Free the GPU and RAM now (the next Generate loads the model again)")
        self.b_unload.clicked.connect(self.unload)
        self.b_unload.hide()
        self.comfy_label = label("", role="statusText")
        self.b_prefs = icon_button("gear", S.tip("Preferences: the video engine (ComfyUI), API keys", "preferences"))
        self.b_prefs.clicked.connect(self.show_prefs)
        sb.addPermanentWidget(self.comfy_label)
        sb.addPermanentWidget(self.b_prefs)
        sb.addPermanentWidget(self.model_label)
        sb.addPermanentWidget(self.b_unload)
        sb.addPermanentWidget(self.gpu)
        self._model_timer = timer(self, self._update_model_label, 20000)
        self._model_timer.start()

        self.start_tile.clicked.connect(self._choose_image)
        for lst in (self.refs, self.cast_list):          # pictures dropped on the cast are faces / pictures of it
            FileDrop(lst.viewport(), lambda files: [self.add_reference(f) for f in files],
                     lambda: self.project is not None)
        FileDrop(self, self._dropped_on_window)         # a project folder dropped anywhere on the window
        self.start_tile.dropped.connect(lambda f: self.set_image(f, confirm=True))
        for spin in (self.exact_width, self.exact_height):
            spin.valueChanged.connect(self._options_typing)
            spin.editingFinished.connect(self._exact_shown)     # (typed in full: the nearest size into the field)
        for sig in (self.frames.currentIndexChanged, self.resolution.currentIndexChanged, self.exact_size.toggled,
                    self.aspect.currentIndexChanged, self.audio.toggled, self.keep_rep.toggled,
                    self.speed.currentIndexChanged, self.veda.toggled, self.turbo_lora.currentIndexChanged,
                    self.h3_distilled.toggled):
            sig.connect(self._options_changed)
        self.h3_model.currentIndexChanged.connect(self._h3_model_chosen)
        for sig in (self.steps.valueChanged, self.turbo_steps.valueChanged, self.turbo_strength.valueChanged,
                    self.clip_cfg.valueChanged):             # (spin boxes tick many times: settled after a pause)
            sig.connect(self._options_typing)

    def _build_shortcuts(self, shared):
        """Build the shortcuts."""
        for sid, fn in (("continue", lambda: self.node_action("continue", self.target_id())),
                        ("add_video", lambda: self.node_action("video", self.target_id())),
                        ("branch", lambda: self.node_action("branch", self.target_id())),
                        ("variant", lambda: self.node_action("variant", self.target_id())),
                        ("generate", self.generate_path), ("generate_all", self.generate_all),
                        ("prev_clip", lambda: self._step_focus(-1)), ("next_clip", lambda: self._step_focus(1))):
            shortcut(S.sequence(sid), self, fn)
        for w in (*self._panels, self.prefs):
            guard_wheel(w)
        # plain keys act only while the graph (or a child) has the focus: in a list or a combo, Delete / F are theirs
        self.graph.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for sid, fn in (("graph_delete", lambda: self.node_action("delete", self.target_id())),
                        ("graph_fit", self.graph.fit)):
            sc = shortcut(S.sequence(sid), self.graph, fn)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
