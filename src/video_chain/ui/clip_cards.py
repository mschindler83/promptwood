"""The clip cards of the inspector: the prompt card of a generated clip and the card of a real video."""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QCursor, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QLabel, QFrame, QHBoxLayout, QMenu, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import camera as CAM, chain, tree as T
from . import shortcuts as S
from .clip_inputs import ChipRow
from .keyframe_strip import KeyframeStrip
from .lora_stack import LoraStack
from .mentions import MentionHelper
from .problems import EXAMPLES, ProblemBox, add_example, camera_menu_items, insert_at_cursor
from .prompt_refiner import replace_text
from .theme import icon, repolish, theme
from .view_mode import AdvancedNote, ViewMode, row_widget
from .widgets import FrameTile, format_time, guard_wheel, icon_button, label
from .mw_support import FileDrop, select_frames
from .ui_logic import (
    CARD_WIDTH, CONTINUE_FIRST_TIP, CONTINUE_SHORT_TIP, CONTINUE_TIP, END_TIP, FRAME_CHOICES, FREEZE_TIP,
    GUIDE_TEXT, GUIDE_TIP, MIX_OFF_TIP, MIX_TIP, SOUND_MODES, SPEAK_TIP, STATE_STYLE, STATE_TEXT, STEPS_UNSET,
    continue_text, frames_text, plural)


class _CardAction(QAction):
    """An action of a card's header (its More menu). click() as the header button it replaced had."""

    def click(self):
        self.trigger()


def more_menu(card, nid: str, entries, tip: str = "More actions for this clip: variants, compare, branch, freeze, "
                                                   "delete") -> tuple[QToolButton, dict[str, QAction]]:
    """A card's "More" button: a menu of (action, icon, text, tooltip, shortcut or None) entries (None: a separator)
    that emit card.action(action, nid), the same signal the header buttons had. A shortcut is only shown (and works
    while the menu is open): the window's own shortcuts do the work, one per window, not one per card."""
    b = QToolButton()
    b.setText("More")
    b.setIcon(icon("more"))
    b.setIconSize(QSize(16, 16))
    b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    b.setAutoRaise(True)
    b.setPopupMode(QToolButton.InstantPopup)
    b.setToolTip(tip)
    m = QMenu(b)
    m.setToolTipsVisible(True)
    acts = {}
    for e in entries:
        if e is None:
            m.addSeparator()
            continue
        key, ic, text, tip, keys = e
        a = _CardAction(icon(ic), text, m)
        a.setToolTip(tip)
        if keys:
            a.setShortcut(QKeySequence(keys))
            a.setShortcutContext(Qt.WidgetShortcut)
        a.triggered.connect(lambda _=False, k=key: card.action.emit(k, nid))
        m.addAction(a)
        acts[key] = a
    b.setMenu(m)
    return b, acts


class _ClipCardBase(QFrame):
    """What the card of a made clip and the card of a real video share: the state tag and picture, the border while it
    is worked on, freezing. The subclasses say how they differ in the class attributes and _frozen_tip."""
    _NO_PICTURE = "not made yet"                       # the thumbnail's text while there is nothing to show
    _BUSY = ("generating",)                            # the states in which the card shows the progress bar
    _BACKGROUND = "bgRaised"                           # the theme colour of the card's background

    def set_state(self, state: str, pixmap: QPixmap | None, step: str = ""):
        self.state = state
        self.tag.setText(STATE_TEXT.get(state, state.upper()))
        self.tag.setProperty("state", STATE_STYLE.get(state, ""))
        repolish(self.tag)
        if pixmap is not None and not pixmap.isNull():
            self.thumb.set_image(pixmap.toImage())
        else:
            self.thumb.clear_frame(self._NO_PICTURE)
        self.step.setText(step)
        self.bar.setVisible(state in self._BUSY)
        self._restyle()

    def set_progress(self, i: int, n: int, text: str | None = None):
        """The progress of the work on the card (a real video has none)."""

    def _frozen_tip(self, frozen: bool) -> str:
        raise NotImplementedError

    def set_frozen(self, frozen: bool):
        self.frozen = bool(frozen)
        self.b_freeze.setChecked(self.frozen)
        self.b_freeze.setToolTip(self._frozen_tip(self.frozen))
        self.set_editable(self._on)

    def _restyle(self):
        t = theme()
        border = t.accent.name() if self.state in self._BUSY else t.lineSubtle.name()
        css = (f"QFrame#promptCard {{ background: {getattr(t, self._BACKGROUND).name()}; border: 1px solid {border};"
               f" border-radius: 6px; }} QFrame#promptCard QLabel {{ background: transparent; }}")
        if css != self.styleSheet():                # (a new sheet re-polishes the card and everything in it)
            self.setStyleSheet(css)


class PromptCard(_ClipCardBase):
    """One clip of the selected path: prompt, seed, LoRAs, own length / steps, status."""
    settings_open = False                      # the own-settings part is open (the same on every card)
    edited = Signal(str)
    seedChanged = Signal(str, int)
    overridesChanged = Signal(str, dict)
    chooseLora = Signal(str)
    lora_picked = Signal(str, str)             # node id, LoRA path chosen in the picker
    action = Signal(str, str)                  # action, node id
    keyframe = Signal(str, str, object)        # keyframe action, node id, {"index", "at"}
    end_dropped = Signal(str, str)             # node id, a picture dropped on the end of the clip
    play = Signal(str)
    leadChanged = Signal(str, bool)            # node id, the lead switch (the lead chip)
    modelPrompt = Signal(str)                  # node id: show the exact prompt the model gets
    fix = Signal(str, str)                     # what to fix (a problem's action, "cast" …), node id

    def __init__(self, node: T.Node, parent=None, names=lambda: [], view: ViewMode | None = None):
        shared = SimpleNamespace()          # what the sections of the build hand to each other
        super().__init__(parent)
        self.nid = node.id
        self.names = names
        self.setObjectName("promptCard")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setFixedWidth(CARD_WIDTH)
        self.state = ""
        shared.v = QVBoxLayout(self)
        shared.v.setContentsMargins(10, 8, 8, 10)
        shared.v.setSpacing(6)
        self._build_header(shared, node)
        self._build_prompt(shared, node, names)
        self._build_prompt_tools(shared)
        self._build_problems(shared, node)
        self._build_own_settings(shared)
        self._build_link_rows(shared)
        self._build_sound_rows(shared)
        self._build_end_rows(shared)
        self._build_footer(shared, view)

    def _build_header(self, shared, node):
        """Build the header: title, state tag, variant button, More menu."""
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = label("Clip", type_="heading")
        self.sub = label(f"#{T.short_id(node.id)}", level="tertiary")
        self.tag = label("", role="statusTag")
        head.addWidget(self.title)
        head.addWidget(self.sub)
        head.addWidget(self.tag)
        head.addStretch(1)
        # the header: the most used action with its name, the others in one More menu (delete last, it asks)
        self.act_variant = _CardAction(icon("clone"), "Variant", self)
        self.act_variant.setToolTip(S.tip("Variant: same prompt, other seed, as a new branch", "variant"))
        self.act_variant.triggered.connect(lambda: self.action.emit("variant", self.nid))
        self.b_variant = QToolButton()
        self.b_variant.setDefaultAction(self.act_variant)
        self.b_variant.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.b_variant.setAutoRaise(True)
        self.b_variant.setIconSize(QSize(16, 16))
        head.addWidget(self.b_variant)
        self.b_more, acts = more_menu(self, self.nid, (
            ("variants", "layers", "Variants…", "Variants…: several new branches with the same prompt and settings "
                                                "but other seeds, generated at once", None),
            ("compare", "cmp", "Compare…", "Compare: play this clip and its made variants side by side, keep one",
             None),
            ("branch", "split", "Branch", S.tip("Branch: another continuation from the same point", "branch"),
             S.keys("branch")),
            ("start_rows", "layers", "More start images…",
             "More start images…: choose several pictures; this clip is made once for each, with the same prompt, "
             "seed and settings, each on its own row (for a clip that starts the video)", None),
            ("row_apply", "clone", "Apply to its other start images",
             "Give this clip's prompt and settings to the clips made for its other start images (each keeps its "
             "picture and seed)", None),
            None,
            ("freeze_toggle", "lock", "Freeze", FREEZE_TIP, None),
            None,
            ("delete", "trash", "Delete…", "Delete this clip and everything after it (asks first)",
             S.keys("graph_delete"))))
        head.addWidget(self.b_more)
        self.more_actions = acts                 # action name -> QAction (the More menu's entries)
        for k in ("start_rows", "row_apply"):    # (shown for a clip that starts the video: set_rows)
            acts[k].setVisible(False)
        # (the old header buttons by their names: enabled / clicked as before)
        self.buttons = [self.act_variant, acts["variants"], acts["compare"], acts["branch"], acts["delete"],
                        acts["start_rows"], acts["row_apply"]]
        self.b_compare = acts["compare"]
        self._can_compare = False
        self.frozen, self._on = False, True
        self.has_lines = False                   # its prompt has a spoken line (text to speech can speak it)
        self.b_freeze = acts["freeze_toggle"]
        self.b_freeze.setCheckable(True)
        shared.v.addLayout(head)

    def _build_prompt(self, shared, node, names):
        """Build the picture, the prompt box, the faces line and the chips."""
        self.span = label("", level="tertiary")
        shared.v.addWidget(self.span)
        self.thumb = FrameTile(312, 176)
        self.thumb.setToolTip("Last frame of this clip (or its hand-over frame); click to play the clip")
        self.thumb.clicked.connect(lambda _t: self.play.emit(self.nid))
        shared.v.addWidget(self.thumb)
        self.edit = QPlainTextEdit(node.prompt)
        self.edit.setPlaceholderText("What happens in this clip? It starts from the last frame of the clip before.\n"
                                     "Type @ for a person or place; the Camera button adds a camera move.")
        self.edit.setTabChangesFocus(True)
        self.edit.setMinimumHeight(96)
        shared.v.addWidget(self.edit, 1)
        self.mentions = MentionHelper(self.edit, names)
        self.faces = label("", level="tertiary")
        self.faces.setWordWrap(True)
        self.faces.hide()
        shared.v.addWidget(self.faces)
        self.chips = ChipRow(self)                 # what the clip is made with, in model order (set_inputs)
        self.chips.leadToggled.connect(lambda on: self.leadChanged.emit(self.nid, on))
        self.chips.promptWanted.connect(lambda: self.modelPrompt.emit(self.nid))
        shared.v.addWidget(self.chips)

    def _build_prompt_tools(self, shared):
        """Build the buttons under the prompt (person, camera, example, refine)."""
        # under the prompt: put in a person or place, a camera move, an example; Refine
        foot = QHBoxLayout()
        foot.setSpacing(2)
        self.b_mention = self._text_button("@ Person or place", "Put a person or place from Cast & references into "
                                           "the description at the cursor (as @Name)")
        self.mention_menu = QMenu(self.b_mention)          # (filled when shown: the cast changes)
        self.b_mention.clicked.connect(self._mention_clicked)
        self.b_camera = self._text_button("Camera", "Put a camera move into the description at the cursor (as a "
                                          "#word; the model gets it as a sentence, e.g. “the camera pushes in at "
                                          "slow speed”)")
        self.camera_menu = QMenu(self.b_camera)
        for text, variants in camera_menu_items():
            sub = self.camera_menu.addMenu(text) if len(variants) > 1 else self.camera_menu
            for words, token in variants:
                a = sub.addAction(words)
                a.setToolTip(f"#{token}")
                a.triggered.connect(lambda _=False, t=token: self.insert_text(f"#{t}"))
        self.b_camera.setMenu(self.camera_menu)
        self.b_camera.setPopupMode(QToolButton.InstantPopup)
        self.b_examples = self._text_button("Examples", "Example descriptions: one replaces an empty description, "
                                            "else it is added at the end (Ctrl+Z takes it back)")
        self.examples_menu = QMenu(self.b_examples)
        for ex in EXAMPLES:
            a = self.examples_menu.addAction(ex if len(ex) <= 60 else ex[:57].rstrip() + " …")
            a.setToolTip(ex)
            a.triggered.connect(lambda _=False, t=ex: self.add_example(t))
        self.b_examples.setMenu(self.examples_menu)
        self.b_examples.setPopupMode(QToolButton.InstantPopup)
        for b in (self.b_mention, self.b_camera, self.b_examples):
            foot.addWidget(b)
        foot.addStretch(1)
        self.b_refine = icon_button("gen", "Refine: a language model rewrites this prompt with more detail (motion, "
                                           "camera, sound); you see the new version before it is used")
        self.b_refine.setText("Refine")
        self.b_refine.setProperty("role", "textButton")     # (an icon button's fixed 24 px would cut its name off)
        self.b_refine.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.b_refine.clicked.connect(lambda: self.action.emit("refine", self.nid))
        foot.addWidget(self.b_refine)
        shared.v.addLayout(foot)

    def _build_problems(self, shared, node):
        """Build the problems box and the seed row."""
        self.problems = ProblemBox(self)             # what blocks the clip / what went wrong, with what to do
        self.problems.wanted.connect(lambda a: self.fix.emit(a, self.nid))
        shared.v.addWidget(self.problems)
        foot = QHBoxLayout()
        self.words = label("", level="tertiary")
        foot.addWidget(self.words)
        foot.addStretch(1)
        self.veda_mark = label("⚡ Veda", level="tertiary")
        self.veda_mark.setToolTip("This clip is large enough for Veda sparse attention: it is made with it (much "
                                  "faster for long / large clips, very slightly different from full attention). "
                                  "Switch it off in the project settings: “Fast attention (Veda)”")
        self.veda_mark.hide()
        foot.addWidget(self.veda_mark)
        self.seed_label = label("Seed", level="tertiary")
        foot.addWidget(self.seed_label)
        self.seed = QSpinBox()
        self.seed.setRange(0, T.SEED_MAX)
        self.seed.setValue(node.seed)
        self.seed.setToolTip("Seed: the number the clip's randomness starts from. The same seed and settings make "
                             "the same clip again; another seed gives another version")
        foot.addWidget(self.seed)
        self.new_seed = None
        self.b_seed = icon_button("rotate", "New random seed for this clip (another version when it is made again)")
        self.b_seed.clicked.connect(self.random_seed)
        foot.addWidget(self.b_seed)
        shared.v.addLayout(foot)

        # own settings of this clip; "default" = the project's
        self.ov = dict(node.overrides or {})

    def _build_own_settings(self, shared):
        """Build the clip's own settings: length, steps, LoRAs."""
        self.steps_locked = False                 # turbo mode: the turbo steps apply to every clip
        self.settings_toggle = QToolButton()
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setAutoRaise(True)
        self.settings_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.settings_toggle.setToolTip("This clip's own length and steps (unset: the project's)")
        self.settings_toggle.toggled.connect(self._show_settings)
        shared.v.addWidget(self.settings_toggle)
        self.own = QWidget()
        grid = QFormLayout(self.own)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        self.loras = LoraStack(self)
        self.loras.setToolTip("LoRAs: add-ons that teach the model a look, a person or a motion. The LoRAs of this "
                              "clip and their strengths (none: the base model)")
        self.loras.picked.connect(lambda p: self.lora_picked.emit(self.nid, p))
        self.loras.use_prompt = self._use_example      # (the LoRA window: an example prompt into this clip's)
        self.length = QComboBox()
        self.length.setToolTip("Length of this clip")
        self.steps = QSpinBox()
        self.steps.setRange(STEPS_UNSET, 80)
        self.steps.setSpecialValueText("default")
        self.steps.setToolTip("Steps: how many times the model refines this clip. Lowest value = the project's")
        srow = QHBoxLayout()
        srow.addWidget(self.length, 1)
        self.steps_label = label("Steps", level="tertiary")
        srow.addWidget(self.steps_label)
        srow.addWidget(self.steps)
        grid.addRow("Length", srow)
        shared.v.addWidget(self.own)

    def _build_link_rows(self, shared):
        """Build the rows of carry-over, continue and camera."""
        shared.link = QFormLayout()
        shared.link.setContentsMargins(0, 2, 0, 0)
        shared.link.setHorizontalSpacing(8)
        shared.link.setVerticalSpacing(4)
        shared.link.addRow("LoRAs", self.loras)
        self.guide = QComboBox()
        for n, text in GUIDE_TEXT:
            self.guide.addItem(text, n)
        self.guide.setToolTip(GUIDE_TIP)
        shared.link.addRow("Carry-over", self.guide)
        self.cont = QComboBox()                  # continue the clip before: its end as <Video 1> (h3 format)
        self.cont.addItem("off", None)
        self.cont.setToolTip(CONTINUE_TIP)
        self._cont_choices: list[int] = []       # the lengths that fit (tree.continue_choices)
        self._cont_why = CONTINUE_FIRST_TIP      # why it cannot be used ('': it can)
        shared.link.addRow("Continue", self.cont)
        crow = QHBoxLayout()
        crow.setSpacing(6)
        self.cam_move = QComboBox()
        self.cam_move.addItem("none", None)
        for slug, (text, _w, _m) in CAM.MOVES.items():
            self.cam_move.addItem(text, slug)
        self.cam_move.setToolTip("A camera move for this clip, in the words of MiniMax H3's prompt guide: it is "
                                 "added as a sentence after the prompt (\u201cThe camera pushes in with small "
                                 "amplitude at slow speed.\u201d). For a move at a certain point, type # in the "
                                 "prompt instead (#push-in-slow \u2026)")
        self.cam_amp = QComboBox()
        for text, data in (("medium", None), ("small", "small"), ("large", "large")):
            self.cam_amp.addItem(text, data)
        self.cam_amp.setToolTip("How far the camera moves (amplitude); medium is left out of the prompt")
        self.cam_speed = QComboBox()
        for text, data in (("normal", None), ("slow", "slow"), ("fast", "fast")):
            self.cam_speed.addItem(text, data)
        self.cam_speed.setToolTip("How fast the camera moves; normal is left out of the prompt")
        crow.addWidget(self.cam_move, 2)
        crow.addWidget(self.cam_amp, 1)
        crow.addWidget(self.cam_speed, 1)
        shared.link.addRow("Camera", crow)

    def _build_sound_rows(self, shared):
        """Build the sound rows."""
        self.sound_box = QWidget()
        sb = QVBoxLayout(self.sound_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(2)
        self.src_sound = QComboBox()
        for text, data in SOUND_MODES:
            self.src_sound.addItem(text, data)
        self.src_sound.setToolTip(
            "<p><b>The model makes it</b>: the clip's sound is made with the picture.<br>"
            "<b>Continue the real video's sound</b>: the real video's own sound from right after its range (what was "
            "said next), exactly as recorded.<br><b>Repeat the sound of the clip before</b>: the clip before's sound "
            "once more (looped if this clip is longer), e.g. music or a background that goes on.<br>"
            "With either sound the picture is made to match it (lips in sync)</p>")
        self.src_sound_info = label("", level="tertiary")
        self.src_sound_info.setWordWrap(True)
        self.sound_over = QCheckBox("Lay the sound over (full motion, no lip sync)")
        self.sound_over.setToolTip(
            "<p><b>On</b>: the clip is made with the normal image-to-video model, as without a sound (it moves "
            "freely), and the exact sound is put into it afterwards. Nothing in the picture follows the sound: good "
            "for music, a background or a voice-over.<br><b>Off</b>: the reference model hears the sound and makes "
            "the picture to it (lips in sync), which often keeps the body still.</p>")
        self.sound_mix = QCheckBox("Keep the clip's own sound too (mix both)")
        self.sound_mix.setToolTip(MIX_TIP)
        self.mix_level = QSpinBox()
        self.mix_level.setRange(*chain.MIX_LEVELS)
        self.mix_level.setSuffix(" dB")
        self.mix_level.setToolTip("How loud the given sound is in the mix: 0 dB is as recorded, below 0 quieter "
                                  "(e.g. music behind the clip's voices), above 0 louder. The clip's own sound "
                                  "stays as the model made it.")
        mrow = QHBoxLayout()
        mrow.setContentsMargins(18, 0, 0, 0)                    # (indented: a choice of the box above it)
        mrow.setSpacing(6)
        mrow.addWidget(self.sound_mix, 1)
        lrow = QHBoxLayout()
        lrow.setSpacing(4)
        lrow.addWidget(label("Level", level="secondary"))
        lrow.addWidget(self.mix_level)
        self.mix_level_box = row_widget(lrow)
        mrow.addWidget(self.mix_level_box)
        self.mix_row = QWidget()
        self.mix_row.setLayout(mrow)
        self._model_sound = True                                # the project's sound is on (the model makes one)
        sb.addWidget(self.src_sound)
        self.b_speak = QPushButton("Speak the lines (text to speech)")
        self.b_speak.setToolTip(SPEAK_TIP)
        self.b_speak.clicked.connect(lambda: self.action.emit("speak", self.nid))
        sb.addWidget(self.b_speak)
        sb.addWidget(self.sound_over)
        sb.addWidget(self.mix_row)
        sb.addWidget(self.src_sound_info)
        shared.link.addRow("Sound", self.sound_box)
        self._link = shared.link
        shared.link.setRowVisible(self.sound_box, True)           # (the sound choice in it only when there is one)
        self._max_guide = 0

    def _build_end_rows(self, shared):
        """Build the end image and keyframe rows."""
        erow = QHBoxLayout()
        erow.setSpacing(6)
        self.end_thumb = QLabel()
        self.end_thumb.setFixedSize(48, 27)
        self.end_thumb.setScaledContents(True)
        self.end_text = label("", level="secondary")
        self.b_end = icon_button("target", END_TIP)
        self.b_end.clicked.connect(lambda: self.action.emit("end_image", self.nid))
        self.b_end_clear = icon_button("x", "No end image: the clip ends freely")
        self.b_end_clear.clicked.connect(lambda: self.action.emit("end_clear", self.nid))
        self.b_end_ai = icon_button("brush", "Make an end image with AI: edit this clip's frame (or any image) by a "
                                             "prompt, e.g. “she now sits on the sofa”")
        self.b_end_ai.clicked.connect(lambda: self.action.emit("end_ai", self.nid))
        self.b_end_video = icon_button("frame", "End image from a video: choose a project or any video, then the "
                                                "frame of it to end at")
        self.b_end_video.clicked.connect(lambda: self.action.emit("end_video", self.nid))
        erow.addWidget(self.end_thumb)
        erow.addWidget(self.end_text, 1)
        erow.addWidget(self.b_end)
        erow.addWidget(self.b_end_video)
        erow.addWidget(self.b_end_ai)
        erow.addWidget(self.b_end_clear)
        shared.link.addRow("End", erow)
        for w in (self.end_text, self.end_thumb):          # (a picture dropped here is the end image)
            FileDrop(w, lambda files: self.end_dropped.emit(self.nid, files[0]),
                     lambda: self._end_free and not self.edit.isReadOnly())
        kbox = QVBoxLayout()
        kbox.setSpacing(2)
        krow = QHBoxLayout()
        krow.setSpacing(6)
        self.keys = KeyframeStrip()
        self.keys.addAt.connect(self._add_key_menu)
        self.keys.moved.connect(lambda i, at: self.keyframe.emit("move", self.nid, {"index": i, "at": at}))
        self.keys.wanted.connect(lambda act, i: self.keyframe.emit(act, self.nid, {"index": i}))
        self.b_key = icon_button("plus", "Add a middle keyframe: a picture the clip passes through at a moment "
                                         "(or click the strip where it should be)")
        self.b_key.clicked.connect(lambda: self._add_key_menu(None))
        krow.addWidget(self.keys, 1)
        krow.addWidget(self.b_key)
        kbox.addLayout(krow)
        self.key_warn = label("", level="secondary")
        self.key_warn.setWordWrap(True)
        self.key_warn.hide()
        kbox.addWidget(self.key_warn)
        self.keys_row = row_widget(kbox)
        shared.link.addRow("Keyframes", self.keys_row)
        hrow = QHBoxLayout()
        hrow.setSpacing(6)
        self.hand_text = label("last frame", level="secondary")
        self.b_hand = icon_button("keyf", "Hand over at an earlier frame: the clips after this one start from the "
                                          "frame you choose; the rest of this clip is left out")
        self.b_hand.clicked.connect(lambda: self.action.emit("handover", self.nid))
        self.b_hand_clear = icon_button("x", "Hand over at the last frame again")
        self.b_hand_clear.clicked.connect(lambda: self.action.emit("handover_clear", self.nid))
        hrow.addWidget(self.hand_text, 1)
        hrow.addWidget(self.b_hand)
        hrow.addWidget(self.b_hand_clear)
        self.hand_row = row_widget(hrow)
        shared.link.addRow("Hand-over", self.hand_row)
        self._can_hand, self._has_hand = False, False
        self._end_free = True                    # no video after this clip: its own end image can be set
        self._has_end = False
        self._guide_useful = True                # there is a clip before it or a video after it
        shared.v.addLayout(shared.link)

    def _build_footer(self, shared, view):
        """Build the Simple view, the status rows and the signal wiring."""
        # Simple view: the expert parts of the card are hidden (their values still apply: the note says so)
        self.view = view if view is not None else ViewMode(False, self)
        self.adv_note = AdvancedNote(self.view)
        shared.v.addWidget(self.adv_note)
        self.view.advanced(self.seed_label, self.seed, self.b_seed, self.veda_mark, self.steps_label, self.steps,
                           (shared.link, self.loras), (shared.link, self.guide), (shared.link,
                                                                                  self.cont), (shared.link,
                                                                                               self.keys_row),
                           (shared.link, self.hand_row), self.mix_level_box)
        self._show_mix()
        self._show_settings(PromptCard.settings_open)
        self.guide.currentIndexChanged.connect(self._ov_changed)
        self.cont.currentIndexChanged.connect(self._ov_changed)
        for w_ in (self.cam_move, self.cam_amp, self.cam_speed):
            w_.currentIndexChanged.connect(self._ov_changed)
        self.src_sound.currentIndexChanged.connect(self._sound_toggled)
        self.sound_over.toggled.connect(self._sound_toggled)
        self.sound_mix.toggled.connect(self._sound_toggled)
        self.mix_level.valueChanged.connect(self._ov_changed)
        self.loras.changed.connect(self._ov_changed)
        self.length.currentIndexChanged.connect(self._ov_changed)
        self.steps.valueChanged.connect(self._ov_changed)

        strow = QHBoxLayout()
        self.step = label("", level="tertiary")
        self.step.setWordWrap(True)
        self.b_revert = QPushButton("Undo changes")
        self.b_revert.clicked.connect(lambda: self.action.emit("revert", self.nid))
        self.b_revert.hide()
        strow.addWidget(self.step, 1)
        strow.addWidget(self.b_revert, 0, Qt.AlignTop)
        shared.v.addLayout(strow)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.hide()
        shared.v.addWidget(self.bar)
        self.edit.textChanged.connect(self._text)
        self.seed.valueChanged.connect(lambda s: self.seedChanged.emit(self.nid, s))
        self.guide.currentIndexChanged.connect(lambda *_: self._enable_guide())
        self.cont.currentIndexChanged.connect(lambda *_: self._enable_cont())
        guard_wheel(self)
        self._words()
        self._restyle()

    def set_choices(self, project_frames: int, project_steps: int, known: list[tuple[str, str]]):
        """Fill the setting widgets (without emitting changes). known: (path, label) of the LoRAs to offer."""
        widgets = (self.loras, self.length, self.steps, self.guide, self.src_sound, self.sound_over, self.cam_move,
                   self.cam_amp, self.cam_speed, self.sound_mix, self.mix_level, self.cont)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.loras.set_items(known, self.ov.get("loras") or [])
            self.length.clear()
            self.length.addItem(f"default ({project_frames / chain.FPS:.1f} s)", None)
            for n in FRAME_CHOICES:
                self.length.addItem(frames_text(n), n)
            select_frames(self.length, self.ov.get("frames"))
            self.steps.setValue(int(self.ov.get("steps", STEPS_UNSET)))
            while self.guide.count() > len(GUIDE_TEXT):         # (an odd value shown earlier is not kept)
                self.guide.removeItem(self.guide.count() - 1)
            g = self.guide.findData(self.ov.get("guide"))
            if g < 0:
                self.guide.addItem(f"{self.ov['guide']} frames", self.ov["guide"])
                g = self.guide.count() - 1
            self.guide.setCurrentIndex(g)
            self._fill_cont()
            k = self.src_sound.findData(self.ov.get("source_sound") or None)
            self.src_sound.setCurrentIndex(max(k, 0))
            self.sound_over.setChecked(bool(self.ov.get("sound_over")))
            self.sound_mix.setChecked(self.ov.get("sound_over") == "mix")
            self.mix_level.setValue(round(float(self.ov.get("sound_mix_level") or 0)))
            self._show_mix()
            cam = CAM.clean(self.ov.get("camera")) or {}
            self.cam_move.setCurrentIndex(max(0, self.cam_move.findData(cam.get("move"))))
            self.cam_amp.setCurrentIndex(max(0, self.cam_amp.findData(cam.get("amplitude"))))
            self.cam_speed.setCurrentIndex(max(0, self.cam_speed.findData(cam.get("speed"))))
            self._enable_camera()
            self._enable_guide()
            self.steps.setToolTip(f"Sampling steps of this clip (lowest value = the project's: {project_steps})")
            self.steps.setEnabled(not self.edit.isReadOnly() and not self.steps_locked)
            if self.steps_locked:
                self.steps.setToolTip("Turbo / HyperFlow / Pika H3 Flex: every clip uses their own steps (Clips → "
                                      "Speed)")
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.loras.setEnabled(not self.edit.isReadOnly())
        self._update_note()

    def _ov_changed(self, *_):
        ov = {}
        loras = self.loras.items()
        if loras:
            ov["loras"] = loras
        if self.length.currentData() is not None:
            ov["frames"] = self.length.currentData()
        if self.steps.value() > STEPS_UNSET:
            ov["steps"] = self.steps.value()
        if self.guide.currentData():
            ov["guide"] = self.guide.currentData()
        if self.cont.currentData():
            ov["continue"] = self.cont.currentData()
        if self.src_sound.currentData():
            ov["source_sound"] = self.src_sound.currentData()
            if self.sound_over.isChecked():
                ov["sound_over"] = "mix" if self.sound_mix.isChecked() else True
                if self.sound_mix.isChecked() and self.mix_level.value():
                    ov["sound_mix_level"] = self.mix_level.value()
        cam = CAM.clean({"move": self.cam_move.currentData(), "amplitude": self.cam_amp.currentData(),
                         "speed": self.cam_speed.currentData()})
        if cam:
            ov["camera"] = cam
        if self.ov.get("lead") is False:          # (the lead chip's switch, not a widget here)
            ov["lead"] = False
        self._enable_camera()
        self.ov = ov
        self._update_note()
        self.overridesChanged.emit(self.nid, ov)

    def _enable_camera(self):
        """Amplitude and speed only for a move that takes them (not for none, static, shake, POV)."""
        move = self.cam_move.currentData()
        on = bool(move) and CAM.MOVES[move][2] and not self.edit.isReadOnly()
        self.cam_amp.setEnabled(on)
        self.cam_speed.setEnabled(on)
        self.cam_move.setEnabled(not self.edit.isReadOnly())

    def text(self) -> str:
        return " ".join(self.edit.toPlainText().split())

    @staticmethod
    def _text_button(text: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setProperty("role", "textButton")
        b.setText(text)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _fill_mentions(self) -> list[str]:
        """The @ menu: the cast's names now."""
        self.mention_menu.clear()
        names = list(self.names())
        for name in names:
            act = self.mention_menu.addAction(f"@{name}")
            act.triggered.connect(lambda _=False, n=name: self.insert_text(f"@{n}"))
        return names

    def _mention_clicked(self):
        """@ Person or place: the cast's names to put in; without any, Cast & references opens to add one."""
        if not self._fill_mentions():
            self.fix.emit("cast_new", self.nid)
            return
        self.mention_menu.popup(self.b_mention.mapToGlobal(self.b_mention.rect().bottomLeft()))

    def insert_text(self, text: str):
        """Put a @Name or #camera token in at the cursor (one step Ctrl+Z takes back)."""
        if not self.edit.isReadOnly():
            insert_at_cursor(self.edit, text)

    def add_example(self, text: str):
        if not self.edit.isReadOnly():
            add_example(self.edit, text)

    def set_problems(self, items: list[dict]):
        """What blocks this clip or went wrong when it was made, each with what to do (ProblemBox)."""
        self.problems.set_items(items)

    def random_seed(self):
        """A new random seed in the field (new_seed(): one no other clip has, else any)."""
        self.seed.setValue(self.new_seed() if self.new_seed is not None else random.randint(1, T.SEED_MAX))

    def _text(self):
        self._words()
        self._enable_speak()
        self.edited.emit(self.nid)

    def _enable_speak(self):
        """"Speak the lines" is always on the card; it can be used once the prompt has a line in quotes."""
        self.has_lines = bool(T.spoken_lines(self.text()))
        self.b_speak.setEnabled(not self.edit.isReadOnly() and self.has_lines)
        self.b_speak.setToolTip(SPEAK_TIP if self.has_lines else
                                "Write what is said in quotes first, e.g. @Anna says \"Hello, nice to see you.\" "
                                "Then this speaks it with text to speech and makes it the clip's sound.")

    def _words(self):
        n = len(self.text().split())
        self.words.setText(f"{n} words" if n else "empty prompt")

    def set_progress(self, i: int, n: int, text: str | None = None):
        if n:
            self.bar.setValue(round(100 * i / n))
            self.step.setText(text if text is not None else f"step {i} / {n}")

    def _use_example(self, text: str):
        """An example prompt from the LoRA window replaces the description (one step Ctrl+Z takes back)."""
        if not self.edit.isReadOnly():
            replace_text(self.edit, text)

    def set_editable(self, on: bool):
        """on: nothing runs (the card can be used); a frozen clip's own inputs stay locked anyway."""
        self._on = bool(on)
        edit = self._on and not self.frozen
        self.edit.setReadOnly(not edit)
        for w in (self.seed, self.b_seed, self.length, self.steps, self.loras, self.guide, self.src_sound,
                  self.sound_over, self.b_refine, self.b_speak, self.b_mention, self.b_camera, self.b_examples):
            w.setEnabled(edit)
        self._enable_speak()
        self._show_mix()
        for b in self.chips.chips:                 # (the lead chip is an own setting of the clip)
            if b.isCheckable():
                b.setEnabled(edit)
        for w in self.buttons:
            w.setEnabled(self._on)
        self.steps.setEnabled(edit and not self.steps_locked)
        self._enable_camera()
        self._enable_guide()
        self._enable_cont()
        self._enable_end()
        self._enable_hand()
        self.b_compare.setEnabled(self._on and self._can_compare)
        self.b_revert.setEnabled(edit)
        self.b_freeze.setEnabled(self._on and (self.frozen or self.state == "done"))

    def _frozen_tip(self, frozen: bool) -> str:
        return FREEZE_TIP if not frozen else (
            "Frozen: kept as made, whatever the project's settings say. Click to unfreeze it "
            "(and the frozen clips after it): it can be edited and follows the settings again")

    def _show_settings(self, on: bool):
        PromptCard.settings_open = bool(on)
        self.own.setVisible(bool(on))
        self.settings_toggle.blockSignals(True)
        self.settings_toggle.setChecked(bool(on))
        self.settings_toggle.blockSignals(False)
        self.settings_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def set_faces(self, text: str):
        self.faces.setText(text)
        self.faces.setVisible(bool(text))

    def set_inputs(self, specs: list[dict] | None):
        """The chips under the prompt (clip_inputs.chip_specs; None: none)."""
        self.chips.set_specs(specs)
        for b in self.chips.chips:
            if b.isCheckable():
                b.setEnabled(not self.edit.isReadOnly())
        self.view.advanced(self.chips.model_chip, self.chips.b_prompt)    # (new chips follow the view too)

    def set_veda(self, on: bool):
        self.view.show(self.veda_mark, bool(on))

    def advanced_active(self) -> list[str]:
        """The clip's own settings that are set (not a clip's defaults: no overrides, no keyframes, hand-over at
        the last frame) and whose control the Simple view hides. The seed is not one: every clip has one."""
        ov, names = self.ov, []
        for key, name, target in (("loras", "LoRAs", (self._link, self.loras)),
                                  ("guide", "Carry-over", (self._link, self.guide)),
                                  ("continue", "Continue", (self._link, self.cont)), ("steps", "Steps", self.steps)):
            if ov.get(key) and self.view.is_advanced(target):
                names.append(name)
        if self.keys.items and self.view.is_advanced((self._link, self.keys_row)):
            names.append("Keyframes")
        if self._has_hand and self.view.is_advanced((self._link, self.hand_row)):
            names.append("Hand-over")
        if ov.get("sound_over") == "mix" and ov.get("sound_mix_level") and self.view.is_advanced(self.mix_level_box):
            names.append("Sound level")
        return names

    def _update_note(self):
        self.adv_note.set_active(self.advanced_active())

    def set_summary(self, text: str):
        """The own settings in a few words, shown on the (closed) settings toggle."""
        self.settings_toggle.setText("Own settings: " + (text or "project defaults"))

    def set_revert(self, edited: list[str]):
        """Offer Undo changes when the prompt / seed / settings / end image differ from the made clip."""
        self.b_revert.setVisible(bool(edited))
        self.b_revert.setEnabled(not self.edit.isReadOnly())
        if edited:
            self.b_revert.setToolTip(f"Back to the {' and '.join(edited)} this clip was made with, so it counts as "
                                     "done again and is not made again")

    def set_handover(self, text: str, can: bool, has: bool, tip: str = ""):
        """Where the clips after this one start: 'last frame' or an earlier frame. can: it can be chosen (made,
        no video after it); has: an earlier frame is set."""
        self.hand_text.setText(text)
        self.hand_text.setToolTip(tip)
        self._can_hand, self._has_hand = bool(can), bool(has)
        self._enable_hand()
        self._update_note()

    def _enable_hand(self):
        on = not self.edit.isReadOnly()
        self.b_hand.setEnabled(on and self._can_hand)
        self.b_hand_clear.setEnabled(on and self._has_hand)

    def set_rows(self, first: bool, others: int):
        """first: the clip starts the video (it can have several start images); others: the clips made for its
        other start images."""
        self.more_actions["start_rows"].setVisible(first)
        self.more_actions["row_apply"].setVisible(first and others > 0)

    def set_compare(self, n: int):
        """n: made clips next to this one (its variants / branches) that can be compared with it."""
        self._can_compare = n > 0
        self.b_compare.setEnabled(self._can_compare and self._on)
        self.b_compare.setToolTip(f"Compare: play this clip and its {plural(n, 'made variant')} side by "
                                  "side, keep one" if n else
                                  "Compare: needs this clip and at least one other made clip from the same point "
                                  "(Variant / Variants…)")

    def set_guide_useful(self, useful: bool):
        """Carry-over needs a clip before this one or a real video after it; a first clip without a video after it
        has nothing to carry over."""
        self._guide_useful = bool(useful)
        self._enable_guide()

    def set_continue(self, choices: list[int], why: str = ""):
        """The continuation lengths that fit this clip (tree.continue_choices) and why it cannot continue the clip
        before ('': it can; else in words: the classic format, a first clip). A set value stays choosable, so it can
        be taken back."""
        self._cont_choices = list(choices)
        self._cont_why = why or ("" if choices else CONTINUE_SHORT_TIP)
        self._fill_cont()
        self._enable_cont()

    def _fill_cont(self):
        """The Continue choices: off, the lengths that fit, and the clip's own value if it does not fit (any more)."""
        cur = (getattr(self, "ov", None) or {}).get("continue") or None
        fits = [] if self._cont_why else self._cont_choices
        want = [None] + fits + ([cur] if cur and cur not in fits else [])
        if [self.cont.itemData(k) for k in range(self.cont.count())] == want and self.cont.currentData() == cur:
            return
        self.cont.blockSignals(True)
        try:
            self.cont.clear()
            for c in want:
                self.cont.addItem("off" if c is None else continue_text(c) + ("" if c in fits or self._cont_why
                                                                              else " (does not fit)"),
                                  c)
            self.cont.setCurrentIndex(max(0, self.cont.findData(cur)))
        finally:
            self.cont.blockSignals(False)

    def _enable_cont(self):
        on = not self.edit.isReadOnly()
        has = bool(self.cont.currentData())
        self.cont.setEnabled(on and (not self._cont_why or has))   # a set value can always be taken back
        self.cont.setToolTip(CONTINUE_TIP if not self._cont_why else
                             self._cont_why + (" It is set: choose 'off' to take it back." if has else ""))

    def _sound_toggled(self, *_):
        """With the video's sound the picture should continue too: no carry-over yet -> the longest one that fits
        (the clip before must be that long, and this clip longer)."""
        if not self.src_sound.currentData() and self.sound_over.isChecked():
            self.sound_over.blockSignals(True)         # (no sound: nothing to lay over; off when one is chosen again)
            self.sound_over.setChecked(False)
            self.sound_over.blockSignals(False)
        if not self.sound_over.isChecked() and self.sound_mix.isChecked():
            self.sound_mix.blockSignals(True)          # (not laid over: nothing to mix it into)
            self.sound_mix.setChecked(False)
            self.sound_mix.blockSignals(False)
        self._show_mix()
        if self.src_sound.currentData() is True and not self.sound_over.isChecked() \
                and not self.guide.currentData() and self._max_guide:
            k = self.guide.findData(self._max_guide)
            if k >= 0:
                self.guide.blockSignals(True)
                self.guide.setCurrentIndex(k)
                self.guide.blockSignals(False)
                self._enable_guide()
        self._ov_changed()

    def set_tracks(self, tracks: list[tuple[str, str]]):
        """The project's audio tracks as choices of the Sound dropdown: (id, name)."""
        names = [name for _tid, name in tracks]
        want = [(f"Audio track: {name}" + (f" ({names[:k].count(name) + 1})" if names.count(name) > 1 else ""),
                 T.TRACK_PREFIX + tid) for k, (tid, name) in enumerate(tracks)]
        have = [(self.src_sound.itemText(k), self.src_sound.itemData(k)) for k in range(len(SOUND_MODES),
                                                                                       self.src_sound.count())]
        cur = (getattr(self, "ov", None) or {}).get("source_sound") or None     # (the clip's own setting)
        if isinstance(cur, str) and cur.startswith(T.TRACK_PREFIX) and cur not in [d for _t, d in want]:
            want.append(("Audio track: (removed)", cur))   # kept, so the setting shows and can be taken back
        if want == have:
            if self.src_sound.currentData() != cur and self.src_sound.findData(cur) >= 0:
                self.src_sound.blockSignals(True)
                self.src_sound.setCurrentIndex(self.src_sound.findData(cur))
                self.src_sound.blockSignals(False)
            return
        self.src_sound.blockSignals(True)
        try:
            while self.src_sound.count() > len(SOUND_MODES):
                self.src_sound.removeItem(len(SOUND_MODES))
            for text, data in want:
                self.src_sound.addItem(text, data)
            self.src_sound.setCurrentIndex(max(0, self.src_sound.findData(cur)))
        finally:
            self.src_sound.blockSignals(False)

    def set_source_sound(self, available: bool, info: str, max_guide: int = 0, repeat: bool = False,
                         model_sound: bool = True):
        """The 'Sound' row: shown for a clip after another clip (or when it is set). available: a real video comes
        before it (its sound can be continued); repeat: there is a clip before it (its sound can be repeated); info:
        where the sound comes from; max_guide: the longest carry-over that fits (set when the video's sound is
        switched on); model_sound: the project's sound is on (a sound laid over can be mixed with the model's)."""
        self._max_guide = max_guide
        cur = self.src_sound.currentData()
        model = self.src_sound.model()
        for k in range(self.src_sound.count()):        # a choice in use stays choosable (to be taken back)
            d = self.src_sound.itemData(k)
            track = isinstance(d, str) and d.startswith(T.TRACK_PREFIX)
            ok = d is None or d == cur or track or (available if d is True else repeat)
            model.item(k).setEnabled(ok)
        some = available or repeat or bool(cur) or self.src_sound.count() > len(SOUND_MODES)
        self._link.setRowVisible(self.sound_box, True)   # (always: Speak the lines)
        self.src_sound.setVisible(some)
        self.src_sound.setEnabled(not self.edit.isReadOnly() and some)
        self._enable_speak()
        self.src_sound_info.setText(info if cur else "")
        self.src_sound_info.setVisible(bool(cur) and bool(info))
        self.sound_over.setVisible(bool(cur))
        self.sound_over.setEnabled(not self.edit.isReadOnly())
        self._model_sound = bool(model_sound)
        self._show_mix()

    def _show_mix(self):
        """The mix choice only under a checked 'Lay the sound over', its level only in a mix (and in Advanced); with
        the project's sound off the model makes nothing to mix with, so it cannot be chosen (one set stays
        visible, so it can be taken back)."""
        over = self.sound_over.isChecked() and bool(self.src_sound.currentData())
        mix = over and self.sound_mix.isChecked()
        self.mix_row.setVisible(over)
        self.view.show(self.mix_level_box, mix)
        edit = not self.edit.isReadOnly()
        self.sound_mix.setEnabled(edit and (self._model_sound or mix))
        self.sound_mix.setToolTip(MIX_TIP if self._model_sound else MIX_OFF_TIP)
        self.mix_level.setEnabled(edit)

    def _enable_guide(self):
        on = not self.edit.isReadOnly()
        has = bool(self.guide.currentData())
        self.guide.setEnabled(on and (self._guide_useful or has))   # a set value can always be taken back
        self.guide.setToolTip(GUIDE_TIP if self._guide_useful else
                              "The first clip has no clip before it (and no real video after it): there is "
                              "nothing to carry over" + (" — this setting has no effect; set it back to "
                                                         "'last frame only'" if has else ""))

    def _add_key_menu(self, at: float | None):
        """Where the new keyframe's picture comes from (at: the moment clicked; None: the biggest free gap)."""
        if at is None:
            times = [0.0] + [it["at"] for it in self.keys.items] + [self.keys.duration]
            gap = max(range(len(times) - 1), key=lambda i: times[i + 1] - times[i])
            at = round((times[gap] + times[gap + 1]) / 2, 2)
        m = QMenu(self)
        acts = {m.addAction(f"A picture at {at:.2f} s…"): "add", m.addAction("A frame of a video…"): "add_video"}
        if self.keys.made:
            acts[m.addAction("This clip's frame at that moment (keep it, re-roll the rest)…")] = "add_moment"
        a = self._exec_menu(m)
        what = acts.get(a)
        m.deleteLater()                             # (a child of the card: it would stay until the card goes)
        if what:
            self.keyframe.emit(what, self.nid, {"at": at})

    def _exec_menu(self, m):
        return m.exec(QCursor.pos())

    def set_keyframes(self, duration: float, items: list[dict], cuts: list[float], made: bool):
        self.keys.set_data(duration, items, cuts, made)
        if cuts:
            self.key_warn.setText("⚠ The clip cuts at " + ", ".join(f"{c:.2f} s" for c in cuts) +
                                  ": a keyframe (or the end image) is probably too far from what comes before it "
                                  "— move it later, or use a picture closer to the scene.")
        self.key_warn.setVisible(bool(cuts))
        self._update_note()

    def _enable_end(self):
        on = not self.edit.isReadOnly()
        self.keys.setEnabled(on)
        self.b_key.setEnabled(on)
        self.b_end.setEnabled(on and self._end_free)
        self.b_end_video.setEnabled(on and self._end_free)
        self.b_end_ai.setEnabled(on and self._end_free)
        self.b_end_clear.setEnabled(on and self._has_end)

    def set_end(self, image: Path | None, video_after: str | None, pixmap: QPixmap | None,
                present: bool | None = None):
        """How this clip ends: at the video after it, at its own end image, or freely. present: whether the end
        image file exists (the window knows it from its one look at the file; None: this looks itself)."""
        self._end_free, self._has_end = video_after is None, image is not None
        self.b_end.setToolTip(END_TIP if video_after is None else
                              f"This clip ends at the first frame of video #{T.short_id(video_after)}; remove that "
                              "video to give it an end image of its own")
        if video_after is not None:
            self.end_text.setText(f"first frame of video #{T.short_id(video_after)}"
                                  + (" (own end image not used)" if image is not None else ""))
        elif image is not None:
            if present is None:
                present = image.is_file()
            self.end_text.setText("own end image" if present else "end image missing")
        else:
            self.end_text.setText("free")
        self.end_text.setToolTip(str(image) if image is not None else "")
        if pixmap is not None and not pixmap.isNull():
            self.end_thumb.setPixmap(pixmap)
            self.end_thumb.show()
        else:
            self.end_thumb.clear()
            self.end_thumb.hide()
        self._enable_end()


class VideoCard(_ClipCardBase):
    """A real video in the selected path: its file and range, status; edit the range, play, delete."""
    _NO_PICTURE = "not converted yet"
    _BUSY = ("generating", "converting")
    _BACKGROUND = "bgSunken"
    action = Signal(str, str)                  # action, node id
    play = Signal(str)

    def __init__(self, node: T.Node, parent=None, names=lambda: [], view: ViewMode | None = None):
        super().__init__(parent)
        self.nid = node.id
        self.view = view
        self.setObjectName("promptCard")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setFixedWidth(CARD_WIDTH)
        self.state = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 8, 10)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = label("Video", type_="heading")
        self.sub = label(f"#{T.short_id(node.id)}", level="tertiary")
        self.tag = label("", role="statusTag")
        head.addWidget(self.title)
        head.addWidget(self.sub)
        head.addWidget(self.tag)
        head.addStretch(1)
        self.b_more, acts = more_menu(self, self.nid, (
            ("edit", "pen", "Edit range…", "Edit range… (or choose another file)", None),
            ("branch", "split", "Branch", S.tip("Branch: another continuation from the same point", "branch"),
             S.keys("branch")),
            None,
            ("freeze_toggle", "lock", "Freeze", FREEZE_TIP.replace("this clip", "this video"), None),
            None,
            ("delete", "trash", "Delete…", "Delete this video and everything after it (asks first)",
             S.keys("graph_delete"))),
            tip="More actions for this video: edit its range, branch, freeze, delete")
        head.addWidget(self.b_more)
        self.more_actions = acts
        self.buttons = [acts["edit"], acts["branch"], acts["delete"]]    # (by their old names, as the buttons were)
        self.frozen, self._on = False, True
        self.b_freeze = acts["freeze_toggle"]
        self.b_freeze.setCheckable(True)
        v.addLayout(head)
        self.span = label("", level="tertiary")
        v.addWidget(self.span)
        self.thumb = FrameTile(312, 176)
        self.thumb.setToolTip("Last frame of the range; click to play the converted clip")
        self.thumb.clicked.connect(lambda _t: self.play.emit(self.nid))
        v.addWidget(self.thumb)
        self.file = label("", level="secondary")
        self.file.setWordWrap(True)
        v.addWidget(self.file)
        self.range = label("", level="tertiary")
        self.range.setWordWrap(True)
        v.addWidget(self.range)
        self.b_edit = QPushButton("Edit range…")
        self.b_edit.clicked.connect(lambda: self.action.emit("edit", self.nid))
        v.addWidget(self.b_edit, 0, Qt.AlignLeft)
        v.addStretch(1)
        self.step = label("", level="tertiary")
        v.addWidget(self.step)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        v.addWidget(self.bar)
        self.set_source(node)
        self._restyle()

    def set_source(self, node: T.Node, present: bool | None = None):
        """present: whether the video file exists (the window's one look at it; None: this looks itself)."""
        src = node.source or {}
        f = Path(src.get("file", ""))
        if present is None:
            present = f.is_file()
        self.file.setText(f.name + ("" if present else "  (file not found)"))
        self.file.setToolTip(str(f))
        n = chain.range_frames(src.get("start", 0), src.get("end", 0))
        rot = T.source_rotation(src)
        self.range.setText(f"{format_time(src.get('start', 0), 1)} – {format_time(src.get('end', 0), 1)} of the file · "
                           f"{n} frames at {chain.FPS} fps" + (f" · turned {rot}° right" if rot else ""))

    def set_editable(self, on: bool):
        self._on = bool(on)
        for w in self.buttons:
            w.setEnabled(self._on)
        self.b_edit.setEnabled(self._on and not self.frozen)
        self.buttons[0].setEnabled(self._on and not self.frozen)       # (edit range)
        self.b_freeze.setEnabled(self._on and (self.frozen or self.state == "done"))

    def _frozen_tip(self, frozen: bool) -> str:
        return FREEZE_TIP.replace("this clip", "this video") if not frozen else (
            "Frozen: kept as converted, whatever the project's size says. Click to unfreeze it")
