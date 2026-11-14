"""Simple / Advanced view, the guide, the engine banner, the menu and the panel layouts."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit

from .. import tree as T
from . import shortcuts as S
from .clip_cards import PromptCard
from .ui_logic import COMFY_START_TIMEOUT, DEFAULT_COMFY_LAUNCHER, LAYOUTS, LAYOUT_VERSION, project_setting_changes


class ViewMixin:
    """Simple / Advanced view, the guide, the engine banner, the menu and the panel layouts."""

    # ------------------------------------------------------------------ Simple / Advanced view
    def _register_advanced(self):
        """The controls the Simple view hides (the clip cards register their own). The question for each: does a
        beginner need it to make and continue a clip with a start image, people / places and sound?"""
        c, a = self._clips_form, self._comfy_form
        self.advanced_only(
            # project: speed and quality knobs, prompt wording, path joining
            (c, self.steps), (c, self.speed), (c, self._cfg_row), (c, self.veda), (c, self.turbo_lora),
            (c, self._turbo_row), (c, self.keep_rep), (c, self.prompt_format), (c, self.style_prefix),
            (c, self.style_suffix), (c, self.lora_triggers),
            # cast: how the pictures are sent
            (self._tts_form, self.ref_size), self.ref_hint, (self._tts_form, self._eleven_row),
            # preferences: ComfyUI's technical part (the address, start command and auto start stay), RunPod
            (a, self._comfy_dir_row), (a, self.comfy_cleanup), (a, self.triggers_online), (a, self.civitai_site),
            (a, self._civitai_row),               # (the prompt refiner's language model setup stays: UX-4)
            (a, self._keep_row), *self._runpod_section,
            # the GPU's memory (the log stays: Details on a failed clip and bug reports lead there)
            self.gpu)

    def advanced_only(self, *targets):
        """Register controls the Simple view hides: a widget, (QFormLayout, field widget) for a form row, (QTabWidget,
        page) for a tab. Use self.view.show(target, on) for their own visibility afterwards (not setVisible)."""
        self.view.advanced(*targets)

    @property
    def simple(self) -> bool:
        return self.view.simple

    def set_simple(self, on: bool):
        """Switch the view (stored in the app's settings). It only shows / hides controls, never changes a value."""
        self.qs.setValue("viewMode", "simple" if on else "advanced")
        self.view.set_simple(on)
        self._show_view()

    def _view_switched(self, simple: bool):
        self._show_view()
        self.viewChanged.emit(simple)

    def _show_view(self):
        """The switch and the View menu show the current view."""
        simple = self.view.simple
        for w, on in ((self.b_simple, simple), (self.b_advanced, not simple),
                      (self.act_simple, simple), (self.act_advanced, not simple)):
            w.setChecked(on)

    def _preview_failed(self, text: str):
        self.status.setText(text)
        self.log.appendPlainText(text)

    def show_log(self):
        """View → Log: shown in the Simple view too when asked for (until the view is switched)."""
        self.view.reveal((self.tabs, self.log))
        self.tabs.setCurrentWidget(self.log)

    # ------------------------------------------------------------------ guided start
    def _guide_store(self) -> dict:
        try:
            d = json.loads(self.qs.value("guideChoices", "{}", type=str) or "{}")
        except ValueError:
            d = {}
        return d if isinstance(d, dict) else {}

    def _guide_choices(self) -> set[str]:
        """What the guide was told for this project ("text_only", "no_cast"); kept in the app's settings, not in
        the project (it changes nothing the clips are made with)."""
        if self.project is None:
            return set()
        d = self._guide_store()
        return set(d.get(self._guide_key(), [])) | set(d.get(str(self.project), []))    # (+ the key used before)

    def _guide_key(self) -> str:
        return str(self.project.resolve())      # (the same folder spelled another way is the same project)

    def _guide_mark(self, what: str):
        if self.project is None:
            return
        d, key = self._guide_store(), self._guide_key()
        d[key] = sorted(set(d.pop(key, [])) | set(d.pop(str(self.project), [])) | {what})
        self.qs.setValue("guideChoices", json.dumps(dict(list(d.items())[-50:])))     # (the last 50 projects)
        self._update_guide()

    def _guide_text_only(self):
        """Step 1 without a picture: the first clip is made from its description; its picture format is the Aspect
        setting."""
        if self.project is None:
            return
        self._guide_mark("text_only")
        self.inspector.setCurrentWidget(self.proj_scroll)
        QTimer.singleShot(0, lambda: self.proj_scroll.ensureWidgetVisible(self.aspect, 0, 40))
        self.status.setText("Text only: the first clip is made from its description. Choose its Aspect "
                            "(Project tab on the right).")

    def _guide_cast(self):
        """Step 2: Cast & references, with New in focus (it adds a person or a place)."""
        self._show_left_tab(1)
        self.b_char_new.setFocus(Qt.FocusReason.OtherFocusReason)
        self.status.setText("Press New to add a person or a place, then give it pictures.")

    def _first_clip(self) -> T.Node | None:
        if self.tree is None:
            return None
        roots = sorted(self.tree.children(None), key=lambda n: self.tree.number(n.id))
        return roots[0] if roots else None

    def _guide_write(self):
        """Step 3: the first clip's description, with the cursor in it."""
        first = self._first_clip()
        if first is None:
            return
        self.select(first.id)
        self.focus_clip(first.id)
        card = self._card_of(first.id, PromptCard)
        if card is not None:
            card.edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def show_guide(self, on: bool):
        """Hide guide / View → Getting started (stored in the app's settings)."""
        self.qs.setValue("guideHidden", not on)
        if on:
            self._show_left_tab(0)
        self._update_guide()
        if not on:
            self.status.setText("Guide hidden: View → Getting started shows it again.")

    def _update_guide(self):
        """The guide while the project has no made clip (step 0 without a project), its steps ticked from the real
        state; the "how to go on" line once after the first made clip."""
        hidden = self.qs.value("guideHidden", False, type=bool)
        if self.tree is None or self.project is None:
            self.guide_panel.set_state(project=False)
            self.guide_panel.setVisible(not hidden)
            self.next_hint.hide()
            self._guide_ready = False
            return
        made = any(n.made for n in self.tree.nodes.values())
        choices = self._guide_choices()
        first = self._first_clip()
        picture = ((self.project / T.START_FILE).is_file() or "text_only" in choices
                   or bool(first is not None and first.is_video))
        cast = bool(T.cast_of(self.tree)) or "no_cast" in choices
        prompt = first is not None and (first.is_video or bool(first.prompt.strip()))
        ready = self.guide_panel.set_state(True, picture, cast, prompt)
        show = not hidden and not made
        self.guide_panel.setVisible(show)
        if show and ready and not self._guide_ready:      # Generate is next: the keyboard focus goes there (not
            fw = QApplication.focusWidget()                 # away from a description being typed)
            if not isinstance(fw, (QPlainTextEdit, QLineEdit)):
                self.b_gen.setFocus(Qt.FocusReason.OtherFocusReason)
        self._guide_ready = ready
        if len(self.tree.nodes) >= 2 and not self.qs.value("nextHintDone", False, type=bool):
            self.qs.setValue("nextHintDone", True)          # a second clip exists: the user knows how to go on
        self.next_hint.setVisible(made and not self.qs.value("nextHintDone", False, type=bool))

    def _next_hint_done(self):
        self.qs.setValue("nextHintDone", True)
        self.next_hint.hide()

    def _engine_can_start(self) -> bool:
        """The app has a start command for ComfyUI (what start_comfy needs)."""
        return Path(os.path.expanduser(self.comfy_launcher.text().strip() or DEFAULT_COMFY_LAUNCHER)).is_file()

    def _engine_banner_clicked(self):
        if self.engine_banner.can_start:
            self.start_comfy()
        else:
            self.show_prefs()

    def _update_engine_banner(self):
        """The banner while the video engine is not reachable (the connection state the app polls anyway)."""
        starting = bool(self._comfy_starting and time.monotonic() - self._comfy_starting < COMFY_START_TIMEOUT)
        self.engine_banner.set_state(starting, self._engine_can_start())
        self.engine_banner.setVisible(self.comfy_up is False)

    def _project_controls(self) -> dict:
        """project_setting_changes' keys -> the control that sets each."""
        c = self._clips_form
        return {"steps": (c, self.steps), "speed": (c, self.speed), "turbo": (c, self._turbo_row),
                "cfg": (c, self._cfg_row), "veda": (c, self.veda), "keep_repeated": (c, self.keep_rep),
                "style_prefix": (c, self.style_prefix), "style_suffix": (c, self.style_suffix),
                "prompt_format": (c, self.prompt_format), "lora_triggers": (c, self.lora_triggers),
                "ref_size": (self._tts_form, self.ref_size), "ref_hint": self.ref_hint,
                "eleven_model": (self._tts_form, self._eleven_row)}

    def advanced_active(self) -> list[str]:
        """The project's settings that differ from a new project's and whose control the Simple view hides."""
        if self.tree is None:
            return []
        ctl = self._project_controls()
        return [name for key, name in project_setting_changes(self.tree.options)
                if self.view.is_advanced(ctl[key])]

    def _update_adv_note(self):
        self.adv_note.set_active(self.advanced_active())

    # ------------------------------------------------------------------ project
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        self.act_new = m.addAction("New project…", self.new_project)
        self.act_new.setShortcut(S.sequence("new_project"))
        self.act_open = m.addAction("Open project…", self._choose_project)
        self.act_open.setShortcut(S.sequence("open"))
        self.recent_file_menu = m.addMenu("Recent projects")
        m.addSeparator()
        self.act_save = m.addAction("Save", self.save_project)
        self.act_save.setShortcut(S.sequence("save"))
        self.act_save_as = m.addAction("Save as…", self.save_project_as)
        self.act_save_as.setShortcut(S.sequence("save_as"))
        m.addSeparator()
        self.act_close = m.addAction("Close project", self.close_project)
        m.addAction("Show project folder", self._open_dir)
        self.act_ai_edit = m.addAction("Edit an image with AI…", lambda: self.edit_image(start=True, mode="edit"))
        self.act_ai_create = m.addAction("Create a new image with AI…",
                                         lambda: self.edit_image(start=True, mode="create"))
        m.addSeparator()
        m.addAction("Quit", self.close).setShortcut(S.sequence("quit"))
        v = self.menuBar().addMenu("&View")
        for k, name in enumerate(("Clips", "Cast & references", "Sounds")):
            a = v.addAction(f"{name} (left)", lambda _=False, k=k: self._show_left_tab(k))
            a.setShortcut(S.sequence(("tab_clips", "tab_cast", "tab_sounds")[k]))
        v.addAction("Log", self.show_log).setShortcut(S.sequence("log"))
        v.addAction("Getting started", lambda: self.show_guide(True))
        v.addAction("LoRAs (example prompts, pictures, what is known)…", lambda: self.show_loras())
        v.addSeparator()
        self.act_left = v.addAction("Left panel (clips, cast & references, sounds)")
        self.act_left.setCheckable(True)
        self.act_left.setChecked(True)
        self.act_left.setShortcut(S.sequence("left_panel"))
        self.act_left.toggled.connect(self._show_left)   # (the panel is built later, in _build)
        for name in ("edit", "write", "watch"):
            row = S.SHORTCUTS[[s.id for s in S.SHORTCUTS].index(f"layout_{name}")]
            v.addAction(row.text, lambda _=False, n=name: self.apply_layout(n)).setShortcut(S.sequence(row.id))
        v.addAction("Reset layout", lambda: self.apply_layout("edit"))
        v.addSeparator()
        group = QActionGroup(self)                  # (the view itself is made in __init__, after the menu)
        self.act_simple = v.addAction("Simple view (the controls you need to make clips)")
        self.act_advanced = v.addAction("Advanced view (every setting)")
        for a, simple in ((self.act_simple, True), (self.act_advanced, False)):
            a.setCheckable(True)
            group.addAction(a)
            a.triggered.connect(lambda _=False, on=simple: self.set_simple(on))
        st = self.menuBar().addMenu("&Settings")
        st.addAction("Preferences…", self.show_prefs).setShortcut(S.sequence("preferences"))
        h = self.menuBar().addMenu("&Help")
        h.addAction("Getting started", lambda: self.show_guide(True))
        h.addAction("Keyboard shortcuts…", self.show_shortcuts).setShortcut(S.sequence("shortcuts"))
        h.addAction("Show the log", self.show_log)

    def show_shortcuts(self):
        """Help → Keyboard shortcuts: the table every window binds from (not modal: it stays beside the work)."""
        dlg = S.shortcuts_dialog(self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.open()

    def apply_layout(self, name: str):
        """A layout preset, as shares of the window: 'edit' (the default: a big viewer), 'write' (a wide inspector
        for the prompt), 'watch' (no left panel, a narrow inspector)."""
        left, mid, right, view = LAYOUTS.get(name, LAYOUTS["edit"])
        w = max(self.split.width(), 1200)
        if left > 0:
            self._left_w = int(w * left)
        self.act_left.setChecked(left > 0)              # (shows / hides the left panel)
        self.split.setSizes([int(w * left) if left > 0 else 0, int(w * mid), int(w * right)])
        h = max(self.center.height(), 600)
        self.center.setSizes([int(h * view), int(h * (1 - view))])

    def _show_left(self, on: bool):
        """Show / hide the left panel; shown again it gets its last width back (taken from the viewer)."""
        sizes = self.split.sizes()
        if not on and sizes[0] > 0:
            self._left_w = sizes[0]
        self.left_tabs.setVisible(on)
        if on:
            want = max(self._left_w, self.left_tabs.minimumWidth())
            sizes = self.split.sizes()
            if sizes[0] < want:
                give = want - sizes[0]
                self.split.setSizes([want, max(sizes[1] - give, 1), sizes[2]])

    def _show_left_tab(self, k: int):
        self.act_left.setChecked(True)                  # (a hidden panel is shown first)
        self.left_tabs.setCurrentIndex(k)

    def _restore_layout(self):
        """The layout of the last session; the default one if there is none, it was saved by an older version or
        it is damaged."""
        try:
            d = json.loads(self.qs.value("layout", "", type=str) or "{}")
            split, center = d["split"], d["center"]
            ok = (d.get("v") == LAYOUT_VERSION and len(split) == 3 and len(center) == 2
                  and all(isinstance(x, int) and x >= 0 for x in split + center)
                  and split[1] > 0 and split[2] > 0 and all(x > 0 for x in center))
            left_w = int(d.get("left_w") or 0)
        except (TypeError, ValueError, KeyError, AttributeError):
            ok = False
        if not ok:
            self.apply_layout("edit")
            return
        if left_w > 0:
            self._left_w = left_w
        elif split[0] > 0:
            self._left_w = split[0]
        self.split.setSizes(split)
        self.act_left.setChecked(bool(d.get("left", True)))
        self.center.setSizes(center)

    def show_prefs(self):
        """ComfyUI's address, launcher, folder and the model / trigger word settings (not modal: the status bar
        and the window stay usable)."""
        self.prefs.resize(max(self.prefs.width(), 620), self.prefs.sizeHint().height())
        self.prefs.show()
        self.prefs.raise_()
        self.prefs.activateWindow()
