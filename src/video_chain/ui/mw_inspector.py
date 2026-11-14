"""Refresh, node states, problems and the texts of the cards."""
from __future__ import annotations

import os
import re
import stat as stat_module
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog

from .. import camera as CAM, chain, tree as T
from .clip_inputs import ModelPromptDialog
from .graph import START, NodeInfo
from .lora_stack import lora_label
from .widgets import format_time
from . import ui_logic as UL
from .clip_cards import PromptCard
from .ui_logic import REMIX_DELAY_MS, SourceSoundInfo, plain_problem, plural, row_start


class InspectorMixin:
    """Refresh, node states, problems and the texts of the cards."""

    def _stat(self, path) -> os.stat_result | None:
        """os.stat of a path (None: not there), remembered while the cards are being updated."""
        key = str(path)
        memo = self._stat_memo
        if memo is not None and key in memo:
            return memo[key]
        try:
            st = os.stat(key)
        except OSError:
            st = None
        if memo is not None:
            memo[key] = st
        return st

    def _is_file(self, path) -> bool:
        st = self._stat(path)
        return st is not None and stat_module.S_ISREG(st.st_mode)

    def _pixmap(self, path: Path | None) -> QPixmap | None:
        st = self._stat(path) if path is not None else None
        if st is None or not stat_module.S_ISREG(st.st_mode):
            return None
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key not in self._pix:
            while len(self._pix) >= UL.PIXMAP_CACHE_MAX:  # (old ones of changed files must not pile up)
                self._pix.pop(next(iter(self._pix)))
            self._pix[key] = QPixmap.fromImage(QImage(str(path)))
        return self._pix[key]

    def _node_frame(self, n: T.Node) -> Path | None:
        """The frame shown for a node: where the clips after it start (its hand-over frame, else its last)."""
        if n.made and self.project:
            h = T.handover_file(self.project, self.tree, n)
            return h if h is not None and self._is_file(h) else self.project / n.made["frame"]
        return None

    def state(self, nid: str) -> str:
        return self.live.get(nid) or self.status_of.get(nid, "new")

    def refresh(self, rebuild_path: bool = False):
        self._uses_cache = None
        self._update_adv_note()                    # (cheap: the project's options against the defaults)
        if self.tree is None:
            self.graph.show_nodes([])
            self._update_buttons()
            self._update_guide()
            return
        if self.tree.options.get("audio_refs"):
            self._show_sound_info()                # how many clips name a sound (prompts change)
        checker = T.Checker(self.project, self.tree)
        try:
            self.status_of = {nid: checker.status(nid) for nid in self.tree.nodes}
            self.why_of = {nid: checker.why(nid) for nid, st in self.status_of.items() if st == "changed"}
            self.remix_due = {nid for nid in self.tree.nodes if checker.remix_due(nid)}
        except Exception as e:  # noqa: BLE001  (e.g. an unreadable start image)
            # (what was known stays: "all clips new" would hide the real state)
            self.status_of = {nid: self.status_of.get(nid, "new") for nid in self.tree.nodes}
            self.remix_due = set()
            self._report_error(f"Could not check the clips: {e}")
        if self.remix_due and self.remix_due != self._remix_stuck and not self._remix_timer.isActive() \
                and not self._remixing and not self.running:
            self._remix_timer.start(REMIX_DELAY_MS)  # (e.g. one left over by a stopped run: mixed again soon)
        sel = self.selected_id()
        on_path = {n.id for n in self.selected_path()}
        start_pm = self._pixmap(self.project / T.START_FILE)
        infos = [NodeInfo(START, None, "Start", "", Path(self.tree.image).name if self.tree.image else "",
                          "start", start_pm, on_path=True, selected=sel == START)]
        for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id)):
            text = n.prompt
            if n.is_video:
                text = (f"{Path(n.source['file']).name} · {format_time(n.source['start'], 1)}–"
                        f"{format_time(n.source['end'], 1)}"
                        + (f" · ↻{T.source_rotation(n.source)}°" if T.source_rotation(n.source) else ""))
            parent = n.parent or START
            if n.parent is None and n.start_image:          # a first clip with its own start image: its own row
                parent = row_start(n.id)
                infos.append(NodeInfo(parent, None, "Start", f"for #{T.short_id(n.id)}", "The start image of clip "
                                      f"#{T.short_id(n.id)} (its own, instead of the project's start image)", "start",
                                      self._pixmap(self.project / n.start_image), on_path=n.id in on_path))
            infos.append(NodeInfo(n.id, parent,
                                  f"{'Video' if n.is_video else 'Clip'} {self.tree.depth(n.id)}",
                                  f"#{T.short_id(n.id)}",
                                  text, self.state(n.id), self._pixmap(self._node_frame(n)),
                                  self.progress_of.get(n.id, 0.0), n.id in on_path, n.id == sel,
                                  self.badge(n), n.kind))
        self.graph.show_nodes(infos)
        self._show_tracks()                        # (how many clips play each track)
        if rebuild_path or [c.nid for c in self.cards] != [n.id for n in self.selected_path()]:
            self._rebuild_path()
        else:
            self._update_cards()
        made = sum(1 for s in self.status_of.values() if s == "done")
        branches = len(self.tree.leaves())
        videos = sum(1 for n in self.tree.nodes.values() if n.is_video)
        self.tree_info.setText(f"{len(self.tree.nodes)} clips" + (f" ({videos} real)" if videos else "") +
                               f" · {branches} branch{'es' if branches != 1 else ''} · {made} made")
        self._update_scene_card()
        self._update_buttons()
        self._update_guide()

    def _frozen_note(self, n: T.Node) -> str:
        try:
            return T.frozen_note(self.tree, n, T.Checker(self.project, self.tree).base())
        except Exception:  # noqa: BLE001  (an unreadable start image: no note)
            return ""

    def badge(self, n: T.Node) -> str:
        """Short text for a clip's own settings, shown on its node ('' = all project defaults)."""
        if n.is_video:
            return ("🔒 frozen · " if n.frozen else "") + f"real video · {T.clip_length(self.tree, n) / chain.FPS:.1f} s"
        ov, parts = n.overrides or {}, self._own_parts(n)
        if ov.get("guide"):
            parts.append(f"carry-over {ov['guide']}")
        if T.end_image_used(self.tree, n):
            parts.append("end image")
        if n.keyframes:
            parts.append(f"{plural(len(n.keyframes), 'keyframe')}")
        if (n.made or {}).get("cuts"):
            parts.append("⚠ cuts")
        if n.frozen:
            parts.insert(0, "🔒 frozen")
        k = T.handover_index(self.tree, n)
        if k is not None:
            parts.append(f"hands over at {k / chain.FPS:.1f} s")
        if self.veda_used(n):
            parts.append("⚡ Veda")
        return " · ".join(parts)

    def _clip_inputs(self, n: T.Node, checker: T.Checker | None = None) -> dict | None:
        """T.clip_inputs of a generated clip (None: it cannot be worked out now, e.g. an unreadable start image)."""
        if self.project is None or self.tree is None or n.is_video:
            return None
        try:
            return T.clip_inputs(self.project, self.tree, n.id, checker)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ problems where they are
    def clip_problems(self, nid: str, checker: T.Checker | None = None) -> list[dict]:
        """What blocks making this clip, from the checks Generate makes (_start; nothing new is checked here and
        Generate refuses exactly as before), each as a plain sentence with the technical text in its tooltip and,
        where the app can do it, an action (fix_problem). A clip that is made (and unchanged) has none. The prompt
        of a new clip being empty is not one: the placeholder and the guide say that."""
        if self.tree is None or self.project is None or nid not in self.tree.nodes:
            return []
        n = self.tree.nodes[nid]
        if n.is_video or n.frozen or self.status_of.get(nid) == "done":
            return []
        out: list[dict] = []

        def add(text, tip="", action=None, button=None):
            out.append({"text": text, "tip": tip, "action": action, "button": button})
        try:
            turbo = self.tree.options.get("turbo") or {}
            if turbo.get("on") and not turbo.get("lora"):
                add("The fast speed (turbo) needs an add-on that is not chosen yet. Choose it, or set Speed to "
                    "Normal.", "Clips → Turbo LoRA: choose a turbo LoRA (or switch Speed to Normal)", "turbo", "Show")
            checker = checker or T.Checker(self.project, self.tree)
            if checker.base() is None:
                add("Choose a start image for the first clip.", "The project has no start image and no real "
                    "video to start from", "start_image", "Choose…")
                return out
            todo = [nid]
            guide = T.guide_problem(self.tree, checker, todo)
            if guide:
                add("The carry-over of this clip does not fit the clip next to it. Make it shorter, or make the "
                    "clip longer.", guide, "carry", "Show")
            if T.missing_end_images(self.project, self.tree, todo):
                add("The end image of this clip is missing from the project folder. Choose it again.",
                    f"ends/: {n.end_image}", "end_image", "Choose…")
            for raw in T.audio_problems(self.project, self.tree, todo):
                text = ("A sound reference this clip uses is missing from the project folder. Add it again in "
                        "Sounds." if "missing" in raw else
                        plain_problem(raw, nid) + " Add the sound in Sounds, or change the description.")
                add(text, raw + " (Sound reference)", "sounds", "Open Sounds")
            for raw in T.source_sound_problems(self.tree, todo, self.project):
                add(plain_problem(raw, nid), raw + " (the clip's “Sound” setting)", "sound", "Show")
            for raw in T.cast_problems(self.project, self.tree, todo):
                m = re.search(r"mentions @([\w-]+), but there is no such character(?: — did you mean @([\w-]+)\?)?",
                              raw)
                pic = re.match(r"@([\w-]+) \(clip #\d+\) has no (picture|face picture or voice) yet", raw)
                if m:
                    add(f"The description names @{m.group(1)}, but Cast & references has no {m.group(1)}."
                        + (f" Did you mean @{m.group(2)}?" if m.group(2) else " Add it there, or change the name."),
                        raw, "cast_new", "Open Cast & references")
                elif pic:
                    add(f"@{pic.group(1)} has no picture yet. Add one, so the clip can show "
                        + ("this person." if pic.group(2) != "picture" else "it."), raw, f"cast:{pic.group(1)}",
                        "Add a picture")
                else:
                    add(plain_problem(raw, nid), raw + " (Cast & references)", "cast", "Open Cast & references")
            for lo in T.loras_needed(self.tree, checker, [nid]):
                if not Path(lo).is_file():
                    add(f"An add-on (LoRA) of this clip is missing: {Path(lo).name}. Choose it again or remove it.",
                        f"The LoRA file does not exist: {lo}", "loras", "Show")
                    break
        except Exception:  # noqa: BLE001  (Generate says what it could not check)
            pass
        return out

    def _note_failure(self, code: int):
        """A run failed: what went wrong stays on the card of the clip it failed at (the first clip still to make
        when it failed before any clip began), with Details (the log) and, where it helps, the engine's start."""
        if self.tree is None:
            return
        err = self._error or f"the generator exited with code {code}"
        nids = [n for n, s_ in self.live.items() if s_ == "failed" and n in self.tree.nodes]
        if not nids:
            todo = [n.id for t in self._targets if t in self.tree.nodes for n in self.tree.path_to(t)
                    if self.status_of.get(n.id) != "done" and not n.is_video]
            nids = todo[:1]
        down = code == 4 or "not running" in err.lower() or "connection refused" in err.lower()
        missing = not down and any(w in err.lower() for w in ("missing", "not found", "not in comfyui"))
        if down:
            text = ("The video engine (ComfyUI) stopped or is not running, so this clip was not made. Start it, "
                    "then press Generate again.")
            more = [("start_engine", "Start the video engine") if self._engine_can_start()
                    else ("prefs", "Open preferences")]
        elif missing:
            text = ("The video engine is missing something this clip needs (a model or an add-on), so it was not "
                    "made. Details shows what.")
            more = []
        else:
            text = "Making this clip failed. Details shows why; press Generate to try again."
            more = []
        for nid in nids:
            self._failures[nid] = {"text": text, "tip": err, "error": True, "action": "log", "button": "Details",
                                   "more": more}
        if nids and self.simple:                    # (Simple has no Log tab in sight: the card shows it)
            self.focus_clip(nids[0])
        self._update_problems()

    def _update_problems(self):
        """The problems of the card in focus (worked out for it only: the others are not in sight, and some checks
        read files); the other cards are cleared."""
        for c in self.cards:
            if not isinstance(c, PromptCard):
                continue
            if c.nid != self.focus or self.tree is None or c.nid not in self.tree.nodes:
                c.set_problems([])
                continue
            fail = self._failures.get(c.nid)
            c.set_problems(([fail] if fail else []) + ([] if self.running else self.clip_problems(c.nid)))

    def _clear_failure(self, nid: str):
        if self._failures.pop(nid, None) is not None:
            self._update_problems()

    def fix_problem(self, action: str, nid: str):
        """A problem's button on a card: the place that fixes it (or the action itself)."""
        card = self._card_of(nid, PromptCard)
        if action == "log":
            self.show_log()
        elif action == "start_engine":
            self.start_comfy()
        elif action == "prefs":
            self.show_prefs()
        elif action == "start_image":
            self._choose_image()
        elif action in ("cast", "cast_new"):
            self._show_left_tab(1)
            if action == "cast_new":
                self.b_char_new.setFocus(Qt.OtherFocusReason)
                self.status.setText("Press New to add a person or a place (its name is what you write after @).")
        elif action.startswith("cast:"):
            self._show_left_tab(1)
            self._show_references(select=action[5:])
            self.b_ref_add.setFocus(Qt.OtherFocusReason)
            self.status.setText(f"Add a picture of @{action[5:]}: Image…, Start image or Selected clip.")
        elif action == "sounds":
            self._show_left_tab(2)
        elif action == "end_image":
            self.node_action("end_image", nid)
        elif action == "turbo":
            c = self._clips_form
            for t in ((c, self.speed), (c, self.turbo_lora), (c, self._turbo_row)):
                self.view.reveal(t)
            self.inspector.setCurrentWidget(self.proj_scroll)
            QTimer.singleShot(0, lambda: self.proj_scroll.ensureWidgetVisible(self.turbo_lora, 0, 40))
            self.turbo_lora.setFocus(Qt.OtherFocusReason)
        elif action in ("sound", "carry", "loras") and card is not None:
            w = {"sound": card.src_sound, "carry": card.guide, "loras": card.loras}[action]
            if action != "sound":
                self.view.reveal((card.link_form, w))         # (an advanced setting: shown in Simple until a switch)
            self.focus_clip(nid)
            w.setFocus(Qt.OtherFocusReason)

    def _sent_guess(self, n: T.Node, use: dict) -> str:
        """The prompt the model gets, from the clip alone (when clip_inputs cannot be worked out)."""
        words = T.lora_trigger_words([p for p, _s in T.node_loras(n)]) \
            if self.tree.options.get("lora_triggers", True) and T.node_loras(n) else []
        sounds = [None] * (T.audio_mentioned(T.full_prompt(self.tree.options, n.prompt)) + len(use.get("voices") or []))
        cont = (n.overrides or {}).get("source_sound") \
            if T.source_sound_span(self.tree, n) is not None and not (n.overrides or {}).get("sound_over") else None
        try:
            return T.model_prompt(self.tree.options, n, {"lora_triggers": words, "audio_refs": sounds,
                                                         "source_sound": cont,
                                                         "camera": CAM.clean((n.overrides or {}).get("camera"))})
        except Exception:  # noqa: BLE001
            return n.prompt.strip()

    def _lead_off(self, n: T.Node) -> str | None:
        """The lead's name when the clip's lead switch is off and the clip would get the lead with it on (its
        prompt mentions nobody), so its chip can switch it on again; else None."""
        lead = T.lead_of(self.tree)
        if T.lead_on(n) or lead is None:
            return None
        use = T.clip_cast(self.tree.options, n.prompt, voices=not T.sound_given(n), lead=True)
        mentioned = {c["name"] for c in T.mentions(self.tree.options, T.full_prompt(self.tree.options, n.prompt))[0]}
        return lead["name"] if any(c["name"] == lead["name"] for c in use["characters"]) \
            and lead["name"] not in mentioned else None

    def set_clip_lead(self, nid: str, on: bool):
        """The lead chip of a clip: switch its lead on / off (overrides "lead"), like its other own settings."""
        node = self.tree.nodes.get(nid) if self.tree is not None else None
        if node is None or node.is_video or self.running or T.lead_on(node) == bool(on):
            return
        if node.frozen:
            self.status.setText(f"Clip #{T.short_id(nid)} is frozen; unfreeze it first.")
            self._update_cards()
            return
        lead = T.lead_of(self.tree)
        T.set_lead_on(node, bool(on))
        card = self._card_of(nid)
        if isinstance(card, PromptCard):
            card.ov = dict(node.overrides or {})
        before = self._done_clips()
        self._changed()
        self._update_cards()
        who = f"@{lead['name']}" if lead else "the lead"
        if not self._report_remade(before, f"Clip #{T.short_id(nid)}: lead {'on' if on else 'off'}"):
            self.status.setText(f"Clip #{T.short_id(nid)} is made with {who} again (its prompt mentions nobody)."
                                if on else
                                f"Clip #{T.short_id(nid)} is made without {who}.")

    def show_model_prompt(self, nid: str) -> QDialog | None:
        """The Model prompt window of a clip: the exact text the model gets and its numbered pictures / sounds.
        Returns the dialog (kept modeless and not run in tests: _dialogs_modeless)."""
        node = self.tree.nodes.get(nid) if self.tree is not None else None
        info = self._clip_inputs(node) if node is not None else None
        if info is None:
            self.status.setText("The model prompt cannot be worked out now (is the start image readable?).")
            return None
        dlg = ModelPromptDialog(self, nid, info)
        self._model_prompt_dialog = dlg            # (tests look at it)
        if not self._dialogs_modeless:
            dlg.exec()
            dlg.deleteLater()
        return dlg

    def _faces_text(self, use: dict) -> str:
        """Which faces (and voices) a clip is made with, for its card ('' without a cast); with spoken lines, who
        says each one (as the app reads it: the voice it gets)."""
        if not T.cast_of(self.tree):
            return ""
        text = self._cast_line(use)
        if use["mode"] == "voices":
            said = [(f"@{d['name']}" if d["name"] else "nobody")
                    + f": \u201c{d['line'][:24]}{'…' if len(d['line']) > 24 else ''}\u201d" for d in use["lines"]]
            text += "\nLines: " + " · ".join(said)
        return text

    @staticmethod
    def _seen_line(chars: list[dict]) -> str:
        """"Faces: @Anna · Pictures: @Kitchen" for the cast entries a clip sends pictures of ('' for none)."""
        faces = [f"@{c['name']}" for c in chars if c["images"] and T.kind_of(c) == "person"]
        pics = [f"@{c['name']}" for c in chars if c["images"] and T.kind_of(c) != "person"]
        return " · ".join(([("Faces: " + ", ".join(faces))] if faces else [])
                          + ([("Pictures: " + ", ".join(pics))] if pics else []))

    def _cast_line(self, use: dict) -> str:
        if use["mode"] == "voices":
            seen = self._seen_line(use["characters"])
            voices = [f"@{c['name']}" for c in use["characters"]
                      if c.get("voice") and c["voice"]["file"] in use["voices"]]
            return ((seen + " · ") if seen else "") + "Voices: " + ", ".join(voices)
        if use["mode"] == "mention":
            return self._seen_line(use["characters"])
        if use["mode"] == "lead":
            return f"Faces: @{use['characters'][0]['name']} (the lead; the prompt mentions nobody)"
        if use["mode"] == "tags":
            return f"Faces: all {len(use['images'])} pictures of the cast, in cast order (<Picture N> written)"
        return "Faces: none (the prompt mentions nobody and there is no lead)"

    @staticmethod
    def _laid_over_text(mixed: bool) -> str:
        if not mixed:
            return (" Laid over: the clip is made without it (full motion, no lip sync) and gets it afterwards; it "
                    "may be shorter than 2 s.")
        return (" Mixed in: the clip is made without it (full motion, no lip sync), then it is mixed with the sound "
                "the model made for the clip; it may be shorter than 2 s.")

    def _source_sound_text(self, n: T.Node, laid: str = "") -> SourceSoundInfo:
        """For the clip card's 'Sound' row: whether a real video comes before this clip, where the clip's sound comes
        from, the longest carry-over that fits, and whether there is a clip before it whose sound can be repeated.
        laid: the sentence about a sound laid over / mixed in, added at the end."""
        def t(x: float) -> str:
            return f"{int(x // 60)}:{x % 60:04.1f}"
        video = T.source_video(self.tree, n)
        has_video = video is not None
        repeat = n.parent is not None

        def made(text: str, guide: int = 0, can_repeat: bool = repeat, tail: str | None = None) -> SourceSoundInfo:
            """tail: what follows a sentence that ends 'exactly as it is' (a comma instead of the full stop when
            something is laid over it)."""
            if tail is not None:
                text += ("," if laid else ".") + tail
            return SourceSoundInfo(has_video, text + laid, guide, can_repeat)
        tid = T.track_id(n)
        if tid is not None:
            span = T.track_sound_span(self.tree, n)
            tr = T.track_of(self.tree, tid)
            if span is None or tr is None:
                return made("This audio track was removed: choose another sound.")
            end = span["start"] + span["secs"] - span["pad"]
            prev = self.tree.nodes.get(n.parent) if n.parent else None
            how = (f"carries it on from clip #{T.short_id(span['from'])}" if span["from"] else
                   "from its beginning" + (f": {'video' if prev.is_video else 'clip'} #{T.short_id(prev.id)} before it "
                                           "does not play it" if prev is not None else ""))
            past = (f"; the track ends at {t(float(tr['secs']))}, silence after it" if end > float(tr["secs"]) + 0.05
                    else "")
            if span["start"] >= float(tr["secs"]) - 0.05:
                return made(f"\u201c{tr['name']}\u201d has ended before this clip ({t(float(tr['secs']))}): "
                            "choose another sound or cut the track longer.")
            return made(f"Plays {t(span['start'])}\u2013{t(min(end, float(tr['secs'])))} of \u201c{tr['name']}\u201d"
                        f" ({how}{past}), exactly as it is",
                        tail=" Write in the prompt what happens; the sound comes from the track.")
        if (n.overrides or {}).get("source_sound") == "previous":
            span = T.previous_sound_span(self.tree, n)
            if span is None:
                return made("This is the first clip: there is no clip before it to repeat.", can_repeat=False)
            p = self.tree.nodes[span["video"]]
            what = (f"video #{T.short_id(p.id)} ({Path(span['file']).name} {t(span['start'])}\u2013"
                    f"{t(span['start'] + span['loop'])})" if p.is_video else f"clip #{T.short_id(p.id)}")
            loop = (f", looped to fill this clip's {span['secs']:.1f} s" if span["secs"] > span["loop"] + 0.05 else "")
            later = "" if p.is_video or p.made else " (it is made first)"
            return made(f"Repeats the sound of {what}{later}: {span['loop']:.1f} s as the film plays "
                        f"it{loop}, exactly as it is",
                        can_repeat=True, tail=" Write in the prompt what happens; the sound comes from the clip "
                                              "before.")
        if video is None:
            return made("There is no real video before this clip any more: choose another sound.")
        parent = self.tree.nodes.get(n.parent) if n.parent else None
        have = 0
        if parent is not None:
            pn = T.made_frames(parent) or T.clip_length(self.tree, parent)
            h = T.handover_index(self.tree, parent, pn)
            have = h + 1 if h is not None else pn
        fits = [g for g in T.GUIDE_CHOICES if g <= have and g < T.clip_length(self.tree, n)]
        span = T.source_sound_span(self.tree, n)
        return made(f"Uses {t(span['start'])}\u2013{t(span['start'] + span['secs'])} of "
                    f"{Path(span['file']).name} (video #{T.short_id(video.id)}): the sound right after its range, "
                    "exactly as recorded", guide=max(fits, default=0),
                    tail=" Write in the prompt what happens; the words come from the video.")

    def _sound_suffix(self, n: T.Node, ov: dict) -> str:
        """" (laid over)" / " (mixed with the clip's own sound, +3 dB)" after the name of a sound, or ""."""
        if not (ov.get("source_sound") and ov.get("sound_over")):
            return ""
        if not T.sound_mixed(n, (self.tree.options if self.tree else {}).get("audio", True)):
            return " (laid over)"
        level = f", {T.mix_level(n):+g} dB" if T.mix_level(n) else ""
        return f" (mixed with the clip's own sound{level})"

    def _own_parts(self, n: T.Node) -> list[str]:
        """A clip's own LoRA / strength / length / steps in a few words."""
        ov, parts = n.overrides or {}, []
        over = self._sound_suffix(n, ov)
        for p, st in T.node_loras(n):
            parts.append(f"{lora_label(p)[:18]} ×{st:g}")
        if "frames" in ov:
            parts.append(f"{ov['frames'] / chain.FPS:.1f} s")
        if "steps" in ov:
            parts.append(f"{ov['steps']} steps")
        if CAM.describe(ov.get("camera")):
            parts.append("🎥 " + CAM.describe(ov.get("camera")))
        if ov.get("lead") is False:
            parts.append("no lead")
        if ov.get("source_sound") == "previous":
            parts.append("repeats the sound before" + over)
        elif T.track_id(n) is not None:
            tr = T.track_of(self.tree, T.track_id(n)) if self.tree else None
            parts.append((f"track “{tr['name'][:16]}”" if tr else "removed track") + over)
        elif ov.get("source_sound"):
            parts.append("video's sound" + over)
        return parts
