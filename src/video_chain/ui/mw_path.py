"""The path of cards and the timeline."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer

from .. import chain, tree as T
from . import shortcuts as S
from .clip_inputs import chip_specs
from .graph import START
from .timeline import Block
from .widgets import format_time
from .clip_cards import PromptCard, VideoCard
from .ui_logic import (
    CARD_MIN_WIDTH, CONTINUE_CLASSIC_TIP, CONTINUE_FIRST_TIP, QWIDGETSIZE_MAX, STATE_TIP, THUMB_CACHE_MAX,
    TIMELINE_THUMB_HEIGHT, plural, row_start_of)
from .mw_host import MainWindowHost


class PathMixin(MainWindowHost):
    """The path of cards and the timeline."""

    def _rebuild_path(self):
        for c in self.cards:
            self.card_stack.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        path = self.selected_path()
        self.path_hint.setText("The start image is selected. Select a clip to see its path, or press Continue "
                               f"({S.keys('continue')}) to add a clip after the start image.")
        known, lora_dir = self.known_loras(), self._lora_dir()      # (once for all cards: each reads files / settings)
        for k, n in enumerate(path):
            if n.is_video:
                vcard = VideoCard(n, view=self.view)
                vcard.action.connect(self.node_action)
                vcard.play.connect(self._play_node)
                self._fit_card(vcard)
                self.card_stack.addWidget(vcard)
                self.cards.append(vcard)
                vcard.set_editable(not self.running)
                continue
            card = PromptCard(n, names=self._cast_names, view=self.view)
            card.edited.connect(self._card_edited)
            card.seedChanged.connect(self._seed_changed)
            card.new_seed = self._new_seed
            card.overridesChanged.connect(self._overrides_changed)
            card.chooseLora.connect(self._choose_node_lora)
            card.lora_picked.connect(self._lora_picked)
            card.loras.file_dir = lora_dir
            card.settings_toggle.toggled.connect(self._show_card_settings)
            assert self.tree is not None
            card.set_choices(int(self.tree.options.get("frames", 73)), int(self.tree.options.get("steps", 28)),
                             known)
            card.action.connect(self.node_action)
            card.keyframe.connect(self.keyframe_action)
            card.end_dropped.connect(self.set_end_image)
            card.play.connect(self._play_node)
            card.leadChanged.connect(self.set_clip_lead)
            card.modelPrompt.connect(self.show_model_prompt)
            card.fix.connect(self.fix_problem)
            self._fit_card(card)
            self.card_stack.addWidget(card)
            self.cards.append(card)
            card.set_editable(not self.running)
            card.set_speech_busy(self._tts_busy)
        ids = [c.nid for c in self.cards]
        end = ids[-1] if ids else None
        if self.focus not in ids or end != self._path_end:
            self.focus = end                       # a new path (or a new clip at its end): where one goes on
            if end is not None:
                self.inspector.setCurrentIndex(0)
        self._path_end = end
        self._update_cards()
        self._show_focus()

    def _card_of(self, nid: str | None, kind: type | None = None):
        """The card of a clip on the path (of this kind, e.g. PromptCard), or None."""
        return next((c for c in self.cards if c.nid == nid and (kind is None or isinstance(c, kind))), None)

    @staticmethod
    def _fit_card(card):
        """A card in the inspector takes its width (it had a fixed width in a row)."""
        card.setMinimumWidth(CARD_MIN_WIDTH)
        card.setMaximumWidth(QWIDGETSIZE_MAX)

    def _show_focus(self):
        card = self._card_of(self.focus)
        self.card_stack.setCurrentWidget(card if card is not None else self.path_hint)
        self.card_stack.updateGeometry()
        ids = [c.nid for c in self.cards]
        k = ids.index(self.focus) if self.focus in ids else -1
        self.b_prev_clip.setEnabled(k > 0)
        self.b_next_clip.setEnabled(0 <= k < len(ids) - 1)
        if k >= 0 and self.tree is not None and self.focus in self.tree.nodes:
            n = self.tree.nodes[self.focus]
            what = "Video" if n.is_video else "Clip"
            self.focus_info.setText(f"{what} #{T.short_id(n.id)} · {k + 1} of {len(ids)} on the path")
        else:
            self.focus_info.setText("")
        self.timeline.set_focus(self.focus)
        self.graph.set_focus(self.focus)
        self._update_problems()
        self._update_buttons()                     # (the actions follow the clip in focus)

    def focus_clip(self, nid: str):
        """Show a clip of the path in the inspector (the path stays as it is)."""
        if any(c.nid == nid for c in self.cards):
            self.focus = nid
            self._show_focus()
            self.inspector.setCurrentIndex(0)

    def _step_focus(self, d: int):
        ids = [c.nid for c in self.cards]
        if self.focus in ids:
            k = max(0, min(len(ids) - 1, ids.index(self.focus) + d))
            self.focus_clip(ids[k])

    def target_id(self):
        """The clip the toolbar / shortcut actions work on: the one shown in the inspector, else the selected."""
        if self.focus and self.tree is not None and self.focus in self.tree.nodes:
            return self.focus
        return self.selected_id()

    def _graph_clicked(self, nid):
        """A click in the graph: a clip of the shown path is shown in the inspector; another one is selected (its
        path is shown, it in focus)."""
        nid = row_start_of(nid) or nid             # (a row's start image: its clip)
        if nid != START and any(c.nid == nid for c in self.cards):
            self.focus_clip(nid)
            return
        self.select(nid)
        if nid != START:
            self.focus_clip(nid)
        else:
            self.show_start_image_settings()

    def show_start_image_settings(self):
        """The Project tab of the inspector, scrolled to the start image (a click on the Start node)."""
        self.inspector.setCurrentWidget(self.proj_scroll)
        QTimer.singleShot(0, lambda: self.proj_scroll.ensureWidgetVisible(self.start_tile, 0, 40))

    def _show_card_of(self, nid: str):
        self.focus_clip(nid)
        card = self._card_of(nid)
        if isinstance(card, PromptCard):
            card.edit.setFocus()

    def _update_timeline(self, path_nodes: list, cuts: dict):
        """The path as blocks as long as its clips are in the joined video; the carry-over hatched."""
        blocks = []
        for n in path_nodes:
            assert self.tree is not None
            frames = T.clip_length(self.tree, n)
            a, b = cuts.get(n.id, (0, 0))
            secs = max(1, frames - a - b) / chain.FPS
            lead = int((n.overrides or {}).get("guide") or 0) / chain.FPS if not n.is_video and n.parent else 0.0
            pm = self._pixmap(self._node_frame(n))
            if pm is not None:                      # scaled once per picture (the timeline is redrawn often)
                cache = self.__dict__.setdefault("_tl_thumbs", {})
                key = pm.cacheKey()
                if key not in cache:
                    if len(cache) > THUMB_CACHE_MAX:
                        cache.clear()
                    cache[key] = pm.scaledToHeight(TIMELINE_THUMB_HEIGHT, Qt.TransformationMode.SmoothTransformation)
                pm = cache[key]
            what = (Path(n.source["file"]).name if n.is_video else (n.prompt or "(no prompt yet)"))
            tip = (f"{'Video' if n.is_video else 'Clip'} #{T.short_id(n.id)} · {secs:.1f} s · {self.state(n.id)}\n"
                   f"{what[:160]}")
            blocks.append(Block(n.id, f"#{T.short_id(n.id)}", secs, self.state(n.id), n.is_video, pm, lead, tip))
        self.timeline.show_path(blocks, self.focus)

    def _show_card_settings(self, on: bool):
        """Open / close the own settings on every card at once."""
        for c in self.cards:
            if isinstance(c, PromptCard):
                c._show_settings(on)

    def _update_cards(self):
        outer = self._stat_memo is not None
        if not outer:
            self._stat_memo, self._cmp_memo = {}, {}
        try:
            self._update_cards_now()
        finally:
            if not outer:
                self._stat_memo = self._cmp_memo = None

    def _update_cards_now(self):
        assert self.tree is not None
        path_nodes = [self.tree.nodes[c.nid] for c in self.cards if c.nid in self.tree.nodes]
        self.cast_strip.show_path(self.tree.options, path_nodes)
        cuts = dict(zip([n.id for n in path_nodes],
                        T.path_cuts(self.tree, path_nodes, bool(self.tree.options.get("keep_repeated")),
                                    planned=True)))
        self._update_timeline(path_nodes, cuts)
        t, checker = 0.0, None
        for k, card in enumerate(self.cards):
            n = self.tree.nodes.get(card.nid)
            if n is None:
                continue
            frames = T.clip_length(self.tree, n)
            cut = cuts.get(n.id, (0, 0))
            length = max(0, frames - cut[0] - cut[1]) / chain.FPS
            card.title.setText(f"{'Video' if n.is_video else 'Clip'} {k + 1}")
            span = f"{format_time(t, 1)} – {format_time(t + length, 1)} in this path"
            vc = self.tree.video_child(n.id)
            if vc is not None:
                span += f" · ends at video #{T.short_id(vc.id)}"
            card.span.setText(span)
            t += length
            st = self.state(n.id)
            if n.is_video:
                assert isinstance(card, VideoCard)
                there = self._is_file(n.source_dict()["file"])
                card.set_source(n, there)
                step = ("will be converted again" if st == "changed" else
                        "" if there or st == "done" else "the video file is missing")
            else:
                assert isinstance(card, PromptCard)
                why = self.why_of.get(n.id) if st == "changed" else ""
                step = f"Will be made again: {why}." if why else ("will be made again" if st == "changed" else "")
                if st == "done" and n.id in self.remix_due:
                    step = "Its sound is mixed again from the kept sounds (no new generation)."
                card.set_revert(T.changed_inputs(self.tree, n.id) if st == "changed" else [])
                card.set_guide_useful(n.parent is not None or vc is not None)
                card.set_continue(T.continue_choices(self.tree, n),
                                  CONTINUE_FIRST_TIP if n.parent is None else
                                  "" if T.is_h3(self.tree) else CONTINUE_CLASSIC_TIP)
                handover = T.handover_index(self.tree, n)
                total = T.made_frames(n) or T.clip_length(self.tree, n)
                if vc is not None:
                    card.set_handover(f"first frame of video #{T.short_id(vc.id)}", False, False,
                                      "A clip that leads into a real video hands over at the video's start")
                elif handover is not None:
                    card.set_handover(f"frame {handover + 1} of {total} ({handover / chain.FPS:.2f} s)", True, True,
                                      f"The clips after it start from this frame; the last {total - 1 - handover} "
                                      "frames are left out of the path")
                else:
                    card.set_handover("last frame", bool(n.made), n.end_at is not None,
                                      "" if n.made else "Make the clip first, then choose an earlier frame if its "
                                                        "last one is not good")
                card.set_compare(len(self.compare_candidates(n.id)) - 1 if n.made else 0)
                card.set_rows(n.parent is None, max(0, len(T.row_members(self.tree, n.id)) - 1))
                card.words.setText(card.words.text().split(" + ")[0] +
                                   (" + style" if T.full_prompt(self.tree.options, "x") != "x" else ""))
                use = T.node_cast(self.tree.options, n)
                if checker is None and self.project is not None:
                    checker = T.Checker(self.project, self.tree)     # (one for all cards: it caches the hashes)
                inputs = self._clip_inputs(n, checker)
                sent = inputs["prompt"] if inputs is not None else self._sent_guess(n, use)
                card.edit.setToolTip("Sent to the model: " + sent if sent != n.prompt.strip() else "")
                card.set_inputs(chip_specs(inputs, self._lead_off(n)) if inputs is not None else None)
                card.set_faces(self._faces_text(use))
                card.set_tracks([(tr["id"], tr["name"]) for tr in T.tracks(self.tree)])
                model_sound = bool(self.tree.options.get("audio", True))     # (no sound of its own: nothing to mix)
                ov = n.overrides or {}
                laid = ("" if not (ov.get("sound_over") and ov.get("source_sound")) else
                        self._laid_over_text(T.sound_mixed(n, model_sound)))
                avail, sound_text, max_guide, rep_ok = self._source_sound_text(n, laid)
                card.set_source_sound(avail, sound_text, max_guide, rep_ok, model_sound)
                card.set_summary(" · ".join(self._own_parts(n)))
                card.set_veda(self.veda_used(n))
                assert self.project is not None
                end = self.project / n.end_image if n.end_image else None
                if vc is not None:
                    card.set_end(end, vc.id, self._pixmap(self.project / vc.made["first"]) if vc.made else None,
                                 end is not None and self._is_file(end))
                else:
                    card.set_end(end, None, self._pixmap(end), end is not None and self._is_file(end))
                card.set_keyframes(T.clip_length(self.tree, n) / chain.FPS,
                                   [{"at": k["at"], "pixmap": self._pixmap(self.project / k["image"]),
                                     "note": k["note"], "ok": self._is_file(self.project / k["image"])}
                                    for k in n.keyframes],
                                   [c / chain.FPS for c in (n.made or {}).get("cuts") or []],
                                   bool(n.made and self._is_file(self.project / n.made["video"])))
            card.set_state(st, self._pixmap(self._node_frame(n)), step)
            changed_why = self.why_of.get(n.id) if st == "changed" else ""
            card.tag.setToolTip(f"Changed: {changed_why}. It is made again with the path." if changed_why
                                else STATE_TIP.get(st, ""))
            frozen = n.frozen and st == "done"
            if frozen:
                note = self._frozen_note(n)
                card.tag.setText("FROZEN")
                card.tag.setToolTip("Kept as made, whatever the project's settings say; not made again")
                card.step.setText("Frozen: kept as made" + (f" ({note})" if note else "") +
                                  ". Unfreeze it (lock button) to edit it or make it again.")
            card.set_frozen(frozen)
            if st == "generating" and n.id in self.progress_of and not n.is_video:
                card.bar.setValue(round(100 * self.progress_of[n.id]))
        path = self.selected_path()
        if path:
            todo = sum(1 for n in path if self.state(n.id) != "done")
            self.path_info.setText(f"{len(path)} clips · {format_time(t, 1)}" + (f" · {todo} to make" if todo
                                                                           else " · all made"))
        else:
            self.path_info.setText("")
        self._update_problems()

    def _busy_with(self) -> str:
        """What keeps Generate from starting now ('' = nothing): the reason, as it is told on the greyed button."""
        if self._joins:
            return "a path is being joined"
        if self._copying:
            return "the project is being copied"
        if self._converts:
            return "a video is being converted"
        if self._tts_busy:
            return "text to speech is working"
        return ""

    def _update_buttons(self):
        tree = self.tree
        has = tree is not None
        sel = self.target_id() if tree is not None else None     # the actions work on the clip in the inspector
        is_clip = bool(tree is not None and sel and sel != START and sel in tree.nodes)
        idle = not self.running
        self.b_continue.setEnabled(has and idle)
        self.b_add_video.setEnabled(has and idle)
        for act in (self.act_save, self.act_save_as, self.act_close):
            act.setEnabled(has and idle)
        for w in (self.act_new, self.act_open, self.b_new, self.b_open, self.b_recent, self.recent_file_menu):
            w.setEnabled(idle)
        self.b_branch.setEnabled(is_clip and idle)
        sel_node = tree.nodes.get(sel) if tree is not None and is_clip and sel else None
        self.b_variant.setEnabled(sel_node is not None and idle and not sel_node.is_video)
        path = self.selected_path()
        all_done = bool(path) and all(self.state(n.id) == "done" for n in path)
        free = idle and not self._joins and not self._converts   # a join reads clips a new run would replace
        busy_with = self._busy_with()                  # (what else works now: Generate is greyed, with the reason)
        can_gen = free and not busy_with
        self.b_gen.setEnabled(can_gen and bool(path))
        self.b_gen_all.setEnabled(can_gen and tree is not None and bool(tree.nodes))
        for b, text in ((self.b_gen, "Make every clip on the selected path that is not made yet"),
                        (self.b_gen_all, "Make every clip of every branch that is not made yet")):
            b.setToolTip(S.tip(text, "generate" if b is self.b_gen else "generate_all") if not busy_with
                         else f"Not now: {busy_with}")
        self.b_delete.setEnabled(is_clip and free)
        num = f" #{T.short_id(sel)}" if is_clip and sel else ""
        self.b_branch.setToolTip(S.tip(f"Branch: another continuation from the same point as clip{num}", "branch"))
        self.b_variant.setToolTip(S.tip(f"Variant of clip{num}: same prompt, other seed", "variant"))
        self.b_delete.setToolTip(S.tip(f"Delete clip{num} (the one in the inspector) and everything after it",
                                       "graph_delete"))
        self.b_play_path.setEnabled(all_done and idle)
        self.b_export.setEnabled(all_done and idle)
        fn = tree.nodes.get(self.focus) if tree is not None and self.focus else None
        self.b_play_clip.setEnabled(bool(fn and fn.made))
        self.b_export_clip.setEnabled(fn is not None and bool(fn.made) and idle and self.state(fn.id) == "done")
        self.b_export_clip.setToolTip(f"Save clip #{T.short_id(fn.id)} on its own (as made, or upscaled)"
                                      if fn and fn.made
                                      else "The clip in the inspector is not made yet" if fn else
                                      "Show a clip in the inspector first")
        self.b_play_clip.setToolTip(f"Play clip #{T.short_id(fn.id)}, the one in the inspector" if fn and fn.made else
                                    "The clip in the inspector is not made yet" if fn else
                                    "Show a clip in the inspector first")
        self.b_play_film.setToolTip("Join the scenes of the film and play it (all scenes must be made)")
        self.b_play_film.setEnabled(self.film is not None and idle and not (self._copying or self._joins
                                                                          or self._converts))
        todo = sum(1 for n in path if self.state(n.id) != "done")
        why = ("Select a clip first: its path is played" if not path else
               f"Not ready: {plural(todo, 'clip')} on this path {'are' if todo != 1 else 'is'} new or "
               "changed. Generate path first" if todo else
               "Wait until generating has finished" if not idle else "")
        self.b_play_path.setToolTip(why or "Join the clips of the selected path and play them")
        self.b_export.setToolTip(why or "Save the selected path as one video")
        self.b_stop.setEnabled(not idle or self._waiting_comfy is not None)
        self.b_stop.setToolTip("Stop generating; finished clips are kept" if not idle else
                               "Do not generate when ComfyUI has started" if self._waiting_comfy is not None else
                               "Stop generating")
        self._update_comfy_label()
        for widget in (self.start_tile, self.frames, self.resolution, self.steps,
                  self.audio, self.keep_rep, self.b_project,
                  self.comfy_url, self.ref_size, self.ref_hint, self.b_ref_add, self.b_ref_create,
                  self.b_ref_start,
                  self.b_ref_del, self.b_ref_crop, self.b_char_new, self.b_char_rename, self.b_char_del, self.char_desc,
                  self.char_lead, self.b_voice_file, self.b_voice_del, self.dialogue_lang, self.speed,
                  self.style_prefix, self.style_suffix, self.prompt_format, self.lora_triggers, self.veda,
                  self.b_ai_edit, self.b_ai_create,
                  self.b_film_cast):                # (storyboard and the film buttons: below, with the busy check)
            widget.setEnabled(idle)
        self.aspect.setEnabled(idle and self.project is not None and T.size_origin(self.project, self.tree) == "aspect")
        self.b_start_clear.setEnabled(idle and self.project is not None and (self.project / T.START_FILE).is_file())
        busy = bool(self._copying or self._joins or self._converts)
        self.storyboard.setEnabled(idle and not busy)  # a scene switch would pull the files from under them
        self.b_film_play.setEnabled(idle and not busy)
        self.b_film_export.setEnabled(idle and not busy)
        self.b_film_gen.setEnabled(idle and not busy)
        self._turbo_widgets()                      # after the loop: turbo locks the steps fields again
        self._update_ref_start_button()            # after the loop too: they depend on the selection
        self._update_cast_buttons()
        self._update_hint_enabled()                # (only for the classic format)

    def select(self, nid):
        if self.tree is None:
            return
        self.tree.selected = nid if (nid == START or nid in self.tree.nodes) else START
        self._save_timer.start()
        self.refresh()

    def _card_edited(self, nid: str):
        card = self._card_of(nid, PromptCard)
        if card is None or self.tree is None or nid not in self.tree.nodes:
            return
        if self.tree.nodes[nid].frozen:
            return
        node, text = self.tree.nodes[nid], card.text()
        made = (node.made or {}).get("prompt")
        # the card keeps words, not spacing: a made prompt that differs only in spaces / line breaks stays as made
        node.prompt = made if isinstance(made, str) and " ".join(made.split()) == text else text
        self._failures.pop(nid, None)              # (edited: what went wrong before may no longer apply)
        self._changed()
