"""What the node menu does: freeze, delete, branch, variants."""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import QDialog, QMenu, QToolTip

from .. import chain, swap, tree as T
from . import dialogs as D
from .clip_dialogs import CompareDialog
from .graph import START
from .prompt_refiner import replace_text
from .clip_cards import PromptCard
from .project_dialogs import VariantsDialog
from .ui_logic import FREEZE_TIP, plural, row_start_of, variant_strengths
from .mw_host import MainWindowHost


@dataclass(frozen=True)
class NodeAction:
    """One node action: what it needs (applies: "any" node or none, "node", "clip" = a generated clip, "video" = a
    real video), whether a frozen clip refuses it (frozen_check; FROZEN_LOCKED names which), and what runs."""
    run: Callable[[Any, str, T.Node | None, str | None], None]
    applies: str = "node"
    frozen_check: bool = True

    def fits(self, node: T.Node | None) -> bool:
        if self.applies == "any":
            return True
        if node is None:
            return False
        return {"node": True, "clip": not node.is_video, "video": node.is_video}[self.applies]


NODE_ACTIONS: dict[str, NodeAction] = {}


def _node_action(*names: str, applies: str = "node", frozen_check: bool = True):
    """Register the decorated method as the action(s) `names` (the one place that maps a name to what it does)."""
    def register(fn):
        for name in names:
            NODE_ACTIONS[name] = NodeAction(fn, applies, frozen_check)
        return fn
    return register


class NodeMixin(MainWindowHost):
    """What the node menu does: freeze, delete, branch, variants."""

    def node_action(self, action: str, nid: str | None):
        """Do a node action by name (the node menu, the card buttons, the shortcuts): the registry NODE_ACTIONS says
        what it needs (a node, a clip, a video) and what runs; an action that does not fit the node does nothing."""
        if self.tree is None or self._wait_note():
            return
        spec = NODE_ACTIONS.get(action)
        if spec is None:
            return
        node = self.tree.nodes.get(nid) if nid and nid != START else None
        if not spec.fits(node):
            return
        if node is not None and spec.frozen_check and self._frozen_block(node, action):
            return
        spec.run(self, action, node, nid)

    # ---- the actions (registered in NODE_ACTIONS by their names)
    @_node_action("take", "take_clear", applies="any", frozen_check=False)
    def _act_take(self, action: str, node: T.Node | None, nid: str | None):
        """End the scene at this clip (take), or clear that (take_clear)."""
        if self.film is None:
            return
        assert self.project is not None
        cur = self.film.scene_of(self.project)
        if cur is None or not nid:
            return
        try:
            self.film.set_take(cur.id, nid if action == "take" else None)
        except OSError as e:
            self._report_error(f"Could not save the scene's ending: {e}")
            return
        self._update_continues()               # a scene continuing this one starts from the new end
        self._show_storyboard()
        self.status.setText(f"The film plays “{cur.title}” up to clip #{T.short_id(nid)}." if action == "take"
                            else f"The film plays “{cur.title}” up to its selected clip.")

    @_node_action("freeze_toggle", "freeze", "unfreeze", frozen_check=False)
    def _act_freeze(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        if action == "unfreeze" or action == "freeze_toggle" and node.frozen:
            self.unfreeze_clip(node.id)
        else:
            self.freeze_clip(node.id)

    @_node_action("start_rows", "row_apply", "start_own", "start_own_clear", "start_cut")
    def _act_start_images(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        if node.is_video or node.parent is not None:
            self.status.setText("Start images are for a first generated clip (one that starts the video).")
        elif action == "start_rows":
            self.add_start_rows(node.id)
        elif action == "row_apply":
            self.apply_to_row(node.id)
        elif action == "start_cut":
            self.edit_start_cut(node.id)
        else:
            self.set_row_start(node.id, clear=action == "start_own_clear")

    @_node_action("video", applies="any")
    def _act_video(self, action: str, node: T.Node | None, nid: str | None):
        self._ask_video(node.id if node else None)

    @_node_action("swap", applies="video")
    def _act_swap(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.replace_person(node.id)

    @_node_action("swap_undo", applies="video")
    def _act_swap_undo(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.undo_replacement(node.id)

    @_node_action("edit", applies="video")
    def _act_edit(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.edit_video(node.id)

    @_node_action("split", applies="video")
    def _act_split(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.split_video(node.id)

    @_node_action("end_image", applies="clip")
    def _act_end_image(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.choose_end_image(node.id)

    @_node_action("end_video", applies="clip")
    def _act_end_video(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.end_from_video(node.id)

    @_node_action("end_ai", applies="clip")
    def _act_end_ai(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.edit_image(node.id)

    @_node_action("edit_ai")
    def _act_edit_ai(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.edit_image(node.id, from_node=True)

    @_node_action("end_clear", applies="clip")
    def _act_end_clear(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.set_end_image(node.id, None)

    @_node_action("revert", applies="clip")
    def _act_revert(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.revert_clip(node.id)

    @_node_action("speak", applies="clip")
    def _act_speak(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.speak_clip_window(node.id)

    @_node_action("refine", applies="clip")
    def _act_refine(self, action: str, node: T.Node | None, nid: str | None):
        """Refine the clip's prompt with the language model (the pictures the clip is made from go with it)."""
        assert node is not None and self.tree is not None and self.project is not None
        card = self._card_of(node.id, PromptCard)
        if card is None:
            return
        ck = T.Checker(self.project, self.tree)
        start = ck.start_frame(node)
        end = T.end_image_used(self.tree, node)
        pics = [("The frame the clip starts from", str(start), True)] if start and start.is_file() else []
        for k in node.keyframes:
            if (self.project / k["image"]).is_file():
                pics.append((f"The picture the clip passes through at {k['at']:.1f} s",
                             str(self.project / k["image"]), True))
        if end and (self.project / end).is_file():
            pics.append(("The image the clip should end at", str(self.project / end), True))
        h3 = self.tree.options.get("prompt_format") == "h3"   # the official H3 format: its own refine
        frames = int((node.overrides or {}).get("frames", self.tree.options.get("frames", 73)))
        new = self.refine_text(card.edit.toPlainText(), "video_h3" if h3 else "video", pictures=pics,
                               seconds=frames / chain.FPS, start_image=start,
                               loras=[p for p, _s in T.node_loras(node)])
        if new is not None and new.text:          # (a clip's prompt is one text; ideas are the image window's)
            replace_text(card.edit, new.text)          # Ctrl+Z in the prompt takes it back
            self.status.setText(f"Clip #{T.short_id(node.id)}: prompt refined (Ctrl+Z in the prompt undoes it).")

    @_node_action("handover", applies="clip")
    def _act_handover(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.choose_handover(node.id)

    @_node_action("handover_clear", applies="clip")
    def _act_handover_clear(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.set_handover(node.id, None)

    @_node_action("compare")
    def _act_compare(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        self.compare_variants(node.id)

    @_node_action("variants")
    def _act_variants(self, action: str, node: T.Node | None, nid: str | None):
        assert node is not None
        if node.is_video:
            self.status.setText("Variants are for generated clips; a real video stays as it is.")
            return
        dlg = VariantsDialog(self, node, settings=self.qs)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.create_variants(node.id, dlg.count.value(), dlg.generate.isChecked(),
                                 strength_steps=dlg.strength_steps(), keep_seed=dlg.keep_seed())

    @_node_action("continue", "branch", "variant", applies="any")
    def _act_add_clip(self, action: str, node: T.Node | None, nid: str | None):
        """A new clip: after this one (continue), next to it (branch), or next to it with the same prompt (variant)."""
        assert self.tree is not None
        if action != "continue" and node is None:
            return
        if action == "variant" and node is not None and node.is_video:
            self.status.setText("Variants are for generated clips; a real video stays as it is.")
            return
        if action == "continue":
            new = self.tree.add(node.id if node else None, "")
        else:
            assert node is not None
            new = self.tree.add(node.parent, node.prompt if action == "variant" else "",
                                self._new_seed() if action == "variant" else None)
            if action == "variant":                # the same clip with another seed: its own settings too
                new.overrides = copy.deepcopy(node.overrides)
                self._copy_video_child(node.id, new.id)
                self._copy_row_start(node.id, new.id)
        self.tree.selected = new.id
        self._write_tree()
        self.refresh()
        self.graph.reveal(new.id)
        if self.cards and self.cards[-1].nid == new.id and isinstance(self.cards[-1], PromptCard):
            self.cards[-1].edit.setFocus()

    @_node_action("delete")
    def _act_delete(self, action: str, node: T.Node | None, nid: str | None):
        """Delete the clip with everything after it (its files move to old/ once tree.json is saved)."""
        assert node is not None and self.tree is not None
        if self._busy_files():
            return
        below = len(self.tree.descendants(node.id))
        made = node.made is not None or any(d.made for d in self.tree.descendants(node.id))
        after = f" and the {plural(below, 'clip')} after it" if below else ""
        msg = f"Delete {'video' if node.is_video else 'clip'} #{T.short_id(node.id)}{after}?"
        if made:
            msg += "\nTheir video files are not deleted: they are moved to old/ in the project folder."
        if not D.confirm(self, "Delete", msg):
            return
        self.tree.remove(node.id, self.project, later=True)      # (its files move once tree.json is saved)
        if self.tree.selected is None:
            self.tree.selected = START
        self._write_tree()
        self.refresh()

    @classmethod
    def _is_locked(cls, node: T.Node, action: str) -> bool:
        """The action would change this frozen clip / video. (A video may follow a frozen video: nothing of the
        frozen one changes then.)"""
        if not node.frozen or action not in cls.FROZEN_LOCKED:
            return False
        return not (action == "video" and node.is_video)

    def _frozen_block(self, node: T.Node, action: str) -> bool:
        """True (with a word in the status bar and at the mouse) when the action would change a frozen clip."""
        if not self._is_locked(node, action):
            return False
        what = "Video" if node.is_video else "Clip"
        text = (f"{what} #{T.short_id(node.id)} is frozen (kept as made): unfreeze it first (the lock button on its "
                "card, or right-click → Unfreeze).")
        self.status.setText(text)
        # (where the user just pressed: the status bar is easy to miss)
        QToolTip.showText(QCursor.pos(), text, self)
        return True

    def freeze_clip(self, nid: str, confirm: bool = True) -> list[str]:
        """Keep a made clip as it is, with every clip before it (tree.freeze). Returns the ids frozen now."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        what = "video" if self.tree.nodes[nid].is_video else "clip"
        assert self.project is not None
        why = T.freeze_problem(self.project, self.tree, nid)
        if why:
            self._report_error(f"Cannot freeze {what} #{T.short_id(nid)}: {why}.")
            return []
        edits = T.freeze_edits(self.tree, nid)
        if confirm and edits and not D.confirm(
                self, "Freeze",
                f"Freeze {what} #{T.short_id(nid)} and the clips before it?\n\nThese edits are not made yet and are "
                "undone (back to what the clips were made with):\n" +
                "\n".join(f"#{T.short_id(i)}: {', '.join(e)}" for i, e in edits.items())):
            return []
        try:
            ids = T.freeze(self.project, self.tree, nid)
        except (ValueError, OSError) as e:
            self._report_error(f"Cannot freeze {what} #{T.short_id(nid)}: {e}.")
            return []
        self._pix.clear()
        self._write_tree()
        self.refresh(rebuild_path=True)
        more = [i for i in ids if i != nid]
        self.status.setText(f"{what.capitalize()} #{T.short_id(nid)} frozen" +
                            (f", with {len(more)} before it" if more else "") +
                            ": kept as made whatever the project's settings say; the clips after it follow them.")
        return ids

    def unfreeze_clip(self, nid: str) -> list[str]:
        """A frozen clip (and the frozen clips after it) follows the settings again. Returns the ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        ids = T.unfreeze(self.tree, nid)
        if not ids:
            return []
        self._write_tree()
        self.refresh(rebuild_path=True)
        changed = [i for i in ids if self.status_of.get(i) == "changed"]
        more = [i for i in ids if i != nid]
        text = (f"#{T.short_id(nid)} unfrozen" + (f", with {', '.join('#' + T.short_id(i) for i in more)}" if more
                                                  else ""))
        if changed:
            text += (f"; {len(changed)} of them count{'s' if len(changed) == 1 else ''} as CHANGED now (the settings "
                     "differ): Generate makes them again. Freeze again to keep them.")
        self.status.setText(text + ".")
        return ids

    def _busy_files(self, what: str = "delete") -> bool:
        """A path is being joined or a video converted (they read / write clip files): no deleting (or changing
        hand-over frames) meanwhile."""
        if self._joins:
            self.status.setText(f"Wait until the {'sound is mixed again' if self._remixing else 'path is joined'}, "
                                f"then {what}.")
        elif self._converts:
            self.status.setText(f"Wait until the video is converted, then {what}.")
        return bool(self._joins or self._converts)

    def _copy_video_child(self, source: str, target: str) -> None:
        """A variant of a clip that leads into a real video leads into its own copy of that video; one with an
        end image gets its own copy of the image."""
        assert self.tree is not None
        vc = self.tree.video_child(source)
        if vc is not None and self.tree.can_add_video(target):
            assert self.project is not None
            T.copy_video(self.project, self.tree, vc, target)
        try:
            assert self.project is not None
            T.copy_end_image(self.project, self.tree, source, target)
            T.copy_keyframes(self.project, self.tree, source, target)
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not copy the end image or the keyframes: {e}")

    def _copy_row_start(self, source: str, target: str) -> None:
        """A variant of a first clip with its own start image starts from its own copy of that picture."""
        try:
            assert self.tree is not None
            assert self.project is not None
            T.copy_node_start(self.project, self.tree, source, target)
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not copy the start image: {e}")

    def revert_clip(self, nid: str) -> list[str]:
        """Undo the edits of a made clip (prompt, seed, own settings, end image) so it counts as done again."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        assert self.project is not None
        done = T.revert(self.project, self.tree, nid)
        if not done:
            return []
        self._pix.clear()
        self._write_tree()
        self.refresh(rebuild_path=True)
        why = self.why_of.get(nid)
        self.status.setText(f"Clip #{T.short_id(nid)}: {' and '.join(done)} back as made" +
                            (f"; it is still changed: {why}." if why else "; it is done again."))
        return done

    # ---- compare
    def compare_candidates(self, nid: str) -> list[str]:
        """The made clips from the same point as nid (nid first), in number order."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None:
            return []
        memo = self._cmp_memo                      # (while the cards are updated: sorted once per point, not per card)
        if memo is not None and node.parent in memo:
            made = memo[node.parent]
        else:
            tree = self.tree
            assert tree is not None
            sibs = [n for n in tree.children(node.parent) if n.made and not n.is_video
                    and self.status_of.get(n.id) == "done"]
            made = [n.id for n in sorted(sibs, key=lambda n: tree.number(n.id))]
            if memo is not None:
                memo[node.parent] = made
        own = [nid] if node.made else []
        return own + [i for i in made if i != nid]

    def compare_variants(self, nid: str, player: bool = True) -> CompareDialog | None:
        """Play nid and the other made clips from the same point side by side; 'Keep this one' deletes the others
        (and what follows them)."""
        if self.tree is None or self.running:
            return None
        ids = self.compare_candidates(nid)[:9]
        if len(ids) < 2:
            self.status.setText("Nothing to compare: make variants of this clip first (Variant / Variants…).")
            return None
        items = []
        assert self.project is not None
        for i in ids:
            n = self.tree.nodes[i]
            detail = " · ".join([f"seed {n.seed}"] + self._own_parts(n) +
                                ([] if n.prompt == self.tree.nodes[nid].prompt else [f"“{n.prompt[:60]}”"]))
            assert n.made is not None                  # (compare_candidates: made clips)
            items.append((i, f"#{T.short_id(i)}", detail, self.project / n.made["video"]))
        self.preview.stop()
        tree = self.tree

        def keep(kid: str) -> bool:
            others = [i for i in ids if i != kid and i in tree.nodes]
            return bool(self.keep_only(kid, others))
        dlg = CompareDialog(self, items, keep, player=player,
                            title=f"Compare {len(ids)} clips from the same point as #{T.short_id(nid)}",
                            settings=self.qs)
        if player:
            dlg.exec()
            dlg.deleteLater()                       # its video players
        return dlg

    def _delete_others(self, nid: str, roots: list[str], title: str, question, done: str,
                       confirm: bool) -> list[str]:
        """The shared part of keep_only / keep_branch: delete the branches that start at roots (with what follows
        them), keep nid and select it. question(doomed): the text asked (before it, the made clips are counted);
        done(gone): the status line. Returns the deleted ids."""
        assert self.tree is not None
        doomed = roots + [d.id for r in roots for d in self.tree.descendants(r)]
        made = sum(1 for d in doomed if self.tree.nodes[d].made)
        if confirm:
            msg = question(doomed)
            if made:
                msg += (f"\n{made} of them are made; their video files are not deleted: they are moved to old/ in the "
                        "project folder.")
            if not D.confirm(self, title, msg):
                return []
        gone = []
        for r in roots:
            if r in self.tree.nodes:
                gone += self.tree.remove(r, self.project, later=True)
        self.tree.selected = nid
        self._write_tree()
        self.refresh()
        self.status.setText(f"Kept clip #{T.short_id(nid)}; deleted {plural(len(gone), 'clip')}{done}.")
        return gone

    def keep_only(self, nid: str, others: list[str], confirm: bool = True) -> list[str]:
        """Delete these other clips (with what follows them); nid stays and is selected."""
        if self.tree is None or self.running or not others or self._busy_files():
            return []

        def question(doomed: list[str]) -> str:
            after = " and what follows them" if len(doomed) > len(others) else ""
            return (f"Keep clip #{T.short_id(nid)} and delete {plural(len(doomed), 'clip')} "
                    f"({', '.join('#' + T.short_id(o) for o in others)}{after})?")
        return self._delete_others(nid, others, "Keep this one", question, "", confirm)

    def _activate(self, nid: str):
        """Double-click on a node: a real video opens its range editor, a generated clip plays."""
        if row_start_of(nid):
            return
        if self.tree is not None and nid in self.tree.nodes and self.tree.nodes[nid].is_video and not self.running:
            QTimer.singleShot(0, lambda: self.edit_video(nid))
        else:
            self._play_node(nid)

    def keep_branch(self, nid: str, whole_path: bool = False, confirm: bool = True) -> list[str]:
        """Delete the other branches at this clip's point (its siblings), or along its whole path; the clip and
        what follows it stay. Returns the deleted ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        roots = self.tree.others_than_path(nid) if whole_path else [n.id for n in self.tree.siblings(nid)]
        if not roots or self._busy_files():
            return []
        where = "along the path to" if whole_path else "next to"

        def question(doomed: list[str]) -> str:
            return (f"Delete {plural(len(doomed), 'clip')} in {plural(len(roots), 'other branch', 'other branches')} "
                    f"{where} clip #{T.short_id(nid)}?")
        return self._delete_others(nid, roots, "Keep this branch", question, " in the other branches", confirm)

    def create_variants(self, nid: str, count: int, generate: bool = True, strength_steps: list[float] | None = None,
                        keep_seed: bool = False) -> list[str]:
        """count new branches next to nid: same prompt and clip settings, each its own new seed; then (optionally)
        generate them in one run. strength_steps: per LoRA of the clip's stack, variant k gets its strength + k ×
        step; keep_seed: the clip's seed instead of new ones. Returns the new node ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        node = self.tree.nodes[nid]
        ids = []
        if node.is_video:
            return []
        steps = list(strength_steps or [])
        stack = list((node.overrides or {}).get("loras") or [])
        loras = [e for e in stack if e.get("path")]
        for k in range(1, max(1, int(count)) + 1):
            new = self.tree.add(node.parent, node.prompt, node.seed if keep_seed else self._new_seed())
            new.overrides = copy.deepcopy(node.overrides)
            if any(steps) and loras:
                new.overrides["loras"] = variant_strengths(stack, steps, k)   # (every entry kept)
            self._copy_video_child(nid, new.id)
            self._copy_row_start(nid, new.id)
            ids.append(new.id)
        self._write_tree()
        self.refresh()
        self.graph.reveal(ids[-1])
        sweep = " with stepped LoRA strengths" if any(steps) and loras else ""
        self.status.setText(f"{plural(len(ids), 'variant')} of clip #{T.short_id(nid)} added{sweep}; once "
                            "made, compare them side by side with the Compare button on the card.")
        if generate:
            self._start(ids)
        return ids

    def _context(self, nid: str, pos):
        """The right-click menu of a node. Every entry is added with what it does (a node action's name from
        NODE_ACTIONS, or a function); one dispatch runs the chosen one."""
        if self.tree is None:
            return
        own = row_start_of(nid)
        m = QMenu(self)
        calls: dict[QAction, Callable[[], None]] = {}
        lock = self.tree.nodes.get(own or nid)      # (a frozen clip refuses the node actions that would change it)

        def add(text: str, does: str | Callable[[], None], enabled: bool = True, tip: str = "") -> QAction:
            act = m.addAction(text)
            if isinstance(does, str):
                target = own or nid                 # (a row's start image acts on the row's clip)
                calls[act] = lambda: self.node_action(does, target)
                enabled = enabled and not (lock is not None and self._is_locked(lock, does))
            else:
                calls[act] = does
            act.setEnabled(enabled and not self.running)        # (a run locks the project: greyed, not hidden)
            if tip:
                act.setToolTip(tip)
            return act

        if own:                                    # a row's start image
            if own not in self.tree.nodes:
                return
            add("Another picture for this row…", "start_own")
            add("Cut of this picture… (the part the clip uses)", "start_cut")
            add("Start from the project's start image instead", "start_own_clear")
            add("More start images… (the same clip for each)", "start_rows")
            extra: dict = {}
        else:
            node = self.tree.nodes.get(nid)
            add("Continue from here", "continue")
            add("Add a real video after this…", "video", self.tree.can_add_video(None if nid == START else nid))
            if nid != START:
                self._node_entries(m, add, nid, node)
            extra = self.plugins.context_actions(m, nid) if nid != START else {}   # (the plugins' clip actions)
        chosen = self._exec_menu(m, pos.toPoint() if hasattr(pos, "toPoint") else pos)
        if chosen is None:
            return
        # run it later: this is still inside the node's mouse handler, and the action may rebuild the graph
        # (deleting that very node)
        if chosen in extra:
            QTimer.singleShot(0, extra[chosen])
        elif chosen in calls:
            QTimer.singleShot(0, calls[chosen])

    def _node_entries(self, m: QMenu, add: Callable[..., QAction], nid: str, node: T.Node | None):
        """The entries of a clip's or video's menu (not of START)."""
        assert self.tree is not None
        if node is not None and node.is_video:
            add("Edit range…", "edit")
            add("Split here and insert a clip…", "split", bool(node.made) and self.state(nid) == "done",
                "Put a generated clip into the middle of this video: the video ends at a frame you choose, the clip "
                "follows, then the rest of the video")
            add("Replace a person…", "swap")
            if swap.replaced(node):
                add("Back to the original video (undo the replacement)", "swap_undo")
        add("Branch (another continuation)", "branch")
        if node is not None and not node.is_video:
            vc = self.tree.video_child(nid)
            add("Variant (same prompt, other seed)", "variant")
            add("Variants… (several, generated at once)", "variants")
            if node.parent is None:
                add("More start images… (the same clip for each)", "start_rows")
                if len(T.row_members(self.tree, nid)) > 1:
                    add("Give this prompt and settings to its other start images", "row_apply")
                add("Its own start image…", "start_own")
                if self.project is not None and T.start_source(self.project, node) is not None:
                    add("Cut of its start image… (the part the clip uses)", "start_cut")
                if node.start_image:
                    add("Start from the project's start image instead", "start_own_clear")
            add("End image…" if vc is None else f"End image… (ends at video #{T.short_id(vc.id)})", "end_image",
                vc is None)
            add("End image from a video frame…", "end_video", vc is None)
            if node.end_image:
                add("Clear end image", "end_clear")
            if T.changed_inputs(self.tree, nid):
                add("Undo changes (back to the made clip)", "revert")
            add("Hand-over frame…", "handover", bool(node.made) and vc is None)
            if T.handover_index(self.tree, node) is not None:
                add("Hand over at the last frame", "handover_clear")
            lines = bool(T.spoken_lines(node.prompt))
            add("Speak the lines with text to speech" if lines else
                "Speak the lines with text to speech (write a line in quotes first)", "speak",
                not self._tts_busy and lines)
            add("Edit an image with AI…", "edit_ai")
            add("Compare with its variants…", "compare", len(self.compare_candidates(nid)) > 1)
        if node is not None:
            if node.frozen:
                add("Unfreeze (edit it, follow the settings again)", "unfreeze")
            else:
                add("Freeze (keep it and the clips before it as made)", "freeze", self.state(nid) == "done",
                    FREEZE_TIP)
        m.addSeparator()
        made = bool(self.tree.nodes[nid].made)
        video = node is not None and node.is_video
        add("Play this video" if video else "Play this clip", lambda: self._play_node(nid), made)
        add("Export this video…" if video else "Export this clip…", lambda: self.export_path(nid),
            made and self.state(nid) == "done")
        ref = "face reference" if self._ref_word() == "face picture" else \
            f"picture of @{self._selected_character()['name']}"
        if video:
            add(f"Add first frame as {ref}", lambda: self.reference_from_node(nid, "first"), made)
        add(f"Add last frame as {ref}", lambda: self.reference_from_node(nid, "last"),
            bool(node is not None and node.made))
        add("Generate the path to here", self.generate_path)
        if self.film is not None:
            assert self.project is not None
            cur = self.film.scene_of(self.project)
            if cur is not None and cur.take == nid:
                add("Clear the scene's ending (the film plays its selected path)", "take_clear")
            else:
                add("End the scene here (the film plays up to this clip)", "take")
        m.addSeparator()
        add("Keep this branch, delete the other branches here", lambda: self.keep_branch(nid, whole_path=False),
            bool(self.tree.siblings(nid)))
        add("Keep only this path, delete all other branches", lambda: self.keep_branch(nid, whole_path=True),
            bool(self.tree.others_than_path(nid)))
        add("Delete (with everything after it)", "delete")

    def _exec_menu(self, menu: QMenu, pos):
        return menu.exec(pos)
