"""Real videos in the chain: add, edit, convert, replace a person, split."""
from __future__ import annotations

import copy
import os
from pathlib import Path

from PySide6.QtWidgets import QDialog

from .workers import run_in_background
from .. import animate, chain, comfy as CMF, swap, tree as T
from . import dialogs as D, image_picker as IP
from .clip_dialogs import SplitDialog
from .swap_dialog import SwapDialog
from .video_editor import VideoRangeDialog, choose_video
from .generator_state import ModelSwitch
from .ui_logic import DEFAULT_COMFY_URL, DEFAULT_SIZE
from .mw_host import MainWindowHost


class VideosMixin(MainWindowHost):
    """Real videos in the chain: add, edit, convert, replace a person, split."""

    def _ask_video(self, parent: str | None):
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return
        assert self.tree is not None
        if not self.tree.can_add_video(parent):
            vc = self.tree.video_child(parent)
            assert vc is not None                      # (what can_add_video refused)
            self.status.setText(f"Clip #{T.short_id(str(parent))} already leads into video #{T.short_id(vc.id)}: a "
                                "generated clip can "
                                "end at one video only. Branch from the clip before it for another one.")
            return
        f = choose_video(self, self.qs, places=self._picker_places(), collections=self.project_media("video"))
        if not f:
            return
        dlg = VideoRangeDialog(self, f, canvas=self._canvas_for(f), settings=self.qs,
                               canvas_for=lambda file, rot: self._canvas_for(file, rot))
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.file:
            return
        self.add_video(parent, dlg.file, dlg.start, dlg.end, dlg.rotate)

    def add_video(self, parent: str | None, file: str, start: float, end: float, rotate: int = 0) -> str | None:
        """Put a range of a real video (turned clockwise by rotate degrees) after parent (None: at the start) and
        convert it right away."""
        if self.tree is None or self.running:
            return None
        try:
            node = self.tree.add_video(parent, os.path.abspath(file), start, end, rotate)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self._remember_video(file)
        self.tree.selected = node.id
        self._write_tree()
        self._show_start_image()                  # the size may follow this video now
        self.refresh()
        self.graph.reveal(node.id)
        self.convert_video(node.id)
        return node.id

    def edit_video(self, nid: str):
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running or self._frozen_block(node, "edit"):
            return
        src = node.source_dict()
        f = src["file"]
        if not Path(f).is_file():
            D.tell(self, "Video missing", f"{Path(f).name} is not there any more; choose the file "
               "again in the next window.")
        rot = T.source_rotation(node.source)
        dlg = VideoRangeDialog(self, f if Path(f).is_file() else "", src["start"], src["end"],
                               canvas=self._canvas_for(f, rot, nid), settings=self.qs, on_reference=self.add_reference,
                               rotate=rot, canvas_for=lambda file, r: self._canvas_for(file, r, nid))
        sel = self._selected_character()
        if sel is not None and T.kind_of(sel) != "person":
            dlg.set_reference_kind(sel["name"], T.kind_of(sel))
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.file:
            return
        self.set_video_range(nid, dlg.file, dlg.start, dlg.end, dlg.rotate)

    def set_video_range(self, nid: str, file: str, start: float, end: float, rotate: int | None = None) -> None:
        """A video node's file, range and rotation (None: as it is turned now)."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return
        node = self.tree.nodes[nid]
        if rotate is None:
            rotate = T.source_rotation(node.source)
        new = T.video_source(os.path.abspath(file), start, end, rotate)
        if new["end"] <= new["start"]:
            self.status.setText("The end of the range must be after its start.")
            return
        if new == node.source:
            return
        before = self._done_clips() - {nid}
        node.source = new
        self._remember_video(file)
        self._write_tree()
        self._show_start_image()
        self.refresh(rebuild_path=True)
        self.convert_video(nid)
        if not self._report_remade(before, f"Video #{T.short_id(nid)} has a new range (converting)"):
            self.status.setText(f"Video #{T.short_id(nid)} has a new range: converting.")

    def _remember_video(self, file: str):
        self.qs.setValue("videoDir", str(Path(file).parent))

    def convert_video(self, nid: str) -> None:
        """Convert a video node's range in the background (the generator does it too if this is not done)."""
        if self.tree is None or nid not in self.tree.nodes:
            return
        node = self.tree.nodes[nid]
        try:
            assert self.project is not None
            checker = T.Checker(self.project, self.tree)
            base = checker.base()
            if base is None or checker.status(nid) == "done" or not Path(node.source_dict()["file"]).is_file():
                return
        except Exception as e:  # noqa: BLE001
            self._report_error(f"Could not convert the video: {e}")
            return
        token = os.urandom(6).hex()
        self._converts[nid] = token
        self.live[nid] = "converting"
        work_node, project = copy.deepcopy(node), self.project
        self.status.setText(f"Converting video #{T.short_id(nid)} …")

        run_in_background(lambda: T.make_video(project, work_node, base),
                          on_done=lambda made: self._converted(nid, made, "", token, str(project)),   # (the answer ends "converting")
                          on_error=lambda e: self._converted(nid, None, str(e), token, str(project)), owner=self)
        self.refresh()

    # ---- a sound-only change: the sound is mixed again from the model's kept sound (no new generation)
    def _start_remix(self):
        """Mix the sound of every clip that only needs that again (Checker.remix_due), off the GUI thread; one job
        at a time, and like a join (it rewrites clip files): no generation, deleting or other join meanwhile."""
        if self.tree is None or self.project is None or self.running or self._remixing:
            return
        try:
            ck = T.Checker(self.project, self.tree)
            due = sorted((n for n in self.tree.nodes if ck.remix_due(n)), key=self.tree.depth)
        except Exception:  # noqa: BLE001  (refresh reports what cannot be checked)
            return
        if not due:
            return
        tree, project = T.Tree.from_json(self.tree.to_json()), self.project
        order = sorted(tree.nodes, key=tree.depth)           # (a clip before the ones after it: what repeats it)
        lock = self._join_lock
        self._remixing = True
        self._joins += 1                                     # (generate, delete, join: they wait for it)
        self._update_buttons()
        self.status.setText(f"Mixing the sound of clip #{T.short_id(due[0])} again …")

        revs = {n: (tree.nodes[n].made or {}).get("sound_rev") for n in order}

        def work():
            with lock:
                error = ""
                try:
                    T.remix_sounds(project, tree, order, T.Checker(project, tree), persist=False)
                except BaseException as e:  # noqa: BLE001  (the ones remixed before it are still taken)
                    error = str(e) or type(e).__name__
                done = [n for n in order if (tree.nodes[n].made or {}).get("sound_rev") != revs[n]]
                return [(n, copy.deepcopy(tree.nodes[n].made)) for n in done], error

        run_in_background(work, on_done=lambda r: self._remixed(r[0], r[1], str(project)),
                          on_error=lambda e: self._remixed([], str(e) or type(e).__name__, str(project)), owner=self)

    def _remixed(self, results, error: str, project: str):
        """Take the remixed clips' records (only for the clip version that was remixed: one made again meanwhile
        keeps its own) and say what follows from it: a clip that is made to that sound, or carries over from that
        clip, is made again with the path."""
        self._remixing = False
        self._joins = max(0, self._joins - 1)
        self._update_buttons()
        if project != str(self.project) or self.tree is None:
            return
        before = dict(self.status_of)
        taken = []
        for nid, made in results:
            node = self.tree.nodes.get(nid)
            if node is not None and node.made and node.made.get("video") == made.get("video"):
                node.made = made
                taken.append(nid)
        if taken:
            self._changed()
            self.refresh()
        if error:
            self._remix_timer.stop()
            self._remix_stuck = set(self.remix_due)          # (not tried again by itself until something changes)
            self._report_error(f"Could not mix the sound again: {error}")
            return
        if not taken:
            return
        names = ", ".join(f"#{T.short_id(n)}" for n in taken)
        text = f"Clip{'s' if len(taken) > 1 else ''} {names}: sound mixed again (no new generation)."
        after = [n for n in self.tree.nodes if before.get(n) == "done" and self.status_of.get(n) == "changed"]
        if after:
            text += (f" Clip{'s' if len(after) > 1 else ''} {', '.join('#' + T.short_id(n) for n in after)} "
                     f"{'use' if len(after) > 1 else 'uses'} that sound (made to it, or carried over from it) and "
                     f"{'are' if len(after) > 1 else 'is'} made again with the path.")
        self.status.setText(text)

    def _converted(self, nid: str, made, error: str, token: str, project: str):
        def discard():
            for k in ("video", "frame", "first"):
                if made and made.get(k):
                    chain.keep_aside(Path(project), Path(project) / made[k])
        if self._converts.get(nid) != token:         # a newer conversion of this node is running
            discard()
            return
        self._converts.pop(nid, None)
        if self.live.get(nid) == "converting":
            self.live.pop(nid, None)
        node = self.tree.nodes.get(nid) if self.tree is not None else None
        if project != str(self.project) or node is None or self.running:
            discard()
            self._update_buttons()
            return
        if error:
            self._report_error(f"Could not convert video #{T.short_id(nid)}: {error}")
            self.refresh()
            return
        key = made.get("key") or {}
        if {k: key.get(k) for k in ("file", "start", "end")} != {k: (node.source or {}).get(k)
                                                               for k in ("file", "start", "end")} \
                or chain.rotation(key.get("rotate", 0)) != T.source_rotation(node.source):   # changed meanwhile
            discard()
            self.refresh()
            return
        assert self.project is not None
        old, old_made = T.clip_files(self.project, node), node.made
        node.made = made
        if not self._write_tree():                  # (the status says why) the record stays as it was
            node.made = old_made
            discard()
            self.refresh()
            return
        if not self._joins:                         # a running join may still read them; else the next run does
            for f in old:                           # (never deleted: moved to old/)
                chain.keep_aside(self.project, f)
        self._pix.clear()
        self.status.setText(f"Video #{T.short_id(nid)} is ready ({made['frames']} frames).")
        self.refresh()

    # ---- replace a person in a real video (Wan 2.2 Animate)
    def replace_person(self, nid: str):
        """The swap window for a video node (its converted clip). Not run (returned) when modal_dialogs is off."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running:
            return None
        try:
            assert self.project is not None
            assert self.tree is not None
            done = T.Checker(self.project, self.tree).status(nid) == "done"
        except Exception:  # noqa: BLE001
            done = False
        assert self.project is not None
        if not done or not node.made or not (self.project / node.made["video"]).is_file():
            self.status.setText(f"Video #{T.short_id(nid)} is not converted yet: generate its path (or wait for the "
                                "conversion), then replace the person.")
            return None
        pictures = []
        assert self.tree is not None
        for c in T.cast_of(self.tree):
            if T.kind_of(c) != "person":            # (a head swap takes faces)
                continue
            for k, rel in enumerate(c["images"], start=1):
                assert self.project is not None
                if (self.project / rel).is_file():
                    pictures.append((f"Face of @{c['name']}" + (f" ({k})" if len(c["images"]) > 1 else ""),
                                     str(self.project / rel)))
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL)
        comfy_dir = self._comfy_dir_path()
        cleanup = self.comfy_cleanup.isChecked()
        switch = ModelSwitch()

        def before():                               # GUI thread: the clip model makes room
            self._qwen_timer.stop()
            switch.prepare(self.comfy.model)
            if self.worker_alive and not self.running:
                self._quit_worker(free=False)

        def connect(progress):                      # worker thread
            an = animate.ComfyAnimator(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir,
                                       cleanup=cleanup)
            switch.loaded = True                    # another model in ComfyUI now (unloaded after the window)
            switch.free_clip_model(an)
            return an

        def describe_face(picture):                 # worker thread: the face details for the head swap
            switch.loaded = True
            return CMF.ComfyPromptRefiner(url, log=lambda _m: None, comfy_dir=comfy_dir, cleanup=cleanup).describe_face(
                picture)

        def connect_h3(progress):                   # worker thread: the clip generator (H3 head swap)
            gen = CMF.ComfyGenerator(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            switch.free_clip_model(gen)
            switch.loaded = True
            return gen

        assert self.project is not None and node.made is not None       # (checked above)
        dlg = SwapDialog(self, self.project, nid, self.project / node.made["video"], self.face_detector(), pictures,
                         connect, self.qs, on_start=before,
                         source=dict(node.source_dict()), h3={
                             "lora": self.find_lora("head_swap"),
                             "turbo": self.find_lora(swap.H3_SWAP_TURBO),
                             "settings": T.current_settings(self.project, self.tree.options, self.tree),
                             "connect": connect_h3, "describe": describe_face},
                         groups=[[str(self.project / rel) for rel in c["images"]] for c in T.cast_of(self.tree)],
                         pick=lambda title: IP.pick_image(
                             self, title, self.qs, None, self._picker_places(), self.project_images()))
        self._edit_switch = switch                  # (as the image window: the model is unloaded after it)
        if not self.modal_dialogs:
            return dlg
        try:
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result is not None:
                self.use_replacement(nid, *dlg.result)
        finally:
            self.edits_closed()
        return None

    def use_replacement(self, nid: str, file: Path, meta: dict, frames: int) -> bool:
        """The video node plays the clip with the replaced person (converted again; the clips after it change)."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running:
            return False
        before = self._done_clips() - {nid}
        if not Path(file).is_file():
            self.status.setText(f"The made clip is gone ({Path(file).name}).")
            return False
        swap.apply(node, file, frames, {k: v for k, v in meta.items() if k in (
            "reference", "head_size", "size", "gen_size", "seed", "prompt", "model", "made", "method", "details",
            "lora")}, fps=chain.FPS)
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.convert_video(nid)
        if not self._report_remade(before, f"Video #{T.short_id(nid)}: person replaced"):
            self.status.setText(f"Video #{T.short_id(nid)} plays the clip with the replaced person now (its menu "
                                "goes back to the original).")
        return True

    def split_video(self, nid: str, at: float | None = None) -> str | None:
        """Insert a generated clip into a real video: choose the frame (at: the moment in its file, else asked),
        the video ends there, a new clip follows, then the rest of the video. Returns the new clip's id."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or not node.is_video or self.running:
            return None
        if not node.made or self.state(nid) != "done":
            self.status.setText(f"Video #{T.short_id(nid)} is not converted yet; wait for it, then split it.")
            return None
        if at is None:
            assert self.project is not None
            clip = self.project / node.made["video"]
            try:
                frames = int(node.made.get("frames") or chain.probe(clip)["frames"])
            except (OSError, RuntimeError) as e:       # (media.MediaError is a RuntimeError)
                self._report_error(f"Cannot read video #{T.short_id(nid)}: {e}")
                return None
            key = node.made.get("key") or {}
            size = (int(key.get("width") or 0) or DEFAULT_SIZE[0], int(key.get("height") or 0) or DEFAULT_SIZE[1])
            dlg = SplitDialog(self, clip, frames, size, node.source_dict()["start"], node.source_dict()["end"],
                              f"Split video #{T.short_id(nid)}")
            try:
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return None
                at = dlg.at()
            finally:
                dlg.deleteLater()
        before = self._done_clips() - {nid}
        try:
            assert self.tree is not None
            clip_node, rest = self.tree.split_video(nid, at)
        except ValueError as e:
            self._report_error(f"Could not split video #{T.short_id(nid)}: {e}")
            return None
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.select(clip_node.id)
        self.convert_video(nid)
        self.convert_video(rest.id)
        if not self._report_remade(before, f"Video #{T.short_id(nid)} split"):
            self.status.setText(f"Video #{T.short_id(nid)} split at {at:.2f} s: write the prompt of the new clip "
                                f"#{T.short_id(clip_node.id)}; it ends at the first frame of video "
                                f"#{T.short_id(rest.id)}.")
        return clip_node.id

    def undo_replacement(self, nid: str) -> bool:
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or self.running or not swap.restore(node):
            return False
        before = self._done_clips() - {nid}
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.convert_video(nid)
        if not self._report_remade(before, f"Video #{T.short_id(nid)}: back to the original"):
            self.status.setText(f"Video #{T.short_id(nid)} plays the original video again.")
        return True

    def _play_node(self, nid: str):
        if self.tree is None or nid not in self.tree.nodes:
            return
        n = self.tree.nodes[nid]
        assert self.project is not None
        if n.made and (self.project / n.made["video"]).is_file():
            what = Path(n.source_dict()["file"]).name if n.is_video else n.prompt[:60]
            self.preview.open(self.project / n.made["video"],
                              f"{'Video' if n.is_video else 'Clip'} #{T.short_id(nid)} · {what}")
