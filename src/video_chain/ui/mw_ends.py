"""Start, end and key pictures of a clip, editing a picture, the clip's handover."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog

from .. import chain, comfy as CMF, imagegen, llm_server as LLM, media, tree as T
from . import dialogs as D, shortcuts as S, image_picker as IP, prompt_refiner as PR
from .clip_dialogs import HandoverDialog, VideoFrameDialog
from .image_edit import ImageEditDialog, record_use
from .video_editor import choose_video
from .generator_state import CLIP_MODELS, ComfyModel
from .ui_logic import ASPECT_TOLERANCE, DEFAULT_COMFY_URL, label_loras


class EndsMixin:
    """Start, end and key pictures of a clip, editing a picture, the clip's handover."""

    # ---- prompt refiner
    def refine_text(self, text: str, kind: str, pictures: list | None = None, seconds: float | None = None,
                    start_image: Path | None = None, loras: list[str] | None = None) -> str | list[str] | None:
        """Open the prompt refiner for a prompt ('video' / 'video_h3' / 'image' / 'edit'); the text to use, or None.
        seconds, start_image: the clip's length and the frame it starts from (used by 'video_h3'). loras: the
        paths of the LoRAs the result is made with (the refiner is told what is known of them)."""
        if not text.strip():
            self.status.setText("Write a prompt first; the refiner improves it.")
            return None
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL)
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None

        llama = self.llama_server()

        def connect():
            if llama is None:
                return CMF.ComfyPromptRefiner(url, log=lambda _m: None, comfy_dir=comfy_dir, cleanup=False)
            try:
                comfy, err = CMF.ComfyPromptRefiner(url, log=lambda _m: None, comfy_dir=comfy_dir, cleanup=False), ""
            except Exception as e:  # noqa: BLE001  (ComfyUI down: the llama.cpp model still refines)
                comfy, err = None, str(e)
            return LLM.CombinedRefiner(comfy, llama, err)
        cast = T.refiner_cast(self.tree) if self.tree is not None and kind in ("video", "video_h3") else []
        dlg = PR.RefineDialog(self, text, kind, connect, self.qs, start_comfy=self.start_comfy, cast=cast,
                           pictures=pictures or [], seconds=seconds, start_image=start_image, loras=loras)
        try:
            if dlg.exec() != QDialog.Accepted:
                return None
            ideas = dlg.chosen_ideas() if hasattr(dlg, "chosen_ideas") else None
            return ideas if ideas else dlg.chosen()       # (a list: all ideas, for the image window's variations)
        finally:
            # the llama.cpp model goes with the window: it slows image / clip generation down while it is loaded
            self.stop_llama("the prompt refiner was closed")
            if dlg.refined_any and not self.running and self._comfy_model != ComfyModel.QWEN:
                # a language model is in ComfyUI now: unloaded before the next clips, idle time as usual
                self._comfy_model = ComfyModel.LLM
                if self.keep_loaded.isChecked():
                    self._qwen_timer.start(self.idle_minutes.value() * 60_000)
                self._update_model_label()
            dlg.deleteLater()

    # ---- AI image edit
    def edit_image(self, nid: str | None = None, from_node: bool = False, start: bool = False,
                   dialog: bool = True, mode: str = "auto", character: str | None = None) -> ImageEditDialog | None:
        """The image edit dialog. nid: the clip whose end image a result can become (its frame is offered first);
        start: the start image first; character: opened for a face picture of this character (its name on the
        face button). Returns the dialog (dialog=False: not shown, for tests)."""
        if self.project is None or self.tree is None:
            self.status.setText("Open or create a project folder first.")
            return None
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return None
        if self._waiting_comfy is not None:
            self.status.setText("Clips are generated as soon as ComfyUI runs; wait for them (or Stop) first.")
            return None
        node = self.tree.nodes.get(nid) if nid else None
        sources: list[tuple[str, str]] = []

        def add(text, path):
            if path is not None and Path(path).is_file() and str(path) not in [p for _t, p in sources]:
                sources.append((text, str(path)))
        if node is not None and node.made:
            f = self._node_frame(node)
            add(("First frame of video" if node.is_video else
                 "Hand-over frame of clip" if T.handover_index(self.tree, node) is not None else
                 "Last frame of clip") + f" #{T.short_id(node.id)}",
                self.project / node.made["first"] if node.is_video else f)
            if node.is_video:
                add(f"Last frame of video #{T.short_id(node.id)}", self.project / node.made["frame"])
        if node is not None and node.end_image:
            add(f"End image of clip #{T.short_id(node.id)}", self.project / node.end_image)
        add("Start image", self.project / T.START_FILE)
        for n in self.selected_path():
            if n.made and n.id != nid:
                add(f"{'Video' if n.is_video else 'Clip'} #{T.short_id(n.id)}: its last frame",
                    self._node_frame(n) if not n.is_video else self.project / n.made["frame"])
        for c in T.cast_of(self.tree):
            for k, rel in enumerate(c["images"], start=1):
                add(("Face" if T.kind_of(c) == "person" else "Picture") + f" of @{c['name']}"
                    + (f" ({k})" if len(c["images"]) > 1 else ""), self.project / rel)
        first = None
        if start:
            first = str(self.project / T.START_FILE) if (self.project / T.START_FILE).is_file() else None
        target = nid if node is not None and not node.is_video and self.tree.video_child(nid) is None else None
        url = (self.comfy_url.text().strip() or DEFAULT_COMFY_URL)
        comfy_dir = os.path.expanduser(self.comfy_dir.text().strip()) or None
        cleanup = self.comfy_cleanup.isChecked()

        switch = {"free": False, "loaded": False}

        def before_edit():                          # GUI thread, before each Generate
            self._qwen_timer.stop()
            self.stop_llama("its memory is needed for the image")
            # the clip model is loaded: unload it (once: after the first image of this window it is gone)
            switch["free"] = not switch["loaded"] and self._comfy_model in CLIP_MODELS
            if self.worker_alive and not self.running:
                self._quit_worker(free=False)               # its model goes now; it is started again for clips

        def connect(progress):                      # worker thread
            ed = CMF.ComfyImageEditor(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            if switch["free"]:
                ed.free()
                switch["free"] = False
            switch["loaded"] = True
            return ed

        def connect_create(progress):               # worker thread: the image creator (any image family)
            ed = imagegen.ComfyImageCreator(url, log=lambda _m: None, progress=progress, comfy_dir=comfy_dir, cleanup=cleanup)
            if switch["free"]:
                ed.free()
                switch["free"] = False
            if progress is not None:                # (None: only listing the models)
                switch["loaded"] = True             # an image model is loaded (unloaded before the next clips)
            return ed

        def use(kind: str, png: str):
            if self.running:
                self.status.setText("Wait until generating has finished.")
                return False
            if kind == "ref":                       # one more face reference; the dialog stays open
                if not self.add_reference(png):
                    return False
                c = self._selected_character()
                record_use(png, {"kind": "ref", "file": c["images"][-1]})
                return (f"Added as a {'face' if T.kind_of(c) == 'person' else 'picture'} of @{c['name']} (the Cast & "
                        f"references tab, {S.keys('tab_cast')}); mention it in prompts as "
                        f"@{c['name']}." + (f" {self._remade_note}" if self._remade_note else ""))
            ok = self.set_image(png, confirm=True) if kind == "start" else \
                bool(target) and self.set_end_image(target, png)
            if ok:
                record_use(png, {"kind": "start", "sha1": chain.file_signature(self.project / T.START_FILE)}
                           if kind == "start" else {"kind": "end", "file": self.tree.nodes[target].end_image})
                self.status.setText(("New start image from the AI edit" if kind == "start" else
                                     f"Clip #{T.short_id(target)} now ends at the AI-edited image") +
                                    (f" — {self._remade_note}" if self._remade_note else "."))
            return bool(ok)
        known = label_loras(self.qwen_loras)
        family_loras = {fam: label_loras(items) for fam, items in self.image_loras.items()}
        family_loras["qwen"] = known
        aspect = None
        if (self.project / T.START_FILE).is_file():
            img = QImage(str(self.project / T.START_FILE))
            aspect = img.width() / img.height() if img.height() else None
        dlg = ImageEditDialog(self, self.project, sources, first, target, connect, use, self.qs,
                              self._picker_places(), start_comfy=self.start_comfy, on_start=before_edit,
                              known_loras=known, lora_dir=self._lora_dir(),
                              uses=self._uses,
                              refine=self.refine_text, connect_create=connect_create, family_loras=family_loras,
                              start_aspect=aspect, mode=mode, check_lora=self.in_comfy_loras,
                              collections=self.project_images,
                              use_triggers=bool(self.tree.options.get("lora_triggers", True)))
        self._edit_switch = switch
        if character:
            pc = T.character(self.tree, character)
            dlg.b_use_ref.setText(f"Add as {'face' if T.kind_of(pc) == 'person' else 'picture'} of @{character}")
        if dialog:
            dlg.exec()
            dlg.deleteLater()
            self.edits_closed()
        return dlg

    def edits_closed(self):
        """The edit dialog is closed: the image model stays loaded for the idle time ('Keep the model loaded'), else
        it is unloaded now."""
        sw = self._edit_switch or {}
        if sw.get("loaded"):                      # an image job connected (also when it finished after closing)
            self._comfy_model = ComfyModel.QWEN
        if self._comfy_model != ComfyModel.QWEN or self.running:
            return
        if self.keep_loaded.isChecked():
            self._qwen_timer.start(self.idle_minutes.value() * 60_000)
        else:
            self._free_comfy()
        self._update_model_label()

    # ---- hand-over frame
    def choose_handover(self, nid: str) -> None:
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return
        if self.tree.video_child(nid) is not None:
            self.status.setText(f"Clip #{T.short_id(nid)} ends at the video after it; it hands over there.")
            return
        if not node.made or not (self.project / node.made["video"]).is_file():
            self.status.setText(f"Make clip #{T.short_id(nid)} first; then choose the frame it hands over at.")
            return
        # a child with a motion guide needs that many frames up to the hand-over frame
        need = max([int((c.overrides or {}).get("guide") or 0) for c in self.tree.children(nid)
                    if not c.is_video] + [1])
        dlg = HandoverDialog(self, self.project / node.made["video"], f"Hand-over frame of clip #{T.short_id(nid)}",
                             T.handover_index(self.tree, node), min_frame=max(1, need - 1))
        try:
            if dlg.exec() != QDialog.Accepted:
                return
            frame = dlg.frame()
        finally:
            dlg.deleteLater()                       # it holds every decoded frame
        self.set_handover(nid, frame)

    def set_handover(self, nid: str, frame: int | None) -> bool:
        """The clips after a made clip start from this frame of it (None: its last frame, as usual)."""
        if self.tree is None or self.running or nid not in self.tree.nodes or \
                self._frozen_block(self.tree.nodes[nid], "handover") or \
                self._busy_files("change the hand-over frame"):
            return False
        node = self.tree.nodes[nid]
        if node.is_video:
            return False
        if node.made and not node.made.get("frames") and (self.project / node.made["video"]).is_file():
            try:                                    # an older record: the real count, once
                node.made["frames"] = chain.probe(self.project / node.made["video"])["frames"]
            except (OSError, RuntimeError):
                pass
        n = T.made_frames(node) or T.clip_length(self.tree, node)
        frame = None if frame is None or int(frame) >= n - 1 else max(1, int(frame))
        if frame == node.end_at:
            return False
        before = self._done_clips()
        node.end_at = frame
        try:
            T.ensure_handover(self.project, self.tree, node)
        except (OSError, RuntimeError) as e:
            node.end_at = None
            self._report_error(f"Could not read that frame of clip #{T.short_id(nid)}: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self.refresh()
        what = (f"Clip #{T.short_id(nid)} hands over at frame {frame + 1} ({frame / chain.FPS:.2f} s)" if frame is not None
                else f"Clip #{T.short_id(nid)} hands over at its last frame")
        if not self._report_remade(before, what):
            self.status.setText(what + ".")
        return True

    # ---- end images
    def choose_end_image(self, nid: str) -> None:
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return
        if self.tree.video_child(nid) is not None:
            self.status.setText(f"Clip #{T.short_id(nid)} ends at the first frame of the video after it.")
            return
        start = str(Path(self.project / node.end_image).parent) if node.end_image else None
        f = IP.pick_image(self, f"End image of clip #{T.short_id(nid)} (the frame it steers its end to)", self.qs, start,
                       self._picker_places(), self.project_images())
        if f:
            self.set_end_image(nid, f)

    def _known_info(self, video) -> dict | None:
        """What the frame window needs to know of a video, when it is a clip of this project (the record of making it
        has it: no ffprobe on the UI thread, UIL-27); None for any other file (the window reads it itself)."""
        if self.tree is None or self.project is None:
            return None
        for n in self.tree.nodes.values():
            m = n.made
            if m and m.get("video") and m.get("frames") and str(self.project / m["video"]) == str(video):
                key = m.get("key") or {}
                return {"fps": float(chain.FPS), "frames": int(m["frames"]), "duration": int(m["frames"]) / chain.FPS,
                        "width": key.get("width") or "?", "height": key.get("height") or "?"}
        return None

    def end_from_video(self, nid: str, video: str | None = None, dialog: bool = True):
        """End image from a video: choose a video (the project's real videos and clips first, or any file), then
        one of its frames; that frame at the video's full size becomes the clip's end image."""
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return None
        if self.tree.video_child(nid) is not None:
            self.status.setText(f"Clip #{T.short_id(nid)} ends at the first frame of the video after it.")
            return None
        if video is None:
            clips = [(str(self.project / n.made["video"]), f"Clip #{T.short_id(n.id)} · {n.prompt[:40]}")
                     for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id))
                     if not n.is_video and n.made and (self.project / n.made["video"]).is_file() and n.id != nid]
            cols = self.project_media("video") + ([("Clips of this project", clips)] if clips else [])
            video = choose_video(self, self.qs, places=self._picker_places(), collections=cols)
            if not video:
                return None
        try:
            dlg = VideoFrameDialog(self, Path(video), f"End image of clip #{T.short_id(nid)}: a frame of {Path(video).name}",
                                   self._known_info(video))
        except (RuntimeError, OSError) as e:
            self._report_error(f"Cannot read {Path(video).name}: {e}")
            return None
        if not dialog:
            return dlg
        try:
            if dlg.exec() != QDialog.Accepted:
                return None
            with media.scratch("end") as tmp:
                png = Path(tmp) / f"{Path(video).stem}_f{dlg.slider.value()}.png"
                if not dlg.frame_file(png):
                    self._report_error(f"Could not read that frame of {Path(video).name}.")
                    return None
                if self.set_end_image(nid, str(png)):
                    self.status.setText(f"Clip #{T.short_id(nid)} now ends at frame {dlg.slider.value() + 1} "
                                        f"({dlg.at():.2f} s) of {Path(video).name}.")
        finally:
            dlg.deleteLater()
        return None

    def set_end_image(self, nid: str, image: str | None) -> bool:
        """Give a generated clip its own end image (None: remove it); the clip then counts as changed."""
        if self.tree is None or self.running or nid not in self.tree.nodes or \
                self._frozen_block(self.tree.nodes[nid], "end_image"):
            return False
        try:
            if image is None:
                if not self.tree.nodes[nid].end_image:
                    return False
                T.clear_end_image(self.project, self.tree, nid)
            else:
                if QImage(image).isNull():
                    self.status.setText(f"Not an image: {Path(image).name}")
                    return False
                T.set_end_image(self.project, self.tree, nid, Path(image))
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not set the end image: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self.refresh()
        self.status.setText(f"Clip #{T.short_id(nid)} " + ("now ends at its end image." if image else "ends freely now."))
        return True

    # ---- middle keyframes
    def keyframe_action(self, act: str, nid: str, data: dict | None = None) -> None:
        """A keyframe action of a clip card: add / add_video / add_moment (data "at"), move (data "index", "at"),
        change / video / moment / note / remove (data "index")."""
        if self.tree is not None and nid in self.tree.nodes and self.tree.nodes[nid].frozen:
            self._frozen_block(self.tree.nodes[nid], "keyframe")
            return
        node = self.tree.nodes.get(nid) if self.tree else None
        if node is None or node.is_video or self.running:
            return
        data = data or {}
        i, at = data.get("index"), data.get("at")
        if act in ("note", "remove") and i is None:
            return                                 # (these name the keyframe they work on)
        if i is not None:
            if not 0 <= i < len(node.keyframes):
                return
            if at is None:
                at = node.keyframes[i]["at"]
        if at is None:
            return
        if act == "move":
            self._set_keyframe(nid, None, at, index=i)
        elif act in ("add", "change"):
            f = IP.pick_image(self, f"Keyframe of clip #{T.short_id(nid)} at {at:.2f} s (a picture the clip passes through)",
                           self.qs, None, self._picker_places(), self.project_images())
            if f:
                self._set_keyframe(nid, f, at, index=i)
        elif act in ("add_video", "video", "add_moment", "moment"):
            self._keyframe_from_video(nid, at, i, own=act in ("add_moment", "moment"))
        elif act == "note":
            text, ok = QInputDialog.getText(
                self, f"Keyframe at {at:.2f} s", "What happens at this moment (added to the prompt with its time, "
                "e.g. “she turns to the camera”; empty: nothing):", text=node.keyframes[i]["note"])
            if ok:
                self._set_keyframe(nid, None, at, note=text.strip(), index=i)
        elif act == "remove":
            try:
                T.remove_keyframe(self.project, self.tree, nid, i)
            except (ValueError, OSError, RuntimeError) as e:
                self._report_error(f"Could not remove the keyframe: {e}")
                return
            self._write_tree()
            self.refresh()
            self.status.setText(f"Clip #{T.short_id(nid)}: keyframe at {at:.2f} s removed.")

    def _keyframe_from_video(self, nid: str, at: float, index: int | None, own: bool = False,
                             video: str | None = None, dialog: bool = True):
        """A keyframe from a frame of a video. own: this clip's frame at that moment ("keep this moment": its time
        follows the frame chosen); else any video (the project's first), the keyframe stays at its time."""
        node = self.tree.nodes[nid]
        if own:
            if not node.made or not (self.project / node.made["video"]).is_file():
                self.status.setText(f"Clip #{T.short_id(nid)} is not made yet: there is no moment of it to keep.")
                return None
            video = str(self.project / node.made["video"])
        elif video is None:
            clips = [(str(self.project / n.made["video"]), f"Clip #{T.short_id(n.id)} · {n.prompt[:40]}")
                     for n in sorted(self.tree.nodes.values(), key=lambda n: self.tree.number(n.id))
                     if not n.is_video and n.made and (self.project / n.made["video"]).is_file()]
            cols = self.project_media("video") + ([("Clips of this project", clips)] if clips else [])
            video = choose_video(self, self.qs, places=self._picker_places(), collections=cols)
            if not video:
                return None
        title = (f"Keep a moment of clip #{T.short_id(nid)}: the frame to keep (it is pinned at its time)" if own else
                 f"Keyframe of clip #{T.short_id(nid)} at {at:.2f} s: a frame of {Path(video).name}")
        try:
            dlg = VideoFrameDialog(self, Path(video), title, self._known_info(video))
        except (RuntimeError, OSError) as e:
            self._report_error(f"Cannot read {Path(video).name}: {e}")
            return None
        if own:
            dlg.slider.setValue(min(dlg.frames - 2, max(1, T.keyframe_frame(at))))
        if not dialog:
            return dlg
        try:
            if dlg.exec() != QDialog.Accepted:
                return None
            return self._keyframe_from_dialog(nid, dlg, at, index, own)
        finally:
            dlg.deleteLater()

    def _keyframe_from_dialog(self, nid: str, dlg, at: float, index: int | None, own: bool) -> bool:
        k = dlg.slider.value()
        if own:                                          # (never its first or last frame)
            k = min(dlg.frames - 2, max(1, k))
            dlg.slider.setValue(k)
            at = k / chain.FPS
        with media.scratch("key") as tmp:
            png = Path(tmp) / f"key_f{k}.png"
            if not dlg.frame_file(png):
                self._report_error("Could not read that frame.")
                return False
            return self._set_keyframe(nid, str(png), at, index=index)

    def _set_keyframe(self, nid: str, image: str | None, at: float, note: str | None = None,
                      index: int | None = None) -> bool:
        """Add or change a keyframe (image None: keeps its picture); the clip then counts as changed."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return False
        if image is not None and QImage(image).isNull():
            self.status.setText(f"Not an image: {Path(image).name}")
            return False
        frames = T.clip_length(self.tree, self.tree.nodes[nid])
        at = min((frames - 2) / chain.FPS, max(1 / chain.FPS, round(float(at) * chain.FPS) / chain.FPS))
        try:
            T.set_keyframe(self.project, self.tree, nid, Path(image) if image else None, at, note, index)
        except (ValueError, OSError, RuntimeError) as e:
            self._report_error(f"Could not set the keyframe: {e}")
            return False
        self._write_tree()
        self.refresh()
        msg = f"Clip #{T.short_id(nid)}: keyframe at {at:.2f} s" + (" set." if index is None or image else " changed.")
        canvas = self._canvas_for()
        if image is not None and canvas:
            img = QImage(image)
            if abs(img.width() / img.height() - canvas[0] / canvas[1]) > ASPECT_TOLERANCE * canvas[0] / canvas[1]:
                msg += (f" Its shape ({img.width()}×{img.height()}) differs from the clip's ({canvas[0]}×{canvas[1]}):"
                        " H3 crops it to the middle.")
        self.status.setText(msg + " Keep it reachable by motion from what comes before it, or the video model cuts to it.")
        return True

    # ---- real videos
    def _canvas_for(self, file: str | None = None, rotate: int = 0, nid: str | None = None) -> tuple[int, int] | None:
        """The video size clips are made in; without a start image or video yet, the one this file would give
        (turned clockwise by rotate). nid: the video node being edited, as if it had this file and rotation (the
        size may follow it)."""
        try:
            tree = self.tree
            if nid is not None and nid in tree.nodes and tree.nodes[nid].is_video:
                tree = copy.copy(tree)                       # (only this node differs: not a copy of the whole tree)
                tree.nodes = dict(tree.nodes)
                node = tree.nodes[nid] = copy.copy(tree.nodes[nid])
                src = node.source
                node.source = T.video_source(file or src["file"], src["start"], src["end"], rotate)
            base = T.Checker(self.project, tree).base()
            if base is not None:
                return base["width"], base["height"]
            if file:
                info = chain.probe_source(Path(file))
                w, h = chain.rotated_size(info["width"], info["height"], rotate)
                return chain.canvas_size(w, h, int(self.tree.options.get("resolution", 512)))
        except Exception:  # noqa: BLE001  (an unreadable file: no size)
            logging.getLogger(__name__).debug("canvas of %s", file, exc_info=True)
        return None

    # ---- the same first clip for several start images (rows of the node view)
    def _pick_images(self, title: str) -> list[str]:
        """Several pictures from the file dialog (it opens where the start image came from)."""
        start = str(Path(self.tree.image).parent) if self.tree and self.tree.image and \
            Path(self.tree.image).parent.is_dir() else self.qs.value("startRowsDir", "", type=str)
        files, _f = QFileDialog.getOpenFileNames(self, title, start or "",
                                                 "Pictures (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if files:
            self.qs.setValue("startRowsDir", str(Path(files[0]).parent))
        return list(files)

    def add_start_rows(self, nid: str, images: list[str] | None = None, generate: bool | None = None) -> list[str]:
        """The same first clip for more start images: one new first clip per picture, with nid's prompt, seed
        and own settings, each on its own row of the node view. images None: asked for (several can be chosen).
        generate None: asked whether to make them now. Returns the new ids."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        node = self.tree.nodes[nid]
        if node.is_video or node.parent is not None:
            self.status.setText("More start images are for a first generated clip (one that starts the video).")
            return []
        if images is None:
            images = self._pick_images(f"Start images: clip #{T.short_id(nid)} is made once for each")
        if not images:
            return []
        try:
            new = T.add_start_rows(self.project, self.tree, nid, [Path(f) for f in images])
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not add the start images: {e}")
            return []
        ids = [n.id for n in new]
        if not ids:
            self.status.setText("None of the pictures could be read.")
            return []
        for i in ids:                              # (a video the clip leads into, its end image, its keyframes)
            self._copy_video_child(nid, i)
        self._pix.clear()
        self._write_tree()
        self.refresh()
        self.graph.reveal(ids[-1])
        skipped = len(images) - len(ids)
        size = ""
        try:
            base = T.Checker(self.project, self.tree).base() or {}
            if base.get("width") and base.get("height"):
                size = (f" Each picture is cut from its centre to the project's {base['width']}×{base['height']} "
                        "shape (nothing is stretched).")
        except Exception:  # noqa: BLE001
            pass
        self.status.setText(f"{len(ids)} start image{'s' if len(ids) != 1 else ''} added: clip #{T.short_id(nid)} "
                            f"once for each, with the same prompt, seed and settings, on its own row."
                            + (f" {skipped} could not be read." if skipped else "") + size)
        todo = ([nid] if self.status_of.get(nid) != "done" else []) + ids
        if generate is None:
            generate = D.confirm(
                self, "Start images", f"{self.status.text()}\n\nMake the {len(todo)} "
                f"clip{'s' if len(todo) != 1 else ''} now? They are made one after the other. You can also make "
                "them later: select one and press Generate.")
        if generate:
            self._start(todo)
        return ids

    def apply_to_row(self, nid: str) -> list[str]:
        """Give a first clip's prompt and own settings to the other clips of its set (the same clip for other
        start images). Each keeps its start image, seed and end image; made ones count as changed."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return []
        others = [n for n in T.row_members(self.tree, nid) if n.id != nid]
        if not others:
            self.status.setText(f"Clip #{T.short_id(nid)} has no other start images (More start images… adds some).")
            return []
        changed = T.apply_to_row(self.tree, nid)
        frozen = [n.id for n in others if n.frozen]
        if changed:
            self._write_tree()
            self.refresh(rebuild_path=True)
        self.status.setText(
            (f"Prompt and settings of clip #{T.short_id(nid)} given to {', '.join('#' + T.short_id(i) for i in changed)}; made ones "
             "count as changed and are made again by Generate." if changed else
             "The other start images already have this prompt and these settings.")
            + (f" Frozen and left as made: {', '.join('#' + T.short_id(i) for i in frozen)}." if frozen else ""))
        return changed

    def set_row_start(self, nid: str, image: str | None = None, clear: bool = False) -> bool:
        """A first clip's own start image: another one (image None: asked for), or none (clear: it starts from
        the project's start image again)."""
        if self.tree is None or self.running or nid not in self.tree.nodes:
            return False
        node = self.tree.nodes[nid]
        if node.frozen:
            self.status.setText(f"Clip #{T.short_id(nid)} is frozen: unfreeze it to change its start image.")
            return False
        try:
            if clear:
                T.clear_node_start(self.project, self.tree, nid)
            else:
                if image is None:
                    image = IP.pick_image(self, f"Start image of clip #{T.short_id(nid)}", self.qs, None,
                                       self._picker_places(), self.project_images())
                if not image:
                    return False
                T.set_node_start(self.project, self.tree, nid, Path(image))
        except (OSError, RuntimeError, ValueError) as e:
            self._report_error(f"Could not set the start image: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self.refresh(rebuild_path=True)
        self.status.setText(f"Clip #{T.short_id(nid)} starts from " + ("the project's start image again." if clear else
                                                                 "its own start image.")
                            + (" It counts as changed: Generate makes it again." if node.made else ""))
        return True
