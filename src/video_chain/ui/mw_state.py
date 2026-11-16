"""The tree's state: saving, change marks, the start image."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QImage, QImageReader

from .. import chain, swap, tree as T
from . import dialogs as D, image_picker as IP, image_edit as IE
from .graph import START
from .ui_logic import as_dict
from .image_edit import EDITS_DIR, project_edits, uses_of
from .mw_host import MainWindowHost
from . import start_crop as SC
from .start_crop import StartCropDialog

log = logging.getLogger(__name__)


class StateMixin(MainWindowHost):
    """The tree's state: saving, change marks, the start image."""
    _cut_dialog: StartCropDialog | None = None  # a start's cut check (made, not run, without modal dialogs)

    def _uses(self, meta: IE.ResultRecord) -> list[str]:
        """Where an image-edit result is used (the AI image window asks for every result of a reload): the index of
        the project's end images and face pictures is built once until something changes (UIM-26)."""
        if self._uses_cache is None or self._uses_cache[0] is not self.tree:
            self._uses_cache = (self.tree, IE.uses_index(self.tree))
        assert self.project is not None
        return uses_of(self.project, self.tree, meta, self._uses_cache[1])

    def _changed(self):
        """Something the clips depend on changed: save soon and update the states."""
        self._uses_cache = None
        self._save_timer.start()
        self._refresh_timer.start()

    def _save_tree(self) -> bool:
        """Save the open project now; False if it could not be saved (changed outside, disk error)."""
        if self.project is not None and self.tree is not None and not self.running:
            return bool(self._write_tree())
        return True

    def _may_leave_project(self) -> bool:
        """Save before the project is closed or replaced; if that fails, ask before its edits are dropped."""
        self._options_settled()                     # (a change still waiting for its pause is part of what is saved)
        self._save_timer.stop()
        if self._save_tree():
            return True
        why = self._write_error or self.status.text()
        return D.confirm(
            self, "Unsaved changes", f"The project could not be saved: {why}\n\nClose it anyway and lose the "
            "changes since the last save?")

    def _stamp(self) -> tuple | None:
        if self.project is None:
            return None
        try:
            st = (self.project / T.TREE_FILE).stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def _say(self, msg: str):
        """The result of a change: the status says it, unless the save of that change just failed (its error stays)."""
        if self._write_error:
            self.status.setText(self._write_error)
            return
        self.status.setText(msg)

    def _report_error(self, text: str):
        """A failure: in the status line and, because that line is replaced soon, also in the log."""
        self.status.setText(text)
        self.log.appendPlainText(text)

    def _write_tree(self) -> bool:
        """Save tree.json unless something else (another window, a command-line run) changed it since this window
        read it; then nothing is overwritten and the project has to be reopened."""
        if self.project is None or self.tree is None:
            return False
        now = self._stamp()
        if now is not None and self._tree_stamp is not None and now != self._tree_stamp:
            self._write_error = ("tree.json was changed outside this window (another window or a command-line "
                                 "run). Nothing was saved; reopen the project to continue.")
            self._report_error(self._write_error)
            return False
        try:
            T.save(self.project, self.tree)
        except OSError as e:
            self._write_error = f"Could not save the project: {e}"
            self._report_error(self._write_error)
            return False
        self._write_error = None
        self._tree_stamp = self._stamp()
        self._sync_shared()
        return True

    # ------------------------------------------------------------------ start image
    def _choose_image(self):
        if self.running:
            return
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            self._choose_project()
            if self.project is None:
                return
        start = str(Path(self.tree.image).parent) if self.tree and self.tree.image and \
            Path(self.tree.image).parent.is_dir() else None
        f = IP.pick_image(self, "Start image", self.qs, start, self._picker_places(), self.project_images())
        if f:
            self.set_image(f, confirm=True)

    def _picker_places(self) -> list[tuple[str, str]]:
        """Extra places in the image picker: the open project (its clips' last frames are there)."""
        if self.project is None:
            return []
        out = [(f"Project {self.project.name}", str(self.project))]
        if (self.project / "nodes").is_dir():
            out.append(("Project clips (last frames)", str(self.project / "nodes")))
        if (self.project / EDITS_DIR).is_dir():
            out.append(("Project AI edits", str(self.project / EDITS_DIR)))
        return out

    def project_media(self, kind: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """The open project's videos ('video') or sounds and videos ('media'), for the file browser:
        [(name, [(path, caption)])], kinds without files left out."""
        if self.project is None or self.tree is None:
            return []
        videos, seen = [], set()
        tree = self.tree
        assert tree is not None
        for n in sorted(self.tree.nodes.values(), key=lambda n: tree.number(n.id)):
            if not n.is_video:
                continue
            src = (swap.replaced(n) or {}).get("original") or n.source or {}
            f = src.get("file")
            if f and f not in seen and Path(f).is_file():
                seen.add(f)
                videos.append((f, f"Video #{T.short_id(n.id)} · {Path(f).name}"))
        out = [("Real videos of this project", videos)]
        if kind == "media":
            out.insert(0, ("Audio tracks", [(str(self.project / t["file"]), f"Track \u201c{t['name']}\u201d")
                                            for t in T.tracks(self.tree) if (self.project / t["file"]).is_file()]))
        return [(name, items) for name, items in out if items]

    def project_images(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """The open project's pictures by kind, for the image picker: [(name, [(path, caption)])] (kinds without
        pictures are left out; "All project images" first)."""
        if self.project is None or self.tree is None:
            return []
        p = self.project
        start = [(str(p / T.START_FILE), "Start image")] if (p / T.START_FILE).is_file() else []
        clips, videos, ends = [], [], []
        tree = self.tree
        assert tree is not None
        for n in sorted(self.tree.nodes.values(), key=lambda n: tree.number(n.id)):
            m = n.made or {}
            if n.is_video:
                for key, what in (("first", "first frame"), ("frame", "last frame")):
                    rel = as_dict(m).get(key)
                    if rel and (p / rel).is_file():
                        videos.append((str(p / rel), f"Video #{T.short_id(n.id)} · {what}"))
                continue
            if m.get("frame") and (p / m["frame"]).is_file():
                clips.append((str(p / m["frame"]), f"Clip #{T.short_id(n.id)} · last frame"))
            h = T.handover_file(p, self.tree, n) if m else None
            if h is not None and h.is_file() and str(h) != str(p / m.get("frame", "")):
                clips.append((str(h), f"Clip #{T.short_id(n.id)} · hand-over frame"))
            if n.end_image and (p / n.end_image).is_file():
                ends.append((str(p / n.end_image), f"End image of clip #{T.short_id(n.id)}"))
        edits = []
        for png, meta in project_edits(p):
            words = " ".join(str(meta.get("prompt") or "").split())
            what = "New image" if meta.get("mode") == "create" else "Edit"
            edits.append((str(png), f"{what} · {words[:48]}{'…' if len(words) > 48 else ''}" if words else what))
        faces = []
        for c in T.cast_of(self.tree):
            word = "face" if T.kind_of(c) == "person" else T.kind_of(c)
            for k, rel in enumerate(c["images"], start=1):
                faces.append((str(p / rel), f"@{c['name']}" + (f" · {word} {k}" if len(c["images"]) > 1 else
                                                                "" if word == "face" else f" · {word}")))
        group = "Cast faces" if all(T.kind_of(c) == "person" for c in T.cast_of(self.tree)) else "Cast & references"
        kinds = [("Start image", start), ("Image edits & new images", edits), (group, faces),
                 ("Clip frames", clips), ("Video frames", videos), ("End images", ends)]
        kinds = [(name, items) for name, items in kinds if items]
        everything = [x for _n, items in kinds for x in items]
        return ([("All project images", everything)] + kinds) if everything else []

    def made_from_start(self) -> list[str]:
        """Made generated clips that start (directly or along their path) from the start image."""
        if self.tree is None:
            return []
        roots = [n for n in self.tree.children(None) if not n.is_video]
        ids = [r.id for r in roots] + [d.id for r in roots for d in self.tree.descendants(r.id)]
        return [i for i in ids if self.status_of.get(i) == "done"]

    def _start_frozen(self) -> bool:
        """A frozen first clip starts from the start image: it stays (with a word in the status bar)."""
        roots = [n for n in (self.tree.nodes.values() if self.tree else []) if n.frozen and n.parent is None
                 and not n.is_video]
        if roots:
            self.status.setText(f"Clip #{T.short_id(roots[0].id)} is frozen and starts from the start image: "
                                "unfreeze it first to change the start image.")
        return bool(roots)

    def clear_image(self, confirm: bool = False) -> bool:
        """No start image any more. confirm: ask first when made clips depend on it."""
        if self.running or self.project is None or not (self.project / T.START_FILE).is_file():
            return False
        if self._start_frozen():
            return False
        made = [i for i, st in self.status_of.items() if st == "done"] if self.made_from_start() else []
        if confirm and made and not D.confirm(
                self, "Remove the start image",
                f"Remove the start image?\n\n{len(made)} made clip{'s depend' if len(made) != 1 else ' depends'} on it "
                "(it starts from it, or it sets the video size); they will count as CHANGED and be made again with "
                "their paths. The first clip then starts from its prompt."):
            return False
        before = self._done_clips()
        assert self.tree is not None
        T.clear_start_image(self.project, self.tree)
        self._pix.clear()
        self._write_tree()
        self._show_start_image()
        self.refresh()
        if not self._report_remade(before, "Start image removed"):
            self._say("Start image removed: the first clip starts from its prompt.")
        return True

    def set_image(self, path: str, confirm: bool = False) -> bool:
        """Use an image as the start image. confirm: ask first when made clips start from the current one."""
        if self.running:
            return False
        if self.project is None:
            self.status.setText("Open or create a project folder first.")
            return False
        if not QImageReader(str(path)).canRead():
            self.status.setText(f"Not an image: {Path(path).name}")
            return False
        if self._start_frozen():
            return False
        made = self.made_from_start()
        try:                                        # another aspect ratio: another video size for every clip
            assert self.tree is not None
            base = T.Checker(self.project, self.tree).base()
            size = T.video_size(self.tree.options, *chain.image_size(Path(path)))
            if base is not None and size != (base["width"], base["height"]) and not T.shape_of(self.tree.options):
                made = [i for i, st in self.status_of.items() if st == "done"]
        except Exception:  # noqa: BLE001  (the clips stay as they are; the question is asked without this check)
            log.warning("the new start image's video size could not be compared", exc_info=True)
        if confirm and made and not D.confirm(
                self, "Replace the start image",
                f"Use {Path(path).name} as the start image?\n\n{len(made)} made "
                f"clip{'s depend' if len(made) != 1 else ' depends'}"
                " on the current one (it starts from it, or it sets the video size); they will count as CHANGED and "
                "be made again with their paths."):
            return False
        before = self._done_clips()
        try:
            assert self.tree is not None
            T.set_start_image(self.project, Path(path), self.tree)
        except (OSError, RuntimeError) as e:
            self._report_error(f"Could not use {Path(path).name}: {e}")
            return False
        self._pix.clear()
        self._write_tree()
        self._show_start_image()
        self.refresh()
        if not self._report_remade(before, "New start image"):
            self._say(f"Start image: {Path(path).name}.")
        return True

    def _update_ref_start_button(self):
        """Enable the 'Add: Start image / Selected clip' buttons and say exactly which frame they take."""
        self._update_sound_buttons()
        self._update_track_buttons()
        start = self._start_image_there()
        self.b_ref_start.setEnabled(start and not self.running)
        ref = "face reference" if self._ref_word() == "face picture" else \
            f"picture of @{self._selected_character()['name']}"
        self.b_ref_start.setToolTip(f"Add the start image as a {ref}" if start else "The project has no start image")
        nid = self.tree.selected if self.tree else None
        node = self.tree.nodes.get(nid) if self.tree and nid else None
        made = node is not None and bool(node.made)
        self.b_ref_clip.setEnabled(made and not self.running)
        if made and node is not None and nid is not None:
            what = (f"the first frame of video #{T.short_id(nid)}" if node.is_video else f"the last frame of clip "
                                                                                         f"#{T.short_id(nid)}")
            tip = f"Add {what} as a {ref}"
        else:
            tip = "Select a converted video or a made clip"
        self.b_ref_clip.setToolTip(tip + ". For any frame of a video: Edit range… → Add as "
                                   + ("face picture…" if ref == "face reference" else ref))
        has_sound = node is not None and bool((node.made or {}).get("video"))
        self.b_voice_clip.setEnabled(has_sound and self._ref_word() == "face picture"
                                     and self._selected_character() is not None and not self.running)
        self.b_voice_clip.setToolTip(
            f"Set the voice from the sound of {'video' if node.is_video else 'clip'} #{T.short_id(nid)}: choose the "
            "part of it, with or without its background music" if has_sound and node is not None and nid is not None
            else "Select a made clip or a converted video with the voice")

    def _show_start_image(self):
        self._update_ref_start_button()
        start = self.project / T.START_FILE if self.project else None
        self._start_img = QImage(str(start)) if start and start.is_file() else None
        self._tile_cut = None
        if self._start_img is not None:
            assert self.tree is not None
            self.start_tile.setToolTip(self.tree.image or str(start))
        else:
            self.start_tile.clear_frame("Click or drop an image")
        self.b_start_clear.setEnabled(bool(start and start.is_file()) and not self.running)
        self._update_size_info()

    def _show_tile_cut(self):
        """The start tile: the start image, with the part the clips cut away shaded (when they cut)."""
        img = self._start_img
        if img is None or img.isNull() or self.tree is None or self.project is None:
            return
        crop = T.get_crop(self.tree, self.project, self.project / T.START_FILE)
        key = (self._canvas, None if crop is None else (crop["cx"], crop["cy"], crop["zoom"]))
        if key == self._tile_cut:
            return
        self._tile_cut = key
        self.start_tile.set_image(SC.cut_image(img, *self._canvas, crop) if self._canvas else img)

    def _update_size_info(self):
        """The line under the start image: the clips' size and shape and what decides it; whether (and how) the
        start image is cut to it. Reads the start image's size from its header (no decoding per option change)."""
        self._canvas = None                            # (set again below when the size is known)
        self.b_cut.setVisible(False)
        if self.project is None or self.tree is None:
            self.size_info.setText("")
            return
        start = self.project / T.START_FILE
        pic = None
        if start.is_file():
            reader = QImageReader(str(start))
            size = reader.size() if reader.canRead() else None
            pic = (size.width(), size.height()) if size is not None else (0, 0)
            if pic[0] < 1 or pic[1] < 1:
                self.size_info.setText(f"The start image ({T.START_FILE}) cannot be read: choose it again.")
                return
        shape = self.shape.currentData()
        origin = "shape" if shape else "image" if pic else T.size_origin(self.project, self.tree)
        if origin == "shape":
            a = T.ASPECTS[shape]
            dims: tuple[int, int] | None = (a[0] * 120, a[1] * 120)
            why = "the Shape chosen for this project"
        elif origin == "image" and pic is not None:
            dims, why = pic, f"like the start image ({pic[0]}×{pic[1]})"
        else:
            dims = None
            try:
                dims = T.size_source(self.project, self.tree)
            except Exception:  # noqa: BLE001  (no size known)
                log.warning("the size of the project's source could not be read", exc_info=True)
            if dims is None:
                self.size_info.setText("Choose a start image, or begin with a real video (Clips → camera button).")
                return
            why = (f"like the first video ({dims[0]}×{dims[1]})" if origin == "video" else
                   f"no start image or video yet, so {self.tree.options.get('aspect') or '16:9'} (choose a Shape "
                   "to change it)")
        assert dims is not None
        vw, vh = self._chosen_size(*dims)
        if self.exact_size.isChecked():
            why = "the exact size (Shape and Size do not count)"
        self._canvas = (vw, vh)
        text = f"Clips: {vw}×{vh} ({SC.shape_text(vw, vh)}), {why}."
        if pic:
            w, h = pic
            if T.mismatched(w, h, vw, vh):
                crop = T.get_crop(self.tree, self.project, start)
                where = "at the centre" if T.is_centre(crop) else "where you set it"
                text += (f" The start image ({w}×{h}, {SC.shape_text(w, h)}) is cut to {SC.shape_text(vw, vh)} "
                         f"{where}: the frame on it shows the part used (Edit cut…).")
                self.b_cut.setVisible(True)
            if w < vw or h < vh:
                text += f" · smaller than the video: the first frame starts soft (best: at least {vw}×{vh})"
        elif origin == "video":
            text += " A first generated clip starts without a start frame."
        else:
            text += " The first clip starts from its prompt (and the cast's face pictures)."
        self.size_info.setText(text + self._veda_note(vw, vh))
        self._show_tile_cut()

    def _cut_canvas(self) -> tuple[int, int] | None:
        """The clips' size (as the project says now)."""
        if self.project is None or self.tree is None:
            return None
        base = T.Checker(self.project, self.tree).base()
        return (base["width"], base["height"]) if base else None

    def edit_start_cut(self, nid: str | None = None) -> StartCropDialog | bool | None:
        """Choose the part of a start image the clips use: the project's start image, or (nid) a first clip's own
        start image. True: a cut was stored; False: cancelled / nothing to cut. Without modal dialogs the window
        is made and handed back (not run)."""
        if self.project is None or self.tree is None or self.running:
            return False
        node = self.tree.nodes.get(nid) if nid else None
        pic = T.start_source(self.project, node) if node is not None else self.project / T.START_FILE
        canvas = self._cut_canvas()
        if pic is None or not pic.is_file() or canvas is None:
            self.status.setText("There is no start image to cut.")
            return False
        if node is not None and node.frozen:
            self.status.setText(f"Clip #{T.short_id(node.id)} is frozen: unfreeze it to change its start image.")
            return False
        dlg = StartCropDialog(self, pic, *canvas, T.get_crop(self.tree, self.project, pic),
                              title="The start image's cut" + (f": clip #{T.short_id(nid)}" if nid else ""),
                              settings=self.qs)
        dlg.accepted.connect(lambda: self._cut_chosen(pic, canvas, dlg.crop()))
        if not self.modal_dialogs:
            return dlg
        return dlg.exec() == StartCropDialog.DialogCode.Accepted

    def _cut_chosen(self, pic: Path, canvas: tuple[int, int], crop: dict, write: bool = True):
        assert self.tree is not None and self.project is not None
        before = self._done_clips()
        T.set_crop(self.tree, self.project, pic, crop["cx"], crop["cy"], crop["zoom"], seen=T.shape_key(*canvas))
        if not write:
            return
        self._pix.clear()
        self._write_tree()
        self._update_size_info()
        self.refresh()
        if not self._report_remade(before, "New cut of the start image"):
            self._say("Cut of the start image kept.")

    def confirm_start_cuts(self, pending: list[tuple[str, Path]]) -> bool:
        """Before a start: each start image of another shape than the clips whose cut was not looked at for
        this shape is shown, to keep or move its cut. False: the user did not want to start."""
        canvas = self._cut_canvas()
        if canvas is None or self.tree is None or self.project is None:
            return True
        for k, (nid, pic) in enumerate(pending, 1):
            note = (f"Start image {k} of {len(pending)}: " if len(pending) > 1 else "") + \
                   (f"clip #{T.short_id(nid)} starts from it." if pic.name != T.START_FILE else
                    "the project's start image.")
            dlg = StartCropDialog(self, pic, *canvas, T.get_crop(self.tree, self.project, pic),
                                  title="Check the cut before the clips are made", note=note,
                                  cancel_text="Don't start", settings=self.qs)
            if not self.modal_dialogs:
                self._cut_dialog = dlg                 # (tests: made, not run; the start waits)
                return False
            if dlg.exec() != StartCropDialog.DialogCode.Accepted:
                return False
            self._cut_chosen(pic, canvas, dlg.crop(), write=False)
        self._pix.clear()
        self._update_size_info()
        return True

    def _veda_note(self, vw: int, vh: int) -> str:
        """Where Veda is used at this size (for the size line)."""
        if not self.veda.isChecked():
            return ""
        secs = chain.veda_from_seconds(vw, vh)
        return (f" · ⚡ Veda for clips from {secs:.1f} s" if secs is not None else
                " · Veda not used at this size (too small to gain)")

    def veda_used(self, n: T.Node) -> bool:
        """This generated clip is made with Veda sparse attention (see tree.veda_for)."""
        canvas = self._canvas
        if n.is_video or canvas is None or self.tree is None:
            return False
        frames = int((n.overrides or {}).get("frames", self.tree.options.get("frames", 73)))
        return T.veda_for(self.tree.options, canvas[0], canvas[1], frames)

    # ------------------------------------------------------------------ tree view
    def selected_id(self) -> str | None:
        return self.tree.selected if self.tree else None

    def selected_path(self) -> list[T.Node]:
        sel = self.selected_id()
        return self.tree.path_to(sel) if self.tree and sel and sel != START and sel in self.tree.nodes else []
