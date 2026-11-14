"""Project files: new, open, save, copy, close."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressDialog)

from .workers import post, run_in_background
from .. import film as F, project_copy as PC, tree as T
from . import dialogs as D
from .graph import START
from .project_dialogs import NewProjectDialog
from .ui_logic import plural
from .mw_host import MainWindowHost
from .qsvalue import qs_value


class ProjectMixin(MainWindowHost):
    """Project files: new, open, save, copy, close."""

    # ---- new / open / save / close
    def new_project(self):
        if not self._idle_or_say():
            return
        last = self.recent_projects()
        location = str(Path(last[0]).parent) if last else str(Path.home())
        dlg = NewProjectDialog(self, location, self.qs)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        folder = dlg.folder()
        if folder is not None:
            self.create_project(folder, dlg.image.text().strip())

    def create_project(self, folder: Path, image: str = "") -> bool:
        """Create a new project folder (it must not be a project yet) and open it; optionally with a start image."""
        folder = Path(folder).expanduser()
        if self._is_project(folder):
            self.status.setText(f"{folder.name} is already a project.")
            return False
        made = not folder.exists()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._report_error(f"Could not create {folder}: {e}")
            return False
        if not self.open_project(folder):
            if made:
                try:
                    folder.rmdir()                  # (the folder this call made, still empty: not left behind)
                except OSError:
                    pass
            return False
        if image:
            self.set_image(image)
        self.status.setText(f"New project {folder.name}." + ("" if image else " Choose a start image."))
        return True

    def _choose_project(self):
        if not self._idle_or_say():
            return
        last = self.recent_projects()
        start = str(self.project.parent) if self.project else (str(Path(last[0]).parent) if last else "")
        f = QFileDialog.getExistingDirectory(self, "Open project folder", start)
        if not f:
            return
        folder = Path(f)
        if not self._is_project(folder):
            if not D.confirm(self, "Open project", f"{folder.name} is not a Promptwood project yet.\n\n"
                                    "Create a new project in this folder?"):
                return
        self.open_project(folder)

    def save_project(self):
        if self.project is None or self.tree is None:
            self.status.setText("No project open.")
            return
        if self.running:
            self.status.setText("The project is saved by the generator while it runs.")
            return
        self._save_timer.stop()
        if self._write_tree():
            self.status.setText(f"Saved {self.project.name} ({time.strftime('%H:%M:%S')}).")

    def save_project_as(self):
        """Copy the whole project (clips included) to a new folder and continue there."""
        if self.project is None or not self._idle_or_say():
            return
        root = self.film.root if self.film else self.project
        d = QFileDialog.getExistingDirectory(self, "Save the project as (choose the folder to put the copy in)",
                                             str(root.parent))
        if not d:
            return
        name, ok = QInputDialog.getText(self, "Save as", "Name of the copy:", text=f"{root.name}-copy")
        if not ok or not name.strip():
            return
        self.copy_project_to(Path(d) / name.strip())

    def copy_project_to(self, target: Path):
        if self.project is None or not self._idle_or_say():
            return
        target = Path(target).expanduser()
        if target.exists() and any(target.iterdir()):
            self.status.setText(f"{target} exists and is not empty.")
            return
        self._save_timer.stop()
        if not self._write_tree():
            return
        source = self.film.root if self.film else self.project
        scene_rel = os.path.relpath(self.project, source)       # the scene to continue in, inside the copy
        self._copy_scene = scene_rel
        cur = self.film.scene_of(self.project) if self.film else None
        self._copy_scene_id = cur.id if cur else None
        self.status.setText(f"Copying the project to {target} …")
        self.act_save_as.setEnabled(False)
        self._copying = str(source)
        cancel = self._copy_cancel = threading.Event()
        self.act_cancel_copy.setEnabled(True)
        dialog = self._copy_dialog = QProgressDialog(f"Copying the project to {target} …", "Cancel", 0, 0, self)
        dialog.setWindowTitle("Save as")
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.canceled.connect(self.cancel_copy)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)

        def progress(done: int, total: int):
            post(self, self._copy_progress, done, total, str(target))

        def work():
            try:
                PC.copy_project(source, target, progress=progress, cancel=cancel)
            except PC.CopyCancelled:
                return False
            return True
        run_in_background(work, on_done=lambda ok: self._copied(str(target), "" if ok else None),
                          on_error=lambda e: self._copied(str(target), str(e)), owner=self)

    def cancel_copy(self):
        """Stop Save as: the files copied so far are removed."""
        if self._copy_cancel is not None:
            self._copy_cancel.set()
            self.status.setText("Cancelling the copy …")

    def _copy_progress(self, done: int, total: int, target: str):
        if self._copy_cancel is None or self._copy_cancel.is_set():
            return
        self.status.setText(f"Copying the project to {target}: {done} of {total} files …")
        if self._copy_dialog is not None:
            self._copy_dialog.setMaximum(total)
            self._copy_dialog.setValue(done)

    def _copied(self, target: str, error: str | None):
        """The copy ended: error "" is success, None is a cancel, text is what went wrong."""
        source, self._copying = self._copying, None
        self._copy_cancel = None
        self.act_cancel_copy.setEnabled(False)
        if self._copy_dialog is not None:
            self._copy_dialog.reset()
            self._copy_dialog.deleteLater()
            self._copy_dialog = None
        self.act_save_as.setEnabled(True)
        self._update_buttons()
        if error is None:
            self.status.setText("Copying cancelled; nothing was kept.")
            return
        if error:
            self._report_error(f"Could not copy the project: {error}")
            return
        here = str(self.film.root if self.film else self.project) if self.project is not None else None
        if here != source:
            self.status.setText(f"Copied the project to {target}.")
            return
        if self.open_project((Path(target) / self._copy_scene).resolve(),
                             scene=self._copy_scene_id):
            self.status.setText(f"Saved as {Path(target).name}; you are working in the copy now.")

    def _dropped_on_window(self, files: list[str]):
        """A project folder (or its tree.json) dropped on the window opens it."""
        f = Path(files[0])
        folder = f.parent if f.name == T.TREE_FILE else f
        if folder.is_dir() and self._is_project(folder):
            self.open_project(folder)
        else:
            self.status.setText(f"{f.name} is not a project: drop a project folder here to open it, "
                                "or pictures on the cast, sounds on the sounds, a picture on a clip's end.")

    def close_project(self):
        if self.project is None or not self._idle_or_say():
            return
        if not self._may_leave_project():
            return
        self._waiting_comfy = self._waiting_refine = None      # a wait for ComfyUI belongs to this project
        self._film_queue = None
        self.preview.stop()
        self.film = None
        self.storyboard.show_scenes([], None)
        self.film_info.setText("")
        self.project = self.tree = None
        self.project_edit.clear()
        self.qs.setValue("project", "")
        self.focus = self._path_end = None
        self.timeline.show_path([], None)
        self.graph.set_focus(None)
        for c in self.cards:
            self.card_stack.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        self.start_tile.clear_frame("Click or drop an image")
        self.size_info.setText("")
        self.refs.clear()
        for lst in (self.cast_list, self.sounds, self.track_list):
            lst.clear()
        self._tracks_shown = None
        self._canvas = None
        self._failures.clear()                  # (ids like n3 exist in every project)
        self.live, self.progress_of, self.status_of, self.why_of = {}, {}, {}, {}
        self.graph.show_nodes([])
        self.path_info.setText("")
        self.tree_info.setText("")
        self._update_title()
        self._update_buttons()
        self.status.setText("No project open. New… or Open… under Project, or the File menu.")

    def open_project(self, folder: Path, quiet: bool = False, scene: str | None = None) -> bool:
        if not self._idle_or_say():
            return False
        if self.project is not None and not self._may_leave_project():
            return False
        folder = Path(folder).expanduser().resolve()
        try:
            # (after the leave check above; tests and the API open new folders)
            folder.mkdir(parents=True, exist_ok=True)
            root = F.root_of(folder)
            film = F.Film.load(root)               # the film this folder belongs to, and which scene opens
            if len(film.scenes) > 1 and not film.problem:
                try:
                    film.update_continues()        # scenes that continue the one before: its current end
                except (OSError, ValueError) as e:
                    self._report_error(f"Could not update the scenes that continue another: {e}")
            if scene is not None and any(x.id == scene for x in film.scenes):
                film.current = scene
                folder = film.path(scene)
            elif folder == root:
                folder = film.path(film.current or film.scenes[0].id)
            else:
                film.current = (film.scene_of(folder) or film.scenes[0]).id
            tree = T.load(folder)
            T.sweep_scratch(folder)                # (the job folders a killed run left in nodes/)
        except (OSError, ValueError) as e:
            msg = f"Could not open the project: {e}"
            self._report_error(msg)
            if not quiet:
                D.warn(self, "Promptwood", msg + "\n\nThe files were not changed.")
            return False
        if not (folder / T.TREE_FILE).exists() and not tree.nodes:
            try:                                 # a new project starts with the last used settings
                last = json.loads(qs_value(self.qs, "lastOptions", "{}", str))
                tree.options.update(cast(T.Options, {
                    k: v for k, v in last.items()
                    if k in T.DEFAULT_OPTIONS and k not in ("identity", "cast", "style", "audio_refs")}))
            except ValueError:
                pass
        self._clean_options(tree.options)           # (a value that cannot be read must not leave a half-open project)
        if scene is None:
            self._film_queue = None                # (Generate film opens its scenes with scene=…)
        self._waiting_comfy = self._waiting_refine = None     # (only now: a failed open leaves the waits as they were)
        self.project, self.tree = folder, tree
        self.film = film
        self.focus = self._path_end = None         # (ids like n3 exist in every scene: no focus carried over)
        self._failures.clear()
        synced = ""
        if len(film.scenes) > 1 and film.shared_state and film.shared(dict(tree.options)) != film.shared_state:
            tree.options.update(json.loads(json.dumps(film.shared_state)))   # it missed a change of the film's cast
            synced = " Its cast and sound settings were brought up to the film's."
        self._shared_sent = json.dumps(film.shared(dict(tree.options)), sort_keys=True)
        self.project_edit.setText(str(root))
        self.project_edit.setToolTip(str(folder))
        self.preview.stop()
        self.live, self.progress_of = {}, {}
        self._options_to_widgets()
        self._show_start_image()
        self._show_references()
        self._show_sounds()
        self._show_tracks()
        if not tree.nodes:
            tree.selected = tree.add(None, "").id
        if tree.selected is None:
            tree.selected = START
        self._tree_stamp = self._stamp()
        self._write_tree()
        T.ensure_handovers(folder, tree)          # e.g. deleted by hand: the clips after them stay done
        self.qs.setValue("project", str(root))
        self._remember_project(root)
        try:
            film.save()                            # which scene was open (only once there are several)
        except OSError as e:
            self._report_error(f"Could not remember which scene is open: {e}")
        self._show_storyboard()
        self._update_title()
        self._warm_lora_hashes()
        self.refresh(rebuild_path=True)
        QTimer.singleShot(0, self.graph.fit)
        QTimer.singleShot(0, self._focus_graph)
        self.status.setText(f"Project {folder.name}: {plural(len(tree.nodes), 'clip')}."
                            + synced + (f" Note: {film.problem}; the film's scene list is not changed until it is "
                                        "fixed." if film.problem else ""))
        if tree.recovered_from is not None:
            self._tell_recovered(folder, tree.recovered_from)
        self.plugins.emit("project_opened", folder)
        return True

    def _focus_graph(self):
        """The keyboard goes to the clip graph (arrow keys walk the clips) unless something is being typed."""
        fw = QApplication.focusWidget() or self.focusWidget()
        typing = isinstance(fw, QPlainTextEdit) or (isinstance(fw, QLineEdit) and not fw.isReadOnly())
        if not typing and self.tree is not None and self.graph.isVisible():
            self.graph.setFocus(Qt.FocusReason.OtherFocusReason)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._first_shown:
            self._first_shown = True
            QTimer.singleShot(0, lambda: self.graph.setFocus(Qt.FocusReason.OtherFocusReason))

    def _tell_recovered(self, folder: Path, backup: Path):
        """tree.json was damaged and its backup was opened: said in the status line and the log, and in a notice
        that does not block the window (the last changes before the damage may be missing)."""
        text = (f"tree.json of {folder.name} was damaged; the backup {backup.name} (the state before the last save) "
                "was opened instead. The last changes may be missing. The damaged file is kept next to it "
                "(tree.json.damaged) when the project is saved.")
        self._report_error(text)
        self.notify("Project opened from its backup", text)

    def notify(self, title: str, text: str):
        """A notice that does not block the window (a message box opened, not run); kept until it is closed."""
        box = QMessageBox(QMessageBox.Icon.Information, title, text, QMessageBox.StandardButton.Ok, self)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._notices.append(box)
        box.destroyed.connect(lambda _=None, b=box: self._notices.remove(b) if b in self._notices else None)
        box.open()

    def _open_dir(self):
        if self.project and self.project.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project)))
