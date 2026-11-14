"""The film: scenes, storyboard, joining the scenes."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import chain, film as F, tree as T
from . import dialogs as D, shortcuts as S
from .widgets import format_time, label, safe_emit


class FilmMixin:
    """The film: scenes, storyboard, joining the scenes."""

    def join_film(self, purpose: str = "play", out: Path | None = None):
        if self.film is None:
            return
        if self.running:
            self.status.setText("Wait until generating has finished.")
            return
        if self._joins:
            self.status.setText("A join is still running; wait for it.")
            return
        if not self._save_tree():
            return
        self._update_continues()
        film, bridge = F.Film.load(self.film.root), self.bridge
        target = Path(out) if out else self.film.root / "paths" / "film.mp4"
        root = str(self.film.root)
        self.status.setText("Checking and joining the film …")
        self._joins += 1                           # (a run waits for it, a second join is refused)
        self._update_buttons()

        def work():
            try:
                path = F.join_film(film, target, keep=out is None)
                try:
                    secs = chain.joined_seconds(Path(path))      # not in the GUI thread (network shares)
                except Exception:  # noqa: BLE001
                    secs = 0.0
                safe_emit(bridge.film, (path, root, secs), None, purpose)
            except BaseException as e:  # noqa: BLE001  (whatever ends the job: the busy flag is reset by the answer)
                safe_emit(bridge.film, (None, root, 0.0), str(e), purpose)
        threading.Thread(target=work, daemon=True).start()

    def _film_joined(self, result, error, purpose):
        self._joins = max(0, self._joins - 1)
        self._update_buttons()
        path, root, secs = result
        if self.film is None or str(self.film.root) != root:
            return                                 # another film was opened meanwhile
        if error:
            self.status.setText(f"The film cannot be joined yet: {error}.")
            return
        if purpose == "play":
            self.preview.open(Path(path), f"Film · {len(self.film.scenes) if self.film else 1} scenes · "
                                          f"{format_time(secs, 1)}")
            self.status.setText(f"Film joined: {format_time(secs, 1)}.")
        else:
            self.status.setText(f"Exported the film: {Path(path).name} ({format_time(secs, 1)}).")

    def export_film(self):
        if self.film is None or not self._idle_or_say():
            return
        start = str(self.film.root / f"{self.film.root.name}-film.mp4")
        f, _ = QFileDialog.getSaveFileName(self, "Export the film", start, "Video (*.mp4)")
        if f:
            self.join_film("export", Path(f if f.lower().endswith(".mp4") else f + ".mp4"))

    def generate_film(self):
        """Make every scene's missing clips (up to each scene's ending), one scene after the other."""
        if self.film is None or not self._idle_or_say():
            return
        if not self._save_tree():
            return
        self._update_continues()
        todo, empty = [], []
        for sc in self.film.scenes:
            d = self.film.path(sc.id)
            try:
                t = T.load(d)
            except (OSError, ValueError) as e:
                self.status.setText(f"Scene “{sc.title}” cannot be read: {e}")
                return
            take = self.film.take_of(sc.id, t)
            if take is None:
                empty.append(sc.title)
                continue
            ch = T.Checker(d, t)
            if any(ch.status(n.id) != "done" for n in t.path_to(take)):
                todo.append(sc.id)
        if empty:
            self.status.setText(f"Scene “{empty[0]}” has no clip yet: write its first clip, or remove the scene.")
            return
        if not todo:
            self.status.setText("Every scene is made: Play film.")
            return
        ids = [sc.id for sc in self.film.scenes]          # from the first scene with work on, each one checked when
        self._film_queue = ids[ids.index(todo[0]):]       # its turn comes (a continuing one may need it by then)
        self._film_total = len(self._film_queue)
        self._film_next()

    def _film_next(self):
        if not self._film_queue:
            self._film_queue = None
            self.status.setText("The film is made: Play film.")
            self._show_storyboard()
            return
        if self.film is not None and len(self.film.scenes) > 1:
            try:
                self.film.update_continues()       # the scenes after a new end start from it
            except (OSError, ValueError) as e:
                self._report_error(f"Could not update the scenes that continue another: {e}")
        if self.film is None:
            self._film_queue = None                # (the film was closed meanwhile)
            return
        sid = self._film_queue.pop(0)
        done = self._film_total - len(self._film_queue)
        d = self.film.path(sid)
        try:
            t = T.load(d)
            take = self.film.take_of(sid, t)
            ch = T.Checker(d, t)
            ready = take is not None and all(ch.status(n.id) == "done" for n in t.path_to(take))
        except (OSError, ValueError, StopIteration, KeyError):
            ready = False
        if ready:                                  # made already: the next one
            QTimer.singleShot(0, self._film_next)
            return
        self.switch_scene(sid)
        if self.film is None or self.film.path(sid) != self.project:
            self._film_queue = None                # could not open the scene (the status says why)
            return
        take = self.film.take_of(sid, self.tree)
        self._start([take])
        if not self.running and self._waiting_comfy is None:
            self._film_queue = None                # not started (the status says why)
        else:
            self.status.setText(f"Generating the film: scene {done} of {self._film_total} "
                                f"(“{self.film.scene(sid).title}”) … " + self.status.text())

    # ---- the film's scenes (film.py)
    def _show_storyboard(self):
        if self.film is None:
            self.storyboard.show_scenes([], None)
            self.film_info.setText("")
            return
        scenes = []
        for sc in self.film.scenes:
            if self.project is not None and self.film.path(sc.id) == self.project:
                summ = F.summary(self.film, sc.id, self.tree, self.status_of)
            else:
                summ = F.summary(self.film, sc.id)
            scenes.append((sc.id, sc.title, summ))
        self.storyboard.show_scenes(scenes, self.film.current)
        cur = self.film.scene_of(self.project) if self.project else None
        self.film_info.setText(f"› {cur.title}" if cur and len(self.film.scenes) > 1 else "")
        self._update_title()

    def _update_scene_card(self):
        """The open scene's card follows the work on it (its path, states)."""
        if self.film is None or self.project is None or self.tree is None:
            return
        cur = self.film.scene_of(self.project)
        if cur is not None and cur.take and cur.take not in self.tree.nodes:
            try:
                self.film.set_take(cur.id, None)
                self.status.setText(f"The clip “{cur.title}” ended at is gone: the film plays its selected path "
                                    "now (right-click a clip → End the scene here).")
            except OSError:
                pass
        if cur is not None:
            self.storyboard.update_card(cur.id, cur.title, F.summary(self.film, cur.id, self.tree, self.status_of))

    def _sync_shared(self):
        """The cast, sounds and size settings are the film's: written into the other scenes when they change."""
        if self.film is None or len(self.film.scenes) < 2 or self.tree is None:
            return
        now = json.dumps(self.film.shared(self.tree.options), sort_keys=True)
        if now == getattr(self, "_shared_sent", None):
            return
        try:
            self.film.sync_shared(self.project, self.tree.options)
            self._shared_sent = now
        except (OSError, ValueError) as e:
            self._report_error(f"Could not pass the cast / sound settings on to the other scenes: {e}")

    def switch_scene(self, sid: str):
        if self.film is None or self.project is None:
            return
        if not self._idle_or_say():
            self._show_storyboard()
            return
        if self.film.path(sid) == self.project:
            return
        if not self.open_project(self.film.root, scene=sid):
            self._show_storyboard()                # back to the scene still open

    def add_scene(self):
        if self.film is None or self.project is None or not self._idle_or_say():
            return
        title, ok = QInputDialog.getText(self, "New scene", "Title of the new scene:",
                                         text=f"Scene {len(self.film.scenes) + 1}")
        if not ok:
            return
        if not self._may_leave_project():
            return
        cur = self.film.scene_of(self.project)
        try:
            scene = self.film.add_scene(title, after=cur.id if cur else None, like=cur.id if cur else None)
        except (OSError, ValueError) as e:
            self._report_error(f"Could not add the scene: {e}")
            return
        if self.open_project(self.film.root, scene=scene.id):
            self.status.setText(f"Scene “{scene.title}” added: choose its start image (Start image), or begin "
                                "with a real video. The cast and sounds are the film's.")

    def rename_scene(self, sid: str):
        if self.film is None:
            return
        title, ok = QInputDialog.getText(self, "Rename scene", "Title:", text=self.film.scene(sid).title)
        if not ok:
            return
        try:
            self.film.rename(sid, title)
        except (OSError, ValueError) as e:
            self._report_error(f"Could not rename the scene: {e}")
            return
        self._show_storyboard()

    def delete_scene(self, sid: str):
        if self.film is None or not self._idle_or_say():
            return
        sc = self.film.scene(sid)
        if sc.dir == ".":
            self.status.setText("The film folder's own scene cannot be deleted; move it if it should not be first.")
            return
        if len(self.film.scenes) < 2:
            return
        n = F.summary(self.film, sid)["clips"]
        if not D.confirm(self, "Delete scene", f"Delete the scene “{sc.title}” with its {n} clip"
                                f"{'s' if n != 1 else ''} from the film? Its folder with their files is not deleted: "
                                "it is moved to old/ in the film's folder."):
            return
        if self.film.path(sid) == self.project:    # leave it first
            other = next(s for s in self.film.scenes if s.id != sid)
            self.switch_scene(other.id)
            if self.project == self.film.path(sid):
                return
        try:
            self.film.remove(sid)
        except (OSError, ValueError) as e:
            self._report_error(f"Could not delete the scene: {e}")
            return
        self._show_storyboard()
        self.status.setText(f"Scene “{sc.title}” deleted.")

    def _update_continues(self) -> list[str]:
        """Bring continuing scenes to the current end of the scene before them; the open scene is read again
        when its start image changed."""
        if self.film is None or len(self.film.scenes) < 2 or self.film.problem:
            return []
        try:
            updated = self.film.update_continues()
        except (OSError, ValueError) as e:
            self._report_error(f"Could not update the scenes that continue another: {e}")
            return []
        cur = self.film.scene_of(self.project) if self.project else None
        if cur is not None and cur.id in updated:
            self.open_project(self.film.root, scene=cur.id)
        return updated

    def duplicate_scene(self, sid: str):
        if self.film is None:
            return
        if not self._idle_or_say():
            return
        if not self._save_tree():
            return
        root, bridge = self.film.root, self.bridge
        self._copying = str(root)                  # (no scene switch, no other copy meanwhile)
        self._update_buttons()
        self.status.setText(f"Copying “{self.film.scene(sid).title}” with its clips …")

        def work():
            try:
                sc = F.Film.load(root).duplicate(sid)
                safe_emit(bridge.duplicated, (sc.id, sc.title), None, str(root))
            except BaseException as e:  # noqa: BLE001  (the answer ends "copying")
                safe_emit(bridge.duplicated, None, str(e), str(root))
        threading.Thread(target=work, daemon=True).start()

    def _scene_duplicated(self, result, error, root):
        self._copying = None
        self._update_buttons()
        if self.film is None or str(self.film.root) != root:
            return
        if error:
            self._report_error(f"Could not duplicate the scene: {error}")
            return
        current = self.film.current
        try:
            self.film = F.Film.load(self.film.root)
            self.film.current = current
        except (OSError, ValueError):
            pass
        self._show_storyboard()
        self.status.setText(f"“{result[1]}” added: change it freely, the original stays as it is.")

    def scene_notes(self, sid: str):
        if self.film is None:
            return
        text, ok = QInputDialog.getMultiLineText(self, "Scene notes", f"Notes for “{self.film.scene(sid).title}” "
                                                 "(e.g. what happens, a line of script):", self.film.scene(sid).notes)
        if not ok:
            return
        try:
            self.film.set_notes(sid, text)
        except OSError as e:
            self._report_error(f"Could not save the notes: {e}")
            return
        self._show_storyboard()

    def show_film_cast(self):
        """A table: the film's characters and references by scene (face ●, picture ◆, voice ♪), along the clips the
        film plays."""
        if self.film is None:
            return
        if not self._save_tree():
            return
        names, rows = F.cast_by_scene(self.film)
        dlg = QDialog(self)
        dlg.setWindowTitle("Cast in the film")
        v = QVBoxLayout(dlg)
        if not names:
            v.addWidget(label("The film has no characters yet (the Cast & references tab, " + S.keys("tab_cast")
                              + ").", level="secondary"))
        table = QTableWidget(len(names), len(rows))
        table.setHorizontalHeaderLabels([f"{k}  {title}" for k, (title, _m) in enumerate(rows, start=1)])
        table.setVerticalHeaderLabels([f"@{n}" for n in names])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        sym = {"face": "●", "voice": "♪", "face+voice": "●♪", "picture": "◆"}
        for c, (_title, marks) in enumerate(rows):
            for r, n in enumerate(names):
                item = QTableWidgetItem(sym.get(marks.get(n, ""), ""))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        v.addWidget(table)
        v.addWidget(label("● face   ◆ picture (a place, object or style)   ♪ speaks with its voice", level="tertiary"))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        dlg.resize(min(900, 160 + 120 * len(rows)), min(600, 140 + 30 * len(names)))
        self._film_cast_dialog = dlg               # (tests look at it)
        if not getattr(self, "_dialogs_modeless", False):
            dlg.exec()
            dlg.deleteLater()

    def set_scene_join(self, sid: str, join: str):
        """How a scene follows the one before: a cut, a crossfade, or continuing from its last frame."""
        if self.film is None or not self._idle_or_say():
            return
        if not self._save_tree():
            return
        try:
            self.film.set_join(sid, join)
            updated = self.film.update_continues() if join == "continue" else []
        except (OSError, ValueError) as e:
            self._report_error(f"Could not change the join: {e}")
            return
        sc = self.film.scene(sid)
        if self.film.path(sid) == self.project and sid in updated:
            self.open_project(self.film.root, scene=sid)   # its start image changed under the open window
        self._show_storyboard()
        self.status.setText({"cut": f"“{sc.title}” follows with a hard cut.",
                             "fade": f"“{sc.title}” fades in over the scene before (½ s).",
                             "continue": f"“{sc.title}” starts from the last frame of the scene before"
                                         + ("." if updated else " (once that one is made)."),
                             }[join])

    def _scenes_moved(self, order: list[str]):
        if self.film is None:
            return
        reset = []
        try:
            reset = self.film.reorder(order)
        except (OSError, ValueError) as e:
            self._report_error(f"Could not move the scene: {e}")
        self._show_storyboard()
        if reset:
            self.status.setText("Moved. " + ", ".join(f"“{self.film.scene(i).title}”" for i in reset) +
                                " now start" + ("s" if len(reset) == 1 else "") + " with a cut: the scene before "
                                "changed (right-click → Join with the scene before → Continue to continue it).")
