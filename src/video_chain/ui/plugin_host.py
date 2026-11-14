"""What a plugin gets in setup(app) (see plugins.py): the app's side of the plugin interface.

    def setup(app):
        app.add_menu_action("Count the clips", lambda: app.status(f"{len(app.tree.nodes)} clips"))
        app.add_clip_action("Mark as favourite", lambda nid: ..., when=lambda node: not node.is_video)
        app.add_panel("My panel", my_widget)                       # a tab next to Clips / Cast / Sounds
        app.on("clip_made", lambda nid: ...)                       # also: project_opened, run_event, run_finished
        app.tree.nodes[nid].prompt = "…"; app.changed()            # after changing the project: save and show it

Everything here stays as it is between versions of the app. app.window is the main window itself: with it a plugin
can reach (and change) anything, but that can break when the app changes.

Threads: the host's methods must be called from the GUI thread, and the plugin's listeners, actions and setup() run
there. setup() is called once per window (a process can have several windows)."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QMessageBox, QWidget

from .. import plugins

# The listeners run in the GUI thread, after the app has handled the event itself (so a clip_made listener sees the
# clip's new record). run_finished comes once per job: a film made scene by scene gives one per scene.
EVENTS = ("project_opened",      # (folder: Path) a project (or a film's scene) was opened
          "run_event",           # (event: dict) every event of a running generation, as the generator sends it
          "clip_made",           # (clip id: str) a clip of a running generation is done (not the clips of an export
                                 # or an upscale, which are made again at another size)
          "run_finished")        # (result: int) a job ended (0: all made; else failed or stopped), once per job
PANELS = ("left", "bottom", "inspector")


class PluginHost:
    def __init__(self, window):
        self.window = window
        self.plugins: list[plugins.Plugin] = []
        self._menu: QMenu | None = None
        self._clip_actions: list[tuple] = []          # (plugin, text, callback, when, tip)
        self._heard: dict[str, list[tuple]] = {e: [] for e in EVENTS}
        self._current: plugins.Plugin | None = None   # the plugin whose setup runs (what it adds is its own)
        self._state: dict[str, plugins.State] = {}    # by module: what failed in this window (plugins are shared)
        self._menu_actions: list[tuple] = []          # (plugin, QAction)
        self._panels: list[tuple] = []                # (plugin, tab widget, widget)

    # ---- for the app
    def start(self) -> None:
        """Import the plugins and call each one's setup(app). Called once, when the window is built."""
        try:
            self.plugins = plugins.load()
        except BaseException as e:  # noqa: BLE001  (the app starts without plugins)
            plugins.interrupt(e)
            self.plugins = []
        for p in self.plugins:
            self._current = p
            try:
                fn = getattr(p.module, "setup", None) if p.loaded else None
                if callable(fn):
                    fn(self)
            except BaseException as e:  # noqa: BLE001  (the app starts without this plugin's part; also SystemExit)
                plugins.interrupt(e)
                plugins.failed(p, "setup()", e, fatal=True, state=self.state(p))
                self._withdraw(p)
            finally:
                self._current = None
        if self.plugins:
            self.menu().addSeparator()
            self.menu().addAction("About plugins…", self.show_about)
            # not at once: the window opens the last project next, and that writes its own status line
            QTimer.singleShot(0, self._say_not_in_use)

    def _say_not_in_use(self) -> None:
        bad = [p.name for p in self.plugins if self.error(p)]
        if not bad:
            return
        try:
            self.status(f"Plugin{'s' if len(bad) != 1 else ''} not in use: {', '.join(bad)} (Plugins → About "
                        "plugins… says why).")
        except RuntimeError:                          # the window is gone
            pass

    def _withdraw(self, p) -> None:
        """Take back everything plugin p added: its setup failed half way, so it is not in use and nothing of it
        stays in the window."""
        for _, a in [x for x in self._menu_actions if x[0] is p]:
            if self._menu is not None:
                self._menu.removeAction(a)
        self._menu_actions = [x for x in self._menu_actions if x[0] is not p]
        self._clip_actions = [x for x in self._clip_actions if x[0] is not p]
        for event in self._heard:
            self._heard[event] = [x for x in self._heard[event] if x[0] is not p]
        for _, tabs, widget in [x for x in self._panels if x[0] is p]:
            i = tabs.indexOf(widget)
            if i >= 0:
                tabs.removeTab(i)
                if widget.parent() is None:
                    widget.deleteLater()
                else:
                    widget.setParent(None)            # (the plugin may still hold it: it is not deleted)
        self._panels = [x for x in self._panels if x[0] is not p]

    def emit(self, event: str, *args) -> None:
        """Tell the plugins that something happened. A listener that fails is switched off (its plugin is
        listed with the error); the app goes on."""
        for entry in list(self._heard.get(event, ())):
            p, fn = entry
            try:
                fn(*args)
            except BaseException as e:  # noqa: BLE001  (also SystemExit)
                plugins.interrupt(e)
                if entry in self._heard[event]:
                    self._heard[event].remove(entry)
                if p is not None:
                    plugins.failed(p, f"its {event} listener", e, state=self.state(p))
                self.status(f"A plugin's {event} listener failed and was switched off"
                            + (f" ({p.name})" if p is not None else "") + f": {e}")
        # (the clips of an export or an upscale are made again at another size: not new clips)
        if event == "run_event" and isinstance(args[0], dict) and args[0].get("e") == "node" \
                and args[0].get("state") == "done" and not (getattr(self.window, "_job", None) or {}).get("refine"):
            self.emit("clip_made", args[0].get("id"))

    def context_actions(self, menu: QMenu, nid: str) -> dict:
        """Put the plugins' actions for clip nid into its right-click menu: -> {QAction: what to call}."""
        node = self.window.tree.nodes.get(nid) if self.window.tree is not None else None
        out, first = {}, True
        for p, text, fn, when, tip in self._clip_actions:
            try:
                if node is None or (when is not None and not when(node)):
                    continue
            except BaseException as e:  # noqa: BLE001  (a broken condition: the action is not offered)
                plugins.interrupt(e)
                if p is not None:
                    plugins.failed(p, f"the condition of the clip action “{text}”", e, state=self.state(p))
                continue
            if first:
                menu.addSeparator()
                first = False
            a = menu.addAction(text)
            if tip:
                a.setToolTip(tip)
            out[a] = lambda fn=fn, p=p, text=text: self._call(p, text, fn, nid)
        return out

    def _call(self, p, what: str, fn: Callable, *args) -> None:
        try:
            fn(*args)
        except BaseException as e:  # noqa: BLE001  (a plugin's action must not take the app down; also SystemExit)
            plugins.interrupt(e)
            if p is not None:
                plugins.failed(p, f"the action “{what}”", e, state=self.state(p))
            self.status(f"The plugin action “{what}” failed: {type(e).__name__}: {e}")

    # ---- for plugins
    @property
    def tree(self):
        """The open project's clip tree (video_chain.tree.Tree), None without a project."""
        return self.window.tree

    @property
    def project(self):
        """The open project's folder (Path), None without one."""
        return self.window.project

    def status(self, text: str) -> None:
        """A line in the window's status bar."""
        self.window.status.setText(str(text))

    def changed(self, rebuild: bool = True) -> bool:
        """Call after changing the open project's tree (a clip's prompt or settings, the project's options): the
        project is saved and the window shows the new state. rebuild=False: the clip cards are only updated.
        -> whether it was saved. It is not while a generation runs (the app itself does not save then; the status
        bar says so) nor when the project was changed outside this window; then the window is not refreshed."""
        if self.window.running:
            self.status("The project cannot be saved while clips are being generated; the plugin's change was not "
                        "saved.")
            return False
        saved = self.window._write_tree()
        if saved is False:
            return False
        self.window.refresh(rebuild_path=bool(rebuild))
        return bool(saved)

    def menu(self) -> QMenu:
        """The Plugins menu of the menu bar (made when first needed)."""
        if self._menu is None:
            self._menu = self.window.menuBar().addMenu("&Plugins")
            self._menu.setToolTipsVisible(True)
        return self._menu

    def add_menu_action(self, text: str, callback: Callable[[], None], shortcut: str | None = None,
                        tip: str = "") -> QAction:
        """An entry in the Plugins menu."""
        p = self._current
        a = self.menu().addAction(str(text))
        self._menu_actions.append((p, a))
        a.triggered.connect(lambda _=False: self._call(p, str(text), callback))
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
        return a

    def add_clip_action(self, text: str, callback: Callable[[str], None],
                        when: Callable[[object], bool] | None = None, tip: str = "") -> None:
        """An entry in every clip's right-click menu (node view): callback(clip id). when(node) -> bool limits it
        to some clips (node: video_chain.tree.Node)."""
        self._clip_actions.append((self._current, str(text), callback, when, tip))

    def add_panel(self, title: str, widget: QWidget, where: str = "left") -> int:
        """A tab with the plugin's widget: "left" (next to Clips / Cast & references / Sounds), "bottom" (next to
        Selected path / Log) or "inspector" (next to Project, on the right). -> its tab index."""
        if where not in PANELS:
            raise ValueError(f"where must be one of {PANELS}")
        tabs = {"left": self.window.left_tabs, "bottom": self.window.tabs, "inspector": self.window.inspector}[where]
        index = tabs.addTab(widget, str(title))
        self._panels.append((self._current, tabs, widget))
        return index

    def on(self, event: str, callback: Callable) -> None:
        """Hear an event (EVENTS): project_opened(folder), run_event(event dict), clip_made(clip id),
        run_finished(result: 0 = all made)."""
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; there are: {', '.join(EVENTS)}")
        self._heard[event].append((self._current, callback))

    def settings(self, name: str) -> QSettings:
        """The plugin's own settings: the app's settings file, under plugins/<name>/."""
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"the settings name must not be empty or contain / or \\: {name!r}")
        qs = self.window.qs
        s = QSettings(qs.fileName(), qs.format()) if qs.fileName() else QSettings()
        s.beginGroup(f"plugins/{name}")
        return s

    # ---- what is loaded
    def state(self, p: plugins.Plugin) -> plugins.State:
        """What failed for plugin p in this window."""
        return self._state.setdefault(p.target, plugins.State())

    def error(self, p: plugins.Plugin) -> str:
        """Why plugin p is not in use in this window ("" = in use): it could not be imported, or its setup failed."""
        return p.error or self.state(p).error

    def about_text(self) -> str:
        if not self.plugins:
            return "No plugins are installed."
        lines = []
        for p in self.plugins:
            error = self.error(p)
            lines.append(f"{p.name}  ({p.source}, module {p.target})\n    "
                         + (f"not in use: {error}" if error else "in use")
                         + "".join(f"\n    {x}" for x in self.state(p).problems))
        return "\n\n".join(lines)

    def show_about(self) -> None:
        box = QMessageBox(QMessageBox.Information, "Plugins", "Plugins add features that are not part of the app "
                          "itself. They are separately installed Python packages.", QMessageBox.Ok, self.window)
        box.setTextFormat(Qt.PlainText)                  # (what a plugin says may contain <…>)
        box.setInformativeText(self.about_text())
        traces = "\n\n".join(f"{p.name}:\n{self.state(p).trace or p.trace}" for p in self.plugins
                             if self.error(p) or self.state(p).problems)
        if traces:
            box.setDetailedText(traces)
        box.exec()
