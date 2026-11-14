"""What a plugin gets in setup(app) (see plugins.py): the app's side of the plugin interface.

    def setup(app):
        app.add_menu_action("Count the clips", lambda: app.status(f"{len(app.tree.nodes)} clips"))
        app.add_clip_action("Mark as favourite", lambda nid: ..., when=lambda node: not node.is_video)
        app.add_panel("My panel", my_widget)                       # a tab next to Clips / Cast / Sounds
        app.on("clip_made", lambda nid: ...)                       # also: project_opened, run_event, run_finished
        app.tree.nodes[nid].prompt = "…"; app.changed()            # after changing the project: save and show it

Everything here stays as it is between versions of the app. app.window is the main window itself: with it a plugin
can reach (and change) anything, but that can break when the app changes."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QMessageBox, QWidget

from .. import plugins

EVENTS = ("project_opened",      # (folder: Path) a project (or a film's scene) was opened
          "run_event",           # (event: dict) every event of a running generation, as the generator sends it
          "clip_made",           # (clip id: str) a clip of the running generation is done
          "run_finished")        # (result: int) a generation or export ended (0: all made; else failed or stopped)
PANELS = ("left", "bottom", "inspector")


class PluginHost:
    def __init__(self, window):
        self.window = window
        self.plugins: list[plugins.Plugin] = []
        self._menu: QMenu | None = None
        self._clip_actions: list[tuple] = []          # (plugin, text, callback, when, tip)
        self._heard: dict[str, list[tuple]] = {e: [] for e in EVENTS}
        self._current: plugins.Plugin | None = None   # the plugin whose setup runs (what it adds is its own)

    # ---- for the app
    def start(self) -> None:
        """Import the plugins and call each one's setup(app). Called once, when the window is built."""
        self.plugins = plugins.load()
        for p in self.plugins:
            fn = getattr(p.module, "setup", None) if p.loaded else None
            if not callable(fn):
                continue
            self._current = p
            try:
                fn(self)
            except Exception as e:  # noqa: BLE001  (the app starts without this plugin's part)
                plugins.failed(p, "setup()", e, fatal=True)
            finally:
                self._current = None
        if self.plugins:
            self.menu().addSeparator()
            self.menu().addAction("About plugins…", self.show_about)
            bad = [p.name for p in self.plugins if p.error]
            if bad:
                self.status(f"Plugin{'s' if len(bad) != 1 else ''} not in use: {', '.join(bad)} (Plugins → About "
                            "plugins… says why).")

    def emit(self, event: str, *args) -> None:
        """Tell the plugins that something happened. A listener that fails is switched off (its plugin is
        listed with the error); the app goes on."""
        for entry in list(self._heard.get(event, ())):
            p, fn = entry
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001
                self._heard[event].remove(entry)
                if p is not None:
                    plugins.failed(p, f"its {event} listener", e)
                self.status(f"A plugin's {event} listener failed and was switched off"
                            + (f" ({p.name})" if p is not None else "") + f": {e}")
        if event == "run_event" and isinstance(args[0], dict) and args[0].get("e") == "node" \
                and args[0].get("state") == "done":
            self.emit("clip_made", args[0].get("id"))

    def context_actions(self, menu: QMenu, nid: str) -> dict:
        """Put the plugins' actions for clip nid into its right-click menu: -> {QAction: what to call}."""
        node = self.window.tree.nodes.get(nid) if self.window.tree is not None else None
        out, first = {}, True
        for p, text, fn, when, tip in self._clip_actions:
            try:
                if node is None or (when is not None and not when(node)):
                    continue
            except Exception:  # noqa: BLE001  (a broken condition: the action is not offered)
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
        except Exception as e:  # noqa: BLE001  (a plugin's action must not take the app down)
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

    def changed(self, rebuild: bool = True) -> None:
        """Call after changing the open project's tree (a clip's prompt or settings, the project's options): the
        project is saved and the window shows the new state. rebuild=False: the clip cards are only updated."""
        self.window._write_tree()
        self.window.refresh(rebuild_path=bool(rebuild))

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
        return tabs.addTab(widget, str(title))

    def on(self, event: str, callback: Callable) -> None:
        """Hear an event (EVENTS): project_opened(folder), run_event(event dict), clip_made(clip id),
        run_finished(result: 0 = all made)."""
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; there are: {', '.join(EVENTS)}")
        self._heard[event].append((self._current, callback))

    def settings(self, name: str) -> QSettings:
        """The plugin's own settings: the app's settings file, under plugins/<name>/."""
        qs = self.window.qs
        s = QSettings(qs.fileName(), qs.format()) if qs.fileName() else QSettings()
        s.beginGroup(f"plugins/{name}")
        return s

    # ---- what is loaded
    def about_text(self) -> str:
        if not self.plugins:
            return "No plugins are installed."
        lines = []
        for p in self.plugins:
            lines.append(f"{p.name}  ({p.source}, module {p.target})\n    "
                         + (f"not in use: {p.error}" if p.error else "in use")
                         + "".join(f"\n    {x}" for x in p.problems))
        return "\n\n".join(lines)

    def show_about(self) -> None:
        box = QMessageBox(QMessageBox.Information, "Plugins", "Plugins add features that are not part of the app "
                          "itself. They are separately installed Python packages.", QMessageBox.Ok, self.window)
        box.setInformativeText(self.about_text())
        traces = "\n\n".join(f"{p.name}:\n{getattr(p, 'trace', '')}" for p in self.plugins
                             if p.error or p.problems)
        if traces:
            box.setDetailedText(traces)
        box.exec()
