"""Plugins: features that are not part of the app itself, added by separately installed Python packages.

A plugin is a Python module. The app finds it in one of two ways:
  * an entry point of the group "video_chain.plugins" in an installed package, e.g. in its pyproject.toml:
        [project.entry-points."video_chain.plugins"]
        my-features = "my_package.plugin"
  * the environment variable VIDEO_CHAIN_PLUGINS: module names, comma-separated (for trying one out without
    installing it; the module must be importable).
VIDEO_CHAIN_NO_PLUGINS=1 switches all plugins off (the app as it is without them; its own tests run like this).

What a plugin module can define (each is optional):
  setup(app)            called once when the window is ready, with the PluginHost (ui/plugin_host.py): it adds
                        menu entries, actions for clips, panels, and hears what happens (a project was opened, a
                        clip was made, a run finished). app.window is the main window itself, for everything the
                        host has no method for; that part of the app can change between versions.
  generator(gen, args)  called in the generation process (python -m video_chain) with the generator that makes the
                        clips and the run's arguments: return another object to use in its place (e.g. one that
                        wraps it, or another backend with the same methods), or None to keep it.

setup() and generator() are independent. setup() runs in the window, once per window; it is never called in the
generation process. So a plugin whose setup() failed in the window (and is "not in use" there) still has its
generator() used in the generation process, and a plugin without generator() needs nothing from it.
Listeners, actions and setup() run in the GUI thread, and the host's methods must be called from it.

A plugin that cannot be imported, or whose setup fails, is left out and listed with its error (Plugins → About
plugins…); the app starts without it. Whatever fails later (an action, a listener) is listed with the plugin,
which stays in use. A plugin cannot stop the app, not even with sys.exit(). What failed in a window is noted by
that window's host (each window has its own); Plugin.error only says that the module could not be imported."""
from __future__ import annotations

import importlib
import os
import traceback
from dataclasses import dataclass, field
from importlib import metadata
from types import ModuleType

GROUP = "video_chain.plugins"
ENV, OFF = "VIDEO_CHAIN_PLUGINS", "VIDEO_CHAIN_NO_PLUGINS"


@dataclass
class Plugin:
    name: str                       # the entry point's name, or the module's
    source: str                     # where it comes from: "package <name> <version>" / "VIDEO_CHAIN_PLUGINS"
    target: str                     # the module to import
    module: ModuleType | None = None
    error: str = ""                 # why it could not be imported ("" = imported); a window's setup failure is that
                                    # window's host's note (State), as all windows of a process share these objects
    trace: str = ""                 # the traceback of that import error (or of a failure in the generation process)
    problems: list = field(default_factory=list)   # what failed in the generation process (it stays in use)

    @property
    def loaded(self) -> bool:
        return self.module is not None and not self.error


def enabled() -> bool:
    return os.environ.get(OFF, "").strip().lower() not in ("1", "true", "yes", "on")


def discover() -> list[Plugin]:
    """The plugins there are (not imported yet): the entry points, then VIDEO_CHAIN_PLUGINS; each module once."""
    if not enabled():
        return []
    out, seen = [], set()
    try:
        points = sorted(metadata.entry_points(group=GROUP), key=lambda p: p.name)
    except Exception:  # noqa: BLE001  (damaged package metadata: no entry points)
        points = []
    for p in points:
        target = str(p.value).split(":", 1)[0].strip()
        if not target or target in seen:
            continue
        seen.add(target)
        dist = getattr(p, "dist", None)
        where = f"package {dist.name} {dist.version}" if dist is not None else "an installed package"
        out.append(Plugin(p.name, where, target))
    for target in (x.strip() for x in os.environ.get(ENV, "").split(",")):
        if target and target not in seen:
            seen.add(target)
            out.append(Plugin(target, ENV, target))
    return out


_LOADED: list[Plugin] | None = None


def load(again: bool = False) -> list[Plugin]:
    """Import the plugins (once per process). One that fails is kept in the list with its error."""
    global _LOADED
    if _LOADED is not None and not again:
        return _LOADED
    out = discover()
    for p in out:
        try:
            p.module = importlib.import_module(p.target)
        except BaseException as e:  # noqa: BLE001  (also SystemExit in a plugin: the app still starts)
            interrupt(e)
            p.error = f"could not be imported: {type(e).__name__}: {e}"
            p.trace = traceback.format_exc()
    _LOADED = out
    return out


def reset() -> None:
    """Forget the loaded plugins (tests)."""
    global _LOADED
    _LOADED = None


@dataclass
class State:
    """What failed for one plugin in one window (the host keeps one per plugin: the Plugin objects are shared)."""
    error: str = ""                 # why its setup failed ("" = in use)
    problems: list = field(default_factory=list)
    trace: str = ""


def interrupt(e: BaseException) -> None:
    """Call in `except BaseException as e`: a plugin may raise SystemExit (sys.exit()) and the like, which must
    not end the app, but Ctrl+C still does."""
    if isinstance(e, KeyboardInterrupt):
        raise e


def failed(p: Plugin, what: str, e: BaseException, fatal: bool = False, state: State | None = None) -> str:
    """Note that one of a plugin's functions failed: -> the sentence about it. fatal (its setup): the plugin is
    not in use; else it stays in use and the failure is listed with it. state: where a window notes it (without,
    the plugin itself: the generation process has no window)."""
    text = f"{what} failed: {type(e).__name__}: {e}"
    where = state if state is not None else p
    if fatal:
        where.error = text
    elif text not in where.problems:
        where.problems.append(text)
    where.trace = traceback.format_exc()
    return text


def wrap_generator(gen, args, log=None):
    """The generator the run uses: each plugin's generator(gen, args) may give another one (see the module's
    text). A plugin that fails is skipped (logged) and the generator before it is kept."""
    try:
        found = load()
    except BaseException as e:  # noqa: BLE001
        interrupt(e)
        return gen
    for p in found:
        try:
            fn = getattr(p.module, "generator", None) if p.loaded else None
            if not callable(fn):
                continue
            new = fn(gen, args)
        except BaseException as e:  # noqa: BLE001  (also SystemExit: not the generation process's exit code)
            interrupt(e)
            text = failed(p, "generator()", e)
            if log:
                log(f"Plugin {p.name}: {text}")
            continue
        if new is not None:
            gen = new
    return gen
