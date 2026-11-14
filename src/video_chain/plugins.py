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

A plugin that cannot be imported, or whose setup fails, is left out and listed with its error (Plugins → About
plugins…); the app starts without it."""
from __future__ import annotations

import importlib
import os
import traceback
from dataclasses import dataclass
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
    error: str = ""                 # why it is not in use ("" = loaded)

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
            if isinstance(e, KeyboardInterrupt):
                raise
            p.error = f"could not be imported: {type(e).__name__}: {e}"
            p.trace = traceback.format_exc()
    _LOADED = out
    return out


def reset() -> None:
    """Forget the loaded plugins (tests)."""
    global _LOADED
    _LOADED = None


def failed(p: Plugin, what: str, e: BaseException) -> None:
    """Note that a plugin failed in one of its functions (it stays listed with the error)."""
    p.error = f"{what} failed: {type(e).__name__}: {e}"
    p.trace = traceback.format_exc()


def wrap_generator(gen, args, log=None):
    """The generator the run uses: each plugin's generator(gen, args) may give another one (see the module's
    text). A plugin that fails is skipped (logged) and the generator before it is kept."""
    for p in load():
        fn = getattr(p.module, "generator", None) if p.loaded else None
        if not callable(fn):
            continue
        try:
            new = fn(gen, args)
        except Exception as e:  # noqa: BLE001
            failed(p, "generator()", e)
            if log:
                log(f"Plugin {p.name}: {p.error}")
            continue
        if new is not None:
            gen = new
    return gen
