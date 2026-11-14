"""Copy a whole project folder (Save as): file by file, with progress and a way to stop; a stopped copy is removed."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Callable

from . import film as F
from . import tree as T

SKIP = shutil.ignore_patterns("*.part", "*.part.*", "video-chain-*")      # (what is never copied)


class CopyCancelled(Exception):
    """The copy was stopped; what was copied so far is removed."""


def copy_file(src: str, dst: str) -> str:
    """Copy one file (the step the progress counts; a name of its own so a test can slow it down)."""
    return shutil.copy2(src, dst)


def _ignore(d, names):
    out = set(SKIP(d, names))
    if Path(d).parent.name == F.SCENES_DIR and T.REFS_DIR in names:
        out.add(T.REFS_DIR)           # a scene's link to the film's refs/: made again after the copy
    return out


def _count(source: Path) -> int:
    n = 0
    for d, dirs, files in os.walk(source):
        ignored = _ignore(d, dirs + files)
        dirs[:] = [x for x in dirs if x not in ignored]
        n += sum(1 for f in files if f not in ignored)
    return n


def copy_project(source: Path, target: Path, *, progress: Callable[[int, int], None],
                 cancel: threading.Event) -> None:
    """Copy `source` to `target` (a missing or empty folder). progress(done, total) after each file; when `cancel` is
    set the copy stops with CopyCancelled and what it made is removed (a folder that was already there stays, empty).
    Any other failure removes it the same way and is raised."""
    source, target = Path(source), Path(target)
    existed = target.exists()
    total, done = _count(source), 0

    def one(s: str, d: str) -> str:
        nonlocal done
        if cancel.is_set():
            raise CopyCancelled()
        copy_file(s, d)
        done += 1
        progress(done, total)
        return d
    try:
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=_ignore, copy_function=one)
        if cancel.is_set():
            raise CopyCancelled()
        for sc in F.Film.load(target).scenes:  # the copy remembers where its start images came from
            d = (target / sc.dir).resolve()
            if sc.dir != "." and not (d / T.REFS_DIR).exists():
                (target / T.REFS_DIR).mkdir(exist_ok=True)
                os.symlink(os.path.relpath(target / T.REFS_DIR, d), d / T.REFS_DIR, target_is_directory=True)
            if (d / T.TREE_FILE).is_file():
                T.save(d, T.load(d))
    except BaseException:  # noqa: BLE001  (a half copy would block the next try)
        if existed:
            for child in target.iterdir():
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() and not child.is_symlink() \
                    else child.unlink(missing_ok=True)
        else:
            shutil.rmtree(target, ignore_errors=True)
        raise
