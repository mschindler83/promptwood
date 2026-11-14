"""The path-escape guard: a record read from tree.json (possibly damaged or edited by hand) never reaches outside
the folder it belongs to."""
from __future__ import annotations

from pathlib import Path

from .. import chain


def inside(project: Path, sub: str, rel: str | Path) -> Path | None:
    """project/rel resolved, when it lies inside project/sub (nodes/, refs/, ends/, starts/); else None."""
    target = (Path(project) / rel).resolve()
    return target if (Path(project) / sub).resolve() in target.parents else None


def retire(project: Path, sub: str, rel: str | Path, *, resolved: bool = True) -> bool:
    """A file inside project/sub that is no longer used goes to old/ (chain.keep_aside; never deleted). False when
    rel is not inside sub (nothing is touched). resolved: hand keep_aside the resolved path (the default), not the
    path as written (a symlink as written is left alone by keep_aside)."""
    target = inside(project, sub, rel)
    if target is None:
        return False
    chain.keep_aside(project, target if resolved else Path(project) / rel)
    return True
