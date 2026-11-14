"""QSettings.value with its type, typed (kept apart: ui_logic imports modules that read settings)."""
from __future__ import annotations

from typing import Any, TypeVar, cast

_V = TypeVar("_V")


def qs_value(qs: Any, key: str, default: _V, kind: type[_V]) -> _V:
    """QSettings.value(key, default, type=kind); the PySide6 stubs type that call as object."""
    return cast(_V, qs.value(key, default, type=kind))
