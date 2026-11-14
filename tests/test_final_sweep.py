"""Tests of the last sweep (MWB-24 and the others below)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_no_shortcut_needs_ctrl_and_alt_together():
    """On a German keyboard AltGr is Ctrl+Alt: Ctrl+Alt+<key> collides with @ { } [ ] \\ ~ | (MWB-24)."""
    from video_chain.ui import shortcuts as S
    for s in S.SHORTCUTS:
        for k in s.keys.split(" / "):
            parts = {p.strip().lower() for p in k.split("+")}
            assert not {"ctrl", "alt"} <= parts, f"{s.id}: {k}"


def test_no_two_shortcuts_of_one_place_share_keys():
    from video_chain.ui import shortcuts as S
    seen: dict[tuple[str, str], str] = {}
    for s in S.SHORTCUTS:
        for k in s.keys.split(" / "):
            assert (s.where, k) not in seen, f"{s.id} and {seen[(s.where, k)]} both use {k}"
            seen[(s.where, k)] = s.id
