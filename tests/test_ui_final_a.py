"""Final wave A (main window): menu paths in hints, named constants, no dead getattr."""
import re
from pathlib import Path

UI = Path(__file__).resolve().parent.parent / "src" / "video_chain" / "ui"


def test_hints_only_name_menu_entries_that_exist():
    """Settings has one entry, Preferences; every 'Settings → ...' hint must continue with it (UX-5)."""
    bad = []
    for f in UI.glob("*.py"):
        for m in re.finditer(r"Settings → ([^\n\"]{0,40})", f.read_text(encoding="utf-8")):
            if not m.group(1).startswith("Preferences") and "API Keys" not in m.group(1):
                bad.append(f"{f.name}: {m.group(0)}")
    assert not bad


def test_note_failure_uses_the_named_engine_down_code():
    src = (UI / "mw_inspector.py").read_text(encoding="utf-8")
    assert "code == 4" not in src


def test_prompt_card_does_not_guess_its_own_attributes():
    assert 'getattr(self, "ov"' not in (UI / "clip_cards.py").read_text(encoding="utf-8")
