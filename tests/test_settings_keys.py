"""Every QSettings key of the UI is a constant in ui/settings_keys.py, never a string literal at the call (XC-14)."""
import re
from pathlib import Path

UI = Path(__file__).resolve().parent.parent / "src" / "video_chain" / "ui"
LITERAL = re.compile(r"""(qs_value\(\s*[\w.\[\]]+,\s*["']|\.value\(["']|\.setValue\(["']|\.contains\(["']|\._remember\(["'])""")


def test_no_settings_key_is_spelled_as_a_literal_outside_the_keys_module():
    bad = [f"{f.name}:{n}" for f in sorted(UI.glob("*.py")) if f.name != "settings_keys.py"
           for n, line in enumerate(f.read_text().splitlines(), 1) if LITERAL.search(line)]
    assert not bad, bad


def test_the_keys_are_unique_constants():
    from video_chain.ui import settings_keys as SK
    values = [v for k, v in vars(SK).items() if k.isupper()]
    assert values and len(values) == len(set(values))
