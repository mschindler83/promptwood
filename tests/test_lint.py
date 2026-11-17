"""The linter (ruff, configured in pyproject.toml [tool.ruff]) finds nothing in the non-UI code: no silent broad
`except`, no unused import or variable, no undefined name. A broad except that is meant says why with
`# noqa: BLE001  (reason)`."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(importlib.util.find_spec("ruff") is None, reason="ruff is not installed")
def test_ruff_finds_nothing():
    r = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache", "--output-format", "concise",
                        "src", "comfy_nodes"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, "ruff found problems:\n" + (r.stdout + r.stderr)[-6000:]
