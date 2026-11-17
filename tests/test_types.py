"""The type checker (mypy, configured in pyproject.toml [tool.mypy]) finds nothing in the checked code.

Keeps the annotations honest: a new Optional used without a check, a TypedDict key that is not in the schema, a wrong
argument type fail here. The ui/ package is excluded in pyproject.toml until it is annotated."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(importlib.util.find_spec("mypy") is None, reason="mypy is not installed")
def test_mypy_has_no_errors(tmp_path):
    r = subprocess.run([sys.executable, "-m", "mypy", "--no-color-output", "--no-error-summary",
                        "--cache-dir", str(tmp_path / "mypy-cache")],
                       cwd=ROOT, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, "mypy found problems:\n" + (r.stdout + r.stderr)[-6000:]
