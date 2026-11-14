"""No test run can hang or run long: one test is stopped after its time limit, a whole run after its budget."""
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

TESTS = Path(__file__).parent


def _run(tmp_path, body, *args, env=None):
    (tmp_path / "test_slow.py").write_text(textwrap.dedent(body))
    e = {**os.environ, "PYTHONPATH": str(TESTS), **(env or {})}
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "run_limits",
                        "-p", "no:xdist", "--rootdir", str(tmp_path), str(tmp_path / "test_slow.py"), *args],
                       cwd=tmp_path, env=e, capture_output=True, text=True, timeout=60)
    return p, time.monotonic() - t0


def test_a_whole_run_is_stopped_after_its_budget(tmp_path):
    p, took = _run(tmp_path, """
        import time
        def test_forever():
            time.sleep(30)
        """, env={"VIDEO_CHAIN_TEST_BUDGET": "3"})
    assert took < 15 and p.returncode != 0
    assert "test run budget" in (p.stdout + p.stderr)


def test_one_hanging_test_is_stopped_after_its_limit(tmp_path):
    p, took = _run(tmp_path, """
        import time
        def test_forever():
            time.sleep(30)
        """, "-o", "timeout=2", env={"VIDEO_CHAIN_TEST_BUDGET": "0"})
    assert took < 15 and p.returncode != 0
    assert "Timeout" in (p.stdout + p.stderr)


def test_the_project_runs_tests_in_parallel_with_limits():
    cfg = (TESTS.parent / "pyproject.toml").read_text()
    assert "timeout = " in cfg and "-n " in cfg
