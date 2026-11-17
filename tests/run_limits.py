"""pytest plugin: a whole test run has a budget (VIDEO_CHAIN_TEST_BUDGET seconds, default 300; 0 = none).

When it is used up, the stacks of all threads are printed and the run ends at once (exit code 3), also when a test
is stuck inside Qt or a subprocess where nothing else could stop it. One test's own limit is pytest-timeout's
(`timeout` in pyproject.toml)."""
from __future__ import annotations

import faulthandler
import os
import sys
import threading

BUDGET_DEFAULT = 300.0


def pytest_configure(config):
    if hasattr(config, "workerinput"):          # an xdist worker: the controller holds the budget for the run
        return
    try:
        budget = float(os.environ.get("VIDEO_CHAIN_TEST_BUDGET", BUDGET_DEFAULT))
    except ValueError:
        budget = BUDGET_DEFAULT
    if budget <= 0:
        return

    def stop():
        capman = config.pluginmanager.getplugin("capturemanager")
        if capman is not None:                     # (pytest holds stderr: what is written now must reach the user)
            try:
                capman.suspend_global_capture(in_=True)
            except Exception:  # noqa: BLE001
                pass
        sys.stderr.write(f"\n\n*** test run budget of {budget:g} s used up: the run is stopped. A run must stay "
                         "under 5 minutes; select fewer tests (-k / file names) or speed them up. Stacks:\n")
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(3)
    t = threading.Timer(budget, stop)
    t.daemon = True
    t.start()
