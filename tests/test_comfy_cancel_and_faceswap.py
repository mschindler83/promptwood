"""CMF-1: the face swap runs on a generator of frames; CMF-2: Stop on a clip still queued in ComfyUI."""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fake_comfy import FakeComfy

from video_chain import comfy as C

NODES = Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes"
sys.path.insert(0, str(NODES))


def test_faceswapper_run_takes_a_generator_with_progress():
    import faceswap as F
    fs = F.FaceSwapper.__new__(F.FaceSwapper)
    fs.faces = lambda f: [(0, 0, 10, 10, 1.0, None)]
    fs.swap_face = lambda f, lm, ident, enh=0.0: f + 1
    seen, got = [], {}

    def gen():
        for k in range(3):
            yield np.full((4, 4, 3), k, np.uint8)
    _none, done = fs.run(gen(), None, progress=lambda k, n: seen.append((k, n)),
                         out=lambda k, f: got.__setitem__(k, f), total=3)
    assert done == 3 and seen == [(1, 3), (2, 3), (3, 3)]
    assert got[2][0, 0, 0] == 3


def test_stop_ends_the_wait_of_a_job_still_queued(tmp_path):
    srv = FakeComfy(tmp_path / "comfy")
    srv.hold = True
    try:
        g = C.ComfyGenerator(srv.url, timeout=30)
        pid = g._queue({"1": {"class_type": "X", "inputs": {}}})
        result = {}

        def work():
            try:
                g._wait(pid, None)
            except Exception as e:  # noqa: BLE001
                result["err"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        time.sleep(0.7)
        g.interrupt()
        t.join(5)
        assert not t.is_alive(), "_wait kept polling after Stop"
        assert isinstance(result["err"], C.ComfyJobFailed)
        assert pid in srv.prompts_deleted
    finally:
        srv.close()
