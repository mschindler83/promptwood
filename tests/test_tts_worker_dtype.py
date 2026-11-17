"""tts_worker: the Whisper pipeline gets `dtype=` where transformers knows it, `torch_dtype=` before (stubbed libs)."""
import json
import subprocess

import pytest
import test_tts_cli_fixes as base

STUB = '''
__version__ = "{version}"
def pipeline(*a, **k):
    import os
    with open(os.environ["MARK"] + ".kw", "w") as f: f.write(" ".join(sorted(k)))
    return lambda audio, **kw: {{"text": " hello "}}
'''


@pytest.mark.parametrize("version,key", [("4.57.1", "dtype"), ("4.56.0", "dtype"), ("4.45.2", "torch_dtype"),
                                         ("5.0.0.dev0", "dtype")])
def test_the_whisper_pipeline_uses_the_dtype_name_of_its_transformers(tmp_path, monkeypatch, version, key):
    monkeypatch.setitem(base.STUBS, "soundfile.py", base.STUBS["soundfile.py"] + "def read(p, dtype=None): return [0.0] * 100, 16000\n")
    monkeypatch.setitem(base.STUBS, "transformers.py", STUB.format(version=version))
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1", str(wav)], check=True)
    w = base.Worker(tmp_path)
    try:
        a = w.ask({"cmd": "transcribe", "model": "m", "audio": str(wav), "language": None})
        assert a["ok"], a
        used = (tmp_path / "mark.kw").read_text().split()
        assert key in used and ({"dtype", "torch_dtype"} - {key}).isdisjoint(used)
    finally:
        w.p.kill()
        w.p.wait()
    assert json.dumps(a)
