"""set_voice leaves no half-made voice files when its reference cannot be cut."""
import shutil
import subprocess

import pytest

from video_chain import chain
from video_chain import tree as T

if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)


def test_a_failing_reference_cut_leaves_no_voice_file(tmp_path, monkeypatch):
    src = tmp_path / "long.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=12", str(src)],
                   check=True)
    t = T.Tree()
    t.options["cast"] = [{"name": "Anna", "images": [], "desc": "", "lead": False}]
    real = chain.to_wav
    calls = []

    def second_fails(*a, **k):
        calls.append(a)
        if len(calls) == 2:                       # the clone reference (the voice itself was cut before it)
            raise RuntimeError("ffmpeg died")
        return real(*a, **k)
    monkeypatch.setattr(chain, "to_wav", second_fails)
    with pytest.raises(RuntimeError):
        T.set_voice(tmp_path, t, "Anna", src)
    assert not list((tmp_path / "refs").glob("voice*")) and "voice" not in t.options["cast"][0]
