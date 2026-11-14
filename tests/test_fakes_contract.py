"""The fake generator keeps the real generator's call signatures (a new argument on the real one breaks here)."""
import inspect

import pytest

from fakes import FakeGenerator


@pytest.mark.parametrize("name", ["generate", "refine", "seedvr"])
def test_fake_generator_signature_matches_the_real_one(name):
    from video_chain.comfy import ComfyGenerator
    real = [p for p in inspect.signature(getattr(ComfyGenerator, name)).parameters if p != "self"]
    fake = [p for p in inspect.signature(getattr(FakeGenerator, name)).parameters if p != "self"]
    assert fake[:len(real)] == real or set(real) <= set(fake), (real, fake)


def test_a_fresh_fake_has_no_refines_or_seedvrs():
    g = FakeGenerator()
    assert g.refines == [] and g.seedvrs == []
