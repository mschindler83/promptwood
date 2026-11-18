"""Random LoRA variations: which LoRAs an image gets (pinned ones always, up to the most), at which strength."""
import random

import pytest

from video_chain import lora_random as R


def choices(*specs):
    return [R.Choice(path, low, high, pinned) for path, low, high, pinned in specs]


def test_pinned_loras_are_always_in_and_count_towards_the_most():
    pool = choices(("a", 0.5, 1.0, True), ("b", 0.2, 0.4, False), ("c", 0.6, 0.6, False), ("d", -1.0, 0.0, False))
    rng = random.Random(1)
    sizes = set()
    for _ in range(400):
        got = R.pick(pool, 3, rng)
        paths = [p for p, _s in got]
        assert paths[0] == "a" and len(paths) == len(set(paths)) and 1 <= len(paths) <= 3
        sizes.add(len(paths))
        for p, s in got:
            spec = next(c for c in pool if c.path == p)
            assert spec.low - 1e-9 <= s <= spec.high + 1e-9 and round(s, 2) == s
    assert sizes == {1, 2, 3}                       # fewer than the most, too


def test_at_least_one_lora_and_never_more_than_there_are():
    pool = choices(("a", 1.0, 1.0, False), ("b", 1.0, 1.0, False))
    rng = random.Random(2)
    seen = {tuple(sorted(p for p, _s in R.pick(pool, 5, rng))) for _ in range(200)}
    assert seen == {("a",), ("b",), ("a", "b")}


def test_the_list_order_is_kept():
    pool = choices(("a", 1, 1, False), ("b", 1, 1, False), ("c", 1, 1, False))
    rng = random.Random(3)
    for _ in range(100):
        paths = [p for p, _s in R.pick(pool, 3, rng)]
        assert paths == sorted(paths)


def test_a_fixed_range_gives_that_strength_and_a_reversed_range_is_read_low_to_high():
    rng = random.Random(4)
    assert R.strength(0.7, 0.7, rng) == 0.7
    for _ in range(50):
        assert 0.2 <= R.strength(0.9, 0.2, rng) <= 0.9


def test_problems():
    assert R.problems([], 3) == ["Add at least one LoRA to the list"]
    pool = choices(("a", 1, 1, True), ("b", 1, 1, True), ("c", 1, 1, False))
    assert R.problems(pool, 1) == ["2 LoRAs are pinned, but an image may have at most 1"]
    assert R.problems(pool, 2) == []
    assert R.problems(pool, 0) == ["An image needs at least 1 LoRA"]


def test_only_pinned_loras_give_the_pinned_ones_alone():
    pool = choices(("a", 0.1, 0.3, True))
    assert [p for p, _s in R.pick(pool, 4, random.Random(5))] == ["a"]


def test_settings_round_trip_and_defaults():
    saved = R.to_saved(choices(("a", 0.3, 0.9, True)), most=3, n=8, same_seed=True)
    back = R.from_saved(saved, [{"path": "a", "strength": 1.0}, {"path": "b", "strength": 0.8}])
    assert back["choices"] == choices(("a", 0.3, 0.9, True), ("b", 0.4, 0.8, False))   # new: half to its strength
    assert (back["most"], back["n"], back["same_seed"]) == (3, 8, True)
    neg = R.from_saved({}, [{"path": "x", "strength": -0.6}])["choices"][0]
    assert (neg.low, neg.high) == (-0.6, -0.3)                                   # (a negative strength: towards 0)
    assert R.from_saved({"most": "x"}, [])["most"] == R.DEFAULT_MOST              # (broken settings: the defaults)


@pytest.mark.parametrize("most", [1, 2])
def test_the_most_is_never_exceeded(most):
    pool = choices(*[(str(i), 0.5, 1.0, False) for i in range(6)])
    rng = random.Random(6)
    assert all(1 <= len(R.pick(pool, most, rng)) <= most for _ in range(100))


def test_a_lora_listed_twice_is_one_choice():
    """The same file twice in the list must not be picked twice in one image."""
    got = R.from_saved(None, [{"path": "a", "strength": 1.0}, {"path": "b", "strength": 0.5},
                              {"path": "a", "strength": 0.4}])
    assert [c.path for c in got["choices"]] == ["a", "b"]
    for seed in range(50):
        paths = [p for p, _s in R.pick(got["choices"], 3, random.Random(seed))]
        assert len(paths) == len(set(paths))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_remembered_numbers_that_are_not_finite_fall_back_to_the_defaults(bad):
    saved = {"most": bad, "n": bad, "loras": {"a": {"low": bad, "high": bad}}}
    got = R.from_saved(saved, [{"path": "a", "strength": 1.0}])
    assert (got["most"], got["n"]) == (R.DEFAULT_MOST, R.DEFAULT_N)
    assert (got["choices"][0].low, got["choices"][0].high) == (0.5, 1.0)
