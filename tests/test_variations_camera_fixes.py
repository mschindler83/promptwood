"""variations (damaged imports, lazy jobs, random picks' index) and camera (newline starts, odd stored values)."""
import json

import pytest

from video_chain import camera, variations as V


def test_odd_weights_in_a_json_import_do_not_crash():
    vs = V.from_json(json.dumps({"p": [["a", "high"], ["b", [1]], {"text": "c", "weight": "heavy"},
                                       {"text": "d", "weight": [2]}, ["e", -4], ["f", "3"]]}))
    assert [x.weight for x in vs[0].values] == [1, 1, 1, 1, 1, 3]


def test_odd_weights_in_a_saved_state_do_not_crash():
    vs = V.from_state([{"name": "p", "values": [{"text": "a", "weight": "x"}, {"text": "b", "weight": None},
                                                 {"text": "c", "weight": -2}, {"text": "d", "weight": [1]}]}])
    assert [x.weight for x in vs[0].values] == [1, 1, 1, 1]


def test_jobs_can_be_made_one_by_one():
    vs = [V.Variable("p", [V.Value("a"), V.Value("b")])]
    it = V.iter_jobs(vs, "x {p}", count=4)
    assert next(it).prompt == "x a" and next(it).prompt == "x b"
    assert [j.prompt for j in V.plan_jobs(vs, "x {p}", count=3)] == ["x a", "x b", "x a"]


def test_random_job_index_matches_its_combination_even_with_equal_texts():
    vs = [V.Variable("p", [V.Value("a"), V.Value("a"), V.Value("b")]), V.Variable("q", [V.Value("1"), V.Value("2")])]
    jobs = V.plan_jobs(vs, "{p}{q}", mode="random", count=50, seed=3)
    assert all(j.combo["p"] in "ab" and 0 <= j.index < 6 for j in jobs)
    assert len({j.index for j in jobs}) > 3                        # not collapsed onto the first duplicate


def test_camera_token_after_a_line_break_starts_a_sentence():
    assert camera.expand("A cat\n#push-in") == "A cat\nThe camera pushes in"
    assert camera.expand("A cat #push-in") == "A cat the camera pushes in"


@pytest.mark.parametrize("bad", [{"move": ["push-in"]}, {"move": {"a": 1}}, {"move": "arc", "amplitude": ["x"]},
                                 {"move": "pan-left", "speed": {"a": 1}}])
def test_camera_with_unhashable_values_is_ignored_not_a_crash(bad):
    camera.sentence(bad)
    camera.describe(bad)
    camera.clean(bad)
    assert camera.sentence({"move": ["x"]}) == "" and camera.describe({"move": {"a": 1}}) == ""
