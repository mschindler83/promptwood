"""Prompt variations engine (the Dynamic Prompter node's behaviour plus its planned ideas)."""
from collections import Counter

from video_chain import variations as V


def vars_(**kw):
    return [V.Variable(k, [V.Value(*x) if isinstance(x, tuple) else V.Value(x) for x in vals]) for k, vals in kw.items()]


def test_combinations_like_the_node():
    vs = vars_(p1=["cat", "dog"], p2=["tree", "fence"])
    jobs = V.plan_jobs(vs, "the {p1} jumps over the {p2}")
    assert [j.prompt for j in jobs] == ["the cat jumps over the tree", "the dog jumps over the tree",
                                        "the cat jumps over the fence", "the dog jumps over the fence"]
    assert [j.index for j in jobs] == [0, 1, 2, 3]                # first variable fastest (mixed radix)
    rep = V.plan_jobs(vs, "{p1} {p2}", repeat=2, count=5, start=3)   # continues; each combination twice
    assert [(j.step, j.prompt) for j in rep] == [(3, "dog tree"), (4, "cat fence"), (5, "cat fence"),
                                                 (6, "dog fence"), (7, "dog fence")]
    assert [j.prompt for j in V.plan_jobs(vs, "{p1}", count=3)] == ["cat", "dog", "cat"]   # only used vars count


def test_shuffle_covers_all_and_random_weights():
    vs = vars_(a=["1", "2", "3"], b=["x", "y"])
    jobs = V.plan_jobs(vs, "{a}{b}", mode="shuffle", seed=5)
    assert sorted(j.prompt for j in jobs) == sorted(["1x", "2x", "3x", "1y", "2y", "3y"]) and len(jobs) == 6
    assert [j.prompt for j in jobs] != ["1x", "2x", "3x", "1y", "2y", "3y"]
    assert [j.prompt for j in V.plan_jobs(vs, "{a}{b}", mode="shuffle", seed=5)] == [j.prompt for j in jobs]
    w = vars_(c=[("cat", 9), ("dog", 1)])
    picks = Counter(j.prompt for j in V.plan_jobs(w, "{c}", mode="random", count=400, seed=1))
    assert picks["cat"] > 300 and picks["dog"] > 10
    off = [V.Variable("c", [V.Value("cat"), V.Value("dog", on=False)])]
    assert {j.prompt for j in V.plan_jobs(off, "{c}", mode="random", count=20)} == {"cat"}


def test_optional_sections_negative_and_tidy():
    vs = vars_(subject=["cat"], style=["watercolor", ""], light=["dramatic", ""])
    t = "a photo of a {subject}{?style|, in {style} style}{?light|, {light} lighting}"
    assert {j.prompt for j in V.plan_jobs(vs, t)} == {
        "a photo of a cat, in watercolor style, dramatic lighting", "a photo of a cat, in watercolor style",
        "a photo of a cat, dramatic lighting", "a photo of a cat"}
    vs2 = vars_(mood=["happy", ""])
    jobs = V.plan_jobs(vs2, "a {mood} dog, running", negative="sad, {mood}")
    assert [(j.prompt, j.negative) for j in jobs] == [("a happy dog, running", "sad, happy"), ("a dog, running", "sad")]
    p = V.Plan(vars_(a=["1"], b=[]), "{a} {b} {c}")
    assert p.missing == ["b", "c"] and p.total == 1 and V.substitute("{a} {zz}", {"a": "1"}) == "1 {zz}"
    assert V.names_in("x {?a|{a}} {b}", "{c} {a}") == ["a", "b", "c"]


def test_node_json_and_lists():
    vs = V.from_json('{"p1": ["cat", ["dog", 3]], "bad name": ["x"], "p2": []}')
    assert [(v.name, [(x.text, x.weight) for x in v.values]) for v in vs] == [("p1", [("cat", 1), ("dog", 3)]),
                                                                            ("p2", [])]
    assert V.from_json(V.to_json(vs))[0].values[1].weight == 3
    assert V.from_state(V.to_state(vs))[0].values[1].text == "dog"
    assert V.wildcard_values("# animals\ncat\n\n dog \n#x") == ["cat", "dog"]
    assert V.split_values("red, green , blue") == ["red", "green", "blue"]
    assert V.split_values("a, b\nc") == ["a, b", "c"]


def test_review_cases():
    # a value with braces is not read again; normal prompts are not tidied; {cat|dog} is no placeholder
    assert V.substitute("{?c|see {a}} and {b}", {"a": "{b}", "b": "B", "c": "1"}) == "see {b} and B"
    assert V.substitute("Shot 1: {x}\n...then he runs ()", {"x": "cat"}) == "Shot 1: cat\n...then he runs ()"
    assert V.substitute("[Shot 1] @Anna says: hi !\n\n? {x}", {"x": "ok"}) == "[Shot 1] @Anna says: hi !\n\n? ok"
    assert V.substitute("a {m} dog, {s}, running", {"m": "", "s": ""}) == "a dog, running"
    assert V.names_in("{cat|dog} {?s|x} {a}") == ["s", "a"]
    assert [v.name for v in V.from_json('{"a": [[], "x"]}')] == ["a"] and len(V.from_json('{"a": [[], "x"]}')[0].values) == 1
    big = [V.Variable(f"v{i}", [V.Value(str(k)) for k in range(10)]) for i in range(10)]   # 10^10 combinations
    assert len(V.plan_jobs(big, " ".join(f"{{v{i}}}" for i in range(10)))) == V.MAX_IMAGES
    assert len(V.plan_jobs(big, " ".join(f"{{v{i}}}" for i in range(10)), count=5, mode="shuffle")) == 5



def test_tidy_only_the_line_with_the_empty_value():
    t = "a cat,\n  {x}, smile,\nend"
    assert V.substitute(t, {"x": ""}) == "a cat,\n  smile\nend"           # other lines keep their commas
    assert V.substitute("  {x}, b", {"x": "a"}) == "  a, b"
