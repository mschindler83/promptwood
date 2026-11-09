"""Camera moves in MiniMax H3's own words (its prompt guide, 4.3): settings and #tokens."""
from video_chain import camera as C


def test_sentences_follow_the_h3_guide():
    assert C.sentence({"move": "push-in", "amplitude": "small", "speed": "slow"}) == \
        "The camera pushes in with small amplitude at slow speed."          # (the guide's own example)
    assert C.sentence({"move": "pan-right", "amplitude": "large", "speed": "fast"}) == \
        "The camera pans right with large amplitude at fast speed."
    assert C.sentence({"move": "static", "speed": "fast"}) == "The camera holds a static shot."   # (no speed)
    assert C.sentence({"move": "truck-right"}) == "The camera trucks right."  # medium / normal left out
    assert C.sentence(None) == "" and C.sentence({"move": "fly-away"}) == ""
    assert len(C.MOVES) == 20 and all(m in C.MOVES for m in ("arc", "tracking", "pov", "roll-ccw", "shake-strongly"))


def test_tokens_expand_in_place():
    assert C.parse("push-in-small-slow") == ("push-in", "small", "slow")
    assert C.parse("push-in-slow-small") == ("push-in", "small", "slow")   # (either order)
    assert C.parse("pan-left-fast") == ("pan-left", None, "fast")
    assert C.parse("static-slow") is None and C.parse("hashtag") is None
    assert C.expand("#push-in-slow toward the letter") == "The camera pushes in at slow speed toward the letter"
    assert C.expand("She looks up, then #pan-right reveals the door.") == \
        "She looks up, then the camera pans right reveals the door."
    assert C.expand("[Shot 2] #static as she waits") == "[Shot 2] The camera holds a static shot as she waits"
    assert C.expand("room #2 and a#b and #nonsense stay") == "room #2 and a#b and #nonsense stay"
    assert C.apply("She waits", {"move": "push-in", "speed": "slow"}) == \
        "She waits. The camera pushes in at slow speed."
    assert C.apply("", {"move": "static"}) == "The camera holds a static shot."
    assert C.apply("Rain. #tracking-fast", None) == "Rain. The camera follows the subject in a tracking shot at fast speed"
    offered = C.tokens()
    assert "push-in" in offered and "push-in-small-slow" in offered and "static" in offered
    assert "static-slow" not in offered and all(C.is_token(t) for t in offered)


def test_describe_and_clean():
    assert C.describe({"move": "push-in", "amplitude": "small", "speed": "slow"}) == "push in · small · slow"
    assert C.describe({"move": "pov", "speed": "fast"}) == "pov"
    assert C.clean({"move": "static", "speed": "fast"}) == {"move": "static"}
    assert C.clean({"move": "pan-left", "amplitude": "medium", "speed": "slow"}) == {"move": "pan-left", "speed": "slow"}
    assert C.clean({"move": None}) is None and C.clean("push-in") is None
