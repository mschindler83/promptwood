"""One helper for the number shown for a clip id ("n3" -> "3"), UMD-32."""
from video_chain import tree as T


def test_short_id_drops_the_prefix_letter():
    assert T.short_id("n3") == "3" and T.short_id("v12") == "12"
    assert T.short_id("") == "" and T.short_id("x") == ""
