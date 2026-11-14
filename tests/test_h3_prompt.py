"""MiniMax H3's official prompt format (h3_prompt): headers, the three fields, the six reference sections, spoken lines,
sound sections and pass-through. Synthetic text only."""
import pytest

from video_chain import h3_prompt as H
from video_chain import tree as T

SCAPE = "overall_soundscape: Natural ambient sound that fits the scene."
NO_MUSIC = "non_diegetic_music: N/A"


def lines_of(text, cast_names, default=None):
    """The spoken lines of a synthetic text, parsed the way tree.py does (dialogue) with a synthetic cast."""
    options = {"cast": [{"name": n, "images": [], "desc": ""} for n in cast_names]}
    lead = T.character(options, default) if default else None
    return T.dialogue(options, text, default=lead)


def fields(desc, scape=SCAPE, music=NO_MUSIC):
    return f"integrated_multimodal_description: {desc}\n\n{scape}\n\n{music}"


# ----------------------------------------------------------------------------- base formats (the table)

def test_nothing_is_t2va_three_fields_without_header():
    assert H.compose("A baker opens the shutters.") == fields("[Shot 1] A baker opens the shutters.")


def test_start_frame_is_i2va():
    assert H.compose("A woman walks.", start=True) == (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
        "referenced.\n\n" + fields("[Shot 1] A woman walks."))


def test_start_and_end_is_fl2va_with_seconds_and_last_shot():
    out = H.compose("She opens the umbrella.", start=True, end=True, seconds=8)
    assert out == ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
                   "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of "
                   "the target video.\n\n" + fields("[Shot 1] She opens the umbrella."))
    two = H.compose("[Shot 1] A door. [Shot 2] At 00:03.500, the camera cuts to a hall.", start=True, end=True,
                    seconds=5.125)
    assert two.splitlines()[0].endswith("Picture 2 (from Shot 2) aligns with the 5.12-second mark of the target video.")
    assert "integrated_multimodal_description: [Shot 1] A door. [Shot 2] At 00:03.500, the camera cuts to a hall." \
        in two and two.count("[Shot 1]") == 1


def test_end_only_is_l2va():
    out = H.compose("A glass falls.", end=True, seconds=6)
    assert out == ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with "
                   "the 6.00-second mark of the target video.\n\n" + fields("[Shot 1] A glass falls."))
    three = H.compose("A. [Shot 2] At 00:02.000, B. [Shot 3] At 00:04.000, C.", end=True, seconds=6)
    assert three.startswith("How the reference pictures align with the target video — <Picture 1> (from [Shot 3])")
    assert "description: [Shot 1] A. [Shot 2] At 00:02.000, B. [Shot 3] At 00:04.000, C." in three


def test_headers_are_the_guides_lines():
    assert H.header(False, False) == ""
    assert H.header(True, False) == H.I2VA_HEADER
    assert H.header(True, True, 3, 2) == H.FL2VA_HEADER.format(n=2, s="3.00")
    assert H.header(False, True, 2.999, 1) == H.L2VA_HEADER.format(n=1, s="3.00")
    with pytest.raises(ValueError):
        H.header(False, True)


def test_body_with_shot_1_keeps_its_markers_and_text_before_goes_after_it():
    assert H.compose("[Shot 1] A man waits.") == fields("[Shot 1] A man waits.")
    assert H.compose("[shot 1] A man waits.") == fields("[Shot 1] A man waits.")
    assert H.compose("Cinematic, 35mm film [Shot 1] A man waits.") == \
        fields("[Shot 1] Cinematic, 35mm film. A man waits.")      # (the style prefix of full_prompt)


def test_triggers_extra_and_empty_body():
    out = H.compose("ohwx woman walks.", triggers=["ohwx", "sks style"], extra=["The camera pushes in.", "At 2.0 s: she stops."])
    assert out == fields("[Shot 1] sks style. ohwx woman walks. The camera pushes in. At 2.0 s: she stops.")
    assert H.compose("") == fields("[Shot 1]")
    assert H.compose("", triggers=["tok"]) == fields("[Shot 1] tok")


# ----------------------------------------------------------------------------- sound sections and pass-through

def test_trailing_sound_lines_either_order_fill_the_fields():
    body = "A man walks.\noverall_soundscape: Rain on the roof.\nnon_diegetic_music: Slow piano."
    assert H.split_sound_sections(body) == ("A man walks.", "Rain on the roof.", "Slow piano.")
    swapped = "A man walks.\n\nnon_diegetic_music: Slow piano.\n\noverall_soundscape: Rain on the roof.\n"
    assert H.split_sound_sections(swapped) == ("A man walks.", "Rain on the roof.", "Slow piano.")
    assert H.compose(body) == fields("[Shot 1] A man walks.", "overall_soundscape: Rain on the roof.",
                                     "non_diegetic_music: Slow piano.")
    only = H.compose("A man walks.\nnon_diegetic_music: N/A")
    assert only == fields("[Shot 1] A man walks.")
    assert H.split_sound_sections("A man walks.") == ("A man walks.", None, None)
    alone = "A man walks.\noverall_soundscape:\nRain on the roof.\nWind."      # (the ref guide's layout)
    assert H.split_sound_sections(alone) == ("A man walks.", "Rain on the roof. Wind.", None)
    middle = "overall_soundscape: rain\nThen he walks."                       # not at the end: body text
    assert H.split_sound_sections(middle) == (middle, None, None)


def test_explicit_sound_arguments_win():
    out = H.compose("A man walks.\noverall_soundscape: Rain.", soundscape="Wind.", music="Drums.")
    assert out == fields("[Shot 1] A man walks.", "overall_soundscape: Wind.", "non_diegetic_music: Drums.")


def test_music_in_the_body_changes_the_default():
    out = H.compose("A woman hums a song while she cooks.")
    assert out.endswith("non_diegetic_music: As described in the shots.")
    assert H.compose("A woman cooks.\nnon_diegetic_music: Soft strings.").endswith("non_diegetic_music: Soft strings.")
    assert H.talks_about_music("background music") and H.talks_about_music("a fast beat")
    assert not H.talks_about_music("a heartbeat monitor") and not H.talks_about_music("She walks.")


def test_structured_text_passes_through():
    for own in ("  integrated_multimodal_description: [Shot 1] x\n\noverall_soundscape: y\n\nnon_diegetic_music: N/A  ",
                "subject_definitions:\n<Subject 1> is x.",
                H.I2VA_HEADER + "\n\nintegrated_multimodal_description: [Shot 1] x",
                H.L2VA_HEADER.format(n=1, s="2.00") + "\n\nintegrated_multimodal_description: [Shot 1] x"):
        assert H.is_structured(own)
        assert H.compose(own, start=True, subjects=[H.Subject("Anna", pictures=(1,))], triggers=["tok"]) == own.strip()
    assert not H.is_structured("A woman says integrated_multimodal_description: hi")


# ----------------------------------------------------------------------------- spoken lines, base formats

def test_spoken_lines_in_base_format():
    text = '@Anna looks at @Ben and says "Hi." "Hello," says @Ben. "Bye."'
    out = H.compose(text, lines=lines_of(text, ["Anna", "Ben"]),
                    resolve=lambda w: {"anna": "Anna", "ben": "Ben"}.get(w.lower()))
    assert out == fields("[Shot 1] Anna (S1) looks at Ben (S2) and says, <d>[English] Hi.</d> Ben (S2) says, "
                         "<d>[English] Hello</d>. Ben (S2) says, <d>[English] Bye.</d>")
    plain = 'A voice says "Wait!" and @Max runs.'                  # unknown @word stays, unnamed speaker: just <d>
    assert H.compose(plain, lines=lines_of(plain, []), language="Deutsch") == \
        fields("[Shot 1] A voice says <d>[Deutsch] Wait!</d> and @Max runs.")
    spoken = '@Anna says "Call @Ben now."'                         # an @Name inside a line is a spoken word
    assert "<d>[English] Call Ben now.</d>" in H.compose(spoken, lines=lines_of(spoken, ["Anna", "Ben"]),
                                                         resolve=lambda w: w.capitalize())


def test_speaker_ids_and_voice_numbers():
    text = '"Hi," says @Ben. @Anna answers "Yes." @Ben says "Ok."'
    ls = lines_of(text, ["Anna", "Ben"])
    assert H.speaker_ids(ls) == {"Ben": 1, "Anna": 2}
    subs = [{"name": "Anna", "voice": True}, {"name": "Ben", "voice": True}]
    assert H.voice_numbers(subs, ls, sounds=2) == {"Ben": 3, "Anna": 4}
    assert H.voice_numbers([{"name": "Anna", "voice": True}, {"name": "Ben"}], ls) == {"Anna": 1}


# ----------------------------------------------------------------------------- reference format

def test_reference_two_subjects_of_different_kinds_with_numbered_start():
    out = H.compose("@Anna walks into @Kitchen and smiles.", start=True,
                    subjects=[H.Subject("Anna", "person", "a tall woman with red hair.", (1, 2)),
                              H.Subject("Kitchen", "place", "", (3,))],
                    frame_pictures={"start": 4})
    assert out == """subject_definitions:
<Subject 1> is Anna (a tall woman with red hair), the person in <Picture 1> and <Picture 2>.
<Subject 2> is Kitchen, the environment in <Picture 3>.
<Picture 4> is the first frame of [Shot 1].

summary:
[keyframe completion + reference generation] The target video shows <Subject 1> and <Subject 2>. It begins from <Picture 4>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - Anna's face and identity are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the layout and look of Kitchen are retained.
<Picture 4> ([Shot 1] first frame): fully_preserved - the target video begins exactly from this picture.

detailed_description:
[Shot 1] The shot begins from <Picture 4>. <Subject 1>, a tall woman with red hair, walks into <Subject 2> and smiles.

overall_soundscape:
Natural ambient sound that fits the scene.

non_diegetic_music:
N/A"""
    assert [s.split(":\n")[0] for s in out.split("\n\n")] == list(H.SECTIONS)


def test_reference_kinds_object_and_style():
    out = H.compose("@Cup stands on a table.", subjects=[
        {"name": "Cup", "kind": "object", "desc": "a blue mug", "pictures": [1]},
        {"name": "Noir", "kind": "style", "pictures": [2, 3, 4]}])
    assert "<Subject 1> is Cup (a blue mug), the object in <Picture 1>." in out
    assert "<Subject 2> is Noir, the visual style of <Picture 2>, <Picture 3> and <Picture 4>." in out
    assert "<Subject 1> (appears in [Shot 1]): fully_preserved - the shape and look of Cup are retained." in out
    assert "<Subject 2> (appears in [Shot 1]): fully_preserved - the visual style of Noir is retained." in out
    assert "[reference generation] The target video shows <Subject 1>. It is in the visual style of <Subject 2>." in out
    assert "[Shot 1] <Subject 1>, a blue mug, stands on a table." in out
    assert "<Subject 1> is Kim, the person in <Picture 1>." in H.compose("x", subjects=[{"name": "Kim", "kind": "alien",
                                                                                         "pictures": [1]}])


def test_reference_numbered_start_and_end_with_later_shots():
    body = "@Anna sits. [Shot 2] At 00:03.000, the camera cuts to @Anna standing."
    out = H.compose(body, start=True, end=True, seconds=5, subjects=[H.Subject("Anna", pictures=(1,))],
                    frame_pictures={"start": 2, "end": 3})
    assert "<Picture 2> is the first frame of [Shot 1].\n<Picture 3> is the last frame of [Shot 2]." in out
    assert "<Picture 3> ([Shot 2] last frame): fully_preserved - the target video ends exactly on this picture." in out
    assert "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved" in out
    assert "It begins from <Picture 2>. It ends on <Picture 3>." in out
    assert ("detailed_description:\n[Shot 1] The shot begins from <Picture 2>. <Subject 1> sits.\n"
            "[Shot 2] At 00:03.000, the camera cuts to <Subject 1> standing. The shot ends on <Picture 3>.\n") in out
    assert "How the reference" not in out and "integrated_multimodal_description" not in out


def test_reference_frames_without_number_are_words_only():
    out = H.compose("@Anna waves.", start=True, end=True, seconds=4, subjects=[H.Subject("Anna", pictures=(1,))])
    assert "[Shot 1] The video begins exactly from the given start frame. <Subject 1> waves. The video ends exactly " \
           "on the given end frame." in out
    assert "keyframe completion" not in out and "first frame" not in out and "It begins" not in out
    half = H.compose("@Anna waves.", start=True, end=True, seconds=4, subjects=[H.Subject("Anna", pictures=(1,))],
                     frame_pictures={"start": 2})                  # the end frame's number did not fit
    assert "[keyframe completion + reference generation]" in half
    assert "The shot begins from <Picture 2>. <Subject 1> waves. The video ends exactly on the given end frame." in half


def test_reference_implicit_lead_and_style_line():
    text = 'Cinematic, warm light. [Shot 1] She looks up and says "Good morning."'
    ls = lines_of(text, ["Anna"], default="Anna")
    out = H.compose(text, subjects=[H.Subject("Anna", desc="a woman", pictures=(1,), voice=True)], lines=ls,
                    lead_implicit=True, start=True, frame_pictures={"start": 2})
    assert ("detailed_description:\nCinematic, warm light.\nThe person in the video is <Subject 1>.\n[Shot 1] The shot "
            "begins from <Picture 2>. She looks up and says, using the voice timbre referenced from <Audio 1>, "
            "<d>[English] Good morning.</d>\n") in out
    assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1), the voice of one speaker." in out
    assert ("[keyframe completion + reference generation + audio reference] The target video shows <Subject 1>. "
            "<Subject 1> (S1) speaks with the voice timbre of <Audio 1>. It begins from <Picture 2>.") in out
    assert "<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 1> without" in out
    keep = out.split("retention_analysis:\n")[1].split("\n\n")[0]
    assert "(S1)" not in keep                                      # (the guide: no speaker ids there)


def test_reference_voices_numbered_after_named_sounds():
    text = '@Anna hears <Audio 1> and <Audio 2>. "Hi," says @Ben. @Anna says "Hello."'
    ls = lines_of(text, ["Anna", "Ben"])
    out = H.compose(text, sounds=2, lines=ls, subjects=[H.Subject("Anna", pictures=(1,), voice=True),
                                                        H.Subject("Ben", pictures=(2,), voice=True)])
    assert ("<Audio 1> is a sound reference, used as the description says.\n<Audio 2> is a sound reference, used as "
            "the description says.\n<Audio 3> is the voice-timbre reference for <Subject 2> (S1), the voice of one "
            "speaker.\n<Audio 4> is the voice-timbre reference for <Subject 1> (S2), the voice of one speaker.") in out
    assert ('[Shot 1] <Subject 1> (S2) hears <Audio 1> and <Audio 2>. <Subject 2> (S1) says, using the voice timbre '
            'referenced from <Audio 3>, <d>[English] Hi</d>. <Subject 1> (S2) says, using the voice timbre referenced '
            'from <Audio 4>, <d>[English] Hello.</d>') in out
    assert "[reference generation + audio reference]" in out
    assert "<Audio 1>: reference - it is used as the description says" in out


def test_reference_soundtrack_is_audio_reuse():
    out = H.compose("@Anna dances.", subjects=[H.Subject("Anna", pictures=(1,))], soundtrack=1,
                    extra=["<Audio 1> is the complete audio track of the target video, reused 1:1."])
    assert "[reference generation + audio reuse]" in out
    assert "<Audio 1>: fully_copy - <Audio 1> is reused 1:1" in out
    assert "[Shot 1] <Subject 1> dances. <Audio 1> is the complete audio track" in out


def test_reference_task_types_follow_the_guide_order():
    assert H.task_types(keyframe=True, subjects=True, reuse=True, audio=True) == \
        "[keyframe completion + reference generation + audio reuse + audio reference]"
    assert H.task_types(audio=True, subjects=True) == "[reference generation + audio reference]"
    assert H.task_types() == "[reference generation]"
    sound_only = H.compose("Rain like <Audio 1>.", sounds=1, start=True, frame_pictures={"start": 1})
    assert "[keyframe completion + audio reference] The target video follows the detailed description." in sound_only


def test_reference_triggers_extra_and_sound_lines():
    text = "@Anna sings a song.\nnon_diegetic_music: Soft guitar.\noverall_soundscape: Birds."
    out = H.compose(text, subjects=[H.Subject("Anna", pictures=(1,))], triggers=["ohwx"], start=True,
                    frame_pictures={"start": 2}, extra=["The camera pushes in."])
    assert "[Shot 1] The shot begins from <Picture 2>. ohwx. <Subject 1> sings a song. The camera pushes in.\n" in out
    assert out.endswith("overall_soundscape:\nBirds.\n\nnon_diegetic_music:\nSoft guitar.")
    assert H.compose("@Anna sings a song.", subjects=[H.Subject("Anna", pictures=(1,))]).endswith(
        "non_diegetic_music:\nAs described in the shots.")


def test_reference_description_at_first_mention_only():
    subs = [H.Subject("Anna", desc="a tall woman", pictures=(1,))]
    assert "[Shot 1] <Subject 1>'s bag falls. <Subject 1>, a tall woman, picks it up. <Subject 1> leaves." in \
        H.compose("@Anna's bag falls. @Anna picks it up. @anna leaves.", subjects=subs)
    assert "[Shot 1] He waves at <Subject 1>, a tall woman." in H.compose("He waves at @Anna.", subjects=subs)


def test_reference_non_subject_speaker_keeps_a_speaker_id():
    text = '@Anna smiles. "Hi," says @Ben.'
    out = H.compose(text, lines=lines_of(text, ["Anna", "Ben"]), subjects=[H.Subject("Anna", pictures=(1,))],
                    resolve=lambda w: {"anna": "Anna", "ben": "Ben"}.get(w.lower()))
    assert "[Shot 1] <Subject 1> smiles. Ben (S1) says, <d>[English] Hi</d>." in out


def test_lines_as_dicts_with_cast_speakers_and_after_the_cut_ignored():
    text = '@Anna says "Hi."\noverall_soundscape: A "quoted" hum.'
    ls = T.dialogue({"cast": [{"name": "Anna", "images": [], "desc": ""}]}, text)
    assert isinstance(ls[0]["speaker"], dict) and len(ls) == 2
    ls[1]["speaker"] = {"name": "Ben"}                             # (would be S2, but it is cut off)
    out = H.compose(text, lines=ls, subjects=[{"name": "Anna", "pictures": [1]}],
                    resolve=lambda w: {"anna": "Anna", "ben": "Ben"}.get(w.lower()))
    assert "(S2)" not in out
    assert '[Shot 1] <Subject 1> (S1) says, <d>[English] Hi.</d>\n' in out and 'overall_soundscape:\nA "quoted" hum.' in out


# ----------------------------------------------------------------------------- the same joining as _voiced_prompt

@pytest.mark.parametrize("text", [
    '@Anna looks at the door and says "Hi there."',
    '"Hi," says @Anna.',
    '@Anna waves. "Hello!" @Ben nods. @Ben says: "Fine," and leaves.',
    '@Anna and @Ben sit. She whispers "Quiet." Then "Why?" asks @Ben.',
    '@Ben says, "One." @Anna replies "Two." "Three," adds @Ben.',
])
def test_body_matches_voiced_prompt(text):
    cast = [{"name": "Anna", "images": ["refs/a.png"], "desc": "", "voice": {"file": "refs/va.wav"}},
            {"name": "Ben", "images": ["refs/b.png"], "desc": "", "voice": {"file": "refs/vb.wav"}}]
    options = {"cast": cast, "identity": {"hint": False, "language": "English"}}
    found, _ = T.mentions(options, text)
    ls = T.dialogue(options, text)
    old, _pics, _voices = T._voiced_prompt(options, text, found, ls, 0, False)
    subs, k = [], 1
    for c in found:
        subs.append(H.Subject(c["name"], pictures=tuple(range(k, k + len(c["images"]))), voice=True))
        k += len(c["images"])
    new = H.compose(text, subjects=subs, lines=ls)
    shot = lambda s: s.split("detailed_description:\n")[1].split("\n\n")[0]   # noqa: E731
    assert shot(new) == shot(old)


def test_patterns_match_tree():
    assert H.MENTION.pattern == T.MENTION.pattern and H.SAID_BY.pattern == T.SAID_BY.pattern
    assert H.SAID_BY.flags == T.SAID_BY.flags


def test_module_is_pure():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(H))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods <= {"__future__", "re", "dataclasses", "typing"}
