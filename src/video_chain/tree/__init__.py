"""Branching chains: a tree of clips. Every clip (node) starts from the last frame of its parent; clips directly
under the start image start from that image. A path from the start to a node is one video.

A node is a generated clip ("ai") or a piece of a real video ("video": a file and a time range, converted to the
chain's frame rate and size). A generated clip right before a video ends at the video's first frame (H3's
last-frame input), one right after it starts from the video's last frame. So real and generated clips can be
chained in any order. A generated clip can lead into at most one video (its end can only match one).
With a motion guide (override "guide": 5, 22 or 39 frames) a generated clip gets the last frames of the clip
before and the first frames of a following video (with their sound) instead of a single frame at each end.
In the h3 prompt format a generated clip after another clip (or a real video) can also continue it (override
"continue": n frames): the last n frames of the clip before go to the reference model as <Video 1> and the prompt
says the clip continues it ("video continuation"); the pinned frames stay as they are.
A generated clip with no video after it can have an end image of its own (Node.end_image, a copy in ends/): the
frame H3 steers the clip's end to. It is not a frame of the chain, so nothing is cut when the path is joined.
Middle keyframes (Node.keyframes: [{"image": ends/..., "at": seconds, "note": text}]) pin pictures at moments inside
a generated clip (MiniMaxH3AddGuide at that frame); a keyframe the action cannot reach makes H3 cut to it.
A made generated clip without a video after it can hand over at an earlier frame (Node.end_at, a frame index): the
clips after it start from that frame, and the frames after it are left out when a path is joined. The clip itself
stays as made (nothing is generated again for it); only what follows it changes.
The project's style text (options "style": prefix / suffix) is put around every prompt sent to the model.
The cast (options "cast"): characters with face pictures in refs/. A prompt mentions them as @Name; the clip is made
with exactly the pictures of the characters it mentions (renumbered <Picture 1> ... for the model, see clip_cast).
A clip that mentions nobody gets the lead's pictures; one that writes <Picture N> itself gets every picture.

Project folder:
  tree.json            nodes (id, parent, prompt, seed, what was made), options, the selected node
  start.png            the start image (copied in)
  nodes/<id>_<tag>.mp4 a node's clip (a new name for every version)
  nodes/<id>_<tag>.png its last frame, the start image of its children
  nodes/<id>_<tag>_first.png   a video node's first frame (the end frame of the generated clip before it)
  nodes/<id>_<tag>_f<k>.png    a clip's hand-over frame k (Node.end_at), the start image of its children
  nodes/<id>_<tag>.own.m4a     the sound the model made for a clip whose given sound is laid over (remix_sound)
  paths/<id>.mp4       a joined path (start -> node), made on demand
  ends/<id>_<tag>.png  a generated clip's own end image
  ends/<id>_k<tag>.png a middle keyframe of a generated clip (Node.keyframes)
  starts/<id>_<tag>.png a first clip's own start picture (Node.start_image), with its prepared copies

A first clip normally starts from the project's start image. It can have a start picture of its own
(Node.start_image); add_start_rows makes the same clip once per picture: a row of the node view each, with the same
prompt, seed and settings (a set, Node.row), which can then go its own way. The video size stays the project's
(it follows start.png): every start picture is cut from its centre to that shape, as start.png is.

Every generated clip has its own LoRA stack (Node.overrides["loras"]: [{"path", "strength"}], applied in order;
none by default) and can override the project's clip length and steps
(Node.overrides); the video size and sound stay per project, so the clips of a path fit together.

A node counts as done when its clip exists and was made from exactly its current prompt, seed, settings and start
frame (the parent's current last frame). If any of that changed, the node and everything below it is "changed"
and is made again when a path through it is generated. Other branches are not touched.
"""
from __future__ import annotations

import logging


from .. import chain, media
from .. import camera as _camera
from .. import h3_prompt as _h3

from .consts import (  # noqa: F401
    TREE_FILE,
    SPEED_MODES,
    DISTILLED_STEPS,
    checkpoint_of,
    speed_mode,
    set_speed_mode,
    TREE_VERSION,
    NewerTreeError,
    START_FILE,
    NO_LORA,
    OVERRIDE_KEYS,
    _OLD_LORA_KEYS,
    AI,
    VIDEO,
    GUIDE_CHOICES,
    CONTINUE_CHOICES,
    DEFAULT_FRAMES,
    DEFAULT_OPTIONS,
    PROMPT_FORMATS,
    CAST_KINDS,
    REFS_DIR,
    ENDS_DIR,
    STARTS_DIR,
    MAX_CAST_PICTURES,
    MAX_PICTURES_PER_CHARACTER,
    MAX_AUDIO_REFS,
    AUDIO_REF_MIN,
    AUDIO_REF_TOTAL,
    AUDIO_TAG,
    MENTION,
    NAME_OK,
    PICTURE_TAG,
    DIALOGUE,
    SAID_BY,
    SPEECH_TAG,
    VOICE_MAX_SECS,
    VOICE_REF_MAX_SECS,
    WRONG_TAG,
    CFG_MAX,
    SEED_LIMIT,
    clean_sampling,
    SEED_MAX,
    REFINED,
    _CLIP_NAME,
    CLIP_KEYS)
from .guard import inside, retire  # noqa: F401
from .schema import (  # noqa: F401
    AudioRef, CastEntry, Eleven, Keyframe, MadeRecord, Options, Overrides, Source, Track, Voice, VoiceRef)
from .model import (  # noqa: F401
    Node,
    short_id,
    Tree)
from .timeline import (  # noqa: F401
    clip_length,
    made_frames,
    handover_index,
    handover_file,
    full_prompt,
    _guide_of,
    path_cuts,
    path_spans,
    SWAPPED_REFINE_PROMPT)
from .files import (  # noqa: F401
    handover_frames,
    _unlink,
    clip_files,
    new_clip_files,
    remove_stale_files,
    node_files,
    load,
    _merged_options,
    _int_or,
    _parses,
    _sync_dir,
    _write_lock,
    save,
    _old_record,
    _extras,
    migrate_loras,
    node_loras,
    refined_file)
from .cast import (  # noqa: F401
    _migrate_cast,
    cast_of,
    character,
    lead_of,
    kind_of,
    refiner_cast,
    is_h3,
    set_prompt_format,
    all_pictures,
    check_name,
    _set_cast,
    source_of,
    _write_picture,
    recrop_picture,
    _copy_picture,
    _check_kind,
    add_character,
    next_character_name,
    add_picture,
    add_reference,
    _drop_file,
    remove_picture,
    remove_character,
    voices_in_use,
    clone_file,
    clone_secs,
    _drop_voice,
    set_voice,
    remove_voice,
    update_character,
    set_kind,
    rename_character,
    _in_lines,
    _unmention,
    mentions,
    audio_refs,
    add_audio_reference,
    remove_audio_reference,
    audio_mentioned)
from .prompts import (  # noqa: F401
    _FEMALE,
    _MALE,
    _PRONOUN,
    without_tags,
    without_speech_tags,
    gender_of,
    dialogue,
    replace_lines,
    spoken_lines,
    refine_problems,
    _picture_tags,
    _voiced_prompt,
    PICTURES_NOTE,
    _kept,
    _pictures_note)
from .clipcast import (  # noqa: F401
    _KIND_WORD,
    _tie,
    sound_given,
    sound_mixed,
    mix_level,
    lead_on,
    set_lead_on,
    node_cast,
    clip_cast,
    _cast_record,
    _language,
    _h3_cast,
    _resolver,
    cast_problems,
    cast_issues,
    CastIssue,
    CastKind,
    _closest)
from .sound import (  # noqa: F401
    audio_problems)
from .keyframes import (  # noqa: F401
    _in_ends,
    set_end_image,
    clear_end_image,
    copy_end_image,
    clean_keyframes,
    keyframe_files,
    _drop_keyframe_file,
    set_keyframe,
    remove_keyframe,
    copy_keyframes,
    keyframe_frame,
    missing_keyframes,
    end_image_used,
    missing_end_images)
from .freezing import (  # noqa: F401
    changed_inputs,
    revert,
    _revert_fields,
    freeze_set,
    freeze_edits,
    unfreeze,
    heal_frozen,
    frozen_note)
from .starts import (  # noqa: F401
    _in_starts,
    _drop_start_file,
    start_source,
    set_node_start,
    clear_node_start,
    copy_node_start,
    row_members,
    add_start_rows,
    apply_to_row,
    prepared_start,
    clear_start_image,
    set_start_image)
from .sources import (  # noqa: F401
    video_source,
    source_rotation,
    _SRC_INFO,
    _IMG_SIZE,
    source_info,
    ASPECTS,
    size_origin,
    size_source, video_size)
from .audio_tracks import (  # noqa: F401
    REPEAT_MIN,
    TRACK_PREFIX,
    TRACK_SOUND_TEXT,
    REPEAT_SOUND_TEXT,
    SOURCE_SOUND_TEXT,
    uses_ref_model,
    source_video,
    source_sound_span,
    _continued_sound,
    joined_sound_start,
    _join_lead,
    tracks,
    track_of,
    track_id,
    _cut_track,
    add_track,
    recut_track,
    remove_track,
    unused_track_files,
    track_users,
    track_sound_span,
    previous_sound_span,
    sound_file,
    source_sound_problems,
    _repeat_sound_problems,
    _sound_room,
    lora_trigger_words,
    _stack_triggers,
    _soundtrack_sentence)
from .inputs import (  # noqa: F401
    model_prompt,
    continue_frames,
    frames_before,
    continue_choices,
    frame_numbers,
    _h3_model_prompt,
    plain_frames,
    node_settings,
    veda_for,
    loras_needed,
    empty_prompts,
    plan)
from .checker import (  # noqa: F401
    freeze_problem,
    freeze,
    keyframe_problem,
    video_key,
    _same_key,
    Checker,
    current_settings)
from .engine import (  # noqa: F401
    clip_inputs,
    make_video,
    copy_video,
    ensure_handover,
    ensure_handovers,
    _clip_frames,
    _guides,
    _continuation,
    guide_problem,
    given_sound,
    _SOUND_OVERRIDES,
    remix_sound,
    remix_sounds,
    sweep_scratch,
    run_tree)
from .export import (  # noqa: F401
    swapped_video,
    refine_record,
    _refined_sound,
    refine_path,
    join_path)

logger = logging.getLogger(__name__)
