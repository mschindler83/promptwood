"""The prompt refiner: a language model in ComfyUI (TextGenerate), its instructions and ComfyPromptRefiner."""
from __future__ import annotations

import re
from pathlib import Path

from .client import ComfyGenerator
from .errors import ComfyError
from .wfb import TEXT_GEN_ID, TEXT_SHOW_ID, combo_options, loader_clip, text_generate_inputs

# ----------------------------------------------------------------------------- prompt refiner (an LLM in ComfyUI)

# Language models ComfyUI can run with its TextGenerate node, loaded as text encoders: (file name contains, loader
# type, label, uncensored). Most preferred first.
# Measured on an RTX 4090 (ComfyUI 0.44): Qwen 2.5-VL 7B cannot generate in ComfyUI (no stop tokens) and the nvfp4
# Qwen3-VL 32B gives garbage on a 4090 (no fp4 hardware), so they are not offered. One file per model.
REFINER_MODELS = (
    ("gemma-3-12b-abliterated", "ltxv", "Gemma 3 12B abliterated (uncensored)", True),
    ("qwen3vl_4b_fp8", "krea2", "Qwen3-VL 4B", False),
    ("qwen-3-vl-4b", "krea2", "Qwen3-VL 4B", False),
    ("qwen3vl_4b", "krea2", "Qwen3-VL 4B", False),
    ("gemma_3_12b_it_fp8_scaled", "ltxv", "Gemma 3 12B", False),
    ("gemma_3_12b_it_fp8", "ltxv", "Gemma 3 12B", False),
)
_SAFETY = ("Every person you describe is an adult. Never write sexual or suggestive content involving minors; if the "
           "input asks for that, return the input unchanged.")
REFINE_INSTRUCTIONS = {
    "video": ("You improve prompts for an image-to-video model (MiniMax H3) that continues a video from a given start "
              "frame and also makes the sound. Rewrite the user's prompt into one clear paragraph in present tense "
              "that describes what happens: the action and motion of each subject, the camera movement and framing, "
              "and the sound (ambient sounds, music, and any spoken words in quotes). Do not describe at length how "
              "things look at the start, the start frame already shows that. Keep every subject, action and detail "
              "the user gave. Do not add new people, animals, objects or events, and do not change what happens; "
              "only make it clearer and more concrete. "
              "At most 120 words. Answer with the prompt only, no preface, no quotes around it, no markdown."),
    # the official MiniMax H3 prompt format (base guide, T2VA / I2VA): the body of
    # integrated_multimodal_description and the two sound fields; the app adds the header, the field label, the
    # <Subject> / <Picture> tags and the speaker ids (h3_prompt), so the refiner writes none of them
    "video_h3": ("You write prompts for MiniMax H3, a model that makes a video together with its sound. Rewrite the "
                 "user's prompt into a description of the video along its timeline, in English and in present "
                 "tense, where every detail is something visible or audible. Begin with the overall style (e.g. "
                 "Live-action, cinematic; 2D-animated; 3D CG; claymation; vintage film) and the initial composition: "
                 "the framing, the subjects, where they are and the place. Then describe what happens in order: how "
                 "the action begins, how it develops, and its result or the reaction to it. Write camera movement as "
                 "a natural sentence with its motion type and, where it matters, its amplitude and speed, e.g. \"The "
                 "camera pushes in with small amplitude at slow speed toward her hands.\" or \"The camera holds a "
                 "static shot.\" Keep every subject, action, detail and spoken line the user gave. Do not add new "
                 "people, animals, objects, events or spoken lines, and do not change what happens; only make it "
                 "clearer, more concrete and more vivid. Spoken lines stay word for word in their language, in "
                 "double quotes. Usually the video is one shot. Only if the user asks for a cut or the action clearly "
                 "needs another view, begin with [Shot 1] and start each later shot on a new line as \"[Shot 2] At "
                 "00:03.500, the camera cuts to ...\" with strictly increasing cut times inside the clip's length; "
                 "a cut shows something new, a slight change of distance or angle is a camera movement instead. "
                 "Write in detail: up to about 350-500 words for a long clip, fewer for a short one, and never more "
                 "action than fits into the clip's length. After the description write exactly two more lines: "
                 "\"overall_soundscape: \" with 1-4 sentences on the ambient sound, the sounds of the actions and "
                 "non-verbal human sounds such as breathing or laughter (not the spoken lines), and "
                 "\"non_diegetic_music: \" with 1-3 sentences on background music only the audience hears "
                 "(instruments, tempo, rhythm, changes in volume), or N/A when there is none. Do not write a header "
                 "line, any other field names, tags in angle brackets or speaker ids like (S1); the app adds them. "
                 "Answer with the prompt only, no preface, no quotes around it, no markdown."),
    "image": ("You improve prompts for text-to-image models. Expand the user's prompt into one cohesive paragraph: the "
              "subject(s) with their attributes and actions, the setting, lighting, composition and framing, the "
              "medium and style. Keep every subject, colour, action and spatial relation the user gave and the medium "
              "they asked for; do not add new people, animals or objects; add only visual details that serve their "
              "intent; if their prompt is already detailed, only polish it. If they want visible text, give the "
              "exact words in quotes. Answer with the prompt only, no preface, no markdown."),
    "edit": ("You improve instructions for an image editing model. Rewrite the user's request as one clear, direct "
             "edit instruction: what exactly to change and how, and what must stay the same (identity, pose, "
             "composition, background) unless they want it changed. Do not add changes they did not ask for. "
             "Answer with the instruction only, no preface, no markdown."),
}


def pick_refiner_models(clip_names: list[str]) -> list[dict]:
    """The language models ComfyUI lists that the refiner can use, most preferred (uncensored) first:
    [{"file", "type", "label", "uncensored"}]."""
    out, seen, labels = [], set(), set()
    for key, kind, text, free in REFINER_MODELS:
        for name in sorted(clip_names, key=len):
            if key in name.lower() and name not in seen and text not in labels:
                seen.add(name)
                labels.add(text)
                out.append({"file": name, "type": kind, "label": text, "uncensored": free})
    return out


SOUND_FIELDS = ("overall_soundscape", "non_diegetic_music")     # the two lines a "video_h3" refine ends with
REFINE_LENGTH = {"video_h3": 1024}       # tokens a refine may write (~500 words + the sound lines) ...
REFINE_LENGTH_DEFAULT = 512              # ... others this many


def clean_refined(text: str, lines: bool = False) -> str:
    """The model's answer without thinking blocks, prefaces, wrapping quotes or markdown, on one line.
    lines (the "video_h3" kind): the body on one line (a line break only before each later [Shot N]), then the
    sound lines the model wrote (overall_soundscape: / non_diegetic_music:), each on its own line; one it left out
    stays out (the prompt builder has defaults). What the app adds itself is taken out when the model wrote it
    anyway: a header line, the integrated_multimodal_description: label, <d> blocks (back to quotes), (S1) ids."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    text = re.sub(r"^\s*<think>.*", "", text, flags=re.S | re.I).strip()     # a thought cut off: no prompt in it
    text = re.sub(r"^(here(?:'s| is)[^:\n]*:|refined prompt:|prompt:)\s*", "", text, flags=re.I).strip()
    if lines:                                  # markdown anywhere: code fences, bold, headings
        text = re.sub(r"^\s*```\w*\s*$", "", text, flags=re.M)
        text = re.sub(r"^\s*#+\s*", "", text.replace("**", "").replace("__", ""), flags=re.M).strip()
    text = re.sub(r"^\*\*|\*\*$", "", text).strip()
    pairs = {'"': '"', "'": "'", "“": "”", "„": "“", "‘": "’", "«": "»"}
    inner = text[1:-1]
    if len(text) > 1 and pairs.get(text[0]) == text[-1] and text[0] not in inner and text[-1] not in inner:
        text = inner.strip()                       # only quotes around the whole answer, not spoken lines
    if not lines:
        return " ".join(text.split())
    text = "\n".join(x for x in text.splitlines()      # an official header line: the app writes its own
                     if not re.match(r"\s*(For the target video, at 0\.00 seconds|How the reference pictures align)",
                                     x, re.I))
    text = " ".join(text.split())
    text = re.sub(r"\bintegrated_multimodal_description\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"<d>\s*(?:\[[^\]]{1,30}\]\s*)?(.*?)\s*</d>", r'"\1"', text, flags=re.S | re.I)
    text = re.sub(r"\s*\(S\d+(?:\s*,\s*S\d+)*\)", "", text)
    parts = re.split(rf"\b({'|'.join(SOUND_FIELDS)})\s*:\s*", text, flags=re.I)
    body, sounds = parts[0].strip(), {}
    for name, value in zip(parts[1::2], parts[2::2]):
        value = value.strip()
        if value and name.lower() not in sounds:
            sounds[name.lower()] = value
    body = re.sub(r"\s*(\[Shot \d+\])", r"\n\1", body).strip()
    return "\n".join([body] + [f"{k}: {sounds[k]}" for k in SOUND_FIELDS if k in sounds]).strip()


_KIND_TEXT = {"place": "a place", "object": "an object", "style": "a visual style"}


def cast_instruction(cast: list[dict]) -> str:
    """What the refiner is told about the cast ([{"name", "desc", "voice": bool, "kind": person (also missing) |
    place | object | style}]): the names stay @Name, and a spoken line goes in quotes into a sentence that starts
    with its speaker's @Name (the app finds the speaker there and gives the line that character's voice). A place,
    object or style is said to be one, so the refiner does not make it a person."""
    if not cast:
        return ""

    def one(c: dict) -> str:
        desc = c.get("desc", "").strip().rstrip(".") if c.get("desc") else ""
        what = _KIND_TEXT.get(c.get("kind") or "person")
        if what:
            return f"@{c['name']} ({what}" + (f": {desc}" if desc else "") + ")"
        return (f"@{c['name']}" + (f" ({c['desc'].strip().rstrip('.')})" if c.get("desc") else "")
                + (", has a voice" if c.get("voice") else ""))
    people = "; ".join(one(c) for c in cast)
    things = [f"@{c['name']}" for c in cast if c.get("kind") in _KIND_TEXT]
    if things:
        people += (f". {', '.join(things)} {'is' if len(things) == 1 else 'are'} not a person: never give "
                   f"{'it' if len(things) == 1 else 'them'} a spoken line or a person's actions")
    return (f"The cast, written as @Name: {people}. The descriptions are for your information; do not repeat "
            "them in the prompt. Write a cast member always as @Name, exactly as written (not "
            "as a description or a plain name), at least once in each sentence about them. Mention a cast member only "
            "if the user's prompt or request does. Each spoken line: in double quotes, in its own sentence that "
            "starts with the speaker's @Name, never with he, she or they. Wrong: @A and @B sit down. She says "
            "\"Hi.\" He answers \"Hello.\" Right: @A and @B sit down. @A says \"Hi.\" @B answers \"Hello.\" "
            "Keep spoken lines in their language and word for word unless the user asks to change them, and do not add "
            "spoken lines. Do not write any tags in angle brackets.")


def clip_instruction(seconds: float | None, start: str | None) -> str:
    """What a "video_h3" refine is told about the clip: its length (cut times and the amount of action fit it) and
    its start frame. start: "seen" (the picture is attached), "unseen" (there is one, the model cannot see it) or
    None (the video starts from nothing given)."""
    out = []
    if seconds:
        s = float(seconds)
        out.append(f"The clip is {s:.2f} seconds long: describe only as much action as fits into {s:.2f} seconds, "
                   f"and every cut time is before {int(s // 60):02d}:{s % 60:06.3f}.")
    if start == "seen":
        out.append("The attached picture is the video's first frame (at 0.00 seconds). Anchor the first sentences "
                   "on what is visible in it: the style, the subjects with their look and clothing, their positions, "
                   "the place and the key objects; keep them consistent, then describe what happens next.")
    elif start == "unseen":
        out.append("The video starts from a given first frame that you cannot see. Do not invent how it looks (faces, "
                   "clothing, colours, the place); describe the style and the initial composition only as far as the "
                   "user's prompt gives them, and concentrate on what happens.")
    return " ".join(out)


def refine_messages(*, prompt: str, kind: str, hint: str = "", cast: list[dict] | None = None,
                    seconds: float | None = None, start: str | None = None, loras: str = "") -> tuple[str, str]:
    """The refiner's (system instruction, user message) for this kind of prompt (any backend).
    cast: the project's characters (video prompts), see cast_instruction. seconds, start: the clip ("video_h3"
    only), see clip_instruction. loras: what is known of the LoRAs the result is made with (lora_notes.guidance),
    put last."""
    system = REFINE_INSTRUCTIONS[kind] + " " + _SAFETY
    if cast and kind in ("video", "video_h3"):
        system += " " + cast_instruction(cast)
    if kind == "video_h3" and (seconds or start):
        system += " " + clip_instruction(seconds, start)
    tokens = re.findall(r"<[^<>]{1,40}>", prompt)          # <Picture 1>, <ref_image_0> …: only then mentioned
    tokens += re.findall(r"(?<![\w@])@\w[\w-]*", prompt)    # @Anna: the cast, mentioned by name
    if tokens:
        system += f" Keep these tokens exactly as they are, where they are: {', '.join(dict.fromkeys(tokens))}."
    if loras.strip():
        system += "\n\n" + loras.strip()
    user = prompt.strip() + (f"\n\nWhat the user wants changed in the prompt: {hint.strip()}" if hint.strip() else "")
    return system, user


def build_refine_prompt_workflow(*, prompt: str, kind: str, model: dict, seed: int, hint: str = "",
                                 temperature: float = 0.5, cast: list[dict] | None = None,
                                 seconds: float | None = None, start: str | None = None,
                                 image: str | None = None, loras: str = "") -> dict:
    """CLIPLoader (the LLM) -> TextGenerate with the instruction for this kind of prompt -> PreviewAny (text out).
    image: an uploaded picture the model sees (a vision model: the clip's first frame, start "seen")."""
    system, user = refine_messages(prompt=prompt, kind=kind, hint=hint, cast=cast, seconds=seconds, start=start,
                                   loras=loras)
    wf = {
        "llm": loader_clip(model["file"], model["type"]),
        TEXT_GEN_ID: {"class_type": "TextGenerate",
                      "inputs": text_generate_inputs(clip=["llm", 0], prompt=user, system_prompt=system,
                                                     max_length=REFINE_LENGTH.get(kind, REFINE_LENGTH_DEFAULT),
                                                     temperature=temperature, seed=seed)},
        TEXT_SHOW_ID: {"class_type": "PreviewAny", "inputs": {"source": [TEXT_GEN_ID, 0]}},
    }
    if image:                                  # (as describe_face passes its picture)
        wf["pic"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        wf[TEXT_GEN_ID]["inputs"]["image"] = ["pic", 0]
    return wf


FACE_MAX_TOKENS = 160
FACE_TEMPERATURE = 0.2
# The head swap LoRA's face description (the creator's instruction, Henge backend/edit_modes.py FACE_INSTRUCTION)
FACE_INSTRUCTION = ('Describe this face in a short comma-separated attribute list, in exactly this style: "Male, oval face '
                    'shape, average-sized head with strong jawline, light brown skin, dark eyes, black tousled hair, silver '
                    'hoop earring." Cover, in order: gender, head/face shape and proportions (e.g. oval/round/square/'
                    'heart-shaped, narrow/wide, jaw structure, whether the head reads as small/average/large relative to '
                    'the shoulders), skin tone, eye color, hair color and style, and any distinctive features (facial '
                    'hair, jewelry, makeup, glasses, etc.). Never name or guess real people, celebrities or fictional '
                    'characters. Only output the description, nothing else.')


def build_describe_face_workflow(*, image: str, model: dict, seed: int = 0) -> dict:
    """A vision language model describes the face in an uploaded picture (TextGenerate with the image)."""
    return {
        "llm": loader_clip(model["file"], model["type"]),
        "pic": {"class_type": "LoadImage", "inputs": {"image": image}},
        TEXT_GEN_ID: {"class_type": "TextGenerate",
                      "inputs": text_generate_inputs(clip=["llm", 0], prompt=FACE_INSTRUCTION, system_prompt="",
                                                     max_length=FACE_MAX_TOKENS, temperature=FACE_TEMPERATURE,
                                                     seed=seed, image=["pic", 0])},
        TEXT_SHOW_ID: {"class_type": "PreviewAny", "inputs": {"source": [TEXT_GEN_ID, 0]}},
    }


class ComfyPromptRefiner(ComfyGenerator):
    """Refines a prompt with a language model in ComfyUI (TextGenerate)."""

    def _check_models(self):
        if not self._object_info("TextGenerate"):
            raise ComfyError("this ComfyUI has no TextGenerate node (update ComfyUI)")
        req = self._object_info("CLIPLoader").get("input", {}).get("required", {})
        self.models = pick_refiner_models(combo_options(req.get("clip_name")))
        if not self.models:
            raise ComfyError("ComfyUI lists no language model the prompt refiner can use (e.g. the Qwen3-VL 4B or a "
                             "Gemma 3 12B text encoder)")

    def vision_models(self) -> list[dict]:
        """The models that can look at a picture (Qwen3-VL)."""
        return [m for m in self.models if "vl" in m["file"].lower().replace("-", "")]

    def describe_face(self, picture: Path, seed: int = 0) -> str:
        """The face in a picture as the head swap LoRA's face details (a short attribute list)."""
        self._begin_job()
        models = self.vision_models()
        if not models:
            raise ComfyError("ComfyUI lists no vision model to describe the face (e.g. the Qwen3-VL 4B text "
                             "encoder); write the face details yourself")
        uploads = [(self.upload(Path(picture)), Path(picture))]
        try:
            wf = build_describe_face_workflow(image=uploads[0][0], model=models[0], seed=seed)
            text = self._ask_text(wf)
        finally:
            if self.cleanup:
                self._drop_uploads(uploads)
        text = " ".join(clean_refined(text).split()).strip().strip('"')
        if not text:
            raise ComfyError("the vision model gave no face description")
        return text[:600]

    def complete(self, system: str, user: str, max_tokens: int = 512, model_file: str | None = None,
                 seed: int = 0, temperature: float = 0.2) -> str:
        """The model's raw answer to one question (text only; e.g. a LoRA's note, lora_notes)."""
        self._begin_job()
        model = next((m for m in self.models if m["file"] == model_file), self.models[0])
        wf = build_refine_prompt_workflow(prompt=user, kind="image", model=model, seed=seed,
                                          temperature=temperature)
        wf[TEXT_GEN_ID]["inputs"].update({"system_prompt": system, "prompt": user, "max_length": int(max_tokens)})
        return self._ask_text(wf)

    def _ask_text(self, wf: dict) -> str:
        """Run a text workflow (a language model; its answer is shown by the "show" node) and return that text raw."""
        entry = self._run_workflow(wf)
        text = (((entry.get("outputs") or {}).get(TEXT_SHOW_ID) or {}).get("text") or [""])[0]
        return text if isinstance(text, str) else str(text)

    def refine_prompt(self, prompt: str, kind: str, model_file: str | None = None, seed: int = 0,
                      hint: str = "", cast: list[dict] | None = None, seconds: float | None = None,
                      start_image: Path | str | None = None, loras: str = "") -> str:
        """seconds, start_image (the "video_h3" kind): the clip's length and the frame it starts from; a vision
        model sees that frame, any other is told not to invent how it looks. loras: lora_notes.guidance."""
        self._begin_job()
        model = next((m for m in self.models if m["file"] == model_file), self.models[0])
        start, uploads = None, []
        if kind == "video_h3" and start_image:     # (a frame still to be made: there, but not to be seen yet)
            seen = Path(start_image).is_file() and model in self.vision_models()
            start = "seen" if seen else "unseen"
            if seen:
                uploads = [(self.upload(Path(start_image)), Path(start_image))]
        try:
            wf = build_refine_prompt_workflow(prompt=prompt, kind=kind, model=model, seed=seed, hint=hint, cast=cast,
                                              seconds=seconds if kind == "video_h3" else None, start=start,
                                              image=uploads[0][0] if uploads else None, loras=loras)
            text = self._ask_text(wf)
        finally:
            if uploads and self.cleanup:
                self._drop_uploads(uploads)
        text = clean_refined(text, lines=kind == "video_h3")
        if not text:
            raise ComfyError("the language model gave no answer")
        return text
