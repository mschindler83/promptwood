"""Qwen3-TTS worker: runs in its own environment (.venv-tts, started by tts.LocalTTS) and answers one JSON request per
line on stdin with one line "@@TTS {json}" on stdout. Everything else the libraries print goes to stderr.

Requests ({"cmd": ...}):
  design     {"model", "text", "language", "instruct", "out", "seed"}   a new voice from a description (VoiceDesign)
  clone      {"model", "text", "language", "ref_audio", "ref_text", "out", "seed"}   a line in a voice sample's voice
  transcribe {"model", "audio", "language"}                             what a voice sample says (Whisper)
  isolate    {"dir", "model", "audio", "out"}                           a recording's voice without its music
  quit
Answers: {"ok": true, "secs": ..., "sr": ...} / {"ok": true, "text": ...} / {"ok": false, "error": "..."}.
One model is kept loaded at a time (the next kind unloads it), so the GPU holds one 1.7B model at most."""
import json
import os
import sys
import traceback

_answer = os.fdopen(os.dup(1), "w", buffering=1)     # the answers; fd 1 now goes to stderr (library prints)
os.dup2(2, 1)
sys.stdout = sys.stderr

import torch  # noqa: E402

_loaded = {"path": None, "model": None}
_prompts: dict = {}                                   # voice clone prompts by (file, size, mtime, text)


def _free():
    _loaded["path"], _loaded["model"] = None, None
    _prompts.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model(path: str, kind: str):
    if _loaded["path"] != path:
        _free()
        if kind == "whisper":
            from transformers import pipeline
            m = pipeline("automatic-speech-recognition", model=path, torch_dtype=torch.float16, device="cuda:0")
        else:
            from qwen_tts import Qwen3TTSModel
            m = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16)
        _loaded["path"], _loaded["model"] = path, m
    return _loaded["model"]


def _seed(s):
    if s is not None:
        torch.manual_seed(int(s))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(s))


def _write(out: str, wav, sr: int) -> dict:
    import soundfile as sf
    tmp = out + ".part.wav"
    sf.write(tmp, wav, sr)
    os.replace(tmp, out)
    return {"ok": True, "secs": round(len(wav) / sr, 3), "sr": sr}


def handle(r: dict) -> dict:
    cmd = r.get("cmd")
    if cmd == "design":
        m = _model(r["model"], "tts")
        _seed(r.get("seed"))
        wavs, sr = m.generate_voice_design(text=r["text"], language=r.get("language") or "Auto",
                                           instruct=r.get("instruct") or "")
        return _write(r["out"], wavs[0], sr)
    if cmd == "clone":
        m = _model(r["model"], "tts")
        st = os.stat(r["ref_audio"])
        key = (r["ref_audio"], st.st_size, st.st_mtime_ns, r.get("ref_text") or "")
        if key not in _prompts:
            text = r.get("ref_text") or None
            _prompts[key] = m.create_voice_clone_prompt(ref_audio=r["ref_audio"], ref_text=text,
                                                        x_vector_only_mode=text is None)
        _seed(r.get("seed"))
        wavs, sr = m.generate_voice_clone(text=r["text"], language=r.get("language") or "Auto",
                                          voice_clone_prompt=_prompts[key])
        return _write(r["out"], wavs[0], sr)
    if cmd == "transcribe":
        import subprocess
        import tempfile

        import soundfile as sf
        p = _model(r["model"], "whisper")
        with tempfile.TemporaryDirectory() as tmp:      # Whisper's own rate, mono (48 kHz stereo is misheard)
            wav16 = os.path.join(tmp, "a.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", r["audio"], "-ar", "16000", "-ac", "1",
                            wav16], check=True, capture_output=True)
            audio, sr = sf.read(wav16, dtype="float32")
        kw = {"generate_kwargs": {"language": r["language"].lower()}} if r.get("language") else {}
        if len(audio) > 30 * sr:                      # (more than Whisper's 30 s window: its long-form mode)
            kw["return_timestamps"] = True
        text = p({"raw": audio, "sampling_rate": sr}, **kw)["text"].strip()
        _free()                                       # (only needed once per voice sample)
        return {"ok": True, "text": text}
    if cmd == "isolate":
        import gc
        import logging
        import shutil
        import tempfile

        import soundfile as sf
        stem = os.path.splitext(r["model"])[0]
        for f in (r["model"], stem + ".yaml"):        # (never downloaded from here: the app says how to get it)
            if not os.path.isfile(os.path.join(r["dir"], f)):
                return {"ok": False, "error": f"the model file {f} is not in {r['dir']}"}
        _free()
        from audio_separator.separator import Separator
        with tempfile.TemporaryDirectory() as tmp:
            sep = Separator(model_file_dir=r["dir"], output_dir=tmp, output_format="WAV",
                            output_single_stem="Vocals", log_level=logging.WARNING)
            sep.load_model(r["model"])
            made = sep.separate(r["audio"], {"Vocals": "vocals"})
            if not made:
                return {"ok": False, "error": "the separation made no file"}
            src = made[0] if os.path.isabs(made[0]) else os.path.join(tmp, made[0])
            info = sf.info(src)
            part = r["out"] + ".part.wav"
            shutil.move(src, part)
            os.replace(part, r["out"])
        del sep
        gc.collect()
        _free()                                       # (only needed once per voice)
        return {"ok": True, "secs": round(info.frames / info.samplerate, 3), "sr": info.samplerate}
    return {"ok": False, "error": f"unknown request {cmd!r}"}


def main():
    _answer.write("@@TTS " + json.dumps({"ok": True, "ready": True}) + "\n")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            _answer.write("@@TTS " + json.dumps({"ok": False, "error": "not JSON"}) + "\n")
            continue
        if r.get("cmd") == "quit":
            break
        try:
            a = handle(r)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            a = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
        _answer.write("@@TTS " + json.dumps(a) + "\n")
    _free()


if __name__ == "__main__":
    main()
