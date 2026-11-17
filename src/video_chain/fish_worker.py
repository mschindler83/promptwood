"""Fish S2 Pro worker: runs in fish-speech's own environment (paths.fish_dir()/.venv, started by tts.FishTTS) and answers
one JSON request per line on stdin with one line "@@TTS {json}" on stdout, as tts_worker.py does. Everything else the
libraries print goes to stderr.

Requests ({"cmd": ...}):
  clone  {"model", "text", "ref_audio", "ref_text", "out", "seed", "temperature"}   a line in a voice sample's voice;
         the text may carry Fish's [tags] (free-form: [angry], [whispering, scared], [laughs] ...)
  quit
Answers: {"ok": true, "secs": ..., "sr": ...} / {"ok": false, "error": "..."}.

The model (about 5B parameters, bf16) is loaded on the first request and kept until quit. fish-speech's own loader
builds the model in fp32 on the CPU and reads every weight into RAM before moving it to the GPU (about 30 GB of
system RAM at its peak, more than a desktop with other programs open has). Here the model is built in bf16 straight
on the GPU (fish-speech casts it to bf16 anyway, so nothing is lost), the weights are copied in one tensor at a
time read straight onto the GPU, and the audio codec is built on the GPU too: about 3 GB of RAM at the peak. The
context is 8192 tokens instead of 32768 (a line with its reference needs well under 1000), which also makes its
caches smaller."""
import json
import os
import sys
import traceback

_answer = os.fdopen(os.dup(1), "w", buffering=1)     # the answers; fd 1 now goes to stderr (library prints)
os.dup2(2, 1)
sys.stdout = sys.stderr
os.environ.setdefault("TQDM_DISABLE", "1")           # (no progress bars in the log)

import torch  # noqa: E402
from loguru import logger  # noqa: E402

logger.remove()                                       # fish-speech logs every token it makes: warnings only
logger.add(sys.stderr, level="WARNING", colorize=False)

CONTEXT = 8192
_state: dict = {"model": None, "engine": None}


def _patch_loaders():
    """fish-speech's loaders, made to use the GPU instead of system RAM (see the module docstring)."""
    from collections import OrderedDict
    from pathlib import Path

    import safetensors
    from fish_speech.models.text2semantic import llama
    from tools.server.model_manager import ModelManager

    original = llama.BaseTransformer.from_pretrained

    def from_pretrained(path, load_weights=False, max_length=None, rope_base=None):
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                model = original(path, False, max_length or CONTEXT, rope_base)
        finally:
            torch.set_default_dtype(old)
        if not load_weights:
            return model
        index = json.load(open(Path(path) / "model.safetensors.index.json"))
        params = dict(model.named_parameters())
        loaded = set()
        for shard in sorted(set(index["weight_map"].values())):
            with safetensors.safe_open(str(Path(path) / shard), framework="pt", device="cuda") as f:
                for key in f.keys():
                    name = next(iter(llama._remap_fish_qwen3_omni_keys(OrderedDict([(key, None)]))))
                    if name not in params:
                        raise RuntimeError(f"the checkpoint has a weight the model does not: {key}")
                    with torch.no_grad():
                        params[name].copy_(f.get_tensor(key))
                    loaded.add(name)
            torch.cuda.empty_cache()
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(f"the checkpoint lacks {len(missing)} weights of the model ({', '.join(missing[:3])})")
        return model

    llama.DualARTransformer.from_pretrained = staticmethod(from_pretrained)
    original_decoder = ModelManager.load_decoder_model

    def load_decoder_model(self, config_name, checkpoint_path, device):
        with torch.device("cuda"):
            return original_decoder(self, config_name, checkpoint_path, device)

    ModelManager.load_decoder_model = load_decoder_model


def _engine(path: str):
    if _state["model"] != path:
        if _state["model"] is not None:
            raise RuntimeError("another Fish model is loaded; stop the worker first")
        if not torch.cuda.is_available():
            raise RuntimeError("Fish S2 Pro needs an NVIDIA GPU (CUDA is not available)")
        _patch_loaders()
        from tools.server.model_manager import ModelManager
        mm = ModelManager(mode="tts", device="cuda", half=False, compile=False, llama_checkpoint_path=path,
                          decoder_checkpoint_path=os.path.join(path, "codec.pth"),
                          decoder_config_name="modded_dac_vq")
        _state["model"], _state["engine"] = path, mm.tts_inference_engine
    return _state["engine"]


def _write(out: str, wav, sr: int) -> dict:
    import soundfile as sf
    tmp = out + ".part.wav"
    sf.write(tmp, wav, sr)
    os.replace(tmp, out)
    return {"ok": True, "secs": round(len(wav) / sr, 3), "sr": sr}


def handle(r: dict) -> dict:
    cmd = r.get("cmd")
    if cmd == "clone":
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
        eng = _engine(r["model"])
        with open(r["ref_audio"], "rb") as f:
            ref = ServeReferenceAudio(audio=f.read(), text=r.get("ref_text") or "")
        kw = {"temperature": float(r["temperature"])} if r.get("temperature") else {}   # (else the model's own)
        # normalize=False: fish-speech's text normalising is for English and Chinese (numbers); it would also
        # touch German text. use_memory_cache: a voice sample is encoded once, by its audio.
        req = ServeTTSRequest(text=r["text"], references=[ref], seed=r.get("seed"), format="wav", normalize=False,
                              use_memory_cache="on", **kw)
        audio = None
        for res in eng.inference(req):
            if res.code == "error":
                raise res.error
            if res.code == "final":
                audio = res.audio
        if audio is None:
            return {"ok": False, "error": "Fish S2 Pro made no sound for the line"}
        sr, wav = audio
        return _write(r["out"], wav, sr)
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
            continue
        if r.get("cmd") == "quit":
            break
        try:
            a = handle(r)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            a = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _answer.write("@@TTS " + json.dumps(a) + "\n")


if __name__ == "__main__":
    main()
