"""ComfyUI's LoRA and model folders, and telling LoRA files apart by the tensor names in their headers."""
from __future__ import annotations

import functools
import json
import logging
import os
import re
from pathlib import Path
from typing import Sequence

import yaml

logger = logging.getLogger(__name__)


def lora_name_for(path: str, names: list[str], roots: Sequence[Path] = ()) -> str | None:
    """ComfyUI's name for a LoRA file: the listed name the path ends with (longest match), else the listed name
    that is the same file (followed links) in one of ComfyUI's LoRA folders (roots)."""
    p = Path(path).as_posix()
    try:
        real = os.path.realpath(path)
    except OSError:
        real = None
    hits = [n for n in names if p.endswith("/" + n.replace("\\", "/")) or p == n]
    if hits and roots:
        # with the folders known, a name only counts if it is this very file (a 'style.safetensors' elsewhere
        # is another LoRA than ComfyUI's 'style.safetensors')
        hits = [n for n in hits if any((Path(r) / n).exists() and os.path.realpath(Path(r) / n) == real
                                       for r in roots)]
    if hits:
        return max(hits, key=len)
    if real is None:
        return None
    for root in roots:
        for n in names:
            cand = Path(root) / n
            if cand.exists() and os.path.realpath(cand) == real:
                return n
    return None


def comfy_lora_roots(comfy_dir: Path) -> list[Path]:
    """ComfyUI's LoRA folders: models/loras plus the 'loras' entries of extra_model_paths.yaml."""
    return comfy_model_roots(comfy_dir, "loras")


def comfy_model_roots(comfy_dir: Path, kind: str) -> list[Path]:
    """ComfyUI's folders for one kind of model: models/<kind> plus the <kind> entries of extra_model_paths.yaml."""
    comfy_dir = Path(comfy_dir).expanduser()
    roots = [comfy_dir / "models" / kind]
    cfg = comfy_dir / "extra_model_paths.yaml"
    if cfg.is_file():
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001  (invalid yaml: only models/<kind>, but say why)
            logger.warning("%s is not readable: %s", cfg, e)
            data = {}
        if not isinstance(data, dict):
            data = {}
        for section in data.values():
            if not isinstance(section, dict) or not section.get(kind):
                continue
            base = Path(os.path.expanduser(str(section.get("base_path", ""))))
            value = section[kind]
            for entry in (value if isinstance(value, list) else str(value).split("\n")):
                entry = str(entry).strip()
                if entry:
                    root = Path(os.path.expanduser(entry))
                    roots.append(root if root.is_absolute() else (comfy_dir / base / root if not base.is_absolute()
                                                                  else base / root))
    return [r for r in roots if r.is_dir()]


def _read_safetensors_keys(path: Path) -> frozenset[str] | None:
    """The tensor names in a .safetensors header (None: not readable / no valid header)."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 100 * 2**20:
                return None
            return frozenset(json.loads(f.read(n)))
    except (OSError, ValueError, TypeError):
        return None


@functools.lru_cache(maxsize=4096)
def _safetensors_keys_cached(path: str, size: int, mtime_ns: int) -> frozenset[str] | None:
    return _read_safetensors_keys(Path(path))


def _safetensors_keys(path: Path) -> frozenset[str] | None:
    """The tensor names of a file, read once per (path, size, mtime) for all the LoRA checks."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return _safetensors_keys_cached(str(path), st.st_size, st.st_mtime_ns)


_safetensors_keys.cache_clear = _safetensors_keys_cached.cache_clear   # type: ignore[attr-defined]  # (tests clear the cache through the plain function)


def is_turbo_lora(path: Path) -> bool:
    """A step-distilled H3 turbo LoRA (needs ComfyUI-MiniMax-H3-Turbo's own loader and sampler): 'turbo' in the
    name, or it trains final_layer, which normal H3 LoRAs do not."""
    if "turbo" in Path(path).name.lower():
        return True
    keys = _safetensors_keys(path)
    return keys is not None and any(k.startswith("final_layer") or ".final_layer" in k for k in keys)


_H3_ATTN = re.compile(r"blocks[._]\d+[._]attn[._]qkv_proj")
_H3_MLP = re.compile(r"blocks[._]\d+[._]mlp[._]fc1")


def is_h3_lora(path: Path) -> bool:
    """Whether a .safetensors file is a MiniMax-H3 LoRA (any format: ai-toolkit/peft, kohya, LoKr), from the
    tensor names in its header only: H3's blocks have attn.qkv_proj and mlp.fc1."""
    names = _safetensors_keys(path)
    return names is not None and any(_H3_ATTN.search(k) for k in names) and any(_H3_MLP.search(k) for k in names)


_H3_VIDEO_IN = re.compile(r"(^|\.)video_patch_proj\.")
_H3_AUDIO_IN = re.compile(r"(^|\.)audio_patch_proj\.")
_ADAPTER = re.compile(r"lora|lokr|loha|\.alpha$", re.IGNORECASE)


def is_h3_checkpoint(path: Path) -> bool:
    """Whether a .safetensors file is a whole MiniMax-H3 model (the standard one or a fine-tune), from the tensor
    names in its header only: H3 alone has both a video and an audio patch layer; a LoRA has adapter tensors."""
    names = _safetensors_keys(path)
    return names is not None and any(_H3_VIDEO_IN.search(k) for k in names) and \
        any(_H3_AUDIO_IN.search(k) for k in names) and not any(_ADAPTER.search(k) for k in names)


def find_h3_checkpoints(roots: list[Path]) -> list[str]:
    """The H3 checkpoints other than the standard files in these folders (ComfyUI's diffusion model folders), as the
    names ComfyUI lists them by (relative to their folder), sorted."""
    from .video import UNET, UNET_REF
    return [n for _p, n in _find_loras(roots, is_h3_checkpoint) if Path(n).name not in (UNET, UNET_REF)]


_QWEN_IMG = re.compile(r"transformer_blocks[._]\d+[._]img_mlp")
_QWEN_TXT = re.compile(r"transformer_blocks[._]\d+[._]txt_mlp")


def is_qwen_lora(path: Path) -> bool:
    """Whether a .safetensors file is a Qwen-Image / Qwen-Image-Edit LoRA, from the tensor names in its header: the
    MMDiT's blocks have img_mlp and txt_mlp (Flux's have no such names; it also has single_transformer_blocks)."""
    names = _safetensors_keys(path)
    return (names is not None and any(_QWEN_IMG.search(k) for k in names) and any(_QWEN_TXT.search(k) for k in names)
            and not any("single_transformer_blocks" in k for k in names))


def find_qwen_loras(roots: list[Path]) -> list[tuple[str, str]]:
    """Qwen Image (Edit) LoRAs in these folders, without the Lightning speed LoRAs (Fast mode applies those)."""
    return [(p, n) for p, n in _find_loras(roots, is_qwen_lora) if "lightning" not in n.lower()]


def is_ic_lora(name: str) -> bool:
    """An Open-H3-Regenerate-2K-IC LoRA (or a converted copy): an upscaler part for the Ref2VA model, never a clip's
    LoRA nor the 2K refiner (e.g. open_h3_regenerate_2k_ic_lora, ..._regen_2k_IC_comfy_fp32)."""
    n = Path(name).name.lower()
    return ("regenerate_2k" in n or "regenerate-2k" in n or "regen_2k" in n) and \
        re.search(r"(^|[_\-.])ic([_\-.]|$)", n) is not None


def find_h3_loras(roots: list[Path]) -> list[tuple[str, str]]:
    """All H3 LoRAs in these folders (links followed) as (path, name relative to its folder), sorted by name.
    Turbo LoRAs are included; is_turbo_lora tells them apart. Pika H3 Flex is left out: its speed mode applies it;
    so are Open-H3-Regenerate-2K-IC LoRAs (upscaler parts, is_ic_lora)."""
    return [(p, n) for p, n in _find_loras(roots, is_h3_lora) if "pika_h3_flex" not in n.lower() and not is_ic_lora(n)]


def _find_loras(roots: list[Path], test) -> list[tuple[str, str]]:
    out, seen, visited = [], set(), set()
    for root in roots:
        for dirpath, dirs, files in os.walk(root, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in visited:                      # a link back into a folder already walked: not again
                dirs[:] = []
                continue
            visited.add(real)
            for f in files:
                if not f.endswith(".safetensors"):
                    continue
                full = os.path.join(dirpath, f)
                if full in seen or not test(Path(full)):
                    continue
                seen.add(full)
                out.append((full, os.path.relpath(full, root)))
    return sorted(out, key=lambda x: x[1].lower())
