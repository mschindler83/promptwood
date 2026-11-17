"""VideoChainH3LoRA: a MiniMax-H3 LoRA in any format ComfyUI reads, also on the pruned H3 files.

The pruned H3 checkpoints (the standard ..._pruned_int8 files and fine-tunes built on them) replace the time
embedder and the full-width adaln projections with a small curve: each block's adaln projection takes an 8-value row
of adaln_t_table instead of the 2688-value silu(t_emb). A LoRA trained on the unpruned model carries its adaln update
for that 2688-value input, so ComfyUI's LoRA loader cannot add it to the pruned weight (it logs "shape ... is invalid"
and skips it). This node applies everything else as ComfyUI's loader does and adds the adaln update at run time:
B @ A @ silu(t_emb), with silu(t_emb) read from the grid that ComfyUI-MiniMax-H3-Turbo ships
(h3_silu_temb_grid.safetensors, one row per adaln_t_table row), interpolated exactly where the model interpolates
its curve row. The idea of re-injecting the adaln update at run time is that of ComfyUI-MiniMax-H3-Turbo
(Apache-2.0); this code is Promptwood's own, works for every LoRA format ComfyUI maps and stacks with the Turbo
LoRA's injection and with further LoRAs instead of replacing them.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

GRID_FILE = "h3_silu_temb_grid.safetensors"     # [rows of adaln_t_table, 2688]: silu(t_emb) on the same t grid
GRID_KEY = "silu_t_emb_grid"
ADALN_WEIGHT = ".adaln_proj.linear.weight"
_GRID: "torch.Tensor | None" = None        # (torch is imported where it is used: the package loads without it)


def silu_rows(t_emb: torch.Tensor, table: torch.Tensor, egrid: torch.Tensor) -> torch.Tensor:
    """The silu(t_emb) rows for the model's curve rows t_emb [M, 8]: each row lies on the polyline through the
    table rows (the model interpolates between two neighbours), so the grid is interpolated the same way. [M, S]"""
    import torch
    x = t_emb.float()
    t = table.to(x.device, torch.float32)
    e = egrid.to(x.device, torch.float32)
    a, seg = t[:-1], t[1:] - t[:-1]                                      # [G-1, 8]: segment starts and directions
    rel = x[:, None, :] - a[None, :, :]                                  # [M, G-1, 8]
    frac = ((rel * seg[None]).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-12)[None]).clamp(0.0, 1.0)   # [M, G-1]
    off = rel - frac[..., None] * seg[None]
    i = (off * off).sum(-1).argmin(dim=1)                                # the segment the row lies on
    f = frac.gather(1, i[:, None])                                       # [M, 1]
    return torch.lerp(e[i], e[i + 1], f)


def adaln_forward(module, previous: Callable | None, up: torch.Tensor, down: torch.Tensor, scale: float,
                  table: Callable[[], torch.Tensor], egrid: torch.Tensor) -> Callable:
    """A replacement forward for one AdalnProj: its output (previous: an earlier replacement, e.g. the Turbo
    LoRA's or another of these; None: the module's own forward) plus scale * up @ down @ silu(t_emb), split as
    AdalnProj splits its projection. table: returns adaln_t_table (read at call time: it may move devices)."""
    import torch
    own = type(module).forward
    cache: dict = {}

    def on(device):
        if device not in cache:
            cache.clear()
            cache[device] = (up.to(device, torch.float32), down.to(device, torch.float32))
        return cache[device]

    def forward(t_emb):
        chunks = previous(t_emb) if previous is not None else own(module, t_emb)
        if scale == 0:
            return chunks
        u, d = on(t_emb.device)
        rows = silu_rows(t_emb, table(), egrid)
        delta = ((rows @ d.T) @ u.T) * scale                             # [M, out]
        delta = delta.to(chunks[0].dtype).view(delta.shape[0] * module.modalities, module.expand * module.hidden)
        return tuple(c + e for c, e in zip(chunks, delta.chunk(module.expand, dim=-1)))

    return forward


def split_adaln(loaded: dict, in_dim: Callable[[str], int | None]) -> tuple[dict, dict]:
    """ComfyUI's loaded LoRA patches ({model key: adapter}) split into (the ones ComfyUI applies, {model key: (up,
    down, scale)} to inject at run time): a plain LoRA on an adaln projection whose input width is not the model's.
    Others (a LoCon mid, DoRA, full weights, a width that fits) stay with ComfyUI, as before."""
    rest, inject = {}, {}
    for key, v in loaded.items():
        if isinstance(key, str) and key.endswith(ADALN_WEIGHT) and type(v).__name__ == "LoRAAdapter":
            up, down, alpha, mid, dora = v.weights[:5]           # (ComfyUI's LoRAAdapter: up, down, alpha, mid, ...)
            if mid is None and dora is None and getattr(down, "ndim", 0) == 2 and in_dim(key) not in (None,
                                                                                                    down.shape[1]):
                inject[key] = (up, down, (float(alpha) / down.shape[0]) if alpha is not None else 1.0)
                continue
        rest[key] = v
    return rest, inject


def silu_grid() -> torch.Tensor:
    """The silu(t_emb) grid of ComfyUI-MiniMax-H3-Turbo (any folder of custom_nodes that has it)."""
    global _GRID
    if _GRID is None:
        import comfy.utils
        import folder_paths
        for root in folder_paths.get_folder_paths("custom_nodes"):
            for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                path = os.path.join(root, name, GRID_FILE)
                if os.path.isfile(path):
                    _GRID = comfy.utils.load_torch_file(path)[GRID_KEY]
                    break
            if _GRID is not None:
                break
        if _GRID is None:
            raise FileNotFoundError(f"{GRID_FILE} not found in ComfyUI's custom_nodes: install ComfyUI-MiniMax-H3-"
                                    "Turbo (it ships the grid this node reads)")
    return _GRID


class VideoChainH3LoRA:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        import folder_paths
        return {"required": {"model": ("MODEL",),
                             "lora_name": (folder_paths.get_filename_list("loras"),),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = ("A MiniMax-H3 LoRA as ComfyUI's LoRA loader applies it; on the pruned H3 files also its adaln "
                   "part (trained on the unpruned model), added at run time.")

    def apply(self, model, lora_name, strength):
        import comfy.lora
        import comfy.lora_convert
        import comfy.utils
        import folder_paths
        if strength == 0:
            return (model,)
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(sd), key_map)
        dm = model.get_model_object("diffusion_model")
        patched = model.clone()
        inject: dict = {}
        if getattr(dm, "use_adaln_curves", False):
            def in_dim(key):
                try:
                    return int(patched.get_model_object(key).shape[1])
                except Exception:  # noqa: BLE001  (not a 2-D weight: left to ComfyUI)
                    return None
            loaded, inject = split_adaln(loaded, in_dim)
        done = set(patched.add_patches(loaded, strength))
        for k in loaded:
            if k not in done:
                log.warning("NOT LOADED %s", k)
        if inject:
            egrid = silu_grid()
            for key, (up, down, scale) in inject.items():
                name = key[: -len(".linear.weight")]
                attr = name + ".forward"
                patched.add_object_patch(attr, adaln_forward(patched.get_model_object(name),
                                                             patched.object_patches.get(attr), up, down,
                                                             scale * float(strength), lambda: dm.adaln_t_table,
                                                             egrid))
            log.info("VideoChainH3LoRA %s: %d adaln projections added at run time (pruned H3)", lora_name,
                     len(inject))
        return (patched,)
