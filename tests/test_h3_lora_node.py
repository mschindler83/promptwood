"""Promptwood's VideoChainH3LoRA node (comfy_nodes/video_chain_nodes/h3_lora.py): a LoRA's adaln update applied at
run time on the pruned H3 files (their adaln projections take an 8-value curve, the LoRA's update the full
2688-value silu(t_emb)). Checked with torch on CPU (.venv-tts) against the formula, with a copy of ComfyUI's
AdalnProj."""
import os
import subprocess
from pathlib import Path

import pytest

NODES = Path(__file__).resolve().parents[1] / "comfy_nodes" / "video_chain_nodes"
TTS_PY = Path(__file__).resolve().parents[1] / ".venv-tts" / "bin" / "python"   # (a venv with torch)


@pytest.mark.skipif(not TTS_PY.is_file(), reason="needs a Python with torch (.venv-tts)")
def test_the_adaln_update_on_a_pruned_model(tmp_path):
    script = tmp_path / "check.py"
    script.write_text(f'''
import sys, torch
from torch import nn
sys.path.insert(0, {str(NODES)!r})
import h3_lora as H

class AdalnProj(nn.Module):          # ComfyUI's comfy/ldm/minimax/model.py AdalnProj (pruned: apply_silu False)
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True):
        super().__init__()
        self.expand, self.modalities, self.hidden, self.apply_silu = expand, modalities, hidden, apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)
    def forward(self, t_emb):
        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)

torch.manual_seed(5)
G, T, S, R = 11, 8, 16, 3                       # grid rows, curve width, silu(t_emb) width, LoRA rank
proj = AdalnProj(T, hidden=4, expand=6, modalities=2, apply_silu=False)
table, egrid = torch.randn(G, T), torch.randn(G, S)
out = proj.linear.out_features
spots = [(0, 0.0), (3, 0.25), (7, 0.9), (9, 1.0)]                 # (grid row, fraction towards the next)
t_emb = torch.stack([torch.lerp(table[i], table[i + 1], f) for i, f in spots])
want_rows = torch.stack([torch.lerp(egrid[i], egrid[i + 1], f) for i, f in spots])
got_rows = H.silu_rows(t_emb, table, egrid)
assert torch.allclose(got_rows, want_rows, atol=1e-5), (got_rows - want_rows).abs().max()

up1, down1, up2, down2 = torch.randn(out, R), torch.randn(R, S), torch.randn(out, R), torch.randn(R, S)
with torch.no_grad():
    base = proj.linear(t_emb)
    d1 = (want_rows @ down1.T @ up1.T) * 0.7
    d2 = (want_rows @ down2.T @ up2.T) * 0.25
    def chunks(x):
        return x.view(x.shape[0] * proj.modalities, proj.expand * proj.hidden).chunk(proj.expand, dim=-1)
    f1 = H.adaln_forward(proj, None, up1, down1, 0.7, lambda: table, egrid)
    for g, w in zip(f1(t_emb), chunks(base + d1)):
        assert torch.allclose(g, w, atol=1e-4), (g - w).abs().max()
    f2 = H.adaln_forward(proj, f1, up2, down2, 0.25, lambda: table, egrid)   # chained on the first (or Turbo's)
    for g, w in zip(f2(t_emb), chunks(base + d1 + d2)):
        assert torch.allclose(g, w, atol=1e-4), (g - w).abs().max()
    for g, w in zip(H.adaln_forward(proj, None, up1, down1, 0.0, lambda: table, egrid)(t_emb), chunks(base)):
        assert torch.equal(g, w)                                            # strength 0: the model as it is

class LoRAAdapter:                                # (ComfyUI's: weights = up, down, alpha, mid, dora, reshape)
    def __init__(self, up, down, alpha=None, mid=None):
        self.weights = (up, down, alpha, mid, None, None)
key = "diffusion_model.blocks.3.adaln_proj.linear.weight"
fits = "diffusion_model.blocks.4.adaln_proj.linear.weight"
qkv = "diffusion_model.blocks.3.attn.qkv_proj.weight"
loaded = {{key: LoRAAdapter(up1, down1, alpha=1.5), fits: LoRAAdapter(torch.randn(out, R), torch.randn(R, T)), qkv: LoRAAdapter(up2, down2),
           "diffusion_model.blocks.5.adaln_proj.linear.weight": LoRAAdapter(up1, down1, mid=torch.randn(R, R, 1, 1))}}
rest, inject = H.split_adaln(loaded, lambda k: T)
assert set(inject) == {{key}} and set(rest) == set(loaded) - {{key}}       # (a LoCon mid: left to ComfyUI)
u, d, scale = inject[key]
assert u is up1 and d is down1 and abs(scale - 1.5 / R) < 1e-9             # alpha / rank, as ComfyUI scales a LoRA
print("same")
''')
    r = subprocess.run([str(TTS_PY), "-I", str(script)], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "OMP_NUM_THREADS": "4"}, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "same", r.stderr[-2000:]
