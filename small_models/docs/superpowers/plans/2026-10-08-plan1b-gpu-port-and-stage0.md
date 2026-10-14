# Plan 1b: GPU port and Stage 0 on the A100 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the judge to PyTorch with flash-linear-attention's fused Kimi Delta Attention kernel, prove the port matches the MLX model's structure and a plain recurrence, and measure Stage 0 on the A100 so the config decision is made on real numbers.

**Architecture:** `vmt/` mirrors `vm/` module for module and reuses `vm/model/config.py` (pure Python, no MLX import), so both implementations share one set of configs and must have identical parameter counts. The delta core is fla's `chunk_kda`, checked against `vmt/kda_ref.py`, a plain step-by-step loop. Code is written on the Mac, synced to the instance with rsync, and tested there.

**Tech Stack:** Python 3.12 (uv-managed `~/vmenv` on the instance), PyTorch 2.11 (CUDA 12.8), flash-linear-attention 0.5.2, Triton 3.6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-verifier-judge-design.md` (sections 2, 4 and 6). Follows plan 1 (`2026-10-08-plan1-model-core-and-stage0.md`).

## Global Constraints

- Hard spending cap $100. The instance bills from 06:08 BST, 8 Oct 2026; every stage adds its hours to `docs/cost_ledger.md`.
- Instance: `ubuntu@150.136.116.45`, NVIDIA A100-SXM4-40GB, 30 vCPUs. Interpreter `~/vmenv/bin/python`.
- Parameter counts must equal the MLX model's exactly: TINY 245,572; DEV 24,324,959; MID 44,509,873; MAIN 116,490,541.
- Gate: per-step log decay `gate_lower * sigmoid(exp(A_log) * (f + dt_bias))`, `gate_lower = -1.0` (the shared config).
- Training precision: parameters in float32, compute under bfloat16 autocast.
- Commits authored `Anil Kumar <anil@hud.ai>`, no `Co-Authored-By` line.

## File structure

| File | Responsibility |
|---|---|
| `scripts/sync.sh` | rsync the repo to the instance |
| `vmt/__init__.py` | package |
| `vmt/kda_ref.py` | `kda_recurrent`: plain step-by-step reference |
| `vmt/layers.py` | `ZRMSNorm`, `ShortConv`, `KDA`, `Attention`, `SwiGLU` |
| `vmt/model.py` | `Block`, `Judge`, `n_params` |
| `vmt/train.py` | `param_groups`, `build_optimizers`, `lm_loss`, `Trainer` |
| `scripts/stage0_gpu.py` | GPU Stage 0 measurements |
| `tests_gpu/test_layers_t.py`, `tests_gpu/test_model_t.py`, `tests_gpu/test_train_t.py`, `tests_gpu/test_stage0_gpu.py` | tests, run on the instance |
| `docs/cost_ledger.md` | GPU hours and dollars |

---

### Task 1: Sync, reference recurrence and layers

**Files:**
- Create: `scripts/sync.sh`, `vmt/__init__.py`, `vmt/kda_ref.py`, `vmt/layers.py`, `docs/cost_ledger.md`
- Test: `tests_gpu/test_layers_t.py`

**Interfaces:**
- Produces: `kda_recurrent(q, k, v, log_g, beta, scale, state=None) -> (o [B,T,H,V], state [B,H,K,V])` in float32; `ZRMSNorm(dim)`, `ShortConv(dim, kernel=4)`, `KDA(d_model, n_heads, head_dim, chunk=32, gate_lower=-1.0)` with `KDA.gates(x) -> (log_g, beta)` and `KDA.use_reference: bool`, `Attention(d_model, n_heads, n_kv_heads, head_dim)`, `SwiGLU(d_model, hidden)`. All take and return `[B, T, d]`.

- [ ] **Step 1: Write the sync script and ledger**

`scripts/sync.sh`:
```bash
#!/bin/sh
# Copy the repository to the GPU instance (code only; experiments stay where they are made).
set -e
HOST=${HOST:-ubuntu@150.136.116.45}
rsync -az --delete --exclude .venv --exclude __pycache__ --exclude experiments --exclude .git \
  "$(dirname "$0")/../" "$HOST:~/verifier_model/"
```

`docs/cost_ledger.md`:
```markdown
# GPU cost ledger

Cap: $100. Instance: Lambda 1x A100 40GB SXM4, launched 8 Oct 2026 06:08 BST. Hourly price: read from the
Lambda console and entered here by the user; until then the ledger uses $1.29/h, Lambda's published A100 40GB
price as remembered, and says so.

| Stage | From | To | Hours | Dollars (at $1.29/h) |
|---|---|---|---:|---:|
```

- [ ] **Step 2: Write the failing tests**

`tests_gpu/test_layers_t.py`:
```python
import pytest
import torch
import torch.nn.functional as F

from vmt.kda_ref import kda_recurrent
from vmt.layers import KDA, Attention, ShortConv, SwiGLU, ZRMSNorm
from fla.ops.kda import chunk_kda

DEV = "cuda"
B, T, D = 2, 70, 128


def causal_leak(fn, x, t0):
    x2 = torch.cat([x[:, :t0], torch.randn_like(x[:, t0:])], dim=1)
    with torch.no_grad():
        return (fn(x)[:, :t0] - fn(x2)[:, :t0]).abs().max().item()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, D, device=DEV)


def kda_inputs(Bn=2, Tn=128, H=3, K=64, lo=-1.0):
    torch.manual_seed(0)
    q = F.normalize(torch.randn(Bn, Tn, H, K, device=DEV), dim=-1)
    k = F.normalize(torch.randn(Bn, Tn, H, K, device=DEV), dim=-1)
    v = torch.randn(Bn, Tn, H, K, device=DEV)
    g = torch.rand(Bn, Tn, H, K, device=DEV) * lo
    b = torch.rand(Bn, Tn, H, device=DEV)
    return q, k, v, g, b


@pytest.mark.parametrize("lo", [-0.1, -1.0])
def test_fused_kernel_matches_plain_recurrence(lo):
    q, k, v, g, b = kda_inputs(lo=lo)
    o1, s1 = chunk_kda(q, k, v, g, b, scale=64**-0.5, output_final_state=True)
    o2, s2 = kda_recurrent(q, k, v, g, b, 64**-0.5)
    assert (o1.float() - o2).abs().max().item() < 2e-3 * o2.abs().max().item() + 1e-4
    assert (s1.float() - s2).abs().max().item() < 2e-3 * s2.abs().max().item() + 1e-3


def test_fused_kernel_gradients_match():
    xs = [t.clone().requires_grad_() for t in kda_inputs(Tn=64)]
    chunk_kda(*xs, scale=64**-0.5)[0].square().sum().backward()
    ys = [t.detach().clone().requires_grad_() for t in xs]
    kda_recurrent(*ys, 64**-0.5)[0].square().sum().backward()
    for a, c in zip(xs, ys):
        assert (a.grad - c.grad).abs().max().item() < 5e-3 * max(1.0, c.grad.abs().max().item())


def test_comparison_has_teeth():
    q, k, v, g, b = kda_inputs()
    o1, _ = chunk_kda(q, k, v, g, b * 0.9, scale=64**-0.5)
    o2, _ = kda_recurrent(q, k, v, g, b, 64**-0.5)
    assert (o1.float() - o2).abs().max().item() > 1e-2 * o2.abs().max().item()


def test_causal_leak_has_teeth(x):
    assert causal_leak(lambda y: y.mean(dim=1, keepdim=True).expand_as(y), x, 30) > 1e-3


@pytest.mark.parametrize("make", [lambda: ShortConv(D), lambda: KDA(D, 2, 64), lambda: Attention(D, 2, 1, 64)])
def test_layers_are_causal_and_keep_shape(make, x):
    layer = make().to(DEV)
    assert layer(x).shape == (B, T, D)
    assert causal_leak(layer, x, 30) < 1e-4


def test_kda_layer_fused_equals_reference_path(x):
    layer = KDA(D, 2, 64).to(DEV)
    with torch.no_grad():
        fused = layer(x)
        layer.use_reference = True
        ref = layer(x)
    assert (fused - ref).abs().max().item() < 2e-3 * ref.abs().max().item() + 1e-4


def test_kda_gates_stay_in_range(x):
    log_g, beta = KDA(D, 2, 64).to(DEV).gates(x)
    assert log_g.max().item() <= 0.0 and log_g.min().item() >= -1.0
    assert 0.0 <= beta.min().item() and beta.max().item() <= 1.0


def test_norm_and_mlp(x):
    out = ZRMSNorm(D).to(DEV)(x)
    assert (out - x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).abs().max().item() < 1e-4
    assert SwiGLU(D, 256).to(DEV)(x).shape == (B, T, D)


def test_attention_matches_naive_softmax(x):
    layer = Attention(D, 4, 2, 32).to(DEV)
    H, Hk, Dh = 4, 2, 32
    with torch.no_grad():
        q = layer.q_norm(layer.q_proj(x).view(B, T, H, Dh)).transpose(1, 2)
        k = layer.k_norm(layer.k_proj(x).view(B, T, Hk, Dh)).transpose(1, 2).repeat_interleave(H // Hk, 1)
        v = layer.v_proj(x).view(B, T, Hk, Dh).transpose(1, 2).repeat_interleave(H // Hk, 1)
        s = (q @ k.transpose(-1, -2)) * Dh**-0.5
        s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV)), float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(layer.g_proj(x))
        assert (layer(x) - layer.o_proj(o)).abs().max().item() < 1e-4
```

- [ ] **Step 3: Sync and run the tests to verify they fail**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu/test_layers_t.py -q'`
Expected: FAIL with `ModuleNotFoundError: No module named 'vmt'`

- [ ] **Step 4: Write the implementation**

`vmt/__init__.py`: empty.

`vmt/kda_ref.py`:
```python
"""Plain step-by-step gated delta rule with a per-channel gate, in float32: the reference the fused kernel
is tested against. Same recurrence as vm/model/delta.py and mlx-lm's gated_delta_ops."""
import torch


def kda_recurrent(q, k, v, log_g, beta, scale, state=None):
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, log_g, beta = (t.float() for t in (q, k, v, log_g, beta))
    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device) if state is None else state.float()
    out = []
    for t in range(T):
        S = S * log_g[:, t].exp()[..., None]
        u = beta[:, t][..., None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S))
        S = S + k[:, t][..., None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(out, 1), S
```

`vmt/layers.py`:
```python
"""PyTorch twins of vm/model/layers.py, with flash-linear-attention's fused KDA kernel for the delta core."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.kda import chunk_kda

from vmt.kda_ref import kda_recurrent


class ZRMSNorm(nn.Module):
    """RMSNorm whose scale is 1 + weight, weight initialised to 0; computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class ShortConv(nn.Module):
    """Causal depthwise convolution over time, then SiLU. Input and output [B, T, dim]."""

    def __init__(self, dim: int, kernel: int = 4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0)))
        return F.silu(y.transpose(1, 2))


class KDA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk: int = 32, gate_lower: float = -1.0):
        super().__init__()
        self.n_heads, self.head_dim, self.gate_lower = n_heads, head_dim, gate_lower
        self.use_reference = False
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.q_conv, self.k_conv, self.v_conv = ShortConv(inner), ShortConv(inner), ShortConv(inner)
        self.f_down = nn.Linear(d_model, head_dim, bias=False)
        self.f_up = nn.Linear(head_dim, inner, bias=False)
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(inner))
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        self.g_proj = nn.Linear(d_model, inner, bias=False)
        self.o_norm = ZRMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, d_model, bias=False)

    def gates(self, x):
        B, T, _ = x.shape
        f = (self.f_up(self.f_down(x)) + self.dt_bias).view(B, T, self.n_heads, self.head_dim)
        log_g = self.gate_lower * torch.sigmoid(self.A_log.exp()[:, None] * f.float())
        beta = torch.sigmoid(self.b_proj(x).float())
        return log_g, beta

    def forward(self, x):
        B, T, _ = x.shape
        H, K = self.n_heads, self.head_dim
        q = F.normalize(self.q_conv(self.q_proj(x)).view(B, T, H, K).float(), dim=-1, eps=1e-6)
        k = F.normalize(self.k_conv(self.k_proj(x)).view(B, T, H, K).float(), dim=-1, eps=1e-6)
        v = self.v_conv(self.v_proj(x)).view(B, T, H, K)
        log_g, beta = self.gates(x)
        if self.use_reference:
            o, _ = kda_recurrent(q, k, v, log_g, beta, K**-0.5)
        else:
            dt = v.dtype
            o, _ = chunk_kda(q.to(dt), k.to(dt), v, log_g, beta, scale=K**-0.5)
        o = self.o_norm(o.to(x.dtype)).reshape(B, T, H * K) * torch.sigmoid(self.g_proj(x))
        return self.o_proj(o)


class Attention(nn.Module):
    """Causal grouped-query attention with QK-norm, a sigmoid output gate and no positional encoding."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.q_norm, self.k_norm = ZRMSNorm(head_dim), ZRMSNorm(head_dim)
        self.g_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, Hk, Dh = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).view(B, T, H, Dh)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, Hk, Dh)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, Dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v.to(q.dtype), is_causal=True, scale=Dh**-0.5, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, T, H * Dh) * torch.sigmoid(self.g_proj(x))
        return self.o_proj(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

- [ ] **Step 5: Sync and run the tests to verify they pass**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu/test_layers_t.py -q'`
Expected: `12 passed`

- [ ] **Step 6: Commit**

```bash
git add scripts/sync.sh vmt docs/cost_ledger.md tests_gpu/test_layers_t.py
git commit -m "GPU layers: fla's fused KDA kernel checked against a plain recurrence in outputs, state and gradients; causal layers; attention equals naive softmax"
```

---

### Task 2: The Judge model in PyTorch

**Files:**
- Create: `vmt/model.py`
- Test: `tests_gpu/test_model_t.py`

**Interfaces:**
- Consumes: `vm.model.config` (`ModelConfig`, `TINY`, `DEV`, `MID`, `MAIN`), `vmt.layers`.
- Produces: `vmt.model.Judge(cfg)` with `forward(tokens, segments) -> hidden`, `lm_weight() -> embed weight`, `lm_logits(hidden)`, `verdict_logit(hidden, pos)`, `pointer_logits(hidden, pos, line_mask)`; `n_params(module) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests_gpu/test_model_t.py`:
```python
import torch

from vm.model.config import DEV, MAIN, MID, TINY
from vmt.model import Judge, n_params

DEV_ = "cuda"


def test_parameter_counts_equal_the_mlx_model():
    assert n_params(Judge(TINY)) == 245_572
    assert n_params(Judge(DEV)) == 24_324_959
    assert n_params(Judge(MID)) == 44_509_873
    assert n_params(Judge(MAIN)) == 116_490_541


def test_layer_pattern():
    m = Judge(MAIN)
    kinds = [type(b.mixer).__name__ for b in m.layers]
    assert [i for i, k in enumerate(kinds) if k == "Attention"] == [3, 7, 11]


def batch(B=2, T=40):
    torch.manual_seed(0)
    return (torch.randint(0, TINY.vocab_size, (B, T), device=DEV_),
            torch.randint(0, 4, (B, T), device=DEV_))


def test_heads_and_masking():
    m = Judge(TINY).to(DEV_)
    tok, seg = batch()
    with torch.no_grad():
        h = m(tok, seg)
        assert h.shape == (2, 40, TINY.d_model)
        assert m.lm_logits(h).shape == (2, 40, TINY.vocab_size)
        pos = torch.tensor([39, 20], device=DEV_)
        assert m.verdict_logit(h, pos).shape == (2,)
        mask = torch.zeros(2, 40, dtype=torch.bool, device=DEV_)
        mask[:, 5] = True
        p = m.pointer_logits(h, pos, mask)
        assert p[0, 6].item() == -1e9 and p[0, 5].item() > -1e9


def test_model_is_causal_and_segments_matter():
    m = Judge(TINY).to(DEV_)
    tok, seg = batch()
    with torch.no_grad():
        h1 = m(tok, seg)
        h2 = m(torch.cat([tok[:, :20], torch.randint(0, TINY.vocab_size, (2, 20), device=DEV_)], 1), seg)
        assert (h1[:, :20] - h2[:, :20]).abs().max().item() < 1e-4
        h3 = m(tok, torch.full_like(seg, 3))
        assert (h1 - h3).abs().max().item() > 1e-3
```

- [ ] **Step 2: Sync and run to verify failure**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu/test_model_t.py -q'`
Expected: FAIL with `ModuleNotFoundError: No module named 'vmt.model'`

- [ ] **Step 3: Write the implementation**

`vmt/model.py`:
```python
"""PyTorch twin of vm/model/model.py. Embeddings are initialised with std d_model**-0.5, as MLX does."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from vm.model.config import ModelConfig
from vmt.layers import KDA, Attention, SwiGLU, ZRMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, attention: bool):
        super().__init__()
        self.norm1 = ZRMSNorm(cfg.d_model)
        self.mixer = (Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) if attention
                      else KDA(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.chunk, cfg.gate_lower))
        self.norm2 = ZRMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Judge(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.segment = nn.Embedding(cfg.n_segments, cfg.d_model)
        for e in (self.embed, self.segment):
            nn.init.normal_(e.weight, std=cfg.d_model**-0.5)
        self.layers = nn.ModuleList(Block(cfg, cfg.is_attention(i)) for i in range(cfg.n_layers))
        self.norm = ZRMSNorm(cfg.d_model)
        self.verdict_head = nn.Linear(cfg.d_model, 1)
        self.pointer_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, tokens, segments):
        x = self.embed(tokens) + self.segment(segments)
        for block in self.layers:
            x = block(x)
        return self.norm(x)

    def lm_weight(self):
        return self.embed.weight

    def lm_logits(self, hidden):
        return hidden @ self.embed.weight.to(hidden.dtype).T

    def _at(self, hidden, pos):
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]

    def verdict_logit(self, hidden, pos):
        return self.verdict_head(self._at(hidden, pos))[:, 0]

    def pointer_logits(self, hidden, pos, line_mask):
        query = self.pointer_query(self._at(hidden, pos))
        scores = (hidden @ query[:, :, None])[..., 0] / math.sqrt(self.cfg.d_model)
        return scores.masked_fill(~line_mask, -1e9)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
```

- [ ] **Step 4: Sync and run the tests to verify they pass**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu/test_model_t.py -q'`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add vmt/model.py tests_gpu/test_model_t.py
git commit -m "GPU Judge model with the MLX model's exact parameter counts (TINY to MAIN), layer pattern, heads, causality and segment effect"
```

---

### Task 3: Training on the GPU

**Files:**
- Create: `vmt/train.py`
- Test: `tests_gpu/test_train_t.py`

**Interfaces:**
- Consumes: `Judge` (Task 2), `fla.modules.FusedLinearCrossEntropyLoss`.
- Produces: `param_groups(model) -> (muon_params, adam_params)`; `lm_loss_with(run_model, model, tokens, segments)`; `build_optimizers(model, lr_muon, lr_adam, total_steps, warmup, weight_decay=0.01) -> (optimizers, schedulers)`; `lm_loss(model, tokens, segments) -> loss` (bfloat16 autocast, fused linear cross-entropy, never materialising the logits); `Trainer(model, optimizers, schedulers, loss_fn=lm_loss)` with `step(*batch) -> float`.

- [ ] **Step 1: Write the failing tests**

`tests_gpu/test_train_t.py`:
```python
import torch

from vm.model.config import TINY
from vmt.model import Judge
from vmt.train import Trainer, build_optimizers, param_groups

DEV = "cuda"


def run(lr_muon, lr_adam, steps=60):
    torch.manual_seed(0)
    model = Judge(TINY).to(DEV)
    opts, scheds = build_optimizers(model, lr_muon, lr_adam, total_steps=steps, warmup=5)
    trainer = Trainer(model, opts, scheds)
    torch.manual_seed(0)
    tok = torch.randint(0, TINY.vocab_size, (4, 65), device=DEV)
    seg = torch.zeros_like(tok)
    return [trainer.step(tok, seg) for _ in range(steps)]


def test_muon_gets_exactly_the_hidden_weight_matrices():
    m = Judge(TINY)
    muon, adam = param_groups(m)
    names = {id(p): n for n, p in m.named_parameters()}
    muon_names = {names[id(p)] for p in muon}
    assert "layers.0.mixer.q_proj.weight" in muon_names
    assert not any(n.startswith(("embed", "segment", "verdict_head", "pointer_query")) for n in muon_names)
    assert all(p.ndim == 2 for p in muon)
    assert len(muon) + len(adam) == len(list(m.parameters()))


def test_training_fits_a_fixed_batch():
    losses = run(0.02, 3e-3)
    assert losses[-1] < 0.1 * losses[0]


def test_zero_learning_rate_learns_nothing():
    losses = run(0.0, 0.0, steps=10)
    assert abs(losses[-1] - losses[0]) < 1e-3
```

- [ ] **Step 2: Sync and run to verify failure**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu/test_train_t.py -q'`
Expected: FAIL with `ModuleNotFoundError: No module named 'vmt.train'`

- [ ] **Step 3: Write the implementation**

`vmt/train.py`:
```python
"""Muon for hidden weight matrices, AdamW for the rest, warm-up then cosine decay to 10%; bfloat16 autocast;
fused linear cross-entropy so the [tokens x vocabulary] logits are never materialised."""
from __future__ import annotations

import math

import torch
from fla.modules import FusedLinearCrossEntropyLoss

_NOT_MUON = ("embed", "segment", "verdict_head", "pointer_query")
_flce = FusedLinearCrossEntropyLoss()


def param_groups(model):
    muon, adam = [], []
    for name, p in model.named_parameters():
        (muon if p.ndim == 2 and not name.startswith(_NOT_MUON) else adam).append(p)
    return muon, adam


def _lr_lambda(total_steps: int, warmup: int):
    if warmup < 1 or total_steps <= warmup:
        raise ValueError("need 1 <= warmup < total_steps")

    def f(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = min(1.0, (step - warmup) / (total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    return f


def build_optimizers(model, lr_muon, lr_adam, total_steps, warmup, weight_decay=0.01):
    muon_p, adam_p = param_groups(model)
    opts = [torch.optim.Muon(muon_p, lr=lr_muon, weight_decay=weight_decay),
            torch.optim.AdamW(adam_p, lr=lr_adam, weight_decay=weight_decay, betas=(0.9, 0.95))]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, _lr_lambda(total_steps, warmup)) for o in opts]
    return opts, scheds


def lm_loss_with(run_model, model, tokens, segments):
    """Next-token loss where run_model (possibly torch.compile'd) runs the forward and model owns the tied weight."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = run_model(tokens[:, :-1], segments[:, :-1])
    flat = hidden.reshape(-1, hidden.shape[-1])
    return _flce(flat, tokens[:, 1:].reshape(-1), model.lm_weight().to(flat.dtype))


def lm_loss(model, tokens, segments):
    return lm_loss_with(model, model, tokens, segments)


class Trainer:
    def __init__(self, model, optimizers, schedulers, loss_fn=lm_loss):
        self.model, self.optimizers, self.schedulers, self.loss_fn = model, optimizers, schedulers, loss_fn

    def step(self, *batch) -> float:
        loss = self.loss_fn(self.model, *batch)
        loss.backward()
        for o in self.optimizers:
            o.step()
            o.zero_grad(set_to_none=True)
        for s in self.schedulers:
            s.step()
        return loss.item()
```

- [ ] **Step 4: Sync and run the tests to verify they pass**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu -q'`
Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
git add vmt/train.py tests_gpu/test_train_t.py
git commit -m "GPU training: Muon and AdamW, warm-up and cosine, bfloat16 autocast, fused linear cross-entropy; fits a fixed batch, learns nothing at zero learning rate"
```

---

### Task 4: Stage 0 on the GPU

**Files:**
- Create: `scripts/stage0_gpu.py`
- Test: `tests_gpu/test_stage0_gpu.py`

**Interfaces:**
- Consumes: `Judge`, `build_optimizers`, `Trainer`, configs.
- Produces: `python scripts/stage0_gpu.py [--quick] [--out DIR] [--steps N] [--price 1.29]` writes `DIR/<stamp>/stage0_gpu.json` with `train` rows (`config, compiled, batch, seq, tokens_per_s` median and `min`/`max` over 3 repetitions, `peak_gb`, `hours_per_1b_tokens`, `dollars_per_1b_tokens`) and `verdict` rows (`config, seq, sec_per_forward`).

- [ ] **Step 1: Write the failing test**

`tests_gpu/test_stage0_gpu.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_run_writes_a_report(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/stage0_gpu.py"), "--quick", "--out", str(tmp_path), "--steps", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(next(tmp_path.glob("*/stage0_gpu.json")).read_text())
    assert rep["train"] and all(r["tokens_per_s_min"] <= r["tokens_per_s"] <= r["tokens_per_s_max"] for r in rep["train"])
    assert {"hours_per_1b_tokens", "dollars_per_1b_tokens", "peak_gb", "compiled"} <= set(rep["train"][0])
    assert rep["verdict"] and rep["verdict"][0]["sec_per_forward"] > 0
```

- [ ] **Step 2: Sync and run to verify failure**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu/test_stage0_gpu.py -q'`
Expected: FAIL (the script does not exist).

- [ ] **Step 3: Write the implementation**

`scripts/stage0_gpu.py`:
```python
"""Stage 0 on the GPU: training throughput per config, eager and torch.compile, at equal tokens per step
(65,536: 32 x 2048 or 128 x 512), three repetitions with median and spread; peak memory; verdict forward time
at 8,192 tokens; hours and dollars per 1B tokens at the given hourly price.

    python scripts/stage0_gpu.py [--quick] [--out experiments/stage0_gpu] [--steps 5] [--price 1.29]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vm.model.config import DEV, MAIN, MID, TINY  # noqa: E402
from vmt.model import Judge, n_params  # noqa: E402
from vmt.train import Trainer, build_optimizers, lm_loss_with  # noqa: E402


def train_speed(name, cfg, batch, seq, steps, compiled, price, reps=3):
    torch.manual_seed(0)
    model = Judge(cfg).cuda()
    run_model = torch.compile(model) if compiled else model
    opts, scheds = build_optimizers(model, 0.02, 3e-4, total_steps=10_000, warmup=10)
    trainer = Trainer(model, opts, scheds)
    if compiled:
        trainer.loss_fn = lambda _m, tok, seg: lm_loss_with(run_model, model, tok, seg)
    tok = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device="cuda")
    seg = torch.zeros_like(tok)
    for _ in range(2):
        trainer.step(tok, seg)  # warm up (and compile)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rates = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(steps):
            trainer.step(tok, seg)
        torch.cuda.synchronize()
        rates.append(batch * seq * steps / (time.perf_counter() - t))
    tps = statistics.median(rates)
    hours = 1e9 / tps / 3600
    return {"config": name, "params": n_params(model), "compiled": compiled, "batch": batch, "seq": seq,
            "tokens_per_s": round(tps), "tokens_per_s_min": round(min(rates)), "tokens_per_s_max": round(max(rates)),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "hours_per_1b_tokens": round(hours, 2), "dollars_per_1b_tokens": round(hours * price, 2)}


@torch.no_grad()
def verdict_speed(name, cfg, seq, reps=5):
    model = Judge(cfg).cuda().eval()
    tok = torch.randint(0, cfg.vocab_size, (1, seq), device="cuda")
    seg = torch.zeros_like(tok)
    pos = torch.tensor([seq - 1], device="cuda")

    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.verdict_logit(model(tok, seg), pos)

    run()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        run()
    torch.cuda.synchronize()
    return {"config": name, "seq": seq, "sec_per_forward": round((time.perf_counter() - t) / reps, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "stage0_gpu")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--price", type=float, default=1.29, help="dollars per GPU hour")
    a = ap.parse_args()
    if a.quick:
        plans = [("tiny", TINY, 4, 128, False), ("tiny", TINY, 4, 128, True)]
        verdicts = [("tiny", TINY, 256)]
    else:
        plans = [(n, c, b, s, comp) for n, c in (("dev", DEV), ("mid", MID), ("main", MAIN))
                 for b, s in ((32, 2048), (128, 512)) for comp in (False, True)]
        verdicts = [("dev", DEV, 8192), ("mid", MID, 8192), ("main", MAIN, 8192)]
    rows = []
    for name, cfg, batch, seq, comp in plans:
        rows.append(train_speed(name, cfg, batch, seq, a.steps, comp, a.price))
        print(json.dumps(rows[-1]), flush=True)
        torch.cuda.empty_cache()
    report = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "price_per_hour": a.price,
              "train": rows, "verdict": [verdict_speed(*v) for v in verdicts]}
    out = a.out / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "stage0_gpu.json").write_text(json.dumps(report, indent=1))
    print("\n| config | params | compiled | batch x seq | tokens/s (median) | min to max | peak GB | hours per 1B | $ per 1B |\n"
          "|---|---:|---|---|---:|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['config']} | {r['params']:,} | {'yes' if r['compiled'] else 'no'} | {r['batch']} x {r['seq']} | "
              f"{r['tokens_per_s']:,} | {r['tokens_per_s_min']:,} to {r['tokens_per_s_max']:,} | {r['peak_gb']} | "
              f"{r['hours_per_1b_tokens']} | {r['dollars_per_1b_tokens']} |")
    for v in report["verdict"]:
        print(f"verdict forward, {v['config']}, {v['seq']} tokens: {v['sec_per_forward']} s")
    print("report", out / "stage0_gpu.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Sync and run the tests to verify they pass**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -m pytest tests_gpu -q'`
Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/stage0_gpu.py tests_gpu/test_stage0_gpu.py
git commit -m "Stage 0 on the GPU: throughput per config, eager and compiled, at equal tokens per step, with hours and dollars per 1B tokens"
```

---

### Task 5: Run Stage 0 on the A100 and report

**Files:**
- Create: `experiments/stage0_gpu/<stamp>/stage0_gpu.json` (copied back from the instance), `docs/stage0_gpu_report.md`
- Modify: `docs/cost_ledger.md`

- [ ] **Step 1: Run**

Run: `sh scripts/sync.sh && ssh ubuntu@150.136.116.45 'cd verifier_model && ~/vmenv/bin/python -u scripts/stage0_gpu.py --steps 5'`, then
`rsync -az ubuntu@150.136.116.45:~/verifier_model/experiments/stage0_gpu experiments/`.
Expected: twelve training rows, three verdict rows. If a row runs out of memory, halve that row's batch and note it.

- [ ] **Step 2: Write `docs/stage0_gpu_report.md`**: the table verbatim; for each config the fastest mode, hours and dollars for 20 tokens per parameter (DEV 0.49B, MID 0.89B, MAIN 2.33B tokens); whether compiling helps; and the config decision for the user, with the remaining budget from the ledger.

- [ ] **Step 3: Update the ledger** with the hours used so far.

- [ ] **Step 4: Commit**, then stop and report to the user.
