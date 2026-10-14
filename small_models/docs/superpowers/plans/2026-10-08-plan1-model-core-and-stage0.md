# Plan 1: Model core and Stage 0 (feasibility) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the judge model's architecture (chunked Kimi Delta Attention, gated full attention, the 3:1 hybrid stack, segment embeddings, verdict and pointer heads), its optimiser and training step, and the Stage 0 measurement that sets every later budget.

**Architecture:** Plain MLX on Apple silicon. The delta rule is computed in chunks of 32 steps with plain MLX ops so autodiff works (mlx-lm's Metal kernel is forward-only), and it is checked against mlx-lm's step-by-step `gated_delta_ops`, an implementation we did not write. Model, optimiser and step are small focused modules; Stage 0 is a script that writes a JSON report.

**Tech Stack:** Python 3.12, MLX 0.32.3, mlx-lm 0.32.0 (reference implementation only), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-verifier-judge-design.md` (sections 4 and 6, stage 0).

**This is plan 1 of 4.** Plan 2 (data: libraries, tasks, attempts, harness, oracle, splits), plan 3 (tokenizer, pretraining, fine-tuning, adversarial rounds, calibration) and plan 4 (evaluation) are written after this plan's Stage 0 report, because their budgets come from it.

## Global Constraints

- Budget $0: no paid API, no HUD, no rented compute. Everything runs on this Mac (Apple M4, 24 GB).
- Interpreter: `/Users/anilkumar/verifier_model/.venv/bin/python` (Python 3.12.12). Run tests with `.venv/bin/pytest`.
- Pinned: `mlx==0.32.3`, `mlx-lm==0.32.0`, `numpy==2.5.3`, `pytest==9.1.1`.
- Commits authored `Anil Kumar <anil@hud.ai>` (already set in the repo's git config). No `Co-Authored-By` line.
- Delta-rule gate: per-step log decay in (−1, 0]; chunk size 32.
- Configs: DEV = d_model 384, 6 layers, 6 heads, 2 KV heads, head_dim 64, FFN 1024. MAIN = d_model 768, 12 layers, 12 heads, 4 KV heads, head_dim 64, FFN 2048. Vocabulary 32,768. Every 4th layer is full attention.
- Segments: task 0, diff 1, evidence 2, untrusted 3.
- Every correctness test that compares against a reference also has a tooth: a deliberately wrong input that must make the comparison fail.

## File structure

| File | Responsibility |
|---|---|
| `requirements.txt`, `pyproject.toml`, `.gitignore` | pins, pytest config, ignores |
| `vm/__init__.py`, `vm/model/__init__.py`, `vm/train/__init__.py` | packages |
| `vm/model/delta.py` | `chunked_delta`: the chunked gated delta rule |
| `vm/model/layers.py` | `ZRMSNorm`, `ShortConv`, `KDA`, `Attention`, `SwiGLU` |
| `vm/model/config.py` | `ModelConfig`, `DEV`, `MAIN`, `TINY`, `SEGMENTS` |
| `vm/model/model.py` | `Block`, `Judge` (hidden states, LM logits, verdict logit, pointer logits) |
| `vm/train/optim.py` | `muon_filter`, `build_optimizer` |
| `vm/train/step.py` | `lm_loss`, `make_train_step` |
| `scripts/stage0.py` | Stage 0 measurements, JSON report |
| `tests/test_delta.py`, `tests/test_layers.py`, `tests/test_model.py`, `tests/test_train.py`, `tests/test_stage0.py` | tests |
| `tests/util.py` | shared causality checker |

---

### Task 1: Scaffold and the chunked delta rule

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.gitignore`, `vm/__init__.py`, `vm/model/__init__.py`, `vm/train/__init__.py`
- Create: `vm/model/delta.py`
- Test: `tests/test_delta.py`

**Interfaces:**
- Consumes: `mlx_lm.models.gated_delta.gated_delta_ops(q, k, v, g, beta, state=None)` (reference; `g` is the multiplicative decay, state layout `[B, H, Dv, Dk]`).
- Produces: `vm.model.delta.chunked_delta(q, k, v, log_g, beta, chunk=32, state=None) -> (o, state)` with `q, k: [B, T, H, Dk]`, `v: [B, T, H, Dv]`, `log_g: [B, T, H, Dk]`, `beta: [B, T, H]`, `state: [B, H, Dk, Dv]`, `o: [B, T, H, Dv]`.

- [ ] **Step 1: Write the scaffold files**

`requirements.txt`:
```
mlx==0.32.3
mlx-lm==0.32.0
numpy==2.5.3
pytest==9.1.1
```

`pyproject.toml`:
```toml
[project]
name = "verifier-model"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
experiments/**/checkpoints/
experiments/**/*.safetensors
```

`vm/__init__.py`, `vm/model/__init__.py`, `vm/train/__init__.py`: empty files.

- [ ] **Step 2: Write the failing tests**

`tests/test_delta.py`:
```python
import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_ops

from vm.model.delta import chunked_delta


def inputs(B=2, T=128, H=3, Dk=32, Dv=48, lo=-1.0, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, H, Dk))
    k = mx.random.normal((B, T, H, Dk))
    v = mx.random.normal((B, T, H, Dv))
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * Dk**-0.5
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    log_g = mx.random.uniform(lo, 0.0, (B, T, H, Dk))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    return q, k, v, log_g, beta


def reference(q, k, v, log_g, beta, state=None):
    ref_state = None if state is None else state.swapaxes(-1, -2)
    o, s = gated_delta_ops(q, k, v, mx.exp(log_g), beta, ref_state)
    return o, s.swapaxes(-1, -2)


def maxdiff(a, b):
    return mx.abs(a - b).max().item()


@pytest.mark.parametrize("lo", [-0.1, -1.0])
def test_matches_reference(lo):
    x = inputs(lo=lo)
    o1, s1 = chunked_delta(*x)
    o2, s2 = reference(*x)
    assert maxdiff(o1, o2) < 1e-5
    assert maxdiff(s1, s2) < 1e-5


def test_padding_when_length_is_not_a_multiple_of_the_chunk():
    x = inputs(T=100)
    o1, s1 = chunked_delta(*x)
    o2, s2 = reference(*x)
    assert o1.shape == (2, 100, 3, 48)
    assert maxdiff(o1, o2) < 1e-5
    assert maxdiff(s1, s2) < 1e-5


def test_initial_state_carries_over():
    q, k, v, g, b = inputs(T=128)
    o_full, s_full = chunked_delta(q, k, v, g, b)
    o_a, s_a = chunked_delta(q[:, :64], k[:, :64], v[:, :64], g[:, :64], b[:, :64])
    o_b, s_b = chunked_delta(q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], b[:, 64:], state=s_a)
    assert maxdiff(mx.concatenate([o_a, o_b], axis=1), o_full) < 1e-5
    assert maxdiff(s_b, s_full) < 1e-5


def test_gradients_match_reference():
    x = inputs(T=64)
    f1 = lambda *a: chunked_delta(*a)[0].square().sum()  # noqa: E731
    f2 = lambda *a: reference(*a)[0].square().sum()  # noqa: E731
    g1 = mx.grad(f1, argnums=(0, 1, 2, 3, 4))(*x)
    g2 = mx.grad(f2, argnums=(0, 1, 2, 3, 4))(*x)
    for a, b in zip(g1, g2):
        assert maxdiff(a, b) < 1e-4 * max(1.0, mx.abs(b).max().item())


def test_comparison_has_teeth():
    """A 10% change in beta must break the comparison, so a pass above is not vacuous."""
    q, k, v, g, b = inputs()
    o1, _ = chunked_delta(q, k, v, g, b * 0.9)
    o2, _ = reference(q, k, v, g, b)
    assert maxdiff(o1, o2) > 1e-3
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_delta.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vm.model.delta'`

- [ ] **Step 4: Write the implementation**

`vm/model/delta.py`:
```python
"""Chunked gated delta rule with a per-channel forget gate (the recurrence of Kimi Delta Attention).

Per head, with state S of shape [Dk, Dv] and a_t = exp(log_g_t) per key channel:
    S <- diag(a_t) S;   u_t = beta_t (v_t - S^T k_t);   S <- S + k_t u_t^T;   o_t = S^T q_t
This is the recurrence of mlx_lm.models.gated_delta.gated_delta_ops (which stores S transposed), the
reference the tests compare against.

Training needs gradients and mlx-lm's Metal kernel is forward-only, so the recurrence is computed here in
chunks of C steps with plain MLX ops. Inside a chunk the triangular system (I + L) U = ... is solved by
forward substitution; only the chunk-to-chunk state pass is sequential. Exponents inside a chunk are
taken relative to the chunk midpoint, so with log_g >= -1 per step and C = 32 no exponent exceeds 16.
"""
from __future__ import annotations

import mlx.core as mx


def _unit_lower_inverse(L: mx.array) -> mx.array:
    """(I + L)^-1 for a strictly lower-triangular L of shape [..., C, C], by forward substitution."""
    C = L.shape[-1]
    eye = mx.eye(C, dtype=L.dtype)
    rows = [mx.broadcast_to(eye[0], L.shape[:-2] + (C,))]
    for t in range(1, C):
        prev = mx.stack(rows, axis=-2)
        rows.append(eye[t] - (L[..., t : t + 1, :t] @ prev)[..., 0, :])
    return mx.stack(rows, axis=-2)


def chunked_delta(q, k, v, log_g, beta, chunk: int = 32, state=None):
    """q, k: [B, T, H, Dk]; v: [B, T, H, Dv]; log_g: [B, T, H, Dk] (<= 0); beta: [B, T, H];
    state: optional [B, H, Dk, Dv]. Returns (o [B, T, H, Dv], final state [B, H, Dk, Dv])."""
    B, T0, H, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk
    pad = (-T0) % C
    if pad:
        # padded steps have k = v = 0, beta = 0 and log_g = 0: no decay and no update
        def pz(x):
            return mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))

        q, k, v, log_g, beta = pz(q), pz(k), pz(v), pz(log_g), pz(beta)
    T = T0 + pad
    n = T // C

    def split(x):
        return x.reshape(B, n, C, H, -1).transpose(0, 3, 1, 2, 4)  # [B, H, n, C, D]

    q, k, v, log_g = split(q), split(k), split(v), split(log_g)
    beta = beta.reshape(B, n, C, H).transpose(0, 3, 1, 2)[..., None]  # [B, H, n, C, 1]
    g = mx.cumsum(log_g, axis=3)
    ref = g[:, :, :, C // 2 - 1 : C // 2]
    k_late, k_early, q_late = k * mx.exp(g - ref), k * mx.exp(ref - g), q * mx.exp(g - ref)
    decay = mx.exp(g)
    idx = mx.arange(C)
    strict = idx[:, None] > idx[None, :]
    incl = idx[:, None] >= idx[None, :]
    L = mx.where(strict, beta * (k_late @ k_early.swapaxes(-1, -2)), 0.0)
    T_inv = _unit_lower_inverse(L)
    W = T_inv @ (beta * k * decay)
    U0 = T_inv @ (beta * v)
    A = mx.where(incl, q_late @ k_early.swapaxes(-1, -2), 0.0)
    q_dec = q * decay
    k_to_end = k * mx.exp(g[..., -1:, :] - g)
    decay_end = mx.exp(g[..., -1, :])[..., None]  # [B, H, n, Dk, 1]
    S = mx.zeros((B, H, Dk, Dv), dtype=q.dtype) if state is None else state
    outs = []
    for c in range(n):
        U = U0[:, :, c] - W[:, :, c] @ S
        outs.append(q_dec[:, :, c] @ S + A[:, :, c] @ U)
        S = decay_end[:, :, c] * S + k_to_end[:, :, c].swapaxes(-1, -2) @ U
    o = mx.stack(outs, axis=2).transpose(0, 2, 3, 1, 4).reshape(B, T, H, Dv)
    return o[:, :T0], S
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_delta.py -q`
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml .gitignore vm tests/test_delta.py
git commit -m "Chunked gated delta rule in plain MLX ops, matched to mlx-lm's step-by-step reference in outputs, state and gradients"
```

---

### Task 2: Layers

**Files:**
- Create: `vm/model/layers.py`, `tests/util.py`
- Test: `tests/test_layers.py`

**Interfaces:**
- Consumes: `chunked_delta` (Task 1).
- Produces (all `mlx.nn.Module`, input and output `[B, T, d]` unless stated):
  - `ZRMSNorm(dim, eps=1e-6)`: zero-centred RMSNorm, scale is `1 + weight`, weight initialised to 0.
  - `ShortConv(dim, kernel=4)`: causal depthwise convolution followed by SiLU.
  - `KDA(d_model, n_heads, head_dim, chunk=32, gate_lower=-1.0)`, with `KDA.gates(x) -> (log_g [B,T,H,Dk], beta [B,T,H])`.
  - `Attention(d_model, n_heads, n_kv_heads, head_dim)`: causal, grouped-query, QK-norm, sigmoid output gate, no positional encoding.
  - `SwiGLU(d_model, hidden)`.
  - `tests/util.py: causal_leak(fn, x, t0) -> float`, the largest change in outputs before position `t0` when inputs from `t0` on change.

- [ ] **Step 1: Write the failing tests**

`tests/util.py`:
```python
import mlx.core as mx


def causal_leak(fn, x, t0: int) -> float:
    """Largest change in outputs at positions < t0 when inputs at positions >= t0 are replaced."""
    noise = mx.random.normal(x[:, t0:].shape).astype(x.dtype)
    x2 = mx.concatenate([x[:, :t0], noise], axis=1)
    return mx.abs(fn(x)[:, :t0] - fn(x2)[:, :t0]).max().item()
```

`tests/test_layers.py`:
```python
import mlx.core as mx
import pytest

from tests.util import causal_leak
from vm.model.layers import KDA, Attention, ShortConv, SwiGLU, ZRMSNorm

B, T, D = 2, 70, 128


@pytest.fixture
def x():
    mx.random.seed(1)
    return mx.random.normal((B, T, D))


def test_causal_leak_has_teeth(x):
    """The checker must catch a function that looks ahead."""
    assert causal_leak(lambda y: mx.broadcast_to(y.mean(axis=1, keepdims=True), y.shape), x, 30) > 1e-3


@pytest.mark.parametrize("make", [
    lambda: ShortConv(D),
    lambda: KDA(D, 2, 64),
    lambda: Attention(D, 2, 1, 64),
])
def test_layers_are_causal_and_keep_shape(make, x):
    layer = make()
    mx.eval(layer.parameters())
    assert layer(x).shape == (B, T, D)
    assert causal_leak(layer, x, 30) < 1e-5


def test_norm_and_mlp_shapes(x):
    assert ZRMSNorm(D)(x).shape == (B, T, D)
    assert SwiGLU(D, 256)(x).shape == (B, T, D)


def test_zero_centred_norm_starts_as_plain_rms_norm(x):
    out = ZRMSNorm(D)(x)
    expected = x * mx.rsqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
    assert mx.abs(out - expected).max().item() < 1e-4


def test_kda_gates_stay_in_range(x):
    layer = KDA(D, 2, 64, gate_lower=-1.0)
    log_g, beta = layer.gates(x)
    assert log_g.shape == (B, T, 2, 64) and beta.shape == (B, T, 2)
    assert log_g.max().item() <= 0.0 and log_g.min().item() >= -1.0
    assert beta.min().item() >= 0.0 and beta.max().item() <= 1.0


def naive_attention(layer, x):
    Bn, Tn, _ = x.shape
    H, Hk, Dh = layer.n_heads, layer.n_kv_heads, layer.head_dim
    q = layer.q_norm(layer.q_proj(x).reshape(Bn, Tn, H, Dh)).transpose(0, 2, 1, 3)
    k = layer.k_norm(layer.k_proj(x).reshape(Bn, Tn, Hk, Dh)).transpose(0, 2, 1, 3)
    v = layer.v_proj(x).reshape(Bn, Tn, Hk, Dh).transpose(0, 2, 1, 3)
    k, v = mx.repeat(k, H // Hk, axis=1), mx.repeat(v, H // Hk, axis=1)
    s = (q @ k.swapaxes(-1, -2)) * Dh**-0.5
    mask = mx.arange(Tn)[:, None] >= mx.arange(Tn)[None, :]
    p = mx.softmax(mx.where(mask, s, -1e9), axis=-1)
    o = (p @ v).transpose(0, 2, 1, 3).reshape(Bn, Tn, -1) * mx.sigmoid(layer.g_proj(x))
    return layer.o_proj(o)


def test_attention_matches_naive_softmax(x):
    layer = Attention(D, 4, 2, 32)
    mx.eval(layer.parameters())
    assert mx.abs(layer(x) - naive_attention(layer, x)).max().item() < 1e-4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_layers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vm.model.layers'`

- [ ] **Step 3: Write the implementation**

`vm/model/layers.py`:
```python
"""Building blocks of the judge: zero-centred RMSNorm, a short causal convolution, Kimi Delta Attention,
gated full attention and a SwiGLU feed-forward."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vm.model.delta import chunked_delta


class ZRMSNorm(nn.Module):
    """RMSNorm whose scale is 1 + weight, with weight initialised to 0 (zero-centred)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class ShortConv(nn.Module):
    """Causal depthwise convolution over time, then SiLU. Input and output [B, T, dim]."""

    def __init__(self, dim: int, kernel: int = 4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def __call__(self, x):
        return nn.silu(self.conv(mx.pad(x, [(0, 0), (self.kernel - 1, 0), (0, 0)])))


class KDA(nn.Module):
    """Kimi Delta Attention: the gated delta rule with a per-channel forget gate bounded to
    (gate_lower, 0] in log space, short convolutions on q, k and v, L2-normalised q and k, and a
    per-head normalised, sigmoid-gated output."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk: int = 32, gate_lower: float = -1.0):
        super().__init__()
        self.n_heads, self.head_dim, self.chunk, self.gate_lower = n_heads, head_dim, chunk, gate_lower
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.q_conv, self.k_conv, self.v_conv = ShortConv(inner), ShortConv(inner), ShortConv(inner)
        self.f_down = nn.Linear(d_model, head_dim, bias=False)
        self.f_up = nn.Linear(head_dim, inner, bias=False)
        self.A_log = mx.zeros((n_heads,))
        self.dt_bias = mx.zeros((inner,))
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        self.g_proj = nn.Linear(d_model, inner, bias=False)
        self.o_norm = ZRMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, d_model, bias=False)

    def gates(self, x):
        B, T, _ = x.shape
        f = (self.f_up(self.f_down(x)) + self.dt_bias).reshape(B, T, self.n_heads, self.head_dim)
        log_g = self.gate_lower * mx.sigmoid(mx.exp(self.A_log)[:, None] * f)
        beta = mx.sigmoid(self.b_proj(x))
        return log_g, beta

    def __call__(self, x):
        B, T, _ = x.shape
        H, Dk = self.n_heads, self.head_dim
        q = self.q_conv(self.q_proj(x)).reshape(B, T, H, Dk)
        k = self.k_conv(self.k_proj(x)).reshape(B, T, H, Dk)
        v = self.v_conv(self.v_proj(x)).reshape(B, T, H, Dk)
        q = q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6) * Dk**-0.5
        k = k * mx.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)
        log_g, beta = self.gates(x)
        f32 = mx.float32
        o, _ = chunked_delta(q.astype(f32), k.astype(f32), v.astype(f32), log_g.astype(f32), beta.astype(f32), self.chunk)
        o = self.o_norm(o.astype(x.dtype)).reshape(B, T, H * Dk) * mx.sigmoid(self.g_proj(x))
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

    def __call__(self, x):
        B, T, _ = x.shape
        H, Hk, Dh = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(B, T, H, Dh)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, T, Hk, Dh)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, Hk, Dh).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=Dh**-0.5, mask="causal")
        o = o.transpose(0, 2, 1, 3).reshape(B, T, H * Dh) * mx.sigmoid(self.g_proj(x))
        return self.o_proj(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_layers.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add vm/model/layers.py tests/util.py tests/test_layers.py
git commit -m "Layers: zero-centred RMSNorm, short causal convolution, Kimi Delta Attention, gated full attention, SwiGLU; causality checked with a checker that is shown to catch a leak"
```

---

### Task 3: Config and the Judge model

**Files:**
- Create: `vm/model/config.py`, `vm/model/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `ZRMSNorm`, `KDA`, `Attention`, `SwiGLU` (Task 2).
- Produces:
  - `vm.model.config.ModelConfig` (frozen dataclass: `vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim, ffn_hidden, attn_period, chunk, gate_lower, n_segments`) with `is_attention(i) -> bool`; constants `DEV`, `MAIN`, `TINY`, `SEGMENTS = {"task": 0, "diff": 1, "evidence": 2, "untrusted": 3}`.
  - `vm.model.model.Judge(cfg)`: `__call__(tokens [B,T] int32, segments [B,T] int32) -> hidden [B,T,d]`; `lm_logits(hidden) -> [B,T,V]`; `verdict_logit(hidden, pos [B]) -> [B]`; `pointer_logits(hidden, pos [B], line_mask [B,T] bool) -> [B,T]` (masked positions get −1e9).
  - `vm.model.model.n_params(module) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import mlx.core as mx

from tests.util import causal_leak
from vm.model.config import DEV, MAIN, TINY
from vm.model.model import Judge, n_params


def test_parameter_counts_match_the_spec():
    assert 20e6 <= n_params(Judge(DEV)) <= 30e6
    assert 100e6 <= n_params(Judge(MAIN)) <= 125e6


def test_three_delta_layers_to_one_attention_layer():
    assert [i for i in range(MAIN.n_layers) if MAIN.is_attention(i)] == [3, 7, 11]
    assert [i for i in range(DEV.n_layers) if DEV.is_attention(i)] == [3]


def batch(B=2, T=40, seed=0):
    mx.random.seed(seed)
    tok = mx.random.randint(0, TINY.vocab_size, (B, T))
    seg = mx.random.randint(0, 4, (B, T))
    return tok, seg


def test_heads_have_the_right_shapes():
    m = Judge(TINY)
    tok, seg = batch()
    h = m(tok, seg)
    assert h.shape == (2, 40, TINY.d_model)
    assert m.lm_logits(h).shape == (2, 40, TINY.vocab_size)
    pos = mx.array([39, 20])
    assert m.verdict_logit(h, pos).shape == (2,)
    line_mask = mx.zeros((2, 40), dtype=mx.bool_)
    line_mask[:, 5] = True
    p = m.pointer_logits(h, pos, line_mask)
    assert p.shape == (2, 40)
    assert p[0, 6].item() == -1e9 and p[0, 5].item() > -1e9


def test_model_is_causal():
    m = Judge(TINY)
    tok, seg = batch()
    emb = lambda t: m(t, seg)  # noqa: E731
    noise = mx.random.randint(0, TINY.vocab_size, (2, 20))
    h1 = emb(tok)
    h2 = emb(mx.concatenate([tok[:, :20], noise], axis=1))
    assert mx.abs(h1[:, :20] - h2[:, :20]).max().item() < 1e-5


def test_segment_ids_change_the_representation():
    """The trust tag must reach the model: same tokens, different segments, different states."""
    m = Judge(TINY)
    tok, seg = batch()
    h1 = m(tok, mx.zeros_like(seg))
    h2 = m(tok, mx.full(seg.shape, 3, dtype=seg.dtype))
    assert mx.abs(h1 - h2).max().item() > 1e-3


def test_language_model_head_is_tied():
    names = [k for k, _ in __import__("mlx.utils", fromlist=["x"]).tree_flatten(Judge(TINY).parameters())]
    assert not any(k.startswith("lm_head") for k in names)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vm.model.config'`

- [ ] **Step 3: Write the implementation**

`vm/model/config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

SEGMENTS = {"task": 0, "diff": 1, "evidence": 2, "untrusted": 3}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_hidden: int = 2048
    attn_period: int = 4  # every attn_period-th layer is full attention: three delta layers to one
    chunk: int = 32
    gate_lower: float = -1.0
    n_segments: int = 4

    def is_attention(self, i: int) -> bool:
        return (i + 1) % self.attn_period == 0


MAIN = ModelConfig()
DEV = ModelConfig(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, ffn_hidden=1024)
TINY = ModelConfig(vocab_size=512, d_model=64, n_layers=4, n_heads=1, n_kv_heads=1, head_dim=64, ffn_hidden=128)
```

`vm/model/model.py`:
```python
"""The judge: a hybrid stack of Kimi Delta Attention and full attention with segment embeddings,
a tied language-model head (pretraining), a verdict head and an evidence-pointer head."""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from vm.model.config import ModelConfig
from vm.model.layers import KDA, Attention, SwiGLU, ZRMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, attention: bool):
        super().__init__()
        self.norm1 = ZRMSNorm(cfg.d_model)
        self.mixer = (Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) if attention
                      else KDA(cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.chunk, cfg.gate_lower))
        self.norm2 = ZRMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def __call__(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Judge(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.segment = nn.Embedding(cfg.n_segments, cfg.d_model)
        self.layers = [Block(cfg, cfg.is_attention(i)) for i in range(cfg.n_layers)]
        self.norm = ZRMSNorm(cfg.d_model)
        self.verdict_head = nn.Linear(cfg.d_model, 1)
        self.pointer_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, tokens, segments):
        x = self.embed(tokens) + self.segment(segments)
        for block in self.layers:
            x = block(x)
        return self.norm(x)

    def lm_logits(self, hidden):
        return self.embed.as_linear(hidden)

    def _at(self, hidden, pos):
        return hidden[mx.arange(hidden.shape[0]), pos]

    def verdict_logit(self, hidden, pos):
        return self.verdict_head(self._at(hidden, pos))[:, 0]

    def pointer_logits(self, hidden, pos, line_mask):
        query = self.pointer_query(self._at(hidden, pos))  # [B, d]
        scores = (hidden @ query[:, :, None])[..., 0] / math.sqrt(self.cfg.d_model)
        return mx.where(line_mask, scores, -1e9)


def n_params(module: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(module.parameters()))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_model.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add vm/model/config.py vm/model/model.py tests/test_model.py
git commit -m "Judge model: 3:1 hybrid of Kimi Delta Attention and gated full attention, segment embeddings, tied LM head, verdict and evidence-pointer heads; DEV about 24M, MAIN about 116M parameters"
```

---

### Task 4: Optimiser and training step

**Files:**
- Create: `vm/train/optim.py`, `vm/train/step.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `Judge`, `TINY` (Task 3).
- Produces:
  - `vm.train.optim.muon_filter(path: str, weight: mx.array) -> bool`: true for 2D weights outside the embeddings and heads.
  - `vm.train.optim.build_optimizer(lr_muon: float, lr_adam: float, total_steps: int, warmup: int, weight_decay: float = 0.01) -> mlx.optimizers.MultiOptimizer` (linear warm-up from 0, cosine decay to 10%).
  - `vm.train.step.lm_loss(model, tokens, segments) -> mx.array` (next-token cross-entropy).
  - `vm.train.step.make_train_step(model, optimizer, loss_fn=lm_loss, compile=True) -> (step, state)`; `step(*batch) -> loss`; the caller runs `mx.eval(loss, state)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
import mlx.core as mx
from mlx.utils import tree_flatten

from vm.model.config import TINY
from vm.model.model import Judge
from vm.train.optim import build_optimizer, muon_filter
from vm.train.step import make_train_step


def fixed_batch():
    mx.random.seed(0)
    tok = mx.random.randint(0, TINY.vocab_size, (4, 65))
    return tok, mx.zeros((4, 65), dtype=mx.int32)


def train(lr_muon, lr_adam, steps=60, compile=True):
    mx.random.seed(0)
    model = Judge(TINY)
    mx.eval(model.parameters())
    opt = build_optimizer(lr_muon, lr_adam, total_steps=steps, warmup=5)
    step, state = make_train_step(model, opt, compile=compile)
    tok, seg = fixed_batch()
    losses = []
    for _ in range(steps):
        loss = step(tok, seg)
        mx.eval(loss, state)
        losses.append(loss.item())
    return losses


def test_muon_gets_exactly_the_hidden_weight_matrices():
    params = dict(tree_flatten(Judge(TINY).parameters()))
    muon = {k for k, v in params.items() if muon_filter(k, v)}
    assert "layers.0.mixer.q_proj.weight" in muon
    assert "layers.0.mlp.down.weight" in muon
    assert not any(k.startswith(("embed", "segment", "verdict_head", "pointer_query")) for k in muon)
    assert all(params[k].ndim == 2 for k in muon)


def test_training_fits_a_fixed_batch():
    losses = train(0.02, 3e-3)
    assert losses[-1] < 0.1 * losses[0]


def test_zero_learning_rate_learns_nothing():
    """Tooth for the test above: with no learning rate the loss must not move."""
    losses = train(0.0, 0.0, steps=10)
    assert abs(losses[-1] - losses[0]) < 1e-5


def test_compiled_step_matches_eager():
    a = train(0.02, 3e-3, steps=10, compile=False)
    b = train(0.02, 3e-3, steps=10, compile=True)
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vm.train.optim'`

- [ ] **Step 3: Write the implementation**

`vm/train/optim.py`:
```python
"""Muon for hidden weight matrices, AdamW for everything else (embeddings, heads, norms, gates,
convolutions), both with linear warm-up and cosine decay to 10%: the Moonlight recipe."""
from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim

_NOT_MUON = ("embed", "segment", "verdict_head", "pointer_query")


def muon_filter(path: str, weight: mx.array) -> bool:
    return weight.ndim == 2 and not path.startswith(_NOT_MUON)


def _schedule(base: float, total_steps: int, warmup: int):
    if warmup < 1 or total_steps <= warmup:
        raise ValueError("need 1 <= warmup < total_steps")
    return optim.join_schedules(
        [optim.linear_schedule(0.0, base, warmup), optim.cosine_decay(base, total_steps - warmup, base * 0.1)],
        [warmup],
    )


def build_optimizer(lr_muon: float, lr_adam: float, total_steps: int, warmup: int,
                    weight_decay: float = 0.01) -> optim.MultiOptimizer:
    muon = optim.Muon(learning_rate=_schedule(lr_muon, total_steps, warmup), weight_decay=weight_decay)
    adam = optim.AdamW(learning_rate=_schedule(lr_adam, total_steps, warmup), weight_decay=weight_decay)
    return optim.MultiOptimizer([muon, adam], [muon_filter])
```

`vm/train/step.py`:
```python
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def lm_loss(model, tokens, segments):
    hidden = model(tokens[:, :-1], segments[:, :-1])
    return nn.losses.cross_entropy(model.lm_logits(hidden), tokens[:, 1:]).mean()


def make_train_step(model, optimizer, loss_fn=lm_loss, compile: bool = True):
    """Returns (step, state). Call loss = step(*batch), then mx.eval(loss, state)."""
    optimizer.init(model.trainable_parameters())
    value_and_grad = nn.value_and_grad(model, loss_fn)

    def step(*batch):
        loss, grads = value_and_grad(model, *batch)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state, mx.random.state]
    if compile:
        step = mx.compile(step, inputs=state, outputs=state)
    return step, state
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_train.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add vm/train/optim.py vm/train/step.py tests/test_train.py
git commit -m "Training step: Muon for hidden matrices, AdamW for the rest, warm-up and cosine decay; fits a fixed batch, learns nothing at zero learning rate, compiled equals eager"
```

---

### Task 5: Stage 0 measurement

**Files:**
- Create: `scripts/stage0.py`
- Test: `tests/test_stage0.py`

**Interfaces:**
- Consumes: `chunked_delta`, `Judge`, `DEV`, `MAIN`, `TINY`, `build_optimizer`, `make_train_step`, mlx-lm's `gated_delta_ops`.
- Produces: `python scripts/stage0.py [--quick] [--out DIR] [--steps N]` writes `DIR/<UTC stamp>/stage0.json` with keys `numeric_check` (`max_out_diff`, `max_state_diff` at gate floors −0.1 and −1.0, production head shape), `train` (one row per config and dtype: `config, dtype, batch, seq, sec_per_step, tokens_per_s, peak_gb, days_per_1b_tokens`), `verdict` (one row per config: `seq, sec_per_forward`), plus `mlx_version`, `machine`. Prints a Markdown table.

- [ ] **Step 1: Write the failing test**

`tests/test_stage0.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_run_writes_a_complete_report(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/stage0.py"), "--quick", "--out", str(tmp_path), "--steps", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    reports = list(tmp_path.glob("*/stage0.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    for check in rep["numeric_check"]:
        assert check["max_out_diff"] < 1e-4 and check["max_state_diff"] < 1e-4
    assert rep["train"] and all(row["tokens_per_s"] > 0 for row in rep["train"])
    assert rep["verdict"] and all(row["sec_per_forward"] > 0 for row in rep["verdict"])
    assert {"config", "dtype", "batch", "seq", "peak_gb", "days_per_1b_tokens"} <= set(rep["train"][0])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_stage0.py -q`
Expected: FAIL (`returncode` 2: `can't open file .../scripts/stage0.py`)

- [ ] **Step 3: Write the implementation**

`scripts/stage0.py`:
```python
"""Stage 0: is the architecture trainable on this Mac, and how fast?

1. Numeric check: chunked_delta against mlx-lm's step-by-step gated_delta_ops at the production head shape,
   at a mild gate floor (-0.1) and the hard floor (-1.0).
2. Training throughput: compiled train steps for each config and dtype; tokens per second, peak memory,
   and the days 1B tokens would take at that rate.
3. Verdict time: one forward pass at batch 1 over the full 8,192-token context (model only; the harness
   that gathers evidence is timed in plan 2).

    python scripts/stage0.py [--quick] [--out experiments/stage0] [--steps 3]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_ops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vm.model.config import DEV, MAIN, TINY  # noqa: E402
from vm.model.delta import chunked_delta  # noqa: E402
from vm.model.model import Judge  # noqa: E402
from vm.train.optim import build_optimizer  # noqa: E402
from vm.train.step import make_train_step  # noqa: E402

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}


def numeric_check(T: int, H: int, D: int) -> list[dict]:
    rows = []
    for lo in (-0.1, -1.0):
        mx.random.seed(0)
        q = mx.random.normal((1, T, H, D))
        k = mx.random.normal((1, T, H, D))
        v = mx.random.normal((1, T, H, D))
        q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * D**-0.5
        k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
        g = mx.random.uniform(lo, 0.0, (1, T, H, D))
        b = mx.sigmoid(mx.random.normal((1, T, H)))
        o1, s1 = chunked_delta(q, k, v, g, b)
        o2, s2 = gated_delta_ops(q, k, v, mx.exp(g), b)
        rows.append({"gate_floor": lo, "T": T, "H": H, "D": D,
                     "max_out_diff": mx.abs(o1 - o2).max().item(),
                     "max_state_diff": mx.abs(s1 - s2.swapaxes(-1, -2)).max().item()})
    return rows


def train_speed(name, cfg, dtype, batch, seq, steps) -> dict:
    mx.random.seed(0)
    model = Judge(cfg)
    model.set_dtype(DTYPES[dtype])
    mx.eval(model.parameters())
    opt = build_optimizer(0.02, 3e-4, total_steps=1000, warmup=10)
    step, state = make_train_step(model, opt, compile=True)
    tok = mx.random.randint(0, cfg.vocab_size, (batch, seq + 1))
    seg = mx.zeros((batch, seq + 1), dtype=mx.int32)
    mx.eval(step(tok, seg), state)  # compile and warm up
    mx.reset_peak_memory()
    t = time.perf_counter()
    for _ in range(steps):
        loss = step(tok, seg)
        mx.eval(loss, state)
    sec = (time.perf_counter() - t) / steps
    tps = batch * seq / sec
    return {"config": name, "dtype": dtype, "batch": batch, "seq": seq, "sec_per_step": round(sec, 3),
            "tokens_per_s": round(tps, 1), "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "days_per_1b_tokens": round(1e9 / tps / 86400, 2)}


def verdict_speed(name, cfg, seq, reps=3) -> dict:
    model = Judge(cfg)
    model.eval()
    mx.eval(model.parameters())
    tok = mx.random.randint(0, cfg.vocab_size, (1, seq))
    seg = mx.zeros((1, seq), dtype=mx.int32)
    pos = mx.array([seq - 1])
    run = lambda: model.verdict_logit(model(tok, seg), pos)  # noqa: E731
    mx.eval(run())
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(run())
    return {"config": name, "seq": seq, "sec_per_forward": round((time.perf_counter() - t) / reps, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny config and short sequences (for the test)")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "stage0")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    if a.quick:
        check = numeric_check(T=64, H=2, D=64)
        plans = [("tiny", TINY, "float32", 2, 128)]
        verdicts = [("tiny", TINY, 256)]
    else:
        check = numeric_check(T=512, H=12, D=64)
        plans = [(n, c, d, b, 2048) for n, c, b in (("dev", DEV, 8), ("main", MAIN, 2)) for d in ("float32", "bfloat16")]
        verdicts = [("dev", DEV, 8192), ("main", MAIN, 8192)]
    rows = []
    for name, cfg, dtype, batch, seq in plans:
        rows.append(train_speed(name, cfg, dtype, batch, seq, a.steps))
        print(json.dumps(rows[-1]), flush=True)
    report = {"mlx_version": mx.__version__, "machine": platform.platform(), "numeric_check": check,
              "train": rows, "verdict": [verdict_speed(n, c, s) for n, c, s in verdicts]}
    out = a.out / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "stage0.json").write_text(json.dumps(report, indent=1))
    print("\n| config | dtype | batch x seq | s/step | tokens/s | peak GB | days per 1B tokens |\n|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['config']} | {r['dtype']} | {r['batch']} x {r['seq']} | {r['sec_per_step']} | {r['tokens_per_s']:,.0f} | "
              f"{r['peak_gb']} | {r['days_per_1b_tokens']} |")
    for v in report["verdict"]:
        print(f"verdict forward, {v['config']}, {v['seq']} tokens: {v['sec_per_forward']} s")
    print("numeric check max diffs:", [(c["gate_floor"], c["max_out_diff"], c["max_state_diff"]) for c in check])
    print("report", out / "stage0.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_stage0.py -q`
Expected: `1 passed`

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: `25 passed`

- [ ] **Step 6: Commit**

```bash
git add scripts/stage0.py tests/test_stage0.py
git commit -m "Stage 0 script: numeric check against mlx-lm at production head shape, training throughput per config and dtype, verdict forward time"
```

---

### Task 6: Run Stage 0 and report

**Files:**
- Create: `experiments/stage0/<stamp>/stage0.json` (written by the script; committed)
- Create: `docs/stage0_report.md`

**Interfaces:**
- Consumes: `scripts/stage0.py` (Task 5).
- Produces: the measured numbers plans 2 to 4 use, and a decision for the user.

- [ ] **Step 1: Run the measurement with nothing else using the GPU**

Run: `.venv/bin/python -u scripts/stage0.py --steps 3`
Expected: a table with four training rows (dev and main, float32 and bfloat16), two verdict rows, numeric-check diffs under 1e-4, and the report path. If a main row runs out of memory, rerun with that row's batch set to 1 and note it in the report.

- [ ] **Step 2: Write `docs/stage0_report.md`**

Copy the printed table verbatim. Under it, write:
1. The numeric check's maximum differences.
2. For the faster dtype of each config, the days 1B tokens would take, and the days a 4-day pretraining window allows in tokens (`tokens_per_s × 345,600`).
3. Whether bfloat16 training is usable: compare the first 50 steps' loss on a fixed batch in float32 and bfloat16 for the DEV config (`train(...)` style loop from `tests/test_train.py`, with `model.set_dtype`), and report both curves' final values.
4. Three options for the main config, each with its token budget at the measured rate: keep MAIN; shrink to d_model 512 (8 layers); or pretrain DEV only. This is a DECISION NEEDED for the user; do not choose.

- [ ] **Step 3: Commit**

```bash
git add experiments/stage0 docs/stage0_report.md
git commit -m "Stage 0 measured on this Mac: numeric check, training throughput, verdict time; the main-config decision is left to the user"
```

- [ ] **Step 4: Stop and report to the user** with the table and the decision. Plans 2 to 4 are written only after the user decides.
