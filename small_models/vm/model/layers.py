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
