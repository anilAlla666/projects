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
