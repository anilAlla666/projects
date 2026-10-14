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
