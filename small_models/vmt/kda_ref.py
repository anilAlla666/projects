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
