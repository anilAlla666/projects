"""PHASE 2: time apply_rotary_pos_emb at decode shapes (B=1,8,32; seq=1; D=128)."""
import torch
import time
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

D = 128
Hq = 32
Hkv = 8
N_ITER = 1000

print(f"=== Phase 2: RoPE isolation timing (decode, seq=1) ===\n", flush=True)
print(f"  {'batch':>5}  {'q_shape':<24}  {'k_shape':<24}  {'us/call':>8}  {'us total Q+K':>14}", flush=True)
for B in [1, 8, 32]:
    q = torch.randn(B, Hq, 1, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Hkv, 1, D, device="cuda", dtype=torch.float16)
    cos = torch.randn(B, 1, D, device="cuda", dtype=torch.float16)
    sin = torch.randn(B, 1, D, device="cuda", dtype=torch.float16)
    # Warmup
    for _ in range(20):
        _ = apply_rotary_pos_emb(q, k, cos, sin)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_ITER):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        q_, k_ = apply_rotary_pos_emb(q, k, cos, sin)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    median = times[N_ITER // 2]
    print(f"  {B:>5}  {tuple(q.shape)!s:<24}  {tuple(k.shape)!s:<24}  {median:>8.1f}  {median:>14.1f}", flush=True)
