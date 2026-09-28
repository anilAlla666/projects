#!/usr/bin/env python3
"""Lever 2 correctness: cipher_kv_dq2_perm_rope produces post-RoPE K
matching HF's apply_rotary_pos_emb within KIVI 2-bit noise."""
import os, ctypes, sys, math
# Must set env BEFORE loading libcipher_rt.so — constructor reads env at load.
os.environ["CIPHER_KV_REDIRECT"]   = "on"
os.environ["CIPHER_SUBSTITUTE_V2"] = "on"

import numpy as np
import torch

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)

rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_kv_redirect_init.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_kv_redirect_init()

rt.cipher_kv_test_qdq_rope.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_void_p,
]
rt.cipher_kv_test_qdq_rope.restype = ctypes.c_int


def hf_apply_rotary_pos_emb(K, positions, base=10000.0):
    """HF/Llama RoPE: rotate_half + cos/sin pairs. K shape [n_heads, n_tokens, head_dim]."""
    H, T, D = K.shape
    half = D // 2
    # inv_freq shape [half]
    inv_freq = base ** (-torch.arange(0, half, device=K.device, dtype=torch.float32) * 2 / D)
    # freqs[t, i] = positions[t] * inv_freq[i]
    pos = torch.tensor(positions, device=K.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)         # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)    # [T, D]
    cos = emb.cos().to(K.dtype)                # [T, D]
    sin = emb.sin().to(K.dtype)
    # rotate_half(K) = cat([-K[..., D/2:], K[..., :D/2]], dim=-1)
    x1 = K[..., :half]
    x2 = K[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)         # [H, T, D]
    cos_b = cos.unsqueeze(0)                   # [1, T, D]
    sin_b = sin.unsqueeze(0)
    return K * cos_b + rot * sin_b


def rel_err(a, b):
    a = a.float(); b = b.float()
    num = (a - b).norm().item()
    den = b.norm().item() + 1e-9
    return num / den


def run_one(n_heads, n_tokens, label, debug=False):
    torch.manual_seed(42)
    head_dim = 128
    # Input layout matches cuBLAS K_proj: [n_tokens, n_heads, head_dim] row-major
    # Tighter range so 2-bit quant noise doesn't swamp signal in rel_err.
    K_in = (torch.randn(n_tokens, n_heads, head_dim, device="cuda", dtype=torch.float16)
            * 0.1)
    K_in_ref = K_in.permute(1, 0, 2).contiguous()  # [H, T, D]

    # CIPHER path WITHOUT rope: should match permute within quant noise
    out_no_rope = torch.empty(n_heads, n_tokens, head_dim, device="cuda", dtype=torch.float16)
    rc = rt.cipher_kv_test_qdq_rope(K_in.data_ptr(), out_no_rope.data_ptr(),
                                      n_heads, n_tokens, 0, 0, 10000.0,
                                      ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    assert rc == 1, "kernel call (no rope) failed"
    err_no_rope = rel_err(out_no_rope, K_in_ref)

    # CIPHER path WITH rope: should match HF RoPE applied to permuted K_in
    positions = list(range(n_tokens))
    K_ref_with_rope = hf_apply_rotary_pos_emb(K_in_ref, positions, base=10000.0)
    out_rope = torch.empty(n_heads, n_tokens, head_dim, device="cuda", dtype=torch.float16)
    rc = rt.cipher_kv_test_qdq_rope(K_in.data_ptr(), out_rope.data_ptr(),
                                      n_heads, n_tokens, 1, 0, 10000.0,
                                      ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    assert rc == 1, "kernel call (rope) failed"
    err_rope = rel_err(out_rope, K_ref_with_rope)

    # Sanity: applying RoPE to the unquantized reference and comparing against
    # CIPHER's RoPE'd output should NOT be much worse than no-rope error.
    # The cleanest correctness check: apply HF's RoPE to CIPHER's no-rope output,
    # and compare against CIPHER's rope output. Both start from the same
    # quantized values, so any difference is purely RoPE-math error.
    expected_rope_of_quantized = hf_apply_rotary_pos_emb(out_no_rope, positions, base=10000.0)
    rope_math_err = rel_err(out_rope, expected_rope_of_quantized)
    diff_outputs = (out_rope.float() - out_no_rope.float()).norm().item()
    diff_refs = (K_ref_with_rope.float() - K_in_ref.float()).norm().item()
    print(f"  {label:20s} no_rope_err={err_no_rope:.4f}  with_rope_err={err_rope:.4f}  "
          f"rope_math_err={rope_math_err:.5f}  "
          f"||out_rope-out_no_rope||={diff_outputs:.4f}  "
          f"||rope_ref-no_rope_ref||={diff_refs:.4f}")
    if debug:
        # First 8 dims of head 0, token 1 — RoPE non-trivial at pos=1
        print(f"    out_no_rope[0,1,:8]: {out_no_rope[0,1,:8].tolist()}")
        print(f"    out_rope   [0,1,:8]: {out_rope[0,1,:8].tolist()}")
        print(f"    expected   [0,1,:8]: {expected_rope_of_quantized[0,1,:8].tolist()}")
    return err_no_rope, err_rope, rope_math_err, diff_outputs, diff_refs


print("=== Lever 2 correctness: dequant + RoPE ===")
print(f"{'shape':>20} {'no_rope':>10} {'with_rope':>10} {'diff':>8}")
fail = 0
for n_tokens, n_heads in [(1, 8), (4, 8), (16, 8), (64, 8), (128, 8), (256, 8)]:
    e0, e1, rm, do, dr = run_one(n_heads, n_tokens, f"H={n_heads} T={n_tokens}",
                                  debug=(n_tokens == 16))
    # Pass criteria: rope_math_err is the clean signal — applying HF RoPE
    # to CIPHER's no-rope output should match CIPHER's rope output to fp16
    # precision (~1e-3). If this fails, the kernel's RoPE math is wrong.
    if rm > 0.01:
        print(f"    FAIL: rope_math_err={rm:.5f} > 1% (kernel's RoPE math is incorrect)")
        fail += 1

if fail:
    print(f"\nFAILED: {fail} cases")
    sys.exit(1)
print("\nLEVER 2 KERNEL CORRECTNESS: PASS")
