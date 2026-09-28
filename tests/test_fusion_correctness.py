#!/usr/bin/env python3
"""Verify fused NVRTC kernels match PyTorch reference within 0.1% relative error."""
import os, ctypes
import torch
import torch.nn as nn

ROOT = "/home/ubuntu/op31-prod-fix"
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"))
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_fusion_kernels_init.restype = ctypes.c_int
rt.cipher_fusion_kernels_enabled.restype = ctypes.c_int

rt.cipher_fused_rmsnorm.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
rt.cipher_fused_rmsnorm.restype = ctypes.c_int

rt.cipher_fused_silu_mul.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_void_p]
rt.cipher_fused_silu_mul.restype = ctypes.c_int

rt.cipher_fused_residual_add.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_void_p]
rt.cipher_fused_residual_add.restype = ctypes.c_int

rt.cipher_substitute_v2_init()
rt.cipher_fusion_kernels_init()
print(f"fusion enabled: {rt.cipher_fusion_kernels_enabled()}")


def stats(label, ours, ref):
    abs_err = (ours.float() - ref.float()).abs()
    rel_err = abs_err / (ref.float().abs() + 1e-6)
    return (f"  [{label}] rel_err: mean={rel_err.mean():.6f} "
            f"median={rel_err.median():.6f} max={rel_err.max():.6f}  "
            f"abs_err mean={abs_err.mean():.6f}")


# ── Pattern 1: RMSNorm vs HF LlamaRMSNorm ───────────────────────────────────
print("\n=== Pattern 1: Fused RMSNorm ===")
torch.manual_seed(0)
B, T, D = 1, 1, 4096
x_ref = torch.randn(B*T, D, dtype=torch.float16, device='cuda')
w_ref = torch.randn(D, dtype=torch.float16, device='cuda') * 0.5 + 1.0
eps = 1e-6


def hf_rmsnorm(x, weight, eps):
    """Mistral / Llama RMSNorm in fp32 then cast back, matching HF impl."""
    in_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight.to(torch.float32) * x).to(in_dtype)


y_ref  = hf_rmsnorm(x_ref, w_ref, eps)
y_ours = torch.empty_like(x_ref)
rc = rt.cipher_fused_rmsnorm(
    x_ref.data_ptr(), w_ref.data_ptr(), y_ours.data_ptr(),
    B*T, D, eps, None)
torch.cuda.synchronize()
print(f"  rc={rc}  (1=success)")
print(stats("RMSNorm[B=1,T=1,D=4096]", y_ours, y_ref))

# Higher-rank batch
B, T, D = 4, 32, 4096
x_ref = torch.randn(B*T, D, dtype=torch.float16, device='cuda')
y_ref  = hf_rmsnorm(x_ref, w_ref, eps)
y_ours = torch.empty_like(x_ref)
rt.cipher_fused_rmsnorm(x_ref.data_ptr(), w_ref.data_ptr(), y_ours.data_ptr(),
                         B*T, D, eps, None)
torch.cuda.synchronize()
print(stats(f"RMSNorm[B={B},T={T},D={D}]", y_ours, y_ref))

# ── Pattern 2: SiLU(gate) * up ──────────────────────────────────────────────
print("\n=== Pattern 2: Fused SiLU·Mul ===")
B, T, D = 1, 1, 14336    # Mistral up_proj output
gate = torch.randn(B*T, D, dtype=torch.float16, device='cuda')
up   = torch.randn(B*T, D, dtype=torch.float16, device='cuda')
y_ref = nn.functional.silu(gate) * up
y_ours = torch.empty_like(gate)
rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), y_ours.data_ptr(),
                          gate.numel(), None)
torch.cuda.synchronize()
print(stats(f"SiLU·Mul[numel={gate.numel()}]", y_ours, y_ref))

# ── Pattern 3: Residual add ────────────────────────────────────────────────
print("\n=== Pattern 3: Fused Residual Add ===")
B, T, D = 1, 1, 4096
x = torch.randn(B*T, D, dtype=torch.float16, device='cuda')
r = torch.randn(B*T, D, dtype=torch.float16, device='cuda')
y_ref  = x + r
y_ours = torch.empty_like(x)
rt.cipher_fused_residual_add(x.data_ptr(), r.data_ptr(), y_ours.data_ptr(),
                              x.numel(), None)
torch.cuda.synchronize()
print(stats(f"ResAdd[numel={x.numel()}]", y_ours, y_ref))


# ── Throughput timing ──────────────────────────────────────────────────────
import time

print("\n=== Throughput vs PyTorch (decode-shape: B=1, T=1, D=4096) ===")

# RMSNorm
x = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
w = torch.randn(4096, dtype=torch.float16, device='cuda')
y = torch.empty_like(x)
N = 1000
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    _ = hf_rmsnorm(x, w, 1e-6)
torch.cuda.synchronize()
t_torch_rms = (time.perf_counter() - t0) / N * 1e6

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    rt.cipher_fused_rmsnorm(x.data_ptr(), w.data_ptr(), y.data_ptr(), 1, 4096, 1e-6, None)
torch.cuda.synchronize()
t_ours_rms = (time.perf_counter() - t0) / N * 1e6

print(f"  RMSNorm:   PyTorch={t_torch_rms:.2f}us  CIPHER={t_ours_rms:.2f}us  "
      f"({t_torch_rms/t_ours_rms:.2f}x)")

# SiLU_mul
gate = torch.randn(1, 14336, dtype=torch.float16, device='cuda')
up   = torch.randn(1, 14336, dtype=torch.float16, device='cuda')
out  = torch.empty_like(gate)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    _ = nn.functional.silu(gate) * up
torch.cuda.synchronize()
t_torch_silu = (time.perf_counter() - t0) / N * 1e6

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    rt.cipher_fused_silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(), gate.numel(), None)
torch.cuda.synchronize()
t_ours_silu = (time.perf_counter() - t0) / N * 1e6

print(f"  SiLU·Mul:  PyTorch={t_torch_silu:.2f}us  CIPHER={t_ours_silu:.2f}us  "
      f"({t_torch_silu/t_ours_silu:.2f}x)")
