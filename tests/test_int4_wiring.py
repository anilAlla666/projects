"""Check if our kernel can substitute for nn.Linear's GEMM cleanly.
nn.Linear stores weight as (out_features, in_features) and computes
  Y = X @ W.T  where X has shape (..., in_features).
Our cipher_int4_gemm computes C = A @ B for B of shape (K, N).
To match nn.Linear: K=in_features, N=out_features, B=W.T.
But W.T isn't contiguous unless we re-quantize W.T.
"""
import os, ctypes, time
import torch
import warnings
warnings.filterwarnings("ignore")
ROOT = "/home/ubuntu/op31-prod-fix"
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"))
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup.restype = ctypes.c_int
rt.cipher_weight_compress_int4_gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
rt.cipher_weight_compress_int4_gemm.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()

# Use a Mistral q_proj weight: shape (4096, 4096). For Y = X @ W.T:
#   X shape: (M, 4096) — input activation
#   Y shape: (M, 4096) — output
# Our kernel computes A @ B with B(K,N). For nn.Linear's W.T:
#   K = in_features = 4096
#   N = out_features = 4096
#   B = W.T which is (4096, 4096) but non-contiguous.
# We must quantize a CONTIGUOUS copy of W.T = W's transpose.

torch.manual_seed(0)
W = torch.randn(4096, 4096, dtype=torch.float16, device='cuda') * 0.02
W_T = W.t().contiguous()    # (4096, 4096) but transposed view materialized

K, N = W_T.shape   # K = original out (4096), N = original in (4096) — actually same here

for _ in range(1001):
    rt.cipher_weight_compress_observe(W_T.data_ptr(), K * N * 2)
rt.cipher_weight_compress_quantize(W_T.data_ptr(), K, N)

b4 = ctypes.c_void_p(); bs = ctypes.c_void_p(); br = ctypes.c_int(); bc = ctypes.c_int()
rt.cipher_weight_compress_lookup(W_T.data_ptr(),
    ctypes.byref(b4), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))

# Run as nn.Linear: Y = X @ W.T   ≡   Y = X_as_A @ W_T_as_B
M = 16   # batch
X = torch.randn(M, 4096, dtype=torch.float16, device='cuda') * 0.5
Y_ref  = X @ W.t()    # (16, 4096)
Y_int4 = torch.empty(M, N, dtype=torch.float16, device='cuda')
rt.cipher_weight_compress_int4_gemm(
    X.data_ptr(), b4.value, bs.value, Y_int4.data_ptr(),
    M, N, K, None)
torch.cuda.synchronize()

abs_err = (Y_int4.float() - Y_ref.float()).abs()
rel_err = abs_err / (Y_ref.float().abs() + 1e-6)
print(f"nn.Linear-style substitution (X @ W.T) on (4096, 4096) Mistral-shape weight:")
print(f"  rel_err: mean={rel_err.mean():.4f} median={rel_err.median():.4f} p99={rel_err.flatten().kthvalue(int(0.99*rel_err.numel())).values.item():.4f}")
print(f"  abs_err: mean={abs_err.mean():.4f} max={abs_err.max():.4f}")
print(f"  C_ref max={Y_ref.abs().max():.3f}  C_int4 max={Y_int4.abs().max():.3f}")

# Throughput
torch.cuda.synchronize()
t0 = time.perf_counter()
N_ITER = 100
for _ in range(N_ITER):
    rt.cipher_weight_compress_int4_gemm(
        X.data_ptr(), b4.value, bs.value, Y_int4.data_ptr(),
        M, N, K, None)
torch.cuda.synchronize()
t_int4 = (time.perf_counter() - t0) / N_ITER * 1e6

t0 = time.perf_counter()
for _ in range(N_ITER):
    Y_ref = X @ W.t()
torch.cuda.synchronize()
t_cublas = (time.perf_counter() - t0) / N_ITER * 1e6

print(f"\nThroughput (M=16, K=N=4096):")
print(f"  cuBLAS:    {t_cublas:.1f} us")
print(f"  CIPHER int4: {t_int4:.1f} us  ({t_cublas/t_int4:.2f}x vs cuBLAS)")
