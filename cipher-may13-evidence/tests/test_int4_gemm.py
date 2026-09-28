import os, ctypes, time
import torch
ROOT = "/home/ubuntu/op31-prod-fix"
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"))

# Bind APIs
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
print(f"sub_v2 enabled: {rt.cipher_substitute_v2_enabled()}")
print(f"wcomp enabled:  {rt.cipher_weight_compress_enabled()}")

torch.manual_seed(42)
# Small-ish so the verification is cheap. K must equal weight rows since
# Stage-7 quantizes with per-row scales over the K axis.
M, K, N = 4, 1024, 1024
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
W = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.1   # (K, N) weight

# Stage 7 quantize wants (rows, cols) = (K, N). Force compression.
for _ in range(1001):
    rt.cipher_weight_compress_observe(W.data_ptr(), K * N * 2)
rc = rt.cipher_weight_compress_quantize(W.data_ptr(), K, N)
print(f"quantize rc={rc}")

# Lookup the buffers
b4 = ctypes.c_void_p(); bs = ctypes.c_void_p()
br = ctypes.c_int(); bc = ctypes.c_int()
ok = rt.cipher_weight_compress_lookup(W.data_ptr(),
    ctypes.byref(b4), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
print(f"lookup ok={ok} rows={br.value} cols={bc.value} int4_buf={b4.value:#x} scale_buf={bs.value:#x}")

# Run our INT4 kernel
C_int4 = torch.empty(M, N, dtype=torch.float16, device='cuda')
torch.cuda.synchronize()
rc = rt.cipher_weight_compress_int4_gemm(
    A.data_ptr(), b4.value, bs.value, C_int4.data_ptr(), M, N, K, None)
torch.cuda.synchronize()
print(f"int4_gemm rc={rc}")

# Reference: cuBLAS A @ W
C_ref = A @ W

# Compare
abs_err = (C_int4.float() - C_ref.float()).abs()
rel_err = abs_err / (C_ref.float().abs() + 1e-6)
print(f"\nReference max abs: {C_ref.abs().max().item():.3f}")
print(f"INT4    max abs:   {C_int4.abs().max().item():.3f}")
print(f"Abs err:    mean={abs_err.mean().item():.4f} max={abs_err.max().item():.4f}")
print(f"Rel err:    mean={rel_err.mean().item():.4f} median={rel_err.median().item():.4f} p99={rel_err.flatten().kthvalue(int(0.99*rel_err.numel())).values.item():.4f}")

# Print a few sample values
print("\nSample values (first 5 of [0, :5]):")
print(f"  ref:  {C_ref[0, :5].tolist()}")
print(f"  int4: {C_int4[0, :5].tolist()}")
