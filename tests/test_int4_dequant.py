import os, ctypes
import torch
import numpy as np
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

rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()

torch.manual_seed(42)
K, N = 1024, 1024
W = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.1
for _ in range(1001):
    rt.cipher_weight_compress_observe(W.data_ptr(), K * N * 2)
rt.cipher_weight_compress_quantize(W.data_ptr(), K, N)

b4 = ctypes.c_void_p(); bs = ctypes.c_void_p()
br = ctypes.c_int(); bc = ctypes.c_int()
rt.cipher_weight_compress_lookup(W.data_ptr(),
    ctypes.byref(b4), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))

# Wrap GPU buffers as torch tensors via from_blob (needs reinterpret)
half_cols = N // 2
int4_t = torch.empty(K * half_cols, dtype=torch.uint8, device='cuda')
scales = torch.empty(K, dtype=torch.float16, device='cuda')
torch.cuda.synchronize()
# cudaMemcpy device-to-device using torch's underlying data
import ctypes
libcudart = ctypes.CDLL("libcudart.so.12")
libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libcudart.cudaMemcpy(int4_t.data_ptr(), b4.value, K * half_cols, 3)  # 3 = D2D
libcudart.cudaMemcpy(scales.data_ptr(),  bs.value, K * 2, 3)
torch.cuda.synchronize()

# Dequantize
buf = int4_t.view(K, half_cols)
low  = (buf & 0xF).int()
high = ((buf >> 4) & 0xF).int()
low  = torch.where(low  >= 8, low  - 16, low)
high = torch.where(high >= 8, high - 16, high)
W_dq = torch.zeros(K, N, dtype=torch.float16, device='cuda')
W_dq[:, 0::2] = low.half()  * scales.unsqueeze(1)
W_dq[:, 1::2] = high.half() * scales.unsqueeze(1)

abs_err = (W_dq.float() - W.float()).abs()
rel_err = abs_err / (W.float().abs() + 1e-6)
print(f"=== Dequant W vs original W ===")
print(f"  W abs max: {W.abs().max().item():.4f}")
print(f"  abs err mean: {abs_err.mean().item():.4f} max: {abs_err.max().item():.4f}")
print(f"  rel err mean: {rel_err.mean().item():.4f} median: {rel_err.median().item():.4f}")
print(f"  W[0, :5]:    {W[0, :5].tolist()}")
print(f"  W_dq[0, :5]: {W_dq[0, :5].tolist()}")
print(f"  scale[0]:    {scales[0].item():.6f}")

# GEMM check
A = torch.randn(4, K, dtype=torch.float16, device='cuda')
C_ref    = A @ W
C_ref_dq = A @ W_dq
print(f"\n=== GEMM: A@W vs A@W_dequant ===")
diff = (C_ref.float() - C_ref_dq.float()).abs()
print(f"  abs err mean: {diff.mean().item():.4f} max: {diff.max().item():.4f}")
print(f"  C_ref[0,:5]:    {C_ref[0,:5].tolist()}")
print(f"  C_ref_dq[0,:5]: {C_ref_dq[0,:5].tolist()}")
