import os, ctypes
import torch
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

GROUP = 128
libcudart = ctypes.CDLL("libcudart.so.12")
libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def quant_and_check(W, label, M=4):
    K, N = W.shape
    num_kg = (K + GROUP - 1) // GROUP
    for _ in range(1001):
        rt.cipher_weight_compress_observe(W.data_ptr(), K * N * 2)
    rc = rt.cipher_weight_compress_quantize(W.data_ptr(), K, N)
    if rc != 1:
        print(f"  [{label}] quantize failed"); return None

    b4 = ctypes.c_void_p(); bs = ctypes.c_void_p(); br = ctypes.c_int(); bc = ctypes.c_int()
    rt.cipher_weight_compress_lookup(W.data_ptr(),
        ctypes.byref(b4), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))

    int4 = torch.empty(K * (N // 2), dtype=torch.uint8, device='cuda')
    scales = torch.empty(num_kg * N, dtype=torch.float16, device='cuda')
    libcudart.cudaMemcpy(int4.data_ptr(), b4.value, K * (N // 2), 3)
    libcudart.cudaMemcpy(scales.data_ptr(), bs.value, num_kg * N * 2, 3)
    torch.cuda.synchronize()

    # Dequant per along-K layout
    buf = int4.view(K, N // 2)
    low  = (buf & 0xF).int(); high = ((buf >> 4) & 0xF).int()
    low  = torch.where(low  >= 8, low  - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    scales_2d = scales.view(num_kg, N).float()  # (K/128, N)
    k_to_g = (torch.arange(K, device='cuda') // GROUP).long()
    scale_per_kn = scales_2d[k_to_g]  # (K, N)
    W_dq = torch.zeros(K, N, dtype=torch.float16, device='cuda')
    W_dq[:, 0::2] = (low.float()  * scale_per_kn[:, 0::2]).half()
    W_dq[:, 1::2] = (high.float() * scale_per_kn[:, 1::2]).half()
    abs_err = (W_dq.float() - W.float()).abs()
    rel_err = abs_err / (W.float().abs() + 1e-6)
    print(f"  [{label}] dequant rel: mean={rel_err.mean():.4f} median={rel_err.median():.4f}")

    # GEMM
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    C_ref = A @ W
    C_int4 = torch.empty(M, N, dtype=torch.float16, device='cuda')
    rt.cipher_weight_compress_int4_gemm(
        A.data_ptr(), b4.value, bs.value, C_int4.data_ptr(), M, N, K, None)
    torch.cuda.synchronize()
    abs_err = (C_int4.float() - C_ref.float()).abs()
    rel_err = abs_err / (C_ref.float().abs() + 1e-6)
    print(f"  [{label}] gemm rel:    mean={rel_err.mean():.4f} median={rel_err.median():.4f}  max_abs_C_ref={C_ref.abs().max():.2f}  max_abs_C_int4={C_int4.abs().max():.2f}")
    return rel_err.median().item()


print("=== Random Gaussian (worst case) ===")
torch.manual_seed(42)
W1 = torch.randn(1024, 1024, dtype=torch.float16, device='cuda') * 0.1
quant_and_check(W1, "random")

print("\n=== Low-rank (LLM-like) ===")
torch.manual_seed(43)
U = torch.randn(1024, 32, dtype=torch.float16, device='cuda') * 0.3
V = torch.randn(32, 1024, dtype=torch.float16, device='cuda') * 0.3
W2 = (U @ V).contiguous()
quant_and_check(W2, "low-rank")

print("\n=== Sparse outliers (5%) ===")
torch.manual_seed(44)
W3 = (torch.randn(1024, 1024, dtype=torch.float16, device='cuda') * 0.05).contiguous()
mask = torch.rand_like(W3) < 0.05
W3[mask] = (torch.randn_like(W3[mask]) * 0.5).half()
quant_and_check(W3, "outliers")

print("\n=== LLM-realistic shape (4096 x 14336, like Mistral up_proj) ===")
torch.manual_seed(45)
W4 = (torch.randn(4096, 14336, dtype=torch.float16, device='cuda') * 0.02).contiguous()
quant_and_check(W4, "mistral-up", M=1)
