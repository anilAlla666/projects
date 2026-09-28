import os, ctypes, time
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
rt.cipher_weight_compress_int4_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
rt.cipher_weight_compress_int4_gemv.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()


def bench(label, K, N, M=1):
    torch.manual_seed(0)
    W = torch.randn(K, N, dtype=torch.float16, device='cuda') * 0.02
    A = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.5

    for _ in range(1001):
        rt.cipher_weight_compress_observe(W.data_ptr(), K * N * 2)
    rt.cipher_weight_compress_quantize(W.data_ptr(), K, N)

    b4 = ctypes.c_void_p(); bs = ctypes.c_void_p(); br = ctypes.c_int(); bc = ctypes.c_int()
    rt.cipher_weight_compress_lookup(W.data_ptr(),
        ctypes.byref(b4), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))

    C_int4 = torch.empty(M, N, dtype=torch.float16, device='cuda')
    rt.cipher_weight_compress_int4_gemv(
        A.data_ptr(), b4.value, bs.value, C_int4.data_ptr(), N, K, None)
    torch.cuda.synchronize()

    C_ref = A @ W
    abs_err = (C_int4.float() - C_ref.float()).abs()
    rel_err = abs_err / (C_ref.float().abs() + 1e-6)
    print(f"  {label} (K={K}, N={N})")
    print(f"    rel_err mean={rel_err.mean():.4f} median={rel_err.median():.4f}")
    print(f"    max_abs ref={C_ref.abs().max():.3f}  int4={C_int4.abs().max():.3f}")

    # Throughput
    N_ITER = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        rt.cipher_weight_compress_int4_gemv(
            A.data_ptr(), b4.value, bs.value, C_int4.data_ptr(), N, K, None)
    torch.cuda.synchronize()
    t_ours = (time.perf_counter() - t0) / N_ITER * 1e6

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        C_ref = A @ W
    torch.cuda.synchronize()
    t_cublas = (time.perf_counter() - t0) / N_ITER * 1e6
    print(f"    cuBLAS={t_cublas:.1f}us  CIPHER int4_gemv={t_ours:.1f}us  ({t_cublas/t_ours:.2f}x)")


print("=== Mistral-shape GEMV benchmarks (M=1) ===")
bench("attn q_proj/o_proj", K=4096, N=4096)
bench("attn k_proj/v_proj", K=4096, N=1024)
bench("mlp gate/up_proj  ", K=4096, N=14336)
bench("mlp down_proj     ", K=14336, N=4096)
