"""ACTION B: L2 cache hot vs cold GEMM timing per Mistral linear shape.

H100: 50 MB L2 @ ~12 TB/s, HBM @ 3.35 TB/s. L2 hits should be ~3.6x faster
than HBM if memory-bound. Test fp16 GEMM and INT4 GEMV.
"""
import os, ctypes, time, json
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_weight_compress_init.restype = ctypes.c_int
rt.cipher_weight_compress_observe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
rt.cipher_weight_compress_observe.restype = ctypes.c_int
rt.cipher_weight_compress_quantize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
rt.cipher_weight_compress_quantize.restype = ctypes.c_int
rt.cipher_weight_compress_lookup_T.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
rt.cipher_weight_compress_lookup_T.restype = ctypes.c_int
rt.cipher_weight_compress_int4_gemv.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
rt.cipher_weight_compress_int4_gemv.restype = ctypes.c_int
rt.cipher_substitute_v2_init()
rt.cipher_weight_compress_init()


def time_op(op_fn, n_iter=100):
    """Run op_fn n_iter times preceded by L2 flush each time. Returns median μs."""
    # Allocate L2 flush buffer (50+ MB to evict L2)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.float16, device="cuda")  # 128 MB

    times = []
    for _ in range(n_iter):
        # Cold: flush L2 first
        flush.zero_()  # write to all 128 MB to evict everything
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        op_fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[n_iter // 2]


def time_op_hot(op_fn, n_iter=100):
    """Run op_fn n_iter times BACK-TO-BACK (weight stays in L2). Returns median μs."""
    # Warmup
    for _ in range(5):
        op_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        op_fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[n_iter // 2]


def fp16_gemm_size_mb(N, K, dtype_bytes=2):
    """Weight bytes."""
    return N * K * dtype_bytes / (1024**2)


def main():
    print(f"=== ACTION B: L2 cache cold vs hot per Mistral shape ===\n", flush=True)
    SHAPES = [
        ("q_proj/o_proj", 4096, 4096),
        ("k_proj/v_proj", 1024, 4096),
        ("gate_proj/up_proj", 14336, 4096),
        ("down_proj", 4096, 14336),
    ]

    print(f"  {'shape':<20} {'fp16_W_MB':>9} {'fp16_cold_us':>13} {'fp16_hot_us':>12} {'fp16_speedup':>13}", flush=True)
    print("  " + "-" * 75, flush=True)

    fp16_results = []
    for name, N, K in SHAPES:
        W = torch.randn(N, K, dtype=torch.float16, device="cuda")
        x = torch.randn(1, K, dtype=torch.float16, device="cuda")
        size_mb = fp16_gemm_size_mb(N, K)
        cold = time_op(lambda: x @ W.T)
        hot  = time_op_hot(lambda: x @ W.T)
        speedup = cold / hot if hot > 0 else 0
        print(f"  {name:<20} {size_mb:>9.1f} {cold:>13.1f} {hot:>12.1f} {speedup:>12.2f}x", flush=True)
        fp16_results.append({"shape": name, "N": N, "K": K, "size_mb": size_mb, "cold_us": cold, "hot_us": hot, "speedup": speedup})
        del W, x

    print(f"\n  {'shape':<20} {'INT4_W_MB':>9} {'INT4_cold_us':>13} {'INT4_hot_us':>12} {'INT4_speedup':>13}", flush=True)
    print("  " + "-" * 75, flush=True)

    int4_results = []
    for name, N, K in SHAPES:
        if K % 128 != 0 or N % 8 != 0:
            print(f"  {name:<20} skip (incompatible shape)", flush=True)
            continue
        # Quantize a weight: cipher_weight_compress takes wt = W.T.contiguous() shape (K, N)
        W = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.02
        wt = W.t().contiguous()
        for _ in range(1001):
            rt.cipher_weight_compress_observe(wt.data_ptr(), wt.numel() * 2)
        rc = rt.cipher_weight_compress_quantize(wt.data_ptr(), K, N)
        if rc != 1:
            print(f"  {name:<20} quantize failed", flush=True)
            continue
        bT = ctypes.c_void_p(); bs = ctypes.c_void_p()
        br = ctypes.c_int(); bc = ctypes.c_int()
        rt.cipher_weight_compress_lookup_T(wt.data_ptr(),
            ctypes.byref(bT), ctypes.byref(bs), ctypes.byref(br), ctypes.byref(bc))
        x = torch.randn(1, K, dtype=torch.float16, device="cuda") * 0.1
        out = torch.empty(1, N, dtype=torch.float16, device="cuda")
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        size_mb = N * K / 2 / (1024 ** 2)  # int4 = 0.5 byte/elem

        def gemv():
            rt.cipher_weight_compress_int4_gemv(
                x.data_ptr(), bT.value, bs.value, out.data_ptr(),
                1, N, K, stream)

        cold = time_op(gemv)
        hot  = time_op_hot(gemv)
        speedup = cold / hot if hot > 0 else 0
        print(f"  {name:<20} {size_mb:>9.1f} {cold:>13.1f} {hot:>12.1f} {speedup:>12.2f}x", flush=True)
        int4_results.append({"shape": name, "N": N, "K": K, "size_mb": size_mb, "cold_us": cold, "hot_us": hot, "speedup": speedup})
        del W, wt, x, out

    print(f"\n=== Interpretation ===", flush=True)
    print(f"  L2 capacity: 50 MB. Weights ≤50 MB should fit; weights >50 MB don't.", flush=True)
    print(f"  Hot/cold speedup > 1.5x → fits in L2; speedup ~1.0x → doesn't fit", flush=True)

    with open("/home/ubuntu/op31-prod-fix/action_b_results.json", "w") as f:
        json.dump({"fp16": fp16_results, "int4": int4_results}, f, indent=2)


if __name__ == "__main__":
    main()
