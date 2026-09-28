"""ACTION E: cross-GEMM L2 prefetch test.

Sequence: q_proj → k_proj → v_proj → o_proj → gate_proj → up_proj → down_proj
(actual Mistral attention+MLP linear sequence)

Variant A: just back-to-back fp16 GEMMs.
Variant B: before each GEMM, "touch" the NEXT weight by reading some bytes (forces L2 fill).

Compare total time.
"""
import torch
import time

# Mistral linear shapes
SHAPES = [
    ("q_proj", 4096, 4096),
    ("k_proj", 1024, 4096),
    ("v_proj", 1024, 4096),
    ("o_proj", 4096, 4096),
    ("gate_proj", 14336, 4096),
    ("up_proj", 14336, 4096),
    ("down_proj", 4096, 14336),
]


def main():
    print(f"=== ACTION E: cross-GEMM L2 prefetch ===\n", flush=True)

    # Allocate weights and inputs once
    Ws = []
    xs = []
    for name, N, K in SHAPES:
        W = torch.randn(N, K, dtype=torch.float16, device="cuda")
        x = torch.randn(1, K, dtype=torch.float16, device="cuda")
        Ws.append((name, W, x))
    total_W_bytes = sum(W.numel() * 2 for _, W, _ in Ws) / (1024 ** 2)
    print(f"  Total weights: {total_W_bytes:.1f} MB (vs L2 cache 50 MB)", flush=True)

    # Variant A: back-to-back GEMMs
    def variant_A():
        for name, W, x in Ws:
            _ = x @ W.T

    # Variant B: prefetch next weight via small read before each GEMM
    def variant_B():
        for i, (name, W, x) in enumerate(Ws):
            # Issue prefetch for NEXT weight (touch first 1MB to fill L2)
            if i + 1 < len(Ws):
                _next_W = Ws[i + 1][1]
                # Read first chunk to pull into L2 - use sum on a slice
                _ = _next_W[:64].sum()
            _ = x @ W.T

    # Variant C: explicit cudaMemPrefetchAsync via torch
    # torch doesn't expose cudaMemPrefetchAsync directly for device→L2.
    # But we can use cudaStreamSynchronize between issuing prefetch via .contiguous() + zero-cost touch
    # Skipped — variant B is the practical proxy.

    # Warmup
    for _ in range(20):
        variant_A()
    torch.cuda.synchronize()

    # Time A
    N_ITER = 200
    times_A = []
    for _ in range(N_ITER):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        variant_A()
        e.record()
        torch.cuda.synchronize()
        times_A.append(s.elapsed_time(e) * 1000.0)
    times_A.sort()
    median_A = times_A[N_ITER // 2]

    # Time B
    times_B = []
    for _ in range(N_ITER):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        variant_B()
        e.record()
        torch.cuda.synchronize()
        times_B.append(s.elapsed_time(e) * 1000.0)
    times_B.sort()
    median_B = times_B[N_ITER // 2]

    print(f"\n  Variant A (no prefetch): median {median_A:.1f} us", flush=True)
    print(f"  Variant B (touch next):   median {median_B:.1f} us", flush=True)
    print(f"  speedup: {median_A / median_B:.3f}x  ({'win' if median_B < median_A else 'no win'})", flush=True)

    # Per-GEMM breakdown
    print(f"\n  Per-GEMM cold timing (without prefetch):", flush=True)
    flush_buf = torch.empty(64 * 1024 * 1024, dtype=torch.float16, device="cuda")
    for name, W, x in Ws:
        flush_buf.zero_()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = x @ W.T
        e.record()
        torch.cuda.synchronize()
        cold_us = s.elapsed_time(e) * 1000.0

        # Hot
        for _ in range(5): _ = x @ W.T
        torch.cuda.synchronize()
        s.record()
        _ = x @ W.T
        e.record()
        torch.cuda.synchronize()
        hot_us = s.elapsed_time(e) * 1000.0
        size_mb = W.numel() * 2 / (1024**2)
        print(f"    {name:<10} {size_mb:>5.1f} MB  cold {cold_us:>5.1f} us  hot {hot_us:>5.1f} us  ratio {cold_us/hot_us:.2f}x", flush=True)

    print(f"\n  Aggregate weight pressure: {total_W_bytes:.0f} MB > 50 MB L2 → not all weights can fit simultaneously.", flush=True)
    print(f"  L2 evictions are inevitable; prefetching one weight evicts another.", flush=True)


if __name__ == "__main__":
    main()
