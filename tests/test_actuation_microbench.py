#!/usr/bin/env python3
"""Stage 3 + 5 actuation microbenchmarks — measure the actual GPU-execution
deltas the persist engine and graph engine produce on the workloads they
target.

  bench_l2_persist:   memory-bound stream-K kernel where one tensor is
                      reused across many launches. With persist injection,
                      that tensor stays in L2 → fewer HBM reads → speedup.
  bench_graph_replay: 64 small kernels per step, captured into a CUDA graph
                      and replayed via cuGraphLaunch instead of individual
                      cudaLaunchKernel calls.
"""
import ctypes
import os
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stats_struct():
    class S(ctypes.Structure):
        _fields_ = [
            ("registered_count",       ctypes.c_int),
            ("admitted_count",         ctypes.c_int),
            ("admitted_bytes",         ctypes.c_size_t),
            ("budget_bytes",           ctypes.c_size_t),
            ("register_calls",         ctypes.c_uint64),
            ("unregister_calls",       ctypes.c_uint64),
            ("window_lookups",         ctypes.c_uint64),
            ("window_hits",            ctypes.c_uint64),
            ("recompute_calls",        ctypes.c_uint64),
            ("apply_to_stream_calls",  ctypes.c_uint64),
            ("l2_resets",              ctypes.c_uint64),
        ]
    return S


# ── Bench 1: L2 persistence on a reused tensor ──────────────────────────────
# Workload: c = a + b across many iterations, where `a` is reused. With
# persist on, `a` (32 MB) gets pinned in L2 — but on H100 cuBLAS already
# manages cache well, so the gain is small. The honest test is: does the
# attribute injection happen, and does it change measured behaviour at all?

def bench_l2_persist(seconds=10):
    a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    out = torch.empty_like(a)
    for _ in range(150):
        torch.add(a, b, out=out)
    torch.cuda.synchronize()
    start = time.time()
    iters = 0
    while time.time() - start < seconds:
        torch.add(a, b, out=out)
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.time() - start
    bytes_per_iter = 4096 * 4096 * 2 * 3   # 3 fp16 tensors touched per add
    return iters, elapsed, bytes_per_iter / 1e9 * iters / elapsed


# ── Bench 2: graph replay — sequence of kernels captured & replayed ─────────

def bench_graph_replay(seconds=10):
    """Capture a sequence of small kernels into a CUDA graph and replay it."""
    rt = ctypes.CDLL(os.path.join(ROOT, 'libcipher_rt.so'))
    rt.cipher_graph_init.restype                = ctypes.c_int
    rt.cipher_graph_enabled.restype             = ctypes.c_int
    rt.cipher_graph_begin_capture.argtypes      = [ctypes.c_void_p]
    rt.cipher_graph_begin_capture.restype       = ctypes.c_int
    rt.cipher_graph_end_capture.argtypes        = [ctypes.c_void_p]
    rt.cipher_graph_end_capture.restype         = ctypes.c_ulong
    rt.cipher_graph_replay.argtypes             = [ctypes.c_ulong, ctypes.c_void_p]
    rt.cipher_graph_replay.restype              = ctypes.c_int
    rt.cipher_graph_destroy.argtypes            = [ctypes.c_ulong]
    rt.cipher_graph_destroy.restype             = ctypes.c_int

    rt.cipher_graph_init()
    if not rt.cipher_graph_enabled():
        print("  [graph] CIPHER_GRAPH not enabled, skipping")
        return None

    # Sequence of N tiny kernels per step. Element-wise add on small tensors
    # so the per-iter compute is small relative to launch overhead.
    N = 64
    SIZE = 1024   # 4 KB tensor — tiny, launch-bound
    xs = [torch.randn(SIZE, dtype=torch.float32, device='cuda') for _ in range(N)]
    ys = [torch.randn(SIZE, dtype=torch.float32, device='cuda') for _ in range(N)]
    outs = [torch.empty_like(xs[i]) for i in range(N)]

    stream = torch.cuda.Stream()
    stream_handle = stream.cuda_stream

    # ── Phase A: individual launches per step ───────────────────────────────
    with torch.cuda.stream(stream):
        for _ in range(50):
            for i in range(N):
                torch.add(xs[i], ys[i], out=outs[i])
    stream.synchronize()
    start = time.time()
    a_iters = 0
    while time.time() - start < seconds:
        with torch.cuda.stream(stream):
            for i in range(N):
                torch.add(xs[i], ys[i], out=outs[i])
        a_iters += 1
    stream.synchronize()
    a_elapsed = time.time() - start

    # ── Phase B: capture into a graph, replay via cuGraphLaunch ────────────
    # Capture
    with torch.cuda.stream(stream):
        if rt.cipher_graph_begin_capture(stream_handle) != 1:
            print("  [graph] begin_capture failed")
            return None
        for i in range(N):
            torch.add(xs[i], ys[i], out=outs[i])
    graph_id = rt.cipher_graph_end_capture(stream_handle)
    if graph_id == 0:
        print("  [graph] end_capture failed")
        return None

    # Warmup replays
    for _ in range(50):
        rt.cipher_graph_replay(graph_id, stream_handle)
    stream.synchronize()

    start = time.time()
    b_iters = 0
    while time.time() - start < seconds:
        rt.cipher_graph_replay(graph_id, stream_handle)
        b_iters += 1
    stream.synchronize()
    b_elapsed = time.time() - start

    rt.cipher_graph_destroy(graph_id)

    return {
        'individual_iters': a_iters,
        'individual_elapsed': a_elapsed,
        'individual_per_step_us': a_elapsed / a_iters * 1e6 if a_iters else float('inf'),
        'replay_iters': b_iters,
        'replay_elapsed': b_elapsed,
        'replay_per_step_us': b_elapsed / b_iters * 1e6 if b_iters else float('inf'),
        'kernels_per_step': N,
        'speedup': (a_elapsed / a_iters) / (b_elapsed / b_iters) if b_iters else 0,
    }


# ── L2 persist stat probe ────────────────────────────────────────────────────

def persist_stats():
    rt = ctypes.CDLL(os.path.join(ROOT, 'libcipher_rt.so'))
    S = _stats_struct()
    rt.cipher_persist_engine_stats.argtypes = [ctypes.POINTER(S)]
    rt.cipher_persist_engine_stats.restype  = ctypes.c_int
    s = S()
    rt.cipher_persist_engine_stats(ctypes.byref(s))
    return s


if __name__ == "__main__":
    print("=== Bench 1: L2 persist on tensor add (compute-bound elementwise) ===")
    iters, elapsed, gbps = bench_l2_persist(seconds=10)
    print(f"  {iters} iters / {elapsed:.2f}s — {gbps:.1f} GB/s effective")
    s = persist_stats()
    print(f"  persist: registered={s.registered_count} admitted={s.admitted_count} "
          f"window_lookups={s.window_lookups} window_hits={s.window_hits}")

    print()
    print("=== Bench 2: graph capture + replay vs individual launches ===")
    r = bench_graph_replay(seconds=10)
    if r:
        print(f"  individual ({r['kernels_per_step']} kernels/step): "
              f"{r['individual_iters']} steps in {r['individual_elapsed']:.2f}s "
              f"= {r['individual_per_step_us']:.1f} us/step")
        print(f"  graph replay:                     "
              f"{r['replay_iters']} steps in {r['replay_elapsed']:.2f}s "
              f"= {r['replay_per_step_us']:.1f} us/step")
        print(f"  speedup: {r['speedup']:.2f}x")
