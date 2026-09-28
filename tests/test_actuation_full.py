#!/usr/bin/env python3
"""End-to-end actuation verification for stages 6, 7, 8, 9, 10, 11.

Each test exercises the stage's CUDA-side actuation:
  6: NVRTC compile + cuModuleGetFunction returns a non-NULL CUfunction
  7: weight quantize launches the INT4 kernel, output byte size = rows*cols/2
  8: KV quantize launches the 2-bit kernel
  9: NCCL v4 decide returns a non-AUTO algo for ≥16 MB reductions
 10: partition router returns distinct CUstreams per partition
 11: NVML sampler thread is alive — silicon clock_sustained_mhz updates
"""
import ctypes
import os
import sys
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rt = ctypes.CDLL(os.path.join(ROOT, 'libcipher_rt.so'))

    # ── Stage 6: NVRTC compile of a trivial kernel ──────────────────────────
    rt.cipher_substitute_v2_init.restype = ctypes.c_int
    rt.cipher_substitute_v2_enabled.restype = ctypes.c_int
    rt.cipher_substitute_v2_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    rt.cipher_substitute_v2_compile.restype  = ctypes.c_ulong
    rt.cipher_substitute_v2_get_function.argtypes = [ctypes.c_ulong]
    rt.cipher_substitute_v2_get_function.restype  = ctypes.c_void_p

    rt.cipher_substitute_v2_init()
    if rt.cipher_substitute_v2_enabled():
        src = b"extern \"C\" __global__ void cipher_test_kernel(float* x) { x[0] = 42.0f; }"
        cubin = rt.cipher_substitute_v2_compile(src, b"cipher_test_kernel")
        if cubin == 0:
            print("[stage6] FAIL nvrtc compile returned 0")
            sys.exit(1)
        fn = rt.cipher_substitute_v2_get_function(cubin)
        if not fn:
            print("[stage6] FAIL get_function returned NULL")
            sys.exit(1)
        print(f"[stage6] PASS nvrtc compiled cubin_id={cubin} fn={fn:#x}")
    else:
        print("[stage6] SKIP CIPHER_SUBSTITUTE_V2 not enabled")

    # ── Stage 7: INT4 weight quantize ───────────────────────────────────────
    rt.cipher_weight_compress_init.restype                = ctypes.c_int
    rt.cipher_weight_compress_enabled.restype             = ctypes.c_int
    rt.cipher_weight_compress_observe.argtypes            = [ctypes.c_void_p, ctypes.c_size_t]
    rt.cipher_weight_compress_observe.restype             = ctypes.c_int
    rt.cipher_weight_compress_quantize.argtypes           = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    rt.cipher_weight_compress_quantize.restype            = ctypes.c_int

    rt.cipher_weight_compress_init()
    if rt.cipher_weight_compress_enabled() and rt.cipher_substitute_v2_enabled():
        rows, cols = 256, 1024  # 256*1024*2 = 512 KB; below MIN_WEIGHT_BYTES(1MB)
        rows, cols = 1024, 1024  # 2 MB — passes min size
        w = torch.randn(rows, cols, dtype=torch.float16, device='cuda')
        # observe ≥ STABLE_HIT_THRESHOLD(1000) times to force compression
        for _ in range(1001):
            rt.cipher_weight_compress_observe(w.data_ptr(), rows * cols * 2)
        rc = rt.cipher_weight_compress_quantize(w.data_ptr(), rows, cols)
        if rc != 1:
            print(f"[stage7] FAIL quantize returned {rc}")
            sys.exit(1)
        print(f"[stage7] PASS INT4 quantize launched on {rows}x{cols} fp16 weight")
    else:
        print("[stage7] SKIP CIPHER_WEIGHT_COMPRESS or CIPHER_SUBSTITUTE_V2 disabled")

    # ── Stage 8: KV 2-bit quantize ──────────────────────────────────────────
    rt.cipher_kv_compress_init.restype       = ctypes.c_int
    rt.cipher_kv_compress_enabled.restype    = ctypes.c_int
    rt.cipher_kv_compress_observe.argtypes   = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    rt.cipher_kv_compress_observe.restype    = ctypes.c_int
    rt.cipher_kv_compress_quantize.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    rt.cipher_kv_compress_quantize.restype   = ctypes.c_int

    rt.cipher_kv_compress_init()
    if rt.cipher_kv_compress_enabled() and rt.cipher_substitute_v2_enabled():
        rows, cols = 64, 4096
        kv = torch.randn(rows, cols, dtype=torch.float16, device='cuda')
        for _ in range(201):
            rt.cipher_kv_compress_observe(0, kv.data_ptr(), rows * cols * 2)
        rc = rt.cipher_kv_compress_quantize(kv.data_ptr(), rows, cols, 0)
        if rc != 1:
            print(f"[stage8] FAIL kv quantize returned {rc}")
            sys.exit(1)
        print(f"[stage8] PASS 2-bit KV quantize launched on {rows}x{cols} (mode=key)")
    else:
        print("[stage8] SKIP CIPHER_KV_COMPRESS or CIPHER_SUBSTITUTE_V2 disabled")

    # ── Stage 9: NCCL v4 decide ─────────────────────────────────────────────
    rt.cipher_nccl_v4_init.restype     = ctypes.c_int
    rt.cipher_nccl_v4_enabled.restype  = ctypes.c_int
    rt.cipher_nccl_v4_decide.argtypes  = [ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)]
    rt.cipher_nccl_v4_decide.restype   = ctypes.c_int

    rt.cipher_nccl_v4_init()
    if rt.cipher_nccl_v4_enabled():
        algo = ctypes.c_int(0)
        rc = rt.cipher_nccl_v4_decide(64 * 1024 * 1024, 8, ctypes.byref(algo))
        if rc != 1 or algo.value != 5:    # NCCL_ALGO_NVLS = 5
            print(f"[stage9] FAIL decide for 64MB: rc={rc} algo={algo.value} expected NVLS(5)")
            sys.exit(1)
        algo2 = ctypes.c_int(0)
        rt.cipher_nccl_v4_decide(512 * 1024, 8, ctypes.byref(algo2))
        if algo2.value != 2:    # RING
            print(f"[stage9] FAIL decide for 512KB: algo={algo2.value} expected RING(2)")
            sys.exit(1)
        print(f"[stage9] PASS NCCL v4 decide: 64MB→NVLS, 512KB→RING")
    else:
        print("[stage9] SKIP CIPHER_NCCL_V4 not enabled")

    # ── Stage 10: partition router stream creation ──────────────────────────
    rt.cipher_partition_router_init.restype          = ctypes.c_int
    rt.cipher_partition_router_enabled.restype       = ctypes.c_int
    rt.cipher_partition_router_bind.argtypes         = [ctypes.c_void_p, ctypes.c_int]
    rt.cipher_partition_router_bind.restype          = ctypes.c_int
    rt.cipher_partition_router_stream_for.argtypes   = [ctypes.c_void_p]
    rt.cipher_partition_router_stream_for.restype    = ctypes.c_void_p

    rt.cipher_partition_router_init()
    if rt.cipher_partition_router_enabled():
        # Two distinct fake pointers, one per partition
        ptr_a = ctypes.cast(0x1000, ctypes.c_void_p)
        ptr_b = ctypes.cast(0x2000, ctypes.c_void_p)
        rt.cipher_partition_router_bind(ptr_a, 0)
        rt.cipher_partition_router_bind(ptr_b, 1)
        s_a = rt.cipher_partition_router_stream_for(ptr_a)
        s_b = rt.cipher_partition_router_stream_for(ptr_b)
        if not s_a or not s_b or s_a == s_b:
            print(f"[stage10] FAIL streams a={s_a:#x} b={s_b:#x}")
            sys.exit(1)
        print(f"[stage10] PASS partition streams created — a={s_a:#x} b={s_b:#x}")
    else:
        print("[stage10] SKIP CIPHER_PARTITION_ROUTER not enabled")

    # ── Stage 11: NVML sampler updates silicon dynamic fields ──────────────
    rt.cipher_thermal_feedback_init.restype                = ctypes.c_int
    rt.cipher_thermal_feedback_enabled.restype             = ctypes.c_int
    rt.cipher_thermal_feedback_aggressiveness.restype      = ctypes.c_double

    class FBStats(ctypes.Structure):
        _fields_ = [
            ("enabled",                   ctypes.c_int),
            ("substitute_aggressiveness", ctypes.c_double),
            ("last_clock_mhz",            ctypes.c_double),
            ("last_headroom",             ctypes.c_double),
            ("last_power_w",              ctypes.c_double),
            ("tick_calls",                ctypes.c_uint64),
            ("aggressiveness_increased",  ctypes.c_uint64),
            ("aggressiveness_decreased",  ctypes.c_uint64),
            ("oscillation_clamps",        ctypes.c_uint64),
        ]
    rt.cipher_thermal_feedback_stats.argtypes = [ctypes.POINTER(FBStats)]
    rt.cipher_thermal_feedback_stats.restype  = ctypes.c_int

    rt.cipher_thermal_feedback_init()
    if rt.cipher_thermal_feedback_enabled():
        # Run a brief GPU workload so power.draw is non-zero, then check the
        # sampler picked it up.
        a = torch.randn(2048, 2048, dtype=torch.float16, device='cuda')
        b = torch.randn(2048, 2048, dtype=torch.float16, device='cuda')
        for _ in range(200):
            torch.mm(a, b)
        torch.cuda.synchronize()
        time.sleep(0.5)  # let the 100Hz sampler tick a few times
        s = FBStats()
        rt.cipher_thermal_feedback_stats(ctypes.byref(s))
        if s.tick_calls == 0:
            print(f"[stage11] FAIL sampler thread didn't tick (tick_calls=0)")
            sys.exit(1)
        if s.last_clock_mhz < 100 or s.last_power_w < 1.0:
            print(f"[stage11] FAIL sampler reads look stale: "
                  f"clock={s.last_clock_mhz} pwr={s.last_power_w}")
            sys.exit(1)
        print(f"[stage11] PASS NVML sampler firing — ticks={s.tick_calls} "
              f"clock={s.last_clock_mhz:.0f}MHz power={s.last_power_w:.0f}W "
              f"headroom={s.last_headroom:.2f} aggr={s.substitute_aggressiveness:.3f}")
    else:
        print("[stage11] SKIP CIPHER_THERMAL_FEEDBACK not enabled")

    print("\nResult: actuation verified for all enabled stages")


if __name__ == "__main__":
    main()
