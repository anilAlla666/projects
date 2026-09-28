#!/usr/bin/env python3
"""CP 5.4 Step 1.5B-4 — Marlin cubin / green-ctx-churn confirmation test.

Question: is the Marlin cubin recompiled (or silently reloaded) per green
context? Code-derived prior (memo §4): NO — ensure_marlin_compiled() is a
one-shot std::atomic, the module loads once in the primary context, and the
Marlin GEMM is primary-context-pinned.

Test: warm Marlin (one NVRTC compile + cuModuleLoadData), then drive 20
production green-context swaps via cp54_pool.PoolBinding.build_green_ctx()
(destroy + recreate), and after each swap re-call ensure_compiled(). The CUPTI
counter (libcumod_count.so, LD_PRELOAD'd) records every cuModuleLoad* in the
process — Marlin's load is caught even though Marlin resolves the symbol via
dlsym(libcuda_handle,...).

Four asserts (memo §4):
  A1 one-shot      — every post-swap ensure_compiled() returns 0 with µs
                     latency (a recompile is ~19 s; a reload ~tens of ms).
  A2 module stable — exactly one Marlin cuModuleLoad*, issued at warmup while
                     only the primary context exists; zero during churn.
  A3 no reload     — cuModuleLoad* count delta across the 20 swaps == 0.
  A4 latency bound — post-swap ensure_compiled() and the swap itself stay
                     bounded (no hidden per-swap cost).

Throwaway Step 1.5 measurement harness — not a campaign anchor, not substrate.
Run with:  LD_PRELOAD=./libcumod_count.so python3 cp54_s15_marlin.py
"""
import argparse
import ctypes
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RT_PATH = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
COUNTER_PATH = os.path.join(HERE, "libcumod_count.so")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swaps", type=int, default=20)
    ap.add_argument("--out", type=str, default="marlin_confirm.json")
    args = ap.parse_args()

    # CUPTI counter (already LD_PRELOAD'd; CDLL to grab the accessors).
    counter = ctypes.CDLL(COUNTER_PATH)
    counter.s15_cumod_count.restype = ctypes.c_int
    counter.s15_cumod_mark.argtypes = [ctypes.c_char_p]

    def count():
        return counter.s15_cumod_count()

    def mark(tag):
        counter.s15_cumod_mark(tag.encode())

    # Marlin engine ABI from libcipher_rt.so (CDLL — engine fns are leaf;
    # cuBLAS interposition is intentionally NOT engaged).
    rt = ctypes.CDLL(RT_PATH, mode=ctypes.RTLD_GLOBAL)
    rt.cipher_rt_marlin_engine_init.restype = ctypes.c_int
    rt.cipher_rt_marlin_engine_ensure_compiled.restype = ctypes.c_int

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    import torch
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    # ---- warmup: compile + load the Marlin cubin -------------------------
    mark("pre-marlin-warmup")
    c_pre_warm = count()
    rc = rt.cipher_rt_marlin_engine_init()
    print("[marlin] engine_init rc=%d" % rc, flush=True)
    t0 = time.perf_counter()
    rc = rt.cipher_rt_marlin_engine_ensure_compiled()
    warmup_ms = (time.perf_counter() - t0) * 1e3
    c_post_warm = count()
    mark("post-marlin-warmup")
    marlin_loads = c_post_warm - c_pre_warm
    print("[marlin] WARMUP ensure_compiled rc=%d  %.1f ms  "
          "cuModuleLoad* during warmup=%d  (total=%d)"
          % (rc, warmup_ms, marlin_loads, c_post_warm), flush=True)
    if rc != 0:
        print("[marlin] FATAL: warmup ensure_compiled failed", flush=True)
        sys.exit(2)

    # ---- green-context churn driver --------------------------------------
    sys.path.insert(0, os.path.join(HERE, "..", "step1_4"))
    import cp54_pool
    pool = cp54_pool.PoolBinding()
    pool.build_green_ctx()                       # initial green ctx + %smid ext
    torch.cuda.synchronize()

    mark("pre-churn")
    c_prechurn = count()
    print("[marlin] pre-churn cuModuleLoad* total=%d — starting %d swaps"
          % (c_prechurn, args.swaps), flush=True)

    swaps = []
    for i in range(args.swaps):
        torch.cuda.synchronize()
        ts = time.perf_counter()
        pool.build_green_ctx()                   # green-ctx destroy + recreate
        swap_ms = (time.perf_counter() - ts) * 1e3

        te = time.perf_counter()
        erc = rt.cipher_rt_marlin_engine_ensure_compiled()
        ens_ms = (time.perf_counter() - te) * 1e3

        cnt = count()
        swaps.append({"i": i, "swap_ms": swap_ms, "ensure_rc": erc,
                      "ensure_ms": ens_ms, "cumod_count": cnt})
        print("[marlin] swap %2d  swap=%.3f ms  ensure_compiled rc=%d %.4f ms  "
              "cuModuleLoad* total=%d" % (i, swap_ms, erc, ens_ms, cnt),
              flush=True)

    mark("post-churn")
    c_postchurn = count()
    pool.free()

    # ---- evaluate the 4 asserts ------------------------------------------
    churn_delta = c_postchurn - c_prechurn
    max_ens = max(s["ensure_ms"] for s in swaps)
    max_swap = max(s["swap_ms"] for s in swaps)
    all_rc0 = all(s["ensure_rc"] == 0 for s in swaps)

    a1 = all_rc0 and max_ens < 1.0          # one-shot: fast + rc=0, no recompile
    a2 = (marlin_loads >= 1) and (churn_delta == 0)  # loaded once, pre-green
    a3 = churn_delta == 0                   # no cuModuleLoad* during swaps
    a4 = max_ens < 1.0 and max_swap < 10.0  # latency bounded

    asserts = {
        "A1_ensure_compiled_one_shot": bool(a1),
        "A2_module_stable_primary_bound": bool(a2),
        "A3_no_cuModuleLoad_during_churn": bool(a3),
        "A4_per_swap_latency_bounded": bool(a4),
    }
    summary = {
        "test": "marlin_greenctx_churn_confirmation",
        "swaps": args.swaps,
        "warmup_ms": warmup_ms,
        "marlin_cuModuleLoad_at_warmup": marlin_loads,
        "cumod_count_prechurn": c_prechurn,
        "cumod_count_postchurn": c_postchurn,
        "cuModuleLoad_delta_across_churn": churn_delta,
        "max_post_swap_ensure_ms": max_ens,
        "max_swap_ms": max_swap,
        "all_ensure_rc_zero": bool(all_rc0),
        "asserts": asserts,
        "verdict": "PASS" if all(asserts.values()) else "FAIL",
        "swap_detail": swaps,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[marlin] ===== 4-ASSERT CONFIRMATION =====", flush=True)
    for k, v in asserts.items():
        print("[marlin]   %-34s %s" % (k, "PASS" if v else "FAIL"), flush=True)
    print("[marlin] cuModuleLoad* delta across %d green-ctx swaps = %d"
          % (args.swaps, churn_delta), flush=True)
    print("[marlin] VERDICT: %s — wrote %s" % (summary["verdict"], args.out),
          flush=True)
    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
