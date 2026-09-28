#!/usr/bin/env python3
"""Stage 3 — Persistence Engine tests.

Verifies the engine API: init, register/unregister, fractional knapsack
budget, get_window, stream apply, and report. Tests run via ctypes against
the live libcipher_rt.so loaded under LD_PRELOAD with both the hook and rt
DSOs preloaded (for cipher_silicon_init to succeed).
"""
import ctypes
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load_engine():
    rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"))
    # Make sure silicon model is alive (init is idempotent).
    rt.cipher_silicon_init.restype = ctypes.c_int
    rt.cipher_silicon_init()

    rt.cipher_persist_engine_init.restype = ctypes.c_int
    rt.cipher_persist_engine_enabled.restype = ctypes.c_int
    rt.cipher_persist_engine_register.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double]
    rt.cipher_persist_engine_register.restype = ctypes.c_int
    rt.cipher_persist_engine_unregister.argtypes = [ctypes.c_void_p]
    rt.cipher_persist_engine_unregister.restype = ctypes.c_int
    rt.cipher_persist_engine_recompute_budget.restype = None
    rt.cipher_persist_engine_report.restype = None

    class Stats(ctypes.Structure):
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
    rt.cipher_persist_engine_stats.argtypes = [ctypes.POINTER(Stats)]
    rt.cipher_persist_engine_stats.restype = ctypes.c_int

    rt.cipher_persist_engine_init()
    return rt, Stats


def _stats(rt, Stats):
    s = Stats()
    rt.cipher_persist_engine_stats(ctypes.byref(s))
    return s


def main():
    rt, Stats = _load_engine()

    if not rt.cipher_persist_engine_enabled():
        print("FAIL  — engine not enabled (CIPHER_PERSIST_ENGINE not set?)")
        sys.exit(1)

    s0 = _stats(rt, Stats)
    print(f"[init] budget={s0.budget_bytes} bytes "
          f"({s0.budget_bytes / 1024 / 1024:.1f} MB)")
    if s0.budget_bytes <= 0:
        print(f"FAIL  — budget is zero (silicon model didn't populate l2_persist_max)")
        sys.exit(1)

    # ── Test 1: fits-in-budget admit ────────────────────────────────────────
    # Three regions, 4 MB each. Total 12 MB << 31.2 MB budget. All admitted.
    pA, pB, pC = 0x1000, 0x2000, 0x3000
    BYTES = 4 * 1024 * 1024
    rt.cipher_persist_engine_register(pA, BYTES, 1.0)
    rt.cipher_persist_engine_register(pB, BYTES, 1.0)
    rt.cipher_persist_engine_register(pC, BYTES, 1.0)
    s1 = _stats(rt, Stats)
    expect = 3
    if s1.registered_count != expect or s1.admitted_count != expect:
        print(f"FAIL  fits-in-budget: registered={s1.registered_count} "
              f"admitted={s1.admitted_count} expected {expect}/{expect}")
        sys.exit(1)
    if s1.admitted_bytes != 3 * BYTES:
        print(f"FAIL  admitted_bytes={s1.admitted_bytes} expected {3 * BYTES}")
        sys.exit(1)
    print(f"[fits-in-budget] PASS  registered={s1.registered_count} "
          f"admitted={s1.admitted_count} bytes={s1.admitted_bytes}")

    # ── Test 2: fractional knapsack — budget cap forces partial admit ───────
    # Add a 40 MB region (> budget) that's the densest. Lower-density
    # neighbors get evicted to make room; the dense one is partially admitted.
    rt.cipher_persist_engine_unregister(pA)
    rt.cipher_persist_engine_unregister(pB)
    rt.cipher_persist_engine_unregister(pC)

    # Two regions: r1 small + high score, r2 huge + medium score.
    pSmall = 0x4000
    pHuge  = 0x5000
    rt.cipher_persist_engine_register(pSmall, 8 * 1024 * 1024, 100.0)  # 8 MB, score 100 → density 12.5
    rt.cipher_persist_engine_register(pHuge, 64 * 1024 * 1024, 1.0)    # 64 MB, score 1 → density 0.0156
    s2 = _stats(rt, Stats)
    if s2.registered_count != 2:
        print(f"FAIL  registered_count={s2.registered_count} expected 2")
        sys.exit(1)
    # Small admitted in full (8 MB). Remaining = budget - 8 MB. Huge gets
    # remaining/64MB fractional admit.
    if s2.admitted_count != 2:
        print(f"FAIL  admitted_count={s2.admitted_count} expected 2 (one full + one fractional)")
        sys.exit(1)
    expected_admitted_bytes = s2.budget_bytes  # full budget consumed
    if s2.admitted_bytes != expected_admitted_bytes:
        print(f"FAIL  admitted_bytes={s2.admitted_bytes} expected {expected_admitted_bytes}")
        sys.exit(1)
    print(f"[fractional knapsack] PASS  small fully admitted + huge fractional, "
          f"total {s2.admitted_bytes} bytes = budget")

    # ── Test 3: unregister updates count ────────────────────────────────────
    rt.cipher_persist_engine_unregister(pSmall)
    rt.cipher_persist_engine_unregister(pHuge)
    s3 = _stats(rt, Stats)
    if s3.registered_count != 0 or s3.admitted_count != 0:
        print(f"FAIL  unregister: registered={s3.registered_count} "
              f"admitted={s3.admitted_count}")
        sys.exit(1)
    print(f"[unregister] PASS  registered={s3.registered_count} admitted={s3.admitted_count}")

    # ── Test 4: report writes valid JSON ────────────────────────────────────
    rt.cipher_persist_engine_register(pA, BYTES, 5.0)
    rt.cipher_persist_engine_register(pB, BYTES, 2.0)
    rt.cipher_persist_engine_report()
    path = "/tmp/cipher_persist_engine_report.json"
    with open(path) as f:
        payload = json.load(f)
    if not (payload["enabled"] == 1 and payload["registered_count"] == 2
            and payload["admitted_count"] == 2 and len(payload["regions"]) == 2):
        print(f"FAIL  report payload: {payload}")
        sys.exit(1)
    print(f"[report] PASS  enabled={payload['enabled']} "
          f"regions={payload['registered_count']} bytes={payload['admitted_bytes']}")

    # ── Test 5: recompute_budget is idempotent ──────────────────────────────
    s_pre  = _stats(rt, Stats)
    rt.cipher_persist_engine_recompute_budget()
    rt.cipher_persist_engine_recompute_budget()
    s_post = _stats(rt, Stats)
    if (s_pre.admitted_count != s_post.admitted_count
            or s_pre.admitted_bytes != s_post.admitted_bytes):
        print(f"FAIL  recompute changed admit set: "
              f"pre admitted_count={s_pre.admitted_count} bytes={s_pre.admitted_bytes} "
              f"post admitted_count={s_post.admitted_count} bytes={s_post.admitted_bytes}")
        sys.exit(1)
    print(f"[recompute idempotent] PASS")

    print("\nResult: 5/5")


if __name__ == "__main__":
    main()
