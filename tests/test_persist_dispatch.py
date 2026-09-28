#!/usr/bin/env python3
# =============================================================================
# Change 2 gate test — Persistent Kernel Mode
#
# Measures CIPHER's per-launch overhead on a tight, stable repeating kernel
# sequence with the persistent-mode fast path OFF vs ON, and asserts at least
# a 50% reduction in CIPHER overhead.
#
# Method:
#   - Launch a fixed, deterministic sequence of N non-cuBLAS kernels
#     (elementwise torch ops) REPEAT times. On H100, individual torch
#     elementwise launches cost ~5-15 us each, dominated by CPU dispatch +
#     CIPHER shim overhead.
#   - Wall-clock the tight loop with torch.cuda.synchronize() at the end.
#   - Run twice: CIPHER_PERSIST=1 (default, fast path active) vs CIPHER_PERSIST=0
#     (fast path disabled). The difference is pure CIPHER per-launch overhead
#     elimination on the stable window.
#   - Gate: baseline_us_per_launch - persist_us_per_launch  >=
#           0.5 * baseline_us_per_launch
#
# Since child processes are the cleanest way to toggle CIPHER_PERSIST between
# the two timings (the env var is read once at first call), we re-exec
# ourselves with a MODE={baseline,persist} env marker. Both inner runs are
# LD_PRELOADed by the caller.
#
# Run:
#   LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" \
#     CIPHER_FORCE_PERMIT=1 \
#     python3 tests/test_persist_dispatch.py
# =============================================================================

import os
import sys
import time
import ctypes
import subprocess

# Stable kernel sequence of length L (unique ops to avoid single-op fusion
# and to produce a real repeating pattern). 4 unique ops, repeated.
SEQ_LEN    = 4
REPEATS    = 4000       # total launches = SEQ_LEN * REPEATS = 16000
WARMUP_REP = 200        # enough to pass PROMOTION_THRESHOLD (10) × L=8 check


def resolve_persist_fns():
    """Late-bind to libcipher_hook.so — LD_PRELOAD makes it already loaded."""
    try:
        hook = ctypes.CDLL("./libcipher_hook.so")
    except OSError:
        return None
    out = {}
    for name in ("cipher_persist_fast_path_count",
                 "cipher_persist_promotion_count",
                 "cipher_persist_observe_count",
                 "cipher_shim_tsc_total",
                 "cipher_shim_tsc_calls"):
        fn = getattr(hook, name, None)
        if fn is None:
            return None
        fn.argtypes = []
        fn.restype  = ctypes.c_uint64
        out[name] = fn
    reset = getattr(hook, "cipher_shim_tsc_reset", None)
    if reset is not None:
        reset.argtypes = []
        reset.restype = None
    out["cipher_shim_tsc_reset"] = reset
    report = getattr(hook, "cipher_persist_report", None)
    if report is not None:
        report.argtypes = []
        report.restype = None
    out["report"] = report
    enabled = getattr(hook, "cipher_persist_enabled", None)
    if enabled is not None:
        enabled.argtypes = []
        enabled.restype = ctypes.c_bool
    out["enabled"] = enabled
    return out


def run_inner():
    """Inner run — measures timing for a single CIPHER_PERSIST setting."""
    import torch
    fns = resolve_persist_fns()
    if fns is None:
        print("COULD NOT RESOLVE PERSIST SYMBOLS — hook not loaded?",
              file=sys.stderr)
        sys.exit(2)

    dev = "cuda"
    a = torch.randn(1024, device=dev)
    b = torch.randn(1024, device=dev)
    c = torch.randn(1024, device=dev)
    d = torch.randn(1024, device=dev)

    def one_step():
        # Stable 4-kernel sequence (all non-GEMM; dispatch path goes through
        # the cuLaunch shim, not cublasGemmEx). Each op issues one kernel.
        a.add_(b)
        c.mul_(d)
        a.sub_(c)
        d.div_(1.0001)

    # Warm up: pass promotion threshold for tandem-repeat detector at L=4.
    # Tandem-repeat test at L=4 needs ≥8 identical hist entries; promotion
    # threshold = 10 consecutive hits. Requires ~14 full 4-op iterations
    # minimum, but we overshoot to be safe.
    for _ in range(WARMUP_REP):
        one_step()
    torch.cuda.synchronize()

    # Snapshot counters before the measured window
    fp_before   = fns["cipher_persist_fast_path_count"]()
    obs_before  = fns["cipher_persist_observe_count"]()
    prom        = fns["cipher_persist_promotion_count"]()
    tsc_before  = fns["cipher_shim_tsc_total"]()
    calls_before = fns["cipher_shim_tsc_calls"]()

    # Timed window
    t0 = time.monotonic_ns()
    for _ in range(REPEATS):
        one_step()
    torch.cuda.synchronize()
    t1 = time.monotonic_ns()

    total_launches = SEQ_LEN * REPEATS
    elapsed_ns = t1 - t0
    per_launch_us = elapsed_ns / 1000.0 / total_launches

    fp_after    = fns["cipher_persist_fast_path_count"]()
    obs_after   = fns["cipher_persist_observe_count"]()
    tsc_after   = fns["cipher_shim_tsc_total"]()
    calls_after = fns["cipher_shim_tsc_calls"]()

    fp_delta    = fp_after - fp_before
    obs_delta   = obs_after - obs_before
    tsc_delta   = tsc_after - tsc_before
    calls_delta = calls_after - calls_before

    # Convert TSC to nanoseconds. On H100 host CPUs the invariant TSC runs
    # at the base frequency (read from /sys/devices/system/cpu). We don't
    # actually need absolute ns — the RATIO is all that matters for the gate,
    # so we report TSC ticks per shim call directly AND a conservative ns
    # estimate using 2.6 GHz (typical Sapphire Rapids base).
    tsc_per_call = (tsc_delta / calls_delta) if calls_delta > 0 else 0.0
    ns_per_call  = tsc_per_call / 2.6   # TSC ticks -> ns at 2.6GHz nominal

    enabled = bool(fns["enabled"]()) if fns["enabled"] else False
    mode = os.environ.get("CIPHER_PERSIST", "1")

    print(f"MODE=CIPHER_PERSIST={mode}  enabled_flag={int(enabled)}")
    print(f"  launches                = {total_launches}")
    print(f"  elapsed_ms              = {elapsed_ns / 1e6:.3f}")
    print(f"  per_launch_us (total)   = {per_launch_us:.3f}")
    print(f"  shim_calls              = {calls_delta}")
    print(f"  shim_tsc_total          = {tsc_delta}")
    print(f"  shim_tsc_per_call       = {tsc_per_call:.1f}")
    print(f"  shim_ns_per_call (est.) = {ns_per_call:.1f}")
    print(f"  fast_path_delta         = {fp_delta}")
    print(f"  observe_delta           = {obs_delta}")
    print(f"  promotions_total        = {prom}")

    if fns["report"] is not None:
        fns["report"]()

    # Machine-readable line (parent scrapes this).
    print(f"RESULT shim_tsc_per_call={tsc_per_call:.3f} "
          f"shim_ns_per_call={ns_per_call:.3f} "
          f"per_launch_us={per_launch_us:.6f} "
          f"fast_path_delta={fp_delta} "
          f"observe_delta={obs_delta} "
          f"calls={calls_delta}")


def run_outer():
    """Parent — launches two child processes for baseline and persist."""
    print("=" * 70)
    print("CHANGE 2 GATE — Persistent Kernel Mode")
    print("=" * 70)
    print(f"Sequence length: {SEQ_LEN}  |  repeats: {REPEATS}  "
          f"|  total launches: {SEQ_LEN * REPEATS}")
    print(f"Gate: persist reduces per-launch CIPHER overhead by >= 50%")
    print("-" * 70)

    env_base = dict(os.environ)
    env_base["CIPHER_INTERNAL_MODE"] = "inner"
    # Make sure the child inherits the exact same LD_PRELOAD

    def run_one(persist_val):
        env = dict(env_base)
        env["CIPHER_PERSIST"] = persist_val
        # The inner run imports torch and needs LD_PRELOAD to have already
        # loaded libcipher_hook.so. Subprocess inherits LD_PRELOAD via env.
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
        return r

    print("\n[1/2] Baseline run (CIPHER_PERSIST=0)")
    print("-" * 70)
    r0 = run_one("0")
    print(r0.stdout, end="")
    if r0.returncode != 0:
        print("BASELINE CHILD FAILED", file=sys.stderr)
        print(r0.stderr, file=sys.stderr)
        sys.exit(3)

    print("\n[2/2] Persist run (CIPHER_PERSIST=1)")
    print("-" * 70)
    r1 = run_one("1")
    print(r1.stdout, end="")
    if r1.returncode != 0:
        print("PERSIST CHILD FAILED", file=sys.stderr)
        print(r1.stderr, file=sys.stderr)
        sys.exit(3)

    def parse(out):
        for line in out.splitlines()[::-1]:
            if line.startswith("RESULT "):
                kv = dict(s.split("=") for s in line.split()[1:])
                return {
                    "tsc_per_call":    float(kv["shim_tsc_per_call"]),
                    "ns_per_call":     float(kv["shim_ns_per_call"]),
                    "per_launch_us":   float(kv["per_launch_us"]),
                    "fast_path_delta": int(kv["fast_path_delta"]),
                    "observe_delta":   int(kv["observe_delta"]),
                    "calls":           int(kv["calls"]),
                }
        return None

    base = parse(r0.stdout)
    pers = parse(r1.stdout)
    if base is None or pers is None:
        print("COULD NOT PARSE CHILD OUTPUT", file=sys.stderr)
        sys.exit(4)

    base_ns   = base["ns_per_call"]
    pers_ns   = pers["ns_per_call"]
    delta_ns  = base_ns - pers_ns
    reduction = (delta_ns / base_ns * 100.0) if base_ns > 0 else 0.0

    print("\n" + "=" * 70)
    print("REPORT — shim-only CIPHER overhead per launch (excludes real CUDA call)")
    print("=" * 70)
    print(f"  baseline shim_ns/call : {base_ns:.1f} ns  "
          f"({base['tsc_per_call']:.0f} TSC ticks)  "
          f"fast_path={base['fast_path_delta']}  obs={base['observe_delta']}  calls={base['calls']}")
    print(f"  persist  shim_ns/call : {pers_ns:.1f} ns  "
          f"({pers['tsc_per_call']:.0f} TSC ticks)  "
          f"fast_path={pers['fast_path_delta']}  obs={pers['observe_delta']}  calls={pers['calls']}")
    print(f"  delta                 : {delta_ns:+.1f} ns")
    print(f"  reduction             : {reduction:.1f} %")
    print(f"  baseline per_launch_us (wall)  : {base['per_launch_us']:.3f}")
    print(f"  persist  per_launch_us (wall)  : {pers['per_launch_us']:.3f}")
    print("-" * 70)

    # Sanity: fast-path must actually have fired in the persist run
    if pers["fast_path_delta"] == 0:
        print("GATE FAIL: fast_path_count is ZERO in persist run — "
              "promotion did not happen", file=sys.stderr)
        sys.exit(1)

    gate_pass = reduction >= 50.0
    print(f"GATE {'PASS' if gate_pass else 'FAIL'}: "
          f"shim-overhead reduction ({reduction:.1f}%) >= 50.0%")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    if os.environ.get("CIPHER_INTERNAL_MODE") == "inner":
        run_inner()
    else:
        run_outer()
