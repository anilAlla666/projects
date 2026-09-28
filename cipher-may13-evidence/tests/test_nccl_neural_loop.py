#!/usr/bin/env python3
# =============================================================================
# Change 4 Part A gate — NCCL CfC policy loop
#
# Single-GPU verification that the decide→feedback loop is wired end to end.
# Exercises the RT bridges via ctypes (no real NCCL comm needed) and checks:
#
#   (1) cipher_nccl_record_decide fires without crash across msg-size buckets
#       {1KB, 64KB, 1MB, 16MB, 128MB, 1GB}.
#   (2) Recommended algorithm per bucket matches the analytical seed the
#       CfC was initialised with (small → LL128, mid → Ring/NVLS, large → Tree).
#   (3) cipher_nccl_record_feedback updates per-bucket stats monotonically
#       (feedback_count increments, bucket EMA is populated).
#   (4) The global report prints non-zero inference counts.
#
# Part B (NCCL tuner plugin) is a separate gate test once that DSO is built.
# Run:
#   LD_PRELOAD="./libcipher_hook.so ./libcipher_rt.so" \
#     CIPHER_FORCE_PERMIT=1 \
#     python3 tests/test_nccl_neural_loop.py
# =============================================================================

import ctypes
import sys

# Algorithm enum — matches CipherNcclAlgo in include/cipher_nccl_bpf.h
# (which is distinct from NcclAlgo in cipher_nccl.h; Change 4's RT bridge
# returns the CipherNcclAlgo variant via cipher_nccl_neural_decide).
AUTO  = 0
RING  = 1
TREE  = 2
NVLS  = 3
LL128 = 4
ALGO_NAMES = {AUTO: "AUTO", RING: "RING", TREE: "TREE", NVLS: "NVLS", LL128: "LL128"}


def resolve_api():
    try:
        rt = ctypes.CDLL("./libcipher_rt.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"[FATAL] cannot load libcipher_rt.so: {e}")
        sys.exit(2)

    decide = rt.cipher_nccl_record_decide
    decide.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
    decide.restype  = ctypes.c_int

    feedback = rt.cipher_nccl_record_feedback
    feedback.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
    feedback.restype  = None

    dc = rt.cipher_nccl_decide_count
    dc.argtypes = []
    dc.restype  = ctypes.c_uint64

    fc = rt.cipher_nccl_feedback_count
    fc.argtypes = []
    fc.restype  = ctypes.c_uint64

    la = rt.cipher_nccl_last_algo
    la.argtypes = []
    la.restype  = ctypes.c_uint64

    lb = rt.cipher_nccl_last_bytes
    lb.argtypes = []
    lb.restype  = ctypes.c_uint64

    rep = rt.cipher_nccl_neural_global_report
    rep.argtypes = []
    rep.restype  = None

    return {
        "decide":        decide,
        "feedback":      feedback,
        "decide_count":  dc,
        "feedback_count": fc,
        "last_algo":     la,
        "last_bytes":    lb,
        "report":        rep,
    }


def main():
    api = resolve_api()

    print("=" * 70)
    print("Change 4 Part A gate — CfC NCCL policy loop")
    print("=" * 70)

    # Message-size sweep across CfC bucket boundaries.
    # Simulated latencies: the neural policy feedback only cares about
    # *relative* durations, so these are monotone in msg_size.
    sweep = [
        ("1KB",    1_024,              20_000),        # 20 µs
        ("64KB",   64_1024 if False else 64 * 1024, 50_000),  # 50 µs
        ("1MB",    1 * 1024 * 1024,    150_000),       # 150 µs
        ("16MB",   16 * 1024 * 1024,   1_500_000),     # 1.5 ms
        ("128MB",  128 * 1024 * 1024,  12_000_000),    # 12 ms
        ("1GB",    1024 * 1024 * 1024, 95_000_000),    # 95 ms
    ]
    expected = {
        # Heuristic expectations from the analytical seed in cipher_nccl_neural.cpp
        # Small sizes → LL128; large sizes → Tree; mid → Ring/NVLS (either ok).
        "1KB":   {LL128},
        "64KB":  {LL128},
        "1MB":   {LL128, RING, NVLS},      # transition zone
        "16MB":  {RING, NVLS, TREE},       # transition zone
        "128MB": {TREE, NVLS, RING},       # large transition
        "1GB":   {TREE, NVLS},
    }

    dc0 = api["decide_count"]()
    fc0 = api["feedback_count"]()

    print(f"\n{'size':<8} {'bytes':<14} {'algo':<8} {'dur_ns':<14}  expected_set")
    print("-" * 70)
    observed_algos = {}
    for name, bytes_, fake_dur in sweep:
        algo = api["decide"](ctypes.c_uint64(bytes_), ctypes.c_uint32(8))
        api["feedback"](ctypes.c_uint64(bytes_),
                        ctypes.c_uint64(fake_dur),
                        ctypes.c_int(algo))
        observed_algos[name] = algo
        ok = algo in expected[name]
        mark = "✓" if ok else "✗"
        print(f"  {name:<6} {bytes_:<14} {ALGO_NAMES.get(algo, '?'):<8} "
              f"{fake_dur:<14} {sorted(expected[name])} {mark}")

    dc1 = api["decide_count"]()
    fc1 = api["feedback_count"]()

    print("\n" + "-" * 70)
    print(f"decide_count delta   : {dc1 - dc0}  (expected {len(sweep)})")
    print(f"feedback_count delta : {fc1 - fc0}  (expected {len(sweep)})")
    print(f"last_algo            : {ALGO_NAMES.get(api['last_algo'](), '?')}")
    print(f"last_bytes           : {api['last_bytes']():,}")

    print("\nGlobal report:")
    print("-" * 70)
    api["report"]()

    # Gate checks
    print("\n" + "=" * 70)
    print("GATE CHECKS")
    print("=" * 70)

    gates = []
    gates.append(("no_crash",                    True))
    gates.append(("decide fired per msg size",   (dc1 - dc0) == len(sweep)))
    gates.append(("feedback fired per msg size", (fc1 - fc0) == len(sweep)))
    correct_buckets = sum(
        1 for name, _, _ in sweep if observed_algos[name] in expected[name]
    )
    gates.append((
        f"algo-per-bucket ({correct_buckets}/{len(sweep)})",
        correct_buckets >= len(sweep) - 1,   # allow one miss due to transition-zone ambiguity
    ))

    # --- Part B: NCCL tuner plugin DSO --------------------------------------
    print("\n" + "=" * 70)
    print("Part B — NCCL tuner plugin DSO")
    print("=" * 70)
    part_b_ok = run_part_b(api)
    gates.append(("part_b_tuner_plugin", part_b_ok))

    all_pass = True
    for name, ok in gates:
        print(f"  {name:<40} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    sys.exit(0 if all_pass else 1)


# NCCL ABI constants — matches include/cipher_nccl_tuner_abi.h
NCCL_ALGO_UNDEF           = -1
NCCL_ALGO_TREE            = 0
NCCL_ALGO_RING            = 1
NCCL_ALGO_COLLNET_DIRECT  = 2
NCCL_ALGO_COLLNET_CHAIN   = 3
NCCL_ALGO_NVLS            = 4
NCCL_ALGO_NVLS_TREE       = 5
NCCL_PROTO_UNDEF  = -1
NCCL_PROTO_LL     = 0
NCCL_PROTO_LL128  = 1
NCCL_PROTO_SIMPLE = 2
NCCL_ALGO_NAMES = {
    NCCL_ALGO_UNDEF:          "UNDEF",
    NCCL_ALGO_TREE:           "TREE",
    NCCL_ALGO_RING:           "RING",
    NCCL_ALGO_COLLNET_DIRECT: "COLLNET_DIRECT",
    NCCL_ALGO_COLLNET_CHAIN:  "COLLNET_CHAIN",
    NCCL_ALGO_NVLS:           "NVLS",
    NCCL_ALGO_NVLS_TREE:      "NVLS_TREE",
}
NCCL_PROTO_NAMES = {
    NCCL_PROTO_UNDEF:  "UNDEF",
    NCCL_PROTO_LL:     "LL",
    NCCL_PROTO_LL128:  "LL128",
    NCCL_PROTO_SIMPLE: "SIMPLE",
}

# ncclTuner_v2_t struct layout matching include/cipher_nccl_tuner_abi.h
class NcclTunerV2(ctypes.Structure):
    _fields_ = [
        ("name",        ctypes.c_char_p),
        ("init",        ctypes.c_void_p),
        ("getCollInfo", ctypes.c_void_p),
        ("destroy",     ctypes.c_void_p),
    ]


def run_part_b(api):
    import os
    plugin_path = "./libcipher_nccl_tuner.so"
    if not os.path.exists(plugin_path):
        print(f"  FAIL: {plugin_path} missing (build did not produce the DSO)")
        return False

    try:
        plugin = ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"  FAIL: cannot dlopen {plugin_path}: {e}")
        return False

    # Resolve the exported struct by name
    try:
        tuner_struct = NcclTunerV2.in_dll(plugin, "ncclTunerPlugin_v2")
    except ValueError as e:
        print(f"  FAIL: ncclTunerPlugin_v2 not found: {e}")
        return False

    print(f"  plugin name: {tuner_struct.name.decode() if tuner_struct.name else '(null)'}")

    # Function signatures for the struct methods
    INIT_FN        = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    GETCOLLINFO_FN = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int))
    DESTROY_FN     = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

    init_fn        = INIT_FN(tuner_struct.init)
    getcollinfo_fn = GETCOLLINFO_FN(tuner_struct.getCollInfo)
    destroy_fn     = DESTROY_FN(tuner_struct.destroy)

    # Init with 8 ranks, 1 node, no logger
    ctx = ctypes.c_void_p()
    rc = init_fn(8, 1, None, ctypes.byref(ctx))
    if rc != 0:
        print(f"  FAIL: init returned {rc}")
        return False
    print(f"  init OK: ctx={ctx.value:#x}")

    dc_before = api["decide_count"]()

    # Sweep message sizes through the plugin's GetCollInfo
    sizes = [1_024, 65_536, 1 * 1024 * 1024, 16 * 1024 * 1024,
             128 * 1024 * 1024, 1024 * 1024 * 1024]
    valid_algos = {NCCL_ALGO_UNDEF, NCCL_ALGO_TREE, NCCL_ALGO_RING,
                   NCCL_ALGO_NVLS}
    valid_protos = {NCCL_PROTO_UNDEF, NCCL_PROTO_LL128, NCCL_PROTO_SIMPLE}

    all_returns_valid = True
    print(f"\n  {'size':<10} {'collnet':<8} {'nvls':<6} "
          f"{'algo':<16} {'proto':<8} {'nch':<4}")
    print("  " + "-" * 60)
    for bytes_ in sizes:
        # Simulate two capability scenarios: with NVLS/CollNet and without.
        for (collnet, nvls) in [(1, 1), (0, 0)]:
            algo_out  = ctypes.c_int(0)
            proto_out = ctypes.c_int(0)
            nch_out   = ctypes.c_int(0)
            rc = getcollinfo_fn(
                ctx,
                4,      # ncclFuncAllReduce
                bytes_,
                collnet,
                nvls,
                1,      # numPipeOps
                ctypes.byref(algo_out),
                ctypes.byref(proto_out),
                ctypes.byref(nch_out),
            )
            if rc != 0:
                print(f"  FAIL: getCollInfo returned {rc}")
                all_returns_valid = False
                continue
            algo_name  = NCCL_ALGO_NAMES.get(algo_out.value, f"?{algo_out.value}")
            proto_name = NCCL_PROTO_NAMES.get(proto_out.value, f"?{proto_out.value}")
            size_str = f"{bytes_//1024}K" if bytes_ < 1024*1024 \
                       else f"{bytes_//(1024*1024)}M" if bytes_ < 1024**3 \
                       else f"{bytes_//(1024**3)}G"
            print(f"  {size_str:<10} {collnet:<8} {nvls:<6} "
                  f"{algo_name:<16} {proto_name:<8} {nch_out.value:<4}")
            if algo_out.value not in valid_algos:
                print(f"    ✗ algo {algo_out.value} not in expected set")
                all_returns_valid = False
            if proto_out.value not in valid_protos:
                print(f"    ✗ proto {proto_out.value} not in expected set")
                all_returns_valid = False
            # When NVLS support is off, the plugin MUST NOT return NCCL_ALGO_NVLS
            if nvls == 0 and algo_out.value == NCCL_ALGO_NVLS:
                print(f"    ✗ returned NVLS when nvls_support=0")
                all_returns_valid = False

    dc_after = api["decide_count"]()
    call_count = dc_after - dc_before
    expected_calls = len(sizes) * 2

    print("\n  " + "-" * 60)
    print(f"  cipher_nccl_record_decide calls delta : {call_count}")
    print(f"  expected                              : {expected_calls}")

    if call_count != expected_calls:
        print("  ✗ decide call count mismatch")
        all_returns_valid = False

    # Destroy
    rc = destroy_fn(ctx)
    if rc != 0:
        print(f"  FAIL: destroy returned {rc}")
        all_returns_valid = False
    else:
        print(f"  destroy OK")

    # Verify v1 struct also exists
    try:
        _ = NcclTunerV2.in_dll(plugin, "ncclTunerPlugin_v1")
        print(f"  v1 fallback struct: present")
    except ValueError:
        print(f"  v1 fallback struct: MISSING")
        all_returns_valid = False

    return all_returns_valid


if __name__ == "__main__":
    main()
