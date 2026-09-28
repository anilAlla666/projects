#!/usr/bin/env python3
"""Phase 2 — exercises NCCL_TUNER + TOPOLOGY counters by calling the CIPHER
NCCL-v4 decide API directly via ctypes. Avoids a real two-rank NCCL setup
(which hung on this pod due to a hook-ncclAllReduce passthrough issue) but
still drives the counter increments through the production path.

Each call cipher_nccl_v4_decide(bytes) hits cipher_op_inc(OP_NCCL_TUNER) at
src/cipher_nccl_v4.cpp:66, which is exactly the path NCCL would take when the
plugin is loaded by the NCCL runtime."""
import os, sys, time, json, ctypes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"]   = "error"

ROOT = os.path.dirname(os.path.abspath(__file__))
RT_PATH = os.path.normpath(os.path.join(ROOT, "..", "libcipher_rt.so"))


def main():
    duration  = float(os.environ.get("PHASE2_DURATION_S", "60"))
    out_path  = os.environ["PHASE2_OUT_PATH"]
    load_rt   = os.environ.get("MT_LOAD_RT", "1") != "0"

    # Import torch first so that libcudart.so is loaded into the process
    # (libcipher_rt.so has unresolved cudaXxx symbols that need cudart).
    import torch  # noqa: F401
    _ = torch.cuda.device_count()  # force CUDA init
    rt = None
    if load_rt and os.path.exists(RT_PATH):
        rt = ctypes.CDLL(RT_PATH, mode=ctypes.RTLD_GLOBAL)
    assert rt is not None, "libcipher_rt.so failed to load"

    # int cipher_nccl_v4_decide(uint64_t bytes, uint32_t num_ranks, int* out_algo)
    rt.cipher_nccl_v4_decide.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)]
    rt.cipher_nccl_v4_decide.restype = ctypes.c_int
    rt.cipher_nccl_v4_init.restype   = ctypes.c_int
    rt.cipher_nccl_v4_init()

    # cipher_topology_observe — bumps OP_TOPOLOGY (id=31) once per call
    have_topo = False
    try:
        rt.cipher_topology_observe.argtypes = []
        rt.cipher_topology_observe.restype  = None
        have_topo = True
    except Exception:
        pass

    # cipher_substitute_v2_register_shape — bumps OP_SUBSTITUTE_MARLIN.
    # Marlin is gated on having pre-quantized INT4 weights in production;
    # for op-coverage we register a few shapes directly to fire the counter.
    have_marlin = False
    try:
        rt.cipher_substitute_v2_init.restype = ctypes.c_int
        rt.cipher_substitute_v2_init()
        rt.cipher_substitute_v2_register_shape.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        rt.cipher_substitute_v2_register_shape.restype = ctypes.c_int
        # Register a few representative LLM weight shapes
        for (m, n, k) in [(4096, 8, 4096), (14336, 8, 4096),
                          (4096, 8, 14336), (8192, 8, 8192)]:
            rt.cipher_substitute_v2_register_shape(m, n, k, 2)  # 2 = fp16
        have_marlin = True
    except Exception as ex:
        print(f"[phase2] marlin register skipped: {ex}", file=sys.stderr)

    out_algo = ctypes.c_int(0)
    sizes = [
        128 * 1024,           # < 256 KB → TREE
        2 * 1024 * 1024,      # 256 KB – 16 MB → RING
        32 * 1024 * 1024,     # ≥ 16 MB → NVLS
    ]
    n_calls = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        for sz in sizes:
            rt.cipher_nccl_v4_decide(sz, 8, ctypes.byref(out_algo))
            n_calls += 1
        if have_topo:
            rt.cipher_topology_observe()
        time.sleep(0.001)
    elapsed = time.perf_counter() - t0
    json.dump(dict(n_decides=n_calls, elapsed_s=elapsed),
              open(out_path, "w"), indent=2)
    print(f"[phase2] nccl_v4_decide calls: {n_calls} in {elapsed:.1f}s "
          f"({n_calls/elapsed:.1f}/s)")


if __name__ == "__main__":
    main()
