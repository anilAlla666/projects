"""V1 Phase A.2 hard gate: cipher_rt_cublas_shim_calls in worker >= 11000
without cipher-vllm-kv plugin installed.

Per V1_PHASE_A_SCOPE_LOCK.md §4.A.2 (HARD GATE per Anil scope verbatim).

Pre-conditions (verified at entry, fail fast if violated):
  1. cipher-vllm-kv NOT installed: pip show returns non-zero
  2. vllm.general_plugins entry point absent: importlib.metadata returns []
  3. CUDA_INJECTION64_PATH not set (LD_PRELOAD-only path under test)
  4. LD_PRELOAD set to /home/ubuntu/cipher_rt_phase4/libcipher_rt.so
  5. CIPHER_RT_COUNTER_DUMP_PATH set (env-gates the substrate counter dump
     mechanism added in A.1)

Workload: TinyLlama-1.1B-Chat-v1.0 FP16 vLLM V1 N=1 128-token decode.

Hard gate: worker subprocess writes cipher_rt_<worker_pid>.json to
CIPHER_RT_COUNTER_DUMP_PATH at atexit; harness reads the JSON and checks
cipher_rt_cublas_shim_calls >= 11000 (Option 2 Step 0 baseline 11658,
noise margin 5.6%).

Soft gates:
  - cipher_rt_matmul_calls_total >= shim_calls (matmul dispatch fired)
  - cipher_rt_koopman_calls_total == 0 (koopman engine off, expected)

Exit codes:
  0 = HARD GATE PASS
  1 = HARD GATE FAIL (surface to Anil per scope-lock §4.A.2)
  2 = pre-condition violation (must fix env before re-running)
"""
import argparse
import glob
import importlib.metadata
import json
import os
import subprocess
import sys
import time


def fail_precondition(msg):
    print(f"[verify-no-plugin] PRECONDITION FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--model", default="/home/ubuntu/.cache/huggingface/hub/"
                    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-shim-calls", type=int, default=11000,
                    help="HARD GATE: cipher_rt_cublas_shim_calls in worker must be >= this")
    ap.add_argument("--allow-plugin-installed", action="store_true",
                    help="DEBUG: skip pip pre-condition (e.g. when using VLLM_PLUGINS empty as proxy)")
    ap.add_argument("--cuda-injection-path", action="store_true",
                    help="Branch D (a): allow CUDA_INJECTION64_PATH deployment (vLLM V1 path)")
    args = ap.parse_args()

    if os.path.isdir(args.model):
        snaps = sorted(glob.glob(os.path.join(args.model, "*")))
        if snaps:
            args.model = snaps[0]

    print(f"[verify-no-plugin] pre-condition check pid={os.getpid()}", file=sys.stderr)

    # 1. cipher-vllm-kv NOT installed in THIS interpreter's site-packages.
    # Use sys.executable -m pip to query the same Python that vLLM runs in,
    # not the shell's PATH-resolved pip (which may belong to a different env).
    if not args.allow_plugin_installed:
        rc = subprocess.run([sys.executable, "-m", "pip", "show", "cipher-vllm-kv"],
                            capture_output=True).returncode
        if rc == 0:
            fail_precondition(f"{sys.executable} -m pip show cipher-vllm-kv returned 0; uninstall first")

    # 2. vllm.general_plugins entry point absent
    eps = list(importlib.metadata.entry_points(group="vllm.general_plugins"))
    cipher_eps = [ep for ep in eps if "cipher" in ep.name.lower()]
    if cipher_eps and not args.allow_plugin_installed:
        fail_precondition(f"cipher entry points still present: {cipher_eps}")

    # 3. + 4. Either CUDA_INJECTION64_PATH (Branch D (a) vLLM V1 path) OR
    # LD_PRELOAD (non-vLLM-V1 path) must point at libcipher_rt.so.
    cuda_inj = os.environ.get("CUDA_INJECTION64_PATH", "")
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if args.cuda_injection_path:
        if "libcipher_rt.so" not in cuda_inj:
            fail_precondition(f"--cuda-injection-path set but CUDA_INJECTION64_PATH does not point at libcipher_rt.so: {cuda_inj!r}")
    else:
        if cuda_inj:
            fail_precondition("CUDA_INJECTION64_PATH is set but --cuda-injection-path not passed; use --cuda-injection-path for Branch D (a) deployment, or unset CUDA_INJECTION64_PATH for LD_PRELOAD-only mode")
        if "libcipher_rt.so" not in ld_preload:
            fail_precondition(f"LD_PRELOAD does not include libcipher_rt.so: {ld_preload!r}")

    # 5. CIPHER_RT_COUNTER_DUMP_PATH set
    dump_path = os.environ.get("CIPHER_RT_COUNTER_DUMP_PATH")
    if not dump_path:
        fail_precondition("CIPHER_RT_COUNTER_DUMP_PATH not set; substrate dump will not fire")
    os.makedirs(dump_path, exist_ok=True)
    # Clear any old dump files so we measure THIS run
    for f in glob.glob(os.path.join(dump_path, "cipher_rt_*.json")):
        os.unlink(f)

    print(f"[verify-no-plugin] pre-conditions OK; LD_PRELOAD={ld_preload}", file=sys.stderr)
    print(f"[verify-no-plugin] VLLM_PLUGINS={os.environ.get('VLLM_PLUGINS', '<unset>')}",
          file=sys.stderr)
    print(f"[verify-no-plugin] dump path={dump_path}", file=sys.stderr)

    # Run vLLM decode
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=0.5,
              max_model_len=2048, enforce_eager=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens,
                        ignore_eos=True, min_tokens=args.tokens)

    per_rep = []
    for rep in range(args.reps):
        t0 = time.perf_counter()
        outs = llm.generate(["Once upon a time"], sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        n_gen = len(outs[0].outputs[0].token_ids)
        per_rep.append({"rep": rep, "elapsed_s": elapsed, "n_gen": n_gen,
                        "tok_s": n_gen / elapsed if elapsed > 0 else 0.0})
        print(f"[verify-no-plugin] rep={rep} elapsed={elapsed:.4f} n_gen={n_gen}",
              file=sys.stderr)

    # Read substrate counters from worker via collective_rpc using an explicit
    # CDLL path. Under CUDA_INJECTION64_PATH the .so is in a separate namespace
    # not visible to RTLD_DEFAULT; explicit path loads into the same address
    # space (refcount-only, no double-mapping of code) and resolves the
    # already-incremented static counters. Under LD_PRELOAD both paths work.
    def _read_substrate(self):
        import ctypes, os as _os
        counters = {}
        try:
            rt = ctypes.CDLL("/home/ubuntu/cipher_rt_phase4/libcipher_rt.so",
                             mode=ctypes.RTLD_GLOBAL)
            for fname in ("cipher_rt_cublas_shim_calls",
                          "cipher_rt_matmul_calls_total",
                          "cipher_rt_matmul_calls_handled",
                          "cipher_rt_matmul_calls_passthrough",
                          "cipher_rt_koopman_calls_total",
                          "cipher_rt_got_slots_patched",
                          "cipher_rt_got_modules_scanned"):
                try:
                    fn = getattr(rt, fname)
                    fn.restype = ctypes.c_uint64
                    fn.argtypes = []
                    counters[fname] = int(fn())
                except AttributeError:
                    counters[fname] = None
        except Exception as exc:
            counters["_load_error"] = str(exc)
        with open(f"/proc/{_os.getpid()}/maps") as f:
            libcipher_rt_loaded = "libcipher_rt.so" in f.read()
        return {"pid": _os.getpid(),
                "libcipher_rt_loaded": libcipher_rt_loaded,
                "counters": counters}

    worker_state = llm.collective_rpc(_read_substrate)

    # Also keep the atexit-dump path for evidence (lands a record per-pid
    # when worker exits gracefully)
    import gc
    del llm
    gc.collect()
    time.sleep(2)

    worker_dumps = []
    for f in sorted(glob.glob(os.path.join(dump_path, "cipher_rt_*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
                d["_file"] = f
                worker_dumps.append(d)
        except Exception as e:
            print(f"[verify-no-plugin] failed to read {f}: {e}", file=sys.stderr)

    # Primary read source: collective_rpc live read (reliable; not subject to
    # SIGKILL'd-before-atexit). dump-file is secondary evidence.
    best_worker = max(worker_state, key=lambda w: w.get("counters", {}).get("cipher_rt_cublas_shim_calls") or 0)
    shim_calls = best_worker["counters"].get("cipher_rt_cublas_shim_calls") or 0
    matmul_total = best_worker["counters"].get("cipher_rt_matmul_calls_total") or 0
    koopman_total = best_worker["counters"].get("cipher_rt_koopman_calls_total") or 0

    gate_pass = shim_calls >= args.gate_shim_calls

    result = {
        "verdict": "HARD_GATE_PASS" if gate_pass else "HARD_GATE_FAIL",
        "gate_shim_calls_required": args.gate_shim_calls,
        "shim_calls_observed": shim_calls,
        "matmul_total_observed": matmul_total,
        "koopman_total_observed": koopman_total,
        "best_worker_pid": best_worker.get("pid"),
        "worker_state_via_collective_rpc": worker_state,
        "per_rep": per_rep,
        "worker_atexit_dumps_secondary": worker_dumps,
        "env": {
            "LD_PRELOAD": ld_preload,
            "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS"),
            "CUDA_INJECTION64_PATH": os.environ.get("CUDA_INJECTION64_PATH"),
            "CIPHER_RT_COUNTER_DUMP_PATH": dump_path,
        },
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[verify-no-plugin] {result['verdict']}", file=sys.stderr)
    print(f"[verify-no-plugin] shim_calls={shim_calls} (gate >= {args.gate_shim_calls})",
          file=sys.stderr)
    print(f"[verify-no-plugin] matmul_total={matmul_total} koopman_total={koopman_total}",
          file=sys.stderr)
    print(f"[verify-no-plugin] wrote {args.out}", file=sys.stderr)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
