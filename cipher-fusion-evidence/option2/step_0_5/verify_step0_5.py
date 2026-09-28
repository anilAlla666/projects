"""W14-followup option-2 Step 0.5 Koopman engine reachability probe.

Pre-condition: Step 0 PASS (worker-subprocess GOT patches confirmed installed
across cipher_rt_cublas_shim). Step 0 found cipher_rt_koopman_calls_total = 0
even at CIPHER_KOOPMAN=1 CIPHER_KOOPMAN_OOD_THRESHOLD=0.99 over an 11658-call
TinyLlama decode burst. This probe localizes WHICH layer drops the call
between cublas_shim (11658 PASS) and koopman_calls_total (0 FAIL).

The probe collects all 7 telemetry counters across the call chain:
  - cipher_rt_cublas_shim_calls       (shim layer; Step 0 = 11658)
  - cipher_rt_matmul_calls_total      (matmul-dispatch entry count)
  - cipher_rt_matmul_calls_handled    (some actuator returned HANDLED)
  - cipher_rt_matmul_calls_passthrough(no actuator handled; fall through)
  - cipher_rt_koopman_calls_total     (koopman actuator entered + passed
                                       early-exit checks; Step 0 = 0)
  - cipher_rt_koopman_calls_handled   (koopman substituted successfully)
  - cipher_rt_koopman_calls_skipped   (shape-not-in-registry skip)

Three diagnostic outcomes per scope-lock Step 0.5 spec:

  (a) Classifier never routes to Koopman.
      matmul_total > 0 AND koopman_total = 0.
      The matmul-dispatch saw the call (so cublas_shim handed it off) but
      koopman's maybe_handle returned PASSTHROUGH via an early-exit BEFORE
      incrementing its own counter (see cipher_rt_koopman_engine.cpp:98-111:
      g_enabled check, FP16-only dtype check, degenerate-dim check,
      null-ptr check). Root cause is classifier eligibility, NOT Koopman
      algorithm. Step 1 β sweep is irrelevant on the substrate as-is.

  (b) Classifier routes BUT Koopman skips at residual gating.
      koopman_total > 0 AND koopman_handled = 0.
      Engine was entered, OOD residual_ratio > threshold, or shape was not
      in the calibration registry. This IS the Branch B preview the
      scope-lock Step 0 verdict line 57 (b) described. Step 1 β sweep is
      the right test.

  (c) Some other layer drops the call.
      matmul_total = 0 OR (matmul_total != shim_calls) OR other anomaly.
      Investigate further before Step 1.

HARD GATE before Step 1: this script's `step_0_5_outcome` field must be one
of {a, b, c} with localized evidence; the `step_1_meaningful` field
adjudicates whether β-sweep is meaningful on substrate as-is.
"""
import asyncio
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

DUMP_DIR = "/tmp/option2_step0_5_probe/worker_dumps"
os.environ["CIPHER_VLLM_COUNTER_DUMP_PATH"] = DUMP_DIR
os.environ.setdefault("CIPHER_KV_ALLOC", "0")  # Step 0 hygiene; bypass KV-bridge
os.environ.setdefault("CIPHER_KOOPMAN", "1")
os.environ.setdefault("CIPHER_KOOPMAN_OOD_THRESHOLD", "0.99")  # always-fire-approx
os.environ.setdefault("CIPHER_REMEMBER", "0")
os.environ.setdefault("CIPHER_REGISTER_MODEL", "0")
os.environ.setdefault("CIPHER_REGISTER_STREAMS", "0")
os.environ.setdefault("CIPHER_EDMD_LIVE", "0")
os.environ.setdefault("CIPHER_KV_OFFLOAD", "0")
os.environ.setdefault("CIPHER_L2_PERSIST", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

print(f"[step0_5] env LD_PRELOAD={os.environ.get('LD_PRELOAD','')}")
print(f"[step0_5] env CUDA_INJECTION64_PATH={os.environ.get('CUDA_INJECTION64_PATH','UNSET')}")
print(f"[step0_5] env CIPHER_KOOPMAN={os.environ.get('CIPHER_KOOPMAN')}")
print(f"[step0_5] env CIPHER_KOOPMAN_OOD_THRESHOLD={os.environ.get('CIPHER_KOOPMAN_OOD_THRESHOLD')}")
print(f"[step0_5] DUMP_DIR={DUMP_DIR}")

Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
for f in glob.glob(f"{DUMP_DIR}/*.json"):
    os.unlink(f)

MODEL = os.environ.get("PROBE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
PROMPT = "Once upon a time, in a kingdom far away, there lived a wise"
N_TOKENS = int(os.environ.get("PROBE_N_TOKENS", "128"))

ALL_COUNTERS = (
    "cipher_rt_cublas_shim_calls",
    "cipher_rt_matmul_calls_total",
    "cipher_rt_matmul_calls_handled",
    "cipher_rt_matmul_calls_passthrough",
    "cipher_rt_koopman_calls_total",
    "cipher_rt_koopman_calls_handled",
    "cipher_rt_koopman_calls_skipped",
)


async def main():
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    args = AsyncEngineArgs(
        model=MODEL,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        disable_log_stats=True,
    )
    print(f"[step0_5] AsyncLLMEngine.from_engine_args({MODEL})...")
    engine = AsyncLLMEngine.from_engine_args(args)

    my_pid = os.getpid()
    time.sleep(2)
    seen_pids_pre = set(int(Path(f).stem.split('_')[1]) for f in glob.glob(f"{DUMP_DIR}/worker_*.json"))
    print(f"[step0_5] dumper pids pre-decode: {sorted(seen_pids_pre)}")

    for pid in seen_pids_pre:
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass
    time.sleep(0.5)

    print(f"[step0_5] decoding {N_TOKENS} tokens...")
    sp = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
    t0 = time.perf_counter()
    out_text = ""
    async for output in engine.generate(PROMPT, sp, request_id="step0_5-probe"):
        out_text = output.outputs[0].text
    t1 = time.perf_counter()
    print(f"[step0_5] decode done {t1-t0:.2f}s; first 80 chars: {out_text[:80]!r}")

    time.sleep(0.5)
    seen_pids_post = set(int(Path(f).stem.split('_')[1]) for f in glob.glob(f"{DUMP_DIR}/worker_*.json"))
    print(f"[step0_5] dumper pids post-decode: {sorted(seen_pids_post)}")
    for pid in seen_pids_post:
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass
    time.sleep(0.5)

    print(f"[step0_5] shutting down engine...")
    del engine
    time.sleep(2)

    # ---- Parse dumps ----
    print(f"\n[step0_5] === PROBE EVALUATION ===")
    dumps = {}
    for f in sorted(glob.glob(f"{DUMP_DIR}/worker_*.json")):
        pid = int(Path(f).stem.split('_')[1])
        try:
            dumps[pid] = json.load(open(f))
        except Exception as e:
            print(f"  could not parse {f}: {e}")
    print(f"[step0_5] worker dump count: {len(dumps)}; pids {sorted(dumps.keys())}")

    parent_pid = my_pid
    workers = []
    for pid, dump in dumps.items():
        is_parent = (pid == parent_pid)
        snaps = dump.get("snapshots", [])
        if not snaps:
            continue
        first = snaps[0]
        last = snaps[-1]
        deltas = {c: (last.get(c) or 0) - (first.get(c) or 0) for c in ALL_COUNTERS}
        absolutes = {c: last.get(c) for c in ALL_COUNTERS}
        workers.append({
            "pid": pid, "is_parent": is_parent,
            "n_snapshots": len(snaps), "deltas": deltas, "absolutes": absolutes,
            "first_tag": first.get("tag"), "last_tag": last.get("tag"),
            "duration_s": last.get("time", 0) - first.get("time", 0),
        })
        print(f"  pid={pid} {'(PARENT)' if is_parent else '(WORKER)'} snaps={len(snaps)}")
        for c in ALL_COUNTERS:
            print(f"    {c:42s} +{deltas[c]:>8d}  (abs={absolutes[c]})")

    worker_only = [w for w in workers if not w["is_parent"]]
    if not worker_only:
        print(f"\n[step0_5] NO WORKER DUMPS — probe could not measure; STOP")
        return 2

    # ---- Outcome classification per scope ----
    print(f"\n[step0_5] === OUTCOME CLASSIFICATION (a/b/c) ===")
    # Pick the worker with the highest cublas activity (the decode worker)
    w = max(worker_only, key=lambda x: x["deltas"]["cipher_rt_cublas_shim_calls"])
    d = w["deltas"]
    shim   = d["cipher_rt_cublas_shim_calls"]
    mtotal = d["cipher_rt_matmul_calls_total"]
    mhand  = d["cipher_rt_matmul_calls_handled"]
    mpass  = d["cipher_rt_matmul_calls_passthrough"]
    ktotal = d["cipher_rt_koopman_calls_total"]
    khand  = d["cipher_rt_koopman_calls_handled"]
    kskip  = d["cipher_rt_koopman_calls_skipped"]

    print(f"  decode worker pid={w['pid']} (Δ over {w['duration_s']:.2f}s)")
    print(f"    shim={shim}  m_total={mtotal} (m_handled={mhand} m_passthrough={mpass})")
    print(f"    k_total={ktotal}  k_handled={khand}  k_skipped={kskip}")

    # Outcome (c) anomalies first
    if mtotal == 0:
        outcome = "c"
        layer_drop = "between cublas_shim and matmul-dispatch (matmul_total=0)"
        step_1 = False
        rationale = ("matmul-dispatch never entered: the cublas shim is not "
                     "calling into cipher_rt_matmul_dispatch_handle. This is "
                     "upstream of Koopman entirely — investigate shim → "
                     "dispatch path before any Koopman work.")
    elif mtotal != shim and mtotal < shim - 10:
        # allow small slack for in-flight calls at snapshot time
        outcome = "c"
        layer_drop = f"between cublas_shim and matmul-dispatch (shim={shim} >> matmul_total={mtotal})"
        step_1 = False
        rationale = (f"shim_calls ({shim}) significantly exceeds matmul_total "
                     f"({mtotal}): the shim is filtering or short-circuiting some "
                     f"calls before they reach matmul-dispatch. Investigate which "
                     f"shim path skips dispatch.")
    elif ktotal == 0:
        # The user spec's outcome (a)
        outcome = "a"
        layer_drop = ("between matmul-dispatch and koopman g_calls_total increment "
                      "(inside maybe_handle_koopman's pre-counter early-exits)")
        step_1 = False
        rationale = (
            "matmul-dispatch saw all calls (m_total=shim) but the Koopman "
            "actuator's maybe_handle returned PASSTHROUGH before its "
            "g_calls_total increment. The early-exits at "
            "cipher_rt_koopman_engine.cpp:98-111 are: g_enabled check, "
            "FP16-only dtype check (Atype/Btype/Ctype == CUDA_R_16F), "
            "degenerate-dim check, null-ptr check. Root cause is classifier "
            "eligibility (most likely the FP16-only dtype check filtering "
            "BF16 tensors — TinyLlama in vLLM defaults to bfloat16). Step 1 "
            "β-sweep is NOT meaningful on the substrate as-is; sweep "
            "classifier-eligibility (dtype gate) instead.")
    elif khand == 0 and (kskip > 0 or ktotal > 0):
        # The user spec's outcome (b)
        outcome = "b"
        layer_drop = ("inside Koopman engine: OOD residual gate or shape-not-in-"
                      "registry skip after entering maybe_handle_koopman")
        step_1 = True
        rationale = (
            "Koopman engine entered (k_total>0) but never substituted "
            "(k_handled=0). Either OOD residual_ratio > threshold (g_calls_ood "
            "path; substrate ships threshold=0.05 default), or the shape was "
            "not in the .cu kernel registry (g_calls_skipped path; "
            "cipher_koopman_fp16_launch_shape rc!=0). This is the Branch B "
            "preview scope-lock §4 Step 0 line 57 (b) described. Step 1 "
            "β-sweep IS the right test.")
    else:
        outcome = "c"
        layer_drop = "indeterminate"
        step_1 = False
        rationale = (f"Unexpected combination: m_total={mtotal} k_total={ktotal} "
                     f"k_handled={khand} k_skipped={kskip}; manual investigation "
                     f"needed before Step 1.")

    print(f"\n[step0_5] OUTCOME: ({outcome})")
    print(f"[step0_5] LAYER DROP: {layer_drop}")
    print(f"[step0_5] STEP 1 MEANINGFUL ON SUBSTRATE AS-IS: {step_1}")
    print(f"[step0_5] RATIONALE: {rationale}")

    # ---- Persist ----
    result_path = "/tmp/option2_step0_5_probe/probe_result.json"
    result = {
        "parent_pid": parent_pid,
        "n_workers": len(worker_only),
        "decode_worker": w,
        "all_workers": worker_only,
        "step_0_5_outcome": outcome,
        "layer_drop": layer_drop,
        "step_1_meaningful": step_1,
        "rationale": rationale,
        "env": {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
            "CUDA_INJECTION64_PATH": os.environ.get("CUDA_INJECTION64_PATH", ""),
            "CIPHER_KOOPMAN": os.environ.get("CIPHER_KOOPMAN"),
            "CIPHER_KOOPMAN_OOD_THRESHOLD": os.environ.get("CIPHER_KOOPMAN_OOD_THRESHOLD"),
        },
        "model": MODEL,
        "n_tokens_requested": N_TOKENS,
        "decode_wall_s": t1 - t0,
        "decode_output_first_80": out_text[:80],
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[step0_5] full result: {result_path}")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
