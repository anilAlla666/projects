"""W14-followup option-2 Step 1 alpha — BF16-hypothesis direct confirmation.

Pre-condition: Step 0.5 outcome (a) localized the drop to maybe_handle_koopman
early-exits but did not directly identify which one. Strong inference (vLLM
log dtype=torch.bfloat16) pointed to the FP16-only dtype gate.

This step adds 3 per-early-exit counters to cipher_rt_koopman_engine.cpp:
  - cipher_rt_koopman_skip_dtype   (FP16-only check fails — BF16 hypothesis)
  - cipher_rt_koopman_skip_dim     (degenerate dimension check)
  - cipher_rt_koopman_skip_nullptr (null B or C pointer)

The g_enabled early-exit deliberately has NO counter — Step 0.5 §5
eliminated it via the engine-registered worker stderr signal.

Substrate change (rotates libcipher_rt anchor):
  cipher_rt_koopman_engine.cpp: +3 atomics + 3 fetch_adds + 3 getters
  libcipher_rt.so md5: 097cf8d9 → 4bedf648

Plugin change:
  cipher_vllm_kv.py: +3 names to existing dump tuple
  md5: 305003d5 → 8330506a

Expected outcome (direct BF16 confirmation):
  skip_dtype ≈ 11658  (matches Step 0.5 shim/m_total)
  skip_dim   = 0
  skip_nullptr = 0
  koopman_calls_total = 0  (unchanged from Step 0.5)

If skip_dtype ≈ 11658: Branch B close-out fires on FP16-vs-BF16 dtype-gate
finding; no need to investigate dim or nullptr further.
"""
import asyncio
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

DUMP_DIR = "/tmp/option2_step1_alpha_probe/worker_dumps"
os.environ["CIPHER_VLLM_COUNTER_DUMP_PATH"] = DUMP_DIR
os.environ.setdefault("CIPHER_KV_ALLOC", "0")
os.environ.setdefault("CIPHER_KOOPMAN", "1")
os.environ.setdefault("CIPHER_KOOPMAN_OOD_THRESHOLD", "0.99")
os.environ.setdefault("CIPHER_REMEMBER", "0")
os.environ.setdefault("CIPHER_REGISTER_MODEL", "0")
os.environ.setdefault("CIPHER_REGISTER_STREAMS", "0")
os.environ.setdefault("CIPHER_EDMD_LIVE", "0")
os.environ.setdefault("CIPHER_KV_OFFLOAD", "0")
os.environ.setdefault("CIPHER_L2_PERSIST", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

print(f"[step1_alpha] env LD_PRELOAD={os.environ.get('LD_PRELOAD','')}")
print(f"[step1_alpha] env CIPHER_KOOPMAN={os.environ.get('CIPHER_KOOPMAN')}")
print(f"[step1_alpha] env CIPHER_KOOPMAN_OOD_THRESHOLD={os.environ.get('CIPHER_KOOPMAN_OOD_THRESHOLD')}")
print(f"[step1_alpha] DUMP_DIR={DUMP_DIR}")

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
    "cipher_rt_koopman_skip_dtype",
    "cipher_rt_koopman_skip_dim",
    "cipher_rt_koopman_skip_nullptr",
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
    print(f"[step1_alpha] AsyncLLMEngine.from_engine_args({MODEL})...")
    engine = AsyncLLMEngine.from_engine_args(args)
    my_pid = os.getpid()
    time.sleep(2)
    seen_pids_pre = set(int(Path(f).stem.split('_')[1]) for f in glob.glob(f"{DUMP_DIR}/worker_*.json"))
    print(f"[step1_alpha] dumper pids pre-decode: {sorted(seen_pids_pre)}")
    for pid in seen_pids_pre:
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass
    time.sleep(0.5)

    print(f"[step1_alpha] decoding {N_TOKENS} tokens...")
    sp = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
    t0 = time.perf_counter()
    out_text = ""
    async for output in engine.generate(PROMPT, sp, request_id="step1_alpha-probe"):
        out_text = output.outputs[0].text
    t1 = time.perf_counter()
    print(f"[step1_alpha] decode done {t1-t0:.2f}s; first 80 chars: {out_text[:80]!r}")

    time.sleep(0.5)
    seen_pids_post = set(int(Path(f).stem.split('_')[1]) for f in glob.glob(f"{DUMP_DIR}/worker_*.json"))
    print(f"[step1_alpha] dumper pids post-decode: {sorted(seen_pids_post)}")
    for pid in seen_pids_post:
        try:
            os.kill(pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass
    time.sleep(0.5)

    print(f"[step1_alpha] shutting down engine...")
    del engine
    time.sleep(2)

    # Parse dumps
    print(f"\n[step1_alpha] === PROBE EVALUATION ===")
    dumps = {}
    for f in sorted(glob.glob(f"{DUMP_DIR}/worker_*.json")):
        pid = int(Path(f).stem.split('_')[1])
        try:
            dumps[pid] = json.load(open(f))
        except Exception as e:
            print(f"  could not parse {f}: {e}")
    print(f"[step1_alpha] worker dump count: {len(dumps)}; pids {sorted(dumps.keys())}")

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
        print(f"\n[step1_alpha] NO WORKER DUMPS — probe could not measure")
        return 2

    # ---- Alpha-hypothesis adjudication ----
    print(f"\n[step1_alpha] === ALPHA-HYPOTHESIS ADJUDICATION ===")
    w = max(worker_only, key=lambda x: x["deltas"]["cipher_rt_cublas_shim_calls"])
    d = w["deltas"]
    shim   = d["cipher_rt_cublas_shim_calls"]
    mtotal = d["cipher_rt_matmul_calls_total"]
    mpass  = d["cipher_rt_matmul_calls_passthrough"]
    ktotal = d["cipher_rt_koopman_calls_total"]
    sdtype = d["cipher_rt_koopman_skip_dtype"]
    sdim   = d["cipher_rt_koopman_skip_dim"]
    snull  = d["cipher_rt_koopman_skip_nullptr"]
    sum_skips = sdtype + sdim + snull

    print(f"  decode worker pid={w['pid']} (Δ over {w['duration_s']:.2f}s)")
    print(f"    shim={shim}  m_total={mtotal} m_passthrough={mpass}")
    print(f"    k_total={ktotal}  (Step 0.5 had k_total=0; expected unchanged)")
    print(f"    skip_dtype={sdtype}  skip_dim={sdim}  skip_nullptr={snull}  sum={sum_skips}")
    print(f"    expected: sum_skips + k_total == m_passthrough (== shim, == m_total)")

    # Algebraic identity: every PASSTHROUGH in the worker either came from a
    # pre-counter early-exit (sum_skips) or was a non-Koopman PASSTHROUGH
    # (sum=m_passthrough - sum_skips). With only Koopman registered,
    # m_passthrough should equal sum_skips + (passthroughs that incremented
    # k_total but were rejected downstream by k_skipped — but k_total=0 so
    # those are zero).
    accounting_ok = (sum_skips == mpass)

    if sdtype == mpass and sdim == 0 and snull == 0 and ktotal == 0:
        verdict = "ALPHA CONFIRMED (direct): all 11658 PASSTHROUGH attributed to FP16-only dtype gate"
        branch_b = True
    elif sdtype > 0 and sdtype == sum_skips:
        verdict = "ALPHA CONFIRMED (partial): dtype gate dominates; minor non-dtype skips also present"
        branch_b = True
    elif sdtype > 0 and sdim > 0:
        verdict = "ALPHA MIXED: dtype gate fires but dim check ALSO non-zero — multiple skip causes"
        branch_b = True
    elif sdtype == 0 and (sdim > 0 or snull > 0):
        verdict = "ALPHA REFUTED: dtype gate fires zero times; drop is dim/nullptr instead"
        branch_b = False  # different early-exit; investigate further
    elif sdtype == 0 and sdim == 0 and snull == 0 and ktotal == 0:
        verdict = "ALPHA REFUTED: no early-exit fired; only g_enabled or no actuator-iteration explains it"
        branch_b = False
    else:
        verdict = f"INDETERMINATE: dtype={sdtype} dim={sdim} null={snull} ktotal={ktotal}"
        branch_b = False

    print(f"\n[step1_alpha] VERDICT: {verdict}")
    print(f"[step1_alpha] ACCOUNTING (sum_skips == m_passthrough): {'OK' if accounting_ok else 'MISMATCH — investigate'}")
    print(f"[step1_alpha] BRANCH B READY: {branch_b}")

    result_path = "/tmp/option2_step1_alpha_probe/probe_result.json"
    result = {
        "parent_pid": parent_pid,
        "n_workers": len(worker_only),
        "decode_worker": w,
        "all_workers": worker_only,
        "alpha_verdict": verdict,
        "alpha_confirmed": branch_b,
        "accounting_ok": accounting_ok,
        "env": {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
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
    print(f"[step1_alpha] full result: {result_path}")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
