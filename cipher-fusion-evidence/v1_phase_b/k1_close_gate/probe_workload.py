#!/usr/bin/env python3
"""K.1 close gate single-cell probe.

Argv: <cell_name> <kind> <model_or_signal> [B] [ctx]
  kind: positive | negative_loadonly | negative_bare | negative_shutdown | transition | calibration
  positive: spawn vLLM offline, run a short prefill+decode, sample class periodically
  negative_loadonly: load vLLM model but DON'T generate
  negative_bare: bare torch.matmul (no model)
  negative_shutdown: load + 1 token + shutdown
  transition: positive-mode but emit class @ T=2s, T=5s, T=15s, T=30s
  calibration: same as positive but emit confidence + observation counts

Always emits a final JSON line: {"cell": ..., "kind": ..., "class": ..., "obs": ..., "conf": ...}.
"""

import sys, os, json, time, ctypes, traceback

_RT_HANDLE = None
def get_classifier():
    """Find cipher_workload_* symbols via ctypes. CUDA_INJECTION64_PATH opens
    libcipher_rt.so with RTLD_LOCAL so RTLD_DEFAULT cannot see its symbols.
    Open it explicitly — same path returns the existing dlopen handle with
    refcount++, and the new handle exposes its global symbols."""
    global _RT_HANDLE
    if _RT_HANDLE is None:
        rt_path = os.environ.get("CIPHER_RT_PATH",
                                 "/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so")
        try:
            _RT_HANDLE = ctypes.CDLL(rt_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return None, None, None, None
    rt = _RT_HANDLE
    try:
        cls = rt.cipher_workload_class_current
        cls.restype = ctypes.c_uint32
        obs = rt.cipher_workload_observations_count
        obs.restype = ctypes.c_uint64
        cnt = rt.cipher_workload_classifications_count
        cnt.restype = ctypes.c_uint64
        conf = rt.cipher_workload_confidence_current
        conf.restype = ctypes.c_uint32
        return cls, obs, cnt, conf
    except AttributeError as e:
        return None, None, None, None

def snapshot(label="final"):
    cls_fn, obs_fn, cnt_fn, conf_fn = get_classifier()
    if cls_fn is None:
        return {"label": label, "class": -1, "obs": -1, "conf": -1, "cnt": -1, "err": "symbol_not_resolved"}
    return {
        "label": label,
        "class": int(cls_fn()),
        "obs": int(obs_fn()),
        "conf": int(conf_fn()),
        "cnt": int(cnt_fn()),
    }

CLASS_NAMES = {
    0: "UNKNOWN", 1: "A1_AGENTIC_MT", 2: "A2_CONT_BATCHED",
    3: "A3_SINGLE_TENANT", 4: "A4_BATCH_INFERENCE",
    5: "B1_PRE_TRAINING", 6: "B2_FINE_TUNING", 7: "C1_RAG_LONG_CTX",
}

def run_positive(model, B, ctx, kind="positive"):
    snaps = [snapshot("preinit")]
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, gpu_memory_utilization=0.3, enforce_eager=True,
              max_model_len=ctx, enable_prefix_caching=False, disable_log_stats=True)
    snaps.append(snapshot("post_load"))
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    prompts = ["Hello, my name is Bob and I work at a startup."] * B
    t0 = time.time()
    for i in range(5):
        _ = llm.generate(prompts, sp, use_tqdm=False)
        if kind == "transition":
            elapsed = time.time() - t0
            if i == 0:
                snaps.append(snapshot(f"transition_after_iter1_t={elapsed:.1f}s"))
            elif elapsed > 5 and elapsed < 6:
                snaps.append(snapshot(f"transition_t=5s_iter{i}"))
    snaps.append(snapshot("final"))
    return snaps

def run_negative_loadonly(model):
    snaps = [snapshot("preinit")]
    from vllm import LLM
    llm = LLM(model=model, gpu_memory_utilization=0.3, enforce_eager=True,
              max_model_len=512, enable_prefix_caching=False, disable_log_stats=True)
    snaps.append(snapshot("post_load"))
    # idle 8s — exceeds IDLE_MIN_ELAPSED_NS (5s) so idle gate should fire
    time.sleep(8)
    snaps.append(snapshot("final_after_idle_8s"))
    return snaps

def run_negative_bare():
    """Bare torch.matmul — no model, no NCCL, just a few GEMMs."""
    snaps = [snapshot("preinit")]
    import torch
    a = torch.randn(64, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.float16)
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    snaps.append(snapshot("post_20_matmul"))
    time.sleep(6)
    snaps.append(snapshot("final_after_idle_6s"))
    return snaps

def run_negative_shutdown(model):
    snaps = [snapshot("preinit")]
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, gpu_memory_utilization=0.3, enforce_eager=True,
              max_model_len=512, enable_prefix_caching=False, disable_log_stats=True)
    snaps.append(snapshot("post_load"))
    _ = llm.generate(["Hi"], SamplingParams(max_tokens=1), use_tqdm=False)
    snaps.append(snapshot("final_after_1tok"))
    return snaps

def run_long_prefill(model, ctx_target):
    """W.1 negative-regime cell: compute-bound large prefill. Expected
    classifier output: large_prefill_detected=1, volt_engage=0."""
    snaps = [snapshot("preinit")]
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, gpu_memory_utilization=0.45, enforce_eager=True,
              max_model_len=max(ctx_target + 64, 4096),
              enable_prefix_caching=False, disable_log_stats=True)
    snaps.append(snapshot("post_load"))
    # Build a long prompt that will force max_n past 4096 at prefill time.
    long_prompt = ("Repeat: alpha beta gamma delta epsilon zeta eta theta iota "
                   "kappa lambda mu nu xi omicron pi rho sigma tau upsilon. " * 600)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    _ = llm.generate([long_prompt], sp, use_tqdm=False)
    snaps.append(snapshot("final_after_prefill"))
    return snaps

def run_calibration(model, B):
    """Same as positive — emit confidence + obs growth over iters."""
    snaps = [snapshot("preinit")]
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, gpu_memory_utilization=0.3, enforce_eager=True,
              max_model_len=512, enable_prefix_caching=False, disable_log_stats=True)
    snaps.append(snapshot("post_load"))
    sp = SamplingParams(max_tokens=16, temperature=0.0)
    prompts = ["Hi"] * B
    for i in range(8):
        _ = llm.generate(prompts, sp, use_tqdm=False)
        snaps.append(snapshot(f"iter{i+1}"))
    return snaps

def main():
    cell_name = sys.argv[1]
    kind = sys.argv[2]
    model_or_signal = sys.argv[3]
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 512

    print(f"=== K.1 CLOSE-GATE CELL {cell_name} kind={kind} model={model_or_signal} B={B} ctx={ctx} ===", flush=True)

    try:
        if kind == "positive":
            snaps = run_positive(model_or_signal, B, ctx)
        elif kind == "transition":
            snaps = run_positive(model_or_signal, B, ctx, kind="transition")
        elif kind == "calibration":
            snaps = run_calibration(model_or_signal, B)
        elif kind == "negative_loadonly":
            snaps = run_negative_loadonly(model_or_signal)
        elif kind == "negative_bare":
            snaps = run_negative_bare()
        elif kind == "negative_shutdown":
            snaps = run_negative_shutdown(model_or_signal)
        elif kind == "long_prefill":
            snaps = run_long_prefill(model_or_signal, ctx)
        else:
            raise ValueError(f"unknown kind {kind}")
    except Exception as e:
        traceback.print_exc()
        final = {"cell": cell_name, "kind": kind, "model": model_or_signal,
                 "B": B, "ctx": ctx, "err": str(e), "snaps": []}
        print("RESULT_JSON " + json.dumps(final), flush=True)
        sys.exit(1)

    final = {"cell": cell_name, "kind": kind, "model": model_or_signal, "B": B, "ctx": ctx, "snaps": snaps}
    final["class_name"] = CLASS_NAMES.get(snaps[-1]["class"], f"UNK_{snaps[-1]['class']}")
    print("RESULT_JSON " + json.dumps(final), flush=True)

if __name__ == "__main__":
    main()
