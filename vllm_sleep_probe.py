#!/usr/bin/env python3
# ASSESSMENT 1 baseline: measure vLLM's OWN sleep/wake (the N-sleep-instance density mechanism CIPHER must beat).
# sleep(level=1) offloads weights to CPU + frees HBM; wake_up restores. Metrics: load HBM, sleep-freed HBM,
# sleep latency, wake latency (the swap cost an orchestrator pays per model activation), correctness across cycle.
import os, time, torch
os.environ["VLLM_USE_DEEP_GEMM"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
GB = 1 << 30
def used(): f, t = torch.cuda.mem_get_info(); return (t - f) / GB

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    torch.cuda.init()
    base = used()
    t0 = time.time()
    llm = LLM(model="/home/ubuntu/models/Mistral-7B-v0.1", enforce_eager=True,
              gpu_memory_utilization=0.55, max_model_len=2048, dtype="float16",
              enable_sleep_mode=True)
    load_t = time.time() - t0
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out0 = llm.generate(["The capital of France is"], sp)[0].outputs[0].text
    after_load = used()

    t = time.time(); llm.sleep(level=1); sleep_t = time.time() - t
    after_sleep = used()

    t = time.time(); llm.wake_up(); wake_t = time.time() - t
    after_wake = used()
    out1 = llm.generate(["The capital of France is"], sp)[0].outputs[0].text

    print(f"RESULT load_time={load_t:.1f}s  after_load_HBM={after_load:.1f}GiB (base {base:.2f})")
    print(f"RESULT sleep(level=1): latency={sleep_t*1000:.0f}ms  HBM {after_load:.1f}->{after_sleep:.1f}GiB  freed={after_load-after_sleep:.1f}GiB")
    print(f"RESULT wake_up():      latency={wake_t*1000:.0f}ms  HBM {after_sleep:.1f}->{after_wake:.1f}GiB")
    print(f"RESULT correctness across sleep/wake: out0={out0!r} out1={out1!r} same={out0==out1}")
    print(f"RESULT resident-while-asleep HBM = {after_sleep:.2f}GiB (the per-instance footprint that does NOT free)")
    import sys; sys.stdout.flush(); os._exit(0)
