#!/usr/bin/env python3
# READ-ONLY prefill kernel-time decomposition: eager vLLM Mistral-7B prefill, VLLM_PLUGINS="" (clean baseline,
# NO substrate, NO actuator). Uses vLLM's built-in worker torch-profiler (profiles the EngineCore worker's CUDA
# kernels, which the parent process cannot see) and dumps a chrome trace; we parse CUDA kernel events and bucket.
import os, sys, json, glob, time, threading
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_PLUGINS"] = ""          # substrate-line guard: clean baseline profile
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
B   = int(os.environ.get("RV_BATCH", "8"))
T   = int(os.environ.get("RV_PLEN", "2048"))
REPS= int(os.environ.get("RV_REPS", "10"))
TRACE_DIR = "/home/ubuntu/cipher-fusion-evidence/mfu_profile/trace"
os.makedirs(TRACE_DIR, exist_ok=True)

import pynvml as NV
NV.nvmlInit(); H = NV.nvmlDeviceGetHandleByIndex(0)
_cl=[]; _stop=threading.Event()
def _smp():
    while not _stop.is_set():
        try: _cl.append(NV.nvmlDeviceGetClockInfo(H, NV.NVML_CLOCK_SM))
        except Exception: pass
        _stop.wait(0.1)

import random
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import ProfilerConfig
rng = random.Random(20260610)
def randp(n): return TokensPrompt(prompt_token_ids=[rng.randrange(1000,28000) for _ in range(n)])

pc = ProfilerConfig(profiler="torch", torch_profiler_dir=TRACE_DIR,
                    torch_profiler_dump_cuda_time_total=True)
llm = LLM(model="mistralai/Mistral-7B-v0.1", enforce_eager=True, gpu_memory_utilization=0.85,
          max_model_len=T+64, dtype="float16", disable_log_stats=True, enable_prefix_caching=False,
          profiler_config=pc)
sp1 = SamplingParams(max_tokens=1, temperature=0.0)
# warmup (distinct tokens) so the profiled window is steady-state prefill
llm.generate([randp(T) for _ in range(B)], sp1, use_tqdm=False)

th=threading.Thread(target=_smp,daemon=True); th.start()
batches=[[randp(T) for _ in range(B)] for _ in range(REPS)]
llm.start_profile()
t0=time.monotonic()
for bp in batches: llm.generate(bp, sp1, use_tqdm=False)
t1=time.monotonic()
llm.stop_profile()
_stop.set(); th.join(timeout=2)
time.sleep(2)  # let the trace flush
cs=sorted(_cl)
meta={"workload":f"eager vLLM Mistral-7B prefill B={B}xT={T}x{REPS}reps, VLLM_PLUGINS='' clean",
      "window_s":t1-t0,"sm_clock_med":cs[len(cs)//2] if cs else None,
      "sm_clock_min":cs[0] if cs else None,"sm_clock_max":cs[-1] if cs else None,
      "trace_dir":TRACE_DIR}
json.dump(meta,open("/home/ubuntu/cipher-fusion-evidence/mfu_profile/profile_meta.json","w"),indent=1)
print("PROFILE_DONE", json.dumps(meta))
traces=sorted(glob.glob(TRACE_DIR+"/*.json")+glob.glob(TRACE_DIR+"/*.json.gz"))
print("TRACES:", traces)
NV.nvmlShutdown()
