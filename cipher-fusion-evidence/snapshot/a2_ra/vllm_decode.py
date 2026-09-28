#!/usr/bin/env python3
# Minimal REAL vLLM eager decode driver (real model, real KV cache, continuous batching, real attention).
# Used for: (a) discovery (which cuBLAS entry fires), (b) the detector/injection/0-FP runs, (c) tok/s.
# enforce_eager so the cuBLAS interceptor sees every steady-state decode GEMM (cudagraph would replay them).
import sys, os, time, json
MODE=sys.argv[1] if len(sys.argv)>1 else "decode"   # decode | bench
EAGER = os.environ.get("RV_EAGER","1")=="1"
MODEL = os.environ.get("RV_MODEL","mistralai/Mistral-7B-v0.1")
B = int(os.environ.get("RV_BATCH","8")); OUT=int(os.environ.get("RV_OUT","64"))
os.environ.setdefault("VLLM_LOGGING_LEVEL","WARNING")
os.environ["VLLM_PLUGINS"] = ""  # CRITICAL substrate-line guard: cipher_vllm_kv/cipher_vllm_kvdedup auto-load via vllm.general_plugins entry points (easy-install.pth -> /home/ubuntu/cipher_vllm_plugin) and pull the cipher_v2 substrate into the engine; empty = load no plugins

from vllm import LLM, SamplingParams

def _relock_clocks():
    # The -lgc 1980 lock is observed to release during vLLM engine init on this box (driver 580.105.08);
    # re-apply right before the measured window so the achieved clock is the locked one. Fail-soft.
    import subprocess
    try: subprocess.run(["sudo","-n","nvidia-smi","-lgc","1980,1980"], capture_output=True, timeout=10)
    except Exception: pass

llm = LLM(model=MODEL, enforce_eager=EAGER, gpu_memory_utilization=0.85, max_model_len=2048,
          dtype="float16", disable_log_stats=True)
prompts=["The history of computing began "*4]*B
sp=SamplingParams(max_tokens=OUT, min_tokens=OUT, ignore_eos=True, temperature=0.0)
_relock_clocks()
llm.generate(prompts, sp, use_tqdm=False)   # warmup (also lets the shim see steady-state decode)
_relock_clocks(); time.sleep(0.5)
if MODE=="bench":
    import threading, pynvml as NV
    NV.nvmlInit(); _H=NV.nvmlDeviceGetHandleByIndex(0); _cl=[]; _st=threading.Event()
    def _csmp():
        while not _st.is_set():
            try: _cl.append(NV.nvmlDeviceGetClockInfo(_H, NV.NVML_CLOCK_SM))
            except Exception: pass
            _st.wait(0.1)
    _t=threading.Thread(target=_csmp,daemon=True); _t.start()
    best=None
    for _ in range(2):
        t0=time.perf_counter(); o=llm.generate(prompts,sp,use_tqdm=False); dt=time.perf_counter()-t0
        gt=sum(len(x.outputs[0].token_ids) for x in o); tps=gt/dt
        best=tps if best is None or tps>best else best
    _st.set(); _t.join(timeout=1); cs=sorted(_cl)
    print("BENCH",json.dumps({"model":MODEL,"eager":EAGER,"batch":B,"out":OUT,"decode_tok_s":round(best,1),
        "sm_clock_min_med_max":[cs[0],cs[len(cs)//2],cs[-1]] if cs else None}))
else:
    o=llm.generate(prompts,sp,use_tqdm=False)
    print("DECODE done, tokens:",sum(len(x.outputs[0].token_ids) for x in o))
