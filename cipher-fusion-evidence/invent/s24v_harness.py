# Unified VARIED-PROMPT cell harness (continuation session 2026-06-12).
# Merges s24_harness.py (dense / fp8 / spec) and s24g_harness.py (torchao route) so every cell of
# the in-harness comparison grid runs the IDENTICAL prompt set (same seeds/order as s24_g2v).
# torchao mechanics carried over verbatim: __main__ guard (spawn) + disclosed monkeypatch forcing
# the ONLINE-quantize path (checkpoint is plain bf16; vLLM marks any "torchao" quant_method as
# pre-serialized).
import os, time, json, threading, subprocess
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

QUANT = os.environ.get("S_QUANT", "")  # "", "fp8", "torchao"
if QUANT == "torchao":
    from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
    _orig_from_config = TorchAOConfig.from_config.__func__
    def _patched_from_config(cls, config):
        obj = _orig_from_config(cls, config)
        obj.is_checkpoint_torchao_serialized = False
        return obj
    TorchAOConfig.from_config = classmethod(_patched_from_config)

from vllm import LLM, SamplingParams

def main():
    M = os.environ["S_MODEL"]; NREQ = int(os.environ.get("S_NREQ", "32")); OUT = os.environ["S_OUT"]
    pw, run = [], threading.Event()
    def smp():
        while True:
            if run.is_set():
                try: pw.append(float(subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()))
                except: pass
            time.sleep(0.1)
    kw = dict(model=M, enforce_eager=os.environ.get("S_EAGER","0")=="1",
              gpu_memory_utilization=0.82, max_model_len=2048, disable_log_stats=True)
    if QUANT: kw["quantization"] = QUANT
    if os.environ.get("S_SPEC","0")=="1":
        kw["speculative_config"]={"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":5}
    try:
        llm = LLM(**kw)
    except Exception as e:
        json.dump({"model": M, "quant": QUANT, "nreq": NREQ, "error": str(e)[:300]}, open(OUT, "w")); raise SystemExit
    seeds = ["The history of the Roman Empire begins with",
             "Photosynthesis is the process by which plants",
             "In 1969, the Apollo program achieved",
             "The fundamental theorem of calculus states that",
             "Deep in the Amazon rainforest, researchers discovered"]
    prompts = [seeds[i % len(seeds)] for i in range(NREQ)]
    sp = SamplingParams(max_tokens=256, temperature=0.0, ignore_eos=True, min_tokens=256)
    llm.generate(prompts, sp, use_tqdm=False)
    th = threading.Thread(target=smp, daemon=True); th.start()
    best = None; bp = None
    for _ in range(3):
        pw.clear(); run.set(); t0 = time.perf_counter(); o = llm.generate(prompts, sp, use_tqdm=False); dt = time.perf_counter() - t0; run.clear()
        g = sum(len(x.outputs[0].token_ids) for x in o); tps = g / dt; mp = sum(pw)/len(pw) if pw else None
        if best is None or tps > best: best, bp = tps, mp
    res = {"model": M, "quant_arg": QUANT or None, "spec": os.environ.get("S_SPEC","0")=="1",
           "nreq": NREQ, "eager": kw["enforce_eager"], "promptset": "varied",
           "plugins_env": os.environ.get("VLLM_PLUGINS", "UNSET"), "tok_s": round(best, 1),
           "tok_w": round(best/bp, 4) if bp else None, "mean_w": round(bp, 1) if bp else None,
           "quant": str(getattr(llm.llm_engine.vllm_config.model_config, 'quantization', None)),
           "out": o[0].outputs[0].text[:80]}
    json.dump(res, open(OUT, "w")); print("S24V", json.dumps(res))

if __name__ == "__main__":
    main()
