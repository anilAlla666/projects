# Driver for the cudagraph-safe FP8 seam gate. DEFAULT config: multiprocessing ON (do NOT disable),
# enforce_eager=False (torch.compile + cudagraph ON). Runs clean gen, then triggers in-worker inject,
# then a second gen, then collects the worker's out-of-band result file. Also times tok/s.
import os, json, time, glob, subprocess
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams

def main():
    M = os.environ["S_MODEL"]; NTOK = int(os.environ.get("S_NTOK", "256"))
    TRIG = os.environ["CGSEAM_TRIG"]; OUTB = os.environ["CGSEAM_OUT"]
    PHASE = os.environ.get("S_PHASE", "gate")   # "gate" = clean+inject+catch; "tput" = timed only
    try:
        if os.path.exists(TRIG): os.remove(TRIG)
    except Exception: pass
    EAGER = os.environ.get("S_EAGER", "0") == "1"   # 0 = default (cudagraph ON); 1 = eager contrast
    llm = LLM(model=M, enforce_eager=EAGER, gpu_memory_utilization=0.82,
              max_model_len=2048, disable_log_stats=True)
    prompts = ["The history of the Roman Empire begins with"]
    sp = SamplingParams(max_tokens=NTOK, temperature=0.0, ignore_eos=True, min_tokens=NTOK)
    # warm generation triggers ALL lazy cudagraph capture for these shapes
    llm.generate(prompts, sp, use_tqdm=False)
    # signal the in-worker reader: init capture done, CUDA reads now safe
    open(os.environ["CGSEAM_READY"], "w").write("1")
    time.sleep(0.4)
    t0 = time.perf_counter(); o1 = llm.generate(prompts, sp, use_tqdm=False); dt1 = time.perf_counter() - t0
    g1 = sum(len(x.outputs[0].token_ids) for x in o1); tps = g1 / dt1
    clean_out = o1[0].outputs[0].text[:50]

    if PHASE == "gate":
        time.sleep(0.5)
        open(TRIG, "w").write("inject")        # worker seam thread flips INJ on next replay
        time.sleep(0.8)
        o2 = llm.generate(prompts, sp, use_tqdm=False)   # should be corrupted + caught under cudagraph
        inj_out = o2[0].outputs[0].text[:50]
        time.sleep(0.6)
    else:
        inj_out = None

    # collect worker out-of-band result (worker pid file)
    worker = None
    for f in sorted(glob.glob(OUTB + ".*"), key=os.path.getmtime, reverse=True):
        if f.endswith(".installed") or f.endswith(".err"): continue
        try:
            d = json.load(open(f))
            if d.get("ppid") == os.getpid() or d.get("armed") is not None:
                worker = d; worker["_file"] = f; break
        except Exception:
            pass
    res = {"model": M, "default_config": True, "cudagraph": True, "mp": True,
           "tok_s": round(tps, 2), "clean_out": clean_out, "inj_out": inj_out,
           "worker": worker}
    json.dump(res, open(os.environ["S_OUT"], "w"), indent=1)
    print("CGGATE", json.dumps({"tok_s": round(tps,2),
          "covered_calls": (worker or {}).get("cnt"),
          "clean_res": (worker or {}).get("clean_res"),
          "res_max": (worker or {}).get("res_max"),
          "caught": (worker or {}).get("first_detect_wall") is not None,
          "clean_out": clean_out[:30], "inj_out": (inj_out or "")[:30]}))

if __name__ == "__main__":
    main()
