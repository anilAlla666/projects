# Driver for the FP8 seam detector. Installs the seam BEFORE engine init (in-process), runs the
# real neuralmagic FP8 checkpoint, dumps coverage/correctness/catch + a tok/s overhead pair.
import os, json, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # path-B operating condition (engine in-process)

import cipher_fp8_seam as seam

MODE = os.environ.get("FP8_MODE", "detect")   # "detect" = seam armed; "off" = baseline (no install)
if MODE != "off":
    seam.install()

from vllm import LLM, SamplingParams

def main():
    M = os.environ["S_MODEL"]; OUT = os.environ["S_OUT"]
    NTOK = int(os.environ.get("S_NTOK", "128"))
    EAGER = os.environ.get("S_EAGER", "1") == "1"
    llm = LLM(model=M, enforce_eager=EAGER, gpu_memory_utilization=0.82,
              max_model_len=2048, disable_log_stats=True)
    prompts = ["The history of the Roman Empire begins with"]
    sp = SamplingParams(max_tokens=NTOK, temperature=0.0, ignore_eos=True, min_tokens=NTOK)
    llm.generate(prompts, sp, use_tqdm=False)   # warm
    # reset counters after warm so coverage/catch reflect the timed pass only
    for k in ["lin","checks","gemms_checked","detections","injected","csmm_total","covered_linears","first_detect_step"]:
        seam.st[k] = 0 if k != "first_detect_step" else -1
    seam.st["max_clean_resid"] = 0.0; seam.st["max_detect_resid"] = 0.0; seam.st["shapes_seen"] = {}
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        o = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        g = sum(len(x.outputs[0].token_ids) for x in o)
        tps = g / dt
        if best is None or tps > best:
            best = tps
    seam_res = seam.dump() if MODE != "off" else {}
    res = {"model": M, "mode": MODE, "eager": EAGER, "ntok": NTOK,
           "tok_s": round(best, 2), "out": o[0].outputs[0].text[:60],
           "seam": seam_res}
    json.dump(res, open(OUT, "w"), indent=1)
    print("FP8SEAM", json.dumps({"mode": MODE, "tok_s": round(best,2),
          "covered": seam_res.get("covered_linears"), "checked": seam_res.get("gemms_checked"),
          "detections": seam_res.get("detections"), "first_detect_step": seam_res.get("first_detect_step"),
          "max_clean_resid": seam_res.get("max_clean_resid")}))

if __name__ == "__main__":
    main()
