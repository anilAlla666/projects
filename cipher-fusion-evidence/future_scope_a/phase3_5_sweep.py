#!/usr/bin/env python3
"""FUTURE_SCOPE/A Phase 3.5 — CIPHER DVFS-actuator lift under vLLM (THROWAWAY).

Measures whether CIPHER's DVFS (VOLT) actuator lifts tok/W beneath vLLM,
single tenant, graph mode (the Phase-3-verified production path).

Configs (5 reps each, fresh vLLM process):
  alone_default — vLLM alone, no injection, default clock — the baseline
  volt1200      — vLLM + CIPHER injection + CIPHER_VOLT clock-lock 1200 MHz
  volt800       — vLLM + CIPHER injection + CIPHER_VOLT clock-lock  800 MHz

`nvidia-smi -rgc` resets the GPU clock before every rep so the baseline runs
at default and no run inherits a stale lock. token_ids checked byte-identical
(DVFS must not perturb output — clock speed does not change FP results).
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = "/home/ubuntu/vllm_env/bin/python"
PROBE = os.path.join(HERE, "phase3_5_vllm_probe.py")
RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
REPS = 5

CONFIGS = [
    {"label": "alone_default", "inject": False, "volt_mhz": None},
    {"label": "volt1200",      "inject": True,  "volt_mhz": 1200},
    {"label": "volt800",       "inject": True,  "volt_mhz": 800},
]


def reset_clock():
    subprocess.run(["sudo", "nvidia-smi", "-rgc"], capture_output=True)
    time.sleep(1)


def run_one(cfg, rep):
    tag = "%s_r%d" % (cfg["label"], rep)
    outj = os.path.join(HERE, "p35_%s.json" % tag)
    env = dict(os.environ)
    for k in ("CUDA_INJECTION64_PATH", "CIPHER_VOLT", "CIPHER_VOLT_MHZ"):
        env.pop(k, None)
    env.update({"PROMPT_KEY": "med", "TAG": tag, "OUT_JSON": outj})
    if cfg["inject"]:
        env["CUDA_INJECTION64_PATH"] = RT
    if cfg["volt_mhz"]:
        env["CIPHER_VOLT"] = "on"
        env["CIPHER_VOLT_MHZ"] = str(cfg["volt_mhz"])
    reset_clock()
    log = os.path.join(HERE, "p35_%s.log" % tag)
    rc = subprocess.run([VENV_PY, PROBE], env=env, stdout=open(log, "w"),
                        stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(outj):
        print("  FAIL %s rc=%d" % (tag, rc), flush=True)
        return None
    return json.load(open(outj))


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 3),
            "std": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def main():
    results = {}
    for cfg in CONFIGS:
        rs = []
        for rep in range(REPS):
            print("  %s rep %d/%d ..." % (cfg["label"], rep + 1, REPS),
                  flush=True)
            r = run_one(cfg, rep)
            if r:
                rs.append(r)
        if not rs:
            results[cfg["label"]] = None
            continue
        tokid = rs[0]["token_ids"]
        results[cfg["label"]] = {
            "label": cfg["label"], "reps": len(rs),
            "tok_s": stat([r["tok_s"] for r in rs]),
            "power_w": stat([r["power_mean_w"] for r in rs]),
            "sm_clock_mhz": stat([r["sm_clock_mean_mhz"] for r in rs]),
            "tok_w": stat([r["tok_w"] for r in rs]),
            "token_ids_consistent": all(r["token_ids"] == tokid for r in rs),
            "token_ids": tokid}
    reset_clock()

    base = results.get("alone_default")
    summary = {"phase": "FUTURE_SCOPE/A Phase 3.5", "configs": results,
               "lift_vs_alone_default": {}}
    if base and base["tok_w"]:
        b = base["tok_w"]["mean"]
        for lbl in ("volt1200", "volt800"):
            c = results.get(lbl)
            if c and c["tok_w"]:
                summary["lift_vs_alone_default"][lbl] = {
                    "tok_w_lift_x": round(c["tok_w"]["mean"] / b, 3),
                    "tok_w_lift_pct": round(100 * (c["tok_w"]["mean"] - b) / b,
                                            1),
                    "tok_s_change_pct": round(
                        100 * (c["tok_s"]["mean"] - base["tok_s"]["mean"])
                        / base["tok_s"]["mean"], 1),
                    "token_ids_identical_vs_baseline":
                        c["token_ids"] == base["token_ids"]}

    json.dump(summary, open(os.path.join(HERE, "phase3_5_sweep_result.json"),
                            "w"), indent=2, default=str)
    print("\n=== PHASE 3.5 — DVFS lift under vLLM ===", flush=True)
    for lbl, c in results.items():
        if c:
            print("  %-14s tok/s=%.1f  power=%.1fW  clock=%.0fMHz  tok/W=%.4f"
                  % (lbl, c["tok_s"]["mean"], c["power_w"]["mean"],
                     c["sm_clock_mhz"]["mean"], c["tok_w"]["mean"]), flush=True)
    for lbl, l in summary["lift_vs_alone_default"].items():
        print("  %-14s tok/W lift = %.3fx (%+.1f%%)  tok/s %+.1f%%  "
              "tokens-identical=%s"
              % (lbl, l["tok_w_lift_x"], l["tok_w_lift_pct"],
                 l["tok_s_change_pct"], l["token_ids_identical_vs_baseline"]),
              flush=True)


if __name__ == "__main__":
    main()
