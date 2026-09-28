#!/usr/bin/env python3
"""FUTURE_SCOPE/A Phase 3 — vLLM parity sweep driver (THROWAWAY).

Drives `phase3_vllm_probe.py` across vLLM-alone vs vLLM-on-CIPHER, eager vs
graph, with reps, and aggregates parity statistics. Each rep is a fresh vLLM
process (clean state). Runs the probe via the vLLM venv.

Headline: 4 configs (alone/inj × eager/graph) × 5 reps at the medium prompt.
Plus an overhead-vs-context scan (inj graph, short/med/long, 2 reps) — the
CUPTI-per-launch attribution evidence.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = "/home/ubuntu/vllm_env/bin/python"
PROBE = os.path.join(HERE, "phase3_vllm_probe.py")
RT = "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
HEADLINE_REPS = 5


def run_one(eager, inject, prompt_key, tag):
    outj = os.path.join(HERE, "p3_%s.json" % tag)
    env = dict(os.environ)
    env.update({"EAGER": "1" if eager else "0", "PROMPT_KEY": prompt_key,
                "TAG": tag, "OUT_JSON": outj})
    if inject:
        env["CUDA_INJECTION64_PATH"] = RT
    else:
        env.pop("CUDA_INJECTION64_PATH", None)
    log = os.path.join(HERE, "p3_%s.log" % tag)
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
    return {"mean": round(statistics.mean(xs), 2),
            "std": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 2), "max": round(max(xs), 2), "n": len(xs)}


def config(eager, inject, prompt_key, reps, label):
    rs = []
    for i in range(reps):
        tag = "%s_r%d" % (label, i)
        print("  %s rep %d/%d ..." % (label, i + 1, reps), flush=True)
        r = run_one(eager, inject, prompt_key, tag)
        if r:
            rs.append(r)
    if not rs:
        return None
    tokid = rs[0]["token_ids"]
    return {"label": label, "eager": eager, "inject": inject,
            "prompt_key": prompt_key, "reps": len(rs),
            "tok_s": stat([r["tok_s"] for r in rs]),
            "power_w": stat([r["power_mean_w"] for r in rs]),
            "ttft_s": stat([r["ttft_s"] for r in rs]),
            "itl_ms": stat([r["itl_mean_ms"] for r in rs]),
            "token_ids_consistent": all(r["token_ids"] == tokid for r in rs),
            "token_ids": tokid}


def main():
    t0 = time.time()
    results = {}

    # ---- headline parity: 4 configs × 5 reps, medium prompt ----
    for eager in (True, False):
        mode = "eager" if eager else "graph"
        for inject in (False, True):
            arm = "inj" if inject else "alone"
            label = "%s_%s" % (mode, arm)
            results[label] = config(eager, inject, "med", HEADLINE_REPS, label)

    # ---- overhead-vs-context scan: inj graph, short/med/long, 2 reps ----
    for pk in ("short", "med", "long"):
        label = "ctxscan_graph_inj_%s" % pk
        results[label] = config(False, True, pk, 2, label)

    # ---- parity analysis ----
    def tw(c):
        return (c["tok_s"]["mean"] / c["power_w"]["mean"]
                if c and c["power_w"] and c["power_w"]["mean"] else None)

    analysis = {}
    for mode in ("eager", "graph"):
        a, i = results["%s_alone" % mode], results["%s_inj" % mode]
        if a and i:
            ov = 100.0 * (i["tok_s"]["mean"] - a["tok_s"]["mean"]) \
                / a["tok_s"]["mean"]
            ta, ti = tw(a), tw(i)
            analysis[mode] = {
                "tok_s_alone": a["tok_s"]["mean"],
                "tok_s_inj": i["tok_s"]["mean"],
                "overhead_pct": round(ov, 2),
                "within_5pct_gate": abs(ov) <= 5.0,
                "tok_w_alone": round(ta, 4) if ta else None,
                "tok_w_inj": round(ti, 4) if ti else None,
                "token_ids_identical_alone_vs_inj":
                    a["token_ids"] == i["token_ids"]}

    summary = {"phase": "FUTURE_SCOPE/A Phase 3",
               "wall_min": round((time.time() - t0) / 60, 1),
               "analysis": analysis, "configs": results}
    json.dump(summary, open(os.path.join(HERE, "phase3_sweep_result.json"),
                            "w"), indent=2, default=str)
    print("\n=== PHASE 3 PARITY ===", flush=True)
    for mode, a in analysis.items():
        print("  %-6s alone %.1f -> inj %.1f tok/s  overhead %+.2f%%  "
              "5%%-gate=%s  tok_id-identical=%s"
              % (mode, a["tok_s_alone"], a["tok_s_inj"], a["overhead_pct"],
                 a["within_5pct_gate"], a["token_ids_identical_alone_vs_inj"]),
              flush=True)
    print("=== overhead-vs-context (inj graph) ===", flush=True)
    for pk in ("short", "med", "long"):
        c = results.get("ctxscan_graph_inj_%s" % pk)
        if c:
            print("  %-6s tok/s=%.1f" % (pk, c["tok_s"]["mean"]), flush=True)


if __name__ == "__main__":
    main()
