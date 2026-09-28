#!/usr/bin/env python3
"""Track 2 SC6 — substrate-value aggregator.

Reads the four sc6_run.py result JSONs (TinyLlama / Mistral-7B × shared /
independent) and produces the memory-measurement summary: the measured
savings, the weights/context decomposition, and the structural formula a
reviewer can project to N=100.

The decomposition (5 co-resident processes, idle subtracted):
  shared:      W + 5C        (one weight copy + 5 process contexts)
  independent: 5W + 5C       (a private weight copy per tenant)
  per-tenant independent increment  P = W + C   (measured incrementally)
  shared loaded delta               S = W + 5C
  => C = (S - P) / 4 ,  W = P - C
Savings is reported against the measured 5-tenant independent total when 5
tenants fit, and against the projected total (idle + 5P) when they did not
(an OOM is a finding, not a gap — see TRACK_2_SC6_MEMORY.md).
"""
import json
import sys

HERE = "/home/ubuntu/cipher-fusion-evidence/phase_c"


def load(model, phase):
    return json.load(open("%s/sc6_%s_%s_result.json" % (HERE, model, phase)))


def analyse(model):
    sh = load(model, "shared")
    ind = load(model, "independent")
    idle = ind["fb_mib"]["idle"]
    S = sh["fb_mib"]["loaded"] - idle                    # W + 5C
    P = ind["per_tenant_mib"]                            # W + C
    C = (S - P) / 4.0
    W = P - C
    ind_5 = (ind["fb_mib"]["loaded"] if ind["n_fit"] == 5
             else ind["fb_loaded_5tenant_projected"])
    shared_loaded = sh["fb_mib"]["loaded"]
    savings_mib = ind_5 - shared_loaded
    savings_pct = 100.0 * savings_mib / ind_5 if ind_5 else 0.0
    # asymptote of N*W/(N*W+(N+1)*C) as N->inf  ==  W/(W+C)
    asymptote = 100.0 * W / (W + C) if (W + C) else 0.0
    return {
        "model": model,
        "shared_PASS": sh.get("PASS"),
        "independent_n_fit": ind["n_fit"],
        "independent_oom": ind["oom"],
        "fb_idle_mib": idle,
        "fb_shared_loaded_mib": shared_loaded,
        "fb_independent_loaded_mib": ind["fb_mib"]["loaded"],
        "fb_independent_5tenant_mib": ind_5,
        "fb_independent_5tenant_measured": ind["n_fit"] == 5,
        "fb_shared_exited_mib": sh["fb_mib"]["exited"],
        "fb_independent_exited_mib": ind["fb_mib"]["exited"],
        "est_weights_W_mib": round(W, 1),
        "est_context_C_mib": round(C, 1),
        "savings_mib": round(savings_mib, 1),
        "savings_pct_at_N4": round(savings_pct, 1),
        "asymptote_pct_large_N": round(asymptote, 1),
        "teardown_clean_shared": abs(sh["fb_mib"]["exited"] - idle) < 500,
        "teardown_clean_independent":
            abs(ind["fb_mib"]["exited"] - idle) < 500,
    }


def main():
    models = sys.argv[1:] or ["TinyLlama", "Mistral-7B"]
    rows = [analyse(m) for m in models]
    summary = {"models": rows,
               "formula": "savings(N) = N*W / (N*W + (N+1)*C); "
                          "asymptote W/(W+C)"}
    json.dump(summary, open(HERE + "/sc6_memory_summary.json", "w"),
              indent=2, default=str)
    print(json.dumps(summary, indent=2))
    for r in rows:
        print("\n%s: N=4 savings %.1f%%  (W=%.0f MiB, C=%.0f MiB, "
              "large-N asymptote %.1f%%)  independent fit %d/5%s"
              % (r["model"], r["savings_pct_at_N4"], r["est_weights_W_mib"],
                 r["est_context_C_mib"], r["asymptote_pct_large_N"],
                 r["independent_n_fit"],
                 "  [OOM]" if r["independent_oom"] else ""))


if __name__ == "__main__":
    main()
