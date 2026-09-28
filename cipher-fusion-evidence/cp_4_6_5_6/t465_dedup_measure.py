"""CP 4.6.5 — cross-tenant KV dedup measurement on the 739 Claude Code
agentic traces (callanjfox/kv-cache-tester).

Methodology note — why this is a structural model, not a hash-overlap count.
The traces are `hash_id_scope: "local"`: hash_ids are per-conversation, so
the T4.6.3 global-hash-table method (used on the Mooncake traces) would
falsely collide ids across traces. For these traces the *measurable*
cross-tenant dedup is the structurally-shared Claude Code system+tool
prefix — every session carries the same system prompt + tool schema. That
is exactly the model the dataset's own `trace_replay_tester.py` uses
("canonical shared prefix used by ALL users"). The result is therefore a
LOWER BOUND on production cross-tenant dedup — codebase / shared-file
overlap is invisible to a `local`-scope dataset.

Model. N tenants, each resident at context C tokens. Of a tenant's C
tokens, prefix_i = system_tokens + tool_tokens is the shared Claude Code
prefix; C - prefix_i is conversation-unique. Cross-tenant deduped footprint
= one canonical prefix copy + Σ unique. Capacity multiplier
  m(C) = naive / deduped = (N·C) / (prefix_canonical + Σ_i (C - prefix_i)).
n=5 windowed (80% subsamples) for CI, matching the T4.6.3 discipline.
"""
import json, glob, random, statistics as st

TRACE_DIR = "/home/ubuntu/kv-cache-tester/traces"
OUT = "/home/ubuntu/cipher-fusion-evidence/cp_4_6_5_6/dedup_measurement.json"
CONTEXTS = [25000, 32000, 137000]
SUBSET_FRAC = 0.8
SUBSET_TRIALS = 5
random.seed(20260516)


def load_traces():
    out = []
    for f in sorted(glob.glob(f"{TRACE_DIR}/*.json")):
        d = json.load(open(f))
        prefix = d.get("system_tokens", 0) + d.get("tool_tokens", 0)
        ins = [r.get("in", 0) for r in d.get("requests", [])]
        out.append({"id": d.get("id"), "prefix": prefix,
                    "max_ctx": max(ins) if ins else 0})
    return out


def m_for_window(traces, C):
    """Capacity multiplier m for one set of tenants resident at context C.
    Canonical-shared-prefix model; prefix copy sized at the window mean
    (a representative canonical Claude Code prefix)."""
    # a tenant can only be resident at C if its conversation reaches C
    ten = [t for t in traces if t["max_ctx"] >= C]
    if len(ten) < 2:
        return None
    N = len(ten)
    naive = N * C
    prefixes = [min(t["prefix"], C) for t in ten]
    prefix_canonical = st.mean(prefixes)            # one shared copy
    unique = sum(C - p for t, p in zip(ten, prefixes))
    deduped = prefix_canonical + unique
    return naive / deduped, N


def windowed(traces, C):
    reach = [t for t in traces if t["max_ctx"] >= C]
    full = m_for_window(traces, C)
    ms = []
    k = max(2, int(len(reach) * SUBSET_FRAC))
    for _ in range(SUBSET_TRIALS):
        sub = random.sample(reach, k)
        r = m_for_window(sub, C)
        if r:
            ms.append(r[0])
    mean = st.mean(ms)
    sd = st.pstdev(ms)
    ci = 1.96 * sd / (len(ms) ** 0.5)
    f_dist = sorted(min(t["prefix"], C) / C for t in reach)
    return {
        "context_tokens": C,
        "tenants_reaching_C": len(reach),
        "m_full": round(full[0], 4) if full else None,
        "m_windowed_mean": round(mean, 4),
        "m_windowed_ci95": [round(mean - ci, 4), round(mean + ci, 4)],
        "m_windowed_sd": round(sd, 5),
        "shared_frac_f_p25": round(f_dist[len(f_dist) // 4], 4),
        "shared_frac_f_median": round(f_dist[len(f_dist) // 2], 4),
        "shared_frac_f_p75": round(f_dist[3 * len(f_dist) // 4], 4),
        "tier": ("L1-only" if mean < 2 else
                 "L3" if mean <= 10 else "RadixAttention"),
    }


def main():
    traces = load_traces()
    print(f"=== CP 4.6.5 cross-tenant dedup — {len(traces)} traces ===\n")
    results = []
    for C in CONTEXTS:
        r = windowed(traces, C)
        results.append(r)
        print(f"  C={C:>7}: m={r['m_windowed_mean']:.3f}x "
              f"CI95{r['m_windowed_ci95']}  "
              f"f_median={r['shared_frac_f_median']:.3f}  "
              f"tenants={r['tenants_reaching_C']}  -> tier {r['tier']}")
    doc = {
        "cp": "4.6.5",
        "dataset": "callanjfox/kv-cache-tester, 739 Claude Code agentic traces",
        "hash_id_scope": "local",
        "method": "structural canonical-shared-prefix model; "
                  "cross-tenant dedup = shared system+tool prefix; "
                  "measured m is a LOWER BOUND (codebase overlap unmeasurable)",
        "n_windowed": SUBSET_TRIALS,
        "subset_frac": SUBSET_FRAC,
        "results": results,
    }
    json.dump(doc, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
