"""CP 2.4 — composed-gate analysis (memo §6c).

Reads composed/p{1..N}_{vanilla,marlin,allon}.json (per-prompt tok/s) and the
matching .watts.csv (sentinel-windowed power). Per (pair, arm):
  tok/s = sum(gen_tokens) / sum(wall_s)      (aggregate over the 5 prompts)
  tok/W = tok/s / mean(power_w)              (decode-window mean power)

Matched-pair lifts (n=5, mean +/- 1.96*SE):
  marlin / vanilla   — Marlin INT4 single-tenant tok/s lift
  allon  / marlin    — spec + DVFS over Marlin (tok/s ~ the n-gram spec lift;
                       DVFS is tok/s-neutral) and the tok/W gain from DVFS
  allon  / vanilla    — the COMPOSED lift (tok/s and tok/W)

Composition check: allon/vanilla tok/s should ~ (marlin/vanilla) x
(allon/marlin) by construction; the honest cross-check is whether
allon/marlin tok/s reproduces the independently measured Mistral n-gram spec
lift (spec_varied_mistral_result.json, 1.639x).
"""
import os
import csv
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
CMP = os.path.join(HERE, "composed")
N = int(os.environ.get("N", "5"))
ARMS = ["vanilla", "marlin", "allon"]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    m, s = mean(xs), std(xs)
    se = s / math.sqrt(len(xs)) if xs else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def load_arm(pair, arm):
    """-> (tok_s, tok_W, mean_power) or None."""
    jp = os.path.join(CMP, "p%d_%s.json" % (pair, arm))
    cp = os.path.join(CMP, "p%d_%s.watts.csv" % (pair, arm))
    try:
        d = json.load(open(jp))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    toks = sum(r["gen_tokens"] for r in d["prompts"])
    secs = sum(r["wall_s"] for r in d["prompts"])
    tok_s = toks / secs if secs else 0.0
    pwr = []
    try:
        for row in csv.DictReader(open(cp)):
            try:
                pwr.append(float(row["power_w"]))
            except (ValueError, KeyError):
                pass
    except FileNotFoundError:
        pass
    mp = mean(pwr) if pwr else None
    tok_w = (tok_s / mp) if mp else None
    return tok_s, tok_w, mp


rows = {a: [] for a in ARMS}          # per-arm list of (tok_s, tok_W, pwr)
lift_ts = {"marlin/vanilla": [], "allon/marlin": [], "allon/vanilla": []}
lift_tw = {"marlin/vanilla": [], "allon/marlin": [], "allon/vanilla": []}

print("pair  arm      tok/s    W      tok/W")
print("-" * 44)
for pair in range(1, N + 1):
    got = {}
    for a in ARMS:
        r = load_arm(pair, a)
        if r is None:
            print("%4d  %-8s MISSING" % (pair, a))
            continue
        got[a] = r
        rows[a].append(r)
        print("%4d  %-8s %7.2f  %6.1f  %.4f"
              % (pair, a, r[0], r[2] or float("nan"),
                 r[1] if r[1] is not None else float("nan")))
    if {"vanilla", "marlin", "allon"} <= set(got):
        v, mar, al = got["vanilla"], got["marlin"], got["allon"]
        lift_ts["marlin/vanilla"].append(mar[0] / v[0])
        lift_ts["allon/marlin"].append(al[0] / mar[0])
        lift_ts["allon/vanilla"].append(al[0] / v[0])
        if v[1] and mar[1] and al[1]:
            lift_tw["marlin/vanilla"].append(mar[1] / v[1])
            lift_tw["allon/marlin"].append(al[1] / mar[1])
            lift_tw["allon/vanilla"].append(al[1] / v[1])

print("-" * 44)
res = {"cp": "2.4", "gate": "composed-6c", "n_pairs": N}


def report(tag, d):
    out = {}
    for k, xs in d.items():
        if len(xs) >= 2:
            m, lo, hi = ci95(xs)
            out[k] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            print("  %-16s %-16s %.3fx   95%% CI [%.3f, %.3f]"
                  % (tag, k, m, lo, hi))
    return out


print("THROUGHPUT lift (tok/s):")
res["tok_s_lift"] = report("tok/s", lift_ts)
print("EFFICIENCY lift (tok/W):")
res["tok_w_lift"] = report("tok/W", lift_tw)

# composition cross-check
if lift_ts["allon/vanilla"] and lift_ts["marlin/vanilla"] and lift_ts["allon/marlin"]:
    comp = mean(lift_ts["allon/vanilla"])
    prod = mean(lift_ts["marlin/vanilla"]) * mean(lift_ts["allon/marlin"])
    res["composition_check"] = {
        "composed_tok_s": round(comp, 4),
        "marlin_x_spec_product": round(prod, 4),
        "allon_over_marlin_vs_mistral_spec_1.639": round(
            mean(lift_ts["allon/marlin"]), 4)}
    print("-" * 44)
    print("composition: composed tok/s %.3fx  vs  marlin x (spec+dvfs) %.3fx"
          % (comp, prod))
    print("spec lever (all-on/marlin tok/s): %.3fx  "
          "(independent Mistral n-gram spec arm: 1.639x)"
          % mean(lift_ts["allon/marlin"]))

outp = os.path.join(HERE, "composed_result.json")
with open(outp, "w") as f:
    json.dump(res, f, indent=2)
print("result -> " + outp)
