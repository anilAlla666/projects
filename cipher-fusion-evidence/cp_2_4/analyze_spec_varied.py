"""CP 2.4 sub-task (iii) — varied-prompt spec-decode measurement analysis.

Reads spec_varied_<TAG>/p{1..N}_{off,on}.json. For each matched (pair, prompt)
it computes lift = on tok/s / off tok/s -> 25 lift values (N=5 pairs x 5
prompts). Reports:

  * lift           — flat mean + 95 % CI (1.96*SE); the gate criterion.
  * tok/round      — gen_tokens / rounds on the spec arm: the HONEST lift
                     driver (mean tokens emitted per target verify forward,
                     >1 == speedup). Works for both arms; unlike accept_rate
                     it counts every round including draft misses.
  * accept (cond.) — accepted / proposed. For the n-gram (Mistral) arm this
                     is CONDITIONAL on the draft having proposed at all — the
                     n-gram draft proposes only on an exact n-gram match, so
                     ~1.0 here is a greedy-loop signature, NOT draft quality.
                     For the model-draft (Llama) arm the draft always proposes
                     k tokens, so accept == per-token acceptance (meaningful).

Per-prompt subgroup means are also printed (the 25 points are 5 prompts x 5
correlated replicates — not i.i.d.; see SPEC_DECODE_METHODOLOGY.md §3.1).

Env: TAG (mistral|llama); GATE (1.75 for the Llama primary gate, 0 = no
target); N (default 5).
"""
import os
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("TAG", "mistral")
GATE = float(os.environ.get("GATE", "0"))
N = int(os.environ.get("N", "5"))
SV = os.path.join(HERE, "spec_varied_" + TAG)
NPROMPT = 5
GENRE = ["factual", "narrative", "code", "reasoning", "conversational"]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load(pair, arm):
    try:
        return json.load(open(os.path.join(SV, "p%d_%s.json" % (pair, arm))))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


lifts = []
by_prompt = {i: [] for i in range(NPROMPT)}
accs = []
tprs = []

print("pair prompt genre          off tok/s  on tok/s   lift  tok/rnd  accept")
print("-" * 73)
for pair in range(1, N + 1):
    o, n = load(pair, "off"), load(pair, "on")
    if o is None or n is None:
        print("%4d  -- MISSING (off=%s on=%s)"
              % (pair, o is not None, n is not None))
        continue
    op = {r["prompt"]: r for r in o["prompts"]}
    npd = {r["prompt"]: r for r in n["prompts"]}
    for pi in range(NPROMPT):
        if pi not in op or pi not in npd:
            continue
        lv = npd[pi]["tok_s"] / max(op[pi]["tok_s"], 1e-9)
        lifts.append(lv)
        by_prompt[pi].append(lv)
        rd = npd[pi].get("rounds")
        tpr = (npd[pi]["gen_tokens"] / rd) if rd else None
        if tpr is not None:
            tprs.append(tpr)
        ar = npd[pi].get("accept_rate")
        if ar is not None:
            accs.append(ar)
        print("%4d %4d   %-14s %8.2f  %8.2f  %5.3fx  %5.2f   %s"
              % (pair, pi, GENRE[pi], op[pi]["tok_s"], npd[pi]["tok_s"], lv,
                 tpr if tpr is not None else float("nan"),
                 ("%.3f" % ar) if ar is not None else "n/a"))

print("-" * 73)
res = {"cp": "2.4", "subtask": "iii-speculative-decode", "tag": TAG,
       "n_pairs": N, "n_points": len(lifts)}

if len(lifts) >= 2:
    m_l, s_l = mean(lifts), std(lifts)
    se = s_l / math.sqrt(len(lifts))
    lo, hi = m_l - 1.96 * se, m_l + 1.96 * se
    res.update(lift_mean=round(m_l, 4), lift_sd=round(s_l, 4),
               lift_95lo=round(lo, 4), lift_95hi=round(hi, 4),
               tok_per_round_mean=round(mean(tprs), 4) if tprs else None,
               accept_cond_mean=round(mean(accs), 4) if accs else None)
    res["by_prompt"] = {GENRE[i]: round(mean(by_prompt[i]), 4)
                        for i in range(NPROMPT) if by_prompt[i]}
    print("per-prompt mean lift:")
    for i in range(NPROMPT):
        if by_prompt[i]:
            print("  %-14s %5.3fx  (n=%d)"
                  % (GENRE[i], mean(by_prompt[i]), len(by_prompt[i])))
    print("-" * 73)
    print("spec lift: %.3fx   95%% CI [%.3f, %.3f]   (n=%d points, SD %.3f)"
          % (m_l, lo, hi, len(lifts), s_l))
    if tprs:
        print("lift driver: %.2f tokens / verify round  (mean over on-arm runs)"
              % mean(tprs))
    if accs:
        print("conditional acceptance (accepted/proposed): %.3f" % mean(accs))
    if GATE > 0:
        res["gate"] = GATE
        res["gate_pass_mean"] = m_l >= GATE
        res["gate_pass_ci"] = lo >= GATE
        print("gate target: %.2fx  ->  mean %s  |  CI-lower %s"
              % (GATE, "PASS" if m_l >= GATE else "FAIL",
                 "PASS" if lo >= GATE else "FAIL"))
    else:
        print("(secondary criterion — lift documented, no pass target)")
else:
    res["error"] = "insufficient data"
    print("INSUFFICIENT DATA")

outp = os.path.join(HERE, "spec_varied_%s_result.json" % TAG)
with open(outp, "w") as f:
    json.dump(res, f, indent=2)
print("result -> " + outp)
