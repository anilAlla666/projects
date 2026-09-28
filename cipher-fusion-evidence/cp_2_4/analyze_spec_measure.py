"""CP 2.4 sub-task (iii) — Mistral-arm spec-decode measurement analysis.

Reads spec_measure/spm_p{1..N}_{off,on}.progress, takes tok/s from each run's
'end' line, computes the matched-pair lift (on/off) with a 95% CI, and the
spec-arm acceptance rate from the .log files. Secondary criterion — reports
the lift; there is no pass target for the Mistral arm.
"""
import os
import re
import math
import json

HERE = os.path.dirname(os.path.abspath(__file__))
SM = os.path.join(HERE, "spec_measure")
N = int(os.environ.get("N", "5"))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def tok_s(progress_path):
    """tok/s from the run_for_duration 'end' line: '... end t=Xs ... tokens=N'."""
    try:
        line = ""
        for ln in open(progress_path):
            if " end t=" in ln:
                line = ln
        if not line:
            return None
        secs = float(re.search(r"end t=([\d.]+)s", line).group(1))
        toks = int(re.search(r"tokens=(\d+)", line).group(1))
        return toks / max(secs, 1e-6)
    except (FileNotFoundError, AttributeError):
        return None


def accept_rate(log_path):
    """Mean accept_rate over the '[cipher-spec] rounds=.. accept_rate=..' lines."""
    rates = []
    try:
        for ln in open(log_path):
            m = re.search(r"accept_rate=([\d.]+)", ln)
            if m:
                rates.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    return mean(rates) if rates else None


off, on, lifts, accs = [], [], [], []
print("pair   off tok/s   on tok/s   lift    accept_rate")
print("-" * 52)
for i in range(1, N + 1):
    o = tok_s(os.path.join(SM, "spm_p%d_off.progress" % i))
    n = tok_s(os.path.join(SM, "spm_p%d_on.progress" % i))
    a = accept_rate(os.path.join(SM, "spm_p%d_on.log" % i))
    if o is None or n is None:
        print("%4d   MISSING DATA" % i)
        continue
    off.append(o); on.append(n); lifts.append(n / o)
    if a is not None:
        accs.append(a)
    print("%4d   %8.2f   %8.2f   %5.3fx   %s"
          % (i, o, n, n / o, ("%.3f" % a) if a is not None else "n/a"))

print("-" * 52)
res = {"cp": "2.4", "subtask": "iii-speculative-decode",
       "arm": "Mistral-7B + n-gram draft (secondary criterion)", "n": len(lifts)}
if len(lifts) >= 2:
    m_l, s_l = mean(lifts), std(lifts)
    se = s_l / math.sqrt(len(lifts))
    res["off_tok_s_mean"] = round(mean(off), 3)
    res["on_tok_s_mean"] = round(mean(on), 3)
    res["lift_mean"] = round(m_l, 4)
    res["lift_95lo"] = round(m_l - 1.96 * se, 4)
    res["lift_95hi"] = round(m_l + 1.96 * se, 4)
    res["accept_rate_mean"] = round(mean(accs), 4) if accs else None
    print("Mistral-arm spec lift: %.3fx  (95%% CI %.3f..%.3f)  accept_rate=%s"
          % (m_l, res["lift_95lo"], res["lift_95hi"],
             ("%.3f" % mean(accs)) if accs else "n/a"))
    print("(secondary criterion — lift documented, no pass target)")
else:
    res["error"] = "insufficient data"
    print("INSUFFICIENT DATA")

with open(os.path.join(HERE, "spec_measure_result.json"), "w") as f:
    json.dump(res, f, indent=2)
print("result -> spec_measure_result.json")
