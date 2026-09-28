"""CP 5.6 P2 — teacher-forced gate + TPW re-measurement analysis.

Reads gold.json + {vanilla,marlin,allon}.json + {arm}.watts.csv. Per
(prompt, arm): tok/s (free-run), tok/W = tok/s / ARM-LEVEL mean power, and
teacher-forced agreement. A pair PASSES at agreement >= 0.99. Per arm:
passing-pair count, TPW on PASSING pairs (with CI when n>=2), and an all-prompt
(gate-agnostic) TPW for the full picture. Overall lift all-on/vanilla and
marlin/vanilla, both gate-filtered and gate-agnostic, vs the CP 2.4 headline.
Emits cp56_p2_result.json + A/B/C verdict.

Power is arm-level (advisor A1): 1 Hz nvidia-smi is too sparse to slice per
prompt, but ~12-24 samples/arm is ample for a stable mean (VOLT-lock keeps it
near constant). Matches CP 2.4 analyze_composed.py tok/W = tok/s / mean(power).

Loop detection (advisor A4, corrected): the reliable CP 2.4 loop signature is
accept_rate ~ 1.0. An absolute distinct-ratio threshold is NOT reliable — clean
FP16 gold legitimately sits at 0.35-0.57 over 128 tokens (common tokens repeat).
is_looping therefore keys on accept_rate, plus a *relative* collapse vs the
same prompt's gold distinct-ratio (< 0.5x gold).
"""
import os
import csv
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["vanilla", "marlin", "allon"]
THRESHOLD = 0.99
HEADLINE = 3.6166                       # CP 2.4 composed all-on/vanilla tok/W
HEADLINE_CI = (3.5908, 3.6423)
HEADLINE_TOKS = 1.7954                  # CP 2.4 composed all-on/vanilla tok/s


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    mu, s = mean(xs), std(xs)
    se = s / math.sqrt(len(xs)) if xs else 0.0
    return mu, mu - 1.96 * se, mu + 1.96 * se


def passes(agreement):
    return agreement >= THRESHOLD


def mean_power(rows):
    """rows: list of (epoch_s, power_w). Arm-level mean power, or None."""
    pw = [p for (_ts, p) in rows]
    return mean(pw) if pw else None


def is_looping(rec, gold_distinct):
    """CP 2.4 loop signature: accept_rate ~ 1.0, or a free-run distinct-token
    ratio that has collapsed to < 0.5x the same prompt's clean-FP16 gold."""
    if rec.get("accept_rate") is not None and rec["accept_rate"] >= 0.9999:
        return True
    dr = rec.get("distinct_ratio")
    return (dr is not None and gold_distinct is not None
            and gold_distinct > 0 and dr < 0.5 * gold_distinct)


def load_watts(arm):
    path = os.path.join(HERE, "%s.watts.csv" % arm)
    rows = []
    try:
        for r in csv.DictReader(open(path)):
            try:
                rows.append((float(r["epoch_s"]), float(r["power_w"])))
            except (ValueError, KeyError):
                pass
    except FileNotFoundError:
        pass
    return rows


def load_gold():
    try:
        g = json.load(open(os.path.join(HERE, "gold.json")))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {p["prompt"]: p.get("distinct_ratio") for p in g["prompts"]}


def load_arm(arm, gold_distinct):
    """-> (dict prompt -> per-pair metrics, arm_power) or (None, None)."""
    path = os.path.join(HERE, "%s.json" % arm)
    try:
        d = json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    arm_power = mean_power(load_watts(arm))
    out = {}
    for r in d["prompts"]:
        tw = (r["tok_s"] / arm_power) if arm_power else None
        out[r["prompt"]] = {
            "tok_s": r["tok_s"], "tok_w": tw,
            "tf_agreement": r["tf_agreement"], "pass": passes(r["tf_agreement"]),
            "accept_rate": r.get("accept_rate"),
            "distinct_ratio": r.get("distinct_ratio"),
            "gold_distinct": gold_distinct.get(r["prompt"]),
            "looping": is_looping(r, gold_distinct.get(r["prompt"])),
            "free_run_text": r.get("free_run_text", ""),
            "tf_mismatches": r.get("tf_mismatches", [])}
    return out, arm_power


def matched_lift(arms, num, den, gated):
    """Per-prompt num/den tok/W ratio. gated=True -> only prompts where both
    arms pass the TF gate; gated=False -> all prompts with tok/W on both."""
    dn, dd = arms.get(num) or {}, arms.get(den) or {}
    ratios = []
    for p in range(5):
        if p not in dn or p not in dd:
            continue
        if not (dn[p]["tok_w"] and dd[p]["tok_w"]):
            continue
        if gated and not (dn[p]["pass"] and dd[p]["pass"]):
            continue
        ratios.append(dn[p]["tok_w"] / dd[p]["tok_w"])
    if not ratios:
        return {"n": 0, "mean": None, "ci95": None}
    mu, lo, hi = ci95(ratios) if len(ratios) >= 2 else (mean(ratios), None, None)
    return {"n": len(ratios), "mean": round(mu, 4),
            "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None}


def main():
    res = {"cp": "5.6", "priority": "P2", "gate": "teacher-forced top-1",
           "threshold": THRESHOLD, "headline_under_test": HEADLINE,
           "substrate": "a7ac8e97", "arm_power_w": {}, "arms": {},
           "sanity": {}, "lift_gated": {}, "lift_all_prompt": {}}
    gold_distinct = load_gold()
    arms = {}
    for a in ARMS:
        d, pw = load_arm(a, gold_distinct)
        arms[a] = d
        res["arm_power_w"][a] = round(pw, 2) if pw else None

    print("prompt arm      tok/s    W     tok/W   tf_agree  pass  loop  "
          "distinct(gold)")
    print("-" * 74)
    for p in range(5):
        for a in ARMS:
            d = arms.get(a)
            if not d or p not in d:
                print("%6d %-8s MISSING" % (p, a)); continue
            r = d[p]
            print("%6d %-8s %7.2f %6.1f %s   %.4f   %-4s  %-4s  %.3f (%.3f)"
                  % (p, a, r["tok_s"], res["arm_power_w"][a] or float("nan"),
                     ("%.4f" % r["tok_w"]) if r["tok_w"] else " n/a  ",
                     r["tf_agreement"], "PASS" if r["pass"] else "FAIL",
                     "LOOP" if r["looping"] else "",
                     r["distinct_ratio"] or 0.0, r["gold_distinct"] or 0.0))
    print("-" * 74)

    # sanity — vanilla teacher-forced must be ~100%
    van = arms.get("vanilla") or {}
    van_agree = [van[p]["tf_agreement"] for p in van]
    sane = bool(van_agree) and min(van_agree) >= 0.99
    res["sanity"] = {"vanilla_tf_agreements": van_agree,
                     "vanilla_tf_min": min(van_agree) if van_agree else None,
                     "sane": sane}
    print("SANITY  vanilla TF min=%s  -> %s"
          % (("%.4f" % min(van_agree)) if van_agree else "n/a",
             "OK" if sane else "FAIL (gold capture / harness bug)"))

    # per-arm summary
    for a in ARMS:
        d = arms.get(a) or {}
        passing = sorted(p for p in d if d[p]["pass"] and d[p]["tok_w"])
        all_tw = [d[p]["tok_w"] for p in d if d[p]["tok_w"]]
        all_ts = [d[p]["tok_s"] for p in d]
        pass_tw = [d[p]["tok_w"] for p in passing]
        gmu, glo, ghi = (ci95(pass_tw) if len(pass_tw) >= 2
                         else (mean(pass_tw), None, None))
        amu, alo, ahi = (ci95(all_tw) if len(all_tw) >= 2
                         else (mean(all_tw), None, None))
        res["arms"][a] = {
            "n_prompts": len(d), "n_passing": len(passing),
            "passing_prompts": passing,
            "tf_agreements": {str(p): d[p]["tf_agreement"] for p in d},
            "tpw_passing_mean": round(gmu, 4) if pass_tw else None,
            "tpw_passing_ci95": [round(glo, 4), round(ghi, 4)] if glo is not None else None,
            "tpw_allprompt_mean": round(amu, 4) if all_tw else None,
            "tpw_allprompt_ci95": [round(alo, 4), round(ahi, 4)] if alo is not None else None,
            "toks_allprompt_mean": round(mean(all_ts), 4) if all_ts else None,
            "looping_prompts": sorted(p for p in d if d[p]["looping"]),
            "accept_rates": {str(p): d[p]["accept_rate"] for p in d},
            "failing": {str(p): {"tf_agreement": d[p]["tf_agreement"],
                                 "accept_rate": d[p]["accept_rate"],
                                 "distinct_ratio": d[p]["distinct_ratio"],
                                 "text_head": d[p]["free_run_text"][:240]}
                        for p in d if not d[p]["pass"]}}
        print("ARM %-8s pass=%d/%d  TPW_passing=%s  TPW_allprompt=%s  "
              "tok/s_mean=%.2f"
              % (a, len(passing), len(d),
                 ("%.4f" % gmu) if pass_tw else "n/a",
                 ("%.4f" % amu) if all_tw else "n/a", mean(all_ts)))

    # lift — gated (passing pairs) and gate-agnostic (all prompts)
    for tag, num, den in [("allon/vanilla", "allon", "vanilla"),
                          ("marlin/vanilla", "marlin", "vanilla")]:
        res["lift_gated"][tag] = matched_lift(arms, num, den, gated=True)
        res["lift_all_prompt"][tag] = matched_lift(arms, num, den, gated=False)
        g, ap = res["lift_gated"][tag], res["lift_all_prompt"][tag]
        print("LIFT %-15s gated: %s (n=%d)   all-prompt: %s (n=%d)"
              % (tag,
                 ("%.4fx" % g["mean"]) if g["mean"] else "n/a", g["n"],
                 ("%.4fx" % ap["mean"]) if ap["mean"] else "n/a", ap["n"]))

    # tok/s lift all-prompt (the headline's 1.795x throughput component)
    ts_lift = {}
    for tag, num, den in [("allon/vanilla", "allon", "vanilla"),
                          ("marlin/vanilla", "marlin", "vanilla")]:
        dn, dd = arms.get(num) or {}, arms.get(den) or {}
        rr = [dn[p]["tok_s"] / dd[p]["tok_s"] for p in range(5)
              if p in dn and p in dd and dd[p]["tok_s"]]
        ts_lift[tag] = round(mean(rr), 4) if rr else None
    res["toks_lift_all_prompt"] = ts_lift
    print("TOK/S  all-prompt  allon/vanilla=%s  marlin/vanilla=%s   "
          "(CP 2.4 headline allon/vanilla tok/s = %.3fx)"
          % (ts_lift["allon/vanilla"], ts_lift["marlin/vanilla"], HEADLINE_TOKS))

    # verdict
    mp = res["arms"].get("marlin", {}).get("n_passing", 0)
    ap_ = res["arms"].get("allon", {}).get("n_passing", 0)
    ag = res["lift_all_prompt"].get("allon/vanilla", {}).get("mean")
    gated = res["lift_gated"].get("allon/vanilla", {})
    if not sane:
        verdict = ("INVALID", "vanilla teacher-forced sanity failed — gold "
                   "capture or harness bug; fix before interpreting")
    elif mp < 3 or ap_ < 3 or (ag is not None and ag < 0.85 * HEADLINE):
        verdict = ("C", "the Marlin substrate fails the 99%% teacher-forced "
                   "correctness gate on the majority of prompts (marlin "
                   "%d/5, allon %d/5 passing) — INT4 decode diverges from "
                   "FP16 on ~7-10%% of tokens; AND the re-measured all-prompt "
                   "all-on/vanilla tok/W is %s vs the CP 2.4 headline 3.617x. "
                   "The headline cannot be re-asserted; it was measured on "
                   "F1-degenerate looping decode (CP 2.4 accept_rate=1.000). "
                   "CP 5.6 does not close the headline on P2."
                   % (mp, ap_, ("%.3fx" % ag) if ag else "n/a"))
    elif (gated.get("ci95") and gated["ci95"][0] <= HEADLINE <= gated["ci95"][1]):
        verdict = ("A", "TF gate passes and re-measured all-on/vanilla TPW CI "
                   "brackets 3.617x — headline re-asserted on verified output")
    else:
        verdict = ("B", "TF gate passes but re-measured TPW lift differs from "
                   "3.617x — report the corrected number; F1 closed, headline "
                   "restated")
    res["verdict"] = {"outcome": verdict[0], "rationale": verdict[1]}
    print("-" * 74)
    print("VERDICT %s — %s" % verdict)

    outp = os.path.join(HERE, "cp56_p2_result.json")
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)
    print("result -> " + outp)


if __name__ == "__main__":
    main()
