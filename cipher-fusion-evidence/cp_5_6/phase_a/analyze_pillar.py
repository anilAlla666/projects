"""Phase A — three-pillar per-workload rollup (PASS 1; PASS 2 when present).

Usage: python3 analyze_pillar.py WL01
Reads phase_a/<WL>/{gold,vanilla,marlin,allon}.json + <arm>.watts.csv.
Per arm: aggregate tok/s, arm-level mean power, tok/W, MFU (analytical FLOPs
/ H100 peak / wall), teacher-forced gate pass count. TPW reported both gated
(passing pairs only, per the operator spec) and all-prompt. Emits
<WL>/<WL>_pillar.json and prints a PILLAR_MATRIX.md-ready section.
"""
import os
import sys
import csv
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["vanilla", "marlin", "allon"]
THRESHOLD = 0.99


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def load_watts(d, arm):
    rows = []
    try:
        for r in csv.DictReader(open(os.path.join(d, "%s.watts.csv" % arm))):
            try:
                rows.append(float(r["power_w"]))
            except (ValueError, KeyError):
                pass
    except FileNotFoundError:
        pass
    return mean(rows) if rows else None


def analyze(wl):
    d = os.path.join(HERE, wl)
    res = {"wl_id": wl, "pass1": {}}
    peak = None
    for arm in ARMS:
        try:
            j = json.load(open(os.path.join(d, "%s.json" % arm)))
        except (FileNotFoundError, json.JSONDecodeError):
            res["pass1"][arm] = None
            continue
        peak = j["peak_flops"]
        pr = j["prompts"]
        power = load_watts(d, arm)
        tot_gen = sum(p["total_gen"] for p in pr)
        tot_wall = sum(p["wall_s"] for p in pr)
        tot_flops = sum(p["flops"] for p in pr)
        tok_s = tot_gen / tot_wall if tot_wall else 0.0
        mfu = (tot_flops / (peak * tot_wall)) if tot_wall else 0.0
        tok_w = (tok_s / power) if power else None
        passing = [p for p in pr if p["tf_agreement"] >= THRESHOLD]
        # gated tok/s and tok/W over passing pairs only
        gw = sum(p["wall_s"] for p in passing)
        gg = sum(p["total_gen"] for p in passing)
        g_tok_s = (gg / gw) if gw else None
        g_tok_w = (g_tok_s / power) if (g_tok_s and power) else None
        res["pass1"][arm] = {
            "n_params": j["n_params"], "batch": j["batch"],
            "tok_s_allprompt": round(tok_s, 3),
            "mean_power_w": round(power, 2) if power else None,
            "tok_w_allprompt": round(tok_w, 5) if tok_w else None,
            "mfu_pct": round(100.0 * mfu, 4),
            "tf_pass": len(passing), "tf_total": len(pr),
            "tf_agreements": [p["tf_agreement"] for p in pr],
            "tok_s_gated": round(g_tok_s, 3) if g_tok_s else None,
            "tok_w_gated": round(g_tok_w, 5) if g_tok_w else None,
            "accept_rates": [p["accept_rate"] for p in pr]}
    v = res["pass1"].get("vanilla")
    a = res["pass1"].get("allon")
    mr = res["pass1"].get("marlin")
    if v and a:
        res["pass1"]["lift_allon_vanilla"] = {
            "mfu": round(a["mfu_pct"] / v["mfu_pct"], 4) if v["mfu_pct"] else None,
            "tok_w_allprompt": round(a["tok_w_allprompt"] / v["tok_w_allprompt"], 4)
            if (v["tok_w_allprompt"] and a["tok_w_allprompt"]) else None}
    with open(os.path.join(d, "%s_pillar.json" % wl), "w") as f:
        json.dump(res, f, indent=2)

    # printed matrix section
    print("### %s" % wl)
    print()
    if v:
        print("- model %s params, batch %d, H100 peak %.1f TFLOP/s"
              % ("%.2fB" % (v["n_params"] / 1e9), v["batch"], peak / 1e12))
    print()
    print("| arm | tok/s | W | tok/W | MFU%% | TF pass |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        r = res["pass1"].get(arm)
        if not r:
            print("| %s | MISSING |" % arm); continue
        print("| %s | %.1f | %s | %s | %.4f | %d/%d |"
              % (arm, r["tok_s_allprompt"],
                 r["mean_power_w"] if r["mean_power_w"] else "n/a",
                 ("%.5f" % r["tok_w_allprompt"]) if r["tok_w_allprompt"] else "n/a",
                 r["mfu_pct"], r["tf_pass"], r["tf_total"]))
    print()
    lift = res["pass1"].get("lift_allon_vanilla", {})
    print("- PILLAR 1 (MFU): all-on/vanilla MFU lift = %s"
          % (("%.3fx" % lift["mfu"]) if lift.get("mfu") else "n/a"))
    print("- PILLAR 2 (TPW): all-on/vanilla tok/W (all-prompt) = %s ; "
          "gated (passing pairs) = %s"
          % (("%.3fx" % lift["tok_w_allprompt"]) if lift.get("tok_w_allprompt") else "n/a",
             "n/a — substrate arms pass 0 pairs" if (a and a["tf_pass"] == 0)
             else "see json"))
    print("- PILLAR 3 (multi-tenant): PASS 2 pending")
    print()
    return res


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "WL01")
