"""CP 2.4 — DVFS envelope analysis.

For each clock point M<mhz>/, read pair{1..N}_{off,on}/{watts.csv,progress.txt}.
watts.csv is sentinel-windowed by run_dvfs_sweep.sh — it already covers exactly
the decode loop (model load + Marlin NVRTC compile + warmup excluded; CP 2.4
DVFS harness fix, 2026-05-15). Power is therefore averaged over the whole file
with only a 1 s lead/tail trim, to drop the first/last sample, which may
straddle the DECODE_START/END sentinel edge or catch a clock transition. tok/s
comes from the run_for_duration progress log; tok/W = tok_s / decode-window W.
matched-pair Δtok/W with 95% CI is aggregated into an envelope table + verdict.

Decode-window power averaging + matched pairs = the T4.3 envelope methodology,
so the result is directly comparable to the T4.3 -2%..-14% Mistral-7B FP16
envelope. The workload here is Mistral-7B INT4 (Marlin on) — the question is
whether INT4 moves the envelope positive.

Verdict rule (memo §3.3, firm): DVFS is IN the composed gate only if some
clock point shows a Δtok/W whose 95% CI lower bound is > 0. Otherwise OUT.
No re-tuning, no cherry-picking.
"""
import os, sys, json, math, re

HERE = "/home/ubuntu/cipher-fusion-evidence/cp_2_4/dvfs_envelope"
MHZ_GRID = [int(x) for x in os.environ.get("MHZ_GRID", "1000 1200 1400 1600 1800").split()]
N = int(os.environ.get("N", "5"))
DUR = int(os.environ.get("DUR", "60"))
# watts.csv is already the decode window (sentinel-windowed by the harness),
# so the trim is just 1 s off each end for sampler/clock-transition settle —
# not the old 5 s/2 s, which was compensating for the now-fixed window bug.
TRIM_LEAD, TRIM_TAIL = 1, 1


def mean(xs): return sum(xs) / len(xs) if xs else 0.0
def std(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_progress(p):
    use = None
    try:
        with open(p) as f:
            for line in f:
                if "tokens=" in line and "iters=" in line:
                    use = line.strip()
                    if line.startswith("ENV") and " end " in line:
                        break
    except FileNotFoundError:
        return None
    if not use: return None
    mt, mi, mtt = (re.search(r"tokens=(\d+)", use), re.search(r"iters=(\d+)", use),
                   re.search(r"t=([\d.]+)s", use))
    if not (mt and mi and mtt): return None
    secs = float(mtt.group(1))
    return {"tokens": int(mt.group(1)), "secs": secs,
            "tok_s": int(mt.group(1)) / max(secs, 1e-6)}


def parse_watts(p):
    rows = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("ts_s"): continue
                a = line.split(",")
                try: rows.append((int(a[0]), float(a[1]), int(a[2])))
                except Exception: pass
    except FileNotFoundError:
        return None
    if not rows: return None
    # watts.csv spans exactly the decode window; window = whole file minus trim.
    end = max(r[0] for r in rows)
    win = [r for r in rows if TRIM_LEAD <= r[0] <= end - TRIM_TAIL]
    if not win: return None
    return {"watts_mean": mean([r[1] for r in win]),
            "clock_mean": mean([r[2] for r in win]), "n_samples": len(win)}


def analyze_clock(mhz):
    root = f"{HERE}/M{mhz}"
    per = {"off": [], "on": []}
    for arm in ("off", "on"):
        for i in range(1, N + 1):
            d = f"{root}/pair{i}_{arm}"
            w, pr = parse_watts(f"{d}/watts.csv"), parse_progress(f"{d}/progress.txt")
            if not w or not pr:
                print(f"  !! missing M{mhz} pair{i}_{arm}")
                continue
            tw = pr["tok_s"] / w["watts_mean"] if w["watts_mean"] else 0
            per[arm].append({"pair": i, "watts": round(w["watts_mean"], 3),
                             "clock_mhz": round(w["clock_mean"], 1),
                             "tok_s": round(pr["tok_s"], 3), "tok_w": round(tw, 6)})
    res = {"clock_mhz": mhz, "per_pair": per}
    n = min(len(per["off"]), len(per["on"]))
    if n >= 2:
        diffs = [per["on"][i]["tok_w"] - per["off"][i]["tok_w"] for i in range(n)]
        off_tw = [per["off"][i]["tok_w"] for i in range(n)]
        m_off = mean(off_tw)
        m_d, s_d = mean(diffs), std(diffs)
        stderr = s_d / math.sqrt(n)
        pct = 100.0 * m_d / m_off if m_off else 0
        pct_se = 100.0 * stderr / m_off if m_off else 0
        res["delta_tok_w"] = {
            "n": n, "off_tok_w_mean": round(m_off, 5),
            "on_tok_w_mean": round(mean([per["on"][i]["tok_w"] for i in range(n)]), 5),
            "off_clock_mhz": round(mean([per["off"][i]["clock_mhz"] for i in range(n)]), 0),
            "on_clock_mhz": round(mean([per["on"][i]["clock_mhz"] for i in range(n)]), 0),
            "pct_delta": round(pct, 3),
            "pct_95lo": round(pct - 1.96 * pct_se, 3),
            "pct_95hi": round(pct + 1.96 * pct_se, 3)}
    return res


def main():
    env = {"cp": "2.4", "test": "DVFS_envelope_post_marlin_int4",
           "workload": "Mistral-7B-v0.1 INT4 (Marlin on), B=1 decode",
           "methodology": "matched-pair (VOLT off vs on), n=%d, decode-window "
                          "power avg, T4.3-comparable" % N,
           "t4_3_reference": "Mistral-7B FP16 envelope was -2%% to -14%% Δtok/W",
           "clocks": []}
    print("clock   off tok/W   on tok/W   off→on clk   Δtok/W %   95%% CI")
    print("-" * 66)
    for mhz in MHZ_GRID:
        c = analyze_clock(mhz)
        env["clocks"].append(c)
        d = c.get("delta_tok_w")
        if d:
            print(f"{mhz:5d}   {d['off_tok_w_mean']:8.5f}   {d['on_tok_w_mean']:8.5f}   "
                  f"{d['off_clock_mhz']:.0f}→{d['on_clock_mhz']:.0f}   "
                  f"{d['pct_delta']:+7.2f}    {d['pct_95lo']:+.2f}..{d['pct_95hi']:+.2f}")
        else:
            print(f"{mhz:5d}   INSUFFICIENT DATA")

    # Verdict: DVFS IN only if some clock's 95% CI lower bound > 0.
    positives = [c for c in env["clocks"]
                 if c.get("delta_tok_w") and c["delta_tok_w"]["pct_95lo"] > 0]
    if positives:
        best = max(positives, key=lambda c: c["delta_tok_w"]["pct_delta"])
        env["verdict"] = {
            "dvfs_in_composed_gate": True,
            "best_clock_mhz": best["clock_mhz"],
            "best_delta_pct": best["delta_tok_w"]["pct_delta"],
            "rationale": "post-Marlin-INT4 moves DVFS positive; CI lower bound > 0"}
    else:
        env["verdict"] = {
            "dvfs_in_composed_gate": False,
            "rationale": "no clock point shows Δtok/W with 95%% CI lower bound > 0 "
                         "on the post-Marlin-INT4 Mistral-7B workload — neutral or "
                         "negative across the envelope. DVFS OUT of the composed "
                         "gate (memo §3.3 disable rule). Not re-tuned."}
    print("-" * 66)
    print("VERDICT:", "DVFS IN" if env["verdict"]["dvfs_in_composed_gate"]
          else "DVFS OUT", "—", env["verdict"]["rationale"])
    with open(f"{HERE}/dvfs_envelope_result.json", "w") as f:
        json.dump(env, f, indent=2)
    print("result ->", f"{HERE}/dvfs_envelope_result.json")


if __name__ == "__main__":
    main()
