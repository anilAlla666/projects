"""Phase A — N-tenant concurrent aggregate rollup.
Usage: python3 analyze_multitenant.py WL01 mt4_allon
Reads phase_a/<WL>/<dir>/tenant*.json + power.csv. Aggregate window =
[min tenant first-prompt t0_epoch, max tenant last-prompt t1_epoch].
aggregate tok/s = sum(all tenant tokens) / window ; power = mean full-GPU
power over window ; aggregate tok/W = tok/s / power ; aggregate MFU =
sum(FLOPs)/(peak*window). Per-tenant breakdown + KL stats. Emits agg json.
"""
import os
import sys
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main(wl, sub):
    d = os.path.join(HERE, wl, sub)
    tenants = []
    i = 1
    while os.path.exists(os.path.join(d, "tenant%d.json" % i)):
        tenants.append(json.load(open(os.path.join(d, "tenant%d.json" % i))))
        i += 1
    if not tenants:
        print("no tenant json in " + d); return
    peak = tenants[0]["peak_flops"]
    pw = []
    try:
        for r in csv.DictReader(open(os.path.join(d, "power.csv"))):
            try:
                pw.append((float(r["epoch_s"]), float(r["power_w"])))
            except (ValueError, KeyError):
                pass
    except FileNotFoundError:
        pass

    per = []
    t0s, t1s = [], []
    for k, t in enumerate(tenants, 1):
        pr = t["prompts"]
        t0 = min(p["t0_epoch"] for p in pr)
        t1 = max(p["t1_epoch"] for p in pr)
        t0s.append(t0); t1s.append(t1)
        toks = sum(p["total_gen"] for p in pr)
        flops = sum(p["flops"] for p in pr)
        kls = [p["kl"]["mean"] for p in pr if p.get("kl")]
        tf = [p["tf_agreement"] for p in pr]
        per.append({"tenant": k, "tokens": toks, "flops": flops,
                    "decode_s": round(t1 - t0, 3),
                    "tok_s": round(toks / (t1 - t0), 3) if t1 > t0 else None,
                    "kl_mean": round(mean(kls), 5) if kls else None,
                    "kl_max": round(max(kls), 5) if kls else None,
                    "tf_min": round(min(tf), 4) if tf else None})

    W0, W1 = min(t0s), max(t1s)
    window = W1 - W0
    tot_tokens = sum(p["tokens"] for p in per)
    tot_flops = sum(p["flops"] for p in per)
    win_pw = [p for (ts, p) in pw if W0 <= ts <= W1]
    mp = mean(win_pw) if win_pw else None
    agg_tok_s = tot_tokens / window if window > 0 else None
    agg_tok_w = (agg_tok_s / mp) if (agg_tok_s and mp) else None
    agg_mfu = (tot_flops / (peak * window)) if window > 0 else None

    res = {"wl_id": wl, "config": sub, "n_tenants": len(tenants),
           "window_s": round(window, 3), "total_tokens": tot_tokens,
           "agg_tok_s": round(agg_tok_s, 3) if agg_tok_s else None,
           "mean_full_gpu_power_w": round(mp, 2) if mp else None,
           "agg_tok_w": round(agg_tok_w, 5) if agg_tok_w else None,
           "agg_mfu_pct": round(100.0 * agg_mfu, 4) if agg_mfu else None,
           "power_samples_in_window": len(win_pw),
           "per_tenant": per}
    with open(os.path.join(d, "aggregate.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
