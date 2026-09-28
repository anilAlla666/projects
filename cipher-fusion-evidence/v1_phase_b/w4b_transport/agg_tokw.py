"""W.4b.6 decode-window tok/W aggregator.

For each (result.json, power.csv, sentinel) triple: clip power samples to the
[DECODE_START, DECODE_END] window (sentinel wall-clock stamps), mean the power
column, and report tok/W = decode_tok_s / mean_power_W (matches W.4b.5 §E:
in-proc B=4 77.2 tok/s / 146.6 W = 0.527 tok/W). Prints a markdown row per cond
and the substrate-attributable + in-proc-lever ratios.
"""
import sys
import json


def window(sent_path):
    start = end = None
    for ln in open(sent_path):
        if ln.startswith("DECODE_START"):
            start = float(ln.split()[1])
        elif ln.startswith("DECODE_END"):
            end = float(ln.split()[1])
    return start, end


def mean_power(csv_path, start, end):
    ps = []
    for ln in open(csv_path):
        parts = ln.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0]); pw = float(parts[1])
        except ValueError:
            continue
        if start is not None and (ts < start or (end is not None and ts > end)):
            continue
        ps.append(pw)
    return (sum(ps) / len(ps), len(ps)) if ps else (float("nan"), 0)


def cond(name, result_json, power_csv, sent):
    r = json.load(open(result_json))
    toks = r.get("decode_tok_s")
    s, e = window(sent)
    mp, n = mean_power(power_csv, s, e)
    tokw = toks / mp if (toks and mp == mp and mp > 0) else float("nan")
    return {"name": name, "tok_s": toks, "W": round(mp, 1), "n_samp": n,
            "tok_w": round(tokw, 4),
            "peak_hbm": r.get("peak_hbm_gib")}


if __name__ == "__main__":
    # argv: name=result.json:power.csv:sentinel  (repeatable); roles cross/inproc/naive
    rows = []
    for a in sys.argv[1:]:
        name, rest = a.split("=", 1)
        rj, pc, st = rest.split(":")
        rows.append(cond(name, rj, pc, st))
    print("| condition | tok/s | W | tok/W | peakHBM | n |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %.1f | %.1f | %.4f | %s | %d |" %
              (r["name"], r["tok_s"], r["W"], r["tok_w"], r["peak_hbm"], r["n_samp"]))
    d = {r["name"]: r["tok_w"] for r in rows}
    if "cross" in d and "naive" in d:
        print("\nsubstrate-attributable (cross / naive) = %.3f" % (d["cross"] / d["naive"]))
    if "inproc" in d and "naive" in d:
        print("in-proc lever (inproc / naive)          = %.3f" % (d["inproc"] / d["naive"]))
    if "cross" in d and "inproc" in d:
        print("cross / inproc (transport-banked check) = %.3f" % (d["cross"] / d["inproc"]))
