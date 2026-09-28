#!/usr/bin/env python3
# CPU-only. Consolidate MFU_JSON from all sweep logs -> markdown tables + Q2 derived metrics + roofline plots.
import json, glob, math, sys
PLOTS = "--plots" in sys.argv

def load(path):
    for line in open(path):
        if line.startswith("MFU_JSON "):
            return json.loads(line[len("MFU_JSON "):])
    return None

runs = {}
for p in glob.glob("/home/ubuntu/mfu_*.log"):
    r = load(p)
    if r: runs[r["tag"]] = r
ridge = next(iter(runs.values()))["ridge"]
PK_BF16_1200 = 989.4 * 1200/1980      # 599.6
PK_FP8_1200  = 1978.9 * 1200/1980     # 1199.3

def rows(tag, reg): return [x for x in runs[tag]["results"] if x["regime"] == reg]

# ---- bf16 prefill2048 curve: master(1,8,32,64) + m128 retry(128) ----
pf = {x["batch"]: x for x in rows("bf16_master", "prefill2048")}
for x in rows("bf16_m128exp", "prefill2048"): pf[x["batch"]] = x
pf2048 = [pf[b] for b in sorted(pf)]

print("## Q1 prefill SEQ=2048 (compute-bound spine), bf16, locked 1200MHz")
print("| M | tok/s | TFLOP/s | MFU% (vs bf16 peak 600) | AI | I/I* | roofline pred | clk | W | mem MB |")
print("|--|--|--|--|--|--|--|--|--|--|")
for x in pf2048:
    print(f"| {x['batch']} | {x['tps']:,.0f} | {x['tflops']:.1f} | {x['mfu']:.1f} | {x['AI']:.0f} | {x['AI_over_Istar']:.2f} | {x['roofline_pred']:.2f} | {x['clk_min']}={x['clk_max']} | {x['powW']} | {x['memMB']:,} |")

for reg, label in [("prefill16", "Positive control: prefill SEQ=16 (memory-bound at low M)"),
                   ("decode", "Decode (engine-realistic, 1 tok/seq, ctx~520, EAGER)")]:
    print(f"\n## {label}, bf16, locked 1200MHz")
    print("| M | tok/s | TFLOP/s | MFU% | AI | I/I* | roofline pred | W |")
    print("|--|--|--|--|--|--|--|--|")
    for x in rows("bf16_master", reg):
        print(f"| {x['batch']} | {x['tps']:,.0f} | {x['tflops']:.2f} | {x['mfu']:.1f} | {x['AI']:.1f} | {x['AI_over_Istar']:.2f} | {x['roofline_pred']:.2f} | {x['powW']} |")

# ---- Q2: FP8 ON vs OFF (both injection), locked 1200 ----
off = {x["batch"]: x for x in rows("fp8off", "prefill2048")}
on  = {x["batch"]: x for x in rows("fp8on",  "prefill2048")}
print("\n## Q2 FP8 substitution ON vs OFF (injection, same locked 1200MHz), prefill SEQ=2048")
print("| M | bf16 tok/s | FP8 tok/s | tok/s ratio | bf16 MFU%(/bf16pk) | FP8 MFU%(/FP8pk) | FP8 'MFU'%(/bf16pk) | bf16 W | FP8 W | tok/W gain | fp8 handled | wq |")
print("|--|--|--|--|--|--|--|--|--|--|--|--|")
for b in sorted(off):
    o, n = off[b], on[b]
    tr = n["tps"]/o["tps"]
    mfu_fp8_vs_bf16pk = n["tflops"]/PK_BF16_1200*100
    tokw = tr * (o["powW"]/n["powW"])
    print(f"| {b} | {o['tps']:,.0f} | {n['tps']:,.0f} | {tr:.3f}x | {o['mfu']:.1f} | {n['mfu']:.1f} | {mfu_fp8_vs_bf16pk:.1f} | {o['powW']} | {n['powW']} | {tokw:.2f}x | {n['fp8_handled']} | {n['fp8_wq']} |")

if not PLOTS:
    print("\n(skip plots; run with --plots once matplotlib is available)"); sys.exit(0)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---- Plot 1: MFU vs batch + roofline overlay, 3 regimes ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
panels = [(pf2048, "Prefill SEQ=2048 (compute-bound spine)"),
          (rows("bf16_master", "prefill16"), "Prefill SEQ=16 (positive control)"),
          (rows("bf16_master", "decode"), "Decode (engine-realistic, EAGER)")]
for ax, (rws, title) in zip(axes, panels):
    bs = [x["batch"] for x in rws]; mfu = [x["mfu"] for x in rws]; pred = [x["roofline_pred"]*100 for x in rws]
    ax.plot(bs, mfu, "o-", color="#1f77b4", lw=2.2, label="measured MFU")
    ax.plot(bs, pred, "s--", color="#d62728", lw=1.8, label="roofline pred  min(1, I(M)/I*)")
    ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs, fontsize=8)
    ax.set_ylim(0, 105); ax.set_xlabel("batch M"); ax.set_ylabel("MFU (%)")
    ax.set_title(title, fontsize=9.5); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
fig.suptitle("MFU vs batch vs roofline  (Mistral-7B, H100 SXM, bf16, locked 1200 MHz)", fontsize=11)
fig.tight_layout(); fig.savefig("/home/ubuntu/mfu_vs_batch.png", dpi=115); print("\nsaved mfu_vs_batch.png")

# ---- Plot 2: classic roofline, MFU vs AI, all regimes, ridge marked ----
fig2, ax = plt.subplots(figsize=(8.2, 5.2))
colmap = {"prefill2048": ("#1f77b4", pf2048),
          "prefill16": ("#2ca02c", rows("bf16_master", "prefill16")),
          "decode": ("#ff7f0e", rows("bf16_master", "decode"))}
for reg, (col, rws) in colmap.items():
    ai = [x["AI"] for x in rws]; mfu = [x["mfu"] for x in rws]
    ax.plot(ai, mfu, "o-", color=col, lw=1.9, label=reg)
    for x in rws: ax.annotate(f"M{x['batch']}", (x["AI"], x["mfu"]), fontsize=6, xytext=(2,3), textcoords="offset points")
ax.axvline(ridge, color="k", ls=":", lw=1.4, label=f"ridge I*={ridge:.0f} FLOP/byte")
xx = np.logspace(0, 3.7, 200); ax.plot(xx, np.minimum(1, xx/ridge)*100, "r--", lw=1.1, alpha=0.7, label="roofline ceiling (bandwidth)")
ax.set_xscale("log"); ax.set_xlabel("arithmetic intensity  I  (FLOP/byte)"); ax.set_ylabel("MFU (%)")
ax.set_ylim(0, 105); ax.set_title("Roofline: MFU vs arithmetic intensity (Mistral-7B, H100, bf16, 1200 MHz)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig("/home/ubuntu/mfu_roofline.png", dpi=115); print("saved mfu_roofline.png")
