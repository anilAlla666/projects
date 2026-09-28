#!/bin/bash
# Extension probe: tok/W was monotonically decreasing with clock above
# 1200 MHz, so the optimum may be below.  Scan 900, 1000, 1100 MHz.
set -u
cd /home/ubuntu/op31-prod-fix
ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
PREFILL=128
MEASURE=8

mkdir -p /tmp/sweep

export CIPHER_SENSE=on CIPHER_SHIELD=on CIPHER_SUSTAIN=on
export CIPHER_THERMOSTAT=on CIPHER_PULSE=on CIPHER_VOLT=on
export CIPHER_HIBERNATE=on CIPHER_LOOP=on CIPHER_CONTINUITY=on
export CIPHER_SUBSTITUTE_V2=on CIPHER_WEIGHT_COMPRESS=on
export CIPHER_FUSION_KERNELS=on CIPHER_FP8_COMPUTE=on
export CIPHER_PREDICT=on CIPHER_GUARD=on CIPHER_DETERMINISM=on
export CIPHER_TOPOLOGY=on CIPHER_TRACE=on CIPHER_FAIRNESS=on
export CIPHER_CARBON=on CIPHER_RECEIPT=on CIPHER_COMPLY=on
export CIPHER_PERSIST_ENGINE=on CIPHER_GRAPH=on
export CIPHER_KV_COMPRESS=on CIPHER_NCCL_V4=on
export CIPHER_PARTITION_ROUTER=on CIPHER_THERMAL_FEEDBACK=on
export CIPHER_VMM=on
export LD_PRELOAD="$LD"

for B in 32 64; do
    for mhz in 900 1000 1100; do
        echo
        echo "→ B=$B  clock=$mhz MHz"
        sudo -n nvidia-smi -lgc $mhz > /dev/null
        python3 step7_fp8_eager.py --mode=full \
            --batches=$B --prefill=$PREFILL --measure=$MEASURE \
            --out=/tmp/sweep/full_b${B}_${mhz}.json 2>&1 | tail -3
    done
done
sudo -n nvidia-smi -rgc > /dev/null

python3 - <<'PY'
import json
print()
print("=" * 100)
print(" CLOCK SWEEP COMBINED — B=32, B=64 (full mode), all probes")
print("=" * 100)
print(f"{'B':>3} {'clock':>5}  {'baseline tok/W':>14}   "
      f"{'full tps':>10} {'full W':>9} {'full tok/W':>12}   "
      f"{'×tps':>5} {'×tok/W':>7}")
all_clocks = [900, 1000, 1100, 1200, 1350, 1500, 1650]
best = {}
for B in (32, 64):
    b = json.load(open(f"/tmp/sweep/base_b{B}.json"))["rows"][0]
    best_tw = 0; best_mhz = 0; best_full = None
    for mhz in all_clocks:
        try:
            c = json.load(open(f"/tmp/sweep/full_b{B}_{mhz}.json"))["rows"][0]
        except (FileNotFoundError, IndexError):
            continue
        s_tps = c['tps']/b['tps']
        s_tw  = c['tok_w']/b['tok_w']
        if s_tw > best_tw:
            best_tw = s_tw; best_mhz = mhz; best_full = c
        marker = " *" if mhz == best_mhz else ""
        print(f"{B:>3} {mhz:>5}  {b['tok_w']:>14.4f}   "
              f"{c['tps']:>10.2f} {c['draw_w']:>9.1f} {c['tok_w']:>12.4f}   "
              f"{s_tps:>5.2f} {s_tw:>7.2f}")
    best[B] = (best_mhz, best_tw, best_full)
    print()

print("=" * 100)
print(" PER-BATCH OPTIMUM (max tok/W across all probed clocks)")
print("=" * 100)
for B, (mhz, tw, c) in best.items():
    print(f"  B={B}: clock={mhz} MHz  tok/W ×{tw:.2f}  ({c['tps']:.0f} tps, "
          f"{c['draw_w']:.1f} W, tok/W={c['tok_w']:.4f})  last={c['last_text']!r}")
PY
