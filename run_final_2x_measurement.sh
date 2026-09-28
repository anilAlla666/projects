#!/bin/bash
# Final 2x stack measurement.
#
# Baseline:  DEFAULT clock (no -lgc), no CIPHER, eager mode.
# CIPHER:    LOCKED clock per-batch optimum, FP8 + fusion + all ops on, eager.
#
# Optimal clocks per batch (revised after first run):
#   B=1   -> 1000 MHz   (decode-bound, low compute, save power)
#   B=8   -> 1200 MHz   (FP8 sweet spot)
#   B=32  -> 1200 MHz   (was 1350; trying 1200 to save more power)
#   B=64  -> 1350 MHz   (was 1000 — too aggressive, -36% tps; need more flops)
#
# Each (B, mode) pair runs in its own process so the clock can be set
# right before the run.

set -u
cd /home/ubuntu/op31-prod-fix

ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
PREFILL=128
MEASURE=8

# Best clocks per batch (MHz).
declare -A BEST=( [1]=1000 [8]=1200 [32]=1200 [64]=1350 )

# ------------ Baseline runs (DEFAULT clock, no LD_PRELOAD, no CIPHER) -----
echo "================================================================"
echo "  BASELINE: default clock, no CIPHER, eager"
echo "================================================================"
sudo -n nvidia-smi -rgc > /dev/null
mkdir -p /tmp/final2x
BASE_OUT=/tmp/final2x/baseline.json
echo "[]" > "$BASE_OUT"

for B in 1 8 32 64; do
    python3 step7_fp8_eager.py --mode=baseline \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/final2x/base_b${B}.json 2>&1 | tail -3
done

# ------------ CIPHER runs (LOCKED clock per batch, all ops on) ------------
echo
echo "================================================================"
echo "  CIPHER FP8 + fusion + all ops, locked clock per batch"
echo "================================================================"

# Same env-var set the regression used.
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

for B in 1 8 32 64; do
    mhz=${BEST[$B]}
    echo
    echo "→ B=$B  locking clock to $mhz MHz"
    sudo -n nvidia-smi -lgc $mhz > /dev/null
    python3 step7_fp8_eager.py --mode=fp8 \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/final2x/cipher_b${B}.json 2>&1 | tail -3
done

sudo -n nvidia-smi -rgc > /dev/null
echo "[final2x] clock restored"

# ------------ Summary -----------------------------------------------------
python3 - <<'PY'
import json
import os
results = {"baseline": {}, "cipher": {}}
for B in (1, 8, 32, 64):
    for tag, prefix in (("baseline","base"), ("cipher","cipher")):
        try:
            r = json.load(open(f"/tmp/final2x/{prefix}_b{B}.json"))["rows"][0]
        except Exception:
            r = {"tps": 0, "draw_w": 0, "tok_w": 0, "fp8_delta": {}}
        results[tag][B] = r

print()
print("=" * 86)
print(" FINAL 2x STACK MEASUREMENT")
print("=" * 86)
print(f"{'B':>3}  {'baseline tps':>12} {'baseline W':>10} {'baseline tok/W':>14}   "
      f"{'cipher tps':>10} {'cipher W':>9} {'cipher tok/W':>13}   "
      f"{'×tps':>5} {'×tok/W':>7} {'fp8_calls':>10}")
for B in (1, 8, 32, 64):
    b = results["baseline"][B]; c = results["cipher"][B]
    s_tps = c['tps']/b['tps'] if b['tps'] else 0
    s_tw  = c['tok_w']/b['tok_w'] if b['tok_w'] else 0
    calls = c.get("fp8_delta", {}).get("calls", 0)
    print(f"{B:>3}  {b['tps']:>12.2f} {b['draw_w']:>10.1f} {b['tok_w']:>14.4f}   "
          f"{c['tps']:>10.2f} {c['draw_w']:>9.1f} {c['tok_w']:>13.4f}   "
          f"{s_tps:>5.2f} {s_tw:>7.2f} {calls:>10}")
PY
