#!/bin/bash
# Re-run B=32 at 1200 MHz and B=64 at 1350 MHz only.
set -u
cd /home/ubuntu/op31-prod-fix

ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
PREFILL=128
MEASURE=8

# Baseline (default clock, no CIPHER) — re-measure to keep the comparison
# fresh on the same cooled-down clock.
echo "=== BASELINE B=32 / B=64 (default clock) ==="
sudo -n nvidia-smi -rgc > /dev/null
for B in 32 64; do
    python3 step7_fp8_eager.py --mode=baseline \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/final2x/base_b${B}.json 2>&1 | tail -3
done

# CIPHER with full ops + new clocks.
echo
echo "=== CIPHER B=32 (1200 MHz) / B=64 (1350 MHz) ==="
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

sudo -n nvidia-smi -lgc 1200 > /dev/null
echo "  B=32 locked 1200 MHz"
python3 step7_fp8_eager.py --mode=fp8 --batches=32 --prefill=$PREFILL \
    --measure=$MEASURE --out=/tmp/final2x/cipher_b32.json 2>&1 | tail -3

sudo -n nvidia-smi -lgc 1350 > /dev/null
echo "  B=64 locked 1350 MHz"
python3 step7_fp8_eager.py --mode=fp8 --batches=64 --prefill=$PREFILL \
    --measure=$MEASURE --out=/tmp/final2x/cipher_b64.json 2>&1 | tail -3

sudo -n nvidia-smi -rgc > /dev/null

python3 - <<'PY'
import json
print()
print("=" * 86)
print(" Re-run B=32, B=64 (revised optimal clocks)")
print("=" * 86)
print(f"{'B':>3} {'mhz':>5} {'baseline tps':>12} {'baseline W':>10} "
      f"{'baseline tok/W':>14}   "
      f"{'cipher tps':>10} {'cipher W':>9} {'cipher tok/W':>13}   "
      f"{'×tps':>5} {'×tok/W':>7}")
for B, mhz in [(32, 1200), (64, 1350)]:
    b = json.load(open(f"/tmp/final2x/base_b{B}.json"))["rows"][0]
    c = json.load(open(f"/tmp/final2x/cipher_b{B}.json"))["rows"][0]
    s_tps = c['tps']/b['tps']; s_tw = c['tok_w']/b['tok_w']
    print(f"{B:>3} {mhz:>5} {b['tps']:>12.2f} {b['draw_w']:>10.1f} "
          f"{b['tok_w']:>14.4f}   {c['tps']:>10.2f} {c['draw_w']:>9.1f} "
          f"{c['tok_w']:>13.4f}   {s_tps:>5.2f} {s_tw:>7.2f}")
PY
