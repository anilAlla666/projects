#!/bin/bash
# Re-measure B=32 and B=64 in "full" mode at 1100 MHz with the wider
# FP8 gate (n<=512 instead of n<=64).
set -u
cd /home/ubuntu/op31-prod-fix

ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
PREFILL=128
MEASURE=8

mkdir -p /tmp/wider

echo "=== BASELINE ==="
sudo -n nvidia-smi -rgc > /dev/null
for B in 32 64; do
    python3 step7_fp8_eager.py --mode=baseline \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/wider/base_b${B}.json 2>&1 | tail -3
done

echo
echo "=== FULL (wider gate, 1100 MHz) ==="
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

sudo -n nvidia-smi -lgc 1100 > /dev/null
for B in 32 64; do
    echo "  B=$B"
    python3 step7_fp8_eager.py --mode=full \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/wider/full_b${B}.json 2>&1 | tail -3
done
sudo -n nvidia-smi -rgc > /dev/null

python3 - <<'PY'
import json
print()
print("=" * 96)
print(" WIDER FP8 GATE (n<=512) — full @ 1100 MHz vs baseline @ default")
print("=" * 96)
print(f"{'B':>3}  {'baseline tps':>12} {'baseline W':>10} {'baseline tok/W':>14}   "
      f"{'full tps':>10} {'full W':>9} {'full tok/W':>13}   "
      f"{'×tps':>5} {'×tok/W':>7} {'fp8_calls':>10}")
for B in (32, 64):
    b = json.load(open(f"/tmp/wider/base_b{B}.json"))["rows"][0]
    c = json.load(open(f"/tmp/wider/full_b{B}.json"))["rows"][0]
    s_tps = c['tps']/b['tps']; s_tw = c['tok_w']/b['tok_w']
    calls = c.get("fp8_delta", {}).get("calls", 0)
    print(f"{B:>3}  {b['tps']:>12.2f} {b['draw_w']:>10.1f} {b['tok_w']:>14.4f}   "
          f"{c['tps']:>10.2f} {c['draw_w']:>9.1f} {c['tok_w']:>13.4f}   "
          f"{s_tps:>5.2f} {s_tw:>7.2f} {calls:>10}")
PY
