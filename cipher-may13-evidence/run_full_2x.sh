#!/bin/bash
# Final 2x measurement with graph-capture "full" stack.
# Baseline: DEFAULT clock, no CIPHER, no LD_PRELOAD, eager.
# FULL:     LOCKED clock per-batch optimum, INT4 GEMV (M=1) +
#           FP8 (M=2..64) + fused RMSNorm + fused SiLU + CUDA graph
#           capture + all ops on.
set -u
cd /home/ubuntu/op31-prod-fix

ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
PREFILL=128
MEASURE=8

declare -A BEST=( [1]=1200 [8]=1200 [32]=1200 [64]=1200 )

mkdir -p /tmp/full2x

echo "=== BASELINE (default clock, no CIPHER) ==="
sudo -n nvidia-smi -rgc > /dev/null
for B in 1 8 32 64; do
    python3 step7_fp8_eager.py --mode=baseline \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/full2x/base_b${B}.json 2>&1 | tail -3
done

echo
echo "=== FULL (locked clock, INT4+FP8+fusion+graph) ==="
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
    python3 step7_fp8_eager.py --mode=full \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/full2x/full_b${B}.json 2>&1 | tail -5
done

sudo -n nvidia-smi -rgc > /dev/null

python3 - <<'PY'
import json
print()
print("=" * 90)
print(" FULL STACK (graph capture) vs BASELINE (default clock, no CIPHER)")
print("=" * 90)
print(f"{'B':>3}  {'baseline tps':>12} {'baseline W':>10} {'baseline tok/W':>14}   "
      f"{'full tps':>10} {'full W':>9} {'full tok/W':>13}   "
      f"{'×tps':>5} {'×tok/W':>7}")
for B in (1, 8, 32, 64):
    b = json.load(open(f"/tmp/full2x/base_b{B}.json"))["rows"][0]
    c = json.load(open(f"/tmp/full2x/full_b{B}.json"))["rows"][0]
    s_tps = c['tps']/b['tps']
    s_tw  = c['tok_w']/b['tok_w']
    print(f"{B:>3}  {b['tps']:>12.2f} {b['draw_w']:>10.1f} {b['tok_w']:>14.4f}   "
          f"{c['tps']:>10.2f} {c['draw_w']:>9.1f} {c['tok_w']:>13.4f}   "
          f"{s_tps:>5.2f} {s_tw:>7.2f}")
    print(f"     baseline last_text={b['last_text']!r}  full last_text={c['last_text']!r}")
PY
