#!/bin/bash
# Full stack measurement v2: residual + RoPE fusion added on top.
# Per-batch optimal clocks: B=1,8 → 1200 MHz; B=32,64 → 1100 MHz.
set -u
cd /home/ubuntu/op31-prod-fix
ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
PREFILL=128
MEASURE=8
declare -A BEST=( [1]=1200 [8]=1200 [32]=1100 [64]=1100 )

mkdir -p /tmp/full2x_v2
echo "=== BASELINE (default clock, no CIPHER) ==="
sudo -n nvidia-smi -rgc > /dev/null
for B in 1 8 32 64; do
    python3 step7_fp8_eager.py --mode=baseline \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/full2x_v2/base_b${B}.json 2>&1 | tail -3
done

echo
echo "=== FULL v2 (residual + RoPE fusion added) ==="
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
    echo "→ B=$B clock=$mhz MHz"
    sudo -n nvidia-smi -lgc $mhz > /dev/null
    python3 step7_fp8_eager.py --mode=full \
        --batches=$B --prefill=$PREFILL --measure=$MEASURE \
        --out=/tmp/full2x_v2/full_b${B}.json 2>&1 | tail -3
done
sudo -n nvidia-smi -rgc > /dev/null

python3 - <<'PY'
import json
print()
print("=" * 100)
print(" FULL v2 (graph + INT4 + FP8 + RMSNorm + SiLU + Residual + RoPE) vs BASELINE")
print("=" * 100)
print(f"{'B':>3}  {'baseline tps':>12} {'baseline W':>10} {'baseline tok/W':>14}   "
      f"{'full tps':>10} {'full W':>9} {'full tok/W':>13}   "
      f"{'×tps':>5} {'×tok/W':>7}  last")
for B in (1, 8, 32, 64):
    try:
        b = json.load(open(f"/tmp/full2x_v2/base_b{B}.json"))["rows"][0]
        c = json.load(open(f"/tmp/full2x_v2/full_b{B}.json"))["rows"][0]
    except Exception as e:
        print(f"{B:>3}  (missing: {e})")
        continue
    s_tps = c['tps']/b['tps']; s_tw = c['tok_w']/b['tok_w']
    print(f"{B:>3}  {b['tps']:>12.2f} {b['draw_w']:>10.1f} {b['tok_w']:>14.4f}   "
          f"{c['tps']:>10.2f} {c['draw_w']:>9.1f} {c['tok_w']:>13.4f}   "
          f"{s_tps:>5.2f} {s_tw:>7.2f}  base={b['last_text']!r} full={c['last_text']!r}")
PY
