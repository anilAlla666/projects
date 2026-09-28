#!/bin/bash
# Long-context measurement: B=8 prefill=2048, full CIPHER stack +
# V3 enabled (gated to B=1; at B=8 it falls through to original FA staging).
set -u
cd /home/ubuntu/op31-prod-fix

ROOT=/home/ubuntu/op31-prod-fix
LD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"

mkdir -p /tmp/final2x

echo "=== BASELINE B=8 P=2048 (default clock, no CIPHER) ==="
sudo -n nvidia-smi -rgc > /dev/null
python3 step7_fp8_eager.py --mode=baseline --batches=8 --prefill=2048 \
    --measure=10 --out=/tmp/final2x/base_b8_p2048.json 2>&1 | tail -3

echo
echo "=== CIPHER B=8 P=2048 (1200 MHz, full stack + KV V3 enabled) ==="
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
# V3 wired in per spec; gated to B=1 internally.
export CIPHER_KV_REDIRECT=on
export CIPHER_KV_RDR_V3=on
export CIPHER_KV_RDR_V3_ROPE=on        # apply RoPE inline (K cached pre-RoPE)
export CIPHER_KV_BATCH=8
export CIPHER_KV_MAX_CACHE_LEN=2400    # prefill 2048 + decode ≤ 320 + slack
export LD_PRELOAD="$LD"

sudo -n nvidia-smi -lgc 1200 > /dev/null
python3 step7_fp8_eager.py --mode=fp8 --batches=8 --prefill=2048 \
    --measure=10 --out=/tmp/final2x/cipher_b8_p2048.json 2>&1 | tail -3

sudo -n nvidia-smi -rgc > /dev/null

python3 - <<'PY'
import json
b = json.load(open("/tmp/final2x/base_b8_p2048.json"))["rows"][0]
c = json.load(open("/tmp/final2x/cipher_b8_p2048.json"))["rows"][0]
calls = c["fp8_delta"].get("calls", 0)
print()
print("=" * 86)
print(" B=8 PREFILL=2048 long-context measurement")
print("=" * 86)
print(f" baseline:  tps={b['tps']:.2f}   W={b['draw_w']:.1f}   tok/W={b['tok_w']:.4f}")
print(f" CIPHER:    tps={c['tps']:.2f}   W={c['draw_w']:.1f}   tok/W={c['tok_w']:.4f}")
print(f" ratios:    ×tps={c['tps']/b['tps']:.2f}   ×tok/W={c['tok_w']/b['tok_w']:.2f}")
print(f" fp8_calls: {calls}")
PY
