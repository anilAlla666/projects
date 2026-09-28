#!/bin/bash
# CP 5.3 STEP 2 — A-tok INVESTIGATION (option ii requires the investigation).
#
# The STEP 2 A-tok gate compared partition-vs-fullgpu and got 3% agreement.
# Decoded text shows the *fullgpu reference itself* is degenerate (TinyLlama
# repetition gibberish; Mistral 122x token-0 <unk>). This script runs the
# missing leg: a GOLD reference (plain FP16, no substrate) to triangulate,
# plus a fresh fullgpu re-run (transient-state check) and a B=8 fullgpu run
# (Marlin's designed regime — localizes whether garbage is M=1-specific).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SHIPPED=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_cp5_3_step2   # c2c5d313
DECODE="$HERE/cp53_atok_decode.py"
DECODE_B="$HERE/cp53_atok_decode_batched.py"
NTOK=128
WD=1800

declare -A MODELS=(
  [tinyllama]=/home/ubuntu/models/TinyLlama-1.1B
  [mistral]=/home/ubuntu/models/Mistral-7B-v0.1
)

for tag in tinyllama mistral; do
  mp="${MODELS[$tag]}"
  echo "================ investigate: $tag ($mp) ================"

  # (1) GOLD — plain FP16, no LD_PRELOAD, no substrate at all.
  rm -f "cp53_atok_${tag}_gold.json"
  timeout $WD python3 "$DECODE" "$mp" "$NTOK" "cp53_atok_${tag}_gold.json" \
      > "cp53_atok_${tag}_gold.log" 2>&1
  echo "  gold(no-substrate) exit=$?"

  # (2) FULLGPU fresh re-run — shipped c2c5d313, Marlin on, B=1 (reproduce).
  rm -f "cp53_atok_${tag}_fullgpu_rerun.json"
  env CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
      CIPHER_TENANT_ID=1 WL_TENANT_ID=1 \
      LD_PRELOAD="$SHIPPED" CUDA_INJECTION64_PATH="$SHIPPED" \
      timeout $WD python3 "$DECODE" "$mp" "$NTOK" \
      "cp53_atok_${tag}_fullgpu_rerun.json" \
      > "cp53_atok_${tag}_fullgpu_rerun.log" 2>&1
  echo "  fullgpu B=1 rerun  exit=$?"

  # (3) FULLGPU B=8 — shipped c2c5d313, Marlin on, batch 8 (designed regime).
  rm -f "cp53_atok_${tag}_fullgpu_b8.json"
  env CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
      CIPHER_TENANT_ID=1 WL_TENANT_ID=1 \
      LD_PRELOAD="$SHIPPED" CUDA_INJECTION64_PATH="$SHIPPED" \
      timeout $WD python3 "$DECODE_B" "$mp" "$NTOK" \
      "cp53_atok_${tag}_fullgpu_b8.json" 8 \
      > "cp53_atok_${tag}_fullgpu_b8.log" 2>&1
  echo "  fullgpu B=8        exit=$?"
done

echo
echo "================ TRIANGULATION ================"
python3 - <<'PY'
import json, os
HERE=os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
def load(p):
    try: return json.load(open(p))["tokens"]
    except Exception as e: return None
def cmp(a,b):
    if a is None or b is None: return "MISSING"
    n=min(len(a),len(b))
    agree=sum(1 for i in range(n) if a[i]==b[i])
    fd=next((i for i in range(n) if a[i]!=b[i]),-1)
    return f"agree={agree}/{n} rate={agree/n:.4f} first_div={fd}"
for tag in ("tinyllama","mistral"):
    gold=load(f"cp53_atok_{tag}_gold.json")
    part=load(f"cp53_atok_{tag}_partition.json")
    full=load(f"cp53_atok_{tag}_fullgpu.json")
    fullr=load(f"cp53_atok_{tag}_fullgpu_rerun.json")
    fullb8=load(f"cp53_atok_{tag}_fullgpu_b8.json")
    print(f"--- {tag} ---")
    print(f"  partition     vs gold : {cmp(part,gold)}")
    print(f"  fullgpu(orig) vs gold : {cmp(full,gold)}")
    print(f"  fullgpu(rerun)vs gold : {cmp(fullr,gold)}")
    print(f"  fullgpu(B=8)  vs gold : {cmp(fullb8,gold)}")
    print(f"  partition     vs fullgpu(B=8): {cmp(part,fullb8)}")
PY
