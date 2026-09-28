#!/bin/bash
# F1 audit — PRIORITY 1: does dc804eb3 produce coherent full-GPU decode,
# or only coherent partition decode?
#
# The green ctx is created unconditionally by the CUPTI launch callback
# (cipher_cupti.c:123) — no env/tenant gate. dc804eb3 wires
# grid = green_sm>0 ? green_sm : g_sm_count. So:
#   - with CUPTI active  -> green ctx exists -> grid=8  (partition path)
#   - CUPTI suppressed   -> no green ctx    -> grid=132 (full-GPU path)
# We exercise both by toggling CUDA_INJECTION64_PATH (CUPTI subscription):
#   cfg INJECT  = LD_PRELOAD + CUDA_INJECTION  -> partition path expected
#   cfg PRELOAD = LD_PRELOAD only              -> full-GPU path expected
# No CIPHER_TENANT_ID in either (per the Priority 1 spec).
# The logs are authoritative for which grid path ran; coherence vs gold
# is the verdict.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
NEW=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so          # dc804eb3
DEC1="$HERE/cp53_atok_decode.py"
DEC8="$HERE/cp53_atok_decode_batched.py"
NTOK=128
WD=1800

declare -A MODELS=(
  [tinyllama]=/home/ubuntu/models/TinyLlama-1.1B
  [mistral]=/home/ubuntu/models/Mistral-7B-v0.1
)

# gold (no substrate) — B=1 reuse from cp_5_3 if present, else generate; B=8 fresh
for tag in tinyllama mistral; do
  mp="${MODELS[$tag]}"
  g1="$HERE/p1_${tag}_gold_b1.json"
  if [ -s "/home/ubuntu/cipher-fusion-evidence/cp_5_3/cp53_atok_${tag}_gold.json" ]; then
    cp "/home/ubuntu/cipher-fusion-evidence/cp_5_3/cp53_atok_${tag}_gold.json" "$g1"
    echo "gold b1 $tag: reused from cp_5_3"
  else
    timeout $WD python3 "$DEC1" "$mp" "$NTOK" "$g1" > "$HERE/p1_${tag}_gold_b1.log" 2>&1
    echo "gold b1 $tag: generated exit=$?"
  fi
  timeout $WD python3 "$DEC8" "$mp" "$NTOK" "$HERE/p1_${tag}_gold_b8.json" 8 \
      > "$HERE/p1_${tag}_gold_b8.log" 2>&1
  echo "gold b8 $tag: exit=$?"
done

# dc804eb3 runs — two configs x two batch sizes
run_dc() {  # <tag> <model> <cfg> <B>
  local tag="$1" mp="$2" cfg="$3" B="$4"
  local oj="$HERE/p1_${tag}_${cfg}_b${B}.json"
  local lg="$HERE/p1_${tag}_${cfg}_b${B}.log"
  local dec; [ "$B" = "1" ] && dec="$DEC1" || dec="$DEC8"
  local args=("$mp" "$NTOK" "$oj"); [ "$B" = "8" ] && args+=("8")
  rm -f "$oj"
  case "$cfg" in
    inject)  env CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
                 LD_PRELOAD="$NEW" CUDA_INJECTION64_PATH="$NEW" \
                 timeout $WD python3 "$dec" "${args[@]}" > "$lg" 2>&1 ;;
    preload) env CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
                 LD_PRELOAD="$NEW" \
                 timeout $WD python3 "$dec" "${args[@]}" > "$lg" 2>&1 ;;
  esac
  local rc=$?
  local grid green marlin
  green=$(grep -c 'GREEN: green context bound' "$lg" 2>/dev/null)
  marlin=$(grep 'MATMUL: exit totals' "$lg" | grep -oE 'handled=[0-9]+' | tail -1)
  echo "  $tag $cfg B=$B: exit=$rc  green_ctx_bound=$green  marlin=$marlin"
}

echo
echo "=== dc804eb3 decode runs (no tenant id) ==="
for tag in tinyllama mistral; do
  for cfg in inject preload; do
    for B in 1 8; do
      run_dc "$tag" "${MODELS[$tag]}" "$cfg" "$B"
    done
  done
done

echo
echo "=== TRIANGULATION vs gold ==="
python3 - <<'PY'
import json, os
H=os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
def load(p):
    try: return json.load(open(p))["tokens"]
    except Exception: return None
def cmp(a,b):
    if a is None or b is None: return "MISSING"
    n=min(len(a),len(b)); agree=sum(1 for i in range(n) if a[i]==b[i])
    fd=next((i for i in range(n) if a[i]!=b[i]),-1)
    return f"agree={agree}/{n} rate={agree/n:.3f} first_div={fd}"
for tag in ("tinyllama","mistral"):
    print(f"--- {tag} ---")
    for B in (1,8):
        gold=load(f"p1_{tag}_gold_b{B}.json")
        for cfg in ("inject","preload"):
            run=load(f"p1_{tag}_{cfg}_b{B}.json")
            print(f"  {cfg:8s} B={B}: {cmp(run,gold)}")
PY
