#!/bin/bash
# CP 5.3 STEP 2 gate A-tok — token-level correctness, §7 option (ii).
#
# Greedy 128-token decode under Marlin INT4, two builds of libcipher_rt.so:
#   FULLGPU   = libcipher_rt.so.pre_cp5_3_step2 (c2c5d313) — Marlin grid=132,
#               primary-pinned (the shipped pre-STEP-2 full-GPU path).
#   PARTITION = libcipher_rt.so (STEP 2 build)            — Marlin grid=8,
#               green-context-confined when the green ctx is active.
# Models: TinyLlama-1.1B (MHA), Mistral-7B-v0.1 (GQA).
#
# Verdict (option ii): A-num is the binding gate; A-tok reports top-1
# agreement rate + first-divergence index, pre-registered expectation
# >= 99% (>=127/128). A miss is an investigation trigger, not an auto-fail.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OLD=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_cp5_3_step2
NEW=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DECODE="$HERE/cp53_atok_decode.py"
NTOK=128
WD=1200

declare -A MODELS=(
  [tinyllama]=/home/ubuntu/models/TinyLlama-1.1B
  [mistral]=/home/ubuntu/models/Mistral-7B-v0.1
)

run_one() {  # <lib> <model_path> <out_json> <log>
  local lib="$1" mp="$2" oj="$3" lg="$4"
  rm -f "$oj"
  env CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
      CIPHER_TENANT_ID=1 WL_TENANT_ID=1 \
      LD_PRELOAD="$lib" CUDA_INJECTION64_PATH="$lib" \
      timeout $WD python3 "$DECODE" "$mp" "$NTOK" "$oj" > "$lg" 2>&1
  return $?
}

for tag in tinyllama mistral; do
  mp="${MODELS[$tag]}"
  echo "================ A-tok: $tag ($mp) ================"

  run_one "$OLD" "$mp" "cp53_atok_${tag}_fullgpu.json" "cp53_atok_${tag}_fullgpu.log"
  RF=$?
  run_one "$NEW" "$mp" "cp53_atok_${tag}_partition.json" "cp53_atok_${tag}_partition.log"
  RP=$?
  echo "  fullgpu exit=$RF  partition exit=$RP"

  # Marlin must actually have engaged for the comparison to be meaningful.
  # Anchor on the MATMUL exit-totals line: a bare `handled=` tail-1 grep
  # catches the trailing `[cipher-attn] exit totals ... handled=0` line and
  # falsely reports Marlin handled nothing.
  MF=$(grep 'MATMUL: exit totals' "cp53_atok_${tag}_fullgpu.log"   | grep -oE 'handled=[0-9]+' | tail -1)
  MP=$(grep 'MATMUL: exit totals' "cp53_atok_${tag}_partition.log" | grep -oE 'handled=[0-9]+' | tail -1)
  GP=$(grep -oE 'selecting group [0-9]+' "cp53_atok_${tag}_partition.log" | tail -1)
  echo "  marlin fullgpu:$MF  partition:$MP  (partition $GP)"

  if [ ! -s "cp53_atok_${tag}_fullgpu.json" ] || [ ! -s "cp53_atok_${tag}_partition.json" ]; then
    echo "RESULT atok_$tag=FAIL detail=missing_json fullgpu_exit=$RF partition_exit=$RP"
    continue
  fi

  python3 - "$tag" "cp53_atok_${tag}_fullgpu.json" "cp53_atok_${tag}_partition.json" <<'PY'
import sys, json
tag, fa, fb = sys.argv[1], sys.argv[2], sys.argv[3]
a = json.load(open(fa))["tokens"]
b = json.load(open(fb))["tokens"]
n = min(len(a), len(b))
agree = sum(1 for i in range(n) if a[i] == b[i])
first_div = next((i for i in range(n) if a[i] != b[i]), -1)
rate = agree / n if n else 0.0
verdict = "PASS" if rate >= 0.99 else "INVESTIGATE"
print(f"RESULT atok_{tag}={verdict} n={n} top1_agree={agree}/{n} "
      f"rate={rate:.4f} first_divergence_index={first_div} "
      f"(pre-registered >=0.99)")
PY
done
