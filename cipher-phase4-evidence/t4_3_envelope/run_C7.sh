#!/usr/bin/env bash
# C7: Mistral-7B B=1 with CIPHER_VOLT_MHZ=1600 (override the table's 1000).
set -u
N="${N:-3}"
DUR="${DUR:-60}"
OUT="/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/C7"
mkdir -p "$OUT"; rm -f "$OUT"/* 2>/dev/null || true
BIN="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"

run_one() {
  local pair=$1 arm=$2
  local d="${OUT}/pair${pair}_${arm}"
  local tid="C7_p${pair}_${arm}"
  local csv="${d}/watts.csv"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  local log="${d}/run.log"
  mkdir -p "$d"; rm -f "$csv" "$prog"
  (
    echo "ts_s,power_w,clock_sm_mhz"
    t0=$(date +%s); end=$((t0+DUR+15))
    while [[ $(date +%s) -lt $end ]]; do
      pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1)
      ck=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i 0 | head -1)
      ts=$(( $(date +%s) - t0 )); echo "$ts,${pw:-0},${ck:-0}"; sleep 1
    done
  ) > "$csv" &
  S=$!
  CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" \
    WL_DURATION="$DUR" WL_BATCH=1 WL_MODEL=mistral7b \
    CIPHER_VOLT="$arm" CIPHER_VOLT_MHZ=1600 \
    LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
    python3 /home/ubuntu/cipher-phase4-evidence/t4_3_envelope/envelope_driver.py > "$log" 2>&1 &
  wait $!
  sleep 2; kill -INT "$S" 2>/dev/null; wait "$S" 2>/dev/null || true
  [[ -f "$prog" ]] && cp "$prog" "$d/progress.txt"
  tail -1 "$prog" 2>/dev/null
}

for ((i=1; i<=N; i++)); do
  echo "--- pair $i  off ---"; run_one $i off
  echo "--- pair $i  on  ---"; run_one $i on
done
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i 0
