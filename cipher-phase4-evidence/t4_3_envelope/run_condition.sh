#!/usr/bin/env bash
# Runs one condition: N matched pairs of (VOLT=off, VOLT=on).
# Mirrors T4.3.2 volt_tpw_npairs.sh structure with one workload-config
# selection. Env knobs:
#   COND=C1..C6              required, picks workload+binary+batch
#   N=3                      matched pairs
#   DUR=60                   seconds per measurement
#   OUT=...                  output dir (default per-condition)
#
# Conditions:
#   C1: Mistral-7B B=1 / current libcipher_rt
#   C2: Mistral-7B B=8 / current libcipher_rt
#   C3: Mistral-7B B=32 / current libcipher_rt
#   C4: Mistral-7B B=1 after pod reset / current libcipher_rt
#   C5: Mistral-7B B=1 / libcipher_rt.so.pre_T4_6_1
#   C6: TinyLlama-1.1B B=1 / current libcipher_rt   (T4.3.2 anchor repro)

set -u
COND="${COND:?must set COND=C1..C6}"
N="${N:-3}"
DUR="${DUR:-60}"
OUT="${OUT:-/home/ubuntu/cipher-phase4-evidence/t4_3_envelope/$COND}"
mkdir -p "$OUT"
rm -f "$OUT"/* 2>/dev/null || true

# Default to current libcipher_rt; C5 overrides.
BIN="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"

case "$COND" in
  C1) MODEL="mistral7b"; WLB=1;  VBATCH=1  ;;
  C2) MODEL="mistral7b"; WLB=8;  VBATCH=8  ;;
  C3) MODEL="mistral7b"; WLB=32; VBATCH=32 ;;
  C4) MODEL="mistral7b"; WLB=1;  VBATCH=1
      echo "--- C4 pod reset before measurement ---"
      sudo nvidia-smi -i 0 -rgc  >/dev/null 2>&1
      sudo nvidia-smi -i 0 -rmc  >/dev/null 2>&1
      sudo nvidia-smi -pm 1      >/dev/null 2>&1
      sudo nvidia-smi -pl 700    >/dev/null 2>&1
      sleep 30  ;;
  C5) MODEL="mistral7b"; WLB=1;  VBATCH=1
      BIN="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so.pre_T4_6_1"  ;;
  C6) MODEL="tinyllama11b"; WLB=1; VBATCH=1 ;;
  *)  echo "unknown COND=$COND"; exit 2 ;;
esac

echo "=== $COND  model=$MODEL  WL_BATCH=$WLB  VOLT_BATCH=$VBATCH  BIN=$(basename $BIN) ==="

run_one() {
  local pair="$1" arm="$2"
  local label_dir="${OUT}/pair${pair}_${arm}"
  local tid="${COND}_p${pair}_${arm}"
  local csv="${label_dir}/watts.csv"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  local log="${label_dir}/run.log"
  mkdir -p "$label_dir"; rm -f "$csv" "$prog"

  # Power sampler (1Hz nvidia-smi, captures DUR+buffer seconds).
  (
    echo "ts_s,power_w,clock_sm_mhz"
    t0=$(date +%s); end=$((t0+DUR+15))
    while [[ $(date +%s) -lt $end ]]; do
      pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1)
      ck=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i 0 | head -1)
      ts=$(( $(date +%s) - t0 ))
      echo "$ts,${pw:-0},${ck:-0}"
      sleep 1
    done
  ) > "$csv" &
  SAMPLER_PID=$!

  CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" \
    WL_DURATION="$DUR" WL_BATCH="$WLB" WL_MODEL="$MODEL" \
    CIPHER_VOLT="$arm" CIPHER_VOLT_BATCH="$VBATCH" \
    LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
    python3 /home/ubuntu/cipher-phase4-evidence/t4_3_envelope/envelope_driver.py \
      > "$log" 2>&1 &
  WL_PID=$!

  wait "$WL_PID"
  sleep 2
  kill -INT "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null || true
  [[ -f "$prog" ]] && cp "$prog" "$label_dir/progress.txt"
  tail -1 "$prog" 2>/dev/null || echo "no progress"
}

for ((i=1; i<=N; i++)); do
  echo "--- pair $i  VOLT=off ---"; run_one "$i" off
  echo "--- pair $i  VOLT=on  ---"; run_one "$i" on
done

# Pod state at end.
echo "=== end pod state ==="
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit --format=csv,noheader -i 0

