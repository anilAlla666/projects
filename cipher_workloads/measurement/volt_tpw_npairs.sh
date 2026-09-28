#!/usr/bin/env bash
# T4.3.2 credibility fix — N matched pairs of (VOLT=off, VOLT=on)
# at 60s each, libcipher_rt-driven actuation.
#
# Usage: bash volt_tpw_npairs.sh [N] [DUR]   (defaults N=5 DUR=60)

set -u
N="${1:-5}"
DUR="${2:-60}"
OUT=/home/ubuntu/cipher-phase4-evidence/t4_3_2_npairs
mkdir -p "$OUT"
rm -f "$OUT"/* 2>/dev/null || true

run_one() {
  local pair="$1"
  local arm="$2"          # off | on
  local label_dir="${OUT}/pair${pair}_${arm}"
  local tid="volt_p${pair}_${arm}"
  local csv="${label_dir}/watts.csv"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  local log="${label_dir}/run.log"

  mkdir -p "$label_dir"
  rm -f "$csv" "$prog"

  (
    echo "ts_s,power_w,clock_sm_mhz"
    t0=$(date +%s); end=$((t0+DUR+10))
    while [[ $(date +%s) -lt $end ]]; do
      pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
      ck=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
      ts=$(( $(date +%s) - t0 ))
      echo "$ts,${pw:-0},${ck:-0}"
      sleep 1
    done
  ) > "$csv" &
  SAMPLER_PID=$!

  CIPHER_TENANT_ID="$tid" WL_DURATION="$DUR" \
    CIPHER_VOLT="$arm" CIPHER_VOLT_BATCH=1 \
    CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
    python3 /home/ubuntu/cipher_workloads/drivers/wl01_decode_b1.py \
    > "$log" 2>&1 &
  WL_PID=$!

  wait "$WL_PID"
  sleep 2
  kill -INT "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null || true

  cp "$prog" "$label_dir/progress.txt"
  tail -1 "$prog"
}

echo "=== ensure baseline state ==="
sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1
sudo nvidia-smi -i 0 -pl 700 >/dev/null 2>&1

echo "=== running $N pairs (off + on per pair), $DUR s each ==="
for ((i=1; i<=N; i++)); do
  echo
  echo "--- pair $i: VOLT=off ---"
  run_one "$i" off
  echo "--- pair $i: VOLT=on ---"
  run_one "$i" on
done

echo
echo "=== final state ==="
sudo nvidia-smi -i 0 -rgc >/dev/null 2>&1
nvidia-smi --query-gpu=clocks.sm,power.limit --format=csv,noheader
