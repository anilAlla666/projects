#!/usr/bin/env bash
# CP 2.4 — DVFS harness NULL-CHECK. Same harness, same Mistral-7B B=1 decode,
# same sentinel windowing as run_dvfs_sweep.sh — but BOTH arms run VOLT=off
# (nominal SM clock, no lock either side). A sound harness must report
# Δtok/W ≈ 0 here; a non-zero result would expose residual measurement bias.
#
# run_one is byte-identical to run_dvfs_sweep.sh's run_one. Output lands in
# M9999/pair{i}_{off,on}/ so the unmodified analyze_dvfs.py reads it via
# MHZ_GRID=9999 — the "on" arm is just a second VOLT=off run, so the
# matched-pair Δtok/W is the harness's noise floor.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
DRIVER="$HERE/envelope_driver.py"
N="${N:-5}"
DUR="${DUR:-60}"

echo "=== CP 2.4 DVFS null-check (both arms VOLT=off, nominal clock) ==="
echo "bin=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
echo "n=$N matched pairs   dur=${DUR}s   expect Δtok/W ≈ 0"
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 && echo "GPU clocks reset to nominal"
echo

# run_one <out_dir> <tid> <volt_mode> <volt_mhz>  — identical to run_dvfs_sweep.sh
run_one() {
  local label_dir="$1" tid="$2" volt="$3" mhz="$4"
  local csv="${label_dir}/watts.csv"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  local sentinel="/tmp/cipher_dvfs_${tid}.decode_window"
  local log="${label_dir}/run.log"
  mkdir -p "$label_dir"; rm -f "$csv" "$prog" "$sentinel"

  CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" \
    WL_DURATION="$DUR" WL_BATCH=1 WL_MODEL=mistral7b \
    CIPHER_MARLIN=on \
    CIPHER_VOLT="$volt" CIPHER_VOLT_MHZ="$mhz" \
    LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
    python3 "$DRIVER" > "$log" 2>&1 &
  local wl=$!

  local waited=0
  while ! grep -q DECODE_START "$sentinel" 2>/dev/null; do
    if ! kill -0 "$wl" 2>/dev/null; then
      wait "$wl" 2>/dev/null
      echo "  !! workload exited before DECODE_START — see run.log"; return 1
    fi
    if (( waited >= 1200 )); then
      kill "$wl" 2>/dev/null; wait "$wl" 2>/dev/null
      echo "  !! timed out (240 s) waiting for DECODE_START"; return 1
    fi
    sleep 0.2; waited=$((waited+1))
  done

  ( echo "ts_s,power_w,clock_sm_mhz"
    t0=$(date +%s)
    while ! grep -q DECODE_END "$sentinel" 2>/dev/null; do
      smp=$(nvidia-smi --query-gpu=power.draw,clocks.sm \
            --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
      echo "$(( $(date +%s) - t0 )),${smp:-0,0}"
      sleep 1
    done ) > "$csv" &
  local sampler=$!

  wait "$wl"
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true
  [[ -f "$prog" ]]     && cp "$prog"     "$label_dir/progress.txt"
  [[ -f "$sentinel" ]] && cp "$sentinel" "$label_dir/decode_window.txt"
  tail -1 "$prog" 2>/dev/null || echo "  (no progress — see run.log)"
}

OUT="${HERE}/M9999"
mkdir -p "$OUT"; rm -rf "$OUT"/pair* 2>/dev/null || true
for ((i=1; i<=N; i++)); do
  echo "--- null pair $i  arm A  VOLT=off (nominal) ---"
  run_one "${OUT}/pair${i}_off" "null_p${i}_off" off 0
  echo "--- null pair $i  arm B  VOLT=off (nominal) ---"
  run_one "${OUT}/pair${i}_on"  "null_p${i}_on"  off 0
done
echo
echo "=== null-check runs complete; analyze with: MHZ_GRID=9999 python3 analyze_dvfs.py ==="
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i 0
