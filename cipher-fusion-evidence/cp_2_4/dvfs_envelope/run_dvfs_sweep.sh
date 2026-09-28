#!/usr/bin/env bash
# CP 2.4 — DVFS envelope sweep on the post-Marlin-INT4 Mistral-7B decode
# workload. For each clock point: n=5 matched pairs of (VOLT off, VOLT on @
# CIPHER_VOLT_MHZ). BOTH arms run CIPHER_MARLIN=on — the workload under test
# is Mistral-7B INT4 decode; only the clock lock is toggled.
#
# Derived from cipher-phase4-evidence/t4_3_envelope/run_condition.sh, with:
#   - CIPHER_MARLIN=on added (post-Marlin-INT4 workload)
#   - CIPHER_VOLT_MHZ clock-point sweep (not CIPHER_VOLT_BATCH)
#   - n=5 (memo §6 / user instruction)
#
# Disable rule (memo §3.3, firm): if Δtok/W is neutral/negative across the
# envelope, DVFS is OUT of the composed gate. This script only measures;
# analyze_dvfs.py + the verdict do the honest accounting. No re-tuning.

set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="/home/ubuntu/cipher_rt_phase4/libcipher_rt.so"
DRIVER="$HERE/envelope_driver.py"

# 1200 first — the M1200 clean re-run is the first verified clock point
# (the pre-kmod-fix M1200 partial was discarded). 5 points span the envelope.
MHZ_GRID="${MHZ_GRID:-1200 1000 1400 1600 1800}"
N="${N:-5}"
DUR="${DUR:-60}"

echo "=== CP 2.4 DVFS envelope sweep ==="
echo "bin=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
echo "clock grid: $MHZ_GRID   n=$N pairs/clock   dur=${DUR}s"
echo "workload: Mistral-7B B=1 decode, CIPHER_MARLIN=on (post-Marlin-INT4)"
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 && echo "GPU clocks reset to nominal"

# NOTE: VOLT actuation uses the kmod CIPHER_SET_CLOCK_MHZ ioctl path (NVML
# clock-lock is NOT_SUPPORTED for a non-root process). /dev/cipher access was
# restored in kmod 0.4.8 — the devnode callback codifies mode 0666, reload-
# stable (see REGRESSION_AUDIT_2026_05_15.md). The sweep runs non-root, as
# T4.3.2 designed. Each on-arm process restores the clock at exit (VOLT
# atexit handler), so the following off-arm starts at nominal.
echo

# run_one <out_dir> <tid> <volt_mode> <volt_mhz>
# Sentinel-windowed (CP 2.4 DVFS harness fix, 2026-05-15): the workload is
# launched first; the power sampler starts only when envelope_driver.py signals
# DECODE_START and stops at DECODE_END — so watts.csv is EXACTLY the decode
# window, with model load + Marlin NVRTC compile + warmup excluded. Both arms
# are windowed by the identical sentinel, so the n=5 matched pairs are clean.
run_one() {
  local label_dir="$1" tid="$2" volt="$3" mhz="$4"
  local csv="${label_dir}/watts.csv"
  local prog="/tmp/cipher_tenant_${tid}_progress"
  local sentinel="/tmp/cipher_dvfs_${tid}.decode_window"
  local log="${label_dir}/run.log"
  mkdir -p "$label_dir"; rm -f "$csv" "$prog" "$sentinel"

  # launch the workload (it writes the sentinel at decode-loop boundaries)
  CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" \
    WL_DURATION="$DUR" WL_BATCH=1 WL_MODEL=mistral7b \
    CIPHER_MARLIN=on \
    CIPHER_VOLT="$volt" CIPHER_VOLT_MHZ="$mhz" \
    LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
    python3 "$DRIVER" > "$log" 2>&1 &
  local wl=$!

  # wait for the decode loop to begin; bail if the workload dies first or
  # 240 s elapse (model load + ~12 s Marlin compile + warmup is well under).
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

  # sampler: power+clock at ~1 Hz, from DECODE_START until DECODE_END
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

for MHZ in $MHZ_GRID; do
  COND="M${MHZ}"
  OUT="${HERE}/${COND}"
  mkdir -p "$OUT"; rm -rf "$OUT"/pair* 2>/dev/null || true
  echo "================ clock point ${MHZ} MHz (${COND}) ================"
  for ((i=1; i<=N; i++)); do
    echo "--- ${COND} pair $i  VOLT=off (nominal) ---"
    run_one "${OUT}/pair${i}_off" "${COND}_p${i}_off" off 0
    echo "--- ${COND} pair $i  VOLT=on @ ${MHZ}MHz ---"
    run_one "${OUT}/pair${i}_on"  "${COND}_p${i}_on"  on "$MHZ"
  done
  echo
done

echo "=== sweep complete; end pod state ==="
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit --format=csv,noheader -i 0
