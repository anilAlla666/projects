#!/usr/bin/env bash
# CP 2.4 — composed gate (memo §6c). Mistral-7B B=1 varied-prompt decode,
# n=5 matched pairs, THREE arms per pair:
#   vanilla  — stock HF FP16, no CIPHER at all
#   marlin   — Marlin INT4 only (CIPHER_MARLIN=on, substrate active)
#   all-on   — Marlin + n-gram spec + DVFS clock-lock @ VOLT_MHZ
# Per (pair,arm): per-prompt tok/s JSON + sentinel-windowed watts.csv.
# analyze_composed.py reports composed tok/s lift + tok/W lift, and validates
# the lever decomposition (all-on/marlin tok/s should ~ the independently
# measured Mistral n-gram spec lift; DVFS is tok/s-neutral, shows up in tok/W).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/spec_varied_driver.py
TARGET=/home/ubuntu/models/Mistral-7B-v0.1
VOLT_MHZ="${VOLT_MHZ:-1000}"          # DVFS sweep best tok/W point (+62%)
N="${N:-5}"
OUT="${HERE}/composed"
mkdir -p "$OUT"; rm -f "$OUT"/*.json "$OUT"/*.log "$OUT"/*.csv

# run_one <pair> <arm>
run_one() {
  local pair="$1" arm="$2"
  local tid="cmp_p${pair}_${arm}"
  local sentinel="/tmp/cipher_specvaried_${tid}.decode_window"
  local json="${OUT}/p${pair}_${arm}.json"
  local log="${OUT}/p${pair}_${arm}.log"
  local csv="${OUT}/p${pair}_${arm}.watts.csv"
  rm -f "$sentinel"
  local env_common="CIPHER_TENANT_ID=$tid WL_TENANT_ID=$tid WL_MODEL=$TARGET OUT_JSON=$json"
  case "$arm" in
    vanilla)  env CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" WL_MODEL="$TARGET" \
                OUT_JSON="$json" CIPHER_SPEC=0 \
                python3 "$DRIVER" > "$log" 2>&1 & ;;
    marlin)   env CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" WL_MODEL="$TARGET" \
                OUT_JSON="$json" CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
                LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
                python3 "$DRIVER" > "$log" 2>&1 & ;;
    allon)    env CIPHER_TENANT_ID="$tid" WL_TENANT_ID="$tid" WL_MODEL="$TARGET" \
                OUT_JSON="$json" CIPHER_MARLIN=on CIPHER_SPEC=on \
                CIPHER_SPEC_DRAFT=ngram CIPHER_VOLT=on CIPHER_VOLT_MHZ="$VOLT_MHZ" \
                LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
                python3 "$DRIVER" > "$log" 2>&1 & ;;
  esac
  local wl=$!
  local waited=0
  while ! grep -q DECODE_START "$sentinel" 2>/dev/null; do
    if ! kill -0 "$wl" 2>/dev/null; then
      wait "$wl" 2>/dev/null; echo "  !! $tid exited before DECODE_START — see log"; return 1
    fi
    (( waited >= 2400 )) && { kill "$wl" 2>/dev/null; echo "  !! $tid timeout"; return 1; }
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
  grep -h "^RESULT" "$log" 2>/dev/null | sed 's/^/  /'
}

echo "=== CP 2.4 composed gate — Marlin + DVFS@${VOLT_MHZ}MHz + n-gram spec vs vanilla HF ==="
echo "bin md5=$(md5sum "$BIN" | cut -c1-8)  target=$TARGET  n=$N pairs x 3 arms"
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 && echo "GPU clocks reset to nominal"
echo
for ((i=1; i<=N; i++)); do
  echo "--- pair $i  vanilla (stock HF FP16) ---"; run_one "$i" vanilla
  echo "--- pair $i  marlin (Marlin INT4 only) ---"; run_one "$i" marlin
  echo "--- pair $i  all-on (Marlin + spec + DVFS) ---"; run_one "$i" allon
done
echo
echo "=== composed gate complete; end pod state ==="
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i 0
echo "analyze: python3 ${HERE}/analyze_composed.py"
