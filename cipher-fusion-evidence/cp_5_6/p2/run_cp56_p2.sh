#!/usr/bin/env bash
# CP 5.6 P2 — teacher-forced correctness gate + TPW re-measurement.
# Pass 1: gold  — clean FP16, NO substrate, MODE=gold -> gold.json
# Pass 2: gate  — arms vanilla/marlin/allon, MODE=gate, each free-run power-
#                 windowed (DECODE_START/END sentinel + nvidia-smi sampler),
#                 then teacher-forced vs gold. Arms run STRICTLY sequentially.
# Arm env toggles are identical to cp_2_4/run_composed.sh.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/tf_gate_driver.py
TARGET=/home/ubuntu/models/Mistral-7B-v0.1
GOLD="${HERE}/gold.json"
VOLT_MHZ="${VOLT_MHZ:-1000}"
SUBSTRATE_MD5="$(md5sum "$BIN" | cut -c1-8)"
mkdir -p "$HERE"

echo "=== CP 5.6 P2 — TF gate + TPW re-measurement ==="
echo "substrate libcipher_rt.so md5=${SUBSTRATE_MD5}  (expect a7ac8e97)"
echo "target=$TARGET  VOLT_MHZ=$VOLT_MHZ"
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 && echo "GPU clocks reset to nominal"
echo

# ---- Pass 1: gold (clean FP16, no substrate) ----
echo "--- pass 1: gold capture (clean FP16, no CIPHER) ---"
env MODE=gold ARM=gold WL_TENANT_ID=cp56p2_gold WL_MODEL="$TARGET" \
    OUT_JSON=/dev/null GOLD_JSON="$GOLD" CIPHER_SPEC=0 \
    python3 "$DRIVER" > "${HERE}/gold.log" 2>&1
if [ ! -s "$GOLD" ]; then
  echo "  !! gold capture failed — see ${HERE}/gold.log"; exit 1
fi
grep -h "^GOLD" "${HERE}/gold.log" | sed 's/^/  /'
echo

# ---- run_one <arm> ----
run_one() {
  local arm="$1"
  local tid="cp56p2_${arm}"
  local sentinel="/tmp/cipher_cp56p2_${tid}.decode_window"
  local json="${HERE}/${arm}.json"
  local log="${HERE}/${arm}.log"
  local csv="${HERE}/${arm}.watts.csv"
  rm -f "$sentinel" "$json"
  local common=( MODE=gate ARM="$arm" WL_TENANT_ID="$tid" WL_MODEL="$TARGET" \
                 OUT_JSON="$json" GOLD_JSON="$GOLD" \
                 CIPHER_SUBSTRATE_MD5="$SUBSTRATE_MD5" )
  case "$arm" in
    vanilla) env "${common[@]}" CIPHER_SPEC=0 \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
    marlin)  env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
               LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
    allon)   env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=on \
               CIPHER_SPEC_DRAFT=ngram CIPHER_VOLT=on CIPHER_VOLT_MHZ="$VOLT_MHZ" \
               LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
  esac
  local wl=$!
  local waited=0
  while ! grep -q DECODE_START "$sentinel" 2>/dev/null; do
    if ! kill -0 "$wl" 2>/dev/null; then
      wait "$wl" 2>/dev/null
      echo "  !! $arm exited before DECODE_START — see $log"; return 1
    fi
    (( waited >= 3000 )) && { kill "$wl" 2>/dev/null; echo "  !! $arm timeout"; return 1; }
    sleep 0.2; waited=$((waited+1))
  done
  # continuous power sampler — ABSOLUTE epoch ts; analyze takes the arm-level mean
  ( echo "epoch_s,power_w,clock_sm_mhz"
    while ! grep -q DECODE_END "$sentinel" 2>/dev/null; do
      smp=$(nvidia-smi --query-gpu=power.draw,clocks.sm \
            --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
      echo "$(date +%s.%N),${smp:-0,0}"
      sleep 0.5
    done ) > "$csv" &
  local sampler=$!
  wait "$wl"
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true
  grep -hE "^(RESULT|TFGATE)" "$log" 2>/dev/null | sed 's/^/  /'
}

for arm in vanilla marlin allon; do
  echo "--- arm: $arm ---"
  run_one "$arm" || echo "  (arm $arm did not complete cleanly)"
  echo
done

echo "=== end pod state ==="
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader -i 0
echo "analyze: python3 ${HERE}/analyze_cp56_p2.py"
