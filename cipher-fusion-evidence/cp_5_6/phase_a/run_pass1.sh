#!/usr/bin/env bash
# Phase A PASS 1 — per-tenant MFU + TPW for one workload.
# Env: WL_ID WL_MODEL [WL_BATCH=1] [WL_MAX_NEW=128]
# Gold (no substrate) + 3 arms (vanilla/marlin/allon), TF gate + free-run.
set -u
PHASE_A=/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/pillar_driver.py
WL_ID="${WL_ID:?set WL_ID}"
WL_MODEL="${WL_MODEL:?set WL_MODEL}"
WL_BATCH="${WL_BATCH:-1}"
WL_MAX_NEW="${WL_MAX_NEW:-128}"
VOLT_MHZ="${VOLT_MHZ:-1000}"
SUBSTRATE_MD5="$(md5sum "$BIN" | cut -c1-8)"
HERE="${PHASE_A}/${WL_ID}"
mkdir -p "$HERE"
GOLD="${HERE}/gold.json"

echo "=== Phase A PASS 1 — ${WL_ID} ==="
echo "model=$WL_MODEL batch=$WL_BATCH max_new=$WL_MAX_NEW substrate=$SUBSTRATE_MD5"
sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1 && echo "GPU clocks reset"

echo "--- gold (clean FP16, no substrate) ---"
env MODE=gold ARM=gold WL_ID="$WL_ID" WL_MODEL="$WL_MODEL" WL_BATCH="$WL_BATCH" \
    WL_MAX_NEW="$WL_MAX_NEW" WL_TENANT_ID="pa_${WL_ID}_gold" \
    OUT_JSON=/dev/null GOLD_JSON="$GOLD" CIPHER_SPEC=0 \
    python3 "$DRIVER" > "${HERE}/gold.log" 2>&1
[ -s "$GOLD" ] || { echo "  !! gold failed — see ${HERE}/gold.log"; exit 1; }
grep -h "^GOLD" "${HERE}/gold.log" | sed 's/^/  /'

run_one() {
  local arm="$1" tid="pa_${WL_ID}_$1"
  local sentinel="/tmp/cipher_phasea_${tid}.decode_window"
  local json="${HERE}/${arm}.json" log="${HERE}/${arm}.log" csv="${HERE}/${arm}.watts.csv"
  rm -f "$sentinel" "$json"
  local common=( MODE=gate ARM="$arm" WL_ID="$WL_ID" WL_MODEL="$WL_MODEL" \
    WL_BATCH="$WL_BATCH" WL_MAX_NEW="$WL_MAX_NEW" WL_TENANT_ID="$tid" \
    OUT_JSON="$json" GOLD_JSON="$GOLD" CIPHER_SUBSTRATE_MD5="$SUBSTRATE_MD5" )
  case "$arm" in
    vanilla) env "${common[@]}" CIPHER_SPEC=0 python3 "$DRIVER" > "$log" 2>&1 & ;;
    marlin)  env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
               LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
    allon)   env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=on \
               CIPHER_SPEC_DRAFT=ngram CIPHER_VOLT=on CIPHER_VOLT_MHZ="$VOLT_MHZ" \
               LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
               python3 "$DRIVER" > "$log" 2>&1 & ;;
  esac
  local wl=$! waited=0
  while ! grep -q DECODE_START "$sentinel" 2>/dev/null; do
    kill -0 "$wl" 2>/dev/null || { wait "$wl" 2>/dev/null; echo "  !! $arm exited early — see $log"; return 1; }
    (( waited >= 9000 )) && { kill "$wl" 2>/dev/null; echo "  !! $arm timeout"; return 1; }
    sleep 0.2; waited=$((waited+1))
  done
  ( echo "epoch_s,power_w,clock_sm_mhz"
    while ! grep -q DECODE_END "$sentinel" 2>/dev/null; do
      smp=$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
      echo "$(date +%s.%N),${smp:-0,0}"; sleep 0.5
    done ) > "$csv" &
  local sampler=$!
  wait "$wl"; kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null || true
  grep -hE "^(RESULT|TFGATE)" "$log" 2>/dev/null | sed 's/^/  /'
}

for arm in vanilla marlin allon; do
  echo "--- arm: $arm ---"
  run_one "$arm" || echo "  ($arm did not complete cleanly)"
done
echo "=== ${WL_ID} PASS 1 done ==="
