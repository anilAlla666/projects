#!/usr/bin/env bash
# Phase A — N-tenant concurrent aggregate for one workload/arm.
# Env: WL_ID WL_MODEL [WL_BATCH=1] [WL_MAX_NEW=128] N ARM(vanilla|allon|marlin)
# Requires phase_a/<WL>/gold.json + gold_logits.pt. Spawns N concurrent
# pillar_driver gate procs (distinct tenant ids), samples FULL-GPU power for
# the whole concurrent run. analyze_multitenant.py does the aggregate.
set -u
PHASE_A=/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a
BIN=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
DRIVER=/home/ubuntu/cipher_rt_phase4/pillar_driver.py
WL_ID="${WL_ID:?}" ; WL_MODEL="${WL_MODEL:?}"
WL_BATCH="${WL_BATCH:-1}" ; WL_MAX_NEW="${WL_MAX_NEW:-128}"
N="${N:?set N}" ; ARM="${ARM:?set ARM}" ; VOLT_MHZ="${VOLT_MHZ:-1000}"
SUB=$(md5sum "$BIN" | cut -c1-8)
WLD="${PHASE_A}/${WL_ID}" ; GOLD="${WLD}/gold.json"
[ -s "$GOLD" ] || { echo "!! no gold at $GOLD — run gold pass first"; exit 1; }
OUT="${WLD}/mt${N}_${ARM}" ; mkdir -p "$OUT" ; rm -f "$OUT"/*
echo "=== ${WL_ID} multi-tenant N=${N} arm=${ARM} substrate=${SUB} ==="

pids=()
for ((i=1;i<=N;i++)); do
  tid="pa_${WL_ID}_mt${N}_${ARM}_t${i}"
  rm -f "/tmp/cipher_phasea_${tid}.decode_window"
  common=( MODE=gate ARM="$ARM" WL_ID="$WL_ID" WL_MODEL="$WL_MODEL" \
    WL_BATCH="$WL_BATCH" WL_MAX_NEW="$WL_MAX_NEW" WL_TENANT_ID="$tid" \
    OUT_JSON="${OUT}/tenant${i}.json" GOLD_JSON="$GOLD" CIPHER_SUBSTRATE_MD5="$SUB" )
  if [ "$ARM" = vanilla ]; then
    env "${common[@]}" CIPHER_SPEC=0 \
      python3 "$DRIVER" > "${OUT}/tenant${i}.log" 2>&1 &
  elif [ "$ARM" = marlin ]; then
    env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=0 CIPHER_VOLT=off \
      LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
      python3 "$DRIVER" > "${OUT}/tenant${i}.log" 2>&1 &
  else
    env "${common[@]}" CIPHER_MARLIN=on CIPHER_SPEC=on CIPHER_SPEC_DRAFT=ngram \
      CIPHER_VOLT=on CIPHER_VOLT_MHZ="$VOLT_MHZ" \
      LD_PRELOAD="$BIN" CUDA_INJECTION64_PATH="$BIN" \
      python3 "$DRIVER" > "${OUT}/tenant${i}.log" 2>&1 &
  fi
  pids+=($!)
  sleep 0.5
done
echo "launched ${N} tenants: ${pids[*]}"

anyalive() { for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
( echo "epoch_s,power_w,clock_sm_mhz"
  while anyalive; do
    smp=$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
    echo "$(date +%s.%N),${smp:-0,0}"
    sleep 0.5
  done ) > "${OUT}/power.csv" &
sampler=$!
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
kill "$sampler" 2>/dev/null || true ; wait "$sampler" 2>/dev/null || true

ok=0
for ((i=1;i<=N;i++)); do [ -s "${OUT}/tenant${i}.json" ] && ok=$((ok+1)); done
echo "tenants completed with json: ${ok}/${N}"
grep -h '^TFGATE' "${OUT}"/tenant*.log 2>/dev/null | sed 's/^/  /' | head -40
echo "=== ${WL_ID} N=${N} ${ARM} done ==="
