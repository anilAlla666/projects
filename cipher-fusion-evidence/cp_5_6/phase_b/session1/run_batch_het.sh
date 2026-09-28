#!/usr/bin/env bash
# Phase B Step 4 — heterogeneous batch run. Env: GENLENS (comma-sep), TAG
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
GENLENS="${GENLENS:?set GENLENS}"
TAG="${TAG:?set TAG}"
IFS=',' read -ra GL <<< "$GENLENS"
N=${#GL[@]}
SOCK=/tmp/cipher_batch_exec_${TAG}.sock
SENT=/tmp/cipher_batch_exec_${TAG}.decode_window
GOLD_JSON=/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01/gold.json
GOLD_LOGITS=/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01/gold_logits.pt
OUT="${HERE}/${TAG}"; mkdir -p "$OUT"; rm -f "$OUT"/* "$SOCK" "$SENT"
env WL_MODEL=/home/ubuntu/models/TinyLlama-1.1B N_CLIENTS=$N \
    BATCH_SOCK="$SOCK" SENTINEL="$SENT" GOLD_JSON="$GOLD_JSON" \
    GOLD_LOGITS="$GOLD_LOGITS" RESULT_JSON="${OUT}/executor_result.json" \
    python3 "${HERE}/cipher_batch_executor.py" > "${OUT}/executor.log" 2>&1 &
EXE=$!
while ! grep -q LISTENING "${OUT}/executor.log" 2>/dev/null; do
  kill -0 $EXE 2>/dev/null || { echo "!! executor died:"; tail -15 "${OUT}/executor.log"; exit 1; }
  sleep 0.3
done
echo "executor listening (TAG=$TAG N=$N genlens=$GENLENS)"
( while ! grep -q DECODE_END "$SENT" 2>/dev/null; do
    kill -0 $EXE 2>/dev/null || break
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    sleep 0.3
  done ) > "${OUT}/power.csv" &
SMP=$!
for ((i=1;i<=N;i++)); do
  env BATCH_SOCK="$SOCK" TENANT="t${i}" GEN_LEN="${GL[i-1]}" \
      OUT_JSON="${OUT}/client_t${i}.json" \
      python3 "${HERE}/batch_client.py" > "${OUT}/client_t${i}.log" 2>&1 &
done
wait $EXE
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
grep -hE '^ROUND' "${OUT}/executor.log" | sed 's/^/  /'
echo "=== $TAG done ==="
