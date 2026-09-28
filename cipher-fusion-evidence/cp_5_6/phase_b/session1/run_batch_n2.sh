#!/usr/bin/env bash
# Phase B Build Session 1 — run the N=2 batched-executor prototype on WL01.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SOCK=/tmp/cipher_batch_exec_n2.sock
SENT=/tmp/cipher_batch_exec_n2.decode_window
GOLD_LOGITS=/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL01/gold_logits.pt
rm -f "$SOCK" "$SENT"
[ -s "$GOLD_LOGITS" ] || { echo "!! missing $GOLD_LOGITS"; exit 1; }

env WL_MODEL=/home/ubuntu/models/TinyLlama-1.1B N_CLIENTS=2 WL_MAX_NEW=128 \
    BATCH_SOCK="$SOCK" SENTINEL="$SENT" GOLD_LOGITS="$GOLD_LOGITS" \
    RESULT_JSON="${HERE}/executor_result.json" \
    python3 "${HERE}/cipher_batch_executor.py" > "${HERE}/executor.log" 2>&1 &
EXE=$!
while ! grep -q LISTENING "${HERE}/executor.log" 2>/dev/null; do
  kill -0 $EXE 2>/dev/null || { echo "!! executor died before LISTENING:"; tail -20 "${HERE}/executor.log"; exit 1; }
  sleep 0.3
done
echo "executor listening (pid $EXE)"

( while ! grep -q DECODE_END "$SENT" 2>/dev/null; do
    kill -0 $EXE 2>/dev/null || break
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    sleep 0.3
  done ) > "${HERE}/power.csv" &
SMP=$!

for t in t1 t2; do
  env BATCH_SOCK="$SOCK" TENANT="$t" OUT_JSON="${HERE}/client_${t}.json" \
      python3 "${HERE}/batch_client.py" > "${HERE}/client_${t}.log" 2>&1 &
done
wait $EXE
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true

echo "=== executor rounds ==="
grep -hE '^ROUND' "${HERE}/executor.log" | sed 's/^/  /'
echo "=== clients ==="
grep -hE '^CLIENT' "${HERE}/client_t1.log" "${HERE}/client_t2.log" 2>/dev/null | sed 's/^/  /'
