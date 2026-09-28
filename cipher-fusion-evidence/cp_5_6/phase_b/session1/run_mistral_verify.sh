#!/usr/bin/env bash
# Mistral-7B N=4 cross-tenant batched verification (Session 2 Step 1 methodology,
# generate-based executor). Env: none — fixed config.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
N=4
SOCK=/tmp/cipher_batch_mistral.sock
SENT=/tmp/cipher_batch_mistral.decode_window
MDIR=/home/ubuntu/cipher-fusion-evidence/cp_5_6/phase_a/WL_MISTRAL
OUT="${HERE}/mistral_n4"; mkdir -p "$OUT"; rm -f "$OUT"/* "$SOCK" "$SENT"
env WL_MODEL=/home/ubuntu/models/Mistral-7B-v0.1 N_CLIENTS=$N WL_MAX_NEW=128 \
    BATCH_SOCK="$SOCK" SENTINEL="$SENT" GOLD_JSON="${MDIR}/gold.json" \
    GOLD_LOGITS="${MDIR}/gold_logits.pt" RESULT_JSON="${OUT}/executor_result.json" \
    python3 "${HERE}/cipher_batch_executor_gen.py" > "${OUT}/executor.log" 2>&1 &
EXE=$!
while ! grep -q LISTENING "${OUT}/executor.log" 2>/dev/null; do
  kill -0 $EXE 2>/dev/null || { echo "!! executor died:"; tail -15 "${OUT}/executor.log"; exit 1; }
  sleep 0.3
done
echo "Mistral executor listening (pid $EXE)"
( while ! grep -q DECODE_END "$SENT" 2>/dev/null; do
    kill -0 $EXE 2>/dev/null || break
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    sleep 0.3
  done ) > "${OUT}/power.csv" &
SMP=$!
for ((i=1;i<=N;i++)); do
  env BATCH_SOCK="$SOCK" TENANT="t${i}" GEN_LEN=128 OUT_JSON="${OUT}/client_t${i}.json" \
      python3 "${HERE}/batch_client.py" > "${OUT}/client_t${i}.log" 2>&1 &
done
wait $EXE
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
grep -hE '^(ROUND|TFGATE)' "${OUT}/executor.log" | sed 's/^/  /'
echo "=== Mistral N=4 batched done ==="
