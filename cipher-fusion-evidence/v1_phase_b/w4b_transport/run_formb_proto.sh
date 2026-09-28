#!/usr/bin/env bash
# W.4b.2-5 Form B' prototype harness (N-parametric).
#   N=<int>      number of tenants (default 2)
#   MODE=positive: N same-fp TinyLlama tenants + 1 vanilla TinyLlama executor
#                  -> gate eligible, batched B=N, KL gate.
#   MODE=negative: (N-1) TinyLlama + 1 Llama-3.2-1B (distinct fp) -> the distinct
#                  tenant is REJECTED (Memory #11 guard); the (N-1) same-fp coalesce
#                  if >=2, else all solo.
# Executor is VANILLA (no injection); only tenants inject libcipher_rt.so.
# CIPHER_FORMB_FAULT=<row> forces a first-coalesce correctness fault (3.3).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${MODE:-positive}"; N="${N:-2}"; GEN="${GEN:-64}"; PLEN="${PLEN:-12}"
TINY="${MODELP:-/home/ubuntu/models/TinyLlama-1.1B}"   # primary (same-fp) model
ALT=/home/ubuntu/models/Llama-3.2-1B-Instruct          # negative-control distinct model
TAG="${MTAG:-}"
RT=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
OUT="${HERE}/proto_${MODE}${TAG}_n${N}"; mkdir -p "$OUT"; rm -f "$OUT"/*
SOCK=/tmp/formb_${MODE}${TAG}_n${N}.sock
SENT=/tmp/formb_${MODE}${TAG}_n${N}.decode_window
FP=/tmp/formb_executor_fp
rm -f "$SOCK" "$SENT"

PROMPTS=(
 "The history of computing spans several distinct technological eras that"
 "Renewable energy adoption worldwide has accelerated rapidly over the recent"
 "Modern distributed systems must carefully balance consistency availability and partition"
 "Advances in materials science have enabled lighter stronger and more durable"
 "The global supply chain experienced unprecedented disruption during the pandemic which"
 "Machine learning models trained on large corpora can exhibit surprising emergent"
 "Urban planners increasingly rely on data driven simulations to forecast traffic"
 "Ocean currents play a critical role in regulating the planet climate by"
)

echo "== one-time injected executor-fp warmup (deployment-match, memo 2#2a) =="
env WL_MODEL="$TINY" FP_OUT="$FP" CUDA_INJECTION64_PATH="$RT" \
    python3 "${HERE}/formb_warmup_fp.py" > "${OUT}/warmup.log" 2>&1
grep -h WARMUP "${OUT}/warmup.log" || { echo "!! warmup failed"; tail "${OUT}/warmup.log"; exit 1; }

echo "== launch VANILLA executor (no injection) N_TENANTS=$N =="
# W.4b.7: CIPHER_RT_DISABLE_AUTO_INIT applies to the EXECUTOR ONLY (the vanilla
# process that ctypes-loads the audited guard via formb_cohort) — it suppresses the
# W.4b.6 CUPTI auto-init artifact. Tenants are INJECTED and must NOT set it (they
# need auto-init to register real fps). CIPHER_FORMB_PROFILE toggles item-1 timing.
env WL_MODEL="$TINY" FORMB_SOCK="$SOCK" N_TENANTS="$N" FP_FILE="$FP" GEN_LEN="$GEN" \
    SENTINEL="$SENT" RESULT_JSON="${OUT}/executor.json" \
    CIPHER_RT_DISABLE_AUTO_INIT="${CIPHER_RT_DISABLE_AUTO_INIT:-0}" \
    CIPHER_FORMB_PROFILE="${CIPHER_FORMB_PROFILE:-0}" \
    python3 "${HERE}/formb_executor.py" > "${OUT}/executor.log" 2>&1 &
EXE=$!
while ! grep -q "EXECUTOR listening" "${OUT}/executor.log" 2>/dev/null; do
  kill -0 $EXE 2>/dev/null || { echo "!! executor died:"; tail -30 "${OUT}/executor.log"; exit 1; }
  sleep 0.3
done
echo "executor listening (pid $EXE)"

( while ! grep -q DECODE_END "$SENT" 2>/dev/null; do
    kill -0 $EXE 2>/dev/null || break
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    sleep 0.1
  done ) > "${OUT}/power.csv" &
SMP=$!

echo "== launch $N injected tenants =="
PIDS=()
for ((i=0;i<N;i++)); do
  if [ "$MODE" = "negative" ] && [ $i -eq $((N-1)) ]; then M="$ALT"; else M="$TINY"; fi
  P="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
  env WL_MODEL="$M" TENANT_ROW=$i PLEN="$PLEN" GEN_LEN="$GEN" PROMPT="$P" \
      FORMB_SOCK="$SOCK" RESULT_JSON="${OUT}/tenant${i}.json" CUDA_INJECTION64_PATH="$RT" \
      CIPHER_FORMB_PROFILE="${CIPHER_FORMB_PROFILE:-0}" \
      python3 "${HERE}/formb_tenant.py" > "${OUT}/tenant${i}.log" 2>&1 &
  PIDS+=($!)
done

wait $EXE; ERC=$?
wait "${PIDS[@]}" 2>/dev/null || true
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
echo "=== MODE=$MODE N=$N executor rc=$ERC ==="
grep -hE 'EXECUTOR gate|EXECUTOR batched|EXECUTOR BLOCK|FIRSTCOALESCE' "${OUT}/executor.log" | sed 's/^/  /'
grep -hE 'TENANT row' "${OUT}"/tenant*.log 2>/dev/null | sort | sed 's/^/  /'
echo "results: ${OUT}/  power: ${OUT}/power.csv"
