#!/usr/bin/env bash
# W.4b.6 Step 3 — N=8 Mistral engagement gate, weight-shared, 3 conditions.
#   N, MODELP, GEN, PLEN, MODE(positive|negative), MTAG, CIPHER_FORMB_FAULT
# Conditions (all weight-shared so the substrate-attrib ratio isolates batching):
#   (cross)  full-handoff: vanilla executor (SHARE) owns weights, N injected
#            tenants hand off decode -> executor B=N decode. power-windowed.
#   (inproc) in-process B=N ceiling (one vanilla process). power-windowed.
#   (naive)  N concurrent B=1 solo, weight-shared (formb6_naive). power-windowed.
# Executor + in-proc are VANILLA (no shim tax); tenants inject for real fps.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${MODE:-positive}"; N="${N:-8}"; GEN="${GEN:-64}"; PLEN="${PLEN:-12}"
MODELP="${MODELP:-/home/ubuntu/models/Mistral-7B-v0.1}"
ALT="${ALT:-/home/ubuntu/models/Llama-3.2-1B-Instruct}"      # distinct-fp negative
MTAG="${MTAG:-}"
RT=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
OUT="${HERE}/g6_${MODE}${MTAG}_n${N}"; mkdir -p "$OUT"; rm -f "$OUT"/*
SOCK=/tmp/g6_${MODE}${MTAG}_n${N}.sock
SENT=/tmp/g6_${MODE}${MTAG}_n${N}.win
FP=/tmp/g6_executor_fp${MTAG}
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
sample() {  # $1=sentinel $2=out.csv $3=guard_pid
  ( while ! grep -q DECODE_END "$1" 2>/dev/null; do
      kill -0 "$3" 2>/dev/null || break
      echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -i 0 | tr -d ' ')"
      sleep 0.1
    done ) > "$2" &
  echo $!
}

echo "== one-time injected executor-fp warmup =="
env WL_MODEL="$MODELP" FP_OUT="$FP" CUDA_INJECTION64_PATH="$RT" \
    python3 "${HERE}/formb_warmup_fp.py" > "${OUT}/warmup.log" 2>&1
grep -h WARMUP "${OUT}/warmup.log" || { echo "!! warmup failed"; tail "${OUT}/warmup.log"; exit 1; }

# ---------- (cross) full-handoff -------------------------------------------
echo "== (cross) full-handoff: vanilla executor (SHARE) + $N injected tenants =="
# CIPHER_RT_DISABLE_AUTO_INIT=1: the executor is VANILLA by design (no shim tax,
# W.4b.1), but ctypes-loading libcipher_rt for the pure gate guard runs its
# constructor and re-arms the CUPTI kernel-launch shim -> taxes the executor's
# OWN decode (~2.6x, the cross-process slowdown). The guard (cipher_rt_pool_*)
# is pure CPU and needs no auto-init, so disable it -> truly-vanilla decode.
env WL_MODEL="$MODELP" FORMB_SOCK="$SOCK" N_TENANTS="$N" FP_FILE="$FP" GEN_LEN="$GEN" \
    SENTINEL="$SENT" RESULT_JSON="${OUT}/executor.json" CIPHER_FORMB_SHARE=1 \
    CIPHER_RT_DISABLE_AUTO_INIT=1 \
    CIPHER_FORMB_FAULT="${CIPHER_FORMB_FAULT:--1}" \
    python3 "${HERE}/formb6_executor.py" > "${OUT}/executor.log" 2>&1 &
EXE=$!
while ! grep -q "EXECUTOR listening" "${OUT}/executor.log" 2>/dev/null; do
  kill -0 $EXE 2>/dev/null || { echo "!! executor died:"; tail -30 "${OUT}/executor.log"; exit 1; }
  sleep 0.3
done
SMP=$(sample "$SENT" "${OUT}/cross_power.csv" "$EXE")
PIDS=()
for ((i=0;i<N;i++)); do
  if [ "$MODE" = "negative" ] && [ $i -eq $((N-1)) ]; then M="$ALT"; SH=0; else M="$MODELP"; SH=1; fi
  P="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
  env WL_MODEL="$M" TENANT_ROW=$i PLEN="$PLEN" GEN_LEN="$GEN" PROMPT="$P" \
      FORMB_SOCK="$SOCK" RESULT_JSON="${OUT}/tenant${i}.json" \
      CIPHER_FORMB_SHARE=$SH CUDA_INJECTION64_PATH="$RT" \
      python3 "${HERE}/formb6_tenant.py" > "${OUT}/tenant${i}.log" 2>&1 &
  PIDS+=($!)
done
wait $EXE; ERC=$?
wait "${PIDS[@]}" 2>/dev/null || true
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
echo "  cross rc=$ERC"
grep -hE 'EXECUTOR gate|EXECUTOR batched|EXECUTOR BLOCK|FIRSTCOALESCE' "${OUT}/executor.log" | sed 's/^/    /'
grep -hE 'TENANT row' "${OUT}"/tenant*.log 2>/dev/null | sort | sed 's/^/    /'

if [ "$MODE" = "negative" ]; then echo "negative-only run -> ${OUT}/"; exit 0; fi

# ---------- (inproc) in-process B=N ceiling --------------------------------
echo "== (inproc) in-process B=$N ceiling (vanilla) =="
S=/tmp/g6_inproc${MTAG}_n${N}.win; rm -f "$S"
env WL_MODEL="$MODELP" MODE=inproc NB="$N" PLEN="$PLEN" GEN_LEN="$GEN" SENTINEL="$S" \
    RESULT_JSON="${OUT}/inproc.json" python3 "${HERE}/formb_baselines.py" \
    > "${OUT}/inproc.log" 2>&1 &
P=$!; SMP=$(sample "$S" "${OUT}/inproc_power.csv" "$P"); wait $P
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
grep -h BASELINE "${OUT}/inproc.log" | sed 's/^/    /'

# ---------- (naive) N concurrent B=1 solo, weight-shared -------------------
echo "== (naive) $N concurrent B=1 solo, weight-shared =="
S=/tmp/g6_naive${MTAG}_n${N}.win; rm -f "$S"
NSOCK=/tmp/g6_naive${MTAG}_n${N}.sock; rm -f "$NSOCK"
env WL_MODEL="$MODELP" N_WORKERS="$N" GEN_LEN="$GEN" PLEN="$PLEN" SENTINEL="$S" \
    RESULT_JSON="${OUT}/naive.json" FORMB_SOCK="$NSOCK" \
    python3 "${HERE}/formb6_naive.py" > "${OUT}/naive.log" 2>&1 &
P=$!; SMP=$(sample "$S" "${OUT}/naive_power.csv" "$P"); wait $P
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
grep -h NAIVE "${OUT}/naive.log" | sed 's/^/    /'

echo "== aggregate tok/W =="
python3 "${HERE}/agg_tokw.py" \
  "cross=${OUT}/executor.json:${OUT}/cross_power.csv:${SENT}" \
  "inproc=${OUT}/inproc.json:${OUT}/inproc_power.csv:/tmp/g6_inproc${MTAG}_n${N}.win" \
  "naive=${OUT}/naive.json:${OUT}/naive_power.csv:/tmp/g6_naive${MTAG}_n${N}.win" \
  | tee "${OUT}/engagement_table.md"
echo "results -> ${OUT}/"
