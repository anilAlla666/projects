#!/usr/bin/env bash
# W.4b plumbing-gap baselines with power sampling, N-parametric.
#   N=<int> : in-process B=N ceiling + naive N-concurrent (N x B=1 solo).
# Vanilla (no injection), same GEN/PLEN as the cross-process prototype.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
N="${N:-2}"; GEN="${GEN:-64}"; PLEN="${PLEN:-12}"
TINY="${MODELP:-/home/ubuntu/models/TinyLlama-1.1B}"
TAG="${MTAG:-}"
OUT="${HERE}/baselines${TAG}_n${N}"; mkdir -p "$OUT"; rm -f "$OUT"/*

sample() {  # $1=sentinel $2=out.csv $3=guard_pid
  ( while ! grep -q DECODE_END "$1" 2>/dev/null; do
      kill -0 "$3" 2>/dev/null || break
      echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -i 0 | tr -d ' ')"
      sleep 0.1
    done ) > "$2" &
  echo $!
}

echo "== in-process B=$N ceiling =="
S=/tmp/formb_inproc_n${N}.sent; rm -f "$S"
env WL_MODEL="$TINY" MODE=inproc NB="$N" PLEN="$PLEN" GEN_LEN="$GEN" SENTINEL="$S" \
    RESULT_JSON="${OUT}/inproc.json" python3 "${HERE}/formb_baselines.py" \
    > "${OUT}/inproc.log" 2>&1 &
P=$!; SMP=$(sample "$S" "${OUT}/inproc_power.csv" "$P"); wait $P
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
grep -h BASELINE "${OUT}/inproc.log" | sed 's/^/  /'

echo "== naive $N-concurrent ($N x B=1 solo) =="
S=/tmp/formb_naive_n${N}.sent; rm -f "$S"
PIDS=()
for ((i=0;i<N;i++)); do
  SS=/tmp/formb_naive_n${N}_${i}.sent; [ $i -eq 0 ] && SS="$S"; rm -f "$SS"
  env WL_MODEL="$TINY" MODE=solo PLEN="$PLEN" GEN_LEN="$GEN" PROMPT_OFFSET=$i \
      SENTINEL="$SS" RESULT_JSON="${OUT}/solo${i}.json" \
      python3 "${HERE}/formb_baselines.py" > "${OUT}/solo${i}.log" 2>&1 &
  PIDS+=($!)
done
SMP=$(sample "$S" "${OUT}/naive_power.csv" "${PIDS[0]}"); wait "${PIDS[@]}"
kill $SMP 2>/dev/null; wait $SMP 2>/dev/null || true
grep -h BASELINE "${OUT}"/solo*.log | sed 's/^/  /'
echo "baselines -> ${OUT}/"
