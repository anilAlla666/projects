#!/usr/bin/env bash
# 3-arm Arm 2 — N vLLM tenant instances, B requests-in-flight each. Env: N B TAG REP
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
N="${N:-8}"; B="${B:-1}"; TAG="${TAG:-floor}"; REP="${REP:?set REP}"
OUT="${HERE}/arm2_${TAG}_rep${REP}"; mkdir -p "$OUT"; rm -f "$OUT"/*
rm -f /tmp/arm2_ready_* /tmp/arm2_go
for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d " "); do kill -9 "$q" 2>/dev/null; done; sleep 3
pids=()
for ((i=1;i<=N;i++)); do
  env TENANT="t${i}" OUT_JSON="${OUT}/t${i}.json" GMU="${GMU:-0.10}" EAGER="${EAGER:-1}" B="$B" VLLM_MODEL="${VLLM_MODEL:-/home/ubuntu/models/TinyLlama-1.1B}" \
    /home/ubuntu/vllm_env/bin/python "${HERE}/vllm_tenant.py" > "${OUT}/t${i}.log" 2>&1 &
  pids+=($!); sleep 3
done
echo "launched $N vLLM tenants (TAG=$TAG B=$B rep $REP)"
waited=0
while [ "$(ls /tmp/arm2_ready_* 2>/dev/null | wc -l)" -lt $N ]; do
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  rdy=$(ls /tmp/arm2_ready_* 2>/dev/null | wc -l)
  if [ "$alive" -lt $N ] && [ "$rdy" -lt $N ]; then
    echo "!! tenant died before ready (alive=$alive ready=$rdy) — see ${OUT}/t*.log"
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; exit 1
  fi
  (( waited >= 1500 )) && { echo "!! timeout"; exit 1; }
  sleep 1; waited=$((waited+1))
done
echo "all $N ready — firing barrier"
( for ((s=0;s<1200;s++)); do
    a=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && a=1; done
    [ "$a" = 0 ] && break
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0|tr -d ' ')"
    sleep 0.5
  done ) > "${OUT}/power.csv" &
SMP=$!
touch /tmp/arm2_go
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
kill "$SMP" 2>/dev/null; wait "$SMP" 2>/dev/null || true
for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d " "); do kill -9 "$q" 2>/dev/null; done; sleep 3
ok=$(ls "$OUT"/t*.json 2>/dev/null | wc -l)
grep -h '^ARM2-TENANT' "$OUT"/t*.log 2>/dev/null | sed 's/^/  /' | head -3
echo "=== arm2 $TAG rep $REP: $ok/$N completed ==="
