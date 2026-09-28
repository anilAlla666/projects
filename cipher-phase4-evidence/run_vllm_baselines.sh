#!/usr/bin/env bash
# Task 1e: re-baseline WL04, WL10, WL16 under cipher_kmod 0.4.3 with
# VLLM_USE_DEEP_GEMM=0 (the fix from advisor's escalation 1).
set -u
EVID=/home/ubuntu/cipher-phase4-evidence
LOG=$EVID/vllm_rebaseline.log
SUMMARY=$EVID/vllm_rebaseline_summary.txt
JSONL=$EVID/baselines_combined.jsonl
: > "$LOG"
: > "$SUMMARY"

note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" "$SUMMARY"; }

note "=== vLLM re-baseline under cipher_kmod $(cat /sys/module/cipher_kmod/version)"
note "srcversion: $(cat /sys/module/cipher_kmod/srcversion)"
note "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

WL_ROOT=/home/ubuntu/cipher_workloads

for WL in WL04 WL10 WL16; do
  note "--- $WL: starting (duration=600s, VLLM_USE_DEEP_GEMM=0)"
  cd $WL_ROOT
  rm -f /tmp/cipher_tenant_${WL,,}_baseline_progress /tmp/cipher_baseline_$WL.log
  bash ./measurement/run_baseline.sh "$WL" --duration 600 >> "$LOG" 2>&1
  rc=$?
  python3 ./measurement/phase4_baseline_postprocess.py "$WL" > /tmp/_pp.json 2>>"$LOG"
  cat /tmp/_pp.json >> "$JSONL"; echo >> "$JSONL"
  note "--- $WL: done rc=$rc"
  cat /tmp/_pp.json | tee -a "$SUMMARY" >/dev/null
done

note "=== vLLM re-baseline done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
