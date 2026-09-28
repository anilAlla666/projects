#!/usr/bin/env bash
# Phase 4 post-reboot all-batches baseline runner under cipher_kmod 0.4.3.
#
# Order (user-prescribed):
#   Batch 1: WL01 WL02 WL06 WL11 WL14 WL15      (6)
#   Batch 2: WL03 WL04 WL07 WL08 WL12           (5)
#   Batch 3: WL09 WL10 WL13 WL16 WL17           (5)
#   Batch 4: WL19 WL20 WL21 WL22 WL23 WL24      (6)
#   Special: WL05 (multi-tenant ×8)             (1)
#   Marker:  WL18 (deferred-multi-gpu on 1-GPU) (1)
#
# After each WL: post-process the JSON to add tokens_per_sec, watts_avg, TPW,
# baseline_valid markers.
#
# Daemons (gpustate + exporter) are launched once before the loop and reused.

set -u
EVID=/home/ubuntu/cipher-phase4-evidence
LOG=$EVID/baseline_runner.log
SUMMARY=$EVID/baseline_runner_summary.txt
JSONL=$EVID/baselines_combined.jsonl
: > "$LOG"
: > "$SUMMARY"
: > "$JSONL"

note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" "$SUMMARY"; }

note "=== Phase 4 all-batches baseline runner under cipher_kmod $(cat /sys/module/cipher_kmod/version)"
note "srcversion: $(cat /sys/module/cipher_kmod/srcversion 2>/dev/null)"
note "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Daemons
if ! pgrep -f cipher-gpustate >/dev/null; then
  sudo /home/ubuntu/cipher_gpustate/cipher-gpustate \
    > $EVID/gpustate_baseline.log 2>&1 &
  GP_PID=$!
  echo $GP_PID > $EVID/gpustate.pid
  note "started cipher-gpustate pid=$GP_PID"
fi
if ! pgrep -f cipher-exporter >/dev/null; then
  python3 /home/ubuntu/cipher_exporter/cipher-exporter.py --port 9402 --bind 127.0.0.1 \
    > $EVID/exporter_baseline.log 2>&1 &
  EX_PID=$!
  echo $EX_PID > $EVID/exporter.pid
  note "started cipher-exporter pid=$EX_PID"
fi
sleep 2

WL_ROOT=/home/ubuntu/cipher_workloads

run_single_wl() {
  local WL="$1"
  local DUR="${2:-600}"
  note "--- $WL: starting (duration=${DUR}s)"
  cd $WL_ROOT
  bash ./measurement/run_baseline.sh "$WL" --duration "$DUR" >> "$LOG" 2>&1
  local rc=$?
  local json="$WL_ROOT/expected/${WL,,}_p41_baseline.json"
  if [[ -f "$json" ]]; then
    python3 ./measurement/phase4_baseline_postprocess.py "$WL" > /tmp/_pp.json 2>>"$LOG"
    cat /tmp/_pp.json >> "$JSONL"; echo >> "$JSONL"
    {
      echo "[$WL summary]"
      cat /tmp/_pp.json
    } | tee -a "$SUMMARY" >/dev/null
  else
    note "$WL: MISSING JSON (driver rc=$rc) — recording placeholder"
    python3 -c "import json,sys; json.dump({'wl_id':'$WL','baseline_valid':False,'baseline_invalid_reason':'json missing (driver rc=$rc)'}, sys.stdout)" >> "$JSONL"
    echo >> "$JSONL"
  fi
  note "--- $WL: done (rc=$rc)"
}

run_wl05() {
  note "--- WL05 (multi-tenant ×8): starting (duration=600s)"
  cd $WL_ROOT
  bash ./measurement/run_baseline_wl05.sh --duration 600 >> "$LOG" 2>&1
  local rc=$?
  local json="$WL_ROOT/expected/wl05_p41_baseline.json"
  if [[ -f "$json" ]]; then
    python3 ./measurement/phase4_baseline_postprocess.py "WL05" > /tmp/_pp.json 2>>"$LOG"
    cat /tmp/_pp.json >> "$JSONL"; echo >> "$JSONL"
    {
      echo "[WL05 summary]"
      cat /tmp/_pp.json
    } | tee -a "$SUMMARY" >/dev/null
  else
    note "WL05: MISSING JSON (rc=$rc)"
  fi
  note "--- WL05: done (rc=$rc)"
}

# ---- Batch 1
note "=== BATCH 1 ==="
for WL in WL01 WL02 WL06 WL11 WL14 WL15; do run_single_wl "$WL"; done

# ---- Batch 2
note "=== BATCH 2 ==="
for WL in WL03 WL04 WL07 WL08 WL12; do run_single_wl "$WL"; done

# ---- Batch 3
note "=== BATCH 3 ==="
for WL in WL09 WL10 WL13 WL16 WL17; do run_single_wl "$WL"; done

# ---- Batch 4
note "=== BATCH 4 ==="
for WL in WL19 WL20 WL21 WL22 WL23 WL24; do run_single_wl "$WL"; done

# ---- WL05 multi-tenant
note "=== WL05 ==="
run_wl05

# ---- WL18 deferred marker
note "=== WL18 (deferred-multi-gpu marker) ==="
run_single_wl "WL18" 60   # short duration; driver exits immediately

note "=== ALL BATCHES DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
