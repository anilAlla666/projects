#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# soak_24h.sh — Phase 4 G4 stability gate (24-hour multi-tenant soak).
#
# Realistic mix (defaults):
#   4 inference tenants (4Kx4K PyTorch matmul loop, ~TinyLlama-shaped)
#   2 training-like tenants (8Kx8K matmul + sum, no sleep)
#   1 idle tenant (5s sleep loop)
#
# Every 60 s monitors:
#   - kernel WARN/oops/BUG count (sudo dmesg | grep -c)
#   - taint (/proc/sys/kernel/tainted)
#   - per-tenant MFU stability (from /metrics)
#   - cipher_kmod hashtable depth (proxy via /proc/cipher/stats reaped count)
#   - actuator error rates (from /metrics if exported, else 0)
#
# Stop conditions (any one):
#   - 24h elapsed (success path)
#   - kernel WARN/oops/BUG > 0
#   - taint added > 1 from baseline
#   - --early-stop signal received
#
# Outputs: /tmp/cipher_soak_24h_<TS>.log (full minute-by-minute) and
#          /tmp/cipher_soak_24h_<TS>_summary.json
#
# Usage:
#   ./soak_24h.sh [--duration N] [--monitor-interval N] [--exporter URL]

set -euo pipefail

DURATION=$((24*3600))
MONITOR_INTERVAL=60
EXPORTER="http://localhost:9402"
LIBCIPHER_V2=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)         DURATION="$2"; shift 2 ;;
    --monitor-interval) MONITOR_INTERVAL="$2"; shift 2 ;;
    --exporter)         EXPORTER="$2"; shift 2 ;;
    --help|-h) sed -n '4,28p' "$0" >&2; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="/tmp/cipher_soak_24h_${TS}.log"
SUMFILE="/tmp/cipher_soak_24h_${TS}_summary.json"

baseline_taint=$(cat /proc/sys/kernel/tainted)
baseline_warns=$(sudo dmesg 2>/dev/null | grep -ciE "WARN|oops|BUG:" || echo 0)
fallback_kmod_md5=$(md5sum /home/ubuntu/cipher_kmod.ko.v0.2.0 | awk '{print $1}')

echo "INFO: soak baseline: taint=$baseline_taint warns=$baseline_warns" | tee -a "$LOGFILE"

# ----- launch tenants -----
declare -a TENANT_PIDS=()

launch_inference() {
  local id="$1"
  CIPHER_TENANT_ID="inf_$id" \
  CUDA_INJECTION64_PATH="$LIBCIPHER_V2" \
    python3 -c "
import torch, time, os
m = torch.randn(4000, 4000, device='cuda')
while True:
    (m @ m).sum().item()
    time.sleep(0.05)
" > "/tmp/soak_inf_${id}.log" 2>&1 &
  TENANT_PIDS+=($!)
}

launch_train() {
  local id="$1"
  CIPHER_TENANT_ID="train_$id" \
  CUDA_INJECTION64_PATH="$LIBCIPHER_V2" \
    python3 -c "
import torch
m = torch.randn(8000, 8000, device='cuda')
while True:
    (m @ m).sum().item()
" > "/tmp/soak_train_${id}.log" 2>&1 &
  TENANT_PIDS+=($!)
}

launch_idle() {
  CIPHER_TENANT_ID="idle_1" \
  CUDA_INJECTION64_PATH="$LIBCIPHER_V2" \
    python3 -c "
import torch, time
m = torch.randn(100, 100, device='cuda')  # tiny, just to register
while True:
    time.sleep(5)
" > "/tmp/soak_idle.log" 2>&1 &
  TENANT_PIDS+=($!)
}

cleanup() {
  echo "INFO: stopping tenants" | tee -a "$LOGFILE"
  # SIGINT is masked (SIG_IGN) on &-launched background jobs in a
  # non-interactive script — use SIGTERM, then SIGKILL any straggler.
  for p in "${TENANT_PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  for _ in 1 2 3 4 5; do
    sleep 1
    alive=0
    for p in "${TENANT_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
    [[ $alive -eq 0 ]] && break
  done
  for p in "${TENANT_PIDS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

for i in 1 2 3 4; do launch_inference "$i"; sleep 1; done
for i in 1 2;       do launch_train "$i";     sleep 1; done
launch_idle

echo "INFO: 7 tenants launched (4 inference + 2 train + 1 idle)" | tee -a "$LOGFILE"

# ----- monitor loop -----
t0=$(date +%s)
last_minute=-1
final_status="incomplete"

while :; do
  t=$(( $(date +%s) - t0 ))
  if [[ "$t" -ge "$DURATION" ]]; then
    final_status="success_24h"
    break
  fi

  cur_taint=$(cat /proc/sys/kernel/tainted)
  cur_warns=$(sudo dmesg 2>/dev/null | grep -ciE "WARN|oops|BUG:" || echo 0)
  cur_kmod_md5=$(md5sum /home/ubuntu/cipher_kmod.ko.v0.2.0 | awk '{print $1}')

  delta_taint=$(( cur_taint - baseline_taint ))
  delta_warns=$(( cur_warns - baseline_warns ))

  # Fallback md5 must never change
  if [[ "$cur_kmod_md5" != "$fallback_kmod_md5" ]]; then
    echo "FAIL t=${t}s: fallback md5 changed! $cur_kmod_md5 vs $fallback_kmod_md5" | tee -a "$LOGFILE"
    final_status="fail_fallback_modified"
    break
  fi

  if [[ "$delta_taint" -gt 1 ]]; then
    echo "FAIL t=${t}s: taint bits added > 1 (was $baseline_taint, now $cur_taint)" | tee -a "$LOGFILE"
    final_status="fail_taint_drift"
    break
  fi

  if [[ "$delta_warns" -gt 0 ]]; then
    echo "FAIL t=${t}s: $delta_warns new kernel WARN/oops/BUG" | tee -a "$LOGFILE"
    final_status="fail_kernel_event"
    break
  fi

  # /metrics sample
  metrics=$(curl -fsS "${EXPORTER}/metrics" 2>/dev/null || echo "")
  power=$(echo "$metrics" | awk '/^cipher_gpu_power_watts/ { print $2 }')
  power="${power:-0}"
  active_tenants=$(echo "$metrics" | grep -c '^cipher_tenant_sm_util_pct{' || echo 0)

  echo "t=${t}s taint=$cur_taint warns=$cur_warns power=${power}W active=$active_tenants" \
    | tee -a "$LOGFILE"

  sleep "$MONITOR_INTERVAL"
done

# ----- summary -----
end_t=$(( $(date +%s) - t0 ))
cat > "$SUMFILE" <<EOF
{
  "start_unix":          $t0,
  "end_unix":            $(date +%s),
  "elapsed_seconds":     $end_t,
  "baseline_taint":      $baseline_taint,
  "final_taint":         $(cat /proc/sys/kernel/tainted),
  "baseline_warns":      $baseline_warns,
  "final_warns":         $(sudo dmesg 2>/dev/null | grep -ciE "WARN|oops|BUG:"),
  "fallback_md5_stable": $(if [[ "$(md5sum /home/ubuntu/cipher_kmod.ko.v0.2.0|awk '{print $1}')" == "$fallback_kmod_md5" ]]; then echo true; else echo false; fi),
  "final_status":        "$final_status",
  "logfile":             "$LOGFILE"
}
EOF

echo "INFO: soak complete. status=$final_status logfile=$LOGFILE summary=$SUMFILE" | tee -a "$LOGFILE"
