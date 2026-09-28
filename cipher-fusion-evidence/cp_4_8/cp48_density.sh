#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# density_pack.sh — Phase 4 tenant-density push for G3 verification.
#
# Incrementally adds tenants every STEP_INTERVAL seconds, at counts
# 2 → 5 → 10 → 20 → 30 → 50 → 100. At each step samples per-tenant
# MFU from /metrics. Stops when ANY tenant's MFU drops below
# MIN_TENANT_MFU_PCT.
#
# Tenant workload: simple python3 matmul loop tagged with
# CIPHER_TENANT_ID=<tenant_NN>. Each tenant launched in background;
# script tracks PIDs to clean up on exit.
#
# Usage:
#   ./density_pack.sh [--step-interval N] [--workload-size N]
#                     [--min-mfu N] [--exporter URL]
#
# Output: CSV: timestamp_s,tenant_count,tenant,pid,sm_util_pct,mfu_pct,verdict

set -euo pipefail

STEP_INTERVAL=30
WORKLOAD_SIZE=4000
MIN_TENANT_MFU=1
EXPORTER="http://localhost:9402"
LIBCIPHER_V2=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
PEAK_CLOCK_MHZ=1980

STEPS=(2 5 10 20 30 50 100)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step-interval) STEP_INTERVAL="$2"; shift 2 ;;
    --workload-size) WORKLOAD_SIZE="$2"; shift 2 ;;
    --min-mfu)       MIN_TENANT_MFU="$2"; shift 2 ;;
    --exporter)      EXPORTER="$2"; shift 2 ;;
    --help|-h) sed -n '4,20p' "$0" >&2; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

declare -a TENANT_PIDS=()

cleanup() {
  echo "INFO: cleaning up ${#TENANT_PIDS[@]} tenant processes" >&2
  # NOTE: tenants are &-launched background jobs; bash masks SIGINT/SIGQUIT
  # (SIG_IGN) on such children in a non-interactive script, and CPython keeps
  # the inherited SIG_IGN. SIGINT is therefore a no-op here — use SIGTERM
  # (not masked), then SIGKILL any straggler still in a CUDA sync.
  for pid in "${TENANT_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for _ in 1 2 3 4 5; do
    sleep 1
    alive=0
    for pid in "${TENANT_PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$((alive+1)); done
    [[ $alive -eq 0 ]] && break
  done
  for pid in "${TENANT_PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

launch_tenant() {
  local id="$1"
  CIPHER_TENANT_ID="tenant_$id" \
  CUDA_INJECTION64_PATH="$LIBCIPHER_V2" \
    python3 -c "
import torch, time, os
m = torch.randn($WORKLOAD_SIZE, $WORKLOAD_SIZE, device='cuda')
while True:
    (m @ m).sum().item()
" > "/tmp/density_tenant_${id}.log" 2>&1 &
  TENANT_PIDS+=($!)
}

echo "timestamp_s,tenant_count,tenant,pid,sm_util_pct,mfu_pct,verdict"

t0=$(date +%s)
prev_count=0

for step in "${STEPS[@]}"; do
  # Add tenants to reach `step` count
  while [[ "${#TENANT_PIDS[@]}" -lt "$step" ]]; do
    id=$(( ${#TENANT_PIDS[@]} + 1 ))
    launch_tenant "$id"
    sleep 0.3   # stagger launches
  done

  # Wait STEP_INTERVAL for steady state
  echo "INFO: $step tenants running. Settling for ${STEP_INTERVAL}s..." >&2
  sleep "$STEP_INTERVAL"

  # Sample /metrics
  t=$(( $(date +%s) - t0 ))
  metrics=$(curl -fsS "${EXPORTER}/metrics" 2>/dev/null || true)
  if [[ -z "$metrics" ]]; then
    echo "WARN: exporter unreachable; skipping sample" >&2
    continue
  fi

  clk=$(echo "$metrics" | awk '/^cipher_gpu_sm_clock_mhz/ { print $2 }')
  clk="${clk:-$PEAK_CLOCK_MHZ}"

  min_mfu_seen=100.0

  while IFS= read -r row; do
    echo "$row"
    mfu=$(echo "$row" | awk -F, '{ print $6 }')
    if awk "BEGIN { exit !($mfu < $min_mfu_seen) }"; then
      min_mfu_seen="$mfu"
    fi
  done < <(echo "$metrics" | awk -v ts="$t" -v cnt="$step" -v clk="$clk" -v peak="$PEAK_CLOCK_MHZ" '
    /^cipher_tenant_sm_util_pct\{/ {
      match($0, /tenant="[^"]+"/); tn=substr($0,RSTART+8,RLENGTH-9);
      match($0, /pid="[^"]+"/);    pd=substr($0,RSTART+5,RLENGTH-6);
      sm=$2;
      mfu = (sm/100.0)*(clk/peak)*100.0;
      printf "%d,%d,%s,%s,%.0f,%.2f,measured\n", ts, cnt, tn, pd, sm, mfu;
    }')

  # G3 gate: any tenant below MIN_TENANT_MFU stops the push
  if awk "BEGIN { exit !($min_mfu_seen < $MIN_TENANT_MFU) }"; then
    echo "$t,$step,SATURATED,-,-,$min_mfu_seen,below_min_mfu_$MIN_TENANT_MFU"
    echo "INFO: density push stopped — tenant MFU dropped below $MIN_TENANT_MFU%" >&2
    break
  fi
done

echo "INFO: density push complete. Max tenants observed: ${#TENANT_PIDS[@]}" >&2
