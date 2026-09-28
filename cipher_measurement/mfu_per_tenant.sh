#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# mfu_per_tenant.sh — Phase 4 MFU-per-tenant measurement.
#
# Samples per-tenant SM% from cipher-exporter's /metrics, computes
# MFU per tenant as: MFU = sm_util_pct/100 × sustained_clock_mhz/peak_clock
#
# Approximation: assumes 100% of utilized cycles are tensor-core ops.
# Refined later if cipher-exporter adds tensor_op_ratio metric.
#
# Usage:
#   ./mfu_per_tenant.sh [--duration N] [--interval N] [--exporter URL]
#
# Defaults: duration 300s, interval 1s, exporter http://localhost:9402
#
# Output: CSV to stdout. Columns:
#   timestamp_s,tenant,pid,sm_util_pct,sustained_mhz,mfu_pct

set -euo pipefail

DURATION=300
INTERVAL=1
EXPORTER="http://localhost:9402"
# H100 SXM5 peak boost clock
PEAK_CLOCK_MHZ=1980
# H100 peak FP16 (sparse) at 132 SMs * 1980 MHz = 989 TFLOPS
PEAK_FP16_TFLOPS=989

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)  DURATION="$2"; shift 2 ;;
    --interval)  INTERVAL="$2"; shift 2 ;;
    --exporter)  EXPORTER="$2"; shift 2 ;;
    --help|-h)
      sed -n '4,18p' "$0" >&2
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "timestamp_s,tenant,pid,sm_util_pct,sustained_mhz,mfu_pct"

t0=$(date +%s)
while :; do
  t=$(( $(date +%s) - t0 ))
  if [[ "$t" -ge "$DURATION" ]]; then break; fi

  # Pull /metrics once, derive per-tenant rows
  metrics=$(curl -fsS "${EXPORTER}/metrics" 2>/dev/null || true)
  if [[ -z "$metrics" ]]; then
    echo "WARN: exporter not reachable at ${EXPORTER}" >&2
    sleep "$INTERVAL"
    continue
  fi

  # Device-wide sustained clock (from cipher_gpu_sm_clock_mhz)
  clk=$(echo "$metrics" | awk '/^cipher_gpu_sm_clock_mhz/ { print $2 }')
  clk="${clk:-$PEAK_CLOCK_MHZ}"

  # Per-tenant SM util
  echo "$metrics" | awk -v ts="$t" -v clk="$clk" -v peak="$PEAK_CLOCK_MHZ" '
    /^cipher_tenant_sm_util_pct\{/ {
      # extract tenant="X",pid="N"
      match($0, /tenant="[^"]+"/); tenant=substr($0, RSTART+8, RLENGTH-9);
      match($0, /pid="[^"]+"/);    pid=substr($0, RSTART+5, RLENGTH-6);
      sm=$2;
      mfu = (sm / 100.0) * (clk / peak) * 100.0;
      printf "%d,%s,%s,%.0f,%.0f,%.2f\n", ts, tenant, pid, sm, clk, mfu;
    }
  '

  sleep "$INTERVAL"
done
