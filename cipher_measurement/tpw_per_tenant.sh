#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# tpw_per_tenant.sh — Phase 4 throughput-per-watt per-tenant measurement.
#
# TPW_tenant = tenant_throughput / tenant_watts_share
#   tenant_watts_share = device_power_w × tenant_sm_pct / sum(all_tenant_sm_pct)
#   tenant_throughput  = tenant launches/sec (CUPTI-flushed)
#     or tokens/sec if workload writes /tmp/cipher_tenant_<ID>_tokens
#
# Baseline TPW from /metrics device-wide is computed as
#   baseline_tpw = baseline_throughput / baseline_power
# Caller provides --baseline-throughput and --baseline-power.
#
# Usage:
#   ./tpw_per_tenant.sh [--duration N] [--interval N] [--exporter URL] \
#                       [--baseline-throughput N] [--baseline-power W]
#
# Output: CSV: timestamp_s,tenant,pid,launches_total,delta_launches,
#              tenant_watts_share,tenant_tpw,tpw_lift_vs_baseline

set -euo pipefail

DURATION=600
INTERVAL=5
EXPORTER="http://localhost:9402"
BASELINE_THROUGHPUT=1000   # launches/sec, caller override
BASELINE_POWER=350         # W, caller override

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)              DURATION="$2"; shift 2 ;;
    --interval)              INTERVAL="$2"; shift 2 ;;
    --exporter)              EXPORTER="$2"; shift 2 ;;
    --baseline-throughput)   BASELINE_THROUGHPUT="$2"; shift 2 ;;
    --baseline-power)        BASELINE_POWER="$2"; shift 2 ;;
    --help|-h) sed -n '4,18p' "$0" >&2; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

baseline_tpw=$(awk "BEGIN { print $BASELINE_THROUGHPUT / $BASELINE_POWER }")

echo "timestamp_s,tenant,pid,launches_total,delta_launches,tenant_watts_share,tenant_tpw,tpw_lift_vs_baseline_x"

declare -A PREV_LAUNCHES
t0=$(date +%s)
last_t=0

while :; do
  t=$(( $(date +%s) - t0 ))
  if [[ "$t" -ge "$DURATION" ]]; then break; fi
  delta_t=$(( t - last_t ))
  if [[ "$delta_t" -le 0 ]]; then delta_t=1; fi
  last_t="$t"

  metrics=$(curl -fsS "${EXPORTER}/metrics" 2>/dev/null || true)
  if [[ -z "$metrics" ]]; then
    sleep "$INTERVAL"; continue
  fi

  power_w=$(echo "$metrics" | awk '/^cipher_gpu_power_watts/ { print $2 }')
  power_w="${power_w:-0}"

  # Collect (tenant,pid,sm,launches) tuples
  while IFS=, read -r tenant pid sm launches; do
    [[ -z "$tenant" ]] && continue
    sum_sm=$(awk "BEGIN { print ${sum_sm:-0} + $sm }")
  done < <(echo "$metrics" | awk '
    /^cipher_tenant_sm_util_pct\{/    { match($0, /tenant="[^"]+"/); tn=substr($0,RSTART+8,RLENGTH-9);
                                          match($0, /pid="[^"]+"/); pd=substr($0,RSTART+5,RLENGTH-6);
                                          sm_map[tn","pd]=$2; }
    /^cipher_tenant_launches_total\{/ { match($0, /tenant="[^"]+"/); tn=substr($0,RSTART+8,RLENGTH-9);
                                          match($0, /pid="[^"]+"/); pd=substr($0,RSTART+5,RLENGTH-6);
                                          la_map[tn","pd]=$2; }
    END {
      for (k in sm_map) {
        split(k, a, ",");
        printf "%s,%s,%s,%s\n", a[1], a[2], sm_map[k], (k in la_map ? la_map[k] : 0);
      }
    }')

  # Recompute per-tenant share
  echo "$metrics" | awk -v ts="$t" -v pw="$power_w" -v base="$baseline_tpw" -v dt="$delta_t" -v exporter="$EXPORTER" '
    function fetch_sum_sm() {
      cmd = "curl -fsS " exporter "/metrics 2>/dev/null | awk \"/cipher_tenant_sm_util_pct/ { s += \\$2 } END { print s }\"";
      cmd | getline s; close(cmd);
      return s + 0;
    }
    BEGIN { sum = fetch_sum_sm(); if (sum < 1) sum = 1; }
    /^cipher_tenant_sm_util_pct\{/    { match($0, /tenant="[^"]+"/); tn=substr($0,RSTART+8,RLENGTH-9);
                                          match($0, /pid="[^"]+"/); pd=substr($0,RSTART+5,RLENGTH-6);
                                          sm[tn","pd]=$2; }
    /^cipher_tenant_launches_total\{/ { match($0, /tenant="[^"]+"/); tn=substr($0,RSTART+8,RLENGTH-9);
                                          match($0, /pid="[^"]+"/); pd=substr($0,RSTART+5,RLENGTH-6);
                                          la[tn","pd]=$2; }
    END {
      for (k in sm) {
        split(k, a, ",");
        share = (sm[k] / sum) * pw;
        if (share < 1) share = 1;
        tput = (la[k]+0) / dt;
        tpw  = (tput > 0) ? tput / share : 0;
        lift = (base > 0) ? tpw / base : 0;
        printf "%d,%s,%s,%s,n/a,%.2f,%.4f,%.2f\n", ts, a[1], a[2], la[k]+0, share, tpw, lift;
      }
    }'

  sleep "$INTERVAL"
done
