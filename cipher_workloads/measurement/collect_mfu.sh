#!/usr/bin/env bash
# Phase 4.0.9 — per-tenant MFU sampling.
#
# Usage:
#   ./collect_mfu.sh <tenant_id> <duration_s> [<exporter_url>]
#
# Samples cipher-exporter /metrics at 1 Hz for duration seconds.
# Outputs CSV to stdout:
#   timestamp_s,tenant,pid,sm_util_pct,mem_util_pct,sustained_clock_mhz,launches_total,mfu_pct,device_power_w

set -u
TENANT="$1"
DUR="$2"
EXPORTER="${3:-http://localhost:9402}"

H100_PEAK=1980

echo "timestamp_s,tenant,pid,sm_util_pct,mem_util_pct,sustained_clock_mhz,launches_total,mfu_pct,device_power_w"

t0=$(date +%s)
while :; do
  t=$(( $(date +%s) - t0 ))
  if [[ "$t" -ge "$DUR" ]]; then break; fi

  metrics=$(curl -fsS "${EXPORTER}/metrics" 2>/dev/null || true)
  [[ -z "$metrics" ]] && { sleep 1; continue; }

  clk=$(echo "$metrics" | awk '/^cipher_gpu_sm_clock_mhz/ { print $2 }')
  clk="${clk:-$H100_PEAK}"
  pw=$(echo "$metrics" | awk '/^cipher_gpu_power_watts/ { print $2 }')
  pw="${pw:-0}"

  echo "$metrics" | awk -v ts="$t" -v tn="$TENANT" -v clk="$clk" -v peak="$H100_PEAK" -v pw="$pw" '
    BEGIN { sm = ""; mem = ""; la = ""; pd = ""; }
    /^cipher_tenant_sm_util_pct\{tenant="/ {
      if (match($0, /tenant="[^"]+"/)) {
        cur_tn=substr($0,RSTART+8,RLENGTH-9);
        if (cur_tn == tn) {
          match($0, /pid="[^"]+"/);
          pd=substr($0,RSTART+5,RLENGTH-6);
          sm=$2;
        }
      }
    }
    /^cipher_tenant_mem_util_pct\{tenant="/ {
      if (match($0, /tenant="[^"]+"/)) {
        cur_tn=substr($0,RSTART+8,RLENGTH-9);
        if (cur_tn == tn) mem=$2;
      }
    }
    /^cipher_tenant_launches_total\{tenant="/ {
      if (match($0, /tenant="[^"]+"/)) {
        cur_tn=substr($0,RSTART+8,RLENGTH-9);
        if (cur_tn == tn) la=$2;
      }
    }
    END {
      if (pd != "" && sm != "") {
        mfu = (sm/100.0)*(clk/peak)*100.0;
        printf "%d,%s,%s,%.0f,%s,%.0f,%s,%.2f,%s\n", ts, tn, pd, sm, (mem==""?"0":mem), clk, (la==""?"0":la), mfu, pw;
      }
    }'

  sleep 1
done
