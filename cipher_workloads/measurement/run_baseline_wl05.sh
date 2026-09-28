#!/usr/bin/env bash
# T4.0.9.C — WL05 multi-tenant ×8 baseline measurement.
#
# WL05 spawns 8 wl01_decode_b1 children with CIPHER_TENANT_ID=wl05_t1..wl05_t8.
# Baseline JSON tracks each sub-tenant + device aggregate MFU.
#
# Usage:
#   ./run_baseline_wl05.sh [--duration 600]

set -u
WL_ROOT=/home/ubuntu/cipher_workloads
EXPECTED_DIR="$WL_ROOT/expected"
mkdir -p "$EXPECTED_DIR"

DURATION=600
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

JSON="$EXPECTED_DIR/wl05_p41_baseline.json"
CSV_DEVICE="$EXPECTED_DIR/wl05_baseline_device.csv"
LOG="/tmp/cipher_baseline_WL05.log"

rm -f /tmp/cipher_tenant_wl05_t*_progress
rm -f "$JSON" "$CSV_DEVICE" "$LOG"
rm -f "$EXPECTED_DIR"/wl05_t*_csv

echo "[baseline-wl05] launching 8-tenant WL05; duration=${DURATION}s"

# Allow CIPHER_INJECTION_OVERRIDE to select a different lib (libcipher_rt
# for Phase 4 actuator measurements). Default preserves the original
# libcipher_v2 behavior.
export CUDA_INJECTION64_PATH="${CIPHER_INJECTION_OVERRIDE:-/home/ubuntu/libcipher_v2/libcipher_v2.so}"
export WL_DURATION="$DURATION"
bash "$WL_ROOT/drivers/wl05_multitenant_x8.sh" > "$LOG" 2>&1 &
DRIVER_PID=$!

# Sample device-wide /metrics + each sub-tenant in parallel.
SAMPLE_DUR=$((DURATION + 30))

(
  echo "timestamp_s,power_watts,sm_util_pct,sm_clock_mhz"
  t0=$(date +%s); end=$((t0+SAMPLE_DUR))
  while [[ $(date +%s) -lt $end ]]; do
    m=$(curl -fsS http://localhost:9402/metrics 2>/dev/null || true)
    [[ -z "$m" ]] && { sleep 1; continue; }
    pw=$(echo "$m"   | awk '/^cipher_gpu_power_watts/ { print $2 }')
    sm=$(echo "$m"   | awk '/^cipher_gpu_sm_util_pct/ { print $2 }')
    clk=$(echo "$m"  | awk '/^cipher_gpu_sm_clock_mhz/ { print $2 }')
    ts=$(( $(date +%s) - t0 ))
    echo "$ts,${pw:-0},${sm:-0},${clk:-0}"
    sleep 1
  done
) > "$CSV_DEVICE" &
DEV_SAMPLER=$!

# Per-tenant samplers
declare -A SAMPLER_PIDS
for i in 1 2 3 4 5 6 7 8; do
  TENANT="wl05_t${i}"
  CSV="$EXPECTED_DIR/wl05_t${i}_baseline.csv"
  "$WL_ROOT/measurement/collect_mfu.sh" "$TENANT" "$SAMPLE_DUR" > "$CSV" 2>/dev/null &
  SAMPLER_PIDS[$i]=$!
done

wait $DRIVER_PID
DRIVER_RC=$?

sleep 2
kill -INT $DEV_SAMPLER ${SAMPLER_PIDS[@]} 2>/dev/null
wait 2>/dev/null

# Compute aggregate baseline
python3 - <<PYEOF
import csv, statistics, json, time, os

expected = "$EXPECTED_DIR"
duration = $DURATION
t_start = 180
t_end = min(600, duration)

per_tenant = []
for i in range(1, 9):
    csv_path = f"{expected}/wl05_t{i}_baseline.csv"
    if not os.path.exists(csv_path):
        per_tenant.append({"sub_tenant": f"wl05_t{i}", "mfu_pct_median": None, "error": "no csv"})
        continue
    mfu = []; sm = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                t = int(row["timestamp_s"])
                if t_start <= t <= t_end:
                    mfu.append(float(row["mfu_pct"]))
                    sm.append(float(row["sm_util_pct"]))
            except (KeyError, ValueError):
                continue
    per_tenant.append({
        "sub_tenant": f"wl05_t{i}",
        "samples_in_window": len(mfu),
        "mfu_pct_median": round(statistics.median(mfu), 2) if mfu else 0.0,
        "sm_util_pct_median": round(statistics.median(sm), 2) if sm else 0.0,
    })

# Device aggregate
dev_csv = f"{expected}/wl05_baseline_device.csv"
dev_pw = []; dev_sm = []; dev_clk = []
if os.path.exists(dev_csv):
    with open(dev_csv) as f:
        for row in csv.DictReader(f):
            try:
                t = int(row["timestamp_s"])
                if t_start <= t <= t_end:
                    dev_pw.append(float(row["power_watts"]))
                    dev_sm.append(float(row["sm_util_pct"]))
                    dev_clk.append(float(row["sm_clock_mhz"]))
            except (KeyError, ValueError):
                continue

def med(xs): return statistics.median(xs) if xs else 0.0

aggregate_mfu = (med(dev_sm) / 100.0) * (med(dev_clk) / 1980.0) * 100.0
out = {
    "wl_id": "WL05",
    "tenant_id": "wl05_aggregate",
    "duration_s": duration,
    "steady_state_window": f"t={t_start}..{t_end}s",
    "per_sub_tenant": per_tenant,
    "device_aggregate": {
        "samples_in_window": len(dev_sm),
        "power_watts_median": round(med(dev_pw), 2),
        "sm_util_pct_median": round(med(dev_sm), 2),
        "sm_clock_mhz_median": round(med(dev_clk), 0),
        "aggregate_mfu_pct": round(aggregate_mfu, 2),
    },
    "substitution": "8x TinyLlama substituted for 8x Llama-3.2-1B",
    "kmod_version": open("/sys/module/cipher_kmod/version").read().strip(),
    "kmod_srcversion": open("/sys/module/cipher_kmod/srcversion").read().strip(),
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open("$JSON", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PYEOF

echo "[baseline-wl05] done. json=$JSON"
