#!/usr/bin/env bash
# T4.0.9.C — P4.1 baseline measurement pass for a single workload.
#
# Usage:
#   ./run_baseline.sh WL01 [--duration 600] [--substitute "TinyLlama for Llama-3.2-1B"]
#
# Drives one workload for `duration` seconds while sampling /metrics at
# 1 Hz. Computes steady-state MFU (median of minute 3 through minute 10).
# Writes expected/wlNN_p41_baseline.json and the raw CSV.
#
# Pre-reqs: cipher_kmod loaded, cipher-gpustate running, cipher-exporter
# listening on :9402.

set -u
WL_ROOT=/home/ubuntu/cipher_workloads
EXPECTED_DIR="$WL_ROOT/expected"
mkdir -p "$EXPECTED_DIR"

WL=""
DURATION=600
SUBST="(none documented)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)   DURATION="$2"; shift 2 ;;
    --substitute) SUBST="$2"; shift 2 ;;
    WL[0-9][0-9]) WL="$1"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$WL" ]]; then
  echo "usage: $0 WLnn [--duration N] [--substitute \"...\"]" >&2
  exit 1
fi

wl_num="${WL#WL}"
DRIVER=""
for c in "$WL_ROOT/drivers/wl${wl_num}_"*.py "$WL_ROOT/drivers/wl${wl_num}_"*.sh; do
  [[ -f "$c" ]] && DRIVER="$c" && break
done
if [[ -z "$DRIVER" ]]; then
  echo "driver missing for $WL" >&2; exit 1
fi

TENANT="${WL,,}_baseline"
CSV="$EXPECTED_DIR/${WL,,}_baseline.csv"
JSON="$EXPECTED_DIR/${WL,,}_p41_baseline.json"
LOG="/tmp/cipher_baseline_${WL}.log"
PROGRESS="/tmp/cipher_tenant_${TENANT}_progress"

rm -f "$CSV" "$JSON" "$LOG" "$PROGRESS"

echo "[baseline] $WL tenant=$TENANT duration=${DURATION}s driver=$(basename "$DRIVER")"

# Launch driver in background
export CIPHER_TENANT_ID="$TENANT"
# Allow CUDA_INJECTION64_PATH override so T4.2.2+ actuator measurements can
# load libcipher_rt instead of libcipher_v2. Defaults preserve the baseline.
export CUDA_INJECTION64_PATH="${CIPHER_INJECTION_OVERRIDE:-/home/ubuntu/libcipher_v2/libcipher_v2.so}"
export WL_DURATION="$DURATION"
# vLLM 0.20.2 deep_gemm warmup throws if the optional deep_gemm package isn't
# installed (graceful-degradation defect upstream). Bypass it; deep_gemm is
# only used for FP8 kernels which our TinyLlama/Mistral substitutions do not
# trigger. See PHASE_4_BACKLOG.md B3.
export VLLM_USE_DEEP_GEMM=0

if [[ "$DRIVER" == *.sh ]]; then
  bash "$DRIVER" > "$LOG" 2>&1 &
else
  python3 "$DRIVER" > "$LOG" 2>&1 &
fi
DRIVER_PID=$!

# Sample MFU in parallel — slightly longer than driver to catch tail
SAMPLE_DUR=$((DURATION + 30))
"$WL_ROOT/measurement/collect_mfu.sh" "$TENANT" "$SAMPLE_DUR" > "$CSV" 2>/dev/null &
SAMPLER_PID=$!

# Wait for driver
wait $DRIVER_PID
DRIVER_RC=$?

# Let sampler drain
sleep 2
kill -INT $SAMPLER_PID 2>/dev/null
wait $SAMPLER_PID 2>/dev/null

# Verify the run was valid
VERIFY=$(python3 "$WL_ROOT/measurement/verify_run.py" "$WL" "$TENANT" 2>&1)
echo "[baseline] verify: $VERIFY"

# Compute steady-state MFU (median of minute 3 through minute 10 = seconds 180..600)
# Only valid if duration >= 600.
T_START=180
T_END=$((DURATION < 600 ? DURATION : 600))

STEADY=$(python3 - <<PYEOF
import csv, statistics, json, time, os, sys
csv_path = "$CSV"
t_start = $T_START
t_end   = $T_END
mfu_vals = []
sm_vals  = []
mem_vals = []
clk_vals = []
last_la  = 0
if not os.path.exists(csv_path):
    print('{"error": "no csv"}'); sys.exit(0)
with open(csv_path) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        try:
            t = int(row["timestamp_s"])
        except (KeyError, ValueError):
            continue
        if t < t_start or t > t_end:
            continue
        try:
            mfu_vals.append(float(row["mfu_pct"]))
            sm_vals.append(float(row["sm_util_pct"]))
            mem_vals.append(float(row["mem_util_pct"]))
            clk_vals.append(float(row["sustained_clock_mhz"]))
            last_la = max(last_la, int(float(row["launches_total"])))
        except (KeyError, ValueError):
            pass

def med(xs): return statistics.median(xs) if xs else 0.0

out = {
    "wl_id":               "$WL",
    "tenant_id":           "$TENANT",
    "duration_s":          $DURATION,
    "samples_in_window":   len(mfu_vals),
    "steady_state_window": f"t={t_start}..{t_end}s",
    "mfu_pct_median":      round(med(mfu_vals), 2),
    "sm_util_pct_median":  round(med(sm_vals), 2),
    "mem_util_pct_median": round(med(mem_vals), 2),
    "sustained_clock_mhz_median": round(med(clk_vals), 0),
    "final_launches_total": last_la,
    "csv_path":            csv_path,
    "substitution":        "$SUBST",
    "verify":              "$VERIFY".strip().split("\n")[0][:200],
    "driver_rc":           $DRIVER_RC,
    "kmod_version":        open("/sys/module/cipher_kmod/version").read().strip(),
    "kmod_srcversion":     open("/sys/module/cipher_kmod/srcversion").read().strip(),
    "timestamp_utc":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open("$JSON", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PYEOF
)

echo "$STEADY"
echo "[baseline] $WL done. json=$JSON"
