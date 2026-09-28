#!/usr/bin/env bash
# T4.2.4e two-prefill noisy-neighbor orchestrator.
#
# Spawns two Mistral-7B prefill B=8 tenants (tp_t1, tp_t2) that compete
# for SMs (same-resource contention, unlike T4.2.4d's compute/HBM mismatch).
# Each tenant captures its own per-prefill wall-clock latencies.
#
# Library binding controlled by caller via CUDA_INJECTION64_PATH:
#   A (partition ON):  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so (T4.2.4d)
#   B (partition OFF): /home/ubuntu/libcipher_v2/libcipher_v2.so
#
# Outputs:
#   /tmp/tp_t1.log /tmp/tp_t2.log         per-tenant stderr/stdout
#   /tmp/cipher_tenant_tp_t1_progress     run_for_duration progress (t1)
#   /tmp/cipher_tenant_tp_t2_progress     run_for_duration progress (t2)
#   /tmp/tp_t1_prefill_latencies.json     per-prefill wall-clock ms (t1)
#   /tmp/tp_t2_prefill_latencies.json     per-prefill wall-clock ms (t2)
#   /tmp/tp_device.csv                    device-wide MFU/power timeline
#
# Usage:
#   WL_DURATION=120 LEAD=5 \
#     CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
#     bash wl_two_prefill_neighbor.sh

set -u
WL_ROOT=/home/ubuntu/cipher_workloads
DURATION="${WL_DURATION:-120}"
LEAD="${LEAD:-5}"
LABEL="${TP_LABEL:-two-prefill}"

[[ -z "${CUDA_INJECTION64_PATH:-}" ]] && {
  echo "[tp] CUDA_INJECTION64_PATH must be set" >&2; exit 2;
}

echo "[tp:$LABEL] duration=${DURATION}s lead=${LEAD}s"
echo "[tp:$LABEL] libcipher=$(basename "$CUDA_INJECTION64_PATH")"

rm -f /tmp/tp_t1.log /tmp/tp_t2.log \
      /tmp/cipher_tenant_tp_t1_progress \
      /tmp/cipher_tenant_tp_t2_progress \
      /tmp/tp_t1_prefill_latencies.json \
      /tmp/tp_t2_prefill_latencies.json \
      /tmp/tp_device.csv

SAMPLE_DUR=$((DURATION + LEAD + 10))
(
  echo "ts_s,power_w,sm_util_pct,sm_clock_mhz"
  t0=$(date +%s); end=$((t0+SAMPLE_DUR))
  while [[ $(date +%s) -lt $end ]]; do
    m=$(curl -fsS http://localhost:9402/metrics 2>/dev/null || true)
    if [[ -n "$m" ]]; then
      pw=$(echo "$m"   | awk '/^cipher_gpu_power_watts/ { print $2 }')
      sm=$(echo "$m"   | awk '/^cipher_gpu_sm_util_pct/ { print $2 }')
      clk=$(echo "$m"  | awk '/^cipher_gpu_sm_clock_mhz/ { print $2 }')
      ts=$(( $(date +%s) - t0 ))
      echo "$ts,${pw:-0},${sm:-0},${clk:-0}"
    fi
    sleep 1
  done
) > /tmp/tp_device.csv &
DEV_SAMPLER=$!

# Both tenants want a full 125 s window so they overlap completely.
T_TOTAL=$((DURATION + LEAD))

echo "[tp:$LABEL] launching tenant 1 (tp_t1) ..."
CIPHER_TENANT_ID="tp_t1" WL_DURATION="$T_TOTAL" \
  python3 "$WL_ROOT/drivers/wl_prefill_with_latency.py" \
  > /tmp/tp_t1.log 2>&1 &
T1_PID=$!
echo "[tp:$LABEL] t1 pid=$T1_PID; warming up ${LEAD}s ..."

sleep "$LEAD"

echo "[tp:$LABEL] launching tenant 2 (tp_t2) ..."
CIPHER_TENANT_ID="tp_t2" WL_DURATION="$DURATION" \
  python3 "$WL_ROOT/drivers/wl_prefill_with_latency.py" \
  > /tmp/tp_t2.log 2>&1 &
T2_PID=$!
echo "[tp:$LABEL] t2 pid=$T2_PID"

wait "$T2_PID"; T2_RC=$?
wait "$T1_PID"; T1_RC=$?

sleep 2
kill -INT "$DEV_SAMPLER" 2>/dev/null
wait "$DEV_SAMPLER" 2>/dev/null || true

echo "[tp:$LABEL] t1_rc=$T1_RC t2_rc=$T2_RC"
echo "[tp:$LABEL] artifacts:"
ls -la /tmp/tp_t1.log /tmp/tp_t2.log \
       /tmp/cipher_tenant_tp_t1_progress /tmp/cipher_tenant_tp_t2_progress \
       /tmp/tp_t1_prefill_latencies.json /tmp/tp_t2_prefill_latencies.json \
       /tmp/tp_device.csv 2>&1 | sed 's/^/[tp]   /'
