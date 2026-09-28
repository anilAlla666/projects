#!/usr/bin/env bash
# T4.2.4c noisy-neighbor A/B orchestrator.
#
# Spawns one Mistral-7B prefill BOMB tenant (tid=nn_bomb) and one
# TinyLlama decode VICTIM tenant (tid=nn_victim, with per-token latency
# capture). Bomb starts first; victim joins after BOMB_LEAD seconds so
# the SM pressure is already in steady state when victim measurement
# begins. Both run for WL_DURATION seconds.
#
# Library binding is controlled by the caller via CUDA_INJECTION64_PATH:
#   A (partition ON):  /home/ubuntu/cipher_rt_phase4/libcipher_rt.so
#   B (partition OFF): /home/ubuntu/libcipher_v2/libcipher_v2.so
#
# Outputs:
#   /tmp/nn_bomb.log        bomb stderr/stdout
#   /tmp/nn_victim.log      victim stderr/stdout
#   /tmp/cipher_tenant_nn_bomb_progress      run_for_duration progress
#   /tmp/cipher_tenant_nn_victim_progress    run_for_duration progress
#   /tmp/nn_victim_latencies.json            per-token wall-clock times
#   /tmp/nn_device.csv                       device-wide MFU/power samples
#
# Usage:
#   WL_DURATION=120 BOMB_LEAD=5 \
#     CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so \
#     bash wl_noisy_neighbor.sh

set -u
WL_ROOT=/home/ubuntu/cipher_workloads
DURATION="${WL_DURATION:-120}"
BOMB_LEAD="${BOMB_LEAD:-5}"
LABEL="${NN_LABEL:-noisy}"

[[ -z "${CUDA_INJECTION64_PATH:-}" ]] && {
  echo "[nn] CUDA_INJECTION64_PATH must be set (libcipher_rt or libcipher_v2)" >&2
  exit 2
}

echo "[nn:$LABEL] duration=${DURATION}s bomb_lead=${BOMB_LEAD}s"
echo "[nn:$LABEL] libcipher=$(basename "$CUDA_INJECTION64_PATH")"

rm -f /tmp/nn_bomb.log /tmp/nn_victim.log \
      /tmp/cipher_tenant_nn_bomb_progress \
      /tmp/cipher_tenant_nn_victim_progress \
      /tmp/nn_victim_latencies.json \
      /tmp/nn_device.csv

# Device-wide sampler (1 Hz, full window including bomb_lead)
SAMPLE_DUR=$((DURATION + BOMB_LEAD + 10))
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
) > /tmp/nn_device.csv &
DEV_SAMPLER=$!

# Launch bomb
echo "[nn:$LABEL] launching bomb (Mistral-7B prefill B=8) ..."
CIPHER_TENANT_ID="nn_bomb" WL_DURATION="$((DURATION + BOMB_LEAD))" \
  python3 "$WL_ROOT/drivers/wl_noisy_neighbor_bomb.py" \
  > /tmp/nn_bomb.log 2>&1 &
BOMB_PID=$!
echo "[nn:$LABEL] bomb pid=$BOMB_PID; warming up ${BOMB_LEAD}s ..."

sleep "$BOMB_LEAD"

# Launch victim. Victim should run for DURATION; bomb finishes ~BOMB_LEAD after.
echo "[nn:$LABEL] launching victim (TinyLlama decode B=1 + latency capture) ..."
CIPHER_TENANT_ID="nn_victim" WL_DURATION="$DURATION" \
  python3 "$WL_ROOT/drivers/wl_noisy_neighbor_victim.py" \
  > /tmp/nn_victim.log 2>&1 &
VICTIM_PID=$!
echo "[nn:$LABEL] victim pid=$VICTIM_PID"

# Wait for both
wait "$VICTIM_PID"
VICTIM_RC=$?
wait "$BOMB_PID"
BOMB_RC=$?

# Stop sampler
sleep 2
kill -INT "$DEV_SAMPLER" 2>/dev/null
wait "$DEV_SAMPLER" 2>/dev/null || true

echo "[nn:$LABEL] bomb_rc=$BOMB_RC victim_rc=$VICTIM_RC"
echo "[nn:$LABEL] artifacts:"
ls -la /tmp/nn_bomb.log /tmp/nn_victim.log \
       /tmp/cipher_tenant_nn_bomb_progress \
       /tmp/cipher_tenant_nn_victim_progress \
       /tmp/nn_victim_latencies.json \
       /tmp/nn_device.csv 2>&1 | sed 's/^/[nn]   /'
