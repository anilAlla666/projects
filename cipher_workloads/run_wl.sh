#!/usr/bin/env bash
# Phase 4.0.9 — universal workload launcher.
# Usage:
#   ./run_wl.sh WL01 [--duration 600] [--tenant-id NAME]
#
# Forks the driver, sets CIPHER_TENANT_ID, returns driver PID.
# Logs to /tmp/cipher_wl_WLnn_<tenant_id>.log.

set -u
WL_ROOT=/home/ubuntu/cipher_workloads
LIBCIPHER_V2=/home/ubuntu/libcipher_v2/libcipher_v2.so

WL=""
DURATION=600
TENANT_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)  DURATION="$2"; shift 2 ;;
    --tenant-id) TENANT_ID="$2"; shift 2 ;;
    --help|-h) sed -n '4,8p' "$0" >&2; exit 0 ;;
    WL[0-9][0-9]) WL="$1"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$WL" ]]; then
  echo "usage: $0 WLnn [--duration N] [--tenant-id NAME]" >&2
  exit 1
fi

[[ -z "$TENANT_ID" ]] && TENANT_ID="${WL,,}_$$"

wl_num="${WL#WL}"
case "$wl_num" in
  01) DRIVER="$WL_ROOT/drivers/wl01_decode_b1.py" ;;
  02) DRIVER="$WL_ROOT/drivers/wl02_decode_b8.py" ;;
  03) DRIVER="$WL_ROOT/drivers/wl03_prefill_b8.py" ;;
  04) DRIVER="$WL_ROOT/drivers/wl04_vllm_serving.py" ;;
  05) DRIVER="$WL_ROOT/drivers/wl05_multitenant_x8.sh" ;;
  06) DRIVER="$WL_ROOT/drivers/wl06_embeddings.py" ;;
  07) DRIVER="$WL_ROOT/drivers/wl07_lora_finetune.py" ;;
  08) DRIVER="$WL_ROOT/drivers/wl08_sdxl.py" ;;
  09) DRIVER="$WL_ROOT/drivers/wl09_whisper.py" ;;
  10) DRIVER="$WL_ROOT/drivers/wl10_speculative.py" ;;
  11) DRIVER="$WL_ROOT/drivers/wl11_agentic.py" ;;
  12) DRIVER="$WL_ROOT/drivers/wl12_batch.py" ;;
  13) DRIVER="$WL_ROOT/drivers/wl13_long_context_32k.py" ;;
  14) DRIVER="$WL_ROOT/drivers/wl14_torch_compile.py" ;;
  15) DRIVER="$WL_ROOT/drivers/wl15_moe.py" ;;
  16) DRIVER="$WL_ROOT/drivers/wl16_prefix_caching.py" ;;
  17) DRIVER="$WL_ROOT/drivers/wl17_training_full.py" ;;
  18) DRIVER="$WL_ROOT/drivers/wl18_multi_gpu_tp.py" ;;
  19) DRIVER="$WL_ROOT/drivers/wl19_clip.py" ;;
  20) DRIVER="$WL_ROOT/drivers/wl20_llava.py" ;;
  21) DRIVER="$WL_ROOT/drivers/wl21_code_generation.py" ;;
  22) DRIVER="$WL_ROOT/drivers/wl22_rag.py" ;;
  23) DRIVER="$WL_ROOT/drivers/wl23_model_switch.py" ;;
  24) DRIVER="$WL_ROOT/drivers/wl24_awq.py" ;;
  *) echo "unknown workload: $WL" >&2; exit 1 ;;
esac

if [[ ! -f "$DRIVER" ]]; then
  echo "driver missing: $DRIVER" >&2; exit 1
fi

LOGFILE="/tmp/cipher_wl_${WL}_${TENANT_ID}.log"
export CIPHER_TENANT_ID="$TENANT_ID"
export CUDA_INJECTION64_PATH="$LIBCIPHER_V2"
export WL_DURATION="$DURATION"

echo "[run_wl] $WL tenant=$TENANT_ID duration=${DURATION}s driver=$(basename "$DRIVER")"
echo "[run_wl] log=$LOGFILE"

# Drivers are either .py or .sh
if [[ "$DRIVER" == *.sh ]]; then
  bash "$DRIVER" > "$LOGFILE" 2>&1 &
else
  python3 "$DRIVER" > "$LOGFILE" 2>&1 &
fi
PID=$!
echo "[run_wl] pid=$PID"
echo "$PID" > "/tmp/cipher_wl_${WL}_${TENANT_ID}.pid"
wait "$PID"
exit $?
