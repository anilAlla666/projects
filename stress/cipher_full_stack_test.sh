#!/bin/bash
# CIPHER FULL-STACK STRESS TEST
# Baseline (1 + 15 tenant) + CIPHER FULL STACK (15 tenant 10 min)
# Llama-3.2-1B fp16, GPU 0, 1200 MHz lock, all 33 ops enabled.

set -u
cd "$(dirname "$0")"/..
ROOT=$(pwd)
source /home/ubuntu/cipher-test-venv/bin/activate

# ============================================================================
# CONFIG
# ============================================================================
RUN_DIR="${RUN_DIR:-$ROOT/stress2/full_stack_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "[ORCH] Run dir: $RUN_DIR"

GPU_ID="${GPU_ID:-0}"
N_TENANTS="${N_TENANTS:-15}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-120}"
SINGLE_BASELINE_DURATION_S="${SINGLE_BASELINE_DURATION_S:-30}"
CIPHER_DURATION_S="${CIPHER_DURATION_S:-600}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/models/Llama-3.2-1B}"
BURST_LEN="${BURST_LEN:-10}"
BATCH="${BATCH:-1}"
PREFILL_LEN="${PREFILL_LEN:-128}"
NVSMI_INTERVAL_S="${NVSMI_INTERVAL_S:-5}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# ============================================================================
# Lock GPU clock at 1200 MHz on GPU 0 (best-effort; may fail without sudo)
# ============================================================================
nvidia-smi -i "$GPU_ID" -lgc 1200 >/dev/null 2>&1 || true
nvidia-smi -i "$GPU_ID" --query-gpu=clocks.gr --format=csv,noheader >> "$RUN_DIR/gpu_clock_initial.txt"

# ============================================================================
# nvidia-smi sampler — backgrounded, dumps CSV per phase
# ============================================================================
sample_nvsmi() {
    local out=$1
    local stop_file=$2
    echo "ts,power_w,clock_mhz,mem_used_mib,util_pct,temp_c" > "$out"
    while [ ! -f "$stop_file" ]; do
        ts=$(date +%s.%N)
        line=$(nvidia-smi -i "$GPU_ID" \
            --query-gpu=power.draw,clocks.gr,memory.used,utilization.gpu,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        echo "$ts,$line" >> "$out"
        sleep "$NVSMI_INTERVAL_S"
    done
}

# ============================================================================
# Spawn N tenants, wait for completion, return success count
# ============================================================================
spawn_tenants() {
    local n=$1
    local dur=$2
    local out_dir=$3
    local with_cipher=$4   # 0 or 1
    local stderr_dir="$out_dir/stderr"
    mkdir -p "$out_dir" "$stderr_dir"

    local pids=()
    for i in $(seq 0 $((n-1))); do
        (
            export CIPHER_TENANT_ID="$i"
            export MT_DURATION_S="$dur"
            export MT_OUT_PATH="$out_dir/t${i}.json"
            export MT_BURST_LEN="$BURST_LEN"
            export MT_BATCH="$BATCH"
            export MT_PREFILL="$PREFILL_LEN"
            export MT_MODEL_PATH="$MODEL_PATH"
            export CUDA_VISIBLE_DEVICES="$GPU_ID"
            if [ "$with_cipher" = "1" ]; then
                export LD_PRELOAD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
                export CIPHER_FP8_COMPUTE=on
                export CIPHER_FUSION_KERNELS=on
                export CIPHER_SUBSTITUTE_V2=on
                export CIPHER_PERSIST_ENGINE=on
                export CIPHER_PERSIST=on
                export CIPHER_FAIRNESS=on
                export CIPHER_FLOW_RECORD=on
                export CIPHER_FLOW_MATCH=on
                export CIPHER_FLOW_SUBSTITUTE=on
                export CIPHER_FUSION=on
                export CIPHER_THERMOSTAT=on
                export CIPHER_NCCL_TUNER=on
                export CIPHER_NCCL_V4=on
                export CIPHER_GRAPH=on
                export CIPHER_CARBON=on
                export CIPHER_COMPLY=on
                export CIPHER_TRACE=on
                export CIPHER_RECEIPT=on
                export CIPHER_DETERMINISM=off
                export CIPHER_TOPOLOGY=on
                export CIPHER_GUARD=on
                export CIPHER_PREDICT=on
                export CIPHER_PIPELINE=on
                export CIPHER_CONTINUITY=on
                export CIPHER_LOOP=on
                export CIPHER_SPECULATE_CHECK=on
                export CIPHER_SENSE=on
                export CIPHER_SHIELD=on
                export CIPHER_SUSTAIN=on
                export CIPHER_PULSE=on
                export CIPHER_VOLT=on
                export CIPHER_HIBERNATE=on
                export CIPHER_PARTITION_ROUTER=on
                export CIPHER_THERMAL_FEEDBACK=on
                export CIPHER_VMM=on
                export CIPHER_KV_COMPRESS=on
                export CIPHER_EDMD_LIVE=on
                export CIPHER_ATTN_KOOPMAN=on
                export CIPHER_WEIGHT_COMPRESS=on
                export CIPHER_DVFS=on
                export CIPHER_WORKLOAD_DETECT=on
                export CIPHER_WORKLOAD_REPORT=on
                export CIPHER_COUNTERS_DUMP_DIR="$out_dir/counters"
                export MT_LOAD_RT=1
                mkdir -p "$out_dir/counters"
            else
                unset LD_PRELOAD
                export MT_LOAD_RT=0
            fi
            python3 stress/full_stack_child.py \
                > "$stderr_dir/t${i}.stdout" 2> "$stderr_dir/t${i}.stderr"
        ) &
        pids+=($!)
        sleep 0.4   # stagger to avoid simultaneous CUDA init thundering herd
    done

    echo "[ORCH] Spawned ${#pids[@]} tenants (cipher=$with_cipher, dur=${dur}s)"
    for pid in "${pids[@]}"; do wait "$pid" || true; done

    local ok=0
    for i in $(seq 0 $((n-1))); do
        [ -s "$out_dir/t${i}.json" ] && ok=$((ok+1))
    done
    echo "[ORCH] $ok/$n tenants produced JSON output"
}

# ============================================================================
# PHASE A — single-tenant baseline (no CIPHER)
# ============================================================================
echo
echo "============================================================"
echo "PHASE A — single-tenant baseline (no CIPHER, ${SINGLE_BASELINE_DURATION_S}s)"
echo "============================================================"
A_DIR="$RUN_DIR/phase_a_single_baseline"
mkdir -p "$A_DIR"
A_STOP="$A_DIR/stop"
sample_nvsmi "$A_DIR/nvsmi.csv" "$A_STOP" &
A_SAMPLER=$!
spawn_tenants 1 "$SINGLE_BASELINE_DURATION_S" "$A_DIR" 0
touch "$A_STOP"
wait "$A_SAMPLER" || true

# ============================================================================
# PHASE B — 15-tenant baseline (no CIPHER)
# ============================================================================
echo
echo "============================================================"
echo "PHASE B — ${N_TENANTS}-tenant baseline (no CIPHER, ${BASELINE_DURATION_S}s)"
echo "============================================================"
B_DIR="$RUN_DIR/phase_b_multi_baseline"
mkdir -p "$B_DIR"
B_STOP="$B_DIR/stop"
sample_nvsmi "$B_DIR/nvsmi.csv" "$B_STOP" &
B_SAMPLER=$!
spawn_tenants "$N_TENANTS" "$BASELINE_DURATION_S" "$B_DIR" 0
touch "$B_STOP"
wait "$B_SAMPLER" || true

# ============================================================================
# PHASE C — 15-tenant CIPHER FULL STACK
# ============================================================================
echo
echo "============================================================"
echo "PHASE C — ${N_TENANTS}-tenant CIPHER FULL STACK (${CIPHER_DURATION_S}s)"
echo "============================================================"
C_DIR="$RUN_DIR/phase_c_multi_cipher"
mkdir -p "$C_DIR"
C_STOP="$C_DIR/stop"
sample_nvsmi "$C_DIR/nvsmi.csv" "$C_STOP" &
C_SAMPLER=$!
spawn_tenants "$N_TENANTS" "$CIPHER_DURATION_S" "$C_DIR" 1
touch "$C_STOP"
wait "$C_SAMPLER" || true

# ============================================================================
# REPORT
# ============================================================================
echo
echo "============================================================"
echo "Generating report → $RUN_DIR/REPORT.md"
echo "============================================================"
python3 "$ROOT/stress/full_stack_report.py" "$RUN_DIR" "$N_TENANTS" \
    > "$RUN_DIR/REPORT.md"

echo
echo "[ORCH] DONE → $RUN_DIR/REPORT.md"
cat "$RUN_DIR/REPORT.md"
