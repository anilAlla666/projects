#!/bin/bash
# K.1 close gate batched runner.
# Sequential — each cell gets a hard wall-clock cap; logs aggregated.
# Memory #25: stock-config customer-workload measurement.
set -uo pipefail

OUT=/home/ubuntu/cipher-fusion-evidence/v1_phase_b/k1_close_gate
RT=/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so
PROBE=$OUT/probe_workload.py

mkdir -p "$OUT/captures"

run_cell() {
    local name="$1" kind="$2" model="$3" B="${4:-1}" ctx="${5:-512}" cap="${6:-300}"
    local log="$OUT/captures/${name}.log"
    echo "=== [$name] kind=$kind model=$model B=$B ctx=$ctx cap=${cap}s ===" | tee -a "$OUT/run_summary.log"
    timeout "${cap}s" sudo docker run --rm --gpus all \
        -e CUDA_INJECTION64_PATH="$RT" \
        -e CIPHER_VERBOSE=0 \
        -e HF_HOME=/home/ubuntu/.cache/huggingface \
        -e TRANSFORMERS_OFFLINE=0 \
        -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        -v /home/ubuntu:/home/ubuntu \
        -v /home/ubuntu/models:/models \
        -v /home/ubuntu/cipher_rt_phase4/build_py312/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so:/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so:ro \
        -w /home/ubuntu \
        --entrypoint /bin/bash \
        vllm/vllm-openai:v0.21.0 \
        -c "python3 $PROBE $name $kind \"$model\" $B $ctx 2>&1" \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 124 ]; then echo "[$name] TIMEOUT after ${cap}s" | tee -a "$OUT/run_summary.log"; fi
    if [ $rc -ne 0 ]; then echo "[$name] EXIT=$rc" | tee -a "$OUT/run_summary.log"; fi
    grep -h "^RESULT_JSON" "$log" | tail -1 >> "$OUT/results.jsonl" || echo "{\"cell\":\"$name\",\"err\":\"no_result\"}" >> "$OUT/results.jsonl"
    echo "[$name] done"
}

rm -f "$OUT/results.jsonl" "$OUT/run_summary.log"
echo "K.1 close gate run: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/run_summary.log"
md5sum "$RT" >> "$OUT/run_summary.log"

# Positive workloads (3 of 3)
run_cell "P1_llama3_8b_bf16" positive "/models/Llama-3.1-8B" 8 512 360
run_cell "P2_mistral_7b_bf16" positive "/models/Mistral-7B-v0.1" 8 512 360
run_cell "P3_tinyllama_awq" positive "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ" 1 512 240

# Negative cases (3 of 3)
run_cell "E1_bare_torch" negative_bare "n/a" 1 0 60
run_cell "E2_load_idle" negative_loadonly "/models/TinyLlama-1.1B" 1 512 120
run_cell "E3_load_1tok" negative_shutdown "/models/TinyLlama-1.1B" 1 512 120

# Transition test (advisor catch: slow ramp via idle gate)
run_cell "T1_llama3_transition" transition "/models/Llama-3.1-8B" 1 512 240

# Calibration
run_cell "C1_llama3_calibration" calibration "/models/Llama-3.1-8B" 8 512 300

# W.1 (2026-05-27) compute-bound prefill negative-regime cell:
# Mistral-7B bf16 B=1 ctx=32K prefill. Classifier should detect
# large_prefill_detected=1; volt_engage MUST stay 0 (cipher-t43-envelope
# -14% regression case).  Engagement counter g_volt_classifier_skipped
# should be > 0 by the time this cell completes.
run_cell "N1_mistral_long_prefill" long_prefill "/models/Mistral-7B-v0.1" 1 32768 360

echo "ALL CELLS COMPLETE: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run_summary.log"
echo ""
echo "=== RESULTS ==="
cat "$OUT/results.jsonl"
