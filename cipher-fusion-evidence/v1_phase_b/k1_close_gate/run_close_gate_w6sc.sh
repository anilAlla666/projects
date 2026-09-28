#!/bin/bash
# W.6 sub-C close gate runner.
# Identical to run_close_gate.sh EXCEPT it bind-mounts the dev build_cuda13
# libcipher_rt.so over the CDI-injected /usr/lib/cipher/libcipher_rt.so path
# inside the container (read-only). This is required because the cipher-platform
# rev8 CDI hook OVERRIDES the -e CUDA_INJECTION64_PATH env var to
# /usr/lib/cipher/libcipher_rt.so (the .deb baseline, which predates the K.1
# classifier). The host file stays UNCHANGED (Memory: cipher-platform rev8
# UNCHANGED) — the override lives only inside the ephemeral container.
set -uo pipefail

OUT=/home/ubuntu/cipher-fusion-evidence/v1_phase_b/k1_close_gate
RT=/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so
CDI_RT=/usr/lib/cipher/libcipher_rt.so       # CDI-injected path inside container
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
        -v "$RT:$CDI_RT:ro" \
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
    # W.6 sub-C: surface the co-residence signal per cell (single-tenant => co_resident=1).
    grep -h "CLASSIFY-CORES" "$log" 2>/dev/null | grep -v "co_resident=0" | tail -1 | sed "s/^/[$name] CORES: /" | tee -a "$OUT/run_summary.log"
    echo "[$name] done"
}

rm -f "$OUT/results.jsonl" "$OUT/run_summary.log"
echo "W.6 sub-C close gate run: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/run_summary.log"
echo "RT (bind-mounted over CDI path): $(md5sum "$RT")" >> "$OUT/run_summary.log"

run_cell "P1_llama3_8b_bf16" positive "/models/Llama-3.1-8B" 8 512 360
run_cell "P2_mistral_7b_bf16" positive "/models/Mistral-7B-v0.1" 8 512 360
run_cell "P3_tinyllama_awq" positive "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ" 1 512 240
run_cell "E1_bare_torch" negative_bare "n/a" 1 0 60
run_cell "E2_load_idle" negative_loadonly "/models/TinyLlama-1.1B" 1 512 120
run_cell "E3_load_1tok" negative_shutdown "/models/TinyLlama-1.1B" 1 512 120
run_cell "T1_llama3_transition" transition "/models/Llama-3.1-8B" 1 512 240
run_cell "C1_llama3_calibration" calibration "/models/Llama-3.1-8B" 8 512 300
run_cell "N1_mistral_long_prefill" long_prefill "/models/Mistral-7B-v0.1" 1 32768 360

echo "ALL CELLS COMPLETE: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run_summary.log"
echo ""
echo "=== RESULTS ==="
cat "$OUT/results.jsonl"
