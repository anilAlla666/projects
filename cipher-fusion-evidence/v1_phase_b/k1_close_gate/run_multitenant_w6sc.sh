#!/bin/bash
# W.6 sub-C multi-tenant engagement test (v2 — long-lived light tenants).
#   M1: 2x TinyLlama-1.1B concurrent          -> both co_resident=2, SAME fp (W.4 coalesces)
#   M2: TinyLlama-1.1B + TinyLlama-AWQ         -> both co_resident=2, DISTINCT fp (W.4 splits)
# Light models (no KV-cache memory-profiling collision that OOMs 2x 8B at
# gpu_memory_utilization=0.3). Each tenant runs continuous light inference for
# RUN_SECS so the two overlap and each observes the other in the kmod cohort
# count. The substrate .so is bind-mounted over the CDI path (host file
# UNCHANGED). The co_resident signal is read from CLASSIFY-CORES stderr lines.
#
# NOTE: the cohort signal is model-agnostic — model-fingerprint discrimination
# for Llama-3-8B (0xd40e) / Mistral-7B (0x9fde) / TinyLlama-AWQ (0x458e) is
# independently verified single-tenant in the 9-cell gate + test_cohort.
set -uo pipefail
OUT=/home/ubuntu/cipher-fusion-evidence/v1_phase_b/k1_close_gate
RT=/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so
PROBE=$OUT/mt_probe.py
KVB=/home/ubuntu/cipher_rt_phase4/build_py312/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so
RUN_SECS=70

launch() {  # name model util
    sudo docker run --rm --gpus all \
        -e CUDA_INJECTION64_PATH="$RT" -e CIPHER_VERBOSE=0 \
        -e HF_HOME=/home/ubuntu/.cache/huggingface -e TRANSFORMERS_OFFLINE=0 \
        -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        -v "$RT:/usr/lib/cipher/libcipher_rt.so:ro" \
        -v /home/ubuntu:/home/ubuntu -v /home/ubuntu/models:/models \
        -v "$KVB:/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so:ro" \
        -w /home/ubuntu --entrypoint /bin/bash vllm/vllm-openai:v0.21.0 \
        -c "python3 $PROBE $1 \"$2\" $3 $RUN_SECS 2>&1" > "$OUT/captures/${1}.log" 2>&1
}

report() {  # name
    local log="$OUT/captures/${1}.log"
    local maxco peers
    maxco=$(grep -hoE "co_resident=[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    peers=$(grep -h "CLASSIFY-CORES" "$log" 2>/dev/null | grep -oE "co_resident=2[^]]*\]" | sort -u | tail -1)
    [ -z "$peers" ] && peers=$(grep -h "CLASSIFY-CORES" "$log" 2>/dev/null | tail -1 | grep -oE "co_resident=[0-9]+ multi=[0-9]+ peers=\[[^]]*\]")
    echo "[$1] max_co_resident=${maxco:-NONE}  sample=[$peers]"
}

mkdir -p "$OUT/captures"
echo "=== W.6 sub-C MULTI-TENANT TEST v2 $(date -u +%H:%M:%SZ) RT=$(md5sum $RT|cut -d' ' -f1) ==="

echo "--- M1: 2x TinyLlama-1.1B concurrent (expect co_resident=2, SAME fp) ---"
launch M1a_tiny /models/TinyLlama-1.1B 0.25 &
sleep 2
launch M1b_tiny /models/TinyLlama-1.1B 0.25 &
wait
report M1a_tiny
report M1b_tiny

echo "--- draining 35s liveness window before M2 ---"
sleep 35

echo "--- M2: TinyLlama-1.1B(fp16) + TinyLlama-AWQ(int4) concurrent (expect co_resident=2, DISTINCT fp) ---"
launch M2a_tiny_fp16 /models/TinyLlama-1.1B 0.25 &
sleep 2
launch M2b_tiny_awq "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ" 0.25 &
wait
report M2a_tiny_fp16
report M2b_tiny_awq
echo "=== MULTI-TENANT TEST COMPLETE $(date -u +%H:%M:%SZ) ==="
