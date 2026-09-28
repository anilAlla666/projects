#!/bin/bash
# W.4a multi-tenant ELIGIBILITY verification (decision-only; no transport).
#   M1: 4x TinyLlama-1.1B (same fp)            -> ONE 4-tenant coalesce-group (eligible)
#   M2: TinyLlama-fp16 + TinyLlama-AWQ (diff fp) -> TWO solo groups; distinct_fp_rejected grows. THE guard.
#   M3: 2x TinyLlama-fp16 + 1x TinyLlama-AWQ   -> one 2-group + one solo; correct partition.
# Long-lived light tenants (W.6sC methodology — 8B OOMs concurrent). The .so is
# bind-mounted over the CDI path (host file UNCHANGED). W.4a does NO coalescing;
# the DECISION (group_size / eligible_groups / distinct_fp_rejected) is verified
# from the CLASSIFY-POOL stderr lines.
set -uo pipefail
OUT=/home/ubuntu/cipher-fusion-evidence/v1_phase_b/k1_close_gate
RT=/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so
PROBE=$OUT/mt_probe.py
KVB=/home/ubuntu/cipher_rt_phase4/build_py312/cipher_kv_bridge.cpython-312-x86_64-linux-gnu.so
SECS=75

launch() {  # name model util
    sudo docker run --rm --gpus all \
        -e CUDA_INJECTION64_PATH="$RT" -e CIPHER_VERBOSE=0 \
        -e HF_HOME=/home/ubuntu/.cache/huggingface -e TRANSFORMERS_OFFLINE=0 \
        -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        -v "$RT:/usr/lib/cipher/libcipher_rt.so:ro" \
        -v /home/ubuntu:/home/ubuntu -v /home/ubuntu/models:/models \
        -v "$KVB:/home/ubuntu/cipher_rt_phase4/cipher_kv_bridge.so:ro" \
        -w /home/ubuntu --entrypoint /bin/bash vllm/vllm-openai:v0.21.0 \
        -c "python3 $PROBE $1 \"$2\" $3 $SECS 2>&1" > "$OUT/captures/${1}.log" 2>&1
}
report() {  # name
    local log="$OUT/captures/${1}.log"
    local co eg gs dfr
    co=$(grep -hoE "co_resident=[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    eg=$(grep -hoE "eligible_groups=[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    gs=$(grep -hoE "group_size=[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    dfr=$(grep -hoE "distinct_fp_rejected=[0-9]+" "$log" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    echo "  [$1] max_co_resident=${co:-NA} max_eligible_groups=${eg:-NA} max_group_size=${gs:-NA} max_distinct_fp_rejected=${dfr:-NA}"
}

mkdir -p "$OUT/captures"
echo "=== W.4a POOL MULTI-TENANT ELIGIBILITY $(date -u +%H:%M:%SZ) RT=$(md5sum $RT|cut -d' ' -f1) ==="

echo "--- M1: 4x TinyLlama-1.1B same fp (expect eligible group_size=4) ---"
for i in 1 2 3 4; do launch P1m_tiny$i /models/TinyLlama-1.1B 0.18 & sleep 2; done
wait
for i in 1 2 3 4; do report P1m_tiny$i; done

echo "--- drain 35s ---"; sleep 35

echo "--- M2: TinyLlama-fp16 + TinyLlama-AWQ distinct fp (expect group_size=1 each, distinct_fp_rejected GROWS, ZERO cross-fp group) ---"
launch P2m_fp16 /models/TinyLlama-1.1B 0.2 & sleep 2
launch P2m_awq "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ" 0.2 & wait
report P2m_fp16; report P2m_awq

echo "--- drain 35s ---"; sleep 35

echo "--- M3: 2x TinyLlama-fp16 + 1x TinyLlama-AWQ (expect fp16 pair group_size=2, awq solo, awq rejected) ---"
launch P3m_fp16a /models/TinyLlama-1.1B 0.18 & sleep 2
launch P3m_fp16b /models/TinyLlama-1.1B 0.18 & sleep 2
launch P3m_awq "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ" 0.18 & wait
report P3m_fp16a; report P3m_fp16b; report P3m_awq
echo "=== COMPLETE $(date -u +%H:%M:%SZ) ==="