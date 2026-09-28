#!/bin/bash
# PROTOCOL: substrate NOT loaded (VLLM_PLUGINS= empty), deep_gemm warmup skipped (no deep_gemm on box)
export VLLM_PLUGINS=
export VLLM_DEEP_GEMM_WARMUP=skip
cd /home/ubuntu/cipher-fusion-evidence/invent
run() {
  name=$1; model=$2; fp8=$3; spec=$4
  echo "=== CELL $name start $(date +%H:%M:%S)"
  S_MODEL=$model S_FP8=$fp8 S_SPEC=$spec S_NREQ=32 S_OUT=s24_${name}.json \
    timeout 900 python3 s24_harness.py >s24_${name}.log 2>s24_${name}.err
  echo "=== CELL $name exit=$? $(date +%H:%M:%S)"
}
run a neuralmagic/Sparse-Llama-3.1-8B-2of4 0 0
run b meta-llama/Llama-3.1-8B-Instruct 0 0
run c neuralmagic/Sparse-Llama-3.1-8B-2of4 1 0
run d meta-llama/Llama-3.1-8B-Instruct 1 0
run e neuralmagic/Sparse-Llama-3.1-8B-2of4 0 1
echo "ALL CELLS DONE"
