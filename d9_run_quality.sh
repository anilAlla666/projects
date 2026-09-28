#!/bin/bash
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L HF_DATASETS_OFFLINE=0
echo "### REF (CIPHER-off bf16) ###"
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 timeout 900 python3 d9_fp8_quality.py 2>/home/ubuntu/d9_q_ref.log | grep -E '\[ref\]|WROTE'
echo "### FP8 (CIPHER-on, transparent) ###"
CUDA_INJECTION64_PATH=$SO CIPHER_FP8=on CIPHER_MARLIN=0 timeout 900 python3 d9_fp8_quality.py 2>/home/ubuntu/d9_q_fp8.log | grep -E '\[fp8\]|WROTE'
echo "### FP8 engaged in quality run (Guard 1) ###"; grep -E 'MATMUL: exit totals|FP8: ENGAGED shape' /home/ubuntu/d9_q_fp8.log | sort -u | tail -8
echo "### compare ###"
python3 d9_fp8_quality_compare.py 2>&1 | tail -20
