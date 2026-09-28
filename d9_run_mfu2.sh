#!/bin/bash
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L
echo "### OFF (bf16, sdpa, 700W) ###"
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 D9_ATTN=sdpa timeout 500 python3 d9_fp8_mfu.py 2>/home/ubuntu/d9_mfu2_off.log | grep -E '\[off\]'
echo "### ON (FP8, sdpa, 700W) ###"
CUDA_INJECTION64_PATH=$SO CIPHER_FP8=on CIPHER_MARLIN=0 D9_ATTN=sdpa timeout 500 python3 d9_fp8_mfu.py 2>/home/ubuntu/d9_mfu2_on.log | grep -E '\[fp8\]'
echo "### FP8 engaged check ###"; grep -E 'MATMUL: exit totals' /home/ubuntu/d9_mfu2_on.log | tail -1
