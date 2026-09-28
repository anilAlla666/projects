#!/bin/bash
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 D9_ATTN=sdpa
unset CIPHER_FP8 CIPHER_FP8_SHARE_ACT; timeout 500 python3 d9_fp8_mfu.py 2>/dev/null | grep '\[off\]'; mv -f d9_fp8_mfu_off.json d9_lk2_bf16.json
CIPHER_FP8=on timeout 500 python3 d9_fp8_mfu.py 2>/dev/null | grep '\[fp8\]'; mv -f d9_fp8_mfu_fp8.json d9_lk2_percall.json
CIPHER_FP8=on CIPHER_FP8_SHARE_ACT=1 timeout 500 python3 d9_fp8_mfu.py 2>/dev/null | grep '\[fp8\]'; mv -f d9_fp8_mfu_fp8.json d9_lk2_shared.json
echo DONE
