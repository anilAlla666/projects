#!/bin/bash
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L
run(){ CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 D9_ATTN=sdpa "$@" timeout 500 python3 d9_fp8_mfu.py 2>/dev/null | grep -E '\[off\]|\[fp8\]'; }
echo "### bf16 (locked 1530) ###";            env -u CIPHER_FP8 run
mv -f /home/ubuntu/d9_fp8_mfu_off.json /home/ubuntu/d9_fp8_mfu_lk_bf16.json
echo "### fp8 per-call (locked 1530) ###";     CIPHER_FP8=on run; mv -f /home/ubuntu/d9_fp8_mfu_fp8.json /home/ubuntu/d9_fp8_mfu_lk_percall.json
echo "### fp8 SHARED quant (locked 1530) ###";  CIPHER_FP8=on CIPHER_FP8_SHARE_ACT=1 run; mv -f /home/ubuntu/d9_fp8_mfu_fp8.json /home/ubuntu/d9_fp8_mfu_lk_shared.json
