#!/bin/bash
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L
echo "### vanilla (no inject) ###"; TAG=vanilla timeout 1200 python3 /home/ubuntu/d9_fp8_close_offreg.py 2>/dev/null | grep -E 'SAVED|DONE'
echo "### cipher_fp8off (inject, FP8 unset) ###"; CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=0 TAG=cipher_fp8off timeout 1200 python3 /home/ubuntu/d9_fp8_close_offreg.py 2>/dev/null | grep -E 'SAVED|DONE'
echo "### cipher_fp8on (inject, FP8 on) ###"; CUDA_INJECTION64_PATH=$SO CIPHER_FP8=on CIPHER_FP8_VERBOSE=1 CIPHER_MARLIN=0 TAG=cipher_fp8on timeout 1200 python3 /home/ubuntu/d9_fp8_close_offreg.py 2>/home/ubuntu/d9_fp8on_telemetry.log | grep -E 'SAVED|DONE'
echo "### compare ###"; python3 /home/ubuntu/d9_fp8_close_compare.py
echo "### FP8-on decline/engage telemetry ###"; grep -E 'MATMUL: exit totals|FP8: ENGAGED|prequant' /home/ubuntu/d9_fp8on_telemetry.log | sort -u | tail -10
