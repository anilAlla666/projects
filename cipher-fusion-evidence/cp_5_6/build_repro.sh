#!/bin/bash
# CP 5.6 / F1 — build + run the minimal single-GEMM reproduction.
set -e
cd /home/ubuntu
CU13=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib

g++ -O2 -std=c++17 cipher-fusion-evidence/cp_5_6/f1_marlin_repro.cpp \
    -o cipher-fusion-evidence/cp_5_6/f1_marlin_repro -ldl
echo "build OK"

# c2c5d313 full-GPU path: no CUDA_INJECTION (no CUPTI green-ctx); engine
# resolves libcudart.so.13 / libnvrtc.so.13 from the cu13 pip wheel.
LD_LIBRARY_PATH="$CU13:$LD_LIBRARY_PATH" \
    ./cipher-fusion-evidence/cp_5_6/f1_marlin_repro
