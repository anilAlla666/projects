#!/bin/bash
# Launch T2 (8 children × 5 min) sequentially: CIPHER first, then baseline.
set -e
cd "$(dirname "$0")"/..
source /home/ubuntu/cipher-test-venv/bin/activate

DURATION="${DURATION:-300}"
N="${N:-8}"

echo "=== T2 CIPHER (N=$N, ${DURATION}s) ==="
LD_PRELOAD="$(pwd)/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so" \
CIPHER_FP8_COMPUTE=on CIPHER_SUBSTITUTE_V2=on CIPHER_FUSION_KERNELS=on \
CIPHER=1 STRESS_T2_N=$N STRESS_T2_DURATION=$DURATION STRESS_SUFFIX="" \
python3 stress/t2_driver.py 2>&1 | tee stress/t2_cipher.log

echo
echo "=== T2 baseline (N=$N, ${DURATION}s) ==="
CIPHER=0 STRESS_T2_N=$N STRESS_T2_DURATION=$DURATION STRESS_SUFFIX=_baseline \
python3 stress/t2_driver.py 2>&1 | tee stress/t2_baseline.log
