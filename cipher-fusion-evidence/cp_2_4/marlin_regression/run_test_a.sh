#!/usr/bin/env bash
# CP 2.4 test A — launch the end-to-end density driver under the v2 lib.
# Activation per T4.5/T4.6.1: LD_PRELOAD + CUDA_INJECTION64_PATH at the same
# libcipher_rt.so; CIPHER_MARLIN=on arms the Marlin actuator.
set -u
RT=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[run_test_a] v2 lib: $RT  md5=$(md5sum "$RT" | cut -d' ' -f1)"

LD_PRELOAD="$RT" \
CUDA_INJECTION64_PATH="$RT" \
CIPHER_MARLIN=on \
CIPHER_MARLIN_VERBOSE=on \
python3 "$HERE/test_a_density.py" 2>&1
