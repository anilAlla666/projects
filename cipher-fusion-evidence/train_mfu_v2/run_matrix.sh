#!/bin/bash
# TRAIN-MFU v2 run matrix (PREREG order). Engaged runs exit 139 at-exit (known, post-results) — tolerated.
cd /home/ubuntu/cipher-fusion-evidence/train_mfu_v2
SUC=/home/ubuntu/cipher_fp8v2_build/libcipher_rt.so
COMMON="--batch 2 --seq 2048"
export CIPHER_LIB=$SUC VLLM_PLUGINS="" CIPHER_VOLT=off

echo "=== RUN 1/6 vanilla baseline (default clock) $(date -u +%H:%M:%S)"
python3 train_step_v2.py $COMMON --steps 50 --warmup 8 --result vanilla_B2_v2.json >vanilla_B2_v2.out 2>vanilla_B2_v2.err
echo "rc=$? json=$(test -s vanilla_B2_v2.json && echo OK || echo MISSING)"

echo "=== RUN 2/6 obs-only successor (hosting tax) $(date -u +%H:%M:%S)"
CUDA_INJECTION64_PATH=$SUC python3 train_step_v2.py $COMMON --steps 50 --warmup 8 --result obsonly_B2_v2.json >obsonly_B2_v2.out 2>obsonly_B2_v2.err
echo "rc=$? json=$(test -s obsonly_B2_v2.json && echo OK || echo MISSING)"

echo "=== RUN 3/6 FP8 successor default full-window $(date -u +%H:%M:%S)"
CUDA_INJECTION64_PATH=$SUC CIPHER_FP8=1 python3 train_step_v2.py $COMMON --steps 50 --warmup 8 --result fp8_B2_default_v2.json >fp8_B2_default_v2.out 2>fp8_B2_default_v2.err
echo "rc=$? json=$(test -s fp8_B2_default_v2.json && echo OK || echo MISSING)"

echo "=== RUN 4/6 FP8 successor steady-state warm20 $(date -u +%H:%M:%S)"
CUDA_INJECTION64_PATH=$SUC CIPHER_FP8=1 python3 train_step_v2.py $COMMON --steps 40 --warmup 20 --result fp8_B2_warm20_v2.json >fp8_B2_warm20_v2.out 2>fp8_B2_warm20_v2.err
echo "rc=$? json=$(test -s fp8_B2_warm20_v2.json && echo OK || echo MISSING)"

echo "=== ISO PAIR: locking 1830 MHz $(date -u +%H:%M:%S)"
sudo -n nvidia-smi -lgc 1830,1830 || echo "LGC FAILED"

echo "=== RUN 5/6 iso vanilla $(date -u +%H:%M:%S)"
python3 train_step_v2.py $COMMON --steps 50 --warmup 8 --pin-mhz 1830 --result isoT_van_v2.json >isoT_van_v2.out 2>isoT_van_v2.err
echo "rc=$? json=$(test -s isoT_van_v2.json && echo OK || echo MISSING)"

echo "=== RUN 6/6 iso FP8 successor $(date -u +%H:%M:%S)"
CUDA_INJECTION64_PATH=$SUC CIPHER_FP8=1 python3 train_step_v2.py $COMMON --steps 50 --warmup 8 --pin-mhz 1830 --result isoT_fp8_v2.json >isoT_fp8_v2.out 2>isoT_fp8_v2.err
echo "rc=$? json=$(test -s isoT_fp8_v2.json && echo OK || echo MISSING)"

echo "=== resetting clocks $(date -u +%H:%M:%S)"
sudo -n nvidia-smi -rgc || echo "RGC FAILED"
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader
echo "=== MATRIX DONE $(date -u +%H:%M:%S)"
