#!/usr/bin/env bash
# Llama-3.1-8B graph-capture clock sweep — find the tok/W maximum.
set -e
cd ~/op31-prod-fix
LDPRE="$(pwd)/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
COMMON_ENV='CIPHER_FP8_COMPUTE=on CIPHER_SUBSTITUTE_V2=on CIPHER_FUSION_KERNELS=on'

# 4 clocks at B=8, plus B=1 at 900/1000/1200.
for clk in 900 1000 1100 1200; do
    sudo nvidia-smi -i 0 -lgc $clk >/dev/null 2>&1
    sudo nvidia-smi -i 1 -lgc $clk >/dev/null 2>&1
    echo "=== clock=${clk}MHz B=8 ==="
    LD_PRELOAD="$LDPRE" CIPHER_FP8_COMPUTE=on CIPHER_SUBSTITUTE_V2=on CIPHER_FUSION_KERNELS=on \
        python3 step9_graph.py --mode=full --batches=8 --measure=10.0 \
        --out=clksweep_b8_${clk}.json 2>&1 \
      | grep -E "  8 |verify|EARLY EOS" | head -3
done

for clk in 900 1000 1200; do
    sudo nvidia-smi -i 0 -lgc $clk >/dev/null 2>&1
    sudo nvidia-smi -i 1 -lgc $clk >/dev/null 2>&1
    echo "=== clock=${clk}MHz B=1 ==="
    LD_PRELOAD="$LDPRE" CIPHER_FP8_COMPUTE=on CIPHER_SUBSTITUTE_V2=on CIPHER_FUSION_KERNELS=on \
        python3 step9_graph.py --mode=full --batches=1 --measure=10.0 \
        --out=clksweep_b1_${clk}.json 2>&1 \
      | grep -E "  1 |verify|EARLY EOS" | head -3
done

sudo nvidia-smi -rgc >/dev/null 2>&1
echo "=== sweep complete, clocks unlocked ==="
