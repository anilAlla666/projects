#!/bin/bash
# Marlin GC-on-free — FULL REGRESSION (Mem #13, cache-modification).
#  §1 OFF byte-identical (HARD STOP): vanilla vs my-.so all-actuators-OFF.
#  §2 additivity: my-.so Marlin-ON (with GC, no churn) == pre-GC .so Marlin-ON.
#  §3 test_step3_{b0,b1,c} W7-12 microbenches (additive — must be unchanged).
SO=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
PRE=/home/ubuntu/cipher_rt_phase4/build_cuda13/libcipher_rt.so.d9fp8_staging   # pre-GC Marlin (227d7973)
TORCHLIB=/home/ubuntu/.local/lib/python3.10/site-packages/torch/lib
CU13L=/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH=$TORCHLIB:$CU13L
OFFREG=/home/ubuntu/d9_fp8_close_offreg.py

echo "### §1a vanilla (no inject) ###"
TAG=vanilla timeout 1200 python3 $OFFREG 2>/dev/null | grep -E 'DONE'
echo "### §1b cipher all-OFF (inject my .so, MARLIN+FP8 unset) ###"
CUDA_INJECTION64_PATH=$SO TAG=gcoff timeout 1200 python3 $OFFREG 2>/dev/null | grep -E 'DONE'
echo "### §2a my .so Marlin-ON (GC active, no churn) ###"
CUDA_INJECTION64_PATH=$SO CIPHER_MARLIN=on TAG=marlon_new timeout 1200 python3 $OFFREG 2>/dev/null | grep -E 'DONE'
echo "### §2b pre-GC .so Marlin-ON ###"
CUDA_INJECTION64_PATH=$PRE CIPHER_MARLIN=on TAG=marlon_pre timeout 1200 python3 $OFFREG 2>/dev/null | grep -E 'DONE'

echo "### §1 compare: byte-identical OFF (HARD STOP) ###"
python3 /home/ubuntu/marlin_gc_compare.py vanilla gcoff off_byte_identical
echo "### §2 compare: additivity (my-Marlin-ON vs pre-GC-Marlin-ON, no churn) ###"
python3 /home/ubuntu/marlin_gc_compare.py marlon_pre marlon_new additivity_marlin

echo "### §3 test_step3 microbenches (W7-12 additive) ###"
cd /home/ubuntu/cipher_rt_phase4
for t in test_step3_b0_producer test_step3_b1_consumer; do
  [ -x "$t" ] && (LD_LIBRARY_PATH=$TORCHLIB:$CU13L timeout 300 ./"$t" 2>&1 | grep -iE 'PASS|FAIL|OVERALL' | tail -3 || echo "$t run error")
done
echo "### regression done ###"
