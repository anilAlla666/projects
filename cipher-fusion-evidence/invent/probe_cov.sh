#!/bin/bash
# Detection-coverage probes: does the FP8+EAGLE bundle's GEMM traffic pass the cuBLAS seam?
export VLLM_PLUGINS=
export VLLM_DEEP_GEMM_WARMUP=skip
cd /home/ubuntu/cipher-fusion-evidence/invent
SHIM=/home/ubuntu/cipher-fusion-evidence/invent/rvcount.so
rm -f cov_bundle.[0-9]* cov_dense.[0-9]*
echo "=== P1 bundle fp8+eagle + rvcount $(date +%H:%M:%S)"
LD_PRELOAD=$SHIM RVC_OUT=/home/ubuntu/cipher-fusion-evidence/invent/cov_bundle \
  E_FP8=1 E_SPEC=1 E_NREQ=32 timeout 900 python3 eagle_test2.py >cov_bundle.log 2>cov_bundle.err
echo "=== P1 exit=$? $(date +%H:%M:%S)"
echo "=== P2 dense bf16 control + rvcount $(date +%H:%M:%S)"
LD_PRELOAD=$SHIM RVC_OUT=/home/ubuntu/cipher-fusion-evidence/invent/cov_dense \
  E_FP8=0 E_SPEC=0 E_NREQ=32 timeout 900 python3 eagle_test2.py >cov_dense.log 2>cov_dense.err
echo "=== P2 exit=$? $(date +%H:%M:%S)"
echo "PROBES DONE"
