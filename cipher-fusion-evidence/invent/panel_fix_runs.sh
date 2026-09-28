#!/bin/bash
# Panel-fix runs: (1) bundle@300W re-run with retained stderr (M4), (2) torchao 2:4 cell G if dir exists (M2)
export VLLM_PLUGINS=
export VLLM_DEEP_GEMM_WARMUP=skip
cd /home/ubuntu/cipher-fusion-evidence/invent
echo "=== bundle@300W rerun $(date +%H:%M:%S)"
sudo -n nvidia-smi -pl 300
E_FP8=1 E_SPEC=1 E_NREQ=32 timeout 900 python3 eagle_test2.py >tpw_cap300_rerun.log 2>tpw_cap300_rerun.err
echo "=== exit=$? $(date +%H:%M:%S)"
grep '^R ' tpw_cap300_rerun.log | sed 's/^R //' > tpw_cap300_rerun.json
sudo -n nvidia-smi -pl 700
echo "POWER RESET 700W"
echo "=== cell G1 torchao bf16 2:4 semi-sparse $(date +%H:%M:%S)"
S_MODEL=/home/ubuntu/cipher-fusion-evidence/invent/sparse24_tao S_QUANT=torchao S_NREQ=32 S_OUT=s24_g1.json \
  timeout 900 python3 s24_harness.py >s24_g1.log 2>s24_g1.err
echo "=== cell G1 exit=$? $(date +%H:%M:%S)"
echo "=== cell G2 torchao fp8 2:4 sparse-cutlass $(date +%H:%M:%S)"
S_MODEL=/home/ubuntu/cipher-fusion-evidence/invent/sparse24_tao_f8 S_QUANT=torchao S_NREQ=32 S_OUT=s24_g2.json \
  timeout 900 python3 s24_harness.py >s24_g2.log 2>s24_g2.err
echo "=== cell G2 exit=$? $(date +%H:%M:%S)"
echo "PANEL FIX RUNS DONE"
