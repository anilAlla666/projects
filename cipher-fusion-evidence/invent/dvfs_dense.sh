#!/bin/bash
# Honest DVFS baseline: dense bf16 on the SAME chat harness as tpw_cap*.json, at 300W and 400W.
export VLLM_PLUGINS=
export VLLM_DEEP_GEMM_WARMUP=skip
cd /home/ubuntu/cipher-fusion-evidence/invent
for CAP in 300 400; do
  sudo -n nvidia-smi -pl $CAP
  echo "=== dense @${CAP}W $(date +%H:%M:%S)"
  E_FP8=0 E_SPEC=0 E_NREQ=32 timeout 900 python3 eagle_test2.py >dvfs_dense${CAP}.log 2>dvfs_dense${CAP}.err
  echo "=== exit=$? $(date +%H:%M:%S)"
  grep '^R ' dvfs_dense${CAP}.log | sed 's/^R //' > dvfs_dense${CAP}.json
done
sudo -n nvidia-smi -pl 700
echo "POWER RESET 700W; DVFS DONE"
