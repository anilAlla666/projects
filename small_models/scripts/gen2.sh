#!/usr/bin/env bash
# After round 4 frees the GPUs: serve the 32B model again, sample 6 more attempts per train and dev task with a new
# seed, stop the server, then (once the second hard pass is done) evaluate them on the CPUs.
set -u
L=~/judge/gen2.log
until grep -q ROUND4_DONE ~/judge/rounds/round4.log 2>/dev/null; do sleep 15; done
echo "$(date -u +%T) starting vLLM" > $L
export PATH=$HOME/vllmenv/bin:$PATH
setsid nohup ~/vllmenv/bin/vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 4 --max-model-len 8192 --port 8000 --gpu-memory-utilization 0.85 > ~/judge/vllm2.log 2>&1 < /dev/null &
VP=$!
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; kill -0 $VP 2>/dev/null || { echo "vLLM died" >> $L; exit 1; }; done
echo "$(date -u +%T) vLLM ready" >> $L
cd ~/verifier_model && ~/vmenv/bin/python -u -W ignore scripts/model_attempts_client.py ~/judge/libs Qwen/Qwen2.5-Coder-32B-Instruct --n 6 --seed 1 --splits train,dev --out model_attempts2.jsonl >> $L 2>&1
echo "$(date -u +%T) sampled $(wc -l < ~/judge/libs/model_attempts2.jsonl) rows" >> $L
pkill -f "vllm serv[e]"; sleep 20
until grep -q HARD2_DONE ~/judge/hard2.log 2>/dev/null; do sleep 15; done
VM_WORKERS=96 ~/vmenv/bin/python -u -W ignore scripts/build_judge_data.py more_model ~/judge/libs model_attempts2.jsonl >> $L 2>&1
echo GEN2_DONE >> $L
