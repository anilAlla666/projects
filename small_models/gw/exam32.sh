#!/usr/bin/env bash
# Is the guessing a size limit? The untuned Qwen2.5-Coder-32B on the round-4 exam (same prompts, same runner).
set -u
L=~/gw/exam32.log; M=Qwen/Qwen2.5-Coder-32B-Instruct
export PATH=$HOME/vllmenv/bin:$PATH
cd ~/verifier_model
echo "$(date -u +%T) serving" > $L
setsid nohup ~/vllmenv/bin/vllm serve $M --tensor-parallel-size 4 --max-model-len 24576 --port 8000 \
  --gpu-memory-utilization 0.85 > ~/gw/vllm32.log 2>&1 < /dev/null &
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
~/vmenv/bin/python gw/generate.py ~/gw/data4/eval_prompts_labelled.jsonl $M ~/gw/gen32/base >> $L 2>&1
pkill -f "vllm serv[e]"
echo "$(date -u +%T) generated" >> $L
~/vmenv/bin/python gw/run_graders.py ~/real/main ~/gw/gen32/base ~/gw/runs32/qwen32b_untuned --workers 32 >> $L 2>&1
~/vmenv/bin/python gw/eval_graders.py ~/real/main ~/gw/runs32/qwen32b_untuned > ~/gw/scores32.json 2>> $L
echo EXAM32_DONE >> $L
