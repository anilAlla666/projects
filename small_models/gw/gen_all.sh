#!/usr/bin/env bash
# After the fine-tune: serve the base model with the trained adapter, write graders with both, stop the server, then
# run every grader (base, fine-tuned, and Sonnet's if present) on the unfixed code and all labelled attempts, and score.
set -u
L=~/gw/gen_all.log; P=~/gw/data/eval_prompts_labelled.jsonl; M=Qwen/Qwen2.5-Coder-7B-Instruct
until grep -qE "SAVED|TRAIN_EXIT" ~/gw/train.log 2>/dev/null; do sleep 15; done
[ -d ~/gw/run1/adapter ] || { echo "no adapter" > $L; exit 1; }
export PATH=$HOME/vllmenv/bin:$PATH
echo "$(date -u +%T) serving" > $L
setsid nohup ~/vllmenv/bin/vllm serve $M --enable-lora --lora-modules gw=$HOME/gw/run1/adapter --max-lora-rank 64 \
  --tensor-parallel-size 4 --max-model-len 16384 --port 8000 --gpu-memory-utilization 0.85 > ~/gw/vllm.log 2>&1 < /dev/null &
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
cd ~/verifier_model
~/vmenv/bin/python gw/generate.py $P $M ~/gw/gen/base >> $L 2>&1
~/vmenv/bin/python gw/generate.py $P gw ~/gw/gen/gw >> $L 2>&1
pkill -f "vllm serv[e]"
echo "$(date -u +%T) generated" >> $L
for m in base gw sonnet; do
  [ -d ~/gw/gen/$m ] && ~/vmenv/bin/python gw/run_graders.py ~/real/main ~/gw/gen/$m ~/gw/runs/$m --workers 32 >> $L 2>&1
done
~/vmenv/bin/python gw/eval_graders.py ~/real/main ~/gw/runs/base ~/gw/runs/gw $( [ -d ~/gw/runs/sonnet ] && echo ~/gw/runs/sonnet ) > ~/gw/scores.json 2>> $L
echo GEN_ALL_DONE >> $L
