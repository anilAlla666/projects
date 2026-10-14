#!/usr/bin/env bash
# Standalone-file round: the writer trained on complete new test files (data4). Same exam, same steps as gen_all3.sh.
# Waits for the fine-tune, writes graders with the base and the adapter, runs them and Sonnet's, and scores.
set -u
L=~/gw/gen_all4.log; P=~/gw/data4/eval_prompts_labelled.jsonl; M=Qwen/Qwen2.5-Coder-7B-Instruct
until tail -n 3 ~/gw/train4.log 2>/dev/null | grep -qE "TRAIN_EXIT"; do sleep 15; done  # the tail: a stopped earlier run left its marker mid-log
[ -d ~/gw/run4/adapter ] || { echo "no adapter" > $L; exit 1; }
export PATH=$HOME/vllmenv/bin:$PATH
echo "$(date -u +%T) serving" > $L
setsid nohup ~/vllmenv/bin/vllm serve $M --enable-lora --lora-modules gw=$HOME/gw/run4/adapter --max-lora-rank 64 \
  --tensor-parallel-size 4 --max-model-len 24576 --port 8000 --gpu-memory-utilization 0.85 > ~/gw/vllm.log 2>&1 < /dev/null &
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
cd ~/verifier_model
~/vmenv/bin/python gw/generate.py $P $M ~/gw/gen4/base >> $L 2>&1
~/vmenv/bin/python gw/generate.py $P gw ~/gw/gen4/gw >> $L 2>&1
pkill -f "vllm serv[e]"
echo "$(date -u +%T) generated" >> $L
for m in base gw sonnet; do
  [ -d ~/gw/gen4/$m ] && ~/vmenv/bin/python gw/run_graders.py ~/real/main ~/gw/gen4/$m ~/gw/runs4/$m --workers 32 >> $L 2>&1
done
~/vmenv/bin/python gw/eval_graders.py ~/real/main ~/gw/runs4/base ~/gw/runs4/gw $( [ -d ~/gw/runs4/sonnet ] && echo ~/gw/runs4/sonnet ) > ~/gw/scores4.json 2>> $L
echo GEN_ALL4_DONE >> $L
