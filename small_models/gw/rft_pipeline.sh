#!/usr/bin/env bash
# Reward-filtered round, after the context model is scored: 4 drafts per training task from the context model, run on
# each task's unfixed code and labelled fixes, keep the drafts that pass every correct fix and catch a wrong one,
# continue training on them, then the same exam as before.
set -u
L=~/gw/rft_pipeline.log; M=Qwen/Qwen2.5-Coder-7B-Instruct; R=~/gw/rft
until grep -q GEN_ALL3_DONE ~/gw/gen_all3.log 2>/dev/null; do sleep 15; done
until grep -q PULL_DONE $R/pull.log 2>/dev/null; do sleep 15; done
export PATH=$HOME/vllmenv/bin:$PATH
cd ~/verifier_model
serve() {
  setsid nohup ~/vllmenv/bin/vllm serve $M --enable-lora --lora-modules gw=$1 --max-lora-rank 64 --tensor-parallel-size 4 \
    --max-model-len 24576 --port 8000 --gpu-memory-utilization 0.85 > ~/gw/vllm_rft.log 2>&1 < /dev/null &
  until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
}
echo "$(date -u +%T) drafting" > $L
serve $HOME/gw/run3/adapter
~/vmenv/bin/python gw/generate.py $R/prompts.jsonl gw $R/drafts --n 4 >> $L 2>&1
pkill -f "vllm serv[e]"; sleep 20
echo "$(date -u +%T) running drafts" >> $L
~/vmenv/bin/python gw/run_graders.py $R $R/drafts $R/runs --workers 40 --all >> $L 2>&1
~/vmenv/bin/python gw/rft_score.py $R $R/runs $R/drafts $R/selected.jsonl >> $L 2>&1
mkdir -p $R/data && cp $R/selected.jsonl $R/data/train.jsonl
echo "$(date -u +%T) training" >> $L
~/vllmenv/bin/python -m torch.distributed.run --nproc_per_node 4 gw/train.py $R/data ~/gw/run4 --max-len 12288 --epochs 2 --lr 1e-4 \
  --init-adapter ~/gw/run3/adapter >> ~/gw/train4.log 2>&1
[ -d ~/gw/run4/adapter ] || { echo "no adapter" >> $L; exit 1; }
echo "$(date -u +%T) exam" >> $L
serve $HOME/gw/run4/adapter
~/vmenv/bin/python gw/generate.py ~/gw/data3/eval_prompts_labelled.jsonl gw ~/gw/gen4/gw >> $L 2>&1
pkill -f "vllm serv[e]"
~/vmenv/bin/python gw/run_graders.py ~/real/main ~/gw/gen4/gw ~/gw/runs4/gw --workers 32 >> $L 2>&1
~/vmenv/bin/python gw/eval_graders.py ~/real/main ~/gw/runs3/base ~/gw/runs3/gw ~/gw/runs4/gw ~/gw/runs3/sonnet > ~/gw/scores4.json 2>> $L
echo RFT_DONE >> $L
