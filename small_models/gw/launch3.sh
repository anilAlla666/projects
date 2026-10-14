#!/usr/bin/env bash
# Train the grader writer on the context data once it exists, then generate, run and score (gen_all3.sh).
set -u
until grep -q CONTEXT_DONE ~/gw/context.log 2>/dev/null; do sleep 15; done
cd ~/verifier_model
echo "$(date -u +%T) start" > ~/gw/train3.log
~/vllmenv/bin/python -m torch.distributed.run --nproc_per_node 4 gw/train.py ~/gw/data3 ~/gw/run3 --max-len 12288 --epochs 1 >> ~/gw/train3.log 2>&1
echo "$(date -u +%T) TRAIN_EXIT $?" >> ~/gw/train3.log
bash ~/verifier_model/gw/gen_all3.sh
