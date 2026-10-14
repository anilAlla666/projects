#!/usr/bin/env bash
# Starts the grader-writer fine-tune as soon as the realistic judge round has freed the GPUs.
set -u
until grep -qE "ROUND_DONE|ROUND_FAILED" ~/judge/rounds/real1.log 2>/dev/null; do sleep 10; done
cd ~/verifier_model
echo "$(date -u +%T) start" > ~/gw/train.log
~/vllmenv/bin/python -m torch.distributed.run --nproc_per_node 4 gw/train.py ~/gw/data ~/gw/run1 >> ~/gw/train.log 2>&1
echo "$(date -u +%T) TRAIN_EXIT $?" >> ~/gw/train.log
