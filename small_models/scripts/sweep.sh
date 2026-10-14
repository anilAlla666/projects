#!/usr/bin/env bash
# Round 1: the learning-rate sweep on a frozen copy of the data, after the data chain frees the GPUs.
set -u
until grep -q CHAIN_DONE ~/judge/chain.log 2>/dev/null; do sleep 10; done
mkdir -p ~/judge/sweep && cp ~/judge/libs/serialised_{train,dev,attack_dev}.jsonl ~/judge/sweep/
for n in lr1 lr2 lr3; do DEV_ROOT=$HOME/judge/sweep bash ~/verifier_model/scripts/round.sh "$n" ~/verifier_model/configs/finetune_$n.toml; done
echo SWEEP_DONE >> ~/judge/rounds/sweep.log
