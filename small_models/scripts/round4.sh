#!/usr/bin/env bash
# Round 4: mine the train-plus-attack set with round 3's best, weight its mistakes and the hard wrong examples,
# and refine from it for 2 epochs at the low learning rate.  Usage: round4.sh BEST_CKPT
set -u
cd ~/verifier_model
BEST=$1; L=~/judge/rounds/round4.log
CUDA_VISIBLE_DEVICES=0 ~/vmenv/bin/python -W ignore scripts/mine.py "$BEST" ~/judge/libs/serialised_train_attack.jsonl ~/judge/mined_r4.json --attack-dev ~/judge/libs/serialised_attack_dev.jsonl > $L 2>&1
~/vmenv/bin/python -W ignore scripts/make_weights.py ~/judge/libs ~/judge/mined_r4.json ~/judge/weights_r4.json --hard-factor 4 >> $L 2>&1
bash scripts/round.sh r4 configs/finetune_r4.toml --weights ~/judge/weights_r4.json --init "$BEST"
echo ROUND4_DONE >> $L
