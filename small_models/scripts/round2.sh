#!/usr/bin/env bash
# Round 2: re-score the chosen round-1 checkpoint on the clean dev (the baseline for comparison), mine the clean
# train set with it, weight hard wrong examples, and fine-tune from it.  Usage: round2.sh BEST_CKPT BEST_CFG
set -u
cd ~/verifier_model
BEST=$1; CFG=$2; L=~/judge/rounds/round2.log
mkdir -p ~/judge/r1best && ln -sf "$BEST" ~/judge/r1best/epoch_0.pt
CUDA_VISIBLE_DEVICES=0 ~/vmenv/bin/python -W ignore scripts/dev_eval.py ~/judge/r1best ~/judge/libs > $L 2>&1 &
CUDA_VISIBLE_DEVICES=1 ~/vmenv/bin/python -W ignore scripts/mine.py "$BEST" ~/judge/libs/serialised_train.jsonl ~/judge/mined.json --attack-dev ~/judge/libs/serialised_attack_dev.jsonl >> $L 2>&1
wait
~/vmenv/bin/python -W ignore scripts/make_weights.py ~/judge/libs ~/judge/mined.json ~/judge/weights_r2.json --hard-factor 4 >> $L 2>&1
sed -e 's/^name = .*/name = "r2"/' -e 's#^train = .*#train = "/home/ubuntu/judge/libs/serialised_train.jsonl"#' -e 's/^epochs = .*/epochs = 3/' "$CFG" > configs/finetune_r2.toml
bash scripts/round.sh r2 configs/finetune_r2.toml --weights ~/judge/weights_r2.json --init "$BEST"
echo ROUND2_DONE >> $L
