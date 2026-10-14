#!/usr/bin/env bash
# Once the realistic examples exist and the GPUs are free: leak check, serialisation, attack sets, weights, free
# baselines on dev, then a fine-tuning round on the realistic evidence from the best strict model.
# Usage: realistic_pipeline.sh INIT_CKPT
set -u
cd ~/verifier_model
INIT=$1; R=~/judge/realistic; L=~/judge/realistic_pipeline.log; PY=~/vmenv/bin/python; TOK=~/data/tokenizer.json
export VM_SETTING=realistic
until grep -q REALISTIC_DONE ~/judge/realistic.log 2>/dev/null; do sleep 15; done
mkdir -p $R
$PY -W ignore scripts/leaks.py ~/judge/libs realistic > $L 2>&1
$PY -W ignore scripts/build_judge_data.py serialise ~/judge/libs $TOK realistic $R >> $L 2>&1
$PY -W ignore scripts/attacks_setting.py ~/judge/libs $TOK realistic $R >> $L 2>&1
cat $R/serialised_train.jsonl $R/serialised_attack_train.jsonl > $R/serialised_train_attack.jsonl
$PY -W ignore scripts/make_weights.py ~/judge/libs ~/judge/mined_r4.json ~/judge/weights_real1.json --hard-factor 4 >> $L 2>&1
$PY -W ignore scripts/dev_baselines.py ~/judge/libs --split dev >> $L 2>&1
until grep -qE "ROUND_DONE|ROUND_FAILED" ~/judge/rounds/r5.log 2>/dev/null; do sleep 15; done
DEV_ROOT=$R bash scripts/round.sh real1 configs/finetune_real1.toml --weights /home/ubuntu/judge/weights_real1.json --init "$INIT"
echo PIPELINE_DONE >> $L
