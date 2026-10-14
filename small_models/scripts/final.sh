#!/usr/bin/env bash
# The final evaluation, run once with the checkpoint and threshold chosen on dev.  Usage: final.sh FINAL_CKPT THRESHOLD
set -u
cd ~/verifier_model
F=$1; TH=$2; O=~/judge/final; mkdir -p $O; L=$O/final.log
echo "checkpoint $F threshold $TH" > $L
CUDA_VISIBLE_DEVICES=0 ~/vmenv/bin/python -W ignore scripts/calibrate.py "$F" ~/judge/libs/serialised_dev.jsonl $O/calib.json >> $L 2>&1 || { echo FINAL_FAILED >> $L; exit 1; }
T=$(python3 -c "import json; print(json.load(open('$O/calib.json'))['temperature'])")
LO=$(python3 -c "import json; print(json.load(open('$O/calib.json'))['abstain_low'])")
HI=$(python3 -c "import json; print(json.load(open('$O/calib.json'))['abstain_high'])")
CUDA_VISIBLE_DEVICES=0 ~/vmenv/bin/python -W ignore scripts/evaluate.py ~/judge/libs "$F" $T $LO $HI $O/eval_report.md --llm-scores ~/judge/llm_baseline_test.jsonl --threshold $TH >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=1 ~/vmenv/bin/python -W ignore scripts/evaluate_real.py ~/real/main "$F" $T ~/judge/libs $O/real_strict.md --setting strict --threshold $TH \
  --llm qwen32b=$HOME/real/main/llm32b_real.jsonl --llm sonnet=$HOME/real/main/sonnet_real.jsonl \
  --llm-attack qwen32b=$HOME/real/main/llm32b_real_attack.jsonl --llm-attack sonnet=$HOME/real/main/sonnet_real_attack.jsonl >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=2 ~/vmenv/bin/python -W ignore scripts/evaluate_real.py ~/real/main "$F" $T ~/judge/libs $O/real_realistic.md --setting realistic --threshold $TH >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=3 ~/vmenv/bin/python -W ignore scripts/adapt.py "$F" ~/judge/libs/serialised_dev.jsonl $O/adapt.json >> $L 2>&1 &
wait
echo FINAL_DONE >> $L
