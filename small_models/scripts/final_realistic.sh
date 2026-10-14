#!/usr/bin/env bash
# The final evaluation of the general-purpose model, once: calibration on realistic dev, the synthetic test split in
# the realistic setting, the real agent fixes in both settings (strict carries the LLM judges on identical inputs),
# and the adaptation curve.  Usage: final_realistic.sh FINAL_CKPT THRESHOLD
set -u
cd ~/verifier_model
F=$1; TH=$2; R=~/judge/realistic; O=~/judge/final; mkdir -p $O; L=$O/final.log; PY=~/vmenv/bin/python
echo "checkpoint $F threshold $TH" > $L
VM_SETTING=realistic CUDA_VISIBLE_DEVICES=0 $PY -W ignore scripts/calibrate.py "$F" $R/serialised_dev.jsonl $O/calib.json >> $L 2>&1 || { echo FINAL_FAILED >> $L; exit 1; }
T=$(python3 -c "import json; print(json.load(open('$O/calib.json'))['temperature'])")
LO=$(python3 -c "import json; print(json.load(open('$O/calib.json'))['abstain_low'])")
HI=$(python3 -c "import json; print(json.load(open('$O/calib.json'))['abstain_high'])")
VM_SETTING=realistic CUDA_VISIBLE_DEVICES=0 $PY -W ignore scripts/evaluate.py ~/judge/libs "$F" $T $LO $HI $O/eval_test_realistic.md --data-dir $R --llm-scores ~/judge/llm_baseline_test.jsonl --threshold $TH >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -W ignore scripts/evaluate_real.py ~/real/main "$F" $T ~/judge/libs $O/real_realistic.md --setting realistic --threshold $TH ${EXTRA_REALISTIC:-} >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -W ignore scripts/evaluate_real.py ~/real/main "$F" $T ~/judge/libs $O/real_strict.md --setting strict --threshold $TH \
  --llm qwen32b=$HOME/real/main/llm32b_real.jsonl --llm sonnet=$HOME/real/main/sonnet_real.jsonl \
  --llm-attack qwen32b=$HOME/real/main/llm32b_real_attack.jsonl --llm-attack sonnet=$HOME/real/main/sonnet_real_attack.jsonl >> $L 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -W ignore scripts/adapt.py "$F" $R/serialised_dev.jsonl $O/adapt.json >> $L 2>&1 &
wait
echo FINAL_DONE >> $L
