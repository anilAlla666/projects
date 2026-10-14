#!/usr/bin/env bash
# One fine-tuning round on the 4 GPUs, then dev evaluation of every epoch.
# Usage: round.sh NAME CONFIG [finetune.py args...]   Log: ~/judge/rounds/NAME.log, ends ROUND_DONE <run dir> or ROUND_FAILED
set -u
cd ~/verifier_model
NAME=$1; CONFIG=$2; shift 2
mkdir -p ~/judge/rounds; LOG=~/judge/rounds/$NAME.log
~/vmenv/bin/python -m torch.distributed.run --nproc_per_node 4 vmt/finetune.py "$CONFIG" --name "$NAME" "$@" > "$LOG" 2>&1
RUN=$(grep -o "RUN_DIR .*" "$LOG" | awk '{print $2}')
[ -n "$RUN" ] || { echo ROUND_FAILED >> "$LOG"; exit 1; }
CUDA_VISIBLE_DEVICES=0 ~/vmenv/bin/python -W ignore scripts/dev_eval.py "$RUN" "${DEV_ROOT:-$HOME/judge/libs}" >> "$LOG" 2>&1 || { echo ROUND_FAILED >> "$LOG"; exit 1; }
echo "ROUND_DONE $RUN" >> "$LOG"
