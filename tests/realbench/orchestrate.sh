#!/bin/bash
# Realbench orchestrator — runs baseline + CIPHER + compare as separate
# processes (LD_PRELOAD must be set fresh for the cipher phase).
#
# Usage:
#   bash tests/realbench/orchestrate.sh [--model PATH] [--batch N] [--limit N]
#                                        [--prompts PATH] [--no-sample]
#                                        [--out RUN_DIR]
set -u
cd "$(dirname "$0")"/../..
ROOT=$(pwd)
source /home/ubuntu/cipher-test-venv/bin/activate

MODEL=/home/ubuntu/models/Llama-3.1-8B
BATCH=8
LIMIT=0
PROMPTS=$ROOT/tests/realbench/prompts.jsonl
SAMPLE_FLAG=""
OUT=$ROOT/stress2/realbench_$(date +%Y%m%d_%H%M%S)

while [ $# -gt 0 ]; do
    case "$1" in
        --model)     MODEL="$2"; shift 2;;
        --batch)     BATCH="$2"; shift 2;;
        --limit)     LIMIT="$2"; shift 2;;
        --prompts)   PROMPTS="$2"; shift 2;;
        --no-sample) SAMPLE_FLAG="--no-sample"; shift;;
        --out)       OUT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$OUT"
echo "[realbench] OUT=$OUT  MODEL=$MODEL  BATCH=$BATCH  LIMIT=$LIMIT  prompts=$PROMPTS"

LIMIT_ARG=""; [ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"

# Phase 1: baseline (no LD_PRELOAD)
echo
echo "============================================================"
echo "Phase: baseline (no CIPHER)"
echo "============================================================"
unset LD_PRELOAD
python3 tests/realbench/run.py \
    --phase baseline \
    --model "$MODEL" --batch "$BATCH" $LIMIT_ARG $SAMPLE_FLAG \
    --prompts "$PROMPTS" --out "$OUT" 2>&1 | tail -10

# Phase 2: CIPHER (LD_PRELOAD set)
echo
echo "============================================================"
echo "Phase: CIPHER"
echo "============================================================"
export LD_PRELOAD="$ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so"
python3 tests/realbench/run.py \
    --phase cipher \
    --model "$MODEL" --batch "$BATCH" $LIMIT_ARG $SAMPLE_FLAG \
    --prompts "$PROMPTS" --out "$OUT" 2>&1 | tail -10
unset LD_PRELOAD

# Phase 3: compare (post-hoc, no LD_PRELOAD needed)
echo
echo "============================================================"
echo "Phase: compare"
echo "============================================================"
python3 tests/realbench/run.py \
    --phase compare \
    --prompts "$PROMPTS" --out "$OUT" 2>&1 | tail -5

echo
echo "[realbench] DONE → $OUT/COMPARISON.md"
echo
cat "$OUT/COMPARISON.md"
