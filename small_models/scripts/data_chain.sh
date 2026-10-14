#!/usr/bin/env bash
# After the open model's attempts are written: examples, checks, serialise, then the attack prose and the LLM baseline
# while the 32B model is still loaded, then free the GPUs for fine-tuning.  Usage: data_chain.sh CLIENT_PID VLLM_PID
set -u
cd ~/verifier_model
PY=~/vmenv/bin/python; R=~/judge/libs; M=Qwen/Qwen2.5-Coder-32B-Instruct; L=~/judge/chain.log
step() { echo "$(date -u +%T) start $1" >> $L; shift; "$@" >> ~/judge/chain_steps.log 2>&1; rc=$?; echo "$(date -u +%T) rc=$rc" >> $L; return $rc; }
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "$(date -u +%T) attempts finished: $(wc -l < $R/model_attempts.jsonl) rows" >> $L
step examples env VM_WORKERS=96 $PY -W ignore scripts/build_judge_data.py examples $R || { echo CHAIN_FAILED >> $L; exit 1; }
step checks $PY -W ignore scripts/build_judge_data.py checks $R
step serialise $PY -W ignore scripts/build_judge_data.py serialise $R ~/data/tokenizer.json || { echo CHAIN_FAILED >> $L; exit 1; }
step attacks $PY -W ignore scripts/adversarial.py $R ~/data/tokenizer.json $M
step llm_baseline $PY -W ignore scripts/llm_baseline.py $R $M ~/judge/llm_baseline_test.jsonl
kill "$2"; sleep 30
echo "$(date -u +%T) vLLM stopped; GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
echo CHAIN_DONE >> $L
