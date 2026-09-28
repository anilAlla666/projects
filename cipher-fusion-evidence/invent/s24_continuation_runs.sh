#!/bin/bash
# Continuation-session serial runner (2026-06-12). Env protocol per HEADROOM_2X_RESULT.md:
# substrate NOT loaded, deep_gemm warmup skipped. One cell at a time — exclusive GPU.
cd /home/ubuntu/cipher-fusion-evidence/invent
export VLLM_PLUGINS=
export VLLM_DEEP_GEMM_WARMUP=skip
S=s24_continuation_status.txt
: > $S

run_cell() {
  name=$1; shift
  echo "[$(date +%H:%M:%S)] START $name" >> $S
  env "$@" timeout 900 python3 "$HARNESS" > ${name}.log 2> ${name}.err
  rc=$?
  echo "[$(date +%H:%M:%S)] DONE $name rc=$rc $(cat ${name%.json}.json 2>/dev/null | head -c 200)" >> $S
}

SP=/home/ubuntu/cipher-fusion-evidence/invent/sparse24_tao_f8

HARNESS=s24v_harness.py
run_cell s24v_hv S_MODEL=$SP S_QUANT=torchao S_SPEC=1 S_OUT=s24v_hv.json
run_cell s24v_iv S_MODEL=meta-llama/Llama-3.1-8B-Instruct S_QUANT=fp8 S_SPEC=1 S_OUT=s24v_iv.json
run_cell s24v_dv S_MODEL=meta-llama/Llama-3.1-8B-Instruct S_QUANT=fp8 S_OUT=s24v_dv.json
run_cell s24v_bv S_MODEL=meta-llama/Llama-3.1-8B-Instruct S_OUT=s24v_bv.json

HARNESS=s24_ppl.py
run_cell ppl_qa Q_MODEL=neuralmagic/Sparse-Llama-3.1-8B-2of4 Q_OUT=ppl_qa.json
run_cell ppl_qb Q_MODEL=$SP Q_QUANT=torchao Q_OUT=ppl_qb.json
run_cell ppl_qc Q_MODEL=meta-llama/Llama-3.1-8B Q_OUT=ppl_qc.json
run_cell ppl_qd Q_MODEL=meta-llama/Llama-3.1-8B Q_QUANT=fp8 Q_OUT=ppl_qd.json

echo "[$(date +%H:%M:%S)] ALL DONE" >> $S
