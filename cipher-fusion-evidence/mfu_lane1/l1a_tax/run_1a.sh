#!/bin/bash
# Lane 1a interleaved V/E hosting-tax driver. Engaged = unmodified anchor via CUDA_INJECTION64_PATH + CIPHER_VOLT=off
# (disarm: source proof cipher_rt_volt.c:344 — unset defaults VOLT to ARMED/classifier-self-engage). No actuator env.
ANCHOR=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
AUD=/home/ubuntu/cipher-fusion-evidence/mfu_audit
HERE=/home/ubuntu/cipher-fusion-evidence/mfu_lane1/l1a_tax
WL=$1   # torch | vllm | train
REPS=3
run_torch(){ # $1=tag(vanilla/engaged) $2=rep
  local tag=$1 rep=$2 env=""
  if [ "$tag" = "engaged" ]; then env="CUDA_INJECTION64_PATH=$ANCHOR CIPHER_VOLT=off"; fi
  env $env VLLM_PLUGINS="" MA_TAG=${tag}_r${rep} MA_BATCH=8 MA_SEQ=2048 MA_REPS=5 \
    MA_RESULT=$HERE/torch_${tag}_r${rep}.json \
    timeout 240 python3 $AUD/m_torch_prefill.py 2>$HERE/torch_${tag}_r${rep}.err >/dev/null
}
run_vllm(){
  local tag=$1 rep=$2 env=""
  if [ "$tag" = "engaged" ]; then env="CUDA_INJECTION64_PATH=$ANCHOR CIPHER_VOLT=off"; fi
  env $env VLLM_PLUGINS="" VLLM_USE_DEEP_GEMM=0 RV_BATCH=8 RV_PLEN=2048 RV_REPS=10 \
    RV_RESULT=$HERE/vllm_${tag}_r${rep}.json \
    timeout 360 python3 $AUD/m_bench.py prefill 2>$HERE/vllm_${tag}_r${rep}.err >/dev/null
}
run_train(){
  local tag=$1 rep=$2 env=""
  if [ "$tag" = "engaged" ]; then env="CUDA_INJECTION64_PATH=$ANCHOR CIPHER_VOLT=off"; fi
  env $env VLLM_PLUGINS="" \
    timeout 300 python3 $AUD/m_train.py --batch 1 --seq 512 --steps 40 \
    --result $HERE/train_${tag}_r${rep}.json 2>$HERE/train_${tag}_r${rep}.err >/dev/null
}
for rep in 1 2 3; do
  for tag in vanilla engaged; do
    echo "[$(date +%H:%M:%S)] $WL $tag rep$rep"
    run_$WL $tag $rep
  done
done
echo "DONE $WL"
