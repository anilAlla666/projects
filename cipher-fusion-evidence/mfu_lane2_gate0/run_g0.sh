#!/bin/bash
ANCHOR=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so
AUD=/home/ubuntu/cipher-fusion-evidence/mfu_audit
HERE=/home/ubuntu/cipher-fusion-evidence/mfu_lane2_gate0
WL=$1
inj(){ if [ "$1" = fp8 ]; then echo "CUDA_INJECTION64_PATH=/home/ubuntu/cipher_rt_phase4/libcipher_rt.so CIPHER_FP8=1 CIPHER_FP8_VERBOSE=1 CIPHER_VOLT=off CIPHER_RT_COUNTER_DUMP_PATH=$HERE/cdump_${2}.json"; fi; }
run_torch(){ local tag=$1 rep=$2
  env $(inj $tag torch_${tag}_r${rep}) VLLM_PLUGINS="" MA_TAG=g0_${tag}_r${rep} MA_BATCH=8 MA_SEQ=2048 MA_REPS=5 \
    MA_RESULT=$HERE/torch_${tag}_r${rep}.json \
    timeout 300 python3 $AUD/m_torch_prefill.py 2>$HERE/torch_${tag}_r${rep}.err >/dev/null; }
run_vllm(){ local tag=$1 rep=$2
  env $(inj $tag vllm_${tag}_r${rep}) VLLM_PLUGINS="" VLLM_USE_DEEP_GEMM=0 RV_BATCH=8 RV_PLEN=2048 RV_REPS=10 \
    RV_RESULT=$HERE/vllm_${tag}_r${rep}.json \
    timeout 420 python3 $AUD/m_bench.py prefill 2>$HERE/vllm_${tag}_r${rep}.err >/dev/null; }
for rep in 1 2 3; do for tag in vanilla fp8; do echo "[$(date +%H:%M:%S)] $WL $tag rep$rep"; run_$WL $tag $rep; done; done
echo "DONE $WL"
