MODEL=${1:-$HOME/models/qwen38-27b-fp8}
export PATH=$HOME/vl/bin:/usr/local/cuda/bin:$PATH
export VLLM_USE_DEEP_GEMM=0
NAME=${2:-qwen38-27b}
exec ~/vl/bin/vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port ${PORT:-8000} \
  --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len 262144 --max-num-seqs 64 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --enable-prefix-caching --enable-lora --max-lora-rank 32 --max-loras 1 $LORA
