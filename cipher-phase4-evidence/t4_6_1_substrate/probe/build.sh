#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TORCH_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
TORCH_INC="$TORCH_DIR/include"
TORCH_API_INC="$TORCH_DIR/include/torch/csrc/api/include"
TORCH_LIB="$TORCH_DIR/lib"

echo "torch dir : $TORCH_DIR"
echo "includes  : $TORCH_INC ; $TORCH_API_INC"

g++ -O2 -std=c++17 -shared -fPIC -fvisibility=default \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$TORCH_INC" -I"$TORCH_API_INC" \
    -Wl,-rpath,"$TORCH_LIB" \
    -L"$TORCH_LIB" \
    -o cipher_attn_probe.so cipher_attn_probe.cpp \
    -ldl -lc10 -ltorch_cpu

echo "built     : $PWD/cipher_attn_probe.so"
nm -D cipher_attn_probe.so | c++filt | grep -i scaled_dot_product || true
