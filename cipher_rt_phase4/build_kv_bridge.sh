#!/usr/bin/env bash
# Build cipher_kv_bridge — the T4.6.2 from_blob bridge (pybind11 + libtorch
# extension module). Output: cipher_kv_bridge.so, importable from Python.
set -euo pipefail
cd "$(dirname "$0")"

TORCH_DIR=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "torch : $TORCH_DIR"
echo "python: $PY_INC"

# C allocator object (compiled as C).
# cipher_rt_kv_alloc.c includes cipher_kvdedup.h — the T4.6.4 ioctl ABI header,
# canonically shared from the kmod tree (CIPHER_KMOD_DIR overrides the default).
CIPHER_KMOD_DIR="${CIPHER_KMOD_DIR:-/home/ubuntu/cipher_kmod}"
gcc -O2 -fPIC -I. -I"$CIPHER_KMOD_DIR" -c cipher_rt_kv_alloc.c -o cipher_rt_kv_alloc.o

g++ -O2 -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"$TORCH_DIR/include" \
    -I"$TORCH_DIR/include/torch/csrc/api/include" \
    -I"$PY_INC" \
    -I. \
    -Wl,-rpath,"$TORCH_DIR/lib" \
    -L"$TORCH_DIR/lib" \
    -o "cipher_kv_bridge${EXT_SUFFIX}" \
    cipher_kv_bridge.cpp cipher_rt_kv_alloc.o \
    -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda -lcuda

# Also expose under the plain name for simple sys.path import.
ln -sf "cipher_kv_bridge${EXT_SUFFIX}" cipher_kv_bridge.so 2>/dev/null || true

echo "built : cipher_kv_bridge${EXT_SUFFIX}"
