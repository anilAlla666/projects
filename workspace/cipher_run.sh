#!/bin/bash
# CIPHER launcher — wraps any python invocation with the LD_PRELOAD hook
# and CIPHER's standard env.  No code changes needed in the customer app.
#
# Usage:
#     ./cipher_run.sh script.py [args...]
#     ./cipher_run.sh -c "import torch; ..."
#     ./cipher_run.sh -m module
#
# Optional env overrides BEFORE invocation:
#     CIPHER_TENANT_ID=N        — for multi-tenant FAIRNESS / CARBON
#     CIPHER_FAIRNESS=on        — enable cross-tenant scheduling
#     CIPHER_WEIGHT_SHARE=on    — enable IPC-based weight sharing
#     CIPHER_PYBIN=/usr/bin/python3.11  — interpreter override

set -e
CIPHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Standard CIPHER stack: hook + driver, FP8 + fusion + NCCL bias.
export LD_PRELOAD="$CIPHER_ROOT/libcipher_hook.so /usr/lib/x86_64-linux-gnu/libcuda.so${LD_PRELOAD:+ $LD_PRELOAD}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CIPHER_FP8_COMPUTE="${CIPHER_FP8_COMPUTE:-on}"
export CIPHER_SUBSTITUTE_V2="${CIPHER_SUBSTITUTE_V2:-on}"
export CIPHER_FUSION_KERNELS="${CIPHER_FUSION_KERNELS:-on}"
export CIPHER_NCCL_V4="${CIPHER_NCCL_V4:-on}"
export CIPHER_CARBON="${CIPHER_CARBON:-on}"
# Multi-tenant ops are off by default (require shm coordination).  Set
# CIPHER_FAIRNESS=on / CIPHER_WEIGHT_SHARE=on to opt in.
PYBIN="${CIPHER_PYBIN:-python3}"
exec "$PYBIN" "$@"
