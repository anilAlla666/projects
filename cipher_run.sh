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

# Default to the production preset (operator can override via
# CIPHER_PROFILE=cloud / experimental / safety / perf / full).
export CIPHER_PROFILE="${CIPHER_PROFILE:-production}"

# Per-feature env vars (these are also set by the preset autoinit, but we
# leave them here for backwards compatibility with callers that override
# the preset choice).
export CIPHER_FP8_COMPUTE="${CIPHER_FP8_COMPUTE:-on}"
export CIPHER_SUBSTITUTE_V2="${CIPHER_SUBSTITUTE_V2:-on}"
export CIPHER_FUSION_KERNELS="${CIPHER_FUSION_KERNELS:-on}"
export CIPHER_NCCL_V4="${CIPHER_NCCL_V4:-on}"
export CIPHER_CARBON="${CIPHER_CARBON:-on}"

# Auto-load the per-framework fusion shim. PYTHONSTARTUP runs at
# Python startup; the shim installs a one-shot hook on the first
# torch.cuda.synchronize() that detects the customer's framework and
# applies the matching class-method patches.
if [ -f "$CIPHER_ROOT/cipher_python/autoload.py" ]; then
    export PYTHONSTARTUP="${CIPHER_PYTHONSTARTUP:-$CIPHER_ROOT/cipher_python/autoload.py}"
    export CIPHER_RT_PATH="${CIPHER_RT_PATH:-$CIPHER_ROOT/libcipher_rt.so}"
fi

# Multi-tenant ops are off by default (require shm coordination).
PYBIN="${CIPHER_PYBIN:-python3}"
exec "$PYBIN" "$@"
