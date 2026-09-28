#!/usr/bin/env bash
# Phase 4.0.9 — framework + model installation script.
#
# This script is intentionally NOT run during P4.0.9 driver-write phase.
# It installs heavy frameworks (vllm, diffusers, whisper, peft, sentence-
# transformers, autoawq) and downloads models. Run before the P4.0.9
# baseline measurement pass.
#
# Pip installs may take 20-40 minutes total (vllm alone is ~5 minutes).
# Model downloads: ~50 GB if all gated models are accessible.
#
# Substitutions for gated/unavailable models are baked into the driver
# defaults so each WL has an open-model fallback.
#
# Usage:
#   ./setup.sh                  # full install + downloads
#   ./setup.sh --frameworks-only
#   ./setup.sh --models-only

set -uo pipefail

FRAMEWORKS=1
MODELS=1
case "${1:-}" in
  --frameworks-only) MODELS=0 ;;
  --models-only)     FRAMEWORKS=0 ;;
esac

if [[ "$FRAMEWORKS" == 1 ]]; then
  echo "=== Installing Python frameworks ==="
  # CPU-side; CUDA already present in the pod's torch install.
  python3 -m pip install --upgrade --user \
    "vllm" \
    "diffusers" "accelerate" \
    "openai-whisper" \
    "peft" \
    "sentence-transformers" \
    "autoawq" \
    "bitsandbytes" 2>&1 | tail -10 || true

  echo "=== Verify imports ==="
  for mod in vllm diffusers whisper peft sentence_transformers awq bitsandbytes; do
    python3 -c "import $mod; print('  ok: $mod', $mod.__version__ if hasattr($mod, '__version__') else '')" \
      2>&1 | head -1
  done
fi

if [[ "$MODELS" == 1 ]]; then
  echo "=== Downloading models (open / gated-fallback) ==="
  # Open models — no HF token required.
  python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo in [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     # WL01/02/05/07/11/12/14/15/17/21/23
    "mistralai/Mistral-7B-v0.1",               # WL03/13/23 (open; no token needed)
    "sentence-transformers/all-MiniLM-L6-v2",  # WL06/22
    "openai/clip-vit-base-patch32",            # WL19
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",   # WL24
]:
    try:
        p = snapshot_download(repo)
        print(f"  ok: {repo} -> {p}")
    except Exception as e:
        print(f"  FAIL: {repo}: {e}")

# Heavy / optional models — only download if frameworks for them are installed.
heavy = [
    ("stabilityai/stable-diffusion-xl-base-1.0",   "diffusers"),
    ("openai/whisper-large-v3",                     "whisper"),
    ("llava-hf/llava-1.5-7b-hf",                    "transformers"),
]
import importlib
for repo, mod in heavy:
    try:
        importlib.import_module(mod)
        p = snapshot_download(repo)
        print(f"  ok: {repo}")
    except ImportError:
        print(f"  skip: {repo} (framework {mod} not installed)")
    except Exception as e:
        print(f"  FAIL: {repo}: {e}")
PYEOF

  echo "=== HF cache disk usage ==="
  du -sh ~/.cache/huggingface 2>/dev/null || echo "no cache"
fi

echo "=== setup.sh done ==="
