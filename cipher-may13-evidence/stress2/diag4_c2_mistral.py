"""Diagnostic 4: run c2_marlin.py UNCHANGED but pointed at Mistral-7B.

c2_marlin.py imports stress_common and calls sc.load_model(...) which
reads sc.MODEL_PATH (hardcoded to Llama-3.1-8B). We monkey-patch
sc.MODEL_PATH to Mistral-7B before importing c2_marlin, then call
c2_marlin.main() unchanged.

N_TOKENS=2200 chosen so the run takes ~60s at the May-2 baseline (119 tps)
or ~60s at the density_harness measured rate (36 tps). Either way the
report is single-tenant tps over a meaningful sample.

NOT using LD_PRELOAD — same as density_harness — so the only thing
exercised is the rt's Python-level Marlin substitution path.
"""
import sys, os

sys.path.insert(0, "/workspace/stress")
sys.path.insert(0, "/workspace/stress2")

# Override model path BEFORE c2_marlin imports it.
import stress_common as sc
sc.MODEL_PATH = "/home/ubuntu/models/Mistral-7B-v0.1"

os.environ.setdefault("N_TOKENS", "2200")
os.environ.setdefault("CIPHER_WEIGHT_COMPRESS", "on")
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")
os.environ.setdefault("CIPHER", "0")    # tells c2_marlin we're not LD_PRELOAD'd

# libcuda must be in global namespace before rt.so
import ctypes
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)

import c2_marlin
c2_marlin.main()
