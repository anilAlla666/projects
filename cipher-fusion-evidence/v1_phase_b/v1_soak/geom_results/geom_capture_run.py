#!/usr/bin/env python3
# Capture vLLM real launch geometries + the geometry-classifier verdict, then
# the name oracle. Run with CUDA_INJECTION64_PATH=<staging .so>, CIPHER_GEOM_CAPTURE=1,
# CIPHER_MARLIN=on. Produces /tmp/cipher_geom_capture.jsonl (fn,geometry,geom_class)
# + /tmp/cipher_kernel_table.json (fn,name,name_class) for the offline join.
import os, ctypes
from vllm import LLM, SamplingParams
M = "/home/ubuntu/models/TinyLlama-1.1B"
# enforce_eager=True so launches go through cuLaunchKernel observably (no CUDA-graph capture).
so = os.environ.get("MARLIN_SO", "/home/ubuntu/cipher_rt_phase4/libcipher_rt.so")
def dump_names():
    lib = ctypes.CDLL(so)
    if hasattr(lib, "cipher_kernel_table_report"):
        lib.cipher_kernel_table_report()
        print("kernel_table_report dumped /tmp/cipher_kernel_table.json", flush=True)
try:
    llm = LLM(model=M, enforce_eager=True, gpu_memory_utilization=0.45, max_model_len=2048,
              dtype="bfloat16")
    out = llm.generate(["The history of computing spans several distinct eras, each defined by"],
                       SamplingParams(max_tokens=32, temperature=0.0))
    print("GEN:", repr(out[0].outputs[0].text[:80]), flush=True)
finally:
    dump_names()
    print("CAPTURE_DONE", flush=True)
