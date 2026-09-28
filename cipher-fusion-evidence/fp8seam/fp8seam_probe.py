# Phase-0 recon (2026-06-12): where do FP8 linears dispatch, and is the cutlass_scaled_mm seam
# reachable by a launcher-side monkeypatch (path B, same class as ra_coophook's CUDAGraphWrapper
# patch) with ZERO vLLM source edits? Runs neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8.
# Simultaneously LD_PRELOAD rvcount.so to show the cuBLAS GemmEx seam sees ~0 of these linears.
import os, json, collections
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import vllm._custom_ops as ops

_real_csmm = ops.cutlass_scaled_mm
shapes = collections.Counter()
calls = {"n": 0}

def _wrapped_csmm(a, b, scale_a, scale_b, out_dtype, bias=None):
    out = _real_csmm(a, b, scale_a, scale_b, out_dtype, bias)
    # a is [M,K] (2D-massaged inside the real fn, but here a is pre-massage); b is [K,N]
    m = a.shape[0] if a.dim() == 2 else a.numel() // a.shape[-1]
    k = a.shape[-1]; n = b.shape[1] if b.dim() == 2 else b.shape[-1]
    shapes[(int(m), int(n), int(k))] += 1
    calls["n"] += 1
    return out

ops.cutlass_scaled_mm = _wrapped_csmm
# vLLM's FP8 kernel imports the symbol into its own module namespace; patch there too if present.
try:
    import vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass as _ctk
    if hasattr(_ctk, "ops"):
        _ctk.ops.cutlass_scaled_mm = _wrapped_csmm
except Exception as e:
    print("note: cutlass kernel module patch skipped:", str(e)[:100])

from vllm import LLM, SamplingParams

def main():
    M = os.environ["S_MODEL"]; OUT = os.environ["S_OUT"]
    llm = LLM(model=M, enforce_eager=True, gpu_memory_utilization=0.82,
              max_model_len=2048, disable_log_stats=True)
    prompts = ["The history of the Roman Empire begins with"]
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True, min_tokens=16)
    calls["n"] = 0; shapes.clear()
    o = llm.generate(prompts, sp, use_tqdm=False)
    res = {
        "model": M, "eager": True,
        "csmm_calls_total": calls["n"],
        "distinct_shapes": len(shapes),
        "shapes_mnk_count": sorted([[list(k), v] for k, v in shapes.items()], key=lambda x: -x[1]),
        "out": o[0].outputs[0].text[:60],
        "quant": str(getattr(llm.llm_engine.vllm_config.model_config, 'quantization', None)),
    }
    json.dump(res, open(OUT, "w"), indent=1); print("FP8PROBE", json.dumps({k: res[k] for k in ["csmm_calls_total","distinct_shapes","quant"]}))

if __name__ == "__main__":
    main()
