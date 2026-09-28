# Phase 1A feasibility: can torch.library override the already-registered _C::cutlass_scaled_mm
# CUDA kernel, AND still call the original (no recursion)? Plus: does a functional (no host branch)
# wrap survive torch.compile? Decides ambitious path vs 1B fallback.
import torch, vllm._custom_ops as ops
print("torch", torch.__version__)
pk = torch.ops._C.cutlass_scaled_mm
orig = pk.default
# 1a-i: try to register a CUDA impl for an op that ALREADY has a CUDA kernel (vLLM registered it)
lib = torch.library.Library("_C", "FRAGMENT")
def override(out, a, b, a_scales, b_scales, bias=None):
    return orig(out, a, b, a_scales, b_scales, bias)  # would recurse if override IS the CUDA kernel
try:
    lib.impl("cutlass_scaled_mm", override, "CUDA")
    print("RESULT 1a-i: impl('CUDA') override of existing op = ACCEPTED (no duplicate-registration error)")
except Exception as e:
    print("RESULT 1a-i: impl('CUDA') override = REJECTED:", repr(e)[:240])
# 1a-ii(orig-access): can we obtain the original kernel to call after override? test redispatch-exclude
try:
    ks = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    print("redispatch-exclude guard available:", hasattr(torch._C, "_ExcludeDispatchKeyGuard"))
except Exception as e:
    print("redispatch test err:", repr(e)[:120])
