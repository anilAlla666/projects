"""vLLM fusion shim.

vLLM uses its own RMSNorm kernel via `vllm._custom_ops.rms_norm` (CUDA
binding) wrapped by `vllm.model_executor.layers.layernorm.RMSNorm`. The
class is detected at runtime; we patch its `forward()` to route through
CIPHER's fused kernel.

Status: stub — class-method patch wired but not yet validated end-to-end.
Validation: M1.T9 24-workload coverage matrix runs vLLM serving with this
shim active and confirms `cipher_fused_rmsnorm` calls > 0.
"""
from __future__ import annotations

import ctypes
import sys


def apply(rt, model=None):
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm
    except ImportError as ex:
        return {"applied": False, "framework": "vllm",
                "reason": f"vllm.layernorm.RMSNorm import failed: {ex}"}

    # Bind argtypes (idempotent with transformers shim)
    rt.cipher_fused_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int

    import torch
    _orig = VLLMRMSNorm.forward

    def _stream_for(dev):
        return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    def _patched_forward(self, x, residual=None):
        # vLLM's RMSNorm.forward signature: (x, residual=None) → tensor or
        # (out, residual). Route only the no-residual fp16 fast path.
        if residual is not None or x.dtype != torch.float16:
            return _orig(self, x, residual)
        if not hasattr(self, 'weight') or self.weight.dtype != torch.float16:
            return _orig(self, x, residual)
        dev = x.device
        with torch.cuda.device(dev):
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            out = torch.empty_like(x)
            out_flat = out.reshape(-1, x.shape[-1])
            rc = rt.cipher_fused_rmsnorm(
                flat.data_ptr(), self.weight.data_ptr(), out_flat.data_ptr(),
                flat.shape[0], flat.shape[-1],
                float(getattr(self, 'variance_epsilon', 1e-6)),
                _stream_for(dev))
        return out if rc == 1 else _orig(self, x, residual)

    VLLMRMSNorm.forward = _patched_forward
    print(f"[CIPHER PY vllm] patched VLLMRMSNorm.forward", file=sys.stderr)
    return {"applied": True, "framework": "vllm",
            "patched_class": "vllm.model_executor.layers.layernorm.RMSNorm"}
