"""SGLang fusion shim. Stub — wired at the class level, validation TODO.

SGLang's RMSNorm is in `sglang.srt.layers.layernorm.RMSNorm` (modeled on
vLLM's). The patch follows the same pattern.
"""
from __future__ import annotations

import ctypes
import sys


def apply(rt, model=None):
    try:
        from sglang.srt.layers.layernorm import RMSNorm as SGRMSNorm
    except ImportError as ex:
        return {"applied": False, "framework": "sglang",
                "reason": f"sglang RMSNorm import failed: {ex}"}

    rt.cipher_fused_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int

    import torch
    _orig = SGRMSNorm.forward

    def _stream_for(dev):
        return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    def _patched_forward(self, x, residual=None):
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

    SGRMSNorm.forward = _patched_forward
    print(f"[CIPHER PY sglang] patched SGRMSNorm.forward", file=sys.stderr)
    return {"applied": True, "framework": "sglang",
            "patched_class": "sglang.srt.layers.layernorm.RMSNorm"}
