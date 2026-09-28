"""HuggingFace transformers fusion shim.

Patches `LlamaRMSNorm.forward` and `LlamaMLP.forward` (also Mistral, Qwen,
DeepSeek that subclass these or use the same shape) to dispatch through
CIPHER's fused CUDA kernels:

  - cipher_fused_rmsnorm(x, weight, out, rows, hidden, eps, stream)
  - cipher_fused_silu_mul(gate, up, out, numel, stream)

Falls back to PyTorch on dtype mismatch or kernel failure. Compatible with
the `accelerate` _old_forward bypass pattern.

This is the SAFE shim — extracted from stress/stress_common.py and
genericized. Used by cipher_python.apply_shims() when
detect_framework() returns 'transformers'.
"""
from __future__ import annotations

import ctypes
import sys
import types


def _bind_argtypes(rt):
    rt.cipher_fused_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    rt.cipher_fused_rmsnorm.restype = ctypes.c_int
    rt.cipher_fused_silu_mul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_silu_mul.restype = ctypes.c_int
    rt.cipher_fused_residual_add.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    rt.cipher_fused_residual_add.restype = ctypes.c_int


def apply(rt, model=None):
    import torch
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm, LlamaMLP)
    except ImportError as ex:
        return {"applied": False, "reason": str(ex)}

    _bind_argtypes(rt)
    _orig_rms = LlamaRMSNorm.forward
    _orig_mlp = LlamaMLP.forward

    def _stream_for(dev):
        return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    def _try_fused_rmsnorm(x, weight, eps):
        dev = x.device
        with torch.cuda.device(dev):
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            out = torch.empty_like(x)
            out_flat = out.reshape(-1, x.shape[-1])
            rc = rt.cipher_fused_rmsnorm(
                flat.data_ptr(), weight.data_ptr(), out_flat.data_ptr(),
                flat.shape[0], flat.shape[-1], float(eps),
                _stream_for(dev))
        return out, (rc == 1)

    def _try_fused_silu_mul(gate, up):
        dev = gate.device
        if not gate.is_contiguous(): gate = gate.contiguous()
        if not up.is_contiguous():   up   = up.contiguous()
        with torch.cuda.device(dev):
            out = torch.empty_like(gate)
            rc = rt.cipher_fused_silu_mul(
                gate.data_ptr(), up.data_ptr(), out.data_ptr(),
                gate.numel(), _stream_for(dev))
        return out, (rc == 1)

    def _patched_rmsnorm_forward(self, x):
        if x.dtype != torch.float16 or self.weight.dtype != torch.float16:
            return _orig_rms(self, x)
        out, ok = _try_fused_rmsnorm(x, self.weight, self.variance_epsilon)
        return out if ok else _orig_rms(self, x)
    LlamaRMSNorm.forward = _patched_rmsnorm_forward

    def _patched_mlp_forward(self, x):
        if x.dtype != torch.float16:
            return _orig_mlp(self, x)
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        if gate.dtype != torch.float16 or up.dtype != torch.float16:
            mul = torch.nn.functional.silu(gate) * up
            return self.down_proj(mul)
        mul, ok = _try_fused_silu_mul(gate, up)
        if not ok:
            mul = torch.nn.functional.silu(gate) * up
        return self.down_proj(mul)
    LlamaMLP.forward = _patched_mlp_forward

    # If the model is provided, also rebind accelerate's _old_forward
    # per-instance — without that, accelerate captures the original forward
    # at hook-install time and our class-method patch never runs.
    n_rms = n_mlp = 0
    if model is not None:
        for m in model.modules():
            if isinstance(m, LlamaRMSNorm) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_patched_rmsnorm_forward, m)
                n_rms += 1
            elif isinstance(m, LlamaMLP) and hasattr(m, '_old_forward'):
                m._old_forward = types.MethodType(_patched_mlp_forward, m)
                n_mlp += 1

    print(f"[CIPHER PY transformers] patched LlamaRMSNorm + LlamaMLP "
          f"(n_rms={n_rms} n_mlp={n_mlp} model-bound)", file=sys.stderr)
    return {"applied": True, "framework": "transformers",
            "n_rms_modules": n_rms, "n_mlp_modules": n_mlp}
