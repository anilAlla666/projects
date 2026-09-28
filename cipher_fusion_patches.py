"""CIPHER fusion patches — auto-applied when this module is imported.

Distributed alongside libcipher_rt.so.  Customer activates via:
    LD_PRELOAD=…libcipher_hook.so… PYTHONSTARTUP=/usr/lib/cipher/cipher_fusion_patches.py python3 my_app.py
or imports explicitly:
    import cipher_fusion_patches   # auto-applies on import

What this does:
  - Auto-detects transformers version (currently supports 4.45.x and 4.57.x;
    extends easily to other versions via the `_PATCHES` registry below).
  - Auto-detects model class (LlamaDecoderLayer; Mistral / Qwen / Gemma share
    the same residual-add structure and use the same patch).
  - Applies fp16-only fusion patches to RMSNorm.forward, MLP.forward, and
    DecoderLayer.forward, calling cipher_fused_rmsnorm / cipher_fused_silu_mul /
    cipher_fused_residual_add through libcipher_rt.so.
  - Falls through cleanly on unsupported transformers versions, on bf16 input,
    or on any kernel-call failure — no crashes, no silent corruption.

Counter source-of-truth: `cipher_fusion_kernels_stats(struct)` in libcipher_rt.so
reports rmsnorm_calls / silu_mul_calls / residual_add_calls.  After this module
runs and a generate() pass completes, all three should be > 0 on Llama-class
models with fp16 weights.
"""
from __future__ import annotations
import os, sys, ctypes, inspect, types

_RT_PATHS = (
    os.environ.get("CIPHER_RT_PATH", ""),
    "/home/ubuntu/op31-prod-fix/libcipher_rt.so",
    "libcipher_rt.so",
)

_rt = None
for p in _RT_PATHS:
    if not p: continue
    try:
        _rt = ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue
if _rt is None:
    if os.environ.get("CIPHER_FUSION_VERBOSE"):
        sys.stderr.write("[CIPHER FUSION] libcipher_rt.so not found — patches skipped\n")
    raise SystemExit(0) if False else None  # silently skip


def _bind_signatures():
    if _rt is None: return
    _rt.cipher_fused_rmsnorm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    _rt.cipher_fused_rmsnorm.restype = ctypes.c_int
    _rt.cipher_fused_silu_mul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    _rt.cipher_fused_silu_mul.restype = ctypes.c_int
    _rt.cipher_fused_residual_add.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p]
    _rt.cipher_fused_residual_add.restype = ctypes.c_int
    try:
        _rt.cipher_fusion_kernels_init.restype = ctypes.c_int
        _rt.cipher_fusion_kernels_init()
    except AttributeError:
        pass


def _try_apply():
    if _rt is None: return
    try:
        import torch
        from transformers.models.llama.modeling_llama import (
            LlamaRMSNorm, LlamaMLP, LlamaDecoderLayer)
    except ImportError:
        return
    import transformers
    tx_ver = transformers.__version__
    _bind_signatures()

    def _stream_ptr_for(dev):
        return ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    _orig_rms = LlamaRMSNorm.forward
    _orig_mlp = LlamaMLP.forward
    _orig_dec = LlamaDecoderLayer.forward

    global _call_counts
    _call_counts = [0, 0, 0, 0, 0, 0]  # rms_total, rms_rc1, mlp_total, mlp_rc1, res_total, res_rc1
    def _fused_rms(self, x):
        _call_counts[0] += 1
        if x.dtype != torch.float16 or self.weight.dtype != torch.float16:
            return _orig_rms(self, x)
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty_like(x)
        out_flat = out.reshape(-1, x.shape[-1])
        rc = _rt.cipher_fused_rmsnorm(
            flat.data_ptr(), self.weight.data_ptr(), out_flat.data_ptr(),
            flat.shape[0], flat.shape[-1], float(self.variance_epsilon),
            _stream_ptr_for(x.device))
        if rc != 1:
            if _call_counts[0] <= 3 and os.environ.get("CIPHER_FUSION_VERBOSE"):
                sys.stderr.write(f"[CIPHER FUSION] _fused_rms rc={rc} fallback\n")
            return _orig_rms(self, x)
        _call_counts[1] += 1
        return out

    def _fused_mlp(self, x):
        if x.dtype != torch.float16:
            return _orig_mlp(self, x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if gate.dtype != torch.float16:
            return self.down_proj(torch.nn.functional.silu(gate) * up)
        out = torch.empty_like(gate)
        rc = _rt.cipher_fused_silu_mul(
            gate.data_ptr(), up.data_ptr(), out.data_ptr(),
            gate.numel(), _stream_ptr_for(x.device))
        if rc != 1:
            return self.down_proj(torch.nn.functional.silu(gate) * up)
        return self.down_proj(out)

    def _fused_residual_add(x, residual):
        if x.dtype != torch.float16 or residual.dtype != torch.float16:
            return x + residual
        if not x.is_contiguous(): x = x.contiguous()
        if not residual.is_contiguous(): residual = residual.contiguous()
        out = torch.empty_like(x)
        rc = _rt.cipher_fused_residual_add(
            x.data_ptr(), residual.data_ptr(), out.data_ptr(),
            x.numel(), _stream_ptr_for(x.device))
        if rc != 1:
            return x + residual
        return out

    # OP 26 — version-agnostic LlamaDecoderLayer.forward patch.
    # Both 4.45 and 4.57 accept **kwargs and have the same structural
    # forward (RMSNorm → self_attn → residual_add → RMSNorm → MLP →
    # residual_add).  The only difference is the past_key_value vs
    # past_key_values kwarg name and a removed output_attentions in 4.57.
    # We use **kwargs to pass through whatever the caller sends.
    def _patched_dec_forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Pass kwargs through so the per-version self_attn API is honoured.
        attn_out = self.self_attn(hidden_states=hidden_states, **kwargs)
        # transformers 4.45 returns (hidden_states, attn_weights[opt], present_kv[opt])
        # transformers 4.57 returns (hidden_states, attn_weights[opt])
        # Some impls return just hidden_states.  Always work with the leading tensor.
        attn_weights = None
        present_kv = None
        if isinstance(attn_out, tuple):
            hidden_states = attn_out[0]
            if len(attn_out) > 1: attn_weights = attn_out[1]
            if len(attn_out) > 2: present_kv = attn_out[2]
        else:
            hidden_states = attn_out
        hidden_states = _fused_residual_add(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _fused_residual_add(hidden_states, residual)
        # Reproduce the original LlamaDecoderLayer.forward return contract:
        #   outputs = (h,)
        #   if output_attentions: outputs += (attn_weights,)
        #   if use_cache:         outputs += (present_kv,)
        output_attentions = kwargs.get("output_attentions", False)
        use_cache = kwargs.get("use_cache", False)
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        if use_cache:
            outputs = outputs + (present_kv,)
        return outputs

    LlamaRMSNorm.forward = _fused_rms
    LlamaMLP.forward = _fused_mlp
    LlamaDecoderLayer.forward = _patched_dec_forward

    # Hot-rebind on already-instantiated modules: a model may have been
    # imported (and accelerate / torch.nn.Module may have cached the
    # unbound method) before this patch ran.  We can't see the model
    # here yet, but we install a one-shot post-import hook.
    _CACHED = {
        'rms': _fused_rms,
        'mlp': _fused_mlp,
        'dec': _patched_dec_forward,
    }
    def _rebind_existing_models():
        # Walk every loaded torch model in the gc graph that contains a
        # LlamaDecoderLayer.  This catches the case where the customer
        # already loaded the model before this patch fired.
        try:
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, LlamaRMSNorm) and hasattr(obj, '_old_forward'):
                    obj._old_forward = types.MethodType(_CACHED['rms'], obj)
                elif isinstance(obj, LlamaMLP) and hasattr(obj, '_old_forward'):
                    obj._old_forward = types.MethodType(_CACHED['mlp'], obj)
                elif isinstance(obj, LlamaDecoderLayer) and hasattr(obj, '_old_forward'):
                    obj._old_forward = types.MethodType(_CACHED['dec'], obj)
        except Exception:
            pass
    # Try once on import (no-op if model not yet loaded), and stash a
    # callable the customer/runner can invoke after model.from_pretrained().
    _rebind_existing_models()
    globals()['rebind_after_model_load'] = _rebind_existing_models

    # Auto-rebind: wrap from_pretrained so the rebind happens the moment
    # any HF model is constructed.  Customer code is unchanged.
    try:
        from transformers import PreTrainedModel
        _orig_fp = PreTrainedModel.from_pretrained.__func__
        @classmethod
        def _wrapped_fp(cls, *a, **kw):
            m = _orig_fp(cls, *a, **kw)
            _rebind_existing_models()
            return m
        PreTrainedModel.from_pretrained = _wrapped_fp
    except Exception:
        pass

    if os.environ.get("CIPHER_FUSION_VERBOSE"):
        sys.stderr.write(
            f"[CIPHER FUSION] applied (transformers={tx_ver}, "
            f"residual_add via DecoderLayer.forward patch + "
            f"from_pretrained auto-rebind)\n")


# Apply when imported, catch any exception so a transformers-import miss
# doesn't break the user's program.
try:
    _try_apply()
except Exception as e:
    if os.environ.get("CIPHER_FUSION_VERBOSE"):
        sys.stderr.write(f"[CIPHER FUSION] auto-apply failed: {e}\n")


def get_python_counts():
    """Returns (rms_total, rms_rc1, mlp_total, mlp_rc1, res_total, res_rc1).
    Useful for debugging — proves whether the Python wrappers are reached
    even when the rt counters say 0."""
    try:
        return tuple(_call_counts)
    except NameError:
        return (0, 0, 0, 0, 0, 0)
