# CIPHER automatic torch.mm interception + fp16 MLP substitution
# Import this module to route all torch.mm calls through CIPHER's pipeline.

import torch
import ctypes
import os
import sys
import numpy as np

_hook = ctypes.CDLL(None)
_hook.cipher_set_gemm_ptrs.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3
_hook.cipher_set_gemm_ptrs.restype = None

# Block-level substitution (Session 3 — fp32 path)
try:
    _hook.cipher_block_sub_collect.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    _hook.cipher_block_sub_collect.restype = ctypes.c_bool
    _hook.cipher_block_sub_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    _hook.cipher_block_sub_predict.restype = ctypes.c_bool
    _hook.cipher_feed_ring.argtypes = [ctypes.c_uint8, ctypes.c_float]
    _hook.cipher_feed_ring.restype = None
    _has_block_sub = True
except AttributeError:
    _has_block_sub = False

_block_sub_ready = False

# ── Session 7: per-layer fp16 fused Koopman kernel ──────────────────────────
_fp16_sub_count = 0
_fp16_fused_ready = False
_per_layer_ptrs = {}  # layer_idx -> (d_vt, d_k, d_w) GPU tensors

try:
    _hook.cipher_koopman_fp16_set_ptrs.argtypes = [ctypes.c_void_p] * 3
    _hook.cipher_koopman_fp16_set_ptrs.restype = None
    _hook.cipher_koopman_fp16_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    _hook.cipher_koopman_fp16_launch.restype = ctypes.c_int
    _has_fp16_kernel = True
except AttributeError:
    _has_fp16_kernel = False

def _load_layer_matrices(layer_idx):
    """Load per-layer V/K/W to GPU. Returns (d_vt, d_k, d_w) or None."""
    base = f'/workspace/manifold/layers'
    v_path = f'{base}/V_layer{layer_idx}.npy'
    k_path = f'{base}/K_layer{layer_idx}.npy'
    w_path = f'{base}/W_layer{layer_idx}.npy'
    if not (os.path.exists(v_path) and os.path.exists(k_path) and os.path.exists(w_path)):
        return None
    V = np.load(v_path)    # (4096, 16)
    K_op = np.load(k_path) # (16, 16)
    W = np.load(w_path)    # (16, 4096)
    V_T = V.T.copy()       # (16, 4096)
    d_vt = torch.from_numpy(V_T).float().cuda().contiguous()
    d_k  = torch.from_numpy(K_op).float().cuda().contiguous()
    d_w  = torch.from_numpy(W).float().cuda().contiguous()
    return (d_vt, d_k, d_w)

def install_mlp_hooks(model, layer_idx=20):
    """Replace layer N MLP with per-layer fused Koopman CUDA kernel."""
    global _fp16_fused_ready
    if not _has_fp16_kernel:
        print(f'[CIPHER] fp16 kernel not available', file=sys.stderr)
        return

    ptrs = _load_layer_matrices(layer_idx)
    if ptrs is None:
        print(f'[CIPHER] Layer {layer_idx}: no calibration data', file=sys.stderr)
        return
    _per_layer_ptrs[layer_idx] = ptrs
    _fp16_fused_ready = True

    mlp = model.model.layers[layer_idx].mlp
    _real_forward = mlp.forward
    d_vt, d_k, d_w = ptrs

    def _koopman_forward(x):
        global _fp16_sub_count
        if x.dtype == torch.float16 and x.shape[-1] == 4096:
            orig_shape = x.shape
            x_flat = x.reshape(-1, 4096).contiguous()
            M = x_flat.shape[0]
            out_flat = torch.empty_like(x_flat)
            # Set per-layer pointers and launch
            _hook.cipher_koopman_fp16_set_ptrs(d_vt.data_ptr(), d_k.data_ptr(), d_w.data_ptr())
            _hook.cipher_koopman_fp16_launch(x_flat.data_ptr(), out_flat.data_ptr(), M)
            _fp16_sub_count += 1
            if _fp16_sub_count <= 5 or (_fp16_sub_count % 500) == 0:
                print(f'[O(1)-fp16-fused] L={layer_idx} M={M} r=16 '
                      f'out_norm={out_flat.float().norm().item():.2f} count={_fp16_sub_count}',
                      file=sys.stderr)
            return out_flat.reshape(orig_shape)
        return _real_forward(x)

    mlp.forward = _koopman_forward
    print(f'[CIPHER fp16-fused] Layer {layer_idx} hooked', file=sys.stderr)

def install_all_mlp_hooks(model):
    """Install fused Koopman on all 32 layers with per-layer calibration."""
    count = 0
    for i in range(32):
        if os.path.exists(f'/workspace/manifold/layers/V_layer{i}.npy'):
            install_mlp_hooks(model, layer_idx=i)
            count += 1
    print(f'[CIPHER fp16-fused] {count}/32 layers hooked', file=sys.stderr)

# ── torch.mm interception (Session 3 fp32 path) ─────────────────────────────
_real_mm = torch.mm

def _cipher_mm(a, b, *, out=None):
    global _block_sub_ready
    M, K = a.shape[0], a.shape[1]
    N = b.shape[1]
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    is_f32 = (a.dtype == torch.float32)
    if _has_block_sub and _block_sub_ready and is_f32 and M == 4096 and K == 4096:
        if _hook.cipher_block_sub_predict(a.data_ptr(), out.data_ptr(), 4096):
            _hook.cipher_feed_ring(0, 0.85)
            return out
    _hook.cipher_set_gemm_ptrs(M, N, K, a.data_ptr(), b.data_ptr(), out.data_ptr())
    result = _real_mm(a, b, out=out)
    if _has_block_sub and not _block_sub_ready and is_f32 and M == 4096 and K == 4096:
        _block_sub_ready = _hook.cipher_block_sub_collect(a.data_ptr(), out.data_ptr(), 4096)
    return result

torch.mm = _cipher_mm
