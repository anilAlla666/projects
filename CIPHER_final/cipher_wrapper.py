# CIPHER automatic torch.mm interception
# Import this module to route all torch.mm calls through CIPHER's pipeline.
# No other CIPHER-specific code needed in user scripts.

import torch
import ctypes

_hook = ctypes.CDLL(None)
_hook.cipher_set_gemm_ptrs.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3
_hook.cipher_set_gemm_ptrs.restype = None

_real_mm = torch.mm

def _cipher_mm(a, b, *, out=None):
    M, K = a.shape
    K2, N = b.shape
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=a.dtype)
    _hook.cipher_set_gemm_ptrs(M, N, K, a.data_ptr(), b.data_ptr(), out.data_ptr())
    return _real_mm(a, b, out=out)

torch.mm = _cipher_mm
