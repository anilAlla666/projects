#!/usr/bin/env python3
"""
cipher_startup.py — Register Koopman matrices at CIPHER startup.

Usage:
    import cipher_startup
    cipher_startup.register_synthetic_matrices()  # for demos/testing
    # cipher_startup.register_edmd_matrices()     # for production (requires live data)

Synthetic matrices prove the substitution mechanism fires end-to-end.
Output is approximate (identity * 0.01 attenuation). For correct output,
use EDMD-derived matrices from live inference data.
"""

import ctypes
import torch

def register_synthetic_matrices():
    """Register synthetic Koopman matrices for all production shapes.

    Substitution fires, output is approximate (not mathematically correct).
    Used for demos, testing, and proving the mechanism works.
    """
    rt = ctypes.CDLL('./libcipher_rt.so')
    rt.cipher_koopman_fp16_register_shape.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]
    rt.cipher_koopman_fp16_register_shape.restype = ctypes.c_int

    r = 16  # Koopman subspace rank

    shapes = [
        (4096,  4096,  "attention projections"),
        (4096,  14336, "FFN gate/up (7B)"),
        (14336, 4096,  "FFN down (7B)"),
        (128,   512,   "decode attention Q@K.T (seq=512)"),
        (128,   1024,  "decode attention Q@K.T (seq=1024)"),
    ]

    for K_dim, N_dim, desc in shapes:
        V_T  = torch.eye(r, K_dim, dtype=torch.float32).cuda() * 0.01
        K_op = torch.eye(r, r,     dtype=torch.float32).cuda()
        W    = torch.eye(r, N_dim, dtype=torch.float32).cuda() * 0.01

        res = rt.cipher_koopman_fp16_register_shape(
            K_dim, N_dim,
            V_T.data_ptr(), K_op.data_ptr(), W.data_ptr()
        )
        status = "OK" if res == 0 else f"FAIL({res})"
        print(f'[CIPHER] Registered K={K_dim:>5} N={N_dim:>5} — {desc} [{status}]')

    print(f'[CIPHER] {len(shapes)} synthetic shapes registered (r={r})')
