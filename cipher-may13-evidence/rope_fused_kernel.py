"""PHASE 5: Fused RoPE kernel that processes Q and K in ONE launch.

Mistral RoPE (rotate-half convention):
  for d in [0, D/2):  rotated[d] = q[d] * cos[d] + (-q[d+D/2]) * sin[d]
  for d in [D/2, D):  rotated[d] = q[d] * cos[d] +   q[d-D/2]  * sin[d]

Where cos/sin are shape [..., D] with cos[d]=cos[d+D/2] and sin[d]=sin[d+D/2]
(the "concatenated halves" pattern in transformers' apply_rotary_pos_emb).

Kernel launch:
  grid = (B * (Hq + Hkv) * S, 1, 1)
  block = (D, 1, 1)
  Each block processes ONE (b, head, s) row of D elements.
  Each thread handles one dim d.
"""
import os, ctypes
os.environ.setdefault("CIPHER_SUBSTITUTE_V2", "on")

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/op31-prod-fix"
ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
rt = ctypes.CDLL(os.path.join(ROOT, "libcipher_rt.so"), mode=ctypes.RTLD_GLOBAL)
rt.cipher_substitute_v2_init.restype = ctypes.c_int
rt.cipher_substitute_v2_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
rt.cipher_substitute_v2_compile.restype = ctypes.c_ulong
rt.cipher_substitute_v2_get_function.argtypes = [ctypes.c_ulong]
rt.cipher_substitute_v2_get_function.restype = ctypes.c_void_p
rt.cipher_substitute_v2_init()

# NVRTC source for fused RoPE on Q+K
ROPE_KERNEL_SRC = r"""
#include <cuda_fp16.h>

extern "C" __global__ void cipher_fused_rope_qk(
    __half*       __restrict__ Q_out,      // [B, Hq, S, D]
    __half*       __restrict__ K_out,      // [B, Hkv, S, D]
    const __half* __restrict__ Q_in,       // [B, Hq, S, D]
    const __half* __restrict__ K_in,       // [B, Hkv, S, D]
    const __half* __restrict__ cos,        // [B, S, D] OR [S, D] (broadcast over batch)
    const __half* __restrict__ sin,
    int B, int Hq, int Hkv, int S, int D,
    int cos_batch_stride                   // 0 means cos/sin broadcast across batch
)
{
    // Block grid: ((Hq+Hkv) * S, B, 1)  — gridDim.x = (Hq+Hkv)*S, gridDim.y = B
    // Block: (D, 1, 1)
    int half_D = D / 2;
    int d = threadIdx.x;
    if (d >= D) return;

    int b = blockIdx.y;
    int row_idx = blockIdx.x;            // 0 .. (Hq+Hkv)*S - 1
    int total_q_rows = Hq * S;

    bool is_q = row_idx < total_q_rows;
    int local = is_q ? row_idx : row_idx - total_q_rows;
    int H = is_q ? Hq : Hkv;
    int h = local / S;
    int s = local % S;

    const __half* x_in;
    __half* x_out;
    if (is_q) {
        x_in  = Q_in  + ((b * Hq  + h) * S + s) * D;
        x_out = Q_out + ((b * Hq  + h) * S + s) * D;
    } else {
        x_in  = K_in  + ((b * Hkv + h) * S + s) * D;
        x_out = K_out + ((b * Hkv + h) * S + s) * D;
    }

    // cos/sin index: [b, s, d] OR [s, d] if broadcast (cos_batch_stride==0)
    int cos_offset = b * cos_batch_stride + s * D + d;
    float c = __half2float(cos[cos_offset]);
    float si = __half2float(sin[cos_offset]);

    // Load self and pair into shared memory so we can swap halves
    extern __shared__ float shm[];
    shm[d] = __half2float(x_in[d]);
    __syncthreads();

    float x_self = shm[d];
    float x_pair, sign;
    if (d < half_D) {
        x_pair = shm[d + half_D];
        sign = -1.0f;            // first half: rotated[d] = x[d]*cos + (-x[d+H/2])*sin
    } else {
        x_pair = shm[d - half_D];
        sign = 1.0f;             // second half: rotated[d] = x[d]*cos +   x[d-H/2] *sin
    }
    float result = x_self * c + sign * x_pair * si;
    x_out[d] = __float2half(result);
}
"""

# Compile via cipher's NVRTC pipeline
_cubin_id = rt.cipher_substitute_v2_compile(ROPE_KERNEL_SRC.encode(), b"cipher_fused_rope_qk")
assert _cubin_id != 0, "Failed to compile fused RoPE kernel"
_kfn = rt.cipher_substitute_v2_get_function(_cubin_id)
assert _kfn, "Failed to get fused RoPE function"

# Resolve cuLaunchKernel
_libcuda = ctypes.CDLL("libcuda.so.1")
_cu_launch = _libcuda.cuLaunchKernel
_cu_launch.argtypes = [
    ctypes.c_void_p,                                          # CUfunction
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,              # grid x, y, z
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,              # block x, y, z
    ctypes.c_uint,                                            # shared mem bytes
    ctypes.c_void_p,                                          # stream
    ctypes.POINTER(ctypes.c_void_p),                          # kernel params
    ctypes.POINTER(ctypes.c_void_p),                          # extra
]
_cu_launch.restype = ctypes.c_int


def fused_rope_qk(q, k, cos, sin, q_out=None, k_out=None):
    """Apply RoPE to q,k via fused kernel. Returns (q_rot, k_rot).

    Shapes:
      q: [B, Hq, S, D]
      k: [B, Hkv, S, D]
      cos: [B, S, D] OR [S, D] OR [1, S, D] (auto-detected by stride)
      sin: same shape as cos
    """
    assert q.dtype == torch.float16 and k.dtype == torch.float16
    B, Hq, S, D = q.shape
    _, Hkv, _, _ = k.shape
    if q_out is None: q_out = torch.empty_like(q)
    if k_out is None: k_out = torch.empty_like(k)

    # cos/sin shape: derive batch stride
    if cos.dim() == 2:           # [S, D]
        cos_bs = 0
    elif cos.shape[0] == 1:      # [1, S, D]
        cos_bs = 0
    else:                        # [B, S, D]
        cos_bs = S * D

    args_list = [
        ctypes.c_void_p(q_out.data_ptr()),
        ctypes.c_void_p(k_out.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(cos.data_ptr()),
        ctypes.c_void_p(sin.data_ptr()),
        ctypes.c_int(B), ctypes.c_int(Hq), ctypes.c_int(Hkv),
        ctypes.c_int(S), ctypes.c_int(D), ctypes.c_int(cos_bs),
    ]
    args_arr = (ctypes.c_void_p * len(args_list))(*[
        ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args_list
    ])

    grid_x = (Hq + Hkv) * S
    grid_y = B
    block_x = D
    shared_bytes = D * 4   # float row buffer
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = _cu_launch(_kfn, grid_x, grid_y, 1, block_x, 1, 1, shared_bytes,
                    stream, args_arr, None)
    if rc != 0:
        raise RuntimeError(f"fused_rope_qk launch failed rc={rc}")
    return q_out, k_out


# ===== Correctness test =====
if __name__ == "__main__":
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

    print("=== Phase 5 fused RoPE kernel ===", flush=True)
    print("[compile] kernel compiled, cubin_id =", _cubin_id, flush=True)

    # Correctness across decode shapes
    print("\n=== Correctness test ===", flush=True)
    print(f"  {'B':>3} {'Hq':>3} {'Hkv':>3} {'S':>3} {'D':>4}  {'q_max_err':>10}  {'k_max_err':>10}  {'status':<6}", flush=True)
    test_cases = [
        (1, 32, 8, 1, 128),
        (8, 32, 8, 1, 128),
        (32, 32, 8, 1, 128),
        (1, 32, 8, 17, 128),  # multi-seq
    ]
    for B, Hq, Hkv, S, D in test_cases:
        torch.manual_seed(0)
        q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16) * 0.5
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16) * 0.5
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.float16) * 0.5
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.float16) * 0.5

        q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)
        q_fused, k_fused = fused_rope_qk(q.contiguous(), k.contiguous(), cos.contiguous(), sin.contiguous())
        torch.cuda.synchronize()
        q_max = (q_ref - q_fused).abs().max().item()
        k_max = (k_ref - k_fused).abs().max().item()
        ok = "PASS" if (q_max < 5e-3 and k_max < 5e-3) else "FAIL"
        print(f"  {B:>3} {Hq:>3} {Hkv:>3} {S:>3} {D:>4}  {q_max:>10.6f}  {k_max:>10.6f}  {ok:<6}", flush=True)

    # Speed test
    print(f"\n=== Speed test (median of 1000 iters) ===", flush=True)
    print(f"  {'B':>3}  {'pytorch_us':>12}  {'fused_us':>10}  {'speedup':>9}", flush=True)
    for B in [1, 8, 32]:
        Hq = 32; Hkv = 8; S = 1; D = 128
        q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.float16)
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        q_out = torch.empty_like(q); k_out = torch.empty_like(k)

        N = 1000
        for _ in range(20):
            _ = apply_rotary_pos_emb(q, k, cos, sin)
            _ = fused_rope_qk(q, k, cos, sin, q_out, k_out)
        torch.cuda.synchronize()

        t_pt = []
        for _ in range(N):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = apply_rotary_pos_emb(q, k, cos, sin)
            e.record()
            torch.cuda.synchronize()
            t_pt.append(s.elapsed_time(e) * 1000.0)

        t_fused = []
        for _ in range(N):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = fused_rope_qk(q, k, cos, sin, q_out, k_out)
            e.record()
            torch.cuda.synchronize()
            t_fused.append(s.elapsed_time(e) * 1000.0)

        t_pt.sort(); t_fused.sort()
        med_pt = t_pt[N // 2]; med_fused = t_fused[N // 2]
        speedup = med_pt / med_fused if med_fused > 0 else 0
        print(f"  {B:>3}  {med_pt:>12.1f}  {med_fused:>10.1f}  {speedup:>8.2f}x", flush=True)
